"""Null calibration of the alpha discovery pipeline: a false-promotion rate with a confidence interval.

Five null panels that promote nothing cannot bound a 5 % familywise error rate:
0 events in 5 campaigns leaves an exact 95 % upper bound of 52 %. This study
runs ``replicates`` independent null campaigns — driftless panels with nothing
planted, each with its own noise seed and search seed derived from the study
seed and the replicate index — through the default alpha policy and reports,
with Clopper–Pearson intervals, the campaign-level rates that matter:

* **promotion at review** — the campaign promoted at least one candidate on data
  with nothing to find (judged with the validation-trial count at that moment);
* **holdout reach** — at least one promotion survives the final selection's
  re-check at the campaign's final trial count, so ``finalize`` would have scored
  a null candidate on the reserved holdout;
* **holdout read** — any holdout access during the campaign. There must be none:
  only ``finalize`` reads the holdout and the study never calls it.

The campaign is the unit because the deflated Sharpe ratio controls the error
rate of one family of validation trials; the pooled per-trial rate is a summary
only. Everything here is about artificial panels and this code: a small panel
is a cheap proxy for the benchmark panel, not the same regime, and the result
is a measurement of the pipeline, not evidence about any market.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, TypedDict

from autoresearch.lab.runtime import limit_blas_threads, requested_threads

STUDY_SEED = 2026
DEFAULTS = {"names": 80, "sessions": 700, "rounds": 8, "proposals": 24}
REPORT_JSON = "benchmark-null.json"
REPORT_MARKDOWN = "benchmark-null.md"


# ------------------------------------------------------------------ binomial interval
def _log_choose(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _binomial_sum(first: int, last: int, n: int, p: float) -> float:
    """P(first <= X <= last) for X ~ Binomial(n, p), summed in log space."""
    if first > last:
        return 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    terms = [_log_choose(n, i) + i * log_p + (n - i) * log_q for i in range(first, last + 1)]
    peak = max(terms)
    return min(1.0, math.exp(peak) * sum(math.exp(t - peak) for t in terms))


def binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if n < 0 or not 0.0 <= p <= 1.0:
        raise ValueError("binomial_cdf needs n >= 0 and 0 <= p <= 1")
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p == 0.0:
        return 1.0
    if p == 1.0:
        return 0.0
    return _binomial_sum(0, k, n, p)


def binomial_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    if n < 0 or not 0.0 <= p <= 1.0:
        raise ValueError("binomial_tail needs n >= 0 and 0 <= p <= 1")
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return 1.0
    return _binomial_sum(k, n, n, p)


def _root(increasing: Callable[[float], float]) -> float:
    """The p in [0, 1] where a monotonically increasing function crosses zero (bisection)."""
    low, high = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if increasing(mid) < 0.0:
            low = mid
        else:
            high = mid
        if high - low <= 1e-15:
            break
    return 0.5 * (low + high)


def clopper_pearson(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact two-sided binomial confidence interval for a proportion (Clopper and Pearson, 1934).

    The lower bound is the p at which ``successes`` or more events have probability ``(1 -
    confidence) / 2``; the upper bound the p at which ``successes`` or fewer do. Closed forms
    at 0 and at ``trials`` events; bisection on the exact binomial sums otherwise. The
    interval's coverage is at least ``confidence`` for every p, usually more.
    """
    if trials < 1:
        raise ValueError("a confidence interval needs at least one trial")
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie between 0 and trials")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    half = (1.0 - confidence) / 2.0
    if successes == 0:
        lower = 0.0
    elif successes == trials:
        lower = half ** (1.0 / trials)
    else:
        lower = _root(lambda p: binomial_tail(successes, trials, p) - half)
    if successes == trials:
        upper = 1.0
    elif successes == 0:
        upper = 1.0 - half ** (1.0 / trials)
    else:
        upper = _root(lambda p: half - binomial_cdf(successes, trials, p))
    return lower, upper


# ----------------------------------------------------------------------- replicates
class PromotedNull(TypedDict):
    """One false promotion: its review numbers and whether it would have reached the holdout."""

    expression: str
    generator: str
    validation_daily_sharpe: float | None
    dsr_at_review: float | None
    final_check: dict[str, Any]
    would_reach_holdout: bool


class ReplicateResult(TypedDict):
    index: int
    panel_seed: int
    search_seed: int
    campaign_id: str
    status: str
    budget: str | None
    rounds: int
    screen_trials: int
    screen_errors: int
    validation_trials: int
    promotions: int
    finalists: int
    holdout_accessed: bool
    ledger_ok: bool
    promoted: list[PromotedNull]
    seconds: float


def replicate_seeds(study_seed: int, index: int) -> tuple[int, int]:
    """(panel noise seed, search seed) of one replicate, a pure function of the study seed and the index."""
    if study_seed < 0 or index < 0:
        raise ValueError("study seed and replicate index must be nonnegative")
    digest = hashlib.sha256(f"alpha-null-calibration:{study_seed}:{index}".encode()).digest()
    panel = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
    search = int.from_bytes(digest[4:8], "big") & 0x7FFFFFFF
    return panel, search


def replicate_root(root: Path, index: int) -> Path:
    return Path(root) / f"replicate-{index:03d}"


def null_replicate(root: Path, index: int, *, study_seed: int = STUDY_SEED, names: int = DEFAULTS["names"],
                   sessions: int = DEFAULTS["sessions"], rounds: int = DEFAULTS["rounds"],
                   proposals: int = DEFAULTS["proposals"]) -> ReplicateResult:
    """One null panel and one default-policy campaign on it; the holdout is never read.

    The replicate's folder holds its data, records and runtime. A finished replicate
    (``result.json`` present) is returned as recorded, so an interrupted study resumes.
    """
    from autoresearch.lab.campaign import LabCampaign

    from .benchmark import _drive, benchmark_config
    from .planted import create_planted_dataset

    folder = replicate_root(root, index)
    recorded = folder / "result.json"
    if recorded.exists():
        loaded: ReplicateResult = json.loads(recorded.read_text(encoding="utf-8"))
        return loaded
    started = time.time()
    panel_seed, search_seed = replicate_seeds(study_seed, index)
    dataset = create_planted_dataset(folder / "data", symbols=names, days=sessions, seed=panel_seed, strength={})
    config = benchmark_config(f"null-calibration-{index:03d}", dataset, folder, symbols=names, rounds=rounds,
                              seed=search_seed, proposals=proposals)
    campaign = LabCampaign(config)
    try:
        _drive(campaign, rounds)
        status = campaign.status()
        accepted = campaign.checkpoint["accepted"]
        context = campaign.final_context()
        finalists = {entry["key"] for entry in campaign.domain.final_selection(accepted, context)}
        promoted: list[PromotedNull] = []
        for entry in accepted:
            check = {k: v for k, v in campaign.domain.final_check(entry, context).items() if k != "eligible"}
            promoted.append({"expression": entry["candidate"]["expression"], "generator": entry["generator"],
                             "validation_daily_sharpe": entry["score"],
                             "dsr_at_review": (entry.get("checks") or {}).get("deflated_sharpe", {}).get("dsr"),
                             "final_check": check, "would_reach_holdout": entry["key"] in finalists})
        ledger = campaign.ledger.verify()
    finally:
        campaign.domain.release()
    result: ReplicateResult = {"index": index, "panel_seed": panel_seed, "search_seed": search_seed,
                               "campaign_id": campaign.campaign_id, "status": status["status"], "budget": status["budget"],
                               "rounds": status["round"], "screen_trials": status["screen_trials"],
                               "screen_errors": status["screen_errors"], "validation_trials": status["validation_trials"],
                               "promotions": status["promotions"], "finalists": len(finalists),
                               "holdout_accessed": bool(status["holdout_accessed"]), "ledger_ok": bool(ledger.get("ok")),
                               "promoted": promoted, "seconds": round(time.time() - started, 2)}
    recorded.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def _job(args: tuple[str, int, int, int, int, int, int]) -> ReplicateResult:
    root, index, study_seed, names, sessions, rounds, proposals = args
    return null_replicate(Path(root), index, study_seed=study_seed, names=names, sessions=sessions, rounds=rounds,
                          proposals=proposals)


# ------------------------------------------------------------------------- summary
def _rate(events: int, total: int) -> dict[str, Any]:
    if total < 1:
        return {"events": events, "of": total, "rate": None, "ci95": None}
    low, high = clopper_pearson(events, total)
    return {"events": events, "of": total, "rate": round(events / total, 6), "ci95": [round(low, 6), round(high, 6)]}


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("no values")
    position = q * (len(sorted_values) - 1)
    low = math.floor(position)
    high = min(low + 1, len(sorted_values) - 1)
    return float(sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low))


def _distribution(values: Sequence[int]) -> dict[str, Any]:
    ordered = sorted(values)
    return {"total": int(sum(ordered)), "min": ordered[0], "p25": round(_quantile(ordered, 0.25), 2),
            "median": round(_quantile(ordered, 0.5), 2), "p75": round(_quantile(ordered, 0.75), 2), "max": ordered[-1],
            "mean": round(sum(ordered) / len(ordered), 3),
            "histogram": {str(k): v for k, v in sorted(Counter(ordered).items())}}


def summarize(results: Sequence[ReplicateResult], nominal_level: float) -> dict[str, Any]:
    """Campaign-level rates with exact intervals, the per-trial summary and the trial distributions."""
    if not results:
        raise ValueError("a null study needs at least one replicate")
    n = len(results)
    validated = [r for r in results if r["validation_trials"] > 0]
    promoted_any = sum(1 for r in results if r["promotions"] > 0)
    reach_any = sum(1 for r in results if r["finalists"] > 0)
    total_trials = sum(r["validation_trials"] for r in results)
    total_promotions = sum(r["promotions"] for r in results)
    total_finalists = sum(r["finalists"] for r in results)
    promotion = _rate(promoted_any, n)
    reach = _rate(reach_any, n)
    return {"replicates": n,
            "promotion_at_review": promotion,
            "holdout_reach": reach,
            "holdout_read": _rate(sum(1 for r in results if r["holdout_accessed"]), n),
            "validated_any": _rate(len(validated), n),
            "promotion_at_review_given_validation": _rate(sum(1 for r in validated if r["promotions"] > 0), len(validated)),
            "holdout_reach_given_validation": _rate(sum(1 for r in validated if r["finalists"] > 0), len(validated)),
            "per_trial": {**_rate(total_promotions, total_trials), "finalists": total_finalists,
                          "note": "pooled over campaigns and treated as independent draws; a summary, not the "
                                  "familywise quantity the deflated Sharpe ratio controls"},
            "validation_trials": _distribution([r["validation_trials"] for r in results]),
            "screen_trials": _distribution([r["screen_trials"] for r in results]),
            "promotions": _distribution([r["promotions"] for r in results]),
            "nominal_familywise_level": nominal_level,
            "promotion_upper_bound_below_nominal": bool(promotion["ci95"] and promotion["ci95"][1] < nominal_level),
            "holdout_reach_upper_bound_below_nominal": bool(reach["ci95"] and reach["ci95"][1] < nominal_level),
            "all_complete": all(r["status"] == "complete" for r in results),
            "all_ledgers_ok": all(r["ledger_ok"] for r in results),
            "seconds_per_replicate": round(sum(r["seconds"] for r in results) / n, 2)}


def study_command(root: Path, *, replicates: int, names: int, sessions: int, rounds: int, proposals: int,
                  study_seed: int, workers: int) -> str:
    return (f"tubby-lab benchmark null --root {root} --replicates {replicates} --names {names} --sessions {sessions} "
            f"--rounds {rounds} --proposals {proposals} --study-seed {study_seed} --workers {workers}")


def run_null_study(root: str | Path, *, replicates: int, names: int = DEFAULTS["names"],
                   sessions: int = DEFAULTS["sessions"], rounds: int = DEFAULTS["rounds"],
                   proposals: int = DEFAULTS["proposals"], study_seed: int = STUDY_SEED, workers: int = 1) -> dict[str, Any]:
    """``replicates`` null campaigns, deterministic per (study seed, index); writes the JSON and Markdown reports.

    ``workers`` runs replicates in separate processes; it changes the wall time, never a number.
    """
    from autoresearch.lab.campaign import LAB_VERSION
    from qlib_yahoo.alphalab.simulate import ENGINE_VERSION

    from .domain import POLICY_DEFAULTS

    for label, value, floor in (("replicates", replicates, 1), ("names", names, 20), ("sessions", sessions, 300),
                                ("rounds", rounds, 1), ("proposals", proposals, 1), ("workers", workers, 1),
                                ("study_seed", study_seed, 0)):
        if type(value) is not int or value < floor:
            raise ValueError(f"{label} must be an integer of at least {floor}")
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    seeds = [replicate_seeds(study_seed, index) for index in range(replicates)]
    if len({panel for panel, _ in seeds}) < replicates:
        raise RuntimeError("two replicates drew the same panel seed; choose another study seed")
    started = time.time()
    jobs = [(str(root), index, study_seed, names, sessions, rounds, proposals) for index in range(replicates)]
    if workers == 1:
        results = [_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, replicates), initializer=limit_blas_threads,
                                 initargs=(requested_threads(),)) as pool:
            results = list(pool.map(_job, jobs))
    results.sort(key=lambda r: r["index"])
    wall = round(time.time() - started, 1)
    nominal = round(1.0 - float(POLICY_DEFAULTS["min_dsr"]), 6)
    import numpy
    import pandas
    report: dict[str, Any] = {
        "kind": "alpha_null_calibration", "lab_version": LAB_VERSION, "engine": ENGINE_VERSION,
        "note": "Artificial driftless null panels with nothing planted; every promotion is a false discovery. "
                "Measures this pipeline on this code, not any market.",
        "study": {"replicates": replicates, "names": names, "sessions": sessions, "rounds": rounds,
                  "proposals_per_round": proposals, "study_seed": study_seed, "workers": workers, "policy": "default",
                  "seeds": "sha256('alpha-null-calibration:<study seed>:<index>'): bytes 0-3 panel, 4-7 search",
                  "command": study_command(root, replicates=replicates, names=names, sessions=sessions, rounds=rounds,
                                           proposals=proposals, study_seed=study_seed, workers=workers)},
        "environment": {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__,
                        "platform": platform.platform(), "executable": sys.executable,
                        "blas_threads": requested_threads()},
        "summary": summarize(results, nominal), "replicates": results, "wall_seconds": wall}
    (root / REPORT_JSON).write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (root / REPORT_MARKDOWN).write_text(render_markdown(report), encoding="utf-8")
    return report


def _interval(rate: dict[str, Any]) -> str:
    if rate["ci95"] is None:
        return "-"
    low, high = rate["ci95"]
    return f"[{low:.4f}, {high:.4f}]"


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.2f} %"


def render_markdown(report: dict[str, Any]) -> str:
    study, summary = report["study"], report["summary"]
    n = summary["replicates"]
    lines = ["# Alpha null calibration", "",
             f"{n} independent null panels ({study['names']} names x {study['sessions']} sessions, driftless, nothing "
             f"planted; noise and search seeds derived from study seed {study['study_seed']} and the replicate index), "
             f"one alpha campaign each with the default policy: {study['rounds']} rounds of "
             f"{study['proposals_per_round']} proposals. Every promotion is a false discovery. Artificial data; this "
             "measures the pipeline, not a market.", "",
             f"Command: `{study['command']}`", "",
             "| Campaign-level rate | Events | Rate | 95 % Clopper-Pearson |", "| --- | --- | --- | --- |"]
    rows = (("promoted at least one null candidate at review", "promotion_at_review"),
            ("at least one promotion would reach the holdout (survives the final re-check)", "holdout_reach"),
            ("read the reserved holdout", "holdout_read"),
            ("validated at least one candidate", "validated_any"),
            ("promoted at review, among campaigns with a validation trial", "promotion_at_review_given_validation"),
            ("would reach the holdout, among campaigns with a validation trial", "holdout_reach_given_validation"))
    for label, key in rows:
        rate = summary[key]
        lines.append(f"| {label} | {rate['events']}/{rate['of']} | {_percent(rate['rate'])} | {_interval(rate)} |")
    per = summary["per_trial"]
    trials, screens, promotions = summary["validation_trials"], summary["screen_trials"], summary["promotions"]
    lines += ["",
              f"Nominal familywise level of the default policy: {_percent(summary['nominal_familywise_level'])} "
              f"(1 - `min_dsr`). The upper bound of the promotion rate is "
              f"{'below' if summary['promotion_upper_bound_below_nominal'] else 'not below'} it; the upper bound of "
              f"the holdout-reach rate is {'below' if summary['holdout_reach_upper_bound_below_nominal'] else 'not below'} it.",
              "",
              f"Per trial (summary only): {per['events']} false promotions in {per['of']} validation trials "
              f"({_percent(per['rate'])}, {_interval(per)}); {per['finalists']} would reach the holdout. {per['note']}.",
              "",
              f"Validation trials per campaign: min {trials['min']}, p25 {trials['p25']}, median {trials['median']}, "
              f"p75 {trials['p75']}, max {trials['max']}, mean {trials['mean']}; "
              "histogram " + ", ".join(f"{k}: {v}" for k, v in trials["histogram"].items()) + ".",
              f"Screens per campaign: mean {screens['mean']} (min {screens['min']}, max {screens['max']}, "
              f"total {screens['total']}). Promotions per campaign: max {promotions['max']}, total {promotions['total']}.",
              ""]
    promoted = [(r["index"], p) for r in report["replicates"] for p in r["promoted"]]
    if promoted:
        lines += ["| Replicate | False promotion | Generator | DSR at review | DSR at final count | Would reach the holdout |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for index, entry in promoted:
            final = entry["final_check"].get("dsr")
            lines.append(f"| {index} | `{entry['expression'][:80]}` | {entry['generator']} | "
                         f"{entry['dsr_at_review']} | {final} | {'yes' if entry['would_reach_holdout'] else 'no'} |")
        lines.append("")
    else:
        lines += ["No null candidate was promoted in any replicate.", ""]
    environment = report["environment"]
    lines += [f"All campaigns complete: {'yes' if summary['all_complete'] else 'no'}; all trial ledgers intact: "
              f"{'yes' if summary['all_ledgers_ok'] else 'no'}. Wall time {report['wall_seconds']} s with "
              f"{study['workers']} worker(s), {summary['seconds_per_replicate']} s per replicate. "
              f"Engine {report['engine']}, lab version {report['lab_version']}, Python {environment['python']}, "
              f"numpy {environment['numpy']}, pandas {environment['pandas']}, {environment['platform']}"
              + (f", {environment['blas_threads']} BLAS thread(s) per worker" if environment.get("blas_threads") else "")
              + ".", ""]
    return "\n".join(lines)
