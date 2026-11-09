"""Research benchmark for the discovery pipeline: recall on planted panels, false promotions on null panels.

``run_benchmark`` builds two artificial panels with identical shape and noise —
one with the three planted signals of :mod:`.planted`, one with none — runs the
same discovery campaign on both, finalizes the planted campaign's selection on
its reserved holdout, and reports:

* **recall**: for each planted signal, the best mean daily cross-sectional rank
  correlation between a promoted alpha and the signal's oracle expression over
  the validation period (recovered when at least ``recovery_threshold``);
* **false promotions**: everything promoted on the null panel;
* **holdout agreement**: confirmed finals among the planted selection;
* the trial counts behind each number.

The numbers measure the pipeline on synthetic data with known answers. They
are not evidence about any market.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

from autoresearch.lab.runtime import requested_threads

from .domain import AlphaDiscoveryDomain
from .planted import ORACLES, SIGNALS, create_planted_dataset

if TYPE_CHECKING:
    from autoresearch.lab.campaign import LabCampaign

RECOVERY_THRESHOLD = 0.5


class RecallRow(TypedDict):
    """How well the promoted set recovered one planted signal."""

    signal: str
    oracle: str
    best_correlation: float | None
    expression: str | None
    generator: str | None
    recovered: bool


def benchmark_config(name: str, dataset: Path, root: Path, *, symbols: int, rounds: int, seed: int,
                     proposals: int, universe: str | None = None) -> dict[str, Any]:
    """Campaign file for a planted panel: 7.5-45 % development, 45-82.5 % validation, the rest reserved."""
    dates = (dataset / "qlib_data/calendars/day.txt").read_text().split()
    n = len(dates)
    dev_start, dev_end = int(n * 0.075), int(n * 0.45) - 1
    val_end = int(n * 0.825) - 1
    return {"name": name, "domain": "alpha", "seed": seed,
            "research_root": str(root / "records"), "runtime_root": str(root / "runtime"),
            "lineage_root": str(root / "lineage"),   # isolated by construction, not by location
            "settings": {"dataset": str(dataset), "universe": universe or f"TOP{symbols}", "cost_bps": 5, "warmup": 130,
                         "truncation": 0.05,
                         "development_start": dates[dev_start], "development_end": dates[dev_end],
                         "validation_start": dates[dev_end + 1], "validation_end": dates[val_end],
                         "holdout_start": dates[val_end + 1]},
            "limits": {"max_rounds": rounds, "proposals_per_round": proposals, "validations_per_round": 3}}


def _spearman_by_day(a: np.ndarray, b: np.ndarray, members: np.ndarray) -> float | None:
    values = []
    for row in range(a.shape[0]):
        ok = members[row] & np.isfinite(a[row]) & np.isfinite(b[row])
        if ok.sum() < 5:
            continue
        ra = np.argsort(np.argsort(a[row, ok])).astype(float)
        rb = np.argsort(np.argsort(b[row, ok])).astype(float)
        if ra.std() == 0 or rb.std() == 0:
            continue
        values.append(float(np.corrcoef(ra, rb)[0, 1]))
    return float(np.mean(values)) if values else None


def recall(campaign: LabCampaign, threshold: float = RECOVERY_THRESHOLD) -> dict[str, RecallRow]:
    """Best match between promoted alphas and each planted oracle on the validation period."""
    domain = campaign.domain
    if not isinstance(domain, AlphaDiscoveryDomain):
        raise ValueError("recall is measured on alpha discovery campaigns")
    evaluator = domain.evaluator
    oracle = {name: evaluator.alpha_matrix(expression, "validation") for name, expression in ORACLES.items()}
    out: dict[str, RecallRow] = {}
    promoted = [a for a in campaign.checkpoint["accepted"]]
    matrices: dict[str, tuple[str, str, np.ndarray]] = {}
    for entry in promoted:
        expression = entry["candidate"]["expression"]
        matrices[entry["key"]] = (expression, entry["generator"], evaluator.alpha_matrix(expression, "validation")[0])
    for name in SIGNALS:
        target, members, _dates = oracle[name]
        best: RecallRow = {"signal": name, "oracle": ORACLES[name], "best_correlation": None, "expression": None,
                           "generator": None, "recovered": False}
        for key, (expression, generator, values) in matrices.items():
            correlation = _spearman_by_day(values, target, members)
            if correlation is not None and (best["best_correlation"] is None or correlation > best["best_correlation"]):
                best["best_correlation"] = round(correlation, 4)
                best["expression"] = expression
                best["generator"] = generator
        best["recovered"] = best["best_correlation"] is not None and best["best_correlation"] >= threshold
        out[name] = best
    return out


def _drive(campaign: LabCampaign, rounds: int) -> list[dict[str, Any]]:
    log: list[dict[str, Any]] = []
    for _ in range(rounds + 1):
        outcome = campaign.tick()
        log.append({key: outcome.get(key) for key in ("round", "status", "screened", "validated", "promoted",
                                                      "best_fitness", "reason")})
        if outcome["status"] != "round":
            break
    return log


def run_benchmark(root: str | Path, *, symbols: int = 300, days: int = 2000, rounds: int = 20, seed: int = 3,
                  proposals: int = 40, panel_seed: int = 11,
                  extra_null_seeds: tuple[int, ...] = (101, 202, 303, 404), null_twin: bool = True,
                  signal_profile: str = "stationary", dated_universe: bool = False,
                  universe_size: int | None = None) -> dict[str, Any]:
    """Planted panel, its null twin and further null panels with other noise (one null run is one draw).

    ``signal_profile`` (stationary, decay, intermittent), ``dated_universe`` and ``universe_size`` are
    passed to every panel (see :mod:`.planted`); the campaigns use the TOP-``universe_size`` universe.
    ``null_twin=False`` with ``extra_null_seeds=()`` runs the planted panel alone.
    """
    from autoresearch.lab.campaign import LAB_VERSION, LabCampaign
    from qlib_yahoo.alphalab.simulate import ENGINE_VERSION

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    panel_options: dict[str, Any] = {"symbols": symbols, "days": days, "signal_profile_mode": signal_profile,
                                     "dated_universe": dated_universe, "universe_size": universe_size}
    universe = f"TOP{universe_size or symbols}"
    planted_data = create_planted_dataset(root / "planted-data", seed=panel_seed, **panel_options)
    report: dict[str, Any] = {"kind": "alpha_discovery_benchmark", "lab_version": LAB_VERSION, "engine": ENGINE_VERSION,
                              "panel": {"symbols": symbols, "days": days, "panel_seed": panel_seed, "search_seed": seed,
                                        "rounds": rounds, "proposals_per_round": proposals,
                                        "extra_null_seeds": list(extra_null_seeds), "null_twin": null_twin,
                                        "signal_profile": signal_profile, "dated_universe": dated_universe,
                                        "universe": universe},
                              "note": "Artificial panels with known answers; no market or investment evidence."}
    runs = [("planted", planted_data)]
    if null_twin:
        runs.append(("null", create_planted_dataset(root / "null-data", seed=panel_seed, strength={}, **panel_options)))
    for extra in extra_null_seeds:
        runs.append((f"null-{extra}", create_planted_dataset(root / f"null-{extra}-data", seed=extra, strength={},
                                                             **panel_options)))
    for label, dataset in runs:
        campaign = LabCampaign(benchmark_config(f"benchmark-{label}", dataset, root / label, symbols=symbols,
                                                rounds=rounds, seed=seed, proposals=proposals, universe=universe))
        log = _drive(campaign, rounds)
        status = campaign.status()
        accepted = campaign.checkpoint["accepted"]
        section: dict[str, Any] = {"campaign_id": campaign.campaign_id, "rounds": status["round"],
                                   "screen_trials": status["screen_trials"], "validation_trials": status["validation_trials"],
                                   "promotions": status["promotions"], "generators": status["generators"],
                                   "ledger": campaign.ledger.verify(), "log": log,
                                   "promoted": [{"expression": a["candidate"]["expression"], "generator": a["generator"],
                                                 "validation_daily_sharpe": a["score"],
                                                 "deflated_sharpe": (a.get("checks") or {}).get("deflated_sharpe", {}).get("dsr")}
                                                for a in accepted]}
        if label == "planted":
            section["recall"] = recall(campaign)
            section["recovered"] = sum(1 for r in section["recall"].values() if r["recovered"])
            finals = campaign.finalize() if accepted else []
            section["finals"] = {"selected": len(finals),
                                 "confirmed": sum(1 for f in finals if f["decision"] == "confirmed"),
                                 "decisions": [f["decision"] for f in finals]}
        else:
            # Promotions are judged with the trial count reached so far; final selection re-applies
            # the deflated Sharpe ratio with the campaign's final count before any holdout access.
            section["false_promotions"] = status["promotions"]
            section["false_finalists"] = len(campaign.domain.final_selection(accepted, campaign.final_context()))
        if label == "planted" or label == "null":
            report[label] = section
        else:
            section["panel_seed"] = int(label.split("-", 1)[1])
            report.setdefault("null_extra", []).append(section)
        campaign.domain.release()
    nulls = ([report["null"]] if "null" in report else []) + report.get("null_extra", [])
    report["null_total"] = {key: sum(n[key] for n in nulls) for key in
                            ("screen_trials", "validation_trials", "false_promotions", "false_finalists")}
    report["null_total"]["panels"] = len(nulls)
    report["runtime_seconds"] = round(time.time() - started, 1)
    report["blas_threads"] = requested_threads()
    (root / "benchmark.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (root / "benchmark.md").write_text(render_markdown(report))
    return report


def render_markdown(report: dict[str, Any]) -> str:
    planted, null, panel = report["planted"], report.get("null"), report["panel"]
    variant = []
    if panel.get("signal_profile", "stationary") != "stationary":
        variant.append(f"{panel['signal_profile']} planted signals")
    if panel.get("dated_universe"):
        variant.append(f"dated {panel.get('universe', '')} universe ranked monthly by trailing dollar volume")
    elif panel.get("universe") and panel["universe"] != f"TOP{panel['symbols']}":
        variant.append(f"static {panel['universe']} universe ranked by full-sample dollar volume")
    lines = ["# Alpha discovery benchmark", "",
             f"Artificial panels: {panel['symbols']} names x {panel['days']} sessions, "
             f"{panel['rounds']} rounds of {panel['proposals_per_round']} proposals"
             + (" (" + "; ".join(variant) + ")" if variant else "") + ". Known answers; no market evidence.", "",
             "| Panel | Screens | Validations | Promotions | Result |", "| --- | --- | --- | --- | --- |",
             f"| planted | {planted['screen_trials']} | {planted['validation_trials']} | {planted['promotions']} | "
             f"{planted['recovered']}/{len(SIGNALS)} planted signals recovered; "
             f"{planted['finals']['confirmed']}/{planted['finals']['selected']} finals confirmed |"]
    if null is not None:
        lines.append(f"| null twin | {null['screen_trials']} | {null['validation_trials']} | {null['promotions']} | "
                     f"{null['false_promotions']} false promotions at review; {null['false_finalists']} would reach the holdout |")
    for extra in report.get("null_extra", []):
        lines.append(f"| null (noise seed {extra['panel_seed']}) | {extra['screen_trials']} | {extra['validation_trials']} | "
                     f"{extra['promotions']} | {extra['false_promotions']} at review; {extra['false_finalists']} would reach the holdout |")
    total = report["null_total"]
    if total["panels"]:
        lines.append(f"| all {total['panels']} null panels | {total['screen_trials']} | {total['validation_trials']} | "
                     f"{total['false_promotions']} | {total['false_promotions']} false promotions at review; "
                     f"{total['false_finalists']} would reach the holdout |")
    lines += ["", "| Planted signal | Best correlation | Recovered | Matching alpha |", "| --- | --- | --- | --- |"]
    for name, row in planted["recall"].items():
        lines.append(f"| {name} | {row['best_correlation']} | {'yes' if row['recovered'] else 'no'} | "
                     f"`{(row['expression'] or '')[:90]}` |")
    threads = report.get("blas_threads")
    lines += ["", f"Engine {report['engine']}, lab version {report['lab_version']}, {report['runtime_seconds']} s"
              + (f" with {threads} BLAS thread(s)" if threads else "") + ".", ""]
    return "\n".join(lines)
