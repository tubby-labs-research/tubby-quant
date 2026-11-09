"""Benchmark for the machine-learning domain: known answers in, measured behaviour out.

Tasks (all offline and deterministic):

* ``friedman1`` regression — the irreducible RMSE is 1.0, so the gap between
  the confirmed model's holdout RMSE and 1.0 measures how much the search left
  on the table;
* ``interaction`` classification — the Bayes-optimal log loss is computed from
  the true probabilities on the holdout rows;
* ``null_regression`` — no relation between features and target; every
  promotion is a false discovery;
* optionally a planted cross-sectional panel and its null twin (``panel=True``),
  where ML predictions are scored by Alpha Lab exactly like FASTEXPR alphas.

These measure the search and its selection control on synthetic tasks. They
are not evidence about any market or real dataset.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from autoresearch.lab.runtime import requested_threads


def tabular_config(name: str, table: str, root: Path, *, rounds: int, seed: int, proposals: int) -> dict[str, Any]:
    return {"name": name, "domain": "ml", "seed": seed, "research_root": str(root / "records"),
            "runtime_root": str(root / "runtime"), "lineage_root": str(root / "lineage"),
            "settings": {"task": "tabular", "table": "builtin:" + table, "rows": 4000},
            "limits": {"max_rounds": rounds, "proposals_per_round": proposals, "validations_per_round": 2,
                       "max_validation_trials": 24, "population_size": 16}}


def panel_config(name: str, dataset: Path, root: Path, *, symbols: int, rounds: int, seed: int, proposals: int) -> dict[str, Any]:
    from autoresearch.domains.alpha.benchmark import benchmark_config

    alpha = benchmark_config(name, dataset, root, symbols=symbols, rounds=rounds, seed=seed, proposals=proposals)
    settings = {"task": "panel", **alpha["settings"], "max_train_rows": 120_000}
    return {"name": name, "domain": "ml", "seed": seed, "research_root": alpha["research_root"],
            "runtime_root": alpha["runtime_root"], "lineage_root": alpha["lineage_root"], "settings": settings,
            "limits": {"max_rounds": rounds, "proposals_per_round": proposals, "validations_per_round": 2,
                       "max_validation_trials": 16, "population_size": 12}}


def _run(config: dict[str, Any], rounds: int) -> dict[str, Any]:
    from autoresearch.lab.campaign import LabCampaign
    from autoresearch.lab.report import summary

    campaign = LabCampaign(config)
    for _ in range(rounds + 1):
        if campaign.tick()["status"] != "round":
            break
    finals = campaign.finalize() if campaign.checkpoint["accepted"] else []
    report = summary(campaign)
    best = next((row for row in report["leaderboard"] if row["decision"] == "promote"), None)
    out: dict[str, Any] = {"campaign_id": campaign.campaign_id, "funnel": report["funnel"], "ledger_ok": report["ledger"]["ok"],
           "generators": report["generators"], "best_promoted": best,
           "finals": [{"decision": f["decision"], "holdout": f["holdout"]} for f in finals]}
    campaign.domain.release()
    return out


def run_benchmark(root: str | Path, *, rounds: int = 8, seed: int = 5, proposals: int = 10, panel: bool = False,
                  symbols: int = 200, days: int = 1600) -> dict[str, Any]:
    from autoresearch.lab.campaign import LAB_VERSION

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    report: dict[str, Any] = {"kind": "ml_benchmark", "lab_version": LAB_VERSION,
              "settings": {"rounds": rounds, "proposals_per_round": proposals, "seed": seed},
              "note": "Synthetic tasks with known answers; no market or real-data evidence.", "tasks": {}}
    for table in ("friedman1", "interaction", "null_regression"):
        report["tasks"][table] = _run(tabular_config(f"ml-benchmark-{table}", table, root / table,
                                                     rounds=rounds, seed=seed, proposals=proposals), rounds)
    if panel:
        from autoresearch.domains.alpha.planted import create_planted_dataset

        for label, strength in (("panel_planted", None), ("panel_null", {})):
            dataset = create_planted_dataset(root / f"{label}-data", symbols=symbols, days=days, strength=strength)
            report["tasks"][label] = _run(panel_config(f"ml-benchmark-{label}", dataset, root / label, symbols=symbols,
                                                       rounds=rounds, seed=seed, proposals=proposals), rounds)
    report["runtime_seconds"] = round(time.time() - started, 1)
    report["blas_threads"] = requested_threads()
    (root / "benchmark.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (root / "benchmark.md").write_text(render_markdown(report))
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Machine-learning domain benchmark", "",
             "Synthetic tasks with known answers; no market or real-data evidence.", "",
             "| Task | Screened | Validated | Promoted | Confirmed finals | Top validated recipe | Best confirmed holdout |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, task in report["tasks"].items():
        funnel = task["funnel"]
        best = task["best_promoted"]
        recipe = best["title"] if best else "-"
        holdout = "-"
        if task["finals"]:
            confirmed = [f["holdout"] for f in task["finals"] if f["decision"] == "confirmed"] or \
                [f["holdout"] for f in task["finals"]]
            if "score" in confirmed[0]:
                first = min(confirmed, key=lambda h: h["score"])          # losses: lower is better
            else:
                first = max(confirmed, key=lambda h: h.get("sharpe") or float("-inf"))
            if "score" in first:
                holdout = f"{first.get('metric', 'score')} {first['score']:.4f} vs baseline {first['baseline']:.4f}"
                if first.get("noise_floor") is not None:
                    holdout += f" (noise floor {first['noise_floor']:.2f})"
                if first.get("bayes_optimal_score") is not None:
                    holdout += f" (Bayes {first['bayes_optimal_score']:.4f})"
            elif "sharpe" in first:
                holdout = f"Sharpe {first['sharpe']:.2f} over {first.get('days')} sessions"
        confirmed_count = sum(1 for f in task["finals"] if f["decision"] == "confirmed")
        lines.append(f"| {name} | {funnel['screened']} | {funnel['validated']} | {funnel['promoted']} | "
                     f"{confirmed_count}/{len(task['finals'])} | {recipe} | {holdout} |")
    threads = report.get("blas_threads")
    lines += ["", f"Lab version {report['lab_version']}, {report['runtime_seconds']} s"
              + (f" with {threads} BLAS thread(s)" if threads else "") + ".", ""]
    return "\n".join(lines)
