"""Opt-in bounded Stage-3 validation, never data acquisition or live trading.

Run with env/qlib Python and an explicit prepared snapshot. Creates one saved
backtest beside that snapshot and optional reports, leaving all inputs unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qlib_yahoo.dataset import sha256
from qlib_yahoo.report_service import ensure_report
from qlib_yahoo.runtime import run_signed_backtest


def tree_identity(root):
    return {p.relative_to(root).as_posix(): sha256(p)
            for p in root.rglob("*") if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    metadata = json.loads((dataset / "manifest.json").read_text())
    before = tree_identity(dataset)
    source_before = {name: sha256(Path(name)) for name in metadata.get("raw_sha256", {})}
    latest = dataset.parent.parent / "latest.json"
    latest_before = latest.read_bytes() if latest.exists() else None
    alpha = "$close/Ref($close,1)-1"  # Generic mechanics probe, not a recommended strategy.
    run = run_signed_backtest(dataset, alpha, holdings="never")
    import numpy as np
    import pandas as pd
    manifest = json.loads((run / "manifest.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    daily = pd.read_csv(run / "daily_report.csv")
    from qlib_yahoo.analysis import read_instrument_table, table_artifact

    positions = read_instrument_table(table_artifact(run, "positions"))
    weights = read_instrument_table(table_artifact(run, "signed_weights"))
    scores = read_instrument_table(table_artifact(run, "alpha_scores"))
    joined = scores.merge(weights, on=["datetime", "instrument"], validate="one_to_one")
    denominator = joined.groupby("datetime")["score"].transform(lambda x: x.abs().sum())
    # The upstream signal adapter stores float32; signed normalization rounds
    # conservatively toward the exposure cap at that precision.
    np.testing.assert_allclose(joined["weight"], joined["score"] / denominator, rtol=3e-7, atol=1e-12)
    assert manifest["schema_version"] in (5, 6) and manifest["status"] == "completed"
    assert "benchmark_return" not in daily
    assert set(summary["metrics"]) == {"strategy_net"}
    assert (positions.amount < 0).any() and (positions.amount > 0).any()
    np.testing.assert_allclose(daily.account, daily.cash_total + daily.long_value - daily.short_liability)
    np.testing.assert_allclose(daily.cost_dollars, daily.transaction_cost_dollars + daily.borrow_cost_dollars)
    gaps = pd.to_datetime(daily.datetime).diff().dt.days.fillna(0)
    expected_borrow = daily.short_liability.shift(1).fillna(0) * .004 * gaps / 365
    np.testing.assert_allclose(daily.borrow_cost_dollars, expected_borrow, rtol=1e-9, atol=1e-8)
    core_before = {name: sha256(run / name) for name in manifest["artifacts_sha256"]}
    reports = {}
    for profile in ("holdings", "overview", "detailed"):
        path, reused = ensure_report(run, profile)
        again, reused_again = ensure_report(run, profile)
        assert path == again and reused_again
        reports[profile] = str(path)
    assert core_before == {name: sha256(run / name) for name in core_before}
    assert before == tree_identity(dataset)
    assert source_before == {name: sha256(Path(name)) for name in source_before}
    assert latest_before == (latest.read_bytes() if latest.exists() else None)
    evidence = {
        "dataset": str(dataset), "run": str(run), "native_console": False,
        "sessions": len(daily), "start": daily.datetime.iloc[0], "end": daily.datetime.iloc[-1],
        "core_bytes": sum((run / name).stat().st_size for name in core_before) + (run / "manifest.json").stat().st_size,
        "transaction_cost": float(daily.transaction_cost_dollars.sum()),
        "borrow_cost": float(daily.borrow_cost_dollars.sum()),
        "both_position_signs": True, "raw_proportions_verified": True,
        "calendar_gap_borrow_verified": True, "inputs_unchanged": True,
        "source_files_checked": len(source_before), "reports_reused": reports,
        "engine": manifest["signed_engine_identity"], "funding_state": summary["funding_state"],
        "dataset_tree_digest": hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
    }
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
