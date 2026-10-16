"""Bounded signed-worker smoke on synthetic bars; no market data or UI required.

Run after install_fast_runtime.py. Outputs live in a temporary folder and are
removed on success or failure. This checks execution, not trading readiness or
runtime performance. An omitted runtime_profile deliberately tests the default.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import struct
import sys
import tempfile
from datetime import date, timedelta
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def synthetic_snapshot(root: Path, sessions: int = 45) -> Path:
    """Write the Qlib float32 daily format, including a verified tree digest."""
    from qlib_yahoo.dataset import tree_sha256

    if sessions < 25:
        raise ValueError("smoke requires at least 25 sessions")
    snapshot = root / "datasets" / "synthetic-smoke"
    bins = snapshot / "qlib_data"
    dates = []
    day = date(2024, 1, 2)
    while len(dates) < sessions:
        if day.weekday() < 5:
            dates.append(day.isoformat())
        day += timedelta(days=1)
    symbols = [f"TST{i}" for i in range(6)]
    for folder in ("calendars", "instruments", "features"):
        (bins / folder).mkdir(parents=True)
    calendar = "".join(f"{day}\n" for day in dates)
    for name in ("day.txt", "day_future.txt"):
        (bins / "calendars" / name).write_text(calendar, encoding="utf-8")
    (bins / "instruments" / "all.txt").write_text(
        "".join(f"{code}\t{dates[0]}\t{dates[-1]}\n" for code in symbols), encoding="utf-8")
    for i, code in enumerate(symbols):
        folder = bins / "features" / code.lower()
        folder.mkdir()
        close = [30 + 3 * i + .025 * n * (i + 1) + .1 * math.sin(n / 5 + i)
                 for n in range(sessions)]
        fields = {"close": close,
                  "open": [p * (1 + .001 * math.sin(n + i)) for n, p in enumerate(close)],
                  "high": [p * 1.01 for p in close], "low": [p * .99 for p in close],
                  "volume": [100000 + 1000 * i + n for n in range(sessions)],
                  "factor": [1.] * sessions}
        for field, values in fields.items():
            (folder / f"{field}.day.bin").write_bytes(struct.pack(f"<{sessions + 1}f", 0., *values))
    digest, _ = tree_sha256(bins)
    (snapshot / "manifest.json").write_text(json.dumps({
        "symbols": symbols, "qlib_binary_tree_sha256": digest,
        "limitations": ["Deterministic synthetic smoke bars; not market data."],
    }), encoding="utf-8")
    return snapshot


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = set()
        self.frames = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.tags.add(tag)
        if tag == "iframe":
            self.frames.append(dict(attrs).get("src"))
        if tag == "a":
            self.links.append(dict(attrs).get("href"))


def check_html_report(run: Path) -> int:
    reports = list((run / "reports").glob("*/report_manifest.json"))
    require(bool(reports), "worker completed simulation without an HTML report")
    charts_checked = 0
    for manifest_path in reports:
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(record["status"] == "completed", "worker report rendering failed")
        index = manifest_path.parent / "report_index.html"
        parser = ReportParser()
        parser.feed(index.read_text(encoding="utf-8"))
        require({"html", "body", "table"}.issubset(parser.tags), "report is incomplete HTML")
        require(bool(record["charts"]), "overview has no charts")
        for name in record["charts"]:
            require(name in parser.frames or name in parser.links, "chart is absent from report index")
            chart = manifest_path.parent / name
            chart_parser = ReportParser()
            chart_parser.feed(chart.read_text(encoding="utf-8"))
            require("script" in chart_parser.tags, "chart has no Plotly script")
            charts_checked += 1
    return charts_checked


def check_default_profile(profile: dict, platform: str) -> None:
    require(profile.get("name") == "accepted-v1", "default signed run did not select the accepted profile")
    if platform == "darwin":
        require(profile.get("platform") == "macos", "default profile misidentified macOS")
        require(profile.get("engine") == "native-signed" and profile.get("ordered_session") is False,
                "macOS selected the Windows ordered-session engine")
        require(profile.get("gzip") == "stdlib-zlib" and profile.get("date_range") == "python",
                "macOS selected incompatible native helpers")
        require(bool(profile.get("fallback_reason")), "macOS native fallback lacks provenance")


def validate_run(run: Path) -> dict:
    import numpy as np
    from qlib_yahoo.analysis import validate_bundle, read_instrument_table, table_artifact
    from qlib_yahoo.runtime import PIN, platform_pin

    bundle = validate_bundle(run)  # hash, metric and signed-ledger reconciliation
    manifest, summary = bundle.manifest, bundle.summary
    require(manifest["status"] == "completed" and manifest["strategy_mode"] == "signed",
            "worker did not complete signed execution")
    check_default_profile(manifest["runtime_profile"], sys.platform)
    pin = platform_pin(json.loads(PIN.read_text(encoding="utf-8")))
    for key in ("version", "commit"):
        require(manifest["signed_engine_identity"][key] == pin[key], f"engine {key} differs from pin")
    require(set(summary["metrics"]) == {"strategy_net"}, "unexpected metric series")
    require(summary["metric_contract"]["version"] == "daily-simple-strategy-net-v1",
            "unexpected metric contract")
    require(summary["backtest"]["cash"] == 100_000, "batch account did not start with fresh capital")
    daily = bundle.frame
    positions = read_instrument_table(table_artifact(run, "positions"))
    require((positions.amount < 0).any() and (positions.amount > 0).any(),
            "synthetic alpha must execute both position signs")
    require(len(daily) >= 20, "worker omitted evaluation sessions")
    require(daily.metrics_defined.all() and daily.funding_feasible.all(), "smoke account failed")
    np.testing.assert_allclose(daily.account, daily.cash_total + daily.long_value - daily.short_liability)
    np.testing.assert_allclose(daily.cost_dollars, daily.transaction_cost_dollars + daily.borrow_cost_dollars)
    require(daily.borrow_cost_dollars.sum() > 0 and daily.transaction_cost_dollars.sum() > 0,
            "signed execution did not charge costs")
    with gzip.open(run / "signed_decisions.json.gz", "rt", encoding="utf-8") as stream:
        require(bool(json.load(stream)), "signed decision history is empty")
    charts = check_html_report(run)
    bundle.assert_unchanged()
    return {"sessions": len(daily), "charts": charts, "engine": manifest["signed_engine_identity"],
            "runtime_profile": manifest["runtime_profile"]}


def run_smoke(root: Path) -> dict:
    from qlib_yahoo.dataset import tree_sha256
    from qlib_yahoo.jobs import Job

    snapshot = synthetic_snapshot(root)
    before = tree_sha256(snapshot)
    options = {"dataset": str(snapshot), "strategy_mode": "signed", "warmup_sessions": 20,
               "report_level": "overview", "holdings": "never"}
    single = Job({"kind": "backtest", "alpha": "$close-40", **options}).run(print)
    results = [validate_run(single)]
    receipt = Job({"kind": "backtest-batch", "data_root": str(root),
                   "alphas": ["$close-40", "$open-40"], **options}).run(print)
    record = json.loads(receipt.read_text(encoding="utf-8"))
    require(record["status"] == "completed" and not record["errors"], "batch has failed alphas")
    require(len(record["runs"]) == 2 and len(set(record["runs"])) == 2, "batch did not create independent runs")
    require(record["preparation"]["fresh_account_per_run"], "batch did not use fresh accounts")
    for path in record["runs"]:
        results.append(validate_run(Path(path)))
    require((single / "daily_report.csv").read_bytes()
            == (Path(record["runs"][0]) / "daily_report.csv").read_bytes(),
            "same alpha changed its daily ledger in the reused-preparation batch")
    require(before == tree_sha256(snapshot), "worker changed synthetic input data")
    return {"status": "passed", "worker_runs": len(results), "runs": results, "inputs_unchanged": True}


def main() -> int:
    # Redirected Windows streams may default to CP1252 even though workers emit
    # UTF-8 progress bars. Keep diagnostic capture from aborting the smoke run.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="qlib-signed-smoke-") as temporary:
        evidence = run_smoke(Path(temporary).resolve())
        print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
