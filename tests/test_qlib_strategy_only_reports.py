"""Offline schema-5 replay and report compatibility checks."""

import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

from qlib_yahoo.analysis import render_report
from qlib_yahoo.metrics import portfolio_metrics
from qlib_yahoo.report_service import ensure_report, validate_run_selection


@unittest.skipUnless(
    importlib.util.find_spec("pandas") and importlib.util.find_spec("plotly"),
    "pandas and plotly are optional outside the Qlib environment",
)
class StrategyOnlyReports(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def reseal(self, *, mode="long-only", schema=5):
        names = ("summary.json", "daily_report.csv", "positions.csv.gz")
        manifest = {"schema_version": schema, "status": "completed",
                    "artifacts_sha256": {name: hashlib.sha256((self.root / name).read_bytes()).hexdigest()
                                         for name in names}}
        if schema == 5:
            manifest["strategy_mode"] = mode
        (self.root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def bundle(self, *, mode="long-only", insolvent=False, legacy=False):
        import pandas as pd

        dates = ["2024-01-02", "2024-01-03"]
        if mode == "long-only":
            accounts = [101.0, 100.495]
            returns = [0.01, -0.005]
            daily = pd.DataFrame({
                "datetime": dates, "account": accounts, "net_return": returns,
                "cost": [0.001, 0.001], "cost_dollars": [0.1, 0.101],
                "turnover": [0.5, 0.2], "cash": [21.0, 20.495], "holdings_value": [80.0, 80.0],
            })
            positions = pd.DataFrame({
                "datetime": dates, "instrument": ["AAA", "AAA"], "amount": [8.0, 8.0],
                "price": [10.0, 10.0], "value": [80.0, 80.0],
                "weight": [80.0 / value for value in accounts],
                "cash": [21.0, 20.495], "account": accounts,
            })
        else:
            second_price = 230.0 if insolvent else 110.0
            free = [99.95, -0.05 if insolvent else 99.95]
            accounts = [99.95, free[1] + 100.0 - second_price]
            returns = [accounts[0] / 100 - 1,
                       float("nan") if insolvent else accounts[1] / accounts[0] - 1]
            borrow = [0.0, 100.0 if insolvent else 0.0]
            daily = pd.DataFrame({
                "datetime": dates, "account": accounts, "net_return": returns,
                "cost": [0.0005, float("nan") if insolvent else 0.0],
                "cost_dollars": [0.05, borrow[1]],
                "turnover": [1.0, float("nan") if insolvent else 0.0],
                "cash": free, "holdings_value": [-100.0, -second_price],
                "free_cash": free, "cash_total": [free[0] + 100, free[1] + 100],
                "restricted_cash": [100.0, 100.0], "long_value": [0.0, 0.0],
                "short_liability": [100.0, second_price],
                "transaction_cost_dollars": [0.05, 0.0], "borrow_cost_dollars": borrow,
                "total_cost_dollars": [0.05, borrow[1]],
                "unfunded_cash_shortfall": [0.0, max(-free[1], 0.0)],
                "funding_feasible": [True, not insolvent],
                "metrics_defined": [True, not insolvent],
                "stale_marks": [False, False], "stale_mark_symbols": ["[]", "[]"],
            })
            positions = pd.DataFrame({
                "datetime": dates, "instrument": ["SHORT", "SHORT"],
                "amount": [-1.0, -1.0], "price": [100.0, second_price],
                "value": [-100.0, -second_price],
                "weight": [-100.0 / accounts[0], float("nan") if insolvent else -second_price / accounts[1]],
                "cash": free, "account": accounts,
            })
            for column in ("cash_total", "free_cash", "restricted_cash", "long_value", "short_liability",
                           "transaction_cost_dollars", "borrow_cost_dollars", "total_cost_dollars",
                           "unfunded_cash_shortfall", "funding_feasible", "metrics_defined", "stale_marks",
                           "stale_mark_symbols"):
                positions[column] = daily[column]
        if legacy:
            daily["benchmark_return"] = [0.001, 0.002]
            metrics = portfolio_metrics([0.0, 0.0] if insolvent else returns, daily["benchmark_return"])
            if insolvent:
                for family in ("strategy_net", "excess_net_vs_benchmark"):
                    metrics[family] = {key: 2 if key == "observations" else None for key in metrics[family]}
        else:
            metrics = portfolio_metrics([] if insolvent else returns, None)
            if insolvent:
                metrics["strategy_net"] = {key: 2 if key == "observations" else None
                                           for key in metrics["strategy_net"]}
        summary = {
            "metric_contract": {"version": ("daily-simple-signed-net-v1" if mode == "signed" else "daily-simple-net-v1")
                                if legacy else "daily-simple-strategy-net-v1",
                                "periods_per_year": 252, "annual_cash_rate": 0.0},
            "metrics": metrics,
            "backtest": {"start": dates[0], "end": dates[1], "cash": 100.0,
                         **({} if legacy else {"strategy_mode": mode,
                                                "execution_contract": {"benchmark": None, "deal_price": "open"}})},
        }
        if mode == "signed":
            summary["metrics_status"] = {"strategy": "unavailable" if insolvent else "available",
                                         "reason": "nonpositive_equity" if insolvent else None,
                                         "undefined_observations": 1 if insolvent else 0}
            summary["funding_state"] = {"feasible_all_sessions": not insolvent,
                                        "unfunded_sessions": 1 if insolvent else 0,
                                        "max_unfunded_cash_shortfall": 0.05 if insolvent else 0.0,
                                        "stale_mark_sessions": 0}
        daily.to_csv(self.root / "daily_report.csv", index=False)
        positions.to_csv(self.root / "positions.csv.gz", index=False, compression="gzip")
        (self.root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        self.reseal(mode=mode, schema=(4 if mode == "signed" else 3) if legacy else 5)

    def test_schema5_both_modes_render_without_benchmark(self):
        for mode in ("long-only", "signed"):
            with self.subTest(mode=mode):
                self.bundle(mode=mode)
                self.assertEqual(validate_run_selection(self.root), self.root.resolve())
                index, reused = ensure_report(self.root, "overview")
                self.assertFalse(reused)
                page = index.read_text(encoding="utf-8")
                self.assertIn("Net Sharpe", page)
                self.assertIn("Equity and drawdown", page)
                self.assertNotIn("benchmark", page.lower())
                self.assertNotIn("benchmark", (index.parent / "account.html").read_text(encoding="utf-8").lower())
                self.assertNotIn("benchmark_return", (self.root / "daily_report.csv").read_text(encoding="utf-8"))
                self.assertEqual(set(json.loads((self.root / "summary.json").read_text())["metrics"]), {"strategy_net"})
                self.assertEqual(ensure_report(self.root, "overview"), (index, True))

    def test_signed_insolvency_preserves_unavailable_metrics_and_gate(self):
        self.bundle(mode="signed", insolvent=True)
        self.assertEqual(validate_run_selection(self.root), self.root.resolve())
        index = render_report(self.root, level="overview", holdings="auto")
        status = json.loads((index.parent / "report_manifest.json").read_text())
        self.assertFalse(status["holdings_gate_eligible"])
        self.assertIn("Unavailable", index.read_text(encoding="utf-8"))
        self.assertNotIn("benchmark", index.read_text(encoding="utf-8").lower())
        self.assertTrue(math.isnan(__import__("pandas").read_csv(self.root / "daily_report.csv").loc[1, "net_return"]))

    def test_holdings_only_has_no_empty_overview_sections(self):
        self.bundle()
        index = render_report(self.root, level="summary", holdings="always")
        page = index.read_text(encoding="utf-8")
        self.assertIn("Holdings and diagnostics", page)
        self.assertNotIn("Equity and drawdown", page)
        self.assertNotIn("Costs and turnover", page)

    def test_legacy_schemas_still_replay_benchmark(self):
        for mode in ("long-only", "signed"):
            with self.subTest(mode=mode):
                self.bundle(mode=mode, legacy=True)
                self.assertEqual(validate_run_selection(self.root), self.root.resolve())
                index = render_report(self.root, level="overview", holdings="never")
                self.assertIn("Benchmark reference", (index.parent / "account.html").read_text(encoding="utf-8"))
                self.assertIn("benchmark_reference", index.read_text(encoding="utf-8"))

    def test_schema5_strict_metadata_and_escaped_settings(self):
        self.bundle()
        summary = json.loads((self.root / "summary.json").read_text())
        summary["backtest"]["start"] = '<script>alert("x")</script>'
        (self.root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        self.reseal()
        index = render_report(self.root, holdings="never")
        page = index.read_text(encoding="utf-8")
        self.assertIn("&lt;script&gt;", page)
        self.assertNotIn("<script>", page)
        for mutation in (lambda m, s: m.pop("strategy_mode"),
                         lambda m, s: s["backtest"].update(strategy_mode="signed"),
                         lambda m, s: s["metrics"].update(benchmark_reference={"observations": 2}),
                         lambda m, s: s["metric_contract"].update(version="daily-simple-net-v1")):
            self.bundle()
            manifest = json.loads((self.root / "manifest.json").read_text())
            summary = json.loads((self.root / "summary.json").read_text())
            mutation(manifest, summary)
            (self.root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            manifest["artifacts_sha256"]["summary.json"] = hashlib.sha256((self.root / "summary.json").read_bytes()).hexdigest()
            (self.root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_run_selection(self.root)


if __name__ == "__main__":
    unittest.main()
