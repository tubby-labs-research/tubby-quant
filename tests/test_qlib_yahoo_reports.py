"""Small artifact-only replay checks; no Qlib engine or network required."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qlib_yahoo.analysis import _chart, _group_returns, _score_turnover, render_report
from qlib_yahoo.metrics import portfolio_metrics


@unittest.skipUnless(
    importlib.util.find_spec("pandas") and importlib.util.find_spec("plotly"),
    "pandas and plotly are optional outside the Qlib environment",
)
class ReplayTests(unittest.TestCase):
    def bundle(self, root, *, pairs=False):
        import pandas as pd

        root = Path(root)
        summary = {
            "metric_contract": {
                "version": "daily-simple-net-v1", "periods_per_year": 252,
                "annual_cash_rate": 0.0,
                "risk_free_convention": "effective annual rate converted per trading session",
                "brain_equivalent": False,
            },
            "metrics": portfolio_metrics([0.001, 0.002], [0.0005, 0.001]),
            "backtest": {"start": "2024-01-02", "end": "2024-01-03", "cash": 100.0},
        }
        (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        pd.DataFrame({
            "datetime": ["2024-01-02", "2024-01-03"],
            "account": [100.1, 100.3002], "net_return": [0.001, 0.002],
            "benchmark_return": [0.0005, 0.001], "cost": [0.0001, 0.0001],
            "cost_dollars": [0.01, 0.01], "turnover": [0.3, 0.2],
            "cash": [20.0, 20.0], "holdings_value": [80.1, 80.3002],
        }).to_csv(root / "daily_report.csv", index=False)
        pd.DataFrame({
            "datetime": ["2024-01-02", "2024-01-03"], "instrument": ["AAA", "AAA"],
            "amount": [8, 8], "price": [80.1 / 8, 80.3002 / 8], "value": [80.1, 80.3002],
            "weight": [80.1 / 100.1, 80.3002 / 100.3002], "cash": [20, 20], "account": [100.1, 100.3002],
        }).to_csv(root / "positions.csv.gz", index=False, compression="gzip")
        files = ["summary.json", "daily_report.csv", "positions.csv.gz"]
        if pairs:
            pd.DataFrame({
                "datetime": ["2024-01-02"] * 4, "instrument": list("ABCD"),
                "score": [1, 2, 3, 4], "label": [0.1, 0.2, 0.3, 0.4],
            }).to_csv(root / "signal_forward_labels.csv.gz", index=False, compression="gzip")
            files.append("signal_forward_labels.csv.gz")
        hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}
        (root / "manifest.json").write_text(json.dumps({
            "schema_version": 3, "status": "completed", "artifacts_sha256": hashes,
        }), encoding="utf-8")
        return {name: (root / name).read_bytes() for name in [*files, "manifest.json"]}

    def reseal(self, root):
        import pandas as pd

        root = Path(root)
        daily = pd.read_csv(root / "daily_report.csv")
        summary = json.loads((root / "summary.json").read_text())
        summary["metrics"] = portfolio_metrics(daily["net_return"], daily["benchmark_return"])
        summary["backtest"]["end"] = daily["datetime"].iloc[-1]
        (root / "summary.json").write_text(json.dumps(summary))
        manifest = json.loads((root / "manifest.json").read_text())
        for name in manifest["artifacts_sha256"]:
            manifest["artifacts_sha256"][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
        (root / "manifest.json").write_text(json.dumps(manifest))

    def test_summary_gate_boundary_and_manual_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            originals = self.bundle(root)
            sharpe = json.loads((root / "summary.json").read_text())["metrics"]["strategy_net"]["sharpe"]
            self.assertEqual(render_report(root, level="summary"), root / "summary.json")
            self.assertFalse((root / "reports").exists())
            self.assertEqual(render_report(root, level="summary", threshold=sharpe, min_observations=2), root / "summary.json")
            index = render_report(root, level="summary", holdings="always", threshold=sharpe, min_observations=2)
            self.assertTrue((index.parent / "holdings.html").exists())
            status = json.loads((index.parent / "report_manifest.json").read_text())
            self.assertFalse(status["holdings_gate_eligible"])
            self.assertTrue(status["holdings_rendered"])
            self.assertEqual(status["threshold"], sharpe)
            self.assertEqual(status["min_observations"], 2)
            for name, contents in originals.items():
                self.assertEqual((root / name).read_bytes(), contents)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.bundle(root)
            self.assertEqual(render_report(root, level="summary", min_observations=3), root / "summary.json")
            self.assertEqual(render_report(root, level="summary", holdings="never"), root / "summary.json")
            self.assertFalse((root / "reports").exists())

    def test_holdings_sale_gap_rebuy_is_zero_on_missing_session(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.bundle(root)
            daily = pd.read_csv(root / "daily_report.csv")
            daily.loc[1, "datetime"] = "2024-01-04"
            gap = daily.iloc[[0]].copy()
            gap.loc[:, "datetime"] = "2024-01-03"
            gap.loc[:, "account"] = 100.2
            gap.loc[:, "cash"] = 100.2
            gap.loc[:, "holdings_value"] = 0.0
            daily = pd.concat([daily.iloc[[0]], gap, daily.iloc[[1]]], ignore_index=True)
            daily["net_return"] = daily["account"] / daily["account"].shift(1).fillna(100.0) - 1
            daily.to_csv(root / "daily_report.csv", index=False)
            positions = pd.read_csv(root / "positions.csv.gz")
            positions.loc[1, "datetime"] = "2024-01-04"
            cash_only = positions.iloc[[0]].copy()
            cash_only.loc[:, "datetime"] = "2024-01-03"
            cash_only.loc[:, "instrument"] = ""
            cash_only.loc[:, "weight"] = 0.0
            cash_only["cash"] = cash_only["cash"].astype(float)
            cash_only.loc[:, "cash"] = 100.2
            cash_only.loc[:, "account"] = 100.2
            positions = pd.concat([positions.iloc[[0]], cash_only, positions.iloc[[1]]], ignore_index=True)
            positions.to_csv(root / "positions.csv.gz", index=False, compression="gzip")
            self.reseal(root)
            captured = []

            def capture(fig, directory, name):
                if name == "holdings.html":
                    captured.append(fig)
                return _chart(fig, directory, name)

            with patch("qlib_yahoo.analysis._chart", side_effect=capture):
                render_report(root, level="summary", holdings="always")
            stock = next(trace for trace in captured[0].data if trace.name == "AAA")
            self.assertEqual(list(stock.y), [80.1 / 100.1, 0.0, 80.3002 / 100.3002])

    def test_detailed_rolling_risk_without_signal_labels(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.bundle(root)
            daily = pd.read_csv(root / "daily_report.csv")
            dates = pd.bdate_range("2024-01-02", periods=253)
            daily = pd.concat([daily.iloc[[0]]] * len(dates), ignore_index=True)
            daily["datetime"] = dates
            daily["net_return"] = [0.01 if index % 2 else -0.005 for index in range(len(dates))]
            account = 100.0
            accounts = []
            for daily_return in daily["net_return"]:
                account *= 1 + daily_return
                accounts.append(account)
            daily["account"] = accounts
            daily["cash"] = daily["account"] * 0.2
            daily["holdings_value"] = daily["account"] * 0.8
            daily.to_csv(root / "daily_report.csv", index=False)
            self.reseal(root)
            index = render_report(root, level="detailed", holdings="never")
            self.assertTrue((index.parent / "rolling_risk.html").exists())
            self.assertIn("Signal diagnostics unavailable", index.read_text(encoding="utf-8"))

    def test_overview_is_net_only_offline_shared_js_and_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.bundle(root)
            with patch.dict("sys.modules", {"qlib": None, "qlib.contrib": None}):
                first = render_report(root, min_observations=2)
                second = render_report(root, min_observations=2)
            self.assertNotEqual(first.parent, second.parent)
            self.assertEqual(len(list(first.parent.glob("plotly.min.js"))), 1)
            report_manifest = json.loads((first.parent / "report_manifest.json").read_text())
            self.assertEqual(report_manifest["core_manifest_sha256"], hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest())
            self.assertEqual(report_manifest["renderer_source_sha256"], hashlib.sha256(Path(render_report.__code__.co_filename).read_bytes()).hexdigest())
            self.assertTrue(report_manifest["plotly_version"])
            self.assertEqual(report_manifest["threshold"], 1.3)
            self.assertEqual(report_manifest["min_observations"], 2)
            page = first.read_text(encoding="utf-8")
            self.assertIn("Cash weight", (first.parent / "holdings.html").read_text(encoding="utf-8"))
            self.assertIn("Net account", (first.parent / "account.html").read_text(encoding="utf-8"))
            self.assertNotIn("strategy_gross", page)
            self.assertNotIn("gross Sharpe", page)
            self.assertTrue(all("plotly.min.js" in p.read_text(encoding="utf-8") for p in first.parent.glob("*.html") if p.name != "report_index.html"))
            self.assertEqual(json.loads((first.parent / "report_manifest.json").read_text())["status"], "completed")

    def test_partial_diagnostics_do_not_fail_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.bundle(root, pairs=True)
            index = render_report(root, level="detailed", holdings="never")
            page = index.read_text(encoding="utf-8")
            self.assertIn("Signal diagnostics unavailable", page)
            self.assertFalse((index.parent / "signal_ic.html").exists())

    def test_hashed_but_inconsistent_summary_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.bundle(root)
            summary = json.loads((root / "summary.json").read_text())
            summary["metrics"]["strategy_net"]["sharpe"] += 0.5
            (root / "summary.json").write_text(json.dumps(summary))
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["artifacts_sha256"]["summary.json"] = hashlib.sha256((root / "summary.json").read_bytes()).hexdigest()
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "sharpe disagrees"):
                render_report(root, level="summary", min_observations=2)
            failure = next((root / "reports").glob("*/report_manifest.json"))
            self.assertEqual(json.loads(failure.read_text())["status"], "failed")

    def test_account_and_saved_net_return_must_reconcile(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.bundle(root)
            daily = pd.read_csv(root / "daily_report.csv")
            daily.loc[1, "account"] += 1.0
            daily.to_csv(root / "daily_report.csv", index=False)
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["artifacts_sha256"]["daily_report.csv"] = hashlib.sha256((root / "daily_report.csv").read_bytes()).hexdigest()
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "does not reconcile"):
                render_report(root)

    def test_tamper_and_legacy_rejected_with_failure_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.bundle(root)
            (root / "daily_report.csv").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                render_report(root)
            failed = list((root / "reports").glob("*/report_manifest.json"))
            self.assertEqual(len(failed), 1)
            self.assertEqual(json.loads(failed[0].read_text())["status"], "failed")
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["schema_version"] = 2
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "schema_version 3"):
                render_report(root)

    def test_render_failure_is_independent_of_core(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            originals = self.bundle(root)
            with patch("qlib_yahoo.analysis._chart", side_effect=RuntimeError("chart failed")):
                with self.assertRaisesRegex(RuntimeError, "chart failed"):
                    render_report(root)
            failed = next((root / "reports").glob("*/report_manifest.json"))
            failure = json.loads(failed.read_text())
            self.assertEqual(failure["status"], "failed")
            self.assertEqual(failure["threshold"], 1.3)
            self.assertEqual(failure["min_observations"], 252)
            self.assertEqual(failure["level"], "overview")
            self.assertEqual(failure["holdings"], "auto")
            self.assertEqual(failure["core_manifest_sha256"], hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest())
            self.assertEqual(failure["renderer_source_sha256"], hashlib.sha256(Path(render_report.__code__.co_filename).read_bytes()).hexdigest())
            for name, contents in originals.items():
                self.assertEqual((root / name).read_bytes(), contents)

    def test_score_groups_include_worst_remainder_and_missing_session_turnover(self):
        import pandas as pd

        date = pd.Timestamp("2024-01-02")
        idx = pd.MultiIndex.from_product([[date], list("ABCDEFGHIJKL")], names=["datetime", "instrument"])
        frame = pd.DataFrame({"score": range(12), "label": range(12)}, index=idx)
        groups = _group_returns(frame)
        self.assertEqual(groups.iloc[0]["group_5"], 0.5)
        self.assertEqual(groups.iloc[0]["eligible_count"], 12)
        sessions = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        idx = pd.MultiIndex.from_tuples(
            [(sessions[0], c) for c in "ABCDEFG"] + [(sessions[1], c) for c in "ABCDEF"] + [(sessions[2], c) for c in "ABCDEFG"],
            names=["datetime", "instrument"],
        )
        scores = pd.DataFrame({"score": list(range(6)) + [100] + list(range(6)) + list(range(6)) + [100]}, index=idx)
        turnover = _score_turnover(scores)
        self.assertEqual(len(turnover), 2)
        self.assertEqual(turnover.iloc[-1]["top_score_bucket_turnover"], 0)


if __name__ == "__main__":
    unittest.main()
