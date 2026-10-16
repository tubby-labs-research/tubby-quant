"""Bounded full-run acceptance using the explicitly installed Windows derivative.

No download or installation is performed. Portable helper tests run separately.
"""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from qlib_yahoo.dataset import DatasetError
from qlib_yahoo.runtime import PIN, ROOT, platform_pin, run_signed_backtest


def _installed():
    try:
        pin = platform_pin(json.loads(PIN.read_text()))
        return (sys.version_info[:2] == (3, 12)
                and importlib.util.find_spec("pandas") is not None
                and (ROOT / pin["local_runtime"] / "qlib/_tubby_build.json").is_file())
    except (OSError, ValueError, KeyError, DatasetError):
        return False


@unittest.skipUnless(_installed(), "explicit local signed build for this platform not installed")
class SignedIntegrationTests(unittest.TestCase):
    def run_case(self, expression, **options):
        from test_qlib_yahoo_execution import _snapshot
        import pandas as pd
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        # Deliberately no benchmark bars: signed execution must not need them,
        # even when the older dataset manifest still names a benchmark.
        dataset = _snapshot(Path(temporary.name), sessions_count=45, include_benchmark=False)
        original = {str(path.relative_to(dataset)): path.read_bytes()
                    for path in dataset.rglob("*") if path.is_file()}
        run = run_signed_backtest(dataset, expression, warmup_sessions=20,
                                  holdings="never", **options)
        self.assertEqual(original, {str(path.relative_to(dataset)): path.read_bytes()
                                   for path in dataset.rglob("*") if path.is_file()})
        manifest = json.loads((run / "manifest.json").read_text())
        self.assertEqual(manifest["schema_version"], 6)
        self.assertEqual(manifest["strategy_mode"], "signed")
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(set(json.loads((run / "summary.json").read_text())["metrics"]), {"strategy_net"})
        self.assertNotIn("benchmark_return", pd.read_csv(run / "daily_report.csv").columns)
        # Reproducible gzip header: no file name flag, zero modification time.
        self.assertEqual((run / "signed_decisions.json.gz").read_bytes()[3:8], bytes(5))
        from qlib_yahoo.analysis import read_instrument_table, table_artifact

        return run, pd.read_csv(run / "daily_report.csv"), read_instrument_table(table_artifact(run, "positions"))

    def test_unbalanced_signed_run_replays_without_engine(self):
        from qlib_yahoo.report_service import ensure_report
        run, daily, positions = self.run_case("$close-45")
        self.assertTrue((positions.amount < 0).any())
        self.assertTrue((positions.amount > 0).any())
        self.assertGreater(daily.borrow_cost_dollars.sum(), 0)
        self.assertGreater(daily.transaction_cost_dollars.sum(), 0)
        for row in daily.itertuples():
            self.assertAlmostEqual(row.account, row.cash_total + row.long_value - row.short_liability, places=5)
            self.assertAlmostEqual(row.cost_dollars, row.transaction_cost_dollars + row.borrow_cost_dollars, places=5)
        report, reused = ensure_report(run, "holdings")
        self.assertFalse(reused)
        again, reused = ensure_report(run, "holdings")
        self.assertTrue(reused)
        self.assertEqual(again, report)

    def test_zero_alpha_cash_only_is_valid(self):
        _, daily, positions = self.run_case("$close-$close")
        self.assertTrue((daily.account == 100000).all())
        self.assertTrue((daily.cost_dollars == 0).all())
        self.assertTrue((positions.amount == 0).all())

    def test_all_negative_does_not_become_long_topk(self):
        _, daily, positions = self.run_case("0-$close", gross_exposure=.5)
        self.assertTrue((positions.amount < 0).any())
        self.assertFalse((positions.amount > 0).any())
        self.assertTrue((daily.long_value == 0).all())

    def test_insolvent_history_is_retained_and_replayable(self):
        from qlib_yahoo.report_service import ensure_report
        # Deliberately extreme synthetic expense, not a recommended borrow rate.
        run, daily, positions = self.run_case("0-$close", annual_borrow_rate=1000)
        self.assertTrue((daily.account <= 0).any())
        self.assertTrue((daily.unfunded_cash_shortfall > 0).any())
        self.assertTrue(daily.loc[~daily.metrics_defined, "net_return"].isna().all())
        self.assertTrue(daily.loc[~daily.metrics_defined, "turnover"].isna().all())
        self.assertTrue((positions.amount < 0).any())
        summary = json.loads((run / "summary.json").read_text())
        self.assertEqual(summary["metrics_status"]["strategy"], "unavailable")
        self.assertIsNone(summary["metrics"]["strategy_net"]["sharpe"])
        self.assertEqual(summary["metrics"]["strategy_net"]["observations"], len(daily))
        report, _ = ensure_report(run, "overview")
        self.assertTrue(report.is_file())


if __name__ == "__main__":
    unittest.main()
