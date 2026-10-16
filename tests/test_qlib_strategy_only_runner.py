"""No benchmark input, lookup or benchmark-relative metrics for new runs."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from qlib_yahoo.metrics import portfolio_metrics


class StrategyMetricTests(unittest.TestCase):
    def test_no_reference_needed_and_strategy_statistic_unchanged(self):
        returns = [.02, -.01, .005]
        actual = portfolio_metrics(returns, annual_cash_rate=.02)
        legacy = portfolio_metrics(returns, [.01, .01, -.01], annual_cash_rate=.02)
        self.assertEqual(set(actual), {"strategy_net"})
        self.assertEqual(actual["strategy_net"], legacy["strategy_net"])


@unittest.skipUnless(importlib.util.find_spec("qlib"), "optional Qlib environment")
class StrategyOnlyRunnerTests(unittest.TestCase):
    def test_legacy_snapshot_benchmark_is_not_requested(self):
        from tests.test_qlib_yahoo_execution import _snapshot
        from qlib_yahoo.backtest import run_backtest
        from qlib.data.data import BaseProvider
        from qlib.backtest.report import PortfolioMetrics
        requests = []
        original_features = BaseProvider.features
        original_benchmark = PortfolioMetrics._cal_benchmark

        def features(provider, instruments, *args, **kwargs):
            if isinstance(instruments, (list, tuple)):
                self.assertNotIn("TSTB", instruments)
                self.assertNotIn("SH000300", instruments)
                requests.append(list(instruments))
            return original_features(provider, instruments, *args, **kwargs)

        def benchmark(config, freq):
            # Existing upstream accepts an empty precomputed series, no provider lookup.
            self.assertTrue(config["benchmark"].empty)
            return original_benchmark(config, freq)

        with tempfile.TemporaryDirectory() as temp:
            snapshot = _snapshot(Path(temp), sessions_count=35)
            with patch.object(BaseProvider, "features", new=features), \
                 patch.object(PortfolioMetrics, "_cal_benchmark", side_effect=benchmark):
                run = run_backtest(snapshot, "$close", warmup_sessions=20, topk=2, holdings="never")
            summary = json.loads((run / "summary.json").read_text())
            self.assertEqual(set(summary["metrics"]), {"strategy_net"})
            self.assertTrue(requests)

    def test_new_snapshot_needs_no_benchmark_field(self):
        from tests.test_qlib_yahoo_execution import _snapshot
        from qlib_yahoo.backtest import run_backtest
        with tempfile.TemporaryDirectory() as temp:
            snapshot = _snapshot(Path(temp), sessions_count=35, include_benchmark=False)
            path = snapshot / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest.pop("benchmark")
            path.write_text(json.dumps(manifest))
            run = run_backtest(snapshot, "$close", warmup_sessions=20, topk=2, holdings="never")
            self.assertEqual(json.loads((run / "manifest.json").read_text())["schema_version"], 6)


if __name__ == "__main__":
    unittest.main()
