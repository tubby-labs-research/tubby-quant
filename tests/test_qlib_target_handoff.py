"""Bounded handoff checks, with real signed execution in an isolated subprocess."""
import gzip
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qlib_yahoo.dataset import DatasetError, sha256
from qlib_yahoo.backtest import run_backtest
from qlib_yahoo.target_bundle import load_bundle, write_bundle

CHILD = "--target-handoff-child" in sys.argv
ROOT = Path(__file__).resolve().parents[1]


def installed():
    try:
        pin = json.loads((ROOT / "docs/qlib-derivative-pin.json").read_text())
        from qlib_yahoo.runtime import platform_pin
        pin = platform_pin(pin)
        return (sys.version_info[:2] == (3, 12)
                and all(importlib.util.find_spec(name) for name in ("numpy", "pandas", "pyarrow"))
                and (ROOT / pin["local_runtime"] / "qlib/_tubby_build.json").is_file())
    except (OSError, ValueError, KeyError, DatasetError):
        return False


@unittest.skipIf(CHILD, "parent process checks")
class BoundaryTests(unittest.TestCase):
    def test_double_transform_fails_before_runtime(self):
        with patch("qlib_yahoo.runtime.ensure_signed_runtime", side_effect=AssertionError("runtime must not load")):
            for options in ({}, {"strategy_mode": "signed", "signal_transform": "rank"},
                            {"strategy_mode": "signed", "gross_exposure": .5},
                            {"strategy_mode": "signed", "gross_exposure": True},
                            {"strategy_mode": "signed", "sec": {}}):
                with self.subTest(options=options), self.assertRaises(DatasetError):
                    run_backtest(Path("absent"), "unused", target_bundle={}, **options)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "optional numpy environment")
    def test_bad_envelope_fails_before_runtime(self):
        with patch("qlib_yahoo.runtime.ensure_signed_runtime", side_effect=AssertionError("runtime must not load")):
            with self.assertRaisesRegex(DatasetError, "bound path/sha256 envelope"):
                run_backtest(Path("absent"), "unused", target_bundle={}, strategy_mode="signed")

    @unittest.skipUnless(installed(), "verified signed derivative/dependencies unavailable")
    def test_installed_derivative_handoff(self):
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                                 "--target-handoff-child"], cwd=ROOT, env=environment,
                                capture_output=True, text=True, encoding="utf-8", timeout=180)
        self.assertEqual(result.returncode, 0, (result.stdout + result.stderr)[-18000:])
        self.assertIn("HANDOFF_CHILD_OK", result.stdout)


@unittest.skipUnless(CHILD, "isolated installed-engine subprocess only")
class InstalledHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from qlib_yahoo.runtime import ensure_signed_runtime
        cls.identity = ensure_signed_runtime()

    def setUp(self):
        import numpy as np
        import pandas as pd
        from test_qlib_yahoo_execution import _snapshot
        self.np, self.pd = np, pd
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = _snapshot(self.root, sessions_count=45, include_benchmark=False)
        self.days = pd.bdate_range("2024-01-02", periods=45)
        self.symbols = [f"TST{i}" for i in range(6)]
        self.values = np.zeros((24, 6), dtype=np.float32)
        self.values[0:2, :2] = [.25, -.125]
        self.values[3:5, :2] = [-.125, .25]  # liquidation then reversal

    def bundle(self, *, values=None, dates=None, metadata_changes=None):
        from test_target_bundle import metadata
        meta = metadata(sha256(self.dataset / "manifest.json"))
        meta.update(metadata_changes or {})
        directory = self.root / ("targets-" + str(len(list(self.root.glob("targets-*")))))
        return write_bundle(directory, dates=dates if dates is not None else
                            [str(day.date()) for day in self.days[19:-2]], symbols=self.symbols,
                            weights=self.values if values is None else values, metadata=meta)

    def run_targets(self, envelope, **options):
        from qlib.backtest import backtest as actual_backtest
        from qlib_yahoo.performance import RunPerformance
        captured = []
        original_call = RunPerformance.call

        def checked_call(performance, stage, function, *args, **kwargs):
            if stage in ("alpha evaluation", "signed target weights"):
                raise AssertionError("supplied targets must bypass score evaluation/normalization")
            return original_call(performance, stage, function, *args, **kwargs)

        def observe(**kwargs):
            strategy = kwargs["strategy"]
            captured.append(strategy)
            loaded = load_bundle(envelope)
            for number, day in enumerate(loaded["dates"]):
                signal = strategy.signal.get_signal(start_time=self.pd.Timestamp(day), end_time=self.pd.Timestamp(day))
                self.np.testing.assert_array_equal(signal.reindex(self.symbols).to_numpy(), loaded["weights"][number])
            return actual_backtest(**kwargs)

        with patch("qlib_yahoo.signed.normalize_scores", side_effect=AssertionError("normalizer called")), \
                patch("qlib_yahoo.backtest.validate_alpha", side_effect=AssertionError("alpha validated")), \
                patch.object(RunPerformance, "call", checked_call), \
                patch("qlib.backtest.backtest", side_effect=observe):
            run = run_backtest(self.dataset, "NOT AN ALPHA", target_bundle=envelope, strategy_mode="signed",
                               warmup_sessions=20, execution_profile=options.pop("execution_profile", "reference"),
                               history_mode=options.pop("history_mode", "memory"), runtime_profile="native",
                               holdings="never", **options)
        return run, captured[0]

    def test_exact_targets_next_open_accounting_and_replay(self):
        from qlib_yahoo.analysis import read_instrument_table, table_artifact, render_report
        for profile in ("reference", "fast"):
            with self.subTest(profile=profile):
                envelope = self.bundle()
                run, strategy = self.run_targets(envelope, execution_profile=profile)
                manifest = json.loads((run / "manifest.json").read_text())
                self.assertEqual(manifest["input_mode"], "target_weights")
                self.assertNotIn("$close", manifest["alpha"])
                self.assertIn("not raw alpha", manifest["score_table_semantics"])
                self.assertEqual(manifest["target_bundle"]["identity"]["sha256"], envelope["sha256"])
                self.assertEqual(manifest["signed_engine_identity"]["commit"], self.identity["commit"])
                self.assertNotIn("alpha evaluation", str(manifest["performance"]))
                weights = read_instrument_table(table_artifact(run, "signed_weights"))
                self.np.testing.assert_array_equal(weights.weight.to_numpy(), self.values.ravel())
                scores = read_instrument_table(table_artifact(run, "alpha_scores"))
                self.np.testing.assert_array_equal(scores.score.to_numpy(), self.values.ravel())
                decisions = json.loads(gzip.decompress((run / "signed_decisions.json.gz").read_bytes()))
                self.assertEqual(len(decisions), 24)
                for index, decision in enumerate(decisions):
                    self.assertEqual(self.pd.Timestamp(decision["session"]), self.days[20 + index])
                    self.assertEqual(self.pd.Timestamp(decision["signal_session"]), self.days[19 + index])
                    self.assertEqual(decision["target_weights"], dict(zip(self.symbols, map(float, self.values[index]))))
                daily = self.pd.read_csv(run / "daily_report.csv")
                self.np.testing.assert_allclose(daily.account, daily.cash_total + daily.long_value - daily.short_liability)
                self.np.testing.assert_allclose(daily.cost_dollars, daily.transaction_cost_dollars + daily.borrow_cost_dollars)
                self.assertGreater(daily.transaction_cost_dollars.sum(), 0)
                self.assertGreater(daily.borrow_cost_dollars.sum(), 0)
                self.assertTrue(daily.funding_feasible.all())
                self.assertGreater(daily.free_cash.min(), 0)
                self.assertTrue(render_report(run, level="summary", holdings="never").is_file())

    def test_zero_targets_are_cash_only(self):
        envelope = self.bundle(values=self.np.zeros_like(self.values))
        run, _ = self.run_targets(envelope, execution_profile="fast", history_mode="stream")
        daily = self.pd.read_csv(run / "daily_report.csv")
        self.assertTrue((daily.account == 100000).all())
        self.assertTrue((daily.cost_dollars == 0).all())

    def test_missing_decision_and_wrong_dataset_binding_rejected(self):
        envelope = self.bundle(values=self.values[1:], dates=[str(x.date()) for x in self.days[20:-2]])
        with self.assertRaisesRegex(DatasetError, "missing target decision"):
            self.run_targets(envelope)
        envelope = self.bundle(metadata_changes={"dataset_manifest_sha256": "0" * 64})
        with self.assertRaisesRegex(DatasetError, "dataset manifest digest"):
            self.run_targets(envelope)


if __name__ == "__main__":
    if CHILD:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(InstalledHandoffTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if result.wasSuccessful():
            print("HANDOFF_CHILD_OK")
        sys.exit(0 if result.wasSuccessful() else 1)
    unittest.main()
