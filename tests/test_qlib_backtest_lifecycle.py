"""Small lifecycle checks at the sealed-run/renderer boundary; no timing claims."""

import contextlib
import gc
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import weakref
from unittest.mock import patch

from qlib_yahoo import backtest
from qlib_yahoo.dataset import DatasetError, sha256

CHILD = "QLIB_BACKTEST_LIFECYCLE_SIGNED_CHILD"


class WrapperTests(unittest.TestCase):
    def test_all_public_parameters_forward_without_changing_defaults(self):
        public = inspect.signature(backtest.run_backtest)
        private = inspect.signature(backtest._run_backtest_to_sealed)
        self.assertEqual(public.return_annotation, "Path")
        self.assertEqual(set(private.parameters) - set(public.parameters), {"performance"})
        for name, parameter in public.parameters.items():
            self.assertEqual(parameter, private.parameters[name])
        values = {name: object() for name in public.parameters}
        values.update(dataset=Path("unused"), annual_cash_rate=0.0, open_report=False)
        received = []

        def seal(*args, **kwargs):
            received.append(private.bind(*args, **kwargs).arguments)
            return Path("sealed-run"), "2024-01-03", "2024-01-04"

        with patch.object(backtest, "_run_backtest_to_sealed", new=seal), patch(
                "qlib_yahoo.analysis.render_report", new=lambda *a, **k: Path("report.html")), contextlib.redirect_stdout(io.StringIO()):
            result = backtest.run_backtest.__wrapped__(**values)
        self.assertEqual(result, Path("sealed-run"))
        for name, value in values.items():
            self.assertIs(received[0][name], value)

    def test_completed_scope_cycles_die_before_render_and_external_callbacks_survive(self):
        class State:
            pass

        refs, retained, observations = [], [], []

        def seal(*args, **kwargs):
            state = State()
            state.cycle = state
            refs.append(weakref.ref(state))
            other = State()
            other.cycle = other
            refs.append(weakref.ref(other))
            # An externally owned callback remains usable; no closure-cell clearing.
            retained.append(lambda: other)
            return Path("sealed-run"), "s", "e"

        def render(*args, **kwargs):
            observations.append((refs[0]() is None, retained[0]() is refs[1]()))
            return Path("report.html")

        with patch.object(backtest, "_run_backtest_to_sealed", new=seal), patch(
                "qlib_yahoo.analysis.render_report", new=render), contextlib.redirect_stdout(io.StringIO()):
            backtest.run_backtest.__wrapped__(Path("unused"), "$close")
        self.assertEqual(observations, [(True, True)])
        retained.clear()
        gc.collect()
        self.assertIsNone(refs[1]())

    @unittest.skipIf(os.environ.get(CHILD) == "1", "parent launcher only")
    def test_signed_child_does_not_replace_parent_qlib(self):
        if not _installed():
            self.skipTest("optional installed signed runtime and table dependencies")
        before = {name: module for name, module in sys.modules.items()
                  if name == "qlib" or name.startswith("qlib.")}
        environment = {**os.environ, CHILD: "1", "PYTHONDONTWRITEBYTECODE": "1"}
        program = (
            "from qlib_yahoo.runtime import ensure_signed_runtime; ensure_signed_runtime(); "
            "import unittest; result=unittest.TextTestRunner(verbosity=2).run("
            "unittest.defaultTestLoader.loadTestsFromName("
            "'tests.test_qlib_backtest_lifecycle.InstalledLifecycleTests')); "
            "raise SystemExit(not result.wasSuccessful())"
        )
        result = subprocess.run([sys.executable, "-B", "-c", program],
                                cwd=Path(__file__).resolve().parents[1], env=environment,
                                capture_output=True, text=True, encoding="utf-8", timeout=120)
        self.assertEqual(result.returncode, 0, (result.stdout + result.stderr)[-16000:])
        self.assertNotIn("skipped", result.stderr)
        self.assertEqual(before, {name: module for name, module in sys.modules.items()
                                  if name == "qlib" or name.startswith("qlib.")})


def _installed():
    if not all(importlib.util.find_spec(name) for name in ("numpy", "pandas", "pyarrow")):
        return False
    from qlib_yahoo.runtime import PIN, ROOT, platform_pin
    try:
        pin = platform_pin(json.loads(PIN.read_text()))
        return (ROOT / pin["local_runtime"] / "qlib/_tubby_build.json").is_file()
    except (OSError, ValueError, KeyError, DatasetError):
        return False


class LifecycleFixture:
    def setUp(self):
        from tests.test_qlib_yahoo_execution import _snapshot
        self.temporary = tempfile.TemporaryDirectory(
            prefix="qlib-lifecycle-", dir=Path(__file__).resolve().parents[1] / "env")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = _snapshot(self.root, sessions_count=10, include_benchmark=False)
        self.refs = {}
        self.rendered = []
        self.preparation = None
        self.report_error = False
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stderr = self.stack.enter_context(contextlib.redirect_stderr(io.StringIO()))

        # Plain wrappers retain weakrefs only, never Mock.call_args/results.
        import qlib.backtest
        from qlib_yahoo.performance import RunPerformance
        original_simulation = qlib.backtest.backtest
        original_call = RunPerformance.call

        def remember(name, value):
            self.refs[name] = weakref.ref(value)

        def simulate(*args, **kwargs):
            strategy, executor = kwargs["strategy"], kwargs["executor"]
            remember("strategy", strategy)
            remember("executor", executor)
            remember("signal", strategy.signal)
            remember("signal_cache", strategy.signal.signal_cache)
            if "generate_trade_decision" in vars(strategy):
                remember("decision_callback", strategy.generate_trade_decision)
            result = original_simulation(*args, **kwargs)
            account = executor.trade_account
            remember("account", account)
            remember("position", account.current_position)
            remember("exchange", executor.trade_exchange)
            remember("quote", executor.trade_exchange.quote)
            remember("indicator", account.indicator)
            if hasattr(account, "history_writer"):
                remember("history_writer", account.history_writer)
            remember("report", result[0]["1day"][0])
            remember("indicator_frame", result[1]["1day"][0])
            return result

        def timed_call(performance, stage, function, *args, **kwargs):
            result = original_call(performance, stage, function, *args, **kwargs)
            if stage == "signed target weights":
                remember("scores", result[0])
                remember("weights", result[1])
            elif stage == "target bundle verification":
                remember("loaded_targets", result["weights"])
            elif stage == "portfolio risk input preparation":
                remember("risk_runtime", result)
            elif stage in ("position table", "diagnostic labels"):
                remember(stage, result)
            return result

        def render(run_dir, **options):
            manifest = json.loads((run_dir / "manifest.json").read_text())
            # Record facts, assert AFTER run_backtest: renderer exceptions are warnings.
            alive = sorted(name for name, ref in self.refs.items() if ref() is not None)
            cached = None
            if self.preparation is not None:
                prep = self.preparation
                cached = (self.refs["exchange"]() is prep.exchange,
                          tuple(id(value) for value in prep._frames.values()),
                          prep.identity, prep.factor_validated)
            hashes_match = all(sha256(run_dir / name) == digest
                               for name, digest in manifest["artifacts_sha256"].items())
            self.rendered.append((manifest, alive, cached, hashes_match, options))
            if self.report_error:
                raise OSError("injected report failure")
            return run_dir / "report.html"

        self.stack.enter_context(patch("qlib.backtest.backtest", new=simulate))
        self.stack.enter_context(patch.object(RunPerformance, "call", new=timed_call))
        self.stack.enter_context(patch("qlib_yahoo.analysis.render_report", new=render))

    def run_case(self, *, strategy_mode="signed", **options):
        return backtest.run_backtest(
            self.dataset, "$close", warmup_sessions=2, topk=2,
            strategy_mode=strategy_mode, runtime_profile="native", execution_profile="fast",
            holdings="never", **options)

    def assert_boundary(self, *, prepared=False, native_quote=False):
        self.assertTrue(self.rendered, "renderer never entered")
        manifest, alive, cached, hashes_match, _ = self.rendered[-1]
        self.assertEqual((manifest["stage"], manifest["status"]), ("sealed", "completed"))
        self.assertTrue(hashes_match)
        self.assertEqual(alive, ["exchange", "quote"] if prepared else ["quote"] if native_quote else [])
        return manifest, cached

    def targets(self):
        import numpy as np
        import pandas as pd
        from qlib_yahoo.target_bundle import write_bundle
        from tests.test_target_bundle import metadata
        days = pd.bdate_range("2024-01-02", periods=10)
        return write_bundle(self.root / "targets", dates=[str(day.date()) for day in days[1:-2]],
                            symbols=[f"TST{i}" for i in range(6)],
                            weights=np.tile([.25, -.125, 0, 0, 0, 0], (7, 1)).astype("float32"),
                            metadata=metadata(sha256(self.dataset / "manifest.json")))

    def assert_failed_without_render(self):
        self.assertEqual(self.rendered, [])
        manifests = list((self.dataset.parent.parent / "runs").glob("*/manifest.json"))
        self.assertEqual(len(manifests), 1)
        manifest = json.loads(manifests[0].read_text())
        self.assertEqual(manifest["status"], "failed")
        return manifest


@unittest.skipUnless(os.environ.get(CHILD) == "1", "isolated signed child only")
class InstalledLifecycleTests(LifecycleFixture, unittest.TestCase):
    def test_signed_stream_releases_graph_including_cancel_closure(self):
        self.run_case(history_mode="stream", _cancel_file=str(self.root / "cancel"),
                      _columnar_history=True, _flat_indicators=True, _buffered_history=True)
        self.assert_boundary()
        self.assertIn("history_writer", self.refs)
        self.assertIn("decision_callback", self.refs)

    def test_signed_memory_releases_history_and_tables(self):
        self.run_case(history_mode="memory")
        self.assert_boundary()
        self.assertIn("position table", self.refs)

    def test_targets_and_risk_callbacks_release_after_risk_is_sealed(self):
        from tests.test_portfolio_runtime import policy
        self.run_case(target_bundle=self.targets(), portfolio_risk=policy(gross=1),
                      history_mode="stream", _cancel_file=str(self.root / "cancel"))
        manifest, _ = self.assert_boundary()
        self.assertIn("loaded_targets", self.refs)
        self.assertIn("risk_runtime", self.refs)
        self.assertIn("portfolio_risk.jsonl.gz", manifest["artifacts_sha256"])

    def test_preparation_keeps_exact_exchange_and_frames_across_runs(self):
        from qlib_yahoo.preparation import BacktestPreparation
        prep = self.preparation = BacktestPreparation()
        self.run_case(history_mode="stream", _preparation=prep)
        _, first = self.assert_boundary(prepared=True)
        exchange_ref = self.refs["exchange"]
        cached_refs = [weakref.ref(frame) for frame in prep._frames.values()]
        self.assertTrue(cached_refs)
        self.run_case(history_mode="stream", _preparation=prep)
        _, second = self.assert_boundary(prepared=True)
        self.assertEqual(first, second)
        self.assertTrue(first[0] and first[3])
        self.assertIs(exchange_ref(), prep.exchange)
        self.assertTrue(all(ref() is not None for ref in cached_refs))
        self.assertGreater(prep.hits, 0)

    def test_report_error_keeps_completed_simulation_and_path(self):
        self.report_error = True
        result = self.run_case(history_mode="stream")
        self.assert_boundary()
        self.assertEqual(result.parent.name, "runs")
        self.assertIn("Report warning (simulation remains completed)", self.stderr.getvalue())

    def test_diagnostic_failure_does_not_retain_label_at_render(self):
        original_write = backtest._write_table

        def write(frame, path):
            if path.name == "signal_forward_labels.parquet.tmp":
                raise OSError("injected diagnostic failure")
            return original_write(frame, path)

        with patch.object(backtest, "_write_table", new=write):
            self.run_case(history_mode="stream")
        manifest, _ = self.assert_boundary()
        self.assertEqual(manifest["diagnostics"]["forward_labels"], "unavailable")

    def test_cancellation_propagates_without_reporting(self):
        marker = self.root / "cancel"
        marker.touch()
        with self.assertRaises(KeyboardInterrupt):
            self.run_case(history_mode="stream", _cancel_file=str(marker))
        manifest = self.assert_failed_without_render()
        self.assertEqual(manifest["error"]["type"], "KeyboardInterrupt")

    def test_mid_run_cancellation_after_three_signed_sessions_cleans_history(self):
        from qlib_yahoo.streaming import SignedHistoryWriter
        marker = self.root / "cancel"
        original_finish = SignedHistoryWriter.finish_session
        original_close = SignedHistoryWriter.close
        sessions, closures = [], []

        def finish(writer, account, day):
            original_finish(writer, account, day)
            sessions.append((day, writer.sessions, account._active_session is None,
                             bool(account.current_position.get_stock_list()),
                             account.current_position.transaction_cost))
            if writer.sessions == 3:
                marker.touch()

        def close(writer, *, completed):
            result = original_close(writer, completed=completed)
            closures.append((completed, writer.closed))
            return result

        with patch.object(SignedHistoryWriter, "finish_session", new=finish), patch.object(
                SignedHistoryWriter, "close", new=close), self.assertRaises(KeyboardInterrupt):
            self.run_case(history_mode="stream", _cancel_file=str(marker))
        manifest = self.assert_failed_without_render()
        self.assertEqual(manifest["error"]["type"], "KeyboardInterrupt")
        self.assertEqual([item[1] for item in sessions], [1, 2, 3])
        self.assertEqual(len({item[0] for item in sessions}), 3)
        self.assertTrue(all(item[2] and item[3] and item[4] > 0 for item in sessions))
        self.assertEqual(closures, [(False, True)])
        run_dir = next((self.dataset.parent.parent / "runs").iterdir())
        self.assertFalse(list(run_dir.glob("*.tmp")))
        self.assertFalse((run_dir / "signed_decisions.json.gz").exists())
        self.assertFalse((run_dir / "positions.parquet").exists())

    def test_history_close_failure_propagates_without_reporting(self):
        from qlib_yahoo.streaming import SignedHistoryWriter
        original_close = SignedHistoryWriter.close

        def close(writer, *, completed):
            if completed:
                raise OSError("injected history close failure")
            return original_close(writer, completed=completed)

        with patch.object(SignedHistoryWriter, "close", new=close), self.assertRaises(OSError):
            self.run_case(history_mode="stream")
        self.assert_failed_without_render()

    def test_risk_close_failure_propagates_without_reporting(self):
        from qlib_yahoo.portfolio_runtime import PortfolioRiskRuntime
        from tests.test_portfolio_runtime import policy
        original_close = PortfolioRiskRuntime.close

        def close(runtime, *, completed=False):
            if completed:
                raise OSError("injected risk close failure")
            return original_close(runtime, completed=completed)

        with patch.object(PortfolioRiskRuntime, "close", new=close), self.assertRaises(OSError):
            self.run_case(portfolio_risk=policy(gross=1), history_mode="stream")
        self.assert_failed_without_render()

    def test_sealed_manifest_write_failure_does_not_start_report(self):
        original_json = backtest._atomic_json

        def write(path, value):
            if path.name == "manifest.json" and value.get("status") == "completed":
                raise OSError("injected sealing failure")
            return original_json(path, value)

        with patch.object(backtest, "_atomic_json", new=write), self.assertRaises(OSError):
            self.run_case(history_mode="stream")
        self.assert_failed_without_render()


@unittest.skipIf(os.environ.get(CHILD) == "1", "normal Qlib parent only")
@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("qlib", "numpy", "pandas", "pyarrow")),
                     "optional normal Qlib environment")
class LongOnlyLifecycleTests(LifecycleFixture, unittest.TestCase):
    def test_long_only_releases_native_account_and_history(self):
        self.run_case(strategy_mode="long-only")
        self.assert_boundary(native_quote=True)
        # Native NumpyQuote's class-level LRU owns this last reference. Check
        # that ownership explicitly without adding global cache clearing to the runner.
        from qlib.backtest.high_performance_ds import NumpyQuote
        NumpyQuote.get_data.cache_clear()
        gc.collect()
        self.assertIsNone(self.refs["quote"]())


if __name__ == "__main__":
    unittest.main()
