"""Small worker-boundary fixtures; no catalog scans, Qlib engine or benchmarks."""

import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from qlib_yahoo import jobs
from qlib_yahoo.dataset import DatasetError

try:
    import numpy as np
    from qlib_yahoo.alphalab import batch
    from qlib_yahoo.alphalab.settings import Settings
except ImportError:
    np = None


class WorkerTimingTests(unittest.TestCase):
    def test_full_worker_includes_parsing_and_dispatch_through_save_report(self):
        original = json.loads
        clock = [10.0]
        output = io.StringIO()

        def parse(text):
            clock[0] += 3
            return original(text)

        def execute(request):
            clock[0] += 7
            return Path("saved.json")

        with patch.object(jobs.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(jobs.json, "loads", side_effect=parse), \
                patch.object(jobs, "_run_request", side_effect=execute), redirect_stdout(output):
            self.assertEqual(jobs.child_main(['{"kind":"backtest"}']), 0)
        lines = output.getvalue().splitlines()
        record = original(lines[0].split(": ", 1)[1])["diagnostics"]["worker"]
        self.assertEqual(record, {"scope": "child_main", "outcome": "completed", "elapsed_seconds": 10.0})
        self.assertTrue(lines[-1].startswith(jobs.RESULT))

    def test_failure_marker_and_elapsed_survive(self):
        output = io.StringIO()
        with patch.object(jobs, "_run_request", side_effect=DatasetError("bad source")), redirect_stdout(output):
            self.assertEqual(jobs.child_main(['{"kind":"backtest"}']), 1)
        self.assertIn(jobs.FAILURE + '"bad source"', output.getvalue())
        self.assertIn('"outcome": "failed"', output.getvalue())
        self.assertNotIn(jobs.RESULT, output.getvalue())

    def test_cancellation_propagates_without_success_marker(self):
        output = io.StringIO()
        with patch.object(jobs, "_run_request", side_effect=KeyboardInterrupt("cancel")), redirect_stdout(output):
            with self.assertRaises(KeyboardInterrupt):
                jobs.child_main(['{"kind":"backtest"}'])
        self.assertIn('"outcome": "interrupted"', output.getvalue())
        self.assertNotIn(jobs.RESULT, output.getvalue())

    def test_invalid_json_records_failed_whole_boundary(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(jobs.child_main(["{"]), 1)
        self.assertIn(jobs.FAILURE, output.getvalue())
        self.assertIn('"outcome": "failed"', output.getvalue())


@unittest.skipIf(np is None, "NumPy unavailable")
class LabTimingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".tmp-sec-timing-", dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = Settings()
        self.identity = {"source_sha256": "price", "symbols": ["A"], "fields": ["close"],
                         "dates": ["2025-01-02"]}
        self.result = {"expression": "close", "daily": {"dates": [], "pnl": []}, "checks": [], "summary": {}}
        self.lib = Mock()
        self.lib.get.side_effect = lambda _: copy.deepcopy(self.result)
        self.lib.pnl_series.return_value = {}
        self.clock = [0.0]
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(batch.time, "monotonic", side_effect=lambda: self.clock[0]))
        self.stack.enter_context(patch.object(batch, "source_identity", return_value=self.identity))
        self.stack.enter_context(patch.object(batch, "Library", return_value=self.lib))
        self.stack.enter_context(patch("qlib_yahoo.alphalab.engine.check_expression", return_value=[]))

    def run_fixture(self, expressions=None, sec=None, **kwargs):
        diagnostics, messages = {}, []
        results = batch.run_batch(self.root, expressions or ["close"], self.settings,
                                  data_root=self.root, sec=sec, progress=messages.append,
                                  diagnostics=diagnostics, **kwargs)
        return results, diagnostics, messages

    def test_all_cached_sec_includes_slow_verification_before_cache_lookup(self):
        config = {"mode": "dictionary", "catalog": str(self.root), "bindings": {"sec_assets": {}},
                  "registry": {"sha256": "same"}}

        def resolve(expressions, sec, *, dialect, verify):
            self.clock[0] += 30 if verify else 2
            return config

        def verify(*args, **kwargs):
            self.clock[0] += 5

        with patch("qlib_yahoo.sec_resolution.resolve_config", side_effect=resolve) as resolver, \
                patch("qlib_yahoo.sec_features._verify_catalog", side_effect=verify) as source, \
                patch("qlib_yahoo.sec_features.selected_aliases", return_value=config["bindings"]), \
                patch("qlib_yahoo.sec_features.feature_identity", return_value={"source": "same"}), \
                patch.object(batch, "Workspace") as workspace:
            results, diagnostics, messages = self.run_fixture(["sec_assets"] * 2, config)
        self.assertEqual([c.kwargs["verify"] for c in resolver.call_args_list], [False, True])
        source.assert_called_once_with(self.root, force=True)
        workspace.assert_not_called()
        self.lib.save.assert_not_called()
        self.assertEqual(diagnostics["elapsed_seconds"], 37)
        self.assertEqual(diagnostics["stages_seconds"], {"sec_resolution": 2, "sec_selected_verification": 30,
                         "sec_source_verification": 5, "saved_result_cache_lookup": 0})
        self.assertEqual(diagnostics["cache_hit_count"], 2)
        self.assertEqual(diagnostics["new_computation_count"], 0)
        self.assertTrue(all(r["cached"] for r in results))
        self.assertIn("run_batch procedure", messages[-1])
        saved = json.loads(batch.save_batch(self.root, results, self.settings, "prices",
                                            diagnostics=diagnostics).read_text())
        self.assertEqual(saved["diagnostics"]["alphalab"], diagnostics)
        self.assertEqual(saved["ids"], [r["id"] for r in results])

    def test_no_sec_path_has_no_sec_stages_or_source_verification(self):
        with patch("qlib_yahoo.sec_features._verify_catalog") as verify:
            _, diagnostics, _ = self.run_fixture()
        verify.assert_not_called()
        self.assertEqual(set(diagnostics["stages_seconds"]), {"saved_result_cache_lookup"})

    def test_force_and_duplicate_requests_count_one_new_computation(self):
        workspace = SimpleNamespace(symbols=["A"], fields={"close": np.ones((1, 1))},
                                    dates=np.array(["2025-01-02"], dtype="datetime64[D]"))
        cached_workspace = Mock()
        with patch.object(batch, "Workspace", return_value=workspace), \
                patch.object(batch, "write_cache", return_value=self.root), \
                patch.object(batch, "worker_count", return_value=1), \
                patch.object(batch, "CachedWorkspace", return_value=cached_workspace), \
                patch.object(batch, "_run_one", return_value=copy.deepcopy(self.result)) as compute:
            results, diagnostics, _ = self.run_fixture(["close"] * 2, force=True)
        self.lib.get.assert_not_called()
        compute.assert_called_once()
        cached_workspace.close.assert_called_once()
        self.assertEqual(diagnostics["cache_hit_count"], 0)
        self.assertEqual(diagnostics["new_computation_count"], 1)
        self.assertEqual(results[0]["id"], results[1]["id"])
        self.assertEqual(results[0]["id"], batch.alpha_id("close", self.settings, "price"))
        self.assertNotIn("diagnostics", self.lib.save.call_args.args[1])

    def test_preflight_errors_do_not_count_as_computations(self):
        with patch("qlib_yahoo.alphalab.engine.check_expression", return_value=["bad call"]):
            results, diagnostics, _ = self.run_fixture()
        self.assertEqual(diagnostics["new_computation_count"], 0)
        self.lib.get.assert_not_called()
        saved = json.loads(batch.save_batch(self.root, results, self.settings, "prices").read_text())
        self.assertEqual(saved["diagnostics"]["alphalab"]["new_computation_count"], 0)

    def test_legacy_save_call_persists_counts_without_changing_results(self):
        results = [{"id": "old", "cached": True}, {"id": "new"}, {"id": "new"},
                   {"id": "invalid", "preflight": True, "error": "bad call"}]
        before = copy.deepcopy(results)
        historical = self.root / "historical.json"
        historical.write_text('{"ids":["old"]}')
        saved = json.loads(batch.save_batch(self.root, results, self.settings, "prices").read_text())
        self.assertEqual(results, before)
        self.assertEqual(historical.read_text(), '{"ids":["old"]}')
        self.assertEqual(saved["diagnostics"]["alphalab"], {
            "scope": "batch_results", "cache_hit_count": 1, "new_computation_count": 1})
        self.assertEqual(saved["errors"], {"invalid": "bad call"})

    def test_failed_source_verification_never_reads_saved_results(self):
        config = {"catalog": str(self.root), "bindings": {"sec_assets": {}}}
        diagnostics = {}
        with patch("qlib_yahoo.sec_resolution.resolve_config", return_value=config), \
                patch("qlib_yahoo.sec_features._verify_catalog", side_effect=DatasetError("tampered")):
            with self.assertRaisesRegex(DatasetError, "tampered"):
                batch.run_batch(self.root, ["sec_assets"], self.settings, data_root=self.root,
                                sec=config, diagnostics=diagnostics)
        self.assertIn("sec_source_verification", diagnostics["stages_seconds"])
        self.lib.get.assert_not_called()

    def test_mixed_preflight_preserves_order_and_aggregate_cached_counts(self):
        def check(expression, **kwargs):
            self.clock[0] += 1
            return ["bad call"] if expression == "invalid" else []

        with patch("qlib_yahoo.alphalab.engine.check_expression", side_effect=check):
            results, diagnostics, _ = self.run_fixture(["invalid", "close", "close"])
        self.assertTrue(results[0]["preflight"])
        self.assertTrue(all(r["cached"] for r in results[1:]))
        self.assertEqual(diagnostics["cache_hit_count"], 2)
        self.assertEqual(diagnostics["new_computation_count"], 0)
        self.assertEqual(diagnostics["elapsed_seconds"], 5)

    def test_resolution_failure_and_cancel_keep_stage_timings(self):
        for error in (DatasetError("bad dictionary"), KeyboardInterrupt("cancel")):
            diagnostics, messages = {}, []

            def fail(*args, **kwargs):
                self.clock[0] += 4
                raise error

            with patch("qlib_yahoo.sec_resolution.resolve_config", side_effect=fail):
                with self.assertRaises(type(error)):
                    batch.run_batch(self.root, ["close"], self.settings, data_root=self.root,
                                    sec={"catalog": "fixture"}, diagnostics=diagnostics, progress=messages.append)
            self.assertEqual(diagnostics["stages_seconds"]["sec_resolution"], 4)
            self.assertEqual(diagnostics["elapsed_seconds"], 4)
            self.assertIn("interrupted/failed", messages[-1])

    def test_jobs_forwards_batch_diagnostics_to_save(self):
        def run(*args, **kwargs):
            kwargs["diagnostics"].update(cache_hit_count=2, new_computation_count=0)
            return []

        with patch.object(batch, "run_batch", side_effect=run), \
                patch.object(batch, "save_batch", return_value=self.root) as save:
            jobs._run_request({"kind": "alpha-batch", "dataset": "fixture", "data_root": str(self.root),
                               "settings": {}, "expressions": ["close"]})
        self.assertEqual(save.call_args.kwargs["diagnostics"], {"cache_hit_count": 2, "new_computation_count": 0})


@unittest.skipIf(np is None, "NumPy unavailable")
class NativeTimingTests(unittest.TestCase):
    def test_initial_sec_resolution_uses_existing_performance_stage_and_error_translation(self):
        from qlib_yahoo.backtest import run_backtest
        from qlib_yahoo.performance import RunPerformance

        perf = RunPerformance()
        output = io.StringIO()
        with patch("qlib_yahoo.performance.RunPerformance", return_value=perf), \
                patch("qlib_yahoo.sec_resolution.resolve_config", side_effect=ValueError("bad recipe")), \
                redirect_stdout(output):
            with self.assertRaisesRegex(DatasetError, "bad recipe"):
                run_backtest(Path("fixture"), "$sec_assets", sec={"catalog": "fixture"})
        self.assertIn("SEC resolution and selected verification", perf.snapshot()["stages"])
        self.assertIn("Failed SEC resolution", output.getvalue())

    def test_no_sec_does_not_create_sec_stage(self):
        from qlib_yahoo.backtest import run_backtest
        from qlib_yahoo.performance import RunPerformance

        perf = RunPerformance()
        with patch("qlib_yahoo.performance.RunPerformance", return_value=perf), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(DatasetError, "strategy_mode"):
                run_backtest(Path("fixture"), "$close", strategy_mode="invalid")
        self.assertEqual(perf.snapshot()["stages"], {})

    def test_forced_source_verification_is_timed_and_preserves_failure(self):
        from qlib_yahoo import backtest
        from qlib_yahoo.performance import RunPerformance

        with tempfile.TemporaryDirectory(prefix=".tmp-sec-timing-", dir=Path(__file__).resolve().parents[1]) as temp:
            dataset = Path(temp)
            (dataset / "manifest.json").write_text(json.dumps({"symbols": ["A", "B"]}))
            modules = {
                "qlib": SimpleNamespace(config=SimpleNamespace()),
                "qlib.backtest": SimpleNamespace(backtest=Mock()),
                "qlib.backtest.executor": SimpleNamespace(SimulatorExecutor=Mock()),
                "qlib.backtest.position": SimpleNamespace(BasePosition=Mock()),
                "qlib.constant": SimpleNamespace(REG_US="us"),
                "qlib.contrib.strategy": SimpleNamespace(TopkDropoutStrategy=Mock()),
                "qlib.data": SimpleNamespace(D=Mock()),
            }
            perf = RunPerformance()
            config = {"catalog": temp, "bindings": {"sec_assets": {}}}
            with patch.dict(sys.modules, modules), \
                    patch("qlib_yahoo.performance.RunPerformance", return_value=perf), \
                    patch("qlib_yahoo.sec_resolution.resolve_config", return_value=config), \
                    patch("qlib_yahoo.expressions.check_alpha_syntax"), \
                    patch.object(backtest, "validate_alpha", return_value=0), \
                    patch.object(backtest, "verify_dataset", return_value={
                        "qlib_binary_tree_sha256": "price", "qlib_binary_file_count": 1}), \
                    patch("qlib_yahoo.sec_features._verify_catalog", side_effect=DatasetError("tampered")) as verify, \
                    redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(DatasetError, "tampered"):
                    backtest.run_backtest(dataset, "$sec_assets", topk=1, n_drop=0, sec=config,
                                          execution_profile="reference")
            verify.assert_called_once_with(dataset, force=True)
            self.assertIn("SEC source verification", perf.snapshot()["stages"])


if __name__ == "__main__":
    unittest.main()
