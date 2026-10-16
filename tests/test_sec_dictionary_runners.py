"""Worker dictionary pins and verification ordering; heavy execution is mocked."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from qlib_yahoo.dataset import DatasetError


class DictionaryRunnerTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.spec = dict(tag="Assets", taxonomy="us-gaap/", unit="USD", qtrs=0, segments="", coreg="")
        self.pin = dict(version="v1", catalog_sha256="catalog", dictionary_sha256="dictionary")
        self.execution_identity = dict(version="sec-execution-1", manifest_sha256="e" * 64)
        self.pin["execution"] = self.execution_identity
        self.config = dict(catalog=str(self.root / "catalog"), mode="dictionary", registry=self.pin,
                           bindings={"sec_assets": self.spec})
        self.status = Mock(return_value=dict(status="ready", **self.pin))
        self.verified = {"identity": self.execution_identity, "fields": {"sec_assets": self.spec}}
        self.verify = Mock(return_value=self.verified)
        for name, value in (("dictionary_status", self.status),
                            ("lookup_fields", Mock(return_value={"sec_assets": {"name": "sec_assets", "spec": self.spec}}))):
            mocked = patch("qlib_yahoo.sec_registry." + name, value)
            mocked.start()
            self.addCleanup(mocked.stop)
        for name, value in (("execution_status", Mock(return_value={"status": "ready", "identity": self.execution_identity})),
                            ("verify_execution_fields", self.verify)):
            mocked = patch("qlib_yahoo.sec_execution." + name, value)
            mocked.start()
            self.addCleanup(mocked.stop)

    def test_native_pin_mismatch_before_runtime_or_source_work(self):
        from qlib_yahoo.backtest import run_backtest

        self.status.return_value["dictionary_sha256"] = "changed"
        with patch("qlib_yahoo.fast_runtime.resolve_fast_profile") as runtime, \
                patch("qlib_yahoo.sec_features._verify_catalog") as source:
            with self.assertRaisesRegex(DatasetError, "identity differs"):
                run_backtest(self.root, "$sec_assets", sec=self.config, strategy_mode="signed")
            runtime.assert_not_called()
            source.assert_not_called()

    def test_native_selected_execution_verification_at_entry(self):
        from qlib_yahoo.backtest import run_backtest

        self.verify.side_effect = ValueError("dictionary bytes differ")
        with self.assertRaisesRegex(DatasetError, "dictionary bytes differ"):
            run_backtest(self.root, "$sec_assets", sec=self.config)
        self.verify.assert_called_once_with(Path(self.config["catalog"]), ['sec_assets'],
                                           expected_identity=self.execution_identity)

    def test_native_unknown_operator_rejected_before_dictionary_verification(self):
        from qlib_yahoo.backtest import run_backtest

        with patch("qlib_yahoo.fast_runtime.resolve_fast_profile") as runtime:
            with self.assertRaises(DatasetError):
                run_backtest(self.root, "Unknown($sec_assets)", sec=self.config, strategy_mode="signed")
            runtime.assert_not_called()
        self.verify.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "optional Lab runtime")
    def test_lab_unknown_operator_rejected_before_dictionary_verification(self):
        from qlib_yahoo.alphalab.batch import run_batch
        from qlib_yahoo.alphalab.settings import Settings

        with patch("qlib_yahoo.alphalab.batch.Library") as library:
            result = run_batch(self.root, ["unknown_operator(sec_assets)"], Settings(), data_root=self.root, sec=self.config)
            self.assertIn("unknown operator", result[0]["error"])
            library.assert_not_called()
        self.verify.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "optional Lab runtime")
    def test_lab_bad_call_signature_rejected_before_dictionary_verification(self):
        from qlib_yahoo.alphalab.batch import run_batch
        from qlib_yahoo.alphalab.settings import Settings

        with patch("qlib_yahoo.alphalab.batch.Library") as library, \
                patch("qlib_yahoo.sec_features._verify_catalog") as source:
            for expression in ("ts_mean(sec_assets)", "rank(sec_assets,bogus=2)"):
                with self.subTest(expression=expression):
                    result = run_batch(self.root, [expression], Settings(), data_root=self.root, sec=self.config)
                    self.assertIn("error", result[0])
            library.assert_not_called()
            source.assert_not_called()
        self.verify.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "optional Lab runtime")
    def test_dictionary_unknown_non_sec_field_rejected_before_any_bulk_work(self):
        from qlib_yahoo.alphalab.batch import run_batch
        from qlib_yahoo.alphalab.settings import Settings

        with patch("qlib_yahoo.alphalab.batch.source_identity", side_effect=AssertionError("price verification")) as prices, \
                patch("qlib_yahoo.sec_features._verify_catalog", side_effect=AssertionError("SEC verification")) as source:
            for expression in ("sec_assets+nonsense", "close+nonsense"):
                with self.subTest(expression=expression), self.assertRaisesRegex(DatasetError, "unknown field 'nonsense'"):
                    run_batch(self.root, [expression], Settings(), data_root=self.root, sec=self.config)
            prices.assert_not_called()
            source.assert_not_called()
        self.verify.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "optional Lab runtime")
    def test_invalid_legacy_and_price_calls_do_not_resolve_or_verify_sources(self):
        from qlib_yahoo.alphalab.batch import run_batch, save_batch
        from qlib_yahoo.alphalab.settings import Settings

        configurations = (None, {"catalog": self.config["catalog"], "bindings": self.config["bindings"]}, self.config)
        with patch("qlib_yahoo.alphalab.batch.source_identity") as prices, \
                patch("qlib_yahoo.sec_resolution.resolve_config") as resolve, \
                patch("qlib_yahoo.sec_features._verify_catalog") as source, \
                patch("qlib_yahoo.alphalab.batch.Workspace") as workspace, \
                patch("qlib_yahoo.alphalab.batch.Library") as library:
            for sec in configurations:
                expressions = (["ts_mean(sec_assets)", "rank(sec_assets,bogus=2)", "sec_assets +"] if sec
                               else ["ts_mean(close)", "rank(close,bogus=2)", "close +"])
                with self.subTest(sec=sec):
                    results = run_batch(self.root, expressions, Settings(), data_root=self.root, sec=sec)
                    self.assertEqual([row["expression"] for row in results], expressions)
                    self.assertTrue(all(row.get("error") and row["preflight"] and row["id"] for row in results))
            prices.assert_not_called()
            resolve.assert_not_called()
            source.assert_not_called()
            workspace.assert_not_called()
            library.assert_not_called()
        # Early errors remain representable in the ordinary batch receipt.
        receipt = save_batch(self.root, results, Settings(), str(self.root))
        import json

        saved = json.loads(receipt.read_text(encoding="utf-8"))
        self.assertEqual(set(saved["errors"]), {row["id"] for row in results})
        self.verify.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "optional Lab runtime")
    def test_mixed_batch_keeps_valid_cached_result_and_partial_error_in_input_order(self):
        from qlib_yahoo.alphalab.batch import run_batch
        from qlib_yahoo.alphalab.settings import Settings

        for sec in (None, {"catalog": self.config["catalog"], "bindings": self.config["bindings"]}, self.config):
            library = Mock()
            library.get.return_value = {"expression": "close", "summary": {}, "checks": [],
                                        "daily": {"dates": [], "pnl": []}}
            library.pnl_series.return_value = {}
            with self.subTest(sec=sec), \
                    patch("qlib_yahoo.alphalab.batch.Library", return_value=library), \
                    patch("qlib_yahoo.alphalab.batch.source_identity", return_value={"source_sha256": "prices"}) as prices, \
                    patch("qlib_yahoo.alphalab.batch.Workspace") as workspace, \
                    patch("qlib_yahoo.sec_features._verify_catalog") as source:
                results = run_batch(self.root, ["ts_mean(close)", "close", "rank(close,bogus=2)"],
                                    Settings(), data_root=self.root, sec=sec)
                self.assertEqual([row["expression"] for row in results], ["ts_mean(close)", "close", "rank(close,bogus=2)"])
                self.assertEqual(["error" in row for row in results], [True, False, True])
                self.assertTrue(results[1]["cached"])
                self.assertEqual(library.get.call_count, 1)
                prices.assert_called_once()
                workspace.assert_not_called()
                source.assert_not_called()
        self.verify.assert_not_called()

    def test_price_only_with_dictionary_does_not_verify_registry(self):
        from qlib_yahoo.sec_resolution import resolve_config

        resolved = resolve_config(["$close"], self.config, dialect="qlib", verify=True)
        self.assertFalse((resolved or {}).get("bindings"))
        self.verify.assert_not_called()
        self.status.assert_not_called()

    def test_batch_dispatch_rejects_pin_before_preparation(self):
        from qlib_yahoo.jobs import _run_request

        self.status.return_value["dictionary_sha256"] = "changed"
        with patch("qlib_yahoo.preparation.run_backtest_batch") as run:
            with self.assertRaisesRegex(ValueError, "identity differs"):
                _run_request(dict(kind="backtest-batch", dataset=str(self.root), data_root=str(self.root),
                                  alphas=["$sec_assets"], sec=self.config))
            run.assert_not_called()

    def test_build_dispatch_returns_sidecar_and_streams_progress(self):
        from qlib_yahoo.jobs import _run_request

        self.status.return_value["path"] = str(self.root / "sec/dictionaries/hash-v1")
        with patch("qlib_yahoo.sec_registry.build_dictionary") as dictionary_build, \
                patch("qlib_yahoo.sec_execution.build_execution_index") as build:
            result = _run_request(dict(kind="build-dictionary", catalog=self.config["catalog"]))
        self.assertEqual(result, Path(self.status.return_value["path"]))
        self.assertTrue(callable(build.call_args.kwargs["progress"]))
        self.assertEqual(build.call_args.args, (Path(self.config["catalog"]),))
        dictionary_build.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "optional Lab runtime")
    def test_lab_verifies_dictionary_and_source_before_cache_lookup(self):
        from qlib_yahoo.alphalab.batch import run_batch
        from qlib_yahoo.alphalab.settings import Settings

        events = []
        self.verify.side_effect = lambda *a, **kw: events.append("dictionary") or self.verified
        source = Mock(side_effect=lambda *a, **k: events.append("source"))
        library = Mock()
        library.get.side_effect = RuntimeError("cache lookup reached")
        with patch("qlib_yahoo.alphalab.batch.source_identity", return_value={"source_sha256": "prices"}), \
                patch("qlib_yahoo.sec_features._verify_catalog", source), \
                patch("qlib_yahoo.sec_features.feature_identity", return_value={"source_sha256": "sec"}), \
                patch("qlib_yahoo.alphalab.batch.Library", return_value=library), \
                patch("qlib_yahoo.sec_features.materialize") as materialize:
            with self.assertRaisesRegex(RuntimeError, "cache lookup reached"):
                run_batch(self.root, ["sec_assets"], Settings(), data_root=self.root, sec=self.config)
            self.assertEqual(events, ["dictionary", "source"])
            materialize.assert_not_called()
            source.assert_called_once_with(Path(self.config["catalog"]), force=True)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "optional Lab runtime")
    def test_lab_pin_mismatch_prevents_cache_lookup(self):
        from qlib_yahoo.alphalab.batch import run_batch
        from qlib_yahoo.alphalab.settings import Settings

        self.status.return_value["dictionary_sha256"] = "changed"
        with patch("qlib_yahoo.alphalab.batch.Library") as library, \
                patch("qlib_yahoo.alphalab.batch.source_identity") as prices:
            with self.assertRaisesRegex(ValueError, "identity differs"):
                run_batch(self.root, ["sec_assets"], Settings(), data_root=self.root, sec=self.config)
            library.assert_not_called()
            prices.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "optional Lab runtime")
    def test_lab_saved_result_carries_registry_provenance(self):
        import numpy as np
        from qlib_yahoo.alphalab.batch import run_batch
        from qlib_yahoo.alphalab.settings import Settings

        dates = np.array(["2024-01-02", "2024-01-03"], dtype="datetime64[D]")
        identity = dict(source_sha256="prices", symbols=["AAA"], fields=["close"], dates=[str(d) for d in dates])
        workspace = SimpleNamespace(symbols=["AAA"], fields={"close": np.ones((2, 1), dtype=np.float32)}, dates=dates)
        panel = dict(identity="panel", path=str(self.root / "panel"))
        result = dict(expression="sec_assets", checks=[], daily=dict(dates=identity["dates"], pnl=[0, 0]), summary={})
        with patch("qlib_yahoo.alphalab.batch.source_identity", return_value=identity), \
                patch("qlib_yahoo.alphalab.batch.Workspace", return_value=workspace), \
                patch("qlib_yahoo.sec_features._verify_catalog"), \
                patch("qlib_yahoo.sec_features.feature_identity", return_value={"source_sha256": "sec"}), \
                patch("qlib_yahoo.sec_features.materialize", return_value=panel) as materialize, \
                patch("qlib_yahoo.sec_features.load_arrays", return_value={"sec_assets": np.ones((2, 1), dtype=np.float32)}), \
                patch("qlib_yahoo.alphalab.batch.cache_dir", return_value=self.root), \
                patch("qlib_yahoo.alphalab.batch.write_cache", return_value=self.root), \
                patch("qlib_yahoo.alphalab.batch.worker_count", return_value=1), \
                patch("qlib_yahoo.alphalab.batch.CachedWorkspace"), \
                patch("qlib_yahoo.alphalab.batch._run_one", return_value=result):
            saved = run_batch(self.root, ["sec_assets"], Settings(), data_root=self.root, sec=self.config)[0]
            cached = run_batch(self.root, ["sec_assets"], Settings(), data_root=self.root, sec=self.config)[0]
            materialize.assert_called_once()
        self.assertEqual(saved["sec"]["registry"], self.pin)
        self.assertEqual(saved["sec"]["mode"], "dictionary")
        self.assertEqual(saved["dataset"]["sec"]["registry"], self.pin)
        self.assertTrue(cached["cached"])
        self.assertEqual(cached["sec"]["registry"], self.pin)
        self.assertEqual(cached["dataset"]["sec"]["registry"]["version"], self.pin["version"])
        from qlib_yahoo.alphalab.batch import Library

        persisted = Library(self.root / "alphalab/library.sqlite").get(saved["id"])
        self.assertEqual(persisted["sec"]["registry"], self.pin)


if __name__ == "__main__":
    unittest.main()
