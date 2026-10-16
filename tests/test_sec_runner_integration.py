"""Selected SEC inputs at runner boundaries; no external data or runtime install."""
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
import tempfile
from contextlib import contextmanager, redirect_stdout
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from qlib_yahoo.dataset import DatasetError
from qlib_yahoo.expressions import check_alpha_syntax, sec_selection, validate_alpha
from qlib_yahoo.preparation import BacktestPreparation, run_backtest_batch

HAVE_NUMPY = importlib.util.find_spec("numpy") is not None
HAVE_QLIB = importlib.util.find_spec("qlib") is not None


@contextmanager
def patched_sec_api(api):
    """Replace only the fixture module; retain imports made by native Qlib.

    patch.dict(sys.modules) restores the entire module table, which can remove
    lazy scientific-library imports and trigger broken reloads in later tests.
    """
    previous = sys.modules.get(api.__name__)
    sys.modules[api.__name__] = api
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop(api.__name__, None)
        else:
            sys.modules[api.__name__] = previous


def sec_api():
    api = ModuleType("qlib_yahoo.sec_features")
    api.load_bindings = Mock(side_effect=lambda config: config["bindings"])
    api._verify_catalog = Mock(return_value="verified-fixture")
    def select(expressions, bindings):
        aliases = set(re.findall(r"\bsec_[A-Za-z0-9_]+\b", " ".join(expressions)))
        missing = aliases - bindings.keys()
        if missing:
            raise DatasetError(f"unbound SEC alias {sorted(missing)[0]}")
        return {name: bindings[name] for name in sorted(aliases)}
    api.selected_aliases = Mock(side_effect=select)
    api.feature_identity = Mock(side_effect=lambda catalog, bindings: {
        "catalog": str(catalog), "bindings": bindings, "source_hash": "fixture-v1"})
    api.materialize = Mock()
    api.load_arrays = Mock()
    return api


def config():
    return {"catalog": "fixture-catalog", "bindings": {
        name: {"tag": tag, "taxonomy": "us-gaap/", "unit": "USD", "qtrs": 0,
               "segments": "", "coreg": ""}
        for name, tag in (("sec_assets", "Assets"), ("sec_debt", "Liabilities"), ("sec_unused", "Cash"))}}


class SyntaxAndPreparationTests(unittest.TestCase):
    def test_extra_features_are_explicit_and_keep_causality_checks(self):
        with self.assertRaisesRegex(DatasetError, "unsupported feature"):
            check_alpha_syntax("$sec_assets / $close")
        check_alpha_syntax("$sec_assets / $close", extra_features={"sec_assets"})
        for text in ("Ref($sec_assets, -1)", "Mean($sec_assets, 0)", "P($$sec_assets)", "__import__('os')"):
            with self.assertRaises(DatasetError):
                check_alpha_syntax(text, extra_features={"sec_assets"})

    @unittest.skipUnless(HAVE_QLIB, "optional Qlib")
    def test_native_window_inspection_accepts_declared_sec_features(self):
        self.assertEqual(validate_alpha("Mean($sec_assets, 3)/$close", extra_features={"sec_assets"}), 2)

    def test_missing_alias_rejected_before_runtime(self):
        from qlib_yahoo.backtest import run_backtest
        with patch("qlib_yahoo.fast_runtime.resolve_fast_profile") as runtime:
            with self.assertRaisesRegex(DatasetError, "SEC alias.*requires"):
                run_backtest(Path("absent"), "$sec_missing", strategy_mode="signed")
            runtime.assert_not_called()
        api = sec_api()
        with patched_sec_api(api):
            with self.assertRaisesRegex(DatasetError, "unbound SEC alias"):
                run_backtest(Path("absent"), "$sec_missing", sec=config(), strategy_mode="signed")
            api.materialize.assert_not_called()

    def test_batch_preflight_forwards_sec_and_prep_identity_invalidates(self):
        api = sec_api()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp, patched_sec_api(api), \
                patch("qlib_yahoo.backtest.run_backtest", return_value=Path(temp)/"run") as run:
            receipt = run_backtest_batch(Path(temp)/"data", ["$sec_assets/$close"], sec=config(), data_root=Path(temp))
            self.assertEqual(run.call_args.kwargs["sec"], config())
            self.assertEqual(json.loads(receipt.read_text())["status"], "completed")
        prep = BacktestPreparation()
        prep.bind({"prices": "same", "sec": "v1"})
        prep.initialized = prep.factor_validated = True
        prep.exchange = object()
        prep._frames["old"] = object()
        prep.bind({"prices": "same", "sec": "v2"})
        self.assertFalse(prep.initialized or prep.factor_validated)
        self.assertIsNone(prep.exchange)
        self.assertFalse(prep._frames)


@unittest.skipUnless(HAVE_NUMPY, "optional numpy")
class LabRunnerTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from qlib_yahoo.alphalab.batch import digest
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dates = np.datetime64("2024-01-02", "D") + np.arange(60).astype("timedelta64[D]")
        self.identity = {"dataset": str(self.root / "prices"), "source_sha256": "prices",
                         "dates": [str(d) for d in self.dates], "symbols": ["AAA", "BBB"],
                         "fields": ["close"]}
        self.api = sec_api()
        self.arrays = {}
        def materialize(catalog, bindings, dates, symbols, cache_root, progress=None, *, force_verify=True):
            self.assertFalse(force_verify)
            self.assertEqual(self.api._verify_catalog.call_args.kwargs, {"force": True})
            self.arrays = {name: np.full((len(dates), len(symbols)), i + 1, dtype=np.float32)
                           for i, name in enumerate(bindings)}
            return {"path": str(self.root / "selected"), "identity": digest(bindings), "fields": list(bindings)}
        self.api.materialize.side_effect = materialize
        self.api.load_arrays.side_effect = lambda handle: self.arrays
        def workspace(dataset):
            return SimpleNamespace(dataset=dataset, dates=self.dates, symbols=["AAA", "BBB"],
                fields={"close": np.ones((60, 2), dtype=np.float32)}, groups={},
                panel=SimpleNamespace(manifest={}), universe=lambda name: np.ones((60, 2), dtype=bool))
        def simulate(expression, settings, ws):
            for name in self.api.selected_aliases([expression], config()["bindings"]):
                self.assertIn(name, ws.fields)
                self.assertEqual(ws.fields[name].shape, (60, 2))
            return {"expression": expression, "settings": vars(settings), "dataset": ws.identity,
                    "summary": {}, "daily": {"dates": self.identity["dates"],
                                              "pnl": (np.sin(np.arange(60)) * 0.01).tolist()}, "checks": []}
        for context in (patched_sec_api(self.api),
                        patch("qlib_yahoo.alphalab.batch.source_identity", return_value=self.identity),
                        patch("qlib_yahoo.alphalab.batch.Workspace", side_effect=workspace),
                        patch("qlib_yahoo.alphalab.batch.simulate", side_effect=simulate)):
            if hasattr(context, "start"):
                context.start()
                self.addCleanup(context.stop)
            else:
                context.__enter__()
                self.addCleanup(context.__exit__, None, None, None)

    def run_lab(self, expressions, sec=None, force=False):
        from qlib_yahoo.alphalab.batch import run_batch
        from qlib_yahoo.alphalab.settings import Settings
        return run_batch(self.root / "prices", expressions, Settings(), data_root=self.root,
                         workers=1, sec=sec, force=force)

    def test_union_only_requested_and_per_expression_ids_survive_batch_composition(self):
        one = self.run_lab(["sec_assets/close"], config())[0]
        both = self.run_lab(["sec_assets/close", "sec_debt/close"], config(), force=True)
        self.assertEqual(one["id"], both[0]["id"])
        self.assertEqual(set(self.api.materialize.call_args.args[1]), {"sec_assets", "sec_debt"})
        self.assertEqual(set(both[0]["dataset"]["sec"]["bindings"]), {"sec_assets"})
        self.assertEqual(set(both[1]["dataset"]["sec"]["bindings"]), {"sec_debt"})
        cached = self.run_lab(["sec_assets/close"], config())[0]
        self.assertTrue(cached["cached"])
        self.assertEqual(self.api.materialize.call_count, 2)

    def test_price_only_ids_and_results_ignore_unused_sec_sources(self):
        original = self.run_lab(["close"])[0]
        changed = config()
        changed["catalog"] = "other-catalog"
        result = self.run_lab(["close"], changed)[0]
        self.assertEqual(original["id"], result["id"])
        self.assertTrue(result["cached"])
        self.assertNotIn("sec", result)
        self.api.materialize.assert_not_called()
        self.api.feature_identity.assert_not_called()
        self.api._verify_catalog.assert_not_called()

    def test_binding_change_invalidates_only_its_expression(self):
        first = self.run_lab(["sec_assets/close", "sec_debt/close"], config())
        changed = config()
        changed["bindings"]["sec_assets"]["unit"] = "EUR"
        second = self.run_lab(["sec_assets/close", "sec_debt/close"], changed)
        self.assertNotEqual(first[0]["id"], second[0]["id"])
        self.assertEqual(first[1]["id"], second[1]["id"])
        self.assertTrue(second[1]["cached"])
        self.assertEqual(set(self.api.materialize.call_args.args[1]), {"sec_assets"})

    def test_correlation_cohort_compares_price_and_different_sec_bindings(self):
        price = self.run_lab(["close"])[0]
        assets = self.run_lab(["sec_assets/close"], config())[0]
        self.assertEqual(assets["most_correlated_id"], price["id"])
        self.assertAlmostEqual(assets["self_correlation"], 1.0)
        debt = self.run_lab(["sec_debt/close"], config())[0]
        self.assertIn(debt["most_correlated_id"], {price["id"], assets["id"]})
        self.assertAlmostEqual(debt["self_correlation"], 1.0)
        from qlib_yahoo.alphalab.batch import Library
        rows = Library(self.root / "alphalab/library.sqlite").rows(dataset_hash="prices")
        self.assertEqual({row["id"] for row in rows}, {price["id"], assets["id"], debt["id"]})

    def test_price_verification_recheck_stays_separate_and_sec_change_is_rejected(self):
        self.api.feature_identity.side_effect = lambda *args: {"version": self.api.feature_identity.call_count}
        with self.assertRaisesRegex(DatasetError, "SEC source changed"):
            self.run_lab(["sec_assets/close"], config())

    def test_all_cached_sec_still_verifies_source_before_library_lookup(self):
        first = self.run_lab(["sec_assets/close"], config())[0]
        cached = self.run_lab(["sec_assets/close"], config())[0]
        self.assertEqual(first["id"], cached["id"])
        self.assertTrue(cached["cached"])
        self.assertEqual(self.api._verify_catalog.call_count, 2)
        self.assertTrue(all(call.kwargs == {"force": True}
                            for call in self.api._verify_catalog.call_args_list))
        self.assertEqual(self.api.materialize.call_count, 1)
        self.api._verify_catalog.side_effect = ValueError("NUM bytes differ")
        with patch("qlib_yahoo.alphalab.batch.Library.get") as lookup:
            with self.assertRaisesRegex(ValueError, "NUM bytes differ"):
                self.run_lab(["sec_assets/close"], config())
            lookup.assert_not_called()
        self.assertEqual(self.api.materialize.call_count, 1)

    def test_runner_releases_selected_maps_on_success_and_cache_failure(self):
        import numpy as np
        path = self.root / "selected.npy"
        np.save(path, np.ones((60, 2), dtype=np.float32), allow_pickle=False)
        opened = []
        def load(handle):
            mapped = np.load(path, mmap_mode="r", allow_pickle=False)
            opened.append(mapped)
            return {"sec_assets": mapped}
        self.api.load_arrays.side_effect = load
        self.run_lab(["sec_assets/close"], config())
        self.assertTrue(opened[-1]._mmap.closed)
        with patch("qlib_yahoo.alphalab.batch.write_cache", side_effect=RuntimeError("cache failure")):
            with self.assertRaisesRegex(RuntimeError, "cache failure"):
                self.run_lab(["sec_assets/close"], config(), force=True)
        self.assertTrue(opened[-1]._mmap.closed)


@unittest.skipUnless(HAVE_QLIB, "optional Qlib")
class NativeRunnerTests(unittest.TestCase):
    def test_native_sec_scores_next_open_manifest_packed_delegation_and_rebind(self):
        import numpy as np
        import qlib
        from qlib_yahoo import sec_provider
        from qlib_yahoo.backtest import run_backtest
        from qlib_yahoo.packed import pack_dataset
        from test_qlib_yahoo_execution import _snapshot, _saved_table

        api = sec_api()
        arrays = {}
        def materialize(catalog, bindings, dates, symbols, cache_root, progress=None, *, force_verify=True):
            self.assertFalse(force_verify)
            self.assertEqual(api._verify_catalog.call_count, api.materialize.call_count)
            self.assertEqual(api._verify_catalog.call_args.kwargs, {"force": True})
            self.assertTrue(callable(progress))
            output = io.StringIO()
            with redirect_stdout(output), patch("builtins.print", wraps=print) as emit:
                progress("Materializing SEC field sec_assets (1/1)")
            self.assertEqual(output.getvalue(), "Materializing SEC field sec_assets (1/1)\n")
            emit.assert_called_once_with("Materializing SEC field sec_assets (1/1)", flush=True)
            values = np.arange(1, len(symbols) + 1, dtype=np.float32)
            if bindings["sec_assets"]["unit"] == "EUR":
                values = values[::-1]
            arrays["sec_assets"] = np.tile(values, (len(dates), 1))
            return {"path": "fixture", "fields": list(bindings),
                    "identity": {"dates": list(map(str, dates)), "symbols": symbols,
                                 "bindings": bindings}, "diagnostics": {"fixture": True}}
        api.materialize.side_effect = materialize
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root = Path(temp)
            source = _snapshot(root, 35, include_benchmark=False)
            from qlib_yahoo.dataset import tree_sha256
            source_manifest = json.loads((source / "manifest.json").read_text())
            source_manifest["status"] = "ready"
            source_manifest["qlib_binary_file_count"] = tree_sha256(source / "qlib_data")[1]
            (source / "manifest.json").write_text(json.dumps(source_manifest))
            packed = root / "data" / "packed"
            pack_dataset(source, packed)
            prep = BacktestPreparation()
            with patched_sec_api(api), \
                    patch.object(sec_provider, "load_arrays", side_effect=lambda handle: dict(arrays)), \
                    patch.object(prep, "bind", wraps=prep.bind) as bind, \
                    patch("qlib.init", wraps=qlib.init) as initialize:
                first = run_backtest(packed, "$sec_assets/$close", sec=config(), topk=2, n_drop=1,
                                     warmup_sessions=5, holdings="never", _preparation=prep)
                provider = initialize.call_args.kwargs["feature_provider"]
                self.assertEqual(provider["class"], "SECFeatureProvider")
                self.assertEqual(provider["kwargs"]["price_provider"]["class"], "PackedFeatureProvider")
                self.assertEqual(api.materialize.call_args.args[2].shape, (35,))
                self.assertEqual(set(api.materialize.call_args.args[1]), {"sec_assets"})
                manifest = json.loads((first / "manifest.json").read_text())
                self.assertEqual(manifest["schema_version"], 6)
                from qlib_yahoo.backtest import sha256
                catalog_hash = sha256(Path(__file__).resolve().parents[1] / "qlib_yahoo/sec_catalog.py")
                self.assertEqual(manifest["runner_source_sha256"]["sec_catalog.py"], catalog_hash)
                self.assertEqual(bind.call_args.args[0]["source"]["sec_catalog.py"], catalog_hash)
                self.assertEqual(manifest["sec"]["source_identity"]["bindings"],
                                 {"sec_assets": config()["bindings"]["sec_assets"]})
                self.assertEqual(manifest["backtest"]["execution_contract"]["deal_price"], "open")
                scores = _saved_table(first, "alpha_scores")
                positions = _saved_table(first, "positions")
                first_day = positions["datetime"].min()
                previous_day = sorted(scores["datetime"].unique())[0]
                self.assertLess(previous_day, first_day)
                expected = set(scores.loc[scores["datetime"] == previous_day].nlargest(2, "score")["instrument"])
                self.assertEqual(set(positions.loc[positions["datetime"] == first_day, "instrument"]), expected)
                old_identity = prep.identity
                changed = config()
                changed["bindings"]["sec_assets"]["unit"] = "EUR"
                second = run_backtest(packed, "$sec_assets/$close", sec=changed, topk=2, n_drop=1,
                                      warmup_sessions=5, holdings="never", _preparation=prep)
                self.assertNotEqual(prep.identity, old_identity)
                self.assertEqual(initialize.call_count, 2)
                self.assertFalse(scores.equals(_saved_table(second, "alpha_scores")))
                api._verify_catalog.side_effect = ValueError("NUM bytes differ")
                with self.assertRaisesRegex(ValueError, "NUM bytes differ"):
                    run_backtest(packed, "$sec_assets/$close", sec=config(), topk=2, n_drop=1,
                                 warmup_sessions=5, holdings="never", _preparation=prep)
                self.assertEqual(api.materialize.call_count, 2)
                price = run_backtest(source, "$close", sec=config(), topk=2, n_drop=1,
                                     warmup_sessions=5, holdings="never", _preparation=prep)
                self.assertNotIn("feature_provider", initialize.call_args.kwargs)
                self.assertNotIn("sec", json.loads((price / "manifest.json").read_text()))
                for name in ("sec_catalog.py", "sec_features.py", "sec_provider.py"):
                    self.assertNotIn(name, bind.call_args.args[0]["source"])
                    self.assertNotIn(name, json.loads((price / "manifest.json").read_text())["runner_source_sha256"])
                self.assertEqual(api.materialize.call_count, 2)
                self.assertEqual(api._verify_catalog.call_count, 3)
                self.assertTrue(all(call.kwargs == {"force": True}
                                    for call in api._verify_catalog.call_args_list))
            prep.clear()


if __name__ == "__main__":
    unittest.main()
