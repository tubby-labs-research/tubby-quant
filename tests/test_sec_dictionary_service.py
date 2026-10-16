"""Dictionary service boundaries, with no fact scans or real worker launches."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from qlib_yahoo.webapp.service import ConsoleService, ServiceError


class DictionaryServiceTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.catalog = self.root / "sec/catalogs/fixture"
        self.catalog.mkdir(parents=True)
        (self.catalog / "manifest.json").write_text("{}", encoding="utf-8")
        self.spec = dict(tag="Assets", taxonomy="us-gaap/", unit="USD", qtrs=0, segments="", coreg="")
        self.pin = dict(version="v1", catalog_sha256="catalog", dictionary_sha256="dictionary")
        self.entry = dict(name="sec_assets", spec=self.spec, label="Assets", kind="common")
        api = ModuleType("qlib_yahoo.sec_registry")
        api.COMMON_FIELDS = {"sec_assets": self.spec}
        api.dictionary_status = Mock(return_value=dict(status="ready", path=str(self.root / "sec/dictionaries/fixture"),
                                                       field_count=1, message="", **self.pin))
        api.lookup_fields = Mock(side_effect=lambda catalog, names: {name: self.entry for name in names if name == "sec_assets"})
        api.search_dictionary = Mock(return_value=dict(items=[self.entry], total=1, offset=0, version="v1"))
        api.verify_dictionary = Mock(return_value=self.pin)
        api.name_for_spec = Mock(return_value=self.entry)
        api.build_dictionary = Mock()
        self.api = api
        previous = sys.modules.get(api.__name__)
        sys.modules[api.__name__] = api
        def restore():
            if previous is None:
                sys.modules.pop(api.__name__, None)
            else:
                sys.modules[api.__name__] = previous
        self.addCleanup(restore)
        execution = patch("qlib_yahoo.sec_execution.execution_status", return_value={"status": "missing"})
        execution.start()
        self.addCleanup(execution.stop)
        self.service = ConsoleService(self.root, settings_path=self.root / "settings.json")
        header = patch("qlib_yahoo.sec_catalog.read_catalog", return_value={"numeric_materialization_supported": True})
        header.start()
        self.addCleanup(header.stop)
        self.config = {"catalog": str(self.catalog)}

    def test_config_preserves_catalog_only_and_dictionary_pin(self):
        with patch("qlib_yahoo.sec_features.load_bindings", side_effect=AssertionError("catalog treated as binding")):
            self.assertEqual(self.service._sec_config(self.config), self.config)
            pinned = {**self.config, "mode": "dictionary", "registry": self.pin}
            self.assertEqual(self.service._sec_config(pinned), pinned)

    def test_native_preview_is_metadata_only_and_resolves_used_inputs(self):
        with patch("qlib_yahoo.sec_features._verify_catalog", side_effect=AssertionError("source hash")):
            result = self.service.check_alpha("$sec_assets / $close", sec=self.config)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["sec_inputs"], [{"name": "sec_assets", "spec": self.spec}])
        self.assertEqual(result["registry"], self.pin)
        self.api.verify_dictionary.assert_not_called()

    def test_native_typo_rejected_before_launch(self):
        with patch.object(self.service, "_start") as launch:
            with self.assertRaisesRegex(ServiceError, "Unknown SEC field"):
                self.service.start_backtest({"alpha": "$sec_asssets", "sec": self.config})
            launch.assert_not_called()

    def test_no_catalog_sec_reference_never_starts_native_worker(self):
        with patch.object(self.service, "_start") as launch:
            with self.assertRaises(ServiceError):
                self.service.start_backtest({"alpha": "$sec_assets"})
            launch.assert_not_called()
        result = self.service.check_alpha("$sec_assets")
        self.assertIn("Choose a SEC catalog", result["message"])
        self.assertIn("shared dictionary", result["message"])
        self.assertNotIn("Add input", result["message"])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "optional Lab runtime")
    def test_no_catalog_sec_reference_never_starts_lab_worker(self):
        with patch.object(self.service, "_price_dataset", return_value=(self.root, {"fields": ["close"]})), \
                patch.object(self.service, "_lab_groups", return_value={"market"}), \
                patch.object(self.service, "_start") as launch:
            with self.assertRaises(ServiceError):
                self.service.start_alpha({"expressions": "sec_assets"})
            launch.assert_not_called()

    def test_native_price_only_ignores_dictionary_registry(self):
        with patch.object(self.service, "_price_dataset", return_value=(self.root, {"fields": ["close"]})), \
                patch("qlib_yahoo.webapp.service.snap_dates", return_value=[]):
            ordinary, _, _ = self.service._backtest_request({"alpha": "$close"})
            selected, _, _ = self.service._backtest_request({"alpha": "$close", "sec": self.config})
        self.assertNotIn("sec", ordinary)
        self.assertFalse(selected.get("sec", {}).get("bindings"))
        self.assertNotIn("registry", selected.get("sec", {}))
        self.api.lookup_fields.assert_not_called()
        self.api.verify_dictionary.assert_not_called()

    def test_native_bad_syntax_and_operator_with_selected_catalog(self):
        for expression in ("$close +", "Unknown($close)", "Unknown($sec_assets)", "$sec_assets +"):
            with self.subTest(expression=expression), patch.object(self.service, "_start") as launch:
                with self.assertRaises(ServiceError):
                    self.service.start_backtest({"alpha": expression, "sec": self.config})
                launch.assert_not_called()
        self.api.verify_dictionary.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "optional Lab runtime")
    def test_lab_typo_bad_syntax_and_operator_never_launch(self):
        dataset = self.root / "datasets/prices"
        with patch.object(self.service, "_price_dataset", return_value=(dataset, {"fields": ["close"]})), \
                patch.object(self.service, "_lab_groups", return_value={"market"}), \
                patch.object(self.service, "_start") as launch:
            for expression in ("sec_asssets", "close +", "unknown_operator(close)", "unknown_operator(sec_assets)",
                               "ts_mean(sec_assets)", "rank(sec_assets,bogus=2)"):
                with self.subTest(expression=expression), self.assertRaises(ServiceError):
                    self.service.start_alpha({"expressions": expression, "sec": self.config})
            launch.assert_not_called()
        self.api.verify_dictionary.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "optional Lab runtime")
    def test_lab_editor_rejects_call_signatures_using_dictionary_metadata_only(self):
        for expression in ("ts_mean(sec_assets)", "rank(sec_assets,bogus=2)"):
            with self.subTest(expression=expression):
                result = self.service.alpha_check(expression, sec=self.config)
                self.assertFalse(result["ok"])
                self.assertTrue(result["lines"][0]["message"])
        self.api.verify_dictionary.assert_not_called()

    def test_search_missing_status_is_not_an_error(self):
        self.api.dictionary_status.return_value["status"] = "missing"
        result = self.service.sec_dictionary(str(self.catalog), "Assets")
        self.assertEqual(result["status"], "missing")
        self.assertEqual(result["items"], [])
        self.api.search_dictionary.assert_not_called()

    def test_search_and_name_use_registry_metadata(self):
        result = self.service.sec_dictionary(str(self.catalog), "Assets", 10, True)
        self.assertEqual(result["items"], [self.entry])
        self.api.search_dictionary.assert_called_once_with(self.catalog, query="Assets", limit=50, offset=10, include_custom=True, view="recipes")
        self.assertEqual(self.service.sec_name({**self.config, "spec": self.spec}), self.entry)
        self.api.verify_dictionary.assert_not_called()

    def test_build_uses_independent_worker_kind(self):
        with patch.object(self.service, "_start", side_effect=lambda kind, label, request, work: request):
            request = self.service.start_sec_dictionary_build(self.config)
        self.assertEqual(request, {"kind": "build-dictionary", **self.config})
        self.api.build_dictionary.assert_not_called()

    def test_company_view_is_explicit_and_invalid_views_fail_before_search(self):
        self.service.sec_dictionary(str(self.catalog), "cash", view="company")
        self.api.search_dictionary.assert_called_once_with(
            self.catalog, query="cash", limit=50, offset=0,
            include_custom=False, view="company")
        self.api.search_dictionary.reset_mock()
        for view, custom in (("nonsense", False), ("company", True), (None, False)):
            with self.subTest(view=view, custom=custom), self.assertRaises(ServiceError):
                self.service.sec_dictionary(str(self.catalog), view=view, include_custom=custom)
        self.api.search_dictionary.assert_not_called()

    def test_company_route_and_invalid_view(self):
        from qlib_yahoo.webapp.server import Handler
        handler = SimpleNamespace(server=SimpleNamespace(service=self.service), _json=Mock(), _error=Mock())
        Handler._api(handler, "GET", "sec-dictionary", {"catalog": str(self.catalog), "view": "company"})
        self.api.search_dictionary.assert_called_once_with(
            self.catalog, query="", limit=50, offset=0, include_custom=False, view="company")
        Handler._api(handler, "GET", "sec-dictionary", {"catalog": str(self.catalog), "view": "wrong"})
        self.assertEqual(handler._error.call_args.args[0], 400)

    def test_routes_parse_dictionary_pages_and_dispatch_post_requests(self):
        from qlib_yahoo.webapp.server import Handler

        handler = SimpleNamespace(server=SimpleNamespace(service=self.service), _json=Mock(), _error=Mock())
        Handler._api(handler, "GET", "sec-dictionary", {"catalog": str(self.catalog), "include_custom": "true", "offset": "5"})
        self.api.search_dictionary.assert_called_once_with(self.catalog, query="", limit=50, offset=5, include_custom=True, view="recipes")
        Handler._api(handler, "GET", "sec-dictionary", {"offset": "-1"})
        self.assertEqual(handler._error.call_args.args[0], 400)
        Handler._api(handler, "POST", "sec-name", {**self.config, "spec": self.spec})
        self.assertEqual(handler._json.call_args.args[0], self.entry)
        with patch.object(self.service, "_start", return_value={"id": "job"}):
            Handler._api(handler, "POST", "sec-dictionary-build", self.config)
        self.assertEqual(handler._json.call_args.args[0], {"id": "job"})

    def test_native_batch_pins_union_when_first_expression_is_price_only(self):
        with patch.object(self.service, "_price_dataset", return_value=(self.root, {"fields": ["close"]})), \
                patch("qlib_yahoo.webapp.service.snap_dates", return_value=[]), \
                patch.object(self.service, "_start", side_effect=lambda kind, label, request, work: request):
            result = self.service.start_backtest({"alpha": "$close\n$sec_assets", "sec": self.config})
        self.assertEqual(result["sec"]["bindings"], {"sec_assets": self.spec})
        self.assertEqual(result["sec"]["registry"], self.pin)

    def test_cli_catalog_only_relative_path_and_mode_inference(self):
        from qlib_yahoo.alphalab.cli import load_sec_config
        from qlib_yahoo.sec_resolution import resolve_config

        path = self.root / "config.json"
        path.write_text(json.dumps({"catalog": "sec/catalogs/fixture"}), encoding="utf-8")
        loaded = load_sec_config(path)
        self.assertNotIn("bindings", loaded)
        self.assertEqual(resolve_config(["$sec_assets"], loaded, dialect="qlib")["mode"], "dictionary")
        legacy = resolve_config(["$sec_assets"], {**loaded, "bindings": {"sec_assets": self.spec}}, dialect="qlib")
        self.assertEqual(legacy["mode"], "legacy")
        self.assertNotIn("registry", legacy)


if __name__ == "__main__":
    unittest.main()
