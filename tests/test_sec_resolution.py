"""Dictionary resolution is metadata-only and cannot silently reinterpret aliases."""
import unittest
from pathlib import Path
from unittest.mock import patch

from qlib_yahoo.dataset import DatasetError
from qlib_yahoo.sec_features import selected_aliases
from qlib_yahoo.sec_resolution import config_mode, field_dependencies, resolve_config


ASSETS = {"tag": "Assets", "taxonomy": "us-gaap/", "unit": "USD", "qtrs": 0,
          "segments": "", "coreg": ""}
INVENTORY = dict(ASSETS, tag="InventoryNet")


class DependencyTests(unittest.TestCase):
    def test_assignments_are_sequential_and_calls_keywords_strings_are_not_fields(self):
        self.assertEqual(field_dependencies([
            'sec_local=sec_assets; x=sec_local; sec_local=close; '
            'bucket(x-sec_local, range="sec_inventory")'
        ], dialect="lab"), {"sec_assets", "close"})
        self.assertEqual(field_dependencies(['sec_function(close, sec_argument=2)']), {"close"})
        self.assertEqual(field_dependencies(['bucket(close, range="$sec_assets")']), {"close"})

    def test_forward_reference_is_not_hidden_by_later_assignment(self):
        self.assertEqual(field_dependencies(['x=sec_missing;sec_missing=close;x']), {"sec_missing", "close"})
        with self.assertRaisesRegex(ValueError, "Unknown SEC field"):
            selected_aliases(['x=sec_missing;sec_missing=close;x'], {})

    def test_native_names_and_comments(self):
        self.assertEqual(field_dependencies(['Mean($sec_assets, 3)/$close # sec_inventory']),
                         {"sec_assets", "close"})
        for invalid in ['__import__("os") + $sec_assets', 'Mean($sec_assets, 0)',
                        '$sec_assets + gibberish', '$sec_assets +', 'Ref($sec_assets, -1)']:
            with self.subTest(invalid=invalid), self.assertRaises(DatasetError):
                field_dependencies([invalid], dialect="qlib")

    def test_reserved_looking_local_without_catalog_is_not_a_datafield(self):
        self.assertIsNone(resolve_config(['sec_temp=close;sec_temp'], None, dialect="lab"))
        with self.assertRaisesRegex(DatasetError, "requires a SEC catalog"):
            resolve_config(['sec_assets'], None)

    def test_legacy_preserves_even_a_different_common_name_definition(self):
        old = {"catalog": "catalog", "bindings": {"sec_assets": INVENTORY, "sec_unused": ASSETS}}
        result = resolve_config(['sec_assets'], old)
        self.assertEqual(result["mode"], "legacy")
        self.assertEqual(result["bindings"], {"sec_assets": INVENTORY})
        self.assertNotIn("registry", result)
        self.assertEqual(old["bindings"]["sec_unused"], ASSETS)

    def test_mode_inference_and_reject_bad_mode(self):
        self.assertEqual(config_mode({"catalog": "catalog"}), "dictionary")
        self.assertEqual(config_mode({"catalog": "catalog", "bindings": {}}), "legacy")
        with self.assertRaises(ValueError):
            resolve_config(['close'], {"catalog": "catalog", "mode": "guess"})
        self.assertEqual(resolve_config(['close'], {"catalog": "catalog"})["bindings"], {})


class ResolutionTests(unittest.TestCase):
    def setUp(self):
        self.provenance = {"version": "test-v1", "catalog_sha256": "a" * 64,
                           "dictionary_sha256": "b" * 64}
        self.status = {"status": "ready", **self.provenance}
        self.config = {"catalog": str(Path("catalog").resolve()), "mode": "dictionary"}
        self.execution_identity = {"version": "sec-execution-1", "manifest_sha256": "e" * 64}
        self.provenance["execution"] = self.execution_identity
        self.entries = {"sec_assets": {"name": "sec_assets", "spec": ASSETS},
                        "sec_inventory": {"name": "sec_inventory", "spec": INVENTORY}}
        for name, kwargs in (
            ("dictionary_status", {"return_value": self.status}),
            ("lookup_fields", {"side_effect": lambda catalog, names: {n: self.entries[n] for n in names if n in self.entries}}),
            ("verify_dictionary", {"return_value": self.status}),
            ("search_dictionary", {"return_value": {"items": []}}),
        ):
            p = patch("qlib_yahoo.sec_registry." + name, **kwargs)
            setattr(self, name, p.start())
            self.addCleanup(p.stop)
        for name, kwargs in (
            ("execution_status", {"return_value": {"status": "ready", "identity": self.execution_identity}}),
            ("verify_execution_fields", {"side_effect": lambda catalog, names, **kw: {
                "fields": {n: self.entries[n]["spec"] for n in names}, "identity": self.execution_identity}}),
        ):
            p = patch("qlib_yahoo.sec_execution." + name, **kwargs)
            setattr(self, name, p.start())
            self.addCleanup(p.stop)

    def test_formula_resolves_only_external_fields_without_verify(self):
        result = resolve_config(['rank(sec_inventory / sec_assets)'], self.config, dialect="lab")
        self.assertEqual(result["bindings"], {"sec_assets": ASSETS, "sec_inventory": INVENTORY})
        self.assertEqual(result["registry"], self.provenance)
        self.verify_dictionary.assert_not_called()
        self.assertNotIn("bindings", self.config)

    def test_typo_is_rejected_not_corrected(self):
        with self.assertRaisesRegex(ValueError, "Unknown SEC field.*Did you mean 'sec_inventory'"):
            resolve_config(['sec_inventroy'], self.config, verify=True)
        self.verify_dictionary.assert_not_called()
        self.verify_execution_fields.assert_not_called()
        self.search_dictionary.assert_not_called()

    def test_known_but_unavailable_is_distinct_from_unknown(self):
        self.entries.pop("sec_inventory")
        with self.assertRaisesRegex(ValueError, "defined but unavailable"):
            resolve_config(['sec_inventory'], self.config)

    def test_dictionary_needs_preparation_or_numeric_adapter(self):
        for status, message in (("missing", "Prepare dictionary"), ("unsupported", "raw inspection only")):
            with self.subTest(status=status):
                self.dictionary_status.return_value = {"status": status}
                with self.assertRaisesRegex(ValueError, message):
                    resolve_config(['sec_assets'], self.config)
        self.lookup_fields.assert_not_called()

    def test_saved_recipe_or_dictionary_mismatch_is_not_overridden(self):
        with self.assertRaisesRegex(ValueError, "different definition"):
            resolve_config(['sec_assets'], {**self.config, "bindings": {"sec_assets": INVENTORY}})
        with self.assertRaisesRegex(ValueError, "identity differs"):
            resolve_config(['sec_assets'], {**self.config, "registry": dict(self.provenance, version="old")})

    def test_worker_verifies_and_detects_dictionary_change(self):
        resolve_config(['sec_assets'], self.config, verify=True)
        self.verify_dictionary.assert_not_called()
        self.verify_execution_fields.assert_called_once()
        self.verify_execution_fields.side_effect = None
        self.verify_execution_fields.return_value = {"identity": self.execution_identity, "fields": {"sec_assets": INVENTORY}}
        with self.assertRaisesRegex(ValueError, "changed during verification"):
            resolve_config(['sec_assets'], self.config, verify=True)

    def test_missing_execution_index_never_falls_back_to_full_dictionary_scan(self):
        self.execution_status.return_value = {"status": "missing"}
        result = resolve_config(['sec_assets'], self.config)
        self.assertNotIn("execution", result["registry"])
        with self.assertRaisesRegex(ValueError, "Prepare run index"):
            resolve_config(['sec_assets'], self.config, verify=True)
        self.verify_dictionary.assert_not_called()
        self.verify_execution_fields.assert_not_called()

    def test_old_parent_pin_accepted_but_new_execution_pin_enforced(self):
        old = {key: value for key, value in self.provenance.items() if key != "execution"}
        result = resolve_config(['sec_assets'], {**self.config, "registry": old}, verify=True)
        self.assertEqual(result["registry"], self.provenance)
        wrong = dict(self.provenance, execution={"manifest_sha256": "wrong"})
        with self.assertRaisesRegex(ValueError, "execution index identity differs"):
            resolve_config(['sec_assets'], {**self.config, "registry": wrong}, verify=True)

    def test_change_during_lookup_and_price_only_bypass(self):
        self.dictionary_status.side_effect = [self.status, dict(self.status, dictionary_sha256="c" * 64)]
        with self.assertRaisesRegex(ValueError, "changed while resolving"):
            resolve_config(['sec_assets'], self.config)
        self.dictionary_status.reset_mock()
        self.assertEqual(resolve_config(['close'], self.config)["bindings"], {})
        self.dictionary_status.assert_not_called()


if __name__ == "__main__":
    unittest.main()
