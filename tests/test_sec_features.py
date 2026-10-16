"""Independent known-answer fixtures for selected SEC filing events."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from qlib_yahoo.sec_features import (feature_identity, load_arrays, load_bindings,
                                     materialize, selected_aliases)

HAVE_NUMPY = importlib.util.find_spec("numpy") is not None
SPEC = {"tag": "NetIncomeLoss", "taxonomy": "us-gaap/", "unit": "USD", "qtrs": 1,
        "segments": "", "coreg": ""}


@unittest.skipUnless(HAVE_NUMPY, "requires NumPy")
class SECFeaturesTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.catalog = self.root / "catalog"
        self.catalog.mkdir()
        self.mapping = {"fields": ["cik", "name", "ticker", "exchange"],
                        "data": [[1, "Issuer", "AAA", "NYSE"], [2, "Second", "BBB", "NYSE"],
                                 [3, "Class A", "BRK.A", "NYSE"], [3, "Class B", "BRK-B", "NYSE"],
                                 [4, "Conflict", "BAD", "NYSE"], [5, "Conflict2", "BAD", "NYSE"]]}
        (self.catalog / "tickers.json").write_text(json.dumps(self.mapping))
        self.rows = []
        self.maps = []
        self.read_calls = []
        self.manifest = {"catalog_sha256": "a" * 64, "numeric_materialization_supported": True,
                         "artifacts_sha256": {"tickers.json": hashlib.sha256(
                             (self.catalog / "tickers.json").read_bytes()).hexdigest()}}
        def read(path, verify=False):
            self.read_calls.append(verify)
            return self.manifest
        self.read = patch("qlib_yahoo.sec_catalog.read_catalog", side_effect=read)
        self.read.start()
        self.addCleanup(self.read.stop)
        self.facts = patch("qlib_yahoo.sec_catalog.iter_facts", side_effect=lambda path, spec: iter(self.rows))
        self.facts.start()
        self.addCleanup(self.facts.stop)

    def fact(self, value, **changes):
        row = {"cik": "1", "tag": "NetIncomeLoss", "version": "us-gaap/2024", "uom": "USD",
               "qtrs": "1", "segments": "", "coreg": "", "ddate": "20241231",
               "accepted": "2025-01-29 16:12:00.0", "filed": "20250129", "value": value,
               "adsh": "0000000001-25-000001"}
        row.update(changes)
        self.rows.append(row)

    def panel(self, dates, symbols=("AAA.US",), spec=None):
        handle = materialize(self.catalog, {"sec_income": spec or SPEC}, dates, symbols, self.root / "cache")
        values = load_arrays(handle)["sec_income"]
        self.maps.append(values)
        self.addCleanup(values._mmap.close)
        return values, handle

    def test_clock_bound_fractional_acceptance_and_filed_later(self):
        self.fact("10")
        values, _ = self.panel(["2025-01-30", "2025-01-31", "2025-02-03"])
        self.assertTrue(self.np.isnan(values[0, 0]))
        self.np.testing.assert_array_equal(values[1:, 0], [10, 10])
        self.fact("20", cik="2", accepted="2025-01-31 06:01:00.0", filed="20250131")
        values, _ = self.panel(["2025-01-31", "2025-02-03"], ("BBB",))
        self.assertTrue(self.np.isnan(values[0, 0]))
        self.assertEqual(values[1, 0], 20)

    def test_latest_period_then_amendment_and_older_restatement(self):
        self.fact("10")
        self.fact("12", accepted="2025-02-03 10:00:00", filed="20250203")
        self.fact("999", ddate="20240930", accepted="2025-02-04 10:00:00", filed="20250204")
        values, _ = self.panel(["2025-01-31", "2025-02-04", "2025-02-05", "2025-02-06"])
        self.np.testing.assert_array_equal(values[:, 0], [10, 10, 12, 12])

    def test_newest_conflict_and_invalid_value_block_no_fallback(self):
        self.fact("10")
        self.fact("10000000000000000000000001", accepted="2025-02-03 10:00:00", filed="20250203")
        self.fact("10000000000000000000000002", accepted="2025-02-03 10:00:00", filed="20250203")
        self.fact("", accepted="2025-02-04 10:00:00", filed="20250204")
        values, handle = self.panel(["2025-01-31", "2025-02-05", "2025-02-06"])
        self.assertEqual(values[0, 0], 10)
        self.assertTrue(self.np.isnan(values[1:, 0]).all())
        self.assertEqual(handle["diagnostics"]["blocked_winners"], 2)

    def test_exact_units_duration_context_and_versions(self):
        self.fact("10")
        for change in ({"uom": "EUR"}, {"qtrs": "2"}, {"segments": "BusinessSegment=One"},
                       {"coreg": "Subsidiary"}, {"version": "custom2024"}):
            self.fact("999", **change)
        values, _ = self.panel(["2025-01-31"])
        self.assertEqual(values[0, 0], 10)
        self.fact("30", version="us-gaap/2025", accepted="2025-02-03 10:00:00", filed="20250203")
        values, _ = self.panel(["2025-02-05"], spec=dict(SPEC, taxonomy="us-gaap/2024"))
        self.assertEqual(values[0, 0], 10)

    def test_month_expiry_no_older_period_fallback(self):
        self.fact("10")
        self.fact("900", ddate="20240930", accepted="2026-03-30 10:00:00", filed="20260330")
        values, _ = self.panel(["2026-03-31", "2026-04-01"])
        self.assertEqual(values[0, 0], 10)
        self.assertTrue(self.np.isnan(values[1, 0]))

    def test_mapping_ambiguity_and_share_classes(self):
        self.fact("3", cik="3")
        self.fact("4", cik="4")
        values, handle = self.panel(["2025-01-31"], ("BRK.A.US", "BRK-B.US", "BAD.US", "NONE"))
        self.np.testing.assert_array_equal(values[0, :2], [3, 3])
        self.assertTrue(self.np.isnan(values[0, 2:]).all())
        self.assertEqual(handle["diagnostics"]["ambiguous_symbols"], ["BAD.US"])
        self.assertEqual(handle["diagnostics"]["unmapped_symbols"], ["NONE"])

    def test_bad_clocks_future_period_sub_conflict_and_overflow(self):
        self.fact("1e1000")
        self.fact("99", cik="2", accepted="2025-01-29")
        self.fact("99", cik="2", filing_missing=True)
        self.fact("99", cik="2", filing_conflict=True)
        self.fact("99", cik="2", ddate="20260101")
        values, handle = self.panel(["2025-01-31"], ("AAA", "BBB"))
        self.assertTrue(self.np.isnan(values).all())
        self.assertEqual(handle["diagnostics"]["ineligible_rows"], 4)
        self.assertEqual(handle["diagnostics"]["overflow_winners"], 1)

    def test_cache_reuse_tamper_and_calendar_identity(self):
        self.fact("10")
        values, first = self.panel(["2025-01-31"])
        self.assertFalse(values.flags.writeable)
        _, second = self.panel(["2025-01-31"])
        self.assertEqual(first, second)
        _, third = self.panel(["2025-01-31", "2025-02-03"])
        self.assertNotEqual(first["path"], third["path"])
        for array in self.maps:
            array._mmap.close()
        file = (Path(first["path"]).parent / "fields-v1" /
                first["field_objects"]["sec_income"] / "values.npy")
        file.write_bytes(file.read_bytes()[:-1] + b"X")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.panel(["2025-01-31"])

    def test_identity_does_not_scan_or_hash_bulk(self):
        with patch("qlib_yahoo.sec_features._hash", side_effect=AssertionError("bulk hash")):
            identity = feature_identity(self.catalog, {"sec_income": SPEC})
        self.assertEqual(identity["catalog_sha256"], "a" * 64)
        self.assertEqual(self.read_calls, [False])

    def test_reject_notes_adapter_and_invalid_calendar(self):
        self.manifest["numeric_materialization_supported"] = False
        with self.assertRaisesRegex(ValueError, "dimension adapter"):
            self.panel(["2025-01-31"])
        self.manifest["numeric_materialization_supported"] = True
        with self.assertRaisesRegex(ValueError, "increasing"):
            self.panel(["2025-02-03", "2025-01-31"])

    def test_failed_array_load_closes_prior_and_invalid_maps(self):
        import numpy as np
        handle = materialize(self.catalog, {"sec_a": SPEC, "sec_b": dict(SPEC, tag="Liabilities")},
                             ["2025-01-31"], ["AAA"], self.root / "cache")
        second = (Path(handle["path"]).parent / "fields-v1" /
                  handle["field_objects"]["sec_b"] / "values.npy")
        original_load = np.load
        for failure in ("hash", "shape"):
            with self.subTest(failure=failure):
                np.save(second, np.zeros((2, 1), dtype=np.float32))
                if failure == "shape":
                    handle["filehashes"]["sec_b"] = hashlib.sha256(second.read_bytes()).hexdigest()
                    receipt_path = second.parent / "field.json"
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    receipt["sha256"] = handle["filehashes"]["sec_b"]
                    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                else:
                    handle["filehashes"]["sec_b"] = "0" * 64
                opened = []
                def capture(*args, **kwargs):
                    values = original_load(*args, **kwargs)
                    opened.append(values)
                    return values
                with patch("numpy.load", side_effect=capture):
                    with self.assertRaisesRegex(ValueError, failure):
                        load_arrays(handle)
                self.assertEqual(len(opened), 1 if failure == "hash" else 2)
                self.assertTrue(all(values._mmap.closed for values in opened))


class BindingTests(unittest.TestCase):
    def test_exact_lexical_tokens_and_optional_context(self):
        binding = dict(SPEC)
        binding.pop("segments")
        binding.pop("coreg")
        bindings = load_bindings({"sec_income": binding, "sec_other": SPEC})
        self.assertEqual(selected_aliases(['$sec_income # sec_other', 'bucket(sec_income, range="sec_other")'], bindings),
                         {"sec_income": SPEC})
        with self.assertRaisesRegex(ValueError, "Unknown SEC"):
            selected_aliases(["sec_income_extra"], bindings)
        self.assertEqual(load_bindings({"sec_ifrs": dict(SPEC, taxonomy="ifrs-full/")})["sec_ifrs"]["taxonomy"],
                         "ifrs-full/")
        with self.assertRaises(ValueError):
            load_bindings({"../escape": SPEC})
        with self.assertRaises(ValueError):
            load_bindings({"sec_x": dict(SPEC, taxonomy="0000320193-25/")})


@unittest.skipUnless(HAVE_NUMPY and importlib.util.find_spec("pyarrow"), "requires NumPy and Arrow")
class ActualCatalogTests(unittest.TestCase):
    def test_warm_cache_rejects_num_mutation_with_unchanged_manifest(self):
        import numpy as np
        import os
        from qlib_yahoo.sec_catalog import build_catalog, read_catalog

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            tables = {
                "sub": "adsh\tcik\taccepted\tfiled\n0000000001-25-000001\t1\t2025-01-29 16:12:00.0\t20250129\n",
                "tag": "tag\tversion\tcustom\nNetIncomeLoss\tus-gaap/2024\t0\n",
                "num": "adsh\ttag\tversion\tuom\tqtrs\tddate\tvalue\tsegments\tcoreg\n0000000001-25-000001\tNetIncomeLoss\tus-gaap/2024\tUSD\t1\t20241231\t10\t\t\n",
                "pre": "adsh\ttag\tversion\n0000000001-25-000001\tNetIncomeLoss\tus-gaap/2024\n",
            }
            for table, text in tables.items():
                (source / f"{table}.txt").write_text(text)
            ticker = root / "tickers.json"
            ticker.write_text(json.dumps({"fields": ["cik", "ticker"], "data": [[1, "MSFT"]]}))
            catalog = root / "catalog"
            build_catalog([source], catalog, ticker_map=ticker)
            args = (catalog, {"sec_income": SPEC}, ["2025-01-30", "2025-01-31"],
                    ["MSFT.US"], root / "cache")
            cold = materialize(*args)
            arrays = load_arrays(cold)
            try:
                expected = arrays["sec_income"].copy()
                self.assertTrue(np.isnan(expected[0, 0]))
                self.assertEqual(expected[1, 0], 10)
            finally:
                arrays["sec_income"]._mmap.close()
            from qlib_yahoo.sec_features import _catalog_guard, _verify_catalog
            with patch("qlib_yahoo.sec_catalog.read_catalog", wraps=read_catalog) as reader:
                _verify_catalog(catalog, force=True)
                with patch("qlib_yahoo.sec_catalog.iter_facts", side_effect=AssertionError("warm cache rescanned facts")):
                    warm = materialize(*args, force_verify=False)
                self.assertEqual(sum(call.kwargs.get("verify", False) for call in reader.call_args_list), 1)
            self.assertEqual(cold, warm)
            arrays = load_arrays(warm)
            try:
                np.testing.assert_array_equal(arrays["sec_income"], expected)
            finally:
                arrays["sec_income"]._mmap.close()

            manifest_bytes = (catalog / "manifest.json").read_bytes()
            identity = feature_identity(catalog, args[1])
            relative = read_catalog(catalog)["tables"]["num"]["files"][0]
            numeric = catalog / relative
            before = numeric.stat()
            guard = _catalog_guard(catalog)
            content = numeric.read_bytes()
            numeric.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
            os.utime(numeric, ns=(before.st_atime_ns, before.st_mtime_ns))
            self.assertEqual(numeric.stat().st_size, before.st_size)
            self.assertEqual((catalog / "manifest.json").read_bytes(), manifest_bytes)
            # Metadata identity intentionally cannot attest to current source bytes.
            self.assertEqual(feature_identity(catalog, args[1]), identity)
            # Mock the stat guard because Linux ctime cannot be restored. Even
            # identical metadata must not suppress the forced execution check.
            with patch("qlib_yahoo.sec_features._catalog_guard", return_value=guard):
                with patch("qlib_yahoo.sec_catalog.read_catalog", wraps=read_catalog) as reader:
                    _verify_catalog(catalog)
                    self.assertFalse(any(call.kwargs.get("verify", False) for call in reader.call_args_list))
                    with self.assertRaisesRegex(ValueError, "artifact differs: tables/num/"):
                        _verify_catalog(catalog, force=True)
                    self.assertTrue(any(call.kwargs.get("verify", False) for call in reader.call_args_list))
                    # Failure invalidates the cached receipt even for an internal
                    # guarded retry; known corruption cannot regain acceptance.
                    with self.assertRaisesRegex(ValueError, "artifact differs: tables/num/"):
                        _verify_catalog(catalog)
                with patch("qlib_yahoo.sec_features.load_arrays", side_effect=AssertionError("returned stale arrays")):
                    with self.assertRaisesRegex(ValueError, "artifact differs: tables/num/"):
                        materialize(*args)  # Standalone API forces verification.
            self.assertTrue((Path(cold["path"]).parent / "fields-v1" /
                             cold["field_objects"]["sec_income"] / "values.npy").is_file())

    def test_joined_real_catalog_interface(self):
        from qlib_yahoo.sec_catalog import build_catalog
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            tables = {"sub": "adsh\tcik\taccepted\tfiled\n0000000001-25-000001\t1\t2025-01-29 16:12:00.0\t20250129\n",
                      "tag": "tag\tversion\tcustom\nNetIncomeLoss\tus-gaap/2024\t0\n",
                      "num": "adsh\ttag\tversion\tuom\tqtrs\tddate\tvalue\tsegments\tcoreg\n0000000001-25-000001\tNetIncomeLoss\tus-gaap/2024\tUSD\t1\t20241231\t24108000000\t\t\n",
                      "pre": "adsh\ttag\tversion\n0000000001-25-000001\tNetIncomeLoss\tus-gaap/2024\n"}
            for table, text in tables.items():
                (source / f"{table}.txt").write_text(text)
            ticker = root / "tickers.json"
            ticker.write_text(json.dumps({"fields": ["cik", "ticker"], "data": [[1, "MSFT"]]}))
            build_catalog([source], root / "catalog", ticker_map=ticker)
            handle = materialize(root / "catalog", {"sec_income": SPEC},
                                 ["2025-01-30", "2025-01-31"], ["MSFT.US"], root / "cache")
            values = load_arrays(handle)["sec_income"]
            import numpy as np
            self.assertTrue(np.isnan(values[0, 0]))
            self.assertAlmostEqual(float(values[1, 0]) / 24108000000, 1, places=6)
            values._mmap.close()
            (root / "catalog" / "tickers.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "artifact differs"):
                materialize(root / "catalog", {"sec_income": SPEC},
                            ["2025-01-30", "2025-01-31"], ["MSFT.US"], root / "cache")
            # sqlite connection lifetimes in the catalog adapter need collection
            # before removing temporary files on Windows.
            import gc
            gc.collect()
