"""Independent field-cache reuse, integrity and legacy-handle fixtures."""
import copy
import hashlib
import importlib.util
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from qlib_yahoo import sec_features as sec

SPEC = {"tag": "Assets", "taxonomy": "us-gaap/", "unit": "USD", "qtrs": 0,
        "segments": "", "coreg": ""}


@unittest.skipUnless(importlib.util.find_spec("numpy"), "requires NumPy")
class SECFieldCacheTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.catalog = self.root / "catalog"
        self.catalog.mkdir()
        (self.catalog / "tickers.json").write_text(json.dumps({
            "fields": ["cik", "ticker"], "data": [[1, "AAA"], [2, "BBB"]]}))
        self.manifest = {"catalog_sha256": "a" * 64,
                         "numeric_materialization_supported": True,
                         "artifacts_sha256": {"tickers.json": sec._hash(self.catalog / "tickers.json")}}
        reader = patch("qlib_yahoo.sec_catalog.read_catalog", return_value=self.manifest)
        self.reader = reader.start()
        self.addCleanup(reader.stop)
        def build(catalog, spec, dates, mapping, diagnostics):
            diagnostics["ineligible_rows"] += 2
            diagnostics["blocked_winners"] += 1
            return np.full((len(dates), diagnostics["symbol_count"]),
                           {"Assets": 10, "Liabilities": 20, "Equity": 30}.get(spec["tag"], 40),
                           dtype=np.float32)
        builder = patch.object(sec, "_build_field", side_effect=build)
        self.builder = builder.start()
        self.addCleanup(builder.stop)
        self.dates = ["2025-01-31", "2025-02-03"]
        self.symbols = ["AAA", "BBB"]
        self.cache = self.root / "cache"

    def run_fields(self, bindings=None, dates=None, symbols=None):
        return sec.materialize(self.catalog, bindings or {"sec_a": SPEC},
                               dates if dates is not None else self.dates,
                               symbols if symbols is not None else self.symbols, self.cache)

    def object_path(self, handle, alias="sec_a"):
        return self.cache / "fields-v1" / handle["field_objects"][alias] / "values.npy"

    def assert_values(self, handle, expected):
        arrays = sec.load_arrays(handle)
        try:
            for alias, number in expected.items():
                self.np.testing.assert_array_equal(arrays[alias], number)
                self.assertFalse(arrays[alias].flags.writeable)
        finally:
            sec.close_arrays(arrays)

    def test_intersection_alias_rename_and_duplicate_definition_reuse(self):
        first = self.run_fields({"sec_a": SPEC, "sec_b": dict(SPEC, tag="Liabilities")})
        self.assertEqual(self.builder.call_count, 2)
        second = self.run_fields({"sec_a": SPEC, "sec_c": dict(SPEC, tag="Equity")})
        self.assertEqual(self.builder.call_count, 3)
        renamed = self.run_fields({"sec_renamed": SPEC, "sec_duplicate": SPEC})
        self.assertEqual(self.builder.call_count, 3)
        self.assertEqual(first["field_objects"]["sec_a"], second["field_objects"]["sec_a"])
        self.assertEqual(first["field_objects"]["sec_a"], renamed["field_objects"]["sec_renamed"])
        self.assertNotEqual(first["identity_sha256"], renamed["identity_sha256"])
        self.assertEqual(first["identity"], dict(sec.feature_identity(self.catalog, first["identity"]["bindings"]),
                                                dates=self.dates, symbols=self.symbols))
        self.assertEqual(len(list(self.cache.rglob("values.npy"))), 3)
        self.assertFalse(list(Path(first["path"]).glob("*.npy")))
        self.assertEqual(second["diagnostics"]["ineligible_rows"], 4)
        self.assert_values(second, {"sec_a": 10, "sec_c": 30})
        self.assert_values(renamed, {"sec_renamed": 10, "sec_duplicate": 10})

    def test_spec_policy_calendar_symbol_and_source_invalidation(self):
        first = self.run_fields()
        for key, value in (("tag", "Liabilities"), ("taxonomy", "us-gaap/2024"),
                           ("unit", "EUR"), ("qtrs", 1), ("segments", "division"),
                           ("coreg", "child")):
            with self.subTest(key=key):
                handle = self.run_fields({"sec_a": dict(SPEC, **{key: value})})
                self.assertNotEqual(handle["field_objects"], first["field_objects"])
        self.run_fields(dates=["2025-01-31"])
        self.run_fields(dates=["2025-01-31", "2025-02-04"])
        self.run_fields(symbols=list(reversed(self.symbols)))
        self.run_fields(symbols=["AAA"])
        with patch.dict(sec.POLICY, {"expiry_months": 16}):
            self.run_fields()
        self.manifest["catalog_sha256"] = "b" * 64
        self.run_fields()
        self.manifest["artifacts_sha256"]["tickers.json"] = "c" * 64
        self.run_fields()
        self.assertEqual(self.builder.call_count, 14)

    def test_corruption_is_rejected_for_same_and_new_alias_sets(self):
        handle = self.run_fields()
        path = self.object_path(handle)
        path.write_bytes(path.read_bytes()[:-1] + b"X")
        for bindings in ({"sec_a": SPEC}, {"sec_renamed": SPEC}):
            with self.subTest(bindings=bindings), self.assertRaisesRegex(ValueError, "hash mismatch"):
                self.run_fields(bindings)
        self.assertEqual(self.builder.call_count, 1)

    def test_shape_dtype_and_reference_tampering(self):
        handle = self.run_fields()
        for array in (self.np.zeros((3, 2), dtype="float32"), self.np.zeros((2, 2), dtype="float64")):
            with self.subTest(dtype=str(array.dtype), shape=array.shape):
                path = self.object_path(handle)
                self.np.save(path, array)
                receipt_path = path.parent / "field.json"
                receipt = json.loads(receipt_path.read_text())
                receipt["sha256"] = sec._hash(path)
                receipt_path.write_text(json.dumps(receipt))
                changed = copy.deepcopy(handle)
                changed["filehashes"]["sec_a"] = receipt["sha256"]
                with self.assertRaisesRegex(ValueError, "shape/dtype"):
                    sec.load_arrays(changed)
        for reference in ("../escape", "/tmp/escape", "0" * 64):
            changed = copy.deepcopy(handle)
            changed["field_objects"]["sec_a"] = reference
            with self.assertRaisesRegex(ValueError, "object identity"):
                sec.load_arrays(changed)

    def test_old_saved_handle_and_historical_directory_remain_untouched(self):
        identity = dict(sec.feature_identity(self.catalog, {"sec_a": SPEC}),
                        dates=self.dates, symbols=self.symbols)
        target = self.cache / sec._digest(identity)
        target.mkdir(parents=True)
        self.np.save(target / "sec_a.npy", self.np.full((2, 2), 7, dtype="float32"))
        handle = {"path": str(target), "identity": identity, "identity_sha256": sec._digest(identity),
                  "fields": ["sec_a"], "filehashes": {"sec_a": sec._hash(target / "sec_a.npy")},
                  "diagnostics": {"historical": True}}
        (target / "handle.json").write_text(json.dumps(handle))
        before = {p.name: p.read_bytes() for p in target.iterdir()}
        self.assert_values(handle, {"sec_a": 7})
        self.assertEqual(self.run_fields(), handle)
        self.run_fields({"sec_new": SPEC})
        self.assertEqual(before, {p.name: p.read_bytes() for p in target.iterdir()})
        self.assertEqual(self.builder.call_count, 1)

    def test_failed_load_closes_prior_and_invalid_maps(self):
        handle = self.run_fields({"sec_a": SPEC, "sec_b": dict(SPEC, tag="Liabilities")})
        path = self.object_path(handle, "sec_b")
        original_load = self.np.load
        original = path.read_bytes()
        for failure in ("hash", "shape/dtype"):
            path.write_bytes(original)
            changed = copy.deepcopy(handle)
            opened = []
            if failure == "hash":
                changed["filehashes"]["sec_b"] = "0" * 64
            else:
                self.np.save(path, self.np.zeros((3, 2), dtype="float32"))
                receipt_path = path.parent / "field.json"
                receipt = json.loads(receipt_path.read_text())
                receipt["sha256"] = sec._hash(path)
                receipt_path.write_text(json.dumps(receipt))
                changed["filehashes"]["sec_b"] = receipt["sha256"]
            def capture(*args, **kwargs):
                values = original_load(*args, **kwargs)
                opened.append(values)
                return values
            with self.subTest(failure=failure), patch("numpy.load", side_effect=capture):
                with self.assertRaisesRegex(ValueError, failure):
                    sec.load_arrays(changed)
            self.assertEqual(len(opened), 1 if failure == "hash" else 2)
            self.assertTrue(all(values._mmap.closed for values in opened))

    def test_panel_mutation_between_hash_and_map_is_rejected(self):
        handle = self.run_fields()
        path = self.object_path(handle)
        original = sec._hash
        def mutate_after_hash(candidate):
            result = original(candidate)
            if Path(candidate) == path:
                data = path.read_bytes()
                path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
            return result
        opened = []
        original_load = self.np.load
        def capture(*args, **kwargs):
            values = original_load(*args, **kwargs)
            opened.append(values)
            return values
        with patch.object(sec, "_hash", side_effect=mutate_after_hash), patch("numpy.load", side_effect=capture):
            with self.assertRaisesRegex(ValueError, "changed during loading"):
                sec.load_arrays(handle)
        self.assertEqual(len(opened), 1)
        self.assertTrue(opened[0]._mmap.closed)

    def test_field_receipt_and_handle_identity_tampering(self):
        handle = self.run_fields()
        changed = copy.deepcopy(handle)
        changed["identity"]["bindings"]["sec_a"]["unit"] = "EUR"
        with self.assertRaisesRegex(ValueError, "handle identity"):
            sec.load_arrays(changed)
        receipt_path = self.object_path(handle).parent / "field.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["identity"]["spec"]["unit"] = "EUR"
        receipt_path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "field identity"):
            self.run_fields({"sec_new": SPEC})

    def test_integrity_precedes_warm_hit_and_source_mutation_prevents_publish(self):
        self.run_fields()
        with patch.object(sec, "_verify_catalog", side_effect=ValueError("source corrupted")):
            with patch.object(sec, "load_arrays", side_effect=AssertionError("hit before verification")):
                with self.assertRaisesRegex(ValueError, "source corrupted"):
                    self.run_fields()
        original = self.builder.side_effect
        def mutate(*args):
            values = original(*args)
            (self.catalog / "tickers.json").write_text("{}")
            return values
        self.builder.side_effect = mutate
        with self.assertRaisesRegex(ValueError, "changed during materialization"):
            self.run_fields({"sec_b": dict(SPEC, tag="Liabilities")})
        self.assertEqual(len(list(self.cache.rglob("values.npy"))), 1)
        self.assertFalse(list(self.cache.rglob(".f-*")))

    def test_concurrent_field_and_handle_publishers(self):
        barrier = threading.Barrier(2)
        original = self.builder.side_effect
        def synchronized(*args):
            barrier.wait(timeout=10)
            return original(*args)
        self.builder.side_effect = synchronized
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(self.run_fields) for _ in range(2)]
            handles = [future.result(timeout=20) for future in futures]
        self.assertEqual(handles[0], handles[1])
        self.assert_values(handles[0], {"sec_a": 10})
        self.assertEqual(len(list(self.cache.rglob("values.npy"))), 1)
        self.assertFalse(list(self.cache.rglob(".f-*")))
        self.assertFalse(list(self.cache.glob(".sec-selected-*")))

    def test_all_owned_maps_are_closed_before_publication(self):
        opened = []
        original_load = self.np.load
        original_publish = sec._publish_directory
        publications = []
        def capture(*args, **kwargs):
            values = original_load(*args, **kwargs)
            opened.append(values)  # Retain references so GC cannot mask a leak.
            return values
        def publish(staging, target):
            self.assertTrue(opened)
            self.assertTrue(all(values._mmap.closed for values in opened))
            publications.append(target)
            return original_publish(staging, target)
        with patch("numpy.load", side_effect=capture), patch.object(sec, "_publish_directory", side_effect=publish):
            self.run_fields({"sec_a": SPEC, "sec_b": dict(SPEC, tag="Liabilities")})
            self.run_fields({"sec_a": SPEC, "sec_c": dict(SPEC, tag="Equity")})
        self.assertEqual(len(publications), 5)
        self.assertTrue(all(values._mmap.closed for values in opened))

    def test_corrupt_competing_publisher_is_checked_and_never_replaced(self):
        original = sec._publish_directory
        poisoned = []
        def compete(staging, target):
            if target.parent.name == "fields-v1":
                shutil.copytree(staging, target)
                path = target / "values.npy"
                path.write_bytes(path.read_bytes()[:-1] + b"X")
                poisoned.append((path, path.read_bytes()))
            original(staging, target)
        with patch.object(sec, "_publish_directory", side_effect=compete):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                self.run_fields()
        self.assertEqual(len(poisoned), 1)
        self.assertEqual(poisoned[0][0].read_bytes(), poisoned[0][1])
        self.assertFalse(list(self.cache.rglob(".f-*")))
        self.assertFalse(list(self.cache.glob(".sec-selected-*")))

    def test_corrupt_winner_during_publication_backoff_is_rejected(self):
        original_rename = os.rename
        denied = PermissionError(13, "temporarily held by another reader")
        denied.winerror = 5
        pending = []
        poisoned = []
        def rename(staging, target):
            if target.parent.name == "fields-v1":
                pending.append((staging, target))
                raise denied
            return original_rename(staging, target)
        def other_publisher(delay):
            staging, target = pending[-1]
            shutil.copytree(staging, target)
            path = target / "values.npy"
            path.write_bytes(path.read_bytes()[:-1] + b"X")
            poisoned.append((path, path.read_bytes()))
        with patch.object(sec, "os", wraps=os) as platform, patch("time.sleep", side_effect=other_publisher):
            platform.name = "nt"
            platform.rename.side_effect = rename
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                self.run_fields()
            self.assertEqual(platform.rename.call_count, 1)
        self.assertEqual(len(poisoned), 1)
        self.assertEqual(poisoned[0][0].read_bytes(), poisoned[0][1])
        self.assertFalse(list(self.cache.rglob(".f-*")))
        self.assertFalse(list(self.cache.glob(".sec-selected-*")))

    def test_cache_redirect_rejected(self):
        handle = self.run_fields()
        path = self.object_path(handle)
        destination = self.root / "elsewhere.npy"
        destination.write_bytes(path.read_bytes())
        path.unlink()
        try:
            path.symlink_to(destination)
        except OSError as exc:
            self.skipTest(f"symlink unavailable: {exc}")
        with self.assertRaisesRegex(ValueError, "redirected"):
            sec.load_arrays(handle)


class SECFieldPublishTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.staging = self.root / "staging"
        self.staging.mkdir()
        (self.staging / "field.json").write_text("complete receipt")
        self.target = self.root / "published"

    @unittest.skipUnless(os.name == "nt", "requires Windows file sharing semantics")
    def test_open_file_reproduces_publication_denial_then_closure_allows_rename(self):
        # A real no-delete-share file handle produces the same OS-level failure.
        with (self.staging / "field.json").open("rb"):
            with self.assertRaises(OSError) as raised:
                os.rename(self.staging, self.target)
        self.assertIn(raised.exception.winerror, (5, 32))
        self.assertFalse(self.target.exists())
        os.rename(self.staging, self.target)
        self.assertEqual((self.target / "field.json").read_text(), "complete receipt")

    @unittest.skipUnless(os.name == "nt", "requires Windows file sharing semantics")
    def test_publication_retries_real_temporary_file_lock(self):
        original = os.rename
        failures = []
        def rename(staging, target):
            if not failures:
                with (staging / "field.json").open("rb"):
                    try:
                        return original(staging, target)
                    except OSError as exc:
                        failures.append(exc.winerror)
                        raise  # File closes before publisher retries.
            return original(staging, target)
        with patch.object(sec, "os", wraps=os) as platform:
            platform.name = "nt"
            platform.rename.side_effect = rename
            sec._publish_directory(self.staging, self.target)
            self.assertEqual(platform.rename.call_count, 2)
        self.assertEqual(len(failures), 1)
        self.assertIn(failures[0], (5, 32))
        self.assertEqual((self.target / "field.json").read_text(), "complete receipt")

    def test_both_windows_lock_errors_retry_with_bounded_backoff(self):
        for code in (5, 32):
            with self.subTest(winerror=code):
                denied = PermissionError(13, "sharing/access denied")
                denied.winerror = code
                with patch.object(sec, "os", wraps=os) as platform, patch("time.sleep") as sleep:
                    platform.name = "nt"
                    platform.rename.side_effect = denied
                    with self.assertRaises(OSError) as raised:
                        sec._publish_directory(self.staging, self.target)
                    self.assertIs(raised.exception, denied)
                    self.assertEqual(platform.rename.call_count, 5)
                    self.assertEqual([call.args[0] for call in sleep.call_args_list], [0.05, 0.1, 0.2, 0.4])
                self.assertFalse(self.target.exists())
                self.assertEqual((self.staging / "field.json").read_text(), "complete receipt")

    def test_other_errors_and_non_windows_fail_immediately(self):
        for name, code in (("nt", None), ("nt", 112), ("nt", 2), ("posix", 5)):
            with self.subTest(platform=name, winerror=code):
                failure = OSError(13, "unrelated failure")
                if code is not None:
                    failure.winerror = code
                with patch.object(sec, "os", wraps=os) as platform, patch("time.sleep") as sleep:
                    platform.name = name
                    platform.rename.side_effect = failure
                    with self.assertRaises(OSError) as raised:
                        sec._publish_directory(self.staging, self.target)
                    self.assertIs(raised.exception, failure)
                    self.assertEqual(platform.rename.call_count, 1)
                    sleep.assert_not_called()

    def test_existing_winner_is_not_overwritten_or_retried(self):
        self.target.mkdir()
        (self.target / "field.json").write_text("other publisher")
        with patch.object(sec, "os", wraps=os) as platform, patch("time.sleep") as sleep:
            sec._publish_directory(self.staging, self.target)
            platform.rename.assert_not_called()
            sleep.assert_not_called()
        self.assertEqual((self.target / "field.json").read_text(), "other publisher")
        self.assertEqual((self.staging / "field.json").read_text(), "complete receipt")


if __name__ == "__main__":
    unittest.main()
