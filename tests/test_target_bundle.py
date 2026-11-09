"""Sealing, validation and unchanged decision-date alignment, without Qlib."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

from qlib_yahoo.dataset import DatasetError
from qlib_yahoo.target_bundle import align_bundle, load_bundle, write_bundle


def metadata(digest="a" * 64):
    return {"dataset_manifest_sha256": digest, "price_identity": {"kind": "synthetic"},
            "candidate_result_digests": ["b" * 64], "attempt_ids": ["attempt-1"],
            "weight_policy": "lab_close_targets_v1", "assumptions": ["synthetic only"]}


@unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("pandas"), "array dependencies optional")
class TargetBundleTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dates = ["2024-01-05", "2024-01-08", "2024-01-09"]
        self.symbols = ["B", "A"]  # supplied order is meaningful, not alphabetized
        self.weights = np.array([[.25, -.125], [0, 0], [-.125, .25]], dtype="float32")

    def write(self, **changes):
        options = dict(dates=self.dates, symbols=self.symbols, weights=self.weights, metadata=metadata())
        options.update(changes)
        return write_bundle(self.root / "bundle", **options)

    def reseal(self, envelope, change):
        path = Path(envelope["path"]) / "manifest.json"
        manifest = json.loads(path.read_text())
        change(manifest)
        encoded = json.dumps(manifest).encode()
        path.write_bytes(encoded)
        return {**envelope, "sha256": hashlib.sha256(encoded).hexdigest()}

    def test_roundtrip_preserves_order_zeros_and_undergross(self):
        envelope = self.write()
        loaded = load_bundle(envelope)
        self.assertEqual(loaded["dates"], self.dates)
        self.assertEqual(loaded["symbols"], self.symbols)
        self.np.testing.assert_array_equal(loaded["weights"], self.weights)
        self.assertFalse(loaded["weights"].flags.writeable)
        self.assertEqual(loaded["identity"]["execution_timing"], "next_session_open")
        with self.assertRaises(DatasetError):
            self.write()

    def test_float32_rounding_only_reduces_cap_overflow(self):
        values = self.np.array([[.4, .6]] * 3, dtype="float32")
        loaded = load_bundle(self.write(weights=values))
        self.assertGreater(loaded["identity"]["precision"]["gross_cap_ulp_reductions"], 0)
        self.assertTrue((loaded["weights"].astype(float).sum(axis=1) <= 1 + 1e-12).all())
        self.assertTrue((loaded["weights"] <= values).all())

    def test_invalid_dates_symbols_matrix_and_binding(self):
        cases = [dict(dates=[]), dict(dates=self.dates[::-1]), dict(dates=[self.dates[0]] * 3),
                 dict(dates=["not-a-day"] * 3), dict(symbols=["A", "A"]), dict(symbols=[]),
                 dict(weights=self.np.zeros((3, 3))), dict(weights=self.weights.astype(int)),
                 dict(weights=self.np.full((3, 2), self.np.nan)),
                 dict(weights=self.np.full((3, 2), .51)), dict(metadata={}),
                 dict(metadata={**metadata(), "dataset_manifest_sha256": "bad"}),
                 dict(metadata={**metadata(), "attempt_ids": []}),
                 dict(metadata={**metadata(), "assumptions": [float("nan")]})]
        for change in cases:
            with self.subTest(change=change), self.assertRaises(DatasetError):
                self.write(**change)
        self.assertFalse((self.root / "bundle").exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_envelope_and_payload_tampering(self):
        envelope = self.write()
        for bad in (None, {}, {"path": envelope["path"]}, {**envelope, "sha256": "0" * 64},
                    {**envelope, "path": "relative"}, {**envelope, "extra": 1}):
            with self.subTest(bad=bad), self.assertRaises(DatasetError):
                load_bundle(bad)
        path = Path(envelope["path"]) / "weights.npz"
        payload = path.read_bytes()
        path.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
        with self.assertRaisesRegex(DatasetError, "checksum"):
            load_bundle(envelope)

    def test_resealed_schema_shape_dtype_and_size_rejected(self):
        envelope = self.write()
        path = Path(envelope["path"]) / "manifest.json"
        original = path.read_bytes()
        changes = [lambda m: m.update(schema_version=True), lambda m: m.update(input_kind="raw_scores"),
                   lambda m: m.update(execution_timing="same_close"), lambda m: m["weights"].update(shape=[2, 3]),
                   lambda m: m["weights"].update(dtype="object"), lambda m: m["weights"].update(size_bytes=2),
                   lambda m: m["weights"].update(file="../weights.npz")]
        for change in changes:
            path.write_bytes(original)
            bad = self.reseal(envelope, change)
            with self.assertRaises(DatasetError):
                load_bundle(bad)

    def test_resealed_invalid_npy_header_and_values_rejected(self):
        envelope = self.write()
        for array in (self.weights.astype("float64"), self.np.zeros((2, 3), dtype="float32"),
                      self.np.full((3, 2), .75, dtype="float32"), self.weights.astype(object)):
            path = Path(envelope["path"]) / "weights.npz"
            self.np.savez(path, weights=array)
            payload = path.read_bytes()
            bad = self.reseal(envelope, lambda m: m["weights"].update(
                sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload)))
            with self.assertRaises(DatasetError):
                load_bundle(bad)

    def test_links_rejected(self):
        envelope = self.write()
        path = Path(envelope["path"]) / "weights.npz"
        other = self.root / "original.npz"
        path.rename(other)
        os.link(other, path)
        with self.assertRaises(DatasetError):
            load_bundle(envelope)

    def test_calendar_coverage_membership_and_zero_nonmembers(self):
        import pandas as pd
        bundle = load_bundle(self.write())
        options = dict(sessions=pd.to_datetime([*self.dates, "2024-01-10"]),
                       signal_days=pd.to_datetime(self.dates), symbols=self.symbols, eligible=self.symbols)
        result = align_bundle(bundle, **options)
        self.np.testing.assert_array_equal(result.to_numpy(), self.weights.ravel())
        self.assertEqual(result.index.get_level_values("datetime")[0], pd.Timestamp(self.dates[0]))
        for changed in (dict(signal_days=pd.to_datetime(["2024-01-10"])),
                        dict(sessions=pd.to_datetime([self.dates[0], *self.dates])),
                        dict(signal_days=pd.to_datetime(self.dates[::-1])),
                        dict(sessions=pd.to_datetime(self.dates[1:])), dict(symbols=["A"]), dict(eligible=["A"]),
                        dict(spans={"B": [(pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-09"))]})):
            with self.subTest(changed=changed), self.assertRaises(DatasetError):
                align_bundle(bundle, **{**options, **changed})
        # The zero middle row is valid even when nobody belongs to the universe.
        spans = {symbol: [(pd.Timestamp(self.dates[0]), pd.Timestamp(self.dates[0])),
                          (pd.Timestamp(self.dates[2]), pd.Timestamp(self.dates[2]))] for symbol in self.symbols}
        self.np.testing.assert_array_equal(align_bundle(bundle, **options, spans=spans), result)


if __name__ == "__main__":
    unittest.main()
