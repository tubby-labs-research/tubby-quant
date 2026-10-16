"""Regression checks for strict and pre-tree-hash Qlib dataset manifests."""

import json
import struct
import tempfile
import unittest
from pathlib import Path

from qlib_yahoo.dataset import DatasetError, sha256, tree_sha256
from qlib_yahoo.dataset_integrity import verify_dataset


def _legacy_snapshot(root: Path) -> tuple[Path, dict]:
    dataset = root / "snapshot"
    qlib = dataset / "qlib_data"
    for path in (dataset / "raw", dataset / "normalized", qlib / "calendars",
                 qlib / "instruments", qlib / "features" / "aaa", qlib / "features" / "spy"):
        path.mkdir(parents=True)
    (qlib / "calendars" / "day.txt").write_text("2024-01-02\n2024-01-03\n2024-01-04\n")
    (qlib / "instruments" / "all.txt").write_text(
        "AAA\t2024-01-02\t2024-01-04\nSPY\t2024-01-02\t2024-01-04\n")
    fields = ("open", "high", "low", "close", "volume", "factor")
    for symbol in ("AAA", "SPY"):
        (dataset / "raw" / f"{symbol}.csv").write_text("raw-source\n")
        lines = ["date,symbol," + ",".join(fields)]
        for date, value in (("2024-01-02", "1.2"), ("2024-01-04", "2.3")):
            lines.append(f"{date},{symbol}," + ",".join([value] * len(fields)))
        (dataset / "normalized" / f"{symbol}.csv").write_text("\n".join(lines) + "\n")
        for field in fields:
            (qlib / "features" / symbol.lower() / f"{field}.day.bin").write_bytes(
                struct.pack("<4f", 0, 1.2, float("nan"), 2.3))
    manifest = {
        "schema_version": 1, "status": "ready", "symbols": ["AAA"], "benchmark": "SPY",
        "sessions": 3,
        "raw_sha256": {name: sha256(dataset / "raw" / name) for name in ("AAA.csv", "SPY.csv")},
        "normalized_sha256": {name: sha256(dataset / "normalized" / name) for name in ("AAA.csv", "SPY.csv")},
        "qlib_calendar_sha256": sha256(qlib / "calendars" / "day.txt"),
        "qlib_instruments_sha256": sha256(qlib / "instruments" / "all.txt"),
    }
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    return dataset, manifest


class DatasetIntegrityTests(unittest.TestCase):
    def test_legacy_verifies_sources_bins_and_provenance_without_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest = _legacy_snapshot(Path(temporary))
            before = tree_sha256(dataset)
            result = verify_dataset(dataset, manifest)
            self.assertEqual(result["mode"], "legacy_source_csv_calendar_bin_float32_v1")
            self.assertEqual(result["qlib_binary_file_count"], 14)
            self.assertEqual(tree_sha256(dataset), before)

    def test_legacy_rejects_bin_tamper_even_when_source_hashes_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest = _legacy_snapshot(Path(temporary))
            path = dataset / "qlib_data" / "features" / "aaa" / "close.day.bin"
            path.write_bytes(struct.pack("<4f", 0, 1.2, 0, 2.3))
            with self.assertRaisesRegex(DatasetError, "bin differs from normalized CSV"):
                verify_dataset(dataset, manifest)

    def test_legacy_rejects_source_and_calendar_tamper(self):
        for relative, message in (("raw/AAA.csv", "raw_sha256"),
                                  ("normalized/AAA.csv", "normalized_sha256"),
                                  ("qlib_data/calendars/day.txt", "qlib_calendar_sha256")):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as temporary:
                dataset, manifest = _legacy_snapshot(Path(temporary))
                with (dataset / relative).open("ab") as handle:
                    handle.write(b"tamper")
                with self.assertRaisesRegex(DatasetError, message):
                    verify_dataset(dataset, manifest)

    def test_strict_tree_hash_still_rejects_bin_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest = _legacy_snapshot(Path(temporary))
            manifest["qlib_binary_tree_sha256"] = tree_sha256(dataset / "qlib_data")[0]
            self.assertEqual(verify_dataset(dataset, manifest)["mode"], "recorded_binary_tree_sha256")
            path = dataset / "qlib_data" / "features" / "aaa" / "close.day.bin"
            with path.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(DatasetError, "binary tree differs"):
                verify_dataset(dataset, manifest)

    def test_folder_metadata_from_finder_or_explorer_is_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest = _legacy_snapshot(Path(temporary))
            manifest["qlib_binary_tree_sha256"] = tree_sha256(dataset / "qlib_data")[0]
            for name in (".DS_Store", "features/aaa/Thumbs.db", "features/._close.day.bin", "desktop.ini"):
                (dataset / "qlib_data" / name).write_bytes(b"metadata")
            self.assertEqual(verify_dataset(dataset, manifest)["mode"], "recorded_binary_tree_sha256")

    def test_failed_manifest_cannot_run_with_copied_valid_tree_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest = _legacy_snapshot(Path(temporary))
            manifest["qlib_binary_tree_sha256"] = tree_sha256(dataset / "qlib_data")[0]
            manifest["status"] = "failed"
            with self.assertRaisesRegex(DatasetError, "status must be ready"):
                verify_dataset(dataset, manifest)

    def test_malformed_manifest_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset, manifest = _legacy_snapshot(Path(temporary))
            for change, message in (({"schema_version": 2}, "requires schema_version 1"),
                                    ({"raw_sha256": {}}, "must list exactly"),
                                    ({"qlib_binary_file_count": 14}, "incomplete Qlib binary-tree provenance"),
                                    ({"qlib_binary_tree_sha256": "broken"}, "invalid qlib_binary_tree_sha256")):
                with self.subTest(change=change), self.assertRaisesRegex(DatasetError, message):
                    verify_dataset(dataset, {**manifest, **change})


if __name__ == "__main__":
    unittest.main()
