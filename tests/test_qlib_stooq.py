"""Offline Stooq parser and snapshot boundary checks."""

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qlib_yahoo import DEFAULT_DATA_ROOT
from qlib_yahoo.dataset import DatasetError, sha256
from qlib_yahoo.stooq import HEADER, _convert, _dump_script, _safe_roots, _source_files, prepare_stooq


REAL_DUMPER = Path(DEFAULT_DATA_ROOT) / "upstream-qlib-v0.9.7" / "scripts" / "dump_bin.py"


class StooqTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # Resolve: macOS temp folders live behind the /var -> /private/var symlink.
        self.root = Path(self.tmp.name).resolve()
        self.source = self.root / "archive" / "Stooq_US_Hist"
        self.output = self.root / "qlib"
        self.folder = self.source / "data" / "daily" / "us" / "nasdaq stocks" / "1"
        self.folder.mkdir(parents=True)
        self.output.mkdir()

    def file(self, symbol="AAA", lines=None, bucket=None):
        folder = bucket or self.folder
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{symbol.lower()}.us.txt"
        rows = lines if lines is not None else [
            [f"{symbol}.US", "D", "20240102", "000000", "10", "11", "9", "10", "100", "0"],
            [f"{symbol}.US", "D", "20240103", "000000", "12", "13", "11", "12", "0", "0"],
            [f"{symbol}.US", "D", "20240104", "000000", "14", "15", "13", "14", "150", "0"],
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            writer.writerows(rows)
        return path

    def test_inclusive_exclusive_zero_volume_and_source_immutable(self):
        path = self.file()
        digest = sha256(path)
        target = self.root / "converted.csv"
        count, zero, days = _convert(path, target, "AAA", __import__("datetime").date(2024, 1, 2),
                                     __import__("datetime").date(2024, 1, 4))
        self.assertEqual((count, zero, days), (2, 1, {"2024-01-02", "2024-01-03"}))
        with target.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["close"], "10.0")
        self.assertEqual(rows[0]["factor"], "1")
        self.assertEqual(rows[1]["close"], "nan")
        self.assertEqual(sha256(path), digest)

    def test_rejects_bad_rows_with_source_and_line(self):
        base = ["AAA.US", "D", "20240102", "000000", "10", "11", "9", "10", "100", "0"]
        mutations = {
            "ticker": (0, "BBB.US"), "period": (1, "W"), "date": (2, "20240231"),
            "time": (3, "120000"), "nonfinite": (4, "nan"), "ohlc": (5, "8"),
            "volume": (8, "-1"), "openint": (9, "1"),
        }
        for name, (position, value) in mutations.items():
            with self.subTest(name=name):
                row = base.copy()
                row[position] = value
                path = self.file(lines=[row])
                with self.assertRaisesRegex(DatasetError, r"aaa.us.txt: row 2:"):
                    _convert(path, self.root / "converted.csv", "AAA", __import__("datetime").date(2024, 1, 1),
                             __import__("datetime").date(2025, 1, 1))
        path = self.file(lines=[base, base])
        with self.assertRaisesRegex(DatasetError, "duplicate or unsorted"):
            _convert(path, self.root / "converted.csv", "AAA", __import__("datetime").date(2024, 1, 1),
                     __import__("datetime").date(2025, 1, 1))

    def test_out_of_range_economic_anomaly_is_not_rejected(self):
        old = ["AAA.US", "D", "19900102", "000000", "0", "0", "0", "0", "-5", "0"]
        current = ["AAA.US", "D", "20240102", "000000", "10", "11", "9", "10", "100", "0"]
        path = self.file(lines=[old, current])
        count, _, _ = _convert(path, self.root / "converted.csv", "AAA", __import__("datetime").date(2024, 1, 1),
                               __import__("datetime").date(2025, 1, 1))
        self.assertEqual(count, 1)

    def test_duplicate_and_empty_source_files(self):
        self.file()
        duplicate_bucket = self.source / "data" / "daily" / "us" / "nyse stocks" / "2"
        self.file(bucket=duplicate_bucket)
        with self.assertRaisesRegex(DatasetError, "ambiguous duplicate"):
            _source_files(self.source, {"AAA"})
        self.file("BBB", lines=[])
        with self.assertRaisesRegex(DatasetError, "no bars"):
            _convert(self.folder / "bbb.us.txt", self.root / "converted.csv", "BBB",
                     __import__("datetime").date(2024, 1, 1), __import__("datetime").date(2025, 1, 1))

    def test_output_protection_and_symbol_validation(self):
        self.file()
        with self.assertRaisesRegex(DatasetError, "overlaps read-only"):
            prepare_stooq(self.source, self.source.parent / "output", ("AAA",), "2024-01-01", "2025-01-01")
        for symbols in (("CON",), ("../AAA",), ("AAA", "AAA")):
            with self.subTest(symbols=symbols), self.assertRaises(DatasetError):
                prepare_stooq(self.source, self.output, symbols, "2024-01-01", "2025-01-01")

    def test_symbol_input_is_trimmed_and_uppercased(self):
        self.file()
        self.file("SPY")
        def inspect_symbols(_source, wanted):
            self.assertEqual(wanted, {"AAA"})
            raise DatasetError("normalized symbols reached source selection")
        with patch("qlib_yahoo.stooq._dump_script", return_value=self.root / "dump.py"), \
             patch("qlib_yahoo.stooq._source_files", side_effect=inspect_symbols):
            with self.assertRaisesRegex(DatasetError, "normalized symbols reached"):
                prepare_stooq(self.source, self.output, (" aaa ",), "2024-01-02", "2024-01-04")

    def test_source_group_redirect_is_rejected(self):
        group = self.source / "data" / "daily" / "us" / "nyse stocks"
        try:
            group.symlink_to(self.folder.parent, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("directory symlinks unavailable")
        with self.assertRaisesRegex(DatasetError, "redirected"):
            _source_files(self.source, {"AAA"})

    def test_source_root_junction_flag_is_rejected_before_resolve(self):
        if not hasattr(Path, "is_junction"):
            self.skipTest("junction API unavailable")
        with patch.object(Path, "is_junction", autospec=True, side_effect=lambda path: path == self.source):
            with patch.object(Path, "resolve", autospec=True, side_effect=AssertionError("resolved root")):
                with self.assertRaisesRegex(DatasetError, "source root is redirected"):
                    _safe_roots(self.source, self.output)

    def test_source_root_symlink_is_rejected_if_available(self):
        alias = self.root / "alias"
        try:
            alias.symlink_to(self.source, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("directory symlinks unavailable")
        with self.assertRaisesRegex(DatasetError, "source root is redirected"):
            _safe_roots(alias, self.output)

    def test_existing_latest_temp_redirect_is_untouched(self):
        self.file()
        self.file("SPY")
        victim = self.root / "victim.txt"
        victim.write_text("retain", encoding="utf-8")
        old_temp = self.output / "latest.json.tmp"
        try:
            old_temp.symlink_to(victim)
        except (OSError, NotImplementedError):
            self.skipTest("file symlinks unavailable")
        from qlib_yahoo.stooq import _write_json_atomically
        _write_json_atomically(self.output / "latest.json", {"dataset": "new-snapshot"})
        self.assertEqual(victim.read_text(encoding="utf-8"), "retain")
        self.assertTrue(old_temp.is_symlink())
        self.assertEqual(json.loads((self.output / "latest.json").read_text()), {"dataset": "new-snapshot"})

    @unittest.skipUnless(REAL_DUMPER.is_file(), "local Qlib checkout unavailable")
    def test_real_pinned_checkout_is_verified(self):
        self.assertEqual(_dump_script(Path(DEFAULT_DATA_ROOT)), REAL_DUMPER)

    def test_mock_dump_failure_keeps_latest_and_failed_manifest(self):
        self.file()
        self.file("SPY")
        latest = self.output / "latest.json"
        latest.write_text('{"dataset":"old"}', encoding="utf-8")
        with patch("qlib_yahoo.stooq._dump_script", return_value=self.root / "dump.py"), \
             patch("qlib_yahoo.stooq.subprocess.run", side_effect=RuntimeError("dump crashed")):
            with self.assertRaisesRegex(RuntimeError, "dump crashed"):
                prepare_stooq(self.source, self.output, ("AAA",), "2024-01-02", "2024-01-04", make_latest=True)
        snapshot = next((self.output / "datasets").iterdir())
        self.assertEqual(json.loads((snapshot / "failed_manifest.json").read_text())["status"], "failed")
        self.assertFalse((snapshot / "manifest.json").exists())
        self.assertEqual(latest.read_text(encoding="utf-8"), '{"dataset":"old"}')

    def test_initial_progress_write_failure_is_marked_failed(self):
        self.file()
        self.file("SPY")
        from qlib_yahoo import stooq
        real_write = stooq._write_json_atomically
        failed_once = False

        def fail_first_write(path, value):
            nonlocal failed_once
            if path.name == "progress.json" and not failed_once:
                failed_once = True
                raise OSError("initial progress denied")
            return real_write(path, value)

        with patch("qlib_yahoo.stooq._dump_script", return_value=self.root / "dump.py"), \
             patch("qlib_yahoo.stooq._write_json_atomically", side_effect=fail_first_write):
            with self.assertRaisesRegex(OSError, "initial progress denied"):
                prepare_stooq(self.source, self.output, ("AAA",), "2024-01-02", "2024-01-04", make_latest=True)
        snapshot = next((self.output / "datasets").iterdir())
        self.assertEqual(json.loads((snapshot / "failed_manifest.json").read_text())["status"], "failed")
        self.assertEqual(json.loads((snapshot / "progress.json").read_text())["status"], "failed")
        self.assertFalse((snapshot / "manifest.json").exists())
        self.assertFalse((self.output / "latest.json").exists())

    def test_partial_dump_cannot_be_ready(self):
        self.file()
        self.file("SPY")

        def partial_dump(args, **_kwargs):
            qlib = Path(args[args.index("--qlib_dir") + 1])
            (qlib / "calendars").mkdir()
            (qlib / "instruments").mkdir()
            (qlib / "calendars" / "day.txt").write_text("2024-01-02\n2024-01-03\n", encoding="utf-8")
            (qlib / "instruments" / "all.txt").write_text(
                "AAA\t2024-01-02\t2024-01-03\n", encoding="utf-8")

        with patch("qlib_yahoo.stooq._dump_script", return_value=self.root / "dump.py"), \
             patch("qlib_yahoo.stooq.subprocess.run", side_effect=partial_dump):
            with self.assertRaisesRegex(DatasetError, "feature bin inventory"):
                prepare_stooq(self.source, self.output, ("AAA",), "2024-01-02", "2024-01-04", make_latest=True)
        snapshot = next((self.output / "datasets").iterdir())
        self.assertFalse((snapshot / "manifest.json").exists())
        self.assertEqual(json.loads((snapshot / "failed_manifest.json").read_text())["status"], "failed")
        self.assertFalse((self.output / "latest.json").exists())

    def test_pointer_failure_marks_main_manifest_failed(self):
        self.file()
        self.file("SPY")
        if not (importlib.util.find_spec("qlib") and REAL_DUMPER.is_file()):
            self.skipTest("local Qlib 0.9.7 dumper unavailable")
        from qlib_yahoo import stooq
        real_write = stooq._write_json_atomically

        def fail_pointer(path, value):
            if path.name == "latest.json":
                raise OSError("pointer denied")
            return real_write(path, value)

        with patch("qlib_yahoo.stooq._dump_script", return_value=REAL_DUMPER), \
             patch("qlib_yahoo.stooq._write_json_atomically", side_effect=fail_pointer):
            with self.assertRaisesRegex(OSError, "pointer denied"):
                prepare_stooq(self.source, self.output, ("AAA",), "2024-01-02", "2024-01-05", make_latest=True)
        snapshot = next((self.output / "datasets").iterdir())
        self.assertEqual(json.loads((snapshot / "manifest.json").read_text())["status"], "failed")
        self.assertEqual(json.loads((snapshot / "failed_manifest.json").read_text())["status"], "failed")

    def test_ready_progress_failure_does_not_publish_latest(self):
        self.file()
        self.file("SPY")
        if not (importlib.util.find_spec("qlib") and REAL_DUMPER.is_file()):
            self.skipTest("local Qlib 0.9.7 dumper unavailable")
        from qlib_yahoo import stooq
        real_write = stooq._write_json_atomically
        latest = self.output / "latest.json"
        latest.write_text('{"dataset":"old"}', encoding="utf-8")

        def fail_ready_progress(path, value):
            if path.name == "progress.json" and value.get("status") == "ready":
                raise OSError("progress denied")
            return real_write(path, value)

        with patch("qlib_yahoo.stooq._dump_script", return_value=REAL_DUMPER), \
             patch("qlib_yahoo.stooq._write_json_atomically", side_effect=fail_ready_progress):
            with self.assertRaisesRegex(OSError, "progress denied"):
                prepare_stooq(self.source, self.output, ("AAA",), "2024-01-02", "2024-01-05", make_latest=True)
        snapshot = next((self.output / "datasets").iterdir())
        self.assertFalse((snapshot / "manifest.json").exists())
        self.assertEqual(json.loads((snapshot / "failed_manifest.json").read_text())["status"], "failed")
        self.assertEqual(latest.read_text(encoding="utf-8"), '{"dataset":"old"}')

    @unittest.skipUnless(importlib.util.find_spec("qlib") and REAL_DUMPER.is_file(), "local Qlib 0.9.7 dumper unavailable")
    def test_real_dumper_roundtrip(self):
        self.file()
        self.file("SPY")
        original = {symbol: sha256(self.folder / f"{symbol.lower()}.us.txt") for symbol in ("AAA", "SPY")}
        with patch("qlib_yahoo.stooq._dump_script", return_value=REAL_DUMPER):
            dataset = prepare_stooq(self.source, self.output, ("AAA",), "2024-01-02", "2024-01-05")
        manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["row_counts"], {"AAA": 3})
        self.assertIsNone(manifest["benchmark"])
        self.assertEqual(manifest["sessions"], 3)
        self.assertEqual(manifest["qlib_binary_file_count"], 8)
        self.assertFalse((self.output / "latest.json").exists())
        self.assertEqual(original, {symbol: sha256(self.folder / f"{symbol.lower()}.us.txt") for symbol in original})


if __name__ == "__main__":
    unittest.main()
