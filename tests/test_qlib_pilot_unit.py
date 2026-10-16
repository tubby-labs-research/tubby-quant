import math
import hashlib
import struct
import tempfile
import unittest
from pathlib import Path

from tests.qlib_pilot_fixture import (
    QLIB_FIELDS,
    load_fixture,
    materialize_qlib_snapshot,
    snapshot_id,
    verify_expected_asof,
)


class QlibPilotUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_dir = Path(__file__).parent / "fixtures" / "qlib_pilot"

    def test_point_in_time_fixture_resolves_late_and_amended_values(self):
        panel = load_fixture(self.fixture_dir)
        self.assertEqual(verify_expected_asof(panel, self.fixture_dir / "expected_asof.csv"), 25)
        self.assertEqual(panel.revisions[(panel.calendar[2], "QLBA")], "rev1")
        self.assertEqual(panel.revisions[(panel.calendar[7], "QLBA")], "rev2")
        self.assertIsNone(panel.revisions[(panel.calendar[5], "QLBB")])

    def test_materialized_snapshot_has_qlib_layout_and_missing_bar(self):
        panel = load_fixture(self.fixture_dir)
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "qlib_data"
            materialize_qlib_snapshot(panel, destination)
            calendar = (destination / "calendars" / "day.txt").read_text(encoding="utf-8").splitlines()
            instruments = (destination / "instruments" / "all.txt").read_text(encoding="utf-8").splitlines()
            future_calendar = (destination / "calendars" / "day_future.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(calendar), 11)
            self.assertEqual(calendar[-1], "2026-01-19")
            self.assertEqual(calendar, future_calendar)
            self.assertIn("QLBB\t2026-01-06\t2026-01-16", instruments)
            for field in QLIB_FIELDS:
                self.assertTrue((destination / "features" / "qlba" / f"{field}.day.bin").is_file())

            payload = (destination / "features" / "qlbb" / "close.day.bin").read_bytes()
            values = struct.unpack(f"<{len(payload) // 4}f", payload)
            self.assertEqual(values[0], 1.0)
            self.assertTrue(math.isnan(values[5]))

            change_payload = (destination / "features" / "qlbb" / "change.day.bin").read_bytes()
            changes = struct.unpack(f"<{len(change_payload) // 4}f", change_payload)
            self.assertTrue(math.isnan(changes[5]))
            self.assertTrue(math.isnan(changes[6]))

            book_payload = (destination / "features" / "qlba" / "book_value.day.bin").read_bytes()
            book_values = struct.unpack(f"<{len(book_payload) // 4}f", book_payload)
            self.assertTrue(math.isnan(book_values[1]))
            self.assertEqual(book_values[3], 100.0)
            self.assertEqual(book_values[8], 96.0)

            early_end_payload = (destination / "features" / "qlbc" / "close.day.bin").read_bytes()
            self.assertEqual(len(early_end_payload) // 4, 7)

            # Preserve the original binary known answer without a pilot runtime.
            digest = hashlib.sha256()
            files = sorted(path for path in destination.rglob("*") if path.is_file())
            for path in files:
                relative = path.relative_to(destination).as_posix().encode("utf-8")
                payload = path.read_bytes()
                digest.update(len(relative).to_bytes(4, "big"))
                digest.update(relative)
                digest.update(len(payload).to_bytes(8, "big"))
                digest.update(payload)
            self.assertEqual(len(files), 35)
            self.assertEqual(digest.hexdigest(), "4a775704cb8eac610363e0d0484264e09ab4a9883e19b9ebcd40a3cae6277c86")

    def test_snapshot_identifier_is_stable(self):
        identifier = snapshot_id(self.fixture_dir)
        self.assertEqual(len(identifier), 16)
        self.assertEqual(identifier, "9e19e3828df8ea9f")
        self.assertEqual(identifier, snapshot_id(self.fixture_dir))


class ProductionRequirementsTests(unittest.TestCase):
    def test_active_platform_requirements_retain_all_pins(self):
        root = Path(__file__).resolve().parents[1]
        expected = {"numpy": "1.26.4", "pandas": "2.2.3", "scipy": "1.14.1",
                    "mlflow": "3.16.1", "pyqlib": "0.9.7", "yahooquery": "2.4.1",
                    "plotly": "5.24.1", "statsmodels": "0.14.4"}

        def expand(path):
            pins = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("-r "):
                    pins.update(expand(path.parent / line[3:].strip()))
                else:
                    name, version = line.split("==")
                    self.assertNotIn(name, pins, f"duplicate pin in {path}")
                    pins[name] = version
            return pins

        self.assertEqual(expand(root / "requirements-qlib-yahoo.txt"), expected)
        for platform in ("windows", "macos"):
            with self.subTest(platform=platform):
                self.assertEqual(expand(root / f"requirements-qlib-{platform}.txt"),
                                 {**expected, "pyarrow": "25.0.1", "pywebview": "6.2.1"})
        self.assertFalse((root / "requirements-qlib-pilot.txt").exists())


if __name__ == "__main__":
    unittest.main()
