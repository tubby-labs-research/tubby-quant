"""The console's artificial demonstration snapshot is readable by every engine and says it is artificial."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy  # noqa: F401
except ImportError:  # pragma: no cover
    numpy = None

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(numpy, "numpy needed to generate prices")
class DemoMarketTests(unittest.TestCase):
    def test_snapshot_is_price_eligible_sealed_and_idempotent(self):
        from qlib_yahoo.demo_market import DEFAULT_NAME, PROVIDER, create_demo_market
        from qlib_yahoo.universe import load_dataset_universe
        from qlib_yahoo.workspace import dataset_summary

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lines = []
            path = create_demo_market(root / "datasets" / DEFAULT_NAME, symbols=24, days=420, log=lines.append)
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["provider"], PROVIDER)
            self.assertTrue(manifest["artificial"])
            self.assertIn("artificial", manifest["note"].lower())
            self.assertEqual(len(manifest["symbols"]), 24)
            self.assertEqual(manifest["fields"], ["open", "high", "low", "close", "volume", "factor"])
            self.assertEqual(manifest["sessions"], 420)
            universe = manifest["universes"]["top24"]
            self.assertEqual(universe["instruments_file"], "instruments/top24.txt")
            self.assertEqual(universe["min_history_sessions"], 120)
            summary = dataset_summary(root, path)
            self.assertEqual(summary["status"], "ready")
            self.assertTrue(summary["price_eligible"], summary["compatibility_note"])
            self.assertEqual(summary["symbols"], 24)
            self.assertEqual(summary["start"], "2018-01-02")
            spans, record = load_dataset_universe(path, manifest, "top24")
            self.assertEqual(len(spans), 24)
            self.assertEqual(record["size"], 24)
            for symbol in manifest["symbols"][:2]:
                folder = path / "qlib_data" / "features" / symbol.lower()
                self.assertTrue((folder / "factor.day.bin").is_file())
                self.assertTrue((folder / "close.day.bin").is_file())
            self.assertTrue(any("ready" in line for line in lines))
            # A second call verifies instead of rewriting; another provider's folder is refused.
            again = create_demo_market(path)
            self.assertEqual(again, path)
            other = root / "datasets" / "other"
            (other).mkdir(parents=True)
            (other / "manifest.json").write_text('{"provider": "yahoo"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                create_demo_market(other)

    def test_rejects_a_non_empty_folder_and_tiny_panels(self):
        from qlib_yahoo.demo_market import create_demo_market

        with tempfile.TemporaryDirectory() as tmp:
            busy = Path(tmp) / "busy"
            busy.mkdir()
            (busy / "note.txt").write_text("x", encoding="utf-8")
            with self.assertRaises(ValueError):
                create_demo_market(busy)
            with self.assertRaises(ValueError):
                create_demo_market(Path(tmp) / "tiny", symbols=5, days=100)


if __name__ == "__main__":
    unittest.main()
