"""The Discover form explains a mistake in the user's words before anything is written."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

HAVE_NUMERIC = bool(importlib.util.find_spec("numpy") and importlib.util.find_spec("pandas"))
if HAVE_NUMERIC:
    from qlib_yahoo.demo_market import create_demo_market
    from qlib_yahoo.webapp.lab_service import LabService


@unittest.skipUnless(HAVE_NUMERIC, "numpy and pandas are optional outside the Qlib environment")
class LabPlanMessageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        create_demo_market(cls.root / "datasets" / "demo-planted-200", symbols=40, days=600, seed=3)
        cls.service = LabService(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.service.close()
        cls._tmp.cleanup()

    def plan(self, **overrides):
        data = {"domain": "alpha", "name": "t", "preset": "quick", "dataset": "demo-planted-200", "universe": "TOP40",
                "cost_bps": 5, "min_sharpe": 1}
        data.update(overrides)
        return self.service.plan(data)

    def test_a_seed_with_a_syntax_error_is_named_before_anything_is_written(self):
        with self.assertRaisesRegex(ValueError, r"Seed idea 2 \('rank\(close'\): .*expected"):
            self.plan(seeds="-rank(ts_delta(close, 5))\nrank(close")
        self.assertFalse(list((self.root / "autoresearch").rglob("*.json")) if (self.root / "autoresearch").exists() else [])

    def test_a_missing_dataset_is_reported_by_name(self):
        with self.assertRaisesRegex(ValueError, r"Dataset 'nope' was not found under .*datasets\. Open Data"):
            self.plan(dataset="nope")

    def test_a_seed_that_needs_a_field_the_dataset_lacks_is_refused(self):
        with self.assertRaisesRegex(ValueError, r"Seed idea 1 .*field 'cap' is not in this dataset \(it has close, factor, high, low, open, volume\)"):
            self.plan(seeds="rank(cap)")
        with self.assertRaisesRegex(ValueError, r"group 'sector' is not available in Discover campaigns"):
            self.plan(seeds="group_neutralize(rank(close), sector)")

    def test_an_extra_factor_is_checked_against_the_dataset_too(self):
        with self.assertRaisesRegex(ValueError, r"Extra factor 'rank\(cap\)': field 'cap' is not in this dataset"):
            self.plan(domain="ml", task="panel", extra_features="rank(cap)", families=["ridge"])

    def test_a_valid_seed_passes_through_to_the_campaign_file(self):
        out = self.plan(seeds="-rank(ts_delta(close, 5))")
        self.assertEqual(out["config"]["settings"]["search"]["seeds"], ["-rank(ts_delta(close, 5))"])


if __name__ == "__main__":
    unittest.main()
