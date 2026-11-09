"""Submitted settings reject nonfinite/ambiguous values without loading data."""
from dataclasses import replace
import unittest

from qlib_yahoo.alphalab.settings import Settings
from qlib_yahoo.dataset import DatasetError


class WorkflowSettingsTests(unittest.TestCase):
    def test_existing_screening_defaults_remain_unchanged(self):
        defaults = Settings().validate()
        self.assertEqual(defaults.cost_bps, 0)
        self.assertEqual(defaults.delay, 1)
        self.assertEqual(defaults.neutralization, "MARKET")
        self.assertEqual(defaults.booksize, 20_000_000)
        self.assertEqual(defaults.truncation, .08)

    def test_ambiguous_or_nonfinite_settings_fail_before_preparation(self):
        invalid = {
            "delay": [True, 1.0, "1", None],
            "decay": [True, 2.0, -1],
            "warmup": [False, 1.0, -1],
            "pasteurization": [0, 1, "true", None],
            "truncation": [True, "0.1", float("nan"), float("inf")],
            "booksize": [False, 0, "100000", float("inf")],
            "cost_bps": [True, -1, float("nan"), float("inf")],
            "neutralization": [None, 2, "not-a-group"],
            "universe": [None, 3000, ""],
        }
        for name, values in invalid.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(DatasetError):
                    replace(Settings(), **{name: value}).validate()

    def test_valid_explicit_values_and_case_normalization(self):
        settings = Settings(universe="top3000", neutralization="industry", delay=0,
                            decay=3, warmup=10, truncation=0, cost_bps=5,
                            booksize=100_000, pasteurization=False).validate()
        self.assertEqual(settings.universe, "TOP3000")
        self.assertEqual(settings.neutralization, "INDUSTRY")
        self.assertEqual(settings.cost_bps, 5)
        self.assertFalse(settings.pasteurization)


if __name__ == "__main__":
    unittest.main()
