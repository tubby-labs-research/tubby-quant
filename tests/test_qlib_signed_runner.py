"""Bounded signed runner checks against the pinned derivative."""

import importlib.util
import math
import unittest

from qlib_yahoo.dataset import DatasetError
from qlib_yahoo.signed import normalize_scores, signed_daily


@unittest.skipUnless(importlib.util.find_spec("pandas"), "pandas optional")
class SignedPolicyTests(unittest.TestCase):
    def test_normalization_preserves_raw_signs_and_zero(self):
        import pandas as pd

        days = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
        symbols = ["A", "B", "C"]
        index = pd.MultiIndex.from_product([symbols, days], names=["instrument", "datetime"])
        for values, expected in (
            ([3, 1, -2], [0.5, 1 / 6, -1 / 3]),
            ([-3, -1, -2], [-0.5, -1 / 6, -1 / 3]),
            ([0, 0, 0], [0, 0, 0]),
            ([1e308, -1e308, 0], [0.5, -0.5, 0]),
        ):
            with self.subTest(values=values):
                frame = pd.DataFrame({"x": [values[symbols.index(symbol)] for symbol, _ in index]}, index=index)
                raw, weights = normalize_scores(frame, "x", symbols, days, 1.0)
                self.assertEqual(raw.loc[days[0]].tolist(), values)
                for actual, target in zip(weights.loc[days[0]].tolist(), expected):
                    self.assertAlmostEqual(actual, target)
                self.assertLessEqual(sum(abs(value) for value in weights.loc[days[0]]), 1 + 1e-12)
                _, zero = normalize_scores(frame, "x", symbols, days, 0.0)
                self.assertEqual(zero.abs().sum(), 0)

    def test_rank_and_zscore_weights_are_dollar_neutral(self):
        import pandas as pd

        days = pd.to_datetime(["2024-01-02"])
        symbols = ["A", "B", "C"]
        index = pd.MultiIndex.from_product([symbols, days], names=["instrument", "datetime"])
        cases = {
            ("rank", (3, 1, -2)): [0.5, 0.0, -0.5],
            ("rank", (2, 2, -1)): [0.25, 0.25, -0.5],          # ties share their average rank
            ("zscore", (3, 1, -2)): [7 / 16, 1 / 16, -0.5],     # the demeaned score: the scale cancels
            ("rank", (5, 5, 5)): [0.0, 0.0, 0.0],              # no dispersion, no relative view
            ("zscore", (5, 5, 5)): [0.0, 0.0, 0.0],
            ("raw", (3, 1, -2)): [0.5, 1 / 6, -1 / 3],
        }
        for (transform, values), expected in cases.items():
            with self.subTest(transform=transform, values=values):
                frame = pd.DataFrame({"x": [values[symbols.index(symbol)] for symbol, _ in index]}, index=index)
                raw, weights = normalize_scores(frame, "x", symbols, days, 0.8, transform=transform)
                self.assertEqual(raw.loc[days[0]].tolist(), list(values))  # saved scores stay the raw alpha
                for actual, target in zip(weights.loc[days[0]].tolist(), expected):
                    self.assertAlmostEqual(actual, 0.8 * target, places=6)
                if transform != "raw":
                    self.assertAlmostEqual(float(weights.sum()), 0.0, places=6)
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=index)
        with self.assertRaisesRegex(DatasetError, "signal transform"):
            normalize_scores(frame, "x", symbols, days, 1.0, transform="softmax")

    def test_single_name_cross_section_has_no_relative_weight(self):
        from qlib_yahoo.signed import transform_section

        self.assertEqual(transform_section([4.2], "rank").tolist(), [0.0])
        self.assertEqual(transform_section([4.2], "zscore").tolist(), [0.0])
        self.assertEqual(transform_section([4.2], "raw").tolist(), [4.2])

    def test_runner_rejects_transform_for_long_only_before_any_work(self):
        from pathlib import Path

        from qlib_yahoo.backtest import run_backtest

        with self.assertRaisesRegex(DatasetError, "long-only Top-k already ranks"):
            run_backtest(Path("unused"), "$close", strategy_mode="long-only", signal_transform="rank")
        with self.assertRaisesRegex(DatasetError, "signal_transform must be one of"):
            run_backtest(Path("unused"), "$close", strategy_mode="signed", signal_transform="softmax")

    def test_rejects_missing_or_nonfinite_eligible_scores(self):
        import pandas as pd

        day = pd.Timestamp("2024-01-02")
        index = pd.MultiIndex.from_tuples([(day, "A"), (day, "B")], names=["datetime", "instrument"])
        for values, symbols in (([1, math.nan], ["A", "B"]), ([1, math.inf], ["A", "B"]),
                                ([1, 2], ["A", "B", "C"])):
            with self.subTest(values=values, symbols=symbols):
                with self.assertRaises(DatasetError):
                    normalize_scores(pd.DataFrame({"x": values}, index=index), "x", symbols, [day], 1.0)

    def test_unfunded_observation_keeps_nan_return_and_costs(self):
        import pandas as pd

        days = pd.to_datetime(["2024-01-05", "2024-01-08"])
        report = pd.DataFrame({
            "account": [100.0, -1.0], "bench": [0.0, 0.0], "turnover": [0.0, math.nan],
            "return": [0.0, math.nan], "cost": [0.0, math.nan], "net_return": [0.0, math.nan],
            "free_cash": [100.0, -1.0], "cash_total": [100.0, -1.0],
            "restricted_cash": [0.0, 0.0], "long_value": [0.0, 0.0], "short_liability": [0.0, 0.0],
            "transaction_cost_amount": [0.0, 0.0], "borrow_cost_amount": [0.0, 101.0],
            "total_cost_amount": [0.0, 101.0], "unfunded_cash_shortfall": [0.0, 1.0],
            "funding_feasible": [True, False], "metrics_defined": [True, False],
            "stale_marks": [(), ()],
        }, index=days)
        position = type("Position", (), {"calculate_value": lambda self: self.value})
        positions = {day: type("P", (position,), {"value": value})() for day, value in zip(days, [100.0, -1.0])}
        daily = signed_daily(report, positions, 100.0)
        self.assertEqual(len(daily), 2)
        self.assertTrue(math.isnan(daily.iloc[1]["net_return"]))
        self.assertTrue(math.isnan(daily.iloc[1]["turnover"]))
        self.assertTrue(math.isnan(daily.iloc[1]["cost"]))
        self.assertEqual(daily.iloc[1]["borrow_cost_dollars"], 101.0)
        self.assertEqual(daily.iloc[1]["unfunded_cash_shortfall"], 1.0)
        report.loc[days[0], "transaction_cost_amount"] = -1.0
        report.loc[days[0], "borrow_cost_amount"] = 1.0
        with self.assertRaisesRegex(DatasetError, "negative expense"):
            signed_daily(report, positions, 100.0)


if __name__ == "__main__":
    unittest.main()
