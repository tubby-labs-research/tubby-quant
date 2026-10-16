"""Differential checks against the installed Qlib NumpyQuote, with an opt-in benchmark."""

import importlib.util
import os
import time
import unittest
from statistics import median
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("qlib"), "optional Qlib runtime unavailable")
class DailyQuoteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import pandas as pd
        from qlib.backtest.high_performance_ds import NumpyQuote
        from qlib_yahoo.daily_quote import DailyQuote

        cls.np = np
        cls.pd = pd
        cls.NumpyQuote = NumpyQuote
        cls.DailyQuote = DailyQuote
        dates = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"])
        index = pd.MultiIndex.from_product(
            [["AAA", "BBB"], dates], names=["instrument", "datetime"]
        )
        cls.frame = pd.DataFrame(
            {"$close": [10.0, np.nan, 12.0, 20.0, 21.0, 22.0],
             "$volume": [100.0, 110.0, 120.0, 200.0, 210.0, 220.0]},
            index=index,
        )

    def setUp(self):
        self.original = self.NumpyQuote(self.frame, "day")
        self.fast = self.DailyQuote(self.frame, "day")

    def assert_same(self, stock, start, end, field="$close", method=None):
        def run(quote):
            try:
                return ("value", quote.get_data(stock, start, end, field, method))
            except Exception as exc:  # Exact exception parity is part of this test.
                return ("error", type(exc), str(exc))

        expected, actual = run(self.original), run(self.fast)
        self.assertEqual(expected[0], actual[0])
        if expected[0] == "error":
            self.assertEqual(expected[1:], actual[1:])
        elif expected[1] is None:
            self.assertIsNone(actual[1])
        elif hasattr(expected[1], "indices"):
            self.assertEqual(type(expected[1]), type(actual[1]))
            self.np.testing.assert_array_equal(expected[1].data, actual[1].data)
            self.np.testing.assert_array_equal(expected[1].index.idx_list, actual[1].index.idx_list)
        elif self.np.isnan(expected[1]):
            self.assertTrue(self.np.isnan(actual[1]))
            self.assertEqual(type(expected[1]), type(actual[1]))
        else:
            self.assertEqual(expected[1], actual[1])
            self.assertEqual(type(expected[1]), type(actual[1]))
            if isinstance(expected[1], self.np.floating):
                self.assertEqual(self.np.signbit(expected[1]), self.np.signbit(actual[1]))

    def test_exact_daily_windows_and_methods(self):
        p = self.pd.Timestamp
        for day in ("2024-01-05", "2024-01-06", "2024-01-08", "2024-01-09", "2024-01-10"):
            start = p(day)
            for end in (start, start + self.pd.Timedelta(hours=12),
                        start + self.pd.Timedelta(hours=23, minutes=59, seconds=59)):
                for stock in ("AAA", "BBB", "MISSING"):
                    for field in ("$close", "$volume", "$missing"):
                        for method in (None, "sum", "all", "ts_data_last", "mean", "last", "invalid"):
                            with self.subTest(day=day, end=end, stock=stock, field=field, method=method):
                                self.assert_same(stock, start, end, field, method)

    def test_fallback_windows_and_exceptions(self):
        p = self.pd.Timestamp
        cases = [
            (p("2024-01-05 12:00"), p("2024-01-05 23:59:59")),
            (p("2024-01-05"), p("2024-01-08")),
            (p("2024-01-05"), p("2024-01-06")),
            (p("2024-01-08"), p("2024-01-07")),
            (p("2024-01-05", tz="UTC"), p("2024-01-05 23:59:59", tz="UTC")),
            (p("2024-01-05"), p("2024-01-05 23:59:59", tz="UTC")),
        ]
        for start, end in cases:
            for method in (None, "sum", "all", "ts_data_last", "mean", "invalid"):
                with self.subTest(start=start, end=end, method=method):
                    self.assert_same("AAA", start, end, method=method)

    def test_missing_day_never_peeks_at_next_row_and_direct_path(self):
        p = self.pd.Timestamp
        with patch.object(self.NumpyQuote, "get_data", side_effect=AssertionError("fallback")):
            self.assertIsNone(self.fast.get_data("AAA", p("2024-01-06"), p("2024-01-06 23:59:59"), "$close"))
            self.assertTrue(self.np.isnan(self.fast.get_data("AAA", p("2024-01-08"), p("2024-01-08 23:59:59"), "$close")))
            self.assertEqual(self.fast.get_data("AAA", p("2024-01-09"), p("2024-01-09 23:59:59"), "$close"), 12.0)

    def test_result_cache_is_bounded_and_per_instance(self):
        p = self.pd.Timestamp
        start, end = p("2024-01-05"), p("2024-01-05 23:59:59")
        self.assertEqual(self.fast.get_data.cache_info().maxsize, 512)
        before = self.fast.get_data.cache_info().hits
        self.assertEqual(self.fast.get_data("AAA", start, end, "$close"), 10.0)
        self.assertEqual(self.fast.get_data("AAA", start, end, "$close"), 10.0)
        self.assertGreater(self.fast.get_data.cache_info().hits, before)

        other_frame = self.frame.copy()
        other_frame.loc[("AAA", start), "$close"] = 99.0
        other = self.DailyQuote(other_frame, "day")
        self.assertEqual(other.get_data("AAA", start, end, "$close"), 99.0)
        self.assertEqual(self.fast.get_data("AAA", start, end, "$close"), 10.0)
        self.assertEqual(other.get_data.cache_info().hits, 0)

        # Fallback values and errors still follow the parent's behavior, even
        # when a request is repeated through the new instance cache.
        for _ in range(2):
            self.assert_same("AAA", start, p("2024-01-09"), method="sum")
            self.assert_same("AAA", start, p("2024-01-09"), method="unsupported")

    def test_weekend_holiday_closed_intervals(self):
        p = self.pd.Timestamp
        dates = self.pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-12", "2024-01-16"])
        index = self.pd.MultiIndex.from_product(
            [["AAA", "BBB"], dates], names=["instrument", "datetime"]
        )
        frame = self.pd.DataFrame(
            {"$close": [10., self.np.nan, -0., self.np.inf,
                        -self.np.inf, 0., self.np.nan, 20.]}, index=index
        )
        self.original = self.NumpyQuote(frame, "day")
        self.fast = self.DailyQuote(frame, "day")
        cases = [
            ("2024-01-05", "2024-01-07 23:59:59.999999999"),
            ("2024-01-12", "2024-01-15 23:59:59.999999999"),
            ("2024-01-08", "2024-01-11 23:59:59.999999999"),
            ("2024-01-06", "2024-01-08"),  # Missing start, one observed bar.
            ("2024-01-06", "2024-01-07 23:59:59.999999999"),
            ("2024-01-04", "2024-01-05"),
            ("2024-01-05", "2024-01-08"),  # Closed end includes Monday.
            ("2024-01-17", "2024-01-20"),
            ("2024-01-16", "2024-01-15"),
        ]
        for start, end in cases:
            for stock in ("AAA", "BBB", "MISSING"):
                for field in ("$close", "$missing"):
                    for method in (None, "sum", "mean", "all", "last", "ts_data_last", "invalid"):
                        with self.subTest(start=start, end=end, stock=stock, field=field, method=method):
                            self.assert_same(stock, p(start), p(end), field, method)

    def test_weekend_aggregation_uses_direct_path_and_bounded_row_cache(self):
        p = self.pd.Timestamp
        start, end = p("2024-01-05"), p("2024-01-07 23:59:59.999999999")
        with patch.object(self.NumpyQuote, "get_data", side_effect=AssertionError("fallback")):
            for field, expected in (("$close", 10.), ("$volume", 100.)):
                for method in ("sum", "mean", "last", "ts_data_last"):
                    self.assertEqual(self.fast.get_data("AAA", start, end, field, method), expected)
            self.assertIs(self.fast.get_data("AAA", start, end, "$close", "all"), self.np.True_)
            with self.assertRaisesRegex(ValueError, "invalid is not supported"):
                self.fast.get_data("AAA", start, end, "$close", "invalid")
            with self.assertRaises(KeyError):
                self.fast.get_data("AAA", start, end, "$missing", "sum")
        self.assertEqual(self.fast._interval_row.cache_info().maxsize, 512)
        self.assertEqual(self.fast._interval_row.cache_info().misses, 1)
        self.assertGreater(self.fast._interval_row.cache_info().hits, 0)
        other = self.DailyQuote(self.frame, "day")
        self.assertEqual(other._interval_row.cache_info().currsize, 0)

    def test_non_daily_or_unsupported_region_retains_native_contract(self):
        p = self.pd.Timestamp
        for frequency, region in (("min", "us"), ("day", "unsupported")):
            self.original = self.NumpyQuote(self.frame, frequency, region=region)
            self.fast = self.DailyQuote(self.frame, frequency, region=region)
            for end in (p("2024-01-05 23:59:59"), p("2024-01-07 23:59:59")):
                for method in (None, "sum", "all", "last", "invalid"):
                    with self.subTest(frequency=frequency, region=region, end=end, method=method):
                        self.assert_same("AAA", p("2024-01-05"), end, method=method)

    def test_weekend_none_multibar_and_timezone_keep_native_fallback(self):
        p = self.pd.Timestamp
        requests = [
            (p("2024-01-05"), p("2024-01-07 23:59:59"), None),
            (p("2024-01-05"), p("2024-01-08"), "sum"),
            (p("2024-01-06"), p("2024-01-07"), "invalid"),
            (p("2024-01-05", tz="UTC"), p("2024-01-07", tz="UTC"), "sum"),
        ]
        for start, end, method in requests:
            self.fast.get_data.cache_clear()
            with patch.object(self.NumpyQuote, "get_data", return_value="native") as native:
                self.assertEqual(self.fast.get_data("AAA", start, end, "$close", method), "native")
                native.assert_called_once()

    @unittest.skipUnless(os.environ.get("QLIB_DAILY_QUOTE_BENCH") == "1", "opt-in bounded benchmark")
    def test_bounded_microbenchmark(self):
        import pandas as pd

        def measure(label, stocks, dates, repetitions, request_dates=None, end_for=None, method=None):
            index = pd.MultiIndex.from_product([stocks, dates], names=["instrument", "datetime"])
            values = self.np.arange(len(index), dtype=float)
            frame = pd.DataFrame({"$close": values, "$volume": values + 1,
                                  "$factor": values + 2}, index=index)
            original, fast = self.NumpyQuote(frame, "day"), self.DailyQuote(frame, "day")
            requests = [(stock, day, (end_for(day) if end_for else
                                      day + pd.Timedelta(hours=23, minutes=59, seconds=59)), field, method)
                        for day in (dates if request_dates is None else request_dates) for stock in stocks
                        for field in ("$close", "$volume", "$factor")
                        for _ in range(repetitions)]
            results = []
            for quote in (original, fast):
                timings = []
                for _ in range(3):
                    quote.get_data.cache_clear()
                    if hasattr(quote, "_interval_row"):
                        quote._interval_row.cache_clear()
                    start = time.perf_counter()
                    checksum = sum(float(quote.get_data(*request)) for request in requests)
                    timings.append(time.perf_counter() - start)
                results.append((median(timings), checksum))
            self.assertEqual(results[0][1], results[1][1])
            print(f"\nDailyQuote {label}: {len(stocks)} stocks x {len(dates)} dates, "
                  f"{len(requests)} lookups x 3 (median); "
                  f"NumpyQuote={results[0][0]:.3f}s DailyQuote={results[1][0]:.3f}s "
                  f"speedup={results[0][0] / results[1][0]:.2f}x")

        measure("cold cross-section", [f"S{i:04d}" for i in range(3000)],
                pd.bdate_range("2024-01-02", periods=20), 1,
                request_dates=pd.bdate_range("2024-01-02", periods=1))
        measure("hot small universe", [f"S{i:02d}" for i in range(6)],
                pd.bdate_range("2023-01-02", periods=239), 5)
        measure("cold weekend cross-section", [f"S{i:04d}" for i in range(3000)],
                pd.bdate_range("2024-01-05", periods=20), 1,
                request_dates=[pd.Timestamp("2024-01-05")],
                end_for=lambda day: pd.Timestamp("2024-01-08") - pd.Timedelta(nanoseconds=1), method="mean")
        holiday_dates = pd.bdate_range("2024-01-12", periods=20).difference(pd.to_datetime(["2024-01-15"]))
        measure("cold holiday cross-section", [f"S{i:04d}" for i in range(3000)],
                holiday_dates, 1, request_dates=[pd.Timestamp("2024-01-12")],
                end_for=lambda day: pd.Timestamp("2024-01-16") - pd.Timedelta(nanoseconds=1), method="sum")


if __name__ == "__main__":
    unittest.main()
