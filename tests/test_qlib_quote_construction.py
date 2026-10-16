"""Actual-native quote construction differentials and an opt-in benchmark."""

import gc
import json
import statistics
import time
import tracemalloc
import unittest

try:
    import numpy as np
    import pandas as pd
    from qlib.backtest.high_performance_ds import BaseQuote, NumpyQuote
    from qlib.utils.index_data import IndexData
    from qlib.utils.time import Freq
    from qlib_yahoo.quote_construction import native_quote_panels
except ImportError:
    NumpyQuote = None


if NumpyQuote is not None:
    class CandidateQuote(NumpyQuote):
        """Test harness: same native accessors, only panel assembly differs."""

        def __init__(self, frame, freq, region="cn"):
            BaseQuote.__init__(self, frame, freq)
            panels = native_quote_panels(frame, freq)
            if panels is None:
                super().__init__(frame, freq, region)
                return
            self.data = panels
            _, unit = Freq.parse(freq)
            self.freq = Freq.get_timedelta(1, unit)
            self.region = region


@unittest.skipIf(NumpyQuote is None, "optional Qlib environment required")
class QuoteConstructionTests(unittest.TestCase):
    def frame(self):
        dates = pd.bdate_range("2021-01-01", periods=6)
        index = pd.MultiIndex.from_tuples(
            [(stock, date) for stock in ("AAA", "BBB", "ZZZ") for date in dates
             if not (stock == "BBB" and date == dates[2])],
            names=["instrument", "datetime"],
        )
        close = np.arange(len(index), dtype=np.float32)
        close[1:3] = np.nan
        close[-6:] = np.nan
        close[0] = -0.0
        return pd.DataFrame({"$volume": np.arange(len(index), dtype=np.int64),
                             "$close": close,
                             "$valid": np.arange(len(index)) % 2 == 0}, index=index)

    def panels_equal(self, actual, expected):
        self.assertEqual(list(actual), list(expected))
        for stock in expected:
            a, e = actual[stock], expected[stock]
            self.assertIs(type(a), type(e))
            self.assertEqual(a.data.dtype, e.data.dtype)
            self.assertEqual(a.data.shape, e.data.shape)
            self.assertEqual(a.data.tobytes(), e.data.tobytes())
            for ai, ei in zip(a.indices, e.indices):
                self.assertEqual(ai.idx_list.dtype, ei.idx_list.dtype)
                self.assertEqual(ai.idx_list.tobytes(), ei.idx_list.tobytes())
                self.assertEqual(ai.index_map, ei.index_map)
                self.assertEqual(ai.is_sorted(), ei.is_sorted())

    def test_panel_bytes_maps_and_input_preservation(self):
        frame = self.frame()
        before = frame.copy(deep=True)
        # Unused levels must not introduce empty groups or alter stock order.
        frame.index = frame.index.set_levels([pd.Index(["AAA", "BBB", "ZZZ", "UNUSED"]),
                                              frame.index.levels[1]])
        before.index = frame.index.copy()
        self.panels_equal(native_quote_panels(frame, "day"), NumpyQuote(frame, "day").data)
        pd.testing.assert_frame_equal(frame, before, check_exact=True)

    def test_dtype_promotion_matches_native(self):
        index = self.frame().index[:5]
        for dtype_a, dtype_b in (("float16", "int8"), ("float32", "int32"),
                                  ("uint64", "int64"), ("float64", "float32"),
                                  ("bool", "int64"), ("bool", "bool"),
                                  ("float16", "float16")):
            with self.subTest(dtype_a=dtype_a, dtype_b=dtype_b):
                frame = pd.DataFrame({"z": np.array([0, 1, 2, 3, 4], dtype=dtype_a),
                                      "a": np.array([4, 3, 2, 1, 0], dtype=dtype_b)}, index=index)
                self.panels_equal(native_quote_panels(frame, "day"), NumpyQuote(frame, "day").data)
        values = np.array([0x8000000000000000, 0x7ff8000000000001, 0x7ff0000000000000,
                           0xfff0000000000000, 0x3ff0000000000001], dtype=np.uint64).view(np.float64)
        frame = pd.DataFrame({"payload": values}, index=index)
        self.panels_equal(native_quote_panels(frame, "day"), NumpyQuote(frame, "day").data)

    def outcome(self, quote, args):
        try:
            value = quote.get_data(*args)
        except Exception as error:
            return (type(error), str(error))
        if isinstance(value, IndexData):
            return (type(value), value.data.tobytes(),
                    tuple((i.idx_list.dtype.str, i.idx_list.tobytes()) for i in value.indices))
        if isinstance(value, np.generic):
            return (type(value), value.tobytes())
        return value

    def test_native_get_data_intervals_aggregations_and_errors(self):
        frame = self.frame()
        for freq in ("day", "1day", "min", "5min"):
            native, candidate = NumpyQuote(frame, freq, "us"), CandidateQuote(frame, freq, "us")
            for stock in ("AAA", "BBB", "ZZZ", "ABSENT"):
                for start, end in (("2021-01-01", "2021-01-01"),
                                   ("2021-01-01", "2021-01-08"),
                                   ("2021-01-02", "2021-01-05"),
                                   ("2020-01-01", "2020-01-05")):
                    for field in ("$close", "$volume", "$valid", "$absent"):
                        for method in (None, "sum", "mean", "last", "all", "ts_data_last", "bad"):
                            args = (stock, pd.Timestamp(start), pd.Timestamp(end), field, method)
                            with self.subTest(freq=freq, args=args):
                                self.assertEqual(self.outcome(candidate, args), self.outcome(native, args))

    def test_intraday_and_fallback_constructor_errors(self):
        dates = pd.date_range("2021-01-04 09:30", periods=4, freq="5min")
        frame = pd.DataFrame({"$close": [1.0, np.nan, -0.0, 4.0]},
                             index=pd.MultiIndex.from_product([["AAA"], dates],
                                                              names=["instrument", "datetime"]))
        native, candidate = NumpyQuote(frame, "5min", "us"), CandidateQuote(frame, "5min", "us")
        self.panels_equal(candidate.data, native.data)
        for method in (None, "sum", "mean", "last", "all", "ts_data_last"):
            args = ("AAA", dates[0], dates[-1], "$close", method)
            self.assertEqual(self.outcome(candidate, args), self.outcome(native, args))
        for freq in ("week", "nonsense"):
            errors = []
            for cls in (NumpyQuote, CandidateQuote):
                try:
                    cls(frame, freq)
                except Exception as error:
                    errors.append((type(error), str(error)))
            self.assertEqual(len(errors), 2)
            self.assertEqual(errors[0], errors[1])

    def test_unsupported_shapes_and_dtypes_fall_back(self):
        good = self.frame()
        bad = [good.iloc[::-1], pd.concat([good, good]).sort_index(), good.reset_index(),
               good.rename_axis(["datetime", "instrument"]),
               good.astype(object), good.astype("Float64"), good.astype(complex),
               good.rename(columns={"$valid": 7}), good.set_axis(["x", "x", "y"], axis=1),
               good.set_axis(pd.MultiIndex.from_tuples([("x", "a"), ("y", "a"), ("z", "a")]), axis=1)]
        for tuples in ([('AAA', pd.NaT)], [(None, pd.Timestamp("2021-01-01"))],
                       [(1, pd.Timestamp("2021-01-01"))],
                       [('AAA', pd.Timestamp("2021-01-01", tz="UTC"))]):
            bad.append(pd.DataFrame({"x": [1]}, index=pd.MultiIndex.from_tuples(
                tuples, names=["instrument", "datetime"])))
        class CustomFrame(pd.DataFrame):
            pass
        bad.append(CustomFrame(good))
        for frame in bad:
            with self.subTest(dtypes=frame.dtypes):
                self.assertIsNone(native_quote_panels(frame, "day"))
        for freq in ("week", "nonsense", None):
            self.assertIsNone(native_quote_panels(good, freq))
        self.assertEqual(native_quote_panels(good.iloc[:0], "day"), {})


def benchmark():
    """Three paired complete-constructor trials; fixture creation is excluded."""
    dates = pd.bdate_range("2021-01-01", periods=1255)
    index = pd.MultiIndex.from_product([[f"S{i:04}" for i in range(250)], dates],
                                       names=["instrument", "datetime"])
    rng = np.random.default_rng(123)
    frame = pd.DataFrame(rng.standard_normal((len(index), 8)).astype(np.float32),
                         index=index, columns=[f"$field{i}" for i in range(8)])
    frame.iloc[::11, 3] = np.nan
    trials = {"native": [], "candidate": []}
    # Qlib imports and initial frequency caches are outside measured trials.
    for cls in (NumpyQuote, CandidateQuote):
        cls(frame.iloc[:1255], "day", "us")
    for trial in range(3):
        order = (("native", NumpyQuote), ("candidate", CandidateQuote))
        for name, cls in (order if trial % 2 == 0 else reversed(order)):
            gc.collect()
            start = time.perf_counter()
            quote = cls(frame, "day", "us")
            trials[name].append(time.perf_counter() - start)
            del quote
    peaks = {}
    for name, cls in (("native", NumpyQuote), ("candidate", CandidateQuote)):
        gc.collect()
        tracemalloc.start()
        quote = cls(frame, "day", "us")
        peaks[name] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        del quote
    print(json.dumps({"shape": list(frame.shape), "paired_seconds": trials,
                      "median_seconds": {k: statistics.median(v) for k, v in trials.items()},
                      "tracemalloc_peak_bytes": peaks}, indent=2))


if __name__ == "__main__":
    import sys
    if "--benchmark" in sys.argv:
        benchmark()
    else:
        unittest.main()
