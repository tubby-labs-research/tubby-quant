"""Differentials against the installed Qlib DatasetProvider, on tiny local bins."""

import copy
import logging
import statistics
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy as np
    import pandas as pd
    import qlib
    from qlib.config import C
    from qlib.data.cache import H
    from qlib.data.data import (Cal, DatasetProvider, ExpressionD,
                               LocalDatasetProvider, LocalExpressionProvider)
    from qlib.data.inst_processor import InstProcessor
    from qlib_yahoo.array_dataset_provider import ArrayDatasetProvider
except ImportError:
    qlib = None


if qlib is not None:
    class MultiplyProcessor(InstProcessor):
        def __call__(self, df, instrument, *args, **kwargs):
            return df * 3


@unittest.skipIf(qlib is None, "optional Qlib environment required")
class ArrayDatasetProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.saved_config = copy.deepcopy(C)
        from qlib.data import data
        cls.saved_wrappers = {name: getattr(data, name)._provider
                              for name in ("Cal", "Inst", "FeatureD", "PITD", "ExpressionD", "DatasetD", "D")}
        cls.dates = pd.bdate_range("2021-01-04", periods=24)
        (cls.root / "calendars").mkdir()
        (cls.root / "calendars/day.txt").write_text(
            "\n".join(d.strftime("%Y-%m-%d") for d in cls.dates) + "\n")
        (cls.root / "instruments").mkdir()
        (cls.root / "instruments/all.txt").write_text(
            "".join(f"{inst}\t{cls.dates[0]:%Y-%m-%d}\t{cls.dates[-1]:%Y-%m-%d}\n"
                    for inst in ("AAA", "BBB", "CCC", "EMPTY")))
        for inst, start, end in (("AAA", 0, 24), ("BBB", 3, 19), ("CCC", 0, 24)):
            path = cls.root / "features" / inst.lower()
            path.mkdir(parents=True)
            for field, scale in (("close", 1), ("open", 2), ("volume", 100)):
                field_start = start + (2 if inst == "CCC" and field == "open" else 0)
                values = np.arange(field_start + 1, end + 1, dtype=np.float32) * scale
                values[4:6] = np.nan
                np.concatenate((np.array([field_start], dtype=np.float32), values)).tofile(
                    path / f"{field}.day.bin")

    @classmethod
    def tearDownClass(cls):
        from qlib.data import data
        C.register_from_C(cls.saved_config)
        for name, provider in cls.saved_wrappers.items():
            getattr(data, name).register(provider)
        H.clear()
        cls.temp.cleanup()

    def setUp(self):
        qlib.init(provider_uri=str(self.root), region="us", kernels=1,
                  expression_cache=None, dataset_cache=None,
                  logging_level=logging.ERROR, clear_mem_cache=True)
        self.native = LocalDatasetProvider()
        self.array = ArrayDatasetProvider()

    def paired(self, instruments, fields, start=2, end=20, **kwargs):
        args = (instruments, fields, self.dates[start], self.dates[end])
        expected = self.native.dataset(*args, **kwargs)
        actual = self.array.dataset(*args, **kwargs)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        return actual

    def test_native_operators_windows_missing_values_and_duplicate_fields(self):
        fields = ["Ref($close,-2)/Ref($close,-1)-1", " Mean( $close, 5 ) ",
                  "$volume", "$close+$open", "$close", "$close",
                  "Std($close,3)", "Ref($close,2)", "Mean($close,5)"]
        actual = self.paired(["BBB", "AAA"], fields)
        self.assertEqual(actual.columns.tolist(), fields)
        self.assertTrue(all(dtype == np.float32 for dtype in actual.dtypes))
        self.assertEqual(actual.index.names, ["instrument", "datetime"])
        self.assertTrue(actual.index.is_monotonic_increasing)
        self.assertTrue(actual.isna().any().any())

    def test_differing_field_spans_and_empty_symbol_match_native_union(self):
        actual = self.paired(["CCC", "EMPTY", "BBB", "AAA"], ["$open", "$close"], start=0)
        self.assertNotIn("EMPTY", actual.index.get_level_values("instrument"))
        self.assertEqual(actual.loc["CCC"].shape[0], 21)
        self.assertTrue(actual.loc[("CCC", self.dates[0]), "$open"] !=
                        actual.loc[("CCC", self.dates[0]), "$open"])

    def test_missing_entire_field_preserves_other_field_rows(self):
        actual = self.paired(["BBB", "AAA"], ["$low", "$close"])
        self.assertTrue(actual["$low"].isna().all())
        self.assertTrue(actual["$close"].notna().any())

    def test_flat_assembly_avoids_per_instrument_dataframes(self):
        # With identical field ranges the optimized path constructs one frame;
        # the native path constructs a frame for each stock plus concatenation.
        constructor = pd.DataFrame
        calls = []
        def tracked(*args, **kwargs):
            calls.append(args[0] if args else None)
            return constructor(*args, **kwargs)
        with mock.patch("qlib_yahoo.array_dataset_provider.pd.DataFrame", side_effect=tracked):
            actual = self.array.dataset(["AAA", "BBB"], ["$close", "$volume"],
                                        self.dates[2], self.dates[20])
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0], np.ndarray)
        expected = self.native.dataset(["AAA", "BBB"], ["$close", "$volume"],
                                       self.dates[2], self.dates[20])
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)

    def test_empty_symbols_and_empty_calendar_range(self):
        self.paired([], ["$close"])
        self.paired(["EMPTY"], ["$close"])
        self.paired(["AAA"], ["$close"], start=20, end=2)
        expected = self.native.dataset(["AAA"], ["$close"], "2030-01-01", "2030-01-05")
        actual = self.array.dataset(["AAA"], ["$close"], "2030-01-01", "2030-01-05")
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)

    def test_dict_spans_and_instrument_processors_delegate_before_evaluation(self):
        spans = {"AAA": [(self.dates[3], self.dates[8]), (self.dates[14], self.dates[17])]}
        with mock.patch.object(DatasetProvider, "dataset_processor",
                               wraps=DatasetProvider.dataset_processor) as delegated:
            self.paired(spans, ["$close"])
            self.assertEqual(delegated.call_count, 2)
        self.paired(["BBB", "AAA"], ["$close"],
                    inst_processors=[{"class": MultiplyProcessor}])

    def test_duplicate_symbols_use_native_last_instrument_behavior(self):
        self.paired(["BBB", "AAA", "BBB"], ["$close"])

    def test_noncontiguous_datetime_and_dtype_results_use_captured_native_frames(self):
        for kind in ("gapped", "datetime", "float64", "empty", "noncontiguous", "mismatch", "negative"):
            with self.subTest(kind=kind):
                def expression(inst, field, start_time, end_time, freq):
                    if kind == "empty":
                        return pd.Series(dtype=np.float32)
                    indexes = pd.RangeIndex(2, 8)
                    values = np.arange(6, dtype=np.float32)
                    if kind == "gapped":
                        indexes = pd.Index([2, 4, 6, 8, 10, 12])
                    elif kind == "datetime":
                        indexes = pd.DatetimeIndex(self.dates[2:8])
                    elif kind == "float64":
                        values = values.astype(np.float64)
                    elif kind == "noncontiguous":
                        values = np.arange(12, dtype=np.float32)[::2]
                    elif kind == "mismatch" and inst == "BBB" and field == "$open":
                        indexes = pd.RangeIndex(3, 9)
                    elif kind == "negative":
                        indexes = pd.RangeIndex(-6, 0)
                    return pd.Series(values, index=indexes)
                with mock.patch.object(ExpressionD._provider, "expression",
                                       side_effect=expression) as evaluated:
                    self.paired(["BBB", "AAA"], ["$open", "$close", "$close"])
                # Normalization evaluates each unique field once per stock,
                # in both paths, including the captured-values fallback.
                self.assertEqual(evaluated.call_count, 8)

    def test_calendar_index_error_is_not_retried(self):
        series = pd.Series(np.ones(3, dtype=np.float32), index=pd.RangeIndex(100, 103))
        for provider in (self.native, self.array):
            with mock.patch.object(ExpressionD._provider, "expression", return_value=series) as evaluated:
                with self.assertRaises(IndexError):
                    provider.dataset(["AAA", "BBB"], ["$close"], self.dates[2], self.dates[8])
                self.assertEqual(evaluated.call_count, 1)

    def test_native_expression_exception_is_not_retried(self):
        for provider in (self.native, self.array):
            with mock.patch.object(ExpressionD._provider, "expression",
                                   side_effect=RuntimeError("native failure")) as evaluated:
                with self.assertRaisesRegex(RuntimeError, "native failure"):
                    provider.dataset(["AAA", "BBB"], ["$close"], self.dates[2], self.dates[8])
                self.assertEqual(evaluated.call_count, 1)

    def test_nan_bits_and_float32_values_are_preserved(self):
        # Include a noncanonical quiet NaN payload and negative zero, since
        # numeric equality alone would miss changes to these float32 bytes.
        values = np.array([0x7FC01234, 0x80000000, 0x3F800001], dtype=np.uint32).view(np.float32)
        def expression(*args, **kwargs):
            return pd.Series(values, index=pd.RangeIndex(3, 6))
        with mock.patch.object(ExpressionD._provider, "expression", side_effect=expression):
            expected = self.native.dataset(["BBB", "AAA"], ["$close", "$close"],
                                           self.dates[3], self.dates[5])
            actual = self.array.dataset(["BBB", "AAA"], ["$close", "$close"],
                                        self.dates[3], self.dates[5])
        np.testing.assert_array_equal(actual.to_numpy().view(np.uint32),
                                      expected.to_numpy().view(np.uint32))

    def test_custom_expression_provider_time2idx_and_configured_kernels_delegate(self):
        class CustomProvider(LocalExpressionProvider):
            pass
        original = ExpressionD._provider
        cases = [("provider", CustomProvider()), ("time2idx", False), ("kernels", 2)]
        for kind, value in cases:
            with self.subTest(kind=kind):
                if kind == "provider":
                    ExpressionD.register(value)
                elif kind == "time2idx":
                    original.time2idx = value
                else:
                    C.kernels = value
                sentinel = object()
                with mock.patch.object(DatasetProvider, "dataset_processor",
                                       return_value=sentinel) as delegated:
                    result = self.array.dataset(["AAA"], ["$close"], self.dates[2], self.dates[8])
                    self.assertIs(result, sentinel)
                    self.assertEqual(delegated.call_count, 1)
                ExpressionD.register(original)
                original.time2idx = True
                C.kernels = 1

    def test_frequency_specific_kernel_configuration_uses_native_count(self):
        C.kernels = lambda freq: 1 if freq == "day" else 2
        self.paired(["BBB", "AAA"], ["$close"])

    def test_supported_configuration_hook(self):
        qlib.init(provider_uri=str(self.root), region="us", kernels=1,
                  expression_cache=None, dataset_cache=None, logging_level=logging.ERROR,
                  dataset_provider={"class": "ArrayDatasetProvider",
                                    "module_path": "qlib_yahoo.array_dataset_provider"})
        from qlib.data import D
        actual = D.features(["BBB", "AAA"], ["Ref($close,-1)/$close-1", "$close"],
                            self.dates[2], self.dates[20], disk_cache=0)
        expected = self.native.dataset(["BBB", "AAA"], ["Ref($close,-1)/$close-1", "$close"],
                                       self.dates[2], self.dates[20])
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def synthetic_microbenchmark(symbol_count=256, sessions=1255, repeats=3):
    """Time native expressions through completed frames; never a backtest claim.

    Run manually with the Qlib interpreter; not an assertion about host speed.
    The temporary fixture and Qlib initialization are outside the measured
    boundary. Qlib memory caches are cleared before every timed dataset call.
    """
    if qlib is None:
        raise RuntimeError("Qlib is required")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        dates = pd.bdate_range("2016-01-04", periods=sessions)
        (root / "calendars").mkdir()
        (root / "calendars/day.txt").write_text(
            "\n".join(d.strftime("%Y-%m-%d") for d in dates) + "\n")
        (root / "instruments").mkdir()
        symbols = [f"STOCK{i:05}" for i in range(symbol_count)]
        (root / "instruments/all.txt").write_text(
            "".join(f"{inst}\t{dates[0]:%Y-%m-%d}\t{dates[-1]:%Y-%m-%d}\n" for inst in symbols))
        values = np.arange(1, sessions + 1, dtype=np.float32)
        values[50:53] = np.nan
        for inst in symbols:
            path = root / "features" / inst.lower()
            path.mkdir(parents=True)
            for field, scale in (("close", 1), ("open", 2), ("volume", 100)):
                np.concatenate((np.array([0], dtype=np.float32), values * scale)).tofile(
                    path / f"{field}.day.bin")
        qlib.init(provider_uri=str(root), region="us", kernels=1,
                  expression_cache=None, dataset_cache=None, logging_level=logging.ERROR)
        native, array = LocalDatasetProvider(), ArrayDatasetProvider()
        fields = ["$open", "$volume", "Mean($close,5)", "Ref($close,-2)/Ref($close,-1)-1"]
        times = {"native": [], "array": []}
        for repeat in range(repeats):
            results = {}
            pairs = [("native", native), ("array", array)]
            if repeat % 2:
                pairs.reverse()
            for name, provider in pairs:
                H.clear()
                began = time.perf_counter()
                results[name] = provider.dataset(symbols[::-1], fields, dates[4], dates[-3])
                times[name].append(time.perf_counter() - began)
            pd.testing.assert_frame_equal(results["array"], results["native"], check_exact=True)
        return {"symbols": symbol_count, "calendar_sessions": sessions,
                "output_rows": len(results["native"]), "fields": fields,
                "seconds": times,
                "median_seconds": {name: statistics.median(samples) for name, samples in times.items()},
                "exact_frame_parity": True,
                "boundary": "native expression evaluation through full dataset DataFrame construction; excludes fixture/init; not a backtest"}


if __name__ == "__main__":
    unittest.main()
