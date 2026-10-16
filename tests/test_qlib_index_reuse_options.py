"""Explicit date-index reuse guards and isolated native quote-hook checks.

The parent never imports Qlib. Native checks activate the verified signed
runtime in a child before importing any Qlib class; no worker/data run occurs.
"""

import importlib.util
import inspect
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

CHILD = "QLIB_INDEX_REUSE_OPTIONS_CHILD"


class OptionTests(unittest.TestCase):
    def test_retired_projection_rejected_before_sec_runtime_or_dataset_access(self):
        from qlib_yahoo.backtest import run_backtest
        from qlib_yahoo.dataset import DatasetError
        with patch("qlib_yahoo.sec_resolution.resolve_config", side_effect=AssertionError("SEC resolution")) as sec, \
                patch("qlib_yahoo.runtime.ensure_signed_runtime", side_effect=AssertionError("runtime")) as runtime, \
                patch("qlib_yahoo.backtest.verify_dataset", side_effect=AssertionError("dataset")) as dataset:
            for value in (True, 0, 1, None, "true", 1., [], {}):
                for profile in ("auto", "native"):
                    with self.subTest(value=value, profile=profile), self.assertRaisesRegex(DatasetError, "V4.*unsupported.*retired"):
                        run_backtest(Path("unused"), "$close", strategy_mode="signed",
                                     runtime_profile=profile, _index_reuse=True, _numeric_projection=value)
            sec.assert_not_called()
            runtime.assert_not_called()
            dataset.assert_not_called()

    def test_backtest_default_and_invalid_options_fail_before_dataset_access(self):
        from qlib_yahoo.backtest import run_backtest
        from qlib_yahoo.dataset import DatasetError
        self.assertIs(inspect.signature(run_backtest).parameters["_index_reuse"].default, False)
        self.assertIs(inspect.signature(run_backtest).parameters["_numeric_projection"].default, False)
        choices = [{"_index_reuse": value} for value in (0, 1, None, "true", 1.0, [], {})]
        choices += [{"_numeric_projection": value} for value in (0, 1, None, "true", 1.0, [], {})]
        choices += [{"_index_reuse": True, "strategy_mode": "long-only"},
                    {"_index_reuse": True, "strategy_mode": "signed", "execution_profile": "reference"},
                    {"_numeric_projection": True, "strategy_mode": "signed"},
                    {"_numeric_projection": True, "_index_reuse": True, "strategy_mode": "long-only"}]
        with patch.object(Path, "read_text", side_effect=AssertionError("dataset access")) as read, \
                patch("qlib_yahoo.backtest.verify_dataset", side_effect=AssertionError("dataset verification")) as verify:
            for options in choices:
                with self.subTest(options=options), self.assertRaises(DatasetError):
                    run_backtest(Path("unused"), "$close", **options)
            read.assert_not_called()
            verify.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("qlib"), "optional native Qlib unavailable")
    @unittest.skipIf(os.environ.get(CHILD) == "1", "parent launcher")
    def test_isolated_verified_native_quote_options(self):
        from qlib_yahoo.runtime import ROOT, signed_runtime_status
        ready, reason = signed_runtime_status()
        if not ready:
            self.skipTest(reason)
        environment = os.environ.copy()
        environment[CHILD] = "1"
        program = (
            "from qlib_yahoo.runtime import ensure_signed_runtime; ensure_signed_runtime(); "
            "import unittest; result=unittest.TextTestRunner(verbosity=2).run("
            "unittest.defaultTestLoader.loadTestsFromName('tests.test_qlib_index_reuse_options.SignedChild')); "
            "raise SystemExit(not result.wasSuccessful())"
        )
        result = subprocess.run([sys.executable, "-c", program], cwd=ROOT, env=environment,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(os.environ.get(CHILD) == "1", "requires verified signed child")
class SignedChild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import pandas as pd
        from qlib.backtest.high_performance_ds import NumpyQuote
        from qlib_yahoo.daily_quote import DailyQuote
        cls.np, cls.pd = np, pd
        cls.Native, cls.Quote = NumpyQuote, DailyQuote

    def setUp(self):
        policy = self.np.errstate(invalid="ignore")
        policy.__enter__()
        self.addCleanup(policy.__exit__, None, None, None)
        dates = self.pd.bdate_range("2025-01-02", periods=4).as_unit("ns")
        index = self.pd.MultiIndex.from_product([["AAA", "BBB", "CCC"], dates],
                                               names=["instrument", "datetime"])
        self.frame = self.pd.DataFrame({"$open": self.np.arange(12, dtype=self.np.float32),
            "$close": self.np.arange(12, dtype=self.np.float64),
            "$factor": self.np.ones(12), "limit_buy": [False, True, False, False] * 3}, index=index)
        self.frame.iloc[1, 1] = self.np.nan
        self.frame.iloc[2, 1] = -0.

    def assert_panels_equal(self, actual, expected):
        self.assertEqual(list(actual), list(expected))
        for symbol in actual:
            a, e = actual[symbol], expected[symbol]
            self.assertIs(type(a), type(e))
            self.assertEqual(a.data.dtype, e.data.dtype)
            self.assertEqual(a.data.tobytes(), e.data.tobytes())
            self.assertEqual(a.data.strides, e.data.strides)
            for ai, ei in zip(a.indices, e.indices):
                self.assertIs(type(ai), type(ei))
                self.assertEqual(ai.idx_list.dtype, ei.idx_list.dtype)
                self.np.testing.assert_array_equal(ai.idx_list, ei.idx_list)
                self.assertEqual(ai.index_map, ei.index_map)
                self.assertEqual(ai.is_sorted(), ei.is_sorted())
                self.assertEqual(ai.idx_list.flags.writeable, ei.idx_list.flags.writeable)

    def test_option_exact_bool_before_construction(self):
        with patch("qlib_yahoo.daily_quote.native_quote_panels", side_effect=AssertionError("default helper")), \
                patch("qlib_yahoo.quote_construction_v3.native_quote_panels_v3", side_effect=AssertionError("V3 helper")):
            for value in (0, 1, None, "true", 1.0, self.np.bool_(True), [], {}):
                with self.subTest(value=value), self.assertRaisesRegex(TypeError, "boolean"):
                    self.Quote(self.frame, "day", _index_reuse=value)

    def test_default_uses_existing_helper_and_does_not_import_candidate(self):
        from qlib_yahoo.daily_quote import native_quote_panels
        with patch.dict(sys.modules, {"qlib_yahoo.quote_construction_v3": None}), \
                patch.dict(sys.modules, {"qlib_yahoo.quote_construction_v4": None}), \
                patch("qlib_yahoo.daily_quote.native_quote_panels", wraps=native_quote_panels) as helper:
            for options in ({}, {"_index_reuse": False}):
                quote = self.Quote(self.frame, "day", **options)
                self.assertIs(type(quote), self.Quote)
                self.assertIsNone(quote.index_reuse_stats)
                self.assert_panels_equal(quote.data, self.Native(self.frame, "day").data)
            self.assertEqual(helper.call_count, 2)

    def test_disabled_legacy_projection_preserves_v1_and_v3(self):
        for index_reuse in (False, True):
            with self.subTest(index_reuse=index_reuse):
                quote = self.Quote(self.frame, "day", _index_reuse=index_reuse, _numeric_projection=False)
                self.assert_panels_equal(quote.data, self.Native(self.frame, "day").data)

    def test_retired_numeric_projection_rejected_before_any_constructor(self):
        with patch("qlib_yahoo.daily_quote.native_quote_panels", side_effect=AssertionError("V1 construction")), \
                patch("qlib_yahoo.quote_construction_v3.native_quote_panels_v3", side_effect=AssertionError("V3 construction")), \
                patch.object(self.Native, "__init__", side_effect=AssertionError("native construction")):
            for index_reuse in (False, True):
                for value in (True, 0, 1, None, "true", 1., self.np.bool_(True), [], {}):
                    with self.subTest(index_reuse=index_reuse, value=value), self.assertRaisesRegex(ValueError, "V4.*unsupported.*retired"):
                        self.Quote(self.frame, "day", _index_reuse=index_reuse, _numeric_projection=value)

    def test_selected_v3_diagnostics_and_exact_type_cache_binding(self):
        from qlib_yahoo.quote_construction_v3 import native_quote_panels_v3
        with patch("qlib_yahoo.daily_quote.native_quote_panels", side_effect=AssertionError("default helper")), \
                patch("qlib_yahoo.quote_construction_v3.native_quote_panels_v3", wraps=native_quote_panels_v3) as helper:
            quote = self.Quote(self.frame, "day", region="us", _index_reuse=True)
        helper.assert_called_once()
        self.assertIs(type(quote), self.Quote)
        self.assert_panels_equal(quote.data, self.Native(self.frame, "day", "us").data)
        stats = quote.index_reuse_stats
        self.assertIs(stats["requested"], True)
        self.assertIs(stats["fast"], True)
        self.assertIs(stats["native_fallback"], False)
        self.assertEqual((stats["hits"], stats["misses"], stats["cap"], stats["peak_entries"]), (2, 1, 32, 1))
        self.assertIs(quote.get_data, quote._native_get_data_cache)
        self.assertIs(quote.get_data.__wrapped__, quote._native_get_data_method)
        self.assertIs(quote._native_get_data_method.__func__, self.Quote.get_data)
        self.assertIs(quote._native_get_data_method.__self__, quote)
        self.assertEqual(quote.get_data.cache_info().maxsize, 512)
        other = self.Quote(self.frame, "day", _index_reuse=True)
        self.assertIsNot(other.get_data, quote.get_data)
        start = self.pd.Timestamp("2025-01-02")
        args = ("AAA", start, start, "$close")
        quote.get_data(*args)
        quote.get_data(*args)
        self.assertEqual(quote.get_data.cache_info().hits, 1)
        self.assertEqual(other.get_data.cache_info().currsize, 0)

    def test_selected_panels_detach_every_mutable_axis_map_and_source(self):
        original = self.frame.copy(deep=True)
        quote = self.Quote(self.frame, "day", _index_reuse=True)
        panels = list(quote.data.values())
        for axis in (0, 1):
            self.assertEqual(len({id(panel.indices[axis].index_map) for panel in panels}), len(panels))
            for a, b in zip(panels, panels[1:]):
                self.assertFalse(self.np.shares_memory(a.indices[axis].idx_list, b.indices[axis].idx_list))
        for panel in panels:
            self.assertFalse(self.np.shares_memory(panel.index.idx_list, self.frame.index.levels[1].to_numpy(copy=False)))
            for column in self.frame.columns:
                self.assertFalse(self.np.shares_memory(panel.data, self.frame[column].to_numpy(copy=False)))
        preserved = self.Native(self.frame, "day").data
        panels[0].data[:] = -123
        panels[0].index.idx_list[0] = self.np.datetime64("1999-01-01", "ns")
        panels[0].index.index_map.clear()
        panels[0].indices[1].idx_list[0] = "$changed"
        panels[0].indices[1].index_map.clear()
        self.assert_panels_equal({key: quote.data[key] for key in ("BBB", "CCC")},
                                 {key: preserved[key] for key in ("BBB", "CCC")})
        self.pd.testing.assert_frame_equal(self.frame, original, check_exact=True)
        fresh = self.Quote(self.frame, "day", _index_reuse=True)
        self.assert_panels_equal(fresh.data, preserved)
        self.assertEqual((fresh.index_reuse_stats["hits"], fresh.index_reuse_stats["misses"]), (2, 1))

    def test_unsupported_policy_declines_before_numeric_callbacks_then_uses_native(self):
        from qlib_yahoo.quote_construction_v3 import native_quote_panels_v3
        from unittest.mock import Mock
        signalling = self.frame.copy(deep=True)
        values = signalling["$open"].to_numpy(copy=False)
        values.view(self.np.uint32)[0] = 0x7f800001
        callback = Mock(side_effect=AssertionError("numeric callback"))
        callback.write.side_effect = AssertionError("numeric log callback")
        old_callback = self.np.geterrcall()
        self.np.seterrcall(callback)
        try:
            for policy in ("warn", "raise", "call", "log"):
                with self.subTest(policy=policy), self.np.errstate(invalid=policy):
                    diagnostics = {}
                    self.assertIsNone(native_quote_panels_v3(signalling, "day", diagnostics=diagnostics))
                    self.assertEqual(diagnostics, {})
                    callback.assert_not_called()
                    callback.write.assert_not_called()
        finally:
            self.np.seterrcall(old_callback)
        original_init = self.Native.__init__
        calls = []
        def observed(instance, *args, **kwargs):
            calls.append(instance)
            original_init(instance, *args, **kwargs)
        with self.np.errstate(invalid="warn"), \
                patch.object(self.Native, "__init__", observed), \
                patch("qlib_yahoo.daily_quote.native_quote_panels", side_effect=AssertionError("default helper")):
            quote = self.Quote(self.frame, "day", _index_reuse=True)
        self.assertEqual(calls, [quote])
        self.assertEqual(quote.index_reuse_stats, {"requested": True, "fast": False, "native_fallback": True})
        self.assert_panels_equal(quote.data, self.Native(self.frame, "day").data)
        self.assertIs(type(quote), self.Quote)
        self.assertIs(quote.get_data, quote._native_get_data_cache)

    def test_lookup_values_types_errors_and_missing_windows_match_native(self):
        quote = self.Quote(self.frame, "day", region="us", _index_reuse=True)
        native = self.Native(self.frame, "day", "us")
        p = self.pd.Timestamp
        windows = ((p("2025-01-02"), p("2025-01-02")),
                   (p("2025-01-03"), p("2025-01-05 23:59:59")),
                   (p("2025-01-04"), p("2025-01-05")),
                   (p("2025-01-02"), p("2025-01-07")))
        def outcome(instance, *args):
            try:
                value = instance.get_data(*args)
                if hasattr(value, "indices"):
                    return (type(value), value.data.tobytes(), value.index.idx_list.tobytes())
                if hasattr(value, "tobytes"):
                    return (type(value), value.tobytes())
                return (type(value), value)
            except Exception as exc:
                return (type(exc), str(exc))
        for start, end in windows:
            for symbol in ("AAA", "BBB", "MISSING"):
                for field in ("$close", "$open", "limit_buy", "$missing"):
                    for method in (None, "sum", "mean", "all", "last", "ts_data_last", "invalid"):
                        with self.subTest(start=start, end=end, symbol=symbol, field=field, method=method):
                            self.assertEqual(outcome(quote, symbol, start, end, field, method),
                                             outcome(native, symbol, start, end, field, method))

    def test_supported_exchange_callable_factory_retains_exact_daily_quote(self):
        from functools import partial
        from qlib.backtest.exchange import Exchange
        from qlib.config import C
        def provide(exchange):
            exchange.quote_df = self.frame
        factory = partial(self.Quote, _index_reuse=True)
        with patch.object(Exchange, "get_quote_from_qlib", provide), \
                patch.dict(C._config, {"trade_unit": 1, "deal_price": "open", "limit_threshold": None, "region": "us"}):
            exchange = Exchange(freq="day", codes=["AAA", "BBB", "CCC"], deal_price="$open",
                                limit_threshold=None, trade_unit=1, quote_cls=factory)
        self.assertIs(exchange.quote_cls, factory)
        self.assertIs(type(exchange.quote), self.Quote)
        self.assertIs(exchange.quote.get_data, exchange.quote._native_get_data_cache)
        self.assertTrue(exchange.quote.index_reuse_stats["fast"])
        self.assert_panels_equal(exchange.quote.data, self.Native(self.frame, "day").data)


if __name__ == "__main__":
    unittest.main()
