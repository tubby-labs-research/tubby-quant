"""Native da54853c differentials for interval reuse and independent sizing."""

import importlib.util
import json
from fractions import Fraction
import os
import subprocess
import sys
import unittest
import warnings
from unittest.mock import patch

from qlib_yahoo.dataset import DatasetError
from qlib_yahoo.runtime import PIN, ROOT, platform_pin


CHILD = "QLIB_FAST_SIGNED_CHILD"


def _signed_available():
    try:
        pin = platform_pin(json.loads(PIN.read_text(encoding="utf-8")))
        return (sys.version_info[:2] == (3, 12)
                and (ROOT / pin["local_runtime"] / "qlib/_tubby_build.json").is_file())
    except (OSError, ValueError, KeyError, DatasetError):
        return False


class LauncherTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("qlib") and _signed_available(),
                         "pinned signed derivative unavailable")
    @unittest.skipIf(os.environ.get(CHILD) == "1", "parent launcher")
    def test_actual_pinned_signed_differentials(self):
        program = (
            "from qlib_yahoo.runtime import ensure_signed_runtime; ensure_signed_runtime(); "
            "import unittest; suite=unittest.defaultTestLoader.loadTestsFromName("
            "'tests.test_qlib_fast_signed.SignedChild'); "
            "result=unittest.TextTestRunner(verbosity=2).run(suite); "
            "raise SystemExit(not result.wasSuccessful())"
        )
        environment = os.environ.copy()
        environment[CHILD] = "1"
        result = subprocess.run([sys.executable, "-c", program], cwd=ROOT,
                                env=environment, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(os.environ.get(CHILD) == "1", "requires verified signed child")
class SignedChild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import pandas as pd
        from qlib.backtest.decision import Order
        from qlib.backtest.high_performance_ds import NumpyQuote, PandasQuote
        from qlib.backtest.signed_exchange import SignedExchange
        from qlib.backtest.signed_position import SignedPosition
        from qlib.contrib.strategy.signed_weight import SignedWeightStrategy
        from qlib_yahoo.daily_quote import DailyQuote
        from qlib_yahoo.fast_signed import FastSignedExchange, FastSignedWeightStrategy

        cls.np, cls.pd, cls.Order = np, pd, Order
        cls.NumpyQuote, cls.PandasQuote, cls.DailyQuote = NumpyQuote, PandasQuote, DailyQuote
        cls.SignedExchange, cls.SignedPosition = SignedExchange, SignedPosition
        cls.NativeStrategy, cls.FastStrategy = SignedWeightStrategy, FastSignedWeightStrategy
        cls.FastExchange = FastSignedExchange
        cls.dates = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"])

    def frame(self, opens=None, factors=None):
        p, np = self.pd, self.np
        opens = opens or {"AAA": 10.0, "BBB": 20.0, "CCC": 30.0}
        index = p.MultiIndex.from_product([list(opens), self.dates], names=["instrument", "datetime"])
        frame = p.DataFrame({"$open": [value for value in opens.values() for _ in self.dates]}, index=index)
        frame["$close"] = frame["$open"] + 1
        frame["$volume"] = 123.0
        frame["$factor"] = [float((factors or {}).get(symbol, 1)) for symbol in opens for _ in self.dates]
        suspended = frame["$open"].isna() | (frame["$open"] <= 0)
        frame["limit_buy"] = suspended
        frame["limit_sell"] = suspended
        return frame

    def exchange(self, fast, frame=None, quote_class=None):
        # All query/check/fill methods are the actual pinned implementations;
        # only Qlib provider loading is avoided by supplying a tiny quote frame.
        cls = self.FastExchange if fast else self.SignedExchange
        exchange = cls.__new__(cls)
        exchange.quote_df = self.frame() if frame is None else frame
        exchange.quote = (quote_class or self.NumpyQuote)(exchange.quote_df, "day", region="us")
        exchange.buy_price = exchange.sell_price = "$open"
        exchange.open_cost = exchange.close_cost = 0.0005
        if fast:
            exchange.reset_session_cache()
        return exchange

    def strategy(self, fast, weights, *, cash=100.0, frame=None, holdings=None, step=1, exchange=None):
        from qlib.backtest.signal import Signal

        dates, pd = self.dates, self.pd

        class SessionSignal(Signal):
            def get_signal(self, start_time, end_time):
                return weights

        class Calendar:
            start_index = 0

            def get_trade_step(self):
                return step

            def get_step_time(self, index=None, shift=0):
                date = dates[(step if index is None else index) - shift]
                return date, date + pd.Timedelta(hours=23)

        position = self.SignedPosition(cash=cash)
        for symbol, amount in (holdings or {}).items():
            price = float(frame.loc[(symbol, dates[0]), "$open"]) if frame is not None else 10.0
            position._init_stock(symbol, amount, price)
            if amount < 0:
                position.restricted_proceeds[symbol] = abs(amount) * price
        strategy = (self.FastStrategy if fast else self.NativeStrategy)(
            signal=SessionSignal(), level_infra={"trade_calendar": Calendar()},
            common_infra={"trade_account": type("AccountStub", (), {"current_position": position})()},
            trade_exchange=exchange or self.exchange(fast, frame),
        )
        return strategy, position

    @staticmethod
    def orders(orders):
        return [(o.stock_id, o.amount, o.direction, o.start_time, o.end_time) for o in orders]

    def outcome(self, strategy):
        try:
            decision = strategy.generate_trade_decision()
            result = ("value", self.orders(decision.get_decision()))
        except Exception as exc:
            result = ("error", type(exc), str(exc))
        records = []
        for record in strategy.decisions:
            records.append({**record, "issued_orders": self.orders(record["issued_orders"])})
        return result, records

    def assert_strategies(self, weights, **kwargs):
        native, _ = self.strategy(False, weights, **kwargs)
        fast, _ = self.strategy(True, weights, **kwargs)
        expected, actual = self.outcome(native), self.outcome(fast)
        self.assertEqual(expected, actual)
        return actual

    def query_outcome(self, function):
        try:
            return "value", function()
        except Exception as exc:
            return "error", type(exc), str(exc)

    def assert_query(self, native, fast, name, *arguments, **kwargs):
        expected = self.query_outcome(lambda: getattr(native, name)(*arguments, **kwargs))
        actual = self.query_outcome(lambda: getattr(fast, name)(*arguments, **kwargs))
        self.assertEqual(expected[0], actual[0])
        if expected[0] == "error":
            self.assertEqual(expected, actual)
        elif expected[1] is None:
            self.assertIsNone(actual[1])
        elif hasattr(expected[1], "indices"):
            self.assertEqual(type(expected[1]), type(actual[1]))
            self.np.testing.assert_array_equal(expected[1].data, actual[1].data)
            self.np.testing.assert_array_equal(expected[1].index.idx_list, actual[1].index.idx_list)
        else:
            self.assertEqual(type(expected[1]), type(actual[1]))
            self.np.testing.assert_equal(expected[1], actual[1])

    def test_query_weekends_gaps_factors_directions_and_errors(self):
        p, np = self.pd.Timestamp, self.np
        frame = self.frame({"AAA": 10.0, "BBB": np.nan}, {"AAA": 2.0})
        frame.loc[("AAA", self.dates[1]), "$factor"] = np.nan
        windows = [(p("2024-01-05"), p("2024-01-07 23:59:59")),
                   (p("2024-01-06"), p("2024-01-07")),
                   (p("2024-01-08"), p("2024-01-08 23:59:59")),
                   (p("2024-01-05"), p("2024-01-09")),
                   (p("2024-01-05", tz="UTC"), p("2024-01-07", tz="UTC"))]
        for quote in (self.NumpyQuote, self.DailyQuote):
            native, fast = self.exchange(False, frame, quote), self.exchange(True, frame, quote)
            for start, end in windows:
                fast.begin_session(start, end)
                for stock in ("AAA", "BBB", "MISSING"):
                    for direction in (None, self.Order.BUY, self.Order.SELL, 13):
                        self.assert_query(native, fast, "is_stock_tradable", stock, start, end, direction)
                        self.assert_query(native, fast, "check_stock_limit", stock, start, end, direction)
                        self.assert_query(native, fast, "get_deal_price", stock, start, end, direction)
                    self.assert_query(native, fast, "get_factor", stock, start, end)
                    self.assert_query(native, fast, "get_close", stock, start, end)
                    self.assert_query(native, fast, "check_stock_suspended", stock, start, end)
                    for method in (None, "sum", "all", "mean", "last", "ts_data_last", "bad"):
                        self.assert_query(native, fast, "get_deal_price", stock, start, end, self.Order.BUY, method)
                        self.assert_query(native, fast, "get_close", stock, start, end, method)
                    for field in ("$open", "$close", "$factor", "$volume", "$missing"):
                        for method in (None, "sum", "all", "mean", "last", "ts_data_last", "bad"):
                            self.assert_query(native, fast, "get_quote_info", stock, start, end, field, method)

    def test_cache_one_interval_size_reset_and_unknown_requests(self):
        fast = self.exchange(True)
        self.assertTrue(fast.supports_session_rows)
        self.assertEqual(fast.quote.freq, self.pd.Timedelta(days=1))
        self.assertIs(type(fast.quote), self.NumpyQuote)
        start, end = self.dates[0], self.dates[0] + self.pd.Timedelta(days=2, hours=23)
        self.assertTrue(fast.begin_session(start, end))
        with patch.object(fast.quote, "get_data", wraps=fast.quote.get_data) as native:
            fast.get_deal_price("AAA", start, end, self.Order.BUY)
            fast.get_deal_price("AAA", start, end, self.Order.SELL)
            self.assertEqual(native.call_count, 0)
            for _ in range(3):
                fast.is_stock_tradable("AAA", start, end)
                fast.check_stock_suspended("AAA", start, end)
            self.assertEqual(list(fast._session_rows), ["AAA"])
            self.assertEqual(native.call_count, 0)
            for i in range(30):
                fast.get_deal_price(f"UNKNOWN{i}", start, end, self.Order.BUY)
            self.assertEqual(list(fast._session_rows), ["AAA"])
            self.assertEqual(native.call_count, 30)
            fast.reset_session_cache()
            self.assertEqual(fast._session_rows, {})
            self.assertIsNone(fast._session_start)
            fast.begin_session(start, end)
            fast.get_deal_price("AAA", start, end, self.Order.BUY)
            self.assertEqual(native.call_count, 30)
            fast.begin_session(self.dates[1], self.dates[1])
            fast.get_factor("BBB", self.dates[1], self.dates[1])
            self.assertEqual(list(fast._session_rows), ["BBB"])
        other = self.exchange(True, self.frame({"AAA": 99.0}))
        other.begin_session(start, end)
        self.assertEqual(other.get_deal_price("AAA", start, end, self.Order.BUY), 99)
        self.assertIsNot(fast._session_rows, other._session_rows)

    def test_multibar_timezone_and_mutable_results_do_not_enter_cache(self):
        p = self.pd.Timestamp
        fast = self.exchange(True)
        for start, end, method in ((p("2024-01-05"), p("2024-01-09"), "sum"),
                                   (p("2024-01-05", tz="UTC"), p("2024-01-07", tz="UTC"), "sum"),
                                   (p("2024-01-05"), p("2024-01-07"), None)):
            fast.reset_session_cache()
            fast.begin_session(start, end)
            with patch.object(fast.quote, "get_data", wraps=fast.quote.get_data) as native:
                for _ in range(2):
                    self.query_outcome(lambda: fast.get_quote_info("AAA", start, end, "$open", method))
                self.assertEqual(native.call_count, 2)
                self.assertEqual(fast._session_rows, {})

    def test_unusual_query_inputs_delegate_without_disturbing_valid_cache(self):
        p, np = self.pd.Timestamp, self.np
        start, end = self.dates[0], self.dates[0] + self.pd.Timedelta(days=2, hours=23)
        unusual = ([start, end], np.array([start, end]), str(start), None, self.pd.NaT,
                   p("5000-01-01"))
        for quote in (self.NumpyQuote, self.DailyQuote):
            native, fast = self.exchange(False, quote_class=quote), self.exchange(True, quote_class=quote)
            fast.begin_session(start, end)
            fast.get_deal_price("AAA", start, end, self.Order.BUY)
            interval, values = (fast._session_start, fast._session_end), fast._session_rows.copy()
            for value in unusual:
                for left, right in ((value, end), (start, value)):
                    for name, tail in (("get_deal_price", (self.Order.BUY,)),
                                       ("get_factor", ()),
                                       ("is_stock_tradable", (self.Order.BUY,))):
                        self.assert_query(native, fast, name, "AAA", left, right, *tail)
                    self.assertEqual((fast._session_start, fast._session_end), interval)
                    self.assertEqual(fast._session_rows, values)
            for stock in (["AAA"], np.array(["AAA", "BBB"]), None, 13):
                self.assert_query(native, fast, "get_deal_price", stock, start, end, self.Order.BUY)
                self.assertEqual((fast._session_start, fast._session_end), interval)
                self.assertEqual(fast._session_rows, values)
            for field, method in ((["$open"], "sum"), ("$open", np.array(["sum", "last"])),
                                  ("$unknown", "sum"), ("$open", "unknown")):
                self.assert_query(native, fast, "get_quote_info", "AAA", start, end, field, method)
                self.assertEqual((fast._session_start, fast._session_end), interval)
                self.assertEqual(fast._session_rows, values)
            for left, right in ((start, self.dates[2]),
                                (start.tz_localize("UTC"), end.tz_localize("UTC")),
                                (start + self.pd.Timedelta(hours=1), end)):
                self.assert_query(native, fast, "get_deal_price", "AAA", left, right, self.Order.BUY)
                self.assertEqual((fast._session_start, fast._session_end), interval)
                self.assertEqual(fast._session_rows, values)
            # Native handles nonscalar directions itself, including its normal
            # suspension short-circuit. Cache guards must not compare arrays.
            for direction in ([self.Order.BUY], np.array([self.Order.BUY, self.Order.SELL]),
                              np.array(self.Order.BUY), "buy", None):
                for name in ("is_stock_tradable", "check_stock_limit", "get_deal_price"):
                    self.assert_query(native, fast, name, "AAA", start, end, direction)

    def test_weights_validation_and_record_reasons_are_native(self):
        p, np = self.pd, self.np
        for weights in (p.Series({"CCC": -0.25, "AAA": 0.5, "BBB": 0.0}),
                        p.Series({"AAA": -1.0}), p.Series({"AAA": -0.0}),
                        p.Series(dtype=float), p.Series({"AAA": np.nan}),
                        p.Series({"AAA": np.inf}), p.Series({"AAA": True}),
                        p.Series({"": 0.1}), p.Series({12: 0.1}),
                        p.Series({"AAA": "0.1"}), p.Series({"AAA": 1.1}),
                        p.Series([0.2, 0.3], index=["AAA", "AAA"]),
                        p.Series([0.1], index=p.MultiIndex.from_tuples([("AAA", 1)]))):
            self.assert_strategies(weights)
        self.assert_strategies(None)
        self.assert_strategies(p.Series({"AAA": 1.0}), step=0)
        self.assert_strategies(p.Series({"AAA": 1.0}), cash=0)
        self.assert_strategies(p.Series({"AAA": 0.5, "BBB": -0.5}),
                               frame=self.frame({"AAA": np.nan, "BBB": 20.0}))

    def test_round_boundaries_and_arbitrary_size_integer_targets(self):
        np, p = self.np, self.pd
        for price in (1., 3., 0.1, np.nextafter(10., 0.), 10., np.nextafter(10., np.inf), 1e-300):
            for equity in (100., np.nextafter(100., 0.), np.nextafter(100., np.inf), float(2**53), 1e100):
                self.assert_strategies(p.Series({"AAA": 0.5, "BBB": -0.5}), cash=float(equity),
                                       frame=self.frame({"AAA": float(price), "BBB": float(price)}))

    def test_factor_and_overflow_exception_precedence(self):
        p, np = self.pd, self.np
        for factor in (2., np.nan, np.inf, 0.):
            result = self.assert_strategies(p.Series({"AAA": 0.5, "BBB": -0.5}),
                                           frame=self.frame({"AAA": 10., "BBB": 20.}, {"AAA": factor}))
            self.assertEqual(result[0][1], NotImplementedError)
        # An unavailable first symbol must not validate its invalid factor.
        self.assert_strategies(p.Series({"AAA": 0.5, "BBB": -0.5}),
                               frame=self.frame({"AAA": np.nan, "BBB": 20.}, {"AAA": 2.}))
        # Earlier overflowing scalar floor beats a later bad factor.
        result = self.assert_strategies(p.Series({"AAA": 0.5, "BBB": -0.5}), cash=1e100,
                                       frame=self.frame({"AAA": 1e-300, "BBB": 20.}, {"BBB": 2.}))
        self.assertEqual(result[0][1], OverflowError)

    def test_scalar_numeric_fallback_and_invalid_holdings(self):
        for price in (self.np.float32(3.0), Fraction(3, 7), 3):
            outcomes = []
            for fast in (False, True):
                strategy, _ = self.strategy(fast, self.pd.Series({"AAA": -0.4, "BBB": 0.5}))
                with patch.object(strategy.trade_exchange, "get_deal_price", return_value=price):
                    outcomes.append(self.outcome(strategy))
            self.assertEqual(outcomes[0], outcomes[1])
        for holding in (0.5, self.np.inf, self.np.nan, "bad"):
            outcomes = []
            for fast in (False, True):
                strategy, position = self.strategy(fast, self.pd.Series({"AAA": 0.5}))
                with patch.object(position, "get_stock_amount_dict", return_value={"AAA": holding}):
                    outcomes.append(self.outcome(strategy))
            self.assertEqual(outcomes[0], outcomes[1])
            self.assertEqual(outcomes[0][0][1], ValueError)

    def test_strict_numpy_errors_and_unavailable_overflow_keep_native_precedence(self):
        with self.np.errstate(all="raise"):
            self.assert_strategies(self.pd.Series({"AAA": 1e-300, "BBB": 0.5}),
                                   frame=self.frame({"AAA": 1e300, "BBB": 20.}, {"BBB": 2.}))
            self.assert_strategies(self.pd.Series({"AAA": 0.5, "BBB": -0.5}), cash=1e100,
                                   frame=self.frame({"AAA": 1e-300, "BBB": 20.}, {"BBB": 2.}))
        # The gross tolerance allows this input. Native never multiplies it
        # when today's open is unavailable; row acceleration must not introduce
        # sizing or emit overflow warnings for it.
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            self.assert_strategies(self.pd.Series({"AAA": 1. + 1e-13}), cash=1.7976931348623157e308,
                                   frame=self.frame({"AAA": self.np.nan}))
        self.assertFalse([item for item in observed if "overflow" in str(item.message)])

    def test_direction_specific_flags_and_field_methods_are_not_shared(self):
        frame = self.frame()
        frame["limit_buy"] = True
        frame["limit_sell"] = False
        frame.loc[("AAA", self.dates[0]), "$volume"] = self.np.nan
        native, fast = self.exchange(False, frame), self.exchange(True, frame)
        start, end = self.dates[0], self.dates[0] + self.pd.Timedelta(days=2, hours=23)
        fast.begin_session(start, end)
        for _ in range(2):
            for direction in (None, self.Order.BUY, self.Order.SELL):
                self.assert_query(native, fast, "is_stock_tradable", "AAA", start, end, direction)
            for method in ("sum", "ts_data_last", "all"):
                self.assert_query(native, fast, "get_volume", "AAA", start, end, method)
        self.assertIs(fast.is_stock_tradable("AAA", start, end, self.Order.SELL), True)
        self.assertIs(fast.is_stock_tradable("AAA", start, end, self.Order.BUY), False)
        self.assertEqual(fast.get_volume("AAA", start, end, "sum"), 0.)
        self.assertIsNone(fast.get_volume("AAA", start, end, "ts_data_last"))

    def test_reversal_reduction_order_and_native_fills(self):
        frame = self.frame({"AAA": 10., "BBB": 20., "CCC": 30.})
        weights = self.pd.Series({"BBB": 0.2, "AAA": -0.4, "CCC": 0.})
        kwargs = dict(frame=frame, holdings={"AAA": 5, "BBB": -4, "CCC": 2})
        strategies = [self.strategy(fast, weights, **kwargs) for fast in (False, True)]
        self.assertEqual(self.outcome(strategies[0][0]), self.outcome(strategies[1][0]))
        fills = []
        for strategy, position in strategies:
            self.assertEqual([o.stock_id for o in strategy.decisions[0]["issued_orders"][:3]],
                             ["AAA", "CCC", "BBB"])
            orders = strategy.decisions[0]["issued_orders"]
            fills.append([(strategy.trade_exchange.deal_order(order, position=position),
                           order.deal_amount, order.signed_skip_reason, order.signed_partial_reason)
                          for order in orders])
        self.assertEqual(fills[0], fills[1])
        self.assertEqual(strategies[0][1].position, strategies[1][1].position)
        self.assertEqual(strategies[0][1].restricted_proceeds, strategies[1][1].restricted_proceeds)
        self.assertEqual(self.FastExchange.deal_order, self.SignedExchange.deal_order)
        self.assertEqual(self.FastExchange._calc_trade_info_by_order,
                         self.SignedExchange._calc_trade_info_by_order)

    def test_native_strategy_fallback_for_unsupported_exchange_quote(self):
        weights = self.pd.Series({"AAA": 0.5})
        for exchange in (self.exchange(False), self.exchange(True)):
            if isinstance(exchange, self.FastExchange):
                # A genuine alternate quote implementation remains native.
                exchange.quote = self.PandasQuote(self.frame(), "day")
                exchange.reset_session_cache()
            strategy, _ = self.strategy(True, weights, exchange=exchange)
            sentinel = object()
            with patch.object(self.NativeStrategy, "generate_trade_decision", return_value=sentinel) as native:
                self.assertIs(strategy.generate_trade_decision(), sentinel)
                native.assert_called_once_with(None)

    def test_missing_fields_empty_and_duplicate_bars_use_original_quote(self):
        start, end = self.dates[0], self.dates[0] + self.pd.Timedelta(days=2, hours=23)
        frames = [self.frame().drop(columns=[field]) for field in ("$open", "$factor", "$close", "limit_sell")]
        frames.append(self.pd.concat([self.frame(), self.frame().iloc[:1]]).sort_index())
        for frame in frames:
            native, fast = self.exchange(False, frame), self.exchange(True, frame)
            fast.begin_session(start, end)
            with patch.object(fast.quote, "get_data", wraps=fast.quote.get_data) as fallback:
                self.assert_query(native, fast, "get_deal_price", "AAA", start, end, self.Order.BUY)
                self.assertGreater(fallback.call_count, 0)
                self.assertEqual(fast._session_rows, {})
        fast = self.exchange(True)
        empty_start = self.pd.Timestamp("2024-01-06")
        fast.begin_session(empty_start, end)
        with patch.object(fast.quote, "get_data", wraps=fast.quote.get_data) as fallback:
            self.assertIsNone(fast.get_deal_price("AAA", empty_start, end, self.Order.BUY))
            self.assertEqual(fallback.call_count, 1)
            self.assertEqual(fast._session_rows, {})

    def test_begin_session_guards_reset_and_equal_timestamp_objects(self):
        start, end = self.dates[0], self.dates[0] + self.pd.Timedelta(days=2, hours=23)
        fast = self.exchange(True)
        self.assertTrue(fast.begin_session(start, end))
        fast.get_deal_price("AAA", start, end, self.Order.BUY)
        for bad in ([start, end], self.np.array([start, end]), None, str(start), self.pd.NaT,
                    start.tz_localize("UTC"), self.pd.Timestamp("5000-01-01")):
            self.assertFalse(fast.begin_session(bad, end))
            self.assertEqual(fast._session_rows, {})
            self.assertIsNone(fast._session_start)
            self.assertTrue(fast.begin_session(start, end))
            fast.get_deal_price("AAA", start, end, self.Order.BUY)
        rows = fast._session_rows
        # Re-created timestamps should reuse the exact row, without calling the
        # native quote or confusing otherwise equal calendar intervals.
        with patch.object(fast.quote, "get_data", side_effect=AssertionError("fallback")):
            self.assertEqual(fast.get_deal_price("AAA", self.pd.Timestamp(str(start)),
                                                self.pd.Timestamp(str(end)), self.Order.BUY), 10.)
        self.assertTrue(fast.begin_session(self.dates[1], self.dates[1]))
        self.assertEqual(fast._session_rows, {})
        self.assertIsNot(fast._session_rows, rows)

    def test_unsupported_panel_types_preserve_native_field_and_error_precedence(self):
        start, end = self.dates[0], self.dates[0] + self.pd.Timedelta(hours=23)
        for dtype, price in ((object, "bad"), (self.np.complex128, 10 + 2j)):
            native, fast = self.exchange(False), self.exchange(True)
            for exchange in (native, fast):
                panel = exchange.quote.data["AAA"]
                panel.data = panel.data.astype(dtype)
                panel.data[0, panel.indices[1].index_map["$open"]] = price
            fast.begin_session(start, end)
            for name, tail in (("get_factor", ()), ("get_close", ()),
                               ("get_deal_price", (self.Order.BUY,)),
                               ("check_stock_suspended", ()),
                               ("check_stock_limit", (self.Order.BUY,)),
                               ("is_stock_tradable", (self.Order.BUY,))):
                self.assert_query(native, fast, name, "AAA", start, end, *tail)
            self.assertEqual(fast._session_rows, {})

    def test_same_day_missing_exact_start_never_selects_later_intraday_bar(self):
        frame = self.frame().reset_index()
        frame["datetime"] += self.pd.Timedelta(hours=12)
        frame = frame.set_index(["instrument", "datetime"])
        start = self.dates[0]
        same_end = start + self.pd.Timedelta(hours=23)
        weekend_end = start + self.pd.Timedelta(days=2, hours=23)
        for quote in (self.NumpyQuote, self.DailyQuote):
            native, fast = self.exchange(False, frame, quote), self.exchange(True, frame, quote)
            fast.begin_session(start, same_end)
            with patch.object(fast.quote, "get_data", wraps=fast.quote.get_data) as fallback:
                self.assert_query(native, fast, "get_deal_price", "AAA", start, same_end, self.Order.BUY)
                self.assertIsNone(fast.get_deal_price("AAA", start, same_end, self.Order.BUY))
                self.assert_query(native, fast, "get_factor", "AAA", start, same_end)
                self.assert_query(native, fast, "get_close", "AAA", start, same_end)
                self.assert_query(native, fast, "is_stock_tradable", "AAA", start, same_end)
                self.assertGreater(fallback.call_count, 0)
                self.assertEqual(fast._session_rows, {})
            # The distinct weekend aggregation includes the one later bar;
            # it can use direct lookup and must retain native scalar types.
            fast.begin_session(start, weekend_end)
            with patch.object(fast.quote, "get_data", side_effect=AssertionError("fallback")):
                self.assert_query(native, fast, "get_deal_price", "AAA", start, weekend_end, self.Order.BUY)
                self.assert_query(native, fast, "get_factor", "AAA", start, weekend_end)
                self.assert_query(native, fast, "get_close", "AAA", start, weekend_end)
            self.assertEqual(list(fast._session_rows), ["AAA"])

    def test_supported_strategy_preserves_native_sizing_after_begin_session(self):
        weights = self.pd.Series({"AAA": 0.5})
        native, _ = self.strategy(False, weights)
        strategy, _ = self.strategy(True, weights)
        self.assertTrue(strategy._can_fuse_targets(strategy.trade_exchange))
        self.assertEqual(self.outcome(native), self.outcome(strategy))
        self.assertEqual(strategy.trade_exchange._session_start, self.dates[1])
        self.assertEqual(list(strategy.trade_exchange._session_rows), ["AAA"])
        self.assertEqual(strategy.decisions[0]["target_quantities"], {"AAA": 5})

    def test_replaced_native_strategy_binding_clears_rows_before_fallback(self):
        strategy, _ = self.strategy(True, self.pd.Series({"AAA": 0.5}))
        exchange = strategy.trade_exchange
        start, end = self.dates[0], self.dates[0] + self.pd.Timedelta(hours=23)
        self.assertTrue(exchange.begin_session(start, end))
        exchange.get_deal_price("AAA", start, end, self.Order.BUY)
        self.assertEqual(list(exchange._session_rows), ["AAA"])
        sentinel = object()
        with patch.object(self.NativeStrategy, "generate_trade_decision", return_value=sentinel) as native:
            # Replacing the reviewed binding deliberately disables fusion.
            self.assertFalse(strategy._can_fuse_targets(exchange))
            self.assertIs(strategy.generate_trade_decision(), sentinel)
            native.assert_called_once_with(None)
        self.assertIsNone(exchange._session_start)
        self.assertIsNone(exchange._session_end)
        self.assertEqual(exchange._session_rows, {})

    def test_unreviewed_pin_uses_native_classes(self):
        from qlib_yahoo.fast_signed import SUPPORTED_COMMIT, classes_for

        self.assertEqual(classes_for(SUPPORTED_COMMIT), (self.FastExchange, self.FastStrategy))
        for pin in (None, "future-pin", SUPPORTED_COMMIT[:8]):
            self.assertEqual(classes_for(pin), (self.SignedExchange, self.NativeStrategy))


if __name__ == "__main__":
    unittest.main()
