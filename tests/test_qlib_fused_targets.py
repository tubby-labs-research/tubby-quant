"""Complete native decision-boundary differentials for fused target queries."""

import importlib.util
import os
import subprocess
import sys
import unittest
from functools import lru_cache
from unittest.mock import patch

from tests import test_qlib_fast_signed as fixtures
from qlib_yahoo.runtime import ROOT


class LauncherTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("qlib") and fixtures._signed_available(),
                         "pinned signed derivative unavailable")
    @unittest.skipIf(os.environ.get(fixtures.CHILD) == "1", "parent launcher")
    def test_actual_pinned_fused_targets(self):
        program = (
            "from qlib_yahoo.runtime import ensure_signed_runtime; ensure_signed_runtime(); "
            "import unittest; suite=unittest.defaultTestLoader.loadTestsFromName("
            "'tests.test_qlib_fused_targets.SignedChild'); "
            "result=unittest.TextTestRunner(verbosity=2).run(suite); "
            "raise SystemExit(not result.wasSuccessful())"
        )
        environment = os.environ.copy()
        environment[fixtures.CHILD] = "1"
        result = subprocess.run([sys.executable, "-c", program], cwd=ROOT,
                                env=environment, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(os.environ.get(fixtures.CHILD) == "1", "requires verified signed child")
class SignedChild(unittest.TestCase):
    # Reuse only fixture helpers, without inheriting/running the original suite.
    setUpClass = classmethod(fixtures.SignedChild.setUpClass.__func__)
    frame = fixtures.SignedChild.frame
    exchange = fixtures.SignedChild.exchange
    strategy = fixtures.SignedChild.strategy
    orders = staticmethod(fixtures.SignedChild.orders)
    outcome = fixtures.SignedChild.outcome
    assert_strategies = fixtures.SignedChild.assert_strategies

    def test_one_row_per_target_vs_three_native_queries(self):
        for quote_class in (self.NumpyQuote, self.DailyQuote):
            self.check_row_count(quote_class)

    def check_row_count(self, quote_class):
        code = self.FastExchange._session_row.__code__
        counts = []
        outcomes = []
        for cls in (self.NativeStrategy, self.FastStrategy):
            exchange = self.exchange(True, quote_class=quote_class)
            strategy, _ = self.strategy(True, self.pd.Series({"AAA": .5, "BBB": -.5}), exchange=exchange)
            strategy.__class__ = cls
            exchange = strategy.trade_exchange
            exchange.begin_session(self.dates[1], self.dates[1] + self.pd.Timedelta(hours=23))
            calls = []

            def count(frame, event, arg):
                if event == "call" and frame.f_code is code:
                    calls.append(frame.f_locals["stock_id"])

            previous = sys.getprofile()
            try:
                sys.setprofile(count)
                outcomes.append(self.outcome(strategy))
            finally:
                sys.setprofile(previous)
            counts.append(calls)
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(counts, [["AAA"] * 3 + ["BBB"] * 3, ["AAA", "BBB"]])

    def test_normal_tiny_extreme_weights_and_prices(self):
        np, pd = self.np, self.pd
        for weights in ({"BBB": -.5, "AAA": .5}, {"AAA": -0.},
                        {"AAA": 1e-300, "BBB": -.5}, {}, {"AAA": 1 + 1e-13}):
            for cash, price in ((100., 3.), (float(2**53), .1), (1e100, 1e-300)):
                self.assert_strategies(pd.Series(weights, dtype=float), cash=cash,
                                       frame=self.frame({"AAA": price, "BBB": price}))
        with np.errstate(all="raise"):
            self.assert_strategies(pd.Series({"AAA": 1e-300, "BBB": .5}),
                                   frame=self.frame({"AAA": 1e300, "BBB": 20.}, {"BBB": 2.}))

    def test_gaps_invalid_factors_and_partial_records(self):
        for price in (self.np.nan, 0., -1.):
            self.assert_strategies(self.pd.Series({"AAA": .5, "BBB": -.5}),
                                   frame=self.frame({"AAA": price, "BBB": 20.}, {"AAA": 2.}))
        for factor in (0., 2., self.np.nan, self.np.inf):
            self.assert_strategies(self.pd.Series({"AAA": .5, "BBB": -.5}),
                                   frame=self.frame(factors={"BBB": factor}))
        self.assert_strategies(self.pd.Series({"AAA": .2, "MISSING": -.2}))
        self.assert_strategies(self.pd.Series({"AAA": .5, "BBB": -.5}), cash=1e100,
                               frame=self.frame({"AAA": 1e-300, "BBB": 20.}, {"BBB": 2.}))

    def test_empty_missing_signal_nonpositive_and_invalid_inputs(self):
        for equity in (0., -1., self.np.inf, self.np.nan):
            results = []
            for fast in (False, True):
                strategy, position = self.strategy(fast, self.pd.Series({"AAA": .5}))
                with patch.object(position, "calculate_value", return_value=equity):
                    results.append(self.outcome(strategy))
            self.assertEqual(*results)
        self.assert_strategies(None)
        self.assert_strategies(self.pd.Series({"AAA": .5}), step=0)
        for weights in (self.pd.Series({"AAA": self.np.nan}), self.pd.Series({"AAA": True}),
                        self.pd.Series({"AAA": 1.1}), self.pd.Series({12: .1}),
                        self.pd.Series([.1, .2], index=["AAA", "AAA"])):
            self.assert_strategies(weights)

    def test_reversal_orders_and_actual_native_fills(self):
        fixtures.SignedChild.test_reversal_reduction_order_and_native_fills(self)

    def test_instance_callbacks_and_subclasses_use_native_boundary(self):
        for name, replacement in (("get_deal_price", lambda *args: 7.),
                                  ("get_factor", lambda *args: 2.),
                                  ("is_stock_tradable", lambda *args: False)):
            outcomes = []
            for cls in (self.NativeStrategy, self.FastStrategy):
                strategy, _ = self.strategy(True, self.pd.Series({"AAA": .5}))
                strategy.__class__ = cls
                with patch.object(strategy.trade_exchange, name, side_effect=replacement) as callback:
                    outcomes.append(self.outcome(strategy))
                    self.assertGreater(callback.call_count, 0)
            self.assertEqual(*outcomes)
        strategy, _ = self.strategy(True, self.pd.Series({"AAA": .5}))
        strategy.__class__ = type("CustomStrategy", (self.FastStrategy,), {})
        with patch.object(self.NativeStrategy, "generate_trade_decision", return_value="native") as native:
            self.assertEqual(strategy.generate_trade_decision(), "native")
            native.assert_called_once()

    def test_changed_quote_and_settings_use_native_boundary(self):
        for mode in ("quote", "price", "quote_method"):
            results = []
            for cls in (self.NativeStrategy, self.FastStrategy):
                strategy, _ = self.strategy(True, self.pd.Series({"AAA": .5}))
                strategy.__class__ = cls
                exchange = strategy.trade_exchange
                if mode == "quote":
                    exchange.quote = self.PandasQuote(self.frame(), "day")
                    exchange.reset_session_cache()
                elif mode == "price":
                    exchange.buy_price = "$close"
                else:
                    exchange.quote.get_data = lambda *args, **kwargs: 2.
                results.append(self.outcome(strategy))
            self.assertEqual(*results)

    def test_signal_callback_change_does_not_reuse_an_earlier_row(self):
        for callback in ("signal", "equity"):
            for quote_class in (self.NumpyQuote, self.DailyQuote):
                self.check_mutating_callback(callback, quote_class)

    def check_mutating_callback(self, callback, quote_class):
        results = []
        for fast in (False, True):
            exchange = self.exchange(fast, quote_class=quote_class)
            strategy, position = self.strategy(fast, self.pd.Series({"AAA": .5}), exchange=exchange)
            start, end = strategy.trade_calendar.get_step_time(1)
            if fast:
                exchange.begin_session(start, end)
                exchange.get_factor("AAA", start, end)
            obj, name = ((strategy.signal, "get_signal") if callback == "signal"
                         else (position, "calculate_value"))
            original = getattr(obj, name)

            def changed(*args, **kwargs):
                exchange.quote.get_data = lambda *args, **kwargs: 2.
                return original(*args, **kwargs)

            setattr(obj, name, changed)
            results.append(self.outcome(strategy))
            self.assertEqual(len(strategy.decisions), 1)
        self.assertEqual(*results)

    def test_daily_quote_complete_boundary_native_parity(self):
        for weights, frame, cash in (
                ({"AAA": .5, "BBB": -.5}, self.frame(), 100.),
                ({"AAA": 1e-300, "BBB": -.5}, self.frame(), 100.),
                ({"AAA": .5, "BBB": -.5}, self.frame({"AAA": 1e-300, "BBB": 20.}), 1e100),
                ({"AAA": .5, "BBB": -.5}, self.frame(factors={"BBB": 2.}), 100.),
                ({"AAA": .5, "BBB": -.5}, self.frame({"AAA": self.np.nan, "BBB": 20.}), 100.),
                ({"AAA": .5}, self.frame(), 0.),
                ({"AAA": -.4, "BBB": .2, "CCC": 0.}, self.frame(), 100.)):
            results = []
            for fast in (False, True):
                exchange = self.exchange(fast, frame, self.DailyQuote)
                strategy, _ = self.strategy(fast, self.pd.Series(weights), cash=cash,
                                           frame=frame, holdings={"AAA": 5, "BBB": -4}, exchange=exchange)
                results.append(self.outcome(strategy))
            self.assertEqual(*results)

    def test_daily_quote_reversal_native_fills_and_final_book(self):
        frame = self.frame({"AAA": 10., "BBB": 20., "CCC": 30.})
        weights = self.pd.Series({"BBB": .2, "AAA": -.4, "CCC": 0.})
        strategies = []
        for fast in (False, True):
            exchange = self.exchange(fast, frame, self.DailyQuote)
            strategies.append(self.strategy(
                fast, weights, frame=frame, holdings={"AAA": 5, "BBB": -4, "CCC": 2},
                exchange=exchange))
        self.assertEqual(self.outcome(strategies[0][0]), self.outcome(strategies[1][0]))
        fills = []
        for strategy, position in strategies:
            orders = strategy.decisions[0]["issued_orders"]
            self.assertEqual([order.stock_id for order in orders[:3]], ["AAA", "CCC", "BBB"])
            fills.append([(strategy.trade_exchange.deal_order(order, position=position),
                           order.deal_amount, order.signed_skip_reason, order.signed_partial_reason)
                          for order in orders])
        self.assertEqual(*fills)
        self.assertEqual(strategies[0][1].position, strategies[1][1].position)
        self.assertEqual(strategies[0][1].restricted_proceeds, strategies[1][1].restricted_proceeds)

    def test_daily_quote_spoofed_cache_metadata_uses_native_callback(self):
        from qlib_yahoo.fast_signed import _reviewed_quote_binding

        results = []
        for fast in (False, True):
            exchange = self.exchange(fast, quote_class=self.DailyQuote)
            native_bound = exchange.quote.get_data.__wrapped__
            callback = lru_cache(maxsize=512)(lambda *args, **kwargs: 2.)
            callback.__wrapped__ = native_bound
            exchange.quote.get_data = callback
            self.assertFalse(_reviewed_quote_binding(exchange.quote))
            strategy, _ = self.strategy(fast, self.pd.Series({"AAA": .5}), exchange=exchange)
            results.append(self.outcome(strategy))
        self.assertEqual(*results)

    def test_weights_instance_class_and_subclass_bindings(self):
        for mode in ("instance", "class", "subclass"):
            results = []
            for fast in (False, True):
                strategy, _ = self.strategy(fast, self.pd.Series({"AAA": .5}))
                replacement = lambda values: {"BBB": -.25}
                if mode == "instance":
                    strategy._weights = replacement
                    results.append(self.outcome(strategy))
                elif mode == "subclass":
                    strategy.__class__ = type("CustomWeights", (type(strategy),),
                                              {"_weights": staticmethod(replacement)})
                    results.append(self.outcome(strategy))
                else:
                    with patch.object(self.NativeStrategy, "_weights", staticmethod(replacement)):
                        results.append(self.outcome(strategy))
            self.assertEqual(*results)

    def test_stateful_calendar_calls_and_errors_match_native_boundary(self):
        for fail in (False, True):
            results, histories = [], []
            for fast in (False, True):
                strategy, _ = self.strategy(fast, self.pd.Series({"AAA": .5}))
                calendar = strategy.trade_calendar
                original = calendar.get_step_time
                calls = []

                def step_time(index=None, shift=0):
                    calls.append((index, shift))
                    if fail and len(calls) == 2:
                        raise ValueError("stateful previous-session failure")
                    start, end = original(index, shift)
                    # Observable state changes the returned time on each call.
                    offset = self.pd.Timedelta(days=len(calls) - 1)
                    return start + offset, end + offset

                calendar.get_step_time = step_time
                results.append(self.outcome(strategy))
                histories.append(calls)
            self.assertEqual(*results)
            self.assertEqual(*histories)
            self.assertEqual(histories[0][:2], [(1, 0), (1, 1)])
            self.assertEqual(len(histories[0]), 2 if fail else 4)

    def test_weekend_multibar_and_missing_fields_per_symbol_fallback(self):
        for frame in (self.frame(), self.frame().drop(columns=["$factor"]),
                      self.pd.concat([self.frame(), self.frame().iloc[:1]]).sort_index()):
            for end in (self.dates[0] + self.pd.Timedelta(days=2, hours=23), self.dates[2]):
                results = []
                for fast in (False, True):
                    strategy, _ = self.strategy(fast, self.pd.Series({"AAA": .5, "BBB": -.5}), frame=frame)
                    calendar = strategy.trade_calendar
                    calendar.get_step_time = lambda index=None, shift=0: (
                        (self.dates[0] - self.pd.Timedelta(days=1), self.dates[0] - self.pd.Timedelta(hours=1))
                        if shift else (self.dates[0], end))
                    results.append(self.outcome(strategy))
                self.assertEqual(*results)


if __name__ == "__main__":
    unittest.main()
