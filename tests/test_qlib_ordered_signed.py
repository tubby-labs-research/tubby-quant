"""Actual pinned-native differentials for the unwired ordered-session bridge."""

import importlib.util
import json
import os
import struct
from collections import defaultdict
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

from qlib_yahoo.runtime import ROOT
from qlib_yahoo.dataset import DatasetError
from tests import test_qlib_fast_signed as quote_fixtures

CHILD = "QLIB_ORDERED_BRIDGE_CHILD"
CANDIDATE_CHILD = "QLIB_ORDERED_EXECUTOR_CHILD"
CANDIDATE_PIN = ROOT / "docs/qlib-derivative-pin.tubby4-candidate.json"


def _candidate_available():
    try:
        from qlib_yahoo.runtime import platform_pin
        pin = platform_pin(json.loads(CANDIDATE_PIN.read_text(encoding="utf-8")))
        return sys.version_info[:2] == (3, 12) and (ROOT / pin["local_runtime"] / "qlib/_tubby_build.json").is_file()
    except (OSError, ValueError, KeyError, DatasetError):
        return False


class LauncherTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("qlib") and quote_fixtures._signed_available(), "pinned signed derivative unavailable")
    @unittest.skipIf(os.environ.get(CHILD) == "1", "parent launcher")
    def test_actual_native_bridge_differentials(self):
        environment = os.environ.copy()
        environment[CHILD] = "1"
        program = (
            "from qlib_yahoo.runtime import ensure_signed_runtime; ensure_signed_runtime(); "
            "import unittest; suite=unittest.defaultTestLoader.loadTestsFromName("
            "'tests.test_qlib_ordered_signed.OrderedChild'); "
            "result=unittest.TextTestRunner(verbosity=2).run(suite); "
            "raise SystemExit(not result.wasSuccessful())"
        )
        result = subprocess.run([sys.executable, "-c", program], cwd=ROOT, env=environment,
                                capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(importlib.util.find_spec("qlib") and importlib.util.find_spec("numba") and _candidate_available(),
                         "candidate signed engine or optional compiled dependencies unavailable")
    @unittest.skipIf(os.environ.get(CANDIDATE_CHILD) == "1", "parent launcher")
    def test_actual_candidate_executor_differentials(self):
        environment = os.environ.copy()
        environment[CANDIDATE_CHILD] = "1"
        program = (
            "from pathlib import Path; from qlib_yahoo.runtime import ensure_signed_runtime; "
            "from qlib_yahoo.runtime import PIN; default_before=PIN.read_bytes(); "
            f"ensure_signed_runtime(pin_path=Path({str(CANDIDATE_PIN)!r})); "
            "import unittest; suite=unittest.defaultTestLoader.loadTestsFromName("
            "'tests.test_qlib_ordered_signed.CandidateExecutorChild'); "
            "result=unittest.TextTestRunner(verbosity=2).run(suite); "
            "assert PIN.read_bytes()==default_before; "
            "raise SystemExit(not result.wasSuccessful())"
        )
        result = subprocess.run([sys.executable, "-c", program], cwd=ROOT, env=environment,
                                capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(os.environ.get(CHILD) == "1", "requires verified signed child")
class OrderedChild(unittest.TestCase):
    frame = quote_fixtures.SignedChild.frame
    exchange = quote_fixtures.SignedChild.exchange

    @classmethod
    def setUpClass(cls):
        fixtures = quote_fixtures.SignedChild
        fixtures.setUpClass()
        for name in ("np", "pd", "Order", "NumpyQuote", "DailyQuote", "SignedExchange", "SignedPosition", "FastExchange", "dates"):
            setattr(cls, name, getattr(fixtures, name))
        from qlib.backtest.account import AccumulatedInfo
        from qlib.backtest.signed_account import SignedAccount
        from qlib_yahoo import ordered_signed as bridge
        cls.AccumulatedInfo, cls.SignedAccount, cls.bridge = AccumulatedInfo, SignedAccount, bridge
        cls.runtime_default_errstate = cls.np.geterr().copy()

    def modes(self):
        return (False, True) if importlib.util.find_spec("numba") else (False,)

    def account(self, cash=1000.0, holdings=None):
        # Exact native classes and methods; avoid provider/metrics initialization.
        account = self.SignedAccount.__new__(self.SignedAccount)
        account.current_position = self.SignedPosition(cash=float(cash))
        account.accum_info = self.AccumulatedInfo()
        account._active_session = self.dates[0]
        for symbol, amount in (holdings or {}).items():
            account.current_position._init_stock(symbol, float(amount), 8.125)
            account.current_position.position[symbol]["count_day"] = 7
            account.current_position.position[symbol]["research_metadata"] = "retained"
            if amount < 0:
                account.current_position.restricted_proceeds[symbol] = float(abs(amount) * 8.125)
        # Nonzero cumulative scalars test native float summation order.
        account.accum_info.cost = .125
        account.accum_info.to = 27.25
        account.accum_info.rtn = -.03125
        account.current_position.transaction_cost = .0625
        account.current_position.borrow_cost = .015625
        return account

    def orders(self, specifications, weekend=False):
        start = self.dates[0]
        end = start + self.pd.Timedelta(days=2, hours=23, minutes=59, seconds=59) if weekend else start
        result = []
        for symbol, amount, direction in specifications:
            order = self.Order(symbol, amount, direction, start, end)
            order.signed_skip_reason = "old_skip"
            order.signed_partial_reason = "old_partial"
            order.research_metadata = {"opaque": "preserve"}
            result.append(order)
        return result

    def freeze(self, value):
        """Dictionary insertion order, scalar types and binary64 bits are evidence."""
        if isinstance(value, dict):
            return ("dict", tuple((key, self.freeze(item)) for key, item in value.items()))
        if isinstance(value, (list, tuple)):
            return (type(value).__name__, tuple(self.freeze(item) for item in value))
        if isinstance(value, (float, self.np.floating)):
            return (type(value).__module__, type(value).__name__, struct.pack("!d", float(value)).hex())
        if isinstance(value, self.pd.Timestamp):
            return ("Timestamp", value.value, str(value.tz))
        return (type(value).__module__, type(value).__name__, value)

    def state(self, account, orders):
        return self.freeze((account.current_position.__dict__, account.accum_info.__dict__,
                            [order.__dict__ for order in orders]))

    def native(self, exchange, account, orders):
        result = []
        error = None
        try:
            for order in orders:
                result.append((order, *exchange.deal_order(order, trade_account=account)))
        except Exception as exception:
            error = (type(exception).__name__, str(exception))
        return [(order.stock_id, self.freeze(values)) for order, *values in result], error

    def assert_differential(self, specifications, *, cash=1000, holdings=None, frame=None, quote=None,
                            weekend=False, compiled=False, fallback=False):
        expected_account, actual_account = self.account(cash, holdings), self.account(cash, holdings)
        expected_orders, actual_orders = self.orders(specifications, weekend), self.orders(specifications, weekend)
        native_exchange = self.exchange(False, frame, quote)
        actual_exchange = self.exchange(True, frame, quote)
        expected_result, expected_error = self.native(native_exchange, expected_account, expected_orders)
        before = self.state(actual_account, actual_orders)
        actual_result = self.bridge.try_execute_orders(actual_exchange, actual_account, actual_orders, compiled=compiled)
        if actual_result is NotImplemented:
            self.assertTrue(fallback, "Unexpected fallback for supported fixture")
            self.assertEqual(self.state(actual_account, actual_orders), before)
            result, error = self.native(actual_exchange, actual_account, actual_orders)
        else:
            self.assertFalse(fallback, "Fixture must exercise fallback")
            self.assertEqual(len(actual_result), len(actual_orders))
            for index, row in enumerate(actual_result):
                self.assertIs(row[0], actual_orders[index])
            result = [(order.stock_id, self.freeze(values)) for order, *values in actual_result]
            error = None
        self.assertEqual(result, expected_result)
        self.assertEqual(error, expected_error)
        self.assertEqual(self.state(actual_account, actual_orders), self.state(expected_account, expected_orders))
        return actual_account, actual_orders

    def test_ordered_reversals_closes_reopens_metadata_and_float_bits(self):
        buy, sell = self.Order.BUY, self.Order.SELL
        specifications = [("AAA", 7, sell), ("BBB", 2, buy), ("AAA", 9, buy),
                          ("BBB", 4, sell), ("AAA", 6, sell), ("AAA", 2, buy), ("CCC", 1, sell)]
        for compiled in self.modes():
            for quote in (self.NumpyQuote, self.DailyQuote):
                for weekend in (False, True):
                    with self.subTest(compiled=compiled, quote=quote.__name__, weekend=weekend):
                        self.assert_differential(specifications, holdings={"CCC": 3, "AAA": 4, "BBB": -2},
                                                 quote=quote, weekend=weekend, compiled=compiled)

    def test_partial_zero_no_cash_and_tradability_reasons(self):
        cases = [([( "AAA", 100, self.Order.BUY), ("BBB", 0, self.Order.SELL)], 25, None),
                 ([("AAA", 1, self.Order.BUY), ("BBB", 1, self.Order.SELL)], 0, None),
                 ([("AAA", 1, self.Order.BUY), ("BBB", 2, self.Order.SELL)], 100,
                  self.frame({"AAA": self.np.nan, "BBB": -1.0})),
                 ([("AAA", 0, self.Order.BUY), ("AAA", 0, self.Order.SELL)], 100, None)]
        limited = self.frame()
        limited.loc[("AAA", self.dates[0]), "limit_buy"] = True
        cases.append(([("AAA", 2, self.Order.BUY), ("AAA", 2, self.Order.SELL)], 100, limited))
        for compiled in self.modes():
            for quote in (self.NumpyQuote, self.DailyQuote):
                for weekend in (False, True):
                    for specifications, cash, frame in cases:
                        with self.subTest(compiled=compiled, quote=quote.__name__, weekend=weekend, cash=cash, specifications=specifications):
                            self.assert_differential(specifications, cash=cash, frame=frame, quote=quote,
                                                     weekend=weekend, compiled=compiled)

    def test_decimal_and_tiny_notional_opens_preserve_native_float_bits(self):
        specifications = [("AAA", 2, self.Order.BUY), ("BBB", 3, self.Order.SELL),
                          ("AAA", 4, self.Order.SELL), ("CCC", 5, self.Order.BUY), ("BBB", 3, self.Order.BUY)]
        for compiled in self.modes():
            for quote in (self.NumpyQuote, self.DailyQuote):
                for opens in ({"AAA": 10.1234, "BBB": 20.33333333333333, "CCC": .123456789},
                              {"AAA": 1e-7, "BBB": 1e-8, "CCC": 1e-9}):
                    with self.subTest(compiled=compiled, quote=quote.__name__, opens=opens):
                        self.assert_differential(specifications, frame=self.frame(opens), quote=quote, compiled=compiled)

    def test_missing_quote_and_late_factor_failure_replay_native_prefix(self):
        for compiled in self.modes():
            for quote in (self.NumpyQuote, self.DailyQuote):
                for weekend in (False, True):
                    with self.subTest(compiled=compiled, quote=quote.__name__, weekend=weekend):
                        self.assert_differential([("AAA", 2, self.Order.BUY), ("MISSING", 1, self.Order.SELL)],
                            quote=quote, weekend=weekend, compiled=compiled, fallback=True)
                        self.assert_differential([("AAA", 2, self.Order.BUY), ("BBB", 1, self.Order.SELL)],
                            quote=quote, frame=self.frame(factors={"BBB": 2}), weekend=weekend,
                            compiled=compiled, fallback=True)

    def test_empty_session_preserves_state(self):
        for compiled in self.modes():
            account = self.account()
            exchange = self.exchange(True)
            before = self.state(account, [])
            self.assertEqual(self.bridge.try_execute_orders(exchange, account, [], compiled=compiled), [])
            self.assertEqual(self.state(account, []), before)

    def test_imported_runtime_numerical_policy_is_actually_eligible(self):
        account, exchange = self.account(), self.exchange(True)
        orders = self.orders([("AAA", 2, self.Order.BUY)])
        self.assertEqual(self.np.geterr(), self.runtime_default_errstate)
        self.assertIsNot(self.bridge.try_execute_orders(exchange, account, orders), NotImplemented)

    def test_guarded_bindings_classes_quotes_and_state_fallback_before_mutation(self):
        def guard_case(kind):
            exchange, account = self.exchange(True), self.account()
            orders = self.orders([("AAA", 1, self.Order.BUY)])
            callback = Mock()
            if kind == "account_callback":
                account.update_order = callback
            elif kind == "position_callback":
                account.current_position.preview_fill = callback
            elif kind == "accum_callback":
                account.accum_info.add_cost = callback
            elif kind == "exchange_callback":
                exchange.check_order = callback
            elif kind == "custom_get_data":
                exchange.quote.get_data = callback
            elif kind == "custom_account_class":
                class CustomAccount(self.SignedAccount):
                    pass
                account.__class__ = CustomAccount
            elif kind == "custom_exchange_class":
                class CustomExchange(self.FastExchange):
                    pass
                exchange.__class__ = CustomExchange
            elif kind == "custom_position_class":
                class CustomPosition(self.SignedPosition):
                    pass
                account.current_position.__class__ = CustomPosition
            elif kind == "custom_quote_class":
                class CustomQuote(self.NumpyQuote):
                    pass
                exchange.quote = CustomQuote(exchange.quote_df, "day", region="us")
            elif kind == "session_mismatch":
                account._active_session = self.dates[1]
            elif kind == "huge_quantity":
                orders[0].amount = self.bridge.kernel.MAX_QUANTITY + 2
            elif kind == "repeated_order":
                orders.append(orders[0])
            elif kind == "settlement":
                account.current_position._settle_type = account.current_position.ST_CASH
            elif kind == "huge_holding":
                account.current_position._init_stock("AAA", float(self.bridge.kernel.MAX_QUANTITY + 2), 10.0)
            elif kind == "fractional_holding":
                account.current_position._init_stock("AAA", 1.5, 10.0)
            return exchange, account, orders, callback
        for compiled in self.modes():
            for kind in ("account_callback", "position_callback", "accum_callback", "exchange_callback", "custom_get_data",
                         "custom_account_class", "custom_exchange_class", "custom_position_class", "custom_quote_class",
                         "session_mismatch", "huge_quantity", "repeated_order", "settlement", "huge_holding", "fractional_holding"):
                with self.subTest(compiled=compiled, kind=kind):
                    exchange, account, orders, callback = guard_case(kind)
                    before = self.state(account, orders)
                    self.assertIs(self.bridge.try_execute_orders(exchange, account, orders, compiled=compiled), NotImplemented)
                    self.assertEqual(self.state(account, orders), before)
                    callback.assert_not_called()
            for policy in ({"over": "raise"}, {"all": "raise"}, {"under": "warn"},
                           {"divide": "ignore"}, {"invalid": "raise"}, {"invalid": "call"}):
                with self.subTest(compiled=compiled, policy=policy):
                    exchange, account = self.exchange(True), self.account()
                    orders = self.orders([("AAA", 1, self.Order.BUY)])
                    before = self.state(account, orders)
                    with self.np.errstate(**policy):
                        self.assertIs(self.bridge.try_execute_orders(exchange, account, orders, compiled=compiled), NotImplemented)
                    self.assertEqual(self.state(account, orders), before)

    @unittest.skipUnless(importlib.util.find_spec("numba"), "optional compiled dependencies unavailable")
    def test_compiled_dependency_mode_is_actually_exercised(self):
        self.assert_differential([("AAA", 2, self.Order.BUY)], compiled=True)
        self.assertIsNotNone(self.bridge.kernel._compiled)
        self.assertTrue(self.bridge.kernel._compiled.signatures)


@unittest.skipUnless(os.environ.get(CANDIDATE_CHILD) == "1", "requires verified candidate child")
class CandidateExecutorChild(unittest.TestCase):
    setUpClass = classmethod(OrderedChild.setUpClass.__func__)
    frame, exchange = OrderedChild.frame, OrderedChild.exchange
    account, orders = OrderedChild.account, OrderedChild.orders
    freeze, state = OrderedChild.freeze, OrderedChild.state

    def setUp(self):
        if importlib.util.find_spec("numba") is None:
            self.skipTest("optional compiled dependencies unavailable")

    def fixture(self, specifications=None, *, weekend=False, handler=True, account_type=None):
        from qlib.backtest.executor import SimulatorExecutor
        from qlib.backtest.utils import TradeCalendarManager
        from qlib.backtest.decision import TradeDecisionWO
        calendar = TradeCalendarManager.__new__(TradeCalendarManager)
        start = self.dates[0]
        next_day = start + self.pd.Timedelta(days=3 if weekend else 1)
        calendar._calendar = self.np.array([start, next_day], dtype=object)
        calendar.start_index = calendar.trade_step = 0
        account = self.account()
        if account_type is not None:
            account.__class__ = account_type
        account._active_session = start
        executor = SimulatorExecutor.__new__(SimulatorExecutor)
        executor.trade_account = account
        executor._trade_exchange = self.exchange(True)
        executor.level_infra = {"trade_calendar": calendar}
        executor.trade_type = SimulatorExecutor.TT_SERIAL
        executor.time_per_step = "day"
        executor.verbose = False
        executor._settle_type = account.current_position.ST_NO
        executor.generate_portfolio_metrics = True
        executor.deal_day = None
        executor.dealt_order_amount = defaultdict(float, {"OLD": 9.0})
        executor._order_batch_executor = self.bridge.OrderedSessionExecutor(account_type or self.SignedAccount) if handler else None
        decision = TradeDecisionWO.__new__(TradeDecisionWO)
        decision.order_list = self.orders(specifications or [], weekend)
        actual_start, actual_end = calendar.get_step_time()
        for order in decision.order_list:
            order.start_time, order.end_time = actual_start, actual_end
        return executor, decision, calendar

    def economic_and_bookkeeping(self, executor, decision, calendar):
        return (self.state(executor.trade_account, decision.order_list), self.freeze(executor.deal_day),
                self.freeze(dict(executor.dealt_order_amount)), self.freeze(calendar._calendar.tolist()),
                calendar.start_index, calendar.trade_step)

    def test_compiled_native_executor_calendar_day_and_dealt_amount_parity(self):
        specifications = [("AAA", 2, self.Order.BUY), ("AAA", 3, self.Order.SELL), ("BBB", 1, self.Order.BUY)]
        for weekend in (False, True):
            for prior in (None, "2024-01-04", "2024-01-05"):
                with self.subTest(weekend=weekend, prior=prior):
                    native, native_decision, native_calendar = self.fixture(specifications, weekend=weekend, handler=False)
                    actual, actual_decision, actual_calendar = self.fixture(specifications, weekend=weekend)
                    for executor in (native, actual):
                        executor.deal_day = self.pd.Timestamp(prior) if prior else None
                        executor.dealt_order_amount["AAA"] = 7.0
                    native_amounts, actual_amounts = native.dealt_order_amount, actual.dealt_order_amount
                    expected, expected_kwargs = native._collect_data(native_decision)
                    result, kwargs = actual._collect_data(actual_decision)
                    self.assertIs(kwargs["trade_info"], result)
                    self.assertIs(expected_kwargs["trade_info"], expected)
                    self.assertEqual([(o.stock_id, self.freeze(values)) for o, *values in result],
                                     [(o.stock_id, self.freeze(values)) for o, *values in expected])
                    self.assertEqual(self.economic_and_bookkeeping(actual, actual_decision, actual_calendar),
                                     self.economic_and_bookkeeping(native, native_decision, native_calendar))
                    self.assertEqual(actual.dealt_order_amount is actual_amounts, native.dealt_order_amount is native_amounts)
                    self.assertEqual(actual._order_batch_executor.sessions, 1)
                    self.assertEqual(actual._order_batch_executor.orders, 3)
                    self.assertEqual(actual._order_batch_executor.fallbacks, 0)
                    self.assertTrue(self.bridge.kernel._compiled.signatures)

    def test_empty_session_preserves_native_day_and_existing_dealt_dictionary(self):
        executor, decision, calendar = self.fixture([])
        before = self.economic_and_bookkeeping(executor, decision, calendar)
        amounts = executor.dealt_order_amount
        self.assertEqual(executor._collect_data(decision), ([], {"trade_info": []}))
        self.assertEqual(self.economic_and_bookkeeping(executor, decision, calendar), before)
        self.assertIs(executor.dealt_order_amount, amounts)
        self.assertEqual(executor._order_batch_executor.sessions, 1)
        self.assertEqual(executor._order_batch_executor.orders, 0)

    def test_late_factor_fallback_replays_native_prefix_and_bookkeeping(self):
        specifications = [("AAA", 2, self.Order.BUY), ("BBB", 1, self.Order.SELL)]
        native, native_decision, native_calendar = self.fixture(specifications, handler=False)
        actual, actual_decision, actual_calendar = self.fixture(specifications)
        for executor in (native, actual):
            executor._trade_exchange = self.exchange(True, self.frame(factors={"BBB": 2}))
        errors = []
        for executor, decision in ((native, native_decision), (actual, actual_decision)):
            with self.assertRaises(NotImplementedError) as caught:
                executor._collect_data(decision)
            errors.append((type(caught.exception), str(caught.exception)))
        self.assertEqual(errors[0], errors[1])
        self.assertEqual(self.economic_and_bookkeeping(actual, actual_decision, actual_calendar),
                         self.economic_and_bookkeeping(native, native_decision, native_calendar))
        self.assertEqual(actual.dealt_order_amount["AAA"], 2.0)
        self.assertNotIn("BBB", actual.dealt_order_amount)
        self.assertEqual(actual._order_batch_executor.fallbacks, 1)
        self.assertEqual(actual._order_batch_executor.sessions, 0)

    def test_candidate_activation_retains_source_only_loader_and_default_pin(self):
        import qlib
        from qlib_yahoo import runtime
        default_bytes = runtime.PIN.read_bytes()
        self.assertIsInstance(qlib.__loader__, runtime._SourceOnlyLoader)
        declared = json.loads(CANDIDATE_PIN.read_text(encoding="utf-8"))
        self.assertEqual(qlib.__version__, declared["version"])
        self.assertEqual(runtime.PIN.name, "qlib-derivative-pin.json")
        self.assertNotEqual(runtime.PIN, CANDIDATE_PIN)
        self.assertEqual(runtime.PIN.read_bytes(), default_bytes)

    def test_custom_and_invalid_decision_executor_calendar_decline_before_callbacks(self):
        for kind in ("iterator_callback", "decision_callback", "calendar_callback", "custom_executor", "custom_calendar",
                     "custom_decision", "custom_order", "fractional_order", "parallel", "verbose", "intraday", "no_metrics",
                     "settlement", "nonlist", "calendar_list", "bad_start_index", "bad_dealt_factory", "bad_dealt_value", "timezone_day", "epsilon_callback"):
            with self.subTest(kind=kind):
                executor, decision, calendar = self.fixture([("AAA", 2, self.Order.BUY)])
                handler = executor._order_batch_executor
                callback = Mock()
                if kind == "iterator_callback": executor._get_order_iterator = callback
                elif kind == "decision_callback": decision.get_decision = callback
                elif kind == "calendar_callback": calendar.get_step_time = callback
                elif kind == "custom_executor": executor.__class__ = type("CustomExecutor", (type(executor),), {})
                elif kind == "custom_calendar": calendar.__class__ = type("CustomCalendar", (type(calendar),), {})
                elif kind == "custom_decision": decision.__class__ = type("CustomDecision", (type(decision),), {})
                elif kind == "custom_order": decision.order_list[0].__class__ = type("CustomOrder", (self.Order,), {})
                elif kind == "fractional_order": decision.order_list[0].amount = .5
                elif kind == "parallel": executor.trade_type = type(executor).TT_PARAL
                elif kind == "verbose": executor.verbose = True
                elif kind == "intraday": executor.time_per_step = "1min"
                elif kind == "no_metrics": executor.generate_portfolio_metrics = False
                elif kind == "settlement": executor._settle_type = self.SignedPosition.ST_CASH
                elif kind == "nonlist": decision.order_list = tuple(decision.order_list)
                elif kind == "bad_start_index": calendar.start_index = self.np.int64(0)
                elif kind == "bad_dealt_factory": executor.dealt_order_amount = defaultdict(int)
                elif kind == "bad_dealt_value":
                    executor.deal_day = self.dates[0]
                    executor.dealt_order_amount["AAA"] = 7
                elif kind == "timezone_day": executor.deal_day = self.pd.Timestamp("2024-01-05", tz="UTC")
                before = self.economic_and_bookkeeping(executor, decision, calendar)
                if kind == "calendar_list":
                    calendar._calendar = calendar._calendar.tolist()
                    self.assertIs(handler.try_collect(executor, decision), NotImplemented)
                    calendar._calendar = self.np.array(calendar._calendar, dtype=object)
                elif kind == "epsilon_callback":
                    with patch.dict(type(calendar).get_step_time.__globals__, {"epsilon_change": callback}):
                        self.assertIs(handler.try_collect(executor, decision), NotImplemented)
                else:
                    self.assertIs(handler.try_collect(executor, decision), NotImplemented)
                self.assertEqual(self.economic_and_bookkeeping(executor, decision, calendar), before)
                callback.assert_not_called()
                self.assertEqual(handler.fallbacks, 1)
                self.assertEqual(handler.sessions, 0)

    def test_touched_aliased_inner_dictionaries_decline_before_mutation(self):
        executor, decision, calendar = self.fixture([("AAA", 2, self.Order.BUY), ("BBB", 1, self.Order.SELL)])
        position = executor.trade_account.current_position
        position._init_stock("AAA", 3.0, 10.0)
        position.position["BBB"] = position.position["AAA"]
        before = self.economic_and_bookkeeping(executor, decision, calendar)
        with patch.object(self.bridge.kernel, "run_ordered_session", wraps=self.bridge.kernel.run_ordered_session) as kernel_call:
            self.assertIs(executor._order_batch_executor.try_collect(executor, decision), NotImplemented)
        kernel_call.assert_not_called()
        self.assertIs(position.position["AAA"], position.position["BBB"])
        self.assertEqual(self.economic_and_bookkeeping(executor, decision, calendar), before)

    def test_exact_explicit_streaming_account_type_and_custom_fill_binding_guard(self):
        from qlib_yahoo.streaming import streaming_account_class
        reviewed_type = streaming_account_class(self.SignedAccount)
        executor, decision, _calendar = self.fixture([("AAA", 2, self.Order.BUY)], account_type=reviewed_type)
        result = executor._order_batch_executor.try_collect(executor, decision)
        self.assertIsNot(result, NotImplemented)
        self.assertEqual(result[0][0][0].deal_amount, 2.0)
        for supplied_type in (self.SignedAccount, streaming_account_class(self.SignedAccount)):
            executor, decision, calendar = self.fixture([("AAA", 2, self.Order.BUY)], account_type=reviewed_type)
            before = self.economic_and_bookkeeping(executor, decision, calendar)
            self.assertIs(self.bridge.OrderedSessionExecutor(supplied_type).try_collect(executor, decision), NotImplemented)
            self.assertEqual(self.economic_and_bookkeeping(executor, decision, calendar), before)
        executor, decision, calendar = self.fixture([("AAA", 2, self.Order.BUY)], account_type=reviewed_type)
        callback = Mock()
        executor.trade_account.update_order = callback
        before = self.economic_and_bookkeeping(executor, decision, calendar)
        self.assertIs(executor._order_batch_executor.try_collect(executor, decision), NotImplemented)
        callback.assert_not_called()
        self.assertEqual(self.economic_and_bookkeeping(executor, decision, calendar), before)


if __name__ == "__main__":
    unittest.main()
