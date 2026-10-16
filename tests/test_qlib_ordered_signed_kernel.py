"""Complete ordered-session differentials against the verified native derivative."""
import importlib.util
import math
import os
import random
import subprocess
import sys
import unittest

from qlib_yahoo import ordered_signed_kernel as kernel
from tests.test_qlib_fast_signed import _signed_available
from qlib_yahoo.runtime import ROOT

CHILD = 'QLIB_ORDERED_SIGNED_CHILD'


class LauncherTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('qlib') and _signed_available(),
                         'pinned signed derivative unavailable')
    @unittest.skipIf(os.environ.get(CHILD) == '1', 'parent launcher')
    def test_actual_native_differentials(self):
        environment = os.environ.copy()
        environment[CHILD] = '1'
        program = (
            'from qlib_yahoo.runtime import ensure_signed_runtime; ensure_signed_runtime(); '
            'import unittest; '
            "suite=unittest.defaultTestLoader.loadTestsFromName('tests.test_qlib_ordered_signed_kernel.SignedChild'); "
            'result=unittest.TextTestRunner(verbosity=2).run(suite); '
            'raise SystemExit(not result.wasSuccessful())'
        )
        result = subprocess.run([sys.executable, '-c', program], cwd=ROOT,
                                env=environment, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_ulp_matches_python(self):
        for value in (0.0, 5e-324, 1e-310, 2.2250738585072014e-308,
                      1e-100, 1.0, -100.0, sys.float_info.max):
            self.assertEqual(kernel._ulp(value), math.ulp(value))


@unittest.skipUnless(os.environ.get(CHILD) == '1', 'requires verified signed child')
class SignedChild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import pandas as pd
        from qlib.backtest.account import AccumulatedInfo
        from qlib.backtest.decision import Order
        from qlib.backtest.signed_account import SignedAccount
        from qlib.backtest.signed_position import SignedPosition
        from tests.test_qlib_fast_signed import SignedChild as NativeFixtures
        NativeFixtures.setUpClass()
        cls.fixtures = NativeFixtures()
        cls.np, cls.pd, cls.Order = np, pd, Order
        cls.Account, cls.Position, cls.Accum = SignedAccount, SignedPosition, AccumulatedInfo

    def assert_float_bits(self, actual, expected):
        np = self.np
        actual = np.asarray(actual, dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        mask = ~np.isnan(expected)
        np.testing.assert_array_equal(actual[mask].view(np.uint64), expected[mask].view(np.uint64))

    def compare(self, orders, *, cash=100.0, holdings=None, compiled=False, rate=.0005):
        np = self.np
        symbols = list(dict.fromkeys([*(holdings or {}), *(item[0] for item in orders)]))
        position = self.Position(cash=max(cash, 0.0))
        position.position['cash'] = cash
        for code, (amount, mark, basis) in (holdings or {}).items():
            position._init_stock(code, amount, mark)
            if amount < 0:
                position.restricted_proceeds[code] = basis
        position.transaction_cost = 0.125
        account = self.Account.__new__(self.Account)
        account.current_position = position
        account._active_session = self.fixtures.dates[0]
        account.accum_info = self.Accum()
        account.accum_info.cost = 0.25
        account.accum_info.to = 0.5
        account.accum_info.rtn = -0.75
        arrays = [np.array([position.get_stock_amount(c) for c in symbols], dtype=np.float64),
                  np.array([position.get_stock_price(c) if position.check_stock(c) else 0.0 for c in symbols], dtype=np.float64),
                  np.array([position.restricted_proceeds.get(c, 0.0) for c in symbols], dtype=np.float64),
                  np.array([position.check_stock(c) for c in symbols], dtype=np.bool_),
                  np.array([symbols.index(o[0]) for o in orders], dtype=np.int64),
                  np.array([o[1] for o in orders], dtype=np.float64),
                  np.array([o[2] for o in orders], dtype=np.int64),
                  np.array([o[3] for o in orders], dtype=np.float64),
                  np.ones(len(orders), dtype=np.float64),
                  np.array([math.isfinite(o[3]) and o[3] > 0 for o in orders], dtype=np.bool_)]
        originals = [a.copy() for a in arrays]
        for a in arrays:
            a.setflags(write=False)
        result = kernel.run_ordered_session(*arrays, cash=cash, rate=rate,
                    transaction_cost=.125, accum_cost=.25, accum_turnover=.5,
                    accum_return=-.75, compiled=compiled)
        self.assertIsNotNone(result)
        for a, original in zip(arrays, originals):
            np.testing.assert_array_equal(a, original)
        expected = []
        for code, ask, direction, price in orders:
            frame = self.fixtures.frame({c: price if c == code else 10.0 for c in symbols})
            exchange = self.fixtures.exchange(False, frame)
            exchange.open_cost = exchange.close_cost = rate
            start = self.fixtures.dates[0]
            order = self.Order(stock_id=code, amount=ask, start_time=start,
                               end_time=start, direction=direction)
            value, cost, trade_price = exchange.deal_order(order, trade_account=account)
            skip = getattr(order, 'signed_skip_reason', None)
            partial = getattr(order, 'signed_partial_reason', None)
            statuses = {'zero_amount': kernel.ZERO, 'insufficient_free_cash': kernel.NO_CASH,
                        'missing_or_invalid_open': kernel.INVALID_OPEN,
                        'suspended_or_limited': kernel.NOT_TRADABLE}
            status = statuses[skip] if skip else kernel.PARTIAL if partial else kernel.FILLED
            expected.append([order.deal_amount, value, cost, trade_price, position.get_cash(),
                position.restricted_proceeds.get(code, 0.0), position.get_stock_amount(code),
                float(position.check_stock(code)), account.accum_info.cost,
                account.accum_info.to, account.accum_info.rtn, position.transaction_cost,
                float(status), float(arrays[9][len(expected)])])
        self.assert_float_bits(result.records, np.array(expected, dtype=np.float64))
        self.assert_float_bits(result.amounts, [position.get_stock_amount(c) for c in symbols])
        self.assert_float_bits(result.bases, [position.restricted_proceeds.get(c, 0) for c in symbols])
        self.assert_float_bits(result.marks, [position.get_stock_price(c) if position.check_stock(c) else 0.0 for c in symbols])
        np.testing.assert_array_equal(result.exists, [position.check_stock(c) for c in symbols])
        return arrays, result

    def test_ordered_duplicate_reversals_and_deletions(self):
        self.compare([('A', 6, 0, 10.), ('A', 2, 1, 7.), ('B', 3, 1, 11.),
                      ('A', 9, 1, 8.), ('B', 3, 0, 12.), ('B', 2, 0, 15.),
                      ('B', 2, 1, 14.), ('A', 0, 1, 8.), ('A', 100, 1, math.nan)],
                     compiled=getattr(self, '_force_compiled', False))

    def test_partial_cover_and_two_interval_affordability(self):
        self.compare([('A', 3, 1, 12.), ('A', 8, 1, 8.), ('A', 100, 0, 10.),
                      ('A', 100, 1, 7.)], cash=10., holdings={'A': (-10., 9., 100.)},
                     compiled=getattr(self, '_force_compiled', False))
        self.compare([('A', 50, 0, 10.), ('A', 10, 1, 8.)],
                     cash=-2., holdings={'A': (4., 11., 0.)},
                     compiled=getattr(self, '_force_compiled', False))

    def test_tiny_prices_and_nonzero_accumulators(self):
        self.compare([('A', 3, 1, 1e-10), ('A', 5, 0, 2e-10),
                      ('A', 1, 1, 1e-10), ('B', 7, 1, 1e-100)],
                     compiled=getattr(self, '_force_compiled', False))

    def test_cancellation_tolerance_and_exact_cash_boundary(self):
        self.compare([('A', 3, 1, .1), ('A', 3, 0, .1), ('A', 3, 1, .1)],
                     cash=.3, rate=0., compiled=getattr(self, '_force_compiled', False))
        self.compare([('A', 1, 1, 5e-324), ('A', 1, 0, 5e-324)], cash=1., rate=0.,
                     compiled=getattr(self, '_force_compiled', False))

    def test_fallback_discards_all_scratch(self):
        compiled = getattr(self, '_force_compiled', False)
        arrays, _ = self.compare([('A', 1, 1, 10.), ('A', 1, 0, 10.)], compiled=compiled)
        originals = [a.copy() for a in arrays]
        for changes in ({5: [1., float(2**53)]}, {8: [1., 2.]},
                        {7: [10., 1e308], 5: [1., 2.]}):
            trial = [a.copy() for a in arrays]
            for index, values in changes.items():
                trial[index][:] = values
            unchanged = [a.copy() for a in trial]
            self.assertIsNone(kernel.run_ordered_session(*trial, cash=100., rate=1., compiled=compiled))
            for a, original in zip(trial, unchanged):
                self.np.testing.assert_array_equal(a, original)
            for a, original in zip(arrays, originals):
                self.np.testing.assert_array_equal(a, original)

    @unittest.skipUnless(importlib.util.find_spec('numba'), 'optional Numba unavailable')
    def test_compiled_complete_session(self):
        self._force_compiled = True
        try:
            self.test_ordered_duplicate_reversals_and_deletions()
            self.test_partial_cover_and_two_interval_affordability()
            self.test_tiny_prices_and_nonzero_accumulators()
            self.test_cancellation_tolerance_and_exact_cash_boundary()
            self.test_fallback_discards_all_scratch()
        finally:
            self._force_compiled = False

    def test_randomized_native_sequences(self):
        for compiled in (False, True) if importlib.util.find_spec('numba') else (False,):
            for seed in range(20):
                rng = random.Random(seed)
                orders = [(rng.choice(('A', 'B', 'C', 'D')), rng.randrange(0, 75),
                           rng.randrange(2), rng.choice((.01, .1, 1., 7.25, 10., 100., 1e-10)))
                          for _ in range(100)]
                with self.subTest(seed=seed, compiled=compiled):
                    self.compare(orders, cash=rng.choice((0., 1., 25., 10000.)),
                                 rate=rng.choice((0., .0005, .001, .05)),
                                 holdings={'A': (-7., 11., 77.), 'B': (3., .1, 0.)},
                                 compiled=compiled)
