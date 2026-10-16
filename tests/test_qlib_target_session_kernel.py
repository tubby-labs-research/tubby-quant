"""Target/order phases compared with actual pinned SignedWeightStrategy."""
import importlib.util
import math
import os
import random
import subprocess
import sys
import unittest

from qlib_yahoo import target_session_kernel as kernel
from qlib_yahoo.runtime import ROOT
from tests import test_qlib_fast_signed as fixtures

CHILD = 'QLIB_TARGET_SESSION_CHILD'


class LauncherTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('qlib') and fixtures._signed_available(),
                         'pinned signed derivative unavailable')
    @unittest.skipIf(os.environ.get(CHILD) == '1', 'parent launcher')
    def test_actual_native_targets(self):
        environment = os.environ.copy()
        environment[CHILD] = '1'
        program = (
            'from qlib_yahoo.runtime import ensure_signed_runtime; ensure_signed_runtime(); '
            'import unittest; '
            "suite=unittest.defaultTestLoader.loadTestsFromName('tests.test_qlib_target_session_kernel.SignedChild'); "
            'result=unittest.TextTestRunner(verbosity=2).run(suite); '
            'raise SystemExit(not result.wasSuccessful())'
        )
        result = subprocess.run([sys.executable, '-c', program], cwd=ROOT,
                                env=environment, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(os.environ.get(CHILD) == '1', 'requires verified signed child')
class SignedChild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.SignedChild.setUpClass()
        cls.native = fixtures.SignedChild()
        cls.np, cls.pd = cls.native.np, cls.native.pd

    def compare(self, weights, prices=None, holdings=None, *, compiled=False, cash=100.):
        np = self.np
        prices = prices or {'A': 10., 'B': 20., 'C': 7.}
        frame = self.native.frame(prices)
        strategy, position = self.native.strategy(False, self.pd.Series(weights, dtype=float),
                       cash=cash, frame=frame, holdings=holdings or {})
        decision = strategy.generate_trade_decision()
        record = strategy.decisions[0]
        symbols = list(record['target_weights'])
        start, end = strategy.trade_calendar.get_step_time(1)
        exchange = strategy.trade_exchange
        arrays = [np.array(list(record['target_weights'].values()), dtype=np.float64),
                  np.array([position.get_stock_amount(s) for s in symbols], dtype=np.float64),
                  np.array([float(prices.get(s, math.nan)) for s in symbols], dtype=np.float64),
                  np.ones(len(symbols), dtype=np.float64),
                  np.array([exchange.is_stock_tradable(s, start, end) for s in symbols], dtype=np.bool_)]
        originals = [a.copy() for a in arrays]
        for a in arrays:
            a.setflags(write=False)
        result = kernel.run_target_session(*arrays, equity=record['sizing_equity'], compiled=compiled)
        self.assertIsNotNone(result)
        for a, original in zip(arrays, originals):
            np.testing.assert_array_equal(a.view(np.uint8), original.view(np.uint8))
        targets = {s: int(result.targets[i]) for i, s in enumerate(symbols) if result.active[i]}
        self.assertEqual(targets, record['target_quantities'])
        self.assertEqual(list(targets), list(record['target_quantities']))
        reasons = {symbols[i]: 'unavailable_for_trade' if r == kernel.UNAVAILABLE else 'missing_or_invalid_open'
                   for i, r in enumerate(result.reasons) if r}
        self.assertEqual(reasons, record['reasons'])
        descriptors = [*result.reductions, *result.increases]
        from qlib.backtest.decision import Order
        orders = [Order(symbols[int(i)], int(amount), int(direction), start, end)
                  for i, amount, direction in descriptors]
        self.assertEqual(self.native.orders(orders), self.native.orders(decision.get_decision()))
        return arrays, result

    def variants(self):
        return (False, True) if importlib.util.find_spec('numba') else (False,)

    def test_reversals_phase_sort_held_only_and_negative_zero(self):
        for compiled in self.variants():
            self.compare({'B': .2, 'A': -.4, 'C': -0.0}, holdings={'A': 5., 'B': -4., 'C': 2.}, compiled=compiled)
            self.compare({'A': -0.0}, holdings={'B': -2., 'C': 2.}, compiled=compiled)
            self.compare({}, holdings={'A': 2.}, compiled=compiled)

    def test_subnormal_boundary_missing_and_untradable(self):
        for compiled in self.variants():
            self.compare({'A': 5e-324, 'B': -.5}, compiled=compiled)
            self.compare({'A': .5, 'B': -.5}, {'A': .1, 'B': 3.}, compiled=compiled)
            self.compare({'A': .5, 'B': -.5}, {'A': math.nan, 'B': 20.}, compiled=compiled)
            self.compare({'MISSING': -.5, 'B': .5}, compiled=compiled)

    def test_exceptional_ranges_and_factor_fall_back_atomically(self):
        np = self.np
        for compiled in self.variants():
            for weight, current, price, factor, equity in (
                (.5, 0., 1e-300, 1., 1e100), (.5, 0., .1, 1., float(2**53)),
                (.5, .5, 10., 1., 100.), (.5, 0., 10., 2., 100.),
                (.5, math.inf, 10., 1., 100.), (.5, 0., 10., 1., 0.)):
                arrays = [np.array([v], dtype=np.float64) for v in (weight, current, price, factor)]
                arrays.append(np.ones(1, dtype=np.bool_))
                originals = [a.copy() for a in arrays]
                self.assertIsNone(kernel.run_target_session(*arrays, equity=equity, compiled=compiled))
                for a, original in zip(arrays, originals):
                    np.testing.assert_array_equal(a.view(np.uint8), original.view(np.uint8))

    def test_deterministic_randomized_native_decisions(self):
        for compiled in self.variants():
            for seed in range(40):
                rng = random.Random(seed)
                weights = {s: rng.uniform(-.2, .2) for s in ('A', 'B', 'C')}
                prices = {s: rng.choice((.1, 1., 7.25, 100.)) for s in weights}
                holdings = {s: rng.randrange(-12, 13) for s in weights}
                with self.subTest(seed=seed, compiled=compiled):
                    self.compare(weights, prices, holdings, compiled=compiled, cash=10000.)

    def test_native_error_phase_precedence_and_warning_policy_fallback(self):
        np = self.np
        for compiled in self.variants():
            for factor, expected_error, targets_written in ((2., NotImplementedError, False),
                                                           (1., ValueError, True)):
                frame = self.native.frame({'A': 10., 'B': 20.}, {'B': factor})
                strategy, position = self.native.strategy(False, self.pd.Series({'A': .5, 'B': -.5}),
                                       frame=frame, holdings={'A': .5})
                equity = position.calculate_value()
                arrays = [np.array(v, dtype=np.float64) for v in
                          ([.5, -.5], [.5, 0.], [10., 20.], [1., factor])]
                arrays.append(np.ones(2, dtype=np.bool_))
                self.assertIsNone(kernel.run_target_session(*arrays, equity=equity, compiled=compiled))
                with self.assertRaises(expected_error):
                    strategy.generate_trade_decision()
                record = strategy.decisions[0]
                self.assertEqual(record['target_quantities'] is not None, targets_written)
                self.assertEqual(record['issued_orders'], [])
            arrays = [np.array(v, dtype=np.float64) for v in ([.5], [0.], [10.], [1.])]
            arrays.append(np.ones(1, dtype=np.bool_))
            with np.errstate(all='raise'):
                self.assertIsNone(kernel.run_target_session(*arrays, equity=100., compiled=compiled))
