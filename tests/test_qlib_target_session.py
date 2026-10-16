"""Complete compiled target adapter/native strategy differentials."""
import importlib.util
import os
import random
import subprocess
import sys
import unittest
from unittest.mock import patch

from qlib_yahoo.runtime import ROOT
from tests import test_qlib_fast_signed as fixtures

CHILD = 'QLIB_TARGET_ADAPTER_CHILD'


class LauncherTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('qlib') and fixtures._signed_available(),
                         'pinned signed derivative unavailable')
    @unittest.skipIf(os.environ.get(CHILD) == '1', 'parent launcher')
    def test_actual_complete_target_adapter(self):
        environment = os.environ.copy()
        environment[CHILD] = '1'
        program = (
            'from qlib_yahoo.runtime import ensure_signed_runtime; ensure_signed_runtime(); '
            'import unittest; '
            "suite=unittest.defaultTestLoader.loadTestsFromName('tests.test_qlib_target_session.SignedChild'); "
            'result=unittest.TextTestRunner(verbosity=2).run(suite); '
            'raise SystemExit(not result.wasSuccessful())'
        )
        result = subprocess.run([sys.executable, '-c', program], cwd=ROOT,
                                env=environment, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(os.environ.get(CHILD) == '1' and importlib.util.find_spec('numba'),
                     'requires verified signed child and optional Numba')
class SignedChild(unittest.TestCase):
    setUpClass = classmethod(fixtures.SignedChild.setUpClass.__func__)
    frame = fixtures.SignedChild.frame
    exchange = fixtures.SignedChild.exchange
    strategy = fixtures.SignedChild.strategy
    orders = staticmethod(fixtures.SignedChild.orders)
    outcome = fixtures.SignedChild.outcome

    def compare(self, weights, *, frame=None, holdings=None, cash=100., quote_class=None, mutation=None):
        outcomes, histories, strategies = [], [], []
        for compiled in (False, True):
            quote = quote_class or self.DailyQuote
            exchange = self.exchange(True, frame=frame, quote_class=quote)
            strategy, position = self.strategy(True, self.pd.Series(weights, dtype=float), frame=frame,
                        holdings=holdings, cash=cash, exchange=exchange)
            if not compiled:
                strategy.__class__ = self.NativeStrategy
            else:
                strategy.compiled_targets = True
            calls = []
            calendar = strategy.trade_calendar
            original = calendar.get_step_time
            def observed(index=None, shift=0):
                calls.append((index, shift))
                return original(index, shift)
            calendar.get_step_time = observed
            if mutation:
                mutation(strategy, position, exchange, calls)
            outcomes.append(self.outcome(strategy))
            histories.append(calls)
            strategies.append(strategy)
            self.assertEqual(len(strategy.decisions), 1)
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(histories[0], histories[1])
        return strategies, outcomes

    def test_default_readers_reversals_zero_tiny_gaps_and_factor_errors(self):
        for quote in (self.NumpyQuote, self.DailyQuote):
            for weights, prices, factors, holdings, cash in (
                ({'AAA': -.4, 'BBB': .2, 'CCC': -0.}, None, None, {'AAA': 5, 'BBB': -4, 'CCC': 2}, 100.),
                ({'AAA': 5e-324, 'BBB': -.5}, None, None, None, 100.),
                ({'AAA': .5, 'BBB': -.5}, {'AAA': self.np.nan, 'BBB': 20.}, None, None, 100.),
                ({'MISSING': .5, 'BBB': -.5}, None, None, None, 100.),
                ({'AAA': .5, 'BBB': -.5}, None, {'BBB': 2.}, {'AAA': .5}, 100.),
                ({'AAA': .5, 'BBB': -.5}, None, None, {'AAA': .5}, 100.),
                ({'AAA': .5, 'BBB': -.5}, {'AAA': 1e-300, 'BBB': 20.}, {'BBB': 2.}, None, 1e100),
                ({'AAA': .5}, None, None, None, 0.),
            ):
                with self.subTest(quote=quote, weights=weights, cash=cash):
                    self.compare(weights, frame=self.frame(prices, factors), holdings=holdings,
                                 cash=cash, quote_class=quote)

    def test_random_3000_members_complete_native_decisions(self):
        rng = random.Random(7345)
        names = [f'S{i:04d}' for i in range(3000)]
        weights = {s: rng.uniform(-1, 1) / 6000 for s in names}
        prices = {s: rng.uniform(.1, 100.) for s in names}
        holdings = {s: rng.randrange(-10, 11) for s in names[::3]}
        strategies, _ = self.compare(weights, frame=self.frame(prices), holdings=holdings, cash=1e6)
        self.assertEqual(strategies[1].target_session_counts['compiled_sessions'], 1)
        self.assertEqual(strategies[1].target_session_counts['target_symbols'], 3000)

    def test_signal_equity_callbacks_changing_binding_keep_single_record(self):
        for callback in ('signal', 'equity'):
            def mutation(strategy, position, exchange, calls):
                obj, name = ((strategy.signal, 'get_signal') if callback == 'signal'
                             else (position, 'calculate_value'))
                original = getattr(obj, name)
                def altered(*args, **kwargs):
                    exchange.quote.get_data = lambda *a, **kw: 2.
                    return original(*args, **kwargs)
                setattr(obj, name, altered)
            self.compare({'AAA': .5}, mutation=mutation)

    def test_mutated_panel_index_and_source_callbacks_fall_back(self):
        for mode in ('data', 'index', 'mapping', 'source'):
            def mutation(strategy, position, exchange, calls):
                panel = exchange.quote.data['AAA']
                if mode == 'data':
                    panel.data = panel.data.view(type('CustomArray', (self.np.ndarray,), {}))
                elif mode == 'index':
                    panel.indices[0].idx_list = panel.indices[0].idx_list.view(type('CustomIndex', (self.np.ndarray,), {}))
                elif mode == 'mapping':
                    panel.indices[0].index_map = type('CustomMap', (dict,), {})(panel.indices[0].index_map)
                else:
                    exchange.quote.get_data = lambda *a, **kw: 10.
            strategies, _ = self.compare({'AAA': .5}, mutation=mutation)
            self.assertEqual(strategies[1].target_session_counts['compiled_sessions'], 0)

    def test_atomic_failure_restores_existing_cache_identity_and_contents(self):
        from qlib_yahoo.target_session import try_target_orders
        exchange = self.exchange(True, self.frame(factors={'BBB': 2.}), self.DailyQuote)
        start, end = self.dates[1], self.dates[1] + self.pd.Timedelta(hours=23)
        exchange.begin_session(start, end)
        exchange.get_factor('AAA', start, end)
        cached = exchange._session_rows
        before = cached.copy()
        result = try_target_orders(exchange, start, end, {'AAA': .5, 'BBB': -.5}, {}, 100.)
        self.assertIs(result, NotImplemented)
        self.assertIs(exchange._session_rows, cached)
        self.assertEqual(exchange._session_rows, before)

    def test_mutated_mapping_value_does_not_speculate_past_native_error(self):
        def mutation(strategy, position, exchange, calls):
            class IndexCallback:
                def __index__(self):
                    calls.append('mapping callback')
                    return 0
            # BBB is inspected only after AAA's sizing overflow in the native loop.
            panel = exchange.quote.data['BBB']
            day = self.dates[1].as_unit('ns').asm8
            panel.indices[0].index_map[day] = IndexCallback()
        self.compare({'AAA': .5, 'BBB': -.5}, cash=1e100,
                     frame=self.frame({'AAA': 1e-300, 'BBB': 20.}), mutation=mutation)

    def test_weekend_and_zero_observed_bar_native_fallback(self):
        def mutation(strategy, position, exchange, calls):
            strategy.trade_calendar.get_step_time = lambda index=None, shift=0: (
                (self.pd.Timestamp('2024-01-06'), self.pd.Timestamp('2024-01-07 23:00'))
                if shift == 0 else (self.dates[0], self.dates[0]))
        self.compare({'AAA': .5, 'BBB': -.5}, mutation=mutation)

    def test_custom_cached_row_does_not_speculate_past_native_error(self):
        def mutation(strategy, position, exchange, calls):
            class CachedRow(tuple):
                def __getitem__(self, index):
                    calls.append('cached row callback')
                    return super().__getitem__(index)
            start, end = strategy.trade_calendar.get_step_time(1)
            exchange.begin_session(start, end)
            row = exchange._session_row('BBB', start, end)
            exchange._session_rows['BBB'] = CachedRow(row)
            calls.clear()
        self.compare({'AAA': .5, 'BBB': -.5}, cash=1e100,
                     frame=self.frame({'AAA': 1e-300, 'BBB': 20.}), mutation=mutation)
