"""Actual native complete flat-indicator differentials and bounded paired probe."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from qlib_yahoo.runtime import ROOT
from tests import test_qlib_fast_signed as fixtures

CHILD = 'QLIB_FLAT_INDICATOR_CHILD'


class LauncherTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('qlib') and fixtures._signed_available(),
                         'pinned signed derivative unavailable')
    @unittest.skipIf(os.environ.get(CHILD) == '1', 'parent launcher')
    def test_actual_native_indicator(self):
        environment = os.environ.copy()
        environment[CHILD] = '1'
        program = (
            'from qlib_yahoo.runtime import ensure_signed_runtime; ensure_signed_runtime(); '
            'import unittest; '
            "suite=unittest.defaultTestLoader.loadTestsFromName('tests.test_qlib_flat_order_indicator.SignedChild'); "
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
        import numpy as np
        import pandas as pd
        from qlib.backtest.decision import Order
        from qlib.backtest.report import Indicator
        from qlib_yahoo.flat_order_indicator import FlatIndicator, try_update_order_indicators
        cls.np, cls.pd, cls.Order = np, pd, Order
        cls.Native, cls.Fast, cls.try_update = Indicator, FlatIndicator, staticmethod(try_update_order_indicators)

    def trade_info(self, count=500, seed=0):
        rng = random.Random(seed)
        day = self.pd.Timestamp('2025-01-06')
        result = []
        for i in range(count):
            amount = rng.randrange(0, 100)
            direction = rng.randrange(2)
            order = self.Order(f'S{i % max(1, count // 2):05d}', amount, direction, day, day)
            order.deal_amount = float(rng.randrange(0, amount + 1))
            price = rng.choice((.01, 10., 100., self.np.nan))
            value = order.deal_amount * 10.
            result.append((order, value, value * .0005, price))
        return result

    def snapshot(self, indicator):
        return [(name, type(metric), metric.index.idx_list.copy(), metric.data.copy(),
                 dict(metric.index.index_map), metric.index.is_sorted())
                for name, metric in indicator.order_indicator.data.items()]

    def assert_same(self, native, fast):
        np = self.np
        expected, actual = self.snapshot(native), self.snapshot(fast)
        self.assertEqual(len(expected), len(actual))
        for a, b in zip(expected, actual):
            self.assertEqual(a[:2], b[:2])
            np.testing.assert_array_equal(a[2], b[2])
            np.testing.assert_array_equal(a[3].view(np.uint64), b[3].view(np.uint64))
            self.assertEqual(a[4:], b[4:])
        self.assertEqual(list(native.trade_indicator), list(fast.trade_indicator))
        for name in native.trade_indicator:
            np.testing.assert_array_equal(np.asarray(native.trade_indicator[name], dtype=np.float64).view(np.uint64),
                                          np.asarray(fast.trade_indicator[name], dtype=np.float64).view(np.uint64))

    def compare(self, info):
        native, fast = self.Native(), self.Fast()
        identity = fast.order_indicator.data
        native.update_order_indicators(info)
        fast.update_order_indicators(info)
        self.assertIs(fast.order_indicator.data, identity)
        self.assert_same(native, fast)
        for method in ('mean', 'amount_weighted', 'value_weighted'):
            config = {'ffr_config': {'weight_method': method}, 'pa_config': {'weight_method': method}}
            native.cal_trade_indicators(self.pd.Timestamp('2025-01-06'), 'day', config)
            fast.cal_trade_indicators(self.pd.Timestamp('2025-01-06'), 'day', config)
            self.assert_same(native, fast)
        return native, fast

    def test_all_metrics_aggregates_duplicates_zero_nan_and_empty(self):
        for count in (0, 1, 20, 500, 3000):
            with self.subTest(count=count):
                _, fast = self.compare(self.trade_info(count, seed=count))
                self.assertEqual(fast.flat_update_counts['fast'], 1)

    def test_signed_zero_nan_deal_fallback_and_none_price(self):
        info = self.trade_info(4)
        for i, (order, value, cost, price) in enumerate(info):
            order.amount = 0.0 if i % 2 else 2.0
            order.deal_amount = -0.0
            order.direction = i % 2
            info[i] = (order, -0.0, -0.0, None if i == 0 else self.np.nan)
        _, fast = self.compare(info)
        self.assertEqual(fast.flat_update_counts['fast'], 1)
        info[0][0].deal_amount = self.np.nan
        _, fast = self.compare(info)
        self.assertEqual(fast.flat_update_counts['fallback'], 1)

    def test_nonempty_namespace_preserves_identity_and_insertion_order(self):
        native, fast = self.Native(), self.Fast()
        for indicator in (native, fast):
            indicator.order_indicator.assign('custom_existing', {'Z': 3.})
            indicator.order_indicator.assign('trade_price', {'OLD': 4.})
        identity = fast.order_indicator.data
        old_custom = identity['custom_existing']
        native.update_order_indicators(self.trade_info(20))
        fast.update_order_indicators(self.trade_info(20))
        self.assert_same(native, fast)
        self.assertIs(fast.order_indicator.data, identity)
        self.assertIs(identity['custom_existing'], old_custom)
        self.assertEqual(fast.flat_update_counts['fallback'], 1)

    def test_custom_order_and_bindings_decline_before_callbacks(self):
        info = self.trade_info(1)
        calls = []
        class CustomOrder(self.Order):
            @property
            def amount_delta(self):
                calls.append('property')
                return super().amount_delta
        info[0][0].__class__ = CustomOrder
        fast = self.Fast()
        self.assertFalse(self.try_update(fast, info))
        self.assertEqual(calls, [])
        self.assertEqual(fast.order_indicator.data, {})
        fast.update_order_indicators(info)
        self.assertEqual(calls, ['property'])
        for target, name in ((self.Native, '_update_order_trade_info'),
                             (self.Order, 'sign')):
            with patch.object(target, name, property(lambda obj: calls.append('bad descriptor'))):
                self.assertFalse(self.try_update(self.Fast(), self.trade_info(1)))
        self.assertEqual(calls, ['property'])

    def test_policy_overflow_and_custom_state_atomic_decline(self):
        info = self.trade_info(1)
        for policy in ('raise', 'call', 'warn'):
            with self.np.errstate(invalid=policy):
                fast = self.Fast()
                self.assertFalse(self.try_update(fast, info))
                self.assertEqual(fast.order_indicator.data, {})
        huge = self.trade_info(1)
        huge[0][0].amount = float('inf')
        fast = self.Fast()
        self.assertFalse(self.try_update(fast, huge))
        self.assertEqual(fast.order_indicator.data, {})
        fast.order_indicator.assign = lambda *a: self.fail('custom assign called speculatively')
        self.assertFalse(self.try_update(fast, info))

    def test_order_field_descriptors_and_flat_getattribute_decline_without_callbacks(self):
        calls = []
        info = self.trade_info(1)
        fast = self.Fast()
        for name in ('amount', 'deal_amount', 'stock_id', 'direction'):
            with patch.object(self.Order, name, property(lambda obj: calls.append('field callback')), create=True):
                self.assertFalse(self.try_update(fast, info))
                self.assertEqual(calls, [])
        original = self.Fast.__getattribute__
        def changed(obj, name):
            calls.append(name)
            return original(obj, name)
        with patch.object(self.Fast, '__getattribute__', changed):
            self.assertFalse(self.try_update(fast, info))
            self.assertEqual(calls, [])
        self.assertEqual(fast.order_indicator.data, {})


def complete_probe(output, *, pairs=20, repeats=5):
    """New durable receipt: complete native/guarded update, no backtest claims."""
    output = Path(output).resolve()
    if output.exists() or not output.parent.is_dir():
        raise ValueError('Use a new receipt path in an existing directory')
    progress = {'schema': 1, 'kind': 'synthetic_complete_flat_indicator_boundary_only',
                'status': 'running', 'completed_pairs': [], 'pairs_per_size': pairs,
                'repeats_per_measurement': repeats}
    # Exclusive creation owns this receipt; retain progress/failure thereafter.
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(progress, indent=2))
    try:
        from qlib_yahoo.runtime import ensure_signed_runtime
        progress['engine'] = ensure_signed_runtime()
        progress['source_sha256'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ('qlib_yahoo/flat_order_indicator.py', 'tests/test_qlib_flat_order_indicator.py')}
        SignedChild.setUpClass()
        fixture = SignedChild()
        progress['numpy_version'] = fixture.np.__version__
        progress['numpy_policy'] = fixture.np.geterr()
        for count in (500, 3000):
            info = fixture.trade_info(count)
            for pair in range(pairs):
                times = {}
                outcomes = {}
                for fast in ((False, True) if pair % 2 == 0 else (True, False)):
                    constructor = fixture.Fast if fast else fixture.Native
                    indicators = [constructor() for _ in range(repeats)]
                    before = time.perf_counter()
                    for indicator in indicators:
                        indicator.update_order_indicators(info)
                    times[fast] = (time.perf_counter() - before) / repeats
                    outcomes[fast] = indicators[-1]
                    if fast and any(i.flat_update_counts['fast'] != 1 for i in indicators):
                        raise AssertionError('Unexpected fallback; reject timing')
                fixture.assert_same(outcomes[False], outcomes[True])
                progress['completed_pairs'].append({'orders': count, 'pair': pair,
                    'native_seconds': times[False], 'guarded_seconds': times[True], 'exact': True})
                output.write_text(json.dumps(progress, indent=2), encoding='utf-8')
        progress['medians'] = {str(count): {
            key: statistics.median(p[key] for p in progress['completed_pairs'] if p['orders'] == count)
            for key in ('native_seconds', 'guarded_seconds')} for count in (500, 3000)}
        progress.update(status='completed', limitations='Synthetic full update boundary; not ordinary full-worker savings.')
        output.write_text(json.dumps(progress, indent=2), encoding='utf-8')
        return progress
    except BaseException as exc:
        progress.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        output.write_text(json.dumps(progress, indent=2), encoding='utf-8')
        raise
