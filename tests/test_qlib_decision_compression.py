"""Compression controls, ownership and real-native continuous-stream probe."""
import copy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from qlib_yahoo.runtime import ROOT
from tests import test_qlib_streaming as ownership
from tests import test_qlib_fast_signed as fixtures

CHILD = 'QLIB_DECISION_COMPRESSION_CHILD'


@unittest.skipUnless(importlib.util.find_spec('pyarrow'), 'optional PyArrow unavailable')
class OwnershipTests(unittest.TestCase):
    setUp = ownership.StreamOwnershipTests.setUp
    session = ownership.StreamOwnershipTests.session
    assert_retained = ownership.StreamOwnershipTests.assert_retained

    def test_strict_level_validation_before_files(self):
        for value in (True, False, 0, 10, 1., '1', None):
            with self.subTest(value=value), patch('pyarrow.parquet.ParquetWriter') as parquet:
                with self.assertRaisesRegex(ValueError, 'integer from 1 through 9'):
                    self.Writer(self.path, self.strategy, decision_gzip_level=value)
                parquet.assert_not_called()
                self.assertEqual(list(self.path.iterdir()), [])

    def save(self, name, level=None):
        path = self.path / name
        path.mkdir()
        self.strategy = SimpleNamespace(decisions=[])
        kwargs = {} if level is None else {'decision_gzip_level': level}
        writer = self.Writer(path, self.strategy, columnar=True, buffered=True, **kwargs)
        logical = []
        from qlib_yahoo.signed import signed_decisions
        for day in ('2025-01-02', '2025-01-03'):
            account, _, _ = self.session(day)
            logical.append(signed_decisions(self.strategy)[0])
            writer.finish_session(account, day)
        writer.close(completed=True)
        compressed = (path / 'signed_decisions.json.gz').read_bytes()
        return writer, compressed, gzip.decompress(compressed), logical

    def test_default_determinism_and_exact_decompressed_json_bytes(self):
        a, a_zip, a_raw, records = self.save('default')
        b, b_zip, b_raw, _ = self.save('six', 6)
        c, _, c_raw, _ = self.save('one', 1)
        self.assertEqual((a.decision_gzip_level, b.decision_gzip_level, c.decision_gzip_level), (6, 6, 1))
        self.assertEqual(a_zip, b_zip)
        self.assertEqual(a_raw, b_raw)
        self.assertEqual(a_raw, c_raw)
        self.assertEqual(json.loads(a_raw), records)
        self.assertEqual(a_zip[4:8], b'\0\0\0\0')

    def test_encoding_and_write_failure_retention_and_abort(self):
        for level in (1, 6):
            for failure in ('encoding', 'write'):
                writer = self.Writer(self.path, self.strategy, columnar=True, buffered=True,
                                     decision_gzip_level=level)
                account, record, raw = self.session('2025-01-02')
                if failure == 'encoding':
                    record['target_weights']['bad'] = float('nan')
                    with patch.object(writer._decisions, 'write', wraps=writer._decisions.write) as write:
                        with self.assertRaises(ValueError):
                            writer.finish_session(account, '2025-01-02')
                        write.assert_not_called()
                else:
                    with patch.object(writer._decisions, 'write', side_effect=OSError('injected write')):
                        with self.assertRaisesRegex(OSError, 'injected write'):
                            writer.finish_session(account, '2025-01-02')
                self.assert_retained(writer, account, '2025-01-02', record, raw)
                writer.close(completed=False)
                self.assertTrue(writer._raw.closed)
                self.assertFalse(list(self.path.iterdir()))
                self.strategy.decisions.clear()

    def test_failed_gzip_close_and_abort_release_all_buffers(self):
        for level in (1, 6):
            writer = self.Writer(self.path, self.strategy, columnar=True, buffered=True,
                                 decision_gzip_level=level)
            original = writer._gzip.close
            def failed_close():
                original()
                raise OSError('injected gzip close')
            with patch.object(writer._gzip, 'close', side_effect=failed_close):
                with self.assertRaisesRegex(OSError, 'injected gzip close'):
                    writer.close(completed=True)
            self.assertTrue(writer._raw.closed)
            self.assertTrue(writer.closed)
            self.assertFalse(list(self.path.iterdir()))
            writer.close(completed=False)


class LauncherTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('qlib') and fixtures._signed_available(),
                         'pinned signed derivative unavailable')
    @unittest.skipIf(os.environ.get(CHILD) == '1', 'parent launcher')
    def test_actual_native_stream(self):
        environment = os.environ.copy()
        environment[CHILD] = '1'
        program = (
            'from qlib_yahoo.runtime import ensure_signed_runtime; ensure_signed_runtime(); '
            'import unittest; '
            "suite=unittest.defaultTestLoader.loadTestsFromName('tests.test_qlib_decision_compression.SignedChild'); "
            'result=unittest.TextTestRunner(verbosity=2).run(suite); '
            'raise SystemExit(not result.wasSuccessful())'
        )
        result = subprocess.run([sys.executable, '-c', program], cwd=ROOT,
                                env=environment, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def native_sessions(stocks=1500, sessions=20):
    """Build real native strategy/fill/bar-close snapshots outside writer timing."""
    import numpy as np
    import pandas as pd
    from qlib.backtest.high_performance_ds import NumpyQuote
    from qlib.backtest.signal import Signal
    from qlib.backtest.signed_account import SignedAccount
    from qlib.backtest.signed_exchange import SignedExchange
    from qlib.contrib.strategy.signed_weight import SignedWeightStrategy
    names = [f'S{i:05d}' for i in range(stocks)]
    dates = pd.date_range('2025-01-02', periods=sessions + 2, freq='B').as_unit('ns')
    index = pd.MultiIndex.from_product([names, dates], names=['instrument', 'datetime'])
    price_values = np.repeat(np.array([1. + i % 50 for i in range(stocks)], dtype=np.float64), len(dates))
    frame = pd.DataFrame({'$open': price_values, '$close': price_values, '$factor': 1.,
                          '$volume': 100000., 'limit_buy': False, 'limit_sell': False}, index=index)
    exchange = SignedExchange.__new__(SignedExchange)
    exchange.quote_df, exchange.quote = frame, NumpyQuote(frame, 'day', region='us')
    exchange.buy_price = exchange.sell_price = '$open'
    exchange.open_cost = exchange.close_cost = .0005
    account = SignedAccount(init_cash=1e6)
    class Calendar:
        start_index = 0
        step = 1
        def get_trade_step(self):
            return self.step
        def get_step_time(self, index=None, shift=0):
            day = dates[(self.step if index is None else index) - shift]
            return day, day + pd.Timedelta(hours=23)
    calendar = Calendar()
    class NativeSignal(Signal):
        def get_signal(self, start_time, end_time):
            return pd.Series({name: (1 if (i + calendar.step) % 2 else -1) * .4 / stocks
                              for i, name in enumerate(names)})
    strategy = SignedWeightStrategy(signal=NativeSignal(), level_infra={'trade_calendar': calendar},
                                    common_infra={'trade_account': account}, trade_exchange=exchange)
    snapshots = []
    for step in range(1, sessions + 2):
        calendar.step = step
        day, end = calendar.get_step_time()
        account.start_bar(day, end, exchange)
        decision = strategy.generate_trade_decision()
        orders = decision.get_decision()
        info = [(order, *exchange.deal_order(order, trade_account=account)) for order in orders]
        account.update_bar_end(day, end, exchange, atomic=True, outer_trade_decision=decision, trade_info=info)
        if step > 1:
            snapshot = copy.copy(account)
            snapshot.current_position = copy.deepcopy(account.current_position)
            snapshot.ledger_records = {day: dict(account.ledger_records[day])}
            snapshot.portfolio_metrics = copy.deepcopy(account.portfolio_metrics)
            snapshot.indicator = copy.copy(account.indicator)
            snapshot.indicator.order_indicator_his = {day: account.indicator.order_indicator_his[day]}
            snapshots.append((day, snapshot, strategy.decisions[0]))
        strategy.decisions.clear()
        account.hist_positions.clear()
        account.indicator.order_indicator_his.clear()
    return snapshots


def write_sessions(path, level, snapshots, *, mutate_after_queue=False):
    from qlib_yahoo.streaming import SignedHistoryWriter
    strategy = SimpleNamespace(decisions=[])
    # Original native snapshot values remain available across levels/pairs.
    accounts = []
    for day, account, record in snapshots:
        owned = copy.copy(account)
        owned.indicator = copy.copy(account.indicator)
        owned.indicator.order_indicator_his = dict(account.indicator.order_indicator_his)
        if mutate_after_queue:
            owned.indicator.order_indicator_his = copy.deepcopy(owned.indicator.order_indicator_his)
        accounts.append((day, owned, record))
    writer = None
    before = time.perf_counter()
    try:
        writer = SignedHistoryWriter(path, strategy, columnar=True, buffered=True, decision_gzip_level=level)
        for day, account, record in accounts:
            strategy.decisions.append(record)
            raw = account.indicator.order_indicator_his[day]
            writer.positions(account, day)
            writer.finish_session(account, day)
            if mutate_after_queue:
                for metric in raw.data.values():
                    metric.data[:] = 999.
        writer.close(completed=True)
    except BaseException:
        if writer is not None:
            writer.close(completed=False)
        raise
    elapsed = time.perf_counter() - before
    return elapsed, writer


def verify_outputs(left, right):
    import pyarrow.parquet as pq
    a = gzip.decompress((left / 'signed_decisions.json.gz').read_bytes())
    b = gzip.decompress((right / 'signed_decisions.json.gz').read_bytes())
    if a != b or json.loads(a) != json.loads(b):
        raise AssertionError('Decompressed JSON bytes/records differ')
    table_rows = {}
    for name in ('positions.parquet', 'qlib_order_indicators.parquet'):
        first, second = pq.read_table(left / name), pq.read_table(right / name)
        if not first.equals(second, check_metadata=True):
            raise AssertionError(f'Native table data/schema differ: {name}')
        table_rows[name] = first.num_rows
    return {'decompressed_sha256': hashlib.sha256(a).hexdigest(), 'sessions': len(json.loads(a)),
            'table_rows': table_rows, 'exact_decompressed_bytes': True}


@unittest.skipUnless(os.environ.get(CHILD) == '1', 'requires verified signed child')
class SignedChild(unittest.TestCase):
    def test_native_continuous_stream_exact_bytes_tables_and_owned_buffers(self):
        snapshots = native_sessions(stocks=12, sessions=3)
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / 'six', Path(directory) / 'one'
            left.mkdir()
            right.mkdir()
            _, first = write_sessions(left, 6, snapshots)
            _, second = write_sessions(right, 1, snapshots, mutate_after_queue=True)
            result = verify_outputs(left, right)
            self.assertEqual(result['sessions'], 3)
            self.assertTrue(first.closed and second.closed)
            self.assertTrue(first._raw.closed and second._raw.closed)
            for writer in (first, second):
                self.assertTrue(writer._buffers)
                for buffer in writer._buffers:
                    self.assertEqual(buffer._tables, [])
                    self.assertEqual(buffer.stats()['pending_bytes'], 0)


def complete_probe(output, *, pairs=5, stocks=1500, sessions=20):
    output = Path(output).resolve()
    if output.exists() or not output.parent.is_dir():
        raise ValueError('Use a new receipt in an existing directory')
    progress = {'schema': 1, 'kind': 'synthetic_native_continuous_history_compression_boundary',
                'status': 'running', 'stocks': stocks, 'sessions': sessions, 'pairs': []}
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(progress, indent=2))
    try:
        from qlib_yahoo.runtime import ensure_signed_runtime
        progress['engine'] = ensure_signed_runtime()
        progress['phase'] = 'native_fixture_preparation_outside_timing'
        output.write_text(json.dumps(progress, indent=2), encoding='utf-8')
        snapshots = native_sessions(stocks, sessions)
        progress['orders_per_session'] = [len(record['issued_orders']) for _, _, record in snapshots]
        with tempfile.TemporaryDirectory(dir=output.parent, prefix='decision-compression-') as directory:
            for pair in range(pairs):
                results, paths = {}, {}
                for level in ((6, 1) if pair % 2 == 0 else (1, 6)):
                    path = Path(directory) / f'pair-{pair}-level-{level}'
                    path.mkdir()
                    elapsed, writer = write_sessions(path, level, snapshots)
                    paths[level] = path
                    results[level] = {'seconds': elapsed,
                        'gzip_bytes': (path / 'signed_decisions.json.gz').stat().st_size,
                        'rows': writer.rows, 'closed': writer.closed}
                parity = verify_outputs(paths[6], paths[1])
                progress['pairs'].append({'pair': pair, 'level6': results[6], 'level1': results[1], 'parity': parity})
                progress['phase'] = 'paired_trials'
                output.write_text(json.dumps(progress, indent=2), encoding='utf-8')
        progress['median_seconds'] = {str(level): statistics.median(p[f'level{level}']['seconds']
                                       for p in progress['pairs']) for level in (6, 1)}
        progress['source_sha256'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ('qlib_yahoo/streaming.py', 'qlib_yahoo/signed.py', 'tests/test_qlib_decision_compression.py')}
        progress.update(status='completed', phase='completed',
            timing_boundary='Writer construction, positions conversion, JSON conversion/encoding, casts, buffered writes, flush and close; actual native fixture prepared outside timing',
            limitations='Synthetic actual native strategy/fills/account/indicators; not full-job timing or accounting acceptance.')
        output.write_text(json.dumps(progress, indent=2), encoding='utf-8')
        return progress
    except BaseException as exc:
        progress.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        output.write_text(json.dumps(progress, indent=2), encoding='utf-8')
        raise
