"""Actual native complete-panel equivalence; no production activation."""
import unittest
from unittest.mock import patch
import warnings

from tests import test_qlib_quote_construction as native

if native.NumpyQuote is not None:
    import numpy as np
    import pandas as pd
    from qlib.utils.index_data import Index, MultiData
    from qlib_yahoo.quote_construction_v2 import native_quote_panels_v2


@unittest.skipIf(native.NumpyQuote is None, 'optional native Qlib required')
class QuoteConstructionV2Tests(unittest.TestCase):
    frame = native.QuoteConstructionTests.frame
    panels_equal = native.QuoteConstructionTests.panels_equal

    def test_native_bits_metadata_and_owned_mutable_panels(self):
        frame = self.frame()
        original = frame.copy(deep=True)
        actual = native_quote_panels_v2(frame, 'day')
        expected = native.NumpyQuote(frame, 'day').data
        self.panels_equal(actual, expected)
        panels = list(actual.values())
        for panel in panels:
            self.assertTrue(panel.data.flags.c_contiguous)
            self.assertTrue(panel.data.flags.owndata)
            self.assertEqual(panel.data.dtype, np.dtype('float64'))
        for a, b in zip(panels, panels[1:]):
            for ai, bi in zip(a.indices, b.indices):
                self.assertFalse(np.shares_memory(ai.idx_list, bi.idx_list))
                self.assertIsNot(ai.index_map, bi.index_map)
        saved = panels[1].data.copy()
        columns = panels[1].columns.idx_list.copy()
        column_map = dict(panels[1].columns.index_map)
        panels[0].data[:] = 999
        panels[0].columns.idx_list[0] = 'MUTATED'
        panels[0].columns.index_map.clear()
        np.testing.assert_array_equal(panels[1].data, saved)
        np.testing.assert_array_equal(panels[1].columns.idx_list, columns)
        self.assertEqual(panels[1].columns.index_map, column_map)
        pd.testing.assert_frame_equal(frame, original, check_exact=True)
        frame['$close'] = np.float32(777)
        frame['$volume'] = np.int64(777)
        frame['$valid'] = False
        np.testing.assert_array_equal(panels[1].data, saved)

    def test_promotions_payloads_and_all_numpy_error_policies(self):
        index = self.frame().index[:5]
        for policy in ('ignore', 'warn', 'raise', 'call', 'print', 'log'):
            # No callback/log sink is reached: the payloads here are quiet NaNs.
            with np.errstate(all=policy):
                for da, db in (('float16', 'int8'), ('float32', 'int32'), ('uint64', 'int64'),
                               ('bool', 'float32'), ('bool', 'float64'), ('bool', 'bool')):
                    frame = pd.DataFrame({'z': np.array([0, 1, 0, 1, 0], dtype=da),
                                          'a': np.array([4, 3, 2, 1, 0], dtype=db)}, index=index)
                    self.panels_equal(native_quote_panels_v2(frame, 'day'), native.NumpyQuote(frame, 'day').data)
                for bits, dtype in (([0x8000000000000000, 0x7ff8000000000001,
                                     0x7ff0000000000000, 0xfff0000000000000, 1], np.uint64),
                                    ([0x80000000, 0x7fc00001, 0x7f800000, 0xff800000, 1], np.uint32)):
                    values = np.array(bits, dtype=dtype).view(np.float64 if dtype is np.uint64 else np.float32)
                    frame = pd.DataFrame({'payload': values, 'flag': [True, False, True, False, True]}, index=index)
                    self.panels_equal(native_quote_panels_v2(frame, 'day'), native.NumpyQuote(frame, 'day').data)

    def test_signalling_nan_conversion_error_and_warning_order(self):
        index = self.frame().index[:1]
        for dtype, bits in ((np.uint32, 0x7f800001), (np.uint64, 0x7ff0000000000001)):
            values = np.array([bits], dtype=dtype).view(np.float32 if dtype is np.uint32 else np.float64)
            for mixed in (False, True):
                frame = pd.DataFrame({'payload': values}, index=index)
                if mixed:
                    frame['flag'] = True
                for policy in ('ignore', 'warn', 'raise'):
                    outcomes = []
                    for constructor in (lambda f: native.NumpyQuote(f, 'day').data,
                                        lambda f: native_quote_panels_v2(f, 'day')):
                        with np.errstate(invalid=policy), warnings.catch_warnings(record=True) as captured:
                            warnings.simplefilter('always')
                            try:
                                result = constructor(frame)
                                outcome = ('ok', result['AAA'].data.tobytes())
                            except Exception as exc:
                                outcome = (type(exc), str(exc))
                            outcomes.append((outcome, [(w.category, str(w.message)) for w in captured]))
                    self.assertEqual(*outcomes)

    def test_custom_bindings_decline_without_callbacks_or_allocation(self):
        frame = self.frame()
        for cls, name in ((Index, 'sort'), (MultiData, '__init__'),
                          (pd.DataFrame, '__getitem__'), (pd.Series, 'to_numpy')):
            with patch.object(cls, name, side_effect=AssertionError('callback')) as callback:
                self.assertIsNone(native_quote_panels_v2(frame, 'day'))
                callback.assert_not_called()
        with patch.object(np, 'take', side_effect=AssertionError('callback')) as callback:
            self.assertIsNone(native_quote_panels_v2(frame, 'day'))
            callback.assert_not_called()

    def test_sorted_identity_does_not_take_and_unsupported_inputs_decline(self):
        frame = self.frame()
        # A profiler hook, not a substituted NumPy binding, observes calls while
        # retaining the guard's original callable identity.
        import sys
        calls = []
        def profile(current, event, arg):
            if event == 'call' and current.f_code.co_name == 'take':
                calls.append(current.f_code.co_filename)
        previous = sys.getprofile()
        try:
            sys.setprofile(profile)
            self.assertIsNotNone(native_quote_panels_v2(frame, 'day'))
        finally:
            sys.setprofile(previous)
        self.assertEqual(calls, [])
        for bad in (frame.iloc[::-1], frame.astype(object), frame.astype('Float64'),
                    pd.concat([frame, frame]).sort_index(), frame.reset_index()):
            self.assertIsNone(native_quote_panels_v2(bad, 'day'))
        self.assertEqual(native_quote_panels_v2(frame.iloc[:0], 'day'), {})


def complete_probe(output, *, pairs=5, stocks=100, days=1500):
    """Complete constructor timings; source creation and full parity excluded."""
    import gc
    import hashlib
    import json
    from pathlib import Path
    import statistics
    import time
    from qlib_yahoo.quote_construction import native_quote_panels
    from qlib_yahoo.runtime import ROOT, ensure_signed_runtime
    output = Path(output).resolve()
    if output.exists() or not output.parent.is_dir():
        raise ValueError('Use a new receipt in an existing directory')
    progress = {'schema': 1, 'kind': 'synthetic_complete_native_quote_constructor',
                'status': 'running', 'stocks': stocks, 'days': days, 'trials': []}
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(progress, indent=2))
    def save():
        output.write_text(json.dumps(progress, indent=2), encoding='utf-8')
    try:
        progress['engine'] = ensure_signed_runtime()
        dates = pd.bdate_range('2019-01-01', periods=days).as_unit('ns')
        index = pd.MultiIndex.from_product([[f'S{i:05}' for i in range(stocks)], dates],
                                           names=['instrument', 'datetime'])
        rng = np.random.default_rng(73)
        values = rng.normal(size=(len(index), 6))
        values[::23, 1] = np.nan
        values[0, 0] = -0.
        frame = pd.DataFrame(values, index=index, columns=['$open', '$close', '$volume',
                                                            '$factor', '$extra1', '$extra2'])
        frame['$extra1'] = frame['$extra1'].astype(np.float32)
        frame['$extra2'] = frame['$extra2'].astype(np.float32)
        mixed = frame.copy(deep=True)
        mixed['limit_buy'] = np.arange(len(index)) % 7 == 0
        mixed['limit_sell'] = np.arange(len(index)) % 11 == 0
        inputs = {'native_float': frame, 'mixed_boolean_float': mixed}
        constructors = {'original': lambda f: native.NumpyQuote(f, 'day', 'us').data,
                        'current': lambda f: native_quote_panels(f, 'day'),
                        'v2': lambda f: native_quote_panels_v2(f, 'day')}
        case = QuoteConstructionV2Tests()
        def verify(actual, expected, source):
            if actual is None:
                raise AssertionError('Unexpected fallback')
            case.panels_equal(actual, expected)
            for axis in (0, 1):
                if len({id(p.indices[axis].index_map) for p in actual.values()}) != stocks:
                    raise AssertionError('Shared mutable native index map')
            panels = list(actual.values())
            for panel, reference in zip(panels, expected.values()):
                for flag in ('c_contiguous', 'f_contiguous', 'owndata', 'writeable', 'aligned'):
                    case.assertEqual(getattr(panel.data.flags, flag), getattr(reference.data.flags, flag))
                case.assertEqual(panel.data.strides, reference.data.strides)
                for column in source.columns:
                    case.assertFalse(np.shares_memory(panel.data, source[column].to_numpy(copy=False)))
            for a, b in zip(panels, panels[1:]):
                for ai, bi in zip(a.indices, b.indices):
                    case.assertFalse(np.shares_memory(ai.idx_list, bi.idx_list))
            return {'panels': len(actual), 'rows': sum(len(p.data) for p in panels),
                    'float64_bytes': sum(p.data.nbytes for p in panels),
                    'bits_maps_metadata_flags_strides_equal': True,
                    'detached_data_label_arrays_maps': True}
        progress['phase'] = 'initial_native_parity_outside_timing'
        save()
        # Prime ordinary native imports/frequency caches, not constructor data.
        for source in inputs.values():
            reference = constructors['original'](source)
            for name in ('current', 'v2'):
                verify(constructors[name](source), reference, source)
            del reference
        for pair in range(pairs):
            for fixture, source in inputs.items():
                trial = {'pair': pair, 'fixture': fixture, 'shape': list(source.shape), 'results': {}}
                names = ('original', 'current', 'v2') if pair % 2 == 0 else ('v2', 'current', 'original')
                progress.update(phase='paired_complete_constructor', active_pair=pair, active_fixture=fixture)
                save()
                panels = {}
                for name in names:
                    gc.collect()
                    before = time.perf_counter()
                    actual = constructors[name](source)
                    elapsed = time.perf_counter() - before
                    panels[name] = actual
                    trial['results'][name] = {'seconds': elapsed}
                for name, actual in panels.items():
                    trial['results'][name]['parity'] = verify(actual, panels['original'], source)
                progress['trials'].append(trial)
                save()
                del panels, actual
        progress['median_seconds'] = {fixture: {name: statistics.median(
            trial['results'][name]['seconds'] for trial in progress['trials'] if trial['fixture'] == fixture)
            for name in constructors} for fixture in inputs}
        progress['source_sha256'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ('qlib_yahoo/quote_construction.py', 'qlib_yahoo/quote_construction_v2.py',
                         'tests/test_qlib_quote_construction_v2.py')}
        progress.update(status='completed', phase='completed',
            timing_boundary='All source guards/frame projection, native MultiData constructors/casts/allocations and native Index.sort; original includes full NumpyQuote constructor. Fixture creation/parity/garbage collection outside timing.',
            limitations='Bounded synthetic constructor-only evidence, not exchange-provider preparation or full-job saving. No direct float64 mixed-boolean projection.')
        save()
        return progress
    except BaseException as exc:
        progress.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        save()
        raise
