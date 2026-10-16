"""Native V3 equivalence including repeated/sparse vectors and mutable state."""
import unittest
import warnings
from unittest.mock import patch
from tests import test_qlib_quote_construction as native
from tests import test_qlib_quote_construction_v2 as v2

if native.NumpyQuote is not None:
    import numpy as np
    import pandas as pd
    from qlib.utils.index_data import Index
    from qlib_yahoo.quote_construction_v3 import native_quote_panels_v3


@unittest.skipIf(native.NumpyQuote is None, 'optional native Qlib required')
class QuoteConstructionV3Tests(v2.QuoteConstructionV2Tests):
    def setUp(self):
        policy = np.errstate(invalid='ignore')
        policy.__enter__()
        self.addCleanup(policy.__exit__, None, None, None)
        # Reuse the whole native V2 primitive/error/callback corpus without
        # modifying either candidate or its source test module permanently.
        replacement = patch.object(v2, 'native_quote_panels_v2', native_quote_panels_v3)
        replacement.start()
        self.addCleanup(replacement.stop)

    def test_sorted_identity_does_not_take_and_unsupported_inputs_decline(self):
        # V3 intentionally retains native data-copy effects, including take.
        frame = self.frame()
        self.panels_equal(native_quote_panels_v3(frame, 'day'), native.NumpyQuote(frame, 'day').data)
        for bad in (frame.iloc[::-1], frame.astype(object), frame.astype('Float64'),
                    pd.concat([frame, frame]).sort_index(), frame.reset_index()):
            self.assertIsNone(native_quote_panels_v3(bad, 'day'))
        self.assertEqual(native_quote_panels_v3(frame.iloc[:0], 'day'), {})

    def test_promotions_payloads_and_all_numpy_error_policies(self):
        index = self.frame().index[:5]
        for policy in ('ignore', 'warn', 'raise', 'print', 'call', 'log'):
            with np.errstate(all=policy):
                for da, db in (('float16', 'int8'), ('float32', 'int32'), ('uint64', 'int64'),
                               ('bool', 'float32'), ('bool', 'float64'), ('bool', 'bool')):
                    frame = pd.DataFrame({'z': np.array([0, 1, 0, 1, 0], dtype=da),
                                          'a': np.array([4, 3, 2, 1, 0], dtype=db)}, index=index)
                    if policy != 'ignore':
                        self.assertIsNone(native_quote_panels_v3(frame, 'day'))
                    else:
                        self.panels_equal(native_quote_panels_v3(frame, 'day'), native.NumpyQuote(frame, 'day').data)
                for bits, dtype in (([0x8000000000000000, 0x7ff8000000000001,
                                     0x7ff0000000000000, 0xfff0000000000000, 1], np.uint64),
                                    ([0x80000000, 0x7fc00001, 0x7f800000, 0xff800000, 1], np.uint32)):
                    values = np.array(bits, dtype=dtype).view(np.float64 if dtype is np.uint64 else np.float32)
                    frame = pd.DataFrame({'payload': values, 'flag': [True, False, True, False, True]}, index=index)
                    if policy != 'ignore':
                        self.assertIsNone(native_quote_panels_v3(frame, 'day'))
                    else:
                        self.panels_equal(native_quote_panels_v3(frame, 'day'), native.NumpyQuote(frame, 'day').data)

    def test_many_repeated_sparse_vectors_and_bounded_eviction(self):
        dates = pd.bdate_range('2020-01-01', periods=80).as_unit('ns')
        # Same endpoints/length but different holes must not share a cache key;
        # >32 distinct vectors also exercise eviction followed by reuse.
        vectors = [dates.delete(i) for i in range(1, 40)]
        vectors += [dates, dates.delete(2), dates, dates.delete(1)] * 4
        tuples = [(f'S{i:03}', day) for i, vector in enumerate(vectors) for day in vector]
        index = pd.MultiIndex.from_tuples(tuples, names=['instrument', 'datetime'])
        frame = pd.DataFrame({'$open': np.arange(len(index), dtype=np.float32),
                              '$flag': np.arange(len(index)) % 2 == 0}, index=index)
        original = frame.copy(deep=True)
        diagnostics = {}
        actual = native_quote_panels_v3(frame, 'day', diagnostics=diagnostics)
        self.assertGreater(diagnostics['hits'], 0)
        self.assertEqual(diagnostics['peak_entries'], 32)
        self.assertEqual(diagnostics['hits'] + diagnostics['misses'], len(vectors))
        self.panels_equal(actual, native.NumpyQuote(frame, 'day').data)
        panels = list(actual.values())
        for axis in (0, 1):
            self.assertEqual(len({id(p.indices[axis].index_map) for p in panels}), len(panels))
            for a, b in zip(panels, panels[1:]):
                self.assertFalse(np.shares_memory(a.indices[axis].idx_list, b.indices[axis].idx_list))
        saved = panels[-1].data.tobytes()
        labels = panels[-1].index.idx_list.tobytes()
        maps = dict(panels[-1].index.index_map)
        panels[-3].data[:] = -123
        panels[-3].index.idx_list[0] = np.datetime64('1999-01-01', 'ns')
        panels[-3].index.index_map.clear()
        self.assertEqual(panels[-1].data.tobytes(), saved)
        self.assertEqual(panels[-1].index.idx_list.tobytes(), labels)
        self.assertEqual(panels[-1].index.index_map, maps)
        pd.testing.assert_frame_equal(frame, original, check_exact=True)

    def test_native_index_field_descriptor_declines_before_callback(self):
        frame = self.frame()
        def fail(instance):
            raise AssertionError('descriptor callback')
        with patch.object(Index, 'idx_list', property(fail), create=True):
            self.assertIsNone(native_quote_panels_v3(frame, 'day'))
        from qlib.utils import index_data
        with patch.object(index_data, 'Index', side_effect=AssertionError('module callback')) as callback:
            self.assertIsNone(native_quote_panels_v3(frame, 'day'))
            callback.assert_not_called()
        for name in ('__getattribute__', '__setattr__'):
            with patch.object(Index, name, side_effect=AssertionError('access callback')) as callback:
                self.assertIsNone(native_quote_panels_v3(frame, 'day'))
                callback.assert_not_called()

    def fallback(self, frame, freq):
        result = native_quote_panels_v3(frame, freq)
        if np.geterr()['invalid'] != 'ignore':
            self.assertIsNone(result)
        return native.NumpyQuote(frame, freq).data if result is None else result

    def test_signalling_nan_conversion_error_and_warning_order(self):
        with patch.object(v2, 'native_quote_panels_v2', self.fallback):
            v2.QuoteConstructionV2Tests.test_signalling_nan_conversion_error_and_warning_order(self)

    def test_cached_vector_later_signalling_nan_preserves_cast_prefix(self):
        dates = pd.bdate_range('2021-01-01', periods=3).as_unit('ns')
        index = pd.MultiIndex.from_product([['AAA', 'BBB', 'CCC'], dates],
                                           names=['instrument', 'datetime'])
        for mixed in (False, True):
            values = np.ones(9, dtype=np.float32)
            values.view(np.uint32)[4] = 0x7f800001
            frame = pd.DataFrame({'payload': values}, index=index)
            if mixed:
                frame['flag'] = True
            for policy in ('ignore', 'warn', 'raise'):
                outcomes = []
                for constructor in (lambda f: native.NumpyQuote(f, 'day').data,
                                    lambda f: self.fallback(f, 'day')):
                    with np.errstate(invalid=policy), warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter('always')
                        try:
                            result = constructor(frame)
                            outcome = ('ok', [(name, p.data.tobytes()) for name, p in result.items()])
                        except Exception as exc:
                            outcome = (type(exc), str(exc))
                        outcomes.append((outcome, [(w.category, str(w.message)) for w in captured]))
                self.assertEqual(*outcomes)


def complete_probe(output, *, pairs=5, stocks=100, days=1500):
    """Draft complete V3/current/original constructor probe; explicit invocation."""
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
    progress = {'schema': 1, 'kind': 'synthetic_complete_native_quote_constructor_v3',
                'status': 'running', 'stocks': stocks, 'days': days, 'trials': []}
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(progress, indent=2))
    def save():
        output.write_text(json.dumps(progress, indent=2), encoding='utf-8')
    try:
        progress['engine'] = ensure_signed_runtime()
        dates = pd.bdate_range('2019-01-01', periods=days).as_unit('ns')
        rng = np.random.default_rng(73)
        inputs = {}
        for kind in ('repeated_float', 'sparse_mixed_boolean'):
            tuples = [(f'S{i:05}', date) for i in range(stocks) for j, date in enumerate(dates)
                      if kind == 'repeated_float' or j % 17 != i % 5]
            index = pd.MultiIndex.from_tuples(tuples, names=['instrument', 'datetime'])
            frame = pd.DataFrame(rng.normal(size=(len(index), 6)), index=index,
                                 columns=['$open', '$close', '$volume', '$factor', '$extra1', '$extra2'])
            frame.iloc[::23, 1] = np.nan
            frame.iloc[0, 0] = -0.
            frame['$extra1'] = frame['$extra1'].astype(np.float32)
            frame['$extra2'] = frame['$extra2'].astype(np.float32)
            if kind == 'sparse_mixed_boolean':
                frame['limit_buy'] = np.arange(len(index)) % 7 == 0
                frame['limit_sell'] = np.arange(len(index)) % 11 == 0
            inputs[kind] = frame
        last_diagnostics = {}
        def construct_v3(source):
            last_diagnostics.clear()
            return native_quote_panels_v3(source, 'day', diagnostics=last_diagnostics)
        constructors = {'original': lambda f: native.NumpyQuote(f, 'day', 'us').data,
                        'current': lambda f: native_quote_panels(f, 'day'),
                        'v3': construct_v3}
        case = QuoteConstructionV3Tests()
        def verify(actual, expected, source):
            if actual is None:
                raise AssertionError('Unexpected fallback')
            case.panels_equal(actual, expected)
            panels = list(actual.values())
            for axis in (0, 1):
                case.assertEqual(len({id(p.indices[axis].index_map) for p in panels}), stocks)
                for a, b in zip(panels, panels[1:]):
                    case.assertFalse(np.shares_memory(a.indices[axis].idx_list, b.indices[axis].idx_list))
            for panel, reference in zip(panels, expected.values()):
                for flag in ('c_contiguous', 'f_contiguous', 'owndata', 'writeable', 'aligned'):
                    case.assertEqual(getattr(panel.data.flags, flag), getattr(reference.data.flags, flag))
                    for ai, ei in zip(panel.indices, reference.indices):
                        case.assertEqual(getattr(ai.idx_list.flags, flag), getattr(ei.idx_list.flags, flag))
                case.assertEqual(panel.data.strides, reference.data.strides)
                case.assertFalse(np.shares_memory(panel.index.idx_list, source.index.levels[1].to_numpy(copy=False)))
                for column in source.columns:
                    case.assertFalse(np.shares_memory(panel.data, source[column].to_numpy(copy=False)))
            return {'panels': len(panels), 'rows': sum(len(p.data) for p in panels),
                    'bits_maps_metadata_flags_strides_equal': True, 'detached_mutable_buffers_maps': True}
        progress['phase'] = 'initial_parity_outside_timing'
        save()
        progress['first_call_seconds'] = {}
        for fixture, source in inputs.items():
            before = time.perf_counter()
            reference = constructors['original'](source)
            progress['first_call_seconds'][fixture] = {'original': time.perf_counter() - before}
            for name in ('current', 'v3'):
                before = time.perf_counter()
                actual = constructors[name](source)
                progress['first_call_seconds'][fixture][name] = time.perf_counter() - before
                verify(actual, reference, source)
            del reference
        for pair in range(pairs):
            for fixture, source in inputs.items():
                progress.update(phase='paired_complete_constructor', active_pair=pair, active_fixture=fixture)
                save()
                result, panels = {}, {}
                order = ('original', 'current', 'v3') if pair % 2 == 0 else ('v3', 'current', 'original')
                for name in order:
                    gc.collect()
                    before = time.perf_counter()
                    actual = constructors[name](source)
                    elapsed = time.perf_counter() - before
                    panels[name], result[name] = actual, {'seconds': elapsed}
                    if name == 'v3':
                        result[name]['cache'] = dict(last_diagnostics)
                for name in order:
                    result[name]['parity'] = verify(panels[name], panels['original'], source)
                progress['trials'].append({'pair': pair, 'fixture': fixture, 'shape': list(source.shape), 'results': result})
                save()
                del panels, actual
        progress['median_seconds'] = {fixture: {name: statistics.median(
            t['results'][name]['seconds'] for t in progress['trials'] if t['fixture'] == fixture)
            for name in constructors} for fixture in inputs}
        progress['source_sha256'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ('qlib_yahoo/quote_construction.py', 'qlib_yahoo/quote_construction_v2.py',
                         'qlib_yahoo/quote_construction_v3.py', 'tests/test_qlib_quote_construction_v3.py')}
        progress.update(status='completed', phase='completed',
            timing_boundary='All guards/frame projection, native numeric casting/constructors, bounded date-vector key/cache/clones, Index sorting and final np.take allocations. Fixture creation/parity/GC outside timing.',
            limitations='Constructor-only synthetic evidence; not full exchange preparation or job saving.')
        save()
        return progress
    except BaseException as exc:
        progress.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        save()
        raise
