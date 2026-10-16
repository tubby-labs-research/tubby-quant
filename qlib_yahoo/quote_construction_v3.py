# Copyright (c) Microsoft Corporation and Tubby Labs contributors.
# Licensed under the MIT License.
"""Opt-in, bounded native date-index reuse experiment.

The cache lives only during one constructor and is seeded after a complete
native panel construction and sort. Byte-identical date vectors reuse native
Index(existing) hooks, with detached label arrays/maps for every panel. Native
numeric promotion, casting and the final np.take data copy remain unchanged.
No direct float64 projection, global cache, mutable cross-panel sharing, or
default activation is provided.
"""
from collections import OrderedDict
import numpy as np
import pandas as pd
import sys
from qlib.utils import index_data as native_index, time as native_time
from qlib.utils.index_data import Index, IndexData, LocIndexer, MultiData
from qlib.utils.time import Freq
from qlib_yahoo.quote_construction_v2 import _BINDINGS, _NUMPY, _FREQ_PARSE, _FREQ_TIMEDELTA

_ABSENT = object()
_FIELDS = [(cls, name, vars(cls).get(name, _ABSENT)) for cls, names in (
    (Index, ('idx_list', 'index_map', '_is_sorted')),
    (IndexData, ('data', 'indices', 'ndim')),
    (MultiData, ('data', 'indices', 'ndim')),
) for name in names]
_CACHE_CAP = 32
_GETERR = np.geterr
_MODULE_BINDINGS = {name: getattr(native_index, name) for name in
                    ('Index', 'IndexData', 'LocIndexer', 'MultiData', 'np', 'pd')}
_ACCESS_BINDINGS = [(cls, name, getattr(cls, name)) for cls in (Index, IndexData, MultiData)
                    for name in ('__getattribute__', '__setattr__')]


def _detached_index(template):
    # Use the public native copy hook; detach both writable containers before
    # exposing the new panel. Only immutable datetime keys/integers are shared.
    clone = Index(template)
    clone.idx_list = template.idx_list.copy()
    clone.index_map = template.index_map.copy()
    return clone


def native_quote_panels_v3(quote_df, freq, *, diagnostics=None):
    """Return native panels or decline before construction on custom inputs."""
    if (type(quote_df) is not pd.DataFrame or type(freq) is not str
            or (diagnostics is not None and (type(diagnostics) is not dict or diagnostics))
            or any(getattr(cls, name) is not original for cls, name, original in _ACCESS_BINDINGS)
            or any(getattr(native_index, name) is not original for name, original in _MODULE_BINDINGS.items())
            or native_time.Freq is not Freq or native_time.pd is not pd
            or IndexData.loc_idx_cls is not LocIndexer or MultiData.loc_idx_cls is not LocIndexer
            or any(getattr(cls, name) is not original for cls, name, original in _BINDINGS)
            or any(vars(cls).get(name, _ABSENT) is not original for cls, name, original in _FIELDS)
            or any(getattr(np, name) is not original for name, original in _NUMPY.items())
            or np.geterr is not _GETERR
            or Freq.parse is not _FREQ_PARSE or Freq.get_timedelta is not _FREQ_TIMEDELTA):
        return None
    # NumPy call/log policies can invoke arbitrary code during conversion. Such
    # code may change bindings/source mid-constructor; only native fallback owns
    # that callback order. Decline before metadata reads or scratch allocation.
    policy = np.geterr()
    if policy['invalid'] != 'ignore' or any(value in ('call', 'log') for value in policy.values()):
        return None
    index, columns = quote_df.index, quote_df.columns
    if (type(index) is not pd.MultiIndex or index.nlevels != 2
            or index.names != ['instrument', 'datetime']
            or type(columns) is not pd.Index or not len(columns)
            or not columns.is_unique or not all(type(c) is str for c in columns)):
        return None
    instruments, dates = index.levels
    if (type(instruments) is not pd.Index or instruments.dtype != np.dtype('object')
            or not all(type(s) is str for s in instruments)
            or type(dates) is not pd.DatetimeIndex
            or dates.dtype != np.dtype('datetime64[ns]') or dates.hasnans
            or any(np.any(codes < 0) for codes in index.codes)
            or not index.is_monotonic_increasing or not index.is_unique):
        return None
    dtypes = list(quote_df.dtypes)
    if any(not isinstance(dtype, np.dtype) or not dtype.isnative
           or dtype.kind not in 'biuf' or dtype.itemsize > 8 for dtype in dtypes):
        return None
    try:
        _, unit = Freq.parse(freq)
        if unit not in Freq.SUPPORT_CAL_LIST:
            return None
        Freq.get_timedelta(1, unit)
    except (TypeError, ValueError, KeyError):
        return None
    if not len(index):
        return {}
    kinds = {dtype.kind for dtype in dtypes}
    block_dtype = np.dtype(object) if 'b' in kinds and len(kinds) > 1 else np.result_type(*dtypes)
    arrays = [quote_df[column].to_numpy(copy=False) for column in columns]
    codes = index.codes[0]
    boundaries = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    starts, stops = np.r_[0, boundaries], np.r_[boundaries, len(index)]
    date_values = dates.to_numpy(copy=False)
    result, cache = {}, OrderedDict()
    hits = misses = peak_entries = peak_key_bytes = peak_map_container_bytes = 0
    for start, stop in zip(starts, stops):
        block = np.empty((stop - start, len(columns)), dtype=block_dtype)
        for column, values in enumerate(arrays):
            block[:, column] = values[start:stop]
        labels = date_values[index.codes[1][start:stop]]
        key = labels.tobytes()  # ns dtype fixed by guards; includes every hole.
        template = cache.get(key)
        if template is None:
            misses += 1
            panel = MultiData(block, labels, columns)
            panel.sort_index()
            # First native cast/validation/sort must succeed before seeding.
            cache[key] = _detached_index(panel.indices[0])
            if len(cache) > _CACHE_CAP:
                cache.popitem(last=False)
        else:
            hits += 1
            # The detached copy is evaluated before native casting. This is
            # safe only in the guarded invalid=ignore primitive cast domain;
            # callback/error policies decline before all scratch allocation.
            panel = MultiData(block, _detached_index(template), columns)
            panel.data = np.take(panel.data, np.arange(len(labels)), axis=0)
            cache.move_to_end(key)
        peak_entries = max(peak_entries, len(cache))
        if diagnostics is not None:
            peak_key_bytes = max(peak_key_bytes, sum(len(k) for k in cache))
            peak_map_container_bytes = max(peak_map_container_bytes,
                                          sum(sys.getsizeof(t.index_map) for t in cache.values()))
        result[instruments[codes[start]]] = panel
    if diagnostics is not None:
        diagnostics.update(hits=hits, misses=misses, cap=_CACHE_CAP,
                           peak_entries=peak_entries, peak_key_bytes=peak_key_bytes,
                           peak_cached_label_bytes=peak_key_bytes,
                           peak_map_container_bytes=peak_map_container_bytes,
                           memory_scope='keys, label buffers and dict containers; excludes immutable scalar objects and Python Index wrappers',
                           cache_scope='per-constructor date bytes and detached native Index templates')
    return result
