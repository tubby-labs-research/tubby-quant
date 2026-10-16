# Copyright (c) Microsoft Corporation and Tubby Labs contributors.
# Licensed under the MIT License.
"""Unwired quote-construction experiment using native constructors and Index.sort.

Only the identity row permutation's redundant np.take is omitted. Each native
MultiData owns its float64 data, both label arrays and both index maps. Mixed
boolean/numeric projection retains Pandas' object promotion and native casting:
direct float64 projection is deliberately not enabled without error-policy and
signalling-NaN conversion proof. Unsupported/custom bindings return None before
panel construction; the caller alone owns the original constructor fallback.
"""
import numpy as np
import pandas as pd

from qlib.utils.index_data import Index, IndexData, LocIndexer, MultiData
from qlib.utils.time import Freq

_BINDINGS = [(cls, name, getattr(cls, name)) for cls, names in (
    (MultiData, ('__init__', '__getattribute__')),
    (IndexData, ('__init__', 'sort_index')),
    (Index, ('__init__', 'sort', '__getattribute__', '__getitem__', '__len__')),
    (LocIndexer, ('proc_idx_l',)),
    (pd.DataFrame, ('__getattribute__', '__getitem__', 'dtypes', 'index', 'columns')),
    (pd.Series, ('to_numpy',)),
    (pd.Index, ('__getattribute__', '__iter__', 'is_unique', 'to_numpy', 'dtype')),
    (pd.MultiIndex, ('__getattribute__', 'is_unique', 'is_monotonic_increasing',
                     'levels', 'codes', 'names', 'nlevels')),
    (pd.DatetimeIndex, ('__getattribute__', 'to_numpy', 'hasnans', 'dtype')),
) for name in names]
_NUMPY = {name: getattr(np, name) for name in ('empty', 'result_type', 'flatnonzero', 'take',
                                             'argsort', 'array', 'array_equal', 'arange', 'any', 'r_')}
_FREQ_PARSE, _FREQ_TIMEDELTA = Freq.parse, Freq.get_timedelta


def native_quote_panels_v2(quote_df, freq):
    """Return detached native panels, or None with no caller/source mutations."""
    if (type(quote_df) is not pd.DataFrame or type(freq) is not str
            or any(getattr(cls, name) is not original for cls, name, original in _BINDINGS)
            or any(getattr(np, name) is not original for name, original in _NUMPY.items())
            or Freq.parse is not _FREQ_PARSE
            or Freq.get_timedelta is not _FREQ_TIMEDELTA):
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
    result = {}
    for start, stop in zip(starts, stops):
        block = np.empty((stop - start, len(columns)), dtype=block_dtype)
        for column, values in enumerate(arrays):
            block[:, column] = values[start:stop]
        panel = MultiData(block, date_values[index.codes[1][start:stop]], columns)
        # Native sort supplies its exact label dtype/map/_is_sorted metadata.
        # Source guards prove these dates strictly increasing, so argsort is an
        # identity permutation. Retain the already detached C float64 data.
        panel.indices[0], permutation = panel.indices[0].sort()
        if not np.array_equal(permutation, np.arange(len(permutation))):
            # Defensive native behavior if an assumption changes in upstream.
            panel.data = np.take(panel.data, permutation, axis=0)
        result[instruments[codes[start]]] = panel
    return result
