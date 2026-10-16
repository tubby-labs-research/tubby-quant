"""Guarded candidate for native Qlib quote-panel construction.

DailyQuote uses this helper. Unsupported inputs return ``None`` so its
caller can run the unchanged NumpyQuote constructor. No native index machinery
is bypassed: each panel uses MultiData's conversion and public sort_index.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from qlib.utils.index_data import MultiData


def native_quote_panels(quote_df: pd.DataFrame, freq: str) -> dict[str, MultiData] | None:
    """Construct native panels without Pandas' groupby/droplevel copies.

    Column arrays are projected once without coercing the whole frame. At most
    one stock's common-dtype block is temporary, in addition to the copies made
    by native MultiData and sorting. Every original row, NaN and field survives.
    """
    from qlib.utils.index_data import MultiData
    from qlib.utils.time import Freq

    if type(quote_df) is not pd.DataFrame or type(freq) is not str:
        return None
    index, columns = quote_df.index, quote_df.columns
    if (type(index) is not pd.MultiIndex or index.nlevels != 2
            or index.names != ["instrument", "datetime"]
            or type(columns) is not pd.Index or not len(columns)
            or not columns.is_unique or not all(type(c) is str for c in columns)):
        return None
    instruments, dates = index.levels
    if (type(instruments) is not pd.Index or instruments.dtype != np.dtype("object")
            or not all(type(s) is str for s in instruments)
            or type(dates) is not pd.DatetimeIndex
            or dates.dtype != np.dtype("datetime64[ns]") or dates.hasnans
            or any(np.any(codes < 0) for codes in index.codes)
            or not index.is_monotonic_increasing or not index.is_unique):
        return None
    dtypes = list(quote_df.dtypes)
    if any(not isinstance(dtype, np.dtype) or not dtype.isnative
           or dtype.kind not in "biuf" or dtype.itemsize > 8 for dtype in dtypes):
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

    # Pandas .values uses object for mixed boolean/numeric frames; for ordinary
    # NumPy numeric dtypes it uses NumPy's common dtype. Preserve that promotion
    # before letting native MultiData cast to float64 (e.g. int8 + float16).
    kinds = {dtype.kind for dtype in dtypes}
    block_dtype = np.dtype(object) if "b" in kinds and len(kinds) > 1 else np.result_type(*dtypes)
    arrays = [quote_df[column].to_numpy(copy=False) for column in columns]
    codes = index.codes[0]
    boundaries = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, len(index)]
    date_values = dates.to_numpy(copy=False)
    result = {}
    for start, stop in zip(starts, stops):
        block = np.empty((stop - start, len(columns)), dtype=block_dtype)
        for column, values in enumerate(arrays):
            block[:, column] = values[start:stop]
        panel = MultiData(block, date_values[index.codes[1][start:stop]], columns)
        panel.sort_index()
        result[instruments[codes[start]]] = panel
    return result
