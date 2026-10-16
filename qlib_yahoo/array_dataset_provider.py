"""Guarded dataset construction hook; expression evaluation remains native Qlib.

This module is intentionally imported only by Qlib's provider configuration.
It replaces per-instrument Pandas construction for ordinary float32 calendar
series, without evaluating expressions in bulk or extending their output rows.
"""

import numpy as np
import pandas as pd

from qlib.config import C
from qlib.data.cache import DiskDatasetCache
from qlib.data.data import (
    Cal, DatasetProvider, ExpressionD, LocalCalendarProvider,
    LocalDatasetProvider, LocalExpressionProvider,
)
from qlib.utils import normalize_cache_fields


class ArrayDatasetProvider(LocalDatasetProvider):
    """Use native expressions with one array/DataFrame construction per dataset.

    Configure through ``qlib.init(dataset_provider={"class":
    "ArrayDatasetProvider", "module_path": "qlib_yahoo.array_dataset_provider"})``.
    No runtime configuration is changed merely by importing this module.
    """

    @staticmethod
    def dataset_processor(instruments_d, column_names, start_time, end_time,
                          freq, inst_processors=None):
        processors = [] if inst_processors is None else inst_processors
        expression_provider = getattr(ExpressionD, "_provider", None)
        calendar_provider = getattr(Cal, "_provider", None)
        if (
            type(instruments_d) not in (list, tuple)
            or any(type(inst) is not str for inst in instruments_d)
            or len(set(instruments_d)) != len(instruments_d)
            or processors
            or C.get_kernels(freq) != 1
            or type(expression_provider) is not LocalExpressionProvider
            or not expression_provider.time2idx
            or type(calendar_provider) is not LocalCalendarProvider
            or not column_names
        ):
            return DatasetProvider.dataset_processor(
                instruments_d, column_names, start_time, end_time, freq,
                inst_processors=processors,
            )

        fields = normalize_cache_fields(column_names)
        captured = {}
        calendar = None
        has_fallback = False
        # Keep native evaluation order, even though output stocks are sorted.
        for inst in instruments_d:
            obj = {field: ExpressionD.expression(inst, field, start_time, end_time, freq)
                   for field in fields}
            first = obj[fields[0]]
            compatible = (
                type(first) is pd.Series
                and type(first.index) is pd.RangeIndex
                and first.index.step == 1
                and all(type(series) is pd.Series
                        and series.dtype == np.dtype(np.float32)
                        and type(series.index) is pd.RangeIndex
                        and series.index.step == 1
                        and series.index.equals(first.index)
                        and series.to_numpy(copy=False).flags.c_contiguous
                        for series in obj.values())
            )
            if compatible and len(first):
                if calendar is None:
                    calendar = Cal.calendar(freq=freq)
                # Native calendar indexing allows negative integer indexes, but
                # this optimized calendar coding deliberately does not.
                compatible = first.index.start >= 0 and first.index.stop <= len(calendar)
            if compatible:
                if len(first):
                    captured[inst] = (first.index, tuple(
                        series.to_numpy(copy=False) for series in obj.values()))
            else:
                # Use the values already evaluated: retrying the whole provider
                # would evaluate custom expressions twice and hide exceptions.
                frame = pd.DataFrame(obj)
                if not frame.empty and not np.issubdtype(frame.index.dtype, np.dtype("M")):
                    if calendar is None:
                        calendar = Cal.calendar(freq=freq)
                    frame.index = calendar[frame.index.values.astype(int)]
                frame.index.names = ["datetime"]
                if len(frame):
                    captured[inst] = frame
                    has_fallback = True

        if not captured:
            return pd.DataFrame(
                index=pd.MultiIndex.from_arrays([[], []], names=("instrument", "datetime")),
                columns=column_names, dtype=np.float32,
            )

        symbols = sorted(captured)
        if has_fallback:
            frames = {}
            for inst in symbols:
                item = captured[inst]
                if isinstance(item, pd.DataFrame):
                    frames[inst] = item
                else:
                    idx, arrays = item
                    frames[inst] = pd.DataFrame(
                        dict(zip(fields, arrays)),
                        index=pd.Index(calendar[idx.start:idx.stop], name="datetime"),
                    )
            data = pd.concat(frames, names=["instrument"], sort=False)
        else:
            lengths = [len(captured[inst][0]) for inst in symbols]
            rows = sum(lengths)
            values = np.empty((rows, len(fields)), dtype=np.float32)
            calendar_indexes = np.empty(rows, dtype=np.int64)
            offset = 0
            for inst, length in zip(symbols, lengths):
                idx, arrays = captured[inst]
                end = offset + length
                calendar_indexes[offset:end] = np.arange(idx.start, idx.stop)
                for col, array in enumerate(arrays):
                    values[offset:end, col] = array
                offset = end
            used_indexes, date_codes = np.unique(calendar_indexes, return_inverse=True)
            index = pd.MultiIndex(
                levels=[pd.Index(symbols), pd.Index(calendar[used_indexes])],
                codes=[np.repeat(np.arange(len(symbols)), lengths), date_codes],
                names=["instrument", "datetime"], verify_integrity=False,
            )
            data = pd.DataFrame(values, index=index, columns=fields, copy=False)
        return DiskDatasetCache.cache_to_origin_data(data, column_names)
