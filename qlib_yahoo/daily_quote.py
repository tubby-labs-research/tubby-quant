"""A narrow daily scalar lookup for Qlib's supported ``quote_cls`` hook.

Timezone-naive daily windows containing one observed bar take the direct array
path when aggregation is requested. Exact single-day queries preserve the native
scalar shortcut (which ignores aggregation). All other queries retain the
installed NumpyQuote behavior, including its aggregation and
exception semantics.  No second copy of the quote panel or index is maintained.
"""

from datetime import time
from functools import lru_cache

import numpy as np
import pandas as pd
from qlib.backtest.high_performance_ds import BaseQuote, NumpyQuote
from qlib.utils.time import Freq

from .quote_construction import native_quote_panels


class DailyQuote(NumpyQuote):
    """NumpyQuote with a direct lookup for one daily bar."""

    def __init__(self, quote_df, freq, region="cn", *, _index_reuse=False, _numeric_projection=False):
        # Tombstone for archived constructor configs: never reinterpret V4 as V3.
        if _numeric_projection is not False:
            raise ValueError("numeric projection (V4) is unsupported: the experiment was retired")
        if type(_index_reuse) is not bool:
            raise TypeError("quote construction options must be booleans")
        self.index_reuse_stats = None
        if _index_reuse:
            from .quote_construction_v3 import native_quote_panels_v3 as construct
            diagnostics = {}
            panels = construct(quote_df, freq, diagnostics=diagnostics)
            self.index_reuse_stats = {"requested": True, "fast": panels is not None,
                                      "native_fallback": panels is None, **diagnostics}
        else:
            panels = native_quote_panels(quote_df, freq)
        if panels is None:
            super().__init__(quote_df, freq, region=region)
        else:
            # Supported quote_cls hook: only assembly changes. Panel conversion,
            # sorting, accessors and all unsupported constructor errors are native.
            BaseQuote.__init__(self, quote_df, freq)
            self.data = panels
            _, unit = Freq.parse(freq)
            self.freq = Freq.get_timedelta(1, unit)
            self.region = region
        self._is_daily = Freq.parse(freq) == (1, "day") and all(
            panel.index.idx_list.dtype == "datetime64[ns]" for panel in self.data.values()
        )
        self._last_row = None
        self._interval_row = lru_cache(maxsize=512)(self._interval_row)
        # Match NumpyQuote's bounded result cache without sharing entries or
        # retaining quote instances in a class-level cache.
        # Retain the constructor-proven binding so guarded consumers can tell
        # this cache from a later custom callback without trusting __wrapped__.
        self._native_get_data_method = self.get_data
        self.get_data = lru_cache(maxsize=512)(self._native_get_data_method)
        self._native_get_data_cache = self.get_data

    def _interval_row(self, stock_id, start_time, end_time):
        """Locate one bar in the closed interval, sharing the native index."""
        index = self.data[stock_id].index.idx_list
        start = start_time.as_unit("ns").asm8
        end = end_time.as_unit("ns").asm8
        row = int(index.searchsorted(start, side="left"))
        if row == len(index) or index[row] > end:
            return None
        if row + 1 < len(index) and index[row + 1] <= end:
            return None
        return row

    def get_data(self, stock_id, start_time, end_time, field, method=None):
        if (
            self._is_daily
            and type(start_time) is pd.Timestamp
            and type(end_time) is pd.Timestamp
            and start_time.tz is None
            and end_time.tz is None
            and start_time.time() == time.min
            and start_time.nanosecond == 0
            and stock_id in self.data
            and self.region in ("cn", "tw", "us")
        ):
            panel = self.data[stock_id]
            if start_time.date() == end_time.date():
                try:
                    if self._last_row is not None and self._last_row[0] == stock_id and self._last_row[1] == start_time:
                        row = self._last_row[2]
                    else:
                        row = panel.indices[0].index_map[start_time.as_unit("ns").asm8]
                        self._last_row = (stock_id, start_time, row)
                    column = panel.indices[1].index_map[field]
                except KeyError:
                    return None
                # The parent single-value branch ignores method, even for sum/all.
                return panel.data[row, column]
            if method is not None and end_time >= start_time:
                row = self._interval_row(stock_id, start_time, end_time)
                if row is not None:
                    # Generic native slicing raises for a missing field, unlike
                    # its single-day shortcut. Do not catch this KeyError.
                    column = panel.indices[1].index_map[field]
                    value = panel.data[row, column]
                    if method == "sum":
                        return np.nansum(panel.data[row:row + 1, column])
                    if method == "mean":
                        return np.nanmean(panel.data[row:row + 1, column])
                    if method == "all":
                        return np.bool_(value)
                    if method == "last":
                        return value
                    if method == "ts_data_last":
                        return None if np.isnan(value) else value
                    raise ValueError(f"{method} is not supported")
        return super().get_data(stock_id, start_time, end_time, field, method)
