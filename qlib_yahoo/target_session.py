"""Experimental complete target boundary: guards, packing, kernel and Orders.

Called only after native signal/weights/equity handling and the reviewed fused
query guard. Scratch failure restores the session cache and resumes the original
scalar loops on their existing diagnostic record, not the entire strategy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlib.backtest.decision import Order
from qlib.utils.index_data import Index, MultiData

from .fast_signed import FastSignedExchange, _reviewed_quote_binding
from . import target_session_kernel as kernel


_ORDER_INIT = Order.__init__
_ORDER_POST_INIT = Order.__post_init__


def _panel_is_plain(panel, start_ns):
    if type(panel) is not MultiData:
        return False
    backing = object.__getattribute__(panel, "__dict__")
    data, indices = backing.get("data"), backing.get("indices")
    if (type(data) is not np.ndarray or data.ndim != 2 or data.dtype != np.float64
            or type(indices) is not list or len(indices) != 2):
        return False
    for axis, index in enumerate(indices):
        if type(index) is not Index:
            return False
        raw = object.__getattribute__(index, "__dict__")
        labels, mapping = raw.get("idx_list"), raw.get("index_map")
        if (type(labels) is not np.ndarray or labels.ndim != 1
                or len(labels) != data.shape[axis] or type(mapping) is not dict):
            return False
        if axis == 0 and labels.dtype != np.dtype("datetime64[ns]"):
            return False
        if axis == 0:
            # Check the queried map value before _session_row could compare or
            # use it as an ndarray index. Do not invoke custom __index__ values
            # speculatively ahead of an earlier native target error.
            location = mapping.get(start_ns)
            if location is not None and (type(location) is not int or not 0 <= location < len(labels)):
                return False
        else:
            if labels.dtype.kind not in ("U", "O"):
                return False
            if labels.dtype.kind == "O" and any(type(label) is not str for label in labels):
                return False
            if any(type(key) not in (str, np.str_) or type(value) is not int or not 0 <= value < len(labels)
                   for key, value in mapping.items()):
                return False
    return True


def try_target_orders(exchange, start, end, target_weights, holdings, equity):
    """Return (targets, reasons, native Orders), or NotImplemented atomically.

    No original diagnostic/order/holding mutation is performed. Plain panels
    and primitive mappings exclude speculative user conversions/query callbacks.
    Missing zero-bar default rows are unavailable, not fabricated prices.
    """
    if (type(exchange) is not FastSignedExchange or not _reviewed_quote_binding(exchange.quote)
            or type(exchange.quote.data) is not dict or type(exchange._session_rows) is not dict
            or type(target_weights) is not dict or type(holdings) is not dict
            or Order.__init__ is not _ORDER_INIT or Order.__post_init__ is not _ORDER_POST_INIT
            or type(start) is not pd.Timestamp or type(end) is not pd.Timestamp
            or not exchange.supports_session_rows):
        return NotImplemented
    names = list(target_weights)
    if (any(type(name) is not str or type(weight) is not float for name, weight in target_weights.items())
            or any(type(name) is not str or type(amount) not in (int, float)
                   for name, amount in holdings.items())):
        return NotImplemented
    for name in names:
        row = exchange._session_rows.get(name)
        if row is not None and (type(row) is not tuple or len(row) != 9
                or any(type(value) not in (float, np.float64, bool, np.bool_, type(None)) for value in row)
                or any(type(value) not in (bool, np.bool_) for value in row[5:])):
            return NotImplemented
    panels = [exchange.quote.data.get(name) for name in names]
    if any(panel is not None and not _panel_is_plain(panel, exchange._session_start_ns) for panel in panels):
        return NotImplemented
    cached = exchange._session_rows
    before = cached.copy()
    succeeded = False
    try:
        prices, factors, tradable = [], [], []
        for name, panel in zip(names, panels):
            row = exchange._session_row(name, start, end)
            if row is None:
                # The default native same-day shortcut requires start exactly;
                # a weekend range with zero observed bars is also suspended.
                absent = panel is None
                if panel is not None:
                    index = panel.indices[0]
                    if exchange._session_raw:
                        absent = index.index_map.get(exchange._session_start_ns) is None
                    else:
                        labels = index.idx_list
                        left = int(labels.searchsorted(exchange._session_start_ns, side="left"))
                        right = int(labels.searchsorted(exchange._session_end_ns, side="right"))
                        absent = left == right
                if not absent:
                    return NotImplemented  # Multi-bar/unsupported row: native owns it.
                prices.append(float("nan"))
                factors.append(float("nan"))
                tradable.append(False)
            else:
                if any(type(value) not in (float, np.float64, type(None)) for value in row[:2]):
                    return NotImplemented
                prices.append(row[0])
                factors.append(row[1])
                tradable.append(row[6])
        result = kernel.run_target_session(
            np.array(list(target_weights.values()), dtype=np.float64),
            np.array([holdings.get(name, 0) for name in names], dtype=np.float64),
            np.array(prices, dtype=np.float64), np.array(factors, dtype=np.float64),
            np.array(tradable, dtype=np.bool_), equity=equity, compiled=True)
        if result is None:
            return NotImplemented
        targets = {name: int(amount) for name, amount, active in
                   zip(names, result.targets.tolist(), result.active.tolist()) if active}
        reasons = {name: "unavailable_for_trade" if reason == kernel.UNAVAILABLE
                   else "missing_or_invalid_open" for name, reason in
                   zip(names, result.reasons.tolist()) if reason != kernel.ACCEPTED}
        orders = [Order(names[index], amount, Order.BUY if direction else Order.SELL, start, end)
                  for descriptors in (result.reductions, result.increases)
                  for index, amount, direction in descriptors.tolist()]
        succeeded = True
        return targets, reasons, orders
    finally:
        if not succeeded:
            cached.clear()
            cached.update(before)
