"""Optional columnar serialization of reviewed native, aligned order metrics.

The caller seeds the schema with the original Pandas path's first nonempty
session. Unsupported layouts return NotImplemented without calling user getters
or importing Qlib; the caller must retain the original conversion as fallback.
"""

from __future__ import annotations

import sys
from collections import OrderedDict


def position_table(position, shared, equity, schema):
    """Serialize positions after the caller's existing reconciliation.

    This is an equivalent column builder, not a speculative fast path: native
    or custom getters run once in the original sorted traversal/order, and
    exceptions propagate without retry. Shared ledger fields/equity are supplied
    by the caller and never fetched or recomputed here. Python-list Arrow input
    retains from_pylist's valid NaNs, including insolvent-book weights.
    """
    import pyarrow as pa

    instruments, amounts, prices, values, weights = [], [], [], [], []
    for symbol in sorted(position.get_stock_list()) or [""]:
        amount = float(position.get_stock_amount(symbol)) if symbol else 0.0
        price = float(position.get_stock_price(symbol)) if symbol else None
        value = amount * price if symbol else 0.0
        instruments.append(symbol)
        amounts.append(amount)
        prices.append(price)
        values.append(value)
        weights.append(value / equity if equity > 0 else float("nan"))
    count = len(instruments)
    columns = {name: [value] * count for name, value in shared.items()}
    columns.update(instrument=instruments, amount=amounts, price=prices,
                   value=values, weight=weights)
    return pa.Table.from_pydict(columns, schema=schema)


def order_indicator_table(raw, day, schema):
    """Return an Arrow table using ``schema``, or ``NotImplemented``.

    Exact native types and plain backing arrays are required. Array indexes
    must agree in value *and order*, so Pandas' union/reindex semantics are never
    replaced. Native insertion order (including duplicate symbols) is retained.
    No session state is mutated and no writer is called.
    """
    native = sys.modules.get("qlib.backtest.high_performance_ds")
    index_data = sys.modules.get("qlib.utils.index_data")
    if native is None or index_data is None or schema is None:
        return NotImplemented
    if type(raw) is not vars(native).get("NumpyOrderIndicator"):
        return NotImplemented
    # Read backing dictionaries directly: no speculative custom .data/.index
    # property, conversion method, or getter may run before fallback.
    raw_dict = object.__getattribute__(raw, "__dict__")
    metrics = raw_dict.get("data")
    if type(metrics) not in (dict, OrderedDict) or not metrics:
        return NotImplemented

    import numpy as np
    import pandas as pd
    import pyarrow as pa

    if type(schema) is not pa.Schema or type(day) is not pd.Timestamp:
        return NotImplemented
    if day.tz is not None or pd.isna(day):
        return NotImplemented
    names = list(metrics)
    if any(type(name) is not str for name in names):
        return NotImplemented
    if schema.names != ["datetime", "instrument", *names]:
        return NotImplemented
    timestamp_type = schema.field(0).type
    if (not pa.types.is_timestamp(timestamp_type) or timestamp_type.tz is not None
            or timestamp_type.unit not in ("s", "ms", "us", "ns")
            or schema.field(1).type != pa.string()):
        return NotImplemented
    if any(field.type != pa.float64() for field in list(schema)[2:]):
        return NotImplemented
    # Use the seeded Pandas datetime resolution, never relabel nanosecond ticks
    # as milliseconds. Reject lossy/overflowing conversions before allocating
    # Arrow output so the caller retains its original Pandas cast/error path.
    try:
        converted_day = day.as_unit(timestamp_type.unit, round_ok=False)
    except (ValueError, OverflowError):
        return NotImplemented

    arrays = []
    symbols = None
    single_type = vars(index_data).get("SingleData")
    index_type = vars(index_data).get("Index")
    for metric in metrics.values():
        if type(metric) is not single_type:
            return NotImplemented
        backing = object.__getattribute__(metric, "__dict__")
        values, indices = backing.get("data"), backing.get("indices")
        if (type(values) is not np.ndarray or values.ndim != 1
                or values.dtype != np.dtype("float64")
                or type(indices) is not list or len(indices) != 1
                or type(indices[0]) is not index_type):
            return NotImplemented
        labels = object.__getattribute__(indices[0], "__dict__").get("idx_list")
        if (type(labels) is not np.ndarray or labels.ndim != 1
                or len(labels) != len(values) or not len(labels)):
            return NotImplemented
        if labels.dtype.kind == "O":
            # Native Pandas-derived indexes contain exact Python strings.
            # Prove this before any NumPy equality or Arrow conversion, since
            # arbitrary object elements can implement user callbacks.
            if not all(type(label) is str for label in labels):
                return NotImplemented
        elif labels.dtype.kind != "U":
            return NotImplemented
        if symbols is None:
            symbols = labels
        elif labels is not symbols and not np.array_equal(labels, symbols):
            return NotImplemented
        arrays.append(values)

    # Every guard completes before allocating Arrow output. from_pandas=True
    # preserves the old NaN->null conversion, including the original float bits
    # for nonnull values (signed zero, infinities and subnormal values).
    dates = np.full(len(symbols), converted_day.asm8.view(np.int64), dtype=np.int64).view(
        f"datetime64[{timestamp_type.unit}]")
    columns = [pa.array(dates, type=schema.field(0).type, from_pandas=True),
               pa.array(symbols, type=schema.field(1).type, from_pandas=True)]
    columns.extend(pa.array(values, type=pa.float64(), from_pandas=True) for values in arrays)
    return pa.Table.from_arrays(columns, schema=schema)
