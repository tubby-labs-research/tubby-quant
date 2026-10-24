"""Guarded construction of signed report holding series from sparse positions.

This helper changes only series preparation. Ranking and chart aggregation
remain the caller's responsibility, with their original Pandas operations.
"""

from typing import Any


def signed_holding_series(positions: Any, frame: Any) -> dict[str, Any] | None:
    """Return native-equivalent signed weights, or ``None`` for native fallback.

    Supported inputs have ordinary float64 marks/equity, object or string-dtype
    string/null symbols, naive datetime64 dates of any resolution (pandas 3
    infers microseconds where pandas 2 used nanoseconds), and at most one mark
    per stock/session.
    No observations are aggregated, and no input is modified. Output Series
    share a single C-contiguous allocation; each stock's row is contiguous.
    """
    import numpy as np
    import pandas as pd

    if (
        type(positions) is not pd.DataFrame
        or type(frame) is not pd.DataFrame
        or not positions.columns.is_unique
        or not frame.columns.is_unique
        or not {"datetime", "instrument", "value"}.issubset(positions.columns)
        or "account" not in frame.columns
        or type(frame.index) is not pd.DatetimeIndex
        or frame.index.tz is not None
        or not _naive_datetime(frame.index.dtype)
        or frame.index.hasnans
        or not frame.index.is_unique
        or frame["account"].dtype != np.dtype(np.float64)
        or positions["value"].dtype != np.dtype(np.float64)
        or not _naive_datetime(positions["datetime"].dtype)
        or not (positions["instrument"].dtype == np.dtype(object)
                or pd.api.types.is_string_dtype(positions["instrument"].dtype))
    ):
        return None

    selected = positions["instrument"].notna().to_numpy()
    instruments = positions["instrument"].to_numpy(dtype=object, copy=False)[selected]
    if any(type(symbol) is not str for symbol in instruments):
        return None
    dates = pd.DatetimeIndex(positions["datetime"].to_numpy(copy=False)[selected])
    if dates.hasnans:
        return None
    if dates.dtype != frame.index.dtype:
        # Same instants at another resolution still identify the same sessions.
        dates = dates.as_unit(frame.index.unit)
    date_codes = frame.index.get_indexer(dates)
    if np.any(date_codes < 0):
        return None
    symbol_codes, symbols = pd.factorize(instruments, sort=True)
    sessions = len(frame.index)
    # A pivot would silently aggregate duplicates. Check sparse cell identities
    # first so all unsupported inputs can use the caller's original behavior.
    cells = symbol_codes.astype(np.int64, copy=False) * sessions + date_codes
    if not pd.Index(cells).is_unique:
        return None

    values = np.zeros((len(symbols), sessions), dtype=np.float64, order="C")
    values[symbol_codes, date_codes] = positions["value"].to_numpy(copy=False)[selected]
    account = frame["account"].to_numpy(copy=False)
    positive = account > 0
    # Keep division as a separate float64 operation, matching the native
    # positive-equity Series division. Missing holdings were initialized to 0;
    # explicit NaN marks stay NaN, while undefined equity overrides all marks.
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(values, account[np.newaxis, :], out=values,
                  where=positive[np.newaxis, :])
    values[:, ~positive] = np.nan
    return {str(symbol): pd.Series(values[row], index=frame.index, dtype=float, copy=False)
            for row, symbol in enumerate(symbols)}


def _naive_datetime(dtype) -> bool:
    """A plain numpy datetime64 of any resolution; timezone-aware extension dtypes are refused."""
    import numpy as np

    return isinstance(dtype, np.dtype) and dtype.kind == "M"

