"""Vectorized FASTEXPR operators on (dates, instruments) matrices.

Conventions, applied uniformly:

* values are float32 matrices; NaN means "no value" and propagates through
  arithmetic; division by zero and other undefined results are NaN, never inf;
* time-series operators look back only (row t uses rows t-d+1..t) and need a
  full window of values: a NaN inside the window gives NaN;
* cross-sectional operators use, on each date, only instruments inside the
  simulation universe (the ``members`` mask); others are NaN;
* comparisons and logical operators return 1 or 0, and NaN when an input is NaN.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from qlib_yahoo.alphalab.fastexpr import ExpressionError

F32 = np.float32


def _clean(values: Any) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).astype(F32)
    # Check finiteness after the float32 cast: float64 values beyond the float32
    # range (exp(100), volume^7) would otherwise survive as inf.
    return np.where(np.isfinite(out), out, np.float32(np.nan))


def _window(value: Any, name: str, low: int = 1, high: int = 1000) -> int:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not float(value).is_integer():
        raise ExpressionError(f"{name}: the window must be a whole number of days")
    days = int(value)
    if not low <= days <= high:
        raise ExpressionError(f"{name}: the window must be between {low} and {high} days")
    return days


def _frame(x: np.ndarray):
    import pandas as pd

    return pd.DataFrame(np.asarray(x, dtype=np.float64))


def row_mean(values: np.ndarray) -> np.ndarray:
    """Mean of each row over non-NaN entries (NaN for an empty row), without warnings."""
    present = ~np.isnan(values)
    count = present.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, np.where(present, values, 0.0).sum(axis=1, keepdims=True) / count, np.nan)


def row_std(values: np.ndarray) -> np.ndarray:
    """Population standard deviation of each row over non-NaN entries."""
    mean = row_mean(values)
    with np.errstate(invalid="ignore"):
        return np.sqrt(row_mean((values - mean) ** 2))


# ----------------------------------------------------------------- time series
def ts_delay(x, d):
    d = _window(d, "ts_delay", 0)
    out = np.full_like(x, np.nan)
    if d == 0:
        return x.copy()
    out[d:] = x[:-d]
    return out


def ts_delta(x, d):
    return _clean(x.astype(np.float64) - ts_delay(x, _window(d, "ts_delta")).astype(np.float64))


def _rolling(x, d, name, method, **kwargs):
    d = _window(d, name, 2 if method in ("std", "var", "skew", "kurt", "corr", "cov") else 1)
    rolled = getattr(_frame(x).rolling(d, min_periods=d), method)(**kwargs)
    return _clean(rolled.to_numpy())


def ts_mean(x, d):
    return _rolling(x, d, "ts_mean", "mean")


def ts_sum(x, d):
    return _rolling(x, d, "ts_sum", "sum")


def ts_std_dev(x, d):
    return _rolling(x, d, "ts_std_dev", "std", ddof=1)


def ts_max(x, d):
    return _rolling(x, d, "ts_max", "max")


def ts_min(x, d):
    return _rolling(x, d, "ts_min", "min")


def ts_skewness(x, d):
    return _rolling(x, max(_window(d, "ts_skewness", 3), 3), "ts_skewness", "skew")


def ts_kurtosis(x, d):
    return _rolling(x, max(_window(d, "ts_kurtosis", 4), 4), "ts_kurtosis", "kurt")


def ts_rank(x, d, constant=0):
    """Where today's value falls among the last d values, from 0 (lowest) to 1 (highest)."""
    d = _window(d, "ts_rank", 2)
    ranks = _frame(x).rolling(d, min_periods=d).rank(pct=True).to_numpy()
    # pandas pct rank is in (0, 1]; map to [0, 1] like the cross-sectional rank
    return _clean((ranks * d - 1) / (d - 1) + float(constant))


def ts_zscore(x, d):
    mean, std = ts_mean(x, d).astype(np.float64), ts_std_dev(x, d).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return _clean((x - mean) / std)


def ts_ir(x, d):
    with np.errstate(invalid="ignore", divide="ignore"):
        return _clean(ts_mean(x, d).astype(np.float64) / ts_std_dev(x, d).astype(np.float64))


def ts_product(x, d):
    d = _window(d, "ts_product")
    with np.errstate(divide="ignore"):
        logs = _frame(np.log(np.abs(np.asarray(x, dtype=np.float64)))).rolling(d, min_periods=d).sum().to_numpy()
    negatives = _frame((np.asarray(x) < 0).astype(np.float64)).rolling(d, min_periods=d).sum().to_numpy()
    zeros = _frame((np.asarray(x) == 0).astype(np.float64)).rolling(d, min_periods=d).sum().to_numpy()
    with np.errstate(over="ignore", invalid="ignore"):
        product = np.exp(logs) * np.where(np.mod(negatives, 2) == 1, -1.0, 1.0)
    return _clean(np.where(zeros > 0, 0.0, product))


_ARG_CHUNK_CELLS = 8_000_000  # float64 cells per temporary window block (64 MB)


def _arg(x, d, name, pick):
    d = _window(d, name, 1)
    values = np.asarray(x, dtype=np.float64)
    out = np.full(values.shape, np.nan)
    if len(values) >= d:
        windows = np.lib.stride_tricks.sliding_window_view(values, d, axis=0)  # (T-d+1, N, d) view
        fill = -np.inf if pick == "max" else np.inf
        choose = np.argmax if pick == "max" else np.argmin
        # One (rows, N, d) temporary at a time: the whole panel is gigabytes for TOP3000 and d=250.
        step = max(1, _ARG_CHUNK_CELLS // max(1, values.shape[1] * d))
        for start in range(0, len(windows), step):
            block = windows[start:start + step]
            missing = np.isnan(block)
            position = choose(np.where(missing, fill, block)[:, :, ::-1], axis=2)  # days since; 0 = today
            out[d - 1 + start:d - 1 + start + len(block)] = np.where(missing.any(axis=2), np.nan, position)
    return _clean(out)


def ts_arg_max(x, d):
    return _arg(x, d, "ts_arg_max", "max")


def ts_arg_min(x, d):
    return _arg(x, d, "ts_arg_min", "min")


def ts_corr(x, y, d):
    d = _window(d, "ts_corr", 2)
    return _clean(_frame(x).rolling(d, min_periods=d).corr(_frame(y)).to_numpy())


def ts_covariance(x, y, d):
    d = _window(d, "ts_covariance", 2)
    return _clean(_frame(x).rolling(d, min_periods=d).cov(_frame(y)).to_numpy())


def ts_decay_linear(x, d, dense=False):
    """Weighted mean of the last d values with weights d, d-1, ..., 1 (today heaviest).

    Like the other time-series operators a missing value in the window gives NaN;
    with ``dense`` the values present are averaged (NaN only when none is).
    """
    d = _window(d, "ts_decay_linear")
    values = np.asarray(x, dtype=np.float64)
    total = np.zeros_like(values)
    weight = np.zeros_like(values)
    present = np.zeros(values.shape, dtype=np.int32)
    for lag in range(d):
        shifted = np.full_like(values, np.nan)
        shifted[lag:] = values[:len(values) - lag] if lag else values
        w = float(d - lag)
        ok = ~np.isnan(shifted)
        total += np.where(ok, shifted * w, 0.0)
        weight += np.where(ok, w, 0.0)
        present += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / weight
    out = np.where(present > 0 if dense else present == d, out, np.nan)
    out[: d - 1] = np.nan
    return _clean(out)


def ts_backfill(x, d):
    """Fill a missing value with the most recent value from the last d days."""
    d = _window(d, "ts_backfill")
    return _clean(_frame(x).ffill(limit=d).to_numpy())


def ts_count_nans(x, d):
    d = _window(d, "ts_count_nans")
    return _clean(_frame(np.isnan(x).astype(np.float64)).rolling(d, min_periods=d).sum().to_numpy())


def ts_regression(y, x, d, lag=0, rettype=0):
    """Rolling OLS of y on x (lagged by ``lag``): 0 residual, 1 intercept, 2 slope, 3 fitted value, 6 R-squared."""
    d = _window(d, "ts_regression", 2)
    lag = _window(lag, "ts_regression lag", 0)
    x = ts_delay(x, lag) if lag else x
    fy, fx = _frame(y), _frame(x)
    both = fy.notna() & fx.notna()
    fy, fx = fy.where(both), fx.where(both)
    mean_x = fx.rolling(d, min_periods=d).mean()
    mean_y = fy.rolling(d, min_periods=d).mean()
    cov = fx.rolling(d, min_periods=d).cov(fy)
    var = fx.rolling(d, min_periods=d).var()
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = cov / var
        intercept = mean_y - slope * mean_x
        fitted = intercept + slope * fx
        residual = fy - fitted
    kind = int(rettype)
    if kind == 0:
        result = residual
    elif kind == 1:
        result = intercept
    elif kind == 2:
        result = slope
    elif kind == 3:
        result = fitted
    elif kind == 6:
        corr = fx.rolling(d, min_periods=d).corr(fy)
        result = corr * corr
    else:
        raise ExpressionError("ts_regression: rettype must be 0 (residual), 1 (intercept), 2 (slope), 3 (fitted) or 6 (R-squared)")
    return _clean(result.to_numpy())


def hump(x, hump=0.01):
    """Keep yesterday's value unless today's differs by more than ``hump`` x the day's mean |value|."""
    values = np.asarray(x, dtype=np.float64)
    out = np.full_like(values, np.nan)
    previous = np.full(values.shape[1], np.nan)
    for row in range(len(values)):
        today = values[row]
        scale = np.nanmean(np.abs(today)) if np.isfinite(today).any() else np.nan
        keep = np.isfinite(previous) & np.isfinite(today) & (np.abs(today - previous) < float(hump) * scale)
        current = np.where(keep, previous, today)
        out[row] = current
        previous = current
    return _clean(out)


def trade_when(trigger, alpha, exit):
    """Take ``alpha`` when trigger > 0, drop the position (NaN) when exit > 0, otherwise keep the last value."""
    t, a, e = (np.asarray(v, dtype=np.float64) for v in (trigger, alpha, exit))
    out = np.full_like(a, np.nan)
    previous = np.full(a.shape[1], np.nan)
    for row in range(len(a)):
        current = np.where(t[row] > 0, a[row], previous)
        current = np.where(e[row] > 0, np.nan, current)
        out[row] = current
        previous = current
    return _clean(out)


def days_from_last_change(x):
    values = np.asarray(x, dtype=np.float64)
    out = np.full_like(values, np.nan)
    count = np.full(values.shape[1], np.nan)
    for row in range(len(values)):
        if row == 0:
            count = np.where(np.isnan(values[0]), np.nan, 0.0)
        else:
            changed = values[row] != values[row - 1]
            count = np.where(np.isnan(values[row]), np.nan, np.where(changed | np.isnan(count), 0.0, count + 1))
        out[row] = count
    return _clean(out)


# ------------------------------------------------------------- cross-section
def rank(x, members, rate=2):
    """Rank across members each day, scaled to [0, 1]; ties share the average rank."""
    values = np.where(members, np.asarray(x, dtype=np.float64), np.nan)
    ranks = _frame(values).rank(axis=1, method="average").to_numpy()
    count = np.sum(~np.isnan(values), axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(count > 1, (ranks - 1) / (count - 1), np.where(count == 1, 0.5, np.nan))
    return _clean(np.where(np.isnan(values), np.nan, out))


def zscore(x, members):
    values = np.where(members, np.asarray(x, dtype=np.float64), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = row_mean(values)
        std = row_std(values)
        return _clean((values - mean) / std)


def normalize(x, members, useStd=False, limit=0.0):
    values = np.where(members, np.asarray(x, dtype=np.float64), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = values - row_mean(values)
        if useStd:
            out = out / row_std(values)
    if limit and limit > 0:
        out = np.clip(out, -float(limit), float(limit))
    return _clean(out)


def scale(x, members, scale=1.0, longscale=None, shortscale=None):
    values = np.where(members, np.asarray(x, dtype=np.float64), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        if longscale is not None or shortscale is not None:
            pos, neg = np.where(values > 0, values, 0.0), np.where(values < 0, values, 0.0)
            pos_sum = np.nansum(pos, axis=1, keepdims=True)
            neg_sum = -np.nansum(neg, axis=1, keepdims=True)
            # A side with no positions scales to zero instead of turning the day into 0/0 = NaN.
            longs = np.where(pos_sum > 0, pos / np.where(pos_sum > 0, pos_sum, 1.0), 0.0) * float(longscale or 0)
            shorts = np.where(neg_sum > 0, neg / np.where(neg_sum > 0, neg_sum, 1.0), 0.0) * float(shortscale or 0)
            out = np.where(np.isnan(values), np.nan, longs + shorts)
        else:
            out = values / np.nansum(np.abs(values), axis=1, keepdims=True) * float(scale)
    return _clean(out)


def winsorize(x, members, std=4.0):
    values = np.where(members, np.asarray(x, dtype=np.float64), np.nan)
    with np.errstate(invalid="ignore"):
        mean = row_mean(values)
        sd = row_std(values)
        return _clean(np.clip(values, mean - float(std) * sd, mean + float(std) * sd))


def vector_neut(x, y, members):
    """x minus its cross-sectional projection on y (both demeaned) each day."""
    a = np.where(members, np.asarray(x, dtype=np.float64), np.nan)
    b = np.where(members, np.asarray(y, dtype=np.float64), np.nan)
    both = ~np.isnan(a) & ~np.isnan(b)
    a, b = np.where(both, a, np.nan), np.where(both, b, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = a - row_mean(a)
        b = b - row_mean(b)
        beta = np.nansum(a * b, axis=1, keepdims=True) / np.nansum(b * b, axis=1, keepdims=True)
        return _clean(a - beta * b)


def _group_codes(group, shape) -> tuple[np.ndarray, int]:
    """Compact integer group codes 0..k-1 broadcast to ``shape``; -1 where the group is unknown.

    Labels are compacted so the per-(day, group) tables scale with the number of
    distinct groups, not with the largest label (an arbitrary series used as a
    group would otherwise allocate gigabytes).
    """
    codes = np.asarray(group, dtype=np.float64)
    codes = np.broadcast_to(codes, shape) if codes.ndim == 1 else codes
    known = np.isfinite(codes) & (codes >= 0)
    integer = np.full(codes.shape, -1, dtype=np.int64)
    if not known.any():
        return integer, 0
    labels, inverse = np.unique(np.floor(codes[known]), return_inverse=True)
    integer[known] = inverse
    return integer, len(labels)


def _group_rank(values: np.ndarray, codes: np.ndarray, groups: int) -> np.ndarray:
    """Rank within each (day, group) cell to [0, 1] with averaged ties; a singleton ranks 0.5."""
    out = np.full_like(values, np.nan)
    valid = ~np.isnan(values) & (codes >= 0)
    if not valid.any():
        return _clean(out)
    rows = np.broadcast_to(np.arange(values.shape[0])[:, None], values.shape)
    cell = (rows * max(groups, 1) + codes)[valid]
    data = values[valid]
    order = np.lexsort((data, cell))
    cell_sorted, data_sorted = cell[order], data[order]
    new_cell = np.r_[True, cell_sorted[1:] != cell_sorted[:-1]]
    block_start = np.flatnonzero(new_cell)
    block = np.cumsum(new_cell) - 1
    count = np.diff(np.r_[block_start, len(cell_sorted)])[block]
    new_tie = new_cell | np.r_[True, data_sorted[1:] != data_sorted[:-1]]
    tie_start = np.flatnonzero(new_tie)
    tie = np.cumsum(new_tie) - 1
    tie_length = np.diff(np.r_[tie_start, len(cell_sorted)])
    average = (tie_start + (tie_length - 1) / 2.0)[tie] - block_start[block]  # mean 0-based position in its cell
    with np.errstate(invalid="ignore", divide="ignore"):
        ranks = np.where(count > 1, average / (count - 1), 0.5)
    unsorted = np.empty(len(data))
    unsorted[order] = ranks
    out[valid] = unsorted
    return _clean(out)


def _group_apply(x, group, members, kind):
    """Per-day, per-group statistics via one bincount over (day, group) cells."""
    values = np.where(members, np.asarray(x, dtype=np.float64), np.nan)
    codes, groups = _group_codes(group, values.shape)
    if kind == "rank":
        return _group_rank(values, codes, groups)
    valid = ~np.isnan(values) & (codes >= 0)
    rows = np.broadcast_to(np.arange(values.shape[0])[:, None], values.shape)
    cell = np.where(valid, rows * max(groups, 1) + codes, 0)
    size = values.shape[0] * max(groups, 1)
    flat_valid = valid.ravel()
    index = cell.ravel()[flat_valid]
    data = values.ravel()[flat_valid]
    counts = np.bincount(index, minlength=size)
    sums = np.bincount(index, weights=data, minlength=size)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts
        mean = np.where(valid, means[cell], np.nan)
        if kind == "neutralize":
            return _clean(values - mean)
        if kind == "mean":
            return _clean(mean)
        if kind == "zscore":
            deviation = data - means[index]  # two-pass: E[x^2] - E[x]^2 cancels badly for offset series
            squares = np.bincount(index, weights=deviation * deviation, minlength=size)
            std = np.sqrt(squares / counts)
            return _clean((values - mean) / np.where(valid, std[cell], np.nan))
    raise ExpressionError(f"unknown group operation {kind}")


def group_neutralize(x, group, members):
    return _group_apply(x, group, members, "neutralize")


def group_mean(x, group, members):
    return _group_apply(x, group, members, "mean")


def group_zscore(x, group, members):
    return _group_apply(x, group, members, "zscore")


def group_rank(x, group, members):
    return _group_apply(x, group, members, "rank")


def bucket(x, members, range="0,1,0.1", buckets=None):
    """Group codes from value ranges, e.g. bucket(rank(cap), range="0,1,0.1") gives ten buckets."""
    values = np.where(members, np.asarray(x, dtype=np.float64), np.nan)
    if buckets is not None:
        edges = np.asarray([float(v) for v in str(buckets).split(",")])
    else:
        parts = [float(v) for v in str(range).split(",")]
        if len(parts) != 3 or parts[2] <= 0 or parts[1] <= parts[0]:
            raise ExpressionError('bucket: range must be "start,end,step" with end > start and step > 0')
        edges = np.arange(parts[0], parts[1] + parts[2] / 2, parts[2])
    codes = np.digitize(values, edges[1:-1]).astype(np.float64)
    return np.where(np.isnan(values), np.nan, codes).astype(F32)


# ------------------------------------------------------------------ elementwise
def _binary(op: str, a, b):
    a64, b64 = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        if op == "+":
            return _clean(a64 + b64)
        if op == "-":
            return _clean(a64 - b64)
        if op == "*":
            return _clean(a64 * b64)
        if op == "/":
            return _clean(a64 / b64)
        if op == "^":
            return _clean(np.power(a64, b64))
        nan = np.isnan(a64) | np.isnan(b64)
        if op in ("==", "!=", "<", "<=", ">", ">="):
            result = {"==": a64 == b64, "!=": a64 != b64, "<": a64 < b64, "<=": a64 <= b64,
                      ">": a64 > b64, ">=": a64 >= b64}[op]
        elif op == "&&":
            result = (a64 > 0) & (b64 > 0)
        elif op == "||":
            result = (a64 > 0) | (b64 > 0)
        else:
            raise ExpressionError(f"unknown operator {op}")
    return np.where(nan, np.nan, result.astype(np.float64)).astype(F32)


def if_else(condition, then, otherwise):
    c = np.asarray(condition, dtype=np.float64)
    return _clean(np.where(np.isnan(c), np.nan, np.where(c > 0, np.asarray(then, dtype=np.float64),
                                                        np.asarray(otherwise, dtype=np.float64))))


def _unary(fn: Callable[[np.ndarray], np.ndarray]) -> Callable[[Any], np.ndarray]:
    def apply(x):
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            return _clean(fn(np.asarray(x, dtype=np.float64)))
    return apply


ELEMENTWISE: dict[str, Callable[..., np.ndarray]] = {
    "abs": _unary(np.abs), "sign": _unary(np.sign), "exp": _unary(np.exp), "sqrt": _unary(np.sqrt),
    "log": _unary(lambda v: np.where(v > 0, np.log(np.where(v > 0, v, 1.0)), np.nan)),
    "tanh": _unary(np.tanh), "sigmoid": _unary(lambda v: 1 / (1 + np.exp(-v))),
    "inverse": _unary(lambda v: 1 / v), "reverse": _unary(lambda v: -v),
    "s_log_1p": _unary(lambda v: np.sign(v) * np.log1p(np.abs(v))),
    "is_nan": lambda x: np.isnan(np.asarray(x, dtype=np.float64)).astype(F32),
}


def power(x, e):
    return _binary("^", x, e)


def signed_power(x, e):
    a = np.asarray(x, dtype=np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        return _clean(np.sign(a) * np.power(np.abs(a), np.asarray(e, dtype=np.float64)))


def clamp(x, lower=-math.inf, upper=math.inf):
    lower, upper = float(lower), float(upper)
    if lower > upper:
        raise ExpressionError("clamp: lower must not exceed upper")
    return _clean(np.clip(np.asarray(x, dtype=np.float64), lower, upper))


def elementwise_min(*values):
    stacked = np.stack([np.broadcast_to(np.asarray(v, dtype=np.float64), np.shape(values[0])) for v in values])
    return _clean(np.where(np.isnan(stacked).any(axis=0), np.nan, np.min(stacked, axis=0)))


def elementwise_max(*values):
    stacked = np.stack([np.broadcast_to(np.asarray(v, dtype=np.float64), np.shape(values[0])) for v in values])
    return _clean(np.where(np.isnan(stacked).any(axis=0), np.nan, np.max(stacked, axis=0)))
