"""Independent reference for Alpha Lab's restricted simulation (plain numpy/pandas).

This module re-derives, from the written conventions in docs/ALPHA_LAB_REFERENCE.md
and without importing anything from ``qlib_yahoo``, what the Alpha Lab engine is
documented to compute for the restricted configuration:

    delay 1, decay 0, MARKET neutralization, truncation c (0 = off),
    pasteurization on, zero cost, a fixed book that is not compounded.

It has three layers, each usable on its own:

* a snapshot reader for the Qlib binary layout the planted generator writes
  (``read_snapshot``), so the comparison never goes through the engine's loader;
* the handful of FASTEXPR operators the three test alphas need (``rank``,
  ``ts_delta``, ``ts_mean``, ``ts_std_dev``, ``log``, ``returns_field``), with
  the engine's float32 storage convention applied explicitly through ``f32``;
* the weight pipeline, PnL and the six metrics (``simulate``), written from the
  conventions rather than from the engine's code: direct demeaning, a sort-based
  closed-form water-filling for the stock cap, and metric formulas in one place.

Convention numbers in comments (``C1`` ...) refer to docs/ALPHA_LAB_REFERENCE.md.
"""

from __future__ import annotations

import gzip
import io
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252
TURNOVER_FLOOR = 0.125


# ------------------------------------------------------------------ snapshot
def read_snapshot(dataset: Path, fields: tuple[str, ...] = ("close", "volume")) -> dict[str, Any]:
    """Dates, symbols, float32 fields and the TOP-N membership mask of a planted snapshot."""
    dataset = Path(dataset)
    qlib = dataset / "qlib_data"
    dates = np.asarray((qlib / "calendars/day.txt").read_text(encoding="utf-8").split(), dtype="datetime64[D]")
    symbols = [line.split("\t", 1)[0] for line in
               (qlib / "instruments/all.txt").read_text(encoding="utf-8").splitlines() if line]
    out = {name: np.full((len(dates), len(symbols)), np.nan, dtype=np.float32) for name in fields}
    for column, symbol in enumerate(symbols):
        for name in fields:
            payload = np.fromfile(qlib / "features" / symbol.lower() / f"{name}.day.bin", dtype="<f4")
            first = int(payload[0])  # Qlib stores the calendar offset of the first value, then the values
            out[name][first:first + len(payload) - 1, column] = payload[1:]
    return {"dates": dates, "symbols": symbols, "fields": out}


def membership_mask(dataset: Path, dates: np.ndarray, symbols: list[str], size: int) -> np.ndarray:
    """Boolean (dates, symbols): names ranked 1..size between their effective dates (inclusive)."""
    data = (Path(dataset) / "universe/membership.csv.gz").read_bytes()
    table = pd.read_csv(io.BytesIO(gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data),
                        parse_dates=["effective_start", "effective_end"])
    mask = np.zeros((len(dates), len(symbols)), dtype=bool)
    column = {s: i for i, s in enumerate(symbols)}
    for row in table.itertuples(index=False):
        if int(row.rank) > size or row.instrument not in column:
            continue
        first = int(np.searchsorted(dates, np.datetime64(row.effective_start, "D")))
        last = int(np.searchsorted(dates, np.datetime64(row.effective_end, "D"), side="right"))
        mask[first:last, column[row.instrument]] = True
    return mask


def test_window(members: np.ndarray, warmup: int) -> int:
    """First test row (C3): the session after the universe first has all its members, or warm-up."""
    counts = members.sum(axis=1)
    first_member = int(np.flatnonzero(counts > 0)[0])
    full = int(np.flatnonzero(counts >= counts.max())[0])
    return max(full + 1, first_member + 1, warmup)


# ----------------------------------------------------------------- operators
def f32(values: np.ndarray) -> np.ndarray:
    """C5: every field and operator result is stored as float32; non-finite values are missing."""
    out = np.asarray(values, dtype=np.float64).astype(np.float32).astype(np.float64)
    out[~np.isfinite(out)] = np.nan
    return out


def returns_field(close: np.ndarray) -> np.ndarray:
    """C4: close / previous close - 1, float32 stored."""
    close = np.asarray(close, dtype=np.float64)
    out = np.full(close.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[1:] = close[1:] / close[:-1] - 1
    return f32(out)


def log(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, np.nan)
    positive = x > 0
    out[positive] = np.log(x[positive])
    return f32(out)


def ts_delta(x: np.ndarray, d: int) -> np.ndarray:
    """C6: x[t] - x[t-d]; missing for the first d rows."""
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, np.nan)
    out[d:] = x[d:] - x[:-d]
    return f32(out)


def _windows(x: np.ndarray, d: int) -> np.ndarray:
    # (T - d + 1, N, d) view of rows t-d+1..t for t = d-1..T-1
    return np.lib.stride_tricks.sliding_window_view(np.asarray(x, dtype=np.float64), d, axis=0)


def ts_mean(x: np.ndarray, d: int) -> np.ndarray:
    """C6: mean of rows t-d+1..t; missing while the window is incomplete or has a gap."""
    out = np.full(np.shape(x), np.nan)
    out[d - 1:] = _windows(x, d).mean(axis=-1)  # NaN inside the window propagates
    return f32(out)


def ts_std_dev(x: np.ndarray, d: int) -> np.ndarray:
    """C6/C7: sample standard deviation (ddof = 1) of rows t-d+1..t."""
    out = np.full(np.shape(x), np.nan)
    out[d - 1:] = _windows(x, d).std(axis=-1, ddof=1)
    return f32(out)


def _rank_row(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan)
    present = np.flatnonzero(np.isfinite(values))
    n = len(present)
    if n == 0:
        return out
    if n == 1:
        out[present] = 0.5
        return out
    unique, inverse, counts = np.unique(values[present], return_inverse=True, return_counts=True)
    first_position = np.concatenate(([0], np.cumsum(counts)[:-1])) + 1  # 1-based position of each tie group
    average = first_position + (counts - 1) / 2.0
    out[present] = (average[inverse] - 1) / (n - 1)
    return out


def rank(x: np.ndarray, members: np.ndarray) -> np.ndarray:
    """C8: per day, (average tie rank - 1) / (count - 1) over members with a value; others missing."""
    values = np.where(members, np.asarray(x, dtype=np.float64), np.nan)
    return f32(np.vstack([_rank_row(row) for row in values]))


def negate(x: np.ndarray) -> np.ndarray:
    return f32(-np.asarray(x, dtype=np.float64))


def subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return f32(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))


ALPHAS: dict[str, Callable[[dict[str, np.ndarray], np.ndarray], np.ndarray]] = {
    "-rank(ts_delta(close, 5))":
        lambda fields, members: negate(rank(ts_delta(fields["close"], 5), members)),
    "rank(log(volume) - ts_mean(log(volume), 20))":
        lambda fields, members: rank(subtract(log(fields["volume"]), ts_mean(log(fields["volume"]), 20)), members),
    "-rank(ts_std_dev(returns, 20))":
        lambda fields, members: negate(rank(ts_std_dev(returns_field(fields["close"]), 20), members)),
    # A raw (unranked) signal whose demeaned weights are heavy-tailed, so a 0.2 cap binds.
    "-ts_delta(close, 5)":
        lambda fields, members: negate(ts_delta(fields["close"], 5)),
}


def alpha(expression: str, fields: dict[str, np.ndarray], members: np.ndarray) -> np.ndarray:
    """The reference's own evaluation of one of the supported expressions (float32-stored values)."""
    return ALPHAS[expression](fields, members)


# -------------------------------------------------------------------- weights
def water_fill(values: np.ndarray, budget: float, cap: float) -> np.ndarray:
    """C12: allocate ``budget`` over positive ``values`` in proportion, no name above ``cap``.

    Closed form: a_i = min(cap, lam * v_i) with lam chosen so the allocation sums to
    the budget. Found by sorting descending and saturating the k largest names.
    """
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(values)
    active = np.flatnonzero(values > 0)
    if not len(active) or budget <= 0:
        return out
    if cap <= 0:
        out[active] = values[active] * budget / values[active].sum()
        return out
    if budget >= len(active) * cap:
        out[active] = cap
        return out
    order = active[np.argsort(-values[active], kind="stable")]
    v = values[order]
    tail = np.cumsum(v[::-1])[::-1]  # tail[k] = sum of v[k:]
    for k in range(len(v)):
        lam = (budget - k * cap) / tail[k]
        if lam * v[k] <= cap:
            out[order[:k]] = cap
            out[order[k:]] = lam * v[k:]
            return out
    out[order] = cap  # unreachable when budget < n * cap; kept for completeness
    return out


def market_neutral_row(signal: np.ndarray, cap: float) -> np.ndarray:
    """C9-C13: demean the finite signals, scale to gross 1, then cap each side's names."""
    present = np.isfinite(signal)
    w = np.zeros(signal.shape)
    if present.sum() == 0:
        return w
    centred = signal[present] - signal[present].mean()
    gross = np.abs(centred).sum()
    if gross <= 0:
        return w
    unit = centred / gross
    longs, shorts = np.maximum(unit, 0.0), np.maximum(-unit, 0.0)
    budget = min(longs.sum(), shorts.sum())
    if cap > 0:
        budget = min(budget, np.count_nonzero(longs) * cap, np.count_nonzero(shorts) * cap)
    w[present] = water_fill(longs, budget, cap) - water_fill(shorts, budget, cap)
    return w


def weights(alpha_values: np.ndarray, members: np.ndarray, close: np.ndarray, *, delay: int = 1,
            truncation: float = 0.2) -> np.ndarray:
    """Daily weights for the restricted configuration.

    C9  the alpha is shifted by ``delay`` rows (row t holds the alpha of t - delay);
    C10 pasteurization: non-members on day t are missing; so are names without a close on day t
        (the masks are those of the holding day t, not shifted);
    C11 missing means no position, and is excluded from the demeaning mean;
    C12 MARKET neutralization with the stock cap.
    """
    x = np.asarray(alpha_values, dtype=np.float64).copy()
    if delay:
        shifted = np.full(x.shape, np.nan)
        shifted[delay:] = x[:-delay]
        x = shifted
    x[~members] = np.nan
    x[~np.isfinite(np.asarray(close, dtype=np.float64))] = np.nan
    return np.vstack([market_neutral_row(row, float(truncation)) for row in x])


# ------------------------------------------------------------- PnL and metrics
def simulate(alpha_values: np.ndarray, close: np.ndarray, members: np.ndarray, *, start: int,
             booksize: float = 20_000_000.0, delay: int = 1, truncation: float = 0.2,
             cost_bps: float = 0.0) -> dict[str, Any]:
    """Weights, daily PnL and traded fraction over test rows start..T-1, with the metrics."""
    close64 = np.asarray(close, dtype=np.float64)
    w = weights(alpha_values, members, close64, delay=delay, truncation=truncation)
    held = np.zeros(close64.shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        held[1:] = close64[1:] / close64[:-1] - 1
    held[~np.isfinite(held)] = 0.0  # C14: a name without a return earns nothing that day
    test = np.arange(start, len(close64))
    gross_pnl = np.einsum("ij,ij->i", w[test - 1], held[test]) * booksize  # C15: PnL[t] = w[t-1] . r[t]
    traded = np.abs(w[test] - w[test - 1]).sum(axis=1)  # C16: dollars traded at the close of t, / book
    pnl = gross_pnl - traded * booksize * cost_bps / 1e4  # C17
    return {"weights": w[test], "pnl": pnl, "traded": traded, "metrics": metrics(pnl, traded, booksize),
            "gross_exposure": float(np.abs(w[test]).sum(axis=1).mean()),
            "net_exposure": float(w[test].sum(axis=1).mean()),
            "max_weight": float(np.abs(w[test]).max(initial=0.0)),
            "long_count": float((w[test] > 0).sum(axis=1).mean()),
            "short_count": float((w[test] < 0).sum(axis=1).mean())}


def metrics(pnl: np.ndarray, traded: np.ndarray, booksize: float) -> dict[str, float | None]:
    """C18-C23: the six BRAIN-style metrics on a fixed book."""
    pnl = np.asarray(pnl, dtype=np.float64)
    traded = np.asarray(traded, dtype=np.float64)
    half_book = booksize / 2.0
    mean = pnl.mean()
    std = pnl.std(ddof=1)
    sharpe = mean / std * math.sqrt(PERIODS_PER_YEAR) if std > 0 else None
    returns = mean * PERIODS_PER_YEAR / half_book
    turnover = traded.mean()
    fitness = sharpe * math.sqrt(abs(returns) / max(turnover, TURNOVER_FLOOR)) if sharpe is not None else None
    cumulative = np.cumsum(pnl)
    peak = np.maximum(0.0, np.maximum.accumulate(cumulative))  # the book starts flat at a peak of 0
    drawdown = float(np.max(peak - cumulative, initial=0.0)) / half_book
    total_traded = traded.sum() * booksize
    margin_bps = pnl.sum() / total_traded * 1e4 if total_traded > 0 else None
    return {"sharpe": sharpe, "returns": float(returns), "turnover": float(turnover), "fitness": fitness,
            "margin_bps": margin_bps, "drawdown": drawdown, "pnl": float(pnl.sum()), "days": int(len(pnl))}


METRICS = ("sharpe", "returns", "turnover", "fitness", "margin_bps", "drawdown")


def relative_difference(engine: float | None, reference: float | None) -> float:
    """|engine - reference| / max(|engine|, |reference|); 0 when both are None, inf when one is."""
    if engine is None and reference is None:
        return 0.0
    if engine is None or reference is None:
        return math.inf
    scale = max(abs(engine), abs(reference))
    return abs(engine - reference) / scale if scale > 0 else abs(engine - reference)
