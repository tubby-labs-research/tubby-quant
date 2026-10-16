"""BRAIN-style daily simulation of alpha weights, with its metrics.

Timing: the alpha value for day t uses data through the close of day t. With
``delay`` = 1 the portfolio held from the close of day t to the close of day t+1 is
built from the alpha of day t-1 (a full day to trade); with ``delay`` = 0 from day t
(trading at the close with that close's information). PnL of day t+1 is
``booksize x sum(w[t] x return[t+1])`` using adjusted close-to-close returns.

Weights, each day: delayed alpha -> linear decay -> pasteurization (only universe
members with a price) -> neutralization (market or group demeaning) -> scaled to a
gross of 1 -> stock cap. Neutral modes demean in float64 within each day's
market or classification group. For each group the final dollars on EACH side
are min(pre-cap long dollars, pre-cap short dollars, positive count * cap,
negative count * cap). Each side uses proportional capped water-filling,
preserving signs. Cap 0 omits the count/cap limits. Constrained group budgets
are never transferred or rescaled to force gross 1; unknown and singleton
groups hold nothing. NONE retains historical iterative truncation, so an
expression-level group_neutralize with NONE does not ensure final neutrality.
Metric conventions follow BRAIN's published
definitions as closely as they are known; they are a local approximation, not a
reproduction of BRAIN's numbers:

    sharpe    mean(daily PnL) / std(daily PnL) x sqrt(252)
    returns   annualised PnL / (booksize / 2)
    turnover  mean daily traded dollars / booksize
    fitness   sharpe x sqrt(|returns| / max(turnover, 0.125))
    margin    total PnL / total traded dollars, in basis points
    drawdown  largest peak-to-trough fall of cumulative PnL / (booksize / 2)
"""

from __future__ import annotations

import gzip
import io
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from qlib_yahoo.alphalab import ops
from qlib_yahoo.alphalab.engine import Context, evaluate
from qlib_yahoo.alphalab.fastexpr import ExpressionError
from qlib_yahoo.alphalab.settings import NEUTRALIZATIONS, Settings  # noqa: F401  (re-exported)
from qlib_yahoo.dataset import DatasetError

ENGINE_VERSION = "alphalab-3"
PERIODS_PER_YEAR = 252
NEUTRALIZATION_CHUNK_ROWS = 64


class Workspace:
    """A verified snapshot held in memory: price fields, dated universes and groups."""

    def __init__(self, dataset: Path, *, panel: Any = None):
        from qlib_yahoo.panel import available_fields, load_panel
        from qlib_yahoo.reference import dataset_groups

        if panel is None:
            import json as _json

            manifest = _json.loads((Path(dataset) / "manifest.json").read_text(encoding="utf-8"))
            wanted = [name for name in ("open", "high", "low", "close", "volume", "close_raw", "dividend", "cap", "vwap")
                      if name in available_fields(manifest)]
            panel = load_panel(dataset, wanted)
        self.panel = panel
        self.dataset = Path(panel.dataset)
        self.dates = panel.dates
        self.symbols = panel.symbols
        self.fields = {name: values for name, values in panel.fields.items()}
        self.identity = {"dataset": str(self.dataset),
                         "qlib_binary_tree_sha256": panel.manifest.get("qlib_binary_tree_sha256"),
                         "fields": sorted(self.fields)}
        groups = dataset_groups(self.dataset, panel.manifest)
        self.groups: dict[str, np.ndarray] = {}
        for name in ("sector", "industry", "country"):
            labels = sorted({g[name] for g in groups.values() if g.get(name)})
            if not labels:
                continue
            code = {label: float(i) for i, label in enumerate(labels)}
            self.groups[name] = np.asarray([code.get((groups.get(s) or {}).get(name, ""), np.nan) for s in self.symbols])
        self._universes: dict[str, np.ndarray] = {}

    def universe(self, name: str) -> np.ndarray:
        """Boolean (dates, symbols) membership. TOP<n> are the top n of the snapshot's ranked universe."""
        key = name.upper()
        if key in self._universes:
            return self._universes[key]
        from qlib_yahoo.universe import dataset_universes

        universes = dataset_universes(self.panel.manifest)
        if not universes:
            raise DatasetError("this dataset has no dated universe; import it with a TOP-N universe")
        base_name = sorted(universes)[0]
        record = universes[base_name]
        if key.startswith("TOP") and key[3:].isdigit():
            size = int(key[3:])
            if size > int(record.get("size", 0)):
                raise DatasetError(f"{key} is larger than the dataset's {base_name} universe")
        elif key.lower() in universes:
            size = int(universes[key.lower()]["size"])
            base_name, record = key.lower(), universes[key.lower()]
        else:
            raise DatasetError(f"unknown universe {name}; use TOP3000, TOP1000, TOP500, TOP200 or a dataset universe")
        membership = self._membership(base_name, record)
        if record.get("mode") == "historical_index":
            if key.startswith("TOP") and key[3:].isdigit():
                raise DatasetError("historical index membership has no liquidity ranking; select its named universe")
            chosen = membership
        else:
            chosen = membership[membership["rank"] <= size]
        mask = np.zeros((len(self.dates), len(self.symbols)), dtype=bool)
        column = {s: i for i, s in enumerate(self.symbols)}
        starts = np.searchsorted(self.dates, chosen["effective_start"].to_numpy().astype("datetime64[D]"))
        ends = np.searchsorted(self.dates, chosen["effective_end"].to_numpy().astype("datetime64[D]"), side="right")
        for instrument, first, last in zip(chosen["instrument"], starts, ends):
            if instrument in column:
                mask[first:last, column[instrument]] = True
        if record.get("mode") == "historical_index":
            history = int(record.get("min_history_sessions", 1))
            valid = np.ones_like(mask)
            for field in ("open", "high", "low", "close", "volume"):
                values = self.fields.get(field)
                if values is None:
                    raise DatasetError("historical index eligibility requires OHLCV fields")
                valid &= np.isfinite(values) & (values > 0)
            high, low = self.fields["high"], self.fields["low"]
            valid &= (high >= np.maximum(self.fields["open"], self.fields["close"])) & (low <= np.minimum(self.fields["open"], self.fields["close"]))
            consecutive = np.zeros(len(self.symbols), dtype=int)
            for row in range(len(mask)):
                consecutive = np.where(valid[row], consecutive + 1, 0)
                mask[row] &= consecutive >= history
        self._universes[key] = mask
        return mask

    def _membership(self, name: str, record: dict[str, Any]) -> Any:
        import hashlib

        import pandas as pd

        path = self.dataset / record["membership_csv"]
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != record.get("membership_sha256"):
            raise DatasetError(f"universe {name} membership file differs from the dataset manifest")
        # Historical imports may seal plain CSV; ranked imports normally use gzip.
        payload = gzip.decompress(data) if data.startswith(b"\x1f\x8b") else data
        return pd.read_csv(io.BytesIO(payload), parse_dates=["effective_start", "effective_end"],
                           keep_default_na=False, na_values=[""])

    def window(self, settings: Settings) -> tuple[int, int, int]:
        """(first row evaluated incl. warm-up, first test row, last test row)."""
        members = self.universe(settings.universe)
        populated = np.flatnonzero(members.sum(axis=1) > 0)
        if not len(populated):
            raise DatasetError(f"universe {settings.universe} has no members")
        first_member = int(populated[0])
        if settings.start:
            start = int(np.searchsorted(self.dates, np.datetime64(settings.start, "D")))
        else:
            # like BRAIN, test on a full universe: from the first session the universe has all its members
            counts = members.sum(axis=1)
            full = np.flatnonzero(counts >= counts.max())
            start = max(int(full[0]) + 1 if len(full) else first_member + 1, first_member + 1, settings.warmup)
        end = (int(np.searchsorted(self.dates, np.datetime64(settings.end, "D"), side="right")) - 1
               if settings.end else len(self.dates) - 1)
        if start <= first_member:
            raise DatasetError(f"{settings.universe} first has members on {self.dates[first_member]}; start after it")
        if end - start < 20:
            raise DatasetError("the test period needs at least 20 sessions")
        return max(0, start - settings.warmup), start, end

    def context(self, first: int, last: int, members: np.ndarray) -> Context:
        rows = slice(first, last + 1)
        groups = {}
        for name, values in self.groups.items():
            values = _group_array(values, (len(self.dates), len(self.symbols)), name)
            groups[name] = values if values.ndim == 1 else values[rows]
        return Context(dates=self.dates[rows], symbols=self.symbols,
                       fields={name: values[rows] for name, values in self.fields.items()},
                       members=members[rows], groups=groups)


def _group_array(values: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.shape not in ((shape[1],), shape):
        raise DatasetError(f"{name} groups must have shape ({shape[1]},) or {shape}; got {values.shape}")
    return values


def _capped_side(values: np.ndarray, cells: np.ndarray, target: np.ndarray, cap: float) -> np.ndarray:
    """Water-fill all day/group sides together; each pass removes saturated names."""
    out = np.zeros_like(values)
    active = values > 0
    remaining = target.copy()
    while active.any():
        indices = np.flatnonzero(active)
        cell = cells[indices]
        sums = np.bincount(cell, weights=values[indices], minlength=len(target))
        proposed = values[indices] * np.divide(remaining[cell], sums[cell],
                                              out=np.zeros_like(values[indices]), where=sums[cell] > 0)
        saturated = proposed > cap if cap > 0 else np.zeros(len(indices), dtype=bool)
        out[indices[~saturated]] = proposed[~saturated]
        if not saturated.any():
            break
        fixed = indices[saturated]
        out[fixed] = cap
        remaining -= np.bincount(cells[fixed], weights=np.full(len(fixed), cap), minlength=len(target))
        remaining = np.maximum(remaining, 0.0)
        # Recompute only sides with newly saturated names, leaving finished sides intact.
        unfinished = np.bincount(cells[fixed], minlength=len(target)) > 0
        active[fixed] = False
        active &= unfinished[cells]
    return out


def _market_side(values: np.ndarray, target: np.ndarray, cap: float) -> np.ndarray:
    """Dense row-wise water-filling avoids per-name cell coordinates for MARKET."""
    if cap <= 0:
        sums = values.sum(axis=1)
        return values * np.divide(target, sums, out=np.zeros_like(target), where=sums > 0)[:, None]
    out = np.zeros_like(values)
    remaining = target.copy()
    pool = values.copy()
    while np.any(pool > 0):
        sums = pool.sum(axis=1)
        proposed = pool * np.divide(remaining, sums, out=np.zeros_like(target), where=sums > 0)[:, None]
        saturated = proposed > cap
        if not saturated.any():
            out += proposed
            break
        counts = saturated.sum(axis=1)
        finished = counts == 0
        out[finished] += proposed[finished]
        pool[finished] = 0
        out[saturated] = cap
        pool[saturated] = 0
        remaining = np.maximum(remaining - counts * cap, 0.0)
    return out


def _market_chunk(x: np.ndarray, cap: float) -> np.ndarray:
    """Precise demeaning using dense row reductions, without group sorts or coordinates."""
    valid = np.isfinite(x)
    data = np.where(valid, x, 0.0)
    reference = data[np.arange(len(x)), np.argmax(valid, axis=1)][:, None]
    with np.errstate(over="ignore"):
        shifted = data - reference
    shifted[~valid] = 0
    overflow = ~np.isfinite(shifted)
    magnitude = np.abs(shifted)
    references = np.broadcast_to(reference, x.shape)
    magnitude[overflow] = np.maximum(np.abs(data[overflow]), np.abs(references[overflow]))
    scale = np.maximum(magnitude.max(axis=1, keepdims=True), np.finfo(float).tiny)
    scales = np.broadcast_to(scale, x.shape)
    shifted[overflow] = data[overflow] / scales[overflow] - references[overflow] / scales[overflow]
    data = np.divide(shifted, scale, out=shifted, where=~overflow)
    counts = valid.sum(axis=1)
    for _ in range(2):
        sums = data.sum(axis=1)
        data -= np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)[:, None]
        data[~valid] = 0
    gross = np.abs(data).sum(axis=1, keepdims=True)
    data = np.divide(data, gross, out=data, where=gross > 0)
    positive, negative = np.maximum(data, 0), np.maximum(-data, 0)
    target = np.minimum(positive.sum(axis=1), negative.sum(axis=1))
    if cap > 0:
        target = np.minimum(target, (positive > 0).sum(axis=1) * cap)
        target = np.minimum(target, (negative > 0).sum(axis=1) * cap)
    return _market_side(positive, target, cap) - _market_side(negative, target, cap)


def _neutral_chunk(x: np.ndarray, codes: np.ndarray | None, cap: float) -> np.ndarray:
    """One bounded row block; MARKET needs neither a code matrix nor sorting."""
    if codes is None:
        return _market_chunk(x, cap)
    valid = np.isfinite(x)
    valid &= np.isfinite(codes) & (codes >= 0)
    w = np.zeros_like(x)
    if not valid.any():
        return w
    rows, columns = np.nonzero(valid)
    # Compact codes avoid allocating by a potentially sparse external code's magnitude.
    labels, inverse = np.unique(codes[valid], return_inverse=True)
    cells = rows * len(labels) + inverse
    size = x.shape[0] * len(labels)
    data = x[valid]
    # Scaling and subtracting a reference before the mean protects large offsets
    # and avoids overflow without changing demeaning or relative group budgets.
    # First observation per cell in linear time, rather than a second full sort.
    first = np.full(size, len(rows), dtype=np.int64)
    np.minimum.at(first, cells, np.arange(len(rows)))
    populated = first < len(rows)
    reference = np.zeros(size)
    reference[populated] = data[first[populated]]
    with np.errstate(over="ignore"):
        shifted = data - reference[cells]
    # Subtract before dividing so small differences on large offsets stay precise.
    overflow = ~np.isfinite(shifted)
    magnitude = np.abs(shifted)
    magnitude[overflow] = np.maximum(np.abs(data[overflow]), np.abs(reference[cells[overflow]]))
    scale = np.zeros(x.shape[0])
    np.maximum.at(scale, rows, magnitude)
    scale = np.maximum(scale, np.finfo(float).tiny)
    shifted[overflow] = (data[overflow] / scale[rows[overflow]]
                         - reference[cells[overflow]] / scale[rows[overflow]])
    data = np.divide(shifted, scale[rows], out=shifted, where=~overflow)
    counts = np.bincount(cells, minlength=size)
    sums = np.bincount(cells, weights=data, minlength=size)
    data -= sums[cells] / counts[cells]
    # Remove the small residual of floating-point summation before sign allocation.
    sums = np.bincount(cells, weights=data, minlength=size)
    data -= sums[cells] / counts[cells]
    gross = np.bincount(rows, weights=np.abs(data), minlength=x.shape[0])
    data = np.divide(data, gross[rows], out=np.zeros_like(data), where=gross[rows] > 0)
    positive, negative = np.maximum(data, 0.0), np.maximum(-data, 0.0)
    target = np.minimum(np.bincount(cells, weights=positive, minlength=size),
                        np.bincount(cells, weights=negative, minlength=size))
    if cap > 0:
        target = np.minimum(target, np.bincount(cells[positive > 0], minlength=size) * cap)
        target = np.minimum(target, np.bincount(cells[negative > 0], minlength=size) * cap)
    w[rows, columns] = (_capped_side(positive, cells, target, cap)
                        - _capped_side(negative, cells, target, cap))
    return w


def _neutral_weights(x: np.ndarray, codes: np.ndarray | None, cap: float, *, name: str,
                     diagnostics: dict[str, Any] | None, test_start: int) -> np.ndarray:
    """Bound working arrays/sorts to 64 rows; aggregate only scalar test diagnostics."""
    w = np.empty_like(x)
    covered_count = eligible_count = inactive_days = 0
    max_net = 0.0
    for first in range(0, len(x), NEUTRALIZATION_CHUNK_ROWS):
        last = min(first + NEUTRALIZATION_CHUNK_ROWS, len(x))
        block_codes = None if codes is None else codes[first:last]
        block = _neutral_chunk(x[first:last], block_codes, cap)
        w[first:last] = block
        offset = max(test_start - first, 0)
        if offset >= len(block):
            continue
        selected = np.isfinite(x[first + offset:last])
        if codes is None:
            covered = selected
        else:
            selected_codes = block_codes[offset:]
            covered = selected & np.isfinite(selected_codes) & (selected_codes >= 0)
        eligible_count += int(selected.sum())
        covered_count += int(covered.sum())
        if diagnostics is not None:
            test_w = block[offset:]
            inactive_days += int(np.count_nonzero(~np.any(test_w != 0, axis=1)))
            if codes is None:
                max_net = max(max_net, float(np.abs(test_w.sum(axis=1)).max(initial=0)))
            else:
                for row, row_codes in zip(test_w, selected_codes):
                    active = row != 0
                    if active.any():
                        _, cell = np.unique(row_codes[active], return_inverse=True)
                        max_net = max(max_net, float(np.abs(np.bincount(cell, weights=row[active])).max(initial=0)))
    if codes is not None and not covered_count:
        raise DatasetError(f"{name.upper()} neutralization has no active classification among "
                           "finite eligible names in the selected test period; check classification "
                           "dates, symbol alignment, coverage, or select MARKET/NONE")
    if diagnostics is not None:
        diagnostics.update(max_abs_group_net_exposure=max_net,
                           classification_coverage=covered_count / eligible_count if eligible_count else 0.0,
                           inactive_neutralized_days=inactive_days)
    return w


def _truncate(weights: np.ndarray, cap: float) -> np.ndarray:
    if cap <= 0:
        return weights
    w = weights.copy()
    for _ in range(50):
        w = np.clip(w, -cap, cap)
        gross = np.abs(w).sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            w = np.where(gross > 0, w / gross, 0.0)
        if np.abs(w).max(initial=0) <= cap + 1e-9:
            break
    # too few names to respect the cap at a gross of 1: keep the cap, accept a smaller book
    return np.clip(w, -cap, cap)


def weights(alpha: np.ndarray, members: np.ndarray, tradable: np.ndarray, settings: Settings,
            groups: dict[str, np.ndarray], *, diagnostics: dict[str, Any] | None = None,
            test_start: int = 0) -> np.ndarray:
    """Daily weights, gross at most 1; diagnostics/availability cover test rows only.

    Classification coverage uses finite tradable signals after optional member
    masking: with pasteurization off, eligible nonmembers are included.
    """
    x = np.array(alpha, dtype=np.float64, copy=True)
    if settings.delay:
        shifted = np.full_like(x, np.nan)
        shifted[settings.delay:] = x[:-settings.delay]
        x = shifted
    if settings.decay > 0:
        x = ops.ts_decay_linear(x, settings.decay, dense=True).astype(np.float64)
    if settings.pasteurization:
        x[~members] = np.nan
    x[~tradable] = np.nan
    if settings.neutralization != "NONE":
        name = settings.neutralization.lower()
        if name == "market":
            codes = None
        else:
            group = groups.get(name)
            if group is None:
                raise DatasetError(f"{settings.neutralization} neutralization needs {name} classifications; "
                                   "supply classification groups for the selected dataset and test period")
            codes = np.broadcast_to(_group_array(group, x.shape, name), x.shape)
        return _neutral_weights(x, codes, float(settings.truncation), name=name,
                                diagnostics=diagnostics, test_start=test_start)
    x = np.where(np.isfinite(x), x, 0.0)
    gross = np.abs(x).sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(gross > 0, x / gross, 0.0)
    return _truncate(w, float(settings.truncation))


def _stats(pnl: np.ndarray, traded: np.ndarray, booksize: float) -> dict[str, Any]:
    days = len(pnl)
    if days < 2:
        return {"days": days}
    mean, std = float(np.mean(pnl)), float(np.std(pnl, ddof=1))
    sharpe = mean / std * math.sqrt(PERIODS_PER_YEAR) if std > 0 else None
    returns = mean * PERIODS_PER_YEAR / (booksize / 2)
    turnover = float(np.mean(traded))
    fitness = sharpe * math.sqrt(abs(returns) / max(turnover, 0.125)) if sharpe is not None else None
    cumulative = np.cumsum(pnl)
    drawdown = float(np.max(np.maximum.accumulate(np.concatenate(([0.0], cumulative)))[1:] - cumulative, initial=0.0))
    total_traded = float(np.sum(traded)) * booksize
    return {"days": days, "sharpe": sharpe, "returns": returns, "turnover": turnover, "fitness": fitness,
            "margin_bps": float(np.sum(pnl)) / total_traded * 1e4 if total_traded > 0 else None,
            "drawdown": drawdown / (booksize / 2), "pnl": float(np.sum(pnl))}


def simulate(expression: str, settings: Settings, workspace: Workspace, *, keep_weights: bool = False) -> dict[str, Any]:
    """Evaluate and simulate one alpha; return metrics, yearly table, checks and the daily PnL."""
    settings = settings.validate()
    members = workspace.universe(settings.universe)
    first, start, end = workspace.window(settings)
    context = workspace.context(first, end, members)
    alpha = evaluate(expression, context)
    close = context.fields["close"].astype(np.float64)
    tradable = np.isfinite(close)
    offset = start - first
    diagnostics: dict[str, Any] = {}
    w = weights(alpha, context.members, tradable, settings, context.groups,
                diagnostics=diagnostics, test_start=offset)
    with np.errstate(invalid="ignore", divide="ignore"):
        returns = np.full_like(close, np.nan)
        returns[1:] = close[1:] / close[:-1] - 1
    held = np.nan_to_num(returns, nan=0.0)
    # PnL of day t comes from weights held since day t-1; the test period starts at `start`
    gross_pnl = np.sum(w[offset - 1:-1] * held[offset:], axis=1) * settings.booksize if offset >= 1 else None
    if gross_pnl is None:
        raise DatasetError("the test period must start after the first evaluated session")
    traded = np.abs(np.diff(w[offset - 1:], axis=0)).sum(axis=1)  # traded on each test day
    pnl = gross_pnl - traded * settings.booksize * settings.cost_bps / 1e4
    dates = context.dates[offset:]
    test_w = w[offset:]
    longs = (test_w > 0).sum(axis=1)
    shorts = (test_w < 0).sum(axis=1)
    summary = _stats(pnl, traded, settings.booksize)
    summary.update(diagnostics)
    summary.update({
        "long_count": float(np.mean(longs)), "short_count": float(np.mean(shorts)),
        "gross_exposure": float(np.mean(np.abs(test_w).sum(axis=1))),
        "net_exposure": float(np.mean(test_w.sum(axis=1))),
        "max_weight": float(np.abs(test_w).max(initial=0.0)),
        "coverage": float(np.mean((longs + shorts) / np.maximum(context.members[offset:].sum(axis=1), 1))),
        "start": str(dates[0]), "end": str(dates[-1]),
    })
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    yearly = []
    for year in sorted(set(years.tolist())):
        rows = years == year
        stats = _stats(pnl[rows], traded[rows], settings.booksize)
        stats.update({"year": int(year), "long_count": float(np.mean(longs[rows])), "short_count": float(np.mean(shorts[rows]))})
        yearly.append(stats)
    periods = {}
    if settings.oos_start:
        cut = np.datetime64(settings.oos_start, "D")
        inside, outside = dates < cut, dates >= cut
        periods = {"in_sample": _stats(pnl[inside], traded[inside], settings.booksize),
                   "out_of_sample": _stats(pnl[outside], traded[outside], settings.booksize)}
    result = {
        "engine": ENGINE_VERSION, "expression": expression, "settings": asdict(settings),
        "dataset": workspace.identity, "summary": summary, "yearly": yearly, "periods": periods,
        "checks": checks(summary, settings),
        "daily": {"dates": [str(d) for d in dates], "pnl": pnl.astype(np.float64).tolist(),
                  "turnover": traded.astype(np.float64).tolist()},
    }
    if keep_weights:
        result["weights"] = test_w
        result["symbols"] = workspace.symbols
    return result


def checks(summary: dict[str, Any], settings: Settings) -> list[dict[str, Any]]:
    """BRAIN-like submission checks on local numbers (thresholds as documented by BRAIN; an approximation)."""
    def check(name, value, ok, limit):
        return {"name": name, "value": value, "limit": limit,
                "result": "PASS" if ok else ("FAIL" if value is not None else "ERROR")}

    sharpe, fitness, turnover = summary.get("sharpe"), summary.get("fitness"), summary.get("turnover")
    out = [
        check("LOW_SHARPE", sharpe, sharpe is not None and sharpe >= 1.25, 1.25),
        check("LOW_FITNESS", fitness, fitness is not None and fitness >= 1.0, 1.0),
        check("LOW_TURNOVER", turnover, turnover is not None and turnover >= 0.01, 0.01),
        check("HIGH_TURNOVER", turnover, turnover is not None and turnover <= 0.7, 0.7),
        check("CONCENTRATED_WEIGHT", summary.get("max_weight"),
              summary.get("max_weight") is not None and summary["max_weight"] <= max(settings.truncation, 0.1) + 1e-9,
              max(settings.truncation, 0.1)),
    ]
    return out


def json_safe(value: Any) -> Any:
    """Plain JSON types; non-finite numbers become null."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items() if k != "weights"}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def to_json(result: dict[str, Any]) -> str:
    return json.dumps(json_safe(result), allow_nan=False)


__all__ = ["Settings", "Workspace", "simulate", "weights", "checks", "ExpressionError", "ENGINE_VERSION"]
