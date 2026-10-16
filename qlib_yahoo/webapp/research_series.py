"""Validate retained fixed-notional series before exposing them to charts."""
from __future__ import annotations

import math

from autoresearch.backend import _date


def period_series(artifact, result, period):
    row = artifact[period]
    daily = row["daily"]
    dates, pnl, turnover = [daily[key] for key in ("dates", "pnl", "turnover")]
    if (not all(isinstance(v, list) for v in (dates, pnl, turnover)) or not 1 <= len(dates) <= 20_000
            or len(pnl) != len(dates) or len(turnover) != len(dates)):
        raise ValueError("Invalid daily series dimensions")
    settings = result["settings"]
    start, end = settings[period + "_start"], settings[period + "_end"]
    previous = None
    for day in dates:
        _date(day, "daily date")
        if not start <= day <= end or (previous is not None and day <= previous):
            raise ValueError("Daily dates must match the retained chronological period")
        previous = day
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in pnl + turnover):
        raise ValueError("Daily observations must be finite numbers")
    if any(v < 0 for v in turnover):
        raise ValueError("Daily turnover must be nonnegative")
    summary = result["metrics"][period]
    if (row["summary"] != summary or summary.get("days") != len(dates)
            or type(summary.get("pnl")) not in (int, float) or not math.isfinite(summary["pnl"])
            or not math.isclose(math.fsum(pnl), summary["pnl"], rel_tol=1e-9, abs_tol=1e-6)):
        raise ValueError("Daily PnL does not match the retained test summary")
    cumulative, drawdown = [], []
    total, peak = 0.0, 0.0
    for value in pnl:
        total += value
        peak = max(peak, total)
        if not math.isfinite(total) or not math.isfinite(total - peak):
            raise ValueError("Daily totals exceed chart numeric bounds")
        cumulative.append(total)
        drawdown.append(total - peak)
    return {"dates": dates, "pnl": pnl, "turnover": turnover,
            "cumulative": cumulative, "drawdown": drawdown}
