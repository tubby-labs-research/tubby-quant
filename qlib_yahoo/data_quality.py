"""Measure what a snapshot's price data would do to a backtest, universe member-day by member-day.

A read-only scan: it never changes the snapshot. Checks, counted per year:

* possible unadjusted splits: close x3 or more (or /3) overnight while share volume
  moves the opposite way by 2x and dollar volume stays within 2x, the signature of a
  split the source has not adjusted (a crash also moves volume, but dollar volume jumps);
* other extreme moves: close doubling or halving overnight;
* stale prices: the same close five sessions or more in a row while volume trades;
* price level: member-days below $1 and $5 (unadjusted close when the snapshot has it);
* coverage: members per session, and member-days without a close;
* survivorship: the share of each year's members whose prices stop more than
  ENDED_SESSIONS sessions before the snapshot ends (delisted or acquired). A
  survivors-only source keeps this near zero; with delisted companies it is large
  in early years (roughly 40% of 2010's listed stocks no longer traded by 2026).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EXAMPLES = 15
SPLIT_PRICE = 3.0
SPLIT_VOLUME = 2.0
SPLIT_DOLLARS = 2.0
STALE_RUN = 5
ENDED_SESSIONS = 21


def scan(dataset: Path, universe: str | None = None) -> dict[str, Any]:
    import numpy as np

    from qlib_yahoo.panel import available_fields, load_panel, membership_mask
    from qlib_yahoo.universe import dataset_universes

    manifest = json.loads((Path(dataset) / "manifest.json").read_text(encoding="utf-8"))
    fields = ["open", "high", "low", "close", "volume"]
    has_raw = "close_raw" in available_fields(manifest)
    if has_raw:
        fields.append("close_raw")
    panel = load_panel(dataset, fields)
    universes = dataset_universes(panel.manifest)
    universe = universe or (sorted(universes)[0] if universes else None)
    close = panel.fields["close"].astype(np.float64)
    volume = panel.fields["volume"].astype(np.float64)
    member = membership_mask(panel, universe) if universe else np.isfinite(close)
    years = panel.dates.astype("datetime64[Y]").astype(int) + 1970

    with np.errstate(divide="ignore", invalid="ignore"):
        move = close[1:] / close[:-1]
        volume_move = volume[1:] / volume[:-1]
    both = np.isfinite(move) & np.isfinite(volume_move) & (volume[:-1] > 0) & (volume[1:] > 0)
    with np.errstate(invalid="ignore"):
        dollars_kept = (move * volume_move >= 1 / SPLIT_DOLLARS) & (move * volume_move <= SPLIT_DOLLARS)
    split_up = both & dollars_kept & (move >= SPLIT_PRICE) & (volume_move <= 1 / SPLIT_VOLUME)
    split_down = both & dollars_kept & (move <= 1 / SPLIT_PRICE) & (volume_move >= SPLIT_VOLUME)
    split = split_up | split_down
    extreme = np.isfinite(move) & ((move >= 2) | (move <= 0.5)) & ~split
    member_next = member[1:]
    move_years = years[1:]

    stale = np.zeros_like(close, dtype=bool)  # the fifth and later repeats of a traded close
    same = np.zeros_like(close, dtype=bool)
    same[1:] = (close[1:] == close[:-1]) & (volume[1:] > 0)
    run = np.zeros(close.shape[1], dtype=np.int32)
    for row in range(close.shape[0]):
        run = np.where(same[row], run + 1, 0)
        stale[row] = run >= STALE_RUN - 1

    price = panel.fields["close_raw"].astype(np.float64) if has_raw else close
    with np.errstate(invalid="ignore"):
        bad_ohlc = (np.isfinite(close) & ((panel.fields["high"] < np.maximum(panel.fields["open"], panel.fields["close"]))
                                          | (panel.fields["low"] > np.minimum(panel.fields["open"], panel.fields["close"]))))

    listed = np.isfinite(close)
    last_row = np.where(listed.any(axis=0), close.shape[0] - 1 - np.argmax(listed[::-1], axis=0), -1)
    ended = last_row < close.shape[0] - 1 - ENDED_SESSIONS

    def examples(mask: Any, shift: int = 1) -> list[str]:
        rows, columns = np.nonzero(mask)
        order = np.argsort(-rows, kind="stable")[:EXAMPLES]  # most recent first
        out = []
        for r, c in zip(rows[order], columns[order]):
            day = str(panel.dates[r + shift])
            ratio = move[r, c] if shift else float("nan")
            out.append(f"{panel.symbols[c]} {day}" + (f" x{ratio:.3g}" if shift and np.isfinite(ratio) else ""))
        return out

    by_year = []
    for year in sorted(set(years.tolist())):
        rows = years == year
        mrows = move_years == year
        member_days = int(member[rows].sum())
        if not member_days:
            continue
        members_per_day = member[rows].sum(axis=1)
        distinct = member[rows].any(axis=0)
        with np.errstate(invalid="ignore"):
            below_1 = int((member[rows] & (price[rows] < 1)).sum())
            below_5 = int((member[rows] & (price[rows] < 5)).sum())
        by_year.append({
            "year": int(year), "member_days": member_days,
            "members_per_session": {"min": int(members_per_day.min()), "median": int(np.median(members_per_day)),
                                    "max": int(members_per_day.max())},
            "possible_unadjusted_splits": int(split[mrows].sum()),
            "possible_unadjusted_splits_in_universe": int((split & member_next)[mrows].sum()),
            "extreme_moves_in_universe": int((extreme & member_next)[mrows].sum()),
            "stale_member_days": int((stale & member)[rows].sum()),
            "member_days_below_1": below_1, "member_days_below_5": below_5,
            "share_below_1": round(below_1 / member_days, 5), "share_below_5": round(below_5 / member_days, 5),
            "member_days_without_close": int((member[rows] & ~np.isfinite(close[rows])).sum()),
            "distinct_members": int(distinct.sum()), "members_that_ended": int((distinct & ended).sum()),
            "share_ended": round(float((distinct & ended).sum() / max(distinct.sum(), 1)), 4),
        })
    return {
        "dataset": str(panel.dataset), "identity": panel.identity, "universe": universe,
        "price_basis": "unadjusted close (close_raw)" if has_raw else "adjusted close (no close_raw field)",
        "rules": {"possible_unadjusted_split": f"close x{SPLIT_PRICE:g}+ or /{SPLIT_PRICE:g} overnight with volume moving "
                                               f"the opposite way by {SPLIT_VOLUME:g}x and dollar volume within {SPLIT_DOLLARS:g}x",
                  "extreme_move": "close doubles or halves overnight (not counted as a split signature)",
                  "stale": f"the same close {STALE_RUN}+ sessions in a row while volume trades",
                  "ended": f"a member whose last close is more than {ENDED_SESSIONS} sessions before the snapshot's last "
                           "session (delisted, acquired or stopped trading)"},
        "totals": {
            "possible_unadjusted_splits": int(split.sum()),
            "possible_unadjusted_reverse_splits": int(split_up.sum()),
            "possible_unadjusted_splits_in_universe": int((split & member_next).sum()),
            "extreme_moves_in_universe": int((extreme & member_next).sum()),
            "stale_member_days": int((stale & member).sum()),
            "inconsistent_ohlc_bars": int(bad_ohlc.sum()),
            "members_that_ended": int((member.any(axis=0) & ended).sum()),
        },
        "examples": {"possible_unadjusted_splits": examples(split),
                     "extreme_moves_in_universe": examples(extreme & member_next)},
        "by_year": by_year,
    }


def summary_lines(report: dict[str, Any]) -> list[str]:
    totals = report["totals"]
    lines = [f"{report['dataset']} ({report['universe'] or 'all instruments'}; price basis: {report['price_basis']})",
             f"possible unadjusted splits: {totals['possible_unadjusted_splits']} "
             f"({totals['possible_unadjusted_splits_in_universe']} in the universe); "
             f"extreme moves in the universe: {totals['extreme_moves_in_universe']}; "
             f"stale member-days: {totals['stale_member_days']}; inconsistent OHLC bars: {totals['inconsistent_ohlc_bars']}",
             "year  members(med)  splits(univ)  extreme  stale  <$1    <$5    ended"]
    for row in report["by_year"]:
        lines.append(f"{row['year']}  {row['members_per_session']['median']:>12,}  "
                     f"{row['possible_unadjusted_splits_in_universe']:>12}  {row['extreme_moves_in_universe']:>7}  "
                     f"{row['stale_member_days']:>5}  {row['share_below_1']:>5.1%}  {row['share_below_5']:>5.1%}  "
                     f"{row['share_ended']:>5.1%}")
    if report["examples"]["possible_unadjusted_splits"]:
        lines.append("recent split signatures: " + ", ".join(report["examples"]["possible_unadjusted_splits"][:8]))
    return lines

