"""Reconcile Stooq's adjusted bars with Yahoo split/dividend events before a snapshot is built.

Stooq back-adjusts prices for splits and dividends, but splits from the last weeks
before a download can still be missing: a 1:125 reverse split then looks like a
+12,400% return. Every Yahoo split inside a ticker's history is checked against
the Stooq bars on either side of it:

* the jump matches the unadjusted ratio: bars before the split are adjusted
  (prices divided by the ratio, volume multiplied by it), the same convention
  Stooq uses for older splits;
* Stooq moves like Yahoo's split-adjusted close: already adjusted, nothing changes;
* anything else: ambiguous; the ticker's bars before the split are masked so the
  return across it cannot enter a backtest.

The same pass derives the unadjusted close (Yahoo's split-only close times later
split ratios) for price floors and market capitalisation, the cash dividend on each
ex-date, and a price cross-check against Yahoo's fully adjusted close.
"""

from __future__ import annotations

import math
from typing import Any, Callable

from qlib_yahoo.dataset import DatasetError
from qlib_yahoo.yahoo_chart import ChartArchive, yahoo_symbol
from qlib_yahoo.yahoo_table import normalized

EXAMPLES = 20
# Smaller ratios are stock dividends or spin-offs that Yahoo encodes as splits; they cannot be told
# from ordinary moves (the same threshold as the Yahoo table builder).
MIN_SPLIT_RATIO = 1.5
MISMATCH_LOG = math.log(1.10)  # a close >10% away from the ticker's usual Stooq/Yahoo ratio
# Both sources quote the same trades, so Stooq's move across a split must match Yahoo's (adjusted or
# not) within this factor; it tolerates the 2-decimal rounding Stooq applies to sub-dollar prices.
SPLIT_MOVE_TOLERANCE = math.log(1.25)


def _examples(items: list[Any]) -> list[Any]:
    return items[:EXAMPLES]


def reconcile(frame: Any, invalid: Any, archive: ChartArchive, *,
              progress: Callable[[str], None] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Repair unadjusted splits in ``frame`` in place and return (columns, report).

    ``frame`` is the importer's long frame sorted by (ticker, date) with float64
    OHLCV; ``invalid`` marks unusable rows (left untouched, stays masked). The
    returned columns are row-aligned arrays: ``close_raw`` (unadjusted close),
    ``dividend`` (cash per share on ex-dates, 0 otherwise, unadjusted) and
    ``masked`` (rows to hide because a split could not be reconciled).
    """
    import numpy as np

    rows = len(frame)
    close_raw = np.full(rows, np.nan)
    dividend = np.zeros(rows)
    masked = np.zeros(rows, dtype=bool)
    codes = frame["ticker"].cat.codes.to_numpy()
    names = frame["ticker"].cat.categories
    days = frame["date"].to_numpy().astype("datetime64[D]")
    # Copies: Arrow-backed columns can be read-only; repaired values are written back at the end.
    prices = {name: frame[name].to_numpy(dtype=np.float64, copy=True) for name in ("open", "high", "low", "close")}
    volume = frame["volume"].to_numpy(dtype=np.float64, copy=True)
    edges = np.flatnonzero(np.diff(codes)) + 1
    starts, stops = np.concatenate(([0], edges)), np.concatenate((edges, [rows]))

    applied, adjusted, ambiguous, too_small, outside = [], [], [], [], []
    no_yahoo, raw_from_stooq, checked = [], 0, 0
    mismatch_tickers, mismatch_days, compared_days = [], 0, 0
    for number, (first, stop) in enumerate(zip(starts, stops)):
        ticker = str(names[codes[first]])
        history = archive.history(ticker)
        span = slice(first, stop)
        if history is None or not history.dates:
            no_yahoo.append(ticker)
            close_raw[span] = np.where(invalid[span], np.nan, prices["close"][span])
            raw_from_stooq += int((~invalid[span]).sum())
            continue
        # Yahoo's own series, with the splits Yahoo left unadjusted repaired and dated at the real jump.
        yahoo = normalized(history)
        ydays, yclose, yadj = yahoo["days"], yahoo["close"], yahoo["adjclose"]
        sdays = days[span]
        good = ~invalid[span] & np.isfinite(prices["close"][span])

        for boundary, ratio, _treatment in yahoo["events"]:
            if boundary >= len(ydays):
                continue
            when = ydays[boundary]
            if not (sdays[0] < when <= sdays[-1]):
                continue
            checked += 1
            shares = f"{ratio:.4g}:1" if ratio >= 1 else f"1:{1 / ratio:.4g}"  # new shares per old share
            label = f"{ticker} {when} {shares}"
            if max(ratio, 1 / ratio) < MIN_SPLIT_RATIO:
                too_small.append(label)
                continue
            cut = int(np.searchsorted(sdays, when))
            before = np.flatnonzero(good[:cut])
            after = np.flatnonzero(good[cut:])
            if not len(before) or not len(after):
                outside.append(label)
                continue
            prev, on = first + int(before[-1]), first + cut + int(after[0])
            where = np.searchsorted(ydays, days[[prev, on]])
            ok = (where < len(ydays)) & (ydays[np.minimum(where, len(ydays) - 1)] == days[[prev, on]])
            yp, yo = (yclose[where[0]], yclose[where[1]]) if ok.all() else (np.nan, np.nan)
            stooq_move = math.log(prices["close"][on] / prices["close"][prev])
            if not (np.isfinite(yp) and np.isfinite(yo) and yp > 0 and yo > 0):
                ambiguous.append(f"{label} (no Yahoo close beside the split)")
                masked[first:first + cut] = True
                continue
            # Yahoo's move across the split is the ordinary market move, unless Yahoo's own series
            # jumps there too (a mis-recorded ratio, e.g. AMTX 2016): then assume no market move.
            yahoo_move = math.log(yo / yp)
            adjusted_move = yahoo_move if abs(yahoo_move) < abs(math.log(ratio)) / 2 else 0.0
            unadjusted_move = adjusted_move - math.log(ratio)
            tolerance = min(abs(math.log(ratio)) / 2, SPLIT_MOVE_TOLERANCE)
            if abs(stooq_move - unadjusted_move) < tolerance:
                block = slice(first, first + cut)
                for name in ("open", "high", "low", "close"):
                    prices[name][block] /= ratio
                volume[block] *= ratio
                applied.append(f"{label} (Stooq jump x{math.exp(stooq_move):.3g})")
            elif abs(stooq_move - adjusted_move) < tolerance:
                adjusted.append(label)
            else:
                ambiguous.append(f"{label} (Stooq x{math.exp(stooq_move):.3g}, Yahoo x{math.exp(yahoo_move):.3g})")
                masked[first:first + cut] = True

        # Unadjusted close: Yahoo's split-only close times every later split ratio.
        later = yahoo["later"]
        where = np.minimum(np.searchsorted(ydays, sdays), len(ydays) - 1)
        present = ydays[where] == sdays
        raw = np.where(present, yclose[where] * later[where], np.nan)
        # Days Yahoo lacks: Stooq's (now split-consistent) close scaled by splits after that day.
        fallback = ~np.isfinite(raw) & good
        if fallback.any():
            stooq_later = np.ones(int(fallback.sum()))
            fallback_days = sdays[fallback]
            for boundary, ratio, _treatment in yahoo["events"]:
                if boundary < len(ydays) and ydays[boundary] <= sdays[-1]:  # Stooq's basis lacks later splits
                    stooq_later[fallback_days < ydays[boundary]] *= ratio
            raw[fallback] = prices["close"][span][fallback] * stooq_later
            raw_from_stooq += int(fallback.sum())
        close_raw[span] = np.where(good, raw, np.nan)

        for ex_day, amount in history.dividends:
            slot = int(np.searchsorted(sdays, np.datetime64(ex_day, "D")))
            if slot < len(sdays) and sdays[slot] == np.datetime64(ex_day, "D"):
                position = int(np.searchsorted(ydays, np.datetime64(ex_day, "D")))
                later_ratio = later[position] if position < len(ydays) else 1.0
                dividend[first + slot] += amount * later_ratio

        # Cross-check against Yahoo's fully adjusted close on shared, valid days.
        adj = np.where(present, yadj[where], np.nan)
        both = good & np.isfinite(adj) & (adj > 0) & (prices["close"][span] > 0)
        if both.sum() >= 20:
            logs = np.log(prices["close"][span][both] / adj[both])
            off = np.abs(logs - np.median(logs)) > MISMATCH_LOG
            compared_days += int(both.sum())
            mismatch_days += int(off.sum())
            if off.mean() > 0.05:
                mismatch_tickers.append((ticker, round(float(off.mean()), 3)))
        if progress and (number + 1) % 1000 == 0:
            progress(f"Reconciled {number + 1:,}/{len(starts):,} tickers with Yahoo events")

    for name, values in prices.items():
        frame[name] = values
    frame["volume"] = volume
    mismatch_tickers.sort(key=lambda item: -item[1])
    report = {
        "source": archive.identity(),
        "splits_in_history": checked,
        "unadjusted_repaired": {"count": len(applied), "examples": _examples(applied)},
        "already_adjusted": {"count": len(adjusted), "examples": _examples(adjusted)},
        "ambiguous_masked_before_split": {"count": len(ambiguous), "examples": _examples(ambiguous)},
        "below_ratio_threshold": {"count": len(too_small), "threshold": MIN_SPLIT_RATIO, "examples": _examples(too_small)},
        "no_bars_on_one_side": {"count": len(outside), "examples": _examples(outside)},
        "tickers_without_yahoo_history": {"count": len(no_yahoo), "examples": _examples(no_yahoo)},
        "close_raw": {"rule": "Yahoo split-only close x later split ratios; Stooq close x later split ratios "
                              "where Yahoo lacks the day (dividend-adjusted, slightly low)",
                      "days_from_stooq_fallback": raw_from_stooq},
        "dividend": "cash per share on the ex-date in the unadjusted share basis; 0 on other days",
        "price_cross_check": {
            "rule": "Stooq adjusted close / Yahoo adjusted close, deviation from the ticker's median ratio > 10%",
            "days_compared": compared_days, "days_off": mismatch_days,
            "tickers_over_5pct_of_days": {"count": len(mismatch_tickers),
                                          "examples": [f"{t} {share:.1%}" for t, share in _examples(mismatch_tickers)]},
        },
    }
    if len(applied) + len(adjusted) + len(ambiguous) + len(too_small) + len(outside) != checked:
        raise DatasetError("split reconciliation lost track of an event")
    return {"close_raw": close_raw, "dividend": dividend, "masked": masked}, report


def yahoo_type_exclusions(archive: ChartArchive, tickers: list[str]) -> dict[str, str]:
    """Tickers Yahoo classifies as something other than common equity (ETFs listed as 'stock' by Stooq)."""
    records = archive.manifest["tickers"]
    excluded = {}
    for ticker in tickers:
        kind = (records.get(yahoo_symbol(ticker)) or {}).get("instrument_type")
        if kind and kind != "EQUITY":
            excluded[ticker] = f"yahoo_type_{str(kind).lower()}"
    return excluded
