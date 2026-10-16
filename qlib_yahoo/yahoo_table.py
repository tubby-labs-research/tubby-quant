"""Build the consolidated daily table from a Yahoo chart archive (the preferred price source).

Why Yahoo instead of Stooq's bulk file: Stooq stopped dividend-adjusting most
histories around late 2022 (returns switch from total to price returns) and misses
recent splits, while Yahoo's chart data is adjusted consistently and lists every
split and dividend. The table's conventions:

* ``open/high/low/close`` are split- and dividend-adjusted; ``volume`` is scaled the
  other way so close x volume is the traded dollar volume;
* ``close_raw`` is the traded close and ``dividend`` the cash per share on ex-dates;
  ``close / close_raw`` is the adjustment factor. Qlib's own ``factor`` field stays 1:
  the runner's share-rounding contract (and signed mode) requires it.

Securities are classified before import: Yahoo's type drops ETFs, the Nasdaq
screener's security names drop warrants, rights, units, preferreds, closed-end
funds and SPACs, and Nasdaq fifth-letter codes back that up. Bars on or after the
fetch date are dropped (they can be partial), as are isolated one-day spikes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from qlib_yahoo.dataset import DatasetError
from qlib_yahoo.reference import Screener
from qlib_yahoo.yahoo_chart import ChartArchive

EXAMPLES = 20
EXCHANGES = {"NYQ": "NYSE", "NMS": "NASDAQ", "NGM": "NASDAQ", "NCM": "NASDAQ", "ASE": "NYSEMKT",
             "PCX": "NYSEARCA", "BTS": "CBOE"}
NASDAQ_CODES = frozenset({"NMS", "NGM", "NCM"})
# Nasdaq fifth letters that mark a non-common security of a four-letter base ticker.
NASDAQ_NON_COMMON = {"M": "preferred", "N": "preferred", "O": "preferred", "P": "preferred", "R": "right",
                     "T": "with_warrants", "U": "unit", "V": "when_issued", "W": "warrant", "X": "fund"}
NAME_RULES = (  # in order: a SPAC unit's name also mentions its warrant or right
    ("unit", re.compile(r"\bunits?\b", re.I)),
    ("warrant", re.compile(r"\bwarrants?\b", re.I)),
    ("right", re.compile(r"\brights?\b", re.I)),
    ("preferred", re.compile(r"\bpreferred\b|\bpfd\b|\bcumulative\b|\d%|\bnotes? due\b|\bdebentures?\b|\bsenior notes\b", re.I)),
    ("fund", re.compile(r"\bfunds?\b|\bmunicipal\b|shares of beneficial interest|\betf\b|\betn\b", re.I)),
)
# Equity that the name rules would otherwise catch: partnership common units, royalty-trust units,
# and REITs whose shares are "common shares of beneficial interest".
EQUITY_UNITS = re.compile(r"common units?|limited partner|units? of beneficial interest", re.I)
FUND_WORD = re.compile(r"\bfunds?\b|\betf\b|\betn\b", re.I)
SPIKE = math.log(2.0)      # an overnight move of 2x or 1/2 ...
REVERT = math.log(1.25)    # ... that is back within 25% the next day is a bad print


def classify(symbol: str, meta: dict[str, Any], listing: dict[str, str] | None, universe: set[str]) -> str:
    """'stock' for common equity, otherwise the reason it is not."""
    kind = meta.get("instrumentType")
    if kind != "EQUITY":
        return f"yahoo_{str(kind or 'unknown').lower()}"
    if meta.get("currency") != "USD":
        return "non_usd"
    exchange = meta.get("exchangeName")
    if exchange not in EXCHANGES:
        return "not_exchange_listed"
    if listing is not None:
        if (listing.get("industry") or "") == "Blank Checks":
            return "spac"
        name = listing.get("name") or ""
        real_estate = "real estate" in (listing.get("industry") or "").lower() or \
            (listing.get("sector") or "").lower() == "real estate"
        for reason, pattern in NAME_RULES:
            if not pattern.search(name):
                continue
            if reason == "unit" and EQUITY_UNITS.search(name):
                continue
            if reason == "fund" and real_estate and not FUND_WORD.search(name):
                continue
            return reason
    base = symbol.split("-", 1)[0]
    if exchange in NASDAQ_CODES and len(base) == 5 and base[:4] in universe and base[4] in NASDAQ_NON_COMMON:
        return NASDAQ_NON_COMMON[base[4]]
    if len(base) == 6 and base.endswith("WW"):
        return "warrant"
    return "stock"


SPLIT_WINDOW = (10, 3)          # sessions before and after Yahoo's split date searched for the price jump
SPLIT_MATCH = math.log(1.3)     # the jump must match the split ratio within 30%
SPLIT_MINIMUM = 1.5             # smaller ratios (stock dividends) cannot be told from ordinary moves


def _reconcile_splits(days: Any, close: Any, splits: list[tuple[str, float, float]]) -> list[tuple[int, float, str]]:
    """For each split, (first session on the new share basis, ratio, how Yahoo's series treats it).

    Yahoo's split-only close is usually continuous across a split ("adjusted"). Sometimes,
    mostly for small reverse splits, the series still jumps by the split ratio and the event
    is dated days or weeks away from the jump ("unadjusted"): the jump's session is then the
    real boundary and the bars before it must be adjusted here.
    """
    import numpy as np

    events = []
    for split_day, numerator, denominator in splits:
        ratio = numerator / denominator
        at = int(np.searchsorted(days, np.datetime64(split_day, "D")))
        if max(ratio, 1 / ratio) < SPLIT_MINIMUM or at >= len(days):
            events.append((at, ratio, "adjusted"))
            continue
        target = -math.log(ratio)  # a 1:10 reverse split multiplies the unadjusted price by 10
        best, best_gap = None, None
        for t in range(max(1, at - SPLIT_WINDOW[0]), min(len(days), at + SPLIT_WINDOW[1] + 1)):
            if not (close[t] > 0 and close[t - 1] > 0):
                continue
            gap = abs(math.log(close[t] / close[t - 1]) - target)
            if best_gap is None or gap < best_gap:
                best, best_gap = t, gap
        if best is not None and best_gap < SPLIT_MATCH:
            events.append((best, ratio, "unadjusted"))
        else:
            events.append((at, ratio, "adjusted"))
    return events


def normalized(history: Any, cutoff: str | None = None) -> dict[str, Any]:
    """Yahoo's daily values with Yahoo's own unadjusted splits repaired.

    Returns ``days``, ``open high low close adjclose volume`` (split-consistent), ``events``
    as (first session on the new share basis, ratio, treatment) and ``later``: the product
    of the ratios of splits taking effect after each day, so ``close x later`` is the
    traded price.
    """
    import numpy as np

    keep = np.asarray([cutoff is None or day < cutoff for day in history.dates], dtype=bool)
    days = np.asarray(history.dates, dtype="datetime64[D]")[keep]
    values = {name: np.asarray([np.nan if v is None else v for v in getattr(history, name)], dtype=np.float64)[keep]
              for name in ("open", "high", "low", "close", "adjclose", "volume")}
    events = _reconcile_splits(days, values["close"], history.splits)
    later = np.ones(len(days))
    for boundary, ratio, kind in events:
        if kind == "unadjusted":  # put the bars before the jump on the new share basis, as Yahoo does elsewhere
            for name in ("open", "high", "low", "close", "adjclose"):
                values[name][:boundary] /= ratio
            values["volume"][:boundary] *= ratio
        later[:boundary] *= ratio
    return {"days": days, "events": events, "later": later, **values}


def _series(history: Any, cutoff: str) -> dict[str, Any]:
    """Adjusted OHLCV, traded close and dividends for one ticker's history before ``cutoff``."""
    import numpy as np

    values = normalized(history, cutoff)
    days, events, later = values["days"], values["events"], values["later"]
    with np.errstate(invalid="ignore", divide="ignore"):
        dividend_factor = values["adjclose"] / values["close"]  # adjusted / split-only close
        raw_close = values["close"] * later
        out = {name: values[name] * dividend_factor for name in ("open", "high", "low", "close")}
        out["volume"] = values["volume"] / dividend_factor
    out["close_raw"] = raw_close
    dividend = np.zeros(len(days))
    for ex_day, amount in history.dividends:
        slot = int(np.searchsorted(days, np.datetime64(ex_day, "D")))
        if slot < len(days) and days[slot] == np.datetime64(ex_day, "D"):
            dividend[slot] += amount * later[slot]  # split-adjusted amount back to that day's shares
    out["dividend"] = dividend
    out["date"] = days
    out["split_events"] = [(str(days[b]) if b < len(days) else "after", r, k) for b, r, k in events]
    return out


def _level_breaks(close: Any, volume: Any) -> Any:
    """Index of each unexplained upward level shift: the signature of a reverse split Yahoo never recorded.

    The close rises 3x or more overnight and stays up, share volume falls by half or more,
    and dollar volume stays within 2x (a real rally brings a surge in dollar volume).
    Downward shifts are left alone: they are often genuine crashes, and hiding a crash
    would flatter every backtest that held the stock.
    """
    import numpy as np

    breaks = []
    with np.errstate(invalid="ignore", divide="ignore"):
        move = close[1:] / close[:-1]
        shares = volume[1:] / volume[:-1]
    candidates = np.flatnonzero((move >= 3) & (shares <= 0.5) & (move * shares >= 0.5) & (move * shares <= 2))
    for i in candidates:
        after = close[i + 1:i + 7]
        after = after[np.isfinite(after)]
        if len(after) >= 3 and np.median(after) > close[i] * 2:
            breaks.append(int(i + 1))
    return breaks


def _spikes(close: Any) -> Any:
    """Bars that jump 2x (or halve) and are back within 25% of the prior close the next day."""
    import numpy as np

    flags = np.zeros(len(close), dtype=bool)
    if len(close) < 3:
        return flags
    with np.errstate(invalid="ignore", divide="ignore"):
        jump = np.abs(np.log(close[1:-1] / close[:-2]))
        back = np.abs(np.log(close[2:] / close[:-2]))
    flags[1:-1] = (jump > SPIKE) & (back < REVERT)
    return flags


def build_table(archive_path: Path, screener_path: Path, destination: Path, *,
                progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Write ``destination`` (Parquet) and ``destination``.manifest.json; return the manifest."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    archive, screener = ChartArchive(archive_path), Screener(screener_path)
    destination = Path(destination)
    if destination.exists():
        raise DatasetError(f"{destination} exists; choose a new file")
    fetched = sorted(r["fetched_at_utc"] for r in archive.manifest["tickers"].values() if r.get("status") == 200)
    # Bars from the first fetch day on may be partial sessions: drop them. UTC-5 (US Eastern
    # standard time) is used all year; in summer it can only move the cutoff a day earlier.
    first_fetch = datetime.fromisoformat(fetched[0]).astimezone(timezone.utc)
    cutoff_day = datetime.fromtimestamp(first_fetch.timestamp() - 5 * 3600, timezone.utc).date().isoformat()
    symbols = archive.symbols()
    universe = set(symbols)
    schema = pa.schema([("ticker", pa.string()), ("exchange", pa.string()), ("type", pa.string()),
                        ("date", pa.timestamp("s")), *[(name, pa.float64()) for name in
                        ("open", "high", "low", "close", "volume", "close_raw", "dividend")]])
    reasons: dict[str, list[str]] = {}
    spikes, spike_rows, rows_written, missing_listing, breaks = [], 0, 0, [], []
    split_kinds: dict[str, list[str]] = {"adjusted": [], "unadjusted": []}
    temporary = destination.with_name(destination.name + ".tmp")
    with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
        batch: list[Any] = []
        for number, symbol in enumerate(symbols):
            history = archive.history(symbol)
            listing = screener.rows.get(symbol)
            if listing is None:
                missing_listing.append(symbol)
            kind = classify(symbol, history.meta, listing, universe)
            reasons.setdefault(kind, []).append(symbol)
            series = _series(history, cutoff_day)
            for day, ratio, treatment in series.pop("split_events"):
                split_kinds[treatment].append(f"{symbol} {day} x{1 / ratio:.4g}")
            if not len(series["date"]):
                continue
            flags = _spikes(series["close"])
            if flags.any():
                spike_rows += int(flags.sum())
                spikes.extend(f"{symbol} {day}" for day in series["date"][flags].astype(str))
                for name in ("open", "high", "low", "close", "volume", "close_raw"):
                    series[name] = np.where(flags, np.nan, series[name])
            for index in _level_breaks(series["close"], series["volume"]):
                # an unrecorded reverse split: start the series again at the jump rather than guess a ratio
                breaks.append(f"{symbol} {series['date'][index]} x{series['close'][index] / series['close'][index - 1]:.3g}")
                for name in ("open", "high", "low", "close", "volume", "close_raw"):
                    series[name][:index] = np.nan
            count = len(series["date"])
            batch.append(pa.table({
                "ticker": pa.array([symbol] * count, pa.string()),
                "exchange": pa.array([EXCHANGES.get(history.meta.get("exchangeName"), "OTHER")] * count, pa.string()),
                "type": pa.array([kind] * count, pa.string()),
                "date": pa.array(series["date"].astype("datetime64[s]"), pa.timestamp("s")),
                **{name: pa.array(series[name], pa.float64()) for name in
                   ("open", "high", "low", "close", "volume", "close_raw", "dividend")}}, schema=schema))
            rows_written += count
            if len(batch) >= 200:
                writer.write_table(pa.concat_tables(batch))
                batch = []
            if progress and (number + 1) % 1000 == 0:
                progress(f"Built {number + 1:,}/{len(symbols):,} tickers")
        if batch:
            writer.write_table(pa.concat_tables(batch))
    os.replace(temporary, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    manifest = {
        "schema": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "table": str(destination),
        "table_sha256": digest, "rows": rows_written,
        "sources": {"yahoo_chart": archive.identity(), "nasdaq_screener": screener.identity()},
        "conventions": {"prices": "split- and dividend-adjusted (Yahoo adjclose basis)",
                        "volume": "shares scaled so close x volume is traded dollar volume",
                        "close_raw": "traded close; close / close_raw is the adjustment factor",
                        "dividend": "cash per share on the ex-date, traded share basis",
                        "factor": "not supplied: the snapshot keeps Qlib's identity factor (1)"},
        "cutoff": {"rule": "bars dated on or after the first fetch day (UTC-5) are dropped as possibly partial",
                   "first_fetch_utc": fetched[0], "dropped_from": cutoff_day},
        "classification": {reason: {"count": len(names), "examples": names[:EXAMPLES]}
                           for reason, names in sorted(reasons.items())},
        "not_in_nasdaq_screener": {"count": len(missing_listing), "examples": missing_listing[:EXAMPLES]},
        "splits": {"rule": "a split whose price jump (within 10 sessions before to 3 after Yahoo's date) matches the ratio "
                           "within 30% is unadjusted in Yahoo's series; bars before the jump are adjusted here",
                   "adjusted_by_yahoo": len(split_kinds["adjusted"]),
                   "repaired_here": {"count": len(split_kinds["unadjusted"]), "examples": split_kinds["unadjusted"][-EXAMPLES:]}},
        "level_breaks": {"rule": "close x3+ that persists, share volume halved or less, dollar volume within 2x, and no "
                                 "Yahoo split to explain it: bars before the jump are set to NaN (series restarts)",
                         "count": len(breaks), "examples": breaks[-EXAMPLES:]},
        "isolated_spikes": {"rule": "close moves 2x or 1/2 and is back within 25% of the prior close next day; "
                                    "the bar is set to NaN", "rows": spike_rows, "examples": spikes[-EXAMPLES:]},
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    manifest_path = destination.with_name(destination.name + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest
