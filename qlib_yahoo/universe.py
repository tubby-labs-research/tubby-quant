"""Causal, dated liquidity universes (for example a local TOP3000 proxy).

Membership is ranked from data available at a session close and becomes
effective from the next session, so a signal on day t never uses a ranking
that includes information after t. The rule is a documented local proxy,
not a reproduction of WorldQuant Brain TOP3000.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qlib_yahoo.dataset import DatasetError


UNIVERSE_NAME = re.compile(r"[a-z][a-z0-9_]{0,31}\Z")
COMMON_TICKER = re.compile(r"[A-Z][A-Z0-9]*(?:-[A-Z])?\Z")
# Single-letter suffixes that denote warrants, units, rights and when-issued lines.
NON_COMMON_SUFFIXES = frozenset("WURV")
WINDOWS_RESERVED = frozenset({"CON", "PRN", "AUX", "NUL"} | {f"{p}{n}" for p in ("COM", "LPT") for n in range(10)})
# Exchange test instruments carried by public feeds at placeholder prices (Stooq quoted
# ZVZZT at 199,999.99): Nasdaq Z?ZZT, NYSE/Cboe/IEX test lines and *TEST placeholders.
TEST_TICKER = re.compile(r"Z[A-Z]ZZ[TC]\Z")
TEST_TICKERS = frozenset({"ZBZX", "ZVV", "ZXIET", "ZIEXT", "ZEXIT", "ATEST", "CTEST", "MTEST", "NTEST", "PTEST", "ZTEST"})


@dataclass(frozen=True)
class UniverseRule:
    """Top ``size`` stocks by trailing average daily dollar volume, reconstituted monthly."""

    name: str = "top3000"
    size: int = 3000
    adv_sessions: int = 20
    min_history_sessions: int = 252
    min_price: float = 0.0  # unadjusted close on the ranking session; 0 keeps every price

    def validate(self) -> "UniverseRule":
        if not UNIVERSE_NAME.fullmatch(self.name) or self.name == "all":
            raise DatasetError("universe name must be lowercase letters/digits/underscore and not 'all'")
        for label, value, low in (("size", self.size, 1), ("adv_sessions", self.adv_sessions, 1),
                                  ("min_history_sessions", self.min_history_sessions, 1)):
            if type(value) is not int or value < low:
                raise DatasetError(f"universe {label} must be an integer >= {low}")
        if (isinstance(self.min_price, bool) or not isinstance(self.min_price, (int, float))
                or not math.isfinite(self.min_price) or self.min_price < 0):
            raise DatasetError("universe min_price must be a finite number >= 0")
        return self

    def contract(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "ranking": "descending trailing mean of close*volume over adv_sessions calendar sessions; "
                       "sessions without a valid bar contribute zero; ties by ticker",
            "eligibility": "valid bar on the ranking session, at least min_history_sessions valid bars up to it, "
                           "and an unadjusted close >= min_price on the ranking session when min_price > 0",
            "reconstitution": "last session of each calendar month present in the dataset calendar",
            "effective": "from the session after the ranking session through the next ranking session",
            "proxy_for": "liquidity-ranked TOP-N research universe; not WorldQuant Brain TOP3000 membership",
        }


def exclusion_reason(ticker: str, security_type: str) -> str | None:
    """Return why a table row's security is outside the common-stock universe, or None."""
    if security_type != "stock":
        return "not_stock_type"
    base = ticker.split("-", 1)[0]
    if base in TEST_TICKERS or TEST_TICKER.fullmatch(base):
        return "exchange_test_symbol"
    if "_" in ticker:
        return "preferred_or_series"
    if not COMMON_TICKER.fullmatch(ticker):
        return "non_common_symbol"
    if "-" in ticker and ticker.rsplit("-", 1)[1] in NON_COMMON_SUFFIXES:
        return "warrant_unit_right_or_when_issued"
    if ticker in WINDOWS_RESERVED:
        return "windows_reserved_name"
    return None


def rank_sessions(sessions: Any) -> Any:
    """Sessions whose successor lies in another calendar month; the open last month has none."""
    import pandas as pd

    sessions = pd.DatetimeIndex(sessions)
    if len(sessions) < 2:
        return sessions[:0]
    following = sessions[1:]
    return sessions[:-1][(following.month != sessions[:-1].month) | (following.year != sessions[:-1].year)]


def liquidity_membership(dollar_volume: Any, valid: Any, rule: UniverseRule, price: Any = None) -> tuple[Any, Any]:
    """Rank a dates x tickers panel; return (membership rows, per-ranking summary).

    ``dollar_volume`` holds close*volume for usable bars (zero otherwise) and
    ``valid`` marks usable bars. Ranking a session reads only rows up to it.
    ``price`` (unadjusted close, rows for at least the ranking sessions, same
    columns) is required when ``rule.min_price`` is positive.
    """
    import numpy as np
    import pandas as pd

    rule.validate()
    if not (dollar_volume.index.equals(valid.index) and dollar_volume.columns.equals(valid.columns)):
        raise DatasetError("dollar-volume and validity panels must share dates and tickers")
    sessions = pd.DatetimeIndex(dollar_volume.index)
    if not sessions.is_monotonic_increasing or sessions.has_duplicates:
        raise DatasetError("universe calendar must be strictly increasing")
    if rule.min_price > 0:
        if price is None or not price.columns.equals(dollar_volume.columns):
            raise DatasetError("a minimum price needs an unadjusted close panel with the same tickers")
    valid_values = valid.to_numpy(dtype=bool)
    amounts = np.where(valid_values, dollar_volume.to_numpy(dtype=np.float64), 0.0)
    if not np.isfinite(amounts).all() or (amounts < 0).any():
        raise DatasetError("dollar volume must be finite and nonnegative on usable bars")
    # Trailing mean via cumulative sums: row i averages rows i-window+1..i only.
    totals = np.cumsum(amounts, axis=0)
    del amounts
    history = np.cumsum(valid_values, axis=0, dtype=np.int32)
    window = rule.adv_sessions
    tickers = np.asarray(dollar_volume.columns, dtype=object)
    order_by_ticker = np.argsort(tickers, kind="stable")

    rows: list[tuple[Any, ...]] = []
    summary: list[dict[str, Any]] = []
    ranks = rank_sessions(sessions)
    positions = sessions.get_indexer(ranks)
    for number, (ranked_on, position) in enumerate(zip(ranks, positions)):
        effective_start = sessions[position + 1]
        effective_end = ranks[number + 1] if number + 1 < len(ranks) else sessions[-1]
        if position + 1 < window:
            today_adv = np.full(len(tickers), np.nan)
        else:
            earlier = totals[position - window] if position >= window else 0.0
            today_adv = (totals[position] - earlier) / window
        eligible = (valid_values[position] & (history[position] >= rule.min_history_sessions)
                    & np.isfinite(today_adv) & (today_adv > 0))
        if rule.min_price > 0:
            with np.errstate(invalid="ignore"):
                eligible &= price.loc[ranked_on].to_numpy(dtype=np.float64) >= rule.min_price
        candidates = order_by_ticker[eligible[order_by_ticker]]
        # Stable sort on ticker order, then descending ADV: ties resolve alphabetically.
        chosen = candidates[np.argsort(-today_adv[candidates], kind="stable")][: rule.size]
        for rank, column in enumerate(chosen, 1):
            rows.append((ranked_on, effective_start, effective_end, tickers[column], rank, float(today_adv[column])))
        summary.append({
            "ranked_on": ranked_on, "effective_start": effective_start, "effective_end": effective_end,
            "eligible": int(eligible.sum()), "members": int(len(chosen)),
            "cutoff_adv": float(today_adv[chosen[-1]]) if len(chosen) else float("nan"),
        })
    membership = pd.DataFrame(rows, columns=("ranked_on", "effective_start", "effective_end", "instrument", "rank", "adv"))
    return membership, pd.DataFrame(summary)


def membership_spans(membership: Any, rankings: Any = None) -> dict[str, list[tuple[Any, Any]]]:
    """Merge consecutive ranking periods into Qlib instrument spans per ticker.

    Pass the per-ranking summary as ``rankings`` so that a ranking without members
    still separates the periods around it; otherwise only membership periods are known.
    """
    import pandas as pd

    source = membership if rankings is None else rankings
    periods = (source[["effective_start", "effective_end"]].drop_duplicates()
               .sort_values("effective_start").reset_index(drop=True))
    next_start = dict(zip(periods["effective_end"], periods["effective_start"].shift(-1)))
    spans: dict[str, list[tuple[Any, Any]]] = {}
    ordered = membership.sort_values(["instrument", "effective_start"])
    for instrument, group in ordered.groupby("instrument", sort=True):
        merged: list[list[Any]] = []
        for start, end in zip(group["effective_start"], group["effective_end"]):
            # Periods are contiguous: a period ending on ranking day r is followed by one starting after r.
            if merged and next_start.get(merged[-1][1]) == start:
                merged[-1][1] = end
            else:
                merged.append([start, end])
        spans[str(instrument)] = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in merged]
    return spans


def write_instrument_file(path: Path, spans: dict[str, list[tuple[Any, Any]]]) -> None:
    lines = [f"{instrument}\t{start:%Y-%m-%d}\t{end:%Y-%m-%d}\n"
             for instrument in sorted(spans) for start, end in spans[instrument]]
    path.write_bytes("".join(lines).encode("utf-8"))


def read_instrument_file(path: Path) -> dict[str, list[tuple[Any, Any]]]:
    """Parse a Qlib instruments file allowing several ordered, disjoint spans per instrument."""
    import pandas as pd

    spans: dict[str, list[tuple[Any, Any]]] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split("\t")
        if len(parts) != 3:
            raise DatasetError(f"{path.name}: line {number} must be instrument, start and end")
        try:
            start, end = pd.Timestamp(parts[1]), pd.Timestamp(parts[2])
        except ValueError as error:
            raise DatasetError(f"{path.name}: line {number} has an invalid date") from error
        if start > end:
            raise DatasetError(f"{path.name}: line {number} starts after it ends")
        existing = spans.setdefault(parts[0], [])
        if existing and existing[-1][1] >= start:
            raise DatasetError(f"{path.name}: spans for {parts[0]} overlap or are unordered")
        existing.append((start, end))
    if not spans:
        raise DatasetError(f"{path.name} lists no instruments")
    return spans


def members_between(spans: dict[str, list[tuple[Any, Any]]], first: Any, last: Any) -> list[str]:
    """Instruments with membership overlapping the inclusive [first, last] interval."""
    return sorted(name for name, intervals in spans.items()
                  if any(start <= last and end >= first for start, end in intervals))


def mask_to_members(scores: Any, spans: dict[str, list[tuple[Any, Any]]]) -> Any:
    """Keep (datetime, instrument) scores only on dates the instrument is a member."""
    import numpy as np

    if scores.empty:
        return scores
    instruments = scores.index.get_level_values("instrument")
    dates = scores.index.get_level_values("datetime").to_numpy(dtype="datetime64[ns]")
    keep = np.zeros(len(scores), dtype=bool)
    for instrument, locations in scores.groupby(instruments, sort=False).indices.items():
        intervals = spans.get(instrument)
        if not intervals:
            continue
        starts = np.array([start.to_datetime64() for start, _ in intervals], dtype="datetime64[ns]")
        ends = np.array([end.to_datetime64() for _, end in intervals], dtype="datetime64[ns]")
        when = dates[locations]
        slot = np.searchsorted(starts, when, side="right") - 1
        inside = slot >= 0
        inside[inside] = when[inside] <= ends[slot[inside]]
        keep[locations] = inside
    return scores[keep]


def dataset_universes(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Universe records declared by a dataset manifest (empty for explicit-symbol snapshots)."""
    universes = manifest.get("universes") if isinstance(manifest, dict) else None
    return {name: record for name, record in (universes or {}).items()
            if isinstance(name, str) and isinstance(record, dict)}


def load_dataset_universe(dataset: Path, manifest: dict[str, Any], name: str) -> tuple[dict[str, list[tuple[Any, Any]]], dict[str, Any]]:
    """Read a sealed dated universe; its file must match the manifest and list only dataset symbols."""
    from qlib_yahoo.dataset import sha256

    universes = dataset_universes(manifest)
    if name not in universes:
        available = ", ".join(sorted(universes)) or "none"
        raise DatasetError(f"dataset has no '{name}' universe (available: {available})")
    record = universes[name]
    if record.get("instruments_file") != f"instruments/{name}.txt":
        raise DatasetError(f"universe {name} must be stored as instruments/{name}.txt")
    path = Path(dataset) / "qlib_data" / "instruments" / f"{name}.txt"
    if not path.is_file() or path.is_symlink() or sha256(path) != record.get("instruments_sha256"):
        raise DatasetError(f"universe {name} file is missing or differs from the dataset manifest")
    spans = read_instrument_file(path)
    unknown = set(spans) - set(manifest.get("symbols") or ())
    if unknown:
        raise DatasetError(f"universe {name} lists symbols outside the dataset, e.g. {sorted(unknown)[0]}")
    return spans, record
