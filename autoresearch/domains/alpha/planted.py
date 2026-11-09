"""Artificial daily-equity panels with planted, known alphas — a benchmark, not market data.

A discovery pipeline that promotes something on every dataset is useless; one
that finds nothing on any dataset is too. Planted panels make both failure modes
measurable. Returns are generated so that each planted signal, computed from
information through the close of day t-1, predicts the close-to-close return of
day t+1 (Alpha Lab's delay-1 timing)::

    r[s, t+1] = beta[s] * m[t+1] + sigma[s] * (sum_k b_k * g[t+1] * z_k[s, t-1] + e[s, t+1])

with a common, driftless market factor ``m`` (so tilting toward low or high beta
earns nothing in expectation), per-name volatility ``sigma``, fat-tailed
noise ``e`` (Student t, 5 degrees of freedom, unit variance) and cross-sectional
z-scores ``z_k``. ``b_k`` is the information coefficient of signal ``k``: the
correlation between its z-score and the next return's idiosyncratic part.
``g`` is the signal profile over the sample (``signal_profile_mode``): 1 on
every session by default (stationary); ``decay`` falls linearly from 2 at the
first session to 0 at the last; ``intermittent`` is 2 in alternating blocks of
``block_sessions`` sessions and 0 in between. Both keep the sample-average
coefficient at ``b_k``, so they move the signal around the sample without
changing how much of it there is. A null panel (all ``b_k`` = 0) has no
exploitable structure, so every promotion on it is a false discovery.

Planted signals and the FASTEXPR expressions that recover them ("oracles"):

    reversal   -(close[t] / close[t-5] - 1)                  -rank(ts_delta(close, 5) / ts_delay(close, 5))
    volume     log volume[t] minus its 20-day mean           rank(log(volume) - ts_mean(log(volume), 20))
    lowvol     -(20-day standard deviation of returns)       -rank(ts_std_dev(returns, 20))

Written in the snapshot layout Alpha Lab reads (Qlib binary features, a dated
TOP-N universe, a hash-sealed manifest). The default universe ranks every name
once by its full-sample mean dollar volume. That ranking is inert while the
universe is the whole panel (a TOP-N request for all N names selects nothing)
but forward-looking for a smaller TOP-N subset; ``dated_universe`` ranks instead
at the end of every month by trailing mean dollar volume, effective from the
next session, as the real liquidity universe does. No network, no writes
outside the target directory, and the manifest says plainly that it is
artificial.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

PROVIDER = "lab-planted-artificial-v2"
SIGNALS = ("reversal", "volume", "lowvol")
ORACLES = {
    "reversal": "-rank(ts_delta(close, 5) / ts_delay(close, 5))",
    "volume": "rank(log(volume) - ts_mean(log(volume), 20))",
    "lowvol": "-rank(ts_std_dev(returns, 20))",
}
DEFAULT_STRENGTH = {"reversal": 0.025, "volume": 0.025, "lowvol": 0.012}
SIGNAL_PROFILES = ("stationary", "decay", "intermittent")
DEFAULT_BLOCK_SESSIONS = 126
#: Trailing sessions of dollar volume behind a dated ranking (the real universe rule's default).
DEFAULT_ADV_SESSIONS = 20


def _zscore(values: np.ndarray) -> np.ndarray:
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std <= 0:
        return np.zeros_like(values)
    return np.nan_to_num((values - mean) / std)


def signal_profile(mode: str, days: int, block_sessions: int = DEFAULT_BLOCK_SESSIONS) -> np.ndarray:
    """Multiplier of the planted coefficients on each session; its mean is 1 (about 1 for an odd block count)."""
    if mode not in SIGNAL_PROFILES:
        raise ValueError("signal_profile_mode must be one of " + ", ".join(SIGNAL_PROFILES))
    if days < 2:
        raise ValueError("a signal profile needs at least two sessions")
    if mode == "stationary":
        return np.ones(days)
    if mode == "decay":
        return 2.0 * (1.0 - np.arange(days) / (days - 1))
    if block_sessions < 1:
        raise ValueError("block_sessions must be a positive integer")
    blocks = np.arange(days) // int(block_sessions)
    return np.where(blocks % 2 == 0, 2.0, 0.0)


def generate(symbols: int = 150, days: int = 1000, *, strength: dict | None = None, seed: int = 11,
             start: str = "2018-01-01", signal_profile_mode: str = "stationary",
             block_sessions: int = DEFAULT_BLOCK_SESSIONS) -> dict:
    """Arrays for a planted panel: dates, symbols, OHLCV fields and the planted z-scores."""
    if symbols < 20 or days < 300:
        raise ValueError("a planted panel needs at least 20 symbols and 300 sessions")
    strength = dict(DEFAULT_STRENGTH if strength is None else strength)
    unknown = set(strength) - set(SIGNALS)
    if unknown:
        raise ValueError("unknown planted signal: " + ", ".join(sorted(unknown)))
    coefficients = np.array([float(strength.get(name, 0.0)) for name in SIGNALS])
    rng = np.random.default_rng(seed)
    calendar = np.arange(np.datetime64(start, "D"), np.datetime64(start, "D") + int(days * 1.6) + 10)
    calendar = calendar[np.is_busday(calendar)][:days]
    n, t_count = symbols, len(calendar)
    profile = signal_profile(signal_profile_mode, t_count, block_sessions)
    sigma = rng.uniform(0.012, 0.03, n)
    beta = rng.uniform(0.6, 1.4, n)
    # Zero drift: with heterogeneous betas a positive market drift is itself a learnable
    # premium, and a null panel must contain nothing to find.
    market = rng.normal(0.0, 0.009, t_count)
    level = rng.uniform(11.5, 14.5, n)               # log share volume level per name
    log_volume = np.empty((t_count, n))
    log_volume[0] = level + rng.normal(0, 0.3, n)
    returns = np.zeros((t_count, n))
    log_close = np.zeros((t_count, n))
    log_close[0] = np.log(rng.uniform(10, 120, n))
    planted = np.zeros((len(SIGNALS), t_count, n))
    noise = rng.standard_t(5, size=(t_count, n)) / np.sqrt(5 / 3)
    for t in range(1, t_count):
        log_volume[t] = level + 0.8 * (log_volume[t - 1] - level) + rng.normal(0, 0.35, n)
        if t >= 2:
            # The profile is read on the return's session: a decayed or switched-off signal is one
            # whose z-score no longer predicts that session's return.
            signal = (coefficients * profile[t]) @ planted[:, t - 2] if t - 2 >= 0 else 0.0
        else:
            signal = 0.0
        returns[t] = beta * market[t] + sigma * (signal + noise[t])
        log_close[t] = log_close[t - 1] + np.log1p(np.clip(returns[t], -0.5, 1.0))
        # z-scores known at the close of day t predict the return of day t + 2
        if t >= 5:
            change = np.exp(log_close[t] - log_close[t - 5]) - 1
            planted[0, t] = _zscore(-change)
        if t >= 20:
            planted[1, t] = _zscore(log_volume[t] - log_volume[t - 19:t + 1].mean(axis=0))
            planted[2, t] = _zscore(-returns[t - 19:t + 1].std(axis=0, ddof=1))
    close = np.exp(log_close)
    gap = rng.normal(0, 0.3, (t_count, n)) * sigma
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1] * np.exp(gap[1:])
    spread = np.abs(rng.normal(0, 0.5, (t_count, n))) * sigma
    high = np.maximum(open_, close) * np.exp(spread)
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.5, (t_count, n))) * sigma)
    volume = np.round(np.exp(log_volume))
    names = [f"PLT{i:03d}" for i in range(n)]
    return {"dates": calendar, "symbols": names, "planted": {k: planted[i] for i, k in enumerate(SIGNALS)},
            "fields": {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            "strength": {name: float(strength.get(name, 0.0)) for name in SIGNALS}, "seed": seed,
            "signal_profile": signal_profile_mode, "block_sessions": int(block_sessions), "profile": profile}


def month_end_sessions(dates: np.ndarray) -> np.ndarray:
    """Indices of sessions whose successor lies in another calendar month: the ranking sessions."""
    months = np.asarray(dates).astype("datetime64[M]")
    return np.flatnonzero(months[:-1] != months[1:])


def membership_rows(dates: np.ndarray, names: list[str], dollar_volume: np.ndarray, *, universe_size: int,
                    dated: bool, adv_sessions: int = DEFAULT_ADV_SESSIONS) -> list[tuple[str, str, str, int]]:
    """(instrument, effective_start, effective_end, rank) rows of the TOP-``universe_size`` universe.

    Static (the default): one ranking by full-sample mean dollar volume, effective for the whole
    sample. Dated: a ranking at every month-end session by the trailing mean dollar volume over
    the last ``adv_sessions`` sessions (fewer at the start of the sample), effective from the
    next session through the next ranking session. Nothing is a member before the first
    ranking takes effect, and no ranking reads a session after the one it is made on.
    """
    if not 1 <= universe_size <= len(names):
        raise ValueError("universe_size must be between 1 and the number of symbols")
    if adv_sessions < 1:
        raise ValueError("adv_sessions must be a positive integer")
    rows: list[tuple[str, str, str, int]] = []
    if not dated:
        order = np.argsort(-dollar_volume.mean(axis=0))
        rows.extend((names[i], str(dates[0]), str(dates[-1]), rank + 1) for rank, i in enumerate(order[:universe_size]))
        return rows
    ranking = month_end_sessions(dates)
    if not len(ranking):
        raise ValueError("a dated universe needs a sample that spans at least two calendar months")
    for number, session in enumerate(ranking):
        trailing = dollar_volume[max(0, session - adv_sessions + 1):session + 1].mean(axis=0)
        order = np.argsort(-trailing, kind="stable")
        start = dates[session + 1]
        end = dates[ranking[number + 1]] if number + 1 < len(ranking) else dates[-1]
        rows.extend((names[i], str(start), str(end), rank + 1) for rank, i in enumerate(order[:universe_size]))
    return rows


def create_planted_dataset(directory: str | Path, *, symbols: int = 150, days: int = 1000,
                           strength: dict | None = None, seed: int = 11, start: str = "2018-01-01",
                           signal_profile_mode: str = "stationary", block_sessions: int = DEFAULT_BLOCK_SESSIONS,
                           dated_universe: bool = False, universe_size: int | None = None,
                           adv_sessions: int = DEFAULT_ADV_SESSIONS) -> Path:
    """Write a planted panel; a null panel is ``strength={}`` (nothing planted).

    ``universe_size`` (default: every symbol) is the size of the dataset's TOP-N universe;
    ``dated_universe`` ranks it monthly by trailing dollar volume instead of once by the
    full-sample mean. An existing planted dataset at ``directory`` is reopened as it is.
    """
    from qlib_yahoo.alphalab.panel import source_identity
    from qlib_yahoo.dataset import tree_sha256

    directory = Path(directory).resolve()
    if (directory / "manifest.json").exists():
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("provider") != PROVIDER:
            raise ValueError("refusing to replace an existing non-planted dataset")
        source_identity(directory)
        return directory
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("planted dataset directory must be empty")
    size = symbols if universe_size is None else int(universe_size)
    if not 1 <= size <= symbols:
        raise ValueError("universe_size must be between 1 and the number of symbols")
    data = generate(symbols, days, strength=strength, seed=seed, start=start,
                    signal_profile_mode=signal_profile_mode, block_sessions=block_sessions)
    days_list, names = data["dates"], data["symbols"]
    qlib = directory / "qlib_data"
    (qlib / "calendars").mkdir(parents=True)
    (qlib / "instruments").mkdir()
    (qlib / "calendars/day.txt").write_text("\n".join(map(str, days_list)) + "\n")
    (qlib / "instruments/all.txt").write_text("".join(f"{s}\t{days_list[0]}\t{days_list[-1]}\n" for s in names))
    for column, symbol in enumerate(names):
        folder = qlib / "features" / symbol.lower()
        folder.mkdir(parents=True)
        for field, values in data["fields"].items():
            np.concatenate(([0], values[:, column])).astype("<f4").tofile(folder / f"{field}.day.bin")
    dollar = data["fields"]["close"] * data["fields"]["volume"]
    rows = membership_rows(days_list, names, dollar, universe_size=size, dated=dated_universe, adv_sessions=adv_sessions)
    universe = directory / "universe"
    universe.mkdir()
    membership = universe / "membership.csv.gz"
    text = "".join(f"{instrument},{first},{last},{rank}\n" for instrument, first, last, rank in rows)
    membership.write_bytes(gzip.compress(("instrument,effective_start,effective_end,rank\n" + text).encode(), mtime=0))
    universe_name = f"top{size}"
    ranking = ({"mode": "trailing_dollar_volume", "adv_sessions": int(adv_sessions), "cadence": "month_end",
                "effective": "from the session after the ranking session through the next ranking session"}
               if dated_universe else {"mode": "full_sample_dollar_volume"})
    planted_note: dict = {"signals": data["strength"], "seed": seed, "oracles": ORACLES,
                          "timing": "z-score known at close t predicts the close-to-close return of t+2",
                          "signal_profile": data["signal_profile"]}
    if data["signal_profile"] == "intermittent":
        planted_note["block_sessions"] = data["block_sessions"]
    manifest = {"status": "ready", "provider": PROVIDER,
                "note": "Artificial planted-signal benchmark panel; no investment or economic evidence.",
                "planted": planted_note,
                "fields": ["open", "high", "low", "close", "volume"],
                "qlib_binary_tree_sha256": tree_sha256(qlib)[0],
                "universes": {universe_name: {"size": size, "membership_csv": "universe/membership.csv.gz",
                                              "membership_sha256": hashlib.sha256(membership.read_bytes()).hexdigest(),
                                              "ranking": ranking}}}
    (directory / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    source_identity(directory)
    return directory
