"""Independent synthetic availability/membership oracle and test-only Qlib bins.

Retained from the retired initial prototype. Uses only the standard library;
it is neither a production data adapter nor an executable backtest runner.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
import struct
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Iterable, Mapping


BAR_FIELDS = ("open", "high", "low", "close", "volume", "factor")
QLIB_FIELDS = BAR_FIELDS + ("change", "book_value")
INSTRUMENT_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
SIGNAL_TIME_UTC = time(21, 0, tzinfo=timezone.utc)


class FixtureError(ValueError):
    """Raised when the deterministic input fixture violates its contract."""


@dataclass(frozen=True)
class Bar:
    session: date
    instrument: str
    values: Mapping[str, float]


@dataclass(frozen=True)
class UniverseInterval:
    instrument: str
    start: date
    end: date


@dataclass(frozen=True)
class FundamentalEvent:
    instrument: str
    period_end: date
    available_at: datetime
    revision: str
    field: str
    value: float


@dataclass(frozen=True)
class PreparedPanel:
    calendar: tuple[date, ...]
    intervals: tuple[UniverseInterval, ...]
    values: Mapping[str, Mapping[str, tuple[float, ...]]]
    revisions: Mapping[tuple[date, str], str | None]


def _read_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise FixtureError(f"{path} has no CSV header")
        return [dict(row) for row in reader]


def _require_columns(path: Path, rows: list[dict[str, str]], required: Iterable[str]) -> None:
    if not rows:
        raise FixtureError(f"{path} has no data rows")
    missing = set(required) - set(rows[0])
    if missing:
        raise FixtureError(f"{path} is missing columns: {', '.join(sorted(missing))}")


def _instrument(value: str) -> str:
    normalized = value.strip().upper()
    if not INSTRUMENT_PATTERN.fullmatch(normalized):
        raise FixtureError(f"invalid instrument identifier: {value!r}")
    return normalized


def _finite_float(value: str, label: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise FixtureError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise FixtureError(f"{label} must be finite")
    return result


def load_bars(path: Path) -> tuple[Bar, ...]:
    rows = _read_dicts(path)
    _require_columns(path, rows, ("date", "instrument", *BAR_FIELDS))
    bars: list[Bar] = []
    keys: set[tuple[date, str]] = set()
    for row in rows:
        session = date.fromisoformat(row["date"])
        instrument = _instrument(row["instrument"])
        key = (session, instrument)
        if key in keys:
            raise FixtureError(f"duplicate bar: {instrument} {session}")
        keys.add(key)
        values = {field: _finite_float(row[field], f"{instrument} {session} {field}") for field in BAR_FIELDS}
        if values["volume"] < 0 or values["factor"] <= 0:
            raise FixtureError(f"{instrument} {session} volume/factor is invalid")
        if values["low"] > min(values["open"], values["close"]) or values["high"] < max(
            values["open"], values["close"]
        ):
            raise FixtureError(f"{instrument} {session} violates OHLC ordering")
        if values["low"] <= 0 or values["high"] < values["low"]:
            raise FixtureError(f"{instrument} {session} price range is invalid")
        bars.append(Bar(session=session, instrument=instrument, values=values))
    return tuple(sorted(bars, key=lambda item: (item.session, item.instrument)))


def load_universe(path: Path) -> tuple[UniverseInterval, ...]:
    rows = _read_dicts(path)
    _require_columns(path, rows, ("instrument", "start_date", "end_date"))
    intervals: list[UniverseInterval] = []
    seen: set[str] = set()
    for row in rows:
        instrument = _instrument(row["instrument"])
        if instrument in seen:
            raise FixtureError(f"duplicate universe interval for {instrument}")
        seen.add(instrument)
        start = date.fromisoformat(row["start_date"])
        end = date.fromisoformat(row["end_date"])
        if end < start:
            raise FixtureError(f"universe interval ends before it starts for {instrument}")
        intervals.append(UniverseInterval(instrument=instrument, start=start, end=end))
    return tuple(sorted(intervals, key=lambda item: item.instrument))


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise FixtureError(f"availability timestamp lacks a timezone: {value}")
    return parsed.astimezone(timezone.utc)


def load_fundamentals(path: Path) -> tuple[FundamentalEvent, ...]:
    rows = _read_dicts(path)
    _require_columns(path, rows, ("instrument", "period_end", "available_at", "revision", "field", "value"))
    events: list[FundamentalEvent] = []
    keys: set[tuple[str, str, str]] = set()
    for row in rows:
        instrument = _instrument(row["instrument"])
        available_at = _parse_utc(row["available_at"])
        period_end = date.fromisoformat(row["period_end"])
        revision = row["revision"].strip()
        field = row["field"].strip()
        if not revision or field != "book_value":
            raise FixtureError(f"unsupported fundamental event for {instrument}")
        key = (instrument, field, revision)
        if key in keys:
            raise FixtureError(f"duplicate fundamental revision: {key}")
        keys.add(key)
        if period_end > available_at.date():
            raise FixtureError(f"period end follows availability for {instrument} {revision}")
        events.append(
            FundamentalEvent(
                instrument=instrument,
                period_end=period_end,
                available_at=available_at,
                revision=revision,
                field=field,
                value=_finite_float(row["value"], f"{instrument} {revision} value"),
            )
        )
    return tuple(sorted(events, key=lambda item: (item.instrument, item.field, item.available_at, item.revision)))


def prepare_panel(
    bars: Iterable[Bar],
    intervals: Iterable[UniverseInterval],
    fundamentals: Iterable[FundamentalEvent],
) -> PreparedPanel:
    bars = tuple(bars)
    intervals = tuple(intervals)
    fundamentals = tuple(fundamentals)
    if not bars or not intervals:
        raise FixtureError("bars and universe intervals are required")

    interval_by_instrument = {item.instrument: item for item in intervals}
    bar_by_key = {(item.session, item.instrument): item for item in bars}
    calendar = tuple(sorted({item.session for item in bars}))
    calendar_index = {session: index for index, session in enumerate(calendar)}

    for bar in bars:
        interval = interval_by_instrument.get(bar.instrument)
        if interval is None or not interval.start <= bar.session <= interval.end:
            raise FixtureError(f"bar falls outside universe interval: {bar.instrument} {bar.session}")
    for interval in intervals:
        if interval.start not in calendar_index or interval.end not in calendar_index:
            raise FixtureError(f"universe boundary is not in the trading calendar: {interval.instrument}")

    events_by_instrument: dict[str, list[FundamentalEvent]] = {}
    for event in fundamentals:
        if event.instrument not in interval_by_instrument:
            raise FixtureError(f"fundamental instrument is absent from the universe: {event.instrument}")
        events_by_instrument.setdefault(event.instrument, []).append(event)
    for instrument, events in events_by_instrument.items():
        periods = {(event.field, event.period_end) for event in events}
        fields = {event.field for event in events}
        if len(periods) != len(fields):
            raise FixtureError(
                f"pilot fixture supports one reporting period per field for {instrument}; "
                "a production period-selection policy is not implemented"
            )

    values: dict[str, dict[str, tuple[float, ...]]] = {}
    revisions: dict[tuple[date, str], str | None] = {}
    for interval in intervals:
        start_index = calendar_index[interval.start]
        end_index = calendar_index[interval.end]
        sessions = calendar[start_index : end_index + 1]
        columns: dict[str, list[float]] = {field: [] for field in QLIB_FIELDS}
        previous_close: float | None = None
        for session in sessions:
            bar = bar_by_key.get((session, interval.instrument))
            if bar is None:
                for field in QLIB_FIELDS:
                    columns[field].append(math.nan)
                revisions[(session, interval.instrument)] = None
                previous_close = None
                continue

            for field in BAR_FIELDS:
                columns[field].append(bar.values[field])
            close = bar.values["close"]
            columns["change"].append(math.nan if previous_close is None else close / previous_close - 1.0)
            previous_close = close

            signal_at = datetime.combine(session, SIGNAL_TIME_UTC)
            eligible = [event for event in events_by_instrument.get(interval.instrument, []) if event.available_at <= signal_at]
            selected = max(eligible, key=lambda event: (event.available_at, event.revision)) if eligible else None
            columns["book_value"].append(selected.value if selected else math.nan)
            revisions[(session, interval.instrument)] = selected.revision if selected else None
        values[interval.instrument] = {field: tuple(column) for field, column in columns.items()}

    return PreparedPanel(
        calendar=calendar,
        intervals=tuple(sorted(intervals, key=lambda item: item.instrument)),
        values=values,
        revisions=revisions,
    )


def load_fixture(fixture_dir: Path) -> PreparedPanel:
    return prepare_panel(
        load_bars(fixture_dir / "bars.csv"),
        load_universe(fixture_dir / "universe.csv"),
        load_fundamentals(fixture_dir / "fundamentals.csv"),
    )


def fixture_hashes(fixture_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in ("bars.csv", "universe.csv", "fundamentals.csv", "expected_asof.csv"):
        payload = (fixture_dir / name).read_bytes()
        hashes[name] = hashlib.sha256(payload).hexdigest()
    return hashes


def snapshot_id(fixture_dir: Path) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(fixture_hashes(fixture_dir).items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return digest.hexdigest()[:16]


def materialize_qlib_snapshot(panel: PreparedPanel, destination: Path) -> None:
    calendars_dir = destination / "calendars"
    instruments_dir = destination / "instruments"
    features_dir = destination / "features"
    calendars_dir.mkdir(parents=True, exist_ok=True)
    instruments_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    calendar_text = "".join(f"{session.isoformat()}\n" for session in panel.calendar)
    (calendars_dir / "day.txt").write_text(calendar_text, encoding="utf-8", newline="\n")
    # Qlib's backtest calendar requests the next session boundary even on the
    # final simulated day.  The fixture therefore includes one explicit
    # post-period session and publishes the same bounded calendar as `future`.
    (calendars_dir / "day_future.txt").write_text(calendar_text, encoding="utf-8", newline="\n")
    (instruments_dir / "all.txt").write_text(
        "".join(
            f"{interval.instrument}\t{interval.start.isoformat()}\t{interval.end.isoformat()}\n"
            for interval in panel.intervals
        ),
        encoding="utf-8",
        newline="\n",
    )

    calendar_index = {session: index for index, session in enumerate(panel.calendar)}
    for interval in panel.intervals:
        instrument_dir = features_dir / interval.instrument.lower()
        instrument_dir.mkdir(parents=True, exist_ok=True)
        start_index = float(calendar_index[interval.start])
        for field in QLIB_FIELDS:
            payload = (start_index, *panel.values[interval.instrument][field])
            encoded = struct.pack(f"<{len(payload)}f", *payload)
            (instrument_dir / f"{field}.day.bin").write_bytes(encoded)


def load_expected_asof(path: Path) -> dict[tuple[date, str], tuple[float | None, str | None]]:
    rows = _read_dicts(path)
    _require_columns(path, rows, ("date", "instrument", "book_value", "revision"))
    expected: dict[tuple[date, str], tuple[float | None, str | None]] = {}
    for row in rows:
        key = (date.fromisoformat(row["date"]), _instrument(row["instrument"]))
        if key in expected:
            raise FixtureError(f"duplicate expected as-of row: {key}")
        value = _finite_float(row["book_value"], f"{key} book_value") if row["book_value"] else None
        expected[key] = (value, row["revision"] or None)
    return expected


def verify_expected_asof(panel: PreparedPanel, expected_path: Path) -> int:
    expected = load_expected_asof(expected_path)
    checked = 0
    interval_by_instrument = {item.instrument: item for item in panel.intervals}
    calendar_index = {session: index for index, session in enumerate(panel.calendar)}
    for (session, instrument), (expected_value, expected_revision) in expected.items():
        interval = interval_by_instrument[instrument]
        offset = calendar_index[session] - calendar_index[interval.start]
        actual_value = panel.values[instrument]["book_value"][offset]
        actual_revision = panel.revisions[(session, instrument)]
        if expected_value is None:
            if not math.isnan(actual_value):
                raise FixtureError(f"unexpected as-of value for {instrument} {session}: {actual_value}")
        elif not math.isclose(actual_value, expected_value, rel_tol=0.0, abs_tol=1e-8):
            raise FixtureError(f"wrong as-of value for {instrument} {session}: {actual_value}")
        if actual_revision != expected_revision:
            raise FixtureError(f"wrong revision for {instrument} {session}: {actual_revision}")
        checked += 1
    return checked
