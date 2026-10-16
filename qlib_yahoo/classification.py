"""Sealed, point-in-time classification overlays for verified Qlib snapshots.

This module stores dated classification events beside (never inside) a price
snapshot. It does not assign an economic meaning to a provider's labels: callers
must record source and taxonomy caveats in the overlay metadata.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import re
import tempfile
import io
import shutil
from types import MappingProxyType
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from qlib_yahoo.dataset import DatasetError, sha256
from qlib_yahoo.dataset_integrity import verify_dataset
from qlib_yahoo.packed import _reject_redirects


SCHEMA = "dated-classification-overlay-v1"
GROUPS = ("industry", "sector", "country")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_EVENT_FIELDS = frozenset(("instrument", "available_at", *GROUPS, "source_accession", "provenance"))
_COALESCED_KEY = "classification_overlay_source_records"
try:
    _NY = ZoneInfo("America/New_York")
except ZoneInfoNotFoundError:  # Windows Python installations may omit the IANA database.
    _NY = None


@dataclass(frozen=True)
class ClassificationOverlay:
    """Dense read-only code matrices and their immutable source identity."""

    calendar: tuple[str, ...]
    symbols: tuple[str, ...]
    codes: Mapping[str, Any]
    labels: Mapping[str, tuple[str, ...]]
    stale: Any
    identity: Mapping[str, Any]
    coverage: Mapping[str, Any]


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise DatasetError(f"classification metadata is not finite JSON: {error}") from error


def _parse_datetime(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"{label} must be a timezone-aware ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as error:
        raise DatasetError(f"{label} must be a timezone-aware ISO-8601 string") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DatasetError(f"{label} must include an explicit UTC offset")
    return parsed.astimezone(timezone.utc)


def _utc_string(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    first = date(year, month, 1)
    return date(year, month, 1 + (weekday - first.weekday()) % 7 + 7 * (occurrence - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        following = date(year + 1, 1, 1)
    else:
        following = date(year, month + 1, 1)
    last = date.fromordinal(following.toordinal() - 1)
    return date.fromordinal(last.toordinal() - (last.weekday() - weekday) % 7)


def _new_york_open(day: date) -> datetime:
    """Return 09:30 New York as UTC, with a self-contained modern Windows fallback."""
    if _NY is not None:
        return datetime.combine(day, time(9, 30), _NY).astimezone(timezone.utc)
    year = day.year
    if year >= 2007:
        daylight_start = _nth_weekday(year, 3, 6, 2)
        standard_start = _nth_weekday(year, 11, 6, 1)
    elif year >= 1987:
        daylight_start = _nth_weekday(year, 4, 6, 1)
        standard_start = _last_weekday(year, 10, 6)
    elif year >= 1967:
        daylight_start = _last_weekday(year, 4, 6)
        standard_start = _last_weekday(year, 10, 6)
        if year == 1974:
            daylight_start = date(1974, 1, 6)
        elif year == 1975:
            daylight_start = date(1975, 2, 23)
    else:
        raise DatasetError("New York session conversion before 1967 requires an installed IANA timezone database")
    offset = timedelta(hours=-4 if daylight_start <= day < standard_start else -5)
    return datetime.combine(day, time(9, 30), timezone(offset)).astimezone(timezone.utc)


def _normalize_event(value: Mapping[str, Any]) -> tuple[dict[str, Any], datetime]:
    if not isinstance(value, Mapping) or set(value) - _EVENT_FIELDS:
        raise DatasetError("classification event has unsupported fields or is not an object")
    instrument = value.get("instrument")
    if (not isinstance(instrument, str) or not instrument or instrument != instrument.strip()
            or any(character in instrument for character in "\\/\t\r\n\0")):
        raise DatasetError("classification event instrument must be a nonempty Qlib symbol")
    available = _parse_datetime(value.get("available_at"), "classification event available_at")
    event: dict[str, Any] = {"instrument": instrument, "available_at": _utc_string(available)}
    for group in GROUPS:
        label = value.get(group)
        if label is not None and not isinstance(label, str):
            raise DatasetError(f"classification event {group} must be a string or null")
        label = label.strip() if isinstance(label, str) else None
        event[group] = label or None
    accession = value.get("source_accession")
    if accession is not None and (not isinstance(accession, str) or not accession.strip()):
        raise DatasetError("classification event source_accession must be a nonempty string or null")
    if accession is not None:
        event["source_accession"] = accession.strip()
    provenance = value.get("provenance")
    if provenance is not None:
        if not isinstance(provenance, Mapping):
            raise DatasetError("classification event provenance must be an object or null")
        event["provenance"] = dict(provenance)
        _canonical_json(event["provenance"])
    _canonical_json(event)
    return event, available


def _safe_object(value: Mapping[str, Any] | None, label: str) -> dict[str, Any]:
    if value is None:
        result: dict[str, Any] = {}
    elif isinstance(value, Mapping):
        result = dict(value)
    else:
        raise DatasetError(f"{label} must be a JSON object")
    _canonical_json(result)
    return result


def _read_source(dataset: Path, *, full: bool = True) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    dataset = Path(dataset).absolute()
    _reject_redirects(dataset)
    if not dataset.is_dir():
        raise DatasetError(f"price dataset directory does not exist: {dataset}")
    manifest_path = dataset / "manifest.json"
    calendar_path = dataset / "qlib_data" / "calendars" / "day.txt"
    instruments_path = dataset / "qlib_data" / "instruments" / "all.txt"
    for path in (manifest_path, calendar_path, instruments_path):
        _reject_redirects(path)
        if not path.is_file():
            raise DatasetError(f"classification source is missing required file: {path.name}")
    qlib_root = dataset / "qlib_data"
    _reject_redirects(qlib_root)
    for path in qlib_root.rglob("*") if full else ():
        if path.is_symlink() or (path.is_dir() and path.is_junction()):
            raise DatasetError(f"redirected dataset path: {path}")
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetError("price dataset manifest is not valid UTF-8 JSON") from error
    if not isinstance(manifest, dict):
        raise DatasetError("price dataset manifest must be a JSON object")
    if full:
        verified = verify_dataset(dataset, manifest)
    else:
        tree = manifest.get("qlib_binary_tree_sha256")
        count = manifest.get("qlib_binary_file_count")
        if not isinstance(tree, str) or not _DIGEST.fullmatch(tree) or type(count) is not int or count < 1:
            raise DatasetError("price metadata lacks recorded binary-tree provenance")
        if manifest.get("status") != "ready":
            raise DatasetError("price dataset must be ready")
        verified = {"qlib_binary_tree_sha256": tree, "qlib_binary_file_count": count}
    recorded_count = manifest.get("qlib_binary_file_count")
    if recorded_count is not None and recorded_count != verified.get("qlib_binary_file_count"):
        raise DatasetError("price dataset binary file count differs from its manifest")

    calendar_bytes = calendar_path.read_bytes()
    instrument_bytes = instruments_path.read_bytes()
    calendar_lines = calendar_bytes.decode("utf-8").splitlines()
    if not calendar_lines:
        raise DatasetError("price dataset calendar is empty")
    try:
        dates = tuple(date.fromisoformat(line).isoformat() for line in calendar_lines)
    except ValueError as error:
        raise DatasetError("price dataset calendar contains an invalid ISO date") from error
    if (dates != tuple(calendar_lines) or tuple(sorted(set(dates))) != dates):
        raise DatasetError("price dataset calendar must contain unique, ordered ISO dates")
    if _NY is None and any(date.fromisoformat(day).year < 1967 for day in dates):
        raise DatasetError("New York session conversion before 1967 requires an installed IANA timezone database")

    names: list[str] = []
    seen: set[str] = set()
    for line in instrument_bytes.decode("utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) != 3 or not fields[0] or fields[0] in seen:
            raise DatasetError("price dataset instrument inventory is malformed or duplicated")
        try:
            first, last = date.fromisoformat(fields[1]), date.fromisoformat(fields[2])
        except ValueError as error:
            raise DatasetError("price dataset instrument inventory has invalid date bounds") from error
        if first.isoformat() != fields[1] or last.isoformat() != fields[2] or first > last:
            raise DatasetError("price dataset instrument inventory has invalid date bounds")
        seen.add(fields[0])
        names.append(fields[0])
    if not names:
        raise DatasetError("price dataset instrument inventory is empty")
    symbols = tuple(sorted(names))
    calendar_hash = hashlib.sha256(calendar_bytes).hexdigest()
    instruments_hash = hashlib.sha256(instrument_bytes).hexdigest()
    identity = {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "binary_tree_sha256": verified["qlib_binary_tree_sha256"],
        "binary_file_count": verified["qlib_binary_file_count"],
        "calendar_sha256": calendar_hash,
        "calendar_count": len(dates),
        "instrument_inventory_sha256": instruments_hash,
        "symbol_count": len(symbols),
        "symbols_sha256": hashlib.sha256("\n".join(symbols).encode("utf-8")).hexdigest(),
    }
    return identity, dates, symbols


def _normalise_events(events: Iterable[Mapping[str, Any]], symbols: set[str]) -> tuple[list[dict[str, Any]], int]:
    unique: dict[tuple[str, datetime], tuple[dict[str, Any], datetime]] = {}
    duplicates = 0
    try:
        iterator = iter(events)
    except TypeError as error:
        raise DatasetError("classification events must be iterable") from error
    for supplied in iterator:
        event, available = _normalize_event(supplied)
        if _COALESCED_KEY in event.get("provenance", {}):
            raise DatasetError(f"event provenance may not use reserved key {_COALESCED_KEY!r}")
        if event["instrument"] not in symbols:
            raise DatasetError(f"classification event instrument is absent from price inventory: {event['instrument']}")
        key = (event["instrument"], available)
        previous = unique.get(key)
        if previous is not None:
            if previous[0] == event:
                duplicates += 1
                continue
            prior, _ = previous
            if any(prior[group] != event[group] for group in GROUPS):
                raise DatasetError("conflicting classification events share instrument and available_at")
            def source_records(item: dict[str, Any]) -> list[dict[str, Any]]:
                provenance = item.get("provenance", {})
                if _COALESCED_KEY in provenance:
                    return list(provenance[_COALESCED_KEY])
                record: dict[str, Any] = {}
                if "source_accession" in item:
                    record["source_accession"] = item["source_accession"]
                if provenance:
                    record["provenance"] = provenance
                return [record] if record else []

            records = source_records(prior) + source_records(event)
            by_content = {_canonical_json(record): record for record in records}
            merged = dict(prior)
            merged.pop("source_accession", None)
            merged.pop("provenance", None)
            if by_content:
                merged["provenance"] = {_COALESCED_KEY: [by_content[key] for key in sorted(by_content)]}
            unique[key] = (merged, available)
            duplicates += 1
        else:
            unique[key] = (event, available)
    normalized = [item[0] for item in unique.values()]
    normalized.sort(key=lambda event: (event["instrument"], event["available_at"]))
    return normalized, duplicates


def create_classification_overlay(
    price_dataset: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    events: Iterable[Mapping[str, Any]],
    *,
    max_age_days: int | None = None,
    source_metadata: Mapping[str, Any] | None = None,
    taxonomy_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Create a new immutable overlay folder; an existing destination is refused.

    ``max_age_days=None`` explicitly means no age-based expiry. A caller that
    wants stale classifications to become unknown must provide a nonnegative
    integer age limit. Availability is converted to UTC and activated strictly
    after the corresponding New York 09:30 session open.
    """
    if max_age_days is not None and (type(max_age_days) is not int or max_age_days < 0):
        raise DatasetError("max_age_days must be a nonnegative integer or None")
    source_path = Path(price_dataset).absolute()
    dest = Path(destination).absolute()
    _reject_redirects(source_path)
    _reject_redirects(dest)
    source_identity, calendar, symbols = _read_source(source_path)
    if (dest == source_path or source_path in dest.parents or dest in source_path.parents):
        raise DatasetError("classification overlay destination must be separate from the price snapshot")
    if dest.exists():
        raise DatasetError(f"classification overlay destination already exists: {dest}")
    source_metadata = _safe_object(source_metadata, "source_metadata")
    taxonomy_metadata = _safe_object(taxonomy_metadata, "taxonomy_metadata")
    canonical_events, duplicate_count = _normalise_events(events, set(symbols))
    events_bytes = b"".join(_canonical_json(event) + b"\n" for event in canonical_events)
    events_digest = hashlib.sha256(events_bytes).hexdigest()
    event_groups = {group: sum(event[group] is not None for event in canonical_events) for group in GROUPS}
    manifest = {
        "schema": SCHEMA,
        "status": "ready",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "activation": {"timezone": "America/New_York", "time": "09:30", "strictly_after": True},
        "unknown_code": -1,
        "max_age_days": max_age_days,
        "groups": list(GROUPS),
        "price_source": source_identity,
        "events": {"file": "events.jsonl", "sha256": events_digest,
                   "count": len(canonical_events), "coalesced_exact_duplicates": duplicate_count,
                   "nonmissing_values_by_group": event_groups},
        "source_metadata": source_metadata,
        "taxonomy_metadata": taxonomy_metadata,
        "signature_note": "SHA-256 detects accidental modification; this format is not digitally signed.",
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False,
                                allow_nan=False).encode("utf-8") + b"\n"
    parent = dest.parent
    _reject_redirects(parent)
    parent.mkdir(parents=True, exist_ok=True)
    _reject_redirects(parent)
    if dest.exists():
        raise DatasetError(f"classification overlay destination already exists: {dest}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{dest.name}.", dir=parent))
    try:
        (temporary / "events.jsonl").write_bytes(events_bytes)
        (temporary / "manifest.json").write_bytes(manifest_bytes)
        digest = hashlib.sha256(manifest_bytes).hexdigest()
        (temporary / "manifest.sha256").write_text(f"{digest}  manifest.json\n", encoding="ascii")
        if _read_source(source_path)[0] != source_identity:
            raise DatasetError("price snapshot changed during classification overlay creation")
        if dest.exists():
            raise DatasetError(f"classification overlay destination already exists: {dest}")
        os.rename(temporary, dest)
    except Exception:
        if temporary.exists():
            import shutil
            shutil.rmtree(temporary)
        raise
    return dest


def _manifest_metadata(overlay_path: Path) -> tuple[dict[str, Any], str]:
    overlay_path = Path(overlay_path).absolute()
    _reject_redirects(overlay_path)
    if not overlay_path.is_dir():
        raise DatasetError("classification overlay folder does not exist")
    expected_files = {"events.jsonl", "manifest.json", "manifest.sha256"}
    children = list(overlay_path.iterdir())
    for path in children:
        _reject_redirects(path)
    actual_files = {path.name for path in children if path.is_file()}
    if actual_files != expected_files or any(path.is_dir() for path in children):
        raise DatasetError("classification overlay has missing or unexpected files")
    raw_manifest = (overlay_path / "manifest.json").read_bytes()
    digest_line = (overlay_path / "manifest.sha256").read_text(encoding="ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{64}  manifest\.json", digest_line):
        raise DatasetError("classification manifest detached hash is malformed")
    digest = hashlib.sha256(raw_manifest).hexdigest()
    if digest_line.split("  ", 1)[0] != digest:
        raise DatasetError("classification manifest SHA-256 mismatch")
    try:
        manifest = json.loads(raw_manifest.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetError("classification manifest is not valid UTF-8 JSON") from error
    if (not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA
            or manifest.get("status") != "ready" or manifest.get("groups") != list(GROUPS)
            or manifest.get("activation") != {"timezone": "America/New_York", "time": "09:30",
                                               "strictly_after": True}
            or manifest.get("unknown_code") != -1
            or not isinstance(manifest.get("source_metadata"), dict)
            or not isinstance(manifest.get("taxonomy_metadata"), dict)):
        raise DatasetError("unsupported or incomplete classification overlay manifest")
    max_age = manifest.get("max_age_days")
    if max_age is not None and (type(max_age) is not int or max_age < 0):
        raise DatasetError("classification overlay has an invalid max_age_days policy")
    events_record = manifest.get("events")
    if (not isinstance(events_record, dict) or events_record.get("file") != "events.jsonl"
            or not isinstance(events_record.get("sha256"), str)
            or not _DIGEST.fullmatch(events_record["sha256"])
            or type(events_record.get("count")) is not int or events_record["count"] < 0):
        raise DatasetError("classification overlay events record is invalid")
    if not isinstance(manifest.get("price_source"), dict):
        raise DatasetError("classification overlay has no price-source identity")
    counts = events_record.get("nonmissing_values_by_group")
    if (not isinstance(counts, dict) or set(counts) != set(GROUPS)
            or any(type(n) is not int or n < 0 or n > events_record["count"] for n in counts.values())):
        raise DatasetError("classification nonmissing group metadata is invalid")
    _canonical_json(manifest)
    return manifest, digest


def _parse_events(raw_events: bytes, manifest: dict) -> list[dict[str, Any]]:
    events_record = manifest["events"]
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw_events.splitlines(), start=1):
        try:
            event = json.loads(line.decode("utf-8"))
            normalized, _ = _normalize_event(event)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DatasetError(f"classification event line {line_number} is invalid JSON") from error
        if normalized != event:
            raise DatasetError(f"classification event line {line_number} is not canonical")
        if _canonical_json(event) != line:
            raise DatasetError(f"classification event line {line_number} is not canonically encoded")
        rows.append(event)
    if len(rows) != events_record["count"]:
        raise DatasetError("classification event count differs from its manifest")
    if rows != sorted(rows, key=lambda event: (event["instrument"], event["available_at"])):
        raise DatasetError("classification events are not in canonical order")
    seen_event_times: dict[tuple[str, str], tuple[str | None, ...]] = {}
    for event in rows:
        key = (event["instrument"], event["available_at"])
        groups = tuple(event[group] for group in GROUPS)
        if key in seen_event_times:
            if seen_event_times[key] != groups:
                raise DatasetError("conflicting classification events share instrument and available_at")
            raise DatasetError("duplicate classification events were not coalesced")
        seen_event_times[key] = groups
    counts = {g: sum(row[g] is not None for row in rows) for g in GROUPS}
    if counts != events_record["nonmissing_values_by_group"]:
        raise DatasetError("classification nonmissing group counts differ from events")
    return rows


def _read_manifest(overlay_path: Path) -> tuple[dict, list[dict], str]:
    manifest, digest = _manifest_metadata(overlay_path)
    raw = (Path(overlay_path) / "events.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["events"]["sha256"]:
        raise DatasetError("classification events SHA-256 mismatch")
    return manifest, _parse_events(raw, manifest), digest


def classification_source_info(price_dataset) -> dict[str, Any]:
    """Verify price identity once for sidecar discovery, without dense arrays."""
    return _read_source(Path(price_dataset))[0]


def classification_info(price_dataset, overlay_path) -> dict[str, Any]:
    """Verify source/event binding without materializing date-by-symbol arrays."""
    manifest, events, overlay_hash = _read_manifest(Path(overlay_path))
    price_identity, _calendar, full_symbols = _read_source(Path(price_dataset))
    if manifest["price_source"] != price_identity:
        raise DatasetError("classification overlay is bound to a different price snapshot")
    known = set(full_symbols)
    if any(event["instrument"] not in known for event in events):
        raise DatasetError("classification overlay event references a symbol outside its bound inventory")
    source = manifest.get("source_metadata", {})
    return {"path": str(Path(overlay_path).absolute()), "manifest_sha256": overlay_hash,
            "events_sha256": manifest["events"]["sha256"], "price_source": price_identity,
            "groups": [g for g in GROUPS if any(event[g] is not None for event in events)],
            "max_age_days": manifest["max_age_days"],
            "taxonomy_metadata": manifest.get("taxonomy_metadata", {}),
            "availability_policy": source.get("availability_policy"),
            "historical_identity_limit": source.get("historical_identity_limit")}


def read_classification_overlay(
    price_dataset: str | os.PathLike[str],
    overlay_path: str | os.PathLike[str],
    *,
    start: str | None = None,
    end: str | None = None,
    symbols: Iterable[str] | None = None,
    _verified=None,
) -> ClassificationOverlay:
    """Verify and materialize selected rows from a sealed overlay.

    ``start`` and ``end`` are inclusive exact dates from the bound Qlib calendar.
    When ``symbols`` is supplied, output columns use sorted symbol order.
    """
    if _verified is None:
        manifest, events, overlay_hash = _read_manifest(Path(overlay_path))
        price_identity, full_calendar, full_symbols = _read_source(Path(price_dataset))
    else:
        manifest, events, overlay_hash, price_identity, full_calendar, full_symbols = _verified
    if manifest["price_source"] != price_identity:
        raise DatasetError("classification overlay is bound to a different price snapshot")
    known_symbols = set(full_symbols)
    if any(event["instrument"] not in known_symbols for event in events):
        raise DatasetError("classification overlay event references a symbol outside its bound inventory")
    first = 0 if start is None else _calendar_index(full_calendar, start, "start")
    last = len(full_calendar) - 1 if end is None else _calendar_index(full_calendar, end, "end")
    if first > last:
        raise DatasetError("classification selection start is after end")
    calendar = full_calendar[first:last + 1]
    if symbols is None:
        selected_symbols = full_symbols
    else:
        requested = list(symbols)
        if not requested or any(not isinstance(symbol, str) for symbol in requested):
            raise DatasetError("symbols selection must contain one or more string tickers")
        if len(set(requested)) != len(requested):
            raise DatasetError("symbols selection contains duplicates")
        absent = set(requested) - set(full_symbols)
        if absent:
            raise DatasetError(f"classification selection has unknown price symbol(s): {sorted(absent)}")
        selected_symbols = tuple(sorted(requested))

    try:
        import numpy as np
    except ImportError as error:
        raise DatasetError("NumPy is required to materialize classification matrices") from error
    labels = {group: tuple(sorted({event[group] for event in events if event[group] is not None}))
              for group in GROUPS}
    code_lookup = {group: {label: index for index, label in enumerate(values)}
                   for group, values in labels.items()}
    shape = (len(calendar), len(selected_symbols))
    codes = {group: np.full(shape, -1, dtype=np.int32) for group in GROUPS}
    stale = np.zeros(shape, dtype=np.bool_)
    full_opens = [_new_york_open(date.fromisoformat(day)) for day in full_calendar]
    event_by_symbol: dict[str, list[tuple[int, datetime, dict[str, Any]]]] = defaultdict(list)
    beyond_calendar = 0
    for event in events:
        available = _parse_datetime(event["available_at"], "stored event available_at")
        activate_at = bisect.bisect_right(full_opens, available)
        if activate_at >= len(full_opens):
            beyond_calendar += 1
            continue
        event_by_symbol[event["instrument"]].append((activate_at, available, event))
    for symbol in selected_symbols:
        timeline = event_by_symbol.get(symbol, [])
        timeline.sort(key=lambda item: (item[0], item[1]))
        # Several filings may become visible at the same next open. Only the most
        # recently available complete state can be active at that open.
        effective: list[tuple[int, datetime, dict[str, Any]]] = []
        for item in timeline:
            if effective and effective[-1][0] == item[0]:
                effective[-1] = item
            else:
                effective.append(item)
        output_column = bisect.bisect_left(selected_symbols, symbol)
        # A complete event replaces all group states, including null clears.
        for index, (activation, available, event) in enumerate(effective):
            stop = effective[index + 1][0] if index + 1 < len(effective) else len(full_calendar)
            lo, hi = max(first, activation), min(last + 1, stop)
            if lo >= hi:
                continue
            fresh_stop = hi
            if manifest["max_age_days"] is not None:
                expiry = available + timedelta(days=manifest["max_age_days"])
                fresh_stop = max(lo, min(hi, bisect.bisect_right(full_opens, expiry)))
                stale[fresh_stop - first:hi - first, output_column] = True
            for group in GROUPS:
                label = event[group]
                if label is not None:
                    codes[group][lo - first:fresh_stop - first, output_column] = code_lookup[group][label]

    for array in (*codes.values(), stale):
        array.setflags(write=False)
    total_cells = int(shape[0] * shape[1])
    coverage = {
        group: {"labeled_cells": int(np.count_nonzero(matrix >= 0)),
                "unknown_cells": int(total_cells - np.count_nonzero(matrix >= 0)),
                "stale_cells": int(np.count_nonzero(stale)),
                "label_count": len(labels[group])}
        for group, matrix in codes.items()
    }
    coverage["overlay"] = {
        "selected_sessions": len(calendar), "selected_symbols": len(selected_symbols),
        "total_cells": total_cells, "events": len(events),
        "events_after_calendar": beyond_calendar,
        "activated_events": sum(1 for by_symbol in event_by_symbol.values() for _ in by_symbol),
    }
    identity = dict(price_identity)
    identity.update({"overlay_manifest_sha256": overlay_hash,
                     "overlay_events_sha256": manifest["events"]["sha256"],
                     "overlay_path": str(Path(overlay_path).absolute()),
                     "max_age_days": manifest["max_age_days"],
                     "source_metadata": manifest.get("source_metadata", {}),
                     "taxonomy_metadata": manifest.get("taxonomy_metadata", {}),
                     "selection": {"start": calendar[0], "end": calendar[-1],
                                   "symbols": len(selected_symbols),
                                   "symbols_sha256": hashlib.sha256("\n".join(selected_symbols).encode("utf-8")).hexdigest()}})
    return ClassificationOverlay(calendar, selected_symbols, codes, labels, stale, identity, coverage)


def _calendar_index(calendar: tuple[str, ...], value: str, label: str) -> int:
    try:
        canonical = date.fromisoformat(value).isoformat()
    except (TypeError, ValueError) as error:
        raise DatasetError(f"classification {label} must be an ISO calendar date") from error
    if canonical != value or value not in calendar:
        raise DatasetError(f"classification {label} must exactly match a bound price session")
    return calendar.index(value)


def _compact_identity(manifest: dict, digest: str, overlay_path: Path) -> dict:
    source = manifest["source_metadata"]
    return {
        "path": str(Path(overlay_path).absolute()), "manifest_sha256": digest,
        "events_sha256": manifest["events"]["sha256"],
        "price_source": manifest["price_source"],
        "groups": [g for g in GROUPS if manifest["events"]["nonmissing_values_by_group"][g] > 0],
        "max_age_days": manifest["max_age_days"],
        "taxonomy_metadata": manifest["taxonomy_metadata"],
        "availability_policy": source.get("availability_policy"),
        "historical_identity_limit": source.get("historical_identity_limit"),
    }


def _bound_metadata(price_dataset, overlay_path):
    manifest, digest = _manifest_metadata(Path(overlay_path))
    source, calendar, symbols = _read_source(Path(price_dataset), full=False)
    if manifest["price_source"] != source:
        raise DatasetError("classification overlay is bound to a different price snapshot")
    return manifest, digest, source, calendar, symbols


def classification_preview(price_dataset, overlay_path) -> dict:
    """Metadata-only UI check. Never read events or traverse price binaries."""
    manifest, digest, _source, _calendar, _symbols = _bound_metadata(price_dataset, overlay_path)
    return {**_compact_identity(manifest, digest, Path(overlay_path)), "status": "metadata_only"}


def _verified_boundary(price_dataset, overlay_path, *, retain_events=False):
    manifest, digest, source, calendar, symbols = _bound_metadata(price_dataset, overlay_path)
    verified, verified_calendar, verified_symbols = _read_source(Path(price_dataset))
    if (verified, verified_calendar, verified_symbols) != (source, calendar, symbols):
        raise DatasetError("price snapshot changed during classification verification")
    event_path = Path(overlay_path) / "events.jsonl"
    raw = event_path.read_bytes() if retain_events else None
    events_digest = hashlib.sha256(raw).hexdigest() if raw is not None else sha256(event_path)
    if events_digest != manifest["events"]["sha256"]:
        raise DatasetError("classification events SHA-256 mismatch")
    # Recheck the small seals/metadata after the expensive source reads.
    if _bound_metadata(price_dataset, overlay_path) != (manifest, digest, source, calendar, symbols):
        raise DatasetError("classification source changed during verification")
    return manifest, digest, source, calendar, symbols, raw


def classification_identity(price_dataset, overlay_path) -> dict:
    """Fully verify source bytes before a worker's result-cache lookup."""
    manifest, digest, *_ = _verified_boundary(price_dataset, overlay_path)
    return _compact_identity(manifest, digest, Path(overlay_path))


_CACHE_POLICY = "classification-interval-int32-v1"


def _cache_read(directory: Path, identity: dict, calendar: tuple, symbols: tuple) -> ClassificationOverlay:
    """Validate and return owned arrays; no file handles or mmap lifecycle."""
    import numpy as np
    try:
        _reject_redirects(directory)
        for child in directory.iterdir():
            _reject_redirects(child)
        raw = (directory / "metadata.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != (directory / "metadata.sha256").read_text(encoding="ascii").strip():
            raise DatasetError("classification cache metadata checksum mismatch")
        metadata = json.loads(raw)
        if (metadata["policy"] != _CACHE_POLICY or metadata["identity"] != identity
                or metadata["calendar"] != list(calendar) or metadata["symbols"] != list(symbols)
                or set(metadata["arrays"]) != {*GROUPS, "stale"}
                or set(metadata["labels"]) != set(GROUPS)):
            raise DatasetError("classification cache identity or dimensions mismatch")
        arrays = {}
        for name in (*GROUPS, "stale"):
            payload = (directory / f"{name}.npy").read_bytes()
            if hashlib.sha256(payload).hexdigest() != metadata["arrays"][name]:
                raise DatasetError("classification cache array checksum mismatch")
            array = np.load(io.BytesIO(payload), allow_pickle=False)
            expected_dtype = np.dtype("bool") if name == "stale" else np.dtype("int32")
            if array.shape != (len(calendar), len(symbols)) or array.dtype != expected_dtype:
                raise DatasetError("classification cache array shape or dtype mismatch")
            if name != "stale":
                labels = metadata["labels"][name]
                if (not isinstance(labels, list) or any(not isinstance(x, str) or not x for x in labels)
                        or labels != sorted(set(labels))
                        or np.any(array < -1) or np.any(array >= len(labels))):
                    raise DatasetError("classification cache labels or code range mismatch")
            array = array.copy()  # owned memory, with no underlying open file
            array.setflags(write=False)
            arrays[name] = array
        labels = {g: tuple(metadata["labels"][g]) for g in GROUPS}
        total = len(calendar) * len(symbols)
        stale_count = int(np.count_nonzero(arrays["stale"]))
        for group in GROUPS:
            labeled = int(np.count_nonzero(arrays[group] >= 0))
            if metadata["coverage"][group] != {
                    "labeled_cells": labeled, "unknown_cells": total - labeled,
                    "stale_cells": stale_count, "label_count": len(labels[group])}:
                raise DatasetError("classification cache coverage mismatch")
            if np.any(arrays[group][arrays["stale"]] != -1):
                raise DatasetError("classification cache stale cells must be unknown")
        summary = metadata["coverage"]["overlay"]
        if (summary["selected_sessions"] != len(calendar) or summary["selected_symbols"] != len(symbols)
                or summary["total_cells"] != total):
            raise DatasetError("classification cache coverage dimensions mismatch")
        return ClassificationOverlay(calendar, symbols, MappingProxyType({g: arrays[g] for g in GROUPS}),
                                     MappingProxyType(labels), arrays["stale"], identity, metadata["coverage"])
    except DatasetError:
        raise
    except (OSError, ValueError, KeyError, TypeError, EOFError) as error:
        raise DatasetError(f"invalid classification cache: {error}") from error


def load_classification_groups(price_dataset, overlay_path, *, cache_root: Path,
                               expected_identity: dict) -> ClassificationOverlay:
    """Verify sources, then reuse/publish sealed arrays. Corruption fails closed.

    Returned int32 codes use -1 for unknown. Arrays own their memory, are marked
    read-only, and require no close operation. Source files are never modified.
    """
    import numpy as np
    cache_root = Path(cache_root).absolute()
    _reject_redirects(cache_root)
    for source_path in (Path(price_dataset).absolute(), Path(overlay_path).absolute()):
        if cache_root == source_path or source_path in cache_root.parents or cache_root in source_path.parents:
            raise DatasetError("classification cache must be separate from source directories")
    manifest, digest, source, calendar, symbols, raw = _verified_boundary(
        price_dataset, overlay_path, retain_events=True)
    identity = _compact_identity(manifest, digest, Path(overlay_path))
    if identity != expected_identity:
        raise DatasetError("classification source mutation: expected identity differs")
    key = hashlib.sha256(_canonical_json({"policy": _CACHE_POLICY, "identity": identity})).hexdigest()
    directory = cache_root / key
    if directory.exists():
        del raw
        return _cache_read(directory, identity, calendar, symbols)
    events = _parse_events(raw, manifest)
    del raw
    overlay = read_classification_overlay(price_dataset, overlay_path,
        _verified=(manifest, events, digest, source, calendar, symbols))
    del events
    cache_root.mkdir(parents=True, exist_ok=True)
    _reject_redirects(cache_root)
    temporary = Path(tempfile.mkdtemp(prefix=".classification-", dir=cache_root))
    try:
        hashes = {}
        for name, array in {**overlay.codes, "stale": overlay.stale}.items():
            path = temporary / f"{name}.npy"
            np.save(path, array, allow_pickle=False)
            hashes[name] = sha256(path)
        metadata = {"policy": _CACHE_POLICY, "identity": identity, "calendar": calendar,
                    "symbols": symbols, "labels": overlay.labels,
                    "coverage": overlay.coverage, "arrays": hashes}
        payload = _canonical_json(metadata)
        (temporary / "metadata.json").write_bytes(payload)
        (temporary / "metadata.sha256").write_text(hashlib.sha256(payload).hexdigest(), encoding="ascii")
        try:
            os.rename(temporary, directory)
        except OSError:
            if not directory.is_dir():
                raise
            # Another worker may have won publication. Validate its cache.
            return _cache_read(directory, identity, calendar, symbols)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    # Cold results already own their arrays; avoid rereading the just-built cache.
    return ClassificationOverlay(calendar, symbols, MappingProxyType(dict(overlay.codes)),
                                 MappingProxyType(dict(overlay.labels)), overlay.stale, identity, overlay.coverage)
