"""Verify prepared Qlib data before a backtest, including legacy snapshots."""

from __future__ import annotations

import csv
import math
import re
import struct
from pathlib import Path
from typing import Any

from qlib_yahoo.dataset import DatasetError, sha256, tree_sha256


_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_FIELDS = ("open", "high", "low", "close", "volume", "factor")


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise DatasetError(f"dataset manifest has invalid {label}; expected a SHA-256 hex digest")
    return value


def _check_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256(path) != _digest(expected, label):
        raise DatasetError(f"dataset {label} is missing or differs from its manifest")


def _check_csv_hashes(dataset: Path, manifest: dict[str, Any], key: str, expected_names: set[str]) -> None:
    recorded = manifest.get(key)
    if not isinstance(recorded, dict) or set(recorded) != expected_names:
        raise DatasetError(f"legacy dataset manifest {key} must list exactly the symbols and benchmark")
    directory = dataset / key.removesuffix("_sha256")
    actual_names = {path.name for path in directory.glob("*.csv") if path.is_file()}
    if actual_names != expected_names:
        raise DatasetError(f"dataset {directory.name} CSV files differ from its manifest")
    for name in sorted(expected_names):
        _check_file(directory / name, recorded[name], f"{key}[{name}]")


def _legacy_bins(dataset: Path, symbols: list[str], benchmark: str, sessions: list[str]) -> int:
    qlib_dir = dataset / "qlib_data"
    instrument_lines = (qlib_dir / "instruments" / "all.txt").read_text(encoding="utf-8").splitlines()
    expected_symbols = set((*symbols, benchmark))
    instrument_map: dict[str, tuple[str, str]] = {}
    for line in instrument_lines:
        parts = line.split("\t")
        if len(parts) != 3 or parts[0] in instrument_map:
            raise DatasetError("legacy Qlib instrument list is malformed")
        instrument_map[parts[0]] = (parts[1], parts[2])
    if set(instrument_map) != expected_symbols:
        raise DatasetError("legacy Qlib instruments differ from manifest symbols")

    features = qlib_dir / "features"
    expected_paths = {f"{symbol.lower()}/{field}.day.bin"
                      for symbol in expected_symbols for field in _FIELDS}
    actual_paths = {path.relative_to(features).as_posix() for path in features.rglob("*") if path.is_file()}
    if actual_paths != expected_paths:
        raise DatasetError("legacy Qlib feature-bin inventory differs from normalized CSV fields")
    if len(sessions) != len(set(sessions)) or sessions != sorted(sessions):
        raise DatasetError("legacy Qlib calendar is duplicate or unordered")
    calendar_index = {day: index for index, day in enumerate(sessions)}
    for symbol in sorted(expected_symbols):
        path = dataset / "normalized" / f"{symbol}.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["date", "symbol", *_FIELDS]:
                raise DatasetError(f"legacy normalized CSV has unexpected columns: {path.name}")
            rows = list(reader)
        if not rows or any(row["symbol"] != symbol for row in rows):
            raise DatasetError(f"legacy normalized CSV has invalid symbol or no rows: {path.name}")
        dates = [row["date"] for row in rows]
        if len(dates) != len(set(dates)) or dates != sorted(dates) or any(day not in calendar_index for day in dates):
            raise DatasetError(f"legacy normalized CSV dates are duplicate, unordered or outside calendar: {path.name}")
        if instrument_map[symbol] != (dates[0], dates[-1]):
            raise DatasetError(f"legacy Qlib instrument range differs from normalized CSV: {symbol}")
        first = calendar_index[dates[0]]
        last = calendar_index[dates[-1]]
        for field in _FIELDS:
            expected = [float("nan")] * (last - first + 1)
            for row in rows:
                try:
                    expected[calendar_index[row["date"]] - first] = float(row[field]) if row[field] else float("nan")
                except ValueError as error:
                    raise DatasetError(f"legacy normalized CSV has invalid {field}: {path.name}") from error
            bin_path = features / symbol.lower() / f"{field}.day.bin"
            payload = bin_path.read_bytes()
            if len(payload) != 4 * (len(expected) + 1):
                raise DatasetError(f"legacy Qlib bin length differs from normalized CSV: {symbol}/{field}")
            values = struct.unpack(f"<{len(expected) + 1}f", payload)
            if values[0] != first:
                raise DatasetError(f"legacy Qlib bin start index differs from calendar: {symbol}/{field}")
            for offset, (actual, source) in enumerate(zip(values[1:], expected)):
                if math.isnan(source):
                    equal = math.isnan(actual)
                else:
                    try:
                        equal = struct.pack("<f", source) == struct.pack("<f", actual)
                    except OverflowError:
                        equal = False
                if not equal:
                    raise DatasetError(f"legacy Qlib bin differs from normalized CSV: {symbol}/{field} on {sessions[first + offset]}")
    return len(expected_paths)


def verify_dataset(dataset: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Return provenance for a fully verified strict or legacy dataset."""
    if not isinstance(manifest, dict):
        raise DatasetError("dataset manifest must be a JSON object")
    if "status" in manifest and manifest["status"] != "ready":
        raise DatasetError(f"dataset manifest status must be ready before backtesting; got {manifest['status']!r}")
    if "storage" in manifest:
        from qlib_yahoo.packed import verify_packed
        return verify_packed(dataset, manifest)
    tree_hash = manifest.get("qlib_binary_tree_sha256")
    if tree_hash is not None:
        _digest(tree_hash, "qlib_binary_tree_sha256")
        actual_hash, count = tree_sha256(dataset / "qlib_data")
        if actual_hash != tree_hash:
            raise DatasetError("prepared Qlib binary tree differs from dataset manifest")
        return {"mode": "recorded_binary_tree_sha256", "qlib_binary_tree_sha256": actual_hash,
                "qlib_binary_file_count": count}

    if "qlib_binary_tree_sha256" in manifest or "qlib_binary_file_count" in manifest:
        raise DatasetError("dataset manifest has incomplete Qlib binary-tree provenance; expected qlib_binary_tree_sha256")
    if manifest.get("schema_version") != 1 or manifest.get("status") != "ready":
        raise DatasetError("dataset manifest lacks qlib_binary_tree_sha256; legacy verification requires schema_version 1 and status ready")
    symbols, benchmark = manifest.get("symbols"), manifest.get("benchmark")
    if (not isinstance(symbols, list) or not symbols or not isinstance(benchmark, str)
            or any(not isinstance(symbol, str) or not re.fullmatch(r"[A-Z][A-Z0-9-]*", symbol) for symbol in [*symbols, benchmark])
            or len(set([*symbols, benchmark])) != len(symbols) + 1):
        raise DatasetError("legacy dataset manifest has invalid symbols or benchmark")
    names = {f"{symbol}.csv" for symbol in [*symbols, benchmark]}
    for key in ("raw_sha256", "normalized_sha256"):
        _check_csv_hashes(dataset, manifest, key, names)
    qlib_dir = dataset / "qlib_data"
    for key, relative in (("qlib_calendar_sha256", "calendars/day.txt"),
                          ("qlib_instruments_sha256", "instruments/all.txt")):
        _check_file(qlib_dir / relative, manifest.get(key), key)
    allowed = {"calendars/day.txt", "instruments/all.txt"}
    actual_nonbins = {path.relative_to(qlib_dir).as_posix() for path in qlib_dir.rglob("*")
                      if path.is_file() and path.suffix != ".bin"}
    if actual_nonbins != allowed:
        raise DatasetError("legacy Qlib data contains unverified non-bin files")
    sessions = (qlib_dir / "calendars" / "day.txt").read_text(encoding="utf-8").splitlines()
    if not sessions or manifest.get("sessions") != len(sessions):
        raise DatasetError("legacy dataset manifest session count differs from calendar")
    count = _legacy_bins(dataset, symbols, benchmark, sessions)
    actual_hash, file_count = tree_sha256(qlib_dir)
    if file_count != count + 2:
        raise DatasetError("legacy Qlib data contains unexpected files")
    return {"mode": "legacy_source_csv_calendar_bin_float32_v1",
            "qlib_binary_tree_sha256": actual_hash, "qlib_binary_file_count": file_count,
            "note": "Tree hash computed at run time; source CSV, calendar and instruments matched recorded hashes; bins matched normalized CSV float32 values and NaNs."}
