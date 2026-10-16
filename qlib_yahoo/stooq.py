"""Read selected local Stooq daily files into an independently verified Qlib snapshot."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import secrets
import struct
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from qlib_yahoo import QLIB_COMMIT, QLIB_RELEASE
from qlib_yahoo.dataset import DatasetError, sha256, tree_sha256


HEADER = ("<TICKER>", "<PER>", "<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<VOL>", "<OPENINT>")
FIELDS = ("open", "high", "low", "close", "volume", "factor")
SYMBOL = re.compile(r"[A-Z][A-Z0-9-]*\Z")
RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"{p}{n}" for p in ("COM", "LPT") for n in range(10)}


def _write_json_atomically(path: Path, value: dict) -> None:
    """Use an exclusive unpredictable temp name; never open a preexisting redirect."""
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(16)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _redirected(path: Path, root: Path) -> bool:
    return (path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())
            or not _inside(path.resolve(), root))


def _inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _safe_roots(source_root: Path, data_root: Path) -> tuple[Path, Path]:
    if source_root.is_symlink() or (hasattr(source_root, "is_junction") and source_root.is_junction()):
        raise DatasetError(f"Stooq source root is redirected: {source_root}")
    source, output = source_root.resolve(), data_root.resolve()
    protected = source.parent
    if _inside(output, protected) or _inside(protected, output):
        raise DatasetError(f"output {output} overlaps read-only Stooq source parent {protected}")
    if not source.is_dir():
        raise DatasetError(f"Stooq source directory is missing: {source}")
    for child in (output / "datasets", output / "latest.json", output / f"upstream-qlib-{QLIB_RELEASE}"):
        resolved = child.resolve()
        if not _inside(resolved, output):
            raise DatasetError(f"output path redirects outside data root: {child} -> {resolved}")
        if _inside(resolved, protected) or _inside(protected, resolved):
            raise DatasetError(f"output path redirects into read-only Stooq source parent: {child} -> {resolved}")
    return source, output


def _source_files(source: Path, wanted: set[str]) -> dict[str, Path]:
    """Visit only the documented three directory levels; never follow archive links."""
    result: dict[str, Path] = {}
    base = source / "data" / "daily" / "us"
    if not base.is_dir() or any(_redirected(path, source) for path in (source / "data", source / "data" / "daily", base)):
        raise DatasetError(f"Stooq daily US directory is missing or redirected: {base}")
    with os.scandir(base) as groups:
        for group in groups:
            if Path(group.path).is_symlink() or (hasattr(Path(group.path), "is_junction") and Path(group.path).is_junction()):
                raise DatasetError(f"Stooq source directory is redirected: {group.path}")
            if not group.is_dir(follow_symlinks=False):
                continue
            if _redirected(Path(group.path), source):
                raise DatasetError(f"Stooq source directory is redirected: {group.path}")
            with os.scandir(group.path) as buckets:
                for bucket in buckets:
                    if Path(bucket.path).is_symlink() or (hasattr(Path(bucket.path), "is_junction") and Path(bucket.path).is_junction()):
                        raise DatasetError(f"Stooq source directory is redirected: {bucket.path}")
                    if not bucket.is_dir(follow_symlinks=False):
                        continue
                    if _redirected(Path(bucket.path), source):
                        raise DatasetError(f"Stooq source directory is redirected: {bucket.path}")
                    with os.scandir(bucket.path) as entries:
                        for entry in entries:
                            name = entry.name
                            if not name.lower().endswith(".us.txt"):
                                continue
                            symbol = name[:-7].upper()
                            if symbol not in wanted:
                                continue
                            path = Path(entry.path)
                            if not entry.is_file(follow_symlinks=False) or _redirected(path, source):
                                raise DatasetError(f"selected Stooq file is redirected or not regular: {path}")
                            if symbol in result:
                                raise DatasetError(f"ambiguous duplicate Stooq files for {symbol}: {result[symbol]} and {path}")
                            result[symbol] = path
    missing = wanted - result.keys()
    if missing:
        raise DatasetError(f"Stooq source is missing selected symbols: {', '.join(sorted(missing))}")
    return result


def _dump_script(output: Path) -> Path:
    checkout = output / f"upstream-qlib-{QLIB_RELEASE}"
    if not checkout.is_dir():
        raise DatasetError(f"pinned Qlib checkout is missing at {checkout}; install {QLIB_RELEASE} at commit {QLIB_COMMIT} first")
    git = ["git", "-c", f"safe.directory={checkout.resolve()}", "-C", str(checkout)]
    try:
        result = subprocess.run([*git, "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        detail = error.stderr.strip() if isinstance(error, subprocess.CalledProcessError) and error.stderr else str(error)
        raise DatasetError(f"cannot verify pinned Qlib checkout at {checkout}: {detail}") from error
    if result.stdout.strip() != QLIB_COMMIT:
        raise DatasetError(f"Qlib checkout at {checkout} must be pinned to {QLIB_COMMIT}; found {result.stdout.strip()}")
    script = checkout / "scripts" / "dump_bin.py"
    if not script.is_file():
        raise DatasetError(f"pinned Qlib dump script is missing: {script}")
    if _redirected(script, checkout.resolve()):
        raise DatasetError(f"pinned Qlib dump script is redirected: {script}")
    try:
        subprocess.run([*git, "show", "HEAD:scripts/dump_bin.py"], capture_output=True, check=True)
        difference = subprocess.run([*git, "diff", "--quiet", "HEAD", "--", "scripts/dump_bin.py"],
                                    capture_output=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise DatasetError(f"cannot read pinned dump script from Qlib HEAD at {checkout}: {error}") from error
    if difference.returncode != 0:
        raise DatasetError(f"pinned Qlib dump script differs from HEAD: {script}")
    return script


def _convert(path: Path, target: Path, symbol: str, start: date, end: date) -> tuple[int, int, set[str]]:
    count = zero = 0
    dates: set[str] = set()
    previous: date | None = None
    with path.open("r", encoding="utf-8-sig", newline="") as source, target.open("w", encoding="utf-8", newline="") as output:
        reader = csv.reader(source)
        if tuple(next(reader, ())) != HEADER:
            raise DatasetError(f"{path}: row 1: expected exact Stooq daily header")
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(("date", "symbol", *FIELDS))
        for line, row in enumerate(reader, 2):
            if len(row) != len(HEADER):
                raise DatasetError(f"{path}: row {line}: expected {len(HEADER)} columns, got {len(row)}")
            ticker, period, raw_day, clock = row[:4]
            if ticker.upper() != f"{symbol}.US" or period != "D" or clock != "000000":
                raise DatasetError(f"{path}: row {line}: ticker/period/time mismatch")
            try:
                if not re.fullmatch(r"\d{8}", raw_day):
                    raise ValueError("date format")
                day = datetime.strptime(raw_day, "%Y%m%d").date()
            except ValueError as error:
                raise DatasetError(f"{path}: row {line}: invalid YYYYMMDD date {raw_day!r}") from error
            if previous is not None and day <= previous:
                raise DatasetError(f"{path}: row {line}: duplicate or unsorted date {raw_day}")
            previous = day
            try:
                values = [float(value) for value in row[4:]]
            except ValueError as error:
                raise DatasetError(f"{path}: row {line}: invalid numeric field") from error
            if not all(math.isfinite(value) for value in values):
                raise DatasetError(f"{path}: row {line}: nonfinite numeric field")
            if values[5] != 0:
                raise DatasetError(f"{path}: row {line}: nonzero OpenInt is unsupported")
            if not start <= day < end:
                continue
            op, high, low, close, volume, _ = values
            if min(op, high, low, close) <= 0 or high < max(op, low, close) or low > min(op, high, close) or volume < 0:
                raise DatasetError(f"{path}: row {line}: invalid OHLC or negative volume")
            iso = day.isoformat()
            dates.add(iso)
            if volume == 0:
                zero += 1
                writer.writerow((iso, symbol, *("nan",) * len(FIELDS)))
            else:
                writer.writerow((iso, symbol, op, high, low, close, volume, 1))
            count += 1
    if not count:
        raise DatasetError(f"{path}: no bars in requested inclusive/exclusive interval")
    return count, zero, dates


def _verify_bins(snapshot: Path, symbols: tuple[str, ...], observed: set[str]) -> int:
    qlib = snapshot / "qlib_data"
    calendar = qlib / "calendars" / "day.txt"
    instruments = qlib / "instruments" / "all.txt"
    if not calendar.is_file() or not instruments.is_file():
        raise DatasetError("Qlib dump is missing calendar or instruments")
    sessions = calendar.read_text(encoding="utf-8").splitlines()
    if sessions != sorted(observed):
        raise DatasetError("Qlib calendar differs from union of observed Stooq dates")
    index = {day: i for i, day in enumerate(sessions)}
    expected_symbols = set(symbols)
    entries: dict[str, tuple[str, str]] = {}
    for line in instruments.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) != 3 or parts[0] in entries:
            raise DatasetError("Qlib instrument list is malformed")
        entries[parts[0]] = (parts[1], parts[2])
    if set(entries) != expected_symbols:
        raise DatasetError("Qlib instrument list differs from selected symbols")
    feature_root = qlib / "features"
    expected_files = {f"{symbol.lower()}/{field}.day.bin" for symbol in symbols for field in FIELDS}
    actual_files = {p.relative_to(feature_root).as_posix() for p in feature_root.rglob("*") if p.is_file()}
    if actual_files != expected_files:
        raise DatasetError(f"Qlib feature bin inventory mismatch; missing={sorted(expected_files-actual_files)}, extra={sorted(actual_files-expected_files)}")
    for symbol in symbols:
        with (snapshot / "normalized" / f"{symbol}.csv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["date", "symbol", *FIELDS]:
                raise DatasetError(f"normalized CSV schema changed for {symbol}")
            rows = list(reader)
        days = [row["date"] for row in rows]
        if entries[symbol] != (days[0], days[-1]):
            raise DatasetError(f"Qlib instrument range mismatch for {symbol}")
        first, last = index[days[0]], index[days[-1]]
        for field in FIELDS:
            expected = [float("nan")] * (last - first + 1)
            for row in rows:
                expected[index[row["date"]] - first] = float(row[field])
            path = feature_root / symbol.lower() / f"{field}.day.bin"
            payload = path.read_bytes()
            if len(payload) != 4 * (len(expected) + 1):
                raise DatasetError(f"Qlib bin length mismatch: {path}")
            actual = struct.unpack(f"<{len(expected)+1}f", payload)
            if actual[0] != first:
                raise DatasetError(f"Qlib bin calendar offset mismatch: {path}")
            for offset, (got, want) in enumerate(zip(actual[1:], expected)):
                if not (math.isnan(want) and math.isnan(got)):
                    try:
                        same = struct.pack("<f", got) == struct.pack("<f", want)
                    except OverflowError:
                        same = False
                    if not same:
                        raise DatasetError(f"Qlib bin readback mismatch: {symbol}/{field} on {sessions[first+offset]}")
    return len(sessions)


def prepare_stooq(source_root: Path, data_root: Path, symbols: tuple[str, ...], start: str, end: str,
                  *, make_latest: bool = False, max_workers: int = 2) -> Path:
    """Build a unique local dataset; every source file remains in its original location."""
    source, output = _safe_roots(Path(source_root), Path(data_root))
    if not isinstance(symbols, tuple) or not symbols or any(not isinstance(s, str) for s in symbols):
        raise DatasetError("symbols must be a nonempty tuple of US ticker names")
    symbols = tuple(s.strip().upper() for s in symbols)
    if any(not SYMBOL.fullmatch(s) or s in RESERVED for s in symbols):
        raise DatasetError("symbols must be safe US ticker names")
    if len(set(symbols)) != len(symbols):
        raise DatasetError("duplicate symbols are forbidden")
    if not isinstance(max_workers, int) or isinstance(max_workers, bool) or max_workers < 1:
        raise DatasetError("max_workers must be a positive integer")
    try:
        begin, stop = date.fromisoformat(start), date.fromisoformat(end)
        if begin.isoformat() != start or stop.isoformat() != end or begin >= stop:
            raise ValueError("interval")
    except (TypeError, ValueError) as error:
        raise DatasetError("start/end require YYYY-MM-DD with inclusive start and exclusive later end") from error
    script = _dump_script(output)
    selected = symbols
    files = _source_files(source, set(selected))
    dataset_dir = output / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    snapshot = dataset_dir / f"stooq-us-day-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    snapshot.mkdir(exist_ok=False)
    progress = {"status": "building", "source_root": str(source), "source_files": {s: str(files[s]) for s in selected}}
    try:
        _write_json_atomically(snapshot / "progress.json", progress)
        normalized = snapshot / "normalized"
        normalized.mkdir()
        qlib = snapshot / "qlib_data"
        qlib.mkdir()
        read_started = datetime.now(timezone.utc).isoformat()
        before = {s: sha256(files[s]) for s in selected}
        source_stats_before = {s: {"size_bytes": files[s].stat().st_size, "mtime_ns": files[s].stat().st_mtime_ns}
                               for s in selected}
        counts: dict[str, int] = {}
        zeros: dict[str, int] = {}
        observed: set[str] = set()
        for symbol in selected:
            counts[symbol], zeros[symbol], dates = _convert(files[symbol], normalized / f"{symbol}.csv", symbol, begin, stop)
            observed.update(dates)
            progress.update({"row_counts": counts, "zero_volume_rows": zeros})
            _write_json_atomically(snapshot / "progress.json", progress)
        after = {s: sha256(files[s]) for s in selected}
        source_stats_after = {s: {"size_bytes": files[s].stat().st_size, "mtime_ns": files[s].stat().st_mtime_ns}
                              for s in selected}
        if before != after or source_stats_before != source_stats_after:
            raise DatasetError("Stooq source changed during conversion; source checksums differ")
        read_finished = datetime.now(timezone.utc).isoformat()
        try:
            subprocess.run([sys.executable, str(script), "dump_all", "--data_path", str(normalized),
                            "--qlib_dir", str(qlib), "--freq", "day", "--date_field_name", "date",
                            "--symbol_field_name", "symbol", "--exclude_fields", "date,symbol",
                            "--file_suffix", ".csv", "--max_workers", str(max_workers)], check=True)
        except (OSError, subprocess.CalledProcessError) as error:
            raise DatasetError(f"Qlib dump failed: {error}") from error
        sessions = _verify_bins(snapshot, selected, observed)
        if ({s: sha256(files[s]) for s in selected} != before or
                {s: {"size_bytes": files[s].stat().st_size, "mtime_ns": files[s].stat().st_mtime_ns}
                 for s in selected} != source_stats_before):
            raise DatasetError("Stooq source changed during dump verification")
        binary_hash, file_count = tree_sha256(qlib)
        manifest = {
            "schema_version": 1, "status": "ready", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "provider": "Stooq local historical US daily archive", "qlib_source": {"release": QLIB_RELEASE, "commit": QLIB_COMMIT},
            "date_range": {"start_inclusive": start, "end_exclusive": end}, "symbols": list(symbols),
            "benchmark": None, "row_counts": counts, "zero_volume_rows": zeros, "sessions": sessions,
            "raw_sha256": {str(files[s]): before[s] for s in selected},
            "source_sha256_before": before, "source_sha256_after": after,
            "source_stats_before": source_stats_before, "source_stats_after": source_stats_after,
            "source_read_started_at_utc": read_started, "source_read_finished_at_utc": read_finished,
            "normalized_sha256": {f"{s}.csv": sha256(normalized / f"{s}.csv") for s in selected},
            "qlib_calendar_sha256": sha256(qlib / "calendars" / "day.txt"),
            "qlib_instruments_sha256": sha256(qlib / "instruments" / "all.txt"),
            "qlib_binary_tree_sha256": binary_hash, "qlib_binary_file_count": file_count,
            "source_contract": {"format": "Stooq US daily text", "header": list(HEADER), "frequency": "D", "time": "000000",
                                "symbol_mapping": "uppercase ticker; strip .US filename suffix", "source_files": {s: str(files[s]) for s in selected},
                                "parser": "qlib_yahoo.stooq schema-1 strict daily parser",
                                "parser_sha256": sha256(Path(__file__)), "converter_sha256": sha256(script)},
            "normalization": "OHLCV preserved in supplied units; factor=1 identity; zero-volume executable fields NaN",
            "calendar": "union of observed selected-symbol dates; missing bars remain missing",
            "limitations": ["Explicit symbols are not a historical investment universe.",
                            "Raw-versus-adjusted price and corporate actions are unverified; factor=1 is only an identity marker.",
                            "No action, dividend, delisting, publication-time or redistribution guarantee from source archive."],
        }
        if make_latest:
            _safe_roots(source, output)
        progress["status"] = "ready"
        _write_json_atomically(snapshot / "progress.json", progress)
        _write_json_atomically(snapshot / "manifest.json", manifest)
        if make_latest:
            _write_json_atomically(output / "latest.json", {"dataset": str(snapshot)})
    except Exception as error:
        progress["status"] = "failed"
        progress["error"] = str(error)
        manifest_path = snapshot / "manifest.json"
        if manifest_path.is_file():
            recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
            recorded["status"] = "failed"
            recorded["error"] = str(error)
            _write_json_atomically(manifest_path, recorded)
        _write_json_atomically(snapshot / "progress.json", progress)
        _write_json_atomically(snapshot / "failed_manifest.json", progress)
        raise
    print(f"Ready: {snapshot}", flush=True)
    return snapshot
