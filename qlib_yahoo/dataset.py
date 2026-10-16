"""Acquire a frozen, explicitly selected Yahoo sample and convert it to Qlib bins."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from qlib_yahoo import QLIB_COMMIT, QLIB_RELEASE


SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9-]*$")
WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(10)
}
RAW_COLUMNS = ("date", "symbol", "open", "high", "low", "close", "volume", "adjclose")
NORMALIZED_COLUMNS = ("date", "symbol", "open", "high", "low", "close", "volume", "factor")


class DatasetError(RuntimeError):
    """Dataset preparation failed; no latest-dataset pointer was changed."""


def parse_symbols(value: str | Iterable[str]) -> tuple[str, ...]:
    parts = value.split(",") if isinstance(value, str) else value
    symbols = tuple(str(item).strip().upper() for item in parts)
    if not symbols or any(not SYMBOL_PATTERN.fullmatch(item) for item in symbols):
        raise DatasetError("symbols must be comma-separated US tickers, e.g. AAPL,MSFT,BRK-B")
    if any(item in WINDOWS_RESERVED for item in symbols):
        raise DatasetError("Windows-reserved ticker filenames require a separate identifier mapping")
    if len(set(symbols)) != len(symbols):
        raise DatasetError("duplicate symbols are not allowed")
    if len(symbols) < 6:
        raise DatasetError("provide at least six trade symbols for the top-five example")
    return symbols


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_hashes(directory: Path, pattern: str = "*.csv") -> dict[str, str]:
    return {path.name: sha256(path) for path in sorted(directory.glob(pattern))}


# Files that Finder/Explorer create when a folder is opened; they never belong to a snapshot.
OS_METADATA = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})


def is_os_metadata(path: Path) -> bool:
    return path.name in OS_METADATA or path.name.startswith("._")


def tree_sha256(directory: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = sorted(path for path in directory.rglob("*") if path.is_file() and not is_os_metadata(path))
    for path in files:
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest(), len(files)


def _write_json_atomically(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def ensure_upstream(data_root: Path) -> Path:
    checkout = data_root / f"upstream-qlib-{QLIB_RELEASE}"
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", QLIB_RELEASE,
             "https://github.com/microsoft/qlib.git", str(checkout)],
            check=True,
        )
    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    )
    if result.stdout.strip() != QLIB_COMMIT:
        raise DatasetError(f"Qlib source at {checkout} is not the reviewed {QLIB_RELEASE} commit")
    dump_script = checkout / "scripts" / "dump_bin.py"
    if not dump_script.is_file():
        raise DatasetError(f"Qlib dump script is missing: {dump_script}")
    return dump_script


def fetch_yahoo(symbol: str, start: str, end: str, attempts: int, pause: float) -> Any:
    import numpy as np
    import pandas as pd
    from yahooquery import Ticker

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            result = Ticker(symbol, asynchronous=False, timeout=30).history(
                start=start, end=end, interval="1d"
            )
            if not isinstance(result, pd.DataFrame) or result.empty:
                raise DatasetError(f"Yahoo returned no daily frame for {symbol}: {str(result)[:200]}")
            frame = result.reset_index()
            missing = set(RAW_COLUMNS) - set(frame.columns)
            if missing:
                raise DatasetError(f"Yahoo omitted {sorted(missing)} for {symbol}")
            frame = frame.loc[:, list(RAW_COLUMNS)].copy()
            frame["date"] = pd.to_datetime(frame["date"]).dt.date.astype(str)
            frame["symbol"] = symbol
            frame = frame.sort_values("date").reset_index(drop=True)
            if frame["date"].duplicated().any():
                raise DatasetError(f"Yahoo returned duplicate daily bars for {symbol}")
            if len(frame) < 200:
                raise DatasetError(f"only {len(frame)} daily bars for {symbol}; need at least 200")
            for column in ("open", "high", "low", "close", "volume", "adjclose"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            if not np.isfinite(frame[list(RAW_COLUMNS[2:])].to_numpy(dtype=float)).all():
                raise DatasetError(f"Yahoo has non-finite OHLCV/adjusted-close values for {symbol}")
            if ((frame[["open", "high", "low", "close", "adjclose"]] <= 0).any().any()
                    or (frame["volume"] < 0).any()):
                raise DatasetError(f"Yahoo has nonpositive prices or negative volume for {symbol}")
            if ((frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any()
                    or (frame["low"] > frame[["open", "close"]].min(axis=1)).any()):
                raise DatasetError(f"Yahoo has inconsistent OHLC bars for {symbol}")
            return frame
        except Exception as error:
            last_error = error
            if attempt < attempts:
                time.sleep(pause * attempt)
    raise DatasetError(f"failed to fetch/validate {symbol} after {attempts} attempts: {last_error}")


def normalize_frame(raw: Any) -> Any:
    """Match Qlib's Yahoo 1d adjclose/close and first-close scaling for OHLCV."""
    frame = raw.copy()
    factor = frame["adjclose"] / frame["close"]
    first_adjusted_close = float(frame["adjclose"].iloc[0])
    for field in ("open", "high", "low", "close"):
        frame[field] = frame[field] * factor / first_adjusted_close
    frame["volume"] = frame["volume"] / factor * first_adjusted_close
    frame["factor"] = factor / first_adjusted_close
    frame = frame.loc[:, list(NORMALIZED_COLUMNS)]
    # A zero-volume date remains in the calendar but is not an executable bar.
    frame.loc[frame["volume"] <= 0, list(NORMALIZED_COLUMNS[2:])] = float("nan")
    return frame


def prepare(
    data_root: Path, start: str, end: str, symbols: tuple[str, ...],
    attempts: int = 3, pause: float = 1.0, resume_snapshot: Path | None = None,
) -> Path:
    data_root = data_root.resolve()
    if date.fromisoformat(end) <= date.fromisoformat(start):
        raise DatasetError("--end must be after --start (Yahoo end is exclusive)")
    if attempts < 1 or pause < 0:
        raise DatasetError("attempts must be positive and pause nonnegative")
    request = {
        "start_inclusive": start, "end_exclusive": end,
        "symbols": list(symbols), "benchmark": None,
    }
    if resume_snapshot is None:
        dump_script = ensure_upstream(data_root)
        identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        snapshot = data_root / "datasets" / f"yahoo-us-day-{identifier}"
        snapshot.mkdir(parents=True, exist_ok=False)
        _write_json_atomically(snapshot / "request.json", request)
    else:
        snapshot = resume_snapshot.resolve()
        if snapshot.parent != (data_root / "datasets").resolve():
            raise DatasetError("--resume-snapshot must name a dataset directly under the selected data root")
        request_path = snapshot / "request.json"
        if not request_path.is_file() or json.loads(request_path.read_text(encoding="utf-8")) != request:
            raise DatasetError("resume request does not match the snapshot's symbols, dates, or benchmark contract")
        if (snapshot / "manifest.json").is_file():
            print(f"Already ready: {snapshot}")
            return snapshot
        dump_script = ensure_upstream(data_root)
    print(f"Snapshot path: {snapshot}", flush=True)
    raw_dir, normalized_dir, qlib_dir = (snapshot / name for name in ("raw", "normalized", "qlib_data"))
    for path in (raw_dir, normalized_dir, qlib_dir):
        path.mkdir(parents=True, exist_ok=True)

    progress_path = snapshot / "progress.json"
    progress: dict[str, dict[str, Any]] = (
        json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
    )
    counts: dict[str, int] = {}
    for symbol in symbols:
        raw_path = raw_dir / f"{symbol}.csv"
        normalized_path = normalized_dir / f"{symbol}.csv"
        prior = progress.get(symbol)
        if prior is not None:
            if (not raw_path.is_file() or not normalized_path.is_file()
                    or sha256(raw_path) != prior["raw_sha256"]
                    or sha256(normalized_path) != prior["normalized_sha256"]):
                raise DatasetError(f"completed {symbol} files changed; inspect the snapshot before resuming")
            counts[symbol] = int(prior["rows"])
            print(f"Reusing {symbol} ({counts[symbol]} bars)", flush=True)
            continue
        if raw_path.exists() or normalized_path.exists():
            raise DatasetError(f"unrecorded partial files exist for {symbol}; inspect {snapshot}")
        print(f"Downloading {symbol}...", flush=True)
        raw = fetch_yahoo(symbol, start, end, attempts, pause)
        normalized = normalize_frame(raw)
        raw.to_csv(raw_path.with_suffix(".csv.tmp"), index=False, lineterminator="\n")
        normalized.to_csv(normalized_path.with_suffix(".csv.tmp"), index=False, lineterminator="\n")
        raw_path.with_suffix(".csv.tmp").replace(raw_path)
        normalized_path.with_suffix(".csv.tmp").replace(normalized_path)
        counts[symbol] = len(raw)
        progress[symbol] = {
            "rows": len(raw), "raw_sha256": sha256(raw_path),
            "normalized_sha256": sha256(normalized_path),
        }
        _write_json_atomically(progress_path, progress)
        if pause:
            time.sleep(pause)

    subprocess.run(
        [sys.executable, str(dump_script), "dump_all",
         "--data_path", str(normalized_dir), "--qlib_dir", str(qlib_dir),
         "--freq", "day", "--exclude_fields", "date,symbol",
         "--file_suffix", ".csv", "--max_workers", "2"],
        check=True,
    )
    calendar = qlib_dir / "calendars" / "day.txt"
    instruments = qlib_dir / "instruments" / "all.txt"
    if not calendar.is_file() or not instruments.is_file():
        raise DatasetError("Qlib dump did not produce its calendar and instrument list")
    dumped = {line.split("\t", 1)[0].upper() for line in instruments.read_text(encoding="utf-8").splitlines()}
    if dumped != set(symbols):
        raise DatasetError(f"Qlib instrument list mismatch: expected {len(symbols)}, got {len(dumped)}")
    calendar_lines = calendar.read_text(encoding="utf-8").splitlines()
    if len(calendar_lines) < 250:
        raise DatasetError(f"Qlib dumped only {len(calendar_lines)} sessions; need at least 250")
    qlib_hash, qlib_file_count = tree_sha256(qlib_dir)

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ready",
        "provider": "Yahoo Finance via yahooquery; explicit current-symbol convenience sample",
        "qlib_source": {"release": QLIB_RELEASE, "commit": QLIB_COMMIT},
        "date_range": {"start_inclusive": start, "end_exclusive": end},
        "symbols": list(symbols),
        "benchmark": None,
        "row_counts": counts,
        "sessions": len(calendar_lines),
        "raw_sha256": file_hashes(raw_dir),
        "normalized_sha256": file_hashes(normalized_dir),
        "qlib_calendar_sha256": sha256(calendar),
        "qlib_instruments_sha256": sha256(instruments),
        "qlib_binary_tree_sha256": qlib_hash,
        "qlib_binary_file_count": qlib_file_count,
        "normalization": "adjclose/close factor; adjusted OHLC scaled to first adjusted close=1; inverse-adjusted volume",
        "limitations": [
            "Current selected symbols, not a historical universe; survivorship bias.",
            "Yahoo bars and adjusted closes are unverified and may be revised or anomalous.",
            "No independent corporate-action, dividend-cash, or delisting ledger.",
            "Data access/redistribution rights have not been established for sharing this snapshot.",
        ],
    }
    _write_json_atomically(snapshot / "manifest.json", manifest)
    pointer = data_root / "latest.json"
    _write_json_atomically(pointer, {"dataset": str(snapshot)})
    print(f"Ready: {snapshot}")
    return snapshot


def resolve_dataset(data_root: Path, dataset: Path | None) -> Path:
    if dataset is None:
        pointer = data_root / "latest.json"
        if not pointer.is_file():
            raise DatasetError(f"no prepared dataset at {pointer}; run 'prepare' first")
        dataset = Path(json.loads(pointer.read_text(encoding="utf-8"))["dataset"])
    # Preserve the caller's spelling until the packed-storage redirect check.
    # Otherwise resolving a junction here would hide it from run_backtest.
    dataset = Path(dataset).absolute()
    manifest_path = dataset / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest, dict) and "storage" in manifest:
            from qlib_yahoo.packed import _reject_redirects
            _reject_redirects(manifest_path)
    dataset = dataset.resolve()
    if not (dataset / "manifest.json").is_file() or not (dataset / "qlib_data").is_dir():
        raise DatasetError(f"not a completed Qlib dataset: {dataset}")
    return dataset
