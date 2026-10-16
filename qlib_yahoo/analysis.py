"""Offline reporting from immutable legacy and strategy-only run artifacts."""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from uuid import uuid4

CORE_FILES = {"summary.json", "daily_report.csv", "positions.csv.gz"}  # schemas 3-5
CORE_FILES_PARQUET = {"summary.json", "daily_report.csv", "positions.parquet"}  # schema 6
REPLAY_SCHEMAS = (3, 4, 5, 6)
HOLDINGS_TRACES = 25  # individual instruments drawn in the holdings chart; the rest are aggregated


def _redirect(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _file_identity(path: Path) -> tuple[int, ...]:
    if _redirect(path) or not path.is_file():
        raise ValueError(f"Run artifact missing or unsafe: {path.name}")
    stat = path.stat()
    change_time = stat.st_ctime_ns
    if os.name == "nt":
        # Windows stat().st_ctime is creation time, not metadata/content change
        # time. NTFS ChangeTime also detects same-size edits with restored mtime.
        import ctypes
        from ctypes import wintypes

        class FileBasicInfo(ctypes.Structure):
            _fields_ = [(name, ctypes.c_longlong) for name in
                        ("CreationTime", "LastAccessTime", "LastWriteTime", "ChangeTime")] + [
                            ("FileAttributes", wintypes.DWORD)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        create = kernel.CreateFileW
        create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                           wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        create.restype = wintypes.HANDLE
        query = kernel.GetFileInformationByHandleEx
        query.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        query.restype = wintypes.BOOL
        close = kernel.CloseHandle
        close.argtypes = [wintypes.HANDLE]
        close.restype = wintypes.BOOL
        handle = create(str(path), 0x80, 7, None, 3, 0, None)  # attributes only, share read/write/delete
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            info = FileBasicInfo()
            if not query(handle, 0, ctypes.byref(info), ctypes.sizeof(info)):
                raise ctypes.WinError(ctypes.get_last_error())
            change_time = info.ChangeTime
        finally:
            close(handle)
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, change_time)


def _sha256(path: Path) -> str:
    before = _file_identity(path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if _file_identity(path) != before:
        raise ValueError(f"Run artifact changed while reading: {path.name}")
    return digest.hexdigest()


@dataclass
class ValidatedBundle:
    """Request-local validation evidence; never retained as a file timestamp cache."""

    run_dir: Path
    manifest: dict[str, Any]
    summary: dict[str, Any]
    frame: Any
    cash: float
    provenance: dict[str, Any]
    identities: dict[str, tuple[int, ...]]

    def assert_unchanged(self) -> None:
        if _redirect(self.run_dir):
            raise ValueError("Run directory must not be a symlink or junction")
        if _redirect(self.run_dir / "reports"):
            raise ValueError("Reports directory must not be a symlink or junction")
        for name, identity in self.identities.items():
            if _file_identity(self.run_dir / name) != identity:
                raise ValueError(f"Run artifact changed during report request: {name}")


def validate_bundle(run_dir: Path) -> ValidatedBundle:
    run_dir = Path(run_dir)
    # Snapshot before hashing/reconciliation, including the seal itself.
    manifest_path = run_dir / "manifest.json"
    initial_manifest = _file_identity(manifest_path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = raw.get("artifacts_sha256", {})
    if not isinstance(names, dict):
        raise ValueError("Run manifest lacks required core artifact hashes")
    identities = {"manifest.json": initial_manifest}
    for name in names:
        if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Run manifest contains an invalid artifact entry")
        identities[name] = _file_identity(run_dir / name)
    manifest, summary = _bundle(run_dir)
    signed, benchmark = _is_signed(manifest), benchmark_enabled(manifest)
    frame = _daily(run_dir, signed=signed, benchmark_enabled=benchmark)
    cash = (_reconcile_signed(run_dir, frame, summary, benchmark_enabled=benchmark,
                             positions_order=manifest.get("history_storage", {}).get("positions_order")) if signed
            else _reconcile_ledger(frame, summary, benchmark_enabled=benchmark))
    bundle = ValidatedBundle(run_dir, manifest, summary, frame, cash, _provenance(run_dir), identities)
    bundle.assert_unchanged()
    return bundle


def _group_returns(pred_label: Any, groups: int = 5) -> Any:
    """Hypothetical forward-label means in near-equal score buckets."""
    import pandas as pd

    rows = []
    for date, frame in pred_label.dropna(subset=["score", "label"]).groupby(level="datetime"):
        ordered = frame.sort_values("score", ascending=False)
        if len(ordered) < groups:
            continue
        values = ordered["label"].to_numpy()
        base, remainder = divmod(len(values), groups)
        row = {"datetime": date, "eligible_count": len(values), "group_size": base}
        offset = 0
        for index in range(groups):
            size = base + (index < remainder)
            row[f"group_{index + 1}"] = float(values[offset:offset + size].mean())
            offset += size
        row["long_short_spread"] = row["group_1"] - row[f"group_{groups}"]
        row["top_minus_universe"] = row["group_1"] - float(ordered["label"].mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("datetime") if rows else pd.DataFrame()


def _score_turnover(pred_label: Any, groups: int = 5) -> Any:
    """Bucket replacement between adjacent sessions, on common instruments."""
    import pandas as pd

    rows = []
    previous = None
    for date, frame in pred_label[["score"]].groupby(level="datetime", sort=True):
        current = frame.droplevel("datetime")["score"].dropna()
        if previous is not None:
            common = current.index.intersection(previous.index)
            size = len(common) // groups
            if size:
                now, old = current.loc[common], previous.loc[common]
                rows.append({
                    "datetime": date,
                    "top_score_bucket_turnover": 1 - len(set(now.nlargest(size).index) & set(old.nlargest(size).index)) / size,
                    "bottom_score_bucket_turnover": 1 - len(set(now.nsmallest(size).index) & set(old.nsmallest(size).index)) / size,
                })
        previous = current
    return pd.DataFrame(rows).set_index("datetime") if rows else pd.DataFrame()


def table_artifact(run_dir: Path, stem: str) -> Path:
    """Select a sealed table by its manifest, never by an unlisted shadow file.

    The filesystem preference remains available for standalone, unsealed fixtures.
    """
    manifest_path = Path(run_dir) / "manifest.json"
    if manifest_path.exists() or _redirect(manifest_path):
        _file_identity(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        hashes = manifest.get("artifacts_sha256")
        if not isinstance(hashes, dict):
            raise ValueError("Run manifest lacks required core artifact hashes")
        selected = [name for name in (f"{stem}.parquet", f"{stem}.csv.gz") if name in hashes]
        if len(selected) != 1:
            raise ValueError(f"Run manifest must seal exactly one table format: {stem}")
        return Path(run_dir) / selected[0]
    parquet = Path(run_dir) / f"{stem}.parquet"
    return parquet if parquet.is_file() else Path(run_dir) / f"{stem}.csv.gz"


def read_instrument_table(path: Path, columns: list[str] | None = None) -> Any:
    """Read a saved per-instrument table in either format with the same column semantics."""
    import pandas as pd

    if Path(path).suffix != ".parquet":
        return read_instrument_csv(path, columns=columns)
    frame = pd.read_parquet(path, columns=columns)
    if "datetime" in frame:
        frame["datetime"] = pd.to_datetime(frame["datetime"])
    for column in frame.columns[frame.dtypes == object]:
        frame[column] = frame[column].mask(frame[column] == "")  # as in CSV: only empty cells are missing
    return frame


def benchmark_enabled(manifest: dict[str, Any]) -> bool:
    """Schemas 3 and 4 carried a benchmark reference; strategy-only runs (5 and later) do not."""
    return manifest["schema_version"] < 5


def read_instrument_csv(path: Path, columns: list[str] | None = None) -> Any:
    """Read a saved per-instrument table without turning tickers such as NA or TRUE into other values.

    Only empty cells are missing (how the runner writes NaN and cash-only rows).
    """
    import pandas as pd

    return pd.read_csv(path, usecols=columns, parse_dates=["datetime"], keep_default_na=False, na_values=[""], dtype={"instrument": str})


def instrument_batches(path: Path, columns: list[str] | None = None, *, batch_size: int = 65536) -> Any:
    """Projected, bounded saved-table reads with the legacy missing-cell semantics."""
    import pandas as pd

    if Path(path).suffix == ".parquet":
        import pyarrow.parquet as pq

        with pq.ParquetFile(path) as source:
            if columns is not None and not set(columns).issubset(source.schema_arrow.names):
                raise ValueError("Saved position table lacks required columns")
            for batch in source.iter_batches(batch_size=batch_size, columns=columns):
                frame = batch.to_pandas()
                if "datetime" in frame:
                    frame["datetime"] = pd.to_datetime(frame["datetime"])
                for column in frame.columns[frame.dtypes == object]:
                    frame[column] = frame[column].mask(frame[column] == "")
                yield frame
    else:
        with pd.read_csv(path, usecols=columns, parse_dates=["datetime"], keep_default_na=False,
                         na_values=[""], dtype={"instrument": str}, chunksize=batch_size) as source:
            yield from source


def latest_positions(run_dir: Path, date: Any, *, batch_size: int = 65536) -> Any:
    """Read only holdings display columns, retaining only the requested session."""
    import pandas as pd

    columns = ["datetime", "instrument", "weight"]
    selected = []
    for batch in instrument_batches(table_artifact(run_dir, "positions"), columns, batch_size=batch_size):
        day = batch.loc[batch["datetime"] == pd.Timestamp(date)]
        if not day.empty:
            selected.append(day)
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(columns=columns)


def _array_close(actual: Any, expected: Any) -> Any:
    """The same symmetric tolerance as _close, applied to bounded arrays."""
    import numpy as np

    return (np.isfinite(actual) & np.isfinite(expected)
            & (np.abs(actual - expected) <= np.maximum(1e-6, 1e-7 * np.maximum(np.abs(actual), np.abs(expected)))))


def _ordered_signed_position_totals(path: Path, frame: Any, needed: set[str]) -> dict[Any, list[float]]:
    """Vectorized bounded replay for an explicitly promised strict table order."""
    import numpy as np
    import pandas as pd

    totals: dict[Any, list[float]] = {}
    previous_key = None
    scalar_columns = ("free_cash", "cash_total", "restricted_cash", "long_value", "short_liability",
                      "transaction_cost_dollars", "borrow_cost_dollars", "total_cost_dollars",
                      "unfunded_cash_shortfall", "cash", "account")
    for batch in instrument_batches(path, sorted(needed)):
        instruments = batch["instrument"].fillna("").astype(str)
        keys = pd.MultiIndex.from_arrays([batch["datetime"], instruments])
        if len(keys) and (not keys.is_monotonic_increasing or keys.has_duplicates
                         or previous_key is not None and keys[0] <= previous_key):
            raise ValueError("Declared datetime,instrument position order is invalid or duplicated")
        if len(keys):
            previous_key = keys[-1]
        for date, day in batch.groupby("datetime", sort=False, dropna=False):
            if date not in frame.index:
                raise ValueError("schema 4 positions do not cover daily ledger")
            row = frame.loc[date]
            account = float(row.account)
            held = day["instrument"].notna().to_numpy() & (day["instrument"].fillna("") != "").to_numpy()
            amount, price, value, weight = (day[column].to_numpy(dtype=float) for column in
                                          ("amount", "price", "value", "weight"))
            if (not np.isfinite(amount[held]).all() or not np.isfinite(price[held]).all()
                    or not np.isfinite(value[held]).all() or (price[held] <= 0).any()
                    or not _array_close(value[held], amount[held] * price[held]).all()):
                raise ValueError("schema 4 signed position value does not reconcile")
            if (not _array_close(value[~held], 0).all()
                    or not _array_close(amount[~held], 0).all()):
                raise ValueError("schema 4 cash-only position has value")
            if account > 0:
                expected_weight = np.where(held, value / account, 0)
                if not np.isfinite(weight).all() or not _array_close(weight, expected_weight).all():
                    raise ValueError("schema 4 position weight does not reconcile")
            elif not np.isnan(weight).all():
                raise ValueError("schema 4 nonpositive equity requires undefined position weight")
            values = totals.setdefault(date, [0.0, 0.0, 0.0])
            values[0] += float(value[held].sum())
            values[1] += float(np.maximum(value[held], 0).sum())
            values[2] += float(np.maximum(-value[held], 0).sum())
            for column in scalar_columns:
                expected = float(row.free_cash if column == "cash" else row[column])
                if not _array_close(day[column].to_numpy(dtype=float), expected).all():
                    raise ValueError(f"schema 4 position {column} does not reconcile")
            for column in ("funding_feasible", "metrics_defined", "stale_marks"):
                if not (day[column].astype(str).str.lower() == str(bool(row[column])).lower()).all():
                    raise ValueError(f"schema 4 position {column} does not reconcile")
            if not (day["stale_mark_symbols"] == row.stale_mark_symbols).all():
                raise ValueError("schema 4 position stale mark symbols do not reconcile")
    if set(totals) != set(frame.index):
        raise ValueError("schema 4 positions do not cover daily ledger")
    return totals


def _signed_position_totals(path: Path, frame: Any, needed: set[str]) -> dict[Any, list[float]]:
    """Validate every position while retaining only session sums in memory.

    Legacy tables may be arbitrarily ordered. A disk uniqueness index keeps
    duplicate checks bounded without retaining every key in RAM.
    """
    import pandas as pd
    import sqlite3
    from tempfile import TemporaryDirectory

    totals: dict[Any, list[float]] = {}
    scalar_columns = ("free_cash", "cash_total", "restricted_cash", "long_value", "short_liability",
                      "transaction_cost_dollars", "borrow_cost_dollars", "total_cost_dollars",
                      "unfunded_cash_shortfall", "cash", "account")
    with TemporaryDirectory(prefix="qlib-report-positions-") as temporary:
        database = sqlite3.connect(str(Path(temporary) / "keys.sqlite"))
        try:
            database.execute("PRAGMA cache_size=-2048")
            database.execute("CREATE TABLE names (day TEXT, instrument TEXT, PRIMARY KEY(day, instrument)) WITHOUT ROWID")
            for batch in instrument_batches(path, sorted(needed)):
                for date, day in batch.groupby("datetime", sort=False, dropna=False):
                    if date not in frame.index:
                        raise ValueError("schema 4 positions do not cover daily ledger")
                    row = frame.loc[date]
                    account, free_cash = float(row.account), float(row.free_cash)
                    names = day["instrument"].dropna().tolist()
                    try:
                        database.executemany("INSERT INTO names VALUES (?, ?)", [(str(date), str(name)) for name in names])
                    except sqlite3.IntegrityError as exc:
                        raise ValueError("schema 4 duplicate signed position") from exc
                    values = totals.setdefault(date, [0.0, 0.0, 0.0])
                    for item in day.itertuples(index=False):
                        if pd.isna(item.instrument) or item.instrument == "":
                            if not _close(float(item.value), 0) or not _close(float(item.amount), 0):
                                raise ValueError("schema 4 cash-only position has value")
                            weight = float(item.weight)
                            if account > 0:
                                if not math.isfinite(weight) or not _close(weight, 0):
                                    raise ValueError("schema 4 cash-only position weight does not reconcile")
                            elif not math.isnan(weight):
                                raise ValueError("schema 4 cash-only position weight must be undefined at nonpositive equity")
                            continue
                        amount, price, value = float(item.amount), float(item.price), float(item.value)
                        if not all(map(math.isfinite, (amount, price, value))) or price <= 0 or not _close(value, amount * price):
                            raise ValueError("schema 4 signed position value does not reconcile")
                        values[0] += value
                        values[1] += max(value, 0)
                        values[2] += max(-value, 0)
                        weight = float(item.weight)
                        if account > 0:
                            if not math.isfinite(weight) or not _close(weight, value / account):
                                raise ValueError("schema 4 position weight does not reconcile")
                        elif not math.isnan(weight):
                            raise ValueError("schema 4 nonpositive equity requires undefined position weight")
                    for column in scalar_columns:
                        expected = free_cash if column == "cash" else float(row[column])
                        if not all(_close(float(value), expected) for value in day[column]):
                            raise ValueError(f"schema 4 position {column} does not reconcile")
                    for column in ("funding_feasible", "metrics_defined", "stale_marks"):
                        flag = str(bool(row[column])).lower()
                        if not all(str(value).lower() == flag for value in day[column]):
                            raise ValueError(f"schema 4 position {column} does not reconcile")
                    if not all(value == row.stale_mark_symbols for value in day["stale_mark_symbols"]):
                        raise ValueError("schema 4 position stale mark symbols do not reconcile")
            if set(totals) != set(frame.index):
                raise ValueError("schema 4 positions do not cover daily ledger")
        finally:
            database.close()
    return totals


def _bundle(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    schema = manifest.get("schema_version")
    if schema not in REPLAY_SCHEMAS:
        raise ValueError("schema_version 3 to 6 required; legacy outputs are not migrated by this command")
    if manifest.get("status") != "completed":
        raise ValueError("Report replay requires a completed run")
    hashes = manifest.get("artifacts_sha256")
    if not isinstance(hashes, dict) or not (CORE_FILES_PARQUET if schema == 6 else CORE_FILES).issubset(hashes):
        raise ValueError("Run manifest lacks required core artifact hashes")
    for name in hashes:
        if isinstance(name, str) and name.endswith(".parquet") and f"{name[:-8]}.csv.gz" in hashes:
            raise ValueError(f"Run manifest seals ambiguous duplicate table formats: {name[:-8]}")
    for name, expected in hashes.items():
        if not isinstance(name, str) or Path(name).name != name or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("Run manifest contains an invalid artifact entry")
        path = run_dir / name
        if not path.is_file() or _redirect(path):
            raise ValueError(f"Run artifact missing or unsafe: {name}")
        if _sha256(path) != expected.lower():
            raise ValueError(f"Run artifact hash mismatch: {name}")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    contract = summary.get("metric_contract", {})
    if (contract.get("version") not in ("daily-simple-net-v1", "daily-simple-signed-net-v1", "daily-simple-strategy-net-v1")
            or contract.get("periods_per_year") != 252):
        raise ValueError("Unsupported daily net metric contract at 252 sessions")
    if schema == 3 and contract["version"] != "daily-simple-net-v1":
        raise ValueError("Schema 3 requires daily-simple-net-v1")
    if schema == 4 and contract["version"] != "daily-simple-signed-net-v1":
        raise ValueError("Schema 4 requires daily-simple-signed-net-v1")
    if not isinstance(summary.get("metrics"), dict) or not isinstance(summary.get("backtest"), dict):
        raise ValueError("summary.json lacks metrics or backtest metadata")
    if schema in (5, 6):
        if contract["version"] != "daily-simple-strategy-net-v1":
            raise ValueError("Schema 5 and 6 require daily-simple-strategy-net-v1")
        mode = manifest.get("strategy_mode")
        if mode not in ("signed", "long-only") or summary["backtest"].get("strategy_mode") != mode:
            raise ValueError("Schema 5 strategy_mode must agree in manifest and summary")
        if set(summary["metrics"]) != {"strategy_net"}:
            raise ValueError("Schema 5 metrics must contain only strategy_net")
    return manifest, summary


def _is_signed(manifest: dict[str, Any]) -> bool:
    return manifest["schema_version"] == 4 or (
        manifest["schema_version"] in (5, 6) and manifest["strategy_mode"] == "signed"
    )


def _chart(fig: Any, directory: Path, name: str) -> str:
    fig.update_layout(template="plotly_white", height=560, margin=dict(t=65, b=55))
    fig.write_html(directory / name, include_plotlyjs="directory", full_html=True, auto_open=False)
    return name


def _daily(run_dir: Path, *, signed: bool = False, benchmark_enabled: bool = True) -> Any:
    import pandas as pd

    frame = pd.read_csv(run_dir / "daily_report.csv", parse_dates=["datetime"], index_col="datetime")
    required = {"account", "net_return", "cost", "cost_dollars", "turnover", "cash", "holdings_value"}
    if benchmark_enabled:
        required.add("benchmark_return")
    elif "benchmark_return" in frame:
        raise ValueError("Schema 5 daily_report.csv must not contain benchmark_return")
    if frame.empty or not required.issubset(frame.columns) or frame.index.has_duplicates:
        raise ValueError("daily_report.csv has missing columns, dates, or rows")
    frame = frame.sort_index()
    ratios = {"net_return", "cost", "turnover"}
    numbers = frame[list(required - ratios if signed else required)].apply(pd.to_numeric, errors="coerce")
    if not all(math.isfinite(value) for row in numbers.itertuples(index=False, name=None) for value in row):
        raise ValueError("daily_report.csv contains nonfinite or missing values")
    frame[numbers.columns] = numbers
    if signed:
        extra = {"free_cash", "cash_total", "restricted_cash", "long_value", "short_liability",
                 "transaction_cost_dollars", "borrow_cost_dollars", "total_cost_dollars",
                 "unfunded_cash_shortfall", "funding_feasible", "metrics_defined", "stale_marks"}
        if not extra.issubset(frame.columns):
            raise ValueError("schema 4 daily_report.csv lacks signed ledger columns")
        numeric_extra = extra - {"funding_feasible", "metrics_defined", "stale_marks"}
        frame[list(numeric_extra)] = frame[list(numeric_extra)].apply(pd.to_numeric, errors="coerce")
        if not all(math.isfinite(value) for row in frame[list(numeric_extra)].itertuples(index=False, name=None) for value in row):
            raise ValueError("schema 4 daily_report.csv contains nonfinite account or cost values")
        frame[list(ratios)] = frame[list(ratios)].apply(pd.to_numeric, errors="coerce")
        for name in ("funding_feasible", "metrics_defined", "stale_marks"):
            values = frame[name].astype(str).str.lower()
            if not values.isin(("true", "false", "1", "0")).all():
                raise ValueError(f"schema 4 daily_report.csv invalid {name}")
            frame[name] = values.isin(("true", "1"))
        for name in ratios:
            if not frame.loc[frame["metrics_defined"], name].map(math.isfinite).all():
                raise ValueError(f"schema 4 defined {name} must be finite")
            if not frame.loc[~frame["metrics_defined"], name].isna().all():
                raise ValueError(f"schema 4 unavailable {name} must be NaN")
    else:
        frame["net_return"] = pd.to_numeric(frame["net_return"], errors="coerce")
        if not frame["net_return"].map(math.isfinite).all():
            raise ValueError("daily_report.csv contains nonfinite or missing values")
    return frame


def _reconcile_ledger(frame: Any, summary: dict[str, Any], *, benchmark_enabled: bool = True) -> float:
    from qlib_yahoo.metrics import portfolio_metrics

    cash = float(summary["backtest"]["cash"])
    if not math.isfinite(cash) or cash <= 0:
        raise ValueError("summary backtest cash must be positive and finite")
    previous = cash
    for account, net_return in zip(frame["account"], frame["net_return"]):
        if account <= 0 or not math.isclose(account / previous - 1, net_return, rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError("daily_report.csv net_return does not reconcile with account and initial cash")
        previous = account
    annual_cash_rate = float(summary["metric_contract"]["annual_cash_rate"])
    expected = portfolio_metrics(
        frame["net_return"].tolist(), frame["benchmark_return"].tolist() if benchmark_enabled else None,
        annual_cash_rate=annual_cash_rate, periods_per_year=252,
    )["strategy_net"]
    actual = summary["metrics"].get("strategy_net", {})
    if not isinstance(actual, dict) or actual.get("observations") != expected["observations"]:
        raise ValueError("summary strategy_net observations disagree with daily_report.csv")
    if not benchmark_enabled:
        _check_metrics(actual, expected, "strategy_net")
        return cash
    stored_sharpe, computed_sharpe = actual.get("sharpe"), expected["sharpe"]
    if stored_sharpe is None or computed_sharpe is None:
        matched = stored_sharpe is None and computed_sharpe is None
    else:
        matched = (
            isinstance(stored_sharpe, (int, float)) and not isinstance(stored_sharpe, bool)
            and math.isfinite(stored_sharpe)
            and math.isclose(stored_sharpe, computed_sharpe, rel_tol=1e-7, abs_tol=1e-9)
        )
    if not matched:
        raise ValueError("summary strategy_net sharpe disagrees with daily_report.csv")
    return cash


def _check_metrics(actual: Any, expected: dict[str, Any], family: str) -> None:
    if not isinstance(actual, dict) or actual.keys() != expected.keys():
        raise ValueError(f"summary {family} metric shape disagrees with daily_report.csv")
    for name, value in expected.items():
        stored = actual[name]
        matched = stored is None if value is None else (
            isinstance(stored, (int, float)) and not isinstance(stored, bool)
            and math.isfinite(stored) and math.isclose(stored, value, rel_tol=1e-7, abs_tol=1e-6)
        )
        if not matched:
            raise ValueError(f"summary {family}.{name} disagrees with daily_report.csv")


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-6)


def _reconcile_signed(run_dir: Path, frame: Any, summary: dict[str, Any], *, benchmark_enabled: bool = True, positions_order: str | None = None) -> float:
    """Check saved account identity against independent position and cost rows."""
    import pandas as pd

    initial = float(summary["backtest"]["cash"])
    if not math.isfinite(initial) or initial <= 0:
        raise ValueError("summary backtest cash must be positive and finite")
    needed = {"datetime", "instrument", "amount", "price", "value", "weight", "cash", "account",
              "cash_total", "free_cash", "restricted_cash", "long_value", "short_liability",
              "transaction_cost_dollars", "borrow_cost_dollars", "total_cost_dollars",
              "unfunded_cash_shortfall", "funding_feasible", "metrics_defined", "stale_marks",
              "stale_mark_symbols"}
    if (frame["turnover"].dropna() < -1e-9).any():
        raise ValueError("schema 4 turnover must be nonnegative")
    if "stale_mark_symbols" not in frame:
        raise ValueError("schema 4 daily ledger lacks stale mark symbols")
    previous = initial
    if positions_order is not None and positions_order != "datetime,instrument":
        raise ValueError("Unsupported declared position order")
    reducer = _ordered_signed_position_totals if positions_order == "datetime,instrument" else _signed_position_totals
    totals = reducer(table_artifact(run_dir, "positions"), frame, needed)
    for date, row in frame.iterrows():
        try:
            stale_symbols = json.loads(row.stale_mark_symbols)
        except (TypeError, ValueError) as exc:
            raise ValueError("schema 4 stale mark symbols must be a JSON list") from exc
        if (not isinstance(stale_symbols, list) or not all(isinstance(symbol, str) and symbol for symbol in stale_symbols)
                or bool(stale_symbols) != bool(row.stale_marks)):
            raise ValueError("schema 4 stale mark symbols do not reconcile")
        if not _close(row.cash, row.free_cash) or not _close(row.cash_total, row.free_cash + row.restricted_cash):
            raise ValueError("schema 4 free/restricted cash does not reconcile")
        if (row.long_value < -1e-6 or row.short_liability < -1e-6 or row.restricted_cash < -1e-6
                or row.transaction_cost_dollars < -1e-6 or row.borrow_cost_dollars < -1e-6):
            raise ValueError("schema 4 negative liability, restricted cash or expense")
        if not _close(row.account, row.cash_total + row.long_value - row.short_liability):
            raise ValueError("schema 4 equity does not reconcile")
        if not _close(row.holdings_value, row.long_value - row.short_liability):
            raise ValueError("schema 4 signed holdings do not reconcile")
        if not _close(row.total_cost_dollars, row.transaction_cost_dollars + row.borrow_cost_dollars) or not _close(row.cost_dollars, row.total_cost_dollars):
            raise ValueError("schema 4 fee components do not reconcile")
        if not _close(row.unfunded_cash_shortfall, max(-row.free_cash, 0)) or bool(row.funding_feasible) != (row.free_cash >= 0 and row.account > 0):
            raise ValueError("schema 4 funding state does not reconcile")
        defined = previous > 0 and row.account > 0
        if bool(row.metrics_defined) != defined:
            raise ValueError("schema 4 metrics_defined does not reconcile with equity")
        if defined:
            if not math.isfinite(row.net_return) or not _close(row.net_return, row.account / previous - 1):
                raise ValueError("schema 4 net_return does not reconcile with account")
            if not _close(row.cost, row.total_cost_dollars / previous):
                raise ValueError("schema 4 cost ratio does not reconcile with booked dollars")
        elif not math.isnan(row.net_return):
            raise ValueError("schema 4 nonpositive equity requires undefined net_return")
        signed_value, longs, shorts = totals[date]
        if not all(_close(a, b) for a, b in ((signed_value, row.holdings_value), (longs, row.long_value), (shorts, row.short_liability))):
            raise ValueError("schema 4 signed positions do not reconcile with daily ledger")
        previous = row.account
    actual = summary["metrics"].get("strategy_net", {})
    if not isinstance(actual, dict) or actual.get("observations") != len(frame):
        raise ValueError("schema 4 summary observations must retain every session")
    unavailable = not frame["metrics_defined"].all()
    status = summary.get("metrics_status")
    funding = summary.get("funding_state")
    if not isinstance(status, dict) or status.get("strategy") != ("unavailable" if unavailable else "available"):
        raise ValueError("schema 4 metrics_status does not reconcile")
    if status.get("reason") != ("nonpositive_equity" if unavailable else None) or status.get("undefined_observations") != int((~frame["metrics_defined"]).sum()):
        raise ValueError("schema 4 unavailable reason/count does not reconcile")
    if (not isinstance(funding, dict) or funding.get("feasible_all_sessions") is not bool(frame["funding_feasible"].all())
            or funding.get("unfunded_sessions") != int((~frame["funding_feasible"]).sum())
            or not _close(float(funding.get("max_unfunded_cash_shortfall", float("nan"))), float(frame["unfunded_cash_shortfall"].max()))
            or funding.get("stale_mark_sessions") != int(frame["stale_marks"].sum())):
        raise ValueError("schema 4 funding_state does not reconcile")
    if unavailable:
        from qlib_yahoo.metrics import portfolio_metrics

        strategy_keys = portfolio_metrics([], None, annual_cash_rate=float(
            summary["metric_contract"]["annual_cash_rate"]
        ))["strategy_net"].keys() if not benchmark_enabled else None
        for family in (("strategy_net", "excess_net_vs_benchmark") if benchmark_enabled else ("strategy_net",)):
            values = summary["metrics"].get(family, {})
            if not isinstance(values, dict) or (strategy_keys is not None and values.keys() != strategy_keys) or values.get("observations") != len(frame) or any(
                value is not None for key, value in values.items() if key != "observations"
            ):
                raise ValueError("schema 4 unavailable aggregate metrics must be null for the full period")
        if benchmark_enabled:
            benchmark = portfolio_metrics([0.0] * len(frame), frame["benchmark_return"].tolist(),
                                          annual_cash_rate=float(summary["metric_contract"]["annual_cash_rate"]))["benchmark_reference"]
            stored_benchmark = summary["metrics"].get("benchmark_reference", {})
            if not isinstance(stored_benchmark, dict) or stored_benchmark.keys() != benchmark.keys():
                raise ValueError("schema 4 benchmark metric shape disagrees")
            for name, value in benchmark.items():
                actual_value = stored_benchmark[name]
                matched = actual_value is None if value is None else (
                    isinstance(actual_value, (int, float)) and not isinstance(actual_value, bool) and _close(actual_value, value)
                )
                if not matched:
                    raise ValueError(f"schema 4 benchmark_reference.{name} disagrees with ledger")
    else:
        from qlib_yahoo.metrics import portfolio_metrics

        expected = portfolio_metrics(frame["net_return"].tolist(), frame["benchmark_return"].tolist() if benchmark_enabled else None,
                                     annual_cash_rate=float(summary["metric_contract"]["annual_cash_rate"]))
        for family in (("strategy_net", "benchmark_reference", "excess_net_vs_benchmark") if benchmark_enabled else ("strategy_net",)):
            stored = summary["metrics"].get(family, {})
            if not isinstance(stored, dict) or stored.keys() != expected[family].keys():
                raise ValueError(f"schema 4 summary {family} metric shape disagrees")
            for name, value in expected[family].items():
                actual_value = stored[name]
                if value is None:
                    matched = actual_value is None
                else:
                    matched = isinstance(actual_value, (int, float)) and not isinstance(actual_value, bool) and _close(actual_value, value)
                if not matched:
                    raise ValueError(f"schema 4 summary {family}.{name} disagrees with ledger")
    return initial


def _overview(frame: Any, cash: float, directory: Path, *, benchmark_enabled: bool = True) -> list[tuple[str, str]]:
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    dates = [frame.index[0] - pd.Timedelta(days=1), *frame.index]
    account = pd.concat([pd.Series([cash]), frame["account"].reset_index(drop=True)], ignore_index=True)
    fig = go.Figure()
    fig.add_scatter(x=dates, y=account, name="Net account")
    if benchmark_enabled:
        benchmark = cash * (1 + frame["benchmark_return"]).cumprod()
        benchmark = pd.concat([pd.Series([cash]), benchmark.reset_index(drop=True)], ignore_index=True)
        fig.add_scatter(x=dates, y=benchmark, name="Benchmark reference")
    fig.update_layout(title="Net account and benchmark reference" if benchmark_enabled else "Net strategy equity", yaxis_title="Account value")
    charts = [("Account and benchmark" if benchmark_enabled else "Net equity", _chart(fig, directory, "account.html"))]

    fig = go.Figure()
    fig.add_scatter(x=dates, y=account / account.cummax() - 1, name="Net account drawdown")
    fig.update_layout(title="Compounded net drawdown (initial cash baseline)", yaxis_tickformat=".1%")
    charts.append(("Net drawdown", _chart(fig, directory, "drawdown.html")))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=frame.index, y=frame["net_return"], name="Net daily return", secondary_y=False)
    fig.add_scatter(x=frame.index, y=frame["cost_dollars"], name="Trading cost ($)", secondary_y=True)
    fig.update_layout(title="Net returns and trading cost")
    fig.update_yaxes(tickformat=".1%", secondary_y=False)
    charts.append(("Net returns and cost", _chart(fig, directory, "returns_cost.html")))

    fig = go.Figure()
    fig.add_bar(x=frame.index, y=frame["turnover"], name="Turnover")
    fig.update_layout(title="Daily turnover", yaxis_title="Turnover fraction")
    charts.append(("Turnover", _chart(fig, directory, "turnover.html")))
    return charts


def _signed_overview(frame: Any, cash: float, directory: Path, *, benchmark_enabled: bool = True) -> list[tuple[str, str]]:
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    dates = [frame.index[0] - pd.Timedelta(days=1), *frame.index]
    fig = go.Figure()
    fig.add_scatter(x=dates, y=[cash, *frame["account"]], name="Net equity")
    if benchmark_enabled:
        benchmark = cash * (1 + frame["benchmark_return"]).cumprod()
        fig.add_scatter(x=dates, y=[cash, *benchmark], name="Benchmark reference")
    fig.update_layout(title="Signed account equity and benchmark" if benchmark_enabled else "Net signed strategy equity", yaxis_title="Account value")
    charts = [("Account and benchmark" if benchmark_enabled else "Net equity", _chart(fig, directory, "account.html"))]

    peak = pd.Series([cash, *frame["account"]]).cummax()
    drawdown = pd.Series([cash, *frame["account"]]) / peak - 1
    fig = go.Figure()
    fig.add_scatter(x=dates, y=drawdown, name="Net equity drawdown")
    fig.update_layout(title="Net equity drawdown from initial cash", yaxis_tickformat=".1%")
    charts.append(("Net drawdown", _chart(fig, directory, "drawdown.html")))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=frame.index, y=frame["net_return"], name="Net daily return (undefined at nonpositive equity)", secondary_y=False)
    fig.add_scatter(x=frame.index, y=frame["transaction_cost_dollars"], name="Transaction cost ($)", secondary_y=True)
    fig.add_scatter(x=frame.index, y=frame["borrow_cost_dollars"], name="Stock borrow cost ($)", secondary_y=True)
    fig.update_layout(title="Net returns and separate booked costs")
    fig.update_yaxes(tickformat=".1%", secondary_y=False)
    charts.append(("Net returns and costs", _chart(fig, directory, "returns_cost.html")))

    fig = go.Figure()
    fig.add_bar(x=frame.index, y=frame["turnover"], name="Turnover")
    fig.update_layout(title="Daily turnover", yaxis_title="Turnover fraction")
    charts.append(("Turnover", _chart(fig, directory, "turnover.html")))

    fig = go.Figure()
    for column, label, multiplier in (("free_cash", "Free cash", 1), ("restricted_cash", "Restricted short proceeds", 1),
                                      ("long_value", "Long holdings", 1), ("short_liability", "Short liability", -1)):
        fig.add_scatter(x=frame.index, y=frame[column] * multiplier, name=label)
    fig.update_layout(title="Account components in dollars (short liability below zero)", yaxis_title="Dollars")
    charts.append(("Signed account components", _chart(fig, directory, "account_components.html")))

    fig = go.Figure()
    fig.add_scatter(x=frame.index, y=frame["long_value"] + frame["short_liability"], name="Gross position exposure ($)")
    fig.add_scatter(x=frame.index, y=frame["holdings_value"], name="Net position exposure ($)")
    fig.update_layout(title="Actual gross and net position exposure in dollars", yaxis_title="Dollars")
    charts.append(("Gross and net exposure", _chart(fig, directory, "exposure.html")))
    return charts


def _holdings(run_dir: Path, frame: Any, directory: Path, *, signed: bool = False) -> tuple[str, str]:
    import pandas as pd
    import plotly.graph_objects as go

    columns = ["datetime", "instrument", "weight"] + (["value"] if signed else [])
    positions = read_instrument_table(table_artifact(run_dir, "positions"), columns=columns)
    if not {"datetime", "instrument", "weight"}.issubset(positions.columns):
        raise ValueError("positions.csv.gz lacks holdings columns")
    fig = go.Figure()
    positive_equity = frame["account"] > 0 if signed else None

    def signed_weights(values: Any) -> Any:
        weights = pd.Series(float("nan"), index=frame.index, dtype=float)
        weights.loc[positive_equity] = values.loc[positive_equity] / frame.loc[positive_equity, "account"]
        return weights

    series = None
    if signed:
        from qlib_yahoo.holding_series import signed_holding_series

        series = signed_holding_series(positions, frame)
    if series is None:
        series = {}
        for instrument, group in positions.dropna(subset=["instrument"]).groupby("instrument"):
            if signed:
                values = group.set_index("datetime")["value"].reindex(frame.index, fill_value=0.0)
                series[str(instrument)] = signed_weights(values)
            else:
                series[str(instrument)] = group.set_index("datetime")["weight"].reindex(frame.index, fill_value=0.0)
    # Thousands of traces make an unusable page: keep the largest names and aggregate the rest.
    # Every position remains in positions.csv.gz.
    names = sorted(series)
    rest: list[str] = []
    if len(names) > HOLDINGS_TRACES:
        ranked = sorted(names, key=lambda name: (-float(series[name].abs().mean(skipna=True) or 0.0), name))
        names, rest = sorted(ranked[:HOLDINGS_TRACES]), ranked[HOLDINGS_TRACES:]
    for name in names:
        fig.add_scatter(x=frame.index, y=series[name], name=name, mode="lines")
    if rest:
        other = pd.concat([series[name] for name in rest], axis=1)
        undefined = other.isna().all(axis=1)
        longs = other.clip(lower=0).sum(axis=1).mask(undefined)
        fig.add_scatter(x=frame.index, y=longs, name=f"Other {'longs' if signed else 'holdings'} ({len(rest)} names)", mode="lines")
        if signed:
            shorts = other.clip(upper=0).sum(axis=1).mask(undefined)
            fig.add_scatter(x=frame.index, y=shorts, name=f"Other shorts ({len(rest)} names)", mode="lines")
    if signed:
        fig.add_scatter(x=frame.index, y=signed_weights(frame["free_cash"]), name="Free cash weight", mode="lines")
        fig.add_scatter(x=frame.index, y=signed_weights(frame["restricted_cash"]), name="Restricted cash weight", mode="lines")
        fig.update_layout(title="Actual signed holding and cash weights (undefined at nonpositive equity)", yaxis_title="Weight")
        return "Signed holding and cash weights", _chart(fig, directory, "holdings.html")
    fig.add_scatter(x=frame.index, y=frame["cash"] / frame["account"], name="Cash weight", mode="lines")
    fig.update_layout(title="Saved holding and cash weights over time", yaxis_title="Weight")
    return "Holding and cash weights", _chart(fig, directory, "holdings.html")


def _detailed(frame: Any, run_dir: Path, directory: Path, annual_cash_rate: float, labels_available: bool, *, signed: bool = False) -> tuple[list[tuple[str, str]], str]:
    import pandas as pd
    import plotly.graph_objects as go

    charts = []
    if not signed or frame["metrics_defined"].all():
        mean = frame["net_return"].rolling(252, min_periods=252).mean()
        std = frame["net_return"].rolling(252, min_periods=252).std(ddof=1)
        cash_daily = math.expm1(math.log1p(annual_cash_rate) / 252)
        sharpe = (mean - cash_daily) / std.replace(0, float("nan")) * math.sqrt(252)
        if sharpe.notna().any():
            fig = go.Figure()
            fig.add_scatter(x=frame.index, y=sharpe, name="Net rolling Sharpe")
            fig.update_layout(title=f"Net rolling Sharpe (252 sessions; annual cash rate {annual_cash_rate:g})")
            charts.append(("Net rolling risk", _chart(fig, directory, "rolling_risk.html")))
    if not labels_available:
        return charts, "Signal diagnostics unavailable: saved score and forward-label pairs are absent."
    path = table_artifact(run_dir, "signal_forward_labels")
    pairs = read_instrument_table(path)
    if not {"datetime", "instrument", "score", "label"}.issubset(pairs.columns):
        return charts, "Signal diagnostics unavailable: saved pairs lack required columns."
    pairs = pairs.dropna(subset=["score", "label"])
    rows = []
    for date, group in pairs.groupby("datetime"):
        if len(group) < 5:
            continue
        ic = group["score"].corr(group["label"])
        rank_ic = group["score"].corr(group["label"], method="spearman")
        if pd.notna(ic) and pd.notna(rank_ic):
            rows.append((date, ic, rank_ic))
    if not rows:
        return charts, "Signal diagnostics unavailable: fewer than five valid pairs or undefined IC per session."
    fig = go.Figure()
    fig.add_scatter(x=[r[0] for r in rows], y=[r[1] for r in rows], name="IC")
    fig.add_scatter(x=[r[0] for r in rows], y=[r[2] for r in rows], name="Rank IC")
    fig.update_layout(title="Saved signal and forward-label IC (hypothetical, not execution)")
    charts.append(("Signal IC and Rank IC", _chart(fig, directory, "signal_ic.html")))
    valid_dates = {row[0] for row in rows}
    groups = _group_returns(pairs[pairs["datetime"].isin(valid_dates)].set_index(["datetime", "instrument"]))
    if not groups.empty:
        fig = go.Figure()
        for column in [f"group_{n}" for n in range(1, 6)]:
            fig.add_scatter(x=groups.index, y=groups[column], name=column)
        fig.update_layout(title="Hypothetical forward-label means by score group (not executed returns)")
        charts.append(("Score groups", _chart(fig, directory, "score_groups.html")))
    return charts, "Signal diagnostics use the saved requested evaluation horizon; they are not position attribution."


def _index(directory: Path, summary: dict[str, Any], charts: list[tuple[str, str]], notice: str, gate: str, *, signed: bool = False, frame: Any = None, benchmark_enabled: bool = True, positions_name: str = "positions.csv.gz") -> Path:
    backtest = summary["backtest"]
    strategy = summary["metrics"]["strategy_net"]
    def display(value: Any, *, percentage: bool = False) -> str:
        if value is None:
            return "Unavailable"
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return f"{value:.2%}" if percentage else f"{value:.2f}"
        return html.escape(str(value))

    cards = "".join(
        f'<div class="card"><span>{label}</span><strong>{display(strategy.get(key), percentage=percentage)}</strong></div>'
        for label, key, percentage in (("Net Sharpe", "sharpe", False), ("CAGR", "cagr", True),
                                        ("Total return", "cumulative_return", True), ("Max drawdown", "max_drawdown", True))
    )
    chart_map = {name: (label, name) for label, name in charts}
    def links(names: tuple[str, ...]) -> str:
        return "".join(f'<a class="chart-link" href="{html.escape(name, quote=True)}">{html.escape(chart_map[name][0])} <span>↗</span></a>'
                       for name in names if name in chart_map)

    metrics_rows = "".join(
        f"<tr><td>{html.escape(str(family))}</td><td>{html.escape(str(metric))}</td><td>{html.escape(str(value))}</td></tr>"
        for family, values in summary["metrics"].items() if isinstance(values, dict)
        for metric, value in values.items()
    )
    settings = dict(backtest)
    if not benchmark_enabled and isinstance(settings.get("execution_contract"), dict):
        settings["execution_contract"] = {key: value for key, value in settings["execution_contract"].items()
                                          if key != "benchmark"}
    settings_rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in settings.items()
    )
    def panel(title: str, names: tuple[str, ...]) -> str:
        content = links(names)
        return f'<section class="panel"><h2>{title}</h2><div class="chart-grid">{content}</div></section>' if content else ""

    overview_panels = panel("Equity and drawdown", ("account.html", "drawdown.html"))
    cost_panels = panel("Costs and turnover", ("returns_cost.html", "turnover.html"))
    cost_label = "transaction and stock-borrow costs" if signed else "trading costs"
    signed_notice = ""
    if signed:
        unfunded = int((~frame["funding_feasible"]).sum())
        undefined = int((~frame["metrics_defined"]).sum())
        stale = int(frame["stale_marks"].sum())
        availability = ("Full-period strategy metrics are unavailable because equity was nonpositive; "
                        "sessions were retained rather than dropped. " if undefined else "Full-period net metrics are defined. ")
        signed_notice = (f"<p class=\"notice\">Signed-alpha research mode: {unfunded} unfunded sessions, "
                         f"{undefined} sessions with undefined net returns, {stale} sessions with stale marks. "
                         + availability +
                         "Short proceeds are restricted; cash borrowing is disabled. No neutrality, broker margin "
                         "enforcement, forced liquidation, borrow availability history, or trading readiness is claimed. "
                         "Negative free cash is an unfunded research account.</p>")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Backtest report</title>
<style>
:root{{color-scheme:light}}*{{box-sizing:border-box}}body{{font:16px/1.55 system-ui,sans-serif;margin:0;background:#f5f7fa;color:#1c2837}}main{{max-width:1120px;margin:auto;padding:2rem 1.25rem 4rem}}h1{{font-size:2rem;margin:.25rem 0}}h2{{font-size:1.25rem;margin:0 0 .8rem}}p{{margin:.5rem 0 1rem}}.eyebrow{{text-transform:uppercase;letter-spacing:.12em;color:#53677e;font-size:.75rem;font-weight:700}}.muted{{color:#516276}}.cards{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.8rem;margin:1.5rem 0}}.card,.panel{{background:white;border:1px solid #dce3eb;border-radius:12px;padding:1.2rem}}.card span{{display:block;color:#53677e;font-size:.85rem}}.card strong{{display:block;font-size:1.7rem;margin-top:.3rem;font-variant-numeric:tabular-nums}}.panel{{margin-top:1rem}}.chart-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.65rem}}.chart-link{{display:flex;justify-content:space-between;padding:.85rem 1rem;border:1px solid #dce3eb;border-radius:8px;color:#145a85;text-decoration:none}}.chart-link:hover{{background:#eff7fc}}.notice{{background:#fff5de;padding:1rem;border-left:4px solid #b77e00;border-radius:4px}}details{{margin-top:1rem}}summary{{cursor:pointer;font-weight:650}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:.9rem}}th,td{{border-bottom:1px solid #e3e9ef;padding:.45rem .6rem;text-align:left;vertical-align:top}}td{{overflow-wrap:anywhere}}a{{color:#145a85}}@media(max-width:700px){{main{{padding:1.25rem .8rem}}.cards,.chart-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.card strong{{font-size:1.3rem}}}}@media(max-width:440px){{.cards,.chart-grid{{grid-template-columns:1fr}}}}
</style></head><body><main><div class="eyebrow">Completed backtest · {html.escape(str(backtest.get('strategy_mode', 'signed' if signed else 'long-only')))}</div><h1>Strategy report</h1><p class="muted">{html.escape(str(backtest.get('start', 'unspecified')))} to {html.escape(str(backtest.get('end', 'unspecified')))} · {html.escape(str(strategy.get('observations', 'unknown')))} sessions · net of {cost_label}</p>
<div class="cards">{cards}</div><p class="notice">Research output, not trading readiness. {html.escape(gate)}</p>
{overview_panels}{cost_panels}
{f'<section class="panel"><h2>Exposures and funding</h2>{signed_notice}<div class="chart-grid">{links(("account_components.html", "exposure.html"))}</div></section>' if signed else ''}
<section class="panel"><h2>Holdings and diagnostics</h2><div class="chart-grid">{links(('holdings.html', 'rolling_risk.html', 'signal_ic.html', 'score_groups.html'))}</div><p class="muted">{html.escape(notice)}</p><p class="muted">Native position attribution is unavailable: saved signal labels and executed positions use different horizons and costs. No cost-free long-short performance is reported.</p></section>
<section class="panel"><details><summary>Metrics and run settings</summary><p class="muted">Annual cash rate: {html.escape(str(summary['metric_contract']['annual_cash_rate']))}; 252 sessions/year. Brain equivalent: false.</p><div class="scroll"><table><thead><tr><th>Series</th><th>Metric</th><th>Value</th></tr></thead><tbody>{metrics_rows}</tbody></table></div><h2>Backtest settings</h2><div class="scroll"><table>{settings_rows}</table></div></details><details><summary>Saved data and provenance</summary><p>Core data: <a href="../../summary.json">summary</a>, <a href="../../daily_report.csv">daily ledger</a>, <a href="../../{html.escape(positions_name)}">positions</a>, <a href="../../manifest.json">run manifest</a>.</p><p>Metric contract: {html.escape(str(summary['metric_contract']['version']))}</p></details></section>
</main></body></html>"""
    destination = directory / "report_index.html"
    destination.write_text(page, encoding="utf-8")
    return destination


def _provenance(run_dir: Path) -> dict[str, Any]:
    details = {"renderer_source_sha256": _sha256(Path(__file__)),
               "renderer_dependencies_sha256": {
                   "holding_series.py": _sha256(Path(__file__).with_name("holding_series.py"))}}
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        details["core_manifest_sha256"] = _sha256(manifest_path)
    try:
        details["plotly_version"] = version("plotly")
    except PackageNotFoundError:
        details["plotly_version"] = None
    return details


def render_report(run_dir: Path, *, level: str = "overview", holdings: str = "auto", threshold: float = 1.3, min_observations: int = 252, _validated: ValidatedBundle | None = None) -> Path:
    """Render a fresh offline report from a verified completed run."""
    run_dir = Path(run_dir)
    if _redirect(run_dir) or _redirect(run_dir / "reports"):
        raise ValueError("Run or reports directory must not be a symlink or junction")
    if level not in ("summary", "overview", "detailed") or holdings not in ("auto", "always", "never"):
        raise ValueError("level must be summary/overview/detailed and holdings auto/always/never")
    if not math.isfinite(threshold) or min_observations < 1:
        raise ValueError("holdings gate needs a finite threshold and positive observation count")
    request = {"level": level, "holdings": holdings, "threshold": threshold, "min_observations": min_observations}
    report_dir = None
    try:
        bundle = _validated if _validated is not None else validate_bundle(run_dir)
        if bundle.run_dir.resolve() != run_dir.resolve():
            raise ValueError("Validated bundle belongs to a different run")
        bundle.assert_unchanged()
        manifest, summary = bundle.manifest, bundle.summary
        from qlib_yahoo.metrics import holdings_gate

        signed = _is_signed(manifest)
        with_benchmark = benchmark_enabled(manifest)
        frame, cash = bundle.frame, bundle.cash
        eligible = bool(holdings_gate(summary["metrics"], threshold=threshold, min_observations=min_observations))
        if signed:
            eligible = eligible and bool(frame["metrics_defined"].all()) and bool(frame["funding_feasible"].all())
        if level == "summary" and (holdings == "never" or holdings == "auto" and not eligible):
            bundle.assert_unchanged()
            return run_dir / "summary.json"
        report_dir = run_dir / "reports" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8])
        report_dir.mkdir(parents=True, exist_ok=False)
        charts = [] if level == "summary" else (_signed_overview(frame, cash, report_dir, benchmark_enabled=with_benchmark) if signed else _overview(frame, cash, report_dir, benchmark_enabled=with_benchmark))
        show_holdings = holdings == "always" or holdings == "auto" and eligible
        if show_holdings:
            charts.append(_holdings(run_dir, frame, report_dir, signed=signed))
        notice = "Signal diagnostics are available only at detailed level when saved pairs support them."
        if level == "detailed":
            extra, notice = _detailed(frame, run_dir, report_dir, float(summary["metric_contract"]["annual_cash_rate"]), bool({"signal_forward_labels.parquet", "signal_forward_labels.csv.gz"} & set(manifest["artifacts_sha256"])), signed=signed)
            charts.extend(extra)
        gate = f"Reporting holdings gate: {'eligible' if eligible else 'ineligible'} (Sharpe strictly > {threshold}, at least {min_observations} observations{' plus defined metrics and feasible funding throughout' if signed else ''}); manual selection may override. This is not strategy acceptance."
        index = _index(report_dir, summary, charts, notice, gate, signed=signed, frame=frame, benchmark_enabled=with_benchmark,
                       positions_name=table_artifact(run_dir, "positions").name)
        status = {
            "status": "completed", **request,
            "holdings_gate_eligible": eligible, "holdings_rendered": show_holdings,
            "charts": [name for _, name in charts],
            **bundle.provenance,
        }
        bundle.assert_unchanged()
        (report_dir / "report_manifest.json").write_text(json.dumps(status, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return index
    except Exception as exc:
        # A concurrent redirection must not turn failure logging into an external write.
        if _redirect(run_dir) or _redirect(run_dir / "reports") or (report_dir is not None and _redirect(report_dir)):
            raise
        if report_dir is None:
            report_dir = run_dir / "reports" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8])
            report_dir.mkdir(parents=True, exist_ok=False)
        failure = {"status": "failed", **request, **_provenance(run_dir), "error": str(exc)}
        (report_dir / "report_manifest.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        raise
