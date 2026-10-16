"""Shared, UI-independent helpers: datasets, saved runs and validated run/import settings.

Used by the desktop/web workbench; never imports Qlib.
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import date
from pathlib import Path
from typing import Any

from qlib_yahoo import DEFAULT_ALPHA

ALL_SYMBOLS = "All dataset symbols"
SIGNED_LABEL = "Signed alpha weights"
LONG_ONLY_LABEL = "Long-only Top-k (legacy)"
# Below one year of sessions, annualized Sharpe and CAGR extrapolate a short window: 20 sessions at +13%
# annualize to about +365%. The report's Sharpe display gate uses the same minimum
# (report_service.MIN_OBSERVATIONS).
SHORT_WINDOW_SESSIONS = 252
_UNIVERSE = re.compile(r"[a-z][a-z0-9_]{0,31}\Z")
STAGE_PREFIXES = ("Snapshot:", "Fingerprinting", "Scanning", "Classifying", "Loading", "Ranking", "Writing",
                  "Converting", "Reading every", "Hashing", "Ready:", "Verifying dataset", "Scoring",
                  "Simulating", "Rendering", "Started ", "Completed ", "Failed ", "Backtest worker PID")
EXAMPLE_ALPHAS = (
    ("20-day momentum × volume surge", DEFAULT_ALPHA),
    ("5-day reversal", "-1 * ($close / Ref($close, 5) - 1)"),
    ("Distance from 252-day high", "$close / Max($high, 252) - 1"),
    ("12-1 month momentum", "$close / Ref($close, 250) - $close / Ref($close, 20)"),
    ("Low volatility (20-day)", "-1 * Std($close / Ref($close, 1) - 1, 20)"),
    ("20-day z-score reversion", "-1 * ($close - Mean($close, 20)) / Std($close, 20)"),
    ("Price–volume decorrelation", "-1 * Corr($close, $volume, 20)"),
)


def _read_json(path: Path, limit: int = 64 * 1024 * 1024) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"missing or oversized {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return value


def _same_path(a: Path | None, b: Path | None) -> bool:
    """Compare locations, not spellings: a symlinked or relative data root must still match."""
    if a is None or b is None:
        return False
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return Path(a) == Path(b)


def latest_dataset(data_root: Path) -> Path | None:
    try:
        return Path(_read_json(Path(data_root) / "latest.json")["dataset"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def dataset_summary(data_root: Path, dataset: Path | None) -> dict[str, Any]:
    """Describe a prepared snapshot (or the default one) without importing Qlib."""
    chosen = Path(dataset) if dataset is not None else latest_dataset(data_root)
    if chosen is None:
        raise ValueError(f"No default dataset yet ({Path(data_root) / 'latest.json'}). Import one in the Data tab.")
    manifest = _read_json(chosen / "manifest.json")
    universes = {}
    for name, record in (manifest.get("universes") or {}).items():
        if isinstance(record, dict) and isinstance(name, str):
            universes[name] = {key: record.get(key) for key in (
                "size", "adv_sessions", "min_history_sessions", "first_effective", "full_size_from",
                "unique_members", "last_effective")}
    symbols = manifest.get("symbols") if isinstance(manifest.get("symbols"), list) else []
    dates = manifest.get("date_range") or {}
    return {
        "path": chosen, "name": chosen.name, "status": manifest.get("status", "unknown"),
        "provider": manifest.get("provider", "unknown provider"), "symbols": len(symbols),
        "sessions": manifest.get("sessions"), "start": dates.get("start_inclusive"),
        "end": dates.get("end_exclusive"), "universes": universes,
        **dataset_compatibility(manifest),
        "is_default": _same_path(latest_dataset(data_root), chosen),
    }


def dataset_compatibility(manifest: dict[str, Any]) -> dict[str, Any]:
    """Metadata-only recovery gate; the worker still verifies every dataset seal.

    Missing schema/field declarations are supported for older ordinary price
    snapshots and offline fixtures. Explicit pilot declarations never inherit
    that legacy OHLCV contract.
    """
    from qlib_yahoo.panel import available_fields

    role = manifest.get("data_role") or manifest.get("dataset_role")
    role = role or ("combined" if "fundamentals" in manifest else "prices")
    storage = manifest.get("alpha_input_storage", manifest.get("storage"))
    storage_kind = storage.get("format", "unknown") if isinstance(storage, dict) else "qlib-binary-tree"
    declared = manifest.get("fields")
    valid_fields = declared is None or (
        isinstance(declared, list) and bool(declared)
        and all(isinstance(value, str) and bool(value) for value in declared))
    fields = available_fields(manifest) if valid_fields else []
    schema = manifest.get("schema_version")
    unsupported = (
        "fundamentals" in manifest or "alpha_input_storage" in manifest
        or role not in ("prices", "price", "price-only")
        or any(manifest.get(key) not in (None, "", "prices", "price", "price-only")
               for key in ("data_role", "dataset_role"))
        or (schema is not None and (type(schema) is not int or schema != 1))
        or ("storage" in manifest and (not isinstance(storage, dict)
                                      or storage_kind != "qlib-zip-store-v1"))
        or "sec" in str(manifest.get("provider", "")).lower().split()
        or not valid_fields or not {"open", "high", "low", "close", "volume"}.issubset(fields)
    )
    note = ""
    if unsupported:
        note = ("This snapshot uses an unsupported SEC/pilot input, schema, or storage contract. "
                "It is preserved, not deleted; support is rebuild-pending. Select or import a ready ordinary "
                "Yahoo/Stooq price snapshot for backtests, Alpha Lab, and the default dataset.")
    elif manifest.get("status") != "ready":
        note = "Only a ready ordinary price dataset can be used. Finish or repeat the price import; existing files are preserved."
    return {"fields": fields, "price_eligible": not unsupported and manifest.get("status") == "ready",
            "data_role": role, "storage_kind": storage_kind, "compatibility_note": note}


def run_compatibility(manifest: dict[str, Any]) -> dict[str, Any]:
    """Keep newer saved results discoverable without replaying different semantics."""
    schema = manifest.get("schema_version")
    backtest = manifest.get("backtest") if isinstance(manifest.get("backtest"), dict) else {}
    execution = backtest.get("execution_contract") if isinstance(backtest.get("execution_contract"), dict) else {}
    unsupported = (
        (schema is not None and (type(schema) is not int or schema not in (3, 4, 5, 6)))
        or "data_sources" in manifest
        or any(key in section for section in (manifest, backtest, execution)
               for key in ("alpha_frontend", "classification", "alpha_settings"))
        or any(section.get("alpha_dialect", "qlib") != "qlib" for section in (manifest, backtest, execution))
    )
    note = ("This saved run uses a newer schema or PR45 input/execution contract unsupported by this recovery build. "
            "Its results are preserved, not deleted, and listed read-only. Report opening, regeneration, and settings "
            "reuse are unavailable until that support is rebuilt; use a compatible build to inspect or replay it."
            if unsupported else "")
    return {"read_only": bool(unsupported), "reports_supported": not unsupported,
            "reuse_supported": not unsupported, "compatibility_note": note}


def list_datasets(data_root: Path) -> list[dict[str, Any]]:
    folder = Path(data_root) / "datasets"
    if not folder.is_dir():
        return []
    default = latest_dataset(data_root)
    rows = []
    for path in sorted((p for p in folder.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True):
        try:
            summary = dataset_summary(data_root, path)
        except (OSError, ValueError):
            state = "failed" if (path / "failed_manifest.json").is_file() else "incomplete"
            summary = {"path": path, "name": path.name, "status": state, "provider": "", "symbols": 0,
                       "sessions": None, "start": None, "end": None, "universes": {},
                       "fields": [], "price_eligible": False, "data_role": "unknown", "storage_kind": "unknown",
                       "compatibility_note": "No readable ready price manifest. Existing files are preserved; repeat the price import."}
        summary["is_default"] = _same_path(default, path)
        rows.append(summary)
    return rows


def set_default_dataset(data_root: Path, dataset: Path) -> None:
    summary = dataset_summary(data_root, dataset)
    if summary["status"] != "ready":
        raise ValueError("Only a ready dataset can become the default.")
    if not summary["price_eligible"]:
        raise ValueError(summary["compatibility_note"])
    pointer = Path(data_root) / "latest.json"
    temporary = pointer.with_name(pointer.name + ".tmp")
    temporary.write_text(json.dumps({"dataset": str(Path(dataset))}, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, pointer)


def list_runs(data_root: Path, limit: int = 200) -> list[dict[str, Any]]:
    """Newest saved runs first, with headline metrics when the run completed."""
    folder = Path(data_root) / "runs"
    if not folder.is_dir():
        return []
    return [run_row(path) for path in sorted((p for p in folder.iterdir() if p.is_dir()),
                                             key=lambda p: p.name, reverse=True)[:limit]]


def run_row(path: Path) -> dict[str, Any]:
    """One saved run's list entry: settings, headline metrics and its factor key."""
    from qlib_yahoo.workbench_store import factor_key

    row: dict[str, Any] = {"path": path, "name": path.name, "status": "unknown", "mode": "", "universe": "",
                           "alpha": "", "sharpe": None, "cagr": None, "max_drawdown": None, "volatility": None,
                           "total_return": None, "created": "", "sessions": None, "short_window": False,
                           "start": "", "end": "", "dataset": "", "transform": ""}
    try:
        manifest = _read_json(path / "manifest.json")
        row.update(run_compatibility(manifest))
        backtest = manifest.get("backtest") or {}
        execution = backtest.get("execution_contract") or {}
        row.update(status=manifest.get("status", "unknown"), alpha=str(manifest.get("alpha", "")),
                   mode=str(manifest.get("strategy_mode") or backtest.get("strategy_mode") or "long-only"),
                   universe=str((backtest.get("universe") or {}).get("name") or "all"),
                   created=str(manifest.get("created_at_utc", ""))[:16].replace("T", " "),
                   start=str(backtest.get("start") or ""), end=str(backtest.get("end") or ""),
                   dataset=Path(str(manifest.get("dataset") or "")).name,
                   transform=str(execution.get("signal_transform") or ""),
                   period=f"{backtest.get('start', '')} → {backtest.get('end', '')}")
        if row["status"] == "completed":
            metrics = (_read_json(path / "summary.json").get("metrics") or {}).get("strategy_net") or {}
            row.update(sharpe=metrics.get("sharpe"), cagr=metrics.get("cagr"), max_drawdown=metrics.get("max_drawdown"),
                       volatility=metrics.get("annualized_volatility"), total_return=metrics.get("cumulative_return"),
                       sessions=session_count(metrics))
            row["short_window"] = row["sessions"] is not None and row["sessions"] < SHORT_WINDOW_SESSIONS
    except (OSError, ValueError):
        pass
    row["factor"] = factor_key(row["alpha"])
    return row


def session_count(metrics: dict[str, Any]) -> int | None:
    """Sessions behind a run's metrics, or None when an old summary does not record them."""
    count = metrics.get("observations")
    return count if isinstance(count, int) and not isinstance(count, bool) and count >= 0 else None


def _iso_date(text: str, label: str) -> str:
    try:
        value = date.fromisoformat(text)
    except ValueError:
        value = None
    if value is None or value.isoformat() != text:
        raise ValueError(f"{label} must be a date like 2018-01-02, or blank for the default.")
    return text


def _integer(options: dict[str, str], key: str, label: str, default: int, low: int, high: int | None = None) -> int:
    text = options.get(key, "")
    if not text:
        return default
    try:
        value = int(text)
    except ValueError as error:
        raise ValueError(f"{label} must be a whole number.") from error
    if value < low or (high is not None and value > high):
        raise ValueError(f"{label} must be between {low} and {high}." if high is not None else f"{label} must be at least {low}.")
    return value


def _decimal(options: dict[str, str], key: str, label: str, default: float, low: float, high: float | None = None) -> float:
    text = options.get(key, "")
    if not text:
        return default
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(f"{label} must be a number.") from error
    if not math.isfinite(value) or value < low or (high is not None and value > high):
        raise ValueError(f"{label} must be a finite number from {low:g}" + (f" to {high:g}." if high is not None else " upward."))
    return value


def parse_backtest_options(options: dict[str, str] | None, strategy_mode: str) -> dict[str, Any]:
    """Validate window text fields; return only settings that differ from the runner defaults."""
    options = {key: (value or "").strip() for key, value in (options or {}).items()}
    settings: dict[str, Any] = {}
    universe = options.get("universe", "")
    if universe and universe != ALL_SYMBOLS:
        if not _UNIVERSE.fullmatch(universe) or universe == "all":
            raise ValueError("Choose a universe recorded by the selected dataset.")
        settings["universe"] = universe
    for key, label in (("start", "Start date"), ("end", "End date")):
        if options.get(key):
            settings[key] = _iso_date(options[key], label)
    if "start" in settings and "end" in settings and settings["start"] >= settings["end"]:
        raise ValueError("Start date must be before the end date.")
    warmup = _integer(options, "warmup_sessions", "Warm-up sessions", 252, 1, 5000)
    if warmup != 252:
        settings["warmup_sessions"] = warmup
    cost = _decimal(options, "cost_bps", "Cost per side (bps)", 5.0, 0.0, 500.0) / 10_000
    if not math.isclose(cost, 0.0005, rel_tol=0, abs_tol=1e-12):
        settings["cost_per_side"] = cost
    if strategy_mode == "long-only":
        topk = _integer(options, "topk", "Top-k", 5, 1, 5000)
        n_drop = _integer(options, "n_drop", "Replace per day", 1, 0, 5000)
        if n_drop > topk:
            raise ValueError("Replace per day cannot exceed Top-k.")
        if topk != 5:
            settings["topk"] = topk
        if n_drop != 1:
            settings["n_drop"] = n_drop
    else:
        borrow = _decimal(options, "borrow_rate", "Annual borrow rate", 0.004, 0.0, 1.0)
        if not math.isclose(borrow, 0.004, rel_tol=0, abs_tol=1e-12):
            settings["annual_borrow_rate"] = borrow
    return settings


def parse_import_options(table: str, start: str, end: str, size: str, *, adv: str = "20", history: str = "252",
                         coverage: str = "0.5", workers: str = "4", name: str = "", min_price: str = "0",
                         groups: str = "") -> dict[str, Any]:
    path = Path(table.strip()).expanduser()
    if not table.strip() or not path.is_file():
        raise ValueError("Choose the Stooq table (.parquet or .csv) or Stooq's daily US ASCII zip.")
    if path.suffix.lower() not in (".parquet", ".csv", ".zip"):
        raise ValueError("The Stooq source must be a .parquet or .csv table or the daily US ASCII .zip.")
    begin, stop = _iso_date(start.strip(), "Import start"), _iso_date(end.strip(), "Import end (exclusive)")
    if begin >= stop:
        raise ValueError("Import start must be before the exclusive end date.")
    fields = {"size": size, "adv": adv, "history": history, "coverage": coverage, "workers": workers,
              "min_price": min_price}
    size_value = _integer(fields, "size", "Universe size", 3000, 1, 20000)
    universe = name.strip() or f"top{size_value}"
    if not _UNIVERSE.fullmatch(universe) or universe == "all":
        raise ValueError("Universe name must be lowercase letters, digits or underscores (for example top3000).")
    request = {
        "kind": "import-table", "table": str(path), "start": begin, "end": stop,
        "rule": {"name": universe, "size": size_value,
                 "adv_sessions": _integer(fields, "adv", "ADV window", 20, 1, 1000),
                 "min_history_sessions": _integer(fields, "history", "Minimum history", 252, 1, 5000),
                 "min_price": _decimal(fields, "min_price", "Minimum price", 0.0, 0.0, 1000.0)},
        "min_session_coverage": _decimal(fields, "coverage", "Session coverage", 0.5, 0.0, 0.99),
        "max_workers": _integer(fields, "workers", "Workers", 4, 1, 64),
    }
    if groups.strip():
        folder = Path(groups.strip()).expanduser()
        if not (folder / "manifest.json").is_file() or not (folder / "screener.json.gz").is_file():
            raise ValueError("Groups must be a folder made by fetch-nasdaq-screener (manifest.json and screener.json.gz).")
        request["groups"] = str(folder)
    return request
