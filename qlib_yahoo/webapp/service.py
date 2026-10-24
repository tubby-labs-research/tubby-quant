"""Application service behind the desktop web UI: state, jobs, runs and safe file access.

The service never imports Qlib. Backtests and imports run in the isolated,
cancellable worker process from ``qlib_yahoo.jobs``; report rendering runs on a
background thread. Every path received from the UI must stay inside the data
folder's ``datasets`` or ``runs`` directories.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import platform
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict, deque
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from qlib_yahoo.dataset import DatasetError
from qlib_yahoo.workbench_store import Annotations, factor_key
from qlib_yahoo.workflow_submission import current_submission, record_console_submission
from qlib_yahoo.workspace import (
    ALL_SYMBOLS, EXAMPLE_ALPHAS, SHORT_WINDOW_SESSIONS, STAGE_PREFIXES, dataset_summary, latest_dataset, list_datasets,
    list_runs, parse_backtest_options, parse_import_options, session_count, set_default_dataset,
    run_compatibility,
)

ROOT = Path(__file__).resolve().parents[2]
MAX_POINTS = 1200
MAX_DAILY = 20_000  # full daily series for the result view (a 16-year run has about 4,000 sessions)
COUNTED = re.compile(r"^\[(\d+)/(\d+)\] ")
FETCHED = re.compile(r"^(Fetched|Built|Reconciled) ([\d,]+)/([\d,]+) ")
STEP = re.compile(r"^\d/5 ")
# Run files that saved reports link to ("Saved data and provenance"); nothing else outside reports/ is served.
REPORT_LINKED_FILES = frozenset({"summary.json", "daily_report.csv", "positions.csv.gz", "positions.parquet", "manifest.json"})


class ServiceError(ValueError):
    """A user-facing validation or state error (reported as HTTP 400)."""


def _sec_alias_guidance(message: str) -> str:
    match = re.search(r"(?:unsupported feature: \$|unknown field ['\"])(sec_[A-Za-z0-9_]+)", message)
    if match:
        alias = match.group(1)
        return (f"Unknown SEC input {alias!r}. Choose a SEC catalog in SEC fields, then search the shared "
                "dictionary for an existing official field name and definition.")
    return message


def reject_recovery_options(params: dict[str, Any]) -> None:
    """Do not silently ignore execution/input options from the reverted PR45 UI."""
    stale = sorted(key for key in params if key in {
        "classification", "classifications", "price_dataset", "alpha_dialect", "alpha_settings",
        "fast_settings", "execution_phase", "deal_price", "reuse_run", "source_run",
    } or key.startswith(("fast_", "classification_")))
    if stale:
        raise ServiceError("Unsupported PR45 input/execution options: " + ", ".join(stale)
                           + ". This recovery build supports ordinary price datasets and baseline execution. "
                             "Clear the saved pilot/classification/FASTEXPR execution options; existing data and results are preserved.")


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


class JobRecord:
    """Observable state of one task; ``log`` keeps the last lines with a running index."""

    def __init__(self, kind: str, label: str, request: dict[str, Any]):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.label = label
        self.request = request
        self.state = "running"
        self.started = time.time()
        self.finished: float | None = None
        self.progress: float | None = None
        self.progress_label = ""
        self.stage = label
        self.result: str | None = None
        self.error: str | None = None
        self.lines: deque[tuple[int, str]] = deque(maxlen=500)
        self.count = 0
        self.lock = threading.Lock()
        self.cancel_hook: Callable[[], bool] | None = None
        self.attempt_id: str | None = None
        self.receipt: dict[str, Any] | None = None

    def add(self, text: str) -> None:
        with self.lock:
            self.count += 1
            self.lines.append((self.count, text[:400]))

    def public(self, since: int = 0) -> dict[str, Any]:
        with self.lock:
            lines = [text for number, text in self.lines if number > since]
            value = {"id": self.id, "kind": self.kind, "label": self.label, "state": self.state,
                    "started": self.started, "finished": self.finished,
                    "elapsed": (self.finished or time.time()) - self.started,
                    "progress": self.progress, "progress_label": self.progress_label, "stage": self.stage,
                    "result": self.result, "error": self.error, "log": lines, "log_index": self.count,
                    "attempt_id": self.attempt_id,
                    "cancellable": self.cancel_hook is not None and self.state == "running"}
            if self.kind in ("chronology-trial", "chronology-final"):
                value.update(result_kind="chronology", run_path=self.result, receipt=self.receipt)
            return value


class ConsoleService:
    def __init__(self, data_root: Path, *, job_factory=None, settings_path: Path | None = None):
        if job_factory is None:
            from qlib_yahoo.jobs import Job
            job_factory = Job
        # Resolved once, so paths shown to the page and paths checked here agree for symlinked roots.
        self.data_root = Path(data_root).expanduser().resolve()
        self.job_factory = job_factory
        self.settings_path = settings_path
        self._lock = threading.Lock()
        self._sec_variant_lock = threading.Lock()
        self._sec_variant_cache = OrderedDict()
        self.current: JobRecord | None = None
        self.jobs: dict[str, JobRecord] = {}
        from qlib_yahoo.webapp.research_service import ResearchService
        self.research = ResearchService(self.data_root)

    # ------------------------------------------------------------------ paths

    def _inside(self, value: str | Path | None, *areas: str) -> Path:
        """Resolve a UI-supplied path; it must lie inside one of the data folder areas."""
        if not value:
            raise ServiceError("A path is required.")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.data_root / path
        resolved = path.resolve()
        roots = [(self.data_root / area).resolve() for area in (areas or ("datasets", "runs"))]
        if not any(resolved == root or root in resolved.parents for root in roots):
            raise ServiceError("That location is outside the data folder.")
        return resolved

    def report_file(self, relative: str) -> Path:
        """A file inside a run's reports folder, or a saved-data file those reports link to."""
        path = self._inside(self.data_root / "runs" / relative, "runs")
        parts = path.relative_to((self.data_root / "runs").resolve()).parts
        linked = len(parts) == 2 and parts[1] in REPORT_LINKED_FILES
        if not ("reports" in parts or linked) or not path.is_file():
            raise ServiceError("Report file not found.")
        self._require_supported_run((self.data_root / "runs" / parts[0]).resolve())
        return path

    def _require_supported_run(self, run: Path) -> dict[str, Any]:
        try:
            manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
            compatibility = run_compatibility(manifest)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise ServiceError(f"Cannot read this saved run's compatibility metadata: {exc}") from exc
        if compatibility["read_only"]:
            raise ServiceError(compatibility["compatibility_note"])
        return compatibility

    def _price_dataset(self, value: str | Path | None) -> tuple[Path, dict[str, Any]]:
        """Resolve and check headers only, before creating a worker or report job."""
        if value is not None and not isinstance(value, (str, Path)):
            raise ServiceError("Choose a price dataset folder.")
        selected = value or latest_dataset(self.data_root)
        if selected is None:
            raise ServiceError("No prepared dataset. Import a ready ordinary price snapshot on the Data page first.")
        resolved = self._inside(selected, "datasets")
        try:
            summary = dataset_summary(self.data_root, resolved)
        except (OSError, ValueError) as exc:
            raise ServiceError("No readable prepared price dataset. Select or import a ready ordinary price snapshot. "
                               "Existing files are preserved.") from exc
        if not summary["price_eligible"]:
            raise ServiceError(summary["compatibility_note"])
        return resolved, summary

    @staticmethod
    def _lab_groups(dataset: Path) -> set[str]:
        """Static group header only; the worker verifies the actual CSV and seal."""
        manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
        record = manifest.get("groups") or {}
        groups = {"market"}
        if isinstance(record, dict):
            # The importer uses fields. Older records have the fixed CSV
            # contract consumed by reference.read_groups without a declaration.
            declared = record.get("fields", record.get("columns"))
            if declared is None and record.get("file") == "universe/groups.csv":
                declared = ("sector", "industry", "country")
            if isinstance(declared, (list, tuple)):
                groups.update(name for name in declared if name in ("sector", "industry", "country"))
        return groups

    # ------------------------------------------------------------------ read models

    def overview(self) -> dict[str, Any]:
        try:
            from qlib_yahoo.runtime import signed_runtime_status

            signed_ok, signed_detail = signed_runtime_status()
            if signed_ok:
                from qlib_yahoo.fast_runtime import fast_runtime_status
                fast_ok, fast_detail = fast_runtime_status()
                signed_ok = signed_ok and fast_ok
                signed_detail = f"{signed_detail}; {fast_detail}"
        except Exception as exc:  # informational
            signed_ok, signed_detail = False, str(exc)
        try:
            default = dataset_summary(self.data_root, None)
        except (OSError, ValueError) as exc:
            default = {"error": str(exc)}
        version = "local"
        try:
            # An extracted framework may sit beneath an unrelated private Git
            # checkout. Do not discover or report that parent's commit.
            if (ROOT / ".git").exists():
                version = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                                         text=True, timeout=5).stdout.strip() or version
        except (OSError, subprocess.SubprocessError):
            pass
        return _json_safe({
            "data_root": self.data_root, "default_dataset": default,
            "signed": {"available": signed_ok, "detail": signed_detail},
            "platform": f"{platform.system()} {platform.machine()}", "python": sys.version.split()[0],
            "version": version, "examples": [{"label": a, "alpha": b} for a, b in EXAMPLE_ALPHAS],
            "job": self.current.public() if self.current else None, "all_symbols": ALL_SYMBOLS,
            "lab_examples": _lab_examples(),
        })

    def datasets(self) -> list[dict[str, Any]]:
        return _json_safe(list_datasets(self.data_root))

    def _sec_path(self, value: str | Path) -> Path:
        if not isinstance(value, (str, Path)) or not value:
            raise ServiceError("Choose a ready SEC catalog.")
        root = (self.data_root / "sec" / "catalogs").resolve()
        path = self._inside(value, "sec/catalogs")
        if path.parent != root or not path.is_dir() or not (path / "manifest.json").is_file():
            raise ServiceError("Choose a ready SEC catalog directly under sec/catalogs.")
        return path

    def _sec_catalog(self, value: str | Path) -> tuple[Path, dict[str, Any]]:
        from qlib_yahoo.sec_catalog import read_catalog

        path = self._sec_path(value)
        try:
            manifest = read_catalog(path, verify=False)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ServiceError(f"SEC catalog is not ready: {exc}") from exc
        return path, manifest

    def sec_catalogs(self) -> list[dict[str, Any]]:
        rows = []
        root = self.data_root / "sec" / "catalogs"
        if not root.is_dir():
            return rows
        for path in sorted(root.iterdir()):
            if not path.is_dir():
                continue
            try:
                resolved, manifest = self._sec_catalog(path)
            except ServiceError:
                continue
            compact = {key: manifest.get(key) for key in ("status", "product", "numeric_materialization_supported", "catalog_sha256")}
            compact["tables"] = {name: {key: record.get(key) for key in ("columns", "rows")}
                                 for name, record in manifest.get("tables", {}).items()}
            rows.append({"path": str(resolved), "name": path.name, "manifest": compact})
            if manifest.get("catalog_sha256"):
                from qlib_yahoo.sec_registry import dictionary_status

                rows[-1]["dictionary"] = dictionary_status(resolved)
        return _json_safe(rows)

    def sec_dictionary(self, catalog: str, query: str = "", offset: int = 0,
                       include_custom: bool = False, view: str = "recipes") -> dict[str, Any]:
        from qlib_yahoo.sec_registry import dictionary_status, search_dictionary
        from qlib_yahoo.sec_execution import execution_status

        if (not isinstance(query, str) or len(query) > 4096 or type(offset) is not int
                or not 0 <= offset <= 10000000 or type(include_custom) is not bool
                or view not in ("company", "recipes")
                or (view == "company" and include_custom)):
            raise ServiceError("Invalid SEC dictionary search or page.")
        path, _ = self._sec_catalog(catalog)
        status = dictionary_status(path)
        page = (search_dictionary(path, query=query, limit=50, offset=offset, include_custom=include_custom, view=view)
                if status["status"] == "ready" else {"items": [], "total": 0, "offset": offset})
        return _json_safe({**status, **page, "view": view, "execution": execution_status(path)})

    def start_sec_dictionary_build(self, params: dict[str, Any]) -> dict[str, Any]:
        path, _ = self._sec_catalog(params.get("catalog"))
        request = {"kind": "build-dictionary", "catalog": str(path)}
        return self._start("build-dictionary", f"Build SEC dictionary: {path.name}", request,
                           lambda record: self._run_worker(record, request))

    def sec_name(self, params: dict[str, Any]) -> dict[str, Any]:
        from qlib_yahoo.sec_registry import name_for_spec

        path, _ = self._sec_catalog(params.get("catalog"))
        if not isinstance(params.get("spec"), dict):
            raise ServiceError("SEC spec must be an object.")
        return _json_safe(name_for_spec(path, params["spec"]))

    def sec_search(self, catalog: str, query: str = "", offset: int = 0) -> dict[str, Any]:
        from qlib_yahoo.sec_catalog import search_tags

        path, _ = self._sec_catalog(catalog)
        if len(query) > 256 or not 0 <= offset <= 10_000_000:
            raise ServiceError("Invalid SEC search or page.")
        return _json_safe(search_tags(path, query=query, limit=50, offset=offset))

    def sec_fields(self, catalog: str, query: str, offset: int = 0, include_custom: bool = False) -> dict[str, Any]:
        from qlib_yahoo.sec_catalog import search_fields

        if (not isinstance(query, str) or len(query) > 256 or type(offset) is not int
                or not 0 <= offset <= 10_000_000 or type(include_custom) is not bool):
            raise ServiceError("Invalid SEC field search or page.")
        path, _ = self._sec_catalog(catalog)
        return _json_safe(search_fields(path, query=query, limit=50, offset=offset, include_custom=include_custom))

    def sec_variants(self, catalog: str, tag: str, taxonomy: str) -> list[dict[str, Any]]:
        from qlib_yahoo.sec_catalog import field_variants

        path, manifest = self._sec_catalog(catalog)
        if (not isinstance(tag, str) or not isinstance(taxonomy, str)
                or not tag or not taxonomy or max(len(tag), len(taxonomy)) > 512):
            raise ServiceError("Select a tag and exact taxonomy version.")
        identity = manifest.get('catalog_sha256')
        key = (str(path), identity, tag, taxonomy)
        # Validate the ready manifest on every visit, even a cache hit. Do not
        # cache failures or headers without an identity (e.g. legacy mocks).
        with self._sec_variant_lock:
            if identity and key in self._sec_variant_cache:
                self._sec_variant_cache.move_to_end(key)
                return deepcopy(self._sec_variant_cache[key])
        result = _json_safe(field_variants(path, tag, taxonomy))
        with self._sec_variant_lock:
            if identity and len(result) <= 20_000:
                self._sec_variant_cache[key] = deepcopy(result)
                self._sec_variant_cache.move_to_end(key)
                while len(self._sec_variant_cache) > 8:
                    self._sec_variant_cache.popitem(last=False)
        return result

    def sec_rows(self, catalog: str, table: str, filters=None, offset: int = 0) -> dict[str, Any]:
        from qlib_yahoo.sec_catalog import query_rows

        path, manifest = self._sec_catalog(catalog)
        tables = {str(key).lower() for key in manifest.get("tables", {})}
        if table.lower() not in tables or not 0 <= offset <= 10_000_000:
            raise ServiceError("Choose a supplied table and a valid page.")
        if isinstance(filters, str):
            try:
                filters = json.loads(filters or "{}")
            except ValueError as exc:
                raise ServiceError("Table filters must be a JSON object.") from exc
        if filters is not None and not isinstance(filters, dict):
            raise ServiceError("Table filters must be a JSON object.")
        try:
            rows = query_rows(path, table.lower(), filters=filters, limit=100, offset=offset)
        except (ValueError, OSError) as exc:
            raise ServiceError(str(exc)) from exc
        record = manifest["tables"][table.lower()]
        metadata = {"table": table.lower(), "columns": record.get("columns"), "rows": record.get("rows")}
        return _json_safe({"rows": rows, "metadata": metadata, "offset": offset})

    def _sec_config(self, sec, *, ready=True) -> dict[str, Any] | None:
        """Small header and binding checks only. Workers verify bulk content."""
        if sec is None:
            return None
        if not isinstance(sec, dict) or set(sec) - {"catalog", "bindings", "mode", "registry"}:
            raise ServiceError("SEC input must contain catalog, optional bindings, mode and registry.")
        if sec.get("mode") not in (None, "dictionary", "legacy"):
            raise ServiceError("SEC mode must be dictionary or legacy.")
        path = self._sec_path(sec.get("catalog"))
        from qlib_yahoo.sec_features import load_bindings

        try:
            bindings = load_bindings(sec) if "bindings" in sec else {}
        except (ValueError, TypeError, OSError) as exc:
            raise ServiceError(f"Invalid SEC bindings: {exc}") from exc
        if ready:
            _, manifest = self._sec_catalog(path)
            if bindings and not manifest.get("numeric_materialization_supported", False):
                raise ServiceError("This catalog supports source-table browsing only. Numeric notes/dimension inputs need a schema adapter.")
        return {**sec, "catalog": str(path), **({"bindings": bindings} if "bindings" in sec else {})}

    def _resolve_sec(self, expressions, sec, *, dialect):
        if sec is None:
            return None
        from qlib_yahoo.sec_resolution import resolve_config

        try:
            return resolve_config(expressions, self._sec_config(sec, ready=False), dialect=dialect, verify=False)
        except (ValueError, DatasetError, OSError, TypeError) as exc:
            raise ServiceError(str(exc)) from exc

    @staticmethod
    def _sec_preview(config):
        return {"sec_inputs": [{"name": name, "spec": spec} for name, spec in
                               sorted((config or {}).get("bindings", {}).items())],
                "registry": (config or {}).get("registry")}

    def dataset(self, path: str | None) -> dict[str, Any]:
        chosen = self._inside(path, "datasets") if path else None
        try:
            summary = dataset_summary(self.data_root, chosen)
        except (OSError, ValueError) as exc:
            raise ServiceError(str(exc)) from exc
        rankings = {}
        for name, record in (summary.get("universes") or {}).items():
            rankings[name] = self._ranking_series(Path(summary["path"]), name)
        summary["ranking_series"] = rankings
        try:
            manifest = json.loads((Path(summary["path"]) / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = {}
        # Small audit sections only; per-symbol hashes and row counts stay on disk.
        summary["audit"] = {key: manifest.get(key) for key in (
            "security_filter", "session_filter", "invalid_bars", "zero_volume_rows", "limitations",
            "timings_seconds", "normalization", "calendar", "created_at_utc")}
        source = manifest.get("source_table") or {}
        summary["audit"]["source"] = {"path": source.get("path"), "rows": source.get("rows_in_interval"),
                                      "size_bytes": (source.get("before") or {}).get("size_bytes")}
        summary["audit"]["file_count"] = manifest.get("qlib_binary_file_count")
        summary["audit"]["table_builder"] = manifest.get("table_builder")
        summary["audit"]["groups"] = {key: (manifest.get("groups") or {}).get(key) for key in ("coverage", "limitation")} \
            if manifest.get("groups") else None
        calendar = Path(summary["path"]) / "qlib_data" / "calendars" / "day.txt"
        try:
            summary["calendar"] = calendar.read_text(encoding="utf-8").split()
        except OSError:
            summary["calendar"] = []
        return _json_safe(summary)

    def _ranking_series(self, dataset: Path, name: str) -> list[dict[str, Any]]:
        path = dataset / "universe" / f"{name}_rankings.csv"
        if not path.is_file():
            return []
        with path.open(newline="", encoding="utf-8") as handle:
            return [{"date": row["ranked_on"], "members": int(row["members"]), "eligible": int(row["eligible"])}
                    for row in csv.DictReader(handle)]

    def runs(self, limit: int = 1000) -> list[dict[str, Any]]:
        notes = Annotations(self.data_root).load()
        rows = list_runs(self.data_root, limit)
        for row in rows:
            _annotate(row, notes)
        return _json_safe(rows)

    def run_detail(self, path: str) -> dict[str, Any]:
        run = self._inside(path, "runs")
        try:
            manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ServiceError(f"Cannot read this run: {exc}") from exc
        backtest = manifest.get("backtest") or {}
        execution = backtest.get("execution_contract") or {}
        mode = manifest.get("strategy_mode") or backtest.get("strategy_mode") or "long-only"
        detail: dict[str, Any] = {
            "path": run, "name": run.name, "status": manifest.get("status"), "alpha": manifest.get("alpha"),
            "factor": factor_key(str(manifest.get("alpha") or "")),
            "annual_cash_rate": backtest.get("annual_cash_rate") or execution.get("annual_cash_rate") or 0.0,
            "diagnostics": (manifest.get("diagnostics") or {}).get("forward_labels"),
            "mode": mode, "created": manifest.get("created_at_utc"), "error": manifest.get("error"),
            "dataset": Path(str(manifest.get("dataset", ""))).name,
            "dataset_path": manifest.get("dataset"),
            # Saved SEC provenance remains visible even for read-only runs.
            "sec": manifest.get("sec"),
            "period": {"start": backtest.get("start"), "end": backtest.get("end")},
            "universe": (backtest.get("universe") or {}).get("name") or "all",
            "universe_scoring": manifest.get("universe_scoring"),
            "settings": {key: execution.get(key) for key in (
                "topk", "n_drop", "risk_degree", "open_cost", "gross_exposure", "annual_borrow_rate", "signal_transform")},
            "warmup_sessions": backtest.get("warmup_sessions"), "cash": backtest.get("cash"),
            "trade_symbols": len(backtest.get("trade_symbols") or []),
            **run_compatibility(manifest),
        }
        if manifest.get("status") == "completed":
            try:
                summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
                detail["metrics"] = (summary.get("metrics") or {}).get("strategy_net") or {}
                detail["funding_state"] = summary.get("funding_state")
                sessions = session_count(detail["metrics"])
                detail["short_window"] = sessions is not None and sessions < SHORT_WINDOW_SESSIONS
            except (OSError, ValueError):
                detail["metrics"] = {}
            if not detail["read_only"]:
                detail["series"] = self._series(run)
                detail["holdings"] = self._latest_holdings(run)
        if detail["read_only"]:
            detail["settings"] = {}
        detail["reports"] = [] if detail["read_only"] else self._reports(run)
        detail["signal_available"] = not detail["read_only"] and (run / "signal_forward_labels.parquet").is_file()
        _annotate(detail, Annotations(self.data_root).load())
        return _json_safe(detail)

    def _series(self, run: Path) -> dict[str, Any]:
        """The run's daily ledger: equity, net return, turnover, costs and exposure (not downsampled)."""
        path = run / "daily_report.csv"
        if not path.is_file():
            return {}
        columns = {"account": [], "net_return": [], "turnover": [], "cost": [], "long": [], "short": [], "holdings": []}
        dates: list[str] = []
        sources = {"long": "long_value", "short": "short_liability", "holdings": "holdings_value"}
        present: set[str] = set()

        def number(row: dict[str, str], key: str) -> float | None:
            try:
                value = float(row.get(key) or "nan")
            except ValueError:
                return None
            return value if math.isfinite(value) else None

        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            present = set(reader.fieldnames or [])
            for row in reader:
                account = number(row, "account")
                if account is None or not row.get("datetime"):
                    continue
                dates.append(row["datetime"][:10])
                columns["account"].append(account)
                columns["net_return"].append(number(row, "net_return"))
                columns["turnover"].append(number(row, "turnover") or 0.0)
                columns["cost"].append(number(row, "total_cost_dollars") if "total_cost_dollars" in present
                                       else number(row, "cost_dollars") or number(row, "total_cost_amount") or 0.0)
                for key, source in sources.items():
                    columns[key].append(number(row, source))
        if not dates:
            return {}
        account = columns["account"]
        peak, drawdown = -math.inf, []
        for value in account:
            peak = max(peak, value)
            drawdown.append(value / peak - 1 if peak > 0 else 0.0)
        if all(value is None for value in columns["net_return"]):  # older ledgers: rebuild from equity
            columns["net_return"] = [None] + [b / a - 1 if a else None for a, b in zip(account, account[1:])]
        keep = slice(-MAX_DAILY, None)
        series = {"dates": dates[keep], "account": account[keep], "drawdown": drawdown[keep],
                  "net_return": columns["net_return"][keep], "turnover": columns["turnover"][keep],
                  "cost": columns["cost"][keep], "start": account[0], "final": account[-1], "sessions": len(dates),
                  "mean_turnover": sum(columns["turnover"]) / len(dates), "total_cost": sum(c or 0.0 for c in columns["cost"])}
        for key, source in sources.items():
            if source in present and any(value is not None for value in columns[key]):
                series[key] = columns[key][keep]
        return series

    def _latest_holdings(self, run: Path, limit: int = 15) -> dict[str, Any]:
        if (run / "positions.parquet").is_file():
            return self._latest_holdings_parquet(run / "positions.parquet", limit)
        path = run / "positions.csv.gz"
        if not path.is_file():
            return {}
        latest, rows, counts = None, [], []
        with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            day, names = None, 0
            for row in reader:
                if row.get("datetime") != day:
                    if day is not None:
                        counts.append(names)
                    day, names, rows = row.get("datetime"), 0, []
                instrument = row.get("instrument") or ""
                if instrument:
                    names += 1
                    try:
                        weight = float(row.get("weight") or 0)
                    except ValueError:
                        weight = 0.0
                    rows.append({"instrument": instrument, "weight": weight})
                latest = day
            if day is not None:
                counts.append(names)
        rows.sort(key=lambda item: abs(item["weight"]), reverse=True)
        return {"date": (latest or "")[:10], "top": rows[:limit], "count": len(rows),
                "mean_names": sum(counts) / len(counts) if counts else 0}

    @staticmethod
    def _latest_holdings_parquet(path: Path, limit: int) -> dict[str, Any]:
        from qlib_yahoo.analysis import instrument_batches

        counts, latest, rows = {}, None, []
        for batch in instrument_batches(path, ["datetime", "instrument", "weight"]):
            for date, group in batch.groupby("datetime", sort=False):
                held = group[group["instrument"].fillna("") != ""]
                counts[date] = counts.get(date, 0) + len(held)
                if latest is None or date > latest:
                    latest, rows = date, []
                if date == latest:
                    rows.extend({"instrument": str(name), "weight": float(weight or 0.0)}
                                for name, weight in held[["instrument", "weight"]].itertuples(index=False, name=None))
        if latest is None:
            return {}
        rows.sort(key=lambda item: abs(item["weight"]), reverse=True)
        return {"date": latest.strftime("%Y-%m-%d"), "top": rows[:limit], "count": len(rows),
                "mean_names": sum(counts.values()) / len(counts)}

    def _reports(self, run: Path) -> list[dict[str, Any]]:
        folder = run / "reports"
        found = []
        if folder.is_dir():
            for directory in sorted(folder.iterdir(), reverse=True):
                status_path = directory / "report_manifest.json"
                index = directory / "report_index.html"
                if not index.is_file() or not status_path.is_file():
                    continue
                try:
                    status = json.loads(status_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                found.append({"url": "/files/" + index.relative_to((self.data_root / "runs").resolve()).as_posix()
                              if index.resolve().is_relative_to((self.data_root / "runs").resolve()) else None,
                              "path": index, "level": status.get("level"), "holdings": status.get("holdings_rendered"),
                              "created": directory.name[:15]})
        return found

    def check_alpha(self, alpha: str, sec=None) -> dict[str, Any]:
        from qlib_yahoo.expressions import check_alpha_syntax

        lines = _alpha_lines(alpha)
        if not lines:
            return {"ok": False, "message": "Enter an expression.", "count": 0}
        try:
            config = self._resolve_sec([text for _, text in lines], sec, dialect="qlib")
        except ServiceError as exc:
            return {"ok": False, "message": str(exc), "count": len(lines)}
        extra = set(config["bindings"]) if config else set()
        preview = self._sec_preview(config)
        for number, text in lines:
            try:
                check_alpha_syntax(text, extra_features=extra) if extra else check_alpha_syntax(text)
            except Exception as exc:
                prefix = f"Line {number}: " if len(lines) > 1 else ""
                return {"ok": False, "message": prefix + _sec_alias_guidance(str(exc)), "count": len(lines)}
        if len(lines) > 1:
            return {"ok": True, "count": len(lines), **preview,
                    "message": f"{len(lines)} alphas — each line runs as its own backtest, one after another."}
        return {"ok": True, "count": 1, **preview, "message": "Syntax OK — history windows are verified when the run starts."}

    # ------------------------------------------------------------------ jobs

    def job(self, job_id: str, since: int = 0) -> dict[str, Any]:
        record = self.jobs.get(job_id)
        if record is None:
            raise ServiceError("Unknown task.")
        return _json_safe(record.public(since))

    def busy(self) -> bool:
        return (self.current is not None and self.current.state == "running") or self.research.busy()

    def start_research(self, data: dict[str, Any], *, demo: bool = False) -> dict[str, Any]:
        # Share the task gate with imports/backtests; their data folder can be changing.
        with self._lock:
            if self.current is not None and self.current.state == "running":
                raise ServiceError("Another task is running. Stop it before starting research.")
            try:
                return self.research.demo() if demo else self.research.start(data)
            except ValueError as exc:
                raise ServiceError(str(exc)) from exc

    def finalize_research(self, data: dict[str, Any]) -> dict[str, Any]:
        """Score promoted ideas once on a campaign's reserved holdout; shares the task gate like a start."""
        with self._lock:
            if self.current is not None and self.current.state == "running":
                raise ServiceError("Another task is running. Stop it before finalizing research.")
            try:
                return self.research.finalize(data)
            except ValueError as exc:
                raise ServiceError(str(exc)) from exc

    def _start(self, kind: str, label: str, request: dict[str, Any], work: Callable[[JobRecord], str]) -> dict[str, Any]:
        with self._lock:
            if self.busy():
                raise ServiceError("Another task is running. Wait for it to finish or cancel it.")
            record = JobRecord(kind, label, request)
            submission = current_submission()
            if submission:
                record.attempt_id = submission["id"]
            self.current = record
            self.jobs[record.id] = record
            while len(self.jobs) > 50:
                self.jobs.pop(next(iter(self.jobs)))

        def run() -> None:
            from qlib_yahoo.jobs import JobCancelled

            outcome = "failed"
            try:
                record.result = work(record)
                if record.receipt is not None and record.receipt.get("status") not in ("completed", "reused"):
                    raise DatasetError("Chronology numerical outcome: " + str(record.receipt.get("status"))
                                       + ". Inspect the retained receipt and child attempts.")
                outcome = "completed"
                record.stage = "Completed"
                record.add(f"Completed: {record.result}")
            except JobCancelled as exc:
                outcome = "cancelled"
                record.stage = "Cancelled"
                record.error = str(exc)
                record.add(str(exc))
            except Exception as exc:
                outcome = "failed"
                record.error = str(exc) or type(exc).__name__
                record.stage = "Failed"
                record.add(f"Error: {record.error}")
            finally:
                if record.attempt_id:
                    from qlib_yahoo.workflow_records import EvidenceStore
                    try:
                        EvidenceStore(self.data_root).finish(
                            record.attempt_id, outcome,
                            result={"result_path": record.result} if record.result else None,
                            error=record.error,
                        )
                    except Exception as exc:
                        outcome = "failed"
                        record.error = f"Could not seal submission evidence: {exc}"
                        record.stage = "Evidence persistence failed"
                        record.add(record.error)
                # A terminal UI state promises that durable invocation evidence
                # has already been sealed (or its persistence error reported).
                with record.lock:
                    record.finished = time.time()
                    record.cancel_hook = None
                    record.state = outcome

        record.add(label)
        threading.Thread(target=run, name=f"job-{record.id}", daemon=True).start()
        return record.public()

    def _run_worker(self, record: JobRecord, request: dict[str, Any]) -> str:
        from qlib_yahoo.jobs import progress_fraction

        submitted_request = dict(request)
        if record.attempt_id:
            submitted_request["_workflow"] = {"data_root": str(self.data_root),
                                               "parents": [record.attempt_id]}
        job = self.job_factory(submitted_request)
        record.cancel_hook = job.cancel

        def line(text: str) -> None:
            fraction = progress_fraction(text)
            if fraction is not None:
                record.progress_label, record.progress = fraction
                return
            counted = COUNTED.match(text)
            if counted:  # "[3/25] ..." from Alpha Lab batches
                record.progress_label, record.progress = "Alphas", int(counted[1]) / max(int(counted[2]), 1)
            fetched = FETCHED.match(text)
            if fetched:  # "Fetched 1,200/8,957 charts" from market-data refreshes
                done, total = (int(fetched[k].replace(",", "")) for k in (2, 3))
                record.progress_label, record.progress = fetched[1], done / max(total, 1)
                return
            if STEP.match(text):
                record.stage = text
                record.progress, record.progress_label = None, ""
            text = text.strip()
            if text.startswith(STAGE_PREFIXES) or text[:1].isdigit() and "monthly rankings" in text:
                record.stage = text
                record.progress, record.progress_label = None, ""  # a new stage; the old bar is finished
            record.add(text)

        result = job.run(line)
        if record.kind in ("chronology-trial", "chronology-final"):
            from qlib_yahoo.workflow_records import EvidenceStore, _checked
            store = EvidenceStore(self.data_root)
            path = _checked(Path(result))
            if path.parent != store.root / "results" or path.suffix != ".json":
                raise ServiceError("chronology worker returned a receipt outside workflow/results")
            receipt = store.get_result(path.stem)
            phase = "final" if record.kind == "chronology-final" else None
            if (not isinstance(receipt, dict) or receipt.get("schema_version") != 1
                    or receipt.get("phase") not in ((phase,) if phase else ("development", "validation"))):
                raise ServiceError("chronology worker returned an incompatible receipt")
            record.receipt = receipt
        return str(result)

    @record_console_submission("console-backtest")
    def start_backtest(self, params: dict[str, Any]) -> dict[str, Any]:
        reject_recovery_options(params)
        lines = _alpha_lines(str(params.get("alpha") or ""))
        if len(lines) > 1:
            return self._start_backtest_batch(params, lines)
        alpha = lines[0][1] if lines else ""
        request, notes, resolved = self._backtest_request({**params, "alpha": alpha})
        mode = request.get("strategy_mode", "long-only")
        label = f"{'Signed' if mode == 'signed' else 'Long-only'} backtest on {resolved.name}"

        def work(record: JobRecord) -> str:
            for note in notes:
                record.add(note)
            return self._run_worker(record, request)

        return self._start("backtest", label, request, work)

    def _backtest_request(self, params: dict[str, Any]) -> tuple[dict[str, Any], list[str], Path]:
        """Validate one backtest's settings; return the worker request, date-snapping notes and the dataset."""
        from qlib_yahoo.expressions import check_alpha_syntax

        reject_recovery_options(params)
        alpha = str(params.get("alpha") or "").strip()
        sec = self._resolve_sec([alpha], params.get("sec"), dialect="qlib")
        extra = set(sec["bindings"]) if sec else set()
        mode = params.get("strategy_mode", "long-only")
        try:
            check_alpha_syntax(alpha, extra_features=extra) if extra else check_alpha_syntax(alpha)
            if mode not in ("long-only", "signed"):
                raise ValueError("Choose Signed alpha weights or Long-only Top-k.")
            cash_rate = float(params.get("cash_rate") or 0)
            if not math.isfinite(cash_rate) or cash_rate <= -1:
                raise ValueError("Cash rate must be a finite annual decimal above -1.")
            options = {key: "" if params.get(key) is None else str(params[key]) for key in (
                "universe", "start", "end", "warmup_sessions", "cost_bps", "topk", "n_drop", "borrow_rate")}
            settings = parse_backtest_options(options, mode)
            request: dict[str, Any] = {"kind": "backtest", "alpha": alpha, "annual_cash_rate": cash_rate, **settings}
            if mode == "signed":
                raw_gross = params.get("gross_exposure")
                gross = float(1 if raw_gross in (None, "") else raw_gross)
                if not math.isfinite(gross) or not 0 <= gross <= 1:
                    raise ValueError("Gross exposure must be a finite decimal from 0 to 1.")
                transform = str(params.get("signal_transform") or "raw")
                if transform not in ("raw", "rank", "zscore"):
                    raise ValueError("Weighting must be raw, rank or zscore.")
                request.update(strategy_mode="signed", gross_exposure=gross, signal_transform=transform,
                               annual_borrow_rate=request.pop("annual_borrow_rate", 0.004),
                               runtime_profile="auto")
        except (ValueError, TypeError, SyntaxError, RuntimeError) as exc:
            raise ServiceError(str(exc)) from exc
        resolved, summary = self._price_dataset(params.get("dataset"))
        missing = set(re.findall(r"\$([A-Za-z][A-Za-z0-9_]*)", alpha)) - set(summary["fields"]) - extra
        if missing:
            raise ServiceError("Selected price dataset has no field: " + ", ".join(sorted(missing)))
        request["dataset"] = str(resolved)
        if sec:
            request["sec"] = sec
        notes = snap_dates(Path(resolved), request)
        return request, notes, Path(resolved)

    def _start_backtest_batch(self, params: dict[str, Any], lines: list[tuple[int, str]]) -> dict[str, Any]:
        """Independent alphas in one worker with verified preparation reuse."""
        from qlib_yahoo.expressions import check_alpha_syntax

        if len(lines) > 200:
            raise ServiceError("At most 200 alphas per batch.")
        sec = self._resolve_sec([text for _, text in lines], params.get("sec"), dialect="qlib")
        extra = set(sec["bindings"]) if sec else set()
        for number, text in lines:
            try:
                check_alpha_syntax(text, extra_features=extra) if extra else check_alpha_syntax(text)
            except (ValueError, SyntaxError, RuntimeError) as exc:
                raise ServiceError(f"Line {number}: {exc}") from exc
        # Validate the shared settings once, through the single-backtest path.
        template, notes, dataset = self._backtest_request({**params, "alpha": lines[0][1]})
        if sec:
            template["sec"] = sec
        alphas = [text for _, text in lines]
        total = len(alphas)
        label = f"{total} {'signed' if template.get('strategy_mode') == 'signed' else 'long-only'} backtests on {dataset.name}"

        def work(record: JobRecord) -> str:
            for note in notes:
                record.add(note)
            return self._run_worker(record, {**template, "kind": "backtest-batch", "alphas": alphas,
                                             "data_root": str(self.data_root)})

        return self._start("backtest-batch", label, {"alphas": alphas, **template}, work)

    def backtest_batch(self, path: str) -> dict[str, Any]:
        batch = self._inside(path, "workbench")
        try:
            record = json.loads(batch.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ServiceError(f"Cannot read the batch: {exc}") from exc
        return _json_safe(record)

    def start_import(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            request = parse_import_options(
                str(params.get("table") or ""), str(params.get("start") or ""), str(params.get("end") or ""),
                str(params.get("size") or "3000"), adv=str(params.get("adv") or "20"),
                history=str(params.get("history") or "252"), coverage=str(params.get("coverage") or "0.5"),
                workers=str(params.get("workers") or "4"), name=str(params.get("name") or ""),
                min_price=str(params.get("min_price") or "0"), groups=str(params.get("groups") or ""))
        except ValueError as exc:
            raise ServiceError(str(exc)) from exc
        request.update(data_root=str(self.data_root), make_latest=bool(params.get("make_default", True)))
        label = f"Import {Path(request['table']).name} → {request['rule']['name']}"
        if request["rule"]["min_price"]:
            label += f" (min ${request['rule']['min_price']:g})"
        return self._start("import", label, request, lambda record: self._run_worker(record, request))

    # ------------------------------------------------------------ Alpha Lab
    def _alpha_library(self):
        from qlib_yahoo.alphalab.batch import Library

        return Library(self.data_root / "alphalab" / "library.sqlite")

    def _group_preview(self, dataset: Path, group_overlay: str) -> dict[str, Any]:
        """Only sealed headers and price binding; full event verification is worker-owned."""
        try:
            from qlib_yahoo.classification import classification_preview

            path = self._inside(group_overlay, "classifications")
            return classification_preview(dataset, path)
        except (ImportError, OSError, ValueError, DatasetError) as exc:
            raise ServiceError(f"Classification source unavailable: {exc}. Select an overlay bound to the selected "
                               "Trading prices, restore the source, or choose Embedded groups / no overlay. "
                               "MARKET/NONE price-only alphas remain usable.") from exc

    @staticmethod
    def _group_path(value) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ServiceError("group_overlay must be a classification folder path string or empty.")
        return value.strip()

    def alpha_groups(self, dataset: str | None, group_overlay=None) -> dict[str, Any]:
        """Discover direct children only, with compatible metadata previews and concise failures."""
        resolved, _ = self._price_dataset(dataset)
        selected_path = self._group_path(group_overlay)
        embedded = sorted(self._lab_groups(resolved))
        root = self.data_root / "classifications"
        overlays, diagnostics, selected = [], [], None
        try:
            candidates = sorted(root.iterdir()) if root.is_dir() else []
        except OSError as exc:
            candidates = []
            diagnostics.append({"path": str(root), "message": str(exc)[:400]})
        for candidate in candidates:
            if not candidate.is_dir() or not (candidate / "manifest.json").is_file():
                continue
            try:
                preview = self._group_preview(resolved, str(candidate))
                overlays.append(preview)
            except ServiceError as exc:
                diagnostics.append({"path": str(candidate), "message": str(exc)[:500]})
        error = None
        if selected_path:
            try:
                selected = self._group_preview(resolved, selected_path)
            except ServiceError as exc:
                error = str(exc)
        groups = sorted({"market", *selected["groups"]}) if selected else (embedded if not selected_path else ["market"])
        return _json_safe({"dataset": str(resolved), "embedded_groups": embedded, "overlays": overlays,
                           "selected": selected, "groups": groups, "error": error,
                           "diagnostics": diagnostics, "status": "metadata_only"})

    def alpha_check(self, text: str, dataset: str | None = None, sec=None, group_overlay=None) -> dict[str, Any]:
        from qlib_yahoo.alphalab.engine import check_expression
        from qlib_yahoo.alphalab.fastexpr import Binary, Call, Str, Ternary, Unary, names, parse

        overlay_path = self._group_path(group_overlay)
        overlay_checked = False
        overlay_error = None
        group_consumers = set()
        available = None
        try:
            config = self._resolve_sec([line for _, line in _alpha_lines(str(text))], sec, dialect="lab")
        except ServiceError as exc:
            return {"ok": False, "lines": [{"line": number, "ok": False, "message": str(exc)}
                                           for number, _ in _alpha_lines(str(text))], "group_consumers": []}
        extra = set(config["bindings"]) if config else set()
        groups = {"market"}
        if dataset is not None:
            resolved, summary = self._price_dataset(dataset)
            available = set(summary["fields"]) | {"returns", "vwap", "dollar_volume"} | extra
            # Group declarations are small header metadata. Do not hash or read
            # the bar/event/group files while the user types.
            groups = self._lab_groups(resolved)
        lines = []
        for number, line in enumerate(str(text).splitlines(), 1):
            if line.strip() and not line.strip().startswith("#"):
                problems = check_expression(line, fields=extra) if extra else check_expression(line)
                problems = [_sec_alias_guidance(problem) for problem in problems]
                problems = ["subindustry is unsupported; this source does not provide a subindustry taxonomy"
                            if problem == "unknown field 'subindustry'" else problem for problem in problems]
                if not problems and available is not None:
                    program = parse(line)
                    assigned = {"true", "false"}
                    referenced = set()
                    for binding, value in program.assignments:
                        referenced.update(names(value) - assigned)
                        assigned.add(binding)
                    referenced.update(names(program.result) - assigned)
                    group_consumers.update(referenced & {"sector", "industry", "country"})

                    # A selected source is irrelevant to price-only expressions. In a
                    # mixed batch validate once, only after finding a group consumer.
                    if overlay_path and referenced & {"sector", "industry", "country"} and not overlay_checked:
                        overlay_checked = True
                        try:
                            groups = {"market", *self._group_preview(resolved, overlay_path)["groups"]}
                        except ServiceError as exc:
                            groups = {"market"}
                            overlay_error = str(exc)

                    def check_groups(node):
                        if isinstance(node, Call):
                            if node.name.startswith("group_") and len(node.args) >= 2 and isinstance(node.args[1], Str):
                                problems.append("the group must be a bare group name or bucket(...), not quoted text")
                            for child in (*node.args, *(value for _, value in node.kwargs)):
                                check_groups(child)
                        elif isinstance(node, Unary):
                            check_groups(node.operand)
                        elif isinstance(node, Binary):
                            check_groups(node.left)
                            check_groups(node.right)
                        elif isinstance(node, Ternary):
                            for child in (node.condition, node.then, node.otherwise):
                                check_groups(child)
                    for _, node in program.assignments:
                        check_groups(node)
                    check_groups(program.result)
                    for name in sorted(referenced):
                        if name in ("market", "sector", "industry", "country", "subindustry"):
                            if name == "subindustry":
                                problems.append("subindustry is unsupported; this source does not provide a subindustry taxonomy")
                            elif name not in groups:
                                problems.append(overlay_error or f"group {name!r} unavailable; select a compatible classification source or import ordinary prices with --groups")
                        elif name not in available and not re.fullmatch(r"adv\d{1,3}", name):
                            problems.append(f"selected dataset has no field {name!r}")
                lines.append({"line": number, "ok": not problems, "message": "; ".join(problems)})
        return {"ok": all(item["ok"] for item in lines) and bool(lines), "lines": lines,
                "group_consumers": sorted(group_consumers), **self._sec_preview(config)}

    @record_console_submission("console-lab")
    def start_alpha(self, params: dict[str, Any]) -> dict[str, Any]:
        from dataclasses import asdict

        from qlib_yahoo.alphalab.settings import Settings

        reject_recovery_options(params)
        if str(params.get("neutralization") or "").upper() == "SUBINDUSTRY":
            raise ServiceError("Subindustry neutralization is unsupported; no subindustry taxonomy is supplied.")
        expressions = [line.strip() for line in str(params.get("expressions") or "").splitlines()
                       if line.strip() and not line.strip().startswith("#")]
        if not expressions:
            raise ServiceError("Write at least one alpha, one per line.")
        if len(expressions) > 5000:
            raise ServiceError("At most 5,000 alphas per batch.")
        try:
            def number(key, default, kind=float):
                value = params.get(key)
                if isinstance(value, bool):
                    raise ValueError(f"{key} must be a number, not true/false.")
                return default if value in (None, "") else kind(value)

            if type(params.get("pasteurization", True)) is not bool:
                raise ValueError("Pasteurization must be true or false.")
            settings = Settings(
                universe=str(params.get("universe") or "TOP3000"), delay=number("delay", 1, int),
                decay=number("decay", 0, int), neutralization=str(params.get("neutralization") or "MARKET"),
                truncation=number("truncation", 0.08), pasteurization=bool(params.get("pasteurization", True)),
                cost_bps=number("cost_bps", 0.0), start=params.get("start") or None, end=params.get("end") or None,
                warmup=number("warmup", 252, int), oos_start=params.get("oos_start") or None).validate()
            workers = number("workers", 4, int)
            if not 1 <= workers <= 64:
                raise ValueError("Workers must be from 1 to 64.")
        except (ValueError, TypeError, DatasetError) as exc:
            raise ServiceError(str(exc)) from exc
        resolved, _summary = self._price_dataset(params.get("dataset"))
        overlay_path = self._group_path(params.get("group_overlay"))
        neutralization = settings.neutralization.lower()
        if neutralization not in ("market", "none"):
            groups = self._group_preview(resolved, overlay_path)["groups"] if overlay_path else self._lab_groups(resolved)
            if neutralization not in groups:
                raise ServiceError(f"{settings.neutralization} neutralization needs groups; select a compatible classification source or import ordinary prices with --groups.")
        sec = self._resolve_sec(expressions, params.get("sec"), dialect="lab")
        checked = self.alpha_check("\n".join(expressions), str(resolved), sec=sec, group_overlay=overlay_path)
        if not checked["ok"]:
            bad = next(item for item in checked["lines"] if not item["ok"])
            raise ServiceError(f"Line {bad['line']}: {bad['message']}")
        request = {"kind": "alpha-batch", "expressions": expressions, "settings": asdict(settings),
                   "dataset": str(resolved), "data_root": str(self.data_root), "workers": workers,
                   "force": bool(params.get("force")), "group_overlay": overlay_path}
        if sec:
            request["sec"] = sec
        label = f"Alpha Lab: {len(expressions)} alpha{'s' if len(expressions) > 1 else ''} on {Path(resolved).name}"
        return self._start("alpha", label, request, lambda record: self._run_worker(record, request))

    def alpha_library(self, sort: str = "created_at", limit: int = 1000) -> list[dict[str, Any]]:
        notes = Annotations(self.data_root).load()["alphas"]
        rows = []
        for row in self._alpha_library().rows(sort=sort, limit=int(limit)):
            entry = notes.get(row["id"]) or {}
            if entry.get("deleted_at"):
                continue
            row["settings"] = json.loads(row["settings"] or "{}")
            row["starred"] = bool(entry.get("starred"))
            row["label"] = entry.get("label", "")
            rows.append(row)
        return _json_safe(rows)

    def alpha_result(self, identifier: str) -> dict[str, Any]:
        result = self._alpha_library().get(str(identifier), include_errors=True)
        if result is None:
            raise ServiceError("Unknown alpha.")
        daily = result.pop("daily", None) or {}
        pnl = daily.get("pnl") or []
        if pnl:
            cumulative, peak, drawdown = [], 0.0, []
            total = 0.0
            for value in pnl:
                total += value or 0.0
                cumulative.append(total)
                peak = max(peak, total)
                drawdown.append(total - peak)
            step = max(1, len(pnl) // MAX_POINTS)
            result["series"] = {"dates": daily["dates"][::step], "cumulative": cumulative[::step],
                                "drawdown": drawdown[::step]}
        return _json_safe(result)

    def workflow_attempts(self, limit=100):
        from qlib_yahoo.workflow_records import EvidenceStore

        if type(limit) is not int or not 1 <= limit <= 500:
            raise ServiceError("Choose between 1 and 500 recent attempts.")
        store = EvidenceStore(self.data_root)
        records = {}

        def metadata(identity):
            if identity not in records:
                records[identity] = store.get(identity, verify_result=False)
            return records[identity]

        def label(row):
            record = metadata(row["id"])
            request = record["request"]
            params = request.get("parameters", request)
            if isinstance(params, dict):
                for key in ("expression", "alpha"):
                    if isinstance(params.get(key), str) and params[key].strip():
                        return params[key][:160]
                if isinstance(params.get("account_id"), str):
                    return "Portfolio: " + params["account_id"][:120]
            if row["kind"] == "candidate-validation":
                for parent in record["parents"]:
                    source = metadata(parent)
                    if source["kind"] == "lab-evaluation":
                        return str(source["request"].get("expression") or "Saved Lab candidate")[:160]
            expressions = request.get("expressions")
            if isinstance(expressions, list):
                return f"{len(expressions)} submitted expressions"
            return row["kind"].replace("-", " ")

        return [{**row, "display_label": label(row),
                 "integrity_scope": "metadata_only_open_to_verify_result"}
                for row in store.list_attempts(limit=limit, verify_results=False)]

    def workflow_detail(self, attempt_id):
        from qlib_yahoo.workflow_records import EvidenceStore

        store = EvidenceStore(self.data_root)
        attempt = store.get(attempt_id)
        result = store.get_result(attempt["result_digest"]) if attempt["result_digest"] else None
        if isinstance(result, dict):
            result = {key: value for key, value in result.items()
                      if key not in {"daily", "yearly", "weights"}}
        return _json_safe({"attempt": attempt, "result": result})

    def workflow_recover(self):
        from qlib_yahoo.workflow_recovery import recover
        return recover(self.data_root)

    def workflow_preview(self, params):
        from qlib_yahoo.validation_workflow import preview_validation

        preview = preview_validation(self.data_root, params)
        self._inside(preview["dataset"], "datasets")
        return _json_safe(preview)

    @record_console_submission("console-candidate-validation")
    def start_candidate_validation(self, params):
        preview = self.workflow_preview(params)
        request = {"kind": "candidate-validation", "data_root": str(self.data_root),
                   "parameters": params}
        label = f"Validate saved Lab candidate: {preview['expression'][:70]}"
        return self._start("candidate-validation", label, request,
                           lambda record: self._run_worker(record, request))

    def workflow_portfolio_preview(self, params):
        from qlib_yahoo.portfolio_workflow import preview_portfolio

        preview = preview_portfolio(self.data_root, params)
        self._inside(preview["dataset"], "datasets")
        if preview["request"]["group_overlay"] is not None:
            self._inside(preview["request"]["group_overlay"], "classifications")
        return _json_safe(preview)

    @record_console_submission("console-portfolio-validation")
    def start_portfolio_validation(self, params):
        preview = self.workflow_portfolio_preview(params)
        request = {"kind": "portfolio-validation", "data_root": str(self.data_root),
                   "parameters": preview["request"]}
        return self._start("portfolio-validation", f"Portfolio: {preview['policy']['account_id']}", request,
                           lambda record: self._run_worker(record, request))

    def workflow_chronology_preview(self, params):
        from qlib_yahoo.chronology_workflow import preview_chronology
        return preview_chronology(self.data_root, params)

    @record_console_submission("console-chronology-declare")
    def workflow_chronology_declare(self, params):
        from qlib_yahoo.chronology_workflow import declare_chronology
        result = declare_chronology(self.data_root, params)
        return {**result, "submission_attempt_id": current_submission()["id"]}

    @record_console_submission("console-chronology-freeze")
    def workflow_chronology_freeze(self, params):
        from qlib_yahoo.chronology_workflow import freeze_chronology_selection
        result = freeze_chronology_selection(self.data_root, params)
        return {**result, "submission_attempt_id": current_submission()["id"]}

    def workflow_chronology_view(self, params):
        from qlib_yahoo.chronology_workflow import view_chronology
        return view_chronology(self.data_root, params)

    @record_console_submission("console-chronology-trial")
    def start_chronology_trial(self, params):
        return self._start_chronology("chronology-trial", params)

    @record_console_submission("console-chronology-final")
    def start_chronology_final(self, params):
        return self._start_chronology("chronology-final", params)

    def _start_chronology(self, kind, params):
        from qlib_yahoo.workflow_records import canonical_json_bytes
        canonical_json_bytes(params)
        if type(params) is not dict:
            raise ServiceError("chronology parameters must be a JSON object")
        request = {"kind": kind, "data_root": str(self.data_root), "parameters": params}
        return self._start(kind, "Chronology: " + kind.split("-")[1], request,
                           lambda record: self._run_worker(record, request))

    def alpha_batch(self, path: str) -> dict[str, Any]:
        from qlib_yahoo.alphalab.batch import correlation

        batch = self._inside(path, "alphalab")
        try:
            record = json.loads(batch.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ServiceError(f"Cannot read the batch: {exc}") from exc
        library = None
        evidence = record.get("evidence")
        if evidence is not None and (not isinstance(evidence, list) or len(evidence) != len(record["ids"])):
            raise ServiceError("Saved batch evidence does not match its result inventory.")
        if evidence is not None:
            from qlib_yahoo.workflow_records import EvidenceStore
            store = EvidenceStore(self.data_root)
        rows, series = [], {}
        for index, identifier in enumerate(record["ids"]):
            link = evidence[index] if evidence is not None else None
            if link and link.get("attempt_id") and link.get("result_digest"):
                attempt = store.get(link["attempt_id"])
                if attempt["result_digest"] != link["result_digest"]:
                    raise ServiceError("Saved batch attempt/result identity differs.")
                result = store.get_result(link["result_digest"])
                if result.get("id") != identifier:
                    raise ServiceError("Saved batch candidate identity differs.")
            else:
                # Old batches did not retain immutable versions; never invent them.
                library = library or self._alpha_library()
                result = library.get(identifier, include_errors=True) or {"expression": "", "error": "not in the library"}
            row = {"id": identifier, "expression": result.get("expression"), "error": result.get("error"),
                   "engine": result.get("engine"),
                   "summary": result.get("summary"), "checks": result.get("checks"),
                   "self_correlation": result.get("self_correlation"),
                   "attempt_id": link.get("attempt_id") if link else None,
                   "result_digest": link.get("result_digest") if link else None,
                   "evidence_status": "immutable" if link and link.get("result_digest") else "legacy_mutable"}
            rows.append(row)
            if "daily" in result and "error" not in result:
                series[identifier] = (result["daily"]["dates"], result["daily"]["pnl"])
        ids = list(series)
        pairs = []
        for i in range(len(ids)):
            for k in range(i + 1, len(ids)):
                value = correlation(series[ids[i]], series[ids[k]])
                if value is not None:
                    pairs.append({"a": ids[i], "b": ids[k], "correlation": value})
        pairs.sort(key=lambda p: -p["correlation"])
        return _json_safe({"path": batch, "settings": record["settings"], "created": record["created_at_utc"],
                           "rows": rows, "top_pairs": pairs[:20], "diagnostics": record.get("diagnostics")})

    def latest_sources(self) -> dict[str, Any]:
        """The newest Yahoo table and Nasdaq screener under <data root>/sources, if any."""
        sources = self.data_root / "sources"
        tables = sorted(sources.glob("yahoo-table-*/table.parquet")) if sources.is_dir() else []
        screeners = sorted(p for p in sources.glob("nasdaq-screener-*") if (p / "manifest.json").is_file()) if sources.is_dir() else []
        return _json_safe({"table": tables[-1] if tables else None, "screener": screeners[-1] if screeners else None})

    def start_refresh(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            min_price = float(params.get("min_price") or 1.0)
            if not math.isfinite(min_price) or not 0 <= min_price <= 1000:
                raise ValueError("Minimum price must be from 0 to 1000.")
        except (TypeError, ValueError) as exc:
            raise ServiceError(str(exc)) from exc
        latest = self.latest_sources()
        request = {"kind": "refresh-data", "data_root": str(self.data_root), "min_price": min_price,
                   "make_latest": bool(params.get("make_default", True)), "workers": 6,
                   "previous_tables": [latest["table"]] if latest["table"] else []}
        label = "Refresh market data from Yahoo (about an hour)"
        return self._start("refresh", label, request, lambda record: self._run_worker(record, request))

    def start_report(self, run: str, profile: str) -> dict[str, Any]:
        if profile not in ("holdings", "overview", "detailed"):
            raise ServiceError("Unknown report profile.")
        run_dir = self._inside(run, "runs")
        self._require_supported_run(run_dir)

        def work(record: JobRecord) -> str:
            from qlib_yahoo.report_service import ensure_report
            index, reused = ensure_report(run_dir, profile)
            record.add("Reused the existing report." if reused else "Rendered a new report.")
            return str(index)

        return self._start("report", f"{profile.title()} report for {run_dir.name}", {"run": str(run_dir)}, work)

    # ------------------------------------------------------------ organizing runs and alphas

    def annotate(self, params: dict[str, Any]) -> dict[str, Any]:
        """Star, label or annotate a run (by folder name), a factor (by key) or a library alpha (by id)."""
        kind = str(params.get("kind") or "")
        key = str(params.get("key") or "")
        fields = {name: params[name] for name in ("starred", "label", "note") if name in params}
        if kind == "runs" and not ((self.data_root / "runs" / key).is_dir() or (self.data_root / "trash" / "runs" / key).is_dir()):
            raise ServiceError("Unknown run.")
        try:
            entry = Annotations(self.data_root).update(kind, key, fields)
        except ValueError as exc:
            raise ServiceError(str(exc)) from exc
        return _json_safe({"kind": kind, "key": key, "annotation": entry})

    def _run_names(self, paths: Any) -> list[str]:
        if not isinstance(paths, list) or not paths or len(paths) > 1000:
            raise ServiceError("Choose one or more runs.")
        runs_root = (self.data_root / "runs").resolve()
        names = []
        for path in paths:
            run = self._inside(str(path), "runs")
            if run.parent != runs_root:
                raise ServiceError("Only saved run folders can be moved to the trash.")
            names.append(run.name)
        return names

    def trash_runs(self, paths: Any) -> dict[str, Any]:
        from qlib_yahoo.workbench_store import trash_runs

        names = self._run_names(paths)
        try:
            moved = trash_runs(self.data_root, names, busy=self.busy())
        except (OSError, ValueError) as exc:
            raise ServiceError(str(exc)) from exc
        return {"trashed": moved}

    def trash(self) -> dict[str, Any]:
        from qlib_yahoo.workbench_store import list_trash

        notes = Annotations(self.data_root).load()
        runs = list_trash(self.data_root)
        for row in runs:
            _annotate(row, notes)
        alphas = []
        deleted = {key for key, entry in notes["alphas"].items() if entry.get("deleted_at")}
        if deleted:
            for row in self._alpha_library().rows(sort="created_at", limit=100000):
                if row["id"] in deleted:
                    row["settings"] = json.loads(row["settings"] or "{}")
                    row["deleted_at"] = notes["alphas"][row["id"]]["deleted_at"]
                    alphas.append(row)
        return _json_safe({"runs": runs, "alphas": alphas})

    def restore_runs(self, names: Any) -> dict[str, Any]:
        from qlib_yahoo.workbench_store import restore_runs

        if not isinstance(names, list) or not names:
            raise ServiceError("Choose one or more runs to restore.")
        try:
            return {"restored": restore_runs(self.data_root, [str(name) for name in names])}
        except (OSError, ValueError) as exc:
            raise ServiceError(str(exc)) from exc

    def empty_trash(self, names: Any) -> dict[str, Any]:
        from qlib_yahoo.workbench_store import empty_trash

        if names is not None and not isinstance(names, list):
            raise ServiceError("Choose the runs to remove from the trash.")
        try:
            return {"emptied": empty_trash(self.data_root, None if names is None else [str(name) for name in names])}
        except (OSError, ValueError) as exc:
            raise ServiceError(f"Could not move to the system trash: {exc}") from exc

    def alpha_delete(self, ids: Any, deleted: bool = True) -> dict[str, Any]:
        """Hide (or restore) library alphas; the library itself is never rewritten."""
        if not isinstance(ids, list) or not ids or len(ids) > 10000:
            raise ServiceError("Choose one or more alphas.")
        library = self._alpha_library()
        notes = Annotations(self.data_root)
        changed = []
        for identifier in dict.fromkeys(str(value) for value in ids):
            if library.get(identifier, include_errors=True) is None:
                raise ServiceError("Unknown alpha.")
            try:
                notes.update("alphas", identifier, {"deleted": bool(deleted)})
            except ValueError as exc:
                raise ServiceError(str(exc)) from exc
            changed.append(identifier)
        return {"changed": changed, "deleted": bool(deleted)}

    def run_signal(self, path: str) -> dict[str, Any]:
        """IC, rank IC and quantile returns from the run's saved forward labels (cached beside the data)."""
        from qlib_yahoo.signal_analysis import analyze

        run = self._inside(path, "runs")
        self._require_supported_run(run)
        labels = run / "signal_forward_labels.parquet"
        if not labels.is_file():
            raise ServiceError("This run has no saved forward labels (runs before September 2026 or a failed extraction).")
        stat = labels.stat()
        cache = self.data_root / "workbench" / "cache" / f"signal-{run.name}-{stat.st_size}-{stat.st_mtime_ns}.json"
        if cache.is_file():
            try:
                return json.loads(cache.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        try:
            result = _json_safe(analyze(labels))
        except (OSError, ValueError) as exc:
            raise ServiceError(f"Cannot analyse the signal: {exc}") from exc
        cache.parent.mkdir(parents=True, exist_ok=True)
        for old in cache.parent.glob(f"signal-{run.name}-*.json"):
            old.unlink(missing_ok=True)  # an older cache of this run (the labels were rewritten)
        temporary = cache.with_name(cache.name + ".tmp")
        temporary.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
        temporary.replace(cache)
        return result

    def cancel(self) -> bool:
        record = self.current
        if record is None or record.state != "running" or record.cancel_hook is None:
            return False
        record.add("Cancel requested.")
        return bool(record.cancel_hook())

    # ------------------------------------------------------------------ actions

    def make_default(self, path: str) -> dict[str, Any]:
        try:
            set_default_dataset(self.data_root, self._inside(path, "datasets"))
        except (OSError, ValueError) as exc:
            raise ServiceError(str(exc)) from exc
        return {"ok": True}

    def reveal(self, path: str) -> dict[str, Any]:
        from qlib_yahoo.desktop import open_path, reveal_path

        target = self.data_root if path in ("", ".", None) else self._inside(path)
        try:
            (open_path if target == self.data_root else reveal_path)(target)
        except Exception as exc:
            raise ServiceError(f"Could not show {target}: {exc}") from exc
        return {"ok": True}

    def open_external(self, path: str) -> dict[str, Any]:
        from qlib_yahoo.desktop import open_path

        target = self._inside(path, "runs")
        try:
            open_path(target)
        except Exception as exc:
            raise ServiceError(f"Could not open {target.name}: {exc}") from exc
        return {"ok": True}

    def settings(self) -> dict[str, Any]:
        if self.settings_path is None:
            return {}
        from qlib_yahoo.desktop import load_settings

        return load_settings(self.settings_path)

    def save_settings(self, values: dict[str, Any]) -> dict[str, Any]:
        if self.settings_path is not None and isinstance(values, dict):
            from qlib_yahoo.desktop import save_settings

            clean = {str(k): v for k, v in values.items() if isinstance(v, (str, int, float, bool)) or v is None}
            # This file writer replaces its input; preserve unrelated preferences.
            merged = self.settings()
            merged.update(clean)
            save_settings(self.settings_path, merged)
        return {"ok": True}


def _alpha_lines(text: str) -> list[tuple[int, str]]:
    """(line number, expression) for each non-empty line that is not a # comment."""
    return [(number, line.strip()) for number, line in enumerate(str(text or "").splitlines(), 1)
            if line.strip() and not line.strip().startswith("#")]


def _annotate(row: dict[str, Any], notes: dict[str, dict[str, dict[str, Any]]]) -> None:
    """Add a run's and its factor's stars and labels to a run row or detail."""
    run = notes["runs"].get(str(row.get("name") or "")) or {}
    factor = notes["factors"].get(str(row.get("factor") or "")) or {}
    row.update(starred=bool(run.get("starred")), label=run.get("label", ""), note=run.get("note", ""),
               factor_starred=bool(factor.get("starred")), factor_label=factor.get("label", ""))


def _lab_examples() -> str:
    """Generic teaching examples shipped with the framework package."""
    path = Path(__file__).resolve().parents[1] / "examples" / "price_volume.txt"
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def snap_dates(dataset: Path, request: dict[str, Any]) -> list[str]:
    """Move start/end onto dataset sessions (start forward, end backward) and explain any change."""
    import bisect

    calendar = dataset / "qlib_data" / "calendars" / "day.txt"
    if not calendar.is_file() or not ({"start", "end"} & request.keys()):
        return []
    days = calendar.read_text(encoding="utf-8").split()
    notes = []
    if "start" in request:
        index = bisect.bisect_left(days, request["start"])
        if index >= len(days) - 1:
            raise ServiceError("Start date is after the dataset's last usable session.")
        if days[index] != request["start"]:
            notes.append(f"Start {request['start']} is not a session; using {days[index]}.")
            request["start"] = days[index]
    if "end" in request:
        index = min(bisect.bisect_right(days, request["end"]) - 1, len(days) - 2)
        if index < 0:
            raise ServiceError("End date is before the dataset's first session.")
        if days[index] != request["end"]:
            notes.append(f"End {request['end']} moved to session {days[index]} (the runner needs one later session).")
            request["end"] = days[index]
    if "start" in request and "end" in request and request["start"] >= request["end"]:
        raise ServiceError("After aligning to sessions, the start is not before the end.")
    return notes


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
