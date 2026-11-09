"""Test many alphas at once: shared memory-mapped data, worker processes and a local alpha library.

The verified snapshot is loaded once and written to immutable ``.npy`` generations
keyed by all consumed source evidence and array contents. Workers share the maps;
their count is capped using a conservative RAM estimate, not a hard per-alpha
memory guarantee. Each finished alpha goes to a SQLite library under the data
root with its settings, metrics and daily PnL. The same expression with the same
settings on the same data is not simulated again. Correlations are measured
between the batch's alphas and against the library ("self-correlation").
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import closing, contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from qlib_yahoo.alphalab.simulate import ENGINE_VERSION, Settings, Workspace, json_safe, simulate
from qlib_yahoo.dataset import DatasetError
from qlib_yahoo.path_utils import cache_io_path
from qlib_yahoo.alphalab.panel import digest, source_identity

CACHE_VERSION = 2
SELF_CORRELATION_LIMIT = 0.7


# ------------------------------------------------------------------ data cache
def cache_dir(data_root: Path, workspace: Workspace) -> Path:
    return Path(data_root) / "cache" / "alphalab" / f"{digest(workspace.identity)}-v{CACHE_VERSION}"


def _file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _array_hash(array: np.ndarray) -> str:
    # Hash rows separately to avoid allocating a full contiguous matrix copy.
    hasher = hashlib.sha256()
    values = np.asarray(array)
    if not values.size:
        return hasher.hexdigest()
    for row in values.reshape((len(values), -1)):
        hasher.update(np.ascontiguousarray(row).tobytes())
    return hasher.hexdigest()


def _close_sec_arrays(arrays: dict) -> None:
    """Release runner-owned SEC maps after copying into the Lab cache."""
    for values in arrays.values():
        if isinstance(values, np.memmap):
            values._mmap.close()


def _cache_description(workspace: Workspace, universes: Iterable[str]) -> tuple[dict, dict]:
    arrays = {"dates": workspace.dates,
              **{f"field_{k}": v for k, v in workspace.fields.items()},
              **{f"group_{k}": v for k, v in workspace.groups.items()},
              **{f"universe_{k}": workspace.universe(k) for k in sorted({u.upper() for u in universes})}}
    if any(not re.fullmatch(r"[A-Za-z0-9_]+", name) for name in arrays):
        raise DatasetError("invalid Alpha Lab cache array name")
    meta = {"version": CACHE_VERSION, "identity": workspace.identity, "symbols": workspace.symbols,
            "fields": sorted(workspace.fields), "groups": sorted(workspace.groups),
            "universes": sorted(k.removeprefix("universe_") for k in arrays if k.startswith("universe_")),
            "manifest": workspace.panel.manifest,
            "arrays": {k: {"shape": list(v.shape), "dtype": v.dtype.str, "data_sha256": _array_hash(v)}
                       for k, v in arrays.items()}}
    meta["content_sha256"] = digest(meta)
    return meta, arrays


def write_cache(workspace: Workspace, directory: Path, universes: Iterable[str]) -> Path:
    """Write the workspace's arrays once (atomic by folder rename)."""
    directory = Path(directory)
    meta, arrays = _cache_description(workspace, universes)
    legacy = directory.with_name(directory.name + "-" + meta["content_sha256"])
    # The full content digest includes source identity and requested masks.
    # Avoid repeating the source digest inside Research's already nested cache.
    directory = directory.with_name(f"v{CACHE_VERSION}-" + meta["content_sha256"])
    io_directory = cache_io_path(directory)
    io_legacy = cache_io_path(legacy)
    if io_directory.exists():
        with closing(CachedWorkspace(directory, expected=meta)):
            return directory
    try:
        legacy_exists = io_legacy.exists()
    except OSError:
        # An overlong legacy path may be inaccessible on Windows.
        legacy_exists = False
    if legacy_exists:
        with closing(CachedWorkspace(legacy, expected=meta)):
            return legacy
    io_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".tmp-", dir=io_directory.parent))
    try:
        for name, values in arrays.items():
            np.save(staging / f"{name}.npy", values, allow_pickle=False)
        meta["files"] = {name + ".npy": _file_hash(staging / f"{name}.npy") for name in arrays}
        (staging / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        for path in staging.iterdir():
            with path.open("r+b") as handle:
                os.fsync(handle.fileno())
        with closing(CachedWorkspace(staging, expected=meta)):
            pass
        try:
            os.replace(staging, io_directory)
        except OSError:
            if not io_directory.exists():
                raise
            with closing(CachedWorkspace(directory, expected=meta)):
                pass
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return directory


class CachedWorkspace(Workspace):
    """A Workspace backed by read-only memory maps of a cache folder."""

    def __init__(self, directory: Path, *, expected: dict | None = None):
        directory = cache_io_path(Path(directory))
        if directory.is_symlink():
            raise DatasetError("Alpha Lab cache directory must not be a symlink")
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        description = {k: v for k, v in meta.items() if k not in ("files", "content_sha256")}
        if (meta.get("version") != CACHE_VERSION or digest(description) != meta.get("content_sha256")
                or (expected is not None and meta.get("content_sha256") != expected.get("content_sha256"))
                or (expected is None and not directory.name.endswith("-" + meta.get("content_sha256", "")))):
            raise DatasetError("Alpha Lab cache metadata or source identity differs")
        if len(set(meta["symbols"])) != len(meta["symbols"]):
            raise DatasetError("Alpha Lab cache has duplicate symbols")
        expected_names = {"dates", *("field_" + k for k in meta["fields"]),
                          *("group_" + k for k in meta["groups"]), *("universe_" + k for k in meta["universes"])}
        if (set(meta["arrays"]) != expected_names or set(meta["files"]) != {k + ".npy" for k in expected_names}
                or {p.name for p in directory.iterdir()} != {"meta.json", *(k + ".npy" for k in expected_names)}
                or any(not re.fullmatch(r"[A-Za-z0-9_]+", k) for k in expected_names)):
            raise DatasetError("Alpha Lab cache array inventory differs")
        def load(name, *, mapped=False):
            path = directory / (name + ".npy")
            if path.is_symlink() or _file_hash(path) != meta["files"][path.name]:
                raise DatasetError(f"Alpha Lab cache bytes differ: {name}")
            value = np.load(path, mmap_mode="r" if mapped else None, allow_pickle=False)
            try:
                record = meta["arrays"][name]
                if (list(value.shape) != record["shape"] or value.dtype.str != record["dtype"]
                        or _array_hash(value) != record["data_sha256"]):
                    raise DatasetError(f"Alpha Lab cache array differs: {name}")
            except BaseException:
                if isinstance(value, np.memmap):
                    value._mmap.close()
                raise
            return value
        self.dataset = Path(meta["identity"]["dataset"])
        self.identity = meta["identity"]
        self.symbols = meta["symbols"]
        self.dates = load("dates")
        self.fields = {}
        self.groups = {}
        self._universes = {}
        try:
            # Register each map immediately so a later load failure can release it.
            for name in meta["fields"]:
                self.fields[name] = load(f"field_{name}", mapped=True)
            for name in meta["groups"]:
                self.groups[name] = load(f"group_{name}", mapped=True)
            for name in meta["universes"]:
                self._universes[name] = load(f"universe_{name}", mapped=True)
            shape = (len(self.dates), len(self.symbols))
            if (self.dates.dtype != np.dtype("datetime64[D]") or self.dates.ndim != 1
                    or np.isnat(self.dates).any() or np.any(self.dates[1:] <= self.dates[:-1])
                    or any(v.shape != shape or v.dtype.kind != "f" for v in self.fields.values())
                    or any(v.shape != shape or v.dtype != bool for v in self._universes.values())
                    or any(v.shape not in ((shape[1],), shape) or v.dtype.kind not in "fi"
                           for v in self.groups.values())):
                raise DatasetError("Alpha Lab cache shapes, dates, groups or mask types differ")

            class _Panel:  # the pieces of a Panel that Workspace methods read
                manifest = meta["manifest"]
            self.panel = _Panel()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Release owned maps; this workspace and its array views must no longer be used."""
        for values in (*self.fields.values(), *self._universes.values(), *self.groups.values()):
            # NumPy memmap has no close method; close the mapping we own explicitly.
            values._mmap.close()
        self.fields.clear()
        self.groups.clear()
        self._universes.clear()

    def universe(self, name: str) -> np.ndarray:
        key = name.upper()
        if key not in self._universes:
            raise DatasetError(f"universe {name} is not in the shared cache")
        return self._universes[key]


# ------------------------------------------------------------------ library
SCHEMA = """
CREATE TABLE IF NOT EXISTS alphas (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    expression TEXT NOT NULL,
    settings TEXT NOT NULL,
    dataset TEXT NOT NULL,
    engine TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    sharpe REAL, fitness REAL, returns REAL, turnover REAL, drawdown REAL, margin_bps REAL,
    self_correlation REAL,
    result BLOB
);
CREATE INDEX IF NOT EXISTS alphas_dataset ON alphas(dataset, engine);
"""


def alpha_id(expression: str, settings: Settings, dataset_hash: str,
             implementation: dict | None = None) -> str:
    if implementation is None:
        from qlib_yahoo.workflow_provenance import lab_implementation
        implementation = lab_implementation()
    text = json.dumps({"expression": " ".join(expression.split()), "settings": asdict(settings.validate()),
                       "dataset": dataset_hash, "engine": ENGINE_VERSION,
                       "implementation": implementation["sha256"]}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class Library:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db, db:
            db.executescript(SCHEMA)

    def connect(self) -> sqlite3.Connection:
        """Open a caller-owned connection; its transaction context does not close it."""
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def get(self, identifier: str, *, include_errors: bool = False) -> dict[str, Any] | None:
        query = "SELECT result FROM alphas WHERE id = ?" + ("" if include_errors else " AND status = 'ok'")
        with closing(self.connect()) as db, db:
            row = db.execute(query, (identifier,)).fetchone()
        return json.loads(gzip.decompress(row["result"])) if row else None

    def save(self, identifier: str, result: dict[str, Any], dataset_hash: str) -> None:
        summary = result.get("summary") or {}
        with closing(self.connect()) as db, db:
            db.execute(
                "INSERT OR REPLACE INTO alphas VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (identifier, datetime.now(timezone.utc).isoformat(), result["expression"],
                 json.dumps(result.get("settings")), dataset_hash, ENGINE_VERSION,
                 "ok" if "error" not in result else "error", result.get("error"),
                 summary.get("sharpe"), summary.get("fitness"), summary.get("returns"), summary.get("turnover"),
                 summary.get("drawdown"), summary.get("margin_bps"), result.get("self_correlation"),
                 gzip.compress(json.dumps(json_safe(result)).encode(), mtime=0)))

    def rows(self, dataset_hash: str | None = None, sort: str = "fitness", limit: int = 50) -> list[dict[str, Any]]:
        if sort not in ("fitness", "sharpe", "returns", "turnover", "created_at", "self_correlation"):
            raise DatasetError("sort by fitness, sharpe, returns, turnover, created_at or self_correlation")
        query = ("SELECT id, created_at, expression, settings, engine, sharpe, fitness, returns, turnover, drawdown, "
                 "margin_bps, self_correlation, status, error FROM alphas")
        args: tuple[Any, ...] = ()
        if dataset_hash:
            query += " WHERE dataset = ?"
            args = (dataset_hash,)
        query += f" ORDER BY {sort} IS NULL, {sort} DESC LIMIT ?"
        with closing(self.connect()) as db, db:
            return [dict(row) for row in db.execute(query, (*args, int(limit)))]

    def pnl_series(self, dataset_hash: str, settings: Settings, exclude: set[str]) -> dict[str, tuple[list[str], list[float]]]:
        """Daily PnL of saved alphas simulated with the same universe and delay on the same data."""
        found = {}
        with closing(self.connect()) as db, db:
            for row in db.execute("SELECT id, settings, result FROM alphas WHERE dataset = ? AND engine = ? AND status = 'ok'",
                                  (dataset_hash, ENGINE_VERSION)):
                if row["id"] in exclude:
                    continue
                saved = json.loads(row["settings"])
                if saved.get("universe") != settings.universe.upper() or saved.get("delay") != settings.delay:
                    continue
                daily = json.loads(gzip.decompress(row["result"]))["daily"]
                found[row["id"]] = (daily["dates"], daily["pnl"])
        return found


def correlation(a: tuple[list[str], list[float]], b: tuple[list[str], list[float]]) -> float | None:
    """Pearson correlation of two daily PnL series on their shared dates."""
    left = dict(zip(*a))
    shared = [d for d in b[0] if d in left]
    if len(shared) < 60:
        return None
    right = dict(zip(*b))
    x = np.asarray([left[d] for d in shared], dtype=np.float64)
    y = np.asarray([right[d] for d in shared], dtype=np.float64)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


# ------------------------------------------------------------------ workers
_WORKSPACE: CachedWorkspace | None = None


def _init_worker(directory: str) -> None:
    global _WORKSPACE
    _WORKSPACE = CachedWorkspace(Path(directory))


def _available_ram() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except (ImportError, OSError):
        # Unknown machines receive a conservative single-worker budget.
        return 512 * 1024 * 1024


def worker_count(requested: int, tasks: int, workspace: Workspace, ceiling: int) -> int:
    """Allow at most half available RAM for workers' private expression temporaries."""
    cells = len(workspace.dates) * len(workspace.symbols)
    per_worker = max(256 * 1024 * 1024, cells * 8 * 24)
    memory_limit = max(1, _available_ram() // (2 * per_worker))
    return max(1, min(requested, tasks, ceiling, os.cpu_count() or 1, memory_limit))


def _run_one(expression: str, settings: dict[str, Any], workspace: CachedWorkspace | None = None) -> dict[str, Any]:
    started = time.monotonic()
    try:
        result = simulate(expression, Settings(**settings), workspace if workspace is not None else _WORKSPACE)
    except DatasetError as error:
        return {"expression": expression, "settings": settings, "error": str(error)}
    except Exception as error:  # an unexpected failure of one alpha must not stop the batch
        return {"expression": expression, "settings": settings, "error": f"{type(error).__name__}: {error}"}
    result["seconds"] = round(time.monotonic() - started, 2)
    return json_safe(result)


def _uses_classifications(expression: str, settings: Settings) -> bool:
    """Only consumed groups bring an overlay into a result's source identity."""
    if settings.neutralization in ("SECTOR", "INDUSTRY", "COUNTRY"):
        return True
    from qlib_yahoo.alphalab.fastexpr import ExpressionError, names, parse

    try:
        program = parse(expression)
    except ExpressionError:
        return False  # The evaluator returns the original syntax error.
    assigned, referenced = set(), set()
    for binding, value in program.assignments:
        referenced.update(names(value) - assigned)
        assigned.add(binding)
    referenced.update(names(program.result) - assigned)
    return bool(referenced & {"sector", "industry", "country"})


def _aligned_groups(overlay, workspace: Workspace) -> dict[str, np.ndarray]:
    """Bind by symbol, never by coincident column positions or sorting conventions."""
    names = list(overlay.symbols)
    if (len(names) != len(set(names)) or len(names) != len(workspace.symbols)
            or set(names) != set(workspace.symbols)
            or list(overlay.calendar) != [str(day) for day in workspace.dates]):
        raise DatasetError("classification calendar or symbols differ from the Alpha Lab workspace")
    positions = {name: index for index, name in enumerate(names)}
    columns = [positions[name] for name in workspace.symbols]
    same_order = names == list(workspace.symbols)
    result = {}
    for name, values in overlay.codes.items():
        if values.shape != (len(workspace.dates), len(names)):
            raise DatasetError("classification group matrix shape differs from its axes")
        result[name] = values if same_order else np.take(values, columns, axis=1)
    return result


@contextmanager
def _batch_stage(diagnostics: dict, name: str, progress):
    started = time.monotonic()
    completed = False
    try:
        yield
        completed = True
    finally:
        elapsed = max(0.0, time.monotonic() - started)
        stages = diagnostics.setdefault("stages_seconds", {})
        stages[name] = stages.get(name, 0.0) + elapsed
        if progress:
            progress(f"{'Completed' if completed else 'Failed'} Alpha Lab {name} in {elapsed:.2f}s")


def run_batch(dataset: Path, expressions: list[str], settings: Settings, *, data_root: Path, workers: int = 4,
              library: Path | None = None, force: bool = False,
              worker_ceiling: int = 4, sec: dict | None = None,
              group_overlay: str | Path | None = None,
              progress: Callable[[str], None] | None = None,
              diagnostics: dict | None = None,
              parents: list[str] | None = None) -> list[dict[str, Any]]:
    """Simulate in input order; optional diagnostics are separate from alpha identity.

    The procedure timer includes preflight and resolution, but excludes the
    caller's request parsing and save_batch. The jobs worker measures those too.
    Cache hits count input results; computations count distinct uncached alphas,
    including failed simulations but excluding syntax preflight failures.
    """
    started = time.monotonic()
    measured = diagnostics if diagnostics is not None else {}
    measured.clear()
    measured.update(scope="run_batch", cache_hit_count=0, new_computation_count=0,
                    stages_seconds={}, phase="preflight")
    completed = False
    from qlib_yahoo.workflow_records import EvidenceStore
    from qlib_yahoo.workflow_provenance import lab_implementation, request_value, frozen_lab_result

    store = EvidenceStore(data_root)
    if parents is None:
        from qlib_yahoo.workflow_submission import parent_attempts
        parents = parent_attempts()
    submitted = store.begin("lab-batch", request_value({
        "dataset": dataset, "expressions": expressions, "settings": settings,
        "workers": workers, "worker_ceiling": worker_ceiling, "force": force,
        "library": library, "sec": sec, "group_overlay": group_overlay,
    }), parents=parents)
    attempts = []
    try:
        if not isinstance(expressions, list) or not expressions:
            raise DatasetError("expressions must be a nonempty list")
        implementation = lab_implementation()
        for expression in expressions:
            candidate_id = "lab-" + digest(request_value({
                "expression": " ".join(expression.split()) if isinstance(expression, str) else expression,
                "settings": settings, "dialect": "lab-fastexpr",
            }))
            attempts.append(store.begin("lab-evaluation", request_value({
                "dataset": dataset, "expression": expression, "settings": settings,
                "sec": sec, "group_overlay": group_overlay, "force": force,
            }), parents=[submitted["id"]], candidate_id=candidate_id,
                source={"implementation": implementation}))
        valid = [index for index, expression in enumerate(expressions)
                 if isinstance(expression, str) and expression.strip()]
        computed = _run_batch(dataset, [expressions[index] for index in valid], settings,
                             data_root=data_root, workers=workers,
                             library=library, force=force, worker_ceiling=worker_ceiling, sec=sec,
                             group_overlay=group_overlay, progress=progress, diagnostics=measured,
                             implementation=implementation) if valid else []
        if lab_implementation() != implementation:
            raise DatasetError("Alpha Lab implementation changed during the submitted evaluation; rerun")
        by_index = dict(zip(valid, computed))
        results = []
        seen = set()
        for index, attempt in enumerate(attempts):
            result = by_index.get(index, {
                "expression": request_value(expressions[index]), "settings": request_value(settings),
                "error": "expression must be a nonempty string", "preflight": True,
                "id": "rejected-" + attempt["id"],
            })
            result["implementation"] = implementation
            result["candidate_id"] = attempt["candidate_id"]
            identifier = result["id"]
            reused = bool(result.get("cached")) or identifier in seen
            status = ("rejected" if result.get("preflight") else "failed") if "error" in result else (
                "reused" if reused else "completed")
            terminal = store.finish(attempt["id"], status, result=frozen_lab_result(result),
                                    error=result.get("error"))
            result.update(attempt_id=attempt["id"], result_digest=terminal["result_digest"],
                          batch_attempt_id=submitted["id"])
            results.append(result)
            seen.add(identifier)
        store.finish(submitted["id"], "completed", result={
            "attempts": [result["attempt_id"] for result in results],
            "results": [result["result_digest"] for result in results],
            "errors": sum("error" in result for result in results),
        })
        measured["attempt_id"] = submitted["id"]
        completed = True
        return results
    except BaseException as error:
        status = ("cancelled" if isinstance(error, KeyboardInterrupt) else
                  "rejected" if measured["phase"] == "preflight" else "failed")
        details = {"type": type(error).__name__, "message": str(error)}
        # Keep any already completed child immutable if persistence or a callback
        # failed later. Pending ownership also remains recoverable after hard kill.
        for attempt in [*attempts, submitted]:
            if store.get(attempt["id"])["status"] == "pending":
                store.finish(attempt["id"], status, error=details)
        raise
    finally:
        measured["elapsed_seconds"] = max(0.0, time.monotonic() - started)
        if progress:
            progress(f"Alpha Lab run_batch procedure {'completed' if completed else 'interrupted/failed'} "
                     f"in {measured['elapsed_seconds']:.2f}s (excludes caller parsing and batch save); "
                     f"{measured['cache_hit_count']} saved-result cache hits, "
                     f"{measured['new_computation_count']} new computations")


def _run_batch(dataset: Path, expressions: list[str], settings: Settings, *, data_root: Path, workers: int = 4,
              library: Path | None = None, force: bool = False,
              worker_ceiling: int = 4,
              sec: dict | None = None,
              group_overlay: str | Path | None = None,
              progress: Callable[[str], None] | None = None,
              diagnostics: dict, implementation: dict) -> list[dict[str, Any]]:
    """Simulate every expression with ``settings``; return results in input order."""
    settings = settings.validate()
    expressions = [" ".join(e.split()) for e in expressions if e and e.strip()]
    if not expressions:
        raise DatasetError("no expressions to test")
    if type(workers) is not int or not 1 <= workers <= 64:
        raise DatasetError("workers must be a whole number from 1 to 64")
    if type(worker_ceiling) is not int or not 1 <= worker_ceiling <= 64:
        raise DatasetError("worker_ceiling must be a whole number from 1 to 64")
    from qlib_yahoo.alphalab.engine import check_expression

    # Syntax and public call signatures need neither field resolution nor data.
    # Unknown fields retain the existing source/evaluator validation path.
    invalid = {}
    for index, expression in enumerate(expressions):
        problems = [problem for problem in check_expression(expression)
                    if not problem.startswith("unknown field ")]
        if problems:
            invalid[index] = {"expression": expression, "settings": asdict(settings),
                              "error": "; ".join(problems), "preflight": True,
                              "id": alpha_id(expression, settings,
                                             "preflight-unverified:" + str(Path(dataset).resolve()),
                                             implementation)}
    if invalid:
        # Preserve partial failures and input order. Only viable programs may
        # verify sources, acquire a verified cache identity or load panels.
        valid_indices = [index for index in range(len(expressions)) if index not in invalid]
        valid_results = (_run_batch(dataset, [expressions[index] for index in valid_indices], settings,
                                  data_root=data_root, workers=workers, library=library, force=force,
                                  worker_ceiling=worker_ceiling, sec=sec, group_overlay=group_overlay,
                                  progress=progress, diagnostics=diagnostics,
                                  implementation=implementation) if valid_indices else [])
        by_index = {**invalid, **dict(zip(valid_indices, valid_results))}
        return [by_index[index] for index in range(len(expressions))]
    from qlib_yahoo.sec_resolution import resolve_config

    if sec:
        with _batch_stage(diagnostics, "sec_resolution", progress):
            sec = resolve_config(expressions, sec, dialect="lab", verify=False)
    else:
        sec = resolve_config(expressions, sec, dialect="lab", verify=False)
    sec_catalog = Path(sec["catalog"]) if sec else None
    sec_bindings = sec["bindings"] if sec else {}
    if sec and sec.get("mode") == "dictionary":
        # Dictionary mode must reject unknown external fields as well as bad
        # calls before verifying either the dictionary or the price source.
        # A selected dictionary still provides this check for price-only text.
        for expression in expressions:
            problems = check_expression(expression, fields=set(sec_bindings))
            if problems:
                raise DatasetError("; ".join(problems))
        if sec_bindings:
            with _batch_stage(diagnostics, "sec_selected_verification", progress):
                sec = resolve_config(expressions, sec, dialect="lab", verify=True)
    registry = sec.get("registry") if sec and sec.get("mode") == "dictionary" else None
    say = progress or (lambda _message: None)
    started = time.monotonic()
    diagnostics["phase"] = "preparation"
    identity = source_identity(Path(dataset))
    dataset_hash = identity["source_sha256"]
    identities = [identity for _ in expressions]
    group_consumers = [_uses_classifications(expression, settings) for expression in expressions]
    classification = None
    if group_overlay and any(group_consumers):
        from qlib_yahoo.classification import classification_identity

        say("Verifying dated classification source ...")
        # Even an all-cached batch must verify source bytes, but never in the editor.
        classification = classification_identity(dataset, group_overlay)
        combined = {**identity, "classification": classification}
        combined["source_sha256"] = digest(combined)
        identities = [combined if consumed else identity for consumed in group_consumers]
    expression_bindings = [{} for _ in expressions]
    sec_sources = [None for _ in expressions]
    if sec_bindings:
        from qlib_yahoo.sec_features import _verify_catalog, feature_identity, selected_aliases

        # Worker boundary, before any saved-result lookup, including all-cached batches.
        # Keep this byte-verification hook out of editor/syntax validation.
        with _batch_stage(diagnostics, "sec_source_verification", progress):
            _verify_catalog(sec_catalog, force=True)

        for index, expression in enumerate(expressions):
            chosen = selected_aliases([expression], sec_bindings)
            expression_bindings[index] = chosen
            if chosen:
                source = feature_identity(sec_catalog, chosen)
                sec_sources[index] = source
                combined = {**identities[index], "sec": {"catalog": str(sec_catalog), "bindings": chosen,
                                               "source_identity": source,
                                               **({"registry": registry} if registry else {})}}
                combined["source_sha256"] = digest(combined)
                identities[index] = combined
    dataset_hashes = [item["source_sha256"] for item in identities]
    lib = Library(library or Path(data_root) / "alphalab" / "library.sqlite")
    ids = [alpha_id(e, settings, source, implementation) for e, source in zip(expressions, dataset_hashes)]
    results: dict[int, dict[str, Any]] = {}
    todo = []
    first_by_id = {}
    with _batch_stage(diagnostics, "saved_result_cache_lookup", progress):
        for index, (expression, identifier) in enumerate(zip(expressions, ids)):
            cached = None if force else lib.get(identifier)
            if cached is not None:
                cached["cached"] = True
                results[index] = cached
                diagnostics["cache_hit_count"] += 1
            else:
                if identifier not in first_by_id:
                    first_by_id[identifier] = index
                    todo.append(index)
    sec_handle = None
    classification_coverage = None
    if todo:
        workspace = Workspace(Path(dataset))
        sec_arrays = {}
        # Recheck actual source bytes after loading, before associating arrays with an identity.
        if source_identity(Path(dataset)) != identity:
            raise DatasetError("Alpha Lab source changed while loading the workspace")
        if (workspace.symbols != identity["symbols"] or sorted(workspace.fields) != sorted(identity["fields"])
                or [str(day) for day in workspace.dates] != identity["dates"]):
            raise DatasetError("loaded Alpha Lab panel differs from verified source identity")
        workspace.identity = identity
        if classification and any(group_consumers[index] for index in todo):
            from qlib_yahoo.classification import load_classification_groups

            say("Loading dated classification groups (unknown labels stay excluded) ...")
            overlay = load_classification_groups(dataset, group_overlay,
                cache_root=Path(data_root) / "cache" / "classifications", expected_identity=classification)
            # Explicit overlays replace static groups: no future/current-label fallback.
            workspace.groups = _aligned_groups(overlay, workspace)
            classification_coverage = dict(overlay.coverage)
            workspace.identity = {**identity, "classification": classification}
            workspace.identity["source_sha256"] = digest(workspace.identity)
        if sec_bindings:
            from qlib_yahoo.sec_features import materialize, load_arrays

            needed = selected_aliases([expressions[index] for index in todo], sec_bindings)
            if needed:
                source = feature_identity(sec_catalog, needed)
                sec_handle = materialize(sec_catalog, needed, workspace.dates, workspace.symbols,
                                         Path(data_root) / "cache" / "sec", progress=progress,
                                         force_verify=False)
                sec_arrays = load_arrays(sec_handle)
                try:
                    if set(sec_arrays) != set(needed) or any(
                            values.shape != (len(workspace.dates), len(workspace.symbols))
                            or values.dtype != np.dtype("float32") for values in sec_arrays.values()):
                        raise DatasetError("SEC selected panel fields, shapes or dtypes differ")
                    workspace.fields.update(sec_arrays)
                    if feature_identity(sec_catalog, needed) != source or any(
                            feature_identity(sec_catalog, expression_bindings[index]) != sec_sources[index]
                            for index in todo if expression_bindings[index]):
                        raise DatasetError("SEC source changed while loading selected fields")
                except BaseException:
                    _close_sec_arrays(sec_arrays)
                    raise
                workspace.identity = {**workspace.identity, "sec": {"catalog": str(sec_catalog), "bindings": needed,
                                                          "source_identity": source,
                                                          "materialized_identity": sec_handle["identity"],
                                                          **({"registry": registry} if registry else {})}}
                workspace.identity["source_sha256"] = digest(workspace.identity)
        try:
            say(f"Lab source/panel preparation after resolution in {time.monotonic() - started:.1f}s: "
                f"{len(workspace.dates):,} sessions x "
                f"{len(workspace.symbols):,} instruments")
            directory = write_cache(workspace, cache_dir(data_root, workspace), [settings.universe])
        finally:
            _close_sec_arrays(sec_arrays)
        count = worker_count(workers, len(todo), workspace, worker_ceiling)
        del workspace
        say(f"Simulating {len(todo)} alpha(s) with {count} worker process(es) ...")
        diagnostics["phase"] = "evaluation"
        diagnostics["new_computation_count"] += len(todo)
        if count == 1:
            # In-process batches own their maps locally, including on callback failure.
            with closing(CachedWorkspace(directory)) as cached_workspace:
                for done, index in enumerate(todo, 1):
                    results[index] = _run_one(expressions[index], asdict(settings), cached_workspace)
                    say(f"[{done}/{len(todo)}] {_line(results[index])}")
        else:
            with ProcessPoolExecutor(max_workers=count, initializer=_init_worker, initargs=(str(directory),)) as pool:
                futures = {pool.submit(_run_one, expressions[index], asdict(settings)): index for index in todo}
                for done, future in enumerate(as_completed(futures), 1):
                    index = futures[future]
                    results[index] = future.result()
                    say(f"[{done}/{len(todo)}] {_line(results[index])}")
    import copy
    ordered = [copy.deepcopy(results[i if i in results else first_by_id[identifier]])
               for i, identifier in enumerate(ids)]
    # self-correlation against the library, then save the new results
    # Input lineage distinguishes cache IDs; comparison cohorts use the common
    # price source and the library's universe/delay contract. Different SEC
    # bindings are candidates to compare, not disjoint correlation cohorts.
    library_pnl = lib.pnl_series(dataset_hash, settings, exclude=set(ids))
    for index, (identifier, result) in enumerate(zip(ids, ordered)):
        result["id"] = identifier
        result["implementation"] = implementation
        if not result.get("cached"):
            # The shared workspace union must not change an individual alpha's identity.
            result["dataset"] = identities[index]
            if classification and group_consumers[index]:
                result["classification"] = {"source_identity": classification,
                                            "coverage": classification_coverage}
            if expression_bindings[index]:
                result["sec"] = {"catalog": str(sec_catalog), "bindings": expression_bindings[index],
                                 "source_identity": sec_sources[index], "materialized": sec_handle,
                                 **({"mode": "dictionary", "registry": registry} if registry else {})}
        if "error" in result:
            continue
        mine = (result["daily"]["dates"], result["daily"]["pnl"])
        best, best_id = None, None
        for other_id, series in library_pnl.items():
            value = correlation(mine, series)
            if value is not None and (best is None or value > best):
                best, best_id = value, other_id
        result["self_correlation"] = best
        result["most_correlated_id"] = best_id
        result["checks"] = [c for c in result["checks"] if c["name"] != "SELF_CORRELATION"] + [{
            "name": "SELF_CORRELATION", "value": best, "limit": SELF_CORRELATION_LIMIT,
            "result": "PASS" if best is None or best < SELF_CORRELATION_LIMIT else "FAIL"}]
        if not result.get("cached"):
            lib.save(identifier, result, dataset_hash)
    for identifier, result in zip(ids, ordered):
        if "error" in result:
            lib.save(identifier, result, dataset_hash)
    return ordered


def batch_correlations(results: list[dict[str, Any]]) -> tuple[list[str], list[list[float | None]]]:
    ok = [r for r in results if "error" not in r]
    ids = [r["id"] for r in ok]
    series = [(r["daily"]["dates"], r["daily"]["pnl"]) for r in ok]
    matrix = [[1.0 if i == j else correlation(series[i], series[j]) for j in range(len(ok))] for i in range(len(ok))]
    return ids, matrix


def _line(result: dict[str, Any]) -> str:
    if "error" in result:
        return f"ERROR {result['expression'][:60]}: {result['error']}"
    s = result["summary"]

    def fmt(value, spec):
        return format(value, spec) if isinstance(value, (int, float)) and math.isfinite(value) else "-"

    return (f"sharpe {fmt(s.get('sharpe'), '5.2f')} fitness {fmt(s.get('fitness'), '5.2f')} "
            f"returns {fmt(s.get('returns'), '6.1%')} turnover {fmt(s.get('turnover'), '6.1%')}  {result['expression'][:70]}")


def save_batch(data_root: Path, results: list[dict[str, Any]], settings: Settings, dataset: str,
               *, diagnostics: dict | None = None) -> Path:
    """Record IDs and diagnostics in a new batch receipt, never in alpha rows.

    Without the optional run_batch diagnostics, derive counts from transient
    result flags. This preserves the original four-argument calling convention.
    """
    folder = Path(data_root) / "alphalab" / "batches"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.json"
    diagnostic_record = dict(diagnostics) if diagnostics is not None else {
        "scope": "batch_results", "cache_hit_count": sum(bool(r.get("cached")) for r in results),
        "new_computation_count": len({r["id"] for r in results
                                      if not r.get("cached") and not r.get("preflight")}),
    }
    path.write_text(json.dumps({"created_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": str(dataset),
                                "settings": asdict(settings), "ids": [r["id"] for r in results],
                                "evidence": [{"attempt_id": r.get("attempt_id"),
                                              "result_digest": r.get("result_digest"),
                                              "batch_attempt_id": r.get("batch_attempt_id")} for r in results],
                                "diagnostics": {"version": 1, "alphalab": diagnostic_record},
                                "errors": {r["id"]: r["error"] for r in results if "error" in r}}, indent=1),
                    encoding="utf-8")
    return path


def read_expressions(path: Path) -> list[str]:
    """One alpha per line; '#' starts a comment line; blank lines are ignored."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
