"""Bounded, offline report reuse for completed saved Qlib runs."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from qlib_yahoo import analysis


PROFILES = {
    "holdings": ("summary", "always"),
    "overview": ("overview", "auto"),
    "detailed": ("detailed", "auto"),
}
THRESHOLD = 1.3
MIN_OBSERVATIONS = 252
_SELECTIONS = frozenset({"manifest.json", "summary.json", "daily_report.csv"})
_MAX_MANIFEST_BYTES = 64 * 1024


def _is_redirect(path: Path) -> bool:
    """Reject symlinks and Windows junctions before resolving local paths."""
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def resolve_run_selection(path: Path) -> Path:
    """Resolve a selected run path without reading or validating its bundle.

    Report requests validate the seal, bytes and ledger after acquiring the lock.
    """
    selected = Path(path)
    if _is_redirect(selected):
        raise ValueError("Run selection must not be a symlink or junction")
    if selected.is_file():
        if selected.name not in _SELECTIONS:
            raise ValueError("Select a run directory or its root manifest, summary, or daily ledger")
        selected = selected.parent
        if _is_redirect(selected):
            raise ValueError("Run directory must not be a symlink or junction")
    elif not selected.is_dir():
        raise ValueError("Run selection does not exist or is not a directory")
    run_dir = selected.resolve(strict=True)
    if _is_redirect(run_dir / "manifest.json"):
        raise ValueError("Run manifest must not be a symlink or junction")
    return run_dir


def validate_run_selection(path: Path) -> Path:
    """Read-only full validation for callers explicitly inspecting a saved run."""
    run_dir = resolve_run_selection(path)
    analysis.validate_bundle(run_dir)
    return run_dir


def _safe_file(directory: Path, name: str) -> bool:
    if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
        return False
    path = directory / name
    if _is_redirect(path) or not path.is_file():
        return False
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def _reusable(run_dir: Path, request: dict[str, object], bundle: analysis.ValidatedBundle) -> Path | None:
    reports = run_dir / "reports"
    if _is_redirect(reports):
        raise ValueError("Reports directory must not be a symlink or junction")
    if not reports.is_dir():
        return None
    provenance = bundle.provenance
    manifest = bundle.manifest
    signed = analysis._is_signed(manifest)
    # Inspect one directory level in this run only; bound each manifest read.
    for directory in reports.iterdir():
        if _is_redirect(directory) or not directory.is_dir():
            continue
        if not _safe_file(directory, "report_manifest.json"):
            continue
        manifest_path = directory / "report_manifest.json"
        try:
            if manifest_path.stat().st_size > _MAX_MANIFEST_BYTES:
                continue
            status = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            continue
        if not isinstance(status, dict) or status.get("status") != "completed":
            continue
        exact_request = all(status.get(key) == value for key, value in request.items())
        # An automatic summary that actually drew holdings satisfies the
        # holdings button's manual request without drawing the same chart again.
        auto_holdings = (
            request["level"] == "summary" and request["holdings"] == "always"
            and status.get("level") == "summary" and status.get("holdings") == "auto"
            and status.get("holdings_rendered") is True
            and all(status.get(key) == request[key] for key in ("threshold", "min_observations"))
        )
        if not (exact_request or auto_holdings) or any(
            status.get(key) != value for key, value in provenance.items()
        ):
            continue
        charts = status.get("charts")
        if (not isinstance(charts, list) or not charts
                or not all(isinstance(name, str) for name in charts)
                or len(charts) != len(set(charts))):
            continue
        required = {"holdings.html"} if request["level"] == "summary" else {
            "account.html", "drawdown.html", "returns_cost.html", "turnover.html"
        }
        if signed and request["level"] != "summary":
            required |= {"account_components.html", "exposure.html"}
        if not required.issubset(charts):
            continue
        if not all(_safe_file(directory, name) for name in charts):
            continue
        if not _safe_file(directory, "report_index.html") or not _safe_file(directory, "plotly.min.js"):
            continue
        return directory / "report_index.html"
    return None


def _process_alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    if os.name == "nt":  # os.kill would terminate the process on Windows; only query it
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return kernel32.GetLastError() == 5  # access denied: the process exists
        try:
            code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stale_lock(lock_path: Path) -> bool:
    """A lock whose writer has exited (crash, force-quit) is stale; an unreadable one only after an hour."""
    try:
        text = lock_path.read_text(encoding="ascii").strip()
        age = time.time() - lock_path.stat().st_mtime
    except (OSError, UnicodeDecodeError):
        return False
    if text.isdigit():
        return not _process_alive(int(text))
    return age > 3600


@contextmanager
def _run_lock(run_dir: Path) -> Iterator[None]:
    lock_path = run_dir / ".report_reuse.lock"
    for attempt in range(2):
        try:
            descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            break
        except FileExistsError as exc:
            if attempt == 0 and _stale_lock(lock_path):
                lock_path.unlink(missing_ok=True)
                continue
            raise RuntimeError("Report generation is already in progress for this run; lock was not overridden") from exc
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
        yield
    finally:
        lock_path.unlink()


def ensure_report(run_dir: Path, profile: str) -> tuple[Path, bool]:
    """Reuse a complete matching report, or render one; return (index, reused)."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown report profile: {profile}")
    run_dir = resolve_run_selection(run_dir)
    level, holdings = PROFILES[profile]
    request: dict[str, object] = {
        "level": level, "holdings": holdings,
        "threshold": THRESHOLD, "min_observations": MIN_OBSERVATIONS,
    }
    with _run_lock(run_dir):
        bundle = analysis.validate_bundle(run_dir)
        existing = _reusable(run_dir, request, bundle)
        if existing is not None:
            bundle.assert_unchanged()
            return existing, True
        index = analysis.render_report(
            run_dir, level=level, holdings=holdings,
            threshold=THRESHOLD, min_observations=MIN_OBSERVATIONS,
            _validated=bundle,
        )
        return index, False
