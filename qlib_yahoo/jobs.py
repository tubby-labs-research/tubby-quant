"""Run long console tasks in a child Python process with streamed output and cancellation.

The window process never imports Qlib: backtests (long-only or signed) and
dataset imports run in ``python -m qlib_yahoo.jobs --request-file <path>``. A request is JSON
data, never interpolated into code. The child prints one result marker line.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from uuid import uuid4
from pathlib import Path
from typing import Any, Callable

from qlib_yahoo.dataset import DatasetError

ROOT = Path(__file__).resolve().parents[1]
RESULT = "__QUANT_JOB_RESULT__="
FAILURE = "__QUANT_JOB_ERROR__="
KINDS = ("backtest", "backtest-batch", "import-table", "alpha-batch", "refresh-data", "build-dictionary")
COMPLETED = "Backtest completed: "  # printed by run_backtest once the run is sealed, before any report
ACTIVE_RUN = "__QUANT_JOB_ACTIVE_RUN__="
ACTIVE_BATCH = "__QUANT_JOB_ACTIVE_BATCH__="
_PERCENT = re.compile(r"^(?P<label>[^:|]{1,40}):\s*(?P<percent>\d{1,3})%\|")
MAX_REQUEST_BYTES = 16 * 1024 * 1024
# Third-party import banners that say nothing about the task (Qlib imports the unmaintained gym package).
_NOISE = re.compile(r"^(Gym has been unmaintained|Please upgrade to Gymnasium|Users of this version of Gym|"
                    r"See the migration guide at https://gymnasium)")


class JobCancelled(RuntimeError):
    """The user cancelled the running task; partial output is kept, never reported as completed."""


def progress_fraction(line: str) -> tuple[str, float] | None:
    """Recognize a tqdm progress line such as ``backtest loop:  26%|██▋ | 132/500``."""
    match = _PERCENT.match(line.strip())
    if not match:
        return None
    return match["label"].strip(), min(int(match["percent"]), 100) / 100


def _python() -> Path:
    executable = Path(sys.executable)
    if executable.name.lower() == "pythonw.exe":
        executable = executable.with_name("python.exe")
    return executable


def _run_request(request: dict[str, Any]) -> Path:
    kind = request.pop("kind", None)
    if kind == "build-dictionary":
        from qlib_yahoo.sec_registry import build_dictionary, dictionary_status
        from qlib_yahoo.sec_execution import build_execution_index

        catalog = Path(request["catalog"])
        if dictionary_status(catalog)["status"] != "ready":
            build_dictionary(catalog, progress=lambda message: print(message, flush=True))
        build_execution_index(catalog, progress=lambda message: print(message, flush=True))
        status = dictionary_status(catalog)
        if status["status"] != "ready":
            raise DatasetError(status.get("message") or "SEC dictionary build did not complete")
        return Path(status["path"])
    if kind == "backtest":
        from qlib_yahoo.backtest import run_backtest

        request["dataset"] = Path(request["dataset"])
        return Path(run_backtest(**request))
    if kind == "backtest-batch":
        if request.get("sec") and (request["sec"].get("mode") == "dictionary" or "bindings" not in request["sec"]):
            from qlib_yahoo.sec_resolution import resolve_config
            from qlib_yahoo.performance import RunPerformance

            request["sec"] = RunPerformance().call("SEC batch resolution and selected verification", resolve_config,
                                                    request["alphas"], request["sec"], dialect="qlib", verify=True)
        from qlib_yahoo.preparation import run_backtest_batch

        request.pop("alpha", None)
        return run_backtest_batch(Path(request.pop("dataset")), request.pop("alphas"),
                                  data_root=Path(request.pop("data_root")), **request)
    if kind == "import-table":
        from qlib_yahoo.stooq_table import prepare_stooq_table
        from qlib_yahoo.universe import UniverseRule

        rule = UniverseRule(**request.pop("rule"))
        return Path(prepare_stooq_table(Path(request.pop("table")), Path(request.pop("data_root")),
                                        request.pop("start"), request.pop("end"), rule=rule, **request))
    if kind == "refresh-data":
        from qlib_yahoo.refresh import refresh

        record = refresh(Path(request["data_root"]), previous_tables=[Path(p) for p in request.get("previous_tables", [])],
                         min_price=float(request.get("min_price", 1.0)), make_latest=bool(request.get("make_latest")),
                         workers=int(request.get("workers", 6)), progress=lambda message: print(message, flush=True))
        return Path(record["snapshot"])
    if kind == "alpha-batch":
        from qlib_yahoo.alphalab.batch import run_batch, save_batch
        from qlib_yahoo.alphalab.settings import Settings

        data_root = Path(request["data_root"])
        settings = Settings(**request["settings"])
        diagnostics = {}
        results = run_batch(Path(request["dataset"]), list(request["expressions"]), settings, data_root=data_root,
                            workers=int(request.get("workers", 4)), force=bool(request.get("force")),
                            **({"sec": request["sec"]} if request.get("sec") else {}),
                            **({"group_overlay": request["group_overlay"]} if request.get("group_overlay") else {}),
                            progress=lambda message: print(message, flush=True), diagnostics=diagnostics)
        return save_batch(data_root, results, settings, request["dataset"], diagnostics=diagnostics)
    raise DatasetError(f"unknown job kind: {kind!r}")


def _interrupted(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt("cancelled by the user")


def child_main(argv: list[str]) -> int:
    """Entry point inside the worker process."""
    started = time.monotonic()
    outcome = "interrupted"
    if os.name != "nt":
        # Cancel sends SIGTERM first: unwind so runs and snapshots are recorded as failed, not left running.
        signal.signal(signal.SIGTERM, _interrupted)
    try:
        if len(argv) == 2 and argv[0] == "--request-file":
            with Path(argv[1]).open("rb") as source:
                encoded = source.read(MAX_REQUEST_BYTES + 1)
            if len(encoded) > MAX_REQUEST_BYTES:
                raise DatasetError("job request exceeds the 16 MiB transport limit")
            request = json.loads(encoded.decode("utf-8"))
        elif len(argv) == 1:
            # Keep the short legacy child CLI for callers and fixtures.
            if len(argv[0].encode("utf-8")) > MAX_REQUEST_BYTES:
                raise DatasetError("job request exceeds the 16 MiB transport limit")
            request = json.loads(argv[0])
        else:
            raise DatasetError("expected JSON request or --request-file PATH")
        if not isinstance(request, dict) or request.get("kind") not in KINDS:
            raise DatasetError("job request must be a JSON object with a known kind")
        result = _run_request(request)
        outcome = "completed"
    except Exception as error:  # reported to the parent; the traceback stays in this process's output
        outcome = "failed"
        message = str(error) or type(error).__name__
        print(FAILURE + json.dumps(f"{type(error).__name__}: {message}" if not isinstance(error, DatasetError) else message),
              flush=True)
        return 1
    finally:
        print("Full worker elapsed (child_main entry through request parsing, execution, save/report; "
              "excludes process startup/parent UI): " + json.dumps({
                  "diagnostics": {"version": 1, "worker": {
                      "scope": "child_main", "outcome": outcome,
                      "elapsed_seconds": max(0.0, time.monotonic() - started),
                  }}
              }), flush=True)
    print(RESULT + json.dumps(str(result)), flush=True)
    return 0


class Job:
    """One cancellable child process. ``run`` blocks; ``cancel`` may be called from any thread."""

    def __init__(self, request: dict[str, Any]):
        if request.get("kind") not in KINDS:
            raise DatasetError("job request must name a known kind")
        # Validate serializability on the caller's thread so bad input fails before a process starts.
        self.payload = json.dumps(request, allow_nan=False)
        if len(self.payload.encode("utf-8")) > MAX_REQUEST_BYTES:
            raise DatasetError("job request exceeds the 16 MiB transport limit")
        self.process: subprocess.Popen | None = None
        self._process_waited = False
        self.cancelled = False
        self._lock = threading.Lock()
        self._cancel_file = (Path(tempfile.gettempdir()) / f"quant-job-cancel-{uuid4().hex}"
                             if request.get("kind") in ("backtest", "backtest-batch") else None)

    def cancel(self) -> bool:
        with self._lock:
            process = self.process
            if process is not None and process.poll() is not None:
                return False  # already finished: its outcome stands
            self.cancelled = True
        if process is None:
            return False
        cooperative = self._cancel_file is not None
        if cooperative:
            try:
                self._cancel_file.write_text("cancel", encoding="ascii")
            except OSError:
                cooperative = False
        # Escalation may wait several seconds; never block the caller's (UI) thread.
        threading.Thread(target=_terminate_tree, args=(process, cooperative), daemon=True).start()
        return True

    def run(self, on_line: Callable[[str], None]) -> Path:
        # Parent owns the file until the child exits, including forced cancellation.
        # The random path contains no expressions; neither argv nor launch errors
        # expose the private request. TemporaryDirectory also cleans spawn failures.
        try:
            with tempfile.TemporaryDirectory(prefix="quant-job-request-") as directory:
                self._process_waited = False
                try:
                    return self._run(on_line, Path(directory) / "request.json")
                except BaseException:
                    # A callback/read error must not leave an unobserved worker
                    # running after its request and cancellation marker disappear.
                    if (not self._process_waited and self.process is not None
                            and getattr(self.process, "poll", lambda: 0)() is None):
                        _terminate_tree(self.process)
                        self.process.wait()
                    raise
        finally:
            close_output = getattr(getattr(self.process, "stdout", None), "close", None)
            if close_output is not None:
                close_output()
            if self._cancel_file is not None:
                self._cancel_file.unlink(missing_ok=True)

    def _run(self, on_line: Callable[[str], None], request_file: Path) -> Path:
        environment = os.environ.copy()
        environment.update({"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "MLFLOW_DISABLE_AGENT_HINT": "1"})
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT), environment.get("PYTHONPATH", ""))))
        options: dict[str, Any] = {}
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            options["start_new_session"] = True  # own process group: cancel reaches dump_bin workers too
        with self._lock:
            if self.cancelled:
                raise JobCancelled("Cancelled before the task started.")
            payload = json.loads(self.payload)
            if self._cancel_file is not None:
                payload["_cancel_file"] = str(self._cancel_file)
            encoded = json.dumps(payload, allow_nan=False).encode("utf-8")
            if len(encoded) > MAX_REQUEST_BYTES:
                raise DatasetError("job request exceeds the 16 MiB transport limit")
            request_file.write_bytes(encoded)
            self.process = subprocess.Popen(
                [str(_python()), "-m", "qlib_yahoo.jobs", "--request-file", str(request_file)], cwd=ROOT, env=environment,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1, **options)
        result: str | None = None
        failure: str | None = None
        completed: str | None = None
        active_run = active_batch = None
        assert self.process.stdout is not None
        # Universal newlines turn tqdm's carriage returns into separate lines.
        for line in self.process.stdout:
            line = line.rstrip("\n")
            if line.startswith(RESULT):
                result = json.loads(line[len(RESULT):])
            elif line.startswith(FAILURE):
                failure = json.loads(line[len(FAILURE):])
            elif line.startswith(ACTIVE_RUN):
                active_run = json.loads(line[len(ACTIVE_RUN):])
            elif line.startswith(ACTIVE_BATCH):
                active_batch = json.loads(line[len(ACTIVE_BATCH):])
            elif line.strip() and not _NOISE.match(line.strip()):
                if line.startswith(COMPLETED):
                    completed = line[len(COMPLETED):].strip()
                on_line(line)
        code = self.process.wait()
        self._process_waited = True
        close_output = getattr(self.process.stdout, "close", None)
        if close_output is not None:
            close_output()
        if self._cancel_file is not None:
            self._cancel_file.unlink(missing_ok=True)
        if self.cancelled and result is None:
            _record_cancelled(self.payload, active_run, active_batch)
            if completed and json.loads(self.payload).get("kind") == "backtest":
                # A batch is not complete merely because an earlier constituent was sealed.
                return Path(completed)
            raise JobCancelled("Cancelled. Partial output is kept and is not marked completed.")
        if failure is not None:
            raise DatasetError(failure)
        if code or result is None:
            raise DatasetError(f"task process exited with code {code} without a result")
        return Path(result)


def _record_cancelled(payload: str, active_run, active_batch):
    """Repair only this worker's running records after a forced termination.

    Completed constituent runs are preserved. Partial files stay recoverable;
    neither their history nor an interrupted batch is promoted to completed.
    """
    from qlib_yahoo.backtest import _atomic_json, _now
    request = json.loads(payload)
    # Import/refresh jobs have no backtest records (and need not name a
    # dataset). Keep their cancellation outcome independent of this repair.
    if not active_run and not active_batch:
        return
    dataset = Path(request["dataset"]).resolve()
    data_root = Path(request.get("data_root") or dataset.parent.parent).resolve()
    completed_run = None
    for selected, category in ((active_run, "run"), (active_batch, "batch")):
        if not selected:
            continue
        path = Path(selected)
        # run_backtest keeps runs beside its dataset; the receipt can live in
        # an explicitly different data_root. Check every ancestor before any
        # repair so a redirected runs/batches directory cannot widen the target.
        parent = dataset.parent.parent / "runs" if category == "run" else data_root / "workbench/batches"
        if (any(component.is_symlink() or (hasattr(component, "is_junction") and component.is_junction())
                for component in (path, *path.parents)) or path.parent.resolve() != parent.resolve()):
            continue
        manifest_path = path / "manifest.json" if category == "run" else path
        if manifest_path.is_symlink() or not manifest_path.is_file():
            continue
        try:
            record = json.loads(manifest_path.read_text(encoding="utf-8"))
            if category == "run" and record.get("status") == "completed":
                if (Path(record.get("dataset", "")).resolve() == dataset
                        and record.get("alpha") in request.get("alphas", [request.get("alpha")])):
                    completed_run = str(path.resolve())
                continue
            if record.get("status") not in (("running", "cancelled") if category == "batch" else ("running",)):
                continue
            if category == "batch" and completed_run and completed_run not in record.get("runs", []):
                record.setdefault("runs", []).append(completed_run)
            record.update(status="failed" if category == "run" else "cancelled", cancelled_at_utc=_now(),
                          error={"type": "JobCancelled", "message": "cancelled by the user; partial files retained"})
            _atomic_json(manifest_path, record)
        except (OSError, ValueError):
            pass  # UI cancellation still succeeds if an interrupted record cannot be repaired.


def _terminate_tree(process: subprocess.Popen, cooperative=False) -> None:
    try:
        if cooperative:
            try:
                process.wait(timeout=8)
                return
            except subprocess.TimeoutExpired:
                pass
        if os.name == "nt":
            killed = subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)],
                                    capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
            if killed.returncode and process.poll() is None:
                process.kill()
        else:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            process.kill()
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(child_main(sys.argv[1:]))
