"""Explicit ordinary-console acceptance run; importing this module launches nothing.

Use the installed workbench Python. --receipt is required and must be new.
Backtest output follows the existing runner: --data-root must be the dataset's
enclosing root (dataset.parent.parent). Lab uses a fresh folder beneath the
explicit root, preserving the existing library and historical zero-cost results.
This checks completion and report coverage, not numerical parity or a speed gate.
"""
from __future__ import annotations

import argparse
import csv
from contextlib import closing
from dataclasses import asdict
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qlib_yahoo.alphalab.settings import Settings
from qlib_yahoo.jobs import Job, _terminate_tree

START, END = "2021-01-04", "2025-12-31"
EXPRESSIONS = ("-rank(ts_delta(close,5))", "rank(volume)")
ALPHA = "-1 * ($close / Ref($close, 5) - 1)"
CHARTS = frozenset(name + ".html" for name in (
    "account", "drawdown", "returns_cost", "turnover", "account_components",
    "exposure", "holdings", "rolling_risk", "signal_ic", "score_groups"))
ARTIFACTS = frozenset(("summary.json", "daily_report.csv", "alpha_scores.parquet",
    "signed_weights.parquet", "signal_forward_labels.parquet", "positions.parquet",
    "signed_decisions.json.gz", "qlib_trade_indicators.csv", "qlib_order_indicators.parquet"))
METADATA_LIMIT = 8 * 1024 * 1024


def configure_output() -> None:
    """Console/log streams accept tqdm Unicode; StringIO needs no reconfiguration."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def failure_notice(message: str, *, file=None) -> None:
    # A broken output destination must not mask the worker error or its receipt.
    try:
        print(message, file=file, flush=True)
    except (OSError, UnicodeError):
        pass


def stop_failed_job(job) -> dict:
    """Finish Job.cancel's asynchronous tree shutdown before leaving the harness."""
    errors = []
    try:
        job.cancel()
    except BaseException as error:
        errors.append(f"cancel: {type(error).__name__}: {error}")
    process = job.process
    if process is None:
        return {"child_waited": False, "errors": errors}
    waited = False
    try:
        # Job.cancel grants eight seconds of cooperative cleanup before killing
        # the process tree. A failed callback no longer drains its output pipe,
        # so wait only a bounded time before invoking the existing tree fallback.
        try:
            if errors:
                _terminate_tree(process)
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            _terminate_tree(process)
            process.wait(timeout=10)
        waited = True
    except BaseException as error:
        errors.append(f"wait: {type(error).__name__}: {error}")
        try:
            _terminate_tree(process)
            process.kill()
            process.wait(timeout=10)
            waited = True
        except BaseException as cleanup_error:
            errors.append(f"kill/wait: {type(cleanup_error).__name__}: {cleanup_error}")
    finally:
        if waited:
            try:
                if process.stdout is not None:
                    process.stdout.close()
                cancel_file = getattr(job, "_cancel_file", None)
                if cancel_file is not None:
                    cancel_file.unlink(missing_ok=True)
            except BaseException as error:
                errors.append(f"close: {type(error).__name__}: {error}")
    return {"child_waited": waited, "errors": errors}


def safe_path(path: Path) -> Path:
    path = Path(path).absolute()
    for component in (path, *path.parents):
        if component.is_symlink() or (hasattr(component, "is_junction") and component.is_junction()):
            raise ValueError(f"Redirected path: {component}")
    return path.resolve()


def inside(path: Path, root: Path, dataset: Path) -> Path:
    path = safe_path(path)
    if not path.is_relative_to(root) or path.is_relative_to(dataset):
        raise ValueError(f"Output outside explicit root or inside dataset: {path}")
    return path


def read_json(path: Path) -> dict:
    path = safe_path(path)
    if path.stat().st_size > METADATA_LIMIT:
        raise ValueError(f"Metadata exceeds {METADATA_LIMIT} bytes: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return result


def source_identity() -> dict:
    def git(*args):
        return subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    try:
        # Retain the indicator, not names/content of unrelated/private local work.
        result = {"commit": git("rev-parse", "HEAD"),
                  "branch": git("branch", "--show-current"),
                  "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError) as error:
        result = {"commit": None, "branch": None, "dirty": None,
                  "unavailable": type(error).__name__}
    sources = {path for pattern in ("*.py", "*.pyx")
               for path in (ROOT / "qlib_yahoo").rglob(pattern)}
    sources.add(ROOT / "tools/qlib_console_recovery_smoke.py")
    # Include all platform derivative pins: auto selection can choose a different
    # pin from the default. Hash the local profile pointer without exposing it.
    sources.update((ROOT / "docs").glob("qlib-derivative-pin*.json"))
    sources.update(ROOT / name for name in (
        "docs/qlib-derivative-pin.json", "docs/qlib-fast-runtime.v1.json",
        "docs/qlib-row-range-build.candidate.json", "docs/qlib-row-range-buffer-build.candidate.json",
        "env/qlib-fast-runtime.json"))
    result["sha256"] = {path.relative_to(ROOT).as_posix(): (
        hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None)
        for path in sorted(sources)}
    return result


def request_for(args, dataset: Path, output_root: Path) -> dict:
    if args.kind == "lab":
        settings = Settings(start=args.start, end=args.end, neutralization="MARKET",
                            delay=1, warmup=252).validate()
        return {"kind": "alpha-batch", "dataset": str(dataset), "data_root": str(output_root),
                "expressions": list(EXPRESSIONS), "settings": asdict(settings),
                "force": True, "workers": 2}
    return {"kind": "backtest", "dataset": str(dataset), "alpha": ALPHA,
            "start": args.start, "end": args.end, "strategy_mode": "signed",
            "signal_transform": "raw", "universe": "top3000", "warmup_sessions": 252,
            "initial_cash": 100000.0, "cost_per_side": .0005, "annual_borrow_rate": .004,
            "annual_cash_rate": 0.0, "gross_exposure": 1.0, "history_mode": "stream",
            "execution_profile": "fast", "runtime_profile": "auto", "open_report": False,
            "report_level": "detailed", "holdings": "always", "threshold": 1.3,
            "min_observations": 252}


def artifact(path: Path, root: Path, dataset: Path) -> dict:
    path = inside(path, root, dataset)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty artifact: {path}")
    return {"path": str(path), "bytes": path.stat().st_size}


def validate_lab(result: Path, request: dict, root: Path, dataset: Path) -> dict:
    batch = read_json(result)
    ids = batch.get("ids", [])
    if (len(ids) != 2 or len(set(ids)) != 2 or batch.get("errors") != {}
            or batch.get("settings") != request["settings"]
            or safe_path(Path(batch.get("dataset", ""))) != dataset):
        raise ValueError("Lab batch must contain two successful requested results")
    library = Path(request["data_root"]) / "alphalab/library.sqlite"
    paths = [artifact(result, root, dataset), artifact(library, root, dataset)]
    # Read only small columns, never decompress the saved daily-series BLOBs.
    with closing(sqlite3.connect(library.as_uri() + "?mode=ro", uri=True)) as db:
        rows = [db.execute("SELECT expression, status, settings FROM alphas WHERE id = ?",
                           (identifier,)).fetchone() for identifier in ids]
    for row, expression in zip(rows, EXPRESSIONS):
        if (row is None or row[0] != expression or row[1] != "ok"
                or json.loads(row[2]) != request["settings"]):
            raise ValueError("Lab library result missing, failed or settings changed")
    return {"core_completed": True, "successes": 2, "ids": ids, "artifacts": paths,
            "report": {"type": "lab batch/library leaderboard", "path": str(result)}}


def validate_backtest(result: Path, report_path: Path | None, request: dict,
                      root: Path, dataset: Path) -> dict:
    if result.parent != root / "runs":
        raise ValueError("Backtest returned a run outside the ordinary runs folder")
    manifest = read_json(result / "manifest.json")
    if (manifest.get("status") != "completed" or manifest.get("alpha") != ALPHA
            or manifest.get("strategy_mode") != "signed"
            or safe_path(Path(manifest.get("dataset", ""))) != dataset):
        raise ValueError("Backtest core not completed or request identity differs")
    contract = manifest.get("backtest", {})
    for name, expected in (("start", request["start"]), ("end", request["end"]),
                           ("warmup_sessions", 252), ("cash", 100000.0), ("execution_profile", "fast")):
        if contract.get(name) != expected:
            raise ValueError(f"Backtest contract differs: {name}")
    for name, expected in (("cash_borrowing", False), ("restricted_short_proceeds", True),
                           ("open_cost", .0005), ("close_cost", .0005),
                           ("annual_borrow_rate", .004), ("gross_exposure", 1.0),
                           ("signal_transform", "raw"), ("benchmark", None)):
        execution = contract.get("execution_contract", {})
        if name not in execution or execution[name] != expected:
            raise ValueError(f"Signed execution contract differs: {name}")
    if (contract.get("universe", {}).get("name") != "top3000"
            or manifest.get("history_storage", {}).get("mode") != "stream"):
        raise ValueError("Universe/history differs")
    paths = [artifact(result / "manifest.json", root, dataset)]
    for name in sorted(ARTIFACTS):
        if name not in manifest.get("artifacts_sha256", {}):
            raise ValueError(f"Core artifact not sealed: {name}")
        paths.append(artifact(result / name, root, dataset))
    calendar_path = safe_path(dataset / "qlib_data/calendars/day.txt")
    if calendar_path.stat().st_size > METADATA_LIMIT:
        raise ValueError("Calendar exceeds metadata cap")
    calendar = calendar_path.read_text(encoding="utf-8").split()
    expected_dates = [day for day in calendar if request["start"] <= day <= request["end"]]
    if not expected_dates or expected_dates != sorted(set(expected_dates)):
        raise ValueError("Empty or unordered evaluation calendar")
    with (result / "daily_report.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or reader.fieldnames[0] != "datetime":
            raise ValueError("Daily ledger lacks datetime column")
        count = 0
        for row in reader:
            if count >= len(expected_dates) or row["datetime"][:10] != expected_dates[count]:
                raise ValueError("Daily ledger dates differ from selected calendar")
            count += 1
    if count != len(expected_dates):
        raise ValueError("Daily ledger is incomplete")
    five_year = request["start"] == START and request["end"] == END
    if five_year and count != 1255:
        raise ValueError("Five-year workload requires 1,255 sessions")
    if report_path is None:
        raise ValueError("Worker returned no completed Report path")
    report_path = inside(report_path, root, dataset)
    if report_path.name != "report_index.html" or report_path.parent.parent != result / "reports":
        raise ValueError("Report path does not belong to returned run")
    report = read_json(report_path.parent / "report_manifest.json")
    charts = report.get("charts", [])
    required = {"account.html", "drawdown.html", "returns_cost.html", "turnover.html",
                "account_components.html", "exposure.html", "holdings.html"}
    if (report.get("status") != "completed" or report.get("level") != "detailed"
            or report.get("holdings") != "always" or report.get("holdings_rendered") is not True
            or not isinstance(charts, list) or len(charts) != len(set(charts))
            or not required.issubset(charts) or not set(charts).issubset(CHARTS)
            or (five_year and set(charts) != CHARTS)):
        raise ValueError("Required detailed holdings report incomplete")
    for name in ("report_index.html", "report_manifest.json", "plotly.min.js", *charts):
        paths.append(artifact(report_path.parent / name, root, dataset))
    return {"core_completed": True, "sessions": count, "expected_sessions": len(expected_dates),
            "five_year_workload": five_year, "chart_count": len(charts), "artifacts": paths,
            "report": str(report_path),
            "metadata": {key: manifest[key] for key in ("runtime_profile", "execution_acceleration",
                "history_storage", "performance", "signed_engine_identity") if key in manifest}}


def persist(path: Path, record: dict) -> None:
    # This is only called after exclusively creating this invocation's receipt.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(args) -> dict:
    configure_output()
    dataset, root, receipt = map(safe_path, (args.dataset, args.data_root, args.receipt))
    if not dataset.is_dir() or not root.is_dir() or not receipt.parent.is_dir():
        raise ValueError("Dataset, explicit data root and receipt parent must already exist")
    if receipt.is_relative_to(dataset) or receipt.suffix.lower() != ".json":
        raise ValueError("Receipt must be a new .json file outside the dataset")
    record = {"schema_version": 1, "status": "running", "kind": args.kind,
              "dataset": str(dataset), "data_root": str(root), "receipt": str(receipt),
              "parameters": {"kind": args.kind, "start": args.start, "end": args.end},
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_before": source_identity(), "executable": sys.executable,
              "timer_boundary": "monotonic around ordinary Job.run, including report rendering",
              "historical_reference": {"seconds": 121.296907,
                  "scope": "2021-2025 signed TOP3000, 1255 sessions, ten detailed charts",
                  "speed_gate": False, "numerical_parity_checked": False}}
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    report_path = None
    try:
        start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
        if start >= end or args.start != start.isoformat() or args.end != end.isoformat():
            raise ValueError("Use ordered ISO start/end dates")
        if root.is_relative_to(dataset):
            raise ValueError("Output root cannot be inside dataset")
        if args.kind == "backtest" and root != dataset.parent.parent:
            raise ValueError("Ordinary backtest requires --data-root = dataset.parent.parent")
        output_root = root
        if args.kind == "lab":
            output_root = inside(root / ("console-recovery-lab-" + uuid4().hex), root, dataset)
            output_root.mkdir(exist_ok=False)
        # Reject redirected worker destinations before launching any writes.
        inside(output_root / ("alphalab" if args.kind == "lab" else "runs"), root, dataset)
        request = request_for(args, dataset, output_root)
        record["request"] = request
        record["output_root"] = str(output_root)
        persist(receipt, record)

        def on_line(line):
            nonlocal report_path
            print(line, flush=True)
            if line.startswith("Report: "):
                report_path = Path(line[len("Report: "):].strip())

        job = Job(request)
        print(f"Starting {args.kind}: {args.start} through {args.end}", flush=True)
        started = time.monotonic()
        try:
            result = job.run(on_line)
        except BaseException:
            record["worker_cleanup"] = stop_failed_job(job)
            raise
        finally:
            record["job_seconds"] = time.monotonic() - started
        record["result"] = str(result)
        record["report_path"] = str(report_path) if report_path is not None else None
        result = inside(result, root, dataset)
        record["acceptance"] = (validate_lab(result, request, root, dataset) if args.kind == "lab"
                                else validate_backtest(result, report_path, request, root, dataset))
        record["status"] = "completed"
    except (Exception, KeyboardInterrupt) as error:
        record["status"] = "failed"
        record["error"] = {"type": type(error).__name__, "message": str(error)}
        failure_notice(f"Acceptance failed: {error}", file=sys.stderr)
    finally:
        record["source_after"] = source_identity()
        record["source_unchanged"] = record["source_before"] == record["source_after"]
        if not record["source_unchanged"]:
            record["status"] = "failed"
            record["source_error"] = "Source identity changed during acceptance"
        record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        persist(receipt, record)
    failure_notice(f"{record['status']}: receipt {receipt}")
    return record


def main(argv=None) -> int:
    configure_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--kind", choices=("lab", "backtest"), required=True)
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return 0 if run(args)["status"] == "completed" else 1
    except (Exception, KeyboardInterrupt) as error:
        failure_notice(f"Cannot create acceptance receipt: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
