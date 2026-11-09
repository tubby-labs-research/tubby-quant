"""Isolated worker processes: request plumbing, failures, progress parsing and cancellation."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from qlib_yahoo.dataset import DatasetError
from qlib_yahoo.jobs import ACTIVE_BATCH, ACTIVE_RUN, FAILURE, RESULT, Job, JobCancelled, child_main, progress_fraction


class JobUnitTests(unittest.TestCase):
    def test_large_sec_request_file_roundtrip_and_success_cleanup(self):
        bindings = {f"sec_{i}": {"tag": "Assets", "taxonomy": "us-gaap/", "unit": "USD",
                    "qtrs": 0, "segments": "", "coreg": ""} for i in range(1000)}
        for kind in ("backtest", "backtest-batch", "alpha-batch"):
            with self.subTest(kind=kind):
                request = {"kind": kind, "alpha": "$sec_0 / $close /* confidential-marker-9c1e */",
                           "sec": {"catalog": "D:/Qlib/sec/catalogs/fixture", "bindings": bindings}}
                job = Job(request)
                paths = []
                class Child:
                    def __init__(self, args, **kwargs):
                        self.args = args
                        self.assertions = kwargs
                        paths.append(Path(args[-1]))
                        self.payload = json.loads(paths[-1].read_text(encoding="utf-8"))
                        output = io.StringIO()
                        with contextlib.redirect_stdout(output), \
                                patch("qlib_yahoo.jobs._run_request", return_value=Path("receipt")) as dispatch:
                            self.code = child_main(args[3:])
                            self.received = dispatch.call_args.args[0]
                        self.stdout = io.StringIO(output.getvalue())
                    def poll(self):
                        return self.code
                    def wait(self):
                        return self.code
                lines = []
                with patch("qlib_yahoo.jobs.subprocess.Popen", Child):
                    self.assertEqual(job.run(lines.append), Path("receipt"))
                self.assertEqual(job.process.args[3], "--request-file")
                self.assertLess(len(subprocess.list2cmdline(job.process.args)), 32767)
                self.assertNotIn("confidential-marker-9c1e", " ".join(job.process.args))
                self.assertFalse(job.process.assertions.get("shell", False))
                self.assertEqual(job.process.received["sec"], request["sec"])
                self.assertEqual(job.process.received["alpha"], request["alpha"])
                self.assertFalse(paths[0].exists())
                self.assertFalse(paths[0].parent.exists())
                self.assertEqual(len(lines), 1)
                timing_prefix = ("Full worker elapsed (child_main entry through request parsing, execution, "
                                 "save/report; excludes process startup/parent UI): ")
                self.assertTrue(lines[0].startswith(timing_prefix), lines[0])
                self.assertNotIn("confidential-marker-9c1e", lines[0])
                self.assertNotIn(request["alpha"], lines[0])
                diagnostics = json.loads(lines[0][len(timing_prefix):])["diagnostics"]
                self.assertEqual(set(diagnostics), {"version", "worker"})
                self.assertEqual(diagnostics["version"], 1)
                worker = diagnostics["worker"]
                self.assertEqual(set(worker), {"scope", "outcome", "elapsed_seconds"})
                self.assertEqual(worker["scope"], "child_main")
                self.assertEqual(worker["outcome"], "completed")
                self.assertIn(type(worker["elapsed_seconds"]), (int, float))
                self.assertGreaterEqual(worker["elapsed_seconds"], 0)
                if job._cancel_file is not None:
                    self.assertFalse(job._cancel_file.exists())

    def test_request_file_cleanup_on_spawn_failure_without_payload_leak(self):
        job = Job({"kind": "backtest", "alpha": "private-secret-expression"})
        paths = []
        def spawn(args, **kwargs):
            paths.append(Path(args[-1]))
            self.assertTrue(paths[-1].is_file())
            raise OSError("cannot start " + subprocess.list2cmdline(args))
        with patch("qlib_yahoo.jobs.subprocess.Popen", side_effect=spawn):
            with self.assertRaises(OSError) as caught:
                job.run(lambda line: None)
        self.assertNotIn("private-secret-expression", str(caught.exception))
        self.assertFalse(paths[0].parent.exists())
        self.assertFalse(job._cancel_file.exists())

    def test_request_file_cleanup_on_child_error_and_cancel(self):
        for cancelled in (False, True):
            with self.subTest(cancelled=cancelled):
                job = Job({"kind": "backtest", "alpha": "$sec_assets"})
                paths = []
                class Child:
                    def __init__(self, args, **kwargs):
                        paths.append(Path(args[-1]))
                        self.stdout = io.StringIO("paused\n" if cancelled else FAILURE + json.dumps("worker failed") + "\n")
                    def poll(self):
                        return -15 if cancelled else 1
                    def wait(self):
                        return self.poll()
                def line(message):
                    job.cancelled = cancelled
                    if cancelled:
                        job._cancel_file.write_text("cancel")
                with patch("qlib_yahoo.jobs.subprocess.Popen", Child), \
                        self.assertRaises(JobCancelled if cancelled else DatasetError):
                    job.run(line)
                self.assertFalse(paths[0].parent.exists())
                self.assertFalse(job._cancel_file.exists())

    def test_request_file_child_enforces_size_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "request.json"
            path.write_bytes(b"private-data" * 10)
            output = io.StringIO()
            with patch("qlib_yahoo.jobs.MAX_REQUEST_BYTES", 8), contextlib.redirect_stdout(output):
                self.assertEqual(child_main(["--request-file", str(path)]), 1)
            self.assertIn("transport limit", output.getvalue())
            self.assertNotIn("private-data", output.getvalue())
        with patch("qlib_yahoo.jobs.MAX_REQUEST_BYTES", 8), self.assertRaises(DatasetError):
            Job({"kind": "backtest"})

    def test_callback_error_stops_worker_and_cleans_request(self):
        job = Job({"kind": "backtest", "alpha": "private-expression"})
        paths = []
        class Child:
            def __init__(self, args, **kwargs):
                paths.append(Path(args[-1]))
                self.stdout = io.StringIO("progress\n")
                self.ended = False
            def poll(self):
                return 1 if self.ended else None
            def wait(self):
                self.ended = True
                return 1
        def callback(line):
            raise RuntimeError("callback failed")
        with patch("qlib_yahoo.jobs.subprocess.Popen", Child), \
                patch("qlib_yahoo.jobs._terminate_tree") as terminate, \
                self.assertRaisesRegex(RuntimeError, "callback failed"):
            job.run(callback)
        terminate.assert_called_once_with(job.process)
        self.assertTrue(job.process.ended)
        self.assertTrue(job.process.stdout.closed)
        self.assertFalse(paths[0].parent.exists())
        self.assertFalse(job._cancel_file.exists())

    def test_progress_lines(self):
        self.assertEqual(progress_fraction("backtest loop:  26%|██▋       | 132/500 [00:00<00:00]"), ("backtest loop", 0.26))
        self.assertEqual(progress_fraction("backtest loop: 100%|██████████| 500/500"), ("backtest loop", 1.0))
        self.assertIsNone(progress_fraction("Backtest completed: /tmp/run"))
        self.assertIsNone(progress_fraction("Scoring 50% of names"))

    def test_child_reports_unknown_requests_as_failures(self):
        for argv in ([], ["not json"], [json.dumps({"kind": "format-disk"})], [json.dumps(["backtest"])]):
            output = io.StringIO()
            with self.subTest(argv=argv), contextlib.redirect_stdout(output):
                self.assertEqual(child_main(argv), 1)
            self.assertIn(FAILURE, output.getvalue())
            self.assertNotIn(RESULT, output.getvalue())

    def test_requests_are_json_data(self):
        with self.assertRaises(DatasetError):
            Job({"kind": "shell", "command": "rm -rf /"})
        with self.assertRaises(ValueError):
            Job({"kind": "backtest", "cash": float("nan")})
        job = Job({"kind": "backtest", "alpha": "'); import os; os.system('x') #"})
        self.assertEqual(json.loads(job.payload)["alpha"], "'); import os; os.system('x') #")

    def test_worker_failure_is_reported_not_completed(self):
        lines = []
        job = Job({"kind": "backtest", "dataset": str(Path(tempfile.gettempdir()) / "missing-dataset"), "alpha": "$close"})
        with self.assertRaises(DatasetError) as caught:
            job.run(lines.append)
        self.assertTrue(str(caught.exception))
        self.assertFalse(any(line.startswith(RESULT) for line in lines))

    def test_stop_while_rendering_keeps_the_completed_run(self):
        job = Job({"kind": "backtest", "dataset": "x", "alpha": "$close"})

        class Child:
            pid = 0

            def __init__(self, *_args, **_kwargs):
                self.stdout = iter(["Verifying dataset x ...\n", "Backtest completed: /data/runs/r1\n",
                                    "Rendering the summary report ...\n"])

            def poll(self):
                return None

            def wait(self):
                return -15

        def on_line(line):
            if line.startswith("Rendering"):
                job.cancelled = True  # what Stop does while the report renders

        with patch("qlib_yahoo.jobs.subprocess.Popen", Child):
            self.assertEqual(job.run(on_line), Path("/data/runs/r1"))

    def test_cancel_after_the_process_ended_keeps_its_outcome(self):
        job = Job({"kind": "backtest", "dataset": "x", "alpha": "$close"})
        job.process = type("Ended", (), {"poll": lambda self: 0})()
        self.assertFalse(job.cancel())
        self.assertFalse(job.cancelled)

    def test_cancel_after_a_constituent_completed_is_not_batch_completion(self):
        job = Job({"kind": "backtest-batch", "dataset": "x", "alphas": ["$close", "$open"], "data_root": "data"})
        class Child:
            pid = 0
            def __init__(self, *_args, **_kwargs):
                self.stdout = iter(["Backtest completed: /data/runs/r1\n", "[2/2] $open\n"])
            def poll(self):
                return None
            def wait(self):
                return -15
        def on_line(line):
            if line.startswith("[2/2]"):
                job.cancelled = True
        with patch("qlib_yahoo.jobs.subprocess.Popen", Child), self.assertRaises(JobCancelled):
            job.run(on_line)

    def test_child_batch_request_preserves_settings_and_reports_receipt(self):
        output = io.StringIO()
        request = {"kind": "backtest-batch", "dataset": "dataset", "data_root": "data",
                   "alphas": ["$close", "$close"], "alpha": "$close", "strategy_mode": "signed",
                   "start": "2024-01-03", "history_mode": "stream"}
        with patch("qlib_yahoo.preparation.run_backtest_batch", return_value=Path("receipt.json")) as batch:
            with contextlib.redirect_stdout(output):
                self.assertEqual(child_main([json.dumps(request)]), 0)
        batch.assert_called_once_with(Path("dataset"), ["$close", "$close"], data_root=Path("data"),
                                      strategy_mode="signed", start="2024-01-03", history_mode="stream")
        self.assertIn(RESULT, output.getvalue())
        self.assertNotIn(FAILURE, output.getvalue())

    def test_forced_cancel_repairs_only_current_running_records(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            complete, active = root / "runs/complete", root / "runs/active"
            for path, status in ((complete, "completed"), (active, "running")):
                path.mkdir(parents=True)
                (path / "manifest.json").write_text(json.dumps({"status": status}))
                (path / "history.parquet.tmp").write_bytes(b"recoverable partial history")
            receipt = root / "workbench/batches/batch.json"
            receipt.parent.mkdir(parents=True)
            receipt.write_text(json.dumps({"status": "running", "runs": [str(complete)], "errors": []}))
            job = Job({"kind": "backtest-batch", "dataset": str(root / "datasets/snapshot"), "data_root": str(root),
                       "alphas": ["$close", "$open"]})
            class Child:
                def __init__(self, *args, **kwargs):
                    payload = json.loads(Path(args[0][-1]).read_text(encoding="utf-8"))
                    self.cancel_file = Path(payload["_cancel_file"])
                    self.stdout = iter([ACTIVE_BATCH + json.dumps(str(receipt)) + "\n",
                                        ACTIVE_RUN + json.dumps(str(complete)) + "\n",
                                        "Backtest completed: " + str(complete) + "\n",
                                        ACTIVE_RUN + json.dumps(str(active)) + "\n", "paused\n"])
                def wait(self):
                    return -15
            def on_line(line):
                if line == "paused":
                    job.cancelled = True
                    job._cancel_file.write_text("cancel")
            with patch("qlib_yahoo.jobs.subprocess.Popen", Child), self.assertRaises(JobCancelled):
                job.run(on_line)
            self.assertEqual(json.loads((complete / "manifest.json").read_text())["status"], "completed")
            self.assertEqual(json.loads((active / "manifest.json").read_text())["status"], "failed")
            record = json.loads(receipt.read_text())
            self.assertEqual(record["status"], "cancelled")
            self.assertEqual(record["runs"], [str(complete)])
            self.assertEqual((active / "history.parquet.tmp").read_bytes(), b"recoverable partial history")
            self.assertFalse(job._cancel_file.exists())

    def test_cancel_repair_refuses_outside_paths_and_preserves_completed_records(self):
        from qlib_yahoo.jobs import _record_cancelled
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            outside = root / "outside/run"
            outside.mkdir(parents=True)
            outside_manifest = outside / "manifest.json"
            outside_manifest.write_text(json.dumps({"status": "running"}))
            completed = root / "data/runs/complete"
            completed.mkdir(parents=True)
            completed_manifest = completed / "manifest.json"
            completed_manifest.write_text(json.dumps({"status": "completed"}))
            payload = json.dumps({"dataset": str(root / "data/datasets/snapshot"), "data_root": str(root / "data")})
            _record_cancelled(payload, str(outside), None)
            _record_cancelled(payload, str(completed), None)
            self.assertEqual(json.loads(outside_manifest.read_text())["status"], "running")
            self.assertEqual(json.loads(completed_manifest.read_text())["status"], "completed")

    def test_cancel_before_start(self):
        job = Job({"kind": "backtest", "dataset": "x", "alpha": "$close"})
        self.assertFalse(job.cancel())
        with self.assertRaises(JobCancelled):
            job.run(lambda _line: None)

    def test_cancel_import_and_refresh_without_dataset_is_not_a_failure(self):
        for kind in ("import-table", "refresh-data"):
            with self.subTest(kind=kind):
                job = Job({"kind": kind, "data_root": "data"})

                class Child:
                    def __init__(self, *_args, **_kwargs):
                        self.stdout = iter(["paused\n"])

                    def wait(self):
                        return -15

                def on_line(_line):
                    job.cancelled = True

                with patch("qlib_yahoo.jobs.subprocess.Popen", Child), self.assertRaises(JobCancelled):
                    job.run(on_line)

    def test_cancel_during_constituent_report_keeps_sealed_run_in_receipt(self):
        from qlib_yahoo.jobs import _record_cancelled
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            dataset = root / "datasets/snapshot"
            run = root / "runs/sealed"
            run.mkdir(parents=True)
            (run / "manifest.json").write_text(json.dumps({"status": "completed", "dataset": str(dataset),
                                                           "alpha": "$close"}))
            receipt = root / "workbench/batches/batch.json"
            receipt.parent.mkdir(parents=True)
            receipt.write_text(json.dumps({"status": "cancelled", "runs": [], "errors": []}))
            payload = json.dumps({"kind": "backtest-batch", "dataset": str(dataset), "data_root": str(root),
                                  "alphas": ["$close", "$open"]})
            _record_cancelled(payload, str(run), str(receipt))
            self.assertEqual(json.loads(receipt.read_text())["runs"], [str(run)])
            self.assertEqual(json.loads(receipt.read_text())["status"], "cancelled")
            self.assertEqual(json.loads((run / "manifest.json").read_text())["status"], "completed")

    def test_atomic_repair_does_not_overwrite_a_preexisting_temporary_sibling(self):
        from qlib_yahoo.jobs import _record_cancelled
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            run = root / "runs/active"
            run.mkdir(parents=True)
            (run / "manifest.json").write_text(json.dumps({"status": "running"}))
            sentinel = run / "manifest.json.tmp"
            sentinel.write_bytes(b"external sentinel must not be overwritten")
            payload = json.dumps({"dataset": str(root / "datasets/snapshot"), "data_root": str(root)})
            _record_cancelled(payload, str(run), None)
            self.assertEqual(sentinel.read_bytes(), b"external sentinel must not be overwritten")
            self.assertEqual(json.loads((run / "manifest.json").read_text())["status"], "failed")

    def test_cancel_repairs_dataset_run_and_separate_receipt_root(self):
        from qlib_yahoo.jobs import _record_cancelled
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            dataset_root, receipt_root = root / "source", root / "receipts"
            run = dataset_root / "runs/active"
            run.mkdir(parents=True)
            (run / "manifest.json").write_text(json.dumps({"status": "running"}))
            receipt = receipt_root / "workbench/batches/batch.json"
            receipt.parent.mkdir(parents=True)
            receipt.write_text(json.dumps({"status": "running", "runs": []}))
            payload = json.dumps({"dataset": str(dataset_root / "datasets/snapshot"), "data_root": str(receipt_root)})
            _record_cancelled(payload, str(run), str(receipt))
            self.assertEqual(json.loads((run / "manifest.json").read_text())["status"], "failed")
            self.assertEqual(json.loads(receipt.read_text())["status"], "cancelled")

    def test_cancel_repair_refuses_redirected_ancestor(self):
        from qlib_yahoo.jobs import _record_cancelled
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            run = root / "runs/active"
            run.mkdir(parents=True)
            manifest_path = run / "manifest.json"
            manifest_path.write_text(json.dumps({"status": "running"}))
            payload = json.dumps({"dataset": str(root / "datasets/snapshot")})
            original = Path.is_junction
            with patch.object(Path, "is_junction", lambda path: path == run.parent or original(path)):
                _record_cancelled(payload, str(run), None)
            self.assertEqual(json.loads(manifest_path.read_text())["status"], "running")

    def test_cancel_running_subprocess(self):
        job = Job({"kind": "backtest", "dataset": "unused", "alpha": "$close"})
        ready = threading.Event()
        outcome = {}
        real_popen = subprocess.Popen

        def waiting_child(args, **kwargs):
            if len(args) < 3 or args[1:3] != ["-m", "qlib_yahoo.jobs"]:
                return real_popen(args, **kwargs)
            cancel = json.loads(Path(args[-1]).read_text(encoding="utf-8"))["_cancel_file"]
            program = ("from pathlib import Path; import sys,time; marker=Path(sys.argv[1]); "
                       "print('child ready',flush=True)\n"
                       "while not marker.exists(): time.sleep(0.01)\n")
            return real_popen([sys.executable, "-u", "-c", program, cancel],
                              **kwargs)

        def run_job():
            try:
                outcome["result"] = job.run(lambda line: ready.set() if line == "child ready" else None)
            except Exception as error:
                outcome["error"] = error

        with patch("qlib_yahoo.jobs.subprocess.Popen", side_effect=waiting_child):
            thread = threading.Thread(target=run_job)
            thread.start()
            try:
                self.assertTrue(ready.wait(15), "child did not become ready")
                self.assertTrue(job.cancel(), "child exited before cancellation")
                thread.join(30)
                self.assertFalse(thread.is_alive(), "cancel did not stop the child")
                self.assertIsInstance(outcome.get("error"), JobCancelled)
                self.assertNotIn("result", outcome)
                self.assertIsNotNone(job.process.poll())
            finally:
                if job.process is not None and job.process.poll() is None:
                    job.process.kill()
                thread.join(5)


@unittest.skipUnless(importlib.util.find_spec("qlib") and sys.version_info < (3, 13), "optional Qlib runtime unavailable")
class JobProcessTests(unittest.TestCase):
    def test_real_batch_cancel_after_first_completed_keeps_partial_receipt(self):
        from tests.test_qlib_yahoo_execution import _snapshot
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            snapshot = _snapshot(root, sessions_count=35, include_benchmark=False)
            job = Job({"kind": "backtest-batch", "dataset": str(snapshot), "data_root": str(root),
                       "alphas": ["$close", "$open", "$high"], "topk": 2, "n_drop": 1,
                       "warmup_sessions": 20, "holdings": "never"})
            cancelled = []
            def on_line(line):
                if line.startswith("[2/3]"):
                    cancelled.append(job.cancel())
            with self.assertRaises(JobCancelled):
                job.run(on_line)
            self.assertEqual(cancelled, [True])
            receipt = next((root / "workbench/batches").glob("*.json"))
            record = json.loads(receipt.read_text())
            self.assertEqual(record["status"], "cancelled")
            self.assertEqual(len(record["runs"]), 1)
            completed = Path(record["runs"][0])
            self.assertEqual(json.loads((completed / "manifest.json").read_text())["status"], "completed")
            self.assertTrue((completed / "daily_report.csv").is_file())
            self.assertNotIn("running", [json.loads(p.read_text())["status"] for p in (root / "runs").glob("*/manifest.json")])
            self.assertFalse(job._cancel_file.exists())

    def test_real_backtest_in_worker_and_cancellation(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_qlib_yahoo_execution import _snapshot

        with tempfile.TemporaryDirectory() as temporary:
            snapshot = _snapshot(Path(temporary).resolve(), sessions_count=260)
            lines = []
            run = Job({"kind": "backtest", "dataset": str(snapshot), "alpha": "$close/Ref($close, 1)-1",
                       "topk": 2, "n_drop": 1, "warmup_sessions": 20}).run(lines.append)
            self.assertEqual(json.loads((run / "manifest.json").read_text())["status"], "completed")
            self.assertTrue(any(progress_fraction(line) for line in lines), "tqdm progress lines reach the parent")
            self.assertFalse(any(line.startswith("Gym has been unmaintained") for line in lines))

            job = Job({"kind": "backtest", "dataset": str(snapshot), "alpha": "$close/Ref($close, 1)-1",
                       "topk": 2, "n_drop": 1, "warmup_sessions": 20})
            started = threading.Event()
            outcome = {}

            def run_job():
                try:
                    outcome["result"] = job.run(lambda _line: started.set())
                except Exception as error:  # captured for the assertion below
                    outcome["error"] = error

            thread = threading.Thread(target=run_job)
            thread.start()
            self.assertTrue(started.wait(60))
            cancelled = job.cancel()
            thread.join(30)
            self.assertFalse(thread.is_alive())
            if "result" in outcome:
                self.assertNotIn("error", outcome)
                self.assertEqual(json.loads((outcome["result"] / "manifest.json").read_text())["status"], "completed")
            else:
                self.assertTrue(cancelled)
                self.assertIsInstance(outcome.get("error"), JobCancelled)
            time.sleep(0.2)
            self.assertIsNotNone(job.process.poll())
            # Cooperative cancellation or parent fallback seals interrupted records on every OS.
            statuses = [json.loads(path.read_text())["status"]
                        for path in (snapshot.parent.parent / "runs").glob("*/manifest.json")]
            self.assertNotIn("running", statuses)


if __name__ == "__main__":
    unittest.main()
