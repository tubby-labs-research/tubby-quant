"""CLI service parity and cancellation races; no market data or real workers."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from qlib_yahoo import workflow_cli as cli
from qlib_yahoo.dataset import DatasetError
from qlib_yahoo.jobs import Job, JobCancelled
from qlib_yahoo.webapp.service import ConsoleService
from qlib_yahoo.workflow_records import EvidenceStore


def snapshot(state="running", *, finished=None, error=None, **updates):
    return {"id": "job", "state": state, "finished": finished, "error": error,
            "log": [], "log_index": 0, "attempt_id": "attempt",
            "result": "saved/result", "receipt": {"status": "completed"}, **updates}


class ServiceJobWaitTests(unittest.TestCase):
    def service(self, *states):
        service = Mock()
        service.start_chronology_trial.return_value = {"id": "job"}
        service.job.side_effect = states
        return service

    def test_startup_cancel_false_then_true_is_retried_with_pause(self):
        service = self.service(KeyboardInterrupt(), snapshot(), snapshot(),
                               snapshot("cancelled", finished=1, error="stopped"))
        service.cancel.side_effect = [False, True]
        with patch.object(cli.time, "sleep") as pause:
            with self.assertRaisesRegex(JobCancelled, "stopped"):
                cli._chronology_result(service, "trial", {})
        self.assertEqual(service.cancel.call_count, 2)
        self.assertEqual(pause.call_count, 3)
        pause.assert_called_with(.05)

    def test_terminal_before_cancel_keeps_actual_outcome(self):
        service = self.service(snapshot(), snapshot("completed", finished=1))
        with patch.object(cli.time, "sleep", side_effect=KeyboardInterrupt()):
            result = cli._chronology_result(service, "trial", {})
        service.cancel.assert_not_called()
        self.assertEqual(result, {"schema_version": 1, "result_kind": "chronology",
                                 "job_id": "job", "attempt_id": "attempt",
                                 "run_path": "saved/result", "receipt": {"status": "completed"}})

    def test_false_cancel_then_terminal_does_not_retry(self):
        service = self.service(KeyboardInterrupt(), snapshot("completed", finished=1))
        service.cancel.return_value = False
        with patch.object(cli.time, "sleep"):
            self.assertEqual(cli._chronology_result(service, "trial", {})["run_path"], "saved/result")
        service.cancel.assert_called_once()

    def test_terminal_state_without_finished_keeps_waiting(self):
        service = self.service(snapshot("completed"), snapshot("completed", finished=1))
        with patch.object(cli.time, "sleep") as pause:
            cli._chronology_result(service, "trial", {})
        pause.assert_called_once_with(.05)

    def test_repeated_interrupt_during_cancel_stays_pending(self):
        service = self.service(KeyboardInterrupt(), snapshot(), snapshot("cancelled", finished=1))
        service.cancel.side_effect = [KeyboardInterrupt(), True]
        with patch.object(cli.time, "sleep"):
            with self.assertRaises(JobCancelled):
                cli._chronology_result(service, "trial", {})
        self.assertEqual(service.cancel.call_count, 2)

    def test_errors_are_preserved_and_logs_stay_on_stderr(self):
        for state, exception in (("failed", DatasetError), ("cancelled", JobCancelled)):
            with self.subTest(state=state):
                service = self.service(snapshot(state, finished=1, error="seal failure",
                                                log=["worker detail"], log_index=1))
                output = io.StringIO()
                with contextlib.redirect_stderr(output), self.assertRaisesRegex(exception, "seal failure"):
                    cli._chronology_result(service, "trial", {})
                self.assertEqual(output.getvalue(), "worker detail\n")

    def test_incremental_log_cursor(self):
        service = self.service(snapshot(log=["first"], log_index=4),
                               snapshot("completed", finished=1, log=["last"], log_index=5))
        output = io.StringIO()
        with contextlib.redirect_stderr(output), patch.object(cli.time, "sleep"):
            cli._chronology_result(service, "trial", {})
        self.assertEqual([call.args for call in service.job.call_args_list], [("job", 0), ("job", 4)])
        self.assertEqual(output.getvalue(), "first\nlast\n")


class WorkflowCLIEntrypointTests(unittest.TestCase):
    def args(self, command, request=None):
        return SimpleNamespace(workflow_command=command, data_root=Path("unused"), request=request,
                               attempt_id="candidate", initial_cash=100000.,
                               cost_per_side=.0005, annual_borrow_rate=.004)

    def test_all_execution_commands_use_service_starts(self):
        for command, method in (("chronology-trial", "start_chronology_trial"),
                                ("chronology-final", "start_chronology_final"),
                                ("validate", "start_candidate_validation"),
                                ("portfolio-run", "start_portfolio_validation")):
            with self.subTest(command=command):
                request = Mock()
                request.read_bytes.return_value = b'{"fixture": true}'
                service = Mock()
                getattr(service, method).return_value = {"id": "job"}
                service.job.return_value = snapshot("completed", finished=1)
                output = io.StringIO()
                with patch("qlib_yahoo.webapp.service.ConsoleService", return_value=service), \
                        patch("qlib_yahoo.jobs.Job", side_effect=AssertionError("direct Job bypass")), \
                        contextlib.redirect_stdout(output):
                    cli.main(self.args(command, request))
                expected = ({"attempt_id": "candidate", "initial_cash": 100000.,
                             "cost_per_side": .0005, "annual_borrow_rate": .004}
                            if command == "validate" else {"fixture": True})
                getattr(service, method).assert_called_once_with(expected)
                self.assertEqual(json.loads(output.getvalue())["run_path"], "saved/result")
                if command in ("validate", "portfolio-run"):
                    self.assertEqual(json.loads(output.getvalue()), {"run_path": "saved/result"})

    def test_strict_json_for_portfolio_and_chronology(self):
        for command in ("portfolio-preview", "portfolio-run", "chronology-preview", "chronology-trial"):
            for raw in (b'{"x": 1, "x": 2}', b'{"x": NaN}', b'{"x": Infinity}'):
                with self.subTest(command=command, raw=raw):
                    request = Mock()
                    request.read_bytes.return_value = raw
                    service = Mock()
                    with patch("qlib_yahoo.webapp.service.ConsoleService", return_value=service), \
                            self.assertRaises(ValueError):
                        cli.main(self.args(command, request))
                    self.assertEqual(service.method_calls, [])

    def test_chronology_non_job_compatibility(self):
        for action in ("preview", "declare", "freeze", "view"):
            with self.subTest(action=action):
                service = Mock()
                getattr(service, "workflow_chronology_" + action).return_value = {"status": "retained"}
                self.assertEqual(cli._chronology_result(service, action, {}), {"status": "retained"})
                service.job.assert_not_called()

    def test_decorated_rejections_are_retained_without_starting_job(self):
        for method, kind, preview in (
                ("start_candidate_validation", "console-candidate-validation", "workflow_preview"),
                ("start_portfolio_validation", "console-portfolio-validation", "workflow_portfolio_preview"),
                ("start_chronology_trial", "console-chronology-trial", None),
                ("start_chronology_final", "console-chronology-final", None)):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                service = ConsoleService(Path(directory), job_factory=Mock())
                with contextlib.ExitStack() as stack:
                    if preview:
                        stack.enter_context(patch.object(service, preview, side_effect=DatasetError("refused")))
                    with self.assertRaises((DatasetError, ValueError)):
                        cli._service_job_result(service, method, {} if preview else [])
                service.job_factory.assert_not_called()
                records = EvidenceStore(Path(directory)).list_attempts()
                self.assertEqual(len(records), 1)
                self.assertEqual((records[0]["kind"], records[0]["status"]), (kind, "rejected"))


class RealServiceStartupTests(unittest.TestCase):
    def test_real_job_startup_barrier_and_evidence_seal(self):
        # Real Job.run/cancel and ConsoleService boundaries; substitute only the
        # process body and termination, so no subprocess or market data is used.
        constructing, release_factory = threading.Event(), threading.Event()
        process_ready, stopped = threading.Event(), threading.Event()
        seal_entered, release_seal = threading.Event(), threading.Event()
        real_sleep = cli.time.sleep

        def body(job, on_line, request_file):
            job.process = Mock()
            job.process.poll.return_value = None
            process_ready.set()
            if not stopped.wait(5):
                raise RuntimeError("cancel not accepted")
            job._process_waited = True
            raise JobCancelled("barrier stop")

        def factory(request):
            constructing.set()
            if not release_factory.wait(5):
                raise RuntimeError("factory not released")
            return Job(request)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = ConsoleService(root, job_factory=factory)
            store = EvidenceStore(root)
            original_job, original_cancel = service.job, service.cancel
            original_finish = EvidenceStore.finish
            polls, cancellations = [], []

            def poll(identity, since=0):
                if not polls:
                    polls.append(identity)
                    self.assertTrue(constructing.wait(5))
                    raise KeyboardInterrupt()
                return original_job(identity, since)

            def cancel():
                accepted = original_cancel()
                cancellations.append(accepted)
                if len(cancellations) == 1:
                    self.assertFalse(accepted)
                    release_factory.set()
                    self.assertTrue(process_ready.wait(5))
                return accepted

            def finish(evidence, identity, status, **kwargs):
                seal_entered.set()
                if not release_seal.wait(5):
                    raise RuntimeError("seal not released")
                return original_finish(evidence, identity, status, **kwargs)

            def pause(seconds):
                if stopped.is_set() and not release_seal.is_set():
                    self.assertTrue(seal_entered.wait(5))
                    self.assertIsNone(original_job(service.current.id)["finished"])
                    self.assertEqual(store.get(service.current.attempt_id)["status"], "pending")
                    release_seal.set()
                real_sleep(seconds)

            with patch.object(Job, "_run", body), \
                    patch("qlib_yahoo.jobs._terminate_tree", side_effect=lambda *args: stopped.set()), \
                    patch.object(service, "job", side_effect=poll), \
                    patch.object(service, "cancel", side_effect=cancel), \
                    patch.object(EvidenceStore, "finish", finish), \
                    patch.object(cli.time, "sleep", side_effect=pause), \
                    contextlib.redirect_stderr(io.StringIO()):
                try:
                    with self.assertRaisesRegex(JobCancelled, "barrier stop"):
                        cli._chronology_result(service, "trial", {})
                finally:
                    release_factory.set()
                    stopped.set()
                    release_seal.set()
                    # Bound cleanup even if an assertion interrupted the waiter.
                    if service.current:
                        for _ in range(100):
                            if original_job(service.current.id)["finished"] is not None:
                                break
                            real_sleep(.01)
            self.assertEqual(cancellations, [False, True])
            self.assertIsNotNone(original_job(service.current.id)["finished"])
            self.assertEqual(store.get(service.current.attempt_id)["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
