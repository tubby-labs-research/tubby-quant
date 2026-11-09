"""Cheap console rejections, asynchronous outcomes and worker lineage."""
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from qlib_yahoo.webapp.service import ConsoleService, ServiceError
from qlib_yahoo.workflow_records import EvidenceStore
from qlib_yahoo.workflow_submission import parent_attempts, run_worker_submission

NUMERICAL = bool(importlib.util.find_spec("numpy") and importlib.util.find_spec("pandas"))


class WorkflowSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.store = EvidenceStore(self.root)

    @unittest.skipUnless(NUMERICAL, "optional numpy/pandas environment")
    def test_typing_does_not_create_attempt(self):
        service = ConsoleService(self.root)
        service.alpha_check("rank(")
        self.assertFalse((self.root / "workflow").exists())

    def test_invalid_run_creates_rejected_attempt(self):
        service = ConsoleService(self.root)
        with self.assertRaises(ServiceError):
            service.start_alpha({"expressions": ""})
        records = self.store.list_attempts()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "rejected")
        self.assertEqual(records[0]["kind"], "console-lab")

    def test_listing_never_loads_result_payloads_but_detail_verifies(self):
        attempt = self.store.begin("lab-evaluation", {})
        self.store.finish(attempt["id"], "completed", result={"daily": {"pnl": [1]}})
        service = ConsoleService(self.root)
        with patch.object(EvidenceStore, "get_result", side_effect=AssertionError("must not load result")):
            rows = service.workflow_attempts()
            self.assertEqual(rows[0]["integrity_scope"], "metadata_only_open_to_verify_result")
            with self.assertRaises(AssertionError):
                service.workflow_detail(attempt["id"])

    def test_legacy_unsupported_backtest_is_recorded_before_rejection(self):
        service = ConsoleService(self.root)
        with self.assertRaisesRegex(ServiceError, "Unsupported PR45"):
            service.start_backtest({"alpha": "$close", "fast_settings": {}})
        record = self.store.get(self.store.list_attempts()[0]["id"])
        self.assertEqual(record["status"], "rejected")
        self.assertIn("fast_settings", record["request"]["parameters"])

    def test_async_result_and_worker_parent(self):
        captured = []

        class Job:
            def __init__(inner, request):
                captured.append(request)
            def cancel(inner):
                return False
            def run(inner, progress):
                return self.root / "runs" / "fixture"

        service = ConsoleService(self.root, job_factory=Job)
        request = {"kind": "backtest", "dataset": str(self.root / "datasets" / "fixture")}
        with patch.object(service, "_backtest_request", return_value=(request, [], Path("fixture"))):
            reply = service.start_backtest({"alpha": "$close"})
        record = service.jobs[reply["id"]]
        for _ in range(100):
            if record.finished is not None:
                break
            time.sleep(.01)
        self.assertIsNotNone(record.finished)
        self.assertEqual(record.state, "completed")
        self.assertEqual(captured[0]["_workflow"]["parents"], [reply["attempt_id"]])
        stored = self.store.get(reply["attempt_id"])
        self.assertEqual(stored["status"], "completed")
        self.assertEqual(self.store.get_result(stored["result_digest"])["result_path"], str(record.result))

    def test_worker_context_is_scoped_and_persists_failure(self):
        parent = self.store.begin("console-lab", {})
        request = {"kind": "alpha-batch", "data_root": str(self.root),
                   "_workflow": {"data_root": str(self.root), "parents": [parent["id"]]}}
        children = []
        def dispatch(payload):
            self.assertNotIn("_workflow", payload)
            children.extend(parent_attempts())
            raise ValueError("preparation fixture failure")
        with self.assertRaises(ValueError):
            run_worker_submission(request, dispatch)
        self.assertIsNone(parent_attempts())
        self.assertEqual(len(children), 1)
        self.assertEqual(self.store.get(children[0])["status"], "failed")
        self.assertEqual(self.store.get(children[0])["parents"], [parent["id"]])

    @unittest.skipUnless(NUMERICAL, "optional numpy/pandas environment")
    def test_batch_reads_frozen_version_not_replaced_library(self):
        attempt = self.store.begin("lab-evaluation", {})
        result = {"id": "old", "expression": "close", "summary": {"sharpe": -1},
                  "daily": {"dates": ["2020-01-02"], "pnl": [1]}}
        outcome = self.store.finish(attempt["id"], "completed", result=result)
        folder = self.root / "alphalab" / "batches"
        folder.mkdir(parents=True)
        receipt = folder / "fixture.json"
        receipt.write_text(json.dumps({
            "ids": ["old"], "settings": {}, "created_at_utc": "fixture",
            "evidence": [{"attempt_id": attempt["id"], "result_digest": outcome["result_digest"]}],
        }))
        service = ConsoleService(self.root)
        with patch.object(service, "_alpha_library", side_effect=AssertionError("mutable library read")):
            view = service.alpha_batch(str(receipt))
        self.assertEqual(view["rows"][0]["summary"]["sharpe"], -1)
        self.assertEqual(view["rows"][0]["evidence_status"], "immutable")


class WorkflowHTTPTests(unittest.TestCase):
    def setUp(self):
        from qlib_yahoo.webapp.server import start_server
        from test_qlib_webapp import ServerSecurityTests
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.service = ConsoleService(self.root)
        self.server, thread = start_server(self.service)
        self.addCleanup(self.server.server_close)
        self.addCleanup(thread.join, 5)
        self.addCleanup(self.server.shutdown)
        self.port = self.server.server_address[1]
        self.request = ServerSecurityTests.request.__get__(self)
        self.cookie = ServerSecurityTests.cookie.__get__(self)
        self.headers = {**self.cookie(), "X-Token": self.server.token,
                        "Origin": self.server.origin, "Content-Type": "application/json"}

    def post(self, name, params):
        response, data = self.request("POST", "/api/" + name, headers=self.headers,
                                      body=json.dumps(params))
        return response.status, json.loads(data)

    def test_http_pr45_run_refusal_is_recorded_not_discarded_by_router(self):
        for endpoint, params in (
                ("backtest", {"alpha": "$close", "fast_settings": {}}),
                ("alpha-run", {"expressions": "close", "fast_settings": {}})):
            status, value = self.post(endpoint, params)
            self.assertEqual(status, 400, value)
        records = EvidenceStore(self.root).list_attempts()
        self.assertEqual(len(records), 2)
        self.assertTrue(all(row["status"] == "rejected" for row in records))

    def test_preview_is_read_only_and_bad_portfolio_run_is_retained(self):
        status, _ = self.post("workflow-portfolio-preview", {})
        self.assertEqual(status, 400)
        self.assertFalse((self.root / "workflow").exists())
        status, _ = self.post("workflow-portfolio-run", {})
        self.assertEqual(status, 400)
        records = EvidenceStore(self.root).list_attempts()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["kind"], "console-portfolio-validation")
        self.assertEqual(records[0]["status"], "rejected")


if __name__ == "__main__":
    unittest.main()
