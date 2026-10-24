"""Submitted-attempt integration; synthetic results need no provider/Qlib."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

NUMERICAL = bool(importlib.util.find_spec("numpy") and importlib.util.find_spec("pandas"))
if NUMERICAL:
    from qlib_yahoo.alphalab import batch
from qlib_yahoo.alphalab.settings import Settings
from qlib_yahoo.dataset import DatasetError
from qlib_yahoo.workflow_provenance import lab_implementation, request_value
from qlib_yahoo.workflow_records import EvidenceStore


class LabWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.settings = Settings()
        self.result = {
            "id": "candidate", "engine": "fixture", "expression": "rank(close)",
            "settings": {}, "summary": {"sharpe": 0.1},
            "checks": [{"name": "SELF_CORRELATION", "value": 0.3}],
            "daily": {"dates": ["2020-01-02"], "pnl": [0.0], "turnover": [0.0]},
            "self_correlation": 0.3, "most_correlated_id": "other",
        }

    def run_fixture(self, expressions=None, *, cached=False):
        result = copy.deepcopy(self.result)
        if cached:
            result["cached"] = True
            result["self_correlation"] = 0.7
        with patch.object(batch, "_run_batch", return_value=[result]):
            return batch.run_batch(self.root / "dataset",
                                   expressions or ["rank(close)"], self.settings,
                                   data_root=self.root, workers=1)

    @unittest.skipUnless(NUMERICAL, "optional numpy/pandas environment")
    def test_repeat_cache_and_forced_attempts_do_not_replace_evidence(self):
        first = self.run_fixture()[0]
        again = self.run_fixture(cached=True)[0]
        self.assertNotEqual(first["attempt_id"], again["attempt_id"])
        self.assertEqual(first["result_digest"], again["result_digest"])
        store = EvidenceStore(self.root)
        self.assertEqual(store.get(first["attempt_id"])["status"], "completed")
        self.assertEqual(store.get(again["attempt_id"])["status"], "reused")
        self.assertEqual(len(store.list_attempts(candidate_id=first["candidate_id"])), 2)
        self.assertNotIn("self_correlation", store.get_result(first["result_digest"]))
        changed = copy.deepcopy(self.result)
        changed["summary"]["sharpe"] = -0.5
        with patch.object(batch, "_run_batch", return_value=[changed]):
            third = batch.run_batch(self.root, ["rank(close)"], self.settings,
                                    data_root=self.root, force=True)[0]
        self.assertNotEqual(first["result_digest"], third["result_digest"])
        self.assertEqual(store.get_result(first["result_digest"])["summary"]["sharpe"], 0.1)

    @unittest.skipUnless(NUMERICAL, "optional numpy/pandas environment")
    def test_bad_syntax_and_empty_input_are_recorded_without_panels(self):
        with patch.object(batch, "Workspace", side_effect=AssertionError("must not load panels")):
            results = batch.run_batch(self.root / "absent", ["rank(", "", None],
                                      self.settings, data_root=self.root)
        store = EvidenceStore(self.root)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(store.get(r["attempt_id"])["status"] == "rejected" for r in results))
        self.assertTrue(all(r["error"] for r in results))
        with self.assertRaises(DatasetError):
            batch.run_batch(self.root, [], self.settings, data_root=self.root)
        self.assertTrue(any(r["status"] == "rejected" and r["kind"] == "lab-batch"
                            for r in store.list_attempts()))

    @unittest.skipUnless(NUMERICAL, "optional numpy/pandas environment")
    def test_preparation_failure_and_cancellation_preserve_submissions(self):
        def failure(*args, **kwargs):
            kwargs["diagnostics"]["phase"] = "preparation"
            raise DatasetError("fixture missing price source")
        with patch.object(batch, "_run_batch", side_effect=failure), self.assertRaises(DatasetError):
            batch.run_batch(self.root, ["rank(close)"], self.settings, data_root=self.root)
        store = EvidenceStore(self.root)
        self.assertEqual({r["status"] for r in store.list_attempts()}, {"failed"})
        with patch.object(batch, "_run_batch", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            batch.run_batch(self.root, ["rank(close)"], self.settings, data_root=self.root)
        self.assertEqual(sum(r["status"] == "cancelled" for r in store.list_attempts()), 2)

    @unittest.skipUnless(NUMERICAL, "optional numpy/pandas environment")
    def test_source_version_changes_cache_key_and_midrun_change_fails(self):
        first = {"sha256": "a" * 64}
        other = {"sha256": "b" * 64}
        self.assertNotEqual(batch.alpha_id("close", self.settings, "source", first),
                            batch.alpha_id("close", self.settings, "source", other))
        with patch("qlib_yahoo.workflow_provenance.lab_implementation", side_effect=[first, other]):
            with patch.object(batch, "_run_batch", return_value=[self.result]), self.assertRaisesRegex(
                    DatasetError, "implementation changed"):
                batch.run_batch(self.root, ["rank(close)"], self.settings, data_root=self.root)
        self.assertFalse(any(r["status"] == "completed" for r in EvidenceStore(self.root).list_attempts()))

    @unittest.skipUnless(NUMERICAL, "optional numpy/pandas environment")
    def test_receipt_freezes_versions_and_parent(self):
        store = EvidenceStore(self.root)
        parent = store.begin("console-request", {})
        with patch.object(batch, "_run_batch", return_value=[copy.deepcopy(self.result)]):
            results = batch.run_batch(self.root, ["rank(close)"], self.settings,
                                      data_root=self.root, parents=[parent["id"]])
        receipt = json.loads(batch.save_batch(self.root, results, self.settings, str(self.root)).read_text())
        self.assertEqual(receipt["evidence"][0]["result_digest"], results[0]["result_digest"])
        self.assertEqual(store.get(results[0]["batch_attempt_id"])["parents"], [parent["id"]])

    @unittest.skipUnless(NUMERICAL, "optional numpy/pandas environment")
    def test_source_inventory_is_finite(self):
        identity = lab_implementation()
        self.assertIn("alphalab/ops.py", identity["files"])
        self.assertEqual(len(identity["sha256"]), 64)

    def test_invalid_number_is_explicit(self):
        self.assertEqual(request_value(float("nan")), {"invalid_number": "nan"})


if __name__ == "__main__":
    unittest.main()
