"""Offline fault injection across the actual controller, registry and model contract."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoresearch.config import validate_config
from autoresearch.controller import Controller


def candidate(expression="rank(close)", title="Fixture hypothesis"):
    return {"title": title, "mechanism": "Fixture price prediction mechanism", "family": "fixture",
            "expression": expression, "source_ids": ["fixture_source"],
            "evidence": [{"source_id": "fixture_source", "quote": "price patterns"}],
            "required_fields": ["close"], "expected_horizon": "one session", "rationale": "Offline fixture"}


def result(sharpe=-0.5):
    metrics = {"days": 100, "sharpe": sharpe, "returns": 0.1 if sharpe > 0 else -0.1,
               "turnover": 0.2, "drawdown": 0.1}
    return {"status": "ok", "metrics": {"development": metrics, "validation": metrics},
            "periods": {"development": metrics, "validation": metrics}, "settings": {"cost_bps": 5}}


class FakeBackend:
    def __init__(self, values=None):
        self.calls = []
        self.values = values or {}
        self.failures = 0

    def identity(self):
        return {"fixture": "dataset-v1", "cost_bps": 5}

    def validate(self, expression):
        return ["unclosed call"] if expression == "rank(" else []

    def evaluate(self, expression):
        self.calls.append(expression)
        if self.failures:
            self.failures -= 1
            raise RuntimeError("injected worker interruption")
        return copy.deepcopy(self.values.get(expression, result()))


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.cfg = {"name": "fixture-campaign", "research_root": str(root / "research"),
                    "runtime_root": str(root / "runtime"), "backend": {},
                    "discovery": {"network_enabled": False, "fixture_sources": [
                        {"id": "fixture_source", "title": "Price prediction fixture",
                         "url": "https://example.invalid/fixture", "text": "Fixture price patterns support test plumbing."}]},
                    "model": {"provider": "fixture", "hypotheses": [candidate()]},
                    "limits": {"max_experiments": 10, "max_hypotheses": 10, "max_per_family": 10,
                               "max_model_calls": 10, "daily_model_calls": 10},
                    "policy": {"min_validation_days": 60}}

    def drain(self, controller):
        outcomes = []
        for _ in range(30):
            outcome = controller.tick()
            outcomes.append(outcome)
            if not controller.status()["queued"] or outcome["status"] in {"budget", "complete"}:
                return outcomes
        self.fail("campaign did not drain")

    def test_rejected_result_survives_new_campaign_and_does_not_resimulate(self):
        backend = FakeBackend()
        first = Controller(self.cfg, backend=backend)
        self.drain(first)
        self.assertEqual(backend.calls, ["rank(close)"])
        self.assertEqual(first.store.list("reviews")[0]["decision"], "reject")
        cfg = {**self.cfg, "name": "second-campaign"}
        second = Controller(cfg, backend=backend)
        self.drain(second)
        self.assertEqual(backend.calls, ["rank(close)"])
        self.assertEqual(second.status()["evaluations"], 0)
        self.assertTrue(any(e["kind"] == "duplicate_experiment" for e in second.store.list("events")))

    def test_restart_after_result_publication_recovers_without_rerun(self):
        backend = FakeBackend()
        controller = Controller(self.cfg, backend=backend)
        controller.tick()
        original = controller.store.put
        def interrupt_after_write(collection, record):
            saved = original(collection, record)
            if collection == "experiments":
                raise KeyboardInterrupt()
            return saved
        with patch.object(controller.store, "put", side_effect=interrupt_after_write):
            with self.assertRaises(KeyboardInterrupt):
                controller.tick()
        resumed = Controller(self.cfg, backend=backend)
        self.drain(resumed)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(resumed.status()["evaluations"], 1)
        self.assertEqual(len(resumed.store.list("reviews")), 1)

    def test_operational_failure_retries_without_rejecting_hypothesis(self):
        backend = FakeBackend()
        backend.failures = 1
        controller = Controller(self.cfg, backend=backend)
        self.drain(controller)
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual({r["decision"] for r in controller.store.list("reviews")}, {"repair", "reject"})
        self.assertEqual(len(controller.store.list("experiments")), 2)

    def test_invalid_expression_is_recorded_and_repaired_not_lost_in_discovery(self):
        self.cfg["model"]["hypotheses"] = [candidate("rank(")]
        controller = Controller(self.cfg, backend=FakeBackend())
        self.drain(controller)
        records = controller.store.list("experiments")
        self.assertEqual(records[0]["result"]["error_kind"], "expression")
        self.assertEqual(controller.store.list("reviews")[0]["decision"], "repair")

    def test_promoted_peer_does_not_suppress_refinement_and_lineage(self):
        self.cfg["model"]["hypotheses"] = [candidate(), candidate("-rank(close)", "Weak peer")]
        child = {**candidate("rank(ts_delta(close, 5))", "Targeted child"),
                 "parent_experiment_id": "$parent", "question": "Does differencing reduce trend exposure?",
                 "expected_change": "Lower turnover with retained prediction"}
        self.cfg["model"]["refinements"] = [child]
        backend = FakeBackend({"rank(close)": result(2), "-rank(close)": result(.5)})
        controller = Controller(self.cfg, backend=backend)
        self.drain(controller)
        self.assertEqual(len(backend.calls), 3)
        self.assertEqual({r["decision"] for r in controller.store.list("reviews")}, {"promote", "refine", "reject"})
        descendants = [h for h in controller.store.list("hypotheses") if h.get("parent_experiment_id")]
        self.assertEqual(len(descendants), 1)
        self.assertIsNotNone(controller.store.get("experiments", descendants[0]["parent_experiment_id"]))

    def test_duplicate_formula_queued_only_once(self):
        self.cfg["model"]["hypotheses"] = [candidate(), candidate(" rank( (close) ) ", "Paraphrased")]
        backend = FakeBackend()
        controller = Controller(self.cfg, backend=backend)
        self.drain(controller)
        self.assertEqual(len(backend.calls), 1)

    def test_budget_reserved_before_failed_model_call(self):
        self.cfg["limits"]["max_model_calls"] = 1
        self.cfg["model"]["hypotheses"] = [{"invalid": True}]
        controller = Controller(self.cfg, backend=FakeBackend())
        self.assertEqual(controller.tick()["status"], "error")
        self.assertEqual(controller.status()["model_calls"], 1)
        self.assertEqual(controller.tick()["status"], "complete")

    def test_records_are_portable_and_sources_keep_only_metadata(self):
        controller = Controller(self.cfg, backend=FakeBackend())
        self.drain(controller)
        for path in Path(self.cfg["research_root"]).rglob("*.json"):
            self.assertNotIn(self.temp.name, path.read_text())
        source = controller.store.list("sources")[0]
        self.assertNotIn("text", source)
        self.assertIn("text_sha256", source)

    def test_invalid_credential_config_is_rejected(self):
        self.cfg["model"]["api_key"] = "example_secret"
        with self.assertRaisesRegex(ValueError, "environment"):
            validate_config(self.cfg)

    def test_dataset_change_cannot_reuse_stale_identity(self):
        backend = FakeBackend()
        controller = Controller(self.cfg, backend=backend)
        controller.tick()
        with patch.object(backend, "identity", return_value={"fixture": "different-data"}):
            with self.assertRaisesRegex(RuntimeError, "dataset or engine changed"):
                controller.tick()
        self.assertEqual(backend.calls, [])

    def test_unpublished_interrupted_attempt_still_consumes_budget(self):
        self.cfg["limits"]["max_experiments"] = 1
        backend = FakeBackend()
        controller = Controller(self.cfg, backend=backend)
        controller.tick()
        with patch.object(backend, "evaluate", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                controller.tick()
        resumed = Controller(self.cfg, backend=backend)
        self.assertEqual(resumed.tick()["outcome"], "error")
        self.assertEqual(resumed.tick()["decision"], "repair")
        self.assertEqual(resumed.tick()["status"], "budget")
        self.assertEqual(resumed.status()["evaluations"], 1)
        self.assertEqual(backend.calls, [])


if __name__ == "__main__":
    unittest.main()
