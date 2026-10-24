"""Sealed final evaluation: single-use holdout accounting, labelled reuse, hand-off and reports."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import unittest.mock

from autoresearch.controller import Controller
from autoresearch.handoff import campaign_report, export_batch, render_markdown, selected_candidates
from autoresearch.policy import assess_final
from autoresearch.records import ResearchStore
from test_autoresearch_controller import FakeBackend, candidate, result

HAVE_NUMERIC = bool(importlib.util.find_spec("numpy") and importlib.util.find_spec("pandas"))


def holdout_result(sharpe=1.0, days=60, returns=None):
    metrics = {"days": days, "sharpe": sharpe, "returns": (0.08 if sharpe > 0 else -0.08) if returns is None else returns,
               "turnover": 0.2, "drawdown": 0.05}
    return {"status": "ok", "metrics": {"holdout": metrics}, "periods": {"holdout": metrics},
            "settings": {"cost_bps": 5, "holdout_start": "2024-10-01", "holdout_evaluated": True}}


class HoldoutBackend(FakeBackend):
    def __init__(self, values=None, holdouts=None):
        super().__init__(values)
        self.holdouts = holdouts or {}
        self.holdout_calls = []

    def evaluate_holdout(self, expression):
        self.holdout_calls.append(expression)
        value = self.holdouts.get(expression, holdout_result())
        if isinstance(value, Exception):
            raise value
        return json.loads(json.dumps(value))


class FinalPolicyTests(unittest.TestCase):
    def decision(self, value, config=None):
        verdict = assess_final(value, config or {})
        self.assertEqual(set(verdict), {"decision", "reason", "question", "expected_change"})
        return verdict["decision"]

    def test_gates(self):
        self.assertEqual(self.decision(holdout_result()), "confirmed")
        self.assertEqual(self.decision(holdout_result(sharpe=-0.3)), "not_confirmed")
        self.assertEqual(self.decision(holdout_result(sharpe=0.2), {"min_holdout_sharpe": 0.5}), "not_confirmed")
        self.assertEqual(self.decision(holdout_result(days=10)), "insufficient")
        self.assertEqual(self.decision(holdout_result(days=10), {"policy": {"min_holdout_days": 5}}), "confirmed")
        self.assertEqual(self.decision({**holdout_result(), "reused": True}), "not_confirmed")
        self.assertEqual(self.decision({"status": "error", "error": "boom"}), "repair")
        broken = holdout_result()
        broken["metrics"]["holdout"]["sharpe"] = float("nan")
        self.assertEqual(self.decision(broken), "repair")
        self.assertEqual(self.decision(holdout_result(), {"min_holdout_days": 1}), "repair")


class FinalizeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = {"name": "final-campaign", "research_root": str(self.root / "research"),
                    "runtime_root": str(self.root / "runtime"), "backend": {},
                    "discovery": {"network_enabled": False, "fixture_sources": [
                        {"id": "fixture_source", "title": "Price prediction fixture",
                         "url": "https://example.invalid/fixture", "text": "Fixture price patterns support test plumbing."}]},
                    "model": {"provider": "fixture", "hypotheses": [candidate(), candidate("-rank(close)", "Reversal fixture")]},
                    "limits": {"max_experiments": 10, "max_hypotheses": 10, "max_per_family": 10,
                               "max_model_calls": 10, "daily_model_calls": 10, "max_refinement_depth": 0},
                    "policy": {"min_validation_days": 60, "min_sharpe": 1.0}}
        self.backend = HoldoutBackend(values={"rank(close)": result(sharpe=1.5), "-rank(close)": result(sharpe=-0.5)},
                                      holdouts={"rank(close)": holdout_result(sharpe=0.9)})

    def drain(self, controller):
        for _ in range(30):
            outcome = controller.tick()
            if not controller.status()["queued"] or outcome["status"] in {"budget", "complete"}:
                return
        self.fail("campaign did not drain")

    def test_promoted_hypotheses_are_finalized_once_and_reuse_is_labelled(self):
        controller = Controller(self.cfg, backend=self.backend)
        self.drain(controller)
        decisions = {r["hypothesis_id"]: r["decision"] for r in controller.store.list("reviews")}
        self.assertEqual(sorted(decisions.values()), ["promote", "reject"])
        outcomes = controller.finalize()
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(self.backend.holdout_calls, ["rank(close)"])
        final = controller.store.list("finals")[0]
        self.assertEqual(final["decision"], "confirmed")
        self.assertEqual(final["selection"], "promoted")
        self.assertFalse(final["reused"])
        self.assertEqual(final["access_sequence"], 1)
        self.assertEqual(final["result"]["metrics"]["holdout"]["sharpe"], 0.9)
        self.assertTrue(controller.status()["holdout_accessed"])
        self.assertEqual(controller.status()["holdout_access"], {final["experiment_key"]: 1})
        kinds = [e["kind"] for e in controller.store.list("events")]
        self.assertIn("holdout_access", kinds)
        self.assertIn("finalized", kinds)
        with self.assertRaisesRegex(ValueError, "already evaluated"):
            controller.finalize()
        self.assertEqual(self.backend.holdout_calls, ["rank(close)"])  # refused before any evaluation
        again = controller.finalize(allow_reuse=True)
        self.assertEqual(again[0]["access_sequence"], 2)
        self.assertTrue(again[0]["reused"])
        self.assertEqual(again[0]["decision"], "not_confirmed")
        finals = controller.store.list("finals")
        self.assertEqual(len(finals), 2)
        self.assertEqual({f["access_sequence"] for f in finals}, {1, 2})
        # Research decisions never see holdout metrics: the related-history view excludes finals.
        history = controller.store.related_hypotheses("fixture price prediction", limit=5)
        self.assertNotIn("holdout", json.dumps(history))

    def test_access_is_reserved_before_an_interrupted_evaluation(self):
        self.backend.holdouts["rank(close)"] = RuntimeError("worker killed")
        controller = Controller(self.cfg, backend=self.backend)
        self.drain(controller)
        outcomes = controller.finalize()
        self.assertEqual(outcomes[0]["decision"], "repair")
        self.assertEqual(controller.store.list("finals")[0]["result"]["error_kind"], "execution")
        self.assertEqual(controller.status()["holdout_access"][outcomes[0]["experiment_id"] and controller.store.get("experiments", outcomes[0]["experiment_id"])["experiment_key"]], 1)

    def test_explicit_selection_requires_a_successful_evaluation(self):
        controller = Controller(self.cfg, backend=self.backend)
        self.drain(controller)
        rejected = next(r["hypothesis_id"] for r in controller.store.list("reviews") if r["decision"] == "reject")
        outcomes = controller.finalize([rejected])
        self.assertEqual(outcomes[0]["selection"], "explicit")
        self.assertEqual(self.backend.holdout_calls, ["-rank(close)"])
        with self.assertRaisesRegex(ValueError, "unknown hypothesis"):
            controller.finalize(["hypothesis_missing"])

    def test_hypotheses_after_holdout_access_are_labelled_and_identified_separately(self):
        controller = Controller(self.cfg, backend=self.backend)
        self.drain(controller)
        before = {h["id"] for h in controller.store.list("hypotheses")}
        controller.finalize()
        self.cfg["model"]["hypotheses"].append(candidate("rank(volume)", "Late idea"))
        later = Controller(self.cfg, backend=self.backend)
        self.assertNotEqual(later.campaign_id, controller.campaign_id)  # changed inputs: a new campaign
        # Within the same campaign the label applies: simulate by reusing the finalized checkpoint.
        with controller.store.lock():
            controller._load()
            self.assertTrue(controller._enqueue(candidate("rank(volume)", "Late idea")))
            controller._save()
        late = [h for h in controller.store.list("hypotheses") if h["id"] not in before]
        self.assertEqual(len(late), 1)
        self.assertTrue(late[0]["post_holdout"])

    def test_backend_without_holdout_support_is_refused(self):
        controller = Controller(self.cfg, backend=FakeBackend(values={"rank(close)": result(sharpe=1.5)}))
        self.drain(controller)
        with self.assertRaisesRegex(ValueError, "no holdout evaluation"):
            controller.finalize()

    def test_report_handoff_and_markdown(self):
        controller = Controller(self.cfg, backend=self.backend)
        self.drain(controller)
        controller.finalize()
        store = ResearchStore(self.root / "research")
        report = campaign_report(store, controller.campaign_id)
        self.assertEqual(report["counts"]["finals"], 1)
        self.assertEqual(report["decisions"], {"promote": 1, "reject": 1})
        rows = {row["title"]: row for row in report["hypotheses"]}
        self.assertEqual(rows["Fixture hypothesis"]["final"]["decision"], "confirmed")
        self.assertIsNone(rows["Reversal fixture"]["final"])
        self.assertEqual(rows["Fixture hypothesis"]["validation"]["sharpe"], 1.5)
        self.assertEqual([row["decision"] for row in report["hypotheses"]], ["promote", "reject"])
        self.assertEqual([r["expression"] for r in selected_candidates(report, "promote")], ["rank(close)"])
        self.assertEqual([r["expression"] for r in selected_candidates(report, "confirmed")], ["rank(close)"])
        self.assertEqual(len(selected_candidates(report, "all")), 2)
        batch = export_batch(report, self.root / "out" / "promoted.txt")
        text = batch.read_text(encoding="utf-8")
        self.assertIn("rank(close)\n", text)
        self.assertNotIn("-rank(close)", text)
        self.assertIn("# Fixture hypothesis | hypothesis ", text)
        if HAVE_NUMERIC:  # Alpha Lab's reader imports the numerical stack
            from qlib_yahoo.alphalab.batch import read_expressions
            self.assertEqual(read_expressions(batch), ["rank(close)"])
        else:
            self.assertEqual([line for line in text.splitlines() if line and not line.startswith("#")], ["rank(close)"])
        with self.assertRaises(FileExistsError):
            export_batch(report, batch)
        markdown = render_markdown(report)
        self.assertIn("| promote | Fixture hypothesis | 1.50 |", markdown)
        self.assertIn("confirmed, Sharpe 0.90", markdown)
        self.assertIn("not a trading recommendation", markdown)
        self.assertNotIn(str(self.root), json.dumps(report))

    def test_cli_finalize_export_and_report(self):
        from autoresearch.__main__ import main
        import contextlib
        import io
        config_path = self.root / "campaign.json"
        config_path.write_text(json.dumps(self.cfg))
        with unittest.mock.patch("autoresearch.__main__.Controller", lambda cfg: Controller(cfg, backend=self.backend)):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["run", str(config_path), "--once"]), 0)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(main(["finalize", str(config_path)]), 0)
            finals = json.loads(out.getvalue())
            self.assertEqual(finals["finals"][0]["decision"], "confirmed")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(main(["finalize", str(config_path)]), 2)
            self.assertIn("already evaluated", err.getvalue())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["export", str(config_path), "--output", str(self.root / "batch.txt"), "--decision", "confirmed"]), 0)
            self.assertEqual(main(["report", "--research-root", str(self.root / "research"), "--markdown"]), 0)
            self.assertEqual(main(["report", "--research-root", str(self.root / "research")]), 0)
        text = out.getvalue()
        self.assertIn('"status": "written"', text)
        self.assertIn("# Research campaign report", text)
        self.assertEqual((self.root / "batch.txt").read_text().count("rank(close)"), 1)
        self.assertIn('"finals"', text)


@unittest.skipUnless(HAVE_NUMERIC, "requires local numpy and pandas")
class RealHoldoutTests(unittest.TestCase):
    """The real adapter scores only the reserved period, separately from research caches."""

    def test_holdout_evaluation_is_bounded_and_isolated(self):
        from autoresearch.backend import AlphaLabBackend, create_demo_dataset

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = create_demo_dataset(root / "dataset")
            config = {"dataset": str(dataset), "universe": "TOP8", "cost_bps": 5,
                      "development_start": "2023-06-01", "development_end": "2023-12-29",
                      "validation_start": "2024-01-01", "validation_end": "2024-09-30",
                      "holdout_start": "2024-10-01", "warmup": 60, "truncation": 0}
            backend = AlphaLabBackend(config, root / "runtime")
            identity = backend.identity()
            research = backend.evaluate("rank(ts_delta(close, 5))")
            self.assertEqual(research["status"], "ok")
            self.assertFalse(research["settings"]["holdout_evaluated"])
            final = backend.evaluate_holdout("rank(ts_delta(close, 5))")
            self.assertEqual(final["status"], "ok", final.get("error"))
            self.assertTrue(final["settings"]["holdout_evaluated"])
            self.assertEqual(final["settings"]["holdout_start"], "2024-10-01")
            self.assertEqual(set(final["metrics"]), {"holdout"})
            artifact = root / "runtime" / final["artifacts"][0]
            days = json.loads(artifact.read_text())["holdout"]["daily"]["dates"]
            self.assertTrue(days and days[0] >= "2024-10-01" and days[-1] <= "2024-12-31")
            self.assertIn("/holdout/", final["artifacts"][0])
            self.assertNotIn("/holdout/", research["artifacts"][0])
            self.assertEqual(backend.identity(), identity)  # scoring the holdout does not change the research identity
            self.assertEqual(backend.evaluate("rank(ts_delta(close, 5))")["metrics"], research["metrics"])
            no_holdout = AlphaLabBackend({k: v for k, v in config.items() if k != "holdout_start"}, root / "runtime2")
            with self.assertRaisesRegex(ValueError, "reserved no holdout"):
                no_holdout.evaluate_holdout("rank(close)")
            bad = backend.evaluate_holdout("rank(cap)")
            self.assertEqual(bad["status"], "error")
            self.assertEqual(bad["error_kind"], "expression")


if __name__ == "__main__":
    unittest.main()
