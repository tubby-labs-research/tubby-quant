"""The domain-neutral lab harness: trial ledger, selection statistics and the campaign contract.

A deterministic fake domain stands in for alphas or models, so these tests
need only the standard library and run in the dependency-light CI job.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from autoresearch.lab import selection
from autoresearch.lab.campaign import LabCampaign, validate_lab_config
from autoresearch.lab.domain import Domain
from autoresearch.lab.ledger import GENESIS, LedgerError, TrialLedger
from autoresearch.lab.report import render_markdown, summary


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "ledger.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def test_chain_counts_and_head(self):
        ledger = TrialLedger(self.path)
        first = ledger.append("screen", key="a", fitness=1.0)
        second = ledger.append("validate", key="a", score=0.1)
        self.assertEqual(first["prev"], GENESIS)
        self.assertEqual(second["prev"], first["sha256"])
        report = TrialLedger(self.path).verify()
        self.assertTrue(report["ok"])
        self.assertEqual(report["entries"], 2)
        self.assertEqual(report["stages"], {"screen": 1, "validate": 1})
        self.assertEqual(report["head"], second["sha256"])

    def test_editing_deleting_or_reordering_an_entry_breaks_the_chain(self):
        ledger = TrialLedger(self.path)
        for index in range(3):
            ledger.append("screen", key=f"k{index}", fitness=float(index))
        lines = self.path.read_bytes().split(b"\n")[:-1]
        edited = json.loads(lines[1])
        edited["fitness"] = 99.0
        variants = {
            "edit": [lines[0], json.dumps(edited, sort_keys=True, separators=(",", ":")).encode(), lines[2]],
            "delete": [lines[0], lines[2]],
            "reorder": [lines[1], lines[0], lines[2]],
        }
        for name, content in variants.items():
            with self.subTest(name=name):
                self.path.write_bytes(b"\n".join(content) + b"\n")
                self.assertFalse(TrialLedger(self.path).verify()["ok"])
                with self.assertRaises(LedgerError):
                    TrialLedger(self.path).entries()

    def test_a_torn_tail_is_trimmed_and_the_repair_recorded(self):
        ledger = TrialLedger(self.path)
        ledger.append("screen", key="a")
        with self.path.open("ab") as handle:
            handle.write(b'{"seq": 1, "partial')
        reopened = TrialLedger(self.path)
        self.assertEqual(reopened.verify()["torn_tail_bytes"], len(b'{"seq": 1, "partial'))
        reopened.append("screen", key="b")
        entries = TrialLedger(self.path).entries()
        self.assertEqual([e["stage"] for e in entries], ["screen", "torn_tail_repaired", "screen"])
        self.assertEqual(entries[1]["discarded_bytes"], 19)

    def test_unknown_stage_is_refused(self):
        with self.assertRaises(LedgerError):
            TrialLedger(self.path).append("anything")


class SelectionTests(unittest.TestCase):
    def test_expected_max_sharpe_grows_with_trials_and_is_zero_for_one(self):
        self.assertEqual(selection.expected_max_sharpe(1, 0.01), 0.0)
        values = [selection.expected_max_sharpe(n, 0.01) for n in (2, 10, 100, 1000)]
        self.assertEqual(values, sorted(values))
        # Bailey & Lopez de Prado: about 0.1 * 2.53 for 100 trials of unit-variance-0.01 Sharpe estimates
        self.assertAlmostEqual(selection.expected_max_sharpe(100, 0.01), 0.2531, places=3)

    def test_deflated_sharpe_with_one_trial_is_the_probabilistic_sharpe_against_zero(self):
        psr = selection.probabilistic_sharpe(0.1, 0.0, 251)
        self.assertAlmostEqual(psr, 0.5 + 0.5 * math.erf(0.1 * math.sqrt(250) / math.sqrt(1 + 0.5 * 0.01) / math.sqrt(2)), places=9)
        self.assertAlmostEqual(selection.deflated_sharpe(0.1, 251, 1)["dsr"], psr, places=12)

    def test_more_trials_deflate_the_same_sharpe(self):
        few = selection.deflated_sharpe(0.12, 500, 2, 0.003)["dsr"]
        many = selection.deflated_sharpe(0.12, 500, 200, 0.003)["dsr"]
        self.assertGreater(few, many)
        # The null sampling variance 1/T floors an implausibly small cross-trial variance.
        self.assertEqual(selection.deflated_sharpe(0.12, 500, 50, 1e-9)["trial_variance"], 1 / 500)

    def test_fat_left_tails_reduce_confidence(self):
        normal = selection.probabilistic_sharpe(0.1, 0.0, 500, 0.0, 3.0)
        skewed = selection.probabilistic_sharpe(0.1, 0.0, 500, -1.5, 9.0)
        self.assertLess(skewed, normal)

    def test_moments_holm_bonferroni_and_pearson(self):
        stats = selection.moments([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(stats["mean"], 2.5)
        self.assertAlmostEqual(stats["std"], math.sqrt(5 / 3))
        self.assertAlmostEqual(stats["skew"], 0.0)
        self.assertEqual(selection.holm({"a": 0.01, "b": 0.04, "c": 0.03}, 0.05), {"a": True, "c": False, "b": False})
        self.assertEqual(selection.holm({"a": 0.01, "b": 0.02}, 0.05), {"a": True, "b": True})
        self.assertAlmostEqual(selection.bonferroni(0.05, 10), 0.005)
        self.assertAlmostEqual(selection.pearson([1, 2, 3, 4], [2, 4, 6, 8]), 1.0)
        self.assertIsNone(selection.pearson([1, 1, 1], [1, 2, 3]))


class FakeDomain(Domain):
    """Candidates are integers; fitness and validation scores are fixed functions of them."""

    name = "fake"
    title = "fake"

    def __init__(self, config, runtime, policy=None):
        super().__init__(config, runtime, policy)
        self.validations = []
        self.holdouts = []
        self.fail_validation = set()
        # Holdout daily profile per candidate value; the default is a strong, clearly non-null signal.
        self.holdout_profiles = {}
        self.crash_holdout = set()      # values whose first holdout evaluation simulates a crash

    def identity(self):
        return {"fake": self.config.get("data", 1)}

    def propose(self, context):
        base = (context["round"] - 1) * 4
        out = [{"value": base + i, "generator": "counter"} for i in range(4)]
        out.append({"value": 0, "generator": "repeat"})  # always a duplicate after round 1
        for item in context["submitted"]:
            out.append({**item["candidate"], "generator": "inbox"})
        return out

    def key(self, candidate):
        if not isinstance(candidate.get("value"), int):
            raise ValueError("value must be an integer")
        return f"fake_{candidate['value']}"

    def describe_candidate(self, candidate):
        return {"title": f"value {candidate['value']}", "family": "even" if candidate["value"] % 2 == 0 else "odd"}

    def restore_candidate(self, description):
        return {"value": int(description["title"].split()[1])}

    def screen(self, candidates):
        out = []
        for c in candidates:
            v = c["value"]
            # One-hot residue: equal residues correlate perfectly, different ones at -0.5.
            signature = [float(v % 3 == r) for r in range(3)] if self.config.get("signatures") else []
            out.append({"status": "ok", "fitness": float(v), "metrics": {"v": v}, "signature": signature})
        return out

    def validate(self, candidate):
        self.validations.append(candidate["value"])
        if candidate["value"] in self.fail_validation:
            raise RuntimeError("boom")
        return {"status": "ok", "metrics": {"validation": {"score": candidate["value"] / 10}}}

    def selection_score(self, result):
        return result["metrics"]["validation"]["score"]

    def review(self, candidate, result, context):
        promote = result["metrics"]["validation"]["score"] >= self.policy_config.get("threshold", 0.5)
        return {"decision": "promote" if promote else "reject", "reason": "fake gate",
                "checks": {"n": context["n_validation_trials"]}}

    def final_check(self, entry, context):
        # The bar rises with the lineage-wide trial count, like a multiplicity correction.
        required = self.policy_config.get("threshold", 0.5) \
            + self.policy_config.get("final_penalty", 0.0) * context["n_validation_trials"]
        return {"eligible": (entry.get("score") or 0) >= required, "statistic": "fake",
                "score": entry.get("score"), "required": required}

    def evaluate_holdout(self, candidate):
        if candidate["value"] in self.crash_holdout:
            self.crash_holdout.discard(candidate["value"])
            raise KeyboardInterrupt("simulated crash during the holdout read")
        self.holdouts.append(candidate["value"])
        profile = {"daily_sharpe": 0.3, "days": 250, "skew": 0.0, "kurtosis": 3.0,
                   **self.holdout_profiles.get(candidate["value"], {})}
        return {"status": "ok", "metrics": {"holdout": {"days": profile["days"], "sharpe": 1.0, "returns": 0.1}},
                "holdout_profile": profile}

    def assess_final(self, result):
        if result.get("reused"):
            return {"decision": "not_confirmed", "reason": "reused"}
        return {"decision": "confirmed", "reason": "fake"}


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def config(self, **overrides):
        value = {"name": "fake", "domain": "alpha", "seed": 1, "settings": {"data": 1},
                 "research_root": str(self.root / "records"), "runtime_root": str(self.root / "runtime"),
                 "lineage_root": str(self.root / "lineage"),
                 "policy": {"threshold": 0.5}, "selection": {"max_correlation": 0.999},
                 "limits": {"max_rounds": 3, "proposals_per_round": 5, "validations_per_round": 2,
                            "max_validation_trials": 10, "population_size": 6}}
        value.update(overrides)
        return value

    def campaign(self, **overrides):
        config = self.config(**overrides)
        domain = FakeDomain(config["settings"], self.root / "runtime" / "domain", config.get("policy"))
        return LabCampaign(config, domain=domain)

    def test_rounds_dedupe_count_and_record_everything(self):
        campaign = self.campaign()
        first = campaign.tick()
        self.assertEqual(first["status"], "round")
        self.assertEqual(first["screened"], 4)            # value 0 appears twice in round 1
        campaign.tick()
        campaign.tick()
        done = campaign.tick()
        self.assertEqual(done["status"], "complete")
        self.assertEqual(done["reason"], "round budget reached")
        status = campaign.status()
        ledger = campaign.ledger.verify()
        self.assertTrue(ledger["ok"])
        self.assertEqual(ledger["stages"]["screen"], status["screen_trials"])
        self.assertEqual(ledger["stages"]["validate"], status["validation_trials"])
        self.assertEqual(status["validation_trials"], 6)
        # the fittest unvalidated candidates are validated first (values 11, 10, 7, 6, 3, 2 are the round leaders)
        self.assertEqual(campaign.domain.validations, [3, 2, 7, 6, 11, 10])
        experiments = campaign.store.list("experiments")
        self.assertEqual(len(experiments), 6)
        self.assertEqual({r["decision"] for r in campaign.store.list("reviews")}, {"promote", "reject"})

    def test_validation_budget_and_reservation_before_running(self):
        campaign = self.campaign(limits={"max_rounds": 10, "proposals_per_round": 5, "validations_per_round": 2,
                                         "max_validation_trials": 3, "population_size": 6})
        campaign.domain.fail_validation = {3}
        for _ in range(5):
            if campaign.tick()["status"] == "complete":
                break
        status = campaign.status()
        self.assertEqual(status["validation_trials"], 3)
        failed = [e for e in campaign.store.list("experiments") if e["result"]["status"] == "error"]
        self.assertEqual(len(failed), 1)
        self.assertIn("boom", failed[0]["result"]["error"])

    def test_an_interrupted_validation_is_recorded_and_still_counted(self):
        campaign = self.campaign()
        campaign.tick()
        state = json.loads(campaign.state_path.read_text())
        state["validation_trials"] += 1
        state["pending"] = {"key": "fake_99", "hypothesis_id": "hypothesis_x", "experiment_id": "experiment_interrupted",
                            "started_at": "2026-10-24T00:00:00+00:00"}
        campaign.state_path.write_text(json.dumps(state))
        reopened = self.campaign()
        reopened.tick()
        record = reopened.store.get("experiments", "experiment_interrupted")
        self.assertEqual(record["result"]["error_kind"], "interrupted")
        self.assertEqual(reopened.status()["validation_trials"], 2 + 1 + 2)
        self.assertTrue(reopened.ledger.verify()["ok"])

    def test_identical_signatures_are_redundant_and_never_spend_a_validation(self):
        campaign = self.campaign(settings={"data": 1, "signatures": True}, selection={"max_correlation": 0.9})
        campaign.tick()
        campaign.tick()
        validated = campaign.state["validated"]
        decisions = {key: value["decision"] for key, value in validated.items()}
        self.assertEqual(decisions["fake_6"], "redundant")          # same residue as the validated 3
        self.assertNotIn(6, campaign.domain.validations)
        self.assertIn(7, campaign.domain.validations)                # a different residue still validates
        notes = [e for e in campaign.ledger.entries() if e["stage"] == "note"]
        self.assertEqual([n["key"] for n in notes][:1], ["fake_6"])

    def test_holdout_is_scored_once_reuse_is_labelled_and_later_ideas_are_post_holdout(self):
        campaign = self.campaign(policy={"threshold": 0.2})
        campaign.tick()
        accepted = [a["key"] for a in campaign.state["accepted"]]
        self.assertTrue(accepted)
        finals = campaign.finalize()
        self.assertEqual([f["decision"] for f in finals], ["confirmed"] * len(accepted))
        self.assertTrue(all(f["confirmation"]["method"] == "batch-holm" and f["confirmation"]["holm_rejected"]
                            for f in finals))
        with self.assertRaises(ValueError):
            campaign.finalize()
        with self.assertRaisesRegex(ValueError, "retired"):      # the confirmatory batch retired the holdout
            campaign.finalize(allow_reuse=True)
        again = campaign.finalize(allow_reuse=True, non_confirmatory=True)
        self.assertTrue(all(f["reused"] and f["non_confirmatory"] and f["decision"] == "not_confirmed" for f in again))
        self.assertEqual(len(campaign.domain.holdouts), 2 * len(accepted))
        campaign.tick()
        later = [h for h in campaign.store.list("hypotheses") if h["round"] == 2]
        self.assertTrue(later and all(h["post_holdout"] for h in later))
        holdout_entries = [e for e in campaign.ledger.entries() if e["stage"] == "holdout"]
        self.assertEqual([e["access_sequence"] for e in holdout_entries], [1] * len(accepted) + [2] * len(accepted))

    def test_inbox_submissions_are_screened_like_anything_else(self):
        campaign = self.campaign()
        receipt = campaign.submit({"value": 500}, "agent-under-test")
        self.assertEqual(receipt["key"], "fake_500")
        out = campaign.tick()
        self.assertEqual(out["submitted"], 1)
        self.assertIn("inbox", campaign.status()["generators"])
        with self.assertRaises(ValueError):
            campaign.submit({"value": "not a number"})

    def test_policy_is_identity_but_budgets_are_not(self):
        base = self.campaign().campaign_id
        self.assertEqual(self.campaign(limits={"max_rounds": 9}).campaign_id, base)
        self.assertNotEqual(self.campaign(policy={"threshold": 0.9}).campaign_id, base)

    def test_reruns_are_deterministic(self):
        first = self.campaign()
        first.tick()
        keys = [e["key"] for e in first.ledger.entries() if e["stage"] == "screen"]
        self.tmp.cleanup()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        second = self.campaign()
        second.tick()
        self.assertEqual(keys, [e["key"] for e in second.ledger.entries() if e["stage"] == "screen"])

    def test_a_new_campaign_on_the_same_data_shares_holdout_accesses_and_trial_counts(self):
        first = self.campaign(policy={"threshold": 0.2})
        first.tick()
        finals = first.finalize()
        self.assertTrue(finals)
        nudged = self.campaign(policy={"threshold": 0.19})            # a different gate: a new campaign
        self.assertNotEqual(nudged.campaign_id, first.campaign_id)
        self.assertEqual(nudged.validation_lineage, first.validation_lineage)
        nudged.tick()
        review = [r for r in nudged.store.list("reviews") if r["campaign_id"] == nudged.campaign_id][0]
        self.assertEqual(review["checks"]["validation_trials"]["lineage"], first.status()["validation_trials"])
        with self.assertRaisesRegex(ValueError, "retired|another campaign"):
            nudged.finalize()
        with self.assertRaisesRegex(ValueError, "retired"):
            nudged.finalize(allow_reuse=True)
        again = nudged.finalize(allow_reuse=True, non_confirmatory=True)
        self.assertTrue(all(f["reused"] and f["decision"] == "not_confirmed" for f in again))
        other_data = self.campaign(settings={"data": 2}, policy={"threshold": 0.2})
        self.assertNotEqual(other_data.holdout_lineage, first.holdout_lineage)

    def test_candidates_selected_after_the_holdout_was_read_are_never_confirmed(self):
        campaign = self.campaign(policy={"threshold": 0.2}, limits={"max_rounds": 5, "proposals_per_round": 5,
                                                                      "validations_per_round": 2, "max_validation_trials": 20,
                                                                      "population_size": 6})
        campaign.tick()
        first_keys = {f["key"] for f in campaign.finalize()}
        campaign.tick()
        later = [a["key"] for a in campaign.state["accepted"] if a["key"] not in first_keys]
        self.assertTrue(later)
        with self.assertRaisesRegex(ValueError, "retired"):
            campaign.finalize(later)
        outcomes = campaign.finalize(later, non_confirmatory=True)
        self.assertTrue(all(o["post_holdout"] and o["decision"] == "not_confirmed" for o in outcomes))
        stored = [f for f in campaign.store.list("finals") if f["experiment_key"] in later]
        self.assertTrue(all(f["post_holdout"] for f in stored))

    def test_a_repeated_key_in_one_finalize_is_one_access(self):
        campaign = self.campaign(policy={"threshold": 0.2})
        campaign.tick()
        key = campaign.state["accepted"][0]["key"]
        self.assertEqual(len(campaign.finalize([key, key])), 1)
        self.assertEqual(len(campaign.domain.holdouts), 1)

    def test_keyed_finalization_recomputes_the_final_selection_and_refuses_what_no_longer_passes(self):
        # The final bar is 0.2 + 0.0125 per trial: 0.25 after the four validation trials of two rounds.
        campaign = self.campaign(policy={"threshold": 0.2, "final_penalty": 0.0125})
        campaign.tick()
        campaign.tick()
        scores = {a["key"]: a["score"] for a in campaign.state["accepted"]}
        self.assertEqual(campaign.status()["validation_trials"], 4)
        self.assertEqual((scores["fake_2"], scores["fake_3"]), (0.2, 0.3))   # both promoted at review time
        eligibility = campaign.final_eligibility()
        self.assertEqual(eligibility["n_validation_trials"], 4)
        self.assertIn("fake_3", eligibility["eligible"])
        refused = [e for e in eligibility["excluded"] if e["key"] == "fake_2"]
        self.assertEqual(len(refused), 1)
        self.assertEqual((refused[0]["n_validation_trials"], refused[0]["requested"], refused[0]["score"]), (4, False, 0.2))
        self.assertAlmostEqual(refused[0]["required"], 0.25)
        # An explicit key does not bypass the recomputed selection: refused, recorded, never scored.
        self.assertEqual(campaign.finalize(["fake_2"]), [])
        self.assertEqual(campaign.domain.holdouts, [])
        events = [e for e in campaign.store.list("events") if e["kind"] == "final_selection_excluded"]
        self.assertEqual([(e["key"], e["n_validation_trials"], e["requested"], e["score"], e["statistic"]) for e in events],
                         [("fake_2", 4, True, 0.2, "fake")])
        self.assertAlmostEqual(events[0]["required"], 0.25)
        notes = [e for e in campaign.ledger.entries()
                 if e["stage"] == "note" and e.get("decision") == "final_selection_excluded"]
        self.assertEqual([(n["key"], n["n_validation_trials"]) for n in notes], [("fake_2", 4)])
        self.assertTrue(campaign.ledger.verify()["ok"])
        # Explicit keys select a subset of the eligible set: the eligible one is scored, the other refused again.
        finals = campaign.finalize(["fake_3", "fake_2"])
        self.assertEqual([f["key"] for f in finals], ["fake_3"])
        self.assertEqual(campaign.domain.holdouts, [3])
        with self.assertRaisesRegex(ValueError, "not promoted"):
            campaign.finalize(["fake_999"])

    def test_the_command_line_finalize_reports_refused_keys(self):
        import contextlib
        import io
        from unittest import mock

        from autoresearch.lab import cli
        from autoresearch.lab.domain import REGISTRY

        config = self.config(domain="fake", policy={"threshold": 0.2, "final_penalty": 0.0125})
        path = self.root / "fake.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        with mock.patch.dict(REGISTRY, {"fake": (__name__, "FakeDomain")}):
            campaign = LabCampaign(config)
            campaign.tick()
            campaign.tick()
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = cli.main(["finalize", str(path), "--key", "fake_2", "--key", "fake_3"])
        self.assertEqual(code, 0)
        report = json.loads(out.getvalue())
        self.assertEqual([f["key"] for f in report["finals"]], ["fake_3"])
        self.assertEqual([(e["key"], e["n_validation_trials"], e["requested"]) for e in report["excluded"]],
                         [("fake_2", 4, True)])
        self.assertEqual(report["n_validation_trials"], 4)

    def test_the_lineage_registry_counts_across_record_stores_and_confirmation_needs_agreement(self):
        from autoresearch.lab.campaign import LabCampaign

        first = self.campaign(policy={"threshold": 0.2})
        first.tick()
        # The same data, a different gate and a different record store: the store knows nothing of the first
        # campaign, the registry beside the data does.
        config = self.config(policy={"threshold": 0.19}, research_root=str(self.root / "records-b"),
                             runtime_root=str(self.root / "runtime-b"))
        second = LabCampaign(config, domain=FakeDomain(config["settings"], self.root / "runtime-b" / "domain",
                                                        config["policy"]))
        self.assertNotEqual(second.campaign_id, first.campaign_id)
        self.assertEqual(second.registry.path, first.registry.path)
        second.tick()
        review = [r for r in second.store.list("reviews") if r["campaign_id"] == second.campaign_id][0]
        self.assertEqual(review["checks"]["validation_trials"]["lineage"], first.status()["validation_trials"])
        self.assertEqual(second.lineage_summary()["trials"], {"store": 0, "registry": 2, "agree": False})
        self.assertEqual([c["research_root"] for c in second.lineage_summary()["foreign_campaigns"]],
                         [str(first.research_root)])
        registry = json.loads(first.registry.path.read_text())
        campaigns = registry["validation"][first.validation_lineage]["campaigns"]
        self.assertEqual({cid: len(c["trials"]) for cid, c in campaigns.items()},
                         {first.campaign_id: 2, second.campaign_id: 2})
        # Neither store holds all the evidence the registry lists, so neither may confirm.
        with self.assertRaisesRegex(ValueError, "disagrees with the lineage registry"):
            second.finalize()
        with self.assertRaisesRegex(ValueError, "disagrees"):
            first.finalize()
        finals = first.finalize(non_confirmatory=True)
        self.assertTrue(finals)
        self.assertTrue(all(f["non_confirmatory"] and f["decision"] == "not_confirmed" for f in finals))
        self.assertIn("disagrees", finals[0]["reason"])
        # A non-confirmatory read does not retire the holdout, but its accesses are registered beside the data.
        self.assertIsNone(first.lineage_summary()["holdout_retired"])
        shared = {f["key"] for f in finals} & {a["key"] for a in second.state["accepted"]}
        self.assertTrue(shared)
        with self.assertRaisesRegex(ValueError, "another campaign"):
            second.finalize(sorted(shared), non_confirmatory=True)
        again = second.finalize(sorted(shared), non_confirmatory=True, allow_reuse=True)
        self.assertTrue(all(f["reused"] for f in again))
        self.assertIn("DISAGREE", render_markdown(summary(second)))

    def test_batch_confirmation_applies_holm_and_retires_the_holdout(self):
        from autoresearch.lab.campaign import holdout_pvalue

        campaign = self.campaign(policy={"threshold": 0.2})
        # fake_3 has a strong holdout, fake_2 a weak one: p = 1 - PSR(sharpe, 0, 250 sessions).
        campaign.domain.holdout_profiles = {3: {"daily_sharpe": 0.25, "days": 250}, 2: {"daily_sharpe": 0.05, "days": 250}}
        campaign.tick()
        self.assertEqual(sorted(a["key"] for a in campaign.state["accepted"]), ["fake_2", "fake_3"])
        strong = 1 - selection.probabilistic_sharpe(0.25, 0.0, 250)
        weak = 1 - selection.probabilistic_sharpe(0.05, 0.0, 250)
        self.assertLess(strong, 0.025)                  # Holm: the smallest p faces level/2
        self.assertGreater(weak, 0.05)                  # the next faces level/1 and still fails
        self.assertEqual(campaign.holdout_level(), 0.05)
        finals = {f["key"]: f for f in campaign.finalize()}
        self.assertEqual(finals["fake_3"]["decision"], "confirmed")
        self.assertEqual(finals["fake_2"]["decision"], "not_confirmed")
        strong_record, weak_record = finals["fake_3"]["confirmation"], finals["fake_2"]["confirmation"]
        self.assertAlmostEqual(strong_record["p_value"], strong, places=6)   # finals are stored to 6 significant digits
        self.assertAlmostEqual(weak_record["p_value"], weak, places=6)
        self.assertEqual((strong_record["method"], strong_record["batch"], strong_record["statistic"]),
                         ("batch-holm", 2, "psr"))
        self.assertEqual((strong_record["holm_level"], weak_record["holm_level"]), (0.025, 0.05))
        self.assertEqual((strong_record["holm_rejected"], weak_record["holm_rejected"]), (True, False))
        self.assertIn("Holm", finals["fake_2"]["reason"])
        self.assertEqual(holdout_pvalue({"status": "ok", "metrics": {"holdout": {"p_value": 0.01}}}), (0.01, "bootstrap"))
        # The confirmatory batch retired this lineage's holdout: recorded beside the data and in the ledger.
        retired = campaign.lineage_summary()["holdout_retired"]
        self.assertEqual((retired["campaign_id"], sorted(retired["keys"]), retired["level"]),
                         (campaign.campaign_id, ["fake_2", "fake_3"], 0.05))
        self.assertTrue(any(e.get("decision") == "holdout_retired" for e in campaign.ledger.entries()))
        self.assertIn("holdout retired", render_markdown(summary(campaign)))
        # A later campaign on the same lineage can only read the holdout as non-confirmatory.
        nudged = self.campaign(policy={"threshold": 0.19})
        nudged.tick()
        with self.assertRaisesRegex(ValueError, "retired for confirmation"):
            nudged.finalize()
        later = nudged.finalize(non_confirmatory=True, allow_reuse=True)
        self.assertTrue(later)
        self.assertTrue(all(f["decision"] == "not_confirmed" and f["non_confirmatory"] for f in later))
        self.assertIn("retired", later[0]["reason"])
        # The explicit old behaviour: one verdict per candidate, nothing retired.
        separate = self.campaign(policy={"threshold": 0.2, "holdout_confirmation": "per-candidate"},
                                 lineage_root=str(self.root / "lineage-2"), research_root=str(self.root / "records-2"))
        separate.domain.holdout_profiles = campaign.domain.holdout_profiles
        separate.tick()
        finals = {f["key"]: f for f in separate.finalize()}
        self.assertEqual({k: f["decision"] for k, f in finals.items()}, {"fake_2": "confirmed", "fake_3": "confirmed"})
        self.assertEqual(finals["fake_2"]["confirmation"], {"method": "per-candidate"})
        self.assertIsNone(separate.lineage_summary()["holdout_retired"])

    def test_harness_policy_keys_are_validated_and_kept_from_the_domain(self):
        with self.assertRaisesRegex(ValueError, "holdout_confirmation"):
            validate_lab_config(self.config(policy={"holdout_confirmation": "sometimes"}))
        with self.assertRaisesRegex(ValueError, "holdout_level"):
            validate_lab_config(self.config(policy={"holdout_level": 1.5}))
        campaign = self.campaign(policy={"threshold": 0.2, "holdout_level": 0.01})
        self.assertEqual(campaign.holdout_level(), 0.01)
        self.assertEqual(campaign.domain_policy, {"threshold": 0.2})
        self.assertNotEqual(campaign.campaign_id, self.campaign(policy={"threshold": 0.2}).campaign_id)

    def test_a_missing_checkpoint_is_rebuilt_from_the_records_and_the_ledger(self):
        campaign = self.campaign(policy={"threshold": 0.2, "holdout_confirmation": "per-candidate"})
        campaign.tick()
        campaign.tick()
        key = campaign.state["accepted"][0]["key"]
        campaign.finalize([key])
        before = campaign.status()
        accepted = sorted(a["key"] for a in campaign.state["accepted"])
        campaign.state_path.unlink()
        rebuilt = self.campaign(policy={"threshold": 0.2, "holdout_confirmation": "per-candidate"})
        after = rebuilt.status()
        for field in ("round", "screen_trials", "screen_errors", "validation_trials", "promotions", "holdout_accessed"):
            self.assertEqual(after[field], before[field], field)
        self.assertEqual(sorted(a["key"] for a in rebuilt.state["accepted"]), accepted)
        self.assertEqual(rebuilt.state["holdout_access"], {key: 1})
        self.assertEqual(rebuilt.state["seen"], campaign.state["seen"])
        self.assertTrue(any(e.get("decision") == "checkpoint_rebuilt" for e in rebuilt.ledger.entries()))
        self.assertTrue(rebuilt.ledger.verify()["ok"])
        # The restored promotions still work: the finalized key is refused as a repeat, the other is scored.
        with self.assertRaisesRegex(ValueError, "already evaluated"):
            rebuilt.finalize([key])
        others = [k for k in accepted if k != key]
        self.assertEqual([f["key"] for f in rebuilt.finalize(others)], others)
        # The search continues without re-screening what was seen.
        outcome = rebuilt.tick()
        self.assertEqual(outcome["round"], before["round"] + 1)
        screened = [e["key"] for e in rebuilt.ledger.entries() if e["stage"] == "screen"]
        self.assertEqual(len(screened), len(set(screened)))

    def test_first_contact_seeds_the_registry_from_the_store_and_a_smaller_registry_still_disagrees(self):
        from autoresearch.lab.campaign import LabCampaign

        # Two campaigns ran on the same data with their registry elsewhere: to a fresh registry their record
        # store is a pre-registry store with twelve months of history.
        first = self.campaign(policy={"threshold": 0.2, "holdout_confirmation": "per-candidate"})
        first.tick()
        second = self.campaign(policy={"threshold": 0.19})
        second.tick()
        second.tick()                                        # promotes fake_7 and fake_6 as well, before any holdout read
        first.finalize()                                     # accesses fake_3 and fake_2, retires nothing

        def reopen(**overrides):
            config = self.config(lineage_root=str(self.root / "lineage-new"), **overrides)
            return LabCampaign(config, domain=FakeDomain(config["settings"], self.root / "runtime" / "domain",
                                                         config["policy"]))

        reopened = reopen(policy={"threshold": 0.19})
        self.assertEqual(reopened.campaign_id, second.campaign_id)   # where the registry lives is not identity
        self.assertFalse(reopened.registry.path.exists())
        reopened.status()                                    # first contact: seeded from the record store
        self.assertTrue(reopened.registry.path.exists())
        self.assertEqual(reopened.lineage_summary()["trials"], {"store": 2, "registry": 2, "agree": True})
        seeded = [e for e in reopened.ledger.entries() if e.get("decision") == "lineage_seeded"]
        self.assertEqual([(n["trials"], n["holdout_accesses"], n["campaigns"], n["retired"]) for n in seeded],
                         [(6, 2, 2, False)])
        self.assertEqual(len([e for e in reopened.store.list("events") if e["kind"] == "lineage_seeded"]), 1)
        reopened.status()                                    # idempotent: a known lineage is never re-seeded
        self.assertEqual(len([e for e in reopened.ledger.entries() if e.get("decision") == "lineage_seeded"]), 1)
        # The registry remembers that it was seeded, and by whom.
        marker = reopened.lineage_summary()["seeded"]
        self.assertEqual((marker["campaign_id"], marker["trials"], marker["accesses"]), (reopened.campaign_id, 6, 2))
        # The seeded accesses were per-candidate reads: the lineage is exposed (carried over from the store), so a
        # confirmatory batch is refused and the remaining promotions can only be read as evidence.
        self.assertIn(reopened.lineage_summary()["holdout_exposed"]["key"], ("fake_2", "fake_3"))
        self.assertTrue(reopened.registry.holdout_view(reopened.holdout_lineage)["exposed"])
        with self.assertRaisesRegex(ValueError, "outside a confirmatory batch"):
            reopened.finalize(["fake_7", "fake_6"])
        finals = reopened.finalize(["fake_7", "fake_6"], non_confirmatory=True)
        self.assertEqual([f["decision"] for f in finals], ["not_confirmed", "not_confirmed"])
        self.assertTrue(all(f["non_confirmatory"] and f["lineage_seeded"] and f["lineage_exposed"] for f in finals))
        self.assertTrue(all(f["lineage_seeded"] for f in reopened.store.list("finals") if f["campaign_id"] == reopened.campaign_id))
        with self.assertRaisesRegex(ValueError, "another campaign"):          # the seeded access of fake_3 is honoured
            reopened.finalize(["fake_3"], non_confirmatory=True)
        # A registry that exists for the lineage but is smaller than the store is a disagreement, not a first contact.
        document = json.loads(reopened.registry.path.read_text())
        trials = document["validation"][reopened.validation_lineage]["campaigns"][first.campaign_id]["trials"]
        trials.pop(sorted(trials)[0])
        reopened.registry.path.write_text(json.dumps(document))
        third = reopen(policy={"threshold": 0.18})
        third.status()
        self.assertEqual(third.lineage_summary()["trials"], {"store": 6, "registry": 5, "agree": False})
        self.assertFalse(any(e.get("decision") == "lineage_seeded" for e in third.ledger.entries()))
        with self.assertRaisesRegex(ValueError, "disagrees"):
            third.finalize()
        self.assertIn("DISAGREE", render_markdown(summary(third)))

    def test_verify_detects_a_deleted_or_shrunk_registry(self):
        from autoresearch.lab import cli
        from autoresearch.lab.domain import REGISTRY

        patched = mock.patch.dict(REGISTRY, {"fake": (__name__, "FakeDomain")})
        patched.start()
        self.addCleanup(patched.stop)
        campaign = self.campaign(domain="fake", policy={"threshold": 0.2})
        campaign.tick()
        campaign.finalize()
        verified = campaign.verify()
        self.assertTrue(verified["ok"])
        self.assertEqual(verified["registry"]["status"], "ok")
        snapshots = [e for e in campaign.ledger.entries() if e.get("decision") == "lineage_snapshot"]
        self.assertEqual([s["reason"] for s in snapshots], ["batch_opened", "finalize"])
        self.assertEqual((snapshots[-1]["trials"], sorted(snapshots[-1]["accesses"]), snapshots[-1]["registry"]),
                         (2, ["fake_2", "fake_3"], campaign.registry.path.name))
        self.assertGreaterEqual(snapshots[-1]["revision"], 3)
        self.assertEqual((snapshots[0]["batch"] is not None, snapshots[-1]["batch"], snapshots[-1]["closed_batches"]),
                         (True, None, [snapshots[0]["batch"]]))
        document = json.loads(campaign.registry.path.read_text())
        self.assertEqual(document["revision"], snapshots[-1]["revision"])
        self.assertIn("created_at", document)
        # Edited by hand: an access and the retirement removed. Not a reset (same file, same revision), but shrunk.
        document["holdout"][campaign.holdout_lineage]["accesses"].pop("fake_2")
        document["holdout"][campaign.holdout_lineage]["retired"] = None
        campaign.registry.path.write_text(json.dumps(document))
        edited = self.campaign(domain="fake", policy={"threshold": 0.2})
        shrunk = edited.verify()
        self.assertEqual(shrunk["registry"]["status"], "registry_shrunk")
        self.assertFalse(shrunk["ok"])
        self.assertTrue(any("fake_2" in problem for problem in shrunk["registry"]["problems"]))
        self.assertEqual(edited.lineage_summary()["registry_status"], "registry_shrunk")
        with self.assertRaisesRegex(ValueError, "registry_shrunk"):          # not only reported: it cannot confirm
            edited.finalize()
        self.assertIn("REGISTRY registry_shrunk", render_markdown(summary(edited)))
        # A campaign in the same store that never met the registry has no snapshot to compare, but the store's own
        # retirement event outlives the edit: the holdout stays retired for it.
        newcomer = self.campaign(domain="fake", policy={"threshold": 0.19})
        newcomer.tick()
        newcomer.tick()                                     # promotes fake_7 and fake_6 as well
        lineage = newcomer.lineage_summary()
        self.assertEqual((lineage["registry_status"], lineage["holdout_retired"]["source"],
                          sorted(lineage["holdout_retired"]["keys"])), ("no_snapshot", "store", ["fake_2", "fake_3"]))
        with self.assertRaisesRegex(ValueError, "retired for confirmation"):
            newcomer.finalize(["fake_7", "fake_6"])
        # Deleted: the next load does not quietly re-seed a registry this campaign already met; it records a reset.
        campaign.registry.path.unlink()
        reopened = self.campaign(domain="fake", policy={"threshold": 0.2})
        reset = reopened.verify()
        self.assertEqual(reset["registry"]["status"], "registry_reset")
        self.assertFalse(reopened.registry.path.exists())
        self.assertFalse(any(e.get("decision") == "lineage_seeded" for e in reopened.ledger.entries()))
        self.assertEqual(len([e for e in reopened.ledger.entries() if e.get("decision") == "registry_reset"]), 1)
        reopened.verify()                                   # repeated loads do not repeat the note
        self.assertEqual(len([e for e in reopened.ledger.entries() if e.get("decision") == "registry_reset"]), 1)
        with self.assertRaisesRegex(ValueError, "disagrees|registry_reset"):   # both refuse; the store check comes first
            reopened.finalize()
        self.assertEqual(reopened.lineage_summary()["registry_status"], "registry_reset")
        # A campaign in the same store that never snapshotted the lost registry seeds a new one from the store,
        # retirement included, and says so; the holdout stays retired for it.
        newcomer = self.campaign(domain="fake", policy={"threshold": 0.19})
        newcomer.status()                                   # loads: first contact with the recreated registry
        lineage = newcomer.lineage_summary()
        self.assertTrue(lineage["seeded"])
        self.assertEqual((lineage["holdout_retired"]["method"], sorted(lineage["holdout_retired"]["keys"])),
                         ("batch-holm", ["fake_2", "fake_3"]))
        with self.assertRaisesRegex(ValueError, "retired for confirmation"):
            newcomer.finalize(["fake_7", "fake_6"])
        evidence = newcomer.finalize(["fake_7", "fake_6"], non_confirmatory=True)
        self.assertEqual([(f["decision"], f["lineage_seeded"]) for f in evidence],
                         [("not_confirmed", True), ("not_confirmed", True)])
        self.assertEqual(reopened.lineage_summary()["registry_status"], "registry_reset")   # the original still sees the reset
        with self.assertRaises(ValueError):
            reopened.finalize()
        # The command-line verify reports it and fails.
        config = self.config(domain="fake", policy={"threshold": 0.2})
        path = self.root / "fake.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        import contextlib
        import io
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(["verify", str(path)])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out.getvalue())["registry"]["status"], "registry_reset")

    def test_the_registry_follows_the_data_identity_not_the_location_and_custom_registries_are_flagged(self):
        from autoresearch.lab.campaign import LabCampaign, default_lineage_root

        with mock.patch.dict(os.environ, {"TUBBY_QLIB_DATA_ROOT": str(self.root / "data-root")}):
            canonical = default_lineage_root()
            self.assertEqual(canonical, (self.root / "data-root").resolve() / "autoresearch" / "lineage")
            # Two stores, two runtimes, the same data: without lineage_root both meet the canonical registry.
            one = self.config(policy={"threshold": 0.2}, research_root=str(self.root / "records-1"),
                              runtime_root=str(self.root / "runtime-1"))
            two = self.config(policy={"threshold": 0.19}, research_root=str(self.root / "records-2"),
                              runtime_root=str(self.root / "runtime-2"))
            for config in (one, two):
                del config["lineage_root"]
            first = LabCampaign(one, domain=FakeDomain(one["settings"], self.root / "runtime-1" / "domain", one["policy"]))
            second = LabCampaign(two, domain=FakeDomain(two["settings"], self.root / "runtime-2" / "domain", two["policy"]))
            self.assertEqual(first.registry.path, second.registry.path)
            self.assertEqual(first.registry.path.parent, canonical)
            self.assertTrue(first.lineage_summary()["canonical"])
            first.tick()
            self.assertEqual(second.lineage_summary()["trials"]["registry"], 2)
            # A campaign on an explicitly named registry is allowed to confirm, but says so on every final.
            custom = self.campaign(policy={"threshold": 0.2})
            custom.tick()
            summary_ = custom.lineage_summary()
            self.assertFalse(summary_["canonical"])
            self.assertEqual(summary_["default_registry"], str(canonical))
            finals = custom.finalize()
            self.assertTrue(finals)
            self.assertTrue(all(f["registry_canonical"] is False and f["decision"] == "confirmed" for f in finals))
            self.assertTrue(all(f.get("registry_canonical") is False for f in custom.store.list("finals")))
            self.assertIn("custom registry", render_markdown(summary(custom)))

    def test_a_batch_interrupted_after_some_accesses_is_completed_over_the_same_family(self):
        campaign = self.campaign(policy={"threshold": 0.2})
        campaign.tick()
        self.assertEqual(sorted(a["key"] for a in campaign.state["accepted"]), ["fake_2", "fake_3"])
        campaign.domain.crash_holdout = {2}                 # fake_3 is read, then the worker dies
        with self.assertRaises(KeyboardInterrupt):
            campaign.finalize()
        opened = [e for e in campaign.ledger.entries() if e.get("decision") == "batch_opened"]
        self.assertEqual(len(opened), 1)
        batch_id = opened[0]["batch_id"]
        self.assertEqual(sorted(opened[0]["keys"]), ["fake_2", "fake_3"])
        view = campaign.registry.holdout_view(campaign.holdout_lineage)
        self.assertEqual(view["batch"]["batch_id"], batch_id)
        self.assertEqual(sorted(view["accesses"]), ["fake_2", "fake_3"])     # fake_2 was reserved, then the crash
        self.assertEqual(campaign.domain.holdouts, [3])
        self.assertEqual(len(campaign.store.list("finals")), 0)
        scored = [e for e in campaign.store.list("events") if e["kind"] == "holdout_scored"]
        self.assertEqual([(e["key"], e["batch_id"]) for e in scored], [("fake_3", batch_id)])
        # Another campaign cannot confirm on the lineage while the batch is open.
        other = self.campaign(policy={"threshold": 0.19})
        other.tick()
        with self.assertRaisesRegex(ValueError, "unfinished holdout batch"):
            other.finalize()
        # An exploratory read by another campaign while the batch is open exposes the lineage; the batch that was
        # declared before the exposure may still be completed, and its finals are untouched.
        other.tick()                                        # promotes fake_7 and fake_6 as well
        evidence = other.finalize(["fake_7"], non_confirmatory=True)
        self.assertEqual([(f["key"], f["decision"], f["lineage_exposed"]) for f in evidence], [("fake_7", "not_confirmed", False)])
        self.assertEqual(campaign.registry.holdout_view(campaign.holdout_lineage)["exposed"]["key"], "fake_7")
        # The owner's next finalize completes the same family: fake_3 is not read again, fake_2 is read now,
        # and Holm runs over both.
        restarted = self.campaign(policy={"threshold": 0.2})
        self.assertEqual(restarted.lineage_summary()["open_batch"]["batch_id"], batch_id)
        finals = {f["key"]: f for f in restarted.finalize()}
        self.assertEqual(sorted(finals), ["fake_2", "fake_3"])
        self.assertEqual(restarted.domain.holdouts, [2])
        self.assertFalse(any(f["reused"] for f in finals.values()))         # the reserved read was completed, not repeated
        self.assertTrue(any(e.get("decision") == "holdout_resumed" and e["key"] == "fake_2"
                            for e in restarted.ledger.entries()))
        self.assertTrue(all(f["batch_id"] == batch_id and f["confirmation"]["batch"] == 2 for f in finals.values()))
        self.assertEqual([f["decision"] for f in finals.values()], ["confirmed", "confirmed"])
        self.assertFalse(any(f["lineage_exposed"] for f in finals.values()))     # exposed after the declaration
        self.assertEqual(finals["fake_3"]["holdout"]["days"], 250)
        view = restarted.registry.holdout_view(restarted.holdout_lineage)
        self.assertIsNone(view["batch"])
        self.assertEqual([(b["batch_id"], b["status"]) for b in view["batches"]], [(batch_id, "completed")])
        self.assertTrue(view["retired"])
        closed = [e for e in restarted.ledger.entries() if e.get("decision") == "batch_closed"]
        self.assertEqual([(c["batch_id"], c["resumed"]) for c in closed], [(batch_id, True)])
        self.assertEqual(restarted.verify()["registry"]["status"], "ok")
        # Voiding instead: the abandoned batch is kept in the history with its reason, and the next request
        # forms a new family from keys the voided batch never reserved (a reserved read counts as a read).
        roots = {"lineage_root": str(self.root / "lineage-void"), "research_root": str(self.root / "records-void"),
                 "runtime_root": str(self.root / "runtime-void")}
        again = self.campaign(policy={"threshold": 0.2}, **roots)
        again.tick()
        again.tick()                                         # promotes fake_7 and fake_6 as well
        again.domain.crash_holdout = {2}
        with self.assertRaises(KeyboardInterrupt):
            again.finalize(["fake_3", "fake_2"])
        with self.assertRaisesRegex(ValueError, "completed over its own family"):
            again.finalize(["fake_7"])                      # promoted, but outside the open batch's family
        resumed = self.campaign(policy={"threshold": 0.2}, **roots)
        before = resumed.registry.read()["revision"]
        # fake_3 was scored and fake_2 reserved before the crash: part of the batch's answer is known, so voiding
        # retires the lineage's holdout for confirmation and the remaining candidates are evidence only.
        with self.assertRaisesRegex(ValueError, "retired for confirmation"):
            resumed.finalize(["fake_7", "fake_6"], void_batch="family abandoned after a worker crash")
        self.assertEqual(resumed.registry.read()["revision"], before + 1)     # the void and the retirement are one write
        view = resumed.registry.holdout_view(resumed.holdout_lineage)
        self.assertEqual([(b["status"], b["reason"]) for b in view["batches"]],
                         [("voided", "family abandoned after a worker crash")])
        self.assertIsNone(view["batch"])
        self.assertEqual((view["retired"]["method"], sorted(view["retired"]["keys"]), view["retired"]["reason"]),
                         ("voided-after-access", ["fake_2", "fake_3"], "family abandoned after a worker crash"))
        voided = [e for e in resumed.ledger.entries() if e.get("decision") == "batch_voided"]
        self.assertEqual([sorted(e["read"]) for e in voided], [["fake_2", "fake_3"]])
        self.assertTrue(any(e.get("decision") == "holdout_retired" and e.get("method") == "voided-after-access"
                            for e in resumed.ledger.entries()))
        with self.assertRaisesRegex(ValueError, "voided after reading"):
            resumed.finalize(["fake_7", "fake_6"])
        finals = resumed.finalize(["fake_7", "fake_6"], non_confirmatory=True)
        self.assertEqual([f["decision"] for f in finals], ["not_confirmed", "not_confirmed"])
        self.assertTrue(all(f["non_confirmatory"] and "voided after reading" in f["reason"] for f in finals))
        self.assertIn("batch voided after a read", render_markdown(summary(resumed)))
        with self.assertRaisesRegex(ValueError, "already evaluated"):     # fake_2's reserved read still counts
            resumed.finalize(["fake_2"], non_confirmatory=True)

    def test_a_reservation_that_reached_only_the_ledger_still_counts_as_a_read_when_voiding(self):
        campaign = self.campaign(policy={"threshold": 0.2})
        campaign.tick()
        original = campaign._event

        def crash_between_ledger_and_store(kind, **details):
            if kind == "holdout_access":
                raise KeyboardInterrupt("crash between the ledger line and the store event")
            return original(kind, **details)

        campaign._event = crash_between_ledger_and_store
        with self.assertRaises(KeyboardInterrupt):
            campaign.finalize()
        reserved = [e["key"] for e in campaign.ledger.entries() if e.get("stage") == "holdout"]
        self.assertEqual(len(reserved), 1)
        self.assertFalse(any(e["kind"] == "holdout_access" for e in campaign.store.list("events")))
        restarted = self.campaign(policy={"threshold": 0.2})
        with self.assertRaisesRegex(ValueError, "retired for confirmation"):
            restarted.finalize(void_batch="worker lost after reserving a read")
        view = restarted.registry.holdout_view(restarted.holdout_lineage)
        self.assertEqual((view["retired"]["method"], view["retired"]["keys"]), ("voided-after-access", reserved))

    def test_a_registry_that_lost_only_its_open_batch_record_does_not_reopen_the_holdout(self):
        campaign = self.campaign(policy={"threshold": 0.19})
        campaign.tick()
        campaign.tick()                                       # fake_2, fake_3, fake_6, fake_7 promoted
        campaign.domain.crash_holdout = {3}                   # the first read dies
        with self.assertRaises(KeyboardInterrupt):
            campaign.finalize(["fake_2", "fake_3"])
        batch_id = campaign.registry.holdout_view(campaign.holdout_lineage)["batch"]["batch_id"]
        document = json.loads(campaign.registry.path.read_text())
        document["holdout"][campaign.holdout_lineage]["batch"] = None      # edited: only the open batch removed
        campaign.registry.path.write_text(json.dumps(document))
        restarted = self.campaign(policy={"threshold": 0.19})
        lineage = restarted.lineage_summary()
        self.assertEqual((lineage["open_batch"]["batch_id"], lineage["registry_status"]), (batch_id, "registry_shrunk"))
        opened = [e for e in campaign.ledger.entries() if e.get("decision") == "batch_opened"][0]
        self.assertEqual((lineage["open_batch"]["source"], lineage["open_batch"]["trials_at_declaration"]),
                         ("store", opened["trials_at_declaration"]))                  # reconstructed with its declaration
        self.assertTrue(any("open holdout batch" in p for p in restarted.verify()["registry"]["problems"]))
        with self.assertRaisesRegex(ValueError, "registry_shrunk"):
            restarted.finalize(["fake_7", "fake_6"])
        with self.assertRaisesRegex(ValueError, "unfinished holdout batch"):
            restarted.finalize(["fake_7", "fake_6"], non_confirmatory=True)
        # Another campaign in the same store sees the batch as open too.
        other = self.campaign(policy={"threshold": 0.2})
        other.tick()
        with self.assertRaisesRegex(ValueError, "unfinished holdout batch"):
            other.finalize()
        # The owner can still void it (declaring the read non-confirmatory, since its registry no longer matches its
        # snapshot); the void is written to the registry's history and, since fake_2 was reserved, the holdout retires.
        finals = restarted.finalize(["fake_7", "fake_6"], void_batch="registry record lost", non_confirmatory=True)
        self.assertEqual([f["decision"] for f in finals], ["not_confirmed", "not_confirmed"])
        self.assertTrue(all("had already read fake_2" in f["reason"] and "registry_shrunk" in f["reason"] for f in finals))
        view = restarted.registry.holdout_view(restarted.holdout_lineage)
        self.assertEqual([(b["batch_id"], b["status"]) for b in view["batches"]], [(batch_id, "voided")])
        self.assertIn("lost the batch", view["batches"][0]["note"])
        self.assertEqual((view["batches"][0]["trials_at_declaration"], sorted(view["batches"][0]["keys"])),
                         (opened["trials_at_declaration"], ["fake_2", "fake_3"]))     # the history keeps the declaration
        self.assertEqual(view["retired"]["method"], "voided-after-access")

    def test_finals_record_the_trial_count_their_family_was_priced_at(self):
        campaign = self.campaign(policy={"threshold": 0.2})
        campaign.tick()
        priced = campaign.final_eligibility()["n_validation_trials"]
        finals = campaign.finalize()
        self.assertTrue(all(f["trials_at_declaration"] == priced for f in finals))
        self.assertEqual(campaign.registry.holdout_view(campaign.holdout_lineage)["batches"][0]["trials_at_declaration"], priced)
        self.assertEqual(finals[0]["confirmation"]["statistic_note"], "PSR; sessions treated as independent")
        from autoresearch.lab.campaign import statistic_note
        self.assertEqual(statistic_note("moving-block-bootstrap"), "moving-block bootstrap on ordered rows")

    def test_a_batch_voided_before_any_read_leaves_the_holdout_confirmable(self):
        campaign = self.campaign(policy={"threshold": 0.2})
        campaign.tick()
        self.crash_on(campaign, "holdout")                    # the batch is declared, then the first reservation dies
        with self.assertRaises(KeyboardInterrupt):
            campaign.finalize()
        view = campaign.registry.holdout_view(campaign.holdout_lineage)
        self.assertTrue(view["batch"])
        self.assertEqual(view["accesses"], {})
        restarted = self.campaign(policy={"threshold": 0.2})
        finals = restarted.finalize(void_batch="declared by mistake")
        self.assertEqual([f["decision"] for f in finals], ["confirmed", "confirmed"])
        view = restarted.registry.holdout_view(restarted.holdout_lineage)
        self.assertEqual([(b["status"], b["reason"]) for b in view["batches"]],
                         [("voided", "declared by mistake"), ("completed", None)])
        self.assertEqual(view["retired"]["method"], "batch-holm")
        voided = [e for e in restarted.ledger.entries() if e.get("decision") == "batch_voided"]
        self.assertEqual([e["read"] for e in voided], [[]])

    def test_a_read_outside_a_confirmatory_batch_exposes_the_lineage(self):
        campaign = self.campaign(policy={"threshold": 0.19})
        campaign.tick()
        campaign.tick()                                       # fake_2, fake_3, fake_6, fake_7 promoted
        self.assertIsNone(campaign.lineage_summary()["holdout_exposed"])
        evidence = campaign.finalize(["fake_3"], non_confirmatory=True)      # an exploratory read
        self.assertEqual([(f["decision"], f["lineage_exposed"]) for f in evidence], [("not_confirmed", False)])
        view = campaign.registry.holdout_view(campaign.holdout_lineage)
        self.assertEqual((view["exposed"]["key"], view["exposed"]["campaign_id"], view["retired"]),
                         ("fake_3", campaign.campaign_id, None))
        with self.assertRaisesRegex(ValueError, "outside a confirmatory batch"):
            campaign.finalize(["fake_7", "fake_6"])
        finals = campaign.finalize(["fake_7", "fake_6"], non_confirmatory=True)
        self.assertTrue(all(f["decision"] == "not_confirmed" and f["lineage_exposed"] for f in finals))
        self.assertIn("holdout exposed", render_markdown(summary(campaign)))
        # Another record store on the same data sees the exposure through the registry alone.
        other = self.campaign(policy={"threshold": 0.2}, research_root=str(self.root / "records-2"),
                              runtime_root=str(self.root / "runtime-2"))
        other.tick()
        self.assertEqual(other.lineage_summary()["holdout_exposed"]["campaign_id"], campaign.campaign_id)
        with self.assertRaises(ValueError):
            other.finalize(["fake_2"])
        read = other.finalize(["fake_2"], non_confirmatory=True)
        self.assertTrue(all("outside a confirmatory batch" in f["reason"] and f["lineage_exposed"] for f in read))
        # A per-candidate read exposes the lineage too.
        fresh = self.campaign(policy={"threshold": 0.2, "holdout_confirmation": "per-candidate"}, research_root=str(self.root / "records-3"),
                              runtime_root=str(self.root / "runtime-3"), lineage_root=str(self.root / "lineage-3"))
        fresh.tick()
        fresh.finalize(["fake_3"])
        self.assertEqual(fresh.registry.holdout_view(fresh.holdout_lineage)["exposed"]["key"], "fake_3")

    def test_a_registry_written_while_the_batch_is_declared_sends_finalize_round_again(self):
        campaign = self.campaign(policy={"threshold": 0.2})
        campaign.tick()
        experiment = campaign.store.list("experiments")[0]["id"]
        original = campaign._final_eligibility
        calls = {"n": 0}

        def late_score(context, keys):
            calls["n"] += 1
            if calls["n"] == 1:        # a worker of this campaign publishes a score after the family was read
                campaign.registry.record_score(campaign.validation_lineage, campaign.campaign_id, experiment, 0.31,
                                               research_root=str(campaign.research_root), lineage=campaign.lineage["validation"])
            return original(context, keys)

        campaign._final_eligibility = late_score
        finals = campaign.finalize()
        self.assertEqual([f["decision"] for f in finals], ["confirmed", "confirmed"])
        opened = [e for e in campaign.ledger.entries() if e.get("decision") == "batch_opened"]
        self.assertEqual((len(opened), opened[0]["attempt"]), (1, 2))
        self.assertEqual(calls["n"], 2)
        # A change that matters (another campaign's trials appearing) is not retried over: finalize refuses.
        other = self.campaign(policy={"threshold": 0.19}, research_root=str(self.root / "records-2"),
                              runtime_root=str(self.root / "runtime-2"), lineage_root=str(self.root / "lineage-2"))
        other.tick()
        stranger = self.campaign(policy={"threshold": 0.2}, research_root=str(self.root / "records-3"),
                                 runtime_root=str(self.root / "runtime-3"), lineage_root=str(self.root / "lineage-2"))
        original_other = other._final_eligibility
        calls["n"] = 0

        def new_trial(context, keys):
            calls["n"] += 1
            if calls["n"] == 1:
                stranger.registry.record_trial(stranger.validation_lineage, stranger.campaign_id, "experiment_x",
                                               research_root=str(stranger.research_root), lineage=stranger.lineage["validation"])
            return original_other(context, keys)

        other._final_eligibility = new_trial
        with self.assertRaisesRegex(ValueError, "changed while finalizing"):
            other.finalize()
        self.assertFalse(any(e.get("decision") == "batch_opened" for e in other.ledger.entries()))

    def test_the_holdout_level_is_its_own_setting_not_a_validation_threshold(self):
        self.assertEqual(self.campaign(policy={"threshold": 0.2, "min_dsr": 0.9}).holdout_level(), 0.05)
        self.assertEqual(self.campaign(policy={"threshold": 0.2, "alpha_level": 0.1}).holdout_level(), 0.1)
        self.assertEqual(self.campaign(policy={"threshold": 0.2, "alpha_level": 0.1, "holdout_level": 0.02}).holdout_level(), 0.02)

    def test_a_narrowed_family_is_visible_and_the_p_value_is_labelled_nominal(self):
        campaign = self.campaign(policy={"threshold": 0.2})
        campaign.tick()
        self.assertEqual(campaign.final_eligibility(["fake_3"])["eligible_not_requested"], ["fake_2"])
        self.assertEqual(campaign.final_eligibility()["eligible_not_requested"], [])
        finals = campaign.finalize(["fake_3"])
        self.assertEqual([f["decision"] for f in finals], ["confirmed"])
        self.assertIn("nominal p = ", finals[0]["reason"])
        self.assertIn("sessions treated as independent", finals[0]["reason"])
        opened = [e for e in campaign.ledger.entries() if e.get("decision") == "batch_opened"]
        self.assertEqual((opened[0]["eligible_not_requested"], opened[0]["attempt"]), (["fake_2"], 1))
        view = campaign.registry.holdout_view(campaign.holdout_lineage)
        self.assertEqual(view["batches"][0]["eligible_not_requested"], ["fake_2"])
        self.assertEqual(campaign.lineage_summary()["registry_status"], "ok")
        self.assertEqual(campaign.lineage_summary()["holdout_accesses"], {"fake_3": 1})   # what a console can pre-check

    def test_a_store_whose_ledger_does_not_verify_cannot_seed_the_registry(self):
        from autoresearch.lab.campaign import LabCampaign

        first = self.campaign(policy={"threshold": 0.2})
        first.tick()
        second = self.campaign(policy={"threshold": 0.19})
        second.tick()
        lines = second.ledger.path.read_text(encoding="utf-8").splitlines()
        entry = json.loads(lines[1])
        entry["created_at"] = "1999-01-01T00:00:00+00:00"       # an edited line: valid JSON, broken chain
        lines[1] = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        second.ledger.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.assertFalse(second.ledger.verify()["ok"])
        config = self.config(lineage_root=str(self.root / "lineage-new"), policy={"threshold": 0.19})
        reopened = LabCampaign(config, domain=FakeDomain(config["settings"], self.root / "runtime" / "domain", config["policy"]))
        reopened.status()
        self.assertFalse(reopened.registry.path.exists())
        refused = [e for e in reopened.store.list("events") if e["kind"] == "lineage_seed_refused"]
        self.assertEqual(len(refused), 1)
        self.assertIn("hash", refused[0]["reason"])
        reopened.status()                                     # recorded once
        self.assertEqual(len([e for e in reopened.store.list("events") if e["kind"] == "lineage_seed_refused"]), 1)
        self.assertEqual(reopened.lineage_summary()["registry_status"], "ledger_broken")
        with self.assertRaisesRegex(ValueError, "disagrees|does not verify"):
            reopened.finalize()

    def crash_on(self, campaign, stage, nth=1):
        original = campaign.ledger.append
        calls = {"n": 0}

        def append(kind, **fields):
            if kind == stage:
                calls["n"] += 1
                if calls["n"] == nth:
                    raise KeyboardInterrupt("simulated crash")
            return original(kind, **fields)

        campaign.ledger.append = append

    def assert_consistent(self, campaign):
        ledger = campaign.ledger.verify()
        status = campaign.status()
        self.assertTrue(ledger["ok"])
        self.assertEqual(ledger["stages"].get("screen", 0), status["screen_trials"])
        self.assertEqual(ledger["stages"].get("validate", 0), status["validation_trials"])
        self.assertEqual(ledger["stages"].get("review", 0), status["validation_trials"])
        screened = [e["key"] for e in campaign.ledger.entries() if e["stage"] == "screen"]
        self.assertEqual(len(screened), len(set(screened)))

    def test_crash_after_the_experiment_but_before_its_ledger_line_is_completed_on_restart(self):
        campaign = self.campaign(policy={"threshold": 0.2})
        self.crash_on(campaign, "validate")
        with self.assertRaises(KeyboardInterrupt):
            campaign.tick()
        restarted = self.campaign(policy={"threshold": 0.2})
        restarted.tick()
        self.assert_consistent(restarted)
        self.assertEqual(len(restarted.store.list("experiments")), restarted.status()["validation_trials"])

    def test_crash_after_a_promoting_review_keeps_the_promotion(self):
        campaign = self.campaign(policy={"threshold": 0.2})
        self.crash_on(campaign, "review")
        with self.assertRaises(KeyboardInterrupt):
            campaign.tick()
        promoted = [r for r in campaign.store.list("reviews") if r["decision"] == "promote"]
        self.assertEqual(len(promoted), 1)
        restarted = self.campaign(policy={"threshold": 0.2})
        restarted.tick()
        accepted = {a["experiment_id"] for a in restarted.state["accepted"]}
        self.assertIn(promoted[0]["experiment_id"], accepted)
        self.assert_consistent(restarted)

    def test_crash_in_the_middle_of_a_screen_does_not_duplicate_trials(self):
        campaign = self.campaign()
        self.crash_on(campaign, "screen", nth=3)
        with self.assertRaises(KeyboardInterrupt):
            campaign.tick()
        restarted = self.campaign()
        restarted.tick()
        restarted.tick()
        self.assert_consistent(restarted)
        notes = [e for e in restarted.ledger.entries() if e["stage"] == "note" and e.get("decision") == "round_interrupted"]
        self.assertEqual(len(notes), 1)

    def test_the_inbox_is_rotated_so_its_size_cap_applies_to_what_is_waiting(self):
        campaign = self.campaign()
        campaign.submit({"value": 900}, "agent")
        campaign.tick()
        self.assertFalse(campaign.inbox_path.exists())
        self.assertFalse(campaign._inbox_processing.exists())
        campaign.submit({"value": 901}, "agent")
        self.assertTrue(campaign.inbox_path.exists())

    def test_configuration_validation(self):
        with self.assertRaises(ValueError):
            validate_lab_config({"name": "x", "domain": "nope", "settings": {}})
        with self.assertRaises(ValueError):
            validate_lab_config(self.config(limits={"unknown": 1}))
        with self.assertRaises(ValueError):
            validate_lab_config(self.config(settings={"api_key": "x"}))
        with self.assertRaises(ValueError):
            validate_lab_config(self.config(selection={"max_correlation": 0}))
        self.assertIsNone(validate_lab_config(self.config(selection={}))["selection"]["max_correlation"])


if __name__ == "__main__":
    unittest.main()
