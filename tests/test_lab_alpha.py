"""Autonomous alpha discovery: typed grammar, sealed evaluation, planted panels and a full campaign."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HAVE_NUMERIC = bool(importlib.util.find_spec("numpy") and importlib.util.find_spec("pandas"))

if HAVE_NUMERIC:
    import numpy as np


def reachable_arrays(root):
    """Every numpy array reachable from ``root`` through plain object references.

    Walks instance attributes, dicts, lists, tuples and dataclasses, and stops at modules, classes,
    functions and methods, so it covers exactly the data a domain object holds (its caches,
    workspaces, panels and matrices) and nothing of the interpreter around it. The sealing tests
    use it as the memory proof: after screening, validating and finalizing, no array a domain can
    reach may hold a session after the validation end.
    """
    import gc
    import types

    stop = (types.ModuleType, type, types.FunctionType, types.MethodType, types.BuiltinFunctionType,
            types.BuiltinMethodType, types.CodeType, types.FrameType, property, classmethod, staticmethod,
            str, bytes, int, float, bool, type(None))
    seen, queue, arrays = {id(root)}, [root], []
    while queue:
        obj = queue.pop()
        if isinstance(obj, np.ndarray):
            arrays.append(obj)
        for child in gc.get_referents(obj):
            if id(child) in seen or isinstance(child, stop):
                continue
            seen.add(id(child))
            queue.append(child)
    return arrays


def assert_sessions_bounded(case, root, last_session, sessions, symbols):
    """Fail unless every array reachable from ``root`` ends at ``last_session``: date arrays by their
    latest value, (sessions x symbols) arrays and feature stacks by their row count."""
    arrays = reachable_arrays(root)
    case.assertTrue(arrays, "nothing reachable: the walk found no arrays")
    for array in arrays:
        if array.dtype.kind == "M" and array.size:
            case.assertLessEqual(str(array.max().astype("datetime64[D]")), last_session)
        elif array.ndim >= 2 and array.shape[1] == symbols:
            case.assertLessEqual(array.shape[0], sessions, f"an array of shape {array.shape} outlives the seal")


@unittest.skipUnless(HAVE_NUMERIC, "numpy and pandas are optional outside the Qlib environment")
class GrammarTests(unittest.TestCase):
    def test_templates_round_trip_and_are_dimensionless(self):
        from autoresearch.domains.alpha import grammar

        for title, node in grammar.template_trees(every_window=True):
            with self.subTest(title=title):
                self.assertEqual(grammar.from_text(grammar.to_text(node)), node)
                self.assertIn(grammar.unit(node), grammar.DIMENSIONLESS)

    def test_units_reject_meaningless_combinations(self):
        from autoresearch.domains.alpha import grammar

        for text in ("close + volume", "close / returns", "close * volume", "log(returns)", "close - close"):
            with self.subTest(text=text):
                self.assertIsNone(grammar.unit(grammar.from_text(text)))
        self.assertEqual(grammar.unit(grammar.from_text("close / ts_delay(close, 5)")), "R")
        self.assertEqual(grammar.unit(grammar.from_text("ts_mean(returns, 5) / ts_std_dev(returns, 5)")), "R")
        self.assertEqual(grammar.unit(grammar.from_text("log(close) - ts_delay(log(close), 3)")), "R")

    def test_key_is_sign_insensitive_and_formatting_insensitive(self):
        from autoresearch.domains.alpha import grammar

        a = grammar.canonical_key(grammar.from_text("rank(ts_delta(close, 5))"))
        self.assertEqual(a, grammar.canonical_key(grammar.from_text("-rank( ts_delta(close,5) )")))
        self.assertEqual(a, grammar.canonical_key(grammar.from_text("--rank(ts_delta(close, 5))")))
        self.assertNotEqual(a, grammar.canonical_key(grammar.from_text("rank(ts_delta(close, 10))")))

    def test_random_mutation_and_crossover_stay_admissible(self):
        from autoresearch.domains.alpha import grammar

        rng = np.random.default_rng(1)
        space = grammar.Space(max_lookback=120)
        trees = [t for t in (grammar.random_tree(rng, space) for _ in range(80)) if t is not None]
        self.assertGreater(len(trees), 60)
        for tree in trees:
            self.assertTrue(space.admits(tree))
            text = grammar.to_text(tree)
            self.assertEqual(grammar.from_text(text), tree)
        edits = [grammar.mutate(rng, t, space) for t in trees[:40]]
        self.assertGreater(sum(e is not None for e in edits), 30)
        for edit in edits:
            if edit is not None:
                self.assertTrue(space.admits(edit[0]))
        children = [grammar.crossover(rng, trees[i], trees[i + 1], space) for i in range(0, 40, 2)]
        for child in children:
            if child is not None:
                self.assertTrue(space.admits(child))

    def test_constants_round_trip_exactly_and_windows_must_be_whole(self):
        from autoresearch.domains.alpha import grammar

        for text in ("rank(close) * 1234567.5", "signed_power(rank(returns), 0.33333333)", "rank(volume) * 1e-07"):
            with self.subTest(text=text):
                node = grammar.from_text(text)
                self.assertEqual(grammar.from_text(grammar.to_text(node)), node)
        self.assertNotEqual(grammar.canonical_key(grammar.from_text("signed_power(rank(returns), 0.33333333)")),
                            grammar.canonical_key(grammar.from_text("signed_power(rank(returns), 0.333333)")))
        for text in ("ts_corr(close, volume, 5.7)", "ts_mean(close, 2.5)", "rank(close"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                grammar.from_text(text)
        blend = grammar.from_text("add(rank(close), -rank(volume), rank(returns))")
        self.assertEqual(blend.op, "add")
        self.assertEqual(grammar.text_size("add(rank(close), rank(volume), rank(returns))"), 7)

    def test_lookback_adds_nested_windows_and_unsupported_text_is_refused(self):
        from autoresearch.domains.alpha import grammar

        self.assertEqual(grammar.max_window(grammar.from_text("ts_mean(ts_delta(close, 5), 20)")), 25)
        for text in ("a = close; a", "trade_when(close, volume, 1)", "rank(close, rate=0)", "close > open"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                grammar.from_text(text)


@unittest.skipUnless(HAVE_NUMERIC, "numpy and pandas are optional outside the Qlib environment")
class SelectionReviewTests(unittest.TestCase):
    def result(self, sharpe=3.2, daily=0.2, halves=(2.5, 3.5), stress=2.0, perturbations=None, probe="ok"):
        if perturbations is None:
            perturbations = [{"factor": 0.75, "sharpe": 2.6}, {"factor": 1.33, "sharpe": 2.9}]
        return {"status": "ok", "settings": {"cost_bps": 5},
                "metrics": {"validation": {"days": 400, "sharpe": sharpe, "returns": 0.2, "turnover": 0.3,
                                           "drawdown": 0.05}},
                "periods": {"validation": {"days": 400}},
                "robustness": {"daily_sharpe": daily, "days": 400, "skew": 0.0, "kurtosis": 3.0,
                               "half_sharpes": list(halves), "cost_stress_sharpe": stress,
                               "perturbations": perturbations, "perturbation": {"kind": "window", "status": probe}}}

    def test_null_variance_is_a_familywise_test_and_empirical_is_stricter(self):
        from autoresearch.domains.alpha.domain import POLICY_DEFAULTS, review_signal

        mixed = [0.0, 0.01, 0.2, 0.25, 0.3, -0.05, 0.02, 0.18]   # real signals widen the spread
        context = {"n_validation_trials": len(mixed), "validation_scores": mixed}
        null = review_signal(self.result(), context, dict(POLICY_DEFAULTS))
        empirical = review_signal(self.result(), context, {**POLICY_DEFAULTS, "dsr_variance": "empirical"})
        self.assertEqual(null["decision"], "promote")
        self.assertGreater(null["checks"]["deflated_sharpe"]["dsr"], empirical["checks"]["deflated_sharpe"]["dsr"])
        self.assertEqual(empirical["decision"], "reject")
        many = {"n_validation_trials": 5000, "validation_scores": mixed}
        self.assertEqual(review_signal(self.result(daily=0.12), many, dict(POLICY_DEFAULTS))["decision"], "reject")

    def test_vetoes_reject_after_the_gates_pass(self):
        from autoresearch.domains.alpha.domain import POLICY_DEFAULTS, review_signal

        context = {"n_validation_trials": 1, "validation_scores": []}
        self.assertIn("2.0x costs", review_signal(self.result(stress=-0.1), context, dict(POLICY_DEFAULTS))["reason"])
        self.assertIn("validation half", review_signal(self.result(halves=(2.0, -0.2)), context, dict(POLICY_DEFAULTS))["reason"])
        self.assertEqual(review_signal(self.result(sharpe=0.5), context, dict(POLICY_DEFAULTS))["decision"], "reject")

    def test_cost_stress_must_retain_a_fraction_of_the_validation_sharpe(self):
        from autoresearch.domains.alpha.domain import POLICY_DEFAULTS, review_signal

        context = {"n_validation_trials": 1, "validation_scores": []}
        weak = review_signal(self.result(sharpe=3.2, stress=0.8), context, dict(POLICY_DEFAULTS))   # positive, but 25%
        self.assertEqual(weak["decision"], "reject")
        self.assertIn("2.0x costs", weak["reason"])
        self.assertAlmostEqual(weak["checks"]["cost_stress_retention"], 0.25)
        self.assertEqual(review_signal(self.result(sharpe=3.2, stress=1.7), context, dict(POLICY_DEFAULTS))["decision"],
                         "promote")   # 53% survives the default floor of 0.35
        lenient = {**POLICY_DEFAULTS, "min_cost_stress_retention": 0.2}
        self.assertEqual(review_signal(self.result(sharpe=3.2, stress=0.8), context, lenient)["decision"], "promote")
        strict = {**POLICY_DEFAULTS, "min_cost_stress_retention": 0.6}
        self.assertEqual(review_signal(self.result(sharpe=3.2, stress=1.7), context, strict)["decision"], "reject")

    def test_robustness_without_an_applicable_perturbation_is_unavailable_not_a_pass(self):
        from autoresearch.domains.alpha.domain import POLICY_DEFAULTS, review_signal

        context = {"n_validation_trials": 1, "validation_scores": []}
        probed = review_signal(self.result(), context, dict(POLICY_DEFAULTS))
        self.assertEqual((probed["decision"], probed["checks"]["robustness"]), ("promote", "probed"))
        for status, phrase in (("no_window", "no scalable window"), ("unparsable", "could not be parsed"),
                               (None, "no perturbation probe")):
            with self.subTest(status=status):
                result = self.result(perturbations=[], probe=status)
                if status is None:
                    del result["robustness"]["perturbation"]      # a result written before the probe was recorded
                review = review_signal(result, context, dict(POLICY_DEFAULTS))
                self.assertEqual(review["decision"], "park")
                self.assertEqual(review["checks"]["robustness"], "unavailable")
                self.assertIsNone(review["checks"]["perturbation_retention"])
                self.assertIn("Robustness unavailable", review["reason"])
                self.assertIn(phrase, review["reason"])
        # Other vetoes still reject, and are named, before an unavailable probe parks anything.
        rejected = review_signal(self.result(perturbations=[], probe="no_window", stress=-0.1), context,
                                 dict(POLICY_DEFAULTS))
        self.assertEqual((rejected["decision"], rejected["checks"]["robustness"]), ("reject", "unavailable"))
        # A perturbation that collapses the Sharpe is a veto, as before.
        fragile = review_signal(self.result(perturbations=[{"factor": 0.75, "sharpe": 0.4}, {"factor": 1.33, "sharpe": 3.0}]),
                                context, dict(POLICY_DEFAULTS))
        self.assertEqual(fragile["decision"], "reject")
        self.assertIn("perturbations lose too much", fragile["reason"])

    def test_deflated_check_recomputes_the_statistic_at_the_final_count(self):
        from autoresearch.domains.alpha.domain import deflated_check

        entry = {"checks": {"deflated_sharpe": {"sharpe": 0.2, "observations": 400, "skew": 0.0, "kurtosis": 3.0,
                                                "dsr": 0.999, "n_trials": 1}}}
        few = deflated_check(entry, {"n_validation_trials": 1, "validation_scores": []}, 0.95)
        many = deflated_check(entry, {"n_validation_trials": 5000, "validation_scores": []}, 0.95)
        self.assertTrue(few["eligible"])
        self.assertFalse(many["eligible"])
        self.assertGreater(few["dsr"], many["dsr"])
        self.assertGreater(many["benchmark_sharpe"], few["benchmark_sharpe"])
        self.assertEqual((many["statistic"], many["min_dsr"], many["trials_at_review"]), ("deflated_sharpe", 0.95, 1))


@unittest.skipUnless(HAVE_NUMERIC, "numpy and pandas are optional outside the Qlib environment")
class PlantedPanelAndEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from autoresearch.domains.alpha.planted import create_planted_dataset

        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.dataset = create_planted_dataset(root / "planted", symbols=80, days=700, seed=4)
        cls.null = create_planted_dataset(root / "null", symbols=80, days=700, seed=4, strength={})
        cls.dates = (cls.dataset / "qlib_data/calendars/day.txt").read_text().split()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def settings(self, dataset=None):
        d = self.dates
        return {"dataset": str(dataset or self.dataset), "universe": "TOP80", "cost_bps": 5, "warmup": 80,
                "truncation": 0.1, "development_start": d[90], "development_end": d[349],
                "validation_start": d[350], "validation_end": d[599], "holdout_start": d[600]}

    def test_planted_manifest_is_labelled_artificial_and_reopens(self):
        from autoresearch.domains.alpha.planted import create_planted_dataset, PROVIDER

        manifest = json.loads((self.dataset / "manifest.json").read_text())
        self.assertEqual(manifest["provider"], PROVIDER)
        self.assertIn("Artificial", manifest["note"])
        self.assertEqual(create_planted_dataset(self.dataset), self.dataset)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "manifest.json").write_text('{"provider": "real"}')
            with self.assertRaises(ValueError):
                create_planted_dataset(tmp)

    def test_planted_oracles_beat_their_null_twins(self):
        from autoresearch.domains.alpha.evaluator import SealedEvaluator
        from autoresearch.domains.alpha.planted import ORACLES
        from qlib_yahoo.alphalab.settings import Settings

        bounds = {k: v for k, v in self.settings().items() if k.endswith(("_start", "_end")) and not k.startswith("holdout")}
        settings = Settings(universe="TOP80", cost_bps=0, warmup=80, delay=1, truncation=0.1)
        planted = SealedEvaluator(self.dataset, settings, bounds)
        null = SealedEvaluator(self.null, settings, bounds)
        combined = "add(" + ", ".join(ORACLES.values()) + ")"
        self.assertGreater(planted.simulate(combined, "development")["summary"]["sharpe"],
                           null.simulate(combined, "development")["summary"]["sharpe"] + 1.0)

    def test_sealed_workspaces_end_at_their_period_and_match_the_canonical_backend(self):
        from autoresearch.backend import AlphaLabBackend
        from autoresearch.domains.alpha.evaluator import SealedEvaluator

        with tempfile.TemporaryDirectory() as tmp:
            backend = AlphaLabBackend(self.settings(), Path(tmp))
            evaluator = SealedEvaluator(backend.dataset, backend.settings, backend.bounds)
            expression = "-rank(ts_delta(close, 5) / ts_delay(close, 5))"
            canonical = backend.evaluate(expression)
            self.assertEqual(canonical["status"], "ok", canonical.get("error"))
            for period in ("development", "validation"):
                fast = evaluator.simulate(expression, period)["summary"]
                for key in ("sharpe", "returns", "turnover", "drawdown"):
                    self.assertAlmostEqual(fast[key], canonical["metrics"][period][key], places=6, msg=(period, key))
            dev = evaluator._workspace("development")
            val = evaluator._workspace("validation")
            self.assertEqual(str(dev.dates[-1]), self.settings()["development_end"])
            self.assertEqual(str(val.dates[-1]), self.settings()["validation_end"])
            self.assertTrue(all(len(v) == len(val.dates) for v in val.fields.values()))
            # The loader itself stopped at the validation end: the cached panel was never longer.
            self.assertEqual(str(evaluator._panel.dates[-1]), self.settings()["validation_end"])
            self.assertTrue(all(len(v) == len(evaluator._panel.dates) for v in evaluator._panel.fields.values()))
            with self.assertRaises(ValueError):
                evaluator.simulate(expression, "holdout")
            with self.assertRaises(ValueError):
                evaluator._workspace("holdout")

    def test_search_process_holds_no_session_after_the_validation_end(self):
        """Memory proof: after a screen and a validation, no array reachable from the domain holds a
        session after the validation end, because the loader never deserialized one. The holdout
        path, run once, opens its own panel and leaves nothing behind."""
        from autoresearch.lab.campaign import LabCampaign

        settings = self.settings()
        sealed_sessions = self.dates.index(settings["validation_end"]) + 1
        with tempfile.TemporaryDirectory() as tmp:
            domain = LabCampaign(self.campaign_config(Path(tmp))).domain
            expression = "-rank(ts_delta(close, 5) / ts_delay(close, 5))"
            self.assertEqual(domain.screen([{"expression": expression, "complexity": 5}])[0]["status"], "ok")
            validated = domain.validate({"expression": expression})
            self.assertEqual(validated["status"], "ok", validated.get("error"))
            panel = domain.evaluator._panel
            self.assertEqual(str(panel.dates[-1]), settings["validation_end"])
            self.assertEqual({len(values) for values in panel.fields.values()}, {sealed_sessions})
            self.assertEqual(set(domain.evaluator._workspaces), {"development", "validation"})
            assert_sessions_bounded(self, domain, settings["validation_end"], sealed_sessions, symbols=80)
            # Negative control: the walk does reach what the domain holds.
            domain.evaluator.probe = np.zeros((sealed_sessions + 1, 80), dtype=np.float32)
            with self.assertRaises(AssertionError):
                assert_sessions_bounded(self, domain, settings["validation_end"], sealed_sessions, symbols=80)
            del domain.evaluator.probe
            final = domain.evaluate_holdout({"expression": expression})
            self.assertEqual(final["status"], "ok", final.get("error"))
            self.assertEqual(final["metrics"]["holdout"]["start"], settings["holdout_start"])
            self.assertGreater(final["metrics"]["holdout"]["days"], 50)
            assert_sessions_bounded(self, domain, settings["validation_end"], sealed_sessions, symbols=80)
            self.assertIs(domain.evaluator._panel, panel)

    def test_holdout_path_opens_exactly_its_window_and_matches_the_canonical_engine(self):
        from autoresearch.backend import AlphaLabBackend
        from autoresearch.domains.alpha.holdout import holdout_window, open_holdout_panel
        from autoresearch.lab.campaign import LabCampaign
        from qlib_yahoo.panel import load_panel

        d, settings = self.dates, self.settings()
        panel = open_holdout_panel(self.dataset, holdout_start=settings["holdout_start"], warmup=80)
        self.assertEqual((str(panel.dates[0]), str(panel.dates[-1])), (d[600 - 80], d[-1]))
        full = load_panel(self.dataset, list(panel.fields))
        for name, values in panel.fields.items():
            self.assertEqual(values.shape, (len(d) - 520, 80))
            np.testing.assert_array_equal(values, full.fields[name][520:], err_msg=name)
        self.assertEqual(holdout_window(self.dataset, holdout_start=d[600], warmup=80, history_from=d[90]),
                         (d[10], d[-1]))
        self.assertEqual(holdout_window(self.dataset, holdout_start=d[50], warmup=80), (d[0], d[-1]))
        with self.assertRaises(ValueError):
            holdout_window(self.dataset, holdout_start=d[600], warmup=80, history_from=d[601])
        with tempfile.TemporaryDirectory() as tmp:
            domain = LabCampaign(self.campaign_config(Path(tmp))).domain
            expression = "-rank(ts_delta(close, 5) / ts_delay(close, 5))"
            lab = domain.evaluate_holdout({"expression": expression})
            canonical = AlphaLabBackend(settings, Path(tmp) / "canonical").evaluate_holdout(expression)
            self.assertEqual(lab["status"], "ok", lab.get("error"))
            self.assertEqual(canonical["status"], "ok", canonical.get("error"))
            self.assertEqual(lab["metrics"]["holdout"], canonical["metrics"]["holdout"])
            self.assertEqual(lab["settings"], canonical["settings"])
            self.assertEqual((lab["engine"], lab["data_identity"]), (canonical["engine"], canonical["data_identity"]))
            self.assertFalse(lab["cached"])
            artifact = json.loads((domain.backend.runtime_root / lab["artifacts"][0]).read_text())
            self.assertEqual(artifact["holdout"]["daily"]["dates"][0], settings["holdout_start"])
            self.assertEqual(artifact["holdout"]["summary"], lab["metrics"]["holdout"])

    def test_long_decay_windows_validate_in_the_research_backend(self):
        # Regression: the backend's eight-row probe used to reject ts_decay_linear windows above seven.
        from autoresearch.backend import AlphaLabBackend

        with tempfile.TemporaryDirectory() as tmp:
            backend = AlphaLabBackend(self.settings(), Path(tmp))
            self.assertEqual(backend.validate("-rank(ts_decay_linear(returns, 20))"), [])

    def campaign_config(self, root, dataset=None, **policy):
        return {"name": "test", "domain": "alpha", "seed": 2, "research_root": str(root / "records"),
                "runtime_root": str(root / "runtime"), "lineage_root": str(root / "lineage"),
                "settings": self.settings(dataset),
                "policy": {"min_dsr": 0.5, "min_sharpe": 0.5, "min_dev_sharpe": 0.3, **policy},
                "limits": {"max_rounds": 3, "proposals_per_round": 24, "validations_per_round": 2,
                           "max_validation_trials": 8}}

    def test_campaign_discovers_finalizes_once_and_exports_a_readable_batch(self):
        from autoresearch.handoff import campaign_report, export_batch
        from autoresearch.lab.campaign import LabCampaign
        from autoresearch.lab.report import render_markdown, summary
        from qlib_yahoo.alphalab.batch import read_expressions

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = LabCampaign(self.campaign_config(root))
            for _ in range(3):
                campaign.tick()
            status = campaign.status()
            self.assertGreater(status["screen_trials"], 30)
            self.assertGreater(status["validation_trials"], 0)
            self.assertTrue(campaign.ledger.verify()["ok"])
            self.assertGreater(status["promotions"], 0, "a planted panel should yield at least one promotion")
            generators = set(status["generators"])
            self.assertTrue({"template", "random"} <= generators)
            report = summary(campaign)
            self.assertEqual(report["funnel"]["validated"], status["validation_trials"])
            self.assertIn("chain intact", render_markdown(report))
            finals = campaign.finalize()
            self.assertTrue(finals)
            self.assertTrue(all(f["holdout"].get("days", 0) > 0 for f in finals))
            with self.assertRaises(ValueError):
                campaign.finalize()
            batch = export_batch(campaign_report(campaign.store, campaign.campaign_id), root / "out.txt")
            expressions = read_expressions(batch)
            self.assertEqual(len(expressions), status["promotions"])

    def test_a_copied_dataset_meets_the_same_registry(self):
        import os
        import shutil
        from unittest import mock

        from autoresearch.lab.campaign import LabCampaign

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            copy = root / "elsewhere" / "planted-copy"
            shutil.copytree(self.dataset, copy)
            with mock.patch.dict(os.environ, {"TUBBY_QLIB_DATA_ROOT": str(root / "data-root")}):
                original = self.campaign_config(root / "a")
                moved = self.campaign_config(root / "b", dataset=copy)
                for config in (original, moved):
                    del config["lineage_root"]
                first, second = LabCampaign(original), LabCampaign(moved)
                self.assertEqual(first.registry.data_identity, second.registry.data_identity)
                self.assertEqual(first.registry.path, second.registry.path)
                self.assertEqual(first.registry.path.parent, (root / "data-root").resolve() / "autoresearch" / "lineage")
                self.assertTrue(first.lineage_summary()["canonical"])

    def test_a_packed_snapshot_is_refused_by_the_sealed_and_holdout_loaders(self):
        import shutil

        from autoresearch.domains.alpha.evaluator import sealed_panel
        from autoresearch.domains.alpha.holdout import open_holdout_panel

        # A packed snapshot is verified into a module-global cache whole (qlib_yahoo/packed.py), so a bounded
        # window would not bound what the process holds: both loaders refuse it by its manifest.
        with tempfile.TemporaryDirectory() as tmp:
            packed = Path(tmp) / "packed"
            shutil.copytree(self.dataset, packed)
            manifest = json.loads((packed / "manifest.json").read_text(encoding="utf-8"))
            manifest["storage"] = {"archive_sha256": "0" * 64}
            (packed / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "plain snapshot"):
                sealed_panel(packed, self.dates[599])
            with self.assertRaisesRegex(ValueError, "plain snapshot"):
                open_holdout_panel(packed, holdout_start=self.dates[600], warmup=80)
        panel = sealed_panel(self.dataset, self.dates[599])            # the plain snapshot itself is fine
        self.assertEqual(str(panel.dates[-1]), self.dates[599])

    def test_a_packed_snapshot_is_refused_before_anything_verifies_it(self):
        """Review 3, finding 4: the refusal must come before dataset verification, which caches a packed archive whole."""
        import qlib_yahoo.packed as packed
        from autoresearch.domains.alpha.planted import create_planted_dataset
        from autoresearch.domains.ml.tasks import PanelTask
        from autoresearch.lab.campaign import LabCampaign
        from qlib_yahoo.packed import pack_dataset

        # The packer refuses symlinked paths, so the temporary folder is the resolved one (macOS /var -> /private/var).
        with tempfile.TemporaryDirectory(dir=str(Path(tempfile.gettempdir()).resolve())) as tmp:
            root = Path(tmp)
            plain = create_planted_dataset(root / "planted", symbols=20, days=300, seed=4)
            # The packer wants the binary file count the planted manifest does not record; add it to this copy.
            from qlib_yahoo.dataset import tree_sha256
            manifest = json.loads((plain / "manifest.json").read_text(encoding="utf-8"))
            manifest["qlib_binary_file_count"] = tree_sha256(plain / "qlib_data")[1]
            (plain / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
            packed_dataset = pack_dataset(plain, root / "packed")
            self.assertIn("storage", json.loads((packed_dataset / "manifest.json").read_text(encoding="utf-8")))
            before = packed._verified
            packed._verified = None
            try:
                dates = (plain / "qlib_data/calendars/day.txt").read_text().split()
                settings = {"dataset": str(packed_dataset), "universe": "TOP20", "cost_bps": 5, "warmup": 40,
                            "truncation": 0.1, "development_start": dates[50], "development_end": dates[149],
                            "validation_start": dates[150], "validation_end": dates[229], "holdout_start": dates[230]}
                config = {**self.campaign_config(root), "settings": settings}
                with self.assertRaisesRegex(ValueError, "plain snapshot"):
                    LabCampaign(config)
                self.assertIsNone(packed._verified)                   # nothing read the archive into the process
                with self.assertRaisesRegex(ValueError, "plain snapshot"):
                    PanelTask({**settings, "task": "panel"}, runtime=root / "runtime-ml")
                self.assertIsNone(packed._verified)
            finally:
                packed._verified = before

    def test_sealed_screen_never_sees_validation_prices(self):
        from autoresearch.lab.campaign import LabCampaign

        with tempfile.TemporaryDirectory() as tmp:
            campaign = LabCampaign(self.campaign_config(Path(tmp)))
            seen = []
            original = campaign.domain.evaluator.simulate

            def spy(expression, period, **overrides):
                seen.append(period)
                return original(expression, period, **overrides)

            campaign.domain.evaluator.simulate = spy
            outcomes = campaign.domain.screen([{"expression": "rank(volume / ts_mean(volume, 20))", "complexity": 4}])
            self.assertEqual(set(seen), {"development"})
            self.assertEqual(outcomes[0]["status"], "ok")
            self.assertEqual(len(outcomes[0]["metrics"]["fold_sharpes"]), 3)

    def test_validation_records_whether_a_window_perturbation_applied(self):
        from autoresearch.lab.campaign import LabCampaign

        with tempfile.TemporaryDirectory() as tmp:
            domain = LabCampaign(self.campaign_config(Path(tmp))).domain
            windowed = domain.validate({"expression": "rank(volume / ts_mean(volume, 20))"})
            self.assertEqual(windowed["robustness"]["perturbation"], {"kind": "window", "status": "ok"})
            self.assertEqual([p["factor"] for p in windowed["robustness"]["perturbations"]], [0.75, 1.33])
            flat = domain.validate({"expression": "rank(close / open)"})
            self.assertEqual(flat["robustness"]["perturbation"], {"kind": "window", "status": "no_window"})
            self.assertEqual(flat["robustness"]["perturbations"], [])
            review = domain.review({"expression": "rank(close / open)"}, flat,
                                   {"n_validation_trials": 1, "validation_scores": [], "policy": {}})
            self.assertNotEqual(review["decision"], "promote")
            if review["checks"].get("gates") == "promote":
                self.assertEqual(review["checks"]["robustness"], "unavailable")

    def test_orientation_negates_a_backwards_signal_on_development_only(self):
        from autoresearch.lab.campaign import LabCampaign

        with tempfile.TemporaryDirectory() as tmp:
            campaign = LabCampaign(self.campaign_config(Path(tmp)))
            outcome = campaign.domain.screen([{"expression": "rank(ts_delta(close, 5) / ts_delay(close, 5))",
                                               "complexity": 5}])[0]
            self.assertEqual(outcome["candidate"].get("oriented"), "negated")
            self.assertTrue(outcome["candidate"]["expression"].startswith("-"))
            self.assertGreater(outcome["metrics"]["sharpe"], 0)

    def test_bad_seeds_and_submissions_are_counted_not_fatal_and_cannot_buy_priority(self):
        from autoresearch.lab.campaign import LabCampaign

        with tempfile.TemporaryDirectory() as tmp:
            config = self.campaign_config(Path(tmp))
            config["settings"]["search"] = {"seeds": ["rank(close"]}
            campaign = LabCampaign(config)
            campaign.submit({"expression": "rank(volume / ts_mean(volume, 20))", "complexity": -500,
                             "diversity_group": "blend", "rationale": "volume shock"}, "agent")
            outcome = campaign.tick()
            self.assertEqual(outcome["status"], "round")
            screens = {e["generator"]: e for e in campaign.ledger.entries() if e["stage"] == "screen"
                       and e["generator"] in ("seed", "inbox")}
            self.assertEqual(screens["seed"]["status"], "error")
            entry = next(e for e in campaign.state["population"] if e["generator"] == "inbox")
            self.assertEqual(entry["candidate"]["complexity"], 5)   # rank, div, volume, ts_mean, volume
            self.assertNotIn("diversity_group", entry["candidate"])

    def test_unknown_settings_and_policy_keys_are_refused(self):
        from autoresearch.lab.campaign import LabCampaign

        with tempfile.TemporaryDirectory() as tmp:
            config = self.campaign_config(Path(tmp))
            config["settings"]["search"] = {"bogus": 1}
            with self.assertRaises(ValueError):
                LabCampaign(config)
            config = self.campaign_config(Path(tmp), not_a_gate=1)
            with self.assertRaises(ValueError):
                LabCampaign(config)


if __name__ == "__main__":
    unittest.main()
