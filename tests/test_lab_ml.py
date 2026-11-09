"""Machine-learning domain: sealed tasks, recipe search, selection control and full campaigns."""

from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HAVE_NUMERIC = bool(importlib.util.find_spec("numpy") and importlib.util.find_spec("pandas"))
HAVE_NUMPY = bool(importlib.util.find_spec("numpy"))

if HAVE_NUMPY:
    import numpy as np


@unittest.skipUnless(HAVE_NUMPY, "numpy is optional outside the research environments")
class TabularTaskTests(unittest.TestCase):
    def test_builtin_split_is_deterministic_and_the_holdout_is_not_held(self):
        from autoresearch.domains.ml.tasks import TabularTask

        task = TabularTask({"task": "tabular", "table": "builtin:friedman1", "rows": 1000}, seed=3)
        again = TabularTask({"task": "tabular", "table": "builtin:friedman1", "rows": 1000}, seed=3)
        X, y = task.development()
        self.assertEqual(X.shape, (600, 10))
        self.assertEqual(len(task.validation()[1]), 200)
        np.testing.assert_array_equal(X, again.development()[0])
        held_rows = set(task._rows["holdout"].tolist())
        self.assertEqual(len(held_rows), 200)
        self.assertFalse(held_rows & set(task._rows["development"].tolist()))
        self.assertEqual(set(task._arrays), {"development", "validation"})
        self.assertEqual(task.holdout()["X"].shape, (200, 10))
        self.assertEqual(task.identity()["split_mode"], "seeded permutation")

    def test_csv_tasks_split_chronologically_and_read_missing_values(self):
        from autoresearch.domains.ml.tasks import TabularTask

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "table.csv"
            rng = np.random.default_rng(0)
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["date", "a", "b", "label"])
                for i in range(100):
                    a = "" if i == 7 else f"{rng.normal():.4f}"
                    writer.writerow([f"2020-{1 + i // 28:02d}-{1 + i % 28:02d}", a, f"{i}", i % 2])
            task = TabularTask({"task": "tabular", "table": str(path), "target": "label", "problem": "classification",
                                "order_by": "date"})
            X, y = task.development()
            self.assertEqual(task.features, ["a", "b"])
            self.assertTrue(np.isnan(X[:, 0]).any())
            self.assertTrue((np.diff(X[:, 1]) > 0).all())          # chronological: b is the row number
            self.assertLess(X[:, 1].max(), task.validation()[0][:, 1].min())
            with self.assertRaises(ValueError):
                TabularTask({"task": "tabular", "table": str(path), "target": "missing"})
            numeric = Path(tmp) / "numeric.csv"
            with numeric.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["day", "a", "label"])
                for day in range(1, 201):                       # text order would put 100 before 11
                    writer.writerow([day, f"{rng.normal():.4f}", f"{rng.normal():.4f}"])
            ordered = TabularTask({"task": "tabular", "table": str(numeric), "target": "label", "order_by": "day"})
            days = np.arange(1, 201)
            self.assertLess(days[ordered._rows["development"]].max(), days[ordered._rows["validation"]].min())
            self.assertLess(days[ordered._rows["validation"]].max(), days[ordered._rows["holdout"]].min())
            with self.assertRaises(ValueError):
                TabularTask({"task": "tabular", "table": str(path), "target": "label", "split": {"development": 0.9}})

    def test_paired_bootstrap_detects_real_improvement_and_not_noise(self):
        from autoresearch.domains.ml.domain import paired_bootstrap

        rng = np.random.default_rng(1)
        y = rng.normal(size=400)
        good = y + rng.normal(0, 0.5, 400)
        flat = np.zeros(400)
        better = paired_bootstrap("rmse", y, good, flat, resamples=500, seed=0)
        self.assertGreater(better["improvement"], 0)
        # No resample reached zero: the p-value is reported at the resolution floor, never extrapolated below it.
        self.assertEqual(better["p_value"], 1 / 501)
        self.assertEqual((better["p_value_method"], better["p_value_floor"], better["block_length"]),
                         ("iid-bootstrap", 1 / 501, 1))
        noise = paired_bootstrap("rmse", y, rng.normal(0, 0.01, 400), flat, resamples=500, seed=0)
        self.assertGreater(noise["p_value"], 0.05)

    def test_block_bootstrap_widens_the_interval_for_serially_dependent_losses(self):
        from autoresearch.domains.ml.domain import block_length_for, paired_bootstrap

        rng = np.random.default_rng(3)
        n, phi = 1000, 0.9
        shocks = rng.normal(0, 0.2 * np.sqrt(1 - phi ** 2), n)
        a = np.empty(n)
        a[0] = shocks[0]
        for t in range(1, n):
            a[t] = phi * a[t - 1] + shocks[t]
        # Per-row loss difference is exactly -a_t: a zero-mean AR(1) series, like residual losses on ordered data.
        y, base, model = np.zeros(n), -np.ones(n), -(1 + a)
        iid = paired_bootstrap("mae", y, model, base, resamples=1000, seed=0)
        block = paired_bootstrap("mae", y, model, base, resamples=1000, seed=0, block_length=block_length_for(n))
        self.assertEqual(block_length_for(n), 10)
        self.assertEqual((block["p_value_method"], block["block_length"]), ("moving-block-bootstrap", 10))
        self.assertAlmostEqual(block["improvement"], iid["improvement"])
        self.assertGreater(block["ci_high"] - block["ci_low"], 1.8 * (iid["ci_high"] - iid["ci_low"]))

    def test_chronological_tasks_use_the_block_bootstrap(self):
        from autoresearch.domains.ml.domain import MLDomain, block_length_for

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "series.csv"
            rng = np.random.default_rng(0)
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["day", "a", "b", "label"])
                for day in range(1, 401):
                    a, b = rng.normal(), rng.normal()
                    writer.writerow([day, f"{a:.4f}", f"{b:.4f}", f"{2 * a - b + rng.normal(0, 0.5):.4f}"])
            settings = {"task": "tabular", "table": str(path), "target": "label", "order_by": "day",
                        "search": {"families": ["ridge"]}}
            domain = MLDomain(settings, Path(tmp) / "runtime")
            self.assertTrue(domain.task.ordered)
            validation = domain.validate({"family": "ridge", "params": {}})["metrics"]["validation"]
            self.assertEqual(validation["p_value_method"], "moving-block-bootstrap")
            self.assertEqual(validation["block_length"], block_length_for(validation["rows"]))
            self.assertEqual(validation["p_value"], validation["p_value_floor"])    # a real relation: 1 / (bootstrap + 1)
            builtin = MLDomain({"task": "tabular", "table": "builtin:friedman1", "rows": 500,
                                "search": {"families": ["ridge"]}}, Path(tmp) / "runtime2")
            self.assertFalse(builtin.task.ordered)
            self.assertEqual(builtin.validate({"family": "ridge", "params": {}})["metrics"]["validation"]["p_value_method"],
                             "iid-bootstrap")

    def test_recipe_perturbation_scales_numeric_parameters_within_their_ranges(self):
        from autoresearch.domains.ml.search import scale_recipe

        recipe = {"family": "gbdt", "params": {"max_depth": 3, "n_estimators": 100}}
        down, up = scale_recipe(recipe, 0.75, "regression"), scale_recipe(recipe, 1.33, "regression")
        self.assertEqual((down["params"]["max_depth"], down["params"]["n_estimators"], down["params"]["subsample"]), (2, 75, 0.75))
        self.assertEqual((up["params"]["max_depth"], up["params"]["n_estimators"], up["params"]["subsample"]), (4, 133, 1.0))
        self.assertEqual((down["params"]["early_stopping_rounds"], up["params"]["early_stopping_rounds"]), (0, 0))  # fixed
        ridge = scale_recipe({"family": "ridge", "params": {"alpha": 2.0, "fit_intercept": False}}, 0.75, "regression")
        self.assertEqual((ridge["params"]["alpha"], ridge["params"]["fit_intercept"]), (1.5, False))


@unittest.skipUnless(HAVE_NUMPY, "numpy is optional outside the research environments")
class SearchTests(unittest.TestCase):
    def test_sampling_mutation_and_keys(self):
        from autoresearch.domains.ml import search

        rng = np.random.default_rng(2)
        families = search.families_for("regression")
        seen = set()
        for _ in range(40):
            recipe = search.sample(rng, families[int(rng.integers(len(families)))], "regression")
            child = search.mutate(rng, recipe, "regression", families)
            for item in (recipe, child):
                key = search.recipe_key(item, "regression", "tabular")
                self.assertTrue(key.startswith("recipe_"))
                seen.add(key)
                self.assertEqual(key, search.recipe_key(json.loads(json.dumps(item)), "regression", "tabular"))
        self.assertGreater(len(seen), 60)
        default = search.default("gbdt", "regression")
        self.assertEqual(search.recipe_key(default, "regression", "tabular"),
                         search.recipe_key({"family": "gbdt", "params": {}}, "regression", "tabular"))
        with self.assertRaises(ValueError):
            search.families_for("classification", ["ridge"])


@unittest.skipUnless(HAVE_NUMERIC, "numpy and pandas are optional outside the Qlib environment")
class CampaignTests(unittest.TestCase):
    def config(self, root, table, **limits):
        from autoresearch.domains.ml.benchmark import tabular_config

        config = tabular_config("test-" + table, table, root, rounds=3, seed=5, proposals=6)
        config["lineage_root"] = str(root / "lineage")      # built-in tables have no data folder of their own
        config["settings"]["rows"] = 1500
        config["settings"]["search"] = {"families": ["ridge", "gbdt"] if table != "interaction" else ["logistic", "gbdt"]}
        config["limits"].update(limits)
        return config

    def test_friedman_search_beats_the_baseline_and_confirms_on_the_holdout(self):
        from autoresearch.lab.campaign import LabCampaign
        from autoresearch.lab.report import render_markdown, summary

        with tempfile.TemporaryDirectory() as tmp:
            campaign = LabCampaign(self.config(Path(tmp), "friedman1"))
            for _ in range(3):
                campaign.tick()
            status = campaign.status()
            self.assertGreater(status["promotions"], 0)
            self.assertTrue(campaign.ledger.verify()["ok"])
            finals = campaign.finalize()
            confirmed = [f for f in finals if f["decision"] == "confirmed"]
            self.assertTrue(confirmed)
            best = min(f["holdout"]["score"] for f in confirmed)
            self.assertLess(best, 2.5)              # baseline RMSE is about 5; the noise floor is 1.0
            self.assertGreater(best, 0.9)
            self.assertEqual(confirmed[0]["holdout"]["noise_floor"], 1.0)
            report = summary(campaign)
            self.assertIn("recipe", report["leaderboard"][0])
            self.assertIn("p-value", render_markdown(report))
            artifacts = list((Path(tmp) / "runtime" / "domain" / "models").glob("*-final.json"))
            self.assertTrue(artifacts)
            from autoresearch.domains.ml.models import model_from_json
            model = model_from_json(json.loads(artifacts[0].read_text()))
            self.assertEqual(model.predict(np.zeros((2, 10))).shape, (2,))

    def test_null_task_promotes_nothing(self):
        from autoresearch.lab.campaign import LabCampaign

        with tempfile.TemporaryDirectory() as tmp:
            campaign = LabCampaign(self.config(Path(tmp), "null_regression"))
            for _ in range(3):
                campaign.tick()
            self.assertEqual(campaign.status()["promotions"], 0)
            self.assertEqual(campaign.finalize(), [])

    def test_classification_reports_the_bayes_optimal_loss(self):
        from autoresearch.lab.campaign import LabCampaign

        with tempfile.TemporaryDirectory() as tmp:
            campaign = LabCampaign(self.config(Path(tmp), "interaction"))
            for _ in range(3):
                campaign.tick()
            if not campaign.state["accepted"]:
                self.skipTest("no promotion in this short run")
            final = campaign.finalize()[0]
            holdout = final["holdout"]
            self.assertGreater(holdout["score"], holdout["bayes_optimal_score"] - 0.05)
            self.assertLess(holdout["score"], holdout["baseline"])

    def test_agent_recipes_go_through_the_inbox(self):
        from autoresearch.lab.campaign import LabCampaign

        with tempfile.TemporaryDirectory() as tmp:
            campaign = LabCampaign(self.config(Path(tmp), "friedman1"))
            receipt = campaign.submit({"family": "gbdt", "params": {"max_depth": 2, "n_estimators": 60}}, "agent")
            campaign.tick()
            screened = [e for e in campaign.ledger.entries() if e["stage"] == "screen" and e["generator"] == "inbox"]
            self.assertEqual([e["key"] for e in screened], [receipt["key"]])
            with self.assertRaises(ValueError):
                campaign.submit({"family": "not-a-model"})


@unittest.skipUnless(HAVE_NUMERIC, "numpy and pandas are optional outside the Qlib environment")
class PanelTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from autoresearch.domains.alpha.planted import create_planted_dataset

        cls.tmp = tempfile.TemporaryDirectory()
        cls.dataset = create_planted_dataset(Path(cls.tmp.name) / "planted", symbols=60, days=600, seed=8)
        dates = (cls.dataset / "qlib_data/calendars/day.txt").read_text().split()
        cls.settings = {"task": "panel", "dataset": str(cls.dataset), "universe": "TOP60", "cost_bps": 5,
                        "warmup": 80, "truncation": 0.1, "development_start": dates[90], "development_end": dates[319],
                        "validation_start": dates[320], "validation_end": dates[499], "holdout_start": dates[500],
                        "features": ["ret_5", "vol_20", "log_volume_shock", "max_ret_20"]}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_rank_normalize_matches_average_ranks(self):
        from autoresearch.domains.ml.tasks import rank_normalize

        values = np.array([[3.0, 1.0, 2.0, 2.0, np.nan], [1.0, 1.0, 1.0, 5.0, 4.0]])
        members = np.array([[True, True, True, True, True], [True, True, True, True, False]])
        out = rank_normalize(values, members)
        np.testing.assert_allclose(out[0, :4], np.array([3, 0, 1.5, 1.5]) / 3 - 0.5)
        self.assertTrue(np.isnan(out[0, 4]) and np.isnan(out[1, 4]))
        np.testing.assert_allclose(out[1, :4], np.array([1, 1, 1, 3]) / 3 - 0.5)

    def test_training_rows_are_purged_and_the_sealed_panel_ends_at_validation(self):
        from autoresearch.domains.ml.tasks import PanelTask

        with tempfile.TemporaryDirectory() as tmp:
            task = PanelTask(self.settings, runtime=Path(tmp))
            data = task._matrices()
            self.assertEqual(str(data["dates"][-1]), self.settings["validation_end"])
            days = task.rows("development") // data["y"].shape[1]
            last_row = int(np.searchsorted(data["dates"], np.datetime64(self.settings["development_end"])))
            self.assertLessEqual(int(days.max()) + 2, last_row)          # target realized within development
            inner_train = task.rows("development", inner="train") // data["y"].shape[1]
            inner_val = task.rows("development", inner="validate") // data["y"].shape[1]
            # A training target spans closes t+1..t+2; an inner-validation target spans u+1..u+2.
            # The windows never overlap: t + 2 <= u.
            self.assertLessEqual(int(inner_train.max()) + 2, int(inner_val.min()))

    def test_search_process_holds_no_session_after_the_validation_end(self):
        """Memory proof for the panel task: after a screen and a validation every array reachable from
        the domain ends at the validation end; the holdout path adds nothing to the caches."""
        from autoresearch.domains.ml.domain import MLDomain
        try:
            from tests.test_lab_alpha import assert_sessions_bounded
        except ImportError:                      # the tests folder itself on sys.path (pytest, discover -s tests)
            from test_lab_alpha import assert_sessions_bounded

        dates = (self.dataset / "qlib_data/calendars/day.txt").read_text().split()
        sealed_sessions = dates.index(self.settings["validation_end"]) + 1
        recipe = {"family": "ridge", "params": {}}
        with tempfile.TemporaryDirectory() as tmp:
            domain = MLDomain({**self.settings, "search": {"families": ["ridge"], "screen_rows": 8000}}, Path(tmp),
                              policy={"min_sharpe": 0.5, "min_dsr": 0.5, "max_turnover": 1.5})
            self.assertEqual(domain.screen([recipe])[0]["status"], "ok")
            self.assertEqual(domain.validate(recipe)["status"], "ok")
            self.assertEqual(set(domain.task._cache), {"sealed", "matrices"})
            self.assertEqual(str(domain.task._matrices()["dates"][-1]), self.settings["validation_end"])
            self.assertEqual(len(domain.task._workspace().dates), sealed_sessions)
            assert_sessions_bounded(self, domain, self.settings["validation_end"], sealed_sessions, symbols=60)
            # Negative control: the walk does reach what the task caches.
            domain.task._cache["probe"] = np.asarray(dates[-1:], dtype="datetime64[D]")
            with self.assertRaises(AssertionError):
                assert_sessions_bounded(self, domain, self.settings["validation_end"], sealed_sessions, symbols=60)
            del domain.task._cache["probe"]
            final = domain.evaluate_holdout(recipe)
            self.assertEqual(final["status"], "ok", final.get("error"))
            self.assertGreater(final["metrics"]["holdout"]["days"], 50)
            self.assertEqual(set(domain.task._cache), {"sealed", "matrices"})
            assert_sessions_bounded(self, domain, self.settings["validation_end"], sealed_sessions, symbols=60)
            with self.assertRaises(ValueError):
                domain.task.simulate(np.zeros((1, 60)), "holdout")

    def test_holdout_path_opens_the_sessions_it_needs_and_leaves_the_numbers_unchanged(self):
        from autoresearch.domains.alpha.holdout import holdout_window
        from autoresearch.domains.ml.models import make_model
        from autoresearch.domains.ml.tasks import PanelTask
        from qlib_yahoo.alphalab.simulate import Workspace

        dates = (self.dataset / "qlib_data/calendars/day.txt").read_text().split()
        # The window the panel task asks for: warm-up before development_start through the last session.
        self.assertEqual(holdout_window(self.dataset, holdout_start=self.settings["holdout_start"], warmup=80,
                                        history_from=self.settings["development_start"]), (dates[10], dates[-1]))
        with tempfile.TemporaryDirectory() as tmp:
            task = PanelTask(self.settings, runtime=Path(tmp))
            X, y, _ = task.arrays(task.rows("train_final"), limit=20_000, rng=np.random.default_rng(0))
            model = make_model("ridge", {}, task="regression", seed=0).fit(X, y)
            bounded = task.evaluate_holdout(model)["summary"]
            self.assertEqual(set(task._cache), {"sealed", "matrices"})
            # The same model scored the old way, on a workspace of the whole snapshot: identical numbers.
            whole = Workspace(self.dataset)
            data = task._features(whole)
            start, end = task.backend.holdout_bounds()
            reference = task._simulate_on(whole, data, task.predict_matrix(model, data), start, end, 1.0)["summary"]
            self.assertEqual(bounded["start"], self.settings["holdout_start"])
            for key in ("days", "sharpe", "returns", "turnover", "drawdown", "fitness", "start", "end"):
                self.assertEqual(bounded[key], reference[key], key)

    def test_injected_predictions_are_scored_exactly_like_an_expression(self):
        from autoresearch.domains.alpha.evaluator import SealedEvaluator
        from autoresearch.domains.ml.tasks import PanelTask
        from qlib_yahoo.alphalab.engine import evaluate

        with tempfile.TemporaryDirectory() as tmp:
            task = PanelTask(self.settings, runtime=Path(tmp))
            expression = "-rank(ts_std_dev(returns, 20))"
            workspace = task._workspace()
            data = task._matrices()
            members = workspace.universe(task.backend.settings.universe)
            context = workspace.context(data["offset"], len(workspace.dates) - 1, members)
            predictions = np.asarray(evaluate(expression, context), dtype=np.float64)
            injected = task.simulate(predictions, "validation")["summary"]
            bounds = {k: self.settings[k] for k in ("development_start", "development_end",
                                                     "validation_start", "validation_end")}
            direct = SealedEvaluator(self.dataset, task.backend.settings, bounds).simulate(expression, "validation")["summary"]
            for key in ("sharpe", "returns", "turnover"):
                self.assertAlmostEqual(injected[key], direct[key], places=5)
            self.assertNotIn(task.PRED_FIELD, workspace.fields)

    def test_panel_campaign_promotes_and_finalizes_through_alpha_lab(self):
        from autoresearch.lab.campaign import LabCampaign

        with tempfile.TemporaryDirectory() as tmp:
            config = {"name": "panel-test", "domain": "ml", "seed": 5, "research_root": str(Path(tmp) / "records"),
                      "runtime_root": str(Path(tmp) / "runtime"), "lineage_root": str(Path(tmp) / "lineage"),
                      "settings": {**self.settings, "search": {"families": ["ridge"], "screen_rows": 8000}},
                      "policy": {"min_sharpe": 0.5, "min_dsr": 0.5, "max_turnover": 1.5},
                      "limits": {"max_rounds": 2, "proposals_per_round": 3, "validations_per_round": 1}}
            campaign = LabCampaign(config)
            campaign.tick()
            campaign.tick()
            status = campaign.status()
            self.assertGreaterEqual(status["validation_trials"], 1)
            reviews = {r["experiment_id"]: r for r in campaign.store.list("reviews")}
            for experiment in campaign.store.list("experiments"):
                self.assertIn("ic_mean", experiment["result"]["metrics"]["validation"])
                robustness = experiment["result"]["robustness"]
                probe = robustness["perturbation"]
                self.assertEqual(probe["kind"], "recipe")
                if probe["status"] == "ok":
                    self.assertEqual([p["factor"] for p in robustness["perturbations"]], [0.75, 1.33])
                else:   # gate-first: a candidate the review rejects anyway is not refit for the probe
                    self.assertEqual((probe["status"], robustness["perturbations"]), ("skipped", []))
                    self.assertTrue(probe["reason"])
                    self.assertNotEqual(reviews[experiment["id"]]["decision"], "promote")
            if campaign.state["accepted"]:
                final = campaign.finalize()[0]
                self.assertGreater(final["holdout"]["days"], 50)

    def test_gate_failing_panel_candidates_are_not_refit_for_the_perturbation_probe(self):
        from unittest import mock

        from autoresearch.domains.ml import domain as ml_domain

        class Counting:
            """Wraps a model and counts its fits; everything else is delegated."""

            def __init__(self, inner, fits):
                self._inner, self._fits = inner, fits

            def fit(self, *args, **kwargs):
                self._fits.append(1)
                self._inner.fit(*args, **kwargs)
                return self

            def __getattr__(self, name):
                return getattr(self._inner, name)

        fits: list[int] = []
        real = ml_domain.make_model
        counting = mock.patch.object(ml_domain, "make_model", side_effect=lambda *a, **k: Counting(real(*a, **k), fits))
        recipe = {"family": "ridge", "params": {}}
        with tempfile.TemporaryDirectory() as tmp:
            settings = {**self.settings, "search": {"families": ["ridge"]}}
            strict = ml_domain.MLDomain(settings, Path(tmp) / "strict", policy={"min_sharpe": 1e6})
            with counting:
                result = strict.validate(recipe)
            self.assertEqual(len(fits), 1)                      # the base fit only: no refit for a candidate the gates reject
            probe = result["robustness"]["perturbation"]
            self.assertEqual((probe["kind"], probe["status"], result["robustness"]["perturbations"]),
                             ("recipe", "skipped", []))
            self.assertIn("Sharpe", probe["reason"])
            context = {"n_validation_trials": 1, "campaign_validation_trials": 1, "lineage_validation_trials": 0,
                       "validation_scores": [], "policy": {"min_sharpe": 1e6}}
            self.assertEqual(strict.review(recipe, result, context)["decision"], "reject")
            # A candidate the review would otherwise pass is still refit on both sides of its recipe.
            fits.clear()
            lenient = ml_domain.MLDomain(settings, Path(tmp) / "lenient",
                                         policy={"min_sharpe": 0.01, "min_dsr": 0.0, "max_turnover": 10.0, "max_drawdown": 10.0,
                                                 "min_cost_stress_retention": 0.0, "min_half_sharpe": -1e9,
                                                 "min_perturbation_retention": 0.0})
            with counting:
                passing = lenient.validate(recipe)
            probe = passing["robustness"]["perturbation"]
            if probe["status"] == "skipped":
                self.skipTest("the ridge recipe fails even lenient gates on this panel: " + probe["reason"])
            self.assertEqual(len(fits), 3)
            self.assertEqual(probe, {"kind": "recipe", "status": "ok"})
            self.assertEqual([p["factor"] for p in passing["robustness"]["perturbations"]], [0.75, 1.33])


if __name__ == "__main__":
    unittest.main()
