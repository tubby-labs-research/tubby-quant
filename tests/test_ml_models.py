"""Tests for the pure-numpy ML model families and metrics (autoresearch.domains.ml)."""

from __future__ import annotations

import importlib.util
import json
import math
import unittest

if importlib.util.find_spec("numpy") is None:  # the dependency-light CI job has no numpy
    raise unittest.SkipTest("numpy is optional outside the research environments")

import numpy as np

from autoresearch.domains.ml import metrics
from autoresearch.domains.ml.models import FAMILIES, Model, make_model, model_from_json, validate_params


def friedman1(n: int, seed: int, noise: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Friedman #1: five informative uniform features plus five pure-noise features."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, 10))
    y = (
        10.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20.0 * (X[:, 2] - 0.5) ** 2
        + 10.0 * X[:, 3]
        + 5.0 * X[:, 4]
        + noise * rng.standard_normal(n)
    )
    return X, y


def binary_task(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """y = 1[x1 + x2^2 + noise > 1] with four extra noise features."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 6))
    y = (X[:, 0] + X[:, 1] ** 2 + 0.5 * rng.standard_normal(n) > 1.0).astype(np.float64)
    return X, y


def with_nan(X: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    out = X.copy()
    out[np.random.default_rng(seed).random(X.shape) < fraction] = np.nan
    return out


# Parameters giving good held-out quality on n_train = 1,500.
QUALITY = {
    "regression": {
        "constant": {},
        "ridge": {},
        "gbdt": {"n_estimators": 300, "learning_rate": 0.1, "min_samples_leaf": 10},
        "random_forest": {},
        "mlp": {"epochs": 150, "learning_rate": 3e-3, "batch_size": 64, "early_stopping_rounds": 0},
    },
    "classification": {
        "constant": {},
        "logistic": {},
        "gbdt": {"n_estimators": 200, "learning_rate": 0.1},
        "random_forest": {},
        "mlp": {"epochs": 60, "learning_rate": 3e-3, "batch_size": 64, "early_stopping_rounds": 0},
    },
}
MIN_R2 = {"ridge": 0.6, "gbdt": 0.85, "random_forest": 0.7, "mlp": 0.8}
MIN_AUC = {"logistic": 0.7, "gbdt": 0.85, "random_forest": 0.85, "mlp": 0.85}

# Small, fast configurations exercising every stochastic code path.
FAST = {
    "constant": {},
    "ridge": {"alpha": 0.5},
    "logistic": {"C": 2.0},
    "gbdt": {"n_estimators": 25, "learning_rate": 0.2, "subsample": 0.7, "colsample": 0.6, "min_samples_leaf": 5},
    "random_forest": {"n_estimators": 15, "max_depth": 5, "max_features": 0.5},
    "mlp": {"hidden": "32", "epochs": 8, "batch_size": 64, "learning_rate": 3e-3},
}
STOCHASTIC = ("gbdt", "random_forest", "mlp")


def family_tasks():
    for family, spec in FAMILIES.items():
        for task in spec["tasks"]:
            yield family, task


class DataMixin:
    @classmethod
    def setUpClass(cls) -> None:
        X, y = friedman1(2000, seed=0)
        cls.reg = (X[:1500], y[:1500], X[1500:], y[1500:])
        X, y = binary_task(2000, seed=1)
        cls.clf = (X[:1500], y[:1500], X[1500:], y[1500:])

    def data(self, task: str):
        return self.reg if task == "regression" else self.clf


class TestQuality(DataMixin, unittest.TestCase):
    def test_regression_families_beat_constant(self) -> None:
        Xtr, ytr, Xte, yte = self.reg
        base = metrics.r2(yte, make_model("constant", task="regression").fit(Xtr, ytr).predict(Xte))
        self.assertLess(abs(base), 0.05)
        for family, threshold in MIN_R2.items():
            with self.subTest(family=family):
                model = make_model(family, QUALITY["regression"][family], task="regression", seed=0)
                score = metrics.r2(yte, model.fit(Xtr, ytr).predict(Xte))
                self.assertGreater(score, threshold)
                self.assertGreater(score, base + 0.5)

    def test_classification_families_beat_constant(self) -> None:
        Xtr, ytr, Xte, yte = self.clf
        base_pred = make_model("constant", task="classification").fit(Xtr, ytr).predict(Xte)
        self.assertEqual(metrics.auc(yte, base_pred), 0.5)
        base_loss = metrics.log_loss(yte, base_pred)
        for family, threshold in MIN_AUC.items():
            with self.subTest(family=family):
                model = make_model(family, QUALITY["classification"][family], task="classification", seed=0)
                pred = model.fit(Xtr, ytr).predict(Xte)
                self.assertTrue(np.all((pred >= 0.0) & (pred <= 1.0)))
                self.assertGreater(metrics.auc(yte, pred), threshold)
                self.assertLess(metrics.log_loss(yte, pred), base_loss)


class TestDeterminism(DataMixin, unittest.TestCase):
    def test_same_seed_identical_predictions(self) -> None:
        for family, task in family_tasks():
            with self.subTest(family=family, task=task):
                Xtr, ytr, Xte, _ = self.data(task)
                a = make_model(family, FAST[family], task=task, seed=7).fit(Xtr, ytr).predict(Xte)
                b = make_model(family, FAST[family], task=task, seed=7).fit(Xtr, ytr).predict(Xte)
                self.assertTrue(np.array_equal(a, b))
                self.assertTrue(np.isfinite(a).all())

    def test_refit_reproduces(self) -> None:
        Xtr, ytr, Xte, _ = self.reg
        model = make_model("gbdt", FAST["gbdt"], task="regression", seed=3)
        first = model.fit(Xtr, ytr).predict(Xte)
        self.assertTrue(np.array_equal(first, model.fit(Xtr, ytr).predict(Xte)))

    def test_different_seed_changes_stochastic_families(self) -> None:
        for family in STOCHASTIC:
            for task in FAMILIES[family]["tasks"]:
                with self.subTest(family=family, task=task):
                    Xtr, ytr, Xte, _ = self.data(task)
                    a = make_model(family, FAST[family], task=task, seed=0).fit(Xtr, ytr).predict(Xte)
                    b = make_model(family, FAST[family], task=task, seed=1).fit(Xtr, ytr).predict(Xte)
                    self.assertFalse(np.array_equal(a, b))

    def test_seed_irrelevant_for_deterministic_families(self) -> None:
        for family, task in (("ridge", "regression"), ("logistic", "classification")):
            Xtr, ytr, Xte, _ = self.data(task)
            a = make_model(family, task=task, seed=0).fit(Xtr, ytr).predict(Xte)
            b = make_model(family, task=task, seed=99).fit(Xtr, ytr).predict(Xte)
            self.assertTrue(np.array_equal(a, b))


class TestSerialization(DataMixin, unittest.TestCase):
    def test_json_round_trip_identical_predictions(self) -> None:
        for family, task in family_tasks():
            with self.subTest(family=family, task=task):
                Xtr, ytr, Xte, yte = self.data(task)
                Xtr, Xte = with_nan(Xtr, 0.1, 5), with_nan(Xte, 0.1, 6)
                model = make_model(family, FAST[family], task=task, seed=11)
                model.fit(Xtr, ytr, eval_set=(Xte, yte))
                payload = model.to_json()
                self.assertEqual(set(payload), {"family", "task", "params", "seed", "state"})
                text = json.dumps(payload, allow_nan=False)
                clone = model_from_json(json.loads(text))
                self.assertIsInstance(clone, type(model))
                self.assertEqual(clone.params, model.params)
                self.assertTrue(np.array_equal(clone.predict(Xte), model.predict(Xte)))
                self.assertEqual(clone.describe(), model.describe())
                self.assertEqual(json.dumps(clone.to_json(), allow_nan=False), text)

    def test_unfitted_round_trip(self) -> None:
        model = make_model("gbdt", {"max_depth": 2}, task="classification", seed=4)
        self.assertIsNone(model.to_json()["state"])
        clone = model_from_json(json.loads(json.dumps(model.to_json())))
        self.assertEqual(clone.params, model.params)
        self.assertEqual(clone.seed, 4)
        with self.assertRaises(RuntimeError):
            clone.predict(np.zeros((2, 3)))

    def test_corrupted_json_rejected(self) -> None:
        Xtr, ytr, _, _ = self.reg
        payload = make_model("gbdt", {"n_estimators": 3}, task="regression").fit(Xtr, ytr).to_json()
        bad = json.loads(json.dumps(payload))
        bad["state"]["trees"][0]["left"][0] = 0  # child pointing at its own parent: a cycle
        with self.assertRaises(ValueError):
            model_from_json(bad)
        bad = json.loads(json.dumps(payload))
        del bad["state"]["trees"]
        with self.assertRaises(ValueError):
            model_from_json(bad)
        bad = json.loads(json.dumps(payload))
        bad["family"] = "ridge"
        with self.assertRaises(ValueError):
            model_from_json(bad)
        with self.assertRaises(ValueError):
            model_from_json({"family": "gbdt"})


class TestMissingValues(DataMixin, unittest.TestCase):
    def test_nan_inputs_give_finite_predictions(self) -> None:
        for family, task in family_tasks():
            with self.subTest(family=family, task=task):
                Xtr, ytr, Xte, yte = self.data(task)
                Xtr, Xte = with_nan(Xtr, 0.1, 1), with_nan(Xte, 0.1, 2)
                Xtr[:, -1] = np.nan  # an entirely missing feature
                model = make_model(family, FAST[family], task=task, seed=0).fit(Xtr, ytr)
                pred = model.predict(Xte)
                self.assertEqual(pred.shape, (Xte.shape[0],))
                self.assertTrue(np.isfinite(pred).all())
                self.assertTrue(np.isfinite(model.predict(np.full((3, Xte.shape[1]), np.nan))).all())

    def test_gbdt_learns_informative_missingness(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.uniform(size=(2000, 1))
        missing = rng.random(2000) < 0.3
        y = np.where(missing, 5.0, x[:, 0])
        x[missing, 0] = np.nan
        model = make_model("gbdt", {"n_estimators": 100, "learning_rate": 0.3, "max_depth": 2}, task="regression")
        pred = model.fit(x, y).predict(np.array([[np.nan], [0.1], [0.9]]))
        self.assertAlmostEqual(pred[0], 5.0, delta=0.05)
        self.assertAlmostEqual(pred[1], 0.1, delta=0.1)
        self.assertAlmostEqual(pred[2], 0.9, delta=0.1)


class TestValidation(unittest.TestCase):
    def test_defaults_filled(self) -> None:
        for family, spec in FAMILIES.items():
            for task in spec["tasks"]:
                params = validate_params(family, None, task)
                self.assertEqual(params, {name: ps["default"] for name, ps in spec["params"].items()})
                self.assertEqual(validate_params(family, params, task), params)
        self.assertEqual(validate_params("gbdt", {"max_depth": 5.0}, "regression")["max_depth"], 5)
        self.assertIsInstance(validate_params("gbdt", {"max_depth": np.int64(5)}, "regression")["max_depth"], int)
        self.assertEqual(validate_params("ridge", {"alpha": 2}, "regression")["alpha"], 2.0)

    def test_parameter_space_is_well_formed(self) -> None:
        for family, spec in FAMILIES.items():
            self.assertTrue(set(spec["tasks"]) <= {"regression", "classification"})
            self.assertIsInstance(spec["description"], str)
            for name, ps in spec["params"].items():
                with self.subTest(family=family, param=name):
                    if ps["type"] in ("int", "float"):
                        self.assertLessEqual(ps["low"], ps["default"])
                        self.assertLessEqual(ps["default"], ps["high"])
                        self.assertIsInstance(ps["log"], bool)
                    elif ps["type"] == "choice":
                        self.assertIn(ps["default"], ps["choices"])
                    else:
                        self.assertEqual(ps["type"], "bool")

    def test_rejections_name_the_key(self) -> None:
        cases = [
            ("gbdt", {"n_trees": 10}, "regression", "n_trees"),
            ("ridge", {"alpha": "big"}, "regression", "alpha"),
            ("gbdt", {"max_depth": 2.5}, "regression", "max_depth"),
            ("gbdt", {"max_depth": True}, "regression", "max_depth"),
            ("ridge", {"fit_intercept": 1}, "regression", "fit_intercept"),
            ("mlp", {"hidden": 64}, "regression", "hidden"),
            ("mlp", {"activation": "sigmoid"}, "regression", "activation"),
            ("ridge", {"alpha": 0.0}, "regression", "alpha"),
            ("gbdt", {"max_depth": 9}, "regression", "max_depth"),
            ("gbdt", {"subsample": 0.05}, "regression", "subsample"),
            ("logistic", {"C": float("nan")}, "classification", "C"),
            ("ridge", {}, "classification", "ridge"),
            ("logistic", {}, "regression", "logistic"),
            ("gbdt", {}, "ranking", "ranking"),
            ("xgboost", {}, "regression", "xgboost"),
        ]
        for family, params, task, needle in cases:
            with self.subTest(family=family, params=params, task=task):
                with self.assertRaisesRegex(ValueError, needle):
                    validate_params(family, params, task)
                with self.assertRaises(ValueError):
                    make_model(family, params, task=task)
        with self.assertRaises(ValueError):
            validate_params("gbdt", [("max_depth", 2)], "regression")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            make_model("gbdt", task="regression", seed=-1)

    def test_fit_and_predict_input_checks(self) -> None:
        X, y = friedman1(60, seed=2)
        yb = (y > np.median(y)).astype(float)
        model = make_model("gbdt", {"n_estimators": 2, "min_samples_leaf": 5}, task="regression")
        self.assertIsInstance(model, Model)
        with self.assertRaises(RuntimeError):
            model.predict(X)
        bad_fits = [
            (X[:, 0], y, {}),
            (X, y[:-1], {}),
            (X[:0], y[:0], {}),
            (np.where(X == X[0, 0], np.inf, X), y, {}),
            (X, np.where(y == y[0], np.nan, y), {}),
            (X, y, {"sample_weight": -np.ones(60)}),
            (X, y, {"sample_weight": np.zeros(60)}),
            (X, y, {"sample_weight": np.ones(59)}),
            (X, y, {"eval_set": (X[:, :3], y)}),
            (X, y, {"eval_set": (X,)}),
        ]
        for i, (Xa, ya, kwargs) in enumerate(bad_fits):
            with self.subTest(case=i):
                with self.assertRaises(ValueError):
                    model.fit(Xa, ya, **kwargs)
        model.fit(X, y)
        with self.assertRaises(ValueError):
            model.predict(X[:, :5])
        with self.assertRaises(ValueError):
            model.predict(X[0])
        self.assertEqual(model.predict(X[:0]).shape, (0,))
        clf = make_model("logistic", task="classification")
        with self.assertRaises(ValueError):
            clf.fit(X, yb * 2)  # labels must be 0/1
        with self.assertRaises(ValueError):
            clf.fit(X, np.ones(60))  # one class only
        with self.assertRaises(ValueError):
            clf.fit(X, yb, sample_weight=yb)  # only class 1 keeps positive weight


class TestEarlyStopping(unittest.TestCase):
    def test_gbdt_early_stopping_truncates_to_best_iteration(self) -> None:
        rng = np.random.default_rng(4)
        X = rng.standard_normal((2000, 5))
        y = X[:, 0] + 3.0 * rng.standard_normal(2000)  # weak signal, heavy noise
        params = {
            "n_estimators": 500,
            "learning_rate": 0.3,
            "max_depth": 4,
            "min_samples_leaf": 5,
            "early_stopping_rounds": 10,
        }
        model = make_model("gbdt", params, task="regression").fit(X[:1500], y[:1500], eval_set=(X[1500:], y[1500:]))
        info = model.describe()
        self.assertTrue(info["stopped_early"])
        self.assertLess(info["n_trees"], 500)
        self.assertEqual(info["n_trees"], info["best_iteration"])
        self.assertEqual(info["n_trees_built"], info["best_iteration"] + 10)
        val_mse = float(np.mean((model.predict(X[1500:]) - y[1500:]) ** 2))
        self.assertAlmostEqual(val_mse, info["best_eval_loss"], places=9)
        # Without an eval_set the rounds setting is inert and every tree is kept.
        full = make_model("gbdt", params, task="regression").fit(X[:1500], y[:1500]).describe()
        self.assertEqual(full["n_trees"], 500)
        self.assertIsNone(full["best_iteration"])

    def test_gbdt_eval_set_without_patience_keeps_all_trees(self) -> None:
        X, y = binary_task(800, seed=3)
        params = {"n_estimators": 40, "early_stopping_rounds": 0}
        model = make_model("gbdt", params, task="classification")
        info = model.fit(X[:600], y[:600], eval_set=(X[600:], y[600:])).describe()
        self.assertEqual(info["n_trees"], 40)
        self.assertGreaterEqual(info["best_iteration"], 1)
        self.assertFalse(info["stopped_early"])

    def test_mlp_early_stopping_restores_best_weights(self) -> None:
        X, y = friedman1(1200, seed=5, noise=3.0)
        params = {
            "hidden": "128,64",
            "learning_rate": 1e-2,
            "batch_size": 32,
            "epochs": 300,
            "early_stopping_rounds": 5,
        }
        model = make_model("mlp", params, task="regression", seed=0).fit(X[:300], y[:300], eval_set=(X[300:], y[300:]))
        info = model.describe()
        self.assertTrue(info["stopped_early"])
        self.assertEqual(info["epochs_run"], info["best_epoch"] + 5)
        val_mse = float(np.mean((model.predict(X[300:]) - y[300:]) ** 2))
        self.assertAlmostEqual(val_mse, info["best_eval_loss"], places=9)


class TestSampleWeight(DataMixin, unittest.TestCase):
    def test_zero_weight_equals_dropping_rows(self) -> None:
        for family, task in family_tasks():
            with self.subTest(family=family, task=task):
                Xtr, ytr, Xte, _ = self.data(task)
                w = np.zeros(Xtr.shape[0])
                w[::2] = 1.0
                weighted = make_model(family, FAST[family], task=task, seed=2).fit(Xtr, ytr, sample_weight=w)
                subset = make_model(family, FAST[family], task=task, seed=2).fit(Xtr[::2], ytr[::2])
                plain = make_model(family, FAST[family], task=task, seed=2).fit(Xtr, ytr)
                self.assertTrue(np.array_equal(weighted.predict(Xte), subset.predict(Xte)))
                self.assertFalse(np.array_equal(weighted.predict(Xte), plain.predict(Xte)))

    def test_positive_weights(self) -> None:
        Xtr, ytr, Xte, _ = self.reg
        w = np.random.default_rng(0).uniform(0.5, 3.0, Xtr.shape[0])
        const = make_model("constant", task="regression").fit(Xtr, ytr, sample_weight=w)
        self.assertAlmostEqual(const.predict(Xte[:1])[0], float(np.average(ytr, weights=w)), places=10)
        for family in ("ridge", "gbdt", "mlp"):
            with self.subTest(family=family):
                def fit_predict(weights):
                    model = make_model(family, FAST[family], task="regression")
                    return model.fit(Xtr, ytr, sample_weight=weights).predict(Xte)

                a, b, c = fit_predict(w), fit_predict(7.0 * w), fit_predict(None)
                np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)  # only relative weights matter
                self.assertFalse(np.allclose(a, c))


class TestDescribe(DataMixin, unittest.TestCase):
    def test_describe_reports_model_size(self) -> None:
        Xtr, ytr, _, _ = self.reg
        self.assertFalse(make_model("ridge", task="regression").describe()["fitted"])
        ridge = make_model("ridge", task="regression").fit(Xtr, ytr).describe()
        self.assertEqual((ridge["n_features"], ridge["n_coefficients"]), (10, 10))
        gbdt = make_model("gbdt", {"n_estimators": 12, "max_depth": 2}, task="regression").fit(Xtr, ytr).describe()
        self.assertEqual(gbdt["n_trees"], 12)
        self.assertLessEqual(gbdt["max_depth_reached"], 2)
        self.assertLessEqual(gbdt["n_leaves"], 12 * 4)
        importance = np.array(gbdt["feature_importance"])
        self.assertAlmostEqual(float(importance.sum()), 1.0)
        self.assertGreater(importance[:5].sum(), importance[5:].sum())  # informative features dominate
        mlp = make_model("mlp", {"hidden": "64,32", "epochs": 2}, task="regression").fit(Xtr, ytr).describe()
        self.assertEqual(mlp["layers"], [10, 64, 32, 1])
        self.assertEqual(mlp["n_parameters"], 10 * 64 + 64 + 64 * 32 + 32 + 32 + 1)
        json.dumps([ridge, gbdt, mlp], allow_nan=False)


class TestMetrics(unittest.TestCase):
    def test_regression_metrics(self) -> None:
        y, p = [1.0, 2.0, 3.0], [1.0, 2.0, 5.0]
        self.assertAlmostEqual(metrics.rmse(y, p), math.sqrt(4.0 / 3.0))
        self.assertAlmostEqual(metrics.mae(y, p), 2.0 / 3.0)
        self.assertEqual(metrics.r2(y, y), 1.0)
        self.assertAlmostEqual(metrics.r2(y, [2.0, 2.0, 2.0]), 0.0)
        self.assertAlmostEqual(metrics.r2(y, p), 1.0 - 4.0 / 2.0)
        self.assertEqual(metrics.r2([4.0, 4.0], [4.0, 4.0]), 1.0)
        self.assertEqual(metrics.r2([4.0, 4.0], [4.0, 5.0]), 0.0)

    def test_log_loss_clipping(self) -> None:
        self.assertAlmostEqual(metrics.log_loss([1.0, 0.0], [1.0, 0.0]), 1e-12, delta=1e-15)
        self.assertAlmostEqual(metrics.log_loss([1.0], [0.0]), -math.log(1e-12))
        self.assertAlmostEqual(metrics.log_loss([1.0], [0.0], eps=1e-3), -math.log(1e-3))
        self.assertAlmostEqual(metrics.log_loss([1.0, 0.0], [0.8, 0.4]), -(math.log(0.8) + math.log(0.6)) / 2)
        with self.assertRaises(ValueError):
            metrics.log_loss([1.0], [1.5])
        with self.assertRaises(ValueError):
            metrics.log_loss([2.0], [0.5])

    def test_accuracy(self) -> None:
        y, p = [0.0, 1.0, 1.0, 0.0], [0.2, 0.5, 0.9, 0.7]
        self.assertEqual(metrics.accuracy(y, p), 0.75)
        self.assertEqual(metrics.accuracy(y, p, threshold=0.6), 0.5)

    def test_auc_with_ties(self) -> None:
        self.assertEqual(metrics.auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.75)
        # Pairs (pos, neg): (0.5, 0.5) tie -> 0.5, the other three are ordered -> 3.5 / 4.
        self.assertEqual(metrics.auc([0, 1, 0, 1], [0.5, 0.5, 0.2, 0.9]), 0.875)
        self.assertEqual(metrics.auc([0, 1, 0, 1], [0.3, 0.3, 0.3, 0.3]), 0.5)
        rng = np.random.default_rng(0)
        y = (rng.random(300) < 0.4).astype(float)
        p = np.round(rng.random(300) + 0.3 * y, 1)  # heavy ties
        pos, neg = p[y == 1], p[y == 0]
        wins = np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])
        brute = wins / (pos.size * neg.size)
        self.assertAlmostEqual(metrics.auc(y, p), float(brute), places=12)
        with self.assertRaises(ValueError):
            metrics.auc([1.0, 1.0], [0.2, 0.3])

    def test_correlations(self) -> None:
        self.assertAlmostEqual(metrics.pearson([1, 2, 3], [2, 4, 7]), 5.0 / math.sqrt(2.0 * 114.0 / 9.0))
        self.assertAlmostEqual(metrics.pearson([1, 2, 3], [3, 2, 1]), -1.0)
        # Ranks with ties: y -> [1, 2.5, 2.5, 4], p -> [1, 2, 3, 4].
        self.assertAlmostEqual(metrics.spearman([1, 2, 2, 3], [1, 2, 3, 4]), 4.5 / math.sqrt(4.5 * 5.0))
        self.assertAlmostEqual(metrics.spearman([1, 2, 3, 4], [1, 10, 100, 1000]), 1.0)
        self.assertTrue(math.isnan(metrics.pearson([0.1, 0.1, 0.1], [1, 2, 3])))
        self.assertTrue(math.isnan(metrics.spearman([1, 2, 3], [5, 5, 5])))

    def test_input_validation(self) -> None:
        for name in metrics.METRICS:
            with self.subTest(metric=name):
                with self.assertRaises(ValueError):
                    metrics.score(name, [0.0, 1.0], [0.5])
                with self.assertRaises(ValueError):
                    metrics.score(name, [], [])
                with self.assertRaises(ValueError):
                    metrics.score(name, [0.0, float("nan")], [0.5, 0.5])
                with self.assertRaises(ValueError):
                    metrics.score(name, [[0.0, 1.0]], [[0.5, 0.5]])

    def test_score_dispatch(self) -> None:
        y, p = [0.0, 1.0, 1.0, 0.0], [0.1, 0.8, 0.6, 0.3]
        for name, meta in metrics.METRICS.items():
            with self.subTest(metric=name):
                self.assertIn(meta["task"], ("regression", "classification", "any"))
                self.assertIsInstance(meta["greater_is_better"], bool)
                self.assertEqual(metrics.score(name, y, p), getattr(metrics, name)(y, p))
        with self.assertRaises(ValueError):
            metrics.score("f1", y, p)


if __name__ == "__main__":
    unittest.main()
