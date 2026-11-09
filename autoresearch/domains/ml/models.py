"""Pure-numpy model families for the machine-learning research domain.

Every family implements the :class:`Model` contract:

* ``fit(X, y)`` takes float64 ``X`` of shape ``(n, k)`` (NaN marks a missing
  value; +/-inf is rejected) and ``y`` of shape ``(n,)``.  For classification
  ``y`` must hold only the labels 0 and 1, and both must occur.
* ``predict(X)`` returns regression values or, for classification, the
  probability of class 1.  Calling it before ``fit`` raises ``RuntimeError``.
* Fitting is deterministic: all randomness comes from
  ``np.random.default_rng(seed)`` created afresh by each ``fit`` call.
* ``sample_weight`` must be finite, non-negative and not all zero.  Rows with
  zero weight are dropped before fitting (a zero weight is exactly equivalent
  to removing the row) and the remaining weights are rescaled to mean 1, so
  only relative weights matter and regularisation keeps its meaning.
  Imputation medians, standardisation statistics and histogram bin edges are
  learned from the retained rows.
* ``eval_set=(X_val, y_val)`` drives early stopping in ``gbdt`` and ``mlp``
  (and is validated but otherwise unused by the other families).
* ``to_json()`` returns plain JSON data containing finite numbers only, and
  :func:`model_from_json` rebuilds a model with bit-identical predictions.

Invalid shapes, labels, weights or parameters raise ``ValueError``; a fit that
diverges to non-finite parameters raises ``FloatingPointError``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = [
    "FAMILIES",
    "TASKS",
    "ConstantModel",
    "GBDTModel",
    "LogisticModel",
    "MLPModel",
    "Model",
    "RandomForestModel",
    "RidgeModel",
    "make_model",
    "model_from_json",
    "validate_params",
]

TASKS: tuple[str, ...] = ("regression", "classification")
_STATE_FORMAT = 1
# Above this many elements per histogram array a tree level is built in node chunks
# (without the parent-minus-sibling subtraction trick) to bound memory.
_HIST_BUDGET = 1 << 21
# Element budget for each temporary array of the vectorised split search.
_SPLIT_BUDGET = 1 << 19


# --------------------------------------------------------------------------- parameter spaces


def _p_int(low: int, high: int, default: int) -> dict:
    return {"type": "int", "low": low, "high": high, "log": False, "default": default}


def _p_float(low: float, high: float, default: float, *, log: bool = False) -> dict:
    return {"type": "float", "low": low, "high": high, "log": log, "default": default}


def _p_bool(default: bool) -> dict:
    return {"type": "bool", "default": default}


def _p_choice(choices: list[str], default: str) -> dict:
    return {"type": "choice", "choices": list(choices), "default": default}


_HIDDEN_LAYERS: dict[str, tuple[int, ...]] = {"32": (32,), "64": (64,), "64,32": (64, 32), "128,64": (128, 64)}

FAMILIES: dict[str, dict] = {
    "constant": {
        "tasks": TASKS,
        "params": {},
        "description": "Baseline: the weighted training mean (regression) or class-1 prior (classification).",
    },
    "ridge": {
        "tasks": ("regression",),
        "params": {
            "alpha": _p_float(1e-6, 1e4, 1.0, log=True),
            "fit_intercept": _p_bool(True),
        },
        "description": (
            "L2-penalised least squares (penalty alpha*||w||^2, intercept unpenalised) on median-imputed, "
            "standardised features; closed-form solve."
        ),
    },
    "logistic": {
        "tasks": ("classification",),
        "params": {
            "C": _p_float(1e-4, 1e4, 1.0, log=True),
            "max_iter": _p_int(10, 500, 100),
        },
        "description": (
            "L2-penalised logistic regression (penalty ||w||^2/(2C), intercept unpenalised) on median-imputed, "
            "standardised features; damped Newton/IRLS."
        ),
    },
    "gbdt": {
        "tasks": TASKS,
        "params": {
            "n_estimators": _p_int(1, 2000, 200),
            "learning_rate": _p_float(0.001, 1.0, 0.05, log=True),
            "max_depth": _p_int(1, 8, 3),
            "min_samples_leaf": _p_int(1, 10000, 20),
            "l2": _p_float(1e-6, 100.0, 1.0, log=True),
            "subsample": _p_float(0.1, 1.0, 1.0),
            "colsample": _p_float(0.1, 1.0, 1.0),
            "max_bins": _p_int(8, 255, 64),
            "early_stopping_rounds": _p_int(0, 500, 0),
        },
        "description": (
            "Histogram gradient-boosted trees (squared or binary logistic loss, Newton leaves, learned "
            "missing-value direction, row/feature subsampling per tree, optional early stopping on eval_set)."
        ),
    },
    "random_forest": {
        "tasks": TASKS,
        "params": {
            "n_estimators": _p_int(1, 1000, 100),
            "max_depth": _p_int(1, 16, 6),
            "min_samples_leaf": _p_int(1, 10000, 5),
            "max_features": _p_float(0.1, 1.0, 0.5),
            "max_bins": _p_int(8, 255, 64),
        },
        "description": (
            "Bagged histogram regression trees (bootstrap rows, per-split feature subsampling); classification "
            "averages leaf class-1 frequencies."
        ),
    },
    "mlp": {
        "tasks": TASKS,
        "params": {
            "hidden": _p_choice(list(_HIDDEN_LAYERS), "64,32"),
            "activation": _p_choice(["relu", "tanh"], "relu"),
            "learning_rate": _p_float(1e-4, 1e-1, 1e-3, log=True),
            "weight_decay": _p_float(1e-8, 1e-1, 1e-5, log=True),
            "batch_size": _p_int(16, 4096, 256),
            "epochs": _p_int(1, 500, 50),
            "early_stopping_rounds": _p_int(0, 100, 10),
        },
        "description": (
            "Fully connected network trained with AdamW mini-batches on median-imputed, standardised features "
            "(squared loss on standardised y, or binary cross-entropy on logits)."
        ),
    },
}


def _family_spec(family: Any) -> dict:
    if not isinstance(family, str) or family not in FAMILIES:
        raise ValueError(f"unknown model family {family!r}; expected one of {sorted(FAMILIES)}")
    return FAMILIES[family]


def _check_task(family: str, task: Any) -> str:
    if task not in TASKS:
        raise ValueError(f"task must be one of {list(TASKS)}, got {task!r}")
    if task not in FAMILIES[family]["tasks"]:
        supported = list(FAMILIES[family]["tasks"])
        raise ValueError(f"family {family!r} does not support task {task!r} (supports {supported})")
    return task


def _coerce_param(family: str, name: str, spec: dict, value: Any) -> Any:
    kind = spec["type"]
    where = f"{family}: parameter {name!r}"
    if kind == "bool":
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        raise ValueError(f"{where} must be a bool, got {type(value).__name__}")
    if kind == "choice":
        if isinstance(value, str) and value in spec["choices"]:
            return value
        raise ValueError(f"{where} must be one of {spec['choices']}, got {value!r}")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{where} must be a number, got bool")
    if kind == "int":
        if isinstance(value, (int, np.integer)):
            number: int | float = int(value)
        elif isinstance(value, (float, np.floating)) and math.isfinite(value) and float(value).is_integer():
            number = int(value)
        else:
            raise ValueError(f"{where} must be an int, got {value!r}")
    else:
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{where} must be a float, got {type(value).__name__}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{where} must be finite, got {value!r}")
    if not spec["low"] <= number <= spec["high"]:
        raise ValueError(f"{where} must be in [{spec['low']}, {spec['high']}], got {value!r}")
    return number


def validate_params(family: str, params: dict | None, task: str) -> dict:
    """Return a complete, normalised copy of ``params`` for ``family`` on ``task``.

    Missing keys take their defaults; values are converted to plain Python
    types.  Unknown keys, wrong types, out-of-range values, an unknown family
    and an unsupported task raise ``ValueError`` naming the offender.
    """
    spec = _family_spec(family)
    _check_task(family, task)
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError(f"{family}: params must be a dict, got {type(params).__name__}")
    space = spec["params"]
    for key in params:
        if key not in space:
            raise ValueError(f"{family}: unknown parameter {key!r}; allowed: {sorted(space)}")
    return {
        name: _coerce_param(family, name, ps, params[name]) if name in params else ps["default"]
        for name, ps in space.items()
    }


# --------------------------------------------------------------------------- input validation


def _as_matrix(X: Any, name: str, n_features: int | None = None, *, allow_empty: bool = False) -> np.ndarray:
    try:
        # C-contiguous float64 so results never depend on the caller's memory layout.
        A = np.ascontiguousarray(X, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array: {exc}") from None
    if A.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_samples, n_features), got shape {A.shape}")
    if A.shape[1] == 0:
        raise ValueError(f"{name} has no feature columns")
    if A.shape[0] == 0 and not allow_empty:
        raise ValueError(f"{name} has no rows")
    if n_features is not None and A.shape[1] != n_features:
        raise ValueError(f"{name} has {A.shape[1]} features, expected {n_features}")
    if np.isinf(A).any():
        raise ValueError(f"{name} contains +/-inf (NaN is allowed and means missing)")
    return A


def _as_target(y: Any, n: int, task: str, name: str) -> np.ndarray:
    try:
        a = np.ascontiguousarray(y, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array: {exc}") from None
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    if a.shape[0] != n:
        raise ValueError(f"{name} has {a.shape[0]} values but X has {n} rows")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains NaN or inf")
    if task == "classification" and not np.all((a == 0.0) | (a == 1.0)):
        raise ValueError(f"{name} must contain only 0/1 labels for classification")
    return a


def _as_weights(sample_weight: Any, n: int) -> np.ndarray:
    if sample_weight is None:
        return np.ones(n)
    try:
        w = np.ascontiguousarray(sample_weight, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sample_weight must be a numeric array: {exc}") from None
    if w.ndim != 1 or w.shape[0] != n:
        raise ValueError(f"sample_weight must have shape ({n},), got {w.shape}")
    if not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("sample_weight must be finite and non-negative")
    if not w.sum() > 0:
        raise ValueError("sample_weight must not be all zero")
    return w


def _check_seed(seed: Any) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    return int(seed)


# --------------------------------------------------------------------------- shared numerics


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic function without overflow for large |z|."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _logistic_loss(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-row binary cross-entropy of logits ``z`` against 0/1 labels ``y``."""
    return np.logaddexp(0.0, z) - y * z


def _solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def _weighted_mean(v: np.ndarray, w: np.ndarray) -> float:
    return float(w @ v) / float(w.sum())


def _impute(X: np.ndarray, medians: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(X), medians, X)


def _fit_scaler(X: np.ndarray, w: np.ndarray, *, center: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Learn column medians (for imputation), weighted means and weighted standard deviations.

    All-NaN columns impute to 0; constant columns get scale 1 (and, when
    centring, map exactly to 0).  With ``center=False`` the means are zero.
    """
    k = X.shape[1]
    medians = np.zeros(k)
    for j in range(k):
        col = X[:, j]
        observed = col[~np.isnan(col)]
        if observed.size:
            medians[j] = float(np.median(observed))
    Xi = _impute(X, medians)
    lo = Xi.min(axis=0)
    constant = lo == Xi.max(axis=0)
    total = float(w.sum())
    means = np.where(constant, lo, (w @ Xi) / total)
    with np.errstate(over="ignore", invalid="ignore"):
        scales = np.sqrt((w @ (Xi - means) ** 2) / total)
    scales = np.where(constant | ~np.isfinite(scales) | ~(scales > 0), 1.0, scales)
    if not center:
        means = np.zeros(k)
    return medians, means, scales


def _standardize(X: np.ndarray, medians: np.ndarray, means: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return (_impute(X, medians) - means) / scales


def _finite_list(values: Any, what: str) -> list:
    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(f"non-finite values in {what}")
    return arr.tolist()


def _finite_float(value: Any, what: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite value in {what}")
    return number


def _optional_float(value: float | None) -> float | None:
    """``value`` as a float, or None when absent or non-finite (keeps describe()/to_json() finite)."""
    return None if value is None or not math.isfinite(value) else float(value)


def _state_array(
    state: dict, key: str, *, dtype: Any = np.float64, ndim: int = 1, size: int | None = None
) -> np.ndarray:
    arr = np.asarray(state[key], dtype=dtype)
    if arr.ndim != ndim or (size is not None and arr.shape[0] != size):
        raise ValueError(f"state field {key!r} has shape {arr.shape}")
    if arr.dtype.kind == "f" and not np.isfinite(arr).all():
        raise ValueError(f"state field {key!r} contains non-finite values")
    return arr


def _check_finite_arrays(family: str, *arrays: np.ndarray) -> None:
    for arr in arrays:
        if not np.isfinite(arr).all():
            raise FloatingPointError(f"{family} fit produced non-finite parameters; try stronger regularisation")


# --------------------------------------------------------------------------- base class


class Model:
    """Base class of every family; construct instances with :func:`make_model`."""

    family: str = ""

    def __init__(self, params: dict | None = None, *, task: str, seed: int = 0) -> None:
        self.params: dict = validate_params(self.family, params, task)
        self.task: str = task
        self.seed: int = _check_seed(seed)
        self.n_features: int | None = None
        self._fitted = False
        self._info: dict = {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(task={self.task!r}, params={self.params!r}, seed={self.seed})"

    @property
    def fitted(self) -> bool:
        """Whether ``fit`` (or :func:`model_from_json` with a state) has completed."""
        return self._fitted

    def fit(self, X: Any, y: Any, *, sample_weight: Any = None, eval_set: tuple | None = None) -> Model:
        """Fit on ``(X, y)``; returns ``self``.  See the module docstring for semantics."""
        X = _as_matrix(X, "X")
        n, k = X.shape
        y = _as_target(y, n, self.task, "y")
        w = _as_weights(sample_weight, n)
        val = None
        if eval_set is not None:
            if not isinstance(eval_set, (tuple, list)) or len(eval_set) != 2:
                raise ValueError("eval_set must be a (X_val, y_val) pair")
            X_val = _as_matrix(eval_set[0], "eval_set X", k)
            y_val = _as_target(eval_set[1], X_val.shape[0], self.task, "eval_set y")
            val = (X_val, y_val)
        keep = w > 0
        if not keep.all():
            X, y, w = X[keep], y[keep], w[keep]
        w = np.ones(w.shape[0]) if np.all(w == w[0]) else w / w.mean()
        if self.task == "classification" and np.all(y == y[0]):
            raise ValueError("classification needs both classes (0 and 1) among positively weighted rows of y")
        self._fitted = False
        self._info = {}
        self._fit(X, y, w, val)
        self.n_features = k
        self._fitted = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predicted values (regression) or class-1 probabilities (classification), shape ``(n,)``."""
        if not self._fitted:
            raise RuntimeError(f"{self.family} model is not fitted; call fit() first")
        X = _as_matrix(X, "X", self.n_features, allow_empty=True)
        return np.asarray(self._predict(X), dtype=np.float64)

    def describe(self) -> dict:
        """JSON-friendly summary of the fitted model (sizes, iterations, early-stopping outcome)."""
        out: dict = {"family": self.family, "task": self.task, "fitted": self._fitted}
        if self._fitted:
            out["n_features"] = self.n_features
            out.update(self._info)
            out.update(self._describe())
        return out

    def to_json(self) -> dict:
        """``{"family", "task", "params", "seed", "state"}``; ``state`` is None before fitting."""
        state = None
        if self._fitted:
            state = {"format": _STATE_FORMAT, "n_features": self.n_features, "info": dict(self._info)}
            state.update(self._get_state())
        return {
            "family": self.family,
            "task": self.task,
            "params": dict(self.params),
            "seed": self.seed,
            "state": state,
        }

    # hooks implemented by families
    def _fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, val: tuple | None) -> None:
        raise NotImplementedError

    def _predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _describe(self) -> dict:
        return {}

    def _get_state(self) -> dict:
        raise NotImplementedError

    def _set_state(self, state: dict) -> None:
        raise NotImplementedError


# --------------------------------------------------------------------------- constant


class ConstantModel(Model):
    """Predicts the weighted mean of ``y`` (for classification: the class-1 prior)."""

    family = "constant"

    def _fit(self, X, y, w, val):
        self._value = _weighted_mean(y, w)

    def _predict(self, X):
        return np.full(X.shape[0], self._value)

    def _describe(self):
        return {"value": self._value}

    def _get_state(self):
        return {"value": _finite_float(self._value, "constant value")}

    def _set_state(self, state):
        self._value = _finite_float(state["value"], "constant value")


# --------------------------------------------------------------------------- linear models


class _LinearBase(Model):
    """Shared state of ridge/logistic: imputation + standardisation + coefficients."""

    def _scores(self, X: np.ndarray) -> np.ndarray:
        return _standardize(X, self._medians, self._means, self._scales) @ self._coef + self._intercept

    def _get_state(self):
        return {
            "medians": _finite_list(self._medians, "medians"),
            "means": _finite_list(self._means, "means"),
            "scales": _finite_list(self._scales, "scales"),
            "coef": _finite_list(self._coef, "coefficients"),
            "intercept": _finite_float(self._intercept, "intercept"),
        }

    def _set_state(self, state):
        k = int(state["n_features"])
        self._medians = _state_array(state, "medians", size=k)
        self._means = _state_array(state, "means", size=k)
        self._scales = _state_array(state, "scales", size=k)
        self._coef = _state_array(state, "coef", size=k)
        self._intercept = _finite_float(state["intercept"], "intercept")

    def _describe(self):
        return {"n_coefficients": int(self._coef.shape[0]), "intercept": self._intercept}


class RidgeModel(_LinearBase):
    """Minimises ``sum w (y - Zb - c)^2 + alpha ||b||^2`` on standardised ``Z``.

    With ``fit_intercept=False`` features are scaled but not centred and ``c = 0``.
    """

    family = "ridge"

    def _fit(self, X, y, w, val):
        center = self.params["fit_intercept"]
        self._medians, self._means, self._scales = _fit_scaler(X, w, center=center)
        Z = _standardize(X, self._medians, self._means, self._scales)
        intercept = _weighted_mean(y, w) if center else 0.0
        Zw = Z * w[:, None]
        A = Zw.T @ Z
        A[np.diag_indices_from(A)] += self.params["alpha"]
        self._coef = _solve(A, Zw.T @ (y - intercept))
        self._intercept = float(intercept)
        _check_finite_arrays(self.family, self._coef)

    def _predict(self, X):
        return self._scores(X)


class LogisticModel(_LinearBase):
    """Minimises ``sum w logloss(y, sigmoid(Zb + c)) + ||b||^2 / (2C)`` by Newton steps with backtracking.

    Stops when the Newton decrement or the step falls below 1e-10 (relative);
    ``describe()`` reports ``n_iter`` and ``converged``.
    """

    family = "logistic"
    _TOL = 1e-10

    def _fit(self, X, y, w, val):
        self._medians, self._means, self._scales = _fit_scaler(X, w)
        Z = _standardize(X, self._medians, self._means, self._scales)
        n, k = Z.shape
        A = np.empty((n, k + 1))
        A[:, :k] = Z
        A[:, k] = 1.0
        penalty = np.full(k + 1, 1.0 / self.params["C"])
        penalty[k] = 0.0  # intercept is not penalised
        prior = _weighted_mean(y, w)
        beta = np.zeros(k + 1)
        beta[k] = math.log(prior / (1.0 - prior))

        def objective(b: np.ndarray) -> tuple[float, np.ndarray]:
            z = A @ b
            return float(w @ _logistic_loss(z, y)) + 0.5 * float(penalty @ (b * b)), z

        obj, z = objective(beta)
        converged = False
        n_iter = 0
        for n_iter in range(1, self.params["max_iter"] + 1):
            p = _sigmoid(z)
            grad = A.T @ (w * (p - y)) + penalty * beta
            curvature = w * p * (1.0 - p)
            H = (A * curvature[:, None]).T @ A
            H[np.diag_indices_from(H)] += penalty + 1e-12 * max(1.0, float(np.trace(H)) / (k + 1))
            step = _solve(H, grad)
            decrement = float(grad @ step)  # Newton decrement squared
            if not math.isfinite(decrement):
                break
            if 0.5 * decrement <= self._TOL * max(1.0, abs(obj)):
                converged = True
                break
            t = 1.0
            while True:  # backtracking (Armijo) line search keeps Newton monotone on separable data
                candidate = beta - t * step
                new_obj, new_z = objective(candidate)
                if new_obj <= obj - 1e-4 * t * decrement or t < 1e-10:
                    break
                t *= 0.5
            if not new_obj < obj:
                converged = 0.5 * decrement <= 1e-6 * max(1.0, abs(obj))
                break
            moved = float(np.max(np.abs(candidate - beta)))
            beta, obj, z = candidate, new_obj, new_z
            if moved <= self._TOL * (1.0 + float(np.max(np.abs(beta)))):
                converged = True
                break
        self._coef = beta[:k].copy()
        self._intercept = float(beta[k])
        _check_finite_arrays(self.family, self._coef, np.array([self._intercept]))
        self._info = {"n_iter": n_iter, "converged": bool(converged)}

    def _predict(self, X):
        return _sigmoid(self._scores(X))


# --------------------------------------------------------------------------- histogram trees


def _bin_edges(X: np.ndarray, max_bins: int) -> list[np.ndarray]:
    """Per-feature split candidates learned from non-NaN values.

    A feature with at most ``max_bins`` distinct values gets the midpoints
    between consecutive values (exact splits); otherwise the unique interior
    quantiles ``np.quantile(v, i / max_bins)``.  Either way there are at most
    ``max_bins`` non-missing bins.
    """
    probs = np.linspace(0.0, 1.0, max_bins + 1)[1:-1]
    edges = []
    for j in range(X.shape[1]):
        col = X[:, j]
        values = np.sort(col[~np.isnan(col)])
        if values.size == 0:
            edges.append(np.empty(0))
            continue
        distinct = values[np.concatenate(([True], values[1:] != values[:-1]))]
        if distinct.size <= max_bins:
            cut = np.unique(distinct[:-1] * 0.5 + distinct[1:] * 0.5)
        else:
            cut = np.unique(np.quantile(values, probs))
        edges.append(np.ascontiguousarray(cut, dtype=np.float64))
    return edges


def _bin(X: np.ndarray, edges: list[np.ndarray], n_bins: int) -> np.ndarray:
    """Bin ``X`` into a ``(k, n)`` uint8 matrix: bin ``b`` holds ``edges[b-1] < x <= edges[b]``; NaN -> ``n_bins``."""
    n, k = X.shape
    out = np.empty((k, n), dtype=np.uint8)
    for j in range(k):
        col = X[:, j]
        b = np.searchsorted(edges[j], col, side="left")
        b[np.isnan(col)] = n_bins
        out[j] = b
    return out


def _tree_depth(left: np.ndarray, right: np.ndarray, leaf: np.ndarray) -> int:
    depth = 0
    frontier = np.zeros(1, dtype=np.intp)
    while True:
        internal = frontier[~leaf[frontier]]
        if internal.size == 0:
            return depth
        frontier = np.concatenate((left[internal], right[internal]))
        depth += 1


class _Tree:
    """Flat binary tree over binned features.

    Node 0 is the root.  An internal node sends a row left when its bin is
    ``<= threshold`` (non-missing) or, for the missing bin, when
    ``missing_left``; leaves have ``feature == left == right == -1``.
    """

    __slots__ = ("feature", "threshold", "missing_left", "left", "right", "value", "depth", "_feat", "_left", "_right")

    def __init__(self, feature, threshold, missing_left, left, right, value) -> None:
        self.feature = np.asarray(feature, dtype=np.intp)
        self.threshold = np.asarray(threshold, dtype=np.intp)
        self.missing_left = np.asarray(missing_left, dtype=bool)
        self.left = np.asarray(left, dtype=np.intp)
        self.right = np.asarray(right, dtype=np.intp)
        self.value = np.asarray(value, dtype=np.float64)
        leaf = self.feature < 0
        ids = np.arange(self.feature.shape[0], dtype=np.intp)
        # Leaves loop to themselves so traversal needs no masking.
        self._feat = np.where(leaf, 0, self.feature)
        self._left = np.where(leaf, ids, self.left)
        self._right = np.where(leaf, ids, self.right)
        self.depth = _tree_depth(self.left, self.right, leaf)

    @property
    def n_leaves(self) -> int:
        return int(np.count_nonzero(self.feature < 0))

    def leaves(self, flat_bins: np.ndarray, stride: int, rows: np.ndarray, n_bins: int) -> np.ndarray:
        """Leaf index of each row, where row ``r`` / feature ``j`` is ``flat_bins[j * stride + r]``."""
        node = np.zeros(rows.shape[0], dtype=np.intp)
        for _ in range(self.depth):
            b = flat_bins[self._feat[node] * stride + rows]
            go_left = np.where(b == n_bins, self.missing_left[node], b <= self.threshold[node])
            node = np.where(go_left, self._left[node], self._right[node])
        return node

    def to_json(self) -> dict:
        return {
            "feature": self.feature.tolist(),
            "threshold": self.threshold.tolist(),
            "missing_left": self.missing_left.tolist(),
            "left": self.left.tolist(),
            "right": self.right.tolist(),
            "value": _finite_list(self.value, "tree leaf values"),
        }

    @classmethod
    def from_json(cls, data: dict, n_features: int, n_bins: int) -> _Tree:
        keys = ("feature", "threshold", "left", "right", "value")
        arrays = {key: _state_array(data, key, dtype=np.float64) for key in keys}
        size = arrays["value"].shape[0]
        missing_left = np.asarray(data["missing_left"])
        if size == 0 or missing_left.shape != (size,) or missing_left.dtype != bool:
            raise ValueError("tree arrays must be non-empty and of equal length (missing_left boolean)")
        ints = {}
        for key in ("feature", "threshold", "left", "right"):
            arr = arrays[key]
            if arr.shape != (size,) or not np.all(arr == np.round(arr)):
                raise ValueError(f"tree field {key!r} must be {size} integers")
            ints[key] = arr.astype(np.intp)
        feature, left, right = ints["feature"], ints["left"], ints["right"]
        ids = np.arange(size)
        internal = feature >= 0
        if (
            np.any(feature >= n_features)
            or np.any(feature < -1)
            or np.any((ints["threshold"] < 0) | (ints["threshold"] >= n_bins))
            or np.any(internal & ((left <= ids) | (right <= ids) | (left >= size) | (right >= size)))
            or np.any(~internal & ((left != -1) | (right != -1)))
        ):
            raise ValueError("tree structure is inconsistent")
        return cls(feature, ints["threshold"], missing_left, left, right, arrays["value"])


def _histograms(Xs, sel, loc, m, n_bins, g, h) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-node, per-feature histograms ``(sum g, sum h, count)`` of shape ``(m, k, n_bins + 1)``.

    ``Xs`` is the ``(k, n_s)`` binned in-sample matrix; ``sel`` selects rows
    (None = all, then ``m`` must be 1) and ``loc`` is each selected row's node
    position in ``[0, m)``.  ``h=None`` means unit hessians (sum h == count).
    """
    width = n_bins + 1
    size = m * width
    k = Xs.shape[0]
    G = np.empty((m, k, width))
    C = np.empty((m, k, width), dtype=np.int64)
    H = None if h is None else np.empty((m, k, width))
    if sel is None:
        gs, hs, base = g, h, None
    else:
        gs = g[sel]
        hs = None if h is None else h[sel]
        base = loc * width
    for j in range(k):
        idx = Xs[j] if sel is None else base + Xs[j].take(sel)
        G[:, j] = np.bincount(idx, weights=gs, minlength=size).reshape(m, width)
        C[:, j] = np.bincount(idx, minlength=size).reshape(m, width)
        if H is not None:
            H[:, j] = np.bincount(idx, weights=hs, minlength=size).reshape(m, width)
    if H is None:
        H = C.astype(np.float64)
    return G, H, C


def _split_scores(GL, HL, CL, TG, TH, TC, lam, min_leaf):
    GR = TG - GL
    HR = TH - HL
    CR = TC - CL
    dl = HL + lam
    dr = HR + lam
    ok = (CL >= min_leaf) & (CR >= min_leaf) & (dl > 0) & (dr > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = GL * GL / dl + GR * GR / dr
    return np.where(ok, s, -np.inf)


def _best_splits(G, H, C, tg, th, tc, n_bins, lam, min_leaf, allowed):
    """Best split per node by XGBoost gain, trying missing-left and missing-right for every threshold.

    Returns ``(do_split, feature_pos, threshold, missing_left, G_left, H_left, C_left, gain)``.
    A node without missing values (for the chosen feature) sends future
    missing values to its larger child.
    """
    m, k, _ = G.shape
    nb = n_bins
    do = np.zeros(m, dtype=bool)
    feat = np.zeros(m, dtype=np.intp)
    thr = np.zeros(m, dtype=np.intp)
    mleft = np.zeros(m, dtype=bool)
    gl = np.zeros(m)
    hl = np.zeros(m)
    cl = np.zeros(m, dtype=np.int64)
    gain = np.zeros(m)
    step = max(1, _SPLIT_BUDGET // (k * nb))
    for a in range(0, m, step):
        b = min(m, a + step)
        mm = b - a
        cG = np.cumsum(G[a:b, :, :nb], axis=2)
        cH = np.cumsum(H[a:b, :, :nb], axis=2)
        cC = np.cumsum(C[a:b, :, :nb], axis=2)
        mG = G[a:b, :, nb:]
        mH = H[a:b, :, nb:]
        mC = C[a:b, :, nb:]
        TG = tg[a:b, None, None]
        TH = th[a:b, None, None]
        TC = tc[a:b, None, None]
        s_right = _split_scores(cG, cH, cC, TG, TH, TC, lam, min_leaf)
        s_left = _split_scores(cG + mG, cH + mH, cC + mC, TG, TH, TC, lam, min_leaf)
        if allowed is not None:
            banned = ~allowed[a:b, :, None]
            s_right = np.where(banned, -np.inf, s_right)
            s_left = np.where(banned, -np.inf, s_left)
        both = np.concatenate((s_right.reshape(mm, -1), s_left.reshape(mm, -1)), axis=1)
        best = np.argmax(both, axis=1)
        r = np.arange(mm)
        child_score = both[r, best]
        denom = th[a:b] + lam
        with np.errstate(divide="ignore", invalid="ignore"):
            parent_score = np.where(denom > 0, tg[a:b] ** 2 / denom, 0.0)
        improvement = child_score - parent_score
        ok = np.isfinite(child_score) & (improvement > 1e-10 * (np.abs(child_score) + np.abs(parent_score)))
        option, rest = np.divmod(best, k * nb)
        f, t = np.divmod(rest, nb)
        go_missing = option == 1
        n_missing = mC[r, f, 0]
        c_left = cC[r, f, t] + np.where(go_missing, n_missing, 0)
        do[a:b] = ok
        feat[a:b] = f
        thr[a:b] = t
        mleft[a:b] = np.where(n_missing > 0, go_missing, 2 * c_left >= tc[a:b])
        gl[a:b] = cG[r, f, t] + np.where(go_missing, mG[r, f, 0], 0.0)
        hl[a:b] = cH[r, f, t] + np.where(go_missing, mH[r, f, 0], 0.0)
        cl[a:b] = c_left
        gain[a:b] = np.where(ok, 0.5 * improvement, 0.0)
    return do, feat, thr, mleft, gl, hl, cl, gain


def _grow_tree(XbT, rows, g, h, feats, *, n_bins, max_depth, min_leaf, lam, mtry, rng):
    """Grow one depth-wise tree on the binned ``(k, n)`` matrix ``XbT``.

    ``rows`` are the in-sample row indices (None = all rows) and ``g``/``h``
    their gradients/hessians aligned with ``rows`` (``h=None``: unit
    hessians).  ``feats`` are the candidate features; when ``0 < mtry <
    len(feats)`` each node draws ``mtry`` of them.  Leaf values are the Newton
    step ``-G / (H + lam)``.  Returns ``(tree, leaf_of_in_sample_row,
    gain_per_feature)``.
    """
    k_all, n_total = XbT.shape
    n_s = n_total if rows is None else rows.shape[0]
    k_sel = feats.shape[0]
    width = n_bins + 1
    if rows is None:
        Xs = XbT if k_sel == k_all else XbT[feats]
    else:
        Xs = XbT[np.ix_(feats, rows)]
    Xs = np.ascontiguousarray(Xs)
    Xs_flat = Xs.reshape(-1)

    cap = min(2 ** (max_depth + 1) - 1, 2 * (n_s // min_leaf) + 1)
    feat_pos = np.full(cap, -1, dtype=np.intp)
    threshold = np.zeros(cap, dtype=np.intp)
    missing_left = np.zeros(cap, dtype=bool)
    left = np.full(cap, -1, dtype=np.intp)
    right = np.full(cap, -1, dtype=np.intp)
    node_g = np.zeros(cap)
    node_h = np.zeros(cap)
    node_c = np.zeros(cap, dtype=np.int64)
    node_g[0] = g.sum()
    node_h[0] = float(n_s) if h is None else h.sum()
    node_c[0] = n_s
    n_nodes = 1
    node_of_row = np.zeros(n_s, dtype=np.intp)
    importance = np.zeros(k_all)
    use_mtry = 0 < mtry < k_sel

    def rows_of(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        local = np.full(n_nodes, -1, dtype=np.intp)
        local[nodes] = np.arange(nodes.shape[0])
        pos = local[node_of_row]
        sel = np.flatnonzero(pos >= 0)
        return sel, pos[sel]

    level = np.zeros(1, dtype=np.intp)  # nodes created at the current depth, as (left, right) pairs below the root
    prev = None  # histograms of the nodes split at the previous depth, in split order
    for depth in range(max_depth):
        splittable = node_c[level] >= 2 * min_leaf
        if not splittable.any():
            break
        hist: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        if depth == 0:
            nodes = level
            hist = _histograms(Xs, None, None, 1, n_bins, g, h)
        else:
            need = np.flatnonzero(splittable[0::2] | splittable[1::2])
            if prev is not None and 2 * need.shape[0] * k_sel * width <= _HIST_BUDGET:
                # Histogram the smaller sibling directly, derive the larger as parent minus smaller.
                lefts, rights = level[0::2][need], level[1::2][need]
                small_is_left = node_c[lefts] <= node_c[rights]
                small = np.where(small_is_left, lefts, rights)
                large = np.where(small_is_left, rights, lefts)
                Gs, Hs, Cs = _histograms(Xs, *rows_of(small), small.shape[0], n_bins, g, h)
                PG, PH, PC = prev
                cand = np.concatenate((small, large))
                keep = node_c[cand] >= 2 * min_leaf
                nodes = cand[keep]
                hist = (
                    np.concatenate((Gs, PG[need] - Gs))[keep],
                    np.concatenate((Hs, PH[need] - Hs))[keep],
                    np.concatenate((Cs, PC[need] - Cs))[keep],
                )
            else:
                nodes = level[splittable]
        allowed = None
        if use_mtry:
            keys = rng.random((nodes.shape[0], k_sel))
            allowed = np.argsort(np.argsort(keys, axis=1), axis=1) < mtry
        totals = (node_g[nodes], node_h[nodes], node_c[nodes])
        if hist is not None:
            res = _best_splits(*hist, *totals, n_bins, lam, min_leaf, allowed)
        else:  # wide level: build and search in bounded node chunks
            chunk = max(1, _HIST_BUDGET // (k_sel * width))
            parts = []
            for a in range(0, nodes.shape[0], chunk):
                sub = nodes[a : a + chunk]
                sub_hist = _histograms(Xs, *rows_of(sub), sub.shape[0], n_bins, g, h)
                parts.append(
                    _best_splits(
                        *sub_hist,
                        totals[0][a : a + chunk],
                        totals[1][a : a + chunk],
                        totals[2][a : a + chunk],
                        n_bins,
                        lam,
                        min_leaf,
                        None if allowed is None else allowed[a : a + chunk],
                    )
                )
            res = tuple(np.concatenate(cols) for cols in zip(*parts))
        do, fpos, thr, mleft, gl, hl, cl, gain = res
        chosen = np.flatnonzero(do)
        if chosen.shape[0] == 0:
            break
        parents = nodes[chosen]
        s = parents.shape[0]
        first = n_nodes
        L = first + 2 * np.arange(s, dtype=np.intp)
        R = L + 1
        feat_pos[parents] = fpos[chosen]
        threshold[parents] = thr[chosen]
        missing_left[parents] = mleft[chosen]
        left[parents] = L
        right[parents] = R
        node_g[L], node_h[L], node_c[L] = gl[chosen], hl[chosen], cl[chosen]
        node_g[R] = node_g[parents] - node_g[L]
        node_h[R] = node_h[parents] - node_h[L]
        node_c[R] = node_c[parents] - node_c[L]
        importance += np.bincount(feats[fpos[chosen]], weights=gain[chosen], minlength=k_all)
        n_nodes += 2 * s
        # Route the rows of the split nodes to their children.
        is_parent = np.zeros(n_nodes, dtype=bool)
        is_parent[parents] = True
        moving = np.flatnonzero(is_parent[node_of_row])
        nd = node_of_row[moving]
        b = Xs_flat[feat_pos[nd] * n_s + moving]
        go_left = np.where(b == n_bins, missing_left[nd], b <= threshold[nd])
        node_of_row[moving] = np.where(go_left, left[nd], right[nd])
        prev = None if hist is None else (hist[0][chosen], hist[1][chosen], hist[2][chosen])
        level = np.arange(first, n_nodes, dtype=np.intp)

    size = n_nodes
    fp = feat_pos[:size]
    is_leaf = fp < 0
    feature = np.where(is_leaf, -1, feats[np.maximum(fp, 0)])
    denom = node_h[:size] + lam
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.where(is_leaf & (denom > 0), -node_g[:size] / denom, 0.0)
    tree = _Tree(feature, threshold[:size], missing_left[:size], left[:size], right[:size], value)
    return tree, node_of_row, importance


class _TreeEnsemble(Model):
    """Shared binning, serialisation and description of gbdt / random_forest."""

    def _prepare_bins(self, X: np.ndarray) -> np.ndarray:
        self._n_bins = self.params["max_bins"]
        self._edges = _bin_edges(X, self._n_bins)
        return _bin(X, self._edges, self._n_bins)

    def _raw_sum(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        total = np.zeros(n)
        if n == 0:
            return total
        flat = _bin(X, self._edges, self._n_bins).reshape(-1)
        rows = np.arange(n, dtype=np.intp)
        for tree in self._trees:
            total += tree.value[tree.leaves(flat, n, rows, self._n_bins)]
        return total

    def _importance(self, gains: np.ndarray) -> list[float]:
        total = float(gains.sum())
        return (gains / total).tolist() if total > 0 else [0.0] * gains.shape[0]

    def _describe(self):
        return {
            "n_trees": len(self._trees),
            "n_leaves": int(sum(t.n_leaves for t in self._trees)),
            "max_depth_reached": int(max((t.depth for t in self._trees), default=0)),
            "feature_importance": list(self._feature_importance),
        }

    def _get_state(self):
        return {
            "bin_edges": [_finite_list(e, "bin edges") for e in self._edges],
            "trees": [t.to_json() for t in self._trees],
            "feature_importance": _finite_list(self._feature_importance, "feature importance"),
        }

    def _set_state(self, state):
        k = int(state["n_features"])
        self._n_bins = self.params["max_bins"]
        edges = state["bin_edges"]
        if not isinstance(edges, list) or len(edges) != k:
            raise ValueError("bin_edges must hold one list per feature")
        self._edges = []
        for e in edges:
            arr = np.asarray(e, dtype=np.float64).reshape(-1)
            if not np.isfinite(arr).all() or arr.shape[0] > self._n_bins - 1 or np.any(np.diff(arr) <= 0):
                raise ValueError("bin edges must be finite, strictly increasing and fewer than max_bins")
            self._edges.append(arr)
        self._trees = [_Tree.from_json(t, k, self._n_bins) for t in state["trees"]]
        if not self._trees:
            raise ValueError("a tree ensemble needs at least one tree")
        self._feature_importance = _state_array(state, "feature_importance", size=k).tolist()


class GBDTModel(_TreeEnsemble):
    """Gradient boosting of depth-wise histogram trees from a mean / log-odds start.

    Leaves hold ``-learning_rate * G / (H + l2)``.  With an ``eval_set`` the
    eval loss (MSE, or mean log-loss for classification) is tracked per tree;
    with ``early_stopping_rounds > 0`` training stops after that many trees
    without improvement and only the trees up to the best iteration are kept.
    """

    family = "gbdt"

    def _fit(self, X, y, w, val):
        p = self.params
        rng = np.random.default_rng(self.seed)
        n, k = X.shape
        XbT = self._prepare_bins(X)
        flat = XbT.reshape(-1)
        nb = self._n_bins
        clf = self.task == "classification"
        unit_w = bool(np.all(w == 1.0))
        mean_y = _weighted_mean(y, w)
        init = math.log(mean_y / (1.0 - mean_y)) if clf else mean_y
        lr = p["learning_rate"]
        n_sub = n if p["subsample"] >= 1.0 else max(1, int(p["subsample"] * n + 1e-9))
        k_sub = k if p["colsample"] >= 1.0 else max(1, int(p["colsample"] * k + 1e-9))
        all_feats = np.arange(k, dtype=np.intp)
        F = np.full(n, init)
        if val is not None:
            X_val, y_val = val
            val_flat = _bin(X_val, self._edges, nb).reshape(-1)
            n_val = X_val.shape[0]
            val_rows = np.arange(n_val, dtype=np.intp)
            F_val = np.full(n_val, init)
        patience = p["early_stopping_rounds"] if val is not None else 0
        best_loss, best_iter, stopped = math.inf, 0, False
        trees: list[_Tree] = []
        gains: list[np.ndarray] = []
        for it in range(p["n_estimators"]):
            if clf:
                prob = _sigmoid(F)
                g = prob - y
                h = prob * (1.0 - prob)
                if not unit_w:
                    g *= w
                    h *= w
            else:
                g = F - y
                h = None
                if not unit_w:
                    g *= w
                    h = w
            rows = None if n_sub == n else np.sort(rng.permutation(n)[:n_sub])
            feats = all_feats if k_sub == k else np.sort(rng.permutation(k)[:k_sub])
            if rows is not None:
                g = g[rows]
                h = None if h is None else h[rows]
            tree, leaf_of_row, gain = _grow_tree(
                XbT, rows, g, h, feats,
                n_bins=nb, max_depth=p["max_depth"], min_leaf=p["min_samples_leaf"],
                lam=p["l2"], mtry=0, rng=rng,
            )  # fmt: skip
            tree.value *= lr
            trees.append(tree)
            gains.append(gain)
            if rows is None:
                F += tree.value[leaf_of_row]
            else:
                F[rows] += tree.value[leaf_of_row]
                outside = np.ones(n, dtype=bool)
                outside[rows] = False
                others = np.flatnonzero(outside)
                F[others] += tree.value[tree.leaves(flat, n, others, nb)]
            if val is not None:
                F_val += tree.value[tree.leaves(val_flat, n_val, val_rows, nb)]
                loss = float(np.mean(_logistic_loss(F_val, y_val))) if clf else float(np.mean((F_val - y_val) ** 2))
                if loss < best_loss:
                    best_loss, best_iter = loss, it + 1
                elif patience and it + 1 - best_iter >= patience:
                    stopped = True
                    break
        n_built = len(trees)
        if patience and best_iter:
            trees, gains = trees[:best_iter], gains[:best_iter]
        self._init = float(init)
        self._trees = trees
        self._feature_importance = self._importance(np.sum(gains, axis=0))
        _check_finite_arrays(self.family, np.array([self._init]), *(t.value for t in trees))
        self._info = {
            "n_trees_built": n_built,
            "best_iteration": best_iter if val is not None and best_iter else None,
            "best_eval_loss": _optional_float(best_loss if val is not None else None),
            "stopped_early": stopped,
        }

    def _predict(self, X):
        raw = self._init + self._raw_sum(X)
        return _sigmoid(raw) if self.task == "classification" else raw

    def _describe(self):
        return {"init": self._init, **super()._describe()}

    def _get_state(self):
        return {"init": _finite_float(self._init, "init"), **super()._get_state()}

    def _set_state(self, state):
        super()._set_state(state)
        self._init = _finite_float(state["init"], "init")


class RandomForestModel(_TreeEnsemble):
    """Average of squared-loss trees, each on a bootstrap sample with per-split feature sampling.

    Bootstrap multiplicities act as row weights; ``min_samples_leaf`` counts
    distinct rows.  Classification trees average 0/1 labels, and the forest's
    probability is clipped to ``[1e-6, 1 - 1e-6]``.
    """

    family = "random_forest"
    _CLIP = 1e-6

    def _fit(self, X, y, w, val):
        p = self.params
        rng = np.random.default_rng(self.seed)
        n, k = X.shape
        XbT = self._prepare_bins(X)
        mtry = max(1, int(p["max_features"] * k + 1e-9))
        all_feats = np.arange(k, dtype=np.intp)
        trees, gains = [], []
        for _ in range(p["n_estimators"]):
            counts = np.bincount(rng.integers(0, n, size=n), minlength=n)
            rows = np.flatnonzero(counts)
            weight = counts[rows] * w[rows]
            # Squared loss from a zero prediction: leaf = weighted mean of y, gain = SSE reduction.
            tree, _, gain = _grow_tree(
                XbT, rows, -weight * y[rows], weight, all_feats,
                n_bins=self._n_bins, max_depth=p["max_depth"], min_leaf=p["min_samples_leaf"],
                lam=0.0, mtry=mtry, rng=rng,
            )  # fmt: skip
            trees.append(tree)
            gains.append(gain)
        self._trees = trees
        self._feature_importance = self._importance(np.sum(gains, axis=0))
        _check_finite_arrays(self.family, *(t.value for t in trees))
        self._info = {"max_features_per_split": mtry}

    def _predict(self, X):
        mean = self._raw_sum(X) / len(self._trees)
        if self.task == "classification":
            return np.clip(mean, self._CLIP, 1.0 - self._CLIP)
        return mean


# --------------------------------------------------------------------------- multilayer perceptron


class MLPModel(Model):
    """Fully connected network trained by mini-batch AdamW (decoupled decay on weights, not biases).

    Regression fits standardised ``y`` with squared loss; classification uses
    cross-entropy on logits.  With an ``eval_set`` the eval loss (MSE in the
    units of ``y``, or mean log-loss) is tracked per epoch; with
    ``early_stopping_rounds > 0`` training stops after that many epochs without
    improvement and the best epoch's weights are restored.
    """

    family = "mlp"
    _BETA1 = 0.9
    _BETA2 = 0.999
    _EPS = 1e-8

    def _forward(self, Z: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        relu = self.params["activation"] == "relu"
        acts = [Z]
        a = Z
        for W, b in zip(self._weights[:-1], self._biases[:-1]):
            z = a @ W + b
            a = np.maximum(z, 0.0) if relu else np.tanh(z)
            acts.append(a)
        out = (a @ self._weights[-1] + self._biases[-1])[:, 0]
        return out, acts

    def _fit(self, X, y, w, val):
        p = self.params
        rng = np.random.default_rng(self.seed)
        clf = self.task == "classification"
        relu = p["activation"] == "relu"
        self._medians, self._means, self._scales = _fit_scaler(X, w)
        Z = _standardize(X, self._medians, self._means, self._scales)
        n, k = Z.shape
        if clf:
            self._y_mean, self._y_scale = 0.0, 1.0
        elif np.all(y == y[0]):
            self._y_mean, self._y_scale = float(y[0]), 1.0
        else:
            self._y_mean = _weighted_mean(y, w)
            self._y_scale = math.sqrt(_weighted_mean((y - self._y_mean) ** 2, w))
        target = (y - self._y_mean) / self._y_scale

        sizes = (k, *_HIDDEN_LAYERS[p["hidden"]], 1)
        self._weights, self._biases = [], []
        for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            hidden_relu = relu and i < len(sizes) - 2
            std = math.sqrt(2.0 / fan_in) if hidden_relu else math.sqrt(2.0 / (fan_in + fan_out))  # He / Xavier
            self._weights.append(rng.standard_normal((fan_in, fan_out)) * std)
            self._biases.append(np.zeros(fan_out))
        if clf:
            prior = _weighted_mean(y, w)
            self._biases[-1][0] = math.log(prior / (1.0 - prior))
        n_layers = len(self._weights)
        params = self._weights + self._biases
        m1 = [np.zeros_like(a) for a in params]
        m2 = [np.zeros_like(a) for a in params]
        lr, wd = p["learning_rate"], p["weight_decay"]
        batch = min(p["batch_size"], n)
        unit_w = bool(np.all(w == 1.0))

        if val is not None:
            Z_val = _standardize(val[0], self._medians, self._means, self._scales)
            y_val = val[1]
        patience = p["early_stopping_rounds"] if val is not None else 0
        best_loss, best_epoch, best_params, stopped = math.inf, 0, None, False
        steps = epochs_run = 0
        with np.errstate(over="ignore", invalid="ignore"):
            for epoch in range(1, p["epochs"] + 1):
                order = rng.permutation(n)
                for start in range(0, n, batch):
                    idx = order[start : start + batch]
                    out, acts = self._forward(Z[idx])
                    resid = (_sigmoid(out) if clf else out) - target[idx]
                    if not unit_w:
                        resid *= w[idx]
                    delta = (resid / idx.shape[0])[:, None]
                    grads_w: list[np.ndarray] = []  # filled last layer first, then put in layer order
                    grads_b: list[np.ndarray] = []
                    for layer in range(n_layers - 1, -1, -1):
                        grads_w.append(acts[layer].T @ delta)
                        grads_b.append(delta.sum(axis=0))
                        if layer:
                            a = acts[layer]
                            delta = (delta @ self._weights[layer].T) * ((a > 0) if relu else (1.0 - a * a))
                    grads_w.reverse()
                    grads_b.reverse()
                    steps += 1
                    step_size = lr * math.sqrt(1.0 - self._BETA2**steps) / (1.0 - self._BETA1**steps)
                    for i, (param, grad) in enumerate(zip(params, grads_w + grads_b)):
                        m1[i] *= self._BETA1
                        m1[i] += (1.0 - self._BETA1) * grad
                        m2[i] *= self._BETA2
                        m2[i] += (1.0 - self._BETA2) * grad * grad
                        if i < n_layers and wd:
                            param *= 1.0 - lr * wd  # decoupled weight decay (AdamW), weights only
                        param -= step_size * m1[i] / (np.sqrt(m2[i]) + self._EPS)
                epochs_run = epoch
                if not all(np.isfinite(a).all() for a in params):
                    raise FloatingPointError(f"mlp training diverged in epoch {epoch}; lower learning_rate")
                if val is not None:
                    out_val, _ = self._forward(Z_val)
                    if clf:
                        loss = float(np.mean(_logistic_loss(out_val, y_val)))
                    else:
                        loss = float(np.mean((out_val * self._y_scale + self._y_mean - y_val) ** 2))
                    if loss < best_loss:
                        best_loss, best_epoch = loss, epoch
                        if patience:
                            best_params = [a.copy() for a in params]
                    elif patience and epoch - best_epoch >= patience:
                        stopped = True
                        break
        if best_params is not None:
            self._weights = best_params[:n_layers]
            self._biases = best_params[n_layers:]
        self._info = {
            "layers": list(sizes),
            "n_parameters": int(sum(a.size for a in self._weights + self._biases)),
            "epochs_run": epochs_run,
            "n_steps": steps,
            "best_epoch": best_epoch if val is not None and best_epoch else None,
            "best_eval_loss": _optional_float(best_loss if val is not None else None),
            "stopped_early": stopped,
        }

    def _predict(self, X):
        out, _ = self._forward(_standardize(X, self._medians, self._means, self._scales))
        if self.task == "classification":
            return _sigmoid(out)
        return out * self._y_scale + self._y_mean

    def _get_state(self):
        return {
            "medians": _finite_list(self._medians, "medians"),
            "means": _finite_list(self._means, "means"),
            "scales": _finite_list(self._scales, "scales"),
            "y_mean": _finite_float(self._y_mean, "y_mean"),
            "y_scale": _finite_float(self._y_scale, "y_scale"),
            "weights": [_finite_list(W, "weights") for W in self._weights],
            "biases": [_finite_list(b, "biases") for b in self._biases],
        }

    def _set_state(self, state):
        k = int(state["n_features"])
        self._medians = _state_array(state, "medians", size=k)
        self._means = _state_array(state, "means", size=k)
        self._scales = _state_array(state, "scales", size=k)
        self._y_mean = _finite_float(state["y_mean"], "y_mean")
        self._y_scale = _finite_float(state["y_scale"], "y_scale")
        sizes = (k, *_HIDDEN_LAYERS[self.params["hidden"]], 1)
        weights = [np.asarray(W, dtype=np.float64) for W in state["weights"]]
        biases = [np.asarray(b, dtype=np.float64) for b in state["biases"]]
        if len(weights) != len(sizes) - 1 or len(biases) != len(sizes) - 1:
            raise ValueError("mlp state has the wrong number of layers")
        for W, b, fan_in, fan_out in zip(weights, biases, sizes[:-1], sizes[1:]):
            if W.shape != (fan_in, fan_out) or b.shape != (fan_out,):
                raise ValueError("mlp layer shapes do not match the hidden parameter")
            if not (np.isfinite(W).all() and np.isfinite(b).all()):
                raise ValueError("mlp state contains non-finite values")
        self._weights, self._biases = weights, biases


# --------------------------------------------------------------------------- factory / serialisation

_CLASSES: dict[str, type[Model]] = {
    cls.family: cls
    for cls in (ConstantModel, RidgeModel, LogisticModel, GBDTModel, RandomForestModel, MLPModel)
}


def make_model(family: str, params: dict | None = None, *, task: str, seed: int = 0) -> Model:
    """Create an unfitted model; parameters are checked with :func:`validate_params`."""
    _family_spec(family)
    return _CLASSES[family](params, task=task, seed=seed)


def model_from_json(value: dict) -> Model:
    """Rebuild a model from :meth:`Model.to_json` output (predictions are bit-identical)."""
    if not isinstance(value, dict):
        raise ValueError("model JSON must be a dict")
    missing = [key for key in ("family", "task", "params", "seed", "state") if key not in value]
    if missing:
        raise ValueError(f"model JSON is missing {missing}")
    model = make_model(value["family"], value["params"], task=value["task"], seed=value["seed"])
    state = value["state"]
    if state is None:
        return model
    if not isinstance(state, dict):
        raise ValueError("model JSON state must be a dict or null")
    if state.get("format") != _STATE_FORMAT:
        raise ValueError(f"unsupported model state format {state.get('format')!r}")
    try:
        n_features = state["n_features"]
        if isinstance(n_features, bool) or not isinstance(n_features, int) or n_features < 1:
            raise ValueError(f"invalid n_features {n_features!r}")
        model._set_state(state)
        info = state.get("info") or {}
        if not isinstance(info, dict):
            raise ValueError("state info must be a dict")
        model._info = dict(info)
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError(f"invalid {model.family} state: {exc!r}") from exc
    model.n_features = n_features
    model._fitted = True
    return model
