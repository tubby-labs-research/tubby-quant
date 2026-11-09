"""Training-recipe search space: sampling, local mutation and canonical identity.

A recipe is ``{"family": ..., "params": {...}}`` for one model family of
:mod:`.models`. Sampling uses search ranges that are narrower than the
families' validity ranges (no one needs 10,000 trees to start), log-uniform for
scale parameters. Mutation perturbs one or two parameters around a parent and
occasionally jumps to another family, which together act as a simple
evolutionary hyperparameter search; the campaign's two-stage cascade (cheap
screen on a development subsample, full validation for the shortlist) is the
successive-halving step.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

from .models import FAMILIES, validate_params

# Narrower ranges for search; anything outside still validates if supplied explicitly.
SEARCH_RANGES = {
    "ridge": {"alpha": (1e-3, 1e3)},
    "logistic": {"C": (1e-3, 1e2)},
    "gbdt": {"n_estimators": (40, 400), "learning_rate": (0.02, 0.3), "max_depth": (2, 6),
             "min_samples_leaf": (10, 300), "l2": (1e-3, 10.0), "subsample": (0.5, 1.0), "colsample": (0.5, 1.0),
             "max_bins": (32, 128)},
    "random_forest": {"n_estimators": (30, 200), "max_depth": (3, 10), "min_samples_leaf": (5, 200),
                      "max_features": (0.3, 1.0), "max_bins": (32, 128)},
    "mlp": {"learning_rate": (3e-4, 3e-2), "weight_decay": (1e-7, 1e-2), "batch_size": (64, 1024), "epochs": (10, 120)},
}
FIXED = {"gbdt": {"early_stopping_rounds": 0}, "mlp": {"early_stopping_rounds": 0}}
DEFAULT_FAMILIES = {"regression": ["ridge", "gbdt", "random_forest", "mlp"],
                    "classification": ["logistic", "gbdt", "random_forest", "mlp"]}


def families_for(problem: str, requested: list[str] | None = None) -> list[str]:
    allowed = [f for f in FAMILIES if problem in FAMILIES[f]["tasks"] and f != "constant"]
    chosen = list(requested or DEFAULT_FAMILIES[problem])
    bad = [f for f in chosen if f not in allowed]
    if bad or not chosen:
        raise ValueError(f"model families for {problem}: choose from {', '.join(allowed)}")
    return chosen


def _range(family: str, name: str, spec: dict[str, Any]) -> tuple[float, float]:
    """Search bounds: the narrower table above, else the family's own validity range (every numeric
    parameter spec carries ``low`` and ``high``)."""
    low, high = SEARCH_RANGES.get(family, {}).get(name, (spec["low"], spec["high"]))
    return low, high


def _sample_value(rng: np.random.Generator, family: str, name: str, spec: dict[str, Any]) -> Any:
    kind = spec["type"]
    if kind == "bool":
        return bool(rng.random() < 0.5)
    if kind == "choice":
        return spec["choices"][int(rng.integers(len(spec["choices"])))]
    low, high = _range(family, name, spec)
    use_log = spec.get("log") or (low > 0 and high / max(low, 1e-300) > 50)
    if use_log and low > 0:
        value = math.exp(rng.uniform(math.log(low), math.log(high)))
    else:
        value = rng.uniform(low, high)
    return int(round(value)) if kind == "int" else float(value)


def sample(rng: np.random.Generator, family: str, problem: str) -> dict[str, Any]:
    spec = FAMILIES[family]["params"]
    params = {name: _sample_value(rng, family, name, item) for name, item in spec.items()}
    params.update(FIXED.get(family, {}))
    return {"family": family, "params": validate_params(family, params, problem)}


def default(family: str, problem: str) -> dict[str, Any]:
    return {"family": family, "params": validate_params(family, dict(FIXED.get(family, {})), problem)}


def mutate(rng: np.random.Generator, recipe: dict[str, Any], problem: str, families: list[str], jump: float = 0.1) -> dict[str, Any]:
    if rng.random() < jump and len(families) > 1:
        return sample(rng, str(rng.choice([f for f in families if f != recipe["family"]])), problem)
    family = recipe["family"]
    spec = FAMILIES[family]["params"]
    params = dict(recipe["params"])
    tunable = [n for n in spec if n not in FIXED.get(family, {})]
    if not tunable:
        return sample(rng, family, problem)
    for name in rng.choice(tunable, size=min(len(tunable), int(rng.integers(1, 3))), replace=False):
        item = spec[str(name)]
        current = params[str(name)]
        if item["type"] == "bool":
            params[str(name)] = not current
        elif item["type"] == "choice":
            params[str(name)] = item["choices"][int(rng.integers(len(item["choices"])))]
        else:
            low, high = _range(family, str(name), item)
            if low is not None and low > 0 and current > 0:
                value = float(np.clip(current * math.exp(rng.normal(0, 0.5)), low, high))
            else:
                value = float(np.clip(current + rng.normal(0, 0.2) * (high - low), low, high))
            params[str(name)] = int(round(value)) if item["type"] == "int" else value
    return {"family": family, "params": validate_params(family, params, problem)}


def scale_recipe(recipe: dict[str, Any], factor: float, problem: str) -> dict[str, Any]:
    """The recipe with every numeric, non-fixed parameter scaled by ``factor``, clamped to its validity range.

    The model analogue of scaling an expression's windows: one deterministic perturbation of
    the whole recipe, used as a robustness probe. Boolean, categorical and fixed parameters
    stay as they are; a value that rounds or clamps back to itself is unchanged.
    """
    family = recipe["family"]
    spec = FAMILIES[family]["params"]
    params = validate_params(family, dict(recipe.get("params") or {}), problem)
    for name, item in spec.items():
        if item["type"] not in ("int", "float") or name in FIXED.get(family, {}):
            continue
        value = float(np.clip(params[name] * factor, item["low"], item["high"]))
        params[name] = int(round(value)) if item["type"] == "int" else value
    return {"family": family, "params": validate_params(family, params, problem)}


def canonical(recipe: dict[str, Any], problem: str) -> dict[str, Any]:
    params = validate_params(recipe["family"], recipe.get("params") or {}, problem)
    rounded = {k: (float(f"{v:.4g}") if isinstance(v, float) else v) for k, v in sorted(params.items())}
    return {"family": recipe["family"], "params": rounded}


def recipe_key(recipe: dict[str, Any], problem: str, task_kind: str) -> str:
    body = json.dumps({"task": task_kind, "problem": problem, **canonical(recipe, problem)},
                      sort_keys=True, separators=(",", ":"))
    return "recipe_" + hashlib.sha256(body.encode()).hexdigest()[:40]


def title(recipe: dict[str, Any]) -> str:
    p = recipe["params"]
    family = recipe["family"]
    if family == "gbdt":
        return f"gbdt {p['n_estimators']} trees depth {p['max_depth']} lr {p['learning_rate']:.3g}"
    if family == "random_forest":
        return f"forest {p['n_estimators']} trees depth {p['max_depth']}"
    if family == "mlp":
        return f"mlp {p['hidden']} {p['activation']} lr {p['learning_rate']:.2g}"
    if family == "ridge":
        return f"ridge alpha {p['alpha']:.3g}"
    if family == "logistic":
        return f"logistic C {p['C']:.3g}"
    return family
