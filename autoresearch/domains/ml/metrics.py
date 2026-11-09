"""Evaluation metrics for regression and binary classification (pure numpy).

Every metric takes two 1-D arrays of equal, non-zero length: ``y`` (truth) and
``p`` (prediction; for classification metrics, the probability or score of
class 1).  Nothing is ignored silently: a length mismatch, empty input, a
non-1-D array or any non-finite value raises ``ValueError``, as do labels other
than 0/1 where a metric needs binary labels.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

__all__ = [
    "METRICS",
    "accuracy",
    "auc",
    "log_loss",
    "mae",
    "pearson",
    "r2",
    "rmse",
    "score",
    "spearman",
]


def _pair(y: Any, p: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert a (truth, prediction) pair to finite float64 vectors."""
    try:
        ya = np.asarray(y, dtype=np.float64)
        pa = np.asarray(p, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metric inputs must be numeric arrays: {exc}") from None
    if ya.ndim != 1 or pa.ndim != 1:
        raise ValueError(f"metric inputs must be 1-D, got shapes {ya.shape} and {pa.shape}")
    if ya.shape[0] != pa.shape[0]:
        raise ValueError(f"length mismatch: y has {ya.shape[0]} values, p has {pa.shape[0]}")
    if ya.shape[0] == 0:
        raise ValueError("metric inputs are empty")
    if not (np.isfinite(ya).all() and np.isfinite(pa).all()):
        raise ValueError("metric inputs contain NaN or inf")
    return ya, pa


def _binary(y: np.ndarray) -> np.ndarray:
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("y must contain only 0/1 labels")
    return y


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks of ``x`` with tied values sharing their average rank."""
    n = x.shape[0]
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    np.not_equal(xs[1:], xs[:-1], out=new_group[1:])
    starts = np.flatnonzero(new_group)
    ends = np.append(starts[1:], n)  # exclusive
    group_rank = (starts + ends + 1) / 2.0  # mean of 1-based ranks starts+1 .. ends
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = group_rank[np.cumsum(new_group) - 1]
    return ranks


def _is_constant(x: np.ndarray) -> bool:
    return bool(np.all(x == x[0]))


def rmse(y: Any, p: Any) -> float:
    """Root mean squared error."""
    ya, pa = _pair(y, p)
    return float(math.sqrt(np.mean((ya - pa) ** 2)))


def mae(y: Any, p: Any) -> float:
    """Mean absolute error."""
    ya, pa = _pair(y, p)
    return float(np.mean(np.abs(ya - pa)))


def r2(y: Any, p: Any) -> float:
    """Coefficient of determination ``1 - SS_res / SS_tot``.

    For constant ``y`` (``SS_tot == 0``) it is 1.0 when the prediction is exact
    and 0.0 otherwise, so the result is always finite.
    """
    ya, pa = _pair(y, p)
    ss_res = float(np.sum((ya - pa) ** 2))
    if _is_constant(ya):
        return 1.0 if ss_res == 0.0 else 0.0
    ss_tot = float(np.sum((ya - ya.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def log_loss(y: Any, p: Any, eps: float = 1e-12) -> float:
    """Mean binary cross-entropy; ``p`` must lie in [0, 1] and is clipped to [eps, 1 - eps]."""
    if not (isinstance(eps, (int, float)) and 0.0 < eps < 0.5):
        raise ValueError(f"eps must be in (0, 0.5), got {eps!r}")
    ya, pa = _pair(y, p)
    _binary(ya)
    if np.any((pa < 0.0) | (pa > 1.0)):
        raise ValueError("p must be probabilities in [0, 1]")
    pc = np.clip(pa, eps, 1.0 - eps)
    return float(-np.mean(ya * np.log(pc) + (1.0 - ya) * np.log1p(-pc)))


def accuracy(y: Any, p: Any, threshold: float = 0.5) -> float:
    """Fraction of rows whose predicted label ``p >= threshold`` equals the 0/1 label."""
    ya, pa = _pair(y, p)
    _binary(ya)
    return float(np.mean((pa >= threshold) == (ya == 1.0)))


def auc(y: Any, p: Any) -> float:
    """ROC AUC via the Mann-Whitney U statistic (tied scores count one half).

    Raises ``ValueError`` when ``y`` contains only one class.
    """
    ya, pa = _pair(y, p)
    _binary(ya)
    pos = ya == 1.0
    n_pos = int(pos.sum())
    n_neg = ya.shape[0] - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("auc needs both classes in y")
    ranks = _average_ranks(pa)
    u = float(ranks[pos].sum()) - n_pos * (n_pos + 1) / 2.0
    return u / (float(n_pos) * float(n_neg))


def pearson(y: Any, p: Any) -> float:
    """Pearson correlation; NaN when either side is constant."""
    ya, pa = _pair(y, p)
    if _is_constant(ya) or _is_constant(pa):
        return float("nan")
    yc = ya - ya.mean()
    pc = pa - pa.mean()
    r = float(yc @ pc) / math.sqrt(float(yc @ yc) * float(pc @ pc))
    return min(1.0, max(-1.0, r))


def spearman(y: Any, p: Any) -> float:
    """Spearman rank correlation (average ranks for ties); NaN when either side is constant."""
    ya, pa = _pair(y, p)
    return pearson(_average_ranks(ya), _average_ranks(pa))


METRICS: dict[str, dict] = {
    "rmse": {"greater_is_better": False, "task": "regression"},
    "mae": {"greater_is_better": False, "task": "regression"},
    "r2": {"greater_is_better": True, "task": "regression"},
    "log_loss": {"greater_is_better": False, "task": "classification"},
    "accuracy": {"greater_is_better": True, "task": "classification"},
    "auc": {"greater_is_better": True, "task": "classification"},
    "pearson": {"greater_is_better": True, "task": "any"},
    "spearman": {"greater_is_better": True, "task": "any"},
}

_FUNCTIONS: dict[str, Callable[[Any, Any], float]] = {
    "rmse": rmse,
    "mae": mae,
    "r2": r2,
    "log_loss": log_loss,
    "accuracy": accuracy,
    "auc": auc,
    "pearson": pearson,
    "spearman": spearman,
}


def score(metric_name: str, y: Any, p: Any) -> float:
    """Evaluate the metric called ``metric_name`` (a key of ``METRICS``) with default options."""
    fn = _FUNCTIONS.get(metric_name) if isinstance(metric_name, str) else None
    if fn is None:
        raise ValueError(f"unknown metric {metric_name!r}; expected one of {sorted(METRICS)}")
    return fn(y, p)
