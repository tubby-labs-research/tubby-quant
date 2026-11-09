"""Selection statistics for large searches: what a best-of-N result is worth.

Searching many candidates and keeping the best inflates the winner's measured
Sharpe ratio even when nothing works. These functions price that selection:

* :func:`expected_max_sharpe` — the Sharpe ratio the best of ``n`` independent
  null trials is expected to reach, given the cross-trial variance of Sharpe
  estimates (Bailey and López de Prado, "The Deflated Sharpe Ratio", 2014).
* :func:`deflated_sharpe` — the probability that a candidate's true Sharpe
  exceeds that benchmark, correcting for sample length, skewness and fat tails.
* :func:`probabilistic_sharpe` — the same with an explicit benchmark (n = 1).
* :func:`holm` and :func:`bonferroni` — familywise error control for p-values.

Every Sharpe ratio here is per observation (daily), not annualized. Trials that
are strongly correlated make ``n`` an overstatement of independent tries, so the
correction is conservative for families of near-duplicates. Pure standard
library: :class:`statistics.NormalDist` supplies the normal CDF and quantiles.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping
from statistics import NormalDist
from typing import NotRequired, TypedDict, TypeVar

EULER_MASCHERONI = 0.5772156649015329
_NORMAL = NormalDist()

K = TypeVar("K", bound=Hashable)


class Moments(TypedDict):
    """``mean`` is ``None`` for an empty series and ``std`` for fewer than two values."""

    n: int
    mean: float | None
    std: float | None
    skew: float
    kurtosis: float


class DeflatedSharpe(TypedDict):
    """:func:`deflated_sharpe`; callers may add the ``skew`` and ``kurtosis`` they used."""

    dsr: float
    benchmark_sharpe: float
    trial_variance: float
    n_trials: int
    observations: int
    sharpe: float
    annualized_benchmark: float
    skew: NotRequired[float]
    kurtosis: NotRequired[float]


def moments(values: Iterable[float | None]) -> Moments:
    """Mean, sample standard deviation, skewness and (non-excess) kurtosis of a series."""
    data = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(data)
    if n < 2:
        return {"n": n, "mean": data[0] if data else None, "std": None, "skew": 0.0, "kurtosis": 3.0}
    mean = sum(data) / n
    centred = [v - mean for v in data]
    m2 = sum(c * c for c in centred) / n
    if m2 <= 0:
        return {"n": n, "mean": mean, "std": 0.0, "skew": 0.0, "kurtosis": 3.0}
    m3 = sum(c ** 3 for c in centred) / n
    m4 = sum(c ** 4 for c in centred) / n
    return {"n": n, "mean": mean, "std": math.sqrt(m2 * n / (n - 1)),
            "skew": m3 / m2 ** 1.5, "kurtosis": m4 / m2 ** 2}


def daily_sharpe(values: Iterable[float | None]) -> float | None:
    stats = moments(values)
    mean, std = stats["mean"], stats["std"]
    if not std or mean is None:
        return None
    return mean / std


def expected_max_sharpe(n_trials: int, variance: float) -> float:
    """E[max SR] over ``n_trials`` null trials whose Sharpe estimates have ``variance``."""
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if not math.isfinite(variance) or variance < 0:
        raise ValueError("variance must be finite and nonnegative")
    if n_trials == 1:
        return 0.0
    gamma = EULER_MASCHERONI
    first = _NORMAL.inv_cdf(1 - 1 / n_trials)
    second = _NORMAL.inv_cdf(1 - 1 / (n_trials * math.e))
    return math.sqrt(variance) * ((1 - gamma) * first + gamma * second)


def probabilistic_sharpe(sharpe: float, benchmark: float, observations: int,
                         skew: float = 0.0, kurtosis: float = 3.0) -> float:
    """P(true SR > benchmark) for an observed per-period ``sharpe`` over ``observations`` periods."""
    if observations < 2:
        raise ValueError("at least two observations are required")
    denominator = 1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe ** 2
    if denominator <= 0:
        # Extreme skew/kurtosis combinations make the asymptotic variance invalid;
        # refuse to claim significance rather than divide by a nonpositive number.
        return 0.0
    z = (sharpe - benchmark) * math.sqrt(observations - 1) / math.sqrt(denominator)
    return _NORMAL.cdf(z)


def deflated_sharpe(sharpe: float, observations: int, n_trials: int, trial_variance: float | None = None,
                    skew: float = 0.0, kurtosis: float = 3.0) -> DeflatedSharpe:
    """Deflated Sharpe ratio of one selected candidate among ``n_trials`` tried.

    ``trial_variance`` is the variance of per-period Sharpe estimates across the
    trials. With too few trials to estimate it, the null sampling variance
    ``1 / observations`` is used as a floor, which keeps the benchmark positive
    once ``n_trials`` exceeds one.
    """
    if observations < 2:
        raise ValueError("at least two observations are required")
    floor = 1.0 / observations
    variance = floor if trial_variance is None or not math.isfinite(trial_variance) else max(trial_variance, floor)
    benchmark = expected_max_sharpe(max(1, int(n_trials)), variance)
    probability = probabilistic_sharpe(sharpe, benchmark, observations, skew, kurtosis)
    return {"dsr": probability, "benchmark_sharpe": benchmark, "trial_variance": variance,
            "n_trials": int(n_trials), "observations": int(observations), "sharpe": sharpe,
            "annualized_benchmark": benchmark * math.sqrt(252)}


def pearson(a: Iterable[float | None] | None, b: Iterable[float | None] | None) -> float | None:
    """Correlation of two aligned series, skipping missing pairs; None when undefined."""
    pairs = [(float(x), float(y)) for x, y in zip(a or (), b or ()) if x is not None and y is not None
             and math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 3:
        return None
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    sxx = sum((x - mx) ** 2 for x, _ in pairs)
    syy = sum((y - my) ** 2 for _, y in pairs)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def bonferroni(alpha: float, n_tests: int) -> float:
    if not 0 < alpha < 1 or n_tests < 1:
        raise ValueError("alpha must be in (0, 1) and n_tests at least 1")
    return alpha / n_tests


def holm(pvalues: Mapping[K, float], alpha: float = 0.05) -> dict[K, bool]:
    """Holm step-down: which hypotheses are rejected at familywise level ``alpha``."""
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], str(item[0])))
    rejected: dict[K, bool] = {}
    m = len(ordered)
    still = True
    for rank, (name, p) in enumerate(ordered):
        still = still and p <= alpha / (m - rank)
        rejected[name] = still
    return rejected
