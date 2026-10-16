"""Local daily simple-return metrics for an already-computed net return ledger.

These are project definitions, not Qlib risk_analysis or Brain-equivalent results.
Inputs must be aligned account return observations; fees and other costs must
already be reflected in ``net_returns``. No trades or returns are simulated here.
"""

import math
import statistics


METRIC_VERSION = "daily-simple-net-v1"
STRATEGY_METRIC_VERSION = "daily-simple-strategy-net-v1"


def _finite_or_none(value):
    return value if math.isfinite(value) else None


def _returns(values, name):
    try:
        observations = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of numeric returns") from exc
    result = []
    for value in observations:
        if isinstance(value, (str, bytes, bool, complex)):
            raise ValueError(f"{name} must contain finite numeric returns")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must contain finite numeric returns") from exc
        if not math.isfinite(number) or number <= -1:
            raise ValueError(f"{name} returns must be finite and greater than -1")
        result.append(number)
    return result


def _sample_std(values):
    if len(values) < 2:
        return None
    if max(values) == min(values):
        return 0.0  # exact: fmean rounding would otherwise leave a spurious ~1e-19 deviation
    mean = statistics.fmean(values)
    deviations = [value - mean for value in values]
    scale = max(map(abs, deviations))
    if scale == 0:
        return 0.0
    variance = math.fsum((value / scale) ** 2 for value in deviations) / (len(values) - 1)
    return _finite_or_none(scale * math.sqrt(variance))


def _ratio(mean, std, periods_per_year):
    if mean is None or std is None or std == 0:
        return None
    return _finite_or_none(mean / std * math.sqrt(periods_per_year))


def _expm1_or_none(exponent):
    try:
        return _finite_or_none(math.expm1(exponent))
    except OverflowError:
        return None


def _account_metrics(returns, cash_daily, periods_per_year):
    count = len(returns)
    result = {
        "observations": count,
        "mean_daily_return": None,
        "annualized_arithmetic_return": None,
        "annualized_volatility": None,
        "cumulative_return": None,
        "cagr": None,
        "max_drawdown": None,
        "sharpe": None,
        "sharpe_zero_cash_rate": None,
    }
    if not count:
        return result

    mean = statistics.fmean(returns)
    std = _sample_std(returns)
    log_nav = 0.0
    peak_log_nav = 0.0  # Initial NAV is 1, before the first return.
    max_drawdown = 0.0
    for daily_return in returns:
        log_nav += math.log1p(daily_return)
        peak_log_nav = max(peak_log_nav, log_nav)
        max_drawdown = min(max_drawdown, math.expm1(log_nav - peak_log_nav))

    result.update({
        "mean_daily_return": mean,
        "annualized_arithmetic_return": _finite_or_none(mean * periods_per_year),
        "annualized_volatility": (
            _finite_or_none(std * math.sqrt(periods_per_year)) if std is not None else None
        ),
        "cumulative_return": _expm1_or_none(log_nav),
        "cagr": _expm1_or_none(log_nav * periods_per_year / count),
        "max_drawdown": max_drawdown,
        "sharpe": _ratio(mean - cash_daily, std, periods_per_year),
        "sharpe_zero_cash_rate": _ratio(mean, std, periods_per_year),
    })
    return result


def portfolio_metrics(net_returns, benchmark_returns=None, *, annual_cash_rate=0.0, periods_per_year=252):
    """Summarize aligned daily simple returns; all output values are JSON-safe.

    Volatility and ratios use sample standard deviation (ddof=1). Undefined
    statistics for empty, single-observation, or zero-variance series are None.
    Excess returns are daily differences, so no compounded excess CAGR is given.
    Omitting benchmark_returns computes only strategy metrics. A zero-return
    benchmark is not substituted; legacy replay may still supply saved returns.
    """
    if isinstance(periods_per_year, bool) or not isinstance(periods_per_year, int) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer")
    if isinstance(annual_cash_rate, (str, bytes, bool, complex)):
        raise ValueError("annual_cash_rate must be finite and greater than -1")
    try:
        cash_rate = float(annual_cash_rate)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("annual_cash_rate must be finite and greater than -1") from exc
    if not math.isfinite(cash_rate) or cash_rate <= -1:
        raise ValueError("annual_cash_rate must be finite and greater than -1")

    strategy = _returns(net_returns, "net_returns")
    cash_daily = math.expm1(math.log1p(cash_rate) / periods_per_year)
    if benchmark_returns is None:
        return {"strategy_net": _account_metrics(strategy, cash_daily, periods_per_year)}
    benchmark = _returns(benchmark_returns, "benchmark_returns")
    if len(strategy) != len(benchmark):
        raise ValueError("net_returns and benchmark_returns must have matching lengths")
    excess = [net - reference for net, reference in zip(strategy, benchmark)]
    excess_mean = statistics.fmean(excess) if excess else None
    excess_std = _sample_std(excess)
    return {
        "strategy_net": _account_metrics(strategy, cash_daily, periods_per_year),
        "benchmark_reference": _account_metrics(benchmark, cash_daily, periods_per_year),
        "excess_net_vs_benchmark": {
            "observations": len(excess),
            "mean_daily_excess_return": excess_mean,
            "annualized_arithmetic_excess_return": (
                _finite_or_none(excess_mean * periods_per_year) if excess_mean is not None else None
            ),
            "tracking_error": (
                _finite_or_none(excess_std * math.sqrt(periods_per_year))
                if excess_std is not None else None
            ),
            "information_ratio": _ratio(excess_mean, excess_std, periods_per_year),
        },
    }


def holdings_gate(metrics, *, threshold=1.3, min_observations=252):
    """Accept only a finite local net Sharpe strictly above the threshold."""
    if not isinstance(min_observations, int) or isinstance(min_observations, bool) or min_observations < 0:
        raise ValueError("min_observations must be a nonnegative integer")
    try:
        cutoff = float(threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("threshold must be finite") from exc
    if not math.isfinite(cutoff):
        raise ValueError("threshold must be finite")
    if not isinstance(metrics, dict) or not isinstance(metrics.get("strategy_net"), dict):
        return False
    strategy = metrics["strategy_net"]
    sharpe = strategy.get("sharpe")
    count = strategy.get("observations")
    return (
        isinstance(count, int) and not isinstance(count, bool) and count >= min_observations
        and isinstance(sharpe, (int, float)) and not isinstance(sharpe, bool)
        and math.isfinite(sharpe) and sharpe > cutoff
    )
