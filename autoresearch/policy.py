"""Deterministic draft research gates, not universal trading criteria.

Defaults are an explicit starting research policy: 60 validation trading days,
net annual returns > 0, Sharpe >= 1, turnover <= .7 and drawdown <= .3.
Promotion authorizes deeper research only. Production use must choose thresholds
for its universe, costs and risk assumptions. Holdout metrics are never read.
"""
from __future__ import annotations
import math

DEFAULTS = {'min_validation_days': 60, 'min_sharpe': 1.0, 'min_returns': 0.0,
            'max_turnover': 0.7, 'max_drawdown': 0.3, 'min_refinement_improvement': 0.05}


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _answer(decision, reason, question='', expected_change=''):
    return {'decision': decision, 'reason': reason, 'question': question, 'expected_change': expected_change}


def assess(result, config, parent_result=None):
    """Assess chronological, cost-adjusted validation metrics using explicit gates.

    Accept a policy mapping or a complete config containing ``policy``. Missing
    operational evidence is repair; short history parks; nonpositive scientific
    outcomes reject. Positive weak outcomes may refine, unless parent comparison
    shows less than the configured Sharpe improvement. Controller budgets also
    constrain refinement independently.
    """
    if not isinstance(result, dict) or result.get('status') != 'ok':
        return _answer('repair', 'Evaluation failed or has no successful status; repair the operational error.')
    if not isinstance(config, dict):
        return _answer('repair', 'Policy config must be a mapping.')
    supplied = config.get('policy', config)
    if not isinstance(supplied, dict):
        return _answer('repair', 'Policy config must be a mapping.')
    gates = dict(DEFAULTS, **supplied)
    for key in DEFAULTS:
        if not _finite(gates[key]):
            return _answer('repair', 'Invalid policy threshold: ' + key)
    if gates['min_validation_days'] < 2 or gates['max_turnover'] < 0 or gates['max_drawdown'] < 0 or gates['min_refinement_improvement'] < 0:
        return _answer('repair', 'Policy thresholds have invalid ranges.')
    all_metrics, all_periods = result.get('metrics'), result.get('periods')
    if not isinstance(all_metrics, dict) or not isinstance(all_periods, dict):
        return _answer('repair', 'Evaluation lacks structured metrics or period evidence.')
    metrics = all_metrics.get('validation', {})
    periods = all_periods.get('validation', {})
    if not isinstance(metrics, dict) or not isinstance(periods, dict):
        return _answer('repair', 'Validation metrics and period evidence must be mappings.')
    days = periods.get('days', metrics.get('days'))
    if not _finite(days) or days < 0 or int(days) != days:
        return _answer('repair', 'Validation trading-day count is missing or invalid.')
    if days < gates['min_validation_days']:
        return _answer('park', 'Insufficient validation trading days for the configured policy.')
    settings = result.get('settings', {})
    costs = settings.get('cost_bps') if isinstance(settings, dict) else None
    if not _finite(costs) or costs < 0:
        return _answer('repair', 'Backend cost_bps evidence is missing or invalid.')
    required = ('sharpe', 'returns', 'turnover', 'drawdown')
    for key in required:
        if not _finite(metrics.get(key)):
            return _answer('repair', 'Validation metric is missing or nonfinite: ' + key)
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not _finite(value):
            return _answer('repair', 'Validation metric is nonfinite: ' + key)
    if metrics['turnover'] < 0 or metrics['drawdown'] < 0:
        return _answer('repair', 'Validation turnover or drawdown has an invalid range.')
    if metrics['sharpe'] <= 0 or metrics['returns'] <= 0:
        return _answer('reject', 'Validation evidence is nonpositive after configured costs.')
    failed = []
    for name, passed in (
        ('Sharpe', metrics['sharpe'] >= gates['min_sharpe']),
        ('returns', metrics['returns'] > gates['min_returns']),
        ('turnover', metrics['turnover'] <= gates['max_turnover']),
        ('drawdown', metrics['drawdown'] <= gates['max_drawdown']),
    ):
        if not passed:
            failed.append(name)
    if not failed:
        return _answer('promote', 'All configured validation gates passed; eligible for deeper research only.')
    if isinstance(parent_result, dict) and parent_result.get('status') == 'ok':
        parent_all = parent_result.get('metrics')
        parent_metrics = parent_all.get('validation', {}) if isinstance(parent_all, dict) else {}
        parent_sharpe = parent_metrics.get('sharpe') if isinstance(parent_metrics, dict) else None
        if _finite(parent_sharpe) and metrics['sharpe'] - parent_sharpe < gates['min_refinement_improvement']:
            return _answer('reject', 'Refinement stalled: validation Sharpe improvement is below the configured minimum.')
    weak = ', '.join(failed)
    return _answer('refine', 'Positive validation evidence misses configured gates: ' + weak,
                   'Can a mechanism-grounded change improve ' + weak + ' on validation?',
                   'Improve the failed validation gates while retaining positive returns after costs.')


FINAL_DEFAULTS = {'min_holdout_days': 20, 'min_holdout_sharpe': 0.0}


def assess_final(result, config):
    """Deterministic confirmation gate for one sealed holdout evaluation.

    ``confirmed`` means the reserved period agreed with validation after costs
    over at least ``min_holdout_days`` sessions with Sharpe above
    ``min_holdout_sharpe``. It is evidence for further research, not a trading
    recommendation. A repeated access of the same holdout is never confirmed:
    repeated selection turns the holdout into development evidence.
    """
    if not isinstance(result, dict) or result.get('status') != 'ok':
        return _answer('repair', 'Holdout evaluation failed or has no successful status; repair the operational error.')
    if not isinstance(config, dict):
        return _answer('repair', 'Policy config must be a mapping.')
    supplied = config.get('policy', config)
    if not isinstance(supplied, dict):
        return _answer('repair', 'Policy config must be a mapping.')
    gates = dict(FINAL_DEFAULTS, **{k: v for k, v in supplied.items() if k in FINAL_DEFAULTS})
    for key in FINAL_DEFAULTS:
        if not _finite(gates[key]):
            return _answer('repair', 'Invalid policy threshold: ' + key)
    if gates['min_holdout_days'] < 2:
        return _answer('repair', 'Policy thresholds have invalid ranges.')
    metrics = (result.get('metrics') or {}).get('holdout')
    if not isinstance(metrics, dict):
        return _answer('repair', 'Holdout evaluation lacks structured metrics.')
    days = metrics.get('days')
    if not _finite(days) or days < 0 or int(days) != days:
        return _answer('repair', 'Holdout trading-day count is missing or invalid.')
    if days < gates['min_holdout_days']:
        return _answer('insufficient', 'Too few holdout trading days for the configured policy.')
    for key in ('sharpe', 'returns'):
        if not _finite(metrics.get(key)):
            return _answer('repair', 'Holdout metric is missing or nonfinite: ' + key)
    if result.get('reused'):
        return _answer('not_confirmed', 'A reused holdout is further research, not an untouched final test.')
    if metrics['returns'] <= 0 or metrics['sharpe'] <= 0:
        return _answer('not_confirmed', 'Holdout evidence is nonpositive after configured costs.')
    if metrics['sharpe'] < gates['min_holdout_sharpe']:
        return _answer('not_confirmed', 'Holdout Sharpe is below the configured minimum.')
    return _answer('confirmed', 'Reserved holdout agreed with validation after costs; evidence for deeper research only.')

