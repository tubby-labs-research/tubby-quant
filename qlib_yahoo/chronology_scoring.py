"""Daily observation coverage and scoring instructions, not a trading mask.

Signal state is calculated on the full continuous window. Explicit sample
intervals must cover every scored-window session before exclusions can be
applied. Several samples on one day conservatively exclude that day if any
sample is excluded. No samples means calendar-only evaluation, not purging.
"""
from .chronology import PHASES, split_samples


def scoring_split(protocol):
    samples = protocol.get('metadata', {}).get('samples')
    split = split_samples(protocol, samples)
    explicit = samples is not None
    by_id = {row['id']: row['session'] for row in samples or []}
    checked = set(by_id.values())
    excluded = {by_id[row['id']] for row in split['excluded']}
    phases = {}
    for phase in PHASES:
        window = protocol[phase]
        days = [day for day in protocol['sessions'] if window['start'] <= day <= window['end']]
        missing = [day for day in days if day not in checked] if explicit else days
        kept = [day for day in days if day not in excluded]
        # Complete coverage is required across all windows, because intervals
        # belonging to later phases are part of an earlier phase's purge rule.
        phases[phase] = {
            'observed_count': len(days), 'checked_count': len(days) - len(missing),
            'unchecked_sessions': missing, 'excluded_sessions': [day for day in days if day in excluded],
            'evaluation_sessions': kept if explicit else None,
        }
    complete = explicit and all(not row['unchecked_sessions'] for row in phases.values())
    for phase, row in phases.items():
        reason = ('explicit sample intervals require complete session coverage across all phases'
                  if explicit and not complete else
                  'at least two scored observations must remain after exclusions'
                  if explicit and len(row['evaluation_sessions']) < 2 else None)
        row.update(ready=reason is None, refusal=reason)
    split['coverage'] = {'status': 'complete' if complete else 'partial' if explicit else 'not_supplied',
                         'unit': 'daily portfolio return observations', 'phases': phases}
    split['execution_policy'] = 'observation-filter-v1' if explicit else 'calendar-window-v1'
    split['claim'] = ('checked_intervals_for_all_declared_sessions' if complete else
                      'checked_supplied_intervals_only' if explicit else 'split_support_only')
    return split


def phase_scoring(protocol, phase):
    split = scoring_split(protocol)
    row = split['coverage']['phases'][phase]
    if not row['ready']:
        raise ValueError(row['refusal'])
    return row['evaluation_sessions']
