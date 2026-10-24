"""Version-1 chronological evaluation support, without statistical claims.

Dates are strict ISO daily session labels; interval endpoints are inclusive.
The supplied ordered session calendar is authoritative (no holiday inference).
Trial attempts must descend from their protocol via EvidenceStore parents.
Completeness means all such *recorded* trials, including failures, not a claim
that unlogged human searches can be reconstructed. Control kinds below are
reserved for this module; evaluations must not be registered as search trials.

Protocols and selections use EvidenceStore, with two tiny exclusive JSON pointers
under workflow/chronology: one frozen selection per protocol and one first-access
reservation per final interval/input lineage. A new protocol cannot reset access
history; overlapping reserved windows on the same lineage also count as reuse.
Recorded training in other protocols is checked dynamically before final dispatch;
it never changes the intrinsic evidence sealed in a frozen selection. Metadata
views distinguish possible requested access from observed information overlap.
Reservations survive crashes; a reservation without an attempt is
conservatively treated as consumed. No lock expiration, database, or evaluation
execution occurs here. Main must register all trials and gate dispatch through
begin_final_evaluation; direct filesystem/data access is outside this ledger.

Successful trial results should include::

    {"chronology": {"schema_version": 1,
      "evidence_interval": {"start": "2020-01-02", "end": "2020-01-03"},
      "information_end": "2020-01-03"}, ...}

information_end includes the latest label/holding information used. Missing dates
for selected evidence are refused. Missing dates in other successful trials are
an exposure uncertainty, never evidence of an untouched holdout. Labels describe
recorded access only, not out-of-sample performance or point-in-time data quality.
"""
from __future__ import annotations

from datetime import date
import hashlib
import json
import re
import uuid

from qlib_yahoo.workflow_records import (
    EvidenceConflict, EvidenceStore, canonical_json_bytes, read_sealed_record,
    seal_record, _checked, _publish,
)

SCHEMA_VERSION = 1
PHASES = ('development', 'validation', 'final')
_CONTROL = frozenset({'chronology.protocol', 'chronology.selection', 'chronology.final'})
_GROUPING = frozenset({'lab-batch', 'chronology.trial', 'console-lab', 'worker-alpha-batch'})
_SUCCESS = frozenset({'completed', 'reused'})
_ID = re.compile(r'[0-9a-f]{32}\Z')


def _copy(value):
    return json.loads(canonical_json_bytes(value))


def _hash(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _text(value, label):
    if type(value) is not str or not value.strip():
        raise ValueError(label + ' must be a nonempty string')


def _identity(value, pattern=_ID):
    if type(value) is not str or not pattern.fullmatch(value):
        raise ValueError('invalid chronology identity')


def _date(value):
    if type(value) is not str:
        raise ValueError('session dates must be ISO YYYY-MM-DD strings')
    try:
        if date.fromisoformat(value).isoformat() != value:
            raise ValueError('noncanonical date')
    except ValueError as exc:
        raise ValueError('session dates must be ISO YYYY-MM-DD strings') from exc
    return value


def _interval(value, sessions):
    if type(value) is not dict or set(value) != {'start', 'end'}:
        raise ValueError('interval must contain exactly start/end')
    start, end = _date(value['start']), _date(value['end'])
    if start > end:
        raise ValueError('out-of-order interval')
    if start not in sessions or end not in sessions:
        raise ValueError('interval endpoints must be declared sessions')
    return start, end


def _overlap(first, second):
    return first[0] <= second[1] and second[0] <= first[1]


def validate_protocol(protocol: dict):
    """Validate/detach a plan; this does not assert any evaluation occurred.

    Required keys: schema_version=1, sessions (sorted unique ISO dates),
    development/validation/final ({start,end}), input_lineage (nonempty JSON
    object of data/universe/recipe identities), selection_rule (nonempty JSON
    object). Optional metadata is a JSON object. Windows may have gaps.
    """
    protocol = _copy(protocol)
    required = {'schema_version', 'sessions', *PHASES, 'input_lineage', 'selection_rule'}
    if (type(protocol) is not dict or not required <= set(protocol)
            or set(protocol) - required - {'metadata'}
            or type(protocol['schema_version']) is not int
            or protocol['schema_version'] != SCHEMA_VERSION):
        raise ValueError('invalid chronological protocol schema/version')
    sessions = protocol['sessions']
    if type(sessions) is not list or not sessions:
        raise ValueError('sessions must be a nonempty ordered list')
    for session in sessions:
        _date(session)
    if sessions != sorted(set(sessions)):
        raise ValueError('sessions must be strictly ordered and unique')
    calendar = set(sessions)
    windows = [_interval(protocol[phase], calendar) for phase in PHASES]
    if any(left[1] >= right[0] for left, right in zip(windows, windows[1:])):
        raise ValueError('development, validation, final must be chronological and non-overlapping')
    for field in ('input_lineage', 'selection_rule'):
        if type(protocol[field]) is not dict or not protocol[field]:
            raise ValueError(field + ' must be a nonempty JSON object')
    if 'metadata' in protocol and type(protocol['metadata']) is not dict:
        raise ValueError('metadata must be a JSON object')
    return protocol


def split_samples(protocol: dict, samples: list[dict] | None = None):
    """Assign samples and purge actual closed-interval information overlaps.

    Each explicit sample has id, session, information_start, information_end;
    optional feature_available_at must not exceed its session. The information
    interval must contain its session, and all endpoints must be in the supplied
    calendar. Earlier samples are excluded if they intersect a later evaluation
    window OR a later sample's information interval (including its lookback).
    Final samples extending beyond final.end are excluded. No embargo counts.
    Without samples, return session assignments with overlap_check=not_available.
    """
    protocol = validate_protocol(protocol)
    calendar = set(protocol['sessions'])
    windows = {phase: _interval(protocol[phase], calendar) for phase in PHASES}
    assignments = {phase: [] for phase in PHASES}
    if samples is None:
        for phase in PHASES:
            assignments[phase] = [s for s in protocol['sessions']
                                  if windows[phase][0] <= s <= windows[phase][1]]
        return {'schema_version': SCHEMA_VERSION, 'assignments': assignments,
                'excluded': [], 'overlap_check': 'not_available', 'claim': 'split_support_only'}
    samples = _copy(samples)
    if type(samples) is not list:
        raise ValueError('samples must be a JSON list')
    parsed, seen = [], set()
    for sample in samples:
        required = {'id', 'session', 'information_start', 'information_end'}
        if (type(sample) is not dict or not required <= set(sample)
                or set(sample) - required - {'feature_available_at'}):
            raise ValueError('invalid sample interval schema')
        _text(sample['id'], 'sample id')
        if sample['id'] in seen:
            raise ValueError('duplicate sample id')
        seen.add(sample['id'])
        session = _date(sample['session'])
        interval = _interval({'start': sample['information_start'], 'end': sample['information_end']}, calendar)
        if session not in calendar or not interval[0] <= session <= interval[1]:
            raise ValueError('sample information interval must contain its declared session')
        if 'feature_available_at' in sample:
            available = _date(sample['feature_available_at'])
            if available not in calendar or available > session:
                raise ValueError('future/unavailable sample feature information')
        phase = next((p for p in PHASES if windows[p][0] <= session <= windows[p][1]), None)
        if phase is None:
            raise ValueError('sample session is outside all declared windows')
        parsed.append((sample, phase, interval))
    # Every information interval contains a session in its own phase window.
    # Their union WITH that window is therefore one continuous interval. This
    # computes the exact protected union in O(samples), not pairwise O(n**2).
    protected = {phase: list(window) for phase, window in windows.items()}
    for _, phase, interval in parsed:
        protected[phase][0] = min(protected[phase][0], interval[0])
        protected[phase][1] = max(protected[phase][1], interval[1])
    excluded = []
    for sample, phase, interval in parsed:
        later = PHASES[PHASES.index(phase) + 1:]
        overlaps = [p for p in later if _overlap(interval, protected[p])]
        if overlaps:
            excluded.append({'id': sample['id'], 'phase': phase,
                             'reason': 'information_overlap', 'overlaps': overlaps})
        elif phase == 'final' and interval[1] > windows['final'][1]:
            excluded.append({'id': sample['id'], 'phase': phase,
                             'reason': 'information_outside_final', 'overlaps': []})
        else:
            assignments[phase].append(sample['id'])
    return {'schema_version': SCHEMA_VERSION, 'assignments': assignments,
            'excluded': excluded, 'overlap_check': 'information_interval_overlap',
            'claim': 'split_support_only'}


def _reference(attempt):
    outcome = {key: attempt[key] for key in (
        'id', 'schema_version', 'submission_digest', 'status', 'finished_at',
        'result_digest', 'error', 'artifacts',
    )}
    return {key: attempt[key] for key in (
        'id', 'submission_digest', 'status', 'result_digest', 'parents',
    )} | {'outcome_digest': _hash(outcome)}


def _ids(values, label):
    if type(values) is not list or not values:
        raise ValueError(label + ' must be a nonempty list')
    for value in values:
        _identity(value)
    if len(values) != len(set(values)):
        raise ValueError(label + ' contains duplicates')


class ChronologyStore:
    """Versioned support records using an existing EvidenceStore (or data_root).

    freeze_selection returns an ordinary completed EvidenceStore attempt. Its
    request contains the complete frozen population and selected references.
    begin_final_evaluation returns a NEW pending attempt on every permitted call;
    the service/CLI executes it and calls EvidenceStore.finish. Failures and
    cancellations still consume a recorded holdout access. Never dispatch first.
    """

    def __init__(self, evidence):
        self.evidence = evidence if isinstance(evidence, EvidenceStore) else EvidenceStore(evidence)

    def register_protocol(self, protocol: dict, *, parents=None):
        protocol = validate_protocol(protocol)
        attempt = self.evidence.begin('chronology.protocol', protocol, parents=parents)
        self.evidence.finish(attempt['id'], 'completed', result=protocol)
        return self.evidence.get(attempt['id'])

    def get_protocol(self, protocol_id):
        record = self.evidence.get(protocol_id)
        if record['kind'] != 'chronology.protocol' or record['status'] != 'completed' or record['result_digest'] is None:
            raise ValueError('protocol is not a completed immutable declaration')
        protocol = validate_protocol(self.evidence.get_result(record['result_digest']))
        if canonical_json_bytes(protocol) != canonical_json_bytes(record['request']):
            raise ValueError('protocol request/result mismatch')
        return protocol

    def _population(self, protocol_id, *, verify_results=True, records=None):
        # Inspect submission headers first: browsing/freeze must not open final
        # numerical payloads merely to discover the training population.
        all_records = (self.evidence.list_attempts(limit=2**63 - 1, verify_results=False)
                       if records is None else records)
        blocked = {r['id'] for r in all_records if r['kind'] in {'chronology.final', 'chronology.selection'}}
        while True:
            descendants = {r['id'] for r in all_records if any(p in blocked for p in r['parents'])}
            if descendants <= blocked:
                break
            blocked.update(descendants)
        known, trials = {protocol_id}, {}
        while True:
            added = False
            for summary in all_records:
                if (summary['id'] not in known and summary['id'] not in blocked and summary['kind'] not in _CONTROL
                        and any(parent in known for parent in summary['parents'])):
                    known.add(summary['id'])
                    trials[summary['id']] = self.evidence.get(summary['id'], verify_result=verify_results)
                    added = True
            if not added:
                return trials

    def _evidence_dates(self, attempt, protocol):
        payload = self.evidence.get_result(attempt['result_digest'])
        if (attempt['kind'] == 'lab-evaluation' and type(payload) is dict
                and 'chronology' not in payload and payload.get('engine') == 'alphalab-3'):
            # This known engine books close-to-close returns on daily.dates.
            # Earlier warmup features do not shorten the last observed return.
            daily, summary, settings = payload.get('daily'), payload.get('summary'), payload.get('settings')
            if type(daily) is not dict or type(summary) is not dict or type(settings) is not dict:
                raise ValueError('Lab result missing observed daily/settings/summary evidence')
            dates = daily.get('dates')
            if type(dates) is not list or not dates:
                raise ValueError('Lab result missing observed daily dates')
            for day in dates:
                _date(day)
            calendar = protocol['sessions']
            selected = settings.get('evaluation_sessions')
            if selected is not None:
                evaluation = payload.get('evaluation') or {}
                start, end = settings.get('start'), settings.get('end')
                if start not in calendar or end not in calendar or start > end:
                    raise ValueError('filtered Lab evidence missing observed window')
                observed = [day for day in calendar if start <= day <= end]
                expected = {'policy': 'observation-filter-v1', 'observed_start': start,
                            'observed_end': end, 'observed_count': len(observed),
                            'selected_sessions': dates, 'selected_count': len(dates)}
                if (type(selected) is not list or len(selected) < 2 or dates != selected
                        or dates != sorted(set(dates)) or any(day not in observed for day in dates)
                        or any(evaluation.get(key) != value for key, value in expected.items())
                        or summary.get('start') != dates[0] or summary.get('end') != dates[-1]
                        or any(type(daily.get(k)) is not list or len(daily[k]) != len(dates)
                               for k in ('pnl', 'turnover'))):
                    raise ValueError('filtered Lab observed/scored evidence disagrees')
                # Numerical preparation observed the full window even when a
                # metric excludes rows. Do not erase that exposure boundary.
                return {'schema_version': SCHEMA_VERSION,
                        'evidence_interval': {'start': start, 'end': end}, 'information_end': end}
            if (dates != sorted(set(dates)) or any(day not in calendar for day in dates)
                    or dates != [s for s in calendar if dates[0] <= s <= dates[-1]]
                    or summary.get('start') != dates[0] or summary.get('end') != dates[-1]
                    or any(type(daily.get(k)) is not list or len(daily[k]) != len(dates) for k in ('pnl', 'turnover'))
                    or any(settings.get(k) is not None and settings[k] != dates[i]
                           for k, i in (('start', 0), ('end', -1)))):
                raise ValueError('Lab observed evidence dates/settings disagree')
            return {'schema_version': SCHEMA_VERSION,
                    'evidence_interval': {'start': dates[0], 'end': dates[-1]}, 'information_end': dates[-1]}
        if type(payload) is not dict or 'chronology' not in payload:
            return None
        evidence = payload['chronology']
        if (type(evidence) is not dict or set(evidence) != {'schema_version', 'evidence_interval', 'information_end'}
                or type(evidence['schema_version']) is not int or evidence['schema_version'] != SCHEMA_VERSION):
            raise ValueError('invalid selected evidence chronology schema/version')
        _, end = _interval(evidence['evidence_interval'], set(protocol['sessions']))
        information_end = _date(evidence['information_end'])
        if information_end not in protocol['sessions'] or information_end < end:
            raise ValueError('evidence information_end must cover all observed evidence sessions')
        return _copy(evidence)

    def _selection_payload(self, protocol_id, population_ids, selected_ids, decision_reason):
        _ids(population_ids, 'population_ids')
        _ids(selected_ids, 'selected_ids')
        _text(decision_reason, 'decision_reason')
        protocol = self.get_protocol(protocol_id)
        population = self._population(protocol_id)
        if set(population_ids) != set(population):
            raise ValueError('incomplete/mismatched recorded attempt population')
        if not set(selected_ids) <= set(population_ids):
            raise ValueError('selected attempt is not in complete population')
        if any(attempt['status'] == 'pending' for attempt in population.values()):
            raise ValueError('population includes pending trials; outcomes cannot yet be frozen')
        selected, exposure = [], []
        source_id = protocol.get('metadata', {}).get('chronology_workflow', {}).get('source_attempt_id')
        if source_id is not None:
            source = self.evidence.get(source_id)
            dates = self._evidence_dates(source, protocol) if source['result_digest'] else None
            if dates is None or dates['information_end'] >= protocol['final']['start']:
                exposure.append({'attempt_id': source_id, 'reason': 'source_evidence_reaches_final_or_unknown'})
        for identity in sorted(population):
            attempt = population[identity]
            chosen = identity in selected_ids
            if attempt['kind'] in _GROUPING:
                if chosen:
                    raise ValueError('select numerical trial results, not orchestration/grouping records')
                continue  # Retained in population; not a fabricated numerical evaluation.
            if chosen and (attempt['status'] not in _SUCCESS or attempt['result_digest'] is None):
                raise ValueError('selected attempt must have a successful immutable result; legacy/missing results refused')
            if attempt['status'] not in _SUCCESS:
                # Failed/rejected trials with partial results may also expose data.
                dates = self._evidence_dates(attempt, protocol) if attempt['result_digest'] else None
                bounded_lab_failure = False
                if attempt['kind'] == 'lab-evaluation' and dates is None:
                    # Known engine errors/preflight have no observed result to
                    # invent dates for. A registered pre-final window is only
                    # an upper exposure bound, never selectable evidence.
                    settings = attempt['request'].get('settings', {})
                    if type(settings) is dict and settings.get('end') is not None:
                        bounded_lab_failure = _date(settings['end']) < protocol['final']['start']
                if attempt['result_digest'] is not None and dates is None and not bounded_lab_failure:
                    exposure.append({'attempt_id': identity, 'reason': 'partial_trial_dates_unknown'})
            else:
                dates = self._evidence_dates(attempt, protocol) if attempt['result_digest'] else None
                if dates is None:
                    if chosen:
                        raise ValueError('selected evidence dates are required before final evaluation')
                    exposure.append({'attempt_id': identity, 'reason': 'successful_trial_dates_unknown'})
            if dates is not None and dates['information_end'] >= protocol['final']['start']:
                exposure.append({'attempt_id': identity, 'reason': 'evidence_reaches_final_interval'})
            if chosen:
                selected.append({**_reference(attempt), 'chronology': dates})
        ordered_selected = sorted(selected, key=lambda r: selected_ids.index(r['id']))
        return {'record_type': 'chronology.selection', 'schema_version': SCHEMA_VERSION,
                'protocol_id': protocol_id,
                'protocol_result_digest': self.evidence.get(protocol_id)['result_digest'],
                'population': [_reference(population[p]) for p in sorted(population)],
                'selected': ordered_selected, 'decision_reason': decision_reason,
                'exposure_reasons': exposure, 'claim': 'frozen_selection_support_only'}

    def _selection_path(self, protocol_id):
        _identity(protocol_id)
        return _checked(self.evidence.root / 'chronology' / 'selections' / (protocol_id + '.json'))

    def _selection_attempt(self, protocol_id):
        pointer = read_sealed_record(self._selection_path(protocol_id))
        if (type(pointer) is not dict or set(pointer) != {
                'schema_version', 'protocol_id', 'selection_id', 'submission_digest'}
                or type(pointer['schema_version']) is not int or pointer['schema_version'] != SCHEMA_VERSION
                or pointer['protocol_id'] != protocol_id):
            raise ValueError('invalid frozen selection pointer')
        _identity(pointer['selection_id'])
        record = self.evidence.get(pointer['selection_id'])
        if record['kind'] != 'chronology.selection' or pointer['submission_digest'] != record['submission_digest']:
            raise ValueError('frozen selection pointer does not bind its submission')
        return record

    def freeze_selection(self, protocol_id, *, population_ids, selected_ids, decision_reason):
        payload = self._selection_payload(protocol_id, population_ids, selected_ids, decision_reason)
        path = self._selection_path(protocol_id)
        try:
            existing = self._selection_attempt(protocol_id)
        except FileNotFoundError:
            # Only a missing pointer permits a new freeze; missing referenced
            # evidence must not silently create a replacement selection.
            if path.exists():
                raise
        else:
            if canonical_json_bytes(existing['request']) != canonical_json_bytes(payload):
                raise EvidenceConflict('selection already frozen; changed selections require a new protocol')
            self.evidence.finish(existing['id'], 'completed', result=payload)
            return self.get_selection(protocol_id)
        attempt = self.evidence.begin('chronology.selection', payload,
                                      parents=[protocol_id, *sorted(population_ids)])
        pointer = {'schema_version': SCHEMA_VERSION, 'protocol_id': protocol_id,
                   'selection_id': attempt['id'],
                   'submission_digest': self.evidence.get(attempt['id'])['submission_digest']}
        try:
            _publish(path, canonical_json_bytes(seal_record(pointer)))
        except FileExistsError:
            self.evidence.finish(attempt['id'], 'rejected', error='another selection freeze won exclusive publication')
            return self.freeze_selection(protocol_id, population_ids=population_ids,
                                         selected_ids=selected_ids, decision_reason=decision_reason)
        # A newly registered trial invalidates this snapshot; never rewrite it.
        if canonical_json_bytes(self._selection_payload(protocol_id, population_ids, selected_ids, decision_reason)) != canonical_json_bytes(payload):
            raise ValueError('population changed during selection freeze')
        self.evidence.finish(attempt['id'], 'completed', result=payload)
        return self.get_selection(protocol_id)

    def get_selection(self, protocol_id, *, verify_results=True):
        record = self._selection_attempt(protocol_id)
        if record['status'] != 'completed' or record['result_digest'] is None:
            raise ValueError('selection is not frozen/completed')
        payload = self.evidence.get_result(record['result_digest'])
        if canonical_json_bytes(payload) != canonical_json_bytes(record['request']):
            raise ValueError('selection request/result mismatch')
        fields = {'record_type', 'schema_version', 'protocol_id', 'protocol_result_digest',
                  'population', 'selected', 'decision_reason', 'exposure_reasons', 'claim'}
        if (type(payload) is not dict or set(payload) != fields
                or type(payload['population']) is not list or type(payload['selected']) is not list):
            raise ValueError('invalid frozen selection schema')
        reference_fields = {'id', 'submission_digest', 'outcome_digest', 'status', 'result_digest', 'parents'}
        for key in ('population', 'selected'):
            for reference in payload[key]:
                expected_fields = reference_fields | ({'chronology'} if key == 'selected' else set())
                if type(reference) is not dict or set(reference) != expected_fields:
                    raise ValueError('invalid frozen selection reference schema')
                _identity(reference['id'])
        if verify_results:
            # Intrinsic evidence only: external activity is dynamic and must
            # never alter an already sealed selection or its retry identity.
            expected = self._selection_payload(protocol_id,
                [p['id'] for p in payload['population']], [p['id'] for p in payload['selected']], payload['decision_reason'])
            matches = canonical_json_bytes(expected) == canonical_json_bytes(payload)
        else:
            # A view verifies stable outcome references without hydrating daily
            # numerical results. Opening/executing retains full verification.
            self.get_protocol(protocol_id)
            population = self._population(protocol_id, verify_results=False)
            matches = (payload['protocol_id'] == protocol_id
                       and payload['protocol_result_digest'] == self.evidence.get(protocol_id)['result_digest']
                       and payload['population'] == [_reference(population[p]) for p in sorted(population)]
                       and all(r['id'] in population and
                               {k: v for k, v in r.items() if k != 'chronology'} == _reference(population[r['id']])
                               for r in payload['selected']))
        if not matches:
            raise ValueError('frozen selection content/outcome references changed')
        return record

    @staticmethod
    def _price_lineage(protocol):
        """Compare price identity, not settings, recipes, paths or source IDs.

        Canonical workflow plans bind a price tree; older low-level fixtures
        use prices_sha256. Unknown lineage schemas retain exact comparison.
        """
        lineage = protocol['input_lineage']
        for key in ('price_tree_sha256', 'prices_sha256'):
            if key in lineage:
                return {key: lineage[key]}
        return lineage

    def _external_exposure(self, protocol_id, *, metadata_only=False):
        """Current cross-protocol TRAINING exposure, never final payloads.

        Observed dates are authoritative at dispatch. A metadata view uses
        requested/declared bounds only and labels them possible, not observed
        contamination. Pending/failed/unknown attempts are conservative; a
        declaration alone is not a numerical observation. No external reasons
        are inserted retrospectively into a frozen selection.
        """
        protocol = self.get_protocol(protocol_id)
        window = (protocol['final']['start'], protocol['final']['end'])
        rows = self.evidence.list_attempts(limit=2**63 - 1, verify_results=False)
        reasons = []
        for row in sorted(rows, key=lambda r: r['id']):
            if row['kind'] != 'chronology.protocol' or row['id'] == protocol_id or row['status'] != 'completed':
                continue
            other = self.get_protocol(row['id'])
            if self._price_lineage(other) != self._price_lineage(protocol):
                continue
            trials = self._population(row['id'], verify_results=False, records=rows)
            source_id = other.get('metadata', {}).get('chronology_workflow', {}).get('source_attempt_id')
            if source_id is not None:
                # Sources are numerical evidence even when a new protocol has
                # not yet submitted training. Final descendants are never read.
                blocked = {r['id'] for r in rows if r['kind'] in {'chronology.final', 'chronology.selection'}}
                while True:
                    more = {r['id'] for r in rows if any(p in blocked for p in r['parents'])}
                    if more <= blocked:
                        break
                    blocked.update(more)
                if source_id in blocked:
                    reasons.append({'protocol_id': row['id'], 'attempt_id': source_id,
                                    'reason': 'external_source_final_ancestry_unknown', 'certainty': 'possible'})
                else:
                    trials[source_id] = self.evidence.get(source_id, verify_result=False)
            for identity, attempt in sorted(trials.items()):
                # Group records do not add another observation when their
                # numerical children already bound it. A pending/failed empty
                # orchestration can still have begun preparation/access.
                if attempt['kind'] in _GROUPING and any(identity in t['parents'] for t in trials.values()):
                    continue
                dates = None
                if not metadata_only and attempt['result_digest'] and attempt['kind'] not in _GROUPING:
                    try:
                        dates = self._evidence_dates(attempt, other)
                    except ValueError:
                        pass  # Invalid/missing dates are uncertainty, not safety.
                certainty, basis = 'possible', 'unknown'
                interval = None
                if dates is not None and _overlap(window, (dates['evidence_interval']['start'], dates['information_end'])):
                    interval = (dates['evidence_interval']['start'], dates['information_end'])
                    certainty, basis = 'observed', 'recorded_information_interval'
                elif dates is not None and attempt['status'] in _SUCCESS:
                    continue
                else:
                    request = attempt['request']
                    settings = request.get('settings', {})
                    if type(settings) is dict and settings.get('start') is not None and settings.get('end') is not None:
                        try:
                            start, end = _date(settings['start']), _date(settings['end'])
                            if start <= end:
                                interval, basis = (start, end), 'requested_window'
                        except ValueError:
                            pass
                    if interval is None and attempt['kind'] == 'chronology.trial' and request.get('phase') in PHASES[:2]:
                        phase = other[request['phase']]
                        interval, basis = (phase['start'], phase['end']), 'declared_phase_window'
                    # A successful result with unknown observation dates has no
                    # proven bound. Do not infer its actual horizon from settings.
                    if attempt['status'] in _SUCCESS and attempt['kind'] not in _GROUPING:
                        interval = None
                        basis = 'observation_dates_unverified' if metadata_only else 'unknown'
                if interval is None or _overlap(window, interval):
                    reason = {'protocol_id': row['id'], 'attempt_id': identity,
                              'reason': ('external_training_information_overlap' if certainty == 'observed'
                                         else 'external_training_exposure_possible'),
                              'certainty': certainty, 'basis': basis, 'status': attempt['status']}
                    if interval is not None:
                        reason['information_interval'] = {'start': interval[0], 'end': interval[1]}
                    reasons.append(reason)
        return reasons

    def _read_holdout(self, path):
        pointer = read_sealed_record(path)
        if (type(pointer) is not dict or set(pointer) != {
                'schema_version', 'holdout_key', 'reservation', 'final', 'input_lineage'}
                or type(pointer['schema_version']) is not int or pointer['schema_version'] != SCHEMA_VERSION
                or type(pointer['input_lineage']) is not dict or not pointer['input_lineage']):
            raise ValueError('invalid holdout reservation')
        _identity(pointer['reservation'])
        window = pointer['final']
        if type(window) is not dict or set(window) != {'start', 'end'}:
            raise ValueError('invalid holdout interval')
        if _date(window['start']) > _date(window['end']):
            raise ValueError('out-of-order holdout interval')
        key = _hash({'final': window, 'input_lineage': pointer['input_lineage']})
        if pointer['holdout_key'] != key or path.name != key + '.json':
            raise ValueError('holdout reservation lineage/key mismatch')
        return pointer

    def _reserve_holdout(self, protocol):
        key = _hash({'final': protocol['final'], 'input_lineage': protocol['input_lineage']})
        path = _checked(self.evidence.root / 'chronology' / 'holdouts' / (key + '.json'))
        reservation = uuid.uuid4().hex
        pointer = {'schema_version': SCHEMA_VERSION, 'holdout_key': key, 'reservation': reservation,
                   'final': protocol['final'], 'input_lineage': protocol['input_lineage']}
        try:
            _publish(path, canonical_json_bytes(seal_record(pointer)))
            first = True
        except FileExistsError:
            self._read_holdout(path)
            first = False
        # Publication precedes discovery, so two overlapping concurrent windows
        # cannot both miss each other's reservation. Both may conservatively be
        # labeled reused. Disjoint windows remain eligible for first access.
        window = (protocol['final']['start'], protocol['final']['end'])
        for other_path in sorted(_checked(path.parent).iterdir()):
            if other_path.name.startswith('.pending-'):
                continue
            other = self._read_holdout(other_path)
            if (other_path != path and canonical_json_bytes(other['input_lineage']) == canonical_json_bytes(protocol['input_lineage'])
                    and _overlap(window, (other['final']['start'], other['final']['end']))):
                first = False
        return key, reservation, first

    def begin_final_evaluation(self, protocol_id, *, selection_id, request=None, allow_exposed=False, parents=None):
        """Reserve access BEFORE execution; return a pending EvidenceStore attempt.

        request is integration-owned JSON. allow_exposed=True permits known or
        uncertain prior exposure, explicitly labeling it already_exposed_holdout.
        Policy refusals raise ValueError before access; they do not dispatch data.
        A first reservation is only first_recorded_access, never 'untouched'.
        """
        request = {} if request is None else _copy(request)
        if type(request) is not dict or type(allow_exposed) is not bool:
            raise ValueError('request must be a JSON object; allow_exposed must be boolean')
        selection = self.get_selection(protocol_id)
        _identity(selection_id)
        if selection_id != selection['id']:
            raise ValueError('final evaluation requires the frozen selection identity')
        payload = selection['request']
        external = self._external_exposure(protocol_id)
        current_exposure = [*payload['exposure_reasons'], *external]
        exposed = bool(current_exposure)
        if exposed and not allow_exposed:
            raise ValueError('already exposed or unknown holdout: final access refused')
        protocol = self.get_protocol(protocol_id)
        key, reservation, first = self._reserve_holdout(protocol)
        label = 'already_exposed_holdout' if exposed else ('first_recorded_access' if first else 'reused_holdout')
        access = {'record_type': 'chronology.final', 'schema_version': SCHEMA_VERSION,
                  'protocol_id': protocol_id, 'protocol_result_digest': payload['protocol_result_digest'],
                  'selection_id': selection_id, 'selection_result_digest': selection['result_digest'],
                  'final': protocol['final'], 'input_lineage': protocol['input_lineage'],
                  'holdout_key': key, 'reservation': reservation, 'access_label': label,
                  'reused_holdout': exposed or not first, 'exposure_reasons': current_exposure,
                  'frozen_exposure_reasons': payload['exposure_reasons'],
                  'external_exposure_reasons': external,
                  'evaluation_request': request, 'claim': 'evaluation_not_yet_performed'}
        attempt = self.evidence.begin('chronology.final', access,
            parents=list(dict.fromkeys([protocol_id, selection_id, *(parents or [])])))
        # Recheck after registration; a concurrent population change must never
        # leave an apparently dispatchable final request returned to the caller.
        try:
            self.get_selection(protocol_id)
            if canonical_json_bytes(self._external_exposure(protocol_id)) != canonical_json_bytes(external):
                raise ValueError('external exposure changed before final dispatch; request final again')
        except (ValueError, FileNotFoundError):
            self.evidence.finish(attempt['id'], 'rejected', error='population or external exposure changed before final dispatch')
            raise
        return self.evidence.get(attempt['id'])

    def compact_view(self, protocol_id):
        """Read-only declaration/selection/access summary, with no metric claim."""
        protocol = self.get_protocol(protocol_id)
        population = self._population(protocol_id, verify_results=False)
        path = self._selection_path(protocol_id)
        selection = self.get_selection(protocol_id, verify_results=False) if path.exists() else None
        frozen_exposure = [] if selection is None else selection['request']['exposure_reasons']
        external = self._external_exposure(protocol_id, metadata_only=True)
        accesses = [self.evidence.get(row['id'], verify_result=False) for row in self.evidence.list_attempts(limit=2**63 - 1, verify_results=False)
                    if row['kind'] == 'chronology.final' and protocol_id in row['parents']]
        return {'schema_version': SCHEMA_VERSION, 'protocol_id': protocol_id,
                'intervals': {phase: protocol[phase] for phase in PHASES},
                'population': [_reference(population[p]) for p in sorted(population)],
                'selection_id': None if selection is None else selection['id'],
                'selected': [] if selection is None else selection['request']['selected'],
                'frozen_exposure_reasons': frozen_exposure,
                'external_exposure_reasons': external,
                'current_exposure_reasons': [*frozen_exposure, *external],
                'current_exposure_basis': 'metadata_only_conservative_bounds_not_observed_contamination',
                'accesses': [{'id': r['id'], 'status': r['status'],
                              'access_label': r['request']['access_label'],
                              'reused_holdout': r['request']['reused_holdout']} for r in accesses],
                'claim': 'chronological_support_only'}
