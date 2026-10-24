"""JSON service/CLI boundary for supplied chronological Lab evaluations.

No campaign, automatic search, Qlib dispatch, or statistical acceptance rule.
All six APIs take ``(data_root, params: dict)``. Mutating operations reuse the
existing EvidenceStore and run_batch; integration owns scheduling and UI.

preview/declare: source_attempt_id, development/validation/final {start,end},
selection_rule (JSON object), optional samples (split_samples schema).
trial: protocol_id, phase (development|validation), expressions (nonempty list),
optional workers (default 1), force (default False).
freeze: protocol_id, selected_ids, decision_reason, optional population_ids
(otherwise the COMPLETE recorded population is discovered, never a UI page).
final: protocol_id, selection_id, optional workers/force/allow_exposed.
view: protocol_id. Unknown keys, nonfinite values and nonplain JSON are refused.

Protocol source/configuration/calendar/recipes are reconstructed from a sealed
saved candidate, not supplied lineage. Holdout identity uses underlying price
bytes/calendar/symbols: changing settings, recipes, candidate IDs, paths or plan
text cannot manufacture a fresh price-return holdout. Exact execution bindings
remain separately frozen. Source bytes are verified only on explicit execution,
AFTER reserving final access. Preview/view never load a panel or read final
numeric results. Classification/SEC inputs are restricted to recorded recipes.

Explicit sample intervals must cover every session across all three windows.
Exclusions filter scored daily observations after continuous signal calculation;
they do not remove warmup/state or describe an executable trading policy.
cancel (optional callable) is polled before data preparation and on batch progress
events; KeyboardInterrupt is recorded and propagated. Hard kill leaves pending
ownership for ordinary EvidenceStore recovery. Callers must gate all trial
submission/dispatch through this boundary to enforce the frozen population.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

from .chronology import ChronologyStore, _copy, _hash, validate_protocol
from .chronology_scoring import scoring_split, phase_scoring
from .workflow_records import EvidenceStore, _checked
from .workflow_submission import parent_attempts

SCHEMA_VERSION = 1


def _params(params, required, optional=()):
    value = _copy(params)
    if type(value) is not dict or not set(required) <= set(value) or set(value) - set(required) - set(optional):
        raise ValueError('invalid chronology workflow parameters; source/settings/lineage overrides are forbidden')
    return value


def _source_binding(candidate):
    result, attempt = candidate['result'], candidate['attempt']
    identity = result['dataset']
    saved_sec = identity.get('sec')
    sec = None
    if saved_sec:
        sec = {k: saved_sec[k] for k in ('catalog', 'bindings')}
        sec['mode'] = 'dictionary' if 'registry' in saved_sec else 'legacy'
        if 'registry' in saved_sec:
            sec['registry'] = saved_sec['registry']
    return {'schema_version': SCHEMA_VERSION, 'source_attempt_id': attempt['id'],
            'source_result_digest': attempt['result_digest'], 'source_submission_digest': attempt['submission_digest'],
            'source_identity': identity, 'settings': result['settings'],
            'implementation': result['implementation'], 'sec': sec,
            'group_overlay': (identity.get('classification') or {}).get('path'),
            'qualification': {'status': 'exploratory_only', 'economic_pit': 'not_established',
                'scope': 'recorded_snapshot_and_sampled_fixture_support_only',
                'source_bytes': 'pending_until_execution',
                'limitations': ['sampled fixtures and byte seals do not certify complete historical coverage',
                    'identity, delistings, membership, share adjustments and input availability remain source-specific',
                    'Lab fixed-book returns omit borrow and realistic cash/share execution']}}


def _lineage(binding):
    identity = binding['source_identity']
    # Deliberately broader than the exact recipe binding. Selection/tuning on
    # the same underlying returns cannot reset holdout history by editing config.
    return {'schema_version': SCHEMA_VERSION,
            'price_tree_sha256': identity['qlib_binary_tree_sha256'],
            'calendar_sha256': _hash(identity['dates']), 'symbols_sha256': _hash(identity['symbols'])}


def _source_scope(data_root, binding):
    """Apply the console's storage scope equally to direct Python/CLI workers.

    Saved evidence may contain absolute historical paths. It is not authority
    to execute against another root, or to follow a redirected source directory.
    Only path metadata is checked here, never source bytes or dense panels.
    """
    root = EvidenceStore(data_root).root.parent
    values = [(binding['source_identity']['dataset'], 'datasets')]
    if binding['group_overlay'] is not None:
        values.append((binding['group_overlay'], 'classifications'))
    if binding['sec'] is not None:
        values.append((binding['sec']['catalog'], 'sec'))
    for value, area in values:
        if type(value) is not str:
            raise ValueError('recorded chronology source paths must be strings')
        path = Path(value)
        if not path.is_absolute() or '..' in path.parts:
            raise ValueError('recorded chronology source path must be absolute without traversal')
        _checked(path)
        allowed = _checked(root / area)
        if path == allowed or not path.is_relative_to(allowed):
            raise ValueError('recorded chronology source is outside this data root/' + area)


def _window_checks(protocol, binding):
    calendar = protocol['sessions']
    warmup = binding['settings']['warmup']
    for phase in ('development', 'validation', 'final'):
        start, end = (calendar.index(protocol[phase][key]) for key in ('start', 'end'))
        if start < warmup or start < 1:
            raise ValueError(phase + ' lacks the exact preceding warmup sessions')
        if end - start < 20:
            raise ValueError(phase + ' needs at least 21 sessions for the existing Lab engine')


def _read_source(data_root, attempt_id):
    from .candidate_workflow import read_candidate
    store = EvidenceStore(data_root)
    todo, seen = [attempt_id], set()
    while todo:
        identity = todo.pop()
        if identity in seen:
            continue
        seen.add(identity)
        header = store.get(identity, verify_result=False)
        if header['kind'] == 'chronology.final':
            raise ValueError('final evaluation evidence cannot be reintroduced as a protocol source through preview')
        todo.extend(header['parents'])
    return read_candidate(data_root, attempt_id)


def preview_chronology(data_root, params: dict):
    """Read saved source evidence and plan metadata only; do not reserve/access final data."""
    params = _params(params, {'source_attempt_id', 'development', 'validation', 'final', 'selection_rule'}, {'samples'})
    candidate = _read_source(data_root, params['source_attempt_id'])
    binding = _source_binding(candidate)
    _source_scope(data_root, binding)
    plan = {key: params[key] for key in ('development', 'validation', 'final', 'selection_rule')}
    plan.update(schema_version=SCHEMA_VERSION, sessions=binding['source_identity']['dates'],
                input_lineage=_lineage(binding), metadata={'chronology_workflow': binding})
    if 'samples' in params:
        plan['metadata']['samples'] = params['samples']
    plan = validate_protocol(plan)
    _window_checks(plan, binding)
    split = scoring_split(plan)
    return {'schema_version': SCHEMA_VERSION, 'protocol': plan, 'sample_split': split,
            'qualification': binding['qualification'], 'claim': 'declaration_preview_only_no_final_evaluation'}


def declare_chronology(data_root, params: dict):
    """Persist a protocol with a parent link to the exact saved source candidate."""
    preview = preview_chronology(data_root, params)
    plan = preview['protocol']
    source_id = plan['metadata']['chronology_workflow']['source_attempt_id']
    parents = list(dict.fromkeys([source_id, *(parent_attempts() or [])]))
    attempt = ChronologyStore(data_root).register_protocol(plan, parents=parents)
    return {**preview, 'protocol_id': attempt['id'], 'protocol_result_digest': attempt['result_digest'],
            'claim': 'declared_protocol_only_no_evaluation'}


def _bound(data_root, protocol_id):
    chronology = ChronologyStore(data_root)
    plan = chronology.get_protocol(protocol_id)
    binding = plan.get('metadata', {}).get('chronology_workflow')
    if type(binding) is not dict or binding.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('protocol has no canonical chronology workflow source binding')
    _source_scope(data_root, binding)
    candidate = _read_source(data_root, binding['source_attempt_id'])
    if _source_binding(candidate) != binding or _lineage(binding) != plan['input_lineage'] or plan['sessions'] != binding['source_identity']['dates']:
        raise ValueError('chronology protocol source/settings/calendar/lineage binding changed')
    _window_checks(plan, binding)
    return chronology, plan, binding, candidate


def _execution(params, *, cancel, progress):
    workers, force = params.get('workers', 1), params.get('force', False)
    if type(workers) is not int or not 1 <= workers <= 64 or type(force) is not bool:
        raise ValueError('workers must be 1..64 and force must be boolean')
    if cancel is not None and not callable(cancel) or progress is not None and not callable(progress):
        raise ValueError('cancel/progress must be callable')
    return workers, force


def _poll(cancel):
    if cancel is not None and cancel():
        raise KeyboardInterrupt('chronological evaluation cancelled')


def _phase_settings(plan, binding, phase):
    from .candidate_workflow import _settings
    return replace(_settings(binding['settings']), **plan[phase],
                   evaluation_sessions=phase_scoring(plan, phase))


def _assert_phase_settings(result, plan, binding):
    """Only phase-derived windows/masks may differ from the frozen source."""
    from .candidate_workflow import _settings
    settings = _settings(result['settings'])
    phase = next((name for name in ('development', 'validation')
                  if {'start': settings.start, 'end': settings.end} == plan[name]), None)
    if phase is None or asdict(settings) != asdict(_phase_settings(plan, binding, phase)):
        raise ValueError('selected candidate settings differ from the frozen protocol source or phase scoring')
    return settings


def _expression_reason(expression, binding, settings):
    if type(expression) is not str or not expression.strip():
        return None  # Existing run_batch retains its own invalid-expression child.
    from .sec_resolution import field_dependencies
    from .alphalab.batch import _uses_classifications
    from .alphalab.fastexpr import ExpressionError
    try:
        names = {n for n in field_dependencies([expression], dialect='lab') if n.startswith('sec_')}
    except (ValueError, ExpressionError):
        return None  # Ordinary syntax errors belong to run_batch preflight.
    if names - set((binding['sec'] or {}).get('bindings', {})):
        return 'expression requires SEC fields outside the frozen source recipes'
    if _uses_classifications(expression, settings) and not binding['group_overlay']:
        return 'expression requires classification inputs outside the frozen source recipes'
    return None


def _assert_result(result, binding, settings):
    from .candidate_workflow import _combined, _price_binding
    from .alphalab.batch import _uses_classifications
    from .sec_resolution import field_dependencies
    if result.get('settings') != asdict(settings) or result.get('implementation') != binding['implementation']:
        raise ValueError('dispatched Lab result changed frozen settings/implementation')
    identity = binding['source_identity']
    price = _price_binding(identity)
    group = identity.get('classification') if _uses_classifications(result['expression'], settings) else None
    names = {n for n in field_dependencies([result['expression']], dialect='lab') if n.startswith('sec_')}
    sec = None
    if names:
        from .sec_features import feature_identity
        saved = identity['sec']
        recipes = {name: saved['bindings'][name] for name in sorted(names)}
        sec = {'catalog': saved['catalog'], 'bindings': recipes,
               'source_identity': feature_identity(Path(saved['catalog']), recipes)}
        if 'registry' in saved:
            sec['registry'] = saved['registry']
    if result.get('dataset') != _combined(price, group, sec):
        raise ValueError('dispatched Lab result changed frozen source recipes')


def _descendants(store, parent):
    rows = store.list_attempts(limit=2**63 - 1, verify_results=False)
    found = {parent}
    while True:
        more = {r['id'] for r in rows if any(p in found for p in r['parents'])}
        if more <= found:
            break
        found.update(more)
    return [store.get(r['id']) for r in rows if r['id'] in found and r['id'] != parent]


def _receipt(store, attempt_id, phase, qualification):
    children = _descendants(store, attempt_id)
    return {'schema_version': SCHEMA_VERSION, 'phase': phase,
            'attempt_ids': [r['id'] for r in children if r['kind'] == 'lab-evaluation'],
            'batch_attempt_ids': [r['id'] for r in children if r['kind'] == 'lab-batch'],
            'children': [{k: r[k] for k in ('id', 'kind', 'status', 'result_digest', 'parents')} for r in children],
            'qualification': qualification, 'claim': 'sampled_exploratory_Lab_evidence_only'}


def _dispatch(data_root, chronology, plan, binding, candidate, attempt, phase, expressions,
              params, *, cancel, progress):
    from .candidate_workflow import _verify_sources
    from .alphalab.batch import run_batch
    store = chronology.evidence
    qualification = _copy(binding['qualification'])
    try:
        workers, force = _execution(params, cancel=cancel, progress=progress)
        _poll(cancel)
        split = scoring_split(plan)
        settings = _phase_settings(plan, binding, phase)
        if phase == 'final':
            # Low-level stores remain independently usable. Do not assume that
            # every caller froze its selection through this service boundary.
            from .candidate_workflow import _settings
            frozen = chronology.get_selection(attempt['request']['protocol_id'])
            for ref in frozen['request']['selected']:
                selected = store.get_result(ref['result_digest'])
                chosen_settings = _assert_phase_settings(selected, plan, binding)
                _assert_result(selected, binding, chosen_settings)
        _verify_sources(candidate['result'])  # Final reservation already exists.
        qualification['source_bytes'] = 'verified_for_this_execution'

        def report(message):
            _poll(cancel)
            if progress:
                progress(message)

        admitted = []
        for expression in expressions:
            reason = _expression_reason(expression, binding, settings)
            if reason:
                rejected = store.begin('lab-evaluation', {
                    'dataset': binding['source_identity']['dataset'], 'expression': expression,
                    'settings': asdict(settings), 'sec': binding['sec'], 'group_overlay': binding['group_overlay'],
                    'force': force}, parents=[attempt['id']], source={'implementation': binding['implementation']})
                store.finish(rejected['id'], 'rejected', error=reason, result={
                    'expression': expression, 'settings': asdict(settings), 'preflight': True, 'error': reason})
            else:
                admitted.append(expression)
        _poll(cancel)
        if admitted:
            # Conditional observation scores must not appear in the ordinary
            # Lab library as continuous-window alpha results. Keep their cache
            # separate while retaining immutable attempts in the shared ledger.
            scoring_library = (_checked(store.root / 'chronology-cache.sqlite')
                               if settings.evaluation_sessions is not None else None)
            results = run_batch(Path(binding['source_identity']['dataset']), admitted, settings,
                data_root=Path(data_root), workers=workers, worker_ceiling=4, force=force,
                library=scoring_library,
                sec=binding['sec'], group_overlay=binding['group_overlay'], progress=report, parents=[attempt['id']])
            for result in results:
                if 'error' not in result:
                    _assert_result(result, binding, settings)
                    record = store.get(result['attempt_id'])
                    chronology._evidence_dates(record, plan)  # Validate observed dates, not a requested horizon guess.
        _verify_sources(candidate['result'])
        _poll(cancel)
        receipt = _receipt(store, attempt['id'], phase, qualification)
        receipt['sample_split'] = split
        numeric = [c for c in receipt['children'] if c['kind'] == 'lab-evaluation']
        successful = [c for c in numeric if c['status'] in {'completed', 'reused'}]
        status = ('reused' if successful and len(successful) == len(numeric) and all(c['status'] == 'reused' for c in numeric)
                  else 'completed' if successful and (phase != 'final' or len(successful) == len(numeric))
                  else 'rejected' if numeric and all(c['status'] == 'rejected' for c in numeric) else 'failed')
        store.finish(attempt['id'], status, result=receipt,
                     error=None if status in {'completed', 'reused'} else 'one or more numerical trials did not complete')
        return {'schema_version': SCHEMA_VERSION, 'attempt_id': attempt['id'], 'status': status,
                'protocol_id': attempt['request']['protocol_id'], **receipt,
                **({'access_label': attempt['request']['access_label'], 'reused_holdout': attempt['request']['reused_holdout']}
                   if phase == 'final' else {})}
    except BaseException as error:
        if store.get(attempt['id'])['status'] == 'pending':
            status = 'cancelled' if isinstance(error, KeyboardInterrupt) else 'rejected' if isinstance(error, (ValueError, TypeError)) else 'failed'
            receipt = _receipt(store, attempt['id'], phase, qualification)
            store.finish(attempt['id'], status, result=receipt,
                         error={'type': type(error).__name__, 'message': str(error)})
        raise


def submit_chronology_trial(data_root, params: dict, *, cancel=None, progress=None):
    """Submit supplied expressions on development/validation; retain all outcomes."""
    params = _params(params, {'protocol_id', 'phase', 'expressions'}, {'workers', 'force'})
    chronology, plan, binding, candidate = _bound(data_root, params['protocol_id'])
    if chronology._selection_path(params['protocol_id']).exists():
        raise ValueError('population is frozen; declare a new protocol before more training')
    attempt = chronology.evidence.begin('chronology.trial', params,
        parents=list(dict.fromkeys([params['protocol_id'], *(parent_attempts() or [])])))
    try:
        if params['phase'] not in ('development', 'validation'):
            raise ValueError('trial phase must be development or validation; final requires frozen selection')
        if type(params['expressions']) is not list or not params['expressions']:
            raise ValueError('expressions must be a nonempty JSON list')
    except (ValueError, TypeError) as error:
        chronology.evidence.finish(attempt['id'], 'rejected', error=str(error))
        raise
    return _dispatch(data_root, chronology, plan, binding, candidate, attempt,
                     params['phase'], params['expressions'], params, cancel=cancel, progress=progress)


def freeze_chronology_selection(data_root, params: dict):
    """Discover/verify COMPLETE population and freeze only successful numerical results."""
    params = _params(params, {'protocol_id', 'selected_ids', 'decision_reason'}, {'population_ids'})
    chronology, plan, binding, candidate = _bound(data_root, params['protocol_id'])
    selected = params['selected_ids']
    if type(selected) is not list or not selected:
        raise ValueError('selected_ids must be a nonempty list of numerical Lab attempts')
    from .candidate_workflow import read_candidate, _settings
    for identity in selected:
        chosen = read_candidate(data_root, identity, _allow_observation_filter=True)
        settings = _assert_phase_settings(chosen['result'], plan, binding)
        _assert_result(chosen['result'], binding, settings)
    population = params.get('population_ids')
    if population is None:
        population = sorted(chronology._population(params['protocol_id']))
    record = chronology.freeze_selection(params['protocol_id'], population_ids=population,
        selected_ids=selected, decision_reason=params['decision_reason'])
    return {'schema_version': SCHEMA_VERSION, 'protocol_id': params['protocol_id'],
            'selection_id': record['id'], 'selection_result_digest': record['result_digest'],
            'selection': record['request'], 'qualification': binding['qualification']}


def evaluate_chronology_final(data_root, params: dict, *, cancel=None, progress=None):
    """Reserve a NEW access, then execute ONLY the frozen expressions on final."""
    params = _params(params, {'protocol_id', 'selection_id'}, {'workers', 'force', 'allow_exposed'})
    _execution(params, cancel=cancel, progress=progress)
    chronology, plan, binding, candidate = _bound(data_root, params['protocol_id'])
    selection = chronology.get_selection(params['protocol_id'])
    expressions = []
    for ref in selection['request']['selected']:
        result = chronology.evidence.get_result(ref['result_digest'])
        if chronology.evidence.get(ref['id'])['kind'] != 'lab-evaluation':
            raise ValueError('frozen selection must contain numerical Lab trials')
        expressions.append(result['expression'])
    attempt = chronology.begin_final_evaluation(params['protocol_id'], selection_id=params['selection_id'],
        request={'workers': params.get('workers', 1), 'force': params.get('force', False)},
        allow_exposed=params.get('allow_exposed', False), parents=parent_attempts())
    return _dispatch(data_root, chronology, plan, binding, candidate, attempt, 'final', expressions,
                     params, cancel=cancel, progress=progress)


def view_chronology(data_root, params: dict):
    """Compact metadata/evidence view; do not hydrate final numerical payloads."""
    params = _params(params, {'protocol_id'})
    chronology, plan, binding, candidate = _bound(data_root, params['protocol_id'])
    return {**chronology.compact_view(params['protocol_id']), 'qualification': binding['qualification'],
            'source_attempt_id': binding['source_attempt_id'], 'source_result_digest': binding['source_result_digest'],
            'sample_split': scoring_split(plan)}


__all__ = ['preview_chronology', 'declare_chronology', 'submit_chronology_trial',
           'freeze_chronology_selection', 'evaluate_chronology_final', 'view_chronology']
