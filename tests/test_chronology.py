"""Known-answer chronological support and append-only access contracts."""
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

from qlib_yahoo.chronology import ChronologyStore, split_samples, validate_protocol
from qlib_yahoo.workflow_records import EvidenceConflict, EvidenceStore, canonical_json_bytes, seal_record


def protocol():
    return {'schema_version': 1,
            'sessions': ['2020-01-%02d' % day for day in range(1, 16)],
            'development': {'start': '2020-01-01', 'end': '2020-01-04'},
            'validation': {'start': '2020-01-06', 'end': '2020-01-08'},
            'final': {'start': '2020-01-10', 'end': '2020-01-12'},
            'input_lineage': {'prices_sha256': 'a' * 64, 'universe_sha256': 'b' * 64},
            'selection_rule': {'metric': 'declared_net_metric', 'rule': 'human review'}}


def result(start='2020-01-01', end='2020-01-08', information_end=None):
    return {'chronology': {'schema_version': 1, 'evidence_interval': {'start': start, 'end': end},
                           'information_end': information_end or end}, 'value': 0.5}


def sample(identity, session, start, end, **kwargs):
    return {'id': identity, 'session': '2020-01-%02d' % session,
            'information_start': '2020-01-%02d' % start,
            'information_end': '2020-01-%02d' % end, **kwargs}


class ProtocolTests(unittest.TestCase):
    def test_valid_protocol_is_detached_support_only(self):
        plan = protocol()
        saved = validate_protocol(plan)
        plan['selection_rule']['rule'] = 'changed'
        self.assertEqual(saved['selection_rule']['rule'], 'human review')
        split = split_samples(saved)
        self.assertEqual(split['assignments']['development'], saved['sessions'][:4])
        self.assertEqual(split['assignments']['validation'], saved['sessions'][5:8])
        self.assertEqual(split['assignments']['final'], saved['sessions'][9:12])
        self.assertEqual(split['overlap_check'], 'not_available')
        self.assertEqual(split['claim'], 'split_support_only')

    def test_out_of_order_overlapping_and_touching_windows_refused(self):
        for field, interval in (
            ('development', {'start': '2020-01-04', 'end': '2020-01-01'}),
            ('development', {'start': '2020-01-01', 'end': '2020-01-06'}),
            ('validation', {'start': '2020-01-04', 'end': '2020-01-08'}),
            ('validation', {'start': '2020-01-10', 'end': '2020-01-12'}),
            ('final', {'start': '2020-01-01', 'end': '2020-01-02'}),
        ):
            plan = protocol()
            plan[field] = interval
            with self.subTest(field=field, interval=interval), self.assertRaises(ValueError):
                validate_protocol(plan)

    def test_strict_dates_calendar_and_schema(self):
        for bad in ('20200101', '2020-W01-3', '2020-01-01T00:00:00', '2020-02-30',
                    '2020-1-01', date(2020, 1, 1), None):
            plan = protocol()
            plan['development']['start'] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_protocol(plan)
        changes = [
            {'schema_version': 2}, {'schema_version': True}, {'extra': 1},
            {'sessions': []}, {'sessions': list(reversed(protocol()['sessions']))},
            {'sessions': protocol()['sessions'] + ['2020-01-15']},
            {'sessions': protocol()['sessions'][1:]}, {'input_lineage': {}},
            {'selection_rule': []}, {'metadata': 'bad'},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_protocol(protocol() | change)

    def test_nonfinite_and_nonplain_json_refused(self):
        for value in (float('nan'), float('inf'), Path('input'), (1, 2), {1: 'x'}):
            plan = protocol()
            plan['metadata'] = {'bad': value}
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                validate_protocol(plan)

    def test_actual_overlap_purge_includes_future_label_boundary(self):
        split = split_samples(protocol(), [sample('safe', 1, 1, 2),
            sample('future_dev_label', 4, 4, 6), sample('safe_validation', 6, 6, 7),
            sample('future_validation_label', 8, 8, 10), sample('test', 10, 10, 12)])
        self.assertEqual(split['assignments'], {
            'development': ['safe'], 'validation': ['safe_validation'], 'final': ['test']})
        self.assertEqual(split['excluded'], [
            {'id': 'future_dev_label', 'phase': 'development', 'reason': 'information_overlap', 'overlaps': ['validation']},
            {'id': 'future_validation_label', 'phase': 'validation', 'reason': 'information_overlap', 'overlaps': ['final']},
        ])

    def test_later_information_lookback_is_used_not_embargo_count(self):
        split = split_samples(protocol(), [sample('dev', 4, 3, 4),
            sample('validation', 6, 4, 7), sample('final', 10, 9, 12)])
        self.assertEqual(split['assignments']['development'], [])
        self.assertEqual(split['excluded'][0]['overlaps'], ['validation'])
        # A label ending in the gap is retained when it overlaps no later information.
        gap = split_samples(protocol(), [sample('gap_label', 4, 4, 5), sample('v', 6, 6, 7)])
        self.assertEqual(gap['assignments']['development'], ['gap_label'])

    def test_wrong_sample_intervals_and_future_features_refused(self):
        cases = [sample('a', 4, 1, 3), sample('a', 4, 5, 6), sample('a', 4, 4, 16),
                 sample('a', 5, 5, 5), sample('a', 4, 6, 4),
                 sample('a', 4, 4, 5, feature_available_at='2020-01-05'),
                 sample('a', 4, 4, 5, embargo=2), sample('a', 4, 4, 5, information_end=date(2020, 1, 5))]
        for bad in cases:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                split_samples(protocol(), [bad])
        with self.assertRaises(ValueError):
            split_samples(protocol(), [sample('same', 1, 1, 1), sample('same', 2, 2, 2)])
        with self.assertRaises(ValueError):
            split_samples(protocol(), (sample('a', 1, 1, 1),))

    def test_final_labels_cannot_extend_past_declared_evaluation(self):
        split = split_samples(protocol(), [sample('late', 12, 12, 13)])
        self.assertEqual(split['assignments']['final'], [])
        self.assertEqual(split['excluded'][0]['reason'], 'information_outside_final')
        self.assertEqual(split_samples(protocol(), [])['overlap_check'], 'information_interval_overlap')


_DEFAULT = object()


class ChronologyEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.chronology-test-', dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name) / 'data'
        self.evidence = EvidenceStore(self.data)
        self.store = ChronologyStore(self.evidence)

    def register(self, plan=None):
        return self.store.register_protocol(protocol() if plan is None else plan)['id']

    def trial(self, protocol_id, *, status='completed', payload=_DEFAULT, parents=None):
        attempt = self.evidence.begin('validation', {'trial': True},
                                      parents=[protocol_id] if parents is None else parents)
        if status != 'pending':
            payload = result() if payload is _DEFAULT and status in ('completed', 'reused') else (None if payload is _DEFAULT else payload)
            self.evidence.finish(attempt['id'], status, result=payload,
                                 error=None if status in ('completed', 'reused') else {'reason': status})
        return attempt['id']

    def freeze(self, protocol_id, population, selected=None, reason='reviewed declared rule'):
        return self.store.freeze_selection(protocol_id, population_ids=population,
                                           selected_ids=population[:1] if selected is None else selected,
                                           decision_reason=reason)

    def prepared(self):
        protocol_id = self.register()
        trial = self.trial(protocol_id)
        selection = self.freeze(protocol_id, [trial])
        return protocol_id, trial, selection

    def test_protocol_immutable_and_no_evaluation_claim(self):
        plan = protocol()
        identity = self.register(plan)
        plan['selection_rule']['rule'] = 'changed'
        self.assertEqual(self.store.get_protocol(identity), protocol())
        view = self.store.compact_view(identity)
        self.assertEqual(view['claim'], 'chronological_support_only')
        self.assertEqual(view['population'], [])
        self.assertIsNone(view['selection_id'])
        self.assertEqual(view['accesses'], [])
        self.assertEqual(len(self.evidence.list_attempts()), 1)

    def test_freeze_keeps_entire_population_and_selected_digests_and_parents(self):
        identity = self.register()
        chosen = self.trial(identity)
        failed = self.trial(identity, status='failed')
        rejected = self.trial(identity, status='rejected')
        cancelled = self.trial(identity, status='cancelled')
        interrupted = self.trial(identity, status='interrupted_unknown')
        reused = self.trial(identity, status='reused')
        population = [chosen, failed, rejected, cancelled, interrupted, reused]
        frozen = self.freeze(identity, population, [chosen, reused])
        payload = frozen['request']
        self.assertEqual({r['id'] for r in payload['population']}, set(population))
        self.assertEqual({r['status'] for r in payload['population']},
                         {'completed', 'reused', 'failed', 'rejected', 'cancelled', 'interrupted_unknown'})
        self.assertEqual(payload['selected'][0]['result_digest'], self.evidence.get(chosen)['result_digest'])
        self.assertEqual(payload['selected'][0]['parents'], [identity])
        self.assertEqual(set(frozen['parents']), {identity, *population})
        self.assertEqual(payload['decision_reason'], 'reviewed declared rule')
        payload['selected'].clear()
        self.assertEqual(len(self.store.get_selection(identity)['request']['selected']), 2)

    def test_descendant_population_is_complete_and_unrelated_trials_excluded(self):
        identity = self.register()
        parent = self.trial(identity)
        child = self.trial(identity, parents=[parent])
        other = self.register()
        unrelated = self.trial(other)
        with self.assertRaisesRegex(ValueError, 'population'):
            self.freeze(identity, [child])
        frozen = self.freeze(identity, [parent, child], [child])
        self.assertNotIn(unrelated, [r['id'] for r in frozen['request']['population']])

    def test_missing_failed_trial_rejects_incomplete_population(self):
        identity = self.register()
        chosen = self.trial(identity)
        failure = self.trial(identity, status='failed')
        with self.assertRaisesRegex(ValueError, 'population'):
            self.freeze(identity, [chosen])
        self.assertEqual(self.evidence.get(failure)['status'], 'failed')
        self.assertEqual(len(self.evidence.list_attempts()), 3)

    def test_no_implicit_hundred_trial_population_limit(self):
        identity = self.register()
        chosen = self.trial(identity)
        population = [chosen] + [self.trial(identity, status='rejected') for _ in range(100)]
        with self.assertRaisesRegex(ValueError, 'population'):
            self.freeze(identity, population[:100])
        self.assertEqual(len(self.freeze(identity, population)['request']['population']), 101)

    def test_selection_must_be_successful_in_population_with_immutable_result(self):
        for status, payload in (('failed', None), ('rejected', None), ('cancelled', None),
                                ('completed', None), ('reused', None)):
            identity = self.register()
            chosen = self.trial(identity, status=status, payload=payload)
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, 'successful immutable result'):
                self.freeze(identity, [chosen])
        identity = self.register()
        candidate = self.trial(identity)
        foreign = self.trial(self.register())
        with self.assertRaisesRegex(ValueError, 'not in complete population'):
            self.freeze(identity, [candidate], [foreign])

    def test_pending_population_cannot_be_frozen(self):
        identity = self.register()
        chosen = self.trial(identity)
        pending = self.trial(identity, status='pending')
        with self.assertRaisesRegex(ValueError, 'pending'):
            self.freeze(identity, [chosen, pending])
        self.assertEqual(self.evidence.get(pending)['status'], 'pending')

    def test_selected_legacy_dates_missing_and_bad_evidence_intervals_refused(self):
        for payload in ({'value': 1}, result(end='2020-01-17'),
                        result(end='2020-01-08', information_end='2020-01-06'),
                        result() | {'chronology': {'schema_version': 2}}):
            identity = self.register()
            chosen = self.trial(identity, payload=payload)
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.freeze(identity, [chosen])

    def test_frozen_selection_retries_identical_but_changes_refused(self):
        identity = self.register()
        first, second = self.trial(identity), self.trial(identity)
        frozen = self.freeze(identity, [first, second], [first])
        original = self.store.get_selection(identity)
        self.assertEqual(self.freeze(identity, [second, first], [first])['id'], frozen['id'])
        for selected, reason in (([second], 'reviewed declared rule'), ([first], 'changed reason')):
            with self.assertRaises(EvidenceConflict):
                self.freeze(identity, [first, second], selected, reason)
        self.assertEqual(self.store.get_selection(identity), original)

    def test_new_trial_after_freeze_invalidates_final_access(self):
        identity, trial, frozen = self.prepared()
        later = self.trial(identity, status='failed')
        with self.assertRaisesRegex(ValueError, 'population'):
            self.store.begin_final_evaluation(identity, selection_id=frozen['id'])
        with self.assertRaises(EvidenceConflict):
            self.freeze(identity, [trial, later], [trial])
        self.assertEqual(self.evidence.get(frozen['id'])['status'], 'completed')

    def test_first_repeated_access_each_new_attempt_including_failed_access(self):
        identity, trial, frozen = self.prepared()
        first = self.store.begin_final_evaluation(identity, selection_id=frozen['id'], request={'engine': 'fixture'})
        self.assertEqual(first['status'], 'pending')
        self.assertEqual(first['request']['access_label'], 'first_recorded_access')
        self.assertFalse(first['request']['reused_holdout'])
        self.assertEqual(first['request']['claim'], 'evaluation_not_yet_performed')
        self.assertEqual(first['request']['evaluation_request'], {'engine': 'fixture'})
        self.assertEqual(set(first['parents']), {identity, frozen['id']})
        self.evidence.finish(first['id'], 'failed', error='engine stopped')
        second = self.store.begin_final_evaluation(identity, selection_id=frozen['id'])
        third = self.store.begin_final_evaluation(identity, selection_id=frozen['id'])
        self.assertEqual(len({first['id'], second['id'], third['id']}), 3)
        for attempt in (second, third):
            self.assertEqual(attempt['request']['access_label'], 'reused_holdout')
            self.assertTrue(attempt['request']['reused_holdout'])
        view = self.store.compact_view(identity)
        self.assertEqual(len(view['population']), 1)
        self.assertEqual(len(view['accesses']), 3)

    def test_final_requires_exact_frozen_selection(self):
        identity = self.register()
        chosen = self.trial(identity)
        with self.assertRaises(FileNotFoundError):
            self.store.begin_final_evaluation(identity, selection_id=chosen)
        frozen = self.freeze(identity, [chosen])
        with self.assertRaisesRegex(ValueError, 'frozen selection identity'):
            self.store.begin_final_evaluation(identity, selection_id=chosen)
        self.assertEqual(self.store.compact_view(identity)['accesses'], [])

    def test_evidence_reaching_final_refused_or_labeled_exposed(self):
        for payload in (result(end='2020-01-10'), result(information_end='2020-01-10')):
            identity = self.register()
            chosen = self.trial(identity, payload=payload)
            frozen = self.freeze(identity, [chosen])
            self.assertTrue(frozen['request']['exposure_reasons'])
            with self.assertRaisesRegex(ValueError, 'exposed'):
                self.store.begin_final_evaluation(identity, selection_id=frozen['id'])
            attempt = self.store.begin_final_evaluation(identity, selection_id=frozen['id'], allow_exposed=True)
            self.assertEqual(attempt['request']['access_label'], 'already_exposed_holdout')
            self.assertTrue(attempt['request']['reused_holdout'])

    def test_unselected_exposed_or_unknown_success_is_not_untouched(self):
        for other_result in (result(end='2020-01-12'), {'value': 42}):
            identity = self.register()
            chosen = self.trial(identity)
            other = self.trial(identity, payload=other_result)
            frozen = self.freeze(identity, [chosen, other], [chosen])
            with self.assertRaisesRegex(ValueError, 'exposed'):
                self.store.begin_final_evaluation(identity, selection_id=frozen['id'])
            self.assertEqual(frozen['request']['exposure_reasons'][0]['attempt_id'], other)

    def test_failed_partial_result_exposure_is_preserved(self):
        identity = self.register()
        chosen = self.trial(identity)
        failed = self.trial(identity, status='failed', payload=result(end='2020-01-11'))
        frozen = self.freeze(identity, [chosen, failed])
        with self.assertRaisesRegex(ValueError, 'exposed'):
            self.store.begin_final_evaluation(identity, selection_id=frozen['id'])

    def test_new_protocol_or_selection_does_not_reset_holdout(self):
        first_id, trial, first_selection = self.prepared()
        first = self.store.begin_final_evaluation(first_id, selection_id=first_selection['id'])
        second_id, trial, second_selection = self.prepared()
        second = self.store.begin_final_evaluation(second_id, selection_id=second_selection['id'])
        self.assertEqual(first['request']['holdout_key'], second['request']['holdout_key'])
        self.assertEqual(second['request']['access_label'], 'reused_holdout')
        changed = protocol()
        changed['input_lineage']['prices_sha256'] = 'c' * 64
        identity = self.register(changed)
        trial = self.trial(identity)
        frozen = self.freeze(identity, [trial])
        third = self.store.begin_final_evaluation(identity, selection_id=frozen['id'])
        self.assertNotEqual(first['request']['holdout_key'], third['request']['holdout_key'])
        self.assertEqual(third['request']['access_label'], 'first_recorded_access')

    def external_protocol(self, *, lineage=None):
        plan = protocol()
        plan['validation'] = {'start': '2020-01-10', 'end': '2020-01-12'}
        plan['final'] = {'start': '2020-01-13', 'end': '2020-01-15'}
        if lineage is not None:
            plan['input_lineage'] = lineage
        return self.register(plan)

    def external_lab(self, protocol_id, *, status='completed', start='2020-01-10',
                     end='2020-01-12', payload=_DEFAULT):
        attempt = self.evidence.begin('lab-evaluation',
            {'expression': 'supplied', 'settings': {'start': start, 'end': end}}, parents=[protocol_id])
        if status != 'pending':
            if payload is _DEFAULT:
                payload = result(start=start, end=end) if status in ('completed', 'reused') else None
            self.evidence.finish(attempt['id'], status, result=payload,
                                 error=None if status in ('completed', 'reused') else status)
        return attempt['id']

    def test_external_validation_after_freeze_gates_final_without_changing_selection(self):
        identity, _, frozen = self.prepared()
        before = self.store.get_selection(identity)
        other = self.external_protocol()
        external = self.external_lab(other)
        self.assertEqual(self.store.get_selection(identity), before)
        self.assertEqual(self.freeze(identity, [r['id'] for r in frozen['request']['population']])['id'], frozen['id'])
        view = self.store.compact_view(identity)
        self.assertEqual(view['frozen_exposure_reasons'], [])
        self.assertEqual(view['external_exposure_reasons'][0]['attempt_id'], external)
        self.assertEqual(view['external_exposure_reasons'][0]['certainty'], 'possible')
        with self.assertRaisesRegex(ValueError, 'exposed'):
            self.store.begin_final_evaluation(identity, selection_id=frozen['id'])
        self.assertFalse((self.evidence.root / 'chronology' / 'holdouts').exists())
        access = self.store.begin_final_evaluation(identity, selection_id=frozen['id'], allow_exposed=True)
        self.assertEqual(access['request']['access_label'], 'already_exposed_holdout')
        self.assertTrue(access['request']['reused_holdout'])
        self.assertEqual(access['request']['frozen_exposure_reasons'], [])
        self.assertEqual(access['request']['external_exposure_reasons'][0]['certainty'], 'observed')
        self.assertEqual(access['request']['exposure_reasons'], access['request']['external_exposure_reasons'])
        self.assertEqual(self.store.get_selection(identity), before)

    def test_external_pending_failed_and_interrupted_use_conservative_bounds(self):
        for status in ('pending', 'failed', 'cancelled', 'interrupted_unknown', 'rejected'):
            with self.subTest(status=status):
                identity, _, frozen = self.prepared()
                other = self.external_protocol()
                external = self.external_lab(other, status=status)
                with self.assertRaisesRegex(ValueError, 'exposed'):
                    self.store.begin_final_evaluation(identity, selection_id=frozen['id'])
                access = self.store.begin_final_evaluation(identity, selection_id=frozen['id'], allow_exposed=True)
                reason = next(r for r in access['request']['external_exposure_reasons'] if r['attempt_id'] == external)
                self.assertEqual(reason['certainty'], 'possible')
                self.assertEqual(reason['basis'], 'requested_window')
                self.assertEqual(reason['status'], status)

    def test_external_unknown_pending_bounds_are_uncertain_not_observed(self):
        identity, _, frozen = self.prepared()
        other = self.external_protocol()
        external = self.trial(other, status='pending')
        view = self.store.compact_view(identity)
        reason = next(r for r in view['external_exposure_reasons'] if r['attempt_id'] == external)
        self.assertEqual(reason['basis'], 'unknown')
        self.assertEqual(reason['certainty'], 'possible')
        with self.assertRaisesRegex(ValueError, 'exposed'):
            self.store.begin_final_evaluation(identity, selection_id=frozen['id'])

    def test_external_failed_partial_result_does_not_shrink_requested_access(self):
        identity, _, frozen = self.prepared()
        other = self.external_protocol()
        external = self.external_lab(other, status='failed',
                                     payload=result(start='2020-01-01', end='2020-01-04'))
        access = self.store.begin_final_evaluation(identity, selection_id=frozen['id'], allow_exposed=True)
        reason = next(r for r in access['request']['external_exposure_reasons'] if r['attempt_id'] == external)
        self.assertEqual(reason['basis'], 'requested_window')
        self.assertEqual(reason['certainty'], 'possible')

    def test_external_observed_interval_overrides_successful_requested_horizon(self):
        identity, _, frozen = self.prepared()
        other = self.external_protocol()
        external = self.external_lab(other, start='2020-01-01', end='2020-01-04',
                                     payload=result(start='2020-01-10', end='2020-01-12'))
        with self.assertRaisesRegex(ValueError, 'exposed'):
            self.store.begin_final_evaluation(identity, selection_id=frozen['id'])
        access = self.store.begin_final_evaluation(identity, selection_id=frozen['id'], allow_exposed=True)
        self.assertEqual(access['request']['external_exposure_reasons'][0]['attempt_id'], external)
        self.assertEqual(access['request']['external_exposure_reasons'][0]['certainty'], 'observed')

    def test_external_unrelated_lineage_and_bounded_early_failure_are_controls(self):
        identity, _, frozen = self.prepared()
        unrelated = self.external_protocol(lineage={'prices_sha256': 'c' * 64})
        self.external_lab(unrelated)
        same = self.external_protocol()
        self.external_lab(same, status='failed', start='2020-01-01', end='2020-01-04')
        self.assertEqual(self.store.compact_view(identity)['external_exposure_reasons'], [])
        access = self.store.begin_final_evaluation(identity, selection_id=frozen['id'])
        self.assertEqual(access['request']['access_label'], 'first_recorded_access')
        self.assertEqual(access['request']['external_exposure_reasons'], [])

    def test_external_price_identity_survives_settings_and_source_switch(self):
        identity, _, frozen = self.prepared()
        first = self.register()
        source = self.trial(first)
        plan = protocol()
        plan['input_lineage']['universe_sha256'] = 'changed-universe-setting'
        plan['selection_rule'] = {'rule': 'different settings'}
        plan['metadata'] = {'chronology_workflow': {'source_attempt_id': source, 'settings': {'delay': 0}}}
        other = self.register(plan)
        external = self.external_lab(other)
        with self.assertRaisesRegex(ValueError, 'exposed'):
            self.store.begin_final_evaluation(identity, selection_id=frozen['id'])
        view = self.store.compact_view(identity)
        self.assertIn(external, [r['attempt_id'] for r in view['external_exposure_reasons']])
        self.assertEqual(self.store.get_selection(identity), frozen)

    def test_external_empty_declaration_is_not_an_observation(self):
        identity, _, frozen = self.prepared()
        self.external_protocol()
        self.assertEqual(self.store.compact_view(identity)['external_exposure_reasons'], [])
        access = self.store.begin_final_evaluation(identity, selection_id=frozen['id'])
        self.assertEqual(access['request']['access_label'], 'first_recorded_access')

    def test_external_canonical_price_tree_ignores_recipe_and_calendar_settings(self):
        plan = protocol()
        plan['input_lineage'] = {'price_tree_sha256': 'price-tree', 'calendar_sha256': 'calendar-a',
                                 'symbols_sha256': 'symbols-a'}
        identity = self.register(plan)
        chosen = self.trial(identity)
        frozen = self.freeze(identity, [chosen])
        other = self.external_protocol(lineage={'price_tree_sha256': 'price-tree',
            'calendar_sha256': 'calendar-b', 'symbols_sha256': 'symbols-b', 'recipe': 'different'})
        self.external_lab(other)
        with self.assertRaisesRegex(ValueError, 'exposed'):
            self.store.begin_final_evaluation(identity, selection_id=frozen['id'])

    def test_external_success_without_observed_dates_cannot_prove_a_safe_requested_window(self):
        identity, _, frozen = self.prepared()
        other = self.external_protocol()
        external = self.external_lab(other, start='2020-01-01', end='2020-01-04', payload={'value': 1})
        access = self.store.begin_final_evaluation(identity, selection_id=frozen['id'], allow_exposed=True)
        reason = next(r for r in access['request']['external_exposure_reasons'] if r['attempt_id'] == external)
        self.assertEqual(reason['basis'], 'unknown')
        self.assertEqual(reason['certainty'], 'possible')

    def test_external_failed_partial_overlap_is_observed_even_with_safe_requested_window(self):
        identity, _, frozen = self.prepared()
        other = self.external_protocol()
        external = self.external_lab(other, status='failed', start='2020-01-01', end='2020-01-04',
                                     payload=result(start='2020-01-10', end='2020-01-12'))
        access = self.store.begin_final_evaluation(identity, selection_id=frozen['id'], allow_exposed=True)
        reason = next(r for r in access['request']['external_exposure_reasons'] if r['attempt_id'] == external)
        self.assertEqual(reason['certainty'], 'observed')

    def test_external_pending_phase_without_numeric_children_has_declared_bound(self):
        identity, _, frozen = self.prepared()
        other = self.external_protocol()
        pending = self.evidence.begin('chronology.trial', {'protocol_id': other, 'phase': 'validation'}, parents=[other])
        view = self.store.compact_view(identity)
        reason = next(r for r in view['external_exposure_reasons'] if r['attempt_id'] == pending['id'])
        self.assertEqual(reason['basis'], 'declared_phase_window')
        with self.assertRaisesRegex(ValueError, 'exposed'):
            self.store.begin_final_evaluation(identity, selection_id=frozen['id'])

    def test_compact_external_view_reads_no_numerical_payloads_including_previous_final(self):
        identity, chosen, frozen = self.prepared()
        final = self.store.begin_final_evaluation(identity, selection_id=frozen['id'])
        self.evidence.finish(final['id'], 'completed', result={'private_final_numbers': [1, 2]})
        child = self.evidence.begin('lab-evaluation', {'settings': {'start': '2020-01-10', 'end': '2020-01-12'}},
                                    parents=[final['id']])
        self.evidence.finish(child['id'], 'completed', result=result(start='2020-01-10', end='2020-01-12'))
        other = self.external_protocol()
        external = self.external_lab(other)
        forbidden = {self.evidence.get(i)['result_digest'] for i in (chosen, final['id'], child['id'], external)}
        original = self.evidence.get_result

        def metadata_only(digest):
            if digest in forbidden:
                raise AssertionError('view hydrated numerical evidence')
            return original(digest)

        with patch.object(self.evidence, 'get_result', side_effect=metadata_only):
            view = self.store.compact_view(identity)
        self.assertEqual(len(view['accesses']), 1)
        self.assertEqual([r['attempt_id'] for r in view['external_exposure_reasons']], [external])

    def test_external_change_during_registration_rejects_without_rewriting_selection(self):
        identity, _, frozen = self.prepared()
        other = self.external_protocol()
        original = self.evidence.begin

        def racing(kind, *args, **kwargs):
            attempt = original(kind, *args, **kwargs)
            if kind == 'chronology.final':
                self.external_lab(other, status='pending')
            return attempt

        with patch.object(self.evidence, 'begin', side_effect=racing), self.assertRaisesRegex(ValueError, 'external exposure changed'):
            self.store.begin_final_evaluation(identity, selection_id=frozen['id'])
        self.assertEqual(self.store.get_selection(identity), frozen)
        finals = [r for r in self.evidence.list_attempts() if r['kind'] == 'chronology.final']
        self.assertEqual(len(finals), 1)
        self.assertEqual(finals[0]['status'], 'rejected')

    def test_external_source_with_final_ancestry_is_uncertain_without_reading_final_numbers(self):
        identity, chosen, frozen = self.prepared()
        final = self.store.begin_final_evaluation(identity, selection_id=frozen['id'])
        self.evidence.finish(final['id'], 'completed', result={'final_numbers': [42]})
        child = self.evidence.begin('lab-evaluation', {}, parents=[final['id']])
        self.evidence.finish(child['id'], 'completed', result={'final_child_numbers': [43]})
        plan = protocol()
        plan['metadata'] = {'chronology_workflow': {'source_attempt_id': child['id']}}
        self.register(plan)
        forbidden = {self.evidence.get(i)['result_digest'] for i in (final['id'], child['id'])}
        original = self.evidence.get_result

        def no_final_numbers(digest):
            if digest in forbidden:
                raise AssertionError('external source hydrated final evidence')
            return original(digest)

        with patch.object(self.evidence, 'get_result', side_effect=no_final_numbers):
            view = self.store.compact_view(identity)
            with self.assertRaisesRegex(ValueError, 'exposed'):
                self.store.begin_final_evaluation(identity, selection_id=frozen['id'])
            access = self.store.begin_final_evaluation(identity, selection_id=frozen['id'], allow_exposed=True)
        self.assertEqual(view['external_exposure_reasons'][0]['certainty'], 'possible')
        self.assertEqual(access['request']['external_exposure_reasons'][0]['reason'],
                         'external_source_final_ancestry_unknown')

    def test_concurrent_final_access_has_one_first_reservation(self):
        identity, trial, frozen = self.prepared()
        barrier = threading.Barrier(6)

        def access(index):
            store = ChronologyStore(EvidenceStore(self.data))
            barrier.wait(timeout=10)
            return store.begin_final_evaluation(identity, selection_id=frozen['id'], request={'caller': index})

        with ThreadPoolExecutor(max_workers=6) as pool:
            attempts = list(pool.map(access, range(6)))
        self.assertEqual(len({r['id'] for r in attempts}), 6)
        self.assertEqual(sum(r['request']['access_label'] == 'first_recorded_access' for r in attempts), 1)
        self.assertEqual(sum(r['request']['reused_holdout'] for r in attempts), 5)
        self.assertTrue(all(r['status'] == 'pending' for r in attempts))

    def test_changed_overlapping_final_window_remembers_prior_access(self):
        identity, trial, frozen = self.prepared()
        self.store.begin_final_evaluation(identity, selection_id=frozen['id'])
        for start, end, expected in ((11, 13, 'reused_holdout'), (13, 15, 'reused_holdout')):
            plan = protocol()
            plan['final'] = {'start': '2020-01-%02d' % start, 'end': '2020-01-%02d' % end}
            other = self.register(plan)
            candidate = self.trial(other)
            selection = self.freeze(other, [candidate])
            attempt = self.store.begin_final_evaluation(other, selection_id=selection['id'])
            self.assertEqual(attempt['request']['access_label'], expected)
        # Fresh lineage: adjacent but disjoint inclusive intervals are not reuse.
        clean = protocol()
        clean['input_lineage'] = {'prices_sha256': 'c' * 64}
        original = self.register(clean)
        candidate = self.trial(original)
        selected = self.freeze(original, [candidate])
        self.store.begin_final_evaluation(original, selection_id=selected['id'])
        clean['final'] = {'start': '2020-01-13', 'end': '2020-01-15'}
        new = self.register(clean)
        candidate = self.trial(new)
        selected = self.freeze(new, [candidate])
        attempt = self.store.begin_final_evaluation(new, selection_id=selected['id'])
        self.assertEqual(attempt['request']['access_label'], 'first_recorded_access')

    def test_concurrent_overlapping_distinct_windows_are_conservative(self):
        first, trial, first_selection = self.prepared()
        plan = protocol()
        plan['final'] = {'start': '2020-01-11', 'end': '2020-01-13'}
        second = self.register(plan)
        trial = self.trial(second)
        second_selection = self.freeze(second, [trial])
        barrier = threading.Barrier(2)

        def access(pair):
            barrier.wait(timeout=10)
            return ChronologyStore(self.data).begin_final_evaluation(pair[0], selection_id=pair[1])

        with ThreadPoolExecutor(max_workers=2) as pool:
            attempts = list(pool.map(access, [(first, first_selection['id']), (second, second_selection['id'])]))
        self.assertLessEqual(sum(not r['request']['reused_holdout'] for r in attempts), 1)
        self.assertTrue(any(r['request']['reused_holdout'] for r in attempts))

    def test_concurrent_different_selections_never_change_winner(self):
        identity = self.register()
        trials = [self.trial(identity), self.trial(identity)]
        barrier = threading.Barrier(2)

        def freeze(index):
            barrier.wait(timeout=10)
            try:
                return ChronologyStore(EvidenceStore(self.data)).freeze_selection(identity,
                    population_ids=trials, selected_ids=[trials[index]], decision_reason='fixed rule')
            except EvidenceConflict:
                return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(freeze, range(2)))
        winners = [r for r in results if r is not None]
        self.assertEqual(len(winners), 1)
        self.assertEqual(self.store.get_selection(identity)['id'], winners[0]['id'])

    def test_abandoned_access_reservation_conservatively_consumes_first(self):
        identity, trial, frozen = self.prepared()
        original = self.evidence.begin

        def interrupted(kind, *args, **kwargs):
            if kind == 'chronology.final':
                raise KeyboardInterrupt
            return original(kind, *args, **kwargs)

        with patch.object(self.evidence, 'begin', side_effect=interrupted), self.assertRaises(KeyboardInterrupt):
            self.store.begin_final_evaluation(identity, selection_id=frozen['id'])
        resumed = self.store.begin_final_evaluation(identity, selection_id=frozen['id'])
        self.assertEqual(resumed['request']['access_label'], 'reused_holdout')

    def test_population_race_during_final_registration_rejects_before_dispatch(self):
        identity, trial, frozen = self.prepared()
        original = self.evidence.begin

        def racing(kind, *args, **kwargs):
            attempt = original(kind, *args, **kwargs)
            if kind == 'chronology.final':
                self.trial(identity, status='failed')
            return attempt

        with patch.object(self.evidence, 'begin', side_effect=racing), self.assertRaisesRegex(ValueError, 'population'):
            self.store.begin_final_evaluation(identity, selection_id=frozen['id'])
        attempts = [self.evidence.get(r['id']) for r in self.evidence.list_attempts()
                    if r['kind'] == 'chronology.final']
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]['status'], 'rejected')

    def test_freeze_crash_after_pointer_can_resume_identical_request(self):
        identity = self.register()
        trial = self.trial(identity)
        original = self.evidence.finish

        def interrupted(attempt_id, status, **kwargs):
            if self.evidence.get(attempt_id)['kind'] == 'chronology.selection':
                raise KeyboardInterrupt
            return original(attempt_id, status, **kwargs)

        with patch.object(self.evidence, 'finish', side_effect=interrupted), self.assertRaises(KeyboardInterrupt):
            self.freeze(identity, [trial])
        with self.assertRaisesRegex(ValueError, 'not frozen'):
            self.store.get_selection(identity)
        recovered = self.freeze(identity, [trial])
        self.assertEqual(recovered['status'], 'completed')

    def test_json_strictness_for_freeze_and_access(self):
        identity, trial, frozen = self.prepared()
        for request in ([], {'bad': float('nan')}, {'bad': Path('x')}, {'bad': (1, 2)}):
            with self.assertRaises(ValueError):
                self.store.begin_final_evaluation(identity, selection_id=frozen['id'], request=request)
        for kwargs in ({'population_ids': (trial,), 'selected_ids': [trial]},
                       {'population_ids': [trial, trial], 'selected_ids': [trial]},
                       {'population_ids': [trial], 'selected_ids': []},
                       {'population_ids': [trial], 'selected_ids': [trial], 'decision_reason': ''}):
            with self.assertRaises(ValueError):
                self.store.freeze_selection(identity, decision_reason=kwargs.pop('decision_reason', 'valid'), **kwargs)
        with self.assertRaises(ValueError):
            self.store.begin_final_evaluation(identity, selection_id=frozen['id'], allow_exposed=1)
        self.assertEqual(self.store.compact_view(identity)['accesses'], [])

    def test_tamper_protocol_selected_result_selection_pointer_and_holdout(self):
        identity, trial, frozen = self.prepared()
        pointer = self.evidence.root / 'chronology' / 'selections' / (identity + '.json')
        files = [pointer,
                 self.evidence.root / 'results' / (self.evidence.get(identity)['result_digest'] + '.json'),
                 self.evidence.root / 'results' / (self.evidence.get(trial)['result_digest'] + '.json'),
                 self.evidence.root / 'results' / (frozen['result_digest'] + '.json')]
        for path in files:
            original = path.read_bytes()
            path.write_bytes(original + b' ')
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.store.begin_final_evaluation(identity, selection_id=frozen['id'])
            path.write_bytes(original)
        first = self.store.begin_final_evaluation(identity, selection_id=frozen['id'])
        holdout = self.evidence.root / 'chronology' / 'holdouts' / (first['request']['holdout_key'] + '.json')
        holdout.write_bytes(b'{}')
        with self.assertRaises(ValueError):
            self.store.begin_final_evaluation(identity, selection_id=frozen['id'])

    def test_missing_pointer_target_never_replaced(self):
        identity, trial, frozen = self.prepared()
        submission = self.evidence.root / 'submissions' / (frozen['id'] + '.json')
        submission.unlink()
        with self.assertRaises(FileNotFoundError):
            self.freeze(identity, [trial])

    def test_pointer_reparse_and_traversal_fail_closed(self):
        with self.assertRaises(ValueError):
            self.store.get_selection('../escape')
        identity, trial, frozen = self.prepared()
        path = self.evidence.root / 'chronology' / 'selections' / (identity + '.json')
        original = Path.lstat

        def reparse(target, *args, **kwargs):
            info = original(target, *args, **kwargs)
            if target == path.parent:
                return types.SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
            return info

        with patch.object(Path, 'lstat', reparse), self.assertRaisesRegex(ValueError, 'reparse'):
            self.store.get_selection(identity)
        payload = json.loads(path.read_bytes())['payload']
        payload['selection_id'] = '../escape'
        path.write_bytes(canonical_json_bytes(seal_record(payload)))
        with self.assertRaises(ValueError):
            self.store.get_selection(identity)

    def test_changed_trial_outcome_breaks_frozen_reference(self):
        identity, trial, frozen = self.prepared()
        terminal_path = self.evidence.root / 'terminals' / (trial + '.json')
        terminal = json.loads(terminal_path.read_bytes())['payload']
        terminal['error'] = {'changed': 'after freeze'}
        terminal_path.write_bytes(canonical_json_bytes(seal_record(terminal)))
        # Even a coherently resealed individual outcome no longer matches the
        # independent frozen population snapshot.
        with self.assertRaisesRegex(ValueError, 'content/outcome references changed'):
            self.store.get_selection(identity)
        with self.assertRaises(ValueError):
            self.store.begin_final_evaluation(identity, selection_id=frozen['id'])

    def test_empty_constructor_is_read_only_and_import_has_no_scientific_dependency(self):
        empty = Path(self.temp.name) / 'empty'
        with patch.object(Path, 'mkdir', side_effect=AssertionError('constructor wrote data')):
            ChronologyStore(empty)
        self.assertFalse(empty.exists())
        code = ('import sys; import qlib_yahoo.chronology; '
                'assert not ({"numpy","pandas","qlib","sqlite3"} & set(sys.modules))')
        subprocess.run([sys.executable, '-B', '-c', code], check=True, timeout=15)


if __name__ == '__main__':
    unittest.main()
