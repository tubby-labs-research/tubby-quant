"""Real tiny Lab chronology chain; no Qlib initialization or remote inputs."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_workflow_chain import make_snapshot
NUMERICAL = bool(importlib.util.find_spec('numpy') and importlib.util.find_spec('pandas'))
if NUMERICAL:
    from qlib_yahoo.alphalab import batch
from qlib_yahoo.alphalab.settings import Settings
from qlib_yahoo.dataset import DatasetError
from qlib_yahoo.chronology import ChronologyStore
from qlib_yahoo.chronology_workflow import (
    preview_chronology, declare_chronology, submit_chronology_trial,
    freeze_chronology_selection, evaluate_chronology_final, view_chronology,
)
from qlib_yahoo.workflow_records import EvidenceStore, canonical_json_bytes, seal_record


@unittest.skipUnless(NUMERICAL, 'optional numpy/pandas environment')
class ChronologyWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.chronology-workflow-', dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.dataset, days = make_snapshot(self.root)
        self.days = days.strftime('%Y-%m-%d').tolist()
        self.settings = Settings(universe='TOP6', start=self.days[1], end=self.days[21],
                                 warmup=1, truncation=.2, cost_bps=5)
        self.source = batch.run_batch(self.dataset, ['rank(close)'], self.settings,
                                     data_root=self.root, workers=1)[0]
        self.store = EvidenceStore(self.root)

    def params(self, source=None):
        return {'source_attempt_id': source or self.source['attempt_id'],
                'development': {'start': self.days[1], 'end': self.days[21]},
                'validation': {'start': self.days[22], 'end': self.days[42]},
                'final': {'start': self.days[43], 'end': self.days[63]},
                'selection_rule': {'metric': 'declared_net_metric', 'rule': 'review supplied signals'}}

    def declare(self):
        return declare_chronology(self.root, self.params())['protocol_id']

    def trial(self, protocol_id, expressions=None, phase='development', **kwargs):
        return submit_chronology_trial(self.root, {
            'protocol_id': protocol_id, 'phase': phase,
            'expressions': ['rank(close)'] if expressions is None else expressions}, **kwargs)

    def prepare(self):
        identity = self.declare()
        trial = self.trial(identity)
        frozen = freeze_chronology_selection(self.root, {'protocol_id': identity,
            'selected_ids': trial['attempt_ids'], 'decision_reason': 'reviewed supplied candidate'})
        return identity, trial, frozen

    def final(self, protocol_id, frozen, **kwargs):
        return evaluate_chronology_final(self.root, {
            'protocol_id': protocol_id, 'selection_id': frozen['selection_id']}, **kwargs)

    def test_preview_declaration_reads_only_saved_source_no_final_inspection(self):
        before = len(self.store.list_attempts())
        with patch.object(batch, 'run_batch', side_effect=AssertionError('preview executed data')), \
                patch.object(batch, 'Workspace', side_effect=AssertionError('preview loaded a panel')), \
                patch('qlib_yahoo.candidate_workflow._verify_sources', side_effect=AssertionError('preview scanned sources')):
            preview = preview_chronology(self.root, self.params())
            declared = declare_chronology(self.root, self.params())
        self.assertEqual(preview['protocol']['sessions'], self.days)
        self.assertEqual(preview['protocol']['metadata']['chronology_workflow']['settings'], asdict(self.settings))
        self.assertEqual(preview['qualification']['economic_pit'], 'not_established')
        self.assertEqual(preview['sample_split']['overlap_check'], 'not_available')
        self.assertFalse((self.store.root / 'chronology' / 'holdouts').exists())
        self.assertEqual(len(self.store.list_attempts()), before + 1)
        self.assertIn(self.source['attempt_id'], self.store.get(declared['protocol_id'])['parents'])
        for extra in ('input_lineage', 'settings', 'dataset', 'sessions'):
            with self.assertRaises(ValueError):
                preview_chronology(self.root, self.params() | {extra: {}})

    def test_actual_development_validation_invalid_failed_and_reused_trials(self):
        identity = self.declare()
        trial = self.trial(identity, ['rank(close)', 'rank(close)', 'rank(', 'missing_field', 'sec_assets / close'])
        records = [self.store.get(i) for i in trial['attempt_ids']]
        self.assertEqual(len(records), 5)
        self.assertGreaterEqual(sum(r['status'] == 'reused' for r in records), 2)
        self.assertIn('failed', {r['status'] for r in records})
        self.assertGreaterEqual(sum(r['status'] == 'rejected' for r in records), 2)
        validation = self.trial(identity, ['-rank(close)'], phase='validation')
        for response, phase in ((trial, 'development'), (validation, 'validation')):
            for record_id in response['attempt_ids']:
                record = self.store.get(record_id)
                if record['status'] not in ('completed', 'reused'):
                    continue
                result = self.store.get_result(record['result_digest'])
                self.assertEqual(result['daily']['dates'], self.days[1:22] if phase == 'development' else self.days[22:43])
                self.assertEqual(result['settings'], asdict(self.settings) | self.params()[phase])
                dates = ChronologyStore(self.root)._evidence_dates(record, ChronologyStore(self.root).get_protocol(identity))
                self.assertEqual(dates['information_end'], result['daily']['dates'][-1])

    def test_full_population_freeze_grouping_records_do_not_fake_exposure(self):
        identity = self.declare()
        response = self.trial(identity, ['rank(close)', 'rank(', 'missing_field'])
        chosen = [i for i in response['attempt_ids'] if self.store.get(i)['status'] in ('completed', 'reused')]
        with self.assertRaisesRegex(ValueError, 'population'):
            freeze_chronology_selection(self.root, {'protocol_id': identity, 'population_ids': chosen,
                'selected_ids': chosen, 'decision_reason': 'missing failures'})
        frozen = freeze_chronology_selection(self.root, {'protocol_id': identity,
            'selected_ids': chosen, 'decision_reason': 'all submitted trials retained'})
        population = frozen['selection']['population']
        self.assertEqual(len(population), 5)  # workflow + batch + three expressions
        self.assertIn('failed', {r['status'] for r in population})
        self.assertIn('rejected', {r['status'] for r in population})
        self.assertEqual(frozen['selection']['exposure_reasons'], [])
        for ref in population:
            self.assertEqual(ref['result_digest'], self.store.get(ref['id'])['result_digest'])
        with self.assertRaises(ValueError):
            self.trial(identity)
        self.assertEqual(ChronologyStore(self.root).get_selection(identity)['id'], frozen['selection_id'])

    def test_actual_final_dispatch_twice_changes_only_window_and_preserves_selection(self):
        identity, trial, frozen = self.prepare()
        first = self.final(identity, frozen)
        second = self.final(identity, frozen)
        self.assertNotEqual(first['attempt_id'], second['attempt_id'])
        self.assertEqual(first['access_label'], 'first_recorded_access')
        self.assertFalse(first['reused_holdout'])
        self.assertEqual(second['access_label'], 'reused_holdout')
        self.assertTrue(second['reused_holdout'])
        self.assertEqual(second['status'], 'reused')
        first_child = self.store.get(first['attempt_ids'][0])
        second_child = self.store.get(second['attempt_ids'][0])
        self.assertEqual(first_child['result_digest'], second_child['result_digest'])
        payload = self.store.get_result(first_child['result_digest'])
        self.assertEqual(payload['daily']['dates'], self.days[43:64])
        self.assertEqual(payload['settings'], asdict(self.settings) | self.params()['final'])
        self.assertEqual(payload['dataset'], self.source['dataset'])
        self.assertEqual(payload['expression'], 'rank(close)')
        self.assertEqual(len(first['batch_attempt_ids']), 1)
        self.assertEqual(first_child['parents'], first['batch_attempt_ids'])
        self.assertEqual(self.store.get(first['batch_attempt_ids'][0])['parents'], [first['attempt_id']])
        self.assertEqual(ChronologyStore(self.root).get_selection(identity)['id'], frozen['selection_id'])
        self.assertEqual(len(ChronologyStore(self.root)._population(identity)), 3)

    def test_final_overrides_are_rejected_before_reservation(self):
        identity, trial, frozen = self.prepare()
        for key, value in (('settings', asdict(self.settings)), ('expressions', ['-rank(close)']),
                           ('dataset', str(self.dataset)), ('source_attempt_id', self.source['attempt_id']),
                           ('input_lineage', {'arbitrary': 'new'}), ('final', self.params()['development'])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                evaluate_chronology_final(self.root, {'protocol_id': identity, 'selection_id': frozen['selection_id'], key: value})
        self.assertFalse((self.store.root / 'chronology' / 'holdouts').exists())
        with self.assertRaises(ValueError):
            self.trial(identity, phase='final')

    def test_settings_and_new_source_attempt_do_not_reset_holdout_identity(self):
        identity, trial, frozen = self.prepare()
        first = self.final(identity, frozen)
        changed_settings = Settings(**(asdict(self.settings) | {'cost_bps': 8.0}))
        changed_source = batch.run_batch(self.dataset, ['-rank(close)'], changed_settings,
                                        data_root=self.root, workers=1)[0]
        declared = declare_chronology(self.root, self.params(changed_source['attempt_id']))
        other = declared['protocol_id']
        response = self.trial(other, ['-rank(close)'])
        selection = freeze_chronology_selection(self.root, {'protocol_id': other,
            'selected_ids': response['attempt_ids'], 'decision_reason': 'changed settings separate protocol'})
        second = self.final(other, selection)
        self.assertEqual(second['access_label'], 'reused_holdout')
        self.assertEqual(self.store.get(first['attempt_id'])['request']['holdout_key'],
                         self.store.get(second['attempt_id'])['request']['holdout_key'])
        final_result = self.store.get_result(self.store.get(second['attempt_ids'][0])['result_digest'])
        self.assertEqual(final_result['settings']['cost_bps'], 8.0)

    def test_changed_price_bytes_refused_after_final_reservation(self):
        identity, trial, frozen = self.prepare()
        price_file = next((self.dataset / 'qlib_data' / 'features').rglob('close.day.bin'))
        original = price_file.read_bytes()
        price_file.write_bytes(original[:-4] + b'bad!')
        with self.assertRaises(DatasetError):
            self.final(identity, frozen)
        finals = [r for r in self.store.list_attempts() if r['kind'] == 'chronology.final']
        self.assertEqual(len(finals), 1)
        self.assertNotEqual(finals[0]['status'], 'pending')
        self.assertTrue((self.store.root / 'chronology' / 'holdouts').exists())
        price_file.write_bytes(original)
        self.assertEqual(self.final(identity, frozen)['access_label'], 'reused_holdout')

    def test_reservation_exists_before_source_verification_or_batch_execution(self):
        identity, trial, frozen = self.prepare()
        from qlib_yahoo.candidate_workflow import _verify_sources
        called = []

        def verified(result):
            self.assertTrue(list((self.store.root / 'chronology' / 'holdouts').glob('*.json')))
            pending = [r for r in self.store.list_attempts() if r['kind'] == 'chronology.final']
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0]['status'], 'pending')
            called.append(True)
            return _verify_sources(result)

        with patch('qlib_yahoo.candidate_workflow._verify_sources', side_effect=verified):
            self.final(identity, frozen)
        self.assertGreaterEqual(len(called), 2)

    def test_final_cancellation_is_terminal_and_consumes_holdout(self):
        identity, trial, frozen = self.prepare()
        with self.assertRaises(KeyboardInterrupt):
            self.final(identity, frozen, cancel=lambda: True)
        final = [self.store.get(r['id']) for r in self.store.list_attempts() if r['kind'] == 'chronology.final'][0]
        self.assertEqual(final['status'], 'cancelled')
        receipt = self.store.get_result(final['result_digest'])
        self.assertEqual(receipt['attempt_ids'], [])
        self.assertEqual(self.final(identity, frozen)['access_label'], 'reused_holdout')

    def test_mid_batch_cancellation_finishes_existing_nested_attempts(self):
        identity, trial, frozen = self.prepare()
        state = {'cancel': False}

        def progress(message):
            if 'Simulating' in message:
                state['cancel'] = True

        with self.assertRaises(KeyboardInterrupt):
            self.final(identity, frozen, cancel=lambda: state['cancel'], progress=progress)
        finals = [self.store.get(r['id']) for r in self.store.list_attempts() if r['kind'] == 'chronology.final']
        self.assertEqual(finals[0]['status'], 'cancelled')
        receipt = self.store.get_result(finals[0]['result_digest'])
        self.assertTrue(receipt['batch_attempt_ids'])
        self.assertTrue(receipt['attempt_ids'])
        self.assertTrue(all(row['status'] != 'pending' for row in receipt['children']))
        self.assertEqual(ChronologyStore(self.root).get_selection(identity)['id'], frozen['selection_id'])

    def test_runtime_failure_finishes_final_and_actual_children(self):
        identity, trial, frozen = self.prepare()
        with patch.object(batch, '_run_one', return_value={'expression': 'rank(close)',
                'settings': asdict(self.settings) | self.params()['final'], 'error': 'forced numeric failure'}):
            response = self.final(identity, frozen)
        self.assertEqual(response['status'], 'failed')
        self.assertTrue(response['attempt_ids'])
        self.assertEqual(self.store.get(response['attempt_ids'][0])['status'], 'failed')
        self.assertEqual(ChronologyStore(self.root).get_selection(identity)['id'], frozen['selection_id'])

    def test_source_already_observed_final_is_not_hidden(self):
        source = batch.run_batch(self.dataset, ['rank(close)'],
            Settings(**(asdict(self.settings) | {'end': self.days[50]})), data_root=self.root, workers=1)[0]
        identity = declare_chronology(self.root, self.params(source['attempt_id']))['protocol_id']
        trial = self.trial(identity)
        frozen = freeze_chronology_selection(self.root, {'protocol_id': identity,
            'selected_ids': trial['attempt_ids'], 'decision_reason': 'source already used later returns'})
        with self.assertRaisesRegex(ValueError, 'exposed'):
            self.final(identity, frozen)
        response = evaluate_chronology_final(self.root, {'protocol_id': identity,
            'selection_id': frozen['selection_id'], 'allow_exposed': True})
        self.assertEqual(response['access_label'], 'already_exposed_holdout')

    def test_partial_sample_overlap_is_reported_but_incomplete_coverage_refused(self):
        params = self.params()
        params['samples'] = [{'id': 'overlap', 'session': self.days[21],
                              'information_start': self.days[20], 'information_end': self.days[22]}]
        declared = declare_chronology(self.root, params)
        self.assertEqual(declared['sample_split']['excluded'][0]['reason'], 'information_overlap')
        with patch.object(batch, 'run_batch', side_effect=AssertionError('purging was not applied')), \
                self.assertRaisesRegex(ValueError, 'complete session coverage'):
            self.trial(declared['protocol_id'])
        attempted = [r for r in self.store.list_attempts() if r['kind'] == 'chronology.trial']
        self.assertEqual(attempted[0]['status'], 'rejected')

    def test_full_sample_coverage_executes_noncontiguous_scoring_and_final(self):
        import numpy as np
        params = self.params()
        params['samples'] = [{'id': day, 'session': day, 'information_start': day,
                              'information_end': day} for day in self.days[1:64]]
        # Exclude an interior development observation, not just a trailing day.
        params['samples'][5]['information_end'] = self.days[22]
        declared = declare_chronology(self.root, params)
        self.assertEqual(declared['sample_split']['coverage']['status'], 'complete')
        trial = self.trial(declared['protocol_id'])
        record = self.store.get(trial['attempt_ids'][0])
        result = self.store.get_result(record['result_digest'])
        kept = [day for day in self.days[1:22] if day != self.days[6]]
        self.assertEqual(result['daily']['dates'], kept)
        expected = self.store.get_result(self.source['result_digest'])['daily']
        rows = [i for i, day in enumerate(expected['dates']) if day in kept]
        self.assertEqual(result['daily']['pnl'], [expected['pnl'][i] for i in rows])
        self.assertEqual(result['daily']['turnover'], [expected['turnover'][i] for i in rows])
        pnl = np.asarray(result['daily']['pnl'])
        self.assertAlmostEqual(result['summary']['sharpe'], float(pnl.mean()/pnl.std(ddof=1)*np.sqrt(252)))
        self.assertEqual(result['summary']['days'], 20)
        self.assertEqual(result['evaluation']['observed_count'], 21)
        self.assertTrue((self.store.root / 'chronology-cache.sqlite').is_file())
        self.assertIsNone(batch.Library(self.root / 'alphalab' / 'library.sqlite').get(result['id']))
        from qlib_yahoo.candidate_workflow import preview_candidate
        with self.assertRaisesRegex(DatasetError, 'observation-filtered'):
            preview_candidate(self.root, record['id'])
        frozen = freeze_chronology_selection(self.root, {'protocol_id': declared['protocol_id'],
            'selected_ids': [record['id']], 'decision_reason': 'supplied engineering fixture'})
        final = self.final(declared['protocol_id'], frozen)
        evaluated = self.store.get_result(self.store.get(final['attempt_ids'][0])['result_digest'])
        self.assertEqual(evaluated['daily']['dates'], self.days[43:64])
        self.assertEqual(evaluated['settings']['evaluation_sessions'], self.days[43:64])

    def test_empty_samples_do_not_claim_checked_observations(self):
        declared = declare_chronology(self.root, self.params() | {'samples': []})
        split = declared['sample_split']
        self.assertEqual(split['claim'], 'checked_supplied_intervals_only')
        self.assertEqual(split['coverage']['phases']['development']['checked_count'], 0)
        with patch.object(batch, 'run_batch', side_effect=AssertionError('unchecked execution')), \
                self.assertRaisesRegex(ValueError, 'complete session coverage'):
            self.trial(declared['protocol_id'])

    def test_view_and_preview_never_hydrate_prior_final_numeric_results(self):
        identity, trial, frozen = self.prepare()
        final = self.final(identity, frozen)
        protected = {self.store.get(i)['result_digest'] for i in
                     [final['attempt_id'], *final['attempt_ids'], *final['batch_attempt_ids']]}
        original = EvidenceStore.get_result

        def guarded(store, digest):
            if digest in protected:
                raise AssertionError('metadata inspection read final numeric evidence')
            return original(store, digest)

        with patch.object(EvidenceStore, 'get_result', guarded), \
                patch.object(batch, 'run_batch', side_effect=AssertionError('inspection dispatched')):
            preview_chronology(self.root, self.params())
            view = view_chronology(self.root, {'protocol_id': identity})
        self.assertEqual(len(view['accesses']), 1)
        self.assertEqual(view['accesses'][0]['access_label'], 'first_recorded_access')

    def test_preview_cannot_reintroduce_a_final_child_as_source(self):
        identity, trial, frozen = self.prepare()
        final = self.final(identity, frozen)
        child = self.store.get(final['attempt_ids'][0])
        original = EvidenceStore.get_result

        def guarded(store, digest):
            if digest == child['result_digest']:
                raise AssertionError('preview read a final child result')
            return original(store, digest)

        with patch.object(EvidenceStore, 'get_result', guarded), \
                self.assertRaisesRegex(ValueError, 'cannot be reintroduced'):
            preview_chronology(self.root, self.params(child['id']))

    def test_final_checks_low_level_selection_settings_before_dispatch(self):
        identity = self.declare()
        forged = json.loads(canonical_json_bytes(self.source))
        for key in ('cached', 'attempt_id', 'result_digest', 'batch_attempt_id', 'self_correlation', 'most_correlated_id'):
            forged.pop(key, None)
        forged['settings']['cost_bps'] = 8.0
        attempt = self.store.begin('lab-evaluation', {'expression': 'rank(close)', 'settings': forged['settings']},
                                   parents=[identity])
        self.store.finish(attempt['id'], 'completed', result=forged)
        frozen = ChronologyStore(self.root).freeze_selection(identity, population_ids=[attempt['id']],
            selected_ids=[attempt['id']], decision_reason='deliberately incompatible low-level fixture')
        with patch.object(batch, 'run_batch', side_effect=AssertionError('incompatible selection dispatched')), \
                self.assertRaisesRegex(ValueError, 'settings differ'):
            evaluate_chronology_final(self.root, {'protocol_id': identity, 'selection_id': frozen['id']})
        self.assertTrue((self.store.root / 'chronology' / 'holdouts').exists())

    def test_canonical_protocol_tamper_and_nonplain_json_refused(self):
        identity, trial, frozen = self.prepare()
        for bad in ({'workers': float('nan')}, {'force': Path('x')}, {'allow_exposed': 1}):
            with self.assertRaises(ValueError):
                evaluate_chronology_final(self.root, {'protocol_id': identity,
                    'selection_id': frozen['selection_id']} | bad)
        plan = ChronologyStore(self.root).get_protocol(identity)
        plan['input_lineage'] = {'user': 'reset'}
        forged = ChronologyStore(self.root).register_protocol(plan)
        with self.assertRaisesRegex(ValueError, 'binding changed'):
            view_chronology(self.root, {'protocol_id': forged['id']})
        self.assertFalse((self.store.root / 'chronology' / 'holdouts').exists())

    @unittest.skipUnless(importlib.util.find_spec('pyarrow'), 'optional PyArrow unavailable')
    def test_sec_group_recipes_are_frozen_and_new_fields_rejected(self):
        from workflow_fixture_helpers import build_sec_group_fixture
        fixture = build_sec_group_fixture(self.root / 'selected-sources')
        settings = Settings(**(asdict(fixture.settings) | {
            'start': self.days[1], 'end': self.days[21], 'warmup': 1}))
        source = fixture.record(settings=settings)
        params = self.params(source['attempt_id'])
        identity = declare_chronology(fixture.root, params)['protocol_id']
        response = submit_chronology_trial(fixture.root, {'protocol_id': identity, 'phase': 'development',
            'expressions': [fixture.expression, fixture.names['currency'] + ' / close']})
        store = EvidenceStore(fixture.root)
        chosen = [i for i in response['attempt_ids'] if store.get(i)['status'] in ('completed', 'reused')]
        rejected = [i for i in response['attempt_ids'] if store.get(i)['status'] == 'rejected']
        self.assertEqual(len(chosen), 1)
        self.assertEqual(len(rejected), 1)
        self.assertIn('frozen source recipes', store.get(rejected[0])['error'])
        frozen = freeze_chronology_selection(fixture.root, {'protocol_id': identity,
            'selected_ids': chosen, 'decision_reason': 'explicit selected SEC and dated groups'})
        final = evaluate_chronology_final(fixture.root, {'protocol_id': identity, 'selection_id': frozen['selection_id']})
        result = store.get_result(store.get(final['attempt_ids'][0])['result_digest'])
        self.assertEqual(result['dataset'], source['dataset'])
        self.assertEqual(result['sec']['bindings'], source['sec']['bindings'])
        self.assertEqual(result['classification']['source_identity'], source['classification']['source_identity'])
        self.assertEqual(result['settings'], asdict(settings) | self.params()['final'])
        self.assertEqual(result['daily']['dates'], self.days[43:64])
        self.assertEqual(final['qualification']['economic_pit'], 'not_established')


if __name__ == '__main__':
    unittest.main()
