import copy
import errno
import io
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from autoresearch.records import ResearchStore, stable_id
from qlib_yahoo.webapp.research_service import ResearchService


class ResearchServiceTests(unittest.TestCase):
    def symlink_directory(self, link, target):
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError as exc:
            if exc.errno in (errno.EPERM, errno.EACCES, errno.ENOSYS, errno.ENOTSUP) or getattr(exc, 'winerror', None) == 1314:
                self.skipTest('symlink setup unsupported: ' + str(exc))
            raise

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.service = ResearchService(self.root / 'data', self.root / 'source')
        self.addCleanup(self.service.close)
        dataset = self.root / 'data/datasets/sample'
        dataset.mkdir(parents=True)
        (dataset / 'manifest.json').write_text('{}')
        self.cfg = dict(name='Local research', backend=dict(dataset='sample', universe='TOP8', cost_bps=5,
                        development_start='2023-01-01', development_end='2023-12-31',
                        validation_start='2024-01-01', validation_end='2024-12-31', warmup=100),
                        model=dict(provider='anthropic', model='explicit-model-id', research_focus='Price changes'),
                        discovery=dict(arxiv_queries=['all:momentum'], max_results=3), policy={}, limits={})

    def test_safe_save_no_model_or_network_and_no_credential_echo(self):
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'example_private_key_123'}), patch('autoresearch.models.AnthropicModel._generate', side_effect=AssertionError('model called')):
            saved = self.service.save(self.cfg)
            self.assertEqual(self.service.config(saved['id'])['backend']['dataset'], 'sample')
            result = self.service.overview()
            self.assertTrue(result['credential_ready'])
            self.assertTrue(result['configs'][0]['ready'])
            self.assertNotIn('example_private_key_123', json.dumps(result))

    def test_paths_and_fixture_inputs_rejected(self):
        for dataset in ('../outside', '/etc', 'sample/../../outside'):
            cfg = copy.deepcopy(self.cfg)
            cfg['backend']['dataset'] = dataset
            with self.assertRaises(ValueError):
                self.service.save(cfg)
        for section, key in [('model', 'fixture_path'), ('discovery', 'fixture_sources'), ('model', 'api_key_env')]:
            cfg = copy.deepcopy(self.cfg)
            cfg[section][key] = 'anything'
            with self.assertRaises(ValueError):
                self.service.save(cfg)
        for value in ('../x', '/etc/passwd', None):
            with self.assertRaises(ValueError):
                self.service.config(value)

    def test_symlink_dataset_rejected(self):
        self.symlink_directory(self.root / 'data/datasets/escape', self.root)
        cfg = copy.deepcopy(self.cfg)
        cfg['backend']['dataset'] = 'escape'
        with self.assertRaises(ValueError):
            self.service.save(cfg)

    def test_nested_validation(self):
        cases = [('backend', 'warmup', True), ('backend', 'cost_bps', float('nan')),
                 ('backend', 'validation_start', '2022-01-01'), ('model', 'model', ''),
                 ('model', 'research_focus', 'x' * 2001), ('discovery', 'max_results', 101),
                 ('policy', 'max_drawdown', -1), ('limits', 'max_experiments', False)]
        for section, key, value in cases:
            cfg = copy.deepcopy(self.cfg)
            cfg[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.service.save(cfg)

    def test_live_start_requires_explicit_consent_and_credential(self):
        saved = self.service.save(self.cfg)
        with patch('subprocess.Popen') as spawn:
            with self.assertRaises(ValueError):
                self.service.start({'id': saved['id']})
            with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
                self.service.start({'id': saved['id'], 'allow_model': True})
            spawn.assert_not_called()

    def test_concurrency_active_config_and_stop(self):
        saved = self.service.save(self.cfg)
        class Process:
            stdout = io.StringIO('')
            code = None
            def poll(self): return self.code
            def wait(self, timeout=None):
                if timeout is None:
                    gate.wait(3)
                return self.code or 0
            def terminate(self):
                self.code = 0
                gate.set()
            def kill(self): self.terminate()
        gate = threading.Event()
        process = Process()
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'example_credential'}), patch('subprocess.Popen', return_value=process) as spawn:
            worker = self.service.start({'id': saved['id'], 'allow_model': True})
            self.assertTrue(worker['running'])
            with self.assertRaises(ValueError):
                self.service.start({'id': saved['id'], 'allow_model': True})
            with self.assertRaises(ValueError):
                self.service.save(saved)
            self.assertEqual(spawn.call_count, 1)
            args = spawn.call_args.args[0]
            self.assertIn('--allow-model', args)
            effective = json.loads(Path(args[4]).read_text())
            self.assertEqual(effective['research_root'], str(self.service.records_root))
            self.assertFalse(self.service.stop()['running'])
            self.service.save(saved)

    def test_campaign_records_enrichment_resume_and_redaction(self):
        store = ResearchStore(self.root / 'source/research/alphas')
        store.put('campaigns', {'id': 'campaign_one', 'config': {'name': 'Evidence', 'limits': {'max_experiments': 6}}, 'code': {'secret': 'hidden'}})
        store.put('sources', {'id': 'source_one', 'title': 'Paper', 'url': 'https://arxiv.org/abs/1', 'text': 'Paper evidence'})
        store.put('hypotheses', {'id': 'hyp_one', 'title': 'Continuation', 'source_ids': ['source_one'], 'expression': 'rank(close)'})
        store.put('experiments', {'id': 'exp_one', 'campaign_id': 'campaign_one', 'hypothesis_id': 'hyp_one', 'result': {'status': 'ok', 'error': '/private/user/key', 'artifacts': {'report': '/private/path'}}, 'backend_identity': {'code': 'huge'}})
        store.put('reviews', {'id': stable_id('review', {'experiment': 'exp_one', 'hypothesis': 'hyp_one', 'policy': {}}), 'experiment_id': 'exp_one', 'hypothesis_id': 'hyp_one', 'decision': 'refine'})
        runtime = self.root / 'data/autoresearch/runtime/config_one'
        runtime.mkdir(parents=True)
        (runtime / 'campaign_one.json').write_text(json.dumps({'campaign_id': 'campaign_one', 'hypotheses': ['hyp_one'], 'queue': [{'phase': 'refine'}], 'evaluations': 1, 'model_calls': 2, 'daily_calls': {'2026-10-10': 2}}))
        overview = self.service.overview()
        self.assertEqual(overview['campaigns'][0]['counts']['hypotheses'], 1)
        self.assertEqual(overview['campaigns'][0]['config_id'], 'config_one')
        detail = self.service.detail('campaign_one')
        self.assertEqual(detail['hypotheses'][0]['latest_review']['decision'], 'refine')
        self.assertEqual(detail['hypotheses'][0]['result']['status'], 'ok')
        self.assertEqual(detail['checkpoint']['queued'], 1)
        self.assertEqual(len(detail['sources']), 1)
        self.assertNotIn('/private', json.dumps(detail))
        self.assertNotIn('backend_identity', json.dumps(detail))
        self.assertNotIn('"code"', json.dumps(detail))

    def test_reused_experiment_review_stays_with_selected_policy_and_latest_attempt(self):
        store = ResearchStore(self.root / 'source/research/alphas')
        own_policy, other_policy = {'min_sharpe': 1.75}, {'min_sharpe': 2.5}
        store.put('campaigns', {'id': 'campaign_a', 'config': {'name': 'Selected', 'policy': own_policy}})
        store.put('campaigns', {'id': 'campaign_b', 'config': {'name': 'Later reuse', 'policy': other_policy}})
        store.put('hypotheses', {'id': 'hyp_shared', 'expression': 'rank(close)', 'source_ids': []})
        store.put('experiments', {'id': 'exp_shared', 'campaign_id': 'campaign_a', 'hypothesis_id': 'hyp_shared',
                                 'created_at': '2026-10-10T10:00:00', 'result': {'status': 'ok', 'metrics': {'validation': {'sharpe': 2.0}}}})
        for policy, decision, timestamp in ((own_policy, 'promote', '2026-10-10T10:01:00'), (other_policy, 'refine', '2026-10-10T11:01:00')):
            store.put('reviews', {'id': stable_id('review', {'experiment': 'exp_shared', 'hypothesis': 'hyp_shared', 'policy': policy}),
                                 'experiment_id': 'exp_shared', 'hypothesis_id': 'hyp_shared', 'decision': decision, 'created_at': timestamp})
        store.put('events', {'id': 'event_reuse', 'kind': 'duplicate_experiment', 'campaign_id': 'campaign_b',
                             'hypothesis_id': 'hyp_shared', 'reused_experiment_id': 'exp_shared'})
        selected = self.service.detail('campaign_a')
        self.assertEqual(selected['hypotheses'][0]['latest_review']['decision'], 'promote')
        self.assertEqual([r['decision'] for r in selected['reviews']], ['promote'])
        self.assertEqual(selected['campaign']['counts']['reviews'], 1)
        self.assertEqual(self.service.detail('campaign_b')['hypotheses'][0]['latest_review']['decision'], 'refine')
        # A new attempt's metrics must not inherit an older attempt's decision.
        store.put('experiments', {'id': 'exp_pending_review', 'campaign_id': 'campaign_a', 'hypothesis_id': 'hyp_shared',
                                 'created_at': '2026-10-10T12:00:00', 'result': {'status': 'error', 'error': 'Interrupted'}})
        latest = self.service.detail('campaign_a')['hypotheses'][0]
        self.assertEqual(latest['result']['status'], 'error')
        self.assertIsNone(latest['latest_review'])

    def reviewed_config(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['model'].update(provider='reviewed', model='session-reviewed', hypotheses=[{
            'title': 'Continuation', 'family': 'price-continuation', 'mechanism': 'Test past price prediction',
            'expression': 'rank(ts_delta(close, 5))', 'source_ids': ['source_paper'],
            'evidence': [{'source_id': 'source_paper', 'quote': 'past price changes predict future returns'}],
            'expected_horizon': 'one trading session', 'required_fields': ['close'],
            'rationale': 'Falsifiable daily price implementation'}], refinements=[])
        cfg['discovery'] = {'fixture_sources': [{'id': 'source_paper', 'title': 'Reviewed evidence',
            'url': 'https://arxiv.org/abs/1234.5678', 'text': 'Evidence: past price changes predict future returns.'}],
            'arxiv_queries': [], 'network_enabled': False}
        return cfg

    def test_reviewed_validation_immutable_revision_and_no_key_required(self):
        cfg = self.reviewed_config()
        with patch.dict(os.environ, {}, clear=True):
            saved = self.service.save(cfg)
            self.assertTrue(self.service.overview()['configs'][0]['ready'])
            self.assertEqual(self.service.config(saved['id'])['discovery']['fixture_sources'], cfg['discovery']['fixture_sources'])
            revised = self.service.save({**saved, 'name': 'Revision'})
            self.assertNotEqual(saved['id'], revised['id'])
            self.assertEqual(self.service.config(saved['id'])['name'], saved['name'])
        for mutation in ('quote', 'operator', 'fields', 'count', 'network'):
            invalid = self.reviewed_config()
            if mutation == 'quote': invalid['model']['hypotheses'][0]['evidence'][0]['quote'] = 'invented quotation'
            if mutation == 'operator': invalid['model']['hypotheses'][0]['expression'] = 'unknown(close)'
            if mutation == 'fields': invalid['model']['hypotheses'][0]['required_fields'] = ['cap']
            if mutation == 'count': invalid['model']['hypotheses'] *= 21
            if mutation == 'network': invalid['discovery']['network_enabled'] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.service.save(invalid)

    def test_redaction_preserves_division_expressions(self):
        expression = 'returns /rank(volume)'
        self.assertEqual(self.service._clean(expression), expression)
        self.assertEqual(self.service._clean('https://arxiv.org/abs/1234.5678'), 'https://arxiv.org/abs/1234.5678')
        self.assertNotIn('/Users/name/private', self.service._clean('failure /Users/name/private/file.txt'))

    @unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('pandas'), 'Alpha Lab runtime required')
    def test_actual_offline_worker_and_resume(self):
        self.service.source_root = Path(__file__).resolve().parents[1]
        worker = self.service.demo()
        self.assertTrue(worker['demo'])
        end = time.monotonic() + 45
        while self.service.busy() and time.monotonic() < end:
            time.sleep(.05)
        self.assertFalse(self.service.busy())
        overview = self.service.overview()
        self.assertEqual(overview['worker']['status'], 'stopped', overview['worker']['last_error'])
        campaign = next(c for c in overview['campaigns'] if c['demo'])
        self.assertTrue(campaign['demo'])
        detail = self.service.detail(campaign['id'])
        self.assertEqual(len(detail['experiments']), 3)
        self.assertEqual(len(detail['reviews']), 3)
        self.assertTrue(any(h['lineage'] for h in detail['hypotheses']))
        self.service.demo()
        while self.service.busy() and time.monotonic() < end:
            time.sleep(.05)
        self.assertFalse(self.service.busy())
        resumed = self.service.detail(campaign['id'])
        self.assertEqual(len(resumed['experiments']), 3)
        self.assertEqual(resumed['checkpoint']['evaluations'], 3)

    @unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('pandas'), 'Alpha Lab runtime required')
    def test_reviewed_worker_uses_managed_dataset_without_paid_calls(self):
        from autoresearch.backend import create_demo_dataset
        actual_root = Path(__file__).resolve().parents[1]
        self.service.source_root.mkdir(parents=True)
        for package in ('autoresearch', 'qlib_yahoo'):
            self.symlink_directory(self.service.source_root / package, actual_root / package)
        create_demo_dataset(self.service.data_root / 'datasets/reviewed-test')
        cfg = self.reviewed_config()
        cfg['backend']['dataset'] = 'reviewed-test'
        cfg['backend']['development_start'] = '2023-06-01'
        with patch.dict(os.environ, {}, clear=True):
            saved = self.service.save(cfg)
            worker = self.service.start({'id': saved['id']})
            self.assertTrue(worker['reviewed'])
            self.assertFalse(worker['demo'])
            end = time.monotonic() + 45
            while self.service.busy() and time.monotonic() < end:
                time.sleep(.05)
        self.assertFalse(self.service.busy())
        overview = self.service.overview()
        self.assertEqual(overview['worker']['status'], 'stopped', overview['worker']['last_error'])
        self.assertEqual(len(overview['campaigns']), 1)
        campaign = overview['campaigns'][0]
        self.assertFalse(campaign['demo'])
        detail = self.service.detail(campaign['id'])
        self.assertEqual(len(detail['experiments']), 1)
        self.assertEqual(detail['experiments'][0]['result']['status'], 'ok')
        self.assertTrue(detail['campaign']['model']['fixture'])
        self.assertTrue(detail['campaign']['config']['model']['reviewed'])
        runtime_cfg = json.loads((self.service.runtime_root / saved['id'] / 'campaign.json').read_text())
        self.assertEqual(runtime_cfg['model']['provider'], 'fixture')
        self.assertFalse(runtime_cfg['discovery']['network_enabled'])
        self.assertEqual(Path(runtime_cfg['research_root']), self.service.records_root)

    def test_symlink_evidence_outside_source_rejected(self):
        outside = self.root / 'outside'
        outside.mkdir()
        research = self.root / 'source/research'
        research.mkdir(parents=True)
        self.symlink_directory(research / 'alphas', outside)
        with self.assertRaises(ValueError):
            self.service.overview()


if __name__ == '__main__':
    unittest.main()
