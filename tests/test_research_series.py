"""Campaign-scoped charts from actual retained daily after-cost PnL."""
import copy
import errno
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from autoresearch.records import ResearchStore
from qlib_yahoo.webapp.research_service import ResearchService


class SeriesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.service = ResearchService(self.root / 'data', self.root / 'source'); self.addCleanup(self.service.close)
        self.store = ResearchStore(self.root / 'source/research/alphas')
        for cid in ('campaign_one', 'campaign_two'):
            self.store.put('campaigns', {'id': cid, 'config': {'name': cid}})
        self.runtime = self.service.runtime_root / 'config_one'; self.runtime.mkdir(parents=True)
        (self.runtime / 'campaign_one.json').write_text(json.dumps({'campaign_id': 'campaign_one'}))
        self.path = self.runtime / 'engine/backend/identity/evaluations/expression.json'
        self.path.parent.mkdir(parents=True)
        self.artifact = {
            'development': {'summary': {'days': 3, 'pnl': 5.0}, 'daily': {'dates': ['2023-01-02', '2023-01-03', '2023-01-04'], 'pnl': [10., -20., 15.], 'turnover': [0.1, 0.2, 0.3]}},
            'validation': {'summary': {'days': 2, 'pnl': -2.0}, 'daily': {'dates': ['2024-01-02', '2024-01-03'], 'pnl': [-3., 1.], 'turnover': [0.2, 0.1]}}}
        self.experiment = {'id': 'exp_one', 'campaign_id': 'campaign_one', 'hypothesis_id': 'hyp_one',
                           'expression': 'rank(close)', 'result': {'status': 'ok',
                           'metrics': {k: row['summary'] for k, row in self.artifact.items()},
                           'settings': {'booksize': 1000., 'development_start': '2023-01-01', 'development_end': '2023-12-31',
                                        'validation_start': '2024-01-01', 'validation_end': '2024-12-31'},
                           'artifacts': ['backend/identity/evaluations/expression.json']}}
        self.path.write_text(json.dumps(self.artifact)); self.store.put('experiments', self.experiment)

    def test_exact_sum_drawdown_dates_and_denominator(self):
        result = self.service.series('campaign_one', 'exp_one')
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['periods']['development']['cumulative'], [10, -10, 5])
        self.assertEqual(result['periods']['development']['drawdown'], [0, -20, -5])
        self.assertEqual(result['periods']['validation']['cumulative'], [-3, -2])
        self.assertEqual(result['periods']['validation']['drawdown'], [-3, -2])
        self.assertEqual(result['return_denominator'], 500)
        self.assertEqual(result['periods']['development']['dates'][0], '2023-01-02')
        self.assertNotIn(str(self.root), json.dumps(result)); self.assertNotIn('artifacts', result)

    def test_selected_campaign_authorization_and_original_reuse(self):
        with self.assertRaises(ValueError): self.service.series('campaign_two', 'exp_one')
        self.store.put('events', {'id': 'reuse_one', 'campaign_id': 'campaign_two', 'kind': 'duplicate_experiment', 'reused_experiment_id': 'exp_one'})
        result = self.service.series('campaign_two', 'exp_one')
        self.assertTrue(result['reused']); self.assertEqual(result['campaign_id'], 'campaign_two')
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['periods']['validation']['cumulative'][-1], -2)
        for cid, eid in (('../outside', 'exp_one'), ('campaign_one', '../outside'), ('campaign_one', 'other_test')):
            with self.subTest(cid=cid, eid=eid), self.assertRaises(ValueError): self.service.series(cid, eid)

    def test_failed_missing_and_malformed_series_are_explicit(self):
        failed = {**self.experiment, 'id': 'exp_failed', 'result': {'status': 'error', 'error': 'test failed'}}
        self.store.put('experiments', failed)
        self.assertEqual(self.service.series('campaign_one', 'exp_failed')['status'], 'failed')
        self.path.unlink()
        self.assertEqual(self.service.series('campaign_one', 'exp_one')['status'], 'missing')
        self.path.write_text('[]')
        self.assertEqual(self.service.series('campaign_one', 'exp_one')['status'], 'invalid')
        (self.runtime / 'campaign_one.json').unlink()
        self.assertEqual(self.service.series('campaign_one', 'exp_one')['status'], 'missing')

    def test_integrity_chronology_and_finite_values(self):
        for mutation in ('total', 'duplicate', 'holdout', 'nonfinite', 'dimensions', 'summary', 'turnover'):
            data = copy.deepcopy(self.artifact)
            row = data['validation']
            if mutation == 'total': row['daily']['pnl'][0] = 100
            if mutation == 'duplicate': row['daily']['dates'][1] = row['daily']['dates'][0]
            if mutation == 'holdout': row['daily']['dates'][1] = '2025-01-01'
            if mutation == 'nonfinite': row['daily']['pnl'][0] = float('nan')
            if mutation == 'dimensions': row['daily']['pnl'].pop()
            if mutation == 'summary': row['summary']['pnl'] = 200
            if mutation == 'turnover': row['daily']['turnover'][0] = -1
            self.path.write_text(json.dumps(data))
            with self.subTest(mutation=mutation): self.assertEqual(self.service.series('campaign_one', 'exp_one')['status'], 'invalid')

    def test_overflow_totals_are_invalid_instead_of_server_errors(self):
        data = copy.deepcopy(self.artifact)
        data['development']['daily']['pnl'] = [1e308, 1e308, -1e308]
        data['development']['summary']['pnl'] = 1e308
        self.path.write_text(json.dumps(data))
        experiment = copy.deepcopy(self.experiment); experiment['id'] = 'exp_overflow'
        experiment['result']['metrics']['development'] = data['development']['summary']
        self.store.put('experiments', experiment)
        self.assertEqual(self.service.series('campaign_one', 'exp_overflow')['status'], 'invalid')

    def test_series_not_truncated_by_generic_record_cleaner(self):
        from datetime import date, timedelta
        dates = [(date(2023, 1, 1) + timedelta(days=i)).isoformat() for i in range(350)]
        data = copy.deepcopy(self.artifact)
        data['development'] = {'summary': {'days': 350, 'pnl': 350}, 'daily': {'dates': dates, 'pnl': [1.] * 350, 'turnover': [0.] * 350}}
        self.path.write_text(json.dumps(data))
        experiment = copy.deepcopy(self.experiment); experiment['id'] = 'exp_long'
        experiment['result']['metrics']['development'] = data['development']['summary']
        self.store.put('experiments', experiment)
        result = self.service.series('campaign_one', 'exp_long')
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(len(result['periods']['development']['dates']), 350)
        self.assertEqual(result['periods']['development']['cumulative'][-1], 350)

    def test_traversal_artifacts_never_read(self):
        for i, path in enumerate(('../secret.json', str(self.root / 'secret.json'))):
            exp = copy.deepcopy(self.experiment); exp['id'] = 'exp_bad_' + str(i); exp['result']['artifacts'] = [path]
            self.store.put('experiments', exp)
            self.assertEqual(self.service.series('campaign_one', exp['id'])['status'], 'invalid')

    def test_symlink_artifacts_never_read(self):
        outside = self.root / 'outside.json'; outside.write_text(json.dumps(self.artifact))
        self.path.unlink()
        try:
            self.path.symlink_to(outside)
        except OSError as exc:
            if exc.errno in (errno.EPERM, errno.EACCES, errno.ENOSYS, errno.ENOTSUP) or getattr(exc, 'winerror', None) == 1314:
                self.skipTest('symlink setup unsupported: ' + str(exc))
            raise
        self.assertEqual(self.service.series('campaign_one', 'exp_one')['status'], 'invalid')

    @unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("pandas"),
                         "requires local numpy and pandas")
    def test_actual_demo_artifact_sums_match_all_retained_tests(self):
        from autoresearch.demo import create_demo
        from autoresearch.config import load_config
        from autoresearch.controller import Controller
        cfg = load_config(create_demo(self.service.demo_root))
        controller = Controller(cfg)
        for _ in range(30):
            outcome = controller.tick()
            if outcome['status'] in {'budget', 'complete'}: break
        detail = self.service.detail(controller.campaign_id)
        self.assertEqual(len(detail['experiments']), 3)
        for exp in detail['experiments']:
            series = self.service.series(controller.campaign_id, exp['id'])
            self.assertEqual(series['status'], 'ready')
            for period in ('development', 'validation'):
                self.assertAlmostEqual(series['periods'][period]['cumulative'][-1], exp['result']['metrics'][period]['pnl'], places=6)
