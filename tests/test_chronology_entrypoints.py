"""Chronology contracts through real loopback HTTP, CLI and isolated Job workers."""
import http.client
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from test_workflow_chain import make_snapshot
NUMERICAL = bool(importlib.util.find_spec('numpy') and importlib.util.find_spec('pandas'))
if NUMERICAL:
    from qlib_yahoo.alphalab import batch
from qlib_yahoo.alphalab.settings import Settings
from qlib_yahoo.chronology import ChronologyStore
from qlib_yahoo.chronology_workflow import preview_chronology
from qlib_yahoo.jobs import Job, JobCancelled, _run_request
from qlib_yahoo.webapp.server import start_server
from qlib_yahoo.webapp.service import ConsoleService
from qlib_yahoo.workflow_records import EvidenceStore


@unittest.skipUnless(NUMERICAL, 'optional numpy/pandas environment')
class ChronologyEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.chrono-entry-', dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        dataset, days = make_snapshot(self.root)
        self.days = days.strftime('%Y-%m-%d').tolist()
        settings = Settings(universe='TOP6', start=self.days[1], end=self.days[21],
                            warmup=1, truncation=.2, cost_bps=5)
        self.source = batch.run_batch(dataset, ['rank(close)'], settings,
                                     data_root=self.root, workers=1)[0]
        self.store = EvidenceStore(self.root)
        self.service = ConsoleService(self.root)
        self.server, thread = start_server(self.service)
        self.addCleanup(self.server.server_close)
        self.addCleanup(thread.join, 5)
        self.addCleanup(self.server.shutdown)
        self.headers = {'Cookie': 'tqc=' + self.server.token, 'X-Token': self.server.token,
                        'Origin': self.server.origin, 'Content-Type': 'application/json'}

    def params(self):
        return {'source_attempt_id': self.source['attempt_id'],
                'development': {'start': self.days[1], 'end': self.days[21]},
                'validation': {'start': self.days[22], 'end': self.days[42]},
                'final': {'start': self.days[43], 'end': self.days[63]},
                'selection_rule': {'metric': 'declared_net_metric', 'rule': 'supplied candidates only'}}

    def request(self, method, path, params=None, *, raw=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_address[1], timeout=15)
        try:
            body = raw if raw is not None else (json.dumps(params).encode() if params is not None else None)
            # Single send preserves early authorization/error replies on Windows.
            outgoing = []
            with patch.object(connection, 'send', side_effect=outgoing.append):
                connection.request(method, path, body=body, headers=self.headers if headers is None else headers)
            connection.send(b''.join(outgoing))
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def post(self, action, params):
        return self.request('POST', '/api/workflow-chronology-' + action, params)

    def wait(self, started):
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            status, record = self.request('GET', '/api/job?id=' + started['id'])
            self.assertEqual(status, 200, record)
            if record['finished'] is not None:
                return record
            time.sleep(.04)
        self.service.cancel()
        self.fail('tiny chronology job did not finish: ' + str(record))

    def declare(self, params=None):
        status, response = self.post('declare', self.params() if params is None else params)
        self.assertEqual(status, 200, response)
        return response

    def trial(self, identity, expressions=None):
        status, response = self.post('trial', {'protocol_id': identity, 'phase': 'development',
            'expressions': ['rank(close)'] if expressions is None else expressions})
        self.assertEqual(status, 200, response)
        return self.wait(response)

    def freeze(self, identity, trial):
        chosen = [i for i in trial['receipt']['attempt_ids'] if self.store.get(i)['status'] in ('completed', 'reused')]
        status, response = self.post('freeze', {'protocol_id': identity, 'selected_ids': chosen,
                                               'decision_reason': 'human review of supplied trials'})
        self.assertEqual(status, 200, response)
        return response

    def cli(self, action, params, *, root=None):
        path = self.root / ('request-' + action + '.json')
        path.write_text(json.dumps(params), encoding='utf-8')
        return subprocess.run([sys.executable, '-B', '-m', 'qlib_yahoo', '--data-root', str(root or self.root),
            'workflow', 'chronology-' + action, '--request', str(path)], capture_output=True,
            text=True, encoding='utf-8', timeout=45)

    def test_real_http_full_chain_receipts_parentage_and_repeated_final(self):
        before = len(self.store.list_attempts())
        with patch.object(batch, 'run_batch', side_effect=AssertionError('preview executed engine')), \
                patch.object(batch, 'Workspace', side_effect=AssertionError('preview loaded panel')):
            status, preview = self.post('preview', self.params())
        self.assertEqual(status, 200, preview)
        self.assertEqual(len(self.store.list_attempts()), before)
        declared = self.declare()
        protocol_id = declared['protocol_id']
        self.assertEqual(self.store.get(declared['submission_attempt_id'])['status'], 'completed')
        trial = self.trial(protocol_id, ['rank(close)', 'rank('])
        self.assertEqual(trial['state'], 'completed', trial)
        self.assertEqual(trial['result_kind'], 'chronology')
        self.assertEqual(trial['run_path'], trial['result'])
        self.assertEqual(Path(trial['result']).parent, self.store.root / 'results')
        self.assertEqual(len(trial['receipt']['attempt_ids']), 2)
        outer = self.store.get(trial['attempt_id'])
        self.assertEqual(outer['kind'], 'console-chronology-trial')
        worker = next(self.store.get(r['id']) for r in self.store.list_attempts(parent=outer['id'])
                      if r['kind'] == 'worker-chronology-trial')
        actual = self.store.get(trial['receipt']['attempt_id'])
        self.assertIn(worker['id'], actual['parents'])
        frozen = self.freeze(protocol_id, trial)
        self.assertEqual(self.store.get(frozen['submission_attempt_id'])['status'], 'completed')
        self.assertEqual(frozen['selection']['exposure_reasons'], [])
        labels = []
        for _ in range(2):
            status, started = self.post('final', {'protocol_id': protocol_id, 'selection_id': frozen['selection_id']})
            self.assertEqual(status, 200, started)
            final = self.wait(started)
            self.assertEqual(final['state'], 'completed', final)
            labels.append(final['receipt']['access_label'])
            self.assertEqual(final['receipt']['phase'], 'final')
            numeric = self.store.get(final['receipt']['attempt_ids'][0])
            payload = self.store.get_result(numeric['result_digest'])
            self.assertEqual(payload['daily']['dates'], self.days[43:64])
            final_attempt = self.store.get(final['receipt']['attempt_id'])
            worker_final = next(self.store.get(p) for p in final_attempt['parents']
                                if self.store.get(p)['kind'] == 'worker-chronology-final')
            self.assertIn(final['attempt_id'], worker_final['parents'])
        self.assertEqual(labels, ['first_recorded_access', 'reused_holdout'])
        status, view = self.post('view', {'protocol_id': protocol_id})
        self.assertEqual(status, 200, view)
        self.assertEqual(len(view['accesses']), 2)
        self.assertEqual(len(ChronologyStore(self.root)._population(protocol_id)), 4)

    def test_http_invalid_governance_and_submitted_trial_are_retained(self):
        status, response = self.post('declare', {})
        self.assertEqual(status, 400, response)
        bad = next(r for r in self.store.list_attempts() if r['kind'] == 'console-chronology-declare')
        self.assertEqual(bad['status'], 'rejected')
        declared = self.declare()
        status, started = self.post('trial', {'protocol_id': declared['protocol_id'], 'phase': 'final', 'expressions': ['rank(close)']})
        self.assertEqual(status, 200, started)
        job = self.wait(started)
        self.assertEqual(job['state'], 'failed', job)
        actual = [self.store.get(r['id']) for r in self.store.list_attempts() if r['kind'] == 'chronology.trial']
        self.assertEqual(len(actual), 1)
        self.assertEqual(actual[0]['status'], 'rejected')
        self.assertEqual(self.store.get(job['attempt_id'])['status'], 'failed')
        self.assertFalse((self.store.root / 'chronology' / 'holdouts').exists())

    def test_http_metadata_never_dispatches_or_inspects_final_results(self):
        declared = self.declare()
        before = len(self.store.list_attempts())
        with patch.object(self.service, 'job_factory', side_effect=AssertionError('metadata spawned Job')), \
                patch.object(batch, 'run_batch', side_effect=AssertionError('metadata evaluated')), \
                patch('qlib_yahoo.candidate_workflow._verify_sources', side_effect=AssertionError('metadata scanned source')):
            self.assertEqual(self.post('preview', self.params())[0], 200)
            self.assertEqual(self.post('view', {'protocol_id': declared['protocol_id']})[0], 200)
        self.assertEqual(len(self.store.list_attempts()), before)

    def test_http_auth_strict_json_and_existing_routes_preserved(self):
        self.assertEqual(self.request('POST', '/api/workflow-chronology-declare', self.params(), headers={})[0], 403)
        for raw in (b'{"source_attempt_id":"a","source_attempt_id":"b"}', b'{"x":NaN}', b'[]'):
            self.assertEqual(self.request('POST', '/api/workflow-chronology-declare', raw=raw)[0], 400)
        self.assertEqual(self.request('GET', '/api/workflow-chronology-view')[0], 404)
        with patch.object(self.service, 'workflow_portfolio_preview', return_value={'preserved': True}):
            self.assertEqual(self.request('POST', '/api/workflow-portfolio-preview', {})[1], {'preserved': True})
        status, attempts = self.request('GET', '/api/workflow-attempts')
        self.assertEqual(status, 200)
        self.assertTrue(any(r['id'] == self.source['attempt_id'] for r in attempts))

    def test_http_mask_refusal_is_not_silently_bypassed_in_worker(self):
        params = self.params()
        params['samples'] = [{'id': 'overlap', 'session': self.days[21],
                              'information_start': self.days[20], 'information_end': self.days[22]}]
        identity = self.declare(params)['protocol_id']
        job = self.trial(identity)
        self.assertEqual(job['state'], 'failed', job)
        self.assertIn('complete session coverage', job['error'])
        actual = next(self.store.get(r['id']) for r in self.store.list_attempts() if r['kind'] == 'chronology.trial')
        self.assertEqual(actual['status'], 'rejected')
        self.assertEqual(self.store.get_result(actual['result_digest'])['attempt_ids'], [])

    def test_real_http_job_cancellation_cooperatively_seals_worker(self):
        case = self
        cancelled = []

        class CancelOnStart(Job):
            def run(self, on_line):
                def line(message):
                    if message.startswith('Chronology trial:'):
                        status, body = case.request('POST', '/api/cancel', {})
                        case.assertEqual(status, 200)
                        cancelled.append(body['cancelled'])
                    on_line(message)
                return super().run(line)

        self.service.job_factory = CancelOnStart
        identity = self.declare()['protocol_id']
        result = self.trial(identity)
        self.assertEqual(cancelled, [True])
        self.assertEqual(result['state'], 'cancelled', result)
        self.assertEqual(self.store.get(result['attempt_id'])['status'], 'cancelled')
        worker = next(self.store.get(r['id']) for r in self.store.list_attempts() if r['kind'] == 'worker-chronology-trial')
        self.assertEqual(worker['status'], 'cancelled')
        for row in self.store.list_attempts():
            self.assertNotEqual(row['status'], 'pending', row)

    def test_real_cli_same_contract_all_six_commands_and_receipt(self):
        preview = self.cli('preview', self.params())
        self.assertEqual(preview.returncode, 0, preview.stderr)
        self.assertEqual(json.loads(preview.stdout)['claim'], 'declaration_preview_only_no_final_evaluation')
        declared = self.cli('declare', self.params())
        self.assertEqual(declared.returncode, 0, declared.stderr)
        protocol_id = json.loads(declared.stdout)['protocol_id']
        trial = self.cli('trial', {'protocol_id': protocol_id, 'phase': 'development', 'expressions': ['rank(close)']})
        self.assertEqual(trial.returncode, 0, trial.stderr)
        receipt = json.loads(trial.stdout)
        self.assertEqual(receipt['result_kind'], 'chronology')
        self.assertIn('Chronology trial:', trial.stderr)
        self.assertEqual(self.store.get(receipt['attempt_id'])['status'], 'completed')
        frozen = self.cli('freeze', {'protocol_id': protocol_id,
            'selected_ids': receipt['receipt']['attempt_ids'], 'decision_reason': 'same CLI governance contract'})
        self.assertEqual(frozen.returncode, 0, frozen.stderr)
        selection_id = json.loads(frozen.stdout)['selection_id']
        final = self.cli('final', {'protocol_id': protocol_id, 'selection_id': selection_id})
        self.assertEqual(final.returncode, 0, final.stderr)
        outcome = json.loads(final.stdout)
        self.assertEqual(outcome['receipt']['access_label'], 'first_recorded_access')
        self.assertTrue(Path(outcome['run_path']).is_file())
        viewed = self.cli('view', {'protocol_id': protocol_id})
        self.assertEqual(viewed.returncode, 0, viewed.stderr)
        self.assertEqual(len(json.loads(viewed.stdout)['accesses']), 1)

    def test_cli_invalid_submission_is_retained_and_json_is_strict(self):
        identity = self.declare()['protocol_id']
        trial = self.cli('trial', {'protocol_id': identity, 'phase': 'wrong', 'expressions': ['rank(close)']})
        self.assertEqual(trial.returncode, 1, trial.stderr)
        self.assertIn('development or validation', trial.stderr)
        records = [r for r in self.store.list_attempts() if r['kind'] == 'console-chronology-trial']
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['status'], 'failed')
        bad = self.cli('preview', self.params() | {'input_lineage': {'reset': True}})
        self.assertEqual(bad.returncode, 1)

    def test_direct_cli_and_http_scope_refuse_recorded_external_dataset(self):
        other = self.root / 'other-root'
        other.mkdir()
        shutil.copytree(self.store.root, other / 'workflow')
        with self.assertRaisesRegex(ValueError, 'outside this data root'):
            preview_chronology(other, self.params())
        service = ConsoleService(other)
        with self.assertRaisesRegex(ValueError, 'outside this data root'):
            service.workflow_chronology_preview(self.params())
        outcome = self.cli('preview', self.params(), root=other)
        self.assertEqual(outcome.returncode, 1)
        self.assertIn('outside this data root', outcome.stderr)

    def test_worker_kinds_cancel_marker_receipt_and_no_backtest_assumption(self):
        from qlib_yahoo.jobs import KINDS
        for kind in ('chronology-trial', 'chronology-final'):
            self.assertIn(kind, KINDS)
            self.assertIsNotNone(Job({'kind': kind, 'data_root': str(self.root), 'parameters': {}})._cancel_file)
        identity = self.declare()['protocol_id']
        cancel_file = self.root / 'cancel-marker'
        cancel_file.write_text('cancel')
        with self.assertRaises(KeyboardInterrupt):
            _run_request({'kind': 'chronology-trial', 'data_root': str(self.root),
                '_cancel_file': str(cancel_file), 'parameters': {'protocol_id': identity,
                'phase': 'development', 'expressions': ['rank(close)']}})
        self.assertTrue(any(r['kind'] == 'worker-chronology-trial' and r['status'] == 'cancelled'
                            for r in self.store.list_attempts()))

    def test_cli_keyboard_interrupt_uses_normal_service_cancel(self):
        from qlib_yahoo.workflow_cli import _chronology_result
        service = Mock()
        service.start_chronology_trial.return_value = {'id': 'job'}
        service.job.side_effect = [KeyboardInterrupt(), {'log': [], 'log_index': 0,
            'finished': 1, 'state': 'cancelled', 'error': 'cancelled fixture'}]
        with self.assertRaises(JobCancelled):
            _chronology_result(service, 'trial', {})
        service.cancel.assert_called_once()

    def test_job_terminal_state_is_published_only_after_evidence_seal(self):
        entered, release = threading.Event(), threading.Event()
        original_finish = EvidenceStore.finish
        store = self.store

        class ReceiptOnlyJob:
            def __init__(self, request):
                self.request = request

            def run(self, _line):
                digest = store.add_result({'schema_version': 1, 'phase': 'development', 'status': 'completed'})
                return store.root / 'results' / (digest + '.json')

            def cancel(self):
                return False

        def delayed(evidence, identity, status, **kwargs):
            if evidence.get(identity, verify_result=False)['kind'] == 'console-chronology-trial':
                entered.set()
                if not release.wait(10):
                    raise RuntimeError('test seal was not released')
            return original_finish(evidence, identity, status, **kwargs)

        self.service.job_factory = ReceiptOnlyJob
        with patch.object(EvidenceStore, 'finish', delayed):
            started = self.service.start_chronology_trial({'protocol_id': 'fixture', 'phase': 'development', 'expressions': ['close']})
            try:
                self.assertTrue(entered.wait(5))
                self.assertEqual(self.service.job(started['id'])['state'], 'running')
                self.assertEqual(self.store.get(started['attempt_id'])['status'], 'pending')
            finally:
                release.set()
            ended = self.wait(started)
        self.assertEqual(ended['state'], 'completed')
        self.assertEqual(self.store.get(started['attempt_id'])['status'], 'completed')


class CancellationRepairTests(unittest.TestCase):
    """Forced cancellation uses confined manifests, never marker paths alone."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.cancel-repair-', dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.dataset = self.root / 'datasets' / 'synthetic'
        self.dataset.mkdir(parents=True)

    def make_run(self, name, status='running', dataset=None):
        path = self.root / 'runs' / name
        path.mkdir(parents=True)
        manifest = path / 'manifest.json'
        manifest.write_text(json.dumps({'status': status, 'dataset': str(dataset or self.dataset), 'alpha': '$close'}))
        (path / 'history.tmp').write_bytes(b'recoverable partial history')
        return path, manifest

    def test_candidate_portfolio_repair_without_request_dataset_and_preserve_completed(self):
        from qlib_yahoo.jobs import _record_cancelled
        for kind in ('candidate-validation', 'portfolio-validation'):
            with self.subTest(kind=kind):
                run, manifest = self.make_run(kind)
                payload = json.dumps({'kind': kind, 'data_root': str(self.root), 'parameters': {'attempt_id': 'selected'}})
                _record_cancelled(payload, str(run), None)
                saved = json.loads(manifest.read_text())
                self.assertEqual(saved['status'], 'failed')
                self.assertEqual(saved['error']['type'], 'JobCancelled')
                self.assertEqual((run / 'history.tmp').read_bytes(), b'recoverable partial history')
                complete, complete_manifest = self.make_run(kind + '-sealed', 'completed')
                original = complete_manifest.read_bytes()
                _record_cancelled(payload, str(complete), None)
                self.assertEqual(complete_manifest.read_bytes(), original)

    def test_candidate_portfolio_refuse_arbitrary_run_and_manifest_dataset_paths(self):
        from qlib_yahoo.jobs import _record_cancelled
        outside = self.root / 'outside' / 'run'
        outside.mkdir(parents=True)
        outside_manifest = outside / 'manifest.json'
        outside_manifest.write_text(json.dumps({'status': 'running', 'dataset': str(self.dataset)}))
        nested, nested_manifest = self.make_run('nested/run')
        wrong, wrong_manifest = self.make_run('wrong-source', dataset=self.root / 'elsewhere' / 'dataset')
        missing, missing_manifest = self.make_run('missing-source')
        missing_manifest.write_text('{"status":"running"}')
        for kind in ('candidate-validation', 'portfolio-validation'):
            payload = json.dumps({'kind': kind, 'data_root': str(self.root), 'parameters': {}})
            for path, manifest in ((outside, outside_manifest), (nested, nested_manifest),
                                   (wrong, wrong_manifest), (missing, missing_manifest)):
                before = manifest.read_bytes()
                _record_cancelled(payload, str(path), None)
                self.assertEqual(manifest.read_bytes(), before)
            _record_cancelled(payload, str(self.root / 'runs' / '..' / 'outside' / 'run'), None)
        self.assertEqual(json.loads(outside_manifest.read_text())['status'], 'running')

    def test_original_backtest_dataset_guard_remains_when_receipt_root_differs(self):
        from qlib_yahoo.jobs import _record_cancelled
        receipt_root = self.root / 'receipts'
        outside_run = receipt_root / 'runs' / 'not-the-dataset-run'
        outside_run.mkdir(parents=True)
        outside_manifest = outside_run / 'manifest.json'
        outside_manifest.write_text(json.dumps({'status': 'running', 'dataset': str(self.dataset)}))
        run, manifest = self.make_run('real-dataset-run')
        payload = json.dumps({'kind': 'backtest', 'data_root': str(receipt_root),
                              'dataset': str(self.dataset), 'alpha': '$close'})
        _record_cancelled(payload, str(outside_run), None)
        _record_cancelled(payload, str(run), None)
        self.assertEqual(json.loads(outside_manifest.read_text())['status'], 'running')
        self.assertEqual(json.loads(manifest.read_text())['status'], 'failed')

    def test_chronology_lab_receipts_and_unrelated_batch_markers_are_never_repaired(self):
        from qlib_yahoo.jobs import _record_cancelled
        run, manifest = self.make_run('unrelated')
        lab_receipt = self.root / 'alphalab' / 'batches' / 'saved.json'
        lab_receipt.parent.mkdir(parents=True)
        lab_receipt.write_text('{"status":"running","immutable_lab_evidence":true}')
        batch_receipt = self.root / 'workbench' / 'batches' / 'saved.json'
        batch_receipt.parent.mkdir(parents=True)
        batch_receipt.write_text('{"status":"running","runs":[]}')
        originals = {p: p.read_bytes() for p in (manifest, lab_receipt, batch_receipt)}
        for kind in ('chronology-trial', 'chronology-final', 'alpha-batch'):
            for receipt in (lab_receipt, batch_receipt):
                _record_cancelled(json.dumps({'kind': kind, 'data_root': str(self.root)}), str(run), str(receipt))
        for kind in ('candidate-validation', 'portfolio-validation'):
            _record_cancelled(json.dumps({'kind': kind, 'data_root': str(self.root)}), None, str(lab_receipt))
        for path, before in originals.items():
            self.assertEqual(path.read_bytes(), before)

    def test_job_forced_cancel_with_active_run_remains_cancelled_not_keyerror(self):
        from qlib_yahoo.jobs import ACTIVE_RUN
        for kind in ('candidate-validation', 'portfolio-validation'):
            for status in ('running', 'completed'):
                with self.subTest(kind=kind, status=status):
                    run, manifest = self.make_run(kind + '-' + status, status)
                    original = manifest.read_bytes()
                    job = Job({'kind': kind, 'data_root': str(self.root), 'parameters': {}})

                    class KilledWorker:
                        def __init__(self, *_args, **_kwargs):
                            self.stdout = iter([ACTIVE_RUN + json.dumps(str(run)) + '\n', 'worker stopped\n'])

                        def wait(self):
                            return -15

                    def stopped(_line):
                        job.cancelled = True

                    with patch('qlib_yahoo.jobs.subprocess.Popen', KilledWorker), self.assertRaises(JobCancelled):
                        job.run(stopped)
                    if status == 'completed':
                        self.assertEqual(manifest.read_bytes(), original)
                    else:
                        self.assertEqual(json.loads(manifest.read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
