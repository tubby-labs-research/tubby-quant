"""Offline contracts for the lightweight workflow evidence store."""
from concurrent.futures import ThreadPoolExecutor
import errno
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

from qlib_yahoo import workflow_records as records
from qlib_yahoo.workflow_records import (
    EvidenceConflict, EvidenceStore, canonical_json_bytes, read_sealed_record, seal_record,
)


class WorkflowRecordsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.workflow-test-', dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name) / 'data'
        self.store = EvidenceStore(self.data)

    def begin(self, **kwargs):
        return self.store.begin('lab', {'expression': 'rank(close)'}, **kwargs)

    def path(self, collection, identity):
        return self.store.root / collection / (identity + '.json')

    def overwrite_sealed(self, path, payload):
        path.write_bytes(canonical_json_bytes(seal_record(payload)))

    def test_independent_attempt_ids_and_detached_submissions(self):
        request = {'nested': {'x': [1]}}
        first = self.store.begin('lab', request)
        second = self.store.begin('lab', request)
        self.assertNotEqual(first['id'], second['id'])
        self.assertRegex(first['id'], r'^[0-9a-f]{32}$')
        request['nested']['x'].append(2)
        first['request']['nested']['x'].append(3)
        saved = self.store.get(first['id'])
        self.assertEqual(saved['request'], {'nested': {'x': [1]}})
        self.assertEqual(saved['schema_version'], 1)
        self.assertEqual(saved['owner']['pid'], os.getpid())
        self.assertEqual(set(saved['owner']), {'pid', 'create_time', 'host'})
        if os.name == 'nt' or sys.platform.startswith('linux'):
            self.assertIsInstance(saved['owner']['create_time'], (int, float))
            self.assertGreater(saved['owner']['create_time'], 0)
        self.assertEqual(saved['status'], 'pending')
        self.assertIsNone(saved['result_digest'])

    def test_snapshot_dedup_and_detached_reads(self):
        payload = {'b': [2], 'a': 1}
        digest = self.store.add_result(payload)
        self.assertEqual(digest, hashlib.sha256(canonical_json_bytes(payload)).hexdigest())
        path = self.path('results', digest)
        original = path.stat().st_mtime_ns
        self.assertEqual(self.store.add_result({'a': 1, 'b': [2]}), digest)
        self.assertEqual(path.stat().st_mtime_ns, original)
        payload['b'].append(4)
        saved = self.store.get_result(digest)
        saved['b'].append(3)
        self.assertEqual(self.store.get_result(digest), {'a': 1, 'b': [2]})
        self.assertEqual(len(list(path.parent.iterdir())), 1)
        for value in (None, True, 'text', 1, 1.5, [1, {'x': False}]):
            self.assertEqual(self.store.get_result(self.store.add_result(value)), value)

    def test_parent_candidate_and_source_links(self):
        parent = self.begin(candidate_id='candidate-A', source={'code_sha256': 'abc'})
        child = self.store.begin('validation', {'x': 1}, parents=[parent['id']], candidate_id='candidate-A')
        self.assertEqual(child['parent_digests'][parent['id']], seal_record(parent)['sha256'])
        self.assertEqual(self.store.list_attempts(parent=parent['id'])[0]['id'], child['id'])
        summaries = self.store.list_attempts(candidate_id='candidate-A')
        self.assertEqual(len(summaries), 2)
        self.assertNotIn('request', summaries[0])
        self.assertNotIn('result', summaries[0])
        self.assertEqual(self.store.get(parent['id'])['source'], {'code_sha256': 'abc'})
        self.store.finish(parent['id'], 'completed', result={'x': 2})
        self.assertEqual(self.store.get(child['id'])['parents'], [parent['id']])

    def test_invalid_submissions_do_not_create_storage(self):
        bad = [
            ('', {}, {}), ('lab', [], {}), ('lab', {}, {'source': []}),
            ('lab', {}, {'candidate_id': ''}), ('lab', {}, {'parents': 'bad'}),
            ('lab', {}, {'parents': ['../escape']}),
            ('lab', {}, {'parents': ['a' * 32, 'a' * 32]}),
        ]
        for kind, request, kwargs in bad:
            with self.subTest(kind=kind, request=request, kwargs=kwargs), self.assertRaises(ValueError):
                self.store.begin(kind, request, **kwargs)
        with self.assertRaises(FileNotFoundError):
            self.begin(parents=['a' * 32])
        self.assertFalse(self.data.exists())

    def test_strict_json_everywhere(self):
        invalid = [Path('x'), (1, 2), {1: 'bad'}, float('nan'), float('inf'),
                   float('-inf'), {1}, b'bytes', '\ud800']
        attempt = self.begin()['id']
        for value in invalid:
            with self.subTest(value=repr(value)):
                for operation in (
                    lambda: self.store.add_result(value),
                    lambda: self.store.begin('lab', {'value': value}),
                    lambda: self.store.begin('lab', {}, source={'value': value}),
                    lambda: self.store.finish(attempt, 'failed', error=value),
                    lambda: self.store.finish(attempt, 'completed', artifacts=value),
                ):
                    with self.assertRaises(ValueError):
                        operation()
        self.assertEqual(self.store.get(attempt)['status'], 'pending')
        circular = []
        circular.append(circular)
        with self.assertRaises(ValueError):
            self.store.add_result(circular)

    def test_all_terminal_outcomes_and_identical_retry(self):
        for status in sorted(records.TERMINAL_STATUSES):
            with self.subTest(status=status):
                attempt = self.begin()['id']
                terminal = self.store.finish(attempt, status, result={'value': 1},
                                             error={'detail': 'explanation'}, artifacts=[{'path': 'report.html'}])
                raw = self.path('terminals', attempt).read_bytes()
                again = self.store.finish(attempt, status, result={'value': 1},
                                          error={'detail': 'explanation'}, artifacts=[{'path': 'report.html'}])
                self.assertEqual(terminal, again)
                self.assertEqual(self.path('terminals', attempt).read_bytes(), raw)
                self.assertEqual(self.store.get(attempt)['status'], status)
                self.assertEqual(self.store.get_result(terminal['result_digest']), {'value': 1})
                terminal['artifacts'].append('mutated')
                self.assertEqual(len(self.store.get(attempt)['artifacts']), 1)
                with self.assertRaises(EvidenceConflict):
                    self.store.finish(attempt, status, result={'value': 2})
                self.assertEqual(self.path('terminals', attempt).read_bytes(), raw)

    def test_conflict_checks_all_terminal_meaning(self):
        attempt = self.begin()['id']
        self.store.finish(attempt, 'completed', result=False, error={'message': 'a'}, artifacts=['one'])
        original = self.path('terminals', attempt).read_bytes()
        for status, result, error, artifacts in (
            ('failed', False, {'message': 'a'}, ['one']),
            ('completed', None, {'message': 'a'}, ['one']),
            ('completed', False, {'message': 'b'}, ['one']),
            ('completed', False, {'message': 'a'}, ['two']),
        ):
            with self.assertRaises(EvidenceConflict):
                self.store.finish(attempt, status, result=result, error=error, artifacts=artifacts)
            self.assertEqual(self.path('terminals', attempt).read_bytes(), original)

    def test_missing_terminal_is_pending_and_missing_attempt_explicit(self):
        attempt = self.begin()['id']
        with patch.object(Path, 'mkdir', side_effect=AssertionError('read attempted write')):
            self.assertEqual(self.store.get(attempt)['status'], 'pending')
            self.assertEqual(self.store.list_attempts()[0]['status'], 'pending')
        for operation in (lambda: self.store.get('0' * 32),
                          lambda: self.store.finish('0' * 32, 'failed'),
                          lambda: self.store.get_result('0' * 64)):
            with self.assertRaises(FileNotFoundError):
                operation()
        for status in ('pending', 'success', None, []):
            with self.assertRaises(ValueError):
                self.store.finish(attempt, status)

    def test_recover_only_explicit_false(self):
        pending = [self.begin()['id'] for _ in range(5)]
        finished = self.begin()['id']
        self.store.finish(finished, 'completed')
        owner_calls = []
        decisions = iter([True, False, None, 0, 'false'])

        def alive(owner):
            owner_calls.append(owner)
            return next(decisions)

        with patch.object(os, 'kill', side_effect=AssertionError('must not signal processes')):
            recovered = self.store.recover_pending(alive)
        self.assertEqual(len(owner_calls), 5)
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0]['status'], 'interrupted_unknown')
        self.assertEqual(sum(self.store.get(p)['status'] == 'pending' for p in pending), 4)
        self.assertEqual(self.store.get(finished)['status'], 'completed')
        self.assertEqual(self.store.recover_pending(lambda _: None), [])

    def test_recovery_callback_exception_preserves_pending(self):
        attempt = self.begin()['id']
        with self.assertRaisesRegex(RuntimeError, 'unknown owner'):
            self.store.recover_pending(lambda _: (_ for _ in ()).throw(RuntimeError('unknown owner')))
        self.assertEqual(self.store.get(attempt)['status'], 'pending')

    def test_recovery_racing_real_completion_preserves_winner(self):
        attempt = self.begin()['id']

        def callback(owner):
            self.store.finish(attempt, 'completed', result={'real': True})
            return False

        self.assertEqual(self.store.recover_pending(callback), [])
        self.assertEqual(self.store.get(attempt)['status'], 'completed')

    def test_concurrent_finishes_have_one_winner(self):
        attempt = self.begin()['id']
        barrier = threading.Barrier(8)

        def complete(index):
            competitor = EvidenceStore(self.data)
            barrier.wait(timeout=10)
            try:
                return competitor.finish(attempt, 'completed', result={'worker': index})
            except EvidenceConflict:
                return None

        with ThreadPoolExecutor(max_workers=8) as pool:
            outcomes = list(pool.map(complete, range(8)))
        winners = [outcome for outcome in outcomes if outcome is not None]
        self.assertEqual(len(winners), 1)
        self.assertEqual(self.store.get(attempt)['result_digest'], winners[0]['result_digest'])
        self.assertEqual(list(self.store.root.rglob('.pending-*')), [])

    def test_concurrent_identical_completion_and_result_dedup(self):
        attempt = self.begin()['id']
        with ThreadPoolExecutor(max_workers=6) as pool:
            outcomes = list(pool.map(lambda _: EvidenceStore(self.data).finish(
                attempt, 'reused', result={'a': [1]}), range(12)))
        self.assertTrue(all(outcome == outcomes[0] for outcome in outcomes))
        self.assertEqual(len(list((self.store.root / 'results').iterdir())), 1)

    def test_subprocess_competition(self):
        attempt = self.begin()['id']
        code = (
            'import sys; from qlib_yahoo.workflow_records import EvidenceStore, EvidenceConflict; '
            's=EvidenceStore(sys.argv[1]);\n'
            'try: s.finish(sys.argv[2], "completed", result={"worker": sys.argv[3]})\n'
            'except EvidenceConflict: sys.exit(7)\n'
        )
        processes = [subprocess.Popen([sys.executable, '-B', '-c', code, str(self.data), attempt, str(i)],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE) for i in range(3)]
        try:
            codes = []
            for process in processes:
                stdout, stderr = process.communicate(timeout=20)
                self.assertIn(process.returncode, (0, 7), stderr.decode())
                self.assertEqual(stdout, b'')
                codes.append(process.returncode)
            self.assertEqual(codes.count(0), 1)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                process.communicate(timeout=10)
        self.assertEqual(self.store.get(attempt)['status'], 'completed')

    def test_interrupted_publish_cleanup_and_retry(self):
        attempt = self.begin()['id']
        with patch.object(records.os, 'link', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.store.finish(attempt, 'failed', error='preparation')
        self.assertEqual(self.store.get(attempt)['status'], 'pending')
        self.assertEqual(list(self.store.root.rglob('.pending-*')), [])
        self.store.finish(attempt, 'failed', error='preparation')
        self.assertEqual(self.store.get(attempt)['error'], 'preparation')

    def test_process_death_before_and_after_atomic_publication(self):
        code = (
            'import os,sys; from qlib_yahoo import workflow_records as w; '
            's=w.EvidenceStore(sys.argv[1]); link=w.os.link;\n'
            'def crash(source, target):\n'
            ' if sys.argv[3] == "after": link(source, target)\n'
            ' os._exit(23)\n'
            'w.os.link=crash; s.finish(sys.argv[2], "failed", error="worker exited")\n'
        )
        for phase in ('before', 'after'):
            with self.subTest(phase=phase):
                attempt = self.begin()['id']
                outcome = subprocess.run([sys.executable, '-B', '-c', code, str(self.data), attempt, phase],
                                         capture_output=True, timeout=15)
                self.assertEqual(outcome.returncode, 23, outcome.stderr.decode(errors='replace'))
                self.assertEqual(self.store.get(attempt)['status'], 'pending' if phase == 'before' else 'failed')
                self.assertIn(attempt, [r['id'] for r in self.store.list_attempts()])
                self.assertEqual(self.store.get(attempt)['request'], {'expression': 'rank(close)'})

    def test_fsync_failure_does_not_publish_partial_evidence(self):
        with patch.object(records.os, 'fsync', side_effect=OSError('disk failure')):
            with self.assertRaisesRegex(OSError, 'disk failure'):
                self.begin()
        self.assertEqual(self.store.list_attempts(), [])
        self.assertEqual(list(self.store.root.rglob('.pending-*')), [])

    def test_unsupported_exclusive_publish_fails_closed(self):
        with patch.object(records.os, 'link', side_effect=OSError(errno.ENOTSUP, 'no hard links')):
            with self.assertRaises(OSError):
                self.begin()
        self.assertEqual(self.store.list_attempts(), [])

    def test_abandoned_temporary_is_not_a_submission(self):
        attempt = self.begin()['id']
        (self.store.root / 'submissions' / '.pending-abandoned').write_bytes(b'{partial')
        self.assertEqual([r['id'] for r in self.store.list_attempts()], [attempt])
        self.assertEqual(self.store.get(attempt)['status'], 'pending')

    def test_tampered_submission_terminal_and_result(self):
        attempt = self.begin()['id']
        terminal = self.store.finish(attempt, 'completed', result={'value': 1})
        paths = [self.path('submissions', attempt), self.path('terminals', attempt),
                 self.path('results', terminal['result_digest'])]
        for path in paths:
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(original.replace(b'1', b'2', 1))
                with self.assertRaises(ValueError):
                    self.store.get(attempt)
                with self.assertRaises(ValueError):
                    self.store.list_attempts()
                path.write_bytes(original)
        result_path = paths[-1]
        result_path.write_bytes(b'{"value":99}')
        with self.assertRaises(ValueError):
            self.store.get_result(terminal['result_digest'])
        with self.assertRaises(ValueError):
            self.store.add_result({'value': 1})
        result_path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.store.get(attempt)

    def test_parent_and_terminal_content_bindings(self):
        parent = self.begin()
        child = self.begin(parents=[parent['id']])
        parent_path = self.path('submissions', parent['id'])
        changed = dict(parent, request={'changed': True})
        self.overwrite_sealed(parent_path, changed)
        with self.assertRaisesRegex(ValueError, 'parent submission content'):
            self.store.get(child['id'])
        self.overwrite_sealed(parent_path, parent)
        terminal = self.store.finish(child['id'], 'completed')
        terminal['submission_digest'] = '0' * 64
        self.overwrite_sealed(self.path('terminals', child['id']), terminal)
        with self.assertRaisesRegex(ValueError, 'submission digest'):
            self.store.get(child['id'])

    def test_invalid_schema_versions_and_shapes(self):
        attempt = self.begin()
        path = self.path('submissions', attempt['id'])
        for changed in (dict(attempt, schema_version=2), dict(attempt, schema_version=True),
                        dict(attempt, extra=1), dict(attempt, owner={}),
                        dict(attempt, request=[]), dict(attempt, created_at='yesterday'),
                        dict(attempt, id='0' * 32), dict(attempt, parent_digests={'0' * 32: '0' * 64})):
            with self.subTest(changed=changed):
                self.overwrite_sealed(path, changed)
                with self.assertRaises(ValueError):
                    self.store.get(attempt['id'])
        self.overwrite_sealed(path, attempt)
        terminal = self.store.finish(attempt['id'], 'failed')
        for changed in (dict(terminal, schema_version=9), dict(terminal, status=[]),
                        dict(terminal, result_digest='../escape')):
            self.overwrite_sealed(self.path('terminals', attempt['id']), changed)
            with self.assertRaises(ValueError):
                self.store.get(attempt['id'])

    def test_seal_helpers_reject_noncanonical_duplicate_and_unsealed_json(self):
        path = Path(self.temp.name) / 'sealed.json'
        payload = {'source_hash': 'a' * 64}
        self.overwrite_sealed(path, payload)
        self.assertEqual(read_sealed_record(path), payload)
        for raw in (b'{}', b'{"payload":{},"payload":{},"sha256":"bad"}',
                    json.dumps(seal_record(payload), indent=2).encode(), b'{"x":NaN}', b'\xff'):
            path.write_bytes(raw)
            with self.assertRaises(ValueError):
                read_sealed_record(path)

    def test_traversal_invalid_ids_and_invalid_limits(self):
        for identity in ('../escape', '/absolute', 'x/y', 'x\\y', '', 'A' * 32, '0' * 31, '0' * 32 + ':x'):
            for operation in (lambda: self.store.get(identity),
                              lambda: self.store.get_result(identity),
                              lambda: self.store.list_attempts(parent=identity)):
                with self.assertRaises(ValueError):
                    operation()
        with self.assertRaises(ValueError):
            EvidenceStore(self.data / '..' / 'elsewhere')
        for limit in (-1, True, 1.5, None):
            with self.assertRaises(ValueError):
                self.store.list_attempts(limit=limit)
        self.assertFalse(self.data.exists())

    def test_reparse_flag_on_ancestor_rejected(self):
        self.begin()
        original = Path.lstat

        def reparse(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            if path == self.data:
                return types.SimpleNamespace(st_mode=result.st_mode, st_file_attributes=0x400)
            return result

        with patch.object(Path, 'lstat', reparse):
            with self.assertRaisesRegex(ValueError, 'reparse'):
                self.store.list_attempts()
            with self.assertRaisesRegex(ValueError, 'reparse'):
                self.begin()

    def test_reparse_flag_on_record_rejected(self):
        attempt = self.begin()['id']
        original = Path.lstat
        target = self.path('submissions', attempt)

        def reparse(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            if path == target:
                return types.SimpleNamespace(st_mode=result.st_mode, st_file_attributes=0x400)
            return result

        with patch.object(Path, 'lstat', reparse):
            with self.assertRaisesRegex(ValueError, 'reparse'):
                self.store.get(attempt)

    def test_symlink_file_and_ancestor_rejected(self):
        attempt = self.begin()['id']
        outside = Path(self.temp.name) / 'outside'
        outside.mkdir()
        link = outside / 'data-link'
        try:
            link.symlink_to(self.data, target_is_directory=True)
        except OSError as exc:
            if exc.errno in (errno.EPERM, errno.EACCES, errno.ENOSYS, errno.ENOTSUP) or getattr(exc, 'winerror', None) == 1314:
                self.skipTest('symlink privilege unavailable; reparse-flag contract tested separately')
            raise
        with self.assertRaises(ValueError):
            EvidenceStore(link)
        terminal = self.path('terminals', attempt)
        terminal.parent.mkdir()
        terminal.symlink_to(outside / 'missing.json')
        with self.assertRaises(ValueError):
            self.store.get(attempt)
        with self.assertRaises(ValueError):
            self.store.finish(attempt, 'completed')
        self.assertFalse((outside / 'missing.json').exists())

    def test_windows_junction_ancestor_rejected(self):
        if os.name != 'nt':
            self.skipTest('Windows junction contract')
        self.begin()
        junction = Path(self.temp.name) / 'junction'
        result = subprocess.run(['cmd', '/c', 'mklink', '/J', str(junction), str(self.data)],
                                capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
        try:
            with self.assertRaisesRegex(ValueError, 'reparse'):
                EvidenceStore(junction)
        finally:
            junction.rmdir()
        # Replacing an internal collection after construction is also rejected.
        collection = self.store.root / 'results'
        result = subprocess.run(['cmd', '/c', 'mklink', '/J', str(collection), str(self.data)],
                                capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
        try:
            with self.assertRaisesRegex(ValueError, 'reparse'):
                self.store.add_result({'x': 1})
        finally:
            collection.rmdir()

    def test_no_data_browsing_never_creates_directories_or_walks_results(self):
        with patch.object(Path, 'mkdir', side_effect=AssertionError('read created directories')), \
                patch.object(Path, 'iterdir', side_effect=AssertionError('unnecessary scan')):
            store = EvidenceStore(self.data)
            for _ in range(10):
                self.assertEqual(store.list_attempts(), [])
            self.assertEqual(store.recover_pending(lambda _: False), [])
            with self.assertRaises(FileNotFoundError):
                store.get('0' * 32)
        self.assertFalse(self.data.exists())

    def test_stable_listing_filters_and_limit(self):
        with patch.object(records, '_now', return_value='2026-10-20T12:00:00+00:00'):
            attempts = [self.begin(candidate_id='same')['id'] for _ in range(5)]
        expected = sorted(attempts, reverse=True)
        self.assertEqual([r['id'] for r in self.store.list_attempts(limit=3)], expected[:3])
        self.assertEqual(self.store.list_attempts(limit=0), [])
        self.assertEqual(self.store.list_attempts(candidate_id='different'), [])
        self.assertEqual(self.store.list_attempts(), self.store.list_attempts())

    def test_import_is_dependency_free_and_does_not_evaluate_expressions(self):
        code = (
            'import sys; import qlib_yahoo.workflow_records; '
            'assert not ({"numpy", "pandas", "qlib", "sqlite3", "autoresearch.records"} & set(sys.modules))'
        )
        subprocess.run([sys.executable, '-B', '-c', code], check=True, timeout=15)
        attempt = self.store.begin('lab', {'expression': '__import__("os").remove("anything")'})
        self.assertEqual(self.store.get(attempt['id'])['request'], attempt['request'])


if __name__ == '__main__':
    unittest.main()
