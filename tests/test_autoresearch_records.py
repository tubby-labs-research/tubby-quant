from contextlib import closing
import errno
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from qlib_yahoo.alphalab.fastexpr import ExpressionError
from autoresearch.records import ResearchStore, canonical_expression, stable_id


class RecordsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = ResearchStore(self.temp.name)

    def test_immutable_detached_idempotent_records(self):
        value = {'id': 'idea_1', 'nested': {'a': [1]}}
        self.assertEqual(self.store.put('hypotheses', value), value)
        value['nested']['a'].append(2)
        self.assertEqual(self.store.get('hypotheses', 'idea_1')['nested']['a'], [1])
        self.store.put('hypotheses', {'nested': {'a': [1]}, 'id': 'idea_1'})
        with self.assertRaisesRegex(ValueError, 'immutable'):
            self.store.put('hypotheses', value)

    def test_safe_paths_and_json_values(self):
        for identity in ('../escape', '/absolute', 'x/y', '', '.hidden'):
            with self.assertRaises(ValueError):
                self.store.put('sources', {'id': identity})
        with self.assertRaises(ValueError):
            self.store.list('../sources')
        for value in (float('nan'), float('inf'), {1: 'bad'}, (1, 2)):
            with self.assertRaises(ValueError):
                self.store.put('sources', {'id': 'safe', 'value': value})

    def test_symlink_collection_rejected(self):
        try:
            (self.store.root / 'sources').symlink_to(Path(self.temp.name).parent, target_is_directory=True)
        except OSError as exc:
            if exc.errno in (errno.EPERM, errno.EACCES, errno.ENOSYS, errno.ENOTSUP) or getattr(exc, 'winerror', None) == 1314:
                self.skipTest('symlink setup unsupported: ' + str(exc))
            raise
        with self.assertRaises(ValueError):
            self.store.get('sources', 'safe')

    def test_canonical_ast_preserves_semantics(self):
        self.assertEqual(canonical_expression('rank((close)) + 1.00;'), canonical_expression(' rank(close)+1 '))
        self.assertNotEqual(canonical_expression('close / open'), canonical_expression('open / close'))
        self.assertNotEqual(canonical_expression('f("a b")'), canonical_expression('f("ab")'))
        self.assertNotEqual(canonical_expression('close + open'), canonical_expression('close - open'))
        with self.assertRaises(ExpressionError):
            canonical_expression('close$')
        self.assertEqual(stable_id('idea', {'a': 1, 'b': 2}), stable_id('idea', {'b': 2, 'a': 1}))

    def test_reuse_rejection_but_not_operational_error(self):
        self.store.put('experiments', {'id': 'error', 'experiment_key': 'key', 'result': {'status': 'error'}})
        self.assertIsNone(self.store.find_experiment('key'))
        evaluated = {'id': 'rejected', 'experiment_key': 'key', 'result': {'status': 'ok'}, 'decision': 'reject'}
        self.store.put('experiments', evaluated)
        self.assertEqual(self.store.find_experiment('key'), evaluated)
        self.assertIsNone(self.store.find_experiment('other_data_or_cost'))

    def test_merge_primary_json_and_disposable_index(self):
        self.store.put('sources', {'id': 'local'})
        index = self.store.rebuild_index()
        with tempfile.TemporaryDirectory() as peer:
            other = ResearchStore(peer)
            other.put('sources', {'id': 'peer'})
            shutil.copy(Path(peer) / 'sources/peer.json', self.store.root / 'sources/peer.json')
        # Read evidence introduced by merge immediately, despite stale index.
        self.assertEqual([r['id'] for r in self.store.list('sources')], ['local', 'peer'])
        index.write_bytes(b'corrupt disposable database')
        self.store.rebuild_index()
        with closing(sqlite3.connect(index)) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM records').fetchone()[0], 2)

    def test_index_connection_closed_before_replace_and_failure_cleanup(self):
        connect = sqlite3.connect
        connections = []

        def tracked_connect(*args, **kwargs):
            db = connect(*args, **kwargs)
            connections.append(db)
            return db

        def assert_closed():
            with self.assertRaises(sqlite3.ProgrammingError):
                connections[-1].execute('SELECT 1')

        replace = os.replace

        def checked_replace(source, destination):
            assert_closed()
            replace(source, destination)

        self.store.put('sources', {'id': 'local'})
        with patch('autoresearch.records.sqlite3.connect', side_effect=tracked_connect), patch('autoresearch.records.os.replace', side_effect=checked_replace):
            index = self.store.rebuild_index()
            original = index.read_bytes()
            with patch.object(self.store, 'list', side_effect=ValueError('invalid evidence')):
                with self.assertRaisesRegex(ValueError, 'invalid evidence'):
                    self.store.rebuild_index()
            assert_closed()
        self.assertEqual(index.read_bytes(), original)
        self.assertEqual(list(self.store.root.glob('.index-*')), [])

    def test_related_prior_evidence(self):
        self.store.put('hypotheses', {'id': 'h', 'title': 'volume reversal', 'family': 'reversal'})
        self.store.put('experiments', {'id': 'e', 'hypothesis_id': 'h', 'result': {'status': 'ok'}, 'decision': 'reject'})
        self.store.put('reviews', {'id': 'r', 'experiment_id': 'e', 'reason': 'weak validation'})
        related = self.store.related_hypotheses('volume reversal')[0]
        self.assertEqual(related['reviews'][0]['id'], 'r')
        self.assertEqual(related['experiments'][0]['decision'], 'reject')
        self.assertEqual(self.store.related_hypotheses('unrelated'), [])

    def test_lock_contention_and_crash_recovery(self):
        competitor = ResearchStore(self.temp.name)
        with self.store.lock():
            with self.store.lock():
                self.store.put('events', {'id': 'nested'})
            with self.assertRaisesRegex(RuntimeError, 'writer'):
                with competitor.lock():
                    pass
        code = "from autoresearch.records import ResearchStore; import os,sys; s=ResearchStore(sys.argv[1]); c=s.lock(); c.__enter__(); os._exit(0)"
        subprocess.run([sys.executable, '-c', code, self.temp.name], check=True, timeout=10)
        with competitor.lock():
            competitor.put('events', {'id': 'recovered'})


if __name__ == '__main__':
    unittest.main()
