"""Portable immutable JSON evidence; SQLite is a disposable search aid only."""
from __future__ import annotations

from contextlib import closing, contextmanager
from dataclasses import fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import threading

COLLECTIONS = frozenset({'sources', 'hypotheses', 'experiments', 'reviews', 'events', 'campaigns'})
_SAFE_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z')


def _json(value):
    def check(v):
        if isinstance(v, dict):
            if any(not isinstance(k, str) for k in v):
                raise ValueError('JSON object keys must be strings')
            for child in v.values():
                check(child)
        elif isinstance(v, (list, tuple)):
            if isinstance(v, tuple):
                raise ValueError('records must contain JSON arrays, not tuples')
            for child in v:
                check(child)
        elif not (v is None or isinstance(v, (str, bool, int, float))):
            raise ValueError('records must contain JSON-compatible values')
    check(value)
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def stable_id(kind, value):
    """Content identity independent of mapping order and local filesystem paths."""
    if not isinstance(kind, str) or not _SAFE_ID.fullmatch(kind) or len(kind) > 40:
        raise ValueError('invalid identity kind')
    return kind + '_' + hashlib.sha256(_json(value).encode('utf-8')).hexdigest()[:40]


def canonical_expression(expression):
    """Serialize the actual FASTEXPR AST, preserving order and string contents.

    Whitespace, redundant parentheses and number spelling normalize; algebraic
    equivalence, variable renaming, keyword order and commutation do not.
    Invalid syntax raises the existing parser's ExpressionError.
    """
    from qlib_yahoo.alphalab.fastexpr import parse
    def node(v):
        if is_dataclass(v):
            return {'type': type(v).__name__, **{f.name: node(getattr(v, f.name)) for f in fields(v)}}
        if isinstance(v, tuple):
            return [node(x) for x in v]
        return v
    return _json(node(parse(expression)))


class ResearchStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._mutex = threading.RLock()
        self._lock_depth = 0

    def _directory(self, collection):
        if collection not in COLLECTIONS:
            raise ValueError('unknown research collection')
        directory = self.root / collection
        if directory.is_symlink():
            raise ValueError('collection directories must not be symlinks')
        return directory

    def _path(self, collection, record_id):
        if not isinstance(record_id, str) or not _SAFE_ID.fullmatch(record_id):
            raise ValueError('invalid record id')
        path = self._directory(collection) / (record_id + '.json')
        if path.is_symlink():
            raise ValueError('record files must not be symlinks')
        return path

    @contextmanager
    def lock(self):
        """Nonblocking OS advisory single-writer lock, released after process death."""
        with self._mutex:
            if self._lock_depth:
                self._lock_depth += 1
                try:
                    yield self
                finally:
                    self._lock_depth -= 1
                return
            lock_path = self.root / '.writer.lock'
            if lock_path.is_symlink():
                raise ValueError('lock file must not be a symlink')
            handle = open(lock_path, 'a+b')
            try:
                if os.name == 'nt':
                    import msvcrt
                    handle.seek(0)
                    if not handle.read(1):
                        handle.write(b'0')
                        handle.flush()
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                handle.close()
                raise RuntimeError('research store already has a writer') from exc
            self._lock_depth = 1
            try:
                yield self
            finally:
                self._lock_depth = 0
                handle.close()

    def get(self, collection, record_id):
        path = self._path(collection, record_id)
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding='utf-8'))
        _json(record)
        if not isinstance(record, dict) or record.get('id') != record_id:
            raise ValueError('record identity does not match filename')
        return record

    def list(self, collection):
        directory = self._directory(collection)
        if not directory.exists():
            return []
        return [self.get(collection, path.stem) for path in sorted(directory.glob('*.json'))]

    def put(self, collection, record):
        if not isinstance(record, dict):
            raise ValueError('record must be a JSON object with explicit id')
        encoded = _json(record)
        path = self._path(collection, record.get('id'))
        with self.lock():
            existing = self.get(collection, record['id'])
            if existing is not None:
                if _json(existing) != encoded:
                    raise ValueError('immutable record conflict: ' + record['id'])
                return existing
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix='.record-', dir=path.parent)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as output:
                    output.write(encoded + '\n')
                    output.flush()
                    os.fsync(output.fileno())
                # Never replace an existing destination, even if another writer
                # bypassed our advisory lock or a merge introduced this identity.
                os.link(temporary, path)
            finally:
                os.unlink(temporary)
        return json.loads(encoded)

    def find_experiment(self, key):
        for record in self.list('experiments'):
            if record.get('experiment_key') == key and isinstance(record.get('result'), dict) and record['result'].get('status') == 'ok':
                return record
        return None

    def related_hypotheses(self, text, limit=5):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError('limit must be a nonnegative integer')
        words = lambda value: set(re.findall(r'[a-z0-9_]+', value.lower()))
        query = words(text)
        experiments = sorted(self.list('experiments'), key=lambda r: (str(r.get('created_at', '')), r['id']))
        reviews = sorted(self.list('reviews'), key=lambda r: (str(r.get('created_at', '')), r['id']))
        ranked = []
        for hypothesis in self.list('hypotheses'):
            searchable = ' '.join(str(hypothesis.get(k, '')) for k in ('title', 'mechanism', 'family', 'expression', 'rationale'))
            score = len(query & words(searchable))
            if not score:
                continue
            linked = [e for e in experiments if e.get('hypothesis_id') == hypothesis['id']]
            ids = {e['id'] for e in linked}
            def bounded(value):
                if isinstance(value, str):
                    return value if len(value) <= 4000 else value[:4000] + ' [truncated]'
                if isinstance(value, dict):
                    return {k: bounded(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [bounded(v) for v in value[:20]]
                return value
            result = bounded(hypothesis)
            result['experiments'] = [bounded({k: e[k] for k in ('id', 'hypothesis_id', 'parent_experiment_id', 'expression', 'result', 'decision', 'created_at') if k in e}) for e in linked[-5:]]
            result['reviews'] = [bounded(r) for r in reviews if r.get('hypothesis_id') == hypothesis['id'] or r.get('experiment_id') in ids][-10:]
            ranked.append((score, hypothesis['id'], result))
        return [item[2] for item in sorted(ranked, key=lambda item: (-item[0], item[1]))[:limit]]

    def rebuild_index(self):
        """Rebuild after merges; readers and dedup never rely on stale index state."""
        with self.lock():
            fd, temporary = tempfile.mkstemp(prefix='.index-', dir=self.root)
            os.close(fd)
            try:
                # The connection context handles transactions, not closure.
                # Close the handle before replacing or cleaning up on Windows.
                with closing(sqlite3.connect(temporary)) as db, db:
                    db.execute('CREATE TABLE records (collection TEXT, id TEXT, record_json TEXT, PRIMARY KEY(collection,id))')
                    for collection in sorted(COLLECTIONS):
                        db.executemany('INSERT INTO records VALUES (?,?,?)', [(collection, r['id'], _json(r)) for r in self.list(collection)])
                os.replace(temporary, self.root / 'index.sqlite3')
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return self.root / 'index.sqlite3'
