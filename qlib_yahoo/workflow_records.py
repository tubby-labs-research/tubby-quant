"""Dependency-free, append-only workflow evidence under ``data_root/workflow``.

Submissions and terminal outcomes are separately sealed JSON envelopes. Results
are canonical JSON bytes named by their full SHA-256. Seals detect corruption;
they are not signatures against an attacker able to rewrite the whole store.
Returned dictionaries are detached copies, never mutable handles to evidence.

Publication requires a filesystem supporting exclusive hard links; unsupported
filesystems fail rather than falling back to overwriting. Files are fsynced
before publication, and directory entries are fsynced where POSIX supports it.
Filesystem permissions must prevent hostile concurrent replacement of ancestor
directories (particularly on Windows, which lacks portable dir_fd traversal).
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import tempfile
import uuid

SCHEMA_VERSION = 1
TERMINAL_STATUSES = frozenset({
    'completed', 'reused', 'rejected', 'failed', 'cancelled', 'interrupted_unknown',
})
_ID = re.compile(r'[0-9a-f]{32}\Z')
_DIGEST = re.compile(r'[0-9a-f]{64}\Z')
_SUBMISSION_FIELDS = {
    'id', 'schema_version', 'created_at', 'kind', 'request', 'parents',
    'parent_digests', 'candidate_id', 'source', 'owner',
}
_TERMINAL_FIELDS = {
    'id', 'schema_version', 'finished_at', 'status', 'result_digest', 'error',
    'artifacts', 'submission_digest',
}


class EvidenceConflict(ValueError):
    """A published immutable outcome already has different meaning."""


def canonical_json_bytes(value):
    """Strict JSON only: no Paths, tuples, non-string keys or nonfinite floats."""
    def check(item):
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise ValueError('JSON object keys must be strings')
            for child in item.values():
                check(child)
        elif type(item) is list:
            for child in item:
                check(child)
        elif item is not None and type(item) not in (str, bool, int, float):
            raise ValueError('evidence must contain plain JSON values')
    try:
        check(value)
        return json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=False, allow_nan=False).encode('utf-8')
    except (RecursionError, UnicodeError) as exc:
        raise ValueError('invalid JSON evidence') from exc


def seal_record(record):
    """Return a detached envelope whose digest binds canonical record content."""
    encoded = canonical_json_bytes(record)
    return {'payload': json.loads(encoded), 'sha256': hashlib.sha256(encoded).hexdigest()}


def _decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=pairs)
        if canonical_json_bytes(value) != raw:
            raise ValueError('noncanonical or nonfinite JSON evidence')
        return value
    except (UnicodeError, RecursionError) as exc:
        raise ValueError('invalid JSON evidence') from exc


def _identifier(value, pattern=_ID):
    if type(value) is not str or not pattern.fullmatch(value):
        raise ValueError('invalid evidence identity')
    return value


def _text(value, label):
    if type(value) is not str or not value.strip():
        raise ValueError(label + ' must be a nonempty string')


def _timestamp(value):
    _text(value, 'timestamp')
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.utcoffset() is None:
            raise ValueError('timestamp must include a timezone')
    except (ValueError, TypeError) as exc:
        raise ValueError('invalid evidence timestamp') from exc


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='microseconds')


def _checked(path):
    """Check every existing ancestor without resolving away links/junctions."""
    for part in reversed((path, *path.parents)):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('symlink/reparse evidence paths are forbidden')
        if part != path and not stat.S_ISDIR(info.st_mode):
            raise ValueError('evidence ancestor is not a directory')
    return path


def _read_bytes(path):
    _checked(path)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError('evidence must be a regular file')
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, 'rb') as stream:
        opened = os.fstat(stream.fileno())
        current = _checked(path).lstat()
        if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(opened, current):
            raise ValueError('evidence must be an unchanged regular file')
        raw = stream.read()
        if not os.path.samestat(opened, _checked(path).lstat()):
            raise ValueError('evidence path changed while reading')
        return raw


def read_sealed_record(path):
    """Read canonical JSON and verify its embedded content seal (no writes)."""
    if '..' in Path(path).parts:
        raise ValueError('parent traversal in evidence path is forbidden')
    envelope = _decode(_read_bytes(Path(os.path.abspath(path))))
    if type(envelope) is not dict or set(envelope) != {'payload', 'sha256'}:
        raise ValueError('invalid evidence seal')
    _identifier(envelope['sha256'], _DIGEST)
    if seal_record(envelope['payload']) != envelope:
        raise ValueError('evidence seal mismatch')
    return envelope['payload']


def _mkdir(path):
    _checked(path)
    if not path.exists():
        _mkdir(path.parent)
        try:
            path.mkdir()
        except FileExistsError:
            pass
    if not stat.S_ISDIR(_checked(path).lstat().st_mode):
        raise ValueError('evidence directory is not a directory')


def _publish(path, raw):
    """Publish once, atomically; a collision always leaves the winner intact."""
    _mkdir(path.parent)
    _checked(path)
    fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        _checked(path)
        os.link(temporary, path)
        if os.name != 'nt':
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        os.unlink(temporary)


def _owner():
    owner = {'pid': os.getpid(), 'create_time': None, 'host': None}
    try:
        owner['host'] = socket.gethostname() or None
    except OSError:
        pass
    if os.name == 'nt':
        # GetCurrentProcess is a pseudo-handle: no process lookup or signal.
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        times = [wintypes.FILETIME() for _ in range(4)]
        if kernel.GetProcessTimes(kernel.GetCurrentProcess(), *(ctypes.byref(t) for t in times)):
            ticks = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
            owner['create_time'] = ticks / 10_000_000 - 11_644_473_600
    else:
        # Linux start ticks + boot time, compatible with psutil's epoch seconds.
        try:
            start_ticks = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]
            boot = next(line.split()[1] for line in Path('/proc/stat').read_text().splitlines()
                        if line.startswith('btime '))
            owner['create_time'] = int(boot) + int(start_ticks) / os.sysconf('SC_CLK_TCK')
        except (OSError, ValueError, IndexError, StopIteration):
            pass
    return owner


class EvidenceStore:
    """Read-only construction/browsing; only begin/add_result/finish write.

    Missing attempts/results raise FileNotFoundError. Invalid JSON, schemas,
    identities, paths and integrity checks raise ValueError. Conflicting terminal
    meanings raise EvidenceConflict. OS I/O failures propagate without overwrite.
    ``recover_pending(callback)`` passes each recorded owner dictionary to the
    callback and returns only terminal outcomes it successfully recovered.
    """

    def __init__(self, data_root):
        path = Path(data_root).expanduser()
        if '..' in path.parts:
            raise ValueError('parent traversal in data_root is forbidden')
        self.root = Path(os.path.abspath(path)) / 'workflow'
        _checked(self.root)

    def _path(self, collection, identity):
        _identifier(identity, _DIGEST if collection == 'results' else _ID)
        directory = _checked(self.root / collection)
        if directory.exists() and not directory.is_dir():
            raise ValueError('evidence collection is not a directory')
        return _checked(directory / (identity + '.json'))

    @staticmethod
    def _schema(record, fields):
        if (type(record) is not dict or set(record) != fields
                or type(record['schema_version']) is not int
                or record['schema_version'] != SCHEMA_VERSION):
            raise ValueError('invalid evidence schema/version')
        _identifier(record['id'])

    def _submission(self, attempt_id):
        record = read_sealed_record(self._path('submissions', attempt_id))
        self._schema(record, _SUBMISSION_FIELDS)
        if record['id'] != attempt_id:
            raise ValueError('submission identity mismatch')
        _timestamp(record['created_at'])
        _text(record['kind'], 'kind')
        if type(record['request']) is not dict or type(record['source']) is not dict:
            raise ValueError('request/source must be JSON objects')
        if record['candidate_id'] is not None:
            _text(record['candidate_id'], 'candidate_id')
        parents = record['parents']
        if type(parents) is not list or any(type(p) is not str for p in parents):
            raise ValueError('parents must be an array of IDs')
        if len(parents) != len(set(parents)) or attempt_id in parents:
            raise ValueError('duplicate/self parent')
        for parent in parents:
            _identifier(parent)
        if type(record['parent_digests']) is not dict or set(record['parent_digests']) != set(parents):
            raise ValueError('parent digest references mismatch')
        for digest in record['parent_digests'].values():
            _identifier(digest, _DIGEST)
        owner = record['owner']
        if (type(owner) is not dict or set(owner) != {'pid', 'create_time', 'host'}
                or type(owner['pid']) is not int or owner['pid'] <= 0
                or (owner['create_time'] is not None and
                    (type(owner['create_time']) not in (int, float) or owner['create_time'] <= 0))
                or (owner['host'] is not None and type(owner['host']) is not str)):
            raise ValueError('invalid owner identity')
        return record

    def _verified_submission(self, attempt_id):
        first = self._submission(attempt_id)
        todo = [first]
        seen = {attempt_id}
        while todo:
            current = todo.pop()
            for parent in current['parents']:
                record = self._submission(parent)
                if seal_record(record)['sha256'] != current['parent_digests'][parent]:
                    raise ValueError('parent submission content changed')
                if parent not in seen:
                    seen.add(parent)
                    todo.append(record)
        return first

    def begin(self, kind, request: dict, *, parents: list[str] | None = None,
              candidate_id: str | None = None, source: dict | None = None):
        _text(kind, 'kind')
        if type(request) is not dict or (source is not None and type(source) is not dict):
            raise ValueError('request/source must be JSON objects')
        if candidate_id is not None:
            _text(candidate_id, 'candidate_id')
        parents = [] if parents is None else parents
        if type(parents) is not list:
            raise ValueError('parents must be a list')
        for parent in parents:
            _identifier(parent)
        if len(parents) != len(set(parents)):
            raise ValueError('duplicate parents')
        canonical_json_bytes({'request': request, 'source': source})
        parent_digests = {parent: seal_record(self._verified_submission(parent))['sha256']
                          for parent in parents}
        record = {'id': uuid.uuid4().hex, 'schema_version': SCHEMA_VERSION,
                  'created_at': _now(), 'kind': kind, 'request': request,
                  'parents': parents, 'parent_digests': parent_digests,
                  'candidate_id': candidate_id, 'source': {} if source is None else source,
                  'owner': _owner()}
        envelope = seal_record(record)
        _publish(self._path('submissions', record['id']), canonical_json_bytes(envelope))
        return envelope['payload']

    def add_result(self, payload):
        raw = canonical_json_bytes(payload)
        digest = hashlib.sha256(raw).hexdigest()
        path = self._path('results', digest)
        try:
            _publish(path, raw)
        except FileExistsError:
            if _read_bytes(path) != raw:
                raise ValueError('result digest collision or tampering')
        return digest

    def get_result(self, digest):
        raw = _read_bytes(self._path('results', digest))
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('result digest mismatch')
        return _decode(raw)

    def _terminal(self, attempt_id, submission, *, verify_result=True):
        path = self._path('terminals', attempt_id)
        try:
            record = read_sealed_record(path)
        except FileNotFoundError:
            return None
        self._schema(record, _TERMINAL_FIELDS)
        if (record['id'] != attempt_id or type(record['status']) is not str
                or record['status'] not in TERMINAL_STATUSES):
            raise ValueError('invalid terminal identity/status')
        _timestamp(record['finished_at'])
        if record['submission_digest'] != seal_record(submission)['sha256']:
            raise ValueError('terminal submission digest mismatch')
        if record['result_digest'] is not None:
            _identifier(record['result_digest'], _DIGEST)
            if verify_result:
                self.get_result(record['result_digest'])
        return record

    def finish(self, attempt_id, status, *, result=None, error=None, artifacts=None):
        if type(status) is not str or status not in TERMINAL_STATUSES:
            raise ValueError('invalid terminal status')
        submission = self._verified_submission(attempt_id)
        canonical_json_bytes({'result': result, 'error': error, 'artifacts': artifacts})
        digest = None if result is None else self.add_result(result)
        terminal = {'id': attempt_id, 'schema_version': SCHEMA_VERSION,
                    'finished_at': _now(), 'status': status, 'result_digest': digest,
                    'error': error, 'artifacts': artifacts,
                    'submission_digest': seal_record(submission)['sha256']}
        try:
            _publish(self._path('terminals', attempt_id), canonical_json_bytes(seal_record(terminal)))
        except FileExistsError:
            existing = self._terminal(attempt_id, submission)
            meaning = lambda value: {k: v for k, v in value.items() if k != 'finished_at'}
            if existing is None or canonical_json_bytes(meaning(existing)) != canonical_json_bytes(meaning(terminal)):
                raise EvidenceConflict('immutable terminal outcome conflict: ' + attempt_id)
            return existing
        return json.loads(canonical_json_bytes(terminal))

    def get(self, attempt_id, *, verify_result=True):
        submission = self._verified_submission(attempt_id)
        terminal = self._terminal(attempt_id, submission, verify_result=verify_result)
        return {**submission, 'status': 'pending', 'finished_at': None,
                'result_digest': None, 'error': None, 'artifacts': None,
                'submission_digest': seal_record(submission)['sha256'], **(terminal or {})}

    def list_attempts(self, parent=None, candidate_id=None, limit=100, *, verify_results=True):
        """Newest first; metadata-only browsing explicitly does not verify results.

        Verification remains the default for governance and recovery callers.
        The console's list opts out to avoid reading every historical daily-PnL
        payload merely to show submission headers; opening an attempt verifies it.
        """
        if parent is not None:
            _identifier(parent)
        if candidate_id is not None:
            _text(candidate_id, 'candidate_id')
        if type(limit) is not int or limit < 0:
            raise ValueError('limit must be a nonnegative integer')
        directory = _checked(self.root / 'submissions')
        if not directory.exists():
            return []
        if not directory.is_dir():
            raise ValueError('submissions must be a directory')
        records = []
        # Inspect every published filename; interrupted temporary siblings are ignored.
        for path in sorted(directory.iterdir()):
            if path.name.startswith('.pending-'):
                continue
            if path.suffix != '.json':
                raise ValueError('unexpected submission file')
            record = self.get(path.stem, verify_result=verify_results)
            if parent is not None and parent not in record['parents']:
                continue
            if candidate_id is not None and candidate_id != record['candidate_id']:
                continue
            records.append({key: record[key] for key in (
                'id', 'schema_version', 'created_at', 'kind', 'parents', 'candidate_id',
                'status', 'finished_at', 'result_digest', 'submission_digest',
            )})
        return sorted(records, key=lambda r: (r['created_at'], r['id']), reverse=True)[:limit]

    def recover_pending(self, owner_alive):
        """Only an explicit False proves death; age/locks are never evidence."""
        if not callable(owner_alive):
            raise ValueError('owner_alive must be callable')
        recovered = []
        for summary in self.list_attempts(limit=2**63 - 1):
            if summary['status'] != 'pending':
                continue
            record = self.get(summary['id'])
            if record['status'] != 'pending' or owner_alive(record['owner']) is not False:
                continue
            try:
                recovered.append(self.finish(record['id'], 'interrupted_unknown'))
            except EvidenceConflict:
                # A real completion won the exclusive publication race.
                self.get(record['id'])
        return recovered
