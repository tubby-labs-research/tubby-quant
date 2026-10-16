"""Bounded, separately sealed execution recipes derived from a sealed dictionary.

No catalog, dictionary, or daily-feature bytes are written. Preparation reads the
parent twice: a read-only SQLite connection is not an immutability lock, so both
whole-file hashes and unchanged file/metadata identities are required around
export. These guard ordinary concurrent modification and replacement; they are
not a filesystem snapshot or protection against an actor who mutates/restores
bytes and timestamps during export (Windows ctime can be creation time), or
forges metadata and seals. The sealed parent must remain immutable during prep.
Workers trust the prepared parent binding and
rehash only used shards, on every call, including cached-result requests. A saved
execution identity pins the self-sealed manifest; self-seals are not signatures.

Rows are name TAB canonical JSON LF, bucketed by SHA256(name UTF-8)'s first byte.
Memory is bounded by 256 64-KiB writer buffers, one capped row, SQLite's 8-MiB
cache and the capped requested-name set. Each of 256 writers opens once. There
is no whole-dictionary Python name pool. Duplicate selected rows fail verification;
the certified parent's UNIQUE(name/spec) schema enforces global uniqueness at
preparation. No quick_check, database hash, or database lookup runs in workers.
Progress receives strings and may raise (including KeyboardInterrupt) to cancel.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import shutil
import sqlite3
import stat
import tempfile
import time
from contextlib import ExitStack, closing
from pathlib import Path

from . import sec_catalog as source
EXECUTION_VERSION = "sec-execution-1"
BUCKETS = 256
BUFFER_BYTES = 64 * 1024
MAX_ROW_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_BYTES = 256 * 1024
MAX_FIELDS = 4096
_SHARDS = tuple(f"{i:02x}.tsv" for i in range(BUCKETS))
_SCHEMA = dict(format="name-tab-canonical-json-lf", bucket="sha256-utf8-name-first-byte",
               buckets=BUCKETS, max_row_bytes=MAX_ROW_BYTES)
SCHEMA_SHA256 = hashlib.sha256(source._canonical(_SCHEMA)).hexdigest()
_PARENT_KEYS = ("version", "catalog_sha256", "dictionary_sha256",
                "definitions_sha256", "manifest_sha256")


def _registry():
    # Service fixtures replace sys.modules entries. Never retain an import-time
    # stub in this module after their patch context exits.
    return importlib.import_module(f"{__package__}.sec_registry")


def _file_identity(path):
    metadata = source._safe_path(path).stat()
    return (metadata.st_dev, metadata.st_ino, metadata.st_size,
            metadata.st_mtime_ns, metadata.st_ctime_ns)


def _digest(value):
    return hashlib.sha256(source._canonical(value)).hexdigest()


def _json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=pairs)


def _regular(path):
    path = source._safe_path(path)
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"Artifact must be an ordinary unlinked file: {path.name}")
    return path


def _inventory(path, expected, checked=None):
    path = source._safe_path(path)
    if not path.is_dir() or {p.name for p in path.iterdir()} != set(expected):
        raise ValueError("Execution/parent artifact inventory differs")
    return {name: _file_identity(_regular(path / name))
            for name in (expected if checked is None else checked)}


def _small_json(path):
    with _regular(path).open("rb") as stream:
        data = stream.read(MAX_MANIFEST_BYTES + 1)
    if len(data) > MAX_MANIFEST_BYTES:
        raise ValueError("Execution/parent manifest is too large")
    return _json(data)


def _context(catalog):
    registry = _registry()
    catalog, metadata, parent_path = registry._context(catalog)
    _small_json(parent_path / "manifest.json")
    seal = registry._read_seal(parent_path, metadata["catalog_sha256"])
    if not registry._supported(metadata):
        raise ValueError("SEC catalog does not support numeric execution")
    parent = {key: seal[key] for key in _PARENT_KEYS}
    # A sibling of the discovery dictionary, never a child of sealed data.
    output = source._safe_path(parent_path.with_name(
        f"{metadata['catalog_sha256']}-{EXECUTION_VERSION}"))
    return catalog, metadata, parent_path, seal, parent, output


def _identity(manifest):
    return dict(version=manifest["version"], manifest_sha256=manifest["manifest_sha256"],
                parent=dict(manifest["parent"]))


def _read_manifest(output, parent, field_count):
    manifest = _small_json(output / "manifest.json")
    if (not isinstance(manifest, dict)
            or set(manifest) != {"status", "version", "parent", "schema_sha256", "field_count",
                                 "shards", "artifact_sha256", "manifest_sha256"}
            or manifest["status"] != "ready" or manifest["version"] != EXECUTION_VERSION
            or manifest["parent"] != parent or manifest["schema_sha256"] != SCHEMA_SHA256
            or type(manifest["field_count"]) is not int or manifest["field_count"] != field_count):
        raise ValueError("Execution manifest/catalog/parent identity differs")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256")
    if manifest["manifest_sha256"] != _digest(unsigned):
        raise ValueError("Execution manifest self-seal differs")
    shards = manifest["shards"]
    if not isinstance(shards, dict) or set(shards) != set(_SHARDS):
        raise ValueError("Execution shard inventory is malformed")
    total = 0
    for record in shards.values():
        if (not isinstance(record, dict) or set(record) != {"sha256", "count", "bytes"}
                or not isinstance(record["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
                or any(type(record[key]) is not int or record[key] < 0 for key in ("count", "bytes"))
                or record["count"] > record["bytes"]):
            raise ValueError("Execution shard metadata is malformed")
        total += record["count"]
    if total != field_count or manifest["artifact_sha256"] != _digest(shards):
        raise ValueError("Execution artifact count/hash differs")
    return manifest


def execution_status(catalog):
    """Small metadata only: ready means a compatible seal, not verified bytes.

    Does not enumerate the execution directory or stat/open any shard. Workers
    check inventory and selected regular files, then verify their actual bytes.
    """
    result = dict(status="missing", path=None, identity=None, message="Prepare run index")
    try:
        _, _, parent_path, seal, parent, output = _context(catalog)
        result["path"] = str(output)
        if output.exists():
            manifest = _read_manifest(output, parent, seal["field_count"])
            result.update(status="ready", identity=_identity(manifest),
                          message="SEC run index is prepared; selected shards require verification")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result.update(status="invalid", message=f"Prepare run index: {exc}")
    return result


def _bucket(name):
    return hashlib.sha256(name.encode("utf-8")).digest()[0]


def _recipe(name, serialized, registry=None):
    registry = registry if registry is not None else _registry()
    if not isinstance(name, str) or not registry._NAME.fullmatch(name):
        raise ValueError("Invalid SEC execution field name")
    spec = registry._spec(_json(serialized))
    if registry._name_for_recipe(spec) != name:
        raise ValueError("SEC execution name-to-spec mismatch")
    if source._canonical(spec) != (serialized.encode("utf-8") if isinstance(serialized, str) else serialized):
        raise ValueError("SEC execution spec is not canonical JSON")
    return spec


def _hash_parent(database, progress):
    digest = hashlib.sha256()
    with _regular(database).open("rb") as stream:
        size, next_report = 0, 0
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
            if progress is not None and size >= next_report:
                progress(f"Hashing sealed SEC dictionary bytes: {size:,}")
                next_report = size + 1024 * 1024 * 1024
    return digest.hexdigest()


def _parent_snapshot(catalog, metadata, parent_path):
    return (source.read_catalog(catalog),
            _file_identity(_regular(catalog / "manifest.json")),
            _file_identity(source._safe_path(catalog)),
            _file_identity(source._safe_path(parent_path)),
            _inventory(parent_path, ("manifest.json", "dictionary.sqlite")))


def _require_unique_schema(connection):
    # The sealed v2 builder creates both single-column UNIQUE constraints.
    unique = set()
    for index in connection.execute("PRAGMA index_list(fields)"):
        if index[2] and not index[4]:
            columns = connection.execute("SELECT name FROM pragma_index_info(?)", (index[1],)).fetchall()
            if len(columns) == 1:
                unique.add(columns[0][0])
    if not {"name", "spec"} <= unique:
        raise ValueError("Parent dictionary lacks unique name/spec constraints")


def build_execution_index(catalog, progress=None):
    """Prepare once, reuse valid metadata/refuse competing builders; return status.

    Failed/cancelled staging is deleted; a process killed without cleanup leaves
    a lock/staging for explicit operator recovery and blocks another build.
    Publication is one directory rename under an exclusive sibling lock. All
    cooperating builders honor the lock; no existing destination is replaced.
    """
    registry = _registry()
    catalog, metadata, parent_path, seal, parent, output = _context(catalog)
    if output.exists():
        result = execution_status(catalog)
        if result["status"] != "ready":
            raise ValueError(result["message"])
        return result
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = source._safe_path(output.with_name(output.name + ".building-lock"))
    lock.mkdir()  # Atomic exclusive ownership; no stale-lock fallback.
    staging = None
    try:
        if output.exists():
            raise FileExistsError(output)
        before = _parent_snapshot(catalog, metadata, parent_path)
        database = parent_path / "dictionary.sqlite"
        if _hash_parent(database, progress) != seal["dictionary_sha256"]:
            raise ValueError("Parent dictionary hash differs")
        if _parent_snapshot(catalog, metadata, parent_path) != before:
            raise ValueError("Parent/catalog changed during initial hash")
        staging = Path(tempfile.mkdtemp(prefix=output.name + ".building-", dir=output.parent))
        digests = [hashlib.sha256() for _ in _SHARDS]
        records = [dict(count=0, bytes=0) for _ in _SHARDS]
        count = 0
        next_progress = 0.0
        with ExitStack() as stack:
            writers = [stack.enter_context((staging / name).open("xb", buffering=BUFFER_BYTES))
                       for name in _SHARDS]
            connection = stack.enter_context(closing(registry._db(parent_path)))
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_ROW_BYTES)
            connection.execute("PRAGMA cache_size=-8192")
            _require_unique_schema(connection)
            for name, serialized in connection.execute("SELECT name,spec FROM fields ORDER BY id"):
                # Certified v2 bytes + unique schema supply canonical recipes.
                # Do not parse or canonicalize 29.6M already certified rows.
                if (not isinstance(name, str) or not name.startswith("sec_")
                        or not isinstance(serialized, str)
                        or any(char in value for value in (name, serialized) for char in "\t\r\n")):
                    raise ValueError("Malformed parent execution row")
                row = name.encode("utf-8") + b"\t" + serialized.encode("utf-8") + b"\n"
                if len(row) > MAX_ROW_BYTES:
                    raise ValueError("SEC execution row exceeds bounded size")
                bucket = _bucket(name)
                writers[bucket].write(row)
                digests[bucket].update(row)
                records[bucket]["bytes"] += len(row)
                records[bucket]["count"] += 1
                count += 1
                if progress is not None and count % 4096 == 0:
                    now = time.monotonic()
                    if now >= next_progress:
                        progress(f"Exported {count:,} SEC execution recipes")
                        next_progress = now + 2.0
            for bucket, writer in enumerate(writers):
                writer.flush()
                os.fsync(writer.fileno())
                records[bucket]["sha256"] = digests[bucket].hexdigest()
        if count != seal["field_count"]:
            raise ValueError("Parent/export field count differs")
        if progress is not None:
            progress(f"Exported {count:,} SEC execution recipes; verifying parent again")
        if (_hash_parent(database, progress) != seal["dictionary_sha256"]
                or _parent_snapshot(catalog, metadata, parent_path) != before
                or _context(catalog)[4] != parent):
            raise ValueError("Parent/catalog changed during export")
        manifest = dict(status="ready", version=EXECUTION_VERSION, parent=parent,
                        schema_sha256=SCHEMA_SHA256, field_count=count,
                        shards=dict(zip(_SHARDS, records)))
        manifest["artifact_sha256"] = _digest(manifest["shards"])
        manifest["manifest_sha256"] = _digest(manifest)
        with (staging / "manifest.json").open("xb") as stream:
            stream.write(source._canonical(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        _read_manifest(staging, parent, count)
        _inventory(staging, ("manifest.json", *_SHARDS))
        if output.exists():
            raise FileExistsError(output)
        source._safe_path(output)
        staging.rename(output)
        staging = None
        return dict(status="ready", path=str(output), identity=_identity(manifest),
                    message="SEC run index is prepared; selected shards require verification")
    finally:
        if staging is not None:
            shutil.rmtree(source._safe_path(staging))
        lock.rmdir()


def verify_execution_fields(catalog, names, expected_identity=None):
    """Fresh hash/count/byte verification of used shards; return selected recipes.

    Reads every byte of each selected shard once and parses only requested rows
    from those verified bytes. Never returns early after finding names. Unselected
    shard names are inventoried, but their paths are never opened, stat'ed or
    hashed. No database queries, parent hash or TTL/stat cache.
    """
    registry = _registry()
    if isinstance(names, (str, bytes)):
        raise ValueError("names must be an iterable of SEC field names")
    requested = set()
    for name in names:
        if not isinstance(name, str) or not registry._NAME.fullmatch(name):
            raise ValueError("Invalid requested SEC execution name")
        if name in requested:
            raise ValueError("Duplicate requested SEC execution name")
        requested.add(name)
        if len(requested) > MAX_FIELDS:
            raise ValueError("Too many SEC execution fields requested")
    status = execution_status(catalog)
    if status["status"] != "ready":
        raise ValueError(status["message"])
    catalog, metadata, parent_path, seal, parent, output = _context(catalog)
    parent_before = _parent_snapshot(catalog, metadata, parent_path)
    manifest = _read_manifest(output, parent, seal["field_count"])
    identity = _identity(manifest)
    if expected_identity is not None and expected_identity != identity:
        raise ValueError("Pinned SEC execution identity differs")
    buckets = sorted({_bucket(name) for name in requested})
    checked = ("manifest.json", *(_SHARDS[bucket] for bucket in buckets))
    before = _inventory(output, ("manifest.json", *_SHARDS), checked)
    directory_before = _file_identity(output)
    fields = {}
    raw_requested = {name.encode("utf-8"): name for name in requested}
    for bucket in buckets:
        name = _SHARDS[bucket]
        digest, count, size = hashlib.sha256(), 0, 0
        selected = {}
        with _regular(output / name).open("rb") as stream:
            while row := stream.readline(MAX_ROW_BYTES + 1):
                if len(row) > MAX_ROW_BYTES or not row.endswith(b"\n") or row.count(b"\t") != 1:
                    raise ValueError("Malformed/truncated SEC execution row")
                digest.update(row)
                size += len(row)
                count += 1
                raw_name, serialized = row[:-1].split(b"\t")
                if raw_name in raw_requested:
                    field = raw_requested[raw_name]
                    if _bucket(field) != bucket:
                        raise ValueError("SEC execution field in wrong shard")
                    if field in selected or field in fields:
                        raise ValueError("Duplicate selected SEC execution field")
                    selected[field] = serialized
        if dict(sha256=digest.hexdigest(), count=count, bytes=size) != manifest["shards"][name]:
            raise ValueError("SEC execution shard hash/count/bytes differ")
        for field, serialized in selected.items():
            fields[field] = _recipe(field, serialized, registry)
    if set(fields) != requested:
        raise ValueError("Missing requested SEC execution fields")
    if (_inventory(output, ("manifest.json", *_SHARDS), checked) != before
            or _file_identity(source._safe_path(output)) != directory_before
            or _read_manifest(output, parent, seal["field_count"]) != manifest
            or _parent_snapshot(catalog, metadata, parent_path) != parent_before
            or _context(catalog)[4] != parent):
        raise ValueError("SEC execution/parent/catalog changed during verification")
    return dict(fields=fields, identity=identity)
