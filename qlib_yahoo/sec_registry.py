"""Immutable, observed SEC numeric recipes, separate from the raw catalog.

Metadata APIs use only small manifests and indexed SQLite queries. Only build
and verify read bulk artifacts. Presence of a recipe is not PIT eligibility or
coverage: period/filing selection remains the selected-field adapter's job.
Unknown/conflicting TAG.custom metadata permits exact selection only and is
classified as custom for discovery. Null contexts block all recipes matching
their tag/taxonomy/unit/duration, following the numeric adapter's pre-context
validation. They are counted, never interpreted as consolidated blanks.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import time
from collections import OrderedDict
from contextlib import closing
from pathlib import Path

from . import sec_catalog as source

REGISTRY_VERSION = "sec-dictionary-2"
BATCH_SIZE = 65536
RECIPE_CACHE_ENTRIES = 262144
RECIPE_CACHE_BYTES = 128 * 1024 * 1024
_KEYS = ("tag", "taxonomy", "unit", "qtrs", "segments", "coreg")
_NUM_KEYS = ("tag", "version", "uom", "qtrs", "segments", "coreg")
_SELECTOR_KEYS = ("tag", "version", "uom", "qtrs")
_NAME = re.compile(r"sec_[a-z0-9]+(?:_[a-z0-9]+)*\Z")


class _RecipeCache:
    """Build-local LRU of successfully processed raw six-key recipes.

    Charge every retained string in full (even shared ones), tuple/value sizes,
    and the actual OrderedDict allocation. Evict until both limits hold; a key
    too large for the byte budget is never retained. No TAG descriptions live
    here. Fixed empty-cache bookkeeping is excluded from retained-byte counts.
    """
    def __init__(self, max_entries, max_bytes):
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self.entries = OrderedDict()
        self.key_bytes = 0
        self.hits = self.misses = self.evictions = self.peak_bytes = 0

    @property
    def bytes(self):
        return self.key_bytes + sys.getsizeof(self.entries) if self.entries else 0

    def contains(self, key):
        if key in self.entries:
            self.entries.move_to_end(key)
            self.hits += 1
            return True
        self.misses += 1
        return False

    def remember(self, key):
        if self.max_entries <= 0 or self.max_bytes <= 0:
            return
        charge = sys.getsizeof(key) + sum(sys.getsizeof(value) for value in key)
        charge += sys.getsizeof(charge)
        if charge > self.max_bytes or key in self.entries:
            return
        self.entries[key] = charge
        self.key_bytes += charge
        while self.entries and (len(self.entries) > self.max_entries or self.bytes > self.max_bytes):
            _, removed = self.entries.popitem(last=False)
            self.key_bytes -= removed
            self.evictions += 1
        if not self.entries:
            self.entries.clear()
        self.peak_bytes = max(self.peak_bytes, self.bytes)

    def diagnostics(self):
        return dict(recipe_cache_hits=self.hits, recipe_cache_misses=self.misses,
                    recipe_cache_evictions=self.evictions, recipe_cache_entries=len(self.entries),
                    recipe_cache_bytes=self.bytes, recipe_cache_peak_bytes=self.peak_bytes)


def _spec(value):
    if not isinstance(value, dict) or set(value) != set(_KEYS):
        raise ValueError("Recipe requires exactly tag, taxonomy, unit, qtrs, segments and coreg")
    result = {key: value[key] for key in _KEYS}
    for key in ("tag", "taxonomy", "unit", "segments", "coreg"):
        if not isinstance(result[key], str) or (key in {"tag", "taxonomy", "unit"} and not result[key]):
            raise ValueError(f"Recipe {key} must be an explicit string")
    if type(result["qtrs"]) is not int or result["qtrs"] < 0:
        raise ValueError("Recipe qtrs must be a nonnegative integer")
    source._selector(result["tag"], result["taxonomy"])
    return result


with Path(__file__).with_name("sec_registry_definitions.json").open(encoding="utf-8") as _stream:
    COMMON_FIELDS = {name: _spec(spec) for name, spec in json.load(_stream).items()}
if any(not _NAME.fullmatch(name) for name in COMMON_FIELDS):
    raise ValueError("Invalid common SEC name")
_COMMON_BY_SPEC = {source._canonical(spec).decode(): name for name, spec in COMMON_FIELDS.items()}
if len(_COMMON_BY_SPEC) != len(COMMON_FIELDS):
    raise ValueError("Duplicate common SEC recipes")
_DEFINITIONS_SHA256 = hashlib.sha256(source._canonical(COMMON_FIELDS)).hexdigest()


def _name_for_recipe(spec):
    serialized = source._canonical(spec).decode()
    if serialized in _COMMON_BY_SPEC:
        return _COMMON_BY_SPEC[serialized]
    # Spec digest makes punctuation/case/context normalization harmless.
    # The readable prefix is bounded, independent of encounter order.
    def slug(value, length):
        value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
        return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")[:length].rstrip("_") or "field"
    scope = "family" if spec["taxonomy"].endswith("/") else "edition"
    prefix = "_".join(slug(str(spec[key]), length) for key, length in
                      (("tag", 32), ("taxonomy", 24), ("unit", 16)))
    return f"sec_{prefix}_{scope}_q{spec['qtrs']}_{hashlib.sha256(serialized.encode()).hexdigest()[:20]}"


def _context(catalog):
    catalog = source._safe_path(catalog)
    manifest = source.read_catalog(catalog)
    path = source._safe_path(catalog.parent.parent / "dictionaries" /
                             f"{manifest['catalog_sha256']}-{REGISTRY_VERSION}")
    if path == catalog or catalog in path.parents or path in catalog.parents:
        raise ValueError("Dictionary path must be disjoint from catalog")
    return catalog, manifest, path


def _supported(manifest):
    return (manifest.get("numeric_materialization_supported") is True
            and set(_NUM_KEYS) <= set(manifest.get("tables", {}).get("num", {}).get("columns", [])))


def _read_seal(path, catalog_sha256):
    with source._safe_path(path / "manifest.json").open(encoding="utf-8") as stream:
        seal = json.load(stream)
    if (not isinstance(seal, dict) or seal.get("status") != "ready"
            or seal.get("version") != REGISTRY_VERSION
            or seal.get("catalog_sha256") != catalog_sha256
            or seal.get("definitions_sha256") != _DEFINITIONS_SHA256
            or type(seal.get("field_count")) is not int or seal["field_count"] < 0
            or not re.fullmatch(r"[0-9a-f]{64}", str(seal.get("dictionary_sha256")))):
        raise ValueError("Dictionary seal is incompatible or malformed")
    identity = dict(seal)
    digest = identity.pop("manifest_sha256", None)
    if digest != hashlib.sha256(source._canonical(identity)).hexdigest():
        raise ValueError("Dictionary manifest identity differs")
    if not source._safe_path(path / "dictionary.sqlite").is_file():
        raise ValueError("Dictionary database is missing")
    return seal


def dictionary_status(catalog):
    """Small-manifest inspection only; ready does not claim verified bytes."""
    _, manifest, path = _context(catalog)
    result = dict(status="missing", version=REGISTRY_VERSION,
                  catalog_sha256=manifest["catalog_sha256"], dictionary_sha256=None,
                  path=str(path), field_count=0, message="Build the SEC dictionary for this catalog")
    if not _supported(manifest):
        result.update(status="unsupported", message="Raw/notes or missing-context schema requires a numeric adapter")
        return result
    if path.exists():
        try:
            seal = _read_seal(path, manifest["catalog_sha256"])
        except (OSError, ValueError) as exc:
            result["message"] = f"Dictionary is unavailable: {exc}"
        else:
            result.update(status="ready", dictionary_sha256=seal["dictionary_sha256"],
                          field_count=seal["field_count"], message="Observed SEC numeric recipes are available")
    return result


def _file_identity(path):
    stat = source._safe_path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def verify_dictionary(catalog):
    """Force dictionary hashes without writing. Runner verifies source separately.

    The source manifest is identity-checked, but source Parquet bytes are not
    rehashed here. This avoids duplicate multi-GB worker verification.
    """
    before = dictionary_status(catalog)
    if before["status"] != "ready":
        raise ValueError(before["message"])
    catalog, manifest, path = _context(catalog)
    seal = _read_seal(path, manifest["catalog_sha256"])
    database = source._safe_path(path / "dictionary.sqlite")
    identities = {p: _file_identity(p) for p in (database, path / 'manifest.json')}
    if {p.name for p in path.iterdir()} != {"manifest.json", "dictionary.sqlite"}:
        raise ValueError("Dictionary artifact inventory differs")
    if source._hash(database) != seal["dictionary_sha256"]:
        raise ValueError("Dictionary database hash differs")
    with closing(_db(path)) as connection:
        if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise ValueError("Dictionary database integrity differs")
        if connection.execute("SELECT COUNT(*) FROM fields").fetchone()[0] != seal["field_count"]:
            raise ValueError("Dictionary field count differs")
    after = dictionary_status(catalog)
    if (before != after or source.read_catalog(catalog) != manifest
            or any(_file_identity(p) != identity for p, identity in identities.items())):
        raise ValueError("Dictionary/catalog metadata changed during verification")
    return after


def _db(path):
    database = source._safe_path(path / "dictionary.sqlite")
    return sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True)


def _entry(row):
    return dict(name=row[0], spec=json.loads(row[1]), label=row[2], kind=row[3])


def _readable_alias_entry(connection, catalog, name):
    """Resolve lowercase accounting words through exact indexed TAG metadata.

    A snake-case spelling can lose acronym capitalization. Never silently pick
    between distinct observed tags with that spelling. Curated names have an
    explicitly reviewed exact tag and do not need case reconstruction.
    """
    from .sec_names import CURATED_NAMES, accepts_name, preferred_name, spec_for_name

    spec = spec_for_name(name)
    if spec is None:
        return None
    if (spec['taxonomy'] == 'us-gaap/' and spec['tag'] in CURATED_NAMES
            and preferred_name(spec) == name):
        tags = [spec['tag']]
    else:
        with closing(source._db(catalog)) as metadata:
            if not metadata.execute("SELECT 1 FROM sqlite_master WHERE name='tags_key' AND type='index'").fetchone():
                return None  # No unindexed fallback scan for older catalogs.
            deadline = time.monotonic() + .1
            metadata.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
            try:
                tags = [row[0] for row in metadata.execute(
                    'SELECT DISTINCT tag COLLATE BINARY FROM tags INDEXED BY tags_key '
                    'WHERE tag=? COLLATE NOCASE LIMIT 65', (spec['tag'],))]
            except sqlite3.OperationalError as exc:
                if getattr(exc, 'sqlite_errorcode', None) != sqlite3.SQLITE_INTERRUPT:
                    raise
                raise ValueError('SEC accounting-name lookup exceeded its budget; use the qualified storage name') from exc
            finally:
                metadata.set_progress_handler(None, 0)
        if len(tags) > 64:
            raise ValueError('Ambiguous SEC accounting name; use the qualified storage name')
    matches = []
    for tag in tags:
        actual = dict(spec, tag=tag)
        if not accepts_name(name, actual):
            continue
        storage_name = _name_for_recipe(actual)
        row = connection.execute('SELECT name,spec,label,kind FROM fields WHERE name=?',
                                 (storage_name,)).fetchone()
        if row is None:
            continue
        entry = _entry(row)
        if entry['kind'] != 'standard' or entry['spec'] != actual:
            raise ValueError('Readable SEC name disagrees with its stored definition')
        matches.append({**entry, 'name': name, 'storage_name': storage_name})
    if len(matches) > 1:
        raise ValueError('Ambiguous SEC accounting name; use the qualified storage name')
    return matches[0] if matches else None


def readable_entry(entry, catalog, connection, context_names=None):
    """Present a readable alias without renaming a sealed execution recipe."""
    from .sec_names import preferred_name

    name = preferred_name(entry['spec']) if entry.get('kind') == 'standard' else None
    if name and name != entry['name']:
        try:
            resolved = _readable_alias_entry(connection, catalog, name)
        except ValueError as exc:
            return {**entry, 'alias_error': str(exc)}
        if resolved is not None and resolved['spec'] == entry['spec']:
            return resolved
    if name == entry['name']:
        return entry
    from .sec_context_lookup import ContextNames
    context_names = context_names or ContextNames(connection, catalog)
    name = context_names.preferred(entry)
    if name:
        try:
            resolved = context_names.lookup(name)
            if resolved and resolved['spec'] == entry['spec']:
                return resolved
        except ValueError as exc:
            return {**entry, 'alias_error': str(exc)}
    return {**entry, 'alias_error': 'Source metadata does not support an unambiguous descriptive name; inspect the exact definition.'}


def _ready(catalog):
    status = dictionary_status(catalog)
    if status["status"] != "ready":
        raise ValueError(status["message"])
    return Path(status["path"])


def candidate_entry(entry, context_names):
    """Cheap advanced-page label; selection MUST use name_for_spec to verify.

    A natural search can show many unrelated contexts. Certifying all aliases
    eagerly multiplies the per-selector budget by the page size. Mark these
    labels pending; the console verifies the selected exact definition before
    inserting it, and formula resolution independently verifies it again.
    """
    name = context_names.preferred(entry)
    if name:
        return {**entry, 'name': name, 'storage_name': entry['name'], 'alias_pending': True}
    return {**entry, 'alias_error': 'Source metadata does not support a descriptive name; inspect the exact definition.'}


def search_dictionary(catalog, query='', limit=50, offset=0, include_custom=False,
                      view='recipes'):
    """Bounded recipes (legacy default) or grouped company concepts.

    Company items contain tag/taxonomy/label/variants; every variant receives
    the same verified readable-name overlay as a flat recipe. Counts are groups,
    while field_count remains the dictionary's overall recipe metadata.
    """
    from .sec_search import search_dictionary as search
    from .sec_concepts import discover_concepts

    path = _ready(catalog)
    if view not in ('recipes', 'company'):
        raise ValueError('Invalid dictionary view')
    if not isinstance(query, str) or len(query) > 4096:
        raise ValueError('Invalid dictionary query')
    source._bounds(limit, offset)
    if type(include_custom) is not bool:
        raise ValueError('Invalid include_custom')
    query = query.strip()
    concepts = discover_concepts(catalog, query)
    with closing(_db(path)) as connection:
        from .sec_context_lookup import ContextNames
        context_names = ContextNames(connection, catalog)
        if query.startswith('sec_'):
            row = connection.execute('SELECT name,spec,label,kind FROM fields WHERE name=?', (query,)).fetchone()
            exact = (_entry(row) if row else
                     _readable_alias_entry(connection, catalog, query) or context_names.lookup(query))
            if exact is not None:
                # An exact verified identifier is already the answer. Expanding
                # it to unrelated contexts makes specialised lookup needlessly
                # slow and defeats the point of a precise name.
                spec = exact['spec']
                eligible = (include_custom or exact['kind'] == 'standard')
                if view == 'company':
                    eligible = (exact['kind'] == 'standard' and spec['segments'] == spec['coreg'] == ''
                                and bool(re.fullmatch(r'[A-Za-z][A-Za-z0-9_.-]*/', spec['taxonomy'])))
                count = int(eligible)
                items = []
                if eligible and offset == 0:
                    item = readable_entry(exact, catalog, connection, context_names)
                    items = ([dict(tag=spec['tag'], taxonomy=spec['taxonomy'], label=item['label'], variants=[item])]
                             if view == 'company' else [item])
                return dict(items=items, total=count, offset=offset, version=REGISTRY_VERSION,
                            view=view, has_more=False, truncated=False, ranking_complete=True,
                            matched_sample=count, field_count=dictionary_status(catalog)['field_count'],
                            search_scope='company_concepts' if view == 'company' else 'all_fields')
        result = search(connection, query, limit, offset, include_custom, view,
                      common_fields=COMMON_FIELDS, version=REGISTRY_VERSION,
                      field_count=dictionary_status(catalog)["field_count"],
                      concept_seeds=concepts)
        if view == 'company':
            for item in result['items']:
                item['variants'] = [readable_entry(entry, catalog, connection, context_names)
                                    for entry in item['variants']]
        else:
            result['items'] = [candidate_entry(entry, context_names) for entry in result['items']]
        return result


def lookup_fields(catalog, names):
    path = _ready(catalog)
    if isinstance(names, str):
        raise ValueError("names must be an iterable of field names")
    result = {}
    with closing(_db(path)) as connection:
        from .sec_context_lookup import ContextNames
        context_names = ContextNames(connection, catalog)
        for name in names:
            if not isinstance(name, str):
                raise ValueError("Field name must be a string")
            row = connection.execute("SELECT name,spec,label,kind FROM fields WHERE name=?", (name,)).fetchone()
            if row is not None:
                result[name] = _entry(row)
                continue  # Historical names always retain their original definition.
            entry = _readable_alias_entry(connection, catalog, name) or context_names.lookup(name)
            if entry is not None:
                result[name] = entry
    return result


def name_for_spec(catalog, spec):
    spec = _spec(spec)
    path = _ready(catalog)
    with closing(_db(path)) as connection:
        row = connection.execute("SELECT name,spec,label,kind FROM fields WHERE spec=?",
                                 (source._canonical(spec).decode(),)).fetchone()
        if row is None:
            raise ValueError("Numeric recipe is not observed/supported in this catalog")
        return readable_entry(_entry(row), catalog, connection)


def _insert(connection, spec, label, kind, description):
    serialized = source._canonical(spec).decode()
    name = _name_for_recipe(spec)
    old = connection.execute("SELECT spec,kind FROM fields WHERE name=?", (name,)).fetchone()
    if old is not None and old != (serialized, kind):
        raise ValueError(f"SEC dictionary name collision: {name}")
    connection.execute("""INSERT INTO fields(name,spec,label,kind,tag,description,taxonomy,common,unit,qtrs)
        VALUES(?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(spec) DO UPDATE SET label=MIN(fields.label,excluded.label),
        description=MIN(fields.description,excluded.description)""",
                       (name, serialized, label, kind, spec['tag'], description, spec['taxonomy'],
                        int(name in COMMON_FIELDS), spec['unit'], str(spec['qtrs'])))


def _load_tag_metadata(connection, catalog, report):
    """Group TAG in SQLite; malformed members poison the entire group.

    A -1 sentinel violates the standard CHECK even when the group also contains
    valid rows. Unknown scalar custom values remain exact-only, as before.
    Python sees neither source payloads nor an in-memory map of all concepts.
    """
    database = source._safe_path(catalog / 'catalog.sqlite')
    connection.execute("ATTACH DATABASE ? AS source_catalog", (database.as_uri() + '?mode=ro&immutable=1',))
    progress_error = []
    last_report = time.monotonic()

    def heartbeat():
        nonlocal last_report
        if time.monotonic() - last_report >= 5:
            try:
                report("Grouping TAG metadata in SQLite")
            except BaseException as exc:
                progress_error.append(exc)
                return 1
            last_report = time.monotonic()
        return 0

    try:
        report("Grouping TAG metadata in SQLite")
        connection.set_progress_handler(heartbeat, 200000)
        try:
            connection.execute("""INSERT INTO tag_metadata(tag,version,standard,label,description)
                SELECT tag,version,MIN(CASE
                    WHEN NOT json_valid(payload) THEN -1
                    WHEN json_type(payload) <> 'object' THEN -1
                    WHEN typeof(tag) <> 'text' OR typeof(version) <> 'text' THEN -1
                    WHEN COALESCE(json_type(payload,'$.tag'),'null') <> 'text'
                      OR COALESCE(json_type(payload,'$.version'),'null') <> 'text' THEN -1
                    WHEN json_extract(payload,'$.tag') <> tag COLLATE BINARY
                      OR json_extract(payload,'$.version') <> version COLLATE BINARY THEN -1
                    WHEN COALESCE(json_type(payload,'$.tlabel'),'null') NOT IN ('text','null')
                      OR COALESCE(json_type(payload,'$.doc'),'null') NOT IN ('text','null')
                      OR COALESCE(json_type(payload,'$.custom'),'null')
                         NOT IN ('text','integer','real','true','false','null') THEN -1
                    WHEN json_type(payload,'$.custom') IN ('text','integer')
                      AND CAST(json_extract(payload,'$.custom') AS TEXT)='0' THEN 1
                    ELSE 0 END),
                    MIN(COALESCE(NULLIF(json_extract(payload,'$.tlabel'),''),tag)),
                    MIN(COALESCE(json_extract(payload,'$.doc'),''))
                FROM source_catalog.tags GROUP BY tag COLLATE BINARY,version COLLATE BINARY""")
            connection.commit()
        except sqlite3.Error as exc:
            if progress_error:
                raise progress_error[0]
            raise ValueError(f"Malformed or unsupported TAG metadata: {exc}") from exc
        finally:
            connection.set_progress_handler(None, 0)
        count = connection.execute("SELECT COUNT(*) FROM tag_metadata").fetchone()[0]
        report(f"Grouped TAG metadata: {count:,} exact concepts")
        return count
    finally:
        if connection.in_transaction:
            connection.rollback()
        connection.execute("DETACH DATABASE source_catalog")


def _blocking_taxonomies(version):
    # iter_facts uses starts_with, including unusual nested/empty edition
    # strings, so a structural null under any such prefix must block its family.
    family = re.match(r'([A-Za-z][A-Za-z0-9_.-]*/)', version)
    return {version, family.group(1)} if family else {version}


def build_dictionary(catalog, progress=None):
    """Verify a catalog, scan distinct projected NUM keys, atomically seal SQLite.

    Failed/cancelled receipts survive as small sibling JSON files; only this
    invocation's unique staging directory is removed. Existing final paths are
    verified and reused, never replaced. No daily arrays are constructed.
    """
    catalog, manifest, output = _context(catalog)
    if not _supported(manifest):
        raise ValueError("Raw/notes or missing-context schema requires a numeric adapter")
    if output.exists():
        return verify_dictionary(catalog)
    import pyarrow as pa
    import pyarrow.parquet as pq

    def report(message):
        if progress is not None:
            progress(message)

    report(f"Verifying catalog: {len(manifest['tables']['num']['files'])} NUM files, "
           f"{manifest['tables']['num']['rows']:,} records; no daily arrays")
    source.read_catalog(catalog, verify=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="sec-dict-", dir=output.parent))
    connection = None
    diagnostics = dict(distinct_batch_keys=0, invalid_context_keys=0, invalid_selector_keys=0)
    recipe_cache = _RecipeCache(RECIPE_CACHE_ENTRIES, RECIPE_CACHE_BYTES)
    try:
        connection = sqlite3.connect((staging / "dictionary.sqlite").as_uri(), uri=True)
        connection.executescript("""
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=FULL;
            PRAGMA temp_store=FILE;
            CREATE TABLE fields(id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE,
                spec TEXT NOT NULL UNIQUE, label TEXT NOT NULL, kind TEXT NOT NULL,
                tag TEXT NOT NULL, description TEXT NOT NULL, taxonomy TEXT NOT NULL, common INTEGER NOT NULL,
                unit TEXT NOT NULL,qtrs TEXT NOT NULL);
            CREATE TEMP TABLE blocked_families(tag TEXT,taxonomy TEXT,
                PRIMARY KEY(tag,taxonomy)) WITHOUT ROWID;
            CREATE TEMP TABLE blocked_selectors(tag TEXT,taxonomy TEXT,unit TEXT,qtrs TEXT,
                PRIMARY KEY(tag,taxonomy,unit,qtrs)) WITHOUT ROWID;
            CREATE TEMP TABLE tag_metadata(tag TEXT,version TEXT,
                standard INTEGER NOT NULL CHECK(standard IN (0,1)),
                label TEXT NOT NULL,description TEXT NOT NULL,PRIMARY KEY(tag,version)) WITHOUT ROWID;
            PRAGMA main.cache_size=-65536;
            PRAGMA temp.cache_size=-65536;
        """)
        diagnostics['tag_concepts'] = _load_tag_metadata(connection, catalog, report)
        for index, relative in enumerate(manifest['tables']['num']['files']):
            file = pq.ParquetFile(source._safe_path(catalog / relative))
            available = set(file.schema_arrow.names)
            for batch in file.iter_batches(batch_size=BATCH_SIZE, columns=[key for key in _NUM_KEYS if key in available],
                                           use_threads=False):
                table = pa.Table.from_batches([batch])
                for key in _NUM_KEYS:
                    if key not in available:
                        table = table.append_column(key, pa.nulls(table.num_rows, type=pa.string()))
                # Python sees distinct keys only, never the full NUM record stream.
                distinct = table.select(list(_NUM_KEYS)).group_by(list(_NUM_KEYS), use_threads=False).aggregate([])
                for row in distinct.to_pylist():
                    diagnostics['distinct_batch_keys'] += 1
                    if not all(isinstance(row[key], str) for key in _SELECTOR_KEYS):
                        diagnostics['invalid_context_keys'] += 1
                        continue
                    if (not all(row[key] for key in ('tag', 'version', 'uom'))
                            or not re.fullmatch(r'0|[1-9][0-9]*', row['qtrs'])):
                        diagnostics['invalid_selector_keys'] += 1
                        continue
                    if row['segments'] is None or row['coreg'] is None:
                        diagnostics['invalid_context_keys'] += 1
                        connection.executemany("INSERT OR IGNORE INTO blocked_selectors VALUES(?,?,?,?)",
                            ((row['tag'], taxonomy, row['uom'], row['qtrs'])
                             for taxonomy in sorted(_blocking_taxonomies(row['version']))))
                        continue
                    if not all(isinstance(row[key], str) for key in ('segments', 'coreg')):
                        diagnostics['invalid_context_keys'] += 1
                        continue
                    if row['version'].endswith('/'):
                        diagnostics['invalid_selector_keys'] += 1
                        continue
                    try:
                        qtrs = int(row['qtrs'])
                    except ValueError:
                        diagnostics['invalid_selector_keys'] += 1
                        continue
                    # Diagnostics and null-prefix blocking above always run.
                    # Exact edition and both contexts are part of this raw key.
                    recipe_key = tuple(row[key] for key in _NUM_KEYS)
                    if recipe_cache.contains(recipe_key):
                        continue
                    spec = dict(tag=row['tag'], taxonomy=row['version'], unit=row['uom'],
                                qtrs=qtrs, segments=row['segments'], coreg=row['coreg'])
                    metadata = connection.execute("SELECT standard,label,description FROM tag_metadata WHERE tag=? AND version=?",
                                                  (row['tag'], row['version'])).fetchone()
                    standard, label, description = metadata or (0, row['tag'], '')
                    kind = 'standard' if standard else 'custom'
                    _insert(connection, spec, label, kind, description)
                    family = source._standard_family(row['version'])
                    if standard and family != row['version']:
                        _insert(connection, dict(spec, taxonomy=family), label, kind, description)
                    elif not standard and family != row['version']:
                        connection.execute("INSERT OR IGNORE INTO blocked_families VALUES(?,?)", (row['tag'], family))
                    # First encounters finish metadata/classification and both
                    # collision-checked inserts before becoming cacheable.
                    recipe_cache.remember(recipe_key)
                connection.commit()
            diagnostics.update(recipe_cache.diagnostics())
            report(f"Dictionary NUM file {index + 1}/{len(manifest['tables']['num']['files'])}; "
                   f"recipe cache {recipe_cache.hits:,} hits / {recipe_cache.misses:,} misses, "
                   f"{len(recipe_cache.entries):,} entries, {recipe_cache.bytes / (1024 * 1024):.1f} MiB")
        diagnostics['blocked_selector_prefixes'] = connection.execute("SELECT COUNT(*) FROM blocked_selectors").fetchone()[0]
        connection.execute("""DELETE FROM fields WHERE EXISTS (
            SELECT 1 FROM blocked_selectors b WHERE b.tag=fields.tag AND b.taxonomy=fields.taxonomy
                AND b.unit=fields.unit AND b.qtrs=fields.qtrs)""")
        diagnostics['blocked_context_recipes'] = connection.execute("SELECT changes()").fetchone()[0]
        # Keep name/spec uniqueness active throughout ingestion; build only the
        # nonunique discovery indexes after bulk inserts and blocked deletions.
        connection.executescript("""DELETE FROM fields WHERE EXISTS (
            SELECT 1 FROM blocked_families b WHERE b.tag=fields.tag AND b.taxonomy=fields.taxonomy);
            CREATE INDEX fields_kind_name ON fields(kind,common DESC,name);
            CREATE INDEX fields_common_name ON fields(common DESC,name);
            CREATE INDEX fields_tag ON fields(tag);
            CREATE VIRTUAL TABLE fields_fts USING fts5(
            name,tag,label,description,content='fields',content_rowid='id');
            INSERT INTO fields_fts(fields_fts) VALUES('rebuild');""")
        count = connection.execute("SELECT COUNT(*) FROM fields").fetchone()[0]
        connection.commit()
        connection.close()
        connection = None
        report("Verifying source stability and sealing dictionary")
        if source.read_catalog(catalog, verify=True) != manifest:
            raise ValueError("Catalog changed during dictionary build")
        seal = dict(status='ready', version=REGISTRY_VERSION, catalog_sha256=manifest['catalog_sha256'],
                    dictionary_sha256=source._hash(staging / 'dictionary.sqlite'),
                    definitions_sha256=_DEFINITIONS_SHA256, field_count=count, diagnostics=diagnostics)
        seal['manifest_sha256'] = hashlib.sha256(source._canonical(seal)).hexdigest()
        (staging / 'manifest.json').write_bytes(source._canonical(seal) + b'\n')
        report("Publishing SEC dictionary")
        # Directory rename never replaces a populated concurrent winner. Explicit
        # existence checking also prevents POSIX replacing an empty directory.
        lock = source._safe_path(output.parent / (output.name + '.publish.lock'))
        try:
            lock_stream = lock.open('xb')
        except FileExistsError:
            if output.exists():
                return verify_dictionary(catalog)
            raise FileExistsError("Concurrent dictionary publication; retry after it completes") from None
        try:
            with lock_stream:
                if output.exists():
                    return verify_dictionary(catalog)
                source._safe_path(output)
                os.rename(staging, output)
        finally:
            source._safe_path(lock).unlink()
        return verify_dictionary(catalog)
    except BaseException as exc:
        diagnostics.update(recipe_cache.diagnostics())
        receipt = dict(status='cancelled' if isinstance(exc, KeyboardInterrupt) else 'failed',
                       error=str(exc) or type(exc).__name__, catalog_sha256=manifest['catalog_sha256'],
                       version=REGISTRY_VERSION, diagnostics=diagnostics)
        (output.parent / (staging.name + '.error.json')).write_bytes(source._canonical(receipt) + b'\n')
        raise
    finally:
        if connection is not None:
            connection.close()
        if staging.exists():
            source._safe_path(staging)
            shutil.rmtree(staging)
