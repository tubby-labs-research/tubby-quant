"""Lossless, immutable local SEC tables. No numeric/PIT interpretation at import.

Progress callbacks receive messages and may raise KeyboardInterrupt to cancel.
Parquet scans are bounded; metadata calls never verify bulk bytes implicitly.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import tempfile
import zipfile
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

VERSION = 1
CHUNK_BYTES = 128 * 1024 * 1024
CSV_BLOCK_BYTES = 8 * 1024 * 1024
MAX_LIMIT = 1000
PROVENANCE = ("source_archive", "source_sha256", "source_member", "source_row")
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]*\Z")
_REQUIRED = {"sub": {"adsh", "cik", "accepted", "filed"},
             "tag": {"tag", "version"},
             "num": {"adsh", "tag", "version", "uom", "qtrs"},
             "pre": {"adsh", "tag", "version"}}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path(path):
    path = Path(os.path.abspath(path))
    for component in (path, *path.parents):
        if component.exists() or component.is_symlink():
            metadata = component.lstat()
            if component.is_symlink() or getattr(metadata, "st_file_attributes", 0) & 0x400:
                raise ValueError(f"Redirected path is not allowed: {component}")
    return path


def _member(name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or name.startswith("/") or any(p in {"", ".", ".."} for p in name.rstrip("/").split("/"))):
        raise ValueError(f"Unsafe member path: {name!r}")
    return name


def _table(name):
    leaf = PurePosixPath(name).name
    if Path(leaf).suffix.lower() not in {".txt", ".tsv"}:
        return None
    table = Path(leaf).stem.lower()
    if not _IDENTIFIER.fullmatch(table):
        raise ValueError(f"Unsafe table name: {leaf}")
    return table


def _bounds(limit, offset):
    if (isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT
            or isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset <= 1_000_000_000):
        raise ValueError(f"limit must be 1..{MAX_LIMIT}; offset must be 0..1000000000")


def _db(path):
    database = _safe_path(Path(path) / "catalog.sqlite")
    return sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)


def build_catalog(archives: list[Path], output: Path, *, ticker_map: Path, progress=None) -> dict:
    """Import ZIPs or top-level table directories into a newly published catalog.

    All NUM work is columnar. SQLite contains SUB/TAG only. Failure receipts stay
    in a uniquely named sibling staging directory; output is published only ready.
    Do not mix face and notes products in one catalog.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as csv
    import pyarrow.parquet as pq

    output = _safe_path(output)
    ticker_map = _safe_path(ticker_map)
    inputs = [_safe_path(path) for path in archives]
    if not inputs or len(set(inputs)) != len(inputs):
        raise ValueError("Provide distinct input archives/directories")
    if output.exists():
        raise FileExistsError(output)
    if not ticker_map.is_file():
        raise FileNotFoundError(ticker_map)
    for source in [*inputs, ticker_map]:
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("Catalog output must be disjoint from source paths")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".building-", dir=output.parent))
    manifest = {"schema_version": VERSION, "status": "building", "tables": {}, "sources": [],
                "created_at": datetime.now(timezone.utc).isoformat(), "provenance_columns": list(PROVENANCE)}
    connection = None
    products = set()
    def report(message):
        if progress is not None:
            progress(message)
    try:
        ticker_hash = _hash(ticker_map)
        with ticker_map.open("r", encoding="utf-8-sig") as stream:
            if not isinstance(json.load(stream), dict):
                raise ValueError("Ticker JSON must contain an object")
        shutil.copyfile(ticker_map, staging / "tickers.json")
        manifest["ticker_map"] = {"path": str(ticker_map), "sha256": ticker_hash,
                                  "bytes": ticker_map.stat().st_size, "artifact": "tickers.json"}
        connection = sqlite3.connect(staging / "catalog.sqlite")
        connection.executescript("""
            PRAGMA journal_mode=DELETE;
            CREATE TABLE filings(adsh TEXT NOT NULL, payload TEXT NOT NULL, identity TEXT NOT NULL);
            CREATE TABLE tags(tag TEXT NOT NULL, version TEXT NOT NULL, search TEXT NOT NULL,
                              payload TEXT NOT NULL UNIQUE, occurrences INTEGER NOT NULL);
            CREATE VIRTUAL TABLE tags_fts USING fts5(search, content='tags', content_rowid='rowid');
        """)
        sequence = defaultdict(int)
        source_checks = []
        for source in inputs:
            report(f"Reading {source.name}")
            archive = None
            inventory = []
            try:
                if source.is_dir():
                    entries = sorted(source.iterdir())
                    for entry in entries:
                        _safe_path(entry)
                        if not entry.is_file():
                            raise ValueError("Table directory must contain top-level files only")
                        inventory.append({"name": entry.name, "bytes": entry.stat().st_size,
                                          "sha256": _hash(entry)})
                    source_hash = hashlib.sha256(_canonical(inventory)).hexdigest()
                    open_member = lambda name: (source / name).open("rb")
                    source_checks.extend((source / row["name"], row["sha256"]) for row in inventory)
                else:
                    source_hash = _hash(source)
                    source_checks.append((source, source_hash))
                    archive = zipfile.ZipFile(source)
                    seen = set()
                    for info in archive.infolist():
                        _member(info.filename)
                        if (info.filename.casefold() in seen or info.flag_bits & 1
                                or stat.S_ISLNK(info.external_attr >> 16)):
                            raise ValueError("Duplicate, encrypted or symlink ZIP member")
                        seen.add(info.filename.casefold())
                        if not info.is_dir():
                            inventory.append({"name": info.filename, "bytes": info.file_size, "crc32": info.CRC})
                    open_member = archive.open
                tables = {}
                for entry in inventory:
                    table = _table(entry["name"])
                    if table is not None:
                        if table in tables:
                            raise ValueError(f"Duplicate table member: {table}")
                        tables[table] = entry["name"]
                    elif archive is not None:
                        digest = hashlib.sha256()
                        with open_member(entry["name"]) as stream:
                            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                                digest.update(chunk)
                        entry["sha256"] = digest.hexdigest()  # also checks CRC of non-tables
                if not tables:
                    raise ValueError("Source contains no tabular .txt/.tsv members")
                product = "notes" if {"dim", "txt", "cal", "ren"} & tables.keys() else "face"
                products.add(product)
                if len(products) > 1:
                    raise ValueError("Build separate face and notes catalogs")
                source_record = {"path": str(source), "sha256": source_hash,
                                 "kind": "directory" if source.is_dir() else "zip",
                                 "product": product, "members": inventory, "table_counts": {}}
                manifest["sources"].append(source_record)
                for table, member in sorted(tables.items()):
                    report(f"Reading {source.name}: {member}")
                    with open_member(member) as stream:
                        header = stream.readline().decode("utf-8-sig").rstrip("\r\n").split("\t")
                    if (not header or any(not name for name in header) or len(set(header)) != len(header)
                            or set(header) & set(PROVENANCE) or not _REQUIRED.get(table, set()) <= set(header)):
                        raise ValueError(f"Malformed {member} schema")
                    record = manifest["tables"].setdefault(table, {"rows": 0, "columns": [], "files": []})
                    record["columns"] = list(dict.fromkeys([*record["columns"], *header]))
                    folder = staging / "tables" / table
                    folder.mkdir(parents=True, exist_ok=True)
                    buffered, buffered_bytes, offset = [], 0, 0
                    def flush():
                        nonlocal buffered, buffered_bytes
                        if not buffered:
                            return
                        block = pa.Table.from_batches(buffered)
                        if table == "num":
                            block = block.take(pc.sort_indices(block, sort_keys=[("tag", "ascending"), ("version", "ascending")]))
                        relative = f"tables/{table}/part-{sequence[table]:06d}.parquet"
                        sequence[table] += 1
                        pq.write_table(block, staging / relative, compression="zstd", row_group_size=65536,
                                       write_statistics=True)
                        record["files"].append(relative)
                        buffered, buffered_bytes = [], 0
                    with open_member(member) as stream:
                        reader = csv.open_csv(stream,
                            read_options=csv.ReadOptions(block_size=CSV_BLOCK_BYTES, use_threads=False, encoding="utf8"),
                            parse_options=csv.ParseOptions(delimiter="\t", quote_char='"', double_quote=True,
                                                           newlines_in_values=True, ignore_empty_lines=False),
                            convert_options=csv.ConvertOptions(column_types={name: pa.string() for name in header},
                                                               strings_can_be_null=False, null_values=[]))
                        for batch in reader:
                            # No per-NUM-row Python work, including provenance construction.
                            count = batch.num_rows
                            columns = list(batch.columns) + [pa.repeat(str(source), count), pa.repeat(source_hash, count),
                                                            pa.repeat(member, count), pc.add(pc.indices_nonzero(pa.repeat(True, count)), offset + 1)]
                            enriched = pa.RecordBatch.from_arrays(columns, names=header + list(PROVENANCE))
                            if table == "sub":
                                rows = enriched.to_pylist()
                                connection.executemany("INSERT INTO filings VALUES(?,?,?)", (
                                    (row["adsh"], _canonical(row).decode(), _canonical({k: row[k] for k in header}).decode())
                                    for row in rows))
                            elif table == "tag":
                                connection.executemany("INSERT INTO tags VALUES(?,?,?,?,1) ON CONFLICT(payload) DO UPDATE SET occurrences=occurrences+1", (
                                    (row["tag"], row["version"], " ".join(row.values()).casefold(), _canonical(row).decode())
                                    for row in batch.to_pylist()))
                            offset += count
                            record["rows"] += count
                            buffered.append(enriched)
                            buffered_bytes += enriched.nbytes
                            if buffered_bytes >= CHUNK_BYTES:
                                flush()
                            report(f"{source.name}: {table} {offset:,} rows")
                        flush()
                    # Keep a schema-bearing empty file for empty tables.
                    if offset == 0:
                        relative = f"tables/{table}/part-{sequence[table]:06d}.parquet"
                        sequence[table] += 1
                        empty = pa.table({**{name: pa.array([], type=pa.string()) for name in header},
                                          **{name: pa.array([], type=pa.string()) for name in PROVENANCE[:-1]},
                                          "source_row": pa.array([], type=pa.int64())})
                        pq.write_table(empty, staging / relative, compression="zstd")
                        record["files"].append(relative)
                    source_record["table_counts"][table] = offset
                    connection.commit()
            finally:
                if archive is not None:
                    archive.close()
        report("Indexing filing and tag metadata")
        connection.executescript("""CREATE INDEX filings_adsh ON filings(adsh);
            CREATE INDEX tags_key ON tags(tag COLLATE NOCASE,version);
            INSERT INTO tags_fts(tags_fts) VALUES('rebuild');""")
        connection.commit()
        connection.close()
        connection = None
        report("Verifying source stability and sealing catalog")
        for source, expected in [*source_checks, (ticker_map, ticker_hash)]:
            _safe_path(source)
            if _hash(source) != expected:
                raise ValueError(f"Source changed during import: {source}")
        for source in inputs:
            if source.is_dir():
                original = next(row for row in manifest["sources"] if row["path"] == str(source))
                if {p.name for p in source.iterdir()} != {row["name"] for row in original["members"]}:
                    raise ValueError("Directory inventory changed during import")
        if _hash(staging / "tickers.json") != ticker_hash:
            raise ValueError("Ticker copy differs from source")
        manifest["supported_products"] = sorted(products)
        manifest["numeric_materialization_supported"] = products == {"face"}
        manifest["artifacts_sha256"] = {p.relative_to(staging).as_posix(): _hash(p)
                                         for p in sorted(staging.rglob("*")) if p.is_file()}
        manifest["status"] = "ready"
        manifest["catalog_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
        (staging / "manifest.json").write_bytes(_canonical(manifest) + b"\n")
        report("Publishing ready catalog")
        if output.exists():
            raise FileExistsError(output)
        os.rename(staging, output)
        return manifest
    except BaseException as exc:
        if connection is not None:
            connection.close()
        manifest.pop("catalog_sha256", None)
        manifest["status"] = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = str(exc) or type(exc).__name__
        (staging / "manifest.json").write_bytes(_canonical(manifest) + b"\n")
        raise


def read_catalog(path: Path, *, verify=False) -> dict:
    path = _safe_path(path)
    with _safe_path(path / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != VERSION or manifest.get("status") != "ready":
        raise ValueError("Catalog is not a ready supported manifest")
    identity = dict(manifest)
    expected = identity.pop("catalog_sha256", None)
    if expected != hashlib.sha256(_canonical(identity)).hexdigest():
        raise ValueError("Catalog manifest identity differs")
    assets = manifest.get("artifacts_sha256")
    if not isinstance(assets, dict) or not assets:
        raise ValueError("Catalog lacks artifact hashes")
    for relative, digest in assets.items():
        _member(relative)
        if not re.fullmatch(r"[a-f0-9]{64}", str(digest)):
            raise ValueError("Malformed artifact digest")
    for table, record in manifest["tables"].items():
        if not _IDENTIFIER.fullmatch(table):
            raise ValueError("Unsafe catalog table")
        for relative in record["files"]:
            if relative not in assets or not relative.startswith(f"tables/{table}/") or not relative.endswith(".parquet"):
                raise ValueError("Unsealed table file")
    if verify:
        actual = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file() and p != path / "manifest.json"}
        if actual != set(assets):
            raise ValueError("Catalog artifact inventory differs")
        for relative, digest in assets.items():
            if _hash(_safe_path(path / relative)) != digest:
                raise ValueError(f"Catalog artifact differs: {relative}")
    return manifest


def search_tags(path, query='', limit=50, offset=0) -> dict:
    """Rank exact standard tags, exact custom tags, prefixes, then text matches.

    Standard means the supplied TAG.custom is zero, not a taxonomy whitelist.
    Nonempty searches show newer version strings first within each tag/tier.
    """
    _bounds(limit, offset)
    read_catalog(Path(path))
    if not isinstance(query, str) or len(query) > 4096:
        raise ValueError("Tag query must be a string of at most 4096 characters")
    with closing(_db(path)) as connection:
        # FTS is an optional acceleration: pre-FTS version-1 catalogs remain
        # readable without updating their sealed SQLite bytes or manifest.
        has_index = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?", ('tags_key',)).fetchone()
        index_hint = ' INDEXED BY tags_key' if has_index else ''
        matches_cte, membership, params = '', '', ()
        if not query:
            total = connection.execute('SELECT count(*) FROM tags').fetchone()[0]
        else:
            prefix = query.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_') + '%'
            has_fts = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", ('tags_fts',)).fetchone()
            tokens = re.findall(r'\w+', query, re.UNICODE)
            if has_fts and tokens:
                match = ' AND '.join('"' + token.replace('"', '""') + '"*' for token in tokens)
                # UNION deduplicates IDs, not large payloads, and does not hide
                # valid label/doc matches when a tag prefix also matches.
                matches_cte = "matches AS (SELECT rowid FROM tags WHERE tag LIKE ? ESCAPE '\\' UNION SELECT rowid FROM tags_fts WHERE tags_fts MATCH ?), "
                params = (prefix, match)
            elif has_fts:
                matches_cte = "matches AS (SELECT rowid FROM tags WHERE tag LIKE ? ESCAPE '\\'), "
                params = (prefix,)
            else:
                literal = query.casefold().replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
                matches_cte = "matches AS (SELECT rowid FROM tags WHERE tag LIKE ? ESCAPE '\\' UNION SELECT rowid FROM tags WHERE search LIKE ? ESCAPE '\\'), "
                params = (prefix, '%' + literal + '%')
            total = connection.execute('WITH ' + matches_cte[:-2] + ' SELECT count(*) FROM matches', params).fetchone()[0]
            membership = ' WHERE rowid IN (SELECT rowid FROM matches)'
        # Sort compact keys, never full payloads. Only exact-tag candidates need
        # payload access to inspect custom; other tiers use indexed tag keys.
        # Empty browsing remains a covering-index page with no JSON inspection.
        rank, rank_params = '0', ()
        order = 'tag COLLATE NOCASE,version,rowid'
        if query:
            rank = ("CASE WHEN tag = ? COLLATE NOCASE THEN CASE WHEN "
                    "CAST(json_extract(payload,'$.custom') AS TEXT) = '0' THEN 0 ELSE 1 END "
                    "WHEN tag LIKE ? ESCAPE '\\' THEN 2 ELSE 3 END")
            rank_params = (query, prefix)
            order = 'rank,tag COLLATE NOCASE,version DESC,rowid'
        sql = ('WITH ' + matches_cte + 'page AS (SELECT rowid,tag,version,' + rank + ' AS rank FROM tags' + index_hint + membership
               + ' ORDER BY ' + order + ' LIMIT ? OFFSET ?) '
               + 'SELECT t.payload,t.occurrences FROM page p JOIN tags t ON t.rowid=p.rowid '
               + 'ORDER BY ' + ( 'p.rank,p.tag COLLATE NOCASE,p.version DESC,p.rowid' if query
                                else 'p.tag COLLATE NOCASE,p.version,p.rowid'))
        rows = connection.execute(sql, (*params, *rank_params, limit, offset))
        items = [dict(json.loads(payload), occurrences=count) for payload, count in rows]
    return {"items": items, "total": total}


def _standard_family(version):
    """Only collapse versions compatible with the existing family selector."""
    match = re.fullmatch(r'([A-Za-z][A-Za-z0-9_.-]*/)[^/]+', version or '')
    return match.group(1) if match else version


def search_fields(path, query='', limit=50, offset=0, include_custom=False) -> dict:
    """Discover concepts using TAG metadata only; group before paging.

    Matching definitions nominate a concept. Its complete release list and newest
    representative are then obtained by indexed exact-tag lookups for this page.
    No sealed catalog changes, bulk verification, or NUM scans are performed.
    """
    _bounds(limit, offset)
    if (not isinstance(query, str) or len(query) > 256 or offset > 10_000_000
            or type(include_custom) is not bool):
        raise ValueError('Invalid SEC field search or page')
    read_catalog(Path(path))
    query = query.strip()
    if not query:
        return {'items': [], 'total': 0}
    literal = query.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
    prefix = literal + '%'
    with closing(_db(path)) as connection:
        connection.create_function('standard_family', 1, _standard_family, deterministic=True)
        has_fts = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name='tags_fts' AND type='table'").fetchone()
        has_index = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name='tags_key' AND type='index'").fetchone()
        index = ' INDEXED BY tags_key' if has_index else ''
        tokens = re.findall(r'\w+', query, re.UNICODE)
        matches = "SELECT rowid FROM tags WHERE tag LIKE ? ESCAPE '\\'"
        params = [prefix]
        if has_fts and tokens:
            matches += ' UNION SELECT rowid FROM tags_fts WHERE tags_fts MATCH ?'
            params.append(' AND '.join('"' + token + '"*' for token in tokens))
        elif not has_fts:
            # Legacy catalogs have no FTS. Inspect search text, never all JSON.
            matches += " UNION SELECT rowid FROM tags WHERE search LIKE ? ESCAPE '\\'"
            params.append('%' + literal + '%')
        cte = ("WITH matches AS (" + matches + "), candidates AS ("
               "SELECT tag,version,COALESCE(CAST(json_extract(payload,'$.custom') AS TEXT)='0',0) AS standard "
               "FROM tags WHERE rowid IN (SELECT rowid FROM matches)), concepts AS ("
               "SELECT tag,CASE WHEN standard THEN standard_family(version) ELSE version END AS taxonomy,"
               "standard FROM candidates WHERE standard OR ? GROUP BY tag COLLATE BINARY,taxonomy,standard) ")
        params.append(int(include_custom))
        total = connection.execute(cte + 'SELECT count(*) FROM concepts', params).fetchone()[0]
        page = connection.execute(
            cte + "SELECT tag,taxonomy,standard FROM concepts ORDER BY standard DESC,"
            "CASE WHEN tag=? COLLATE NOCASE THEN 0 WHEN tag LIKE ? ESCAPE '\\' THEN 1 ELSE 2 END,"
            "length(tag),tag COLLATE NOCASE,tag COLLATE BINARY,taxonomy LIMIT ? OFFSET ?",
            (*params, query, prefix, limit, offset)).fetchall()
        items = []
        for tag, taxonomy, standard in page:
            # Both predicates keep exact-case concepts distinct while using the
            # catalog's existing NOCASE covering index to bound candidate rows.
            releases = connection.execute(
                'SELECT version,min(rowid) FROM tags' + index +
                " WHERE tag=? COLLATE NOCASE AND tag=? COLLATE BINARY "
                "AND COALESCE(CAST(json_extract(payload,'$.custom') AS TEXT)='0',0)=? "
                'AND ' + ('standard_family(version)=?' if standard else 'version=?') +
                ' GROUP BY version ORDER BY version DESC', (tag, tag, standard, taxonomy)).fetchall()
            version, representative = releases[0]
            payload = json.loads(connection.execute(
                'SELECT payload FROM tags WHERE rowid=?', (representative,)).fetchone()[0])
            item = {key: payload.get(key, '') for key in ('tlabel', 'doc', 'datatype', 'iord')}
            item.update(tag=tag, version=version, custom='0' if standard else str(payload.get('custom') or ''),
                        taxonomy=taxonomy, versions=[release for release, _ in releases],
                        kind='standard' if standard else 'custom')
            items.append(item)
    return {'items': items, 'total': total}


def _scan(path, manifest, table, filters=None, columns=None, taxonomy=None):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    if table not in manifest["tables"]:
        raise ValueError(f"Unknown table: {table}")
    record = manifest["tables"][table]
    names = set(record["columns"]) | set(PROVENANCE)
    expression = None
    if filters is not None and not isinstance(filters, dict):
        raise ValueError("Filters must be a dictionary")
    for key, value in (filters or {}).items():
        if key not in names or not isinstance(value, str):
            raise ValueError("Filters require exact known-column string values")
        term = ds.field(key) == (int(value) if key == "source_row" else value)
        expression = term if expression is None else expression & term
    if taxonomy is not None and taxonomy.endswith('/'):
        term = pc.starts_with(ds.field('version'), pattern=taxonomy)
        expression = term if expression is None else expression & term
    if columns is not None and not set(columns) <= names:
        raise ValueError("Unknown projected columns")
    # Unify header evolution explicitly; absent source columns remain null rather
    # than being confused with original blank strings.
    files = [_safe_path(Path(path) / relative) for relative in record["files"]]
    schema = pa.schema([(name, pa.string()) for name in record["columns"] + list(PROVENANCE[:-1])]
                       + [("source_row", pa.int64())])
    dataset = ds.dataset(files, format="parquet", schema=schema)
    yield from dataset.scanner(filter=expression, columns=columns, batch_size=16384,
                              use_threads=False, batch_readahead=1, fragment_readahead=1).to_batches()


def query_rows(path, table, filters: dict[str, str] | None = None, limit=100, offset=0) -> list[dict]:
    _bounds(limit, offset)
    manifest = read_catalog(Path(path))
    result = []
    for batch in _scan(path, manifest, table, filters):
        if offset >= batch.num_rows:
            offset -= batch.num_rows
            continue
        result.extend(batch.slice(offset, limit - len(result)).to_pylist())
        offset = 0
        if len(result) == limit:
            break
    return result


def _selector(tag, taxonomy):
    if not isinstance(tag, str) or not tag or not isinstance(taxonomy, str) or not taxonomy:
        raise ValueError("An explicit tag and taxonomy selector are required")
    if taxonomy.endswith('/') and not re.fullmatch(r'[A-Za-z][A-Za-z0-9_.-]*/', taxonomy):
        raise ValueError("Custom taxonomy versions must be exact")
    return {"tag": tag, **({} if taxonomy.endswith('/') else {"version": taxonomy})}


def field_variants(path, tag, taxonomy) -> list[dict]:
    manifest = read_catalog(Path(path))
    filters = _selector(tag, taxonomy)
    available = manifest["tables"].get("num", {}).get("columns", [])
    columns = [name for name in ("version", "uom", "qtrs", "segments", "coreg", "dimh", "dimn", "iprx") if name in available]
    variants = set()
    import pyarrow as pa
    for batch in _scan(path, manifest, "num", filters, columns, taxonomy=taxonomy):
        # Projected selected-tag scan is an explicit UI action, not editor validation.
        for row in pa.Table.from_batches([batch]).group_by(columns, use_threads=False).aggregate([]).to_pylist():
            if not taxonomy.endswith('/') or row["version"].startswith(taxonomy):
                variants.add(tuple(row[name] for name in columns))
    return [dict(zip(columns, values)) for values in sorted(variants, key=lambda values: tuple((v is not None, v or '') for v in values))]


def iter_facts(path, spec: dict):
    """Stream NUM+SUB matches; never choose a winner among conflicting filings.

    Original NUM keys/provenance remain top-level; SUB fields are top-level when
    non-colliding, and the complete SUB record is also ``sub``. ``filing_conflict``
    flags distinct SUB content; identical duplicates retain each provenance row.
    Missing matches yield ``sub=None`` and ``filing_missing=True``. Notes remain
    raw-queryable but are rejected here until dimension semantics are defined.
    """
    manifest = read_catalog(Path(path))
    if not manifest.get("numeric_materialization_supported"):
        raise ValueError("Notes/other dimensional products require an explicit schema adapter; raw queries remain supported")
    filters = _selector(spec.get("tag"), spec.get("taxonomy"))
    qtrs = spec.get("qtrs")
    if isinstance(qtrs, bool) or not isinstance(qtrs, int) or qtrs < 0:
        raise ValueError("qtrs must be an explicit nonnegative integer")
    for name, value in (("uom", spec.get("unit")), ("qtrs", str(qtrs)),
                        ("segments", spec.get("segments", "")), ("coreg", spec.get("coreg", ""))):
        if not isinstance(value, str):
            raise ValueError("unit/segments/coreg must be exact strings")
        filters[name] = value
    # Missing source context columns become structural nulls after schema union.
    # Never treat them as consolidated blanks or silently remove selected facts.
    import pyarrow.compute as pc
    contexts = {key: filters.pop(key) for key in ('segments', 'coreg')}
    with closing(_db(path)) as connection:
        for batch in _scan(path, manifest, "num", filters, taxonomy=spec['taxonomy']):
            mask = None
            for key, value in contexts.items():
                index = batch.schema.get_field_index(key)
                if index < 0 or batch.column(index).null_count:
                    raise ValueError(f"Selected NUM rows lack structural context column {key}; explicit schema adapter required")
                term = pc.equal(batch.column(index), value)
                mask = term if mask is None else pc.and_(mask, term)
            batch = batch.filter(mask)
            rows = batch.to_pylist()
            if spec["taxonomy"].endswith('/'):
                rows = [row for row in rows if row["version"].startswith(spec["taxonomy"])]
            accessions = sorted({row["adsh"] for row in rows})
            filings = defaultdict(list)
            for begin in range(0, len(accessions), 500):
                selected = accessions[begin:begin + 500]
                marks = ','.join('?' for _ in selected)
                for accession, payload, identity in connection.execute(f"SELECT adsh,payload,identity FROM filings WHERE adsh IN ({marks})", selected):
                    filings[accession].append((json.loads(payload), identity))
            for row in rows:
                matches = filings[row["adsh"]]
                conflict = len({identity for _, identity in matches}) > 1
                for sub, _ in matches or [(None, None)]:
                    merged = {**(sub or {}), **row, "sub": sub, "filing_conflict": conflict,
                              "filing_missing": sub is None, "sub_match_count": len(matches)}
                    yield merged
