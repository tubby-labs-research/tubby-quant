"""Budgeted discovery on the existing immutable SEC dictionary.

Registry integration: ``search_dictionary(connection, query, limit, offset,
include_custom, common_fields=COMMON_FIELDS, version=REGISTRY_VERSION,
field_count=seal['field_count'])``. The registry still owns readiness and opening
the read-only connection. No schema, recipe or verification changes are made.

FTS retains literal AND token-prefix semantics. Exact names and nominated
concept defaults survive alongside FTS matches. Large results are a bounded sample, never an exact
count or globally ranked page. Sample pages are available only when collection
reached its fixed row boundary; interrupted samples cannot promise stable pages.
Every recipe remains accessible by indexed name or unfiltered indexed browsing.
SQLite progress callbacks bound VM work, not a stalled filesystem read.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from itertools import chain

from .sec_catalog import _bounds

SEARCH_BUDGET_MS = 200
CANDIDATE_LIMIT = 2048
TAG_CANDIDATE_LIMIT = 128
_COLUMNS = 'id,name,label,kind,taxonomy,common,tag'


def _company_filter(prefix=''):
    """Filter before LIMIT using sealed recipe metadata, never raw NUM facts."""
    taxonomy, spec = prefix + 'taxonomy', prefix + 'spec'
    return (f"{prefix}kind='standard' AND {taxonomy} GLOB '[A-Za-z]*/' "
            f"AND substr({taxonomy},1,length({taxonomy})-1) NOT GLOB '*[^A-Za-z0-9_.-]*' "
            f"AND json_extract({spec},'$.segments')='' "
            f"AND json_extract({spec},'$.coreg')=''")


def search_dictionary(connection, query='', limit=50, offset=0,
                      include_custom=False, view='recipes', *, common_fields=None,
                      version='sec-dictionary-2', field_count=None,
                      budget_ms=SEARCH_BUDGET_MS, candidate_limit=CANDIDATE_LIMIT,
                      concept_seeds=()):
    """Return items/total/offset/version plus explicit completeness metadata.

    ``total`` is nullable; ``has_more`` describes another page in the returned
    sample, not exhaustion of all matches. ``truncated`` always signals omitted
    matches or interrupted discovery. ``matched_sample`` counts sampled matches.
    A dedicated connection is required (its progress handler is cleared on exit).
    concept_seeds accepts actual source tag strings or {tag,spec,rank} dicts.
    preferred=True pins an explicit alias spec; expand_defaults=True permits
    default unit/duration probes for a bare reviewed concept base. Source helper
    truncation is carried by its list's ``truncated`` attribute.
    Company items group EXACT tag/taxonomy before pagination; variants are full
    observed blank-context standard-family recipes, possibly a partial sample.
    Counts and paging describe groups. Empty company queries cover reviewed
    starters only, never the global inventory. field_count is recipe metadata.
    """
    _bounds(limit, offset)
    if not isinstance(query, str) or len(query) > 4096 or type(include_custom) is not bool:
        raise ValueError('Invalid dictionary query/include_custom')
    if view not in ('recipes', 'company'):
        raise ValueError('Invalid dictionary view')
    if (type(budget_ms) not in (int, float) or not 0 < budget_ms <= 10000
            or type(candidate_limit) is not int or not 1 <= candidate_limit <= 10000):
        raise ValueError('Invalid search budget/candidate limit')
    query = query.strip()
    company = view == 'company'
    starter = company and not query
    columns = _COLUMNS + ',spec' if company else _COLUMNS
    tokens = re.findall(r'[^\W_]+', query, flags=re.UNICODE)
    result = dict(items=[], total=None, offset=offset, version=version,
                  view=view, has_more=False, truncated=False,
                  search_scope=('company_starter' if starter else 'company_concepts')
                  if company else 'all_fields',
                  ranking_complete=True, matched_sample=0)
    if field_count is not None:
        result['field_count'] = field_count
    if query and not tokens:
        result['total'] = 0
        return result
    deadline = time.monotonic() + budget_ms / 1000
    interrupted = False

    def progress():
        return int(time.monotonic() >= deadline)

    def rows(sql, params=()):
        nonlocal interrupted
        cursor = None
        try:
            cursor = connection.execute(sql, params)
            for row in cursor:
                yield row
        except sqlite3.OperationalError as exc:
            if getattr(exc, 'sqlite_errorcode', None) != sqlite3.SQLITE_INTERRUPT:
                raise
            interrupted = True
        finally:
            if cursor is not None:
                cursor.close()

    def entries(selected):
        # Company predicates already read the (blank-context) specs. Carry
        # those small specs with candidates to avoid per-variant DB lookups
        # outside the discovery budget; parse only the requested group page.
        # Recipes hydrate only the requested page of possibly large contexts.
        for row in selected:
            item = ((row[1], row[7], row[2], row[3]) if company else
                    connection.execute('SELECT name,spec,label,kind FROM fields WHERE id=?', (row[0],)).fetchone())
            yield dict(name=item[0], spec=json.loads(item[1]), label=item[2], kind=item[3])

    connection.set_progress_handler(progress, 1000)
    try:
        if not query and not company:
            index = 'fields_common_name' if include_custom else 'fields_kind_name'
            where = '' if include_custom else " WHERE kind='standard'"
            page = list(rows(f'SELECT {_COLUMNS} FROM fields INDEXED BY {index}' + where +
                             ' ORDER BY common DESC,name LIMIT ? OFFSET ?', (limit + 1, offset)))
            result['has_more'] = len(page) > limit if not interrupted else None
            result['truncated'] = interrupted
            if include_custom and type(field_count) is int:
                result['total'] = field_count
            elif not interrupted:
                count = list(rows('SELECT COUNT(*) FROM fields INDEXED BY fields_kind_name' + where))
                if count:
                    result['total'] = count[0][0]
            # A count timeout does not invalidate an otherwise complete browse page.
            connection.set_progress_handler(None, 0)
            result['items'] = list(entries(page[:limit]))
            result['matched_sample'] = len(result['items'])
            if result['truncated']:
                result['message'] = 'Browsing exceeded the search budget. Use an exact official name or a smaller offset.'
            return result

        match = ' AND '.join('"' + token + '"*' for token in tokens)
        candidates = {}
        relevance = {}
        tag_relevance = {}
        seed_truncated = bool(getattr(concept_seeds, 'truncated', False))
        source_interrupted = bool(getattr(concept_seeds, 'interrupted', False))
        if starter:
            from .sec_names import curated_defs
            concept_seeds = [dict(tag=spec['tag'], spec=spec, rank=rank,
                                  preferred=True, expand_defaults=True)
                             for rank, spec in enumerate(curated_defs().values())]
            seed_truncated = source_interrupted = False
        elif company and len(tokens) == 1:
            # Keep common default name probes even for callers without TAG
            # nominations: advanced rows must not crowd out the blank default.
            concept_seeds = chain(concept_seeds, [dict(tag=spec['tag'], spec=spec)
                for name, spec in (common_fields or {}).items()
                if name.lower() in {query.lower(), 'sec_' + query.lower()}
                or spec['tag'].lower() == query.lower()])
        exact_names = {query.lower(), 'sec_' + query.lower()}
        eligibility = ' AND ' + _company_filter() if company else ''
        # Binary name equality uses the UNIQUE index, unlike a NOCASE scan.
        for name in sorted(exact_names) if not starter else ():
            for row in rows(f'SELECT {columns} FROM fields WHERE name=?' + eligibility, (name,)):
                if include_custom or row[3] == 'standard':
                    candidates[row[0]] = row

        # Source nomination happens before repeated recipe FTS. Canonical names
        # are derived from exact specs and use the dictionary's UNIQUE index.
        # Import lazily: registry calls this module, but owns recipe naming.
        from .sec_registry import _name_for_recipe, _spec
        seed_tags = set()
        for index, seed in enumerate(concept_seeds):
            if index >= 512 or interrupted:
                seed_truncated = True
                break
            if isinstance(seed, str):
                tag, spec, rank = seed, None, 0
            elif isinstance(seed, dict):
                tag, spec, rank = seed.get('tag'), seed.get('spec'), seed.get('rank', 0)
            else:
                raise ValueError('Invalid concept seed')
            if not isinstance(tag, str) or not tag or type(rank) is not int or rank < 0:
                raise ValueError('Invalid concept seed tag/rank')
            if tag not in seed_tags and len(seed_tags) >= 128:
                seed_truncated = True
                break
            seed_tags.add(tag)
            tag_relevance[tag] = min(tag_relevance.get(tag, 10**9), rank)
            if spec is None:
                spec = dict(tag=tag, taxonomy='us-gaap/', unit='USD', qtrs=0, segments='', coreg='')
            spec = _spec(spec)
            if spec['tag'] != tag:
                raise ValueError('Concept seed tag differs from spec')
            recipes = [spec]
            if ((not (isinstance(seed, dict) and seed.get('preferred'))
                 or (isinstance(seed, dict) and seed.get('expand_defaults')))
                    and spec['taxonomy'].endswith('/') and spec['segments'] == spec['coreg'] == ''):
                recipes += [dict(spec, unit=unit, qtrs=qtrs)
                            for unit in ('USD', 'USD/shares', 'shares') for qtrs in range(5)]
            for recipe in recipes:
                if time.monotonic() >= deadline:
                    interrupted = True
                    break
                name = _name_for_recipe(recipe)
                for row in rows(f'SELECT {columns} FROM fields WHERE name=?' + eligibility, (name,)):
                    if include_custom or row[3] == 'standard':
                        candidates[row[0]] = row
                        unit_rank = {'USD': 0, 'USD/shares': 1, 'shares': 2}.get(recipe['unit'], 3)
                        score = (0 if isinstance(seed, dict) and seed.get('preferred') and recipe == spec else 1,
                                 unit_rank, recipe['qtrs'])
                        relevance[row[0]] = min(relevance.get(row[0], (9,9,10**9)), score)

        # Preserve nondefault observations: a concept may have only EUR/q1,
        # USD/q6, or (in recipes view) a nonblank context. Default probes run
        # first; each subsequent tag supplement is a bounded indexed sample.
        for tag in sorted(seed_tags, key=lambda tag: (tag_relevance[tag], tag)):
            if interrupted:
                break
            where = eligibility if company else '' if include_custom else " AND kind='standard'"
            # Reviewed starters use their reviewed family, not other families
            # or concepts that happen to share the same spelling.
            if starter:
                where += " AND taxonomy='us-gaap/'"
            tag_rows = 0
            for row in rows(f'SELECT {columns} FROM fields INDEXED BY fields_tag WHERE tag=?' +
                            where + ' LIMIT ?', (tag, TAG_CANDIDATE_LIMIT + 1)):
                tag_rows += 1
                if tag_rows > TAG_CANDIDATE_LIMIT:
                    seed_truncated = True
                    break
                candidates[row[0]] = row

        # Inventory's common definition points to InventoryNet, not raw Inventory.
        # Seeds only apply to one token; multi-token description semantics stay FTS.
        if len(tokens) == 1 and not interrupted:
            tags = {query}
            for name, spec in (common_fields or {}).items():
                if name.lower() in exact_names or spec['tag'].lower() == query.lower():
                    tags.add(spec['tag'])
            for tag in sorted(tags):
                if tag in seed_tags:
                    continue
                where = eligibility if company else '' if include_custom else " AND kind='standard'"
                tag_rows = 0
                for row in rows(f'SELECT {columns} FROM fields INDEXED BY fields_tag WHERE tag=?' +
                                where + ' LIMIT ?', (tag, TAG_CANDIDATE_LIMIT + 1)):
                    tag_rows += 1
                    if tag_rows > TAG_CANDIDATE_LIMIT:
                        seed_truncated = True
                        break
                    candidates[row[0]] = row
                if interrupted:
                    break

        # No COUNT, global CASE sort or IN materialization of millions of rowids.
        # FTS emits rowids in stable order and the fixed window is independent of offset.
        scanned = 0
        if not interrupted and not starter:
            custom = (' AND ' + _company_filter('f.')) if company else '' if include_custom else " AND f.kind='standard'"
            for row in rows('SELECT ' + ','.join('f.' + col for col in columns.split(',')) +
                            ' FROM fields_fts JOIN fields f ON f.id=fields_fts.rowid'
                            ' WHERE fields_fts MATCH ?' + custom +
                            ' ORDER BY fields_fts.rowid LIMIT ?', (match, candidate_limit + 1)):
                scanned += 1
                if scanned <= candidate_limit:
                    candidates[row[0]] = row
        # Complete is the deduplicated union of the nominated recipe probes and
        # literal AND FTS matches, never a claim after interrupted nomination.
        complete = not interrupted and not source_interrupted and not seed_truncated and scanned <= candidate_limit
        ordered = sorted(candidates.values(), key=lambda r: (
            0 if r[1].lower() in exact_names else 1,
            tag_relevance.get(r[6], 10**9),
            relevance.get(r[0], (9,9,10**9)), -r[5],
            0 if r[3] == 'standard' and r[4].endswith('/') else 1 if r[3] == 'standard' else 2, r[1]))
        if company:
            groups = {}
            for row in ordered:
                groups.setdefault((row[6], row[4]), []).append(row)
            ordered = list(groups.values())
        result.update(total=len(ordered) if complete else None,
                      truncated=not complete, ranking_complete=complete,
                      matched_sample=len(ordered),
                      has_more=(offset + limit < len(ordered))
                      if not interrupted and not source_interrupted else None)
        if not complete:
            result['message'] = ('Partial company concepts and observed variants; total and global ranking are unknown. '
                                 'Narrow the query or use the advanced recipes view.' if company else
                                'Partial search results; total and global ranking are unknown. '
                                 'Narrow the query, use an exact official name, or clear the query to browse all fields.')
        if starter:
            result['message'] = ('Reviewed starter concepts observed in this dictionary; total counts starters, '
                                 'not all company concepts. Search to discover more.' +
                                 (' Starter discovery or observed variants are partial; total is unknown.'
                                  if not complete else ''))
        connection.set_progress_handler(None, 0)
        page = ordered[offset:offset + limit]
        result['items'] = ([dict(tag=group[0][6], taxonomy=group[0][4], label=group[0][2],
                               variants=list(entries(group))) for group in page]
                           if company else list(entries(page)))
        return result
    finally:
        connection.set_progress_handler(None, 0)
