"""Bounded concept nomination from immutable TAG metadata, never NUM facts.

``discover_concepts(catalog, query)`` returns a list accepted by sec_search's
``concept_seeds``. ``aliases`` optionally maps reviewed query names to exact
recipe dictionaries (or tag strings). Alias resolution uses the whole query;
arbitrary words are never expanded independently into an OR search.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from contextlib import closing

from . import sec_catalog as source

CONCEPT_LIMIT = 128
SEARCH_BUDGET_MS = 200
# Whole-query discovery synonyms only; these do not define formulas or recipes.
SEARCH_TERMS = {
    'cashflow': ('NetCashProvidedByUsedInOperatingActivities',
                 'NetCashProvidedByUsedInInvestingActivities',
                 'NetCashProvidedByUsedInFinancingActivities'),
    'operatingcashflow': ('NetCashProvidedByUsedInOperatingActivities',),
    'cashflowoperating': ('NetCashProvidedByUsedInOperatingActivities',),
    'currentassets': ('AssetsCurrent',),
    'netincome': ('NetIncomeLoss',),
    'epsbasic': ('EarningsPerShareBasic',),
}


class ConceptSeeds(list):
    """A bounded list; truncation is retained when forwarded to recipe search."""
    truncated = False
    interrupted = False


def compact_words(query):
    """Join literal words; don't erase punctuation or drop arbitrary terms."""
    if not isinstance(query, str):
        return ''
    query = query.strip()
    if not re.fullmatch(r'[A-Za-z0-9_\s]+', query):
        return ''
    return re.sub(r'[_\s]+', '', query)


def discover_concepts(catalog, query, *, aliases=None, budget_ms=SEARCH_BUDGET_MS,
                      max_tags=CONCEPT_LIMIT):
    """Find exact/compact-prefix TAG concepts using the existing tags_key index.

    Empty queries open no database. Missing indexes return a truncated empty
    list rather than scanning legacy catalogs. Only confirmed textual/integer
    zero metadata nominates family defaults; sealed recipe kind is still checked
    by search. No source metadata is reclassified or rewritten.
    """
    if not isinstance(query, str) or len(query) > 4096:
        raise ValueError('Invalid concept query')
    if (type(max_tags) is not int or not 1 <= max_tags <= CONCEPT_LIMIT
            or type(budget_ms) not in (int, float) or not 0 < budget_ms <= 10000):
        raise ValueError('Invalid concept budget/limit')
    result = ConceptSeeds()
    compact = compact_words(query)
    if not compact:
        return result
    deadline = time.monotonic() + budget_ms / 1000
    explicit = False
    reviewed = {}
    # The optional module is standalone. No registry/resolver imports here.
    if aliases is None:
        try:
            from .sec_names import spec_for_name, curated_defs, CURATED_NAMES
        except ImportError:
            preferred = None
        else:
            preferred = spec_for_name(query.strip())
            explicit = preferred is not None
            if preferred is None:
                candidate = spec_for_name('sec_' + query.strip())
                selectors = '__' in query or bool(re.search(r'_(?:q|[0-9]+q)$', query))
                base = CURATED_NAMES.get(candidate['tag']) if candidate else None
                reviewed_base = base and compact_words(base).lower() == compact.lower()
                # Generic syntax can reconstruct Revenue but cannot choose its
                # taxonomy. Ordinary human words must use source family metadata.
                if candidate and (selectors or reviewed_base):
                    preferred = candidate
                    explicit = bool(selectors)
            if preferred is None:
                preferred = next((spec for name, spec in curated_defs().items()
                                  if compact_words(name[4:]).lower() == compact.lower()), None)
            reviewed = CURATED_NAMES
    else:
        preferred = next((value for name, value in aliases.items()
                          if compact_words(name).lower() == compact.lower()), None)
        explicit = isinstance(preferred, dict)
    target = preferred.get('tag') if isinstance(preferred, dict) else preferred
    if not target and query.strip().lower().startswith('sec_'):
        compact = compact_words(query.strip()[4:])
    # Exact source tag first. Reviewed terms match the entire compact query;
    # reviewed-base prefixes also consume every word, never dropping nonsense.
    nominations = []
    if isinstance(target, str) and target:
        nominations.append((target, 0, preferred if isinstance(preferred, dict) else None, explicit))
    else:
        nominations.append((compact, 0, None, False))
        for rank, tag in enumerate(SEARCH_TERMS.get(compact.lower(), ())):
            nominations.append((tag, rank, None, False))
        for tag, base in reviewed.items():
            words = compact_words(base).lower()
            if words.startswith(compact.lower()):
                nominations.append((tag, int(words != compact.lower()), None, False))
    path = source._safe_path(catalog)
    seen_tags = set()
    with closing(sqlite3.connect(source._safe_path(path / 'catalog.sqlite').as_uri()
                                + '?mode=ro&immutable=1', uri=True)) as connection:
        if not connection.execute("SELECT 1 FROM sqlite_master WHERE type='index' AND name='tags_key'").fetchone():
            result.truncated = True
            return result
        connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        try:
            # LIKE uses the existing NOCASE leading-key range, with no SQL
            # normalization function on tag. DISTINCT collapses repeated editions.
            jobs = [(rank, 'SELECT DISTINCT tag FROM tags INDEXED BY tags_key WHERE tag=? COLLATE NOCASE LIMIT ?',
                     (term,max_tags + 1), spec, pinned)
                    for term,rank,spec,pinned in nominations]
            if not target:
                jobs.append((1, "SELECT DISTINCT tag FROM tags INDEXED BY tags_key WHERE tag LIKE ? ESCAPE '\\' LIMIT ?",
                             (compact.replace('_', '\\_') + '%',max_tags + 1),None,False))
            for rank, sql, params, preferred_spec, pinned in jobs:
                for (tag,) in connection.execute(sql, params):
                    if time.monotonic() >= deadline:
                        result.truncated = True
                        result.interrupted = True
                        break
                    if tag in seen_tags:
                        continue
                    if len(seen_tags) >= max_tags:
                        result.truncated = True
                        break
                    seen_tags.add(tag)
                    if isinstance(preferred_spec, dict):
                        # Names may reconstruct PascalCase while SEC uses an
                        # acronym (EPS). Carry each actual exact TAG identity;
                        # resolver owns ambiguity, never guess storage spelling.
                        result.append(dict(tag=tag, spec=dict(preferred_spec, tag=tag), rank=rank,
                                           preferred=True, expand_defaults=not pinned))
                        # A reviewed alias pins unit/duration/context explicitly.
                        # Do not broaden it to other default recipes.
                        continue
                    # Alphabetic taxonomy range skips numeric accession keys;
                    # WHERE tag + version range is served by tags_key. Limit
                    # metadata per tag as well as the number of distinct tags.
                    try:
                        rows = connection.execute(
                            'SELECT version,payload FROM tags INDEXED BY tags_key '
                            'WHERE tag=? COLLATE NOCASE AND tag=? COLLATE BINARY '
                            'AND version>=? ORDER BY version DESC LIMIT 129', (tag, tag, 'A')).fetchall()
                    except sqlite3.OperationalError as exc:
                        if getattr(exc, 'sqlite_errorcode', None) == sqlite3.SQLITE_INTERRUPT:
                            result.append(dict(tag=tag,rank=rank))
                        raise
                    if len(rows) > 128:
                        result.truncated = True
                    families = set()
                    for version, payload in rows[:128]:
                        flag = json.loads(payload).get('custom')
                        family = source._standard_family(version)
                        if type(flag) not in (str, int) or str(flag) != '0' or family == version:
                            continue
                        families.add(family)
                    for family in sorted(families):
                        result.append(dict(tag=tag, rank=rank, spec=dict(
                            tag=tag, taxonomy=family, unit='USD', qtrs=0, segments='', coreg='')))
                    if not families:
                        result.append(dict(tag=tag, rank=rank))
                if result.truncated:
                    break
            if time.monotonic() >= deadline:
                result.truncated = True
                result.interrupted = True
        except sqlite3.OperationalError as exc:
            if getattr(exc, 'sqlite_errorcode', None) != sqlite3.SQLITE_INTERRUPT:
                raise
            result.truncated = True
            result.interrupted = True
        finally:
            connection.set_progress_handler(None, 0)
    return result
