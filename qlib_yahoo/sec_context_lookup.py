"""Request-scoped, bounded verification of descriptive advanced SEC names.

No persistent alias store, fact scans, or writes. Names are recomputed from
exact observed recipes and defining-filing metadata. Incomplete candidate
collections cannot certify uniqueness and must fail closed.
"""
from contextlib import closing
import json
import re
import sqlite3
import time

from . import sec_catalog as source
from . import sec_context_names as names

CANDIDATE_LIMIT = 32768
LOOKUP_SECONDS = 2.0


class ContextNames:
    def __init__(self, connection, catalog):
        self.connection, self.catalog = connection, catalog
        self.groups, self.filings = {}, {}

    def filing(self, taxonomy):
        if not names._ACCESSION.fullmatch(taxonomy):
            return None
        if taxonomy in self.filings:
            return self.filings[taxonomy]
        result = None
        with closing(source._db(self.catalog)) as db:
            deadline = time.monotonic() + .2
            db.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='filings_adsh' AND type='index'").fetchone():
                rows = db.execute('SELECT payload FROM filings INDEXED BY filings_adsh WHERE adsh=? LIMIT 65',
                                  (taxonomy,)).fetchall()
                if len(rows) > 64:
                    raise ValueError('Descriptive SEC filing-name lookup exceeded its candidate budget')
                if len(rows) <= 64:
                    definitions = {}
                    for row in rows:
                        payload = json.loads(row[0])
                        identity = tuple(payload.get(key) for key in ('adsh', 'name', 'form', 'filed'))
                        definitions[identity] = payload
                    if len(definitions) == 1:
                        result = next(iter(definitions.values()))
        if time.monotonic() >= deadline:
            raise ValueError('Descriptive SEC filing-name lookup exceeded its budget')
        self.filings[taxonomy] = result
        return result

    def preferred(self, entry):
        spec = entry['spec']
        try:
            filing = self.filing(spec['taxonomy'])
        except ValueError:
            return None
        except sqlite3.OperationalError as exc:
            if getattr(exc, 'sqlite_errorcode', None) == sqlite3.SQLITE_INTERRUPT:
                return None
            raise
        return names.preferred_name(spec, kind=entry['kind'], filing=filing)

    def lookup(self, name):
        hint = names.candidate_for_name(name)
        if hint is None:
            return None
        key = tuple(hint.get(k) for k in ('tag', 'taxonomy', 'taxonomy_hint', 'unit', 'unit_hint', 'qtrs')) + (hint['context_labels'],)
        if key not in self.groups:
            self.groups[key] = self._collect(hint)
        candidates = self.groups[key]
        if candidates is None:
            raise ValueError('Descriptive SEC name could not be verified within the lookup budget')
        found = candidates.get(name, [])
        if len(found) > 1:
            raise ValueError('Ambiguous descriptive SEC name; inspect the exact source definition')
        return {**found[0], 'name': name, 'storage_name': found[0]['name']} if found else None

    def _collect(self, hint):
        from .sec_registry import _entry, _name_for_recipe
        deadline = time.monotonic() + LOOKUP_SECONDS
        progress = lambda: int(time.monotonic() >= deadline)
        fragments = tuple(label.split('_', 1)[1].replace('_', '') for label in hint['context_labels'])
        def context_key(segments, coreg):
            if not isinstance(segments, str) or not isinstance(coreg, str):
                return None
            # Cheap necessary condition before full word splitting. Normalizing
            # punctuation/case cannot remove any label's letters/digits. This
            # admits false positives, which the exact normalizer rejects below.
            compact = re.sub('[^a-z0-9]', '', (segments + coreg).lower())
            if any(fragment not in compact for fragment in fragments):
                return None
            labels = names._context_labels({'segments': segments, 'coreg': coreg})
            return json.dumps(labels) if labels is not None else None
        self.connection.create_function('sec_alias_context', 2, context_key)
        try:
            with closing(source._db(self.catalog)) as db:
                if not db.execute("SELECT 1 FROM sqlite_master WHERE name='tags_key' AND type='index'").fetchone():
                    return None
                db.set_progress_handler(progress, 1000)
                tags = [row[0] for row in db.execute(
                    'SELECT DISTINCT tag COLLATE BINARY FROM tags INDEXED BY tags_key '
                    'WHERE tag=? COLLATE NOCASE LIMIT 65', (hint['tag'],))]
            if len(tags) > 64:
                return None
            if not self.connection.execute("SELECT 1 FROM sqlite_master WHERE name='fields_tag' AND type='index'").fetchone():
                return None
            self.connection.set_progress_handler(progress, 1000)
            rows, total = [], 0
            for tag in tags:
                where = 'tag=? AND qtrs=?'
                args = [tag, str(hint['qtrs'])]
                if 'unit' in hint:
                    where += ' AND unit=?'
                    args.append(hint['unit'])
                if 'taxonomy' in hint:
                    where += " AND taxonomy=? AND kind='standard'"
                    args.append(hint['taxonomy'])
                elif 'filing_hint' in hint:
                    # Filing aliases must never reinterpret standard-family rows.
                    where += " AND taxonomy GLOB '??????????-??-??????'"
                else:
                    where += " AND kind IN ('standard','custom')"
                if not hint['context_labels']:
                    where += " AND json_extract(spec,'$.segments')='' AND json_extract(spec,'$.coreg')=''"
                else:
                    # Filter normalized context BEFORE the row cap; an ordinary
                    # tag can have tens of thousands of unrelated dimensions.
                    # Equal normalized contexts all survive for collision checks.
                    where += " AND sec_alias_context(json_extract(spec,'$.segments'),json_extract(spec,'$.coreg'))=?"
                    args.append(json.dumps(hint['context_labels']))
                index = 'fields_tag'
                # The immutable name's readable prefix supplies an existing
                # B-tree range for common scopes. This is only a superset
                # nomination; exact selectors and alias equality still decide.
                # Simple alphabetic unknown units normalize to the same physical
                # prefix (including case/punctuation variants). More complex
                # units use the general bounded tag path, never a guessed one.
                unit = hint.get('unit')
                if unit is None and re.fullmatch('[a-z]+', hint.get('unit_hint', '')):
                    unit = hint['unit_hint']
                if unit is not None and 'taxonomy' in hint:
                    probe = dict(tag=tag, taxonomy=hint['taxonomy'], unit=unit,
                                 qtrs=hint['qtrs'], segments='__prefix_probe__', coreg='')
                    prefix = _name_for_recipe(probe).rsplit('_', 1)[0] + '_'
                    for index_row in self.connection.execute("PRAGMA index_list('fields')"):
                        if not re.fullmatch(r'sqlite_autoindex_fields_[0-9]+', index_row[1]):
                            continue
                        cols = [row[2] for row in self.connection.execute('PRAGMA index_info(' + index_row[1] + ')')]
                        if cols == ['name']:
                            index = index_row[1]
                            where += ' AND name>=? AND name<?'
                            args += [prefix, prefix + '~']
                            break
                selected = self.connection.execute(
                    'SELECT name,spec,label,kind FROM fields INDEXED BY ' + index + ' WHERE ' +
                    where + ' LIMIT ?', (*args, CANDIDATE_LIMIT + 1 - total)).fetchall()
                rows.extend(selected)
                total += len(selected)
                if total > CANDIDATE_LIMIT:
                    return None
            result = {}
            for row in rows:
                if time.monotonic() >= deadline:
                    return None
                entry = _entry(row)
                if _name_for_recipe(entry['spec']) != entry['name']:
                    raise ValueError('Descriptive SEC name disagrees with stored identity')
                filing = self.filing(entry['spec']['taxonomy'])
                aliases = {names.preferred_name(entry['spec'], kind=entry['kind'], filing=filing),
                           names.generic_alias_for_spec(entry['spec'], kind=entry['kind'], filing=filing)}
                for alias in aliases - {None}:
                    result.setdefault(alias, []).append(entry)
            return result if time.monotonic() < deadline else None
        except sqlite3.OperationalError as exc:
            if getattr(exc, 'sqlite_errorcode', None) == sqlite3.SQLITE_INTERRUPT:
                return None
            raise
        finally:
            self.connection.set_progress_handler(None, 0)
            self.connection.create_function('sec_alias_context', 2, None)
