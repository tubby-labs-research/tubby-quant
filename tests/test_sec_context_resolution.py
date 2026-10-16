"""Exact descriptive-alias resolution on indexed, read-only SQLite fixtures.

No catalog build, execution index, Qlib runtime, source writes or real datasets.
The dictionary and TAG/SUB metadata are independent in-memory databases.
"""
from contextlib import ExitStack
import json
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import patch

from qlib_yahoo import sec_catalog as source, sec_concepts, sec_registry as registry
from qlib_yahoo import sec_context_lookup as lookup, sec_context_names as names
from qlib_yahoo import sec_search


def recipe(tag='Assets', **changes):
    return dict(dict(tag=tag, taxonomy='us-gaap/', unit='USD', qtrs=0,
                     segments='', coreg=''), **changes)


class ContextResolutionTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        self.metadata = sqlite3.connect(':memory:')
        self.addCleanup(self.db.close)
        self.addCleanup(self.metadata.close)
        self.db.executescript('''
            CREATE TABLE fields(id INTEGER PRIMARY KEY,name TEXT UNIQUE,spec TEXT UNIQUE,
                label TEXT,kind TEXT,tag TEXT,description TEXT,taxonomy TEXT,common INTEGER,
                unit TEXT,qtrs TEXT);
            CREATE INDEX fields_tag ON fields(tag);
            CREATE INDEX fields_kind_name ON fields(kind,common DESC,name);
            CREATE INDEX fields_common_name ON fields(common DESC,name);
            CREATE VIRTUAL TABLE fields_fts USING fts5(name,tag,label,description,
                content='fields',content_rowid='id');
        ''')
        self.metadata.executescript('''
            CREATE TABLE tags(tag TEXT,version TEXT,payload TEXT,occurrences INTEGER);
            CREATE INDEX tags_key ON tags(tag COLLATE NOCASE,version);
            CREATE TABLE filings(adsh TEXT,cik TEXT,payload TEXT);
            CREATE INDEX filings_adsh ON filings(adsh);
        ''')
        self.catalog = Path('unused-synthetic-catalog')
        self.statements = []
        self.metadata_trace = None

    def add(self, spec, kind='standard'):
        name = registry._name_for_recipe(spec)
        self.db.execute('''INSERT INTO fields(name,spec,label,kind,tag,description,
            taxonomy,common,unit,qtrs) VALUES(?,?,?,?,?,?,?,?,?,?)''',
            (name, source._canonical(spec).decode(), spec['tag'], kind, spec['tag'],
             'Fixture', spec['taxonomy'], int(name in registry.COMMON_FIELDS),
             spec['unit'], str(spec['qtrs'])))
        payload = dict(tag=spec['tag'], version=spec['taxonomy'],
                       custom='0' if kind == 'standard' else '1')
        self.metadata.execute('INSERT INTO tags VALUES(?,?,?,?)',
                              (spec['tag'], spec['taxonomy'], json.dumps(payload), 1))
        return name

    def filing(self, accession, issuer='Acme Corporation', form='10-Q', filed='20240501'):
        payload = dict(adsh=accession, name=issuer, form=form, filed=filed)
        self.metadata.execute('INSERT INTO filings VALUES(?,?,?)',
                              (accession, '1', json.dumps(payload)))
        return payload

    def seal(self):
        self.db.execute("INSERT INTO fields_fts(fields_fts) VALUES('rebuild')")
        self.db.commit()
        self.metadata.commit()
        self.db.execute('PRAGMA query_only=ON')
        self.db.set_trace_callback(lambda sql: self.statements.append(('dictionary', sql)))

    def clone(self, database, label):
        result = sqlite3.connect(':memory:')
        database.backup(result)
        result.execute('PRAGMA query_only=ON')
        def trace(sql):
            self.statements.append((label, sql))
            if label == 'metadata' and self.metadata_trace:
                self.metadata_trace(sql)
        result.set_trace_callback(trace)
        return result

    def patches(self):
        stack = ExitStack()
        stack.enter_context(patch.object(source, '_db', side_effect=lambda _: self.clone(self.metadata, 'metadata')))
        stack.enter_context(patch.object(registry, '_db', side_effect=lambda _: self.clone(self.db, 'dictionary')))
        stack.enter_context(patch.object(registry, '_ready', return_value=Path('unused-dictionary')))
        stack.enter_context(patch.object(registry, 'dictionary_status', return_value={'field_count': 1000000}))
        stack.enter_context(patch.object(source, '_scan', side_effect=AssertionError('NUM scan')))
        stack.enter_context(patch.object(source, '_hash', side_effect=AssertionError('bulk hash')))
        return stack

    def test_advanced_lookup_preserves_all_six_selectors_and_storage_identity(self):
        selected = recipe(tag='InventoryNet', taxonomy='us-gaap/2024', unit='EUR', qtrs=6,
                          segments='LegalEntity=AcmeCompany;Scenario=Successor', coreg='ParentCompany')
        storage = self.add(selected)
        self.add(dict(selected, unit='USD'))
        self.add(dict(selected, qtrs=4))
        self.add(dict(selected, taxonomy='us-gaap/'))
        self.add(dict(selected, coreg='OtherCompany'))
        self.seal()
        alias = names.preferred_name(selected)
        self.assertIsNotNone(alias)
        with self.patches():
            result = registry.lookup_fields(self.catalog, [alias])[alias]
        self.assertEqual(result['spec'], selected)
        self.assertEqual(result['name'], alias)
        self.assertEqual(result['storage_name'], storage)
        self.assertEqual(result['kind'], 'standard')

    def test_generic_original_tag_context_alternate_resolves_exact_recipe(self):
        selected = recipe(tag='InventoryNet', segments='Scenario=Successor')
        storage = self.add(selected)
        self.seal()
        preferred = names.preferred_name(selected)
        alternate = names.generic_alias_for_spec(selected)
        self.assertNotEqual(preferred, alternate)
        with self.patches():
            result = registry.lookup_fields(self.catalog, [preferred, alternate])
        for alias in (preferred, alternate):
            self.assertEqual(result[alias]['spec'], selected)
            self.assertEqual(result[alias]['storage_name'], storage)

    def test_plain_common_and_advanced_historical_names_still_win_direct_lookup(self):
        plain = recipe()
        first = recipe(segments='LegalEntity=AcmeCompany')
        second = dict(first, coreg='AcmeCompany')
        stored = {self.add(spec): spec for spec in (plain, first, second)}
        self.seal()
        self.assertEqual(names.preferred_name(first), names.preferred_name(second))
        with self.patches(), patch.object(lookup.ContextNames, 'lookup', side_effect=AssertionError('alias lookup')):
            result = registry.lookup_fields(self.catalog, list(stored))
        self.assertEqual(set(result), set(stored))
        for name, spec in stored.items():
            self.assertEqual(result[name]['spec'], spec)
            self.assertEqual(result[name]['name'], name)
            self.assertNotIn('storage_name', result[name])
        self.assertFalse(any(label == 'metadata' for label, _ in self.statements))

    def test_punctuation_collision_is_ambiguous_before_display_and_resolution(self):
        first = recipe(segments='BusinessSegment=North-America')
        second = dict(first, segments='BusinessSegment=North America')
        self.add(first)
        self.add(second)
        self.seal()
        alias = names.preferred_name(first)
        self.assertEqual(alias, names.preferred_name(second))
        with self.patches():
            with self.assertRaisesRegex(ValueError, 'Ambiguous'):
                registry.lookup_fields(self.catalog, [alias])
            row = self.db.execute('SELECT name,spec,label,kind FROM fields LIMIT 1').fetchone()
            shown = registry.readable_entry(registry._entry(row), self.catalog, self.db)
        self.assertEqual(shown['name'], row[0])
        self.assertIn('Ambiguous', shown['alias_error'])

    def test_redundant_equal_entity_coreg_cannot_hide_distinct_raw_specs(self):
        first = recipe(segments='LegalEntity=AcmeCompany')
        second = dict(first, coreg='AcmeCompany')
        self.add(first)
        self.add(second)
        self.seal()
        alias = names.preferred_name(first)
        self.assertEqual(alias, names.preferred_name(second))
        with self.patches(), self.assertRaisesRegex(ValueError, 'Ambiguous'):
            registry.lookup_fields(self.catalog, [alias])

    def test_coreg_normalization_is_not_raw_entity_equality(self):
        selected = recipe(segments='LegalEntity=AcmeCompany', coreg='Acme Company')
        storage = self.add(selected)
        self.seal()
        alias = names.preferred_name(selected)
        self.assertIn('__coreg_acme_company', alias)
        self.assertIn('__entity_acme_company', alias)
        with self.patches():
            result = registry.lookup_fields(self.catalog, [alias])[alias]
            incomplete = registry.lookup_fields(self.catalog, ['sec_assets__entity_acme_company'])
        self.assertEqual(result['spec'], selected)
        self.assertEqual(result['storage_name'], storage)
        self.assertEqual(incomplete, {})

    def test_context_order_collision_fails_closed(self):
        first = recipe(segments='Scenario=Successor;LegalEntity=AcmeCompany')
        second = dict(first, segments='LegalEntity=AcmeCompany;Scenario=Successor')
        self.add(first)
        self.add(second)
        self.seal()
        alias = names.preferred_name(first)
        self.assertEqual(alias, names.preferred_name(second))
        with self.patches(), self.assertRaisesRegex(ValueError, 'Ambiguous'):
            registry.lookup_fields(self.catalog, [alias])

    def test_acronym_case_collision_is_included_in_ambiguity_proof(self):
        first = recipe(tag='EPSValue', segments='Scenario=Successor')
        second = dict(first, tag='EpsValue')
        self.add(first)
        self.add(second)
        self.seal()
        alias = names.preferred_name(first)
        self.assertEqual(alias, names.preferred_name(second))
        with self.patches(), self.assertRaisesRegex(ValueError, 'Ambiguous'):
            registry.lookup_fields(self.catalog, [alias])

    def test_custom_issuer_and_filing_date_select_exact_accession_not_family(self):
        first = recipe(tag='Inventory', taxonomy='0000000001-24-000001', qtrs=1,
                       segments='Scenario=Successor')
        second = dict(first, taxonomy='0000000001-24-000002')
        third = dict(first, taxonomy='0000000002-24-000001')
        storage = {spec['taxonomy']: self.add(spec, 'custom') for spec in (first, second, third)}
        filing = self.filing(first['taxonomy'], filed='20240501')
        self.filing(second['taxonomy'], filed='20240801')
        self.filing(third['taxonomy'], issuer='Other Corporation', filed='20240501')
        self.add(recipe(tag='Inventory', qtrs=1, segments='Scenario=Successor'))
        self.seal()
        alias = names.preferred_name(first, kind='custom', filing=filing)
        self.assertIn('__issuer_acme_corporation__form_10_q__filed_2024_05_01', alias)
        with self.patches():
            result = registry.lookup_fields(self.catalog, [alias])[alias]
            old = registry.lookup_fields(self.catalog, storage.values())
        self.assertEqual(result['spec'], first)
        self.assertEqual(result['storage_name'], storage[first['taxonomy']])
        self.assertEqual(result['kind'], 'custom')
        self.assertEqual(len(old), 3)
        self.assertEqual({entry['spec']['taxonomy'] for entry in old.values()}, set(storage))

    def test_same_issuer_form_date_across_distinct_accessions_is_ambiguous(self):
        first = recipe(tag='CompanyMetric', taxonomy='0000000001-24-000001')
        second = dict(first, taxonomy='0000000001-24-000002')
        self.add(first, 'custom')
        self.add(second, 'custom')
        filing = self.filing(first['taxonomy'])
        self.filing(second['taxonomy'])
        self.seal()
        alias = names.preferred_name(first, kind='custom', filing=filing)
        with self.patches(), self.assertRaisesRegex(ValueError, 'Ambiguous'):
            registry.lookup_fields(self.catalog, [alias])

    def test_missing_conflicting_or_wrong_accession_metadata_never_invents_alias(self):
        selected = recipe(tag='CompanyMetric', taxonomy='0000000001-24-000001')
        storage = self.add(selected, 'custom')
        filing = self.filing(selected['taxonomy'])
        self.filing(selected['taxonomy'], issuer='Conflicting Corporation')
        missing = recipe(tag='MissingMetric', taxonomy='0000000001-24-000002')
        missing_storage = self.add(missing, 'custom')
        wrong = recipe(tag='WrongMetric', taxonomy='0000000001-24-000003')
        wrong_storage = self.add(wrong, 'custom')
        wrong_filing = dict(filing, adsh=wrong['taxonomy'])
        # The indexed column nominates the filing, but payload adsh must still
        # match the exact accession; no issuer/title inference is permitted.
        self.metadata.execute('INSERT INTO filings VALUES(?,?,?)',
                              (wrong['taxonomy'], '1', json.dumps(filing)))
        self.seal()
        aliases = [names.preferred_name(selected, kind='custom', filing=filing),
                   names.preferred_name(wrong, kind='custom', filing=wrong_filing)]
        with self.patches():
            self.assertEqual(registry.lookup_fields(self.catalog, aliases), {})
            stored = registry.lookup_fields(self.catalog, [storage, missing_storage, wrong_storage])
            resolver = lookup.ContextNames(self.db, self.catalog)
            self.assertIsNone(resolver.preferred(stored[missing_storage]))
            self.assertIsNone(resolver.preferred(stored[wrong_storage]))
        self.assertEqual(len(stored), 3)

    def test_nonvocabulary_unit_hint_selects_observed_raw_unit_and_detects_collision(self):
        selected = recipe(tag='CompanyMetric', unit='USD/square-foot', segments='Scenario=Successor')
        self.add(selected)
        self.add(dict(selected, unit='EUR'))
        self.seal()
        alias = names.preferred_name(selected)
        self.assertIsNotNone(alias)
        self.assertIn('unit_hint', names.candidate_for_name(alias))
        with self.patches():
            result = registry.lookup_fields(self.catalog, [alias])[alias]
        self.assertEqual(result['spec'], selected)

    def test_normalized_nonvocabulary_unit_collision_cannot_select_first(self):
        first = recipe(tag='CompanyMetric', unit='USD/square-foot', segments='Scenario=Successor')
        second = dict(first, unit='USD/square foot')
        self.add(first)
        self.add(second)
        self.seal()
        alias = names.preferred_name(first)
        self.assertEqual(alias, names.preferred_name(second))
        with self.patches(), self.assertRaisesRegex(ValueError, 'Ambiguous'):
            registry.lookup_fields(self.catalog, [alias])

    def test_simple_unknown_unit_prefix_includes_case_and_punctuation_collisions(self):
        selected = [recipe(unit=unit, segments='Scenario=Successor')
                    for unit in ('RUB', 'rub', 'Rub', ' RUB ', '.RUB.', 'RUB!')]
        for spec in selected:
            self.add(spec)
        self.seal()
        aliases = {names.preferred_name(spec) for spec in selected}
        self.assertEqual(len(aliases), 1)
        alias = aliases.pop()
        self.assertEqual(names.candidate_for_name(alias)['unit_hint'], 'rub')
        with self.patches(), self.assertRaisesRegex(ValueError, 'Ambiguous'):
            registry.lookup_fields(self.catalog, [alias])
        collected = [sql for label, sql in self.statements
                     if label == 'dictionary' and 'SELECT name,spec,label,kind' in sql and 'INDEXED BY' in sql]
        self.assertEqual(len(collected), 1)
        self.assertIn('sqlite_autoindex_fields_', collected[0])
        self.assertIn('name>=', collected[0])
        self.assertNotIn('AND unit=', collected[0])

    def test_inferred_unit_range_keeps_overflow_truthful_for_possible_collisions(self):
        first = recipe(unit='ILS', segments='Scenario=Successor')
        second = dict(first, unit=' ILS ')
        self.add(first)
        self.add(second)
        self.seal()
        alias = names.preferred_name(first)
        self.assertEqual(alias, names.preferred_name(second))
        with self.patches(), patch.object(lookup, 'CANDIDATE_LIMIT', 1):
            with self.assertRaisesRegex(ValueError, 'budget'):
                registry.lookup_fields(self.catalog, [alias])

    def test_complex_units_use_general_tag_path_and_include_raw_unit_collisions(self):
        first = recipe(unit='RUB/per-share', segments='Scenario=Successor')
        second = dict(first, unit='RUB/per share')
        third = dict(first, unit='RUB/perShare')
        for spec in (first, second, third):
            self.add(spec)
        self.seal()
        alias = names.preferred_name(first)
        self.assertEqual({names.preferred_name(spec) for spec in (first, second, third)}, {alias})
        with self.patches(), self.assertRaisesRegex(ValueError, 'Ambiguous'):
            registry.lookup_fields(self.catalog, [alias])
        collected = [sql for label, sql in self.statements if label == 'dictionary' and 'INDEXED BY fields_tag' in sql]
        self.assertEqual(len(collected), 1)
        self.assertNotIn('name>=', collected[0])

    def test_long_physical_prefixes_are_supersets_not_an_alias_identity(self):
        tag = 'LongMeaningfulCompanyAccountingTermExceedingThirtyTwoCharacters'
        first = recipe(tag=tag, unit='LONGALPHABETICCURRENCY', segments='Scenario=Successor')
        second = dict(first, unit='LONGALPHABETICCURRENCY!')
        # Shares the truncated physical unit slug but has a different full
        # public label, and must not be mistaken for a genuine alias collision.
        unrelated = dict(first, unit='LONGALPHABETICCURdifferent')
        for spec in (first, second, unrelated):
            self.add(spec)
        self.seal()
        alias = names.preferred_name(first)
        self.assertIsNotNone(alias)
        self.assertEqual(alias, names.preferred_name(second))
        self.assertNotEqual(alias, names.preferred_name(unrelated))
        with self.patches(), self.assertRaisesRegex(ValueError, 'Ambiguous'):
            registry.lookup_fields(self.catalog, [alias])

    def test_prefix_collisions_from_truncated_unit_do_not_create_false_ambiguity(self):
        tag = 'LongMeaningfulCompanyAccountingTermExceedingThirtyTwoCharacters'
        selected = recipe(tag=tag, unit='LONGALPHABETICCURRENCY', segments='Scenario=Successor')
        unrelated = dict(selected, unit='LONGALPHABETICCURdifferent')
        self.add(selected)
        self.add(unrelated)
        self.seal()
        with self.patches():
            result = registry.lookup_fields(self.catalog, [names.preferred_name(selected)])
        self.assertEqual(result[names.preferred_name(selected)]['spec'], selected)

    def test_prefixes_use_each_canonical_tag_from_source_not_guessed_tag_case(self):
        first = recipe(tag='LongCompanyEPSMetric', unit='RUB', segments='Scenario=Successor')
        second = dict(first, tag='LongCompanyEpsMetric')
        for spec in (first, second):
            self.add(spec)
        self.seal()
        alias = names.preferred_name(first)
        self.assertEqual(alias, names.preferred_name(second))
        with self.patches(), self.assertRaisesRegex(ValueError, 'Ambiguous'):
            registry.lookup_fields(self.catalog, [alias])
        collected = [sql for label, sql in self.statements if label == 'dictionary' and 'name>=' in sql]
        self.assertEqual(len(collected), 2)

    def test_common_physical_names_still_resolve_all_public_ordinary_aliases(self):
        specs = list(registry.COMMON_FIELDS.values())
        for spec in specs:
            self.add(spec)
        self.seal()
        aliases = {names.preferred_name(spec): spec for spec in specs}
        aliases.update({names.generic_alias_for_spec(spec): spec for spec in specs})
        with self.patches():
            result = registry.lookup_fields(self.catalog, list(aliases))
        self.assertEqual(set(result), set(aliases))
        for alias, spec in aliases.items():
            self.assertEqual(result[alias]['spec'], spec)
            self.assertEqual(result[alias].get('storage_name', result[alias]['name']), registry._name_for_recipe(spec))

    def test_whitespace_around_exact_advanced_search_keeps_recipe_available(self):
        selected = recipe(segments='Scenario=Successor')
        self.add(selected)
        self.seal()
        alias = names.preferred_name(selected)
        with self.patches(), patch.object(sec_concepts, 'discover_concepts', side_effect=lambda *_: sec_concepts.ConceptSeeds()):
            direct = registry.search_dictionary(self.catalog, alias, view='recipes')
            spaced = registry.search_dictionary(self.catalog, '  ' + alias + '  ', view='recipes')
        self.assertEqual(direct['total'], 1)
        self.assertEqual(spaced['items'], direct['items'])

    def test_exact_advanced_search_returns_one_recipe_without_secondary_search(self):
        selected = recipe(unit='EUR', qtrs=6, segments='Scenario=Successor')
        storage = self.add(selected)
        for index in range(50):
            self.add(dict(selected, segments=f'BusinessSegment=OtherSegment{index}'))
        self.add(dict(selected, segments='Scenario=Predecessor'))
        self.seal()
        alias = names.preferred_name(selected)
        with self.patches(), patch.object(sec_concepts, 'discover_concepts',
                                         side_effect=lambda *_: sec_concepts.ConceptSeeds()), \
             patch.object(sec_search, 'search_dictionary', side_effect=AssertionError('secondary recipe search')) as secondary:
            exact = registry.search_dictionary(self.catalog, '  ' + alias + '  ', limit=50, view='recipes')
            exhausted = registry.search_dictionary(self.catalog, alias, offset=1, view='recipes')
            excluded = registry.search_dictionary(self.catalog, alias, include_custom=True, view='company')
        secondary.assert_not_called()
        self.assertEqual(len(exact['items']), 1)
        self.assertEqual(exact['items'][0]['name'], alias)
        self.assertEqual(exact['items'][0]['storage_name'], storage)
        self.assertEqual(exact['items'][0]['spec'], selected)
        self.assertEqual((exact['total'], exact['matched_sample']), (1, 1))
        self.assertFalse(exact['truncated'])
        self.assertFalse(exact['has_more'])
        self.assertTrue(exact['ranking_complete'])
        self.assertEqual(exhausted['items'], [])
        self.assertEqual(exhausted['total'], 1)
        self.assertEqual(excluded['items'], [])
        self.assertEqual(excluded['total'], 0)

    def test_exact_company_alias_returns_one_variant_without_secondary_search(self):
        selected = recipe(unit='RUB', qtrs=6)
        storage = self.add(selected)
        self.add(dict(selected, unit='EUR'))
        self.add(dict(selected, qtrs=4))
        self.add(dict(selected, taxonomy='us-gaap/2024'))
        self.seal()
        alias = names.preferred_name(selected)
        with self.patches(), patch.object(sec_concepts, 'discover_concepts',
                                         side_effect=lambda *_: sec_concepts.ConceptSeeds()), \
             patch.object(sec_search, 'search_dictionary', side_effect=AssertionError('secondary company search')) as secondary:
            exact = registry.search_dictionary(self.catalog, alias, view='company')
        secondary.assert_not_called()
        self.assertEqual(len(exact['items']), 1)
        group = exact['items'][0]
        self.assertEqual((group['tag'], group['taxonomy']), (selected['tag'], selected['taxonomy']))
        self.assertEqual(len(group['variants']), 1)
        self.assertEqual(group['variants'][0]['spec'], selected)
        self.assertEqual(group['variants'][0]['storage_name'], storage)
        self.assertEqual(exact['search_scope'], 'company_concepts')
        self.assertEqual((exact['total'], exact['matched_sample']), (1, 1))
        self.assertEqual(exact['field_count'], 1000000)
        self.assertFalse(exact['has_more'])
        self.assertFalse(exact['truncated'])

    def test_nonvocabulary_taxonomy_hint_preserves_observed_exact_scope(self):
        selected = recipe(tag='CompanyMetric', taxonomy='new-family/2024', segments='Scenario=Successor')
        self.add(selected)
        self.add(dict(selected, taxonomy='us-gaap/'))
        self.add(dict(selected, taxonomy='new-family/'))
        self.seal()
        alias = names.preferred_name(selected)
        self.assertIn('taxonomy_hint', names.candidate_for_name(alias))
        with self.patches():
            result = registry.lookup_fields(self.catalog, [alias])[alias]
        self.assertEqual(result['spec'], selected)

    def test_normalized_taxonomy_collision_is_ambiguous(self):
        first = recipe(tag='CompanyMetric', taxonomy='new-family/2024', segments='Scenario=Successor')
        second = dict(first, taxonomy='new_family/2024')
        self.add(first)
        self.add(second)
        self.seal()
        alias = names.preferred_name(first)
        self.assertEqual(alias, names.preferred_name(second))
        with self.patches(), self.assertRaisesRegex(ValueError, 'Ambiguous'):
            registry.lookup_fields(self.catalog, [alias])

    def test_taxonomy_hint_collision_across_standard_and_custom_kinds_fails_closed(self):
        standard = recipe(tag='CompanyMetric', taxonomy='new-family/2024', segments='Scenario=Successor')
        custom = dict(standard, taxonomy='new_family/2024')
        stored = [self.add(standard, 'standard'), self.add(custom, 'custom')]
        self.seal()
        alias = names.preferred_name(standard, kind='standard')
        self.assertEqual(alias, names.preferred_name(custom, kind='custom'))
        self.assertIn('taxonomy_hint', names.candidate_for_name(alias))
        with self.patches():
            with self.assertRaisesRegex(ValueError, 'Ambiguous'):
                registry.lookup_fields(self.catalog, [alias])
            original = registry.lookup_fields(self.catalog, stored)
            for spec in (standard, custom):
                declined = registry.name_for_spec(self.catalog, spec)
                self.assertEqual(declined['spec'], spec)
                self.assertIn('Ambiguous', declined['alias_error'])
        self.assertEqual({item['kind'] for item in original.values()}, {'standard', 'custom'})

    def test_group_collection_is_reused_for_multiple_contexts_in_one_page(self):
        selected = [recipe(segments=f'Scenario={scenario}') for scenario in ('Successor', 'Predecessor')]
        stored = [self.add(spec) for spec in selected]
        self.seal()
        aliases = [names.preferred_name(spec) for spec in selected]
        with self.patches(), patch.object(lookup.ContextNames, '_collect', autospec=True,
                                         side_effect=lookup.ContextNames._collect) as collect:
            result = registry.lookup_fields(self.catalog, aliases + aliases)
        # Normalized-context filtering makes each context its own selector
        # group. Repeated names must reuse the two completed collections.
        self.assertEqual(collect.call_count, 2)
        self.assertEqual({entry['storage_name'] for entry in result.values()}, set(stored))
        collections = [sql for label, sql in self.statements if label == 'dictionary'
                       and 'SELECT name,spec,label,kind' in sql and 'INDEXED BY' in sql]
        self.assertEqual(len(collections), 2)

    def test_recipe_page_marks_candidates_pending_without_collision_lookup(self):
        selected = [recipe(segments=f'Scenario={scenario}') for scenario in ('Successor', 'Predecessor')]
        for spec in selected:
            self.add(spec)
        self.seal()
        with self.patches(), patch.object(sec_concepts, 'discover_concepts', return_value=sec_concepts.ConceptSeeds()), \
             patch.object(lookup.ContextNames, 'lookup', side_effect=AssertionError('eager alias verification')) as verify, \
             patch.object(registry, 'readable_entry', side_effect=AssertionError('eager readable overlay')):
            result = registry.search_dictionary(self.catalog, 'Fixture', view='recipes')
        verify.assert_not_called()
        self.assertEqual(result['total'], 2)
        self.assertTrue(all(entry.get('alias_pending') is True for entry in result['items']))
        self.assertEqual({entry['name'] for entry in result['items']}, {names.preferred_name(spec) for spec in selected})
        self.assertEqual({source._canonical(entry['spec']) for entry in result['items']},
                         {source._canonical(spec) for spec in selected})
        self.assertEqual({entry['storage_name'] for entry in result['items']},
                         {registry._name_for_recipe(spec) for spec in selected})

    def test_pending_candidate_selection_verifies_exact_spec_before_acceptance(self):
        selected = recipe(unit='EUR', qtrs=6, segments='Scenario=Successor', coreg='ParentCompany')
        storage = self.add(selected)
        self.seal()
        with self.patches(), patch.object(sec_concepts, 'discover_concepts', return_value=sec_concepts.ConceptSeeds()):
            page = registry.search_dictionary(self.catalog, 'Fixture', view='recipes')
            candidate = page['items'][0]
            self.assertIs(candidate['alias_pending'], True)
            with patch.object(lookup.ContextNames, 'lookup', autospec=True,
                              side_effect=lookup.ContextNames.lookup) as verify:
                accepted = registry.name_for_spec(self.catalog, candidate['spec'])
            self.assertEqual(verify.call_count, 1)
            resolved = registry.lookup_fields(self.catalog, [accepted['name']])[accepted['name']]
        self.assertEqual(accepted['name'], candidate['name'])
        self.assertEqual(accepted['spec'], selected)
        self.assertEqual(accepted['storage_name'], storage)
        self.assertNotIn('alias_pending', accepted)
        self.assertNotIn('alias_error', accepted)
        self.assertEqual(resolved['spec'], selected)

    def test_pending_colliding_labels_remain_unaccepted_after_exact_spec_selection(self):
        selected = [recipe(segments='BusinessSegment=North-America'),
                    recipe(segments='BusinessSegment=North America')]
        for spec in selected:
            self.add(spec)
        self.seal()
        alias = names.preferred_name(selected[0])
        self.assertEqual(alias, names.preferred_name(selected[1]))
        with self.patches(), patch.object(sec_concepts, 'discover_concepts', return_value=sec_concepts.ConceptSeeds()):
            page = registry.search_dictionary(self.catalog, 'Fixture', view='recipes')
            self.assertEqual(len(page['items']), 2)
            self.assertTrue(all(entry['name'] == alias and entry.get('alias_pending') is True
                                for entry in page['items']))
            for candidate in page['items']:
                declined = registry.name_for_spec(self.catalog, candidate['spec'])
                self.assertEqual(declined['name'], candidate['storage_name'])
                self.assertEqual(declined['spec'], candidate['spec'])
                self.assertIn('Ambiguous', declined['alias_error'])
                self.assertNotIn('alias_pending', declined)
            with self.assertRaisesRegex(ValueError, 'Ambiguous'):
                registry.lookup_fields(self.catalog, [alias])

    def test_candidate_overflow_and_cached_failure_never_certify_uniqueness(self):
        first = recipe(segments='Scenario=Successor')
        self.add(first)
        self.add(dict(first, segments='Scenario=Successor;'))
        self.seal()
        alias = names.preferred_name(first)
        with self.patches(), patch.object(lookup, 'CANDIDATE_LIMIT', 1):
            resolver = lookup.ContextNames(self.db, self.catalog)
            for _ in range(2):
                with self.assertRaisesRegex(ValueError, 'budget'):
                    resolver.lookup(alias)
        collections = [sql for label, sql in self.statements if label == 'dictionary'
                       and 'SELECT name,spec,label,kind' in sql and 'INDEXED BY' in sql]
        self.assertEqual(len(collections), 1)

    def test_normalized_context_predicate_precedes_cap_after_33000_unrelated_rows(self):
        for index in range(33000):
            self.add(recipe(segments=f'BusinessSegment=UnrelatedSegment{index}'))
        first = recipe(segments='BusinessSegment=North-America')
        second = dict(first, segments='business-segment=NORTH AMERICA')
        self.add(first)
        self.add(second)
        # The source TAG table represents definitions/editions, not contexts.
        self.metadata.execute('DELETE FROM tags')
        payload = dict(tag='Assets', version='us-gaap/2024', custom='0')
        self.metadata.execute('INSERT INTO tags VALUES(?,?,?,?)', ('Assets', 'us-gaap/2024', json.dumps(payload), 1))
        self.seal()
        alias = names.preferred_name(first)
        self.assertEqual(alias, names.preferred_name(second))
        # Check row-cap semantics independently of machine timing variation.
        # Both normalized matches must survive; unrelated contexts must not
        # consume the cap, even with a much larger actual indexed inventory.
        with self.patches(), patch.object(lookup, 'CANDIDATE_LIMIT', 2), \
             patch.object(lookup, 'LOOKUP_SECONDS', 10):
            with self.assertRaisesRegex(ValueError, 'Ambiguous'):
                registry.lookup_fields(self.catalog, [alias])
        queries = [sql for label, sql in self.statements if label == 'dictionary' and 'INDEXED BY' in sql]
        self.assertTrue(any('sec_alias_context' in sql and 'LIMIT 3' in sql for sql in queries))

    def test_filing_metadata_is_reused_across_distinct_page_selector_groups(self):
        accession = '0000000001-24-000001'
        selected = [recipe(tag=tag, taxonomy=accession, segments='Scenario=Successor')
                    for tag in ('CompanyMetric', 'OtherCompanyMetric')]
        for spec in selected:
            self.add(spec, 'custom')
        filing = self.filing(accession)
        self.seal()
        aliases = [names.preferred_name(spec, kind='custom', filing=filing) for spec in selected]
        with self.patches():
            result = registry.lookup_fields(self.catalog, aliases)
        self.assertEqual(len(result), 2)
        reads = [sql for label, sql in self.statements if label == 'metadata' and 'SELECT payload FROM filings' in sql]
        self.assertEqual(len(reads), 1)

    def test_missing_source_or_dictionary_index_cannot_prove_uniqueness(self):
        first = recipe(segments='Scenario=Successor')
        storage = self.add(first)
        self.metadata.execute('DROP INDEX tags_key')
        self.seal()
        with self.patches():
            with self.assertRaisesRegex(ValueError, 'budget'):
                registry.lookup_fields(self.catalog, [names.preferred_name(first)])
            self.assertEqual(registry.lookup_fields(self.catalog, [storage])[storage]['spec'], first)

    def test_interrupted_collection_fails_closed_and_clears_progress_handler(self):
        first = recipe(segments='Scenario=Successor')
        self.add(first)
        self.seal()
        tick = iter((0.0, 3.0))
        with self.patches(), patch.object(lookup.time, 'monotonic', side_effect=lambda: next(tick, 3.0)):
            with self.assertRaisesRegex(ValueError, 'budget'):
                lookup.ContextNames(self.db, self.catalog).lookup(names.preferred_name(first))
            # The handler must be cleared even before the patched clock resets.
            self.assertEqual(self.db.execute('SELECT COUNT(*) FROM fields').fetchone()[0], 1)

    def test_final_filing_read_past_deadline_cannot_certify_alias(self):
        selected = recipe(tag='CompanyMetric', taxonomy='0000000001-24-000001')
        self.add(selected, 'custom')
        filing = self.filing(selected['taxonomy'])
        self.seal()
        alias = names.preferred_name(selected, kind='custom', filing=filing)
        clock = [0.0]
        def late_read(sql):
            if 'SELECT payload FROM filings' in sql:
                clock[0] = lookup.LOOKUP_SECONDS + 1
        self.metadata_trace = late_read
        with self.patches(), patch.object(lookup.time, 'monotonic', side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(ValueError, 'budget'):
                lookup.ContextNames(self.db, self.catalog).lookup(alias)

    def test_filing_timeout_within_whole_budget_never_skips_alias_collision(self):
        first = recipe(tag='CompanyMetric', taxonomy='0000000001-24-000001')
        second = dict(first, taxonomy='0000000001-24-000002')
        stored = [self.add(first, 'custom'), self.add(second, 'custom')]
        filing = self.filing(first['taxonomy'])
        other_filing = self.filing(second['taxonomy'])
        self.seal()
        alias = names.preferred_name(first, kind='custom', filing=filing)
        self.assertEqual(alias, names.preferred_name(second, kind='custom', filing=other_filing))
        # A slow first candidate must not disappear. A slow second candidate
        # must not leave the already-collected first candidate certified either.
        for slow in (first, second):
            with self.subTest(slow_accession=slow['taxonomy']):
                clock = [0.0]
                def late_read(sql):
                    if 'SELECT payload FROM filings' in sql and slow['taxonomy'] in sql:
                        clock[0] = .21
                self.metadata_trace = late_read
                with self.patches(), patch.object(lookup.time, 'monotonic', side_effect=lambda: clock[0]):
                    resolver = lookup.ContextNames(self.db, self.catalog)
                    with self.assertRaisesRegex(ValueError, 'filing-name lookup exceeded.*budget'):
                        resolver.lookup(alias)
                    self.assertLess(clock[0], lookup.LOOKUP_SECONDS)
                    self.assertFalse(resolver.groups)
                    self.assertNotIn(slow['taxonomy'], resolver.filings)
                    clock[0] = 0.0
                    row = self.db.execute('SELECT name,spec,label,kind FROM fields WHERE name=?',
                                          (stored[0] if slow == first else stored[1],)).fetchone()
                    self.assertIsNone(lookup.ContextNames(self.db, self.catalog).preferred(registry._entry(row)))

    def test_filing_metadata_overflow_cannot_skip_a_possible_alias_collision(self):
        first = recipe(tag='CompanyMetric', taxonomy='0000000001-24-000001')
        second = dict(first, taxonomy='0000000001-24-000002')
        self.add(first, 'custom')
        self.add(second, 'custom')
        filing = self.filing(first['taxonomy'])
        other_filing = self.filing(second['taxonomy'])
        # Even identical rows cannot establish consensus after a bounded read
        # truncates: a conflicting row could follow the 65 inspected records.
        self.metadata.executemany('INSERT INTO filings VALUES(?,?,?)',
                                  [(first['taxonomy'], '1', json.dumps(filing))] * 64)
        self.seal()
        alias = names.preferred_name(first, kind='custom', filing=filing)
        self.assertEqual(alias, names.preferred_name(second, kind='custom', filing=other_filing))
        with self.patches():
            resolver = lookup.ContextNames(self.db, self.catalog)
            with self.assertRaisesRegex(ValueError, 'filing-name lookup exceeded its candidate budget'):
                resolver.lookup(alias)
            self.assertFalse(resolver.groups)
            self.assertNotIn(first['taxonomy'], resolver.filings)
            row = self.db.execute('SELECT name,spec,label,kind FROM fields LIMIT 1').fetchone()
            self.assertIsNone(lookup.ContextNames(self.db, self.catalog).preferred(registry._entry(row)))

    def test_unknown_names_and_null_contexts_never_invent_recipes(self):
        selected = recipe(segments='Scenario=Successor')
        self.add(selected)
        self.add(dict(selected, segments=None))
        self.add(dict(selected, coreg=None))
        self.seal()
        with self.patches():
            result = registry.lookup_fields(self.catalog, ['sec_fictional_metric__scenario_successor',
                        'sec_assets__scenario_unknown', 'SEC_ASSETS__SCENARIO_SUCCESSOR'])
        self.assertEqual(result, {})

    def test_lookup_statements_are_read_only_and_use_existing_indexes(self):
        selected = recipe(tag='CompanyMetric', taxonomy='0000000001-24-000001', unit='EUR', qtrs=6,
                          segments='Scenario=Successor')
        self.add(selected, 'custom')
        filing = self.filing(selected['taxonomy'])
        self.seal()
        alias = names.preferred_name(selected, kind='custom', filing=filing)
        with self.patches():
            self.assertEqual(registry.lookup_fields(self.catalog, [alias])[alias]['spec'], selected)
        statements = [sql for _, sql in self.statements]
        for index in ('fields_tag', 'tags_key', 'filings_adsh'):
            self.assertTrue(any('INDEXED BY ' + index in sql for sql in statements))
        self.assertFalse(any(sql.lstrip().split()[0].upper() in {'CREATE', 'INSERT', 'UPDATE', 'DELETE', 'REPLACE'}
                             for sql in statements))
        self.assertFalse(any('LOWER(' in sql.upper() for sql in statements))
        # Context normalization reads specs only after indexed nomination;
        # issuer metadata must still use exact accession lookups, not JSON scans.
        for label, sql in self.statements:
            if 'json_extract' in sql.lower():
                self.assertEqual(label, 'dictionary')
                self.assertIn('INDEXED BY', sql)
                self.assertIn('LIMIT', sql)


if __name__ == '__main__':
    unittest.main()
