"""Independent company-discovery fixtures; no source data or sealed DB rebuild."""
import json
import sqlite3
import unittest
from pathlib import Path
from unittest.mock import patch

from qlib_yahoo import sec_concepts, sec_registry as registry, sec_search as search
from qlib_yahoo.sec_names import curated_defs


class CompanySearchTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        self.addCleanup(self.db.close)
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

    def add(self, tag='InventoryNet', taxonomy='us-gaap/', unit='USD', qtrs=0,
            segments='', coreg='', kind='standard', label=None, description='discovery'):
        spec = dict(tag=tag, taxonomy=taxonomy, unit=unit, qtrs=qtrs,
                    segments=segments, coreg=coreg)
        name = registry._name_for_recipe(spec)
        self.db.execute('''INSERT INTO fields(name,spec,label,kind,tag,description,taxonomy,
            common,unit,qtrs) VALUES(?,?,?,?,?,?,?,?,?,?)''',
            (name, json.dumps(spec), tag if label is None else label, kind, tag,
             description, taxonomy, int(name in registry.COMMON_FIELDS), unit, str(qtrs)))
        return name

    def seal(self):
        self.db.execute("INSERT INTO fields_fts(fields_fts) VALUES('rebuild')")
        self.db.commit()

    def run_search(self, query='', **kwargs):
        return search.search_dictionary(self.db, query, view='company',
                                        common_fields=registry.COMMON_FIELDS, **kwargs)

    @staticmethod
    def keys(result):
        return [(item['tag'], item['taxonomy']) for item in result['items']]

    @staticmethod
    def variants(result):
        return [v for item in result['items'] for v in item['variants']]

    def test_exact_tag_family_grouping_precedes_pagination(self):
        for tag, taxonomy in (('InventoryNet', 'us-gaap/'), ('InventoryNet', 'ifrs-full/'),
                              ('Inventory', 'us-gaap/'), ('inventoryNet', 'us-gaap/')):
            for unit in ('USD', 'EUR', 'shares'):
                for qtrs in (0, 1, 6):
                    self.add(tag=tag, taxonomy=taxonomy, unit=unit, qtrs=qtrs)
            for year in (2023, 2024):
                self.add(tag=tag, taxonomy=taxonomy + str(year))
        self.seal()
        whole = self.run_search('discovery')
        self.assertEqual(whole['view'], 'company')
        self.assertEqual(whole['search_scope'], 'company_concepts')
        self.assertEqual(whole['total'], 4)
        self.assertEqual(whole['matched_sample'], 4)
        self.assertEqual(len(set(self.keys(whole))), 4)
        pages = []
        for offset in range(4):
            page = self.run_search('discovery', limit=1, offset=offset)
            self.assertEqual(len(page['items']), 1)
            self.assertEqual(len(page['items'][0]['variants']), 9)
            self.assertEqual(page['has_more'], offset < 3)
            pages += page['items']
        self.assertEqual(pages, whole['items'])
        for item in whole['items']:
            self.assertEqual(set(item), {'tag', 'taxonomy', 'label', 'variants'})
            for variant in item['variants']:
                self.assertEqual(set(variant), {'name', 'spec', 'label', 'kind'})
                self.assertEqual(variant['spec']['taxonomy'], item['taxonomy'])
        self.assertEqual(self.run_search('discovery', offset=4)['items'], [])

    def test_all_special_contexts_and_unconfirmed_kinds_are_excluded(self):
        ordinary = self.add()
        for dimension in ('Scenario=Successor', 'Scenario=Predecessor', 'Axis=Segment',
                          'BusinessSegment=Asia', ' ', 'Restatement=Yes'):
            self.add(segments=dimension)
        self.add(coreg='Subsidiary')
        self.add(segments=None)
        self.add(coreg=None, unit='EUR')
        for index, kind in enumerate(('custom', 'unknown', None)):
            self.add(tag='Other' + str(index), kind=kind)
        self.add(taxonomy='us-gaap/2024')
        self.add(taxonomy='0000000001-24-000001', kind='custom')
        for index, taxonomy in enumerate(('0001/', 'us-gaap/nested/', 'bad family/', '', 'unknown')):
            self.add(tag='Malformed' + str(index), taxonomy=taxonomy)
        self.seal()
        for include_custom in (False, True):
            out = self.run_search('discovery', include_custom=include_custom)
            self.assertEqual(out['total'], 1)
            self.assertEqual([v['name'] for v in self.variants(out)], [ordinary])
            self.assertFalse(out['truncated'])

    def test_advanced_exact_names_and_seed_specs_cannot_bypass_filter(self):
        for spec_changes in ({'segments': 'Scenario=Successor'}, {'coreg': 'Child'},
                             {'taxonomy': 'us-gaap/2024'}, {'kind': 'custom'}):
            with self.subTest(changes=spec_changes):
                name = self.add(tag='Exact' + str(len(spec_changes)) + str(self.db.total_changes),
                                **spec_changes)
                self.seal()
                spec = json.loads(self.db.execute('SELECT spec FROM fields WHERE name=?', (name,)).fetchone()[0])
                out = self.run_search(name.upper(), include_custom=True,
                                      concept_seeds=[dict(tag=spec['tag'], spec=spec, preferred=True)])
                self.assertEqual(out['items'], [])
                self.assertEqual(out['total'], 0)

    def test_ten_thousand_actual_special_rows_before_blank_recipes(self):
        for index in range(10000):
            self.add(segments=f'Scenario=Successor;Axis={index}', description='crowding')
        names = {self.add(description='crowding'), self.add(unit='EUR', qtrs=6, description='crowding')}
        self.seal()
        # Literal FTS and tag supplements must filter BEFORE their candidate
        # caps. This is real SQLite ordering, not a mocked already-filtered list.
        for query, seeds in (('crowding', []), ('inventory net', ['InventoryNet']),
                             ('inventory', [])):
            with self.subTest(query=query):
                out = self.run_search(query, candidate_limit=1 if query == 'inventory net' else 2,
                                      concept_seeds=seeds)
                self.assertEqual(out['total'], 1)
                self.assertFalse(out['truncated'])
                self.assertEqual({v['name'] for v in self.variants(out)}, names)
                self.assertEqual(out['matched_sample'], 1)

    def test_default_name_probe_survives_interrupted_tag_supplement(self):
        for index in range(10000):
            self.add(segments=f'Scenario=Successor;Axis={index}')
        default = self.add()
        self.seal()
        # Allow probes; interrupt the first indexed per-tag scan.
        probes_finished = False
        def trace(sql):
            nonlocal probes_finished
            if 'INDEXED BY fields_tag' in sql:
                probes_finished = True
        self.db.set_trace_callback(trace)
        with patch.object(search.time, 'monotonic', side_effect=lambda: 1 if probes_finished else 0):
            out = self.run_search('inventory net', concept_seeds=['InventoryNet'], budget_ms=1)
        self.assertIn(default, [v['name'] for v in self.variants(out)])
        self.assertTrue(out['truncated'])
        self.assertIsNone(out['total'])
        self.assertIsNone(out['has_more'])
        self.assertFalse(out['ranking_complete'])
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM fields').fetchone()[0], 10001)

    def test_lone_nondefault_currency_and_long_duration_are_observed_only(self):
        tag = 'RevenueFromContractWithCustomerExcludingAssessedTax'
        name = self.add(tag=tag, unit='EUR', qtrs=6)
        self.seal()
        out = self.run_search('revenue from contract with customer excluding assessed tax',
                              concept_seeds=[tag])
        self.assertEqual([v['name'] for v in self.variants(out)], [name])
        self.assertEqual(out['total'], 1)
        self.assertFalse(out['truncated'])
        spec = self.variants(out)[0]['spec']
        self.assertEqual((spec['unit'], spec['qtrs']), ('EUR', 6))

    def test_bounded_fts_counts_groups_and_honestly_marks_variant_overflow(self):
        for qtrs in range(20):
            self.add(qtrs=qtrs)
        self.add(tag='Assets')
        self.seal()
        out = self.run_search('discovery', candidate_limit=5)
        self.assertEqual(out['matched_sample'], 1)
        self.assertEqual(len(self.variants(out)), 5)
        self.assertIsNone(out['total'])
        self.assertTrue(out['truncated'])
        self.assertFalse(out['ranking_complete'])
        self.assertFalse(out['has_more'])
        self.assertIn('variants', out['message'])
        self.assertEqual(self.run_search('discovery', candidate_limit=5, offset=1)['items'], [])

    def test_partial_group_sample_pagination_is_deterministic(self):
        for tag in ('Assets', 'InventoryNet', 'Liabilities'):
            for qtrs in (0, 1):
                self.add(tag=tag, qtrs=qtrs)
        self.seal()
        whole = self.run_search('discovery', candidate_limit=4)
        self.assertEqual(whole['matched_sample'], 2)
        first = self.run_search('discovery', candidate_limit=4, limit=1)
        last = self.run_search('discovery', candidate_limit=4, limit=1, offset=1)
        self.assertTrue(first['has_more'])
        self.assertFalse(last['has_more'])
        self.assertEqual(first['items'] + last['items'], whole['items'])
        self.assertIsNone(first['total'])

    def test_tag_supplement_limit_is_partial_even_if_only_one_group(self):
        tag = 'RevenueFromContractWithCustomerExcludingAssessedTax'
        for qtrs in range(search.TAG_CANDIDATE_LIMIT + 1):
            self.add(tag=tag, unit='EUR', qtrs=qtrs)
        self.seal()
        out = self.run_search('revenue from contract with customer excluding assessed tax',
                              concept_seeds=[tag])
        self.assertEqual(out['matched_sample'], 1)
        self.assertEqual(len(self.variants(out)), search.TAG_CANDIDATE_LIMIT)
        self.assertIsNone(out['total'])
        self.assertTrue(out['truncated'])

    def test_source_nomination_partial_flags_survive_grouping(self):
        self.add()
        self.seal()
        seeds = sec_concepts.ConceptSeeds(['InventoryNet'])
        seeds.truncated = True
        for interrupted in (False, True):
            seeds.interrupted = interrupted
            out = self.run_search('inventory net', concept_seeds=seeds)
            self.assertEqual(out['matched_sample'], 1)
            self.assertIsNone(out['total'])
            self.assertTrue(out['truncated'])
            self.assertFalse(out['ranking_complete'])
            self.assertIs(out['has_more'], None if interrupted else False)

    def test_company_seed_iterator_keeps_existing_tag_bound(self):
        def seeds():
            for index in range(129):
                yield f'Tag{index}'
            raise AssertionError('unbounded seed materialization')
        out = self.run_search('unused', concept_seeds=seeds(), budget_ms=1000)
        self.assertTrue(out['truncated'])
        self.assertIsNone(out['total'])
        self.assertEqual(out['matched_sample'], 0)

    def test_empty_search_is_reviewed_observed_starters_not_global_inventory(self):
        self.add()
        self.add(unit='EUR', qtrs=6)
        self.add(taxonomy='ifrs-full/')
        self.add(tag='NetIncomeLoss', qtrs=1)
        self.add(tag='NetIncomeLoss', qtrs=4)
        self.add(tag='NotReviewed')
        self.add(tag='Assets', segments='Scenario=Successor')
        self.seal()
        out = self.run_search('  ', field_count=29622849)
        self.assertEqual(out['search_scope'], 'company_starter')
        self.assertEqual(out['field_count'], 29622849)
        self.assertEqual(out['total'], 2)
        self.assertEqual(out['matched_sample'], 2)
        self.assertEqual(set(self.keys(out)), {('InventoryNet', 'us-gaap/'), ('NetIncomeLoss', 'us-gaap/')})
        self.assertEqual(len(self.variants(out)), 4)
        self.assertIn('not all company concepts', out['message'])
        self.assertFalse(out['truncated'])
        first = self.run_search('', limit=1)
        last = self.run_search('', limit=1, offset=1)
        self.assertTrue(first['has_more'])
        self.assertFalse(last['has_more'])
        self.assertEqual(first['items'] + last['items'], out['items'])

    def test_no_observed_starters_or_unknown_search_invents_no_names(self):
        self.add(tag='NotReviewed')
        self.seal()
        self.assertEqual(self.run_search()['total'], 0)
        self.assertEqual(self.run_search()['items'], [])
        self.assertEqual(self.run_search('sec_nonsense_fictional')['total'], 0)
        self.assertEqual(self.run_search('sec_nonsense_fictional')['items'], [])
        self.assertEqual(self.run_search('***')['search_scope'], 'company_concepts')

    def test_empty_starter_interruption_keeps_unknown_counts_and_message(self):
        self.add()
        self.seal()
        calls = 0
        def clock():
            nonlocal calls
            calls += 1
            return 0 if calls == 1 else 1
        with patch.object(search.time, 'monotonic', side_effect=clock):
            out = self.run_search('', budget_ms=1)
        self.assertTrue(out['truncated'])
        self.assertIsNone(out['total'])
        self.assertIsNone(out['has_more'])
        self.assertIn('starters', out['message'])

    def test_legacy_recipe_default_and_explicit_view_match(self):
        self.add()
        advanced = self.add(segments='Scenario=Successor')
        self.add(taxonomy='us-gaap/2024')
        self.add(tag='Custom', kind='custom')
        self.seal()
        for query in ('', 'discovery', advanced):
            default = search.search_dictionary(self.db, query, include_custom=True)
            explicit = search.search_dictionary(self.db, query, include_custom=True, view='recipes')
            self.assertEqual(default, explicit)
            self.assertEqual(default['view'], 'recipes')
            self.assertEqual(default['search_scope'], 'all_fields')
            self.assertTrue(all('spec' in item and 'variants' not in item for item in default['items']))

    def fresh_connection(self, _):
        connection = sqlite3.connect(':memory:')
        self.db.backup(connection)
        connection.execute('PRAGMA query_only=ON')
        return connection

    def registry_search(self, query='', **kwargs):
        seeds = sec_concepts.ConceptSeeds([dict(tag='InventoryNet', spec=curated_defs()['sec_inventory'])])
        with patch.object(registry, '_ready', return_value=Path('unused')), \
             patch.object(registry, '_db', side_effect=self.fresh_connection), \
             patch.object(registry, 'dictionary_status', return_value={'field_count': 29622849}), \
             patch.object(sec_concepts, 'discover_concepts', return_value=seeds), \
             patch.object(registry.source, '_scan', side_effect=AssertionError('raw scan')), \
             patch.object(registry.source, '_hash', side_effect=AssertionError('bulk hash')):
            return registry.search_dictionary(Path('unused'), query, **kwargs)

    def test_registry_applies_actual_lowercase_readable_overlay_to_every_variant(self):
        storage = [self.add(unit='EUR', qtrs=6), self.add(qtrs=1), self.add()]
        self.seal()
        out = self.registry_search('inventory net', view='company')
        variants = self.variants(out)
        self.assertEqual({v['name'] for v in variants}, {'sec_inventory', 'sec_inventory_q', 'sec_inventory_6q__eur'})
        self.assertTrue(all(v['name'] == v['name'].lower() for v in variants))
        self.assertEqual({v.get('storage_name', v['name']) for v in variants}, set(storage))
        self.assertEqual(out['field_count'], 29622849)
        flat = self.registry_search('inventory net')
        self.assertEqual({v['name'] for v in flat['items']}, {v['name'] for v in variants})
        for query in ('inventory', 'SEC_INVENTORY', 'inventorynet'):
            self.assertEqual(self.registry_search(query, view='company')['total'], 1)

    def test_registry_empty_company_search_does_not_nominate_source_concepts(self):
        self.add()
        self.seal()
        # Empty TAG nomination already opens nothing, and company starter seeds
        # are taken from the checked-in reviewed definitions in recipe search.
        with patch.object(sec_concepts, 'discover_concepts', wraps=sec_concepts.discover_concepts) as discover, \
             patch.object(registry, '_ready', return_value=Path('unused')), \
             patch.object(registry, '_db', side_effect=self.fresh_connection), \
             patch.object(registry, 'dictionary_status', return_value={'field_count': 1}), \
             patch.object(registry.source, '_db', side_effect=AssertionError('catalog read')):
            out = registry.search_dictionary(Path('unused'), '', view='company')
        discover.assert_called_once_with(Path('unused'), '')
        self.assertEqual(out['total'], 1)
        self.assertEqual(out['search_scope'], 'company_starter')

    def test_company_queries_read_only_bounded_and_use_existing_indexes(self):
        self.add()
        self.seal()
        statements = []
        self.db.set_trace_callback(statements.append)
        self.db.execute('PRAGMA query_only=ON')
        self.run_search('inventory net', concept_seeds=['InventoryNet'])
        self.run_search('discovery')
        self.run_search('')
        queries = [sql for sql in statements if sql.startswith('SELECT')]
        self.assertFalse(any('COUNT(' in sql or 'LOWER(' in sql.upper() for sql in queries))
        self.assertTrue(any('INDEXED BY fields_tag' in sql for sql in queries))
        for sql in queries:
            if 'INDEXED BY fields_tag' in sql:
                self.assertIn('LIMIT 129', sql)
                self.assertIn('fields_tag', str(self.db.execute('EXPLAIN QUERY PLAN ' + sql).fetchall()))
            elif 'fields_fts MATCH' in sql:
                self.assertIn('LIMIT 2049', sql)
                self.assertIn('VIRTUAL TABLE INDEX', str(self.db.execute('EXPLAIN QUERY PLAN ' + sql).fetchall()))
            elif 'WHERE name=' in sql:
                self.assertIn('sqlite_autoindex_fields_1', str(self.db.execute('EXPLAIN QUERY PLAN ' + sql).fetchall()))
        self.assertFalse(any(sql.lstrip().split()[0] in {'CREATE', 'INSERT', 'UPDATE', 'DELETE'} for sql in statements))

    def test_invalid_view_fails_before_source_discovery(self):
        for value in ('', 'companies', None, True, []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                search.search_dictionary(self.db, view=value)
        with patch.object(registry, '_ready', return_value=Path('unused')), \
             patch.object(sec_concepts, 'discover_concepts', side_effect=AssertionError('nomination')):
            with self.assertRaises(ValueError):
                registry.search_dictionary(Path('unused'), view='wrong')


if __name__ == '__main__':
    unittest.main()
