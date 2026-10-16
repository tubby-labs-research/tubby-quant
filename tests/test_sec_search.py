"""Independent SQLite discovery fixtures; no NUM scan or real dictionary build."""
import json
import sqlite3
import unittest
from unittest.mock import patch

from qlib_yahoo import sec_search as search
from qlib_yahoo import sec_registry as registry
from qlib_yahoo.sec_concepts import ConceptSeeds


class SearchTests(unittest.TestCase):
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
        self.common = {'sec_inventory': dict(tag='InventoryNet', taxonomy='us-gaap/',
                                           unit='USD', qtrs=0, segments='', coreg='')}
        self.add('sec_custom_inventory', 'Inventory', 'filing', 'custom')
        self.add('sec_edition_inventory', 'InventoryNet', 'us-gaap/2024')
        self.add('sec_family_inventory', 'InventoryNet', 'us-gaap/', unit='EUR')
        self.add('sec_inventory', 'InventoryNet', 'us-gaap/', common=1)
        self.add('sec_other', 'Other', 'us-gaap/', description='Merchandise balance sheet')
        self.seal()

    def add(self, name, tag, taxonomy, kind='standard', common=0, unit='USD', description='Merchandise'):
        spec = dict(tag=tag,taxonomy=taxonomy,unit=unit,qtrs=0,segments=name if name.startswith('sec_extra') else '',coreg='')
        self.db.execute('INSERT INTO fields(name,spec,label,kind,tag,description,taxonomy,common,unit,qtrs) VALUES(?,?,?,?,?,?,?,?,?,?)',
                        (name,json.dumps(spec),tag,kind,tag,description,taxonomy,common,unit,'0'))

    def seal(self):
        self.db.execute("INSERT INTO fields_fts(fields_fts) VALUES('rebuild')")

    def run_search(self, query='', **kwargs):
        return search.search_dictionary(self.db, query, common_fields=self.common, **kwargs)

    def names(self, result):
        return [item['name'] for item in result['items']]

    def test_complete_scope_ranking_and_common_definition(self):
        result = self.run_search('Inventory', include_custom=True)
        self.assertEqual(self.names(result), ['sec_inventory','sec_family_inventory',
                                            'sec_edition_inventory','sec_custom_inventory'])
        self.assertEqual(result['items'][0]['spec'], self.common['sec_inventory'])
        self.assertEqual(result['total'], 4)
        self.assertFalse(result['truncated'])
        self.assertTrue(result['ranking_complete'])
        self.assertEqual(self.run_search('Inventory')['total'], 3)

    def test_descriptions_and_multiple_tokens_keep_broad_semantics(self):
        self.assertEqual(self.run_search('Merchandise',include_custom=True)['total'], 5)
        self.assertEqual(self.names(self.run_search('balance sheet')), ['sec_other'])
        self.assertEqual(self.run_search('Inventory balance')['total'], 0)
        self.assertEqual(self.names(self.run_search('SEC_INVENTORY'))[0], 'sec_inventory')

    def test_complete_pagination_matches_whole_ranked_result(self):
        for query in ('', 'Inventory', 'Merchandise'):
            all_items = self.run_search(query,include_custom=True)['items']
            paged = []
            for offset in range(len(all_items)):
                page = self.run_search(query,include_custom=True,offset=offset,limit=1)
                paged += page['items']
                self.assertEqual(page['has_more'], offset + 1 < len(all_items))
            self.assertEqual(paged, all_items)

    def test_large_fts_sample_is_explicit_and_paginates_deterministically(self):
        for i in range(30):
            self.add(f'sec_extra_{i:03}', 'InventoryNet', 'us-gaap/')
        self.seal()
        result = self.run_search('Merchandise', candidate_limit=10, limit=3)
        self.assertIsNone(result['total'])
        self.assertTrue(result['truncated'])
        self.assertFalse(result['ranking_complete'])
        self.assertEqual(result['search_scope'], 'all_fields')
        self.assertIn('browse all fields',result['message'])
        sample = self.run_search('Merchandise', candidate_limit=10,limit=50)['items']
        pages = [self.run_search('Merchandise', candidate_limit=10,limit=3,offset=i)['items']
                 for i in (0,3,6,9)]
        self.assertEqual(sum(pages, []), sample)
        self.assertFalse(self.run_search('Merchandise',candidate_limit=10,offset=10)['has_more'])
        # Truncation does not change exact lookup or unfiltered availability.
        self.assertEqual(self.names(self.run_search('sec_extra_029')), ['sec_extra_029'])
        self.assertEqual(self.run_search('',include_custom=True,field_count=35)['total'],35)

    def test_inventory_seed_does_not_replace_description_search(self):
        self.add('sec_extra_description', 'Unrelated', 'us-gaap/', description='Inventory equipment')
        self.seal()
        self.assertIn('sec_extra_description', self.names(self.run_search('Inventory')))
        partial = self.run_search('Inventory',candidate_limit=1)
        self.assertEqual(self.names(partial)[0], 'sec_inventory')
        self.assertTrue(partial['truncated'])

    def test_timeout_is_honest_and_handler_is_cleared(self):
        # Name probes survive; force interruption during the bounded tag/FTS work.
        calls = 0
        def clock():
            nonlocal calls
            calls += 1
            return 0 if calls == 1 else 1
        for i in range(150):
            self.add(f'sec_extra_{i}', 'InventoryNet', 'us-gaap/')
        self.seal()
        with patch.object(search.time, 'monotonic', side_effect=clock):
            result = self.run_search('Inventory',budget_ms=1)
        self.assertTrue(result['truncated'])
        self.assertIsNone(result['total'])
        self.assertIsNone(result['has_more'])
        self.assertIn('sec_inventory', self.names(result))
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM fields').fetchone()[0],155)

    def test_invalid_queries_literal_fts_and_non_interrupt_errors(self):
        for kwargs in ({'query':None},{'query':'x'*4097},{'include_custom':1},
                       {'limit':True},{'limit':0},{'offset':-1},{'candidate_limit':0},{'budget_ms':0}):
            with self.assertRaises(ValueError):
                search.search_dictionary(self.db,**kwargs)
        for query in ('***', "'; DROP TABLE fields; --", 'nonsensezzzz'):
            self.assertEqual(self.run_search(query)['total'],0)
        self.db.execute('DROP TABLE fields_fts')
        with self.assertRaises(sqlite3.OperationalError):
            self.run_search('Inventory')

    def test_search_does_not_write_and_plans_use_existing_indexes(self):
        statements = []
        self.db.set_trace_callback(statements.append)
        self.run_search('Inventory')
        self.run_search('')
        self.assertFalse(any(sql.lstrip().split()[0].upper() in {'CREATE','INSERT','UPDATE','DELETE','REPLACE'} for sql in statements))
        for sql,params,index in (
            ('SELECT name FROM fields WHERE name=?',('sec_inventory',),'sqlite_autoindex_fields_1'),
            ('SELECT name FROM fields INDEXED BY fields_tag WHERE tag=? LIMIT 128',('InventoryNet',),'fields_tag'),
            ("SELECT name FROM fields WHERE kind='standard' ORDER BY common DESC,name LIMIT 50",(),'fields_kind_name')):
            plan = str(self.db.execute('EXPLAIN QUERY PLAN '+sql,params).fetchall())
            self.assertIn(index,plan)

    def add_recipe(self, spec, *, kind='standard', description=''):
        name = registry._name_for_recipe(spec)
        self.db.execute('INSERT INTO fields(name,spec,label,kind,tag,description,taxonomy,common,unit,qtrs) VALUES(?,?,?,?,?,?,?,?,?,?)',
                        (name,registry.source._canonical(spec).decode(),spec['tag'],kind,spec['tag'],
                         description,spec['taxonomy'],0,spec['unit'],str(spec['qtrs'])))
        return name

    def test_phrase_seeds_survive_complete_fts_with_truthful_union_count(self):
        spec = dict(tag='RevenueFromContractWithCustomerExcludingAssessedTax',taxonomy='us-gaap/',unit='USD',qtrs=1,segments='',coreg='')
        name = self.add_recipe(spec)
        self.seal()
        # Unsplit TAG/label/name can't satisfy these literal separate AND words.
        query = 'revenue from contract with customer excluding assessed tax'
        self.assertEqual(self.run_search(query)['total'],0)
        out = self.run_search(query, concept_seeds=[dict(tag=spec['tag'],spec=spec,rank=0)])
        self.assertEqual(self.names(out),[name])
        self.assertEqual(out['total'],1)
        self.assertFalse(out['truncated'])
        self.assertEqual(out['items'][0]['spec'],spec)

    def test_default_units_numeric_duration_and_exact_tag_outrank_related(self):
        base = dict(tag='EarningsPerShareBasic',taxonomy='us-gaap/',unit='USD',qtrs=0,segments='',coreg='')
        specs = [dict(base,unit=unit,qtrs=q) for unit in ('shares','USD/shares','USD') for q in (4,1,0)]
        names = {registry.source._canonical(spec).decode():self.add_recipe(spec) for spec in specs}
        context = dict(base,segments='custom-context')
        context_name = self.add_recipe(context)
        related = dict(base,tag='EarningsPerShareBasicAndDiluted')
        related_name = self.add_recipe(related)
        self.seal()
        seeds = [dict(tag=base['tag'],spec=base,rank=0),dict(tag=related['tag'],spec=related,rank=1)]
        out = self.run_search('EarningsPerShareBasic',concept_seeds=seeds)
        expected = [names[registry.source._canonical(dict(base,unit=unit,qtrs=q)).decode()]
                    for unit in ('USD','USD/shares','shares') for q in (0,1,4)]
        self.assertEqual(self.names(out),expected+[context_name,related_name])
        self.assertEqual(out['total'],11)

    def test_explicit_alias_does_not_broaden_period_or_unit(self):
        base = dict(tag='AssetsCurrent',taxonomy='us-gaap/',unit='USD',qtrs=0,segments='',coreg='')
        name = self.add_recipe(base)
        self.add_recipe(dict(base,qtrs=4))
        self.add_recipe(dict(base,unit='shares'))
        self.seal()
        out = self.run_search('sec_assets_curr',concept_seeds=[dict(tag=base['tag'],spec=base,preferred=True)])
        self.assertEqual(self.names(out)[0],name)
        self.assertEqual(out['items'][0]['spec'],base)
        # Literal FTS may supplement the explicitly pinned alias, but other
        # periods/units must not receive the same exact-recipe relevance.
        self.assertEqual(out['total'],3)

    def test_exact_name_survives_completed_fts_and_deduplicates(self):
        self.db.execute("UPDATE fields SET label='',description='',tag='Other' WHERE name='sec_inventory'")
        self.seal()
        out = self.run_search('SEC_INVENTORY')
        self.assertEqual(self.names(out)[0],'sec_inventory')
        self.assertEqual(self.names(out).count('sec_inventory'),1)
        self.assertEqual(out['total'],3)

    def test_seed_truncation_prevents_exhaustive_count_even_when_fts_complete(self):
        seeds = ConceptSeeds([dict(tag='InventoryNet',spec=self.common['sec_inventory'])])
        seeds.truncated = True
        out = self.run_search('inventory net',concept_seeds=seeds)
        self.assertIn('sec_inventory',self.names(out))
        self.assertIsNone(out['total'])
        self.assertTrue(out['truncated'])
        self.assertFalse(out['ranking_complete'])

    def test_long_phrase_retains_lone_nondefault_unit_and_duration(self):
        tag = 'RevenueFromContractWithCustomerExcludingAssessedTax'
        default = dict(tag=tag,taxonomy='us-gaap/',unit='USD',qtrs=0,segments='',coreg='')
        query = 'revenue from contract with customer excluding assessed tax'
        for recipe in (dict(default,unit='EUR',qtrs=1),dict(default,qtrs=6)):
            with self.subTest(recipe=recipe):
                name = self.add_recipe(recipe)
                self.seal()
                out = self.run_search(query,concept_seeds=[dict(tag=tag,spec=default)])
                self.assertIn(name,self.names(out))
                self.assertFalse(out['truncated'])
                self.assertEqual(out['total'],len(out['items']))
                self.db.execute('DELETE FROM fields WHERE name=?',(name,))

    def test_tag_supplement_overflow_never_claims_complete_count(self):
        tag = 'RevenueFromContractWithCustomerExcludingAssessedTax'
        base = dict(tag=tag,taxonomy='us-gaap/',unit='EUR',qtrs=6,segments='',coreg='')
        for i in range(129):
            self.add_recipe(dict(base,segments=str(i)))
        self.seal()
        out = self.run_search('revenue from contract with customer excluding assessed tax',concept_seeds=[tag])
        self.assertEqual(out['matched_sample'],128)
        self.assertIsNone(out['total'])
        self.assertTrue(out['truncated'])
        self.assertFalse(out['ranking_complete'])

    def test_interrupted_source_nomination_cannot_promise_stable_pages(self):
        seeds = ConceptSeeds(['InventoryNet'])
        seeds.truncated = seeds.interrupted = True
        out = self.run_search('inventory net',concept_seeds=seeds,limit=1)
        self.assertTrue(out['truncated'])
        self.assertIsNone(out['total'])
        self.assertIsNone(out['has_more'])

    def test_custom_filter_never_reclassifies_seeded_recipe(self):
        spec = dict(tag='CustomOnly',taxonomy='0000000001-24-000001',unit='USD',qtrs=1,segments='',coreg='')
        name = self.add_recipe(spec,kind='custom')
        self.seal()
        seeds = [dict(tag=spec['tag'],spec=spec,preferred=True)]
        self.assertEqual(self.run_search('nonfts alias',concept_seeds=seeds)['total'],0)
        out = self.run_search('nonfts alias',concept_seeds=seeds,include_custom=True)
        self.assertEqual(self.names(out),[name])
        self.assertEqual(out['items'][0]['kind'],'custom')

    def test_seed_queries_use_name_equality_and_never_load_all_specs(self):
        statements = []
        self.db.set_trace_callback(statements.append)
        self.run_search('inventory net',concept_seeds=['InventoryNet'])
        probes = [sql for sql in statements if 'WHERE name=' in sql]
        self.assertGreaterEqual(len(probes),15)
        self.assertFalse(any('LOWER(' in sql.upper() or 'json_extract' in sql for sql in statements))

    def test_invalid_seeds_and_fixed_distinct_tag_cap(self):
        for seeds in ([None],[dict(tag='Assets',rank=-1)],
                      [dict(tag='Other',spec=self.common['sec_inventory'])]):
            with self.assertRaises(ValueError):
                self.run_search('inventory',concept_seeds=seeds)
        result = self.run_search('nonfts alias',concept_seeds=[f'Tag{i}' for i in range(129)])
        self.assertTrue(result['truncated'])
        self.assertIsNone(result['total'])


if __name__ == '__main__':
    unittest.main()
