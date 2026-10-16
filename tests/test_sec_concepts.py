"""Indexed, bounded source-concept discovery; no NUM or dictionary builds."""
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from qlib_yahoo import sec_concepts as concepts


class ConceptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        # macOS's temporary root can use /var -> /private/var; catalog paths
        # must have no redirected ancestors, even for this synthetic fixture.
        self.catalog = Path(self.temp.name).resolve()
        self.database = self.catalog / 'catalog.sqlite'
        with closing(sqlite3.connect(self.database)) as db, db:
            db.executescript('CREATE TABLE tags(tag TEXT,version TEXT,payload TEXT);'
                             'CREATE INDEX tags_key ON tags(tag COLLATE NOCASE,version);')
            rows = [('Assets', 'us-gaap/2024', '0'),
                    ('EarningsPerShareBasic', 'us-gaap/2024', '0'),
                    ('EarningsPerShareBasic', 'us-gaap/2025', '0'),
                    ('EarningsPerShareBasicAndDiluted', 'us-gaap/2024', '0'),
                    ('CustomOnly', '0000000001-24-000001', '1'),
                    ('Unknown', 'us-gaap/2024', None),
                    ('AssetsCurrent', 'us-gaap/2024', '0'),
                    ('NetIncomeLoss', 'us-gaap/2024', '0'),
                    ('NetCashProvidedByUsedInOperatingActivities', 'us-gaap/2024', '0'),
                    ('Assets', 'ifrs/2024', '0')]
            db.executemany('INSERT INTO tags VALUES(?,?,?)',
                           [(tag,version,json.dumps(dict(custom=flag))) for tag,version,flag in rows])

    def discover(self, query, **kwargs):
        kwargs.setdefault('aliases', {})
        return concepts.discover_concepts(self.catalog, query, **kwargs)

    def test_compact_phrase_camel_case_and_sec_prefix(self):
        for query in ('earnings per share basic', 'earnings_per_share_basic',
                      'EarningsPerShareBasic', 'sec_EarningsPerShareBasic'):
            seeds = self.discover(query)
            self.assertFalse(seeds.truncated)
            self.assertEqual(seeds[0]['tag'], 'EarningsPerShareBasic')
            self.assertEqual(seeds[0]['rank'], 0)
            self.assertEqual(seeds[0]['spec']['taxonomy'], 'us-gaap/')
            self.assertEqual(seeds[1]['tag'], 'EarningsPerShareBasicAndDiluted')
            self.assertEqual(seeds[1]['rank'], 1)

    def test_whole_query_no_or_expansion_or_punctuation_erasure(self):
        for query in ('assets nonsense', 'assets OR earnings', 'assets*', "'; DROP TABLE tags; --"):
            self.assertEqual(self.discover(query), [])
        self.assertEqual(self.discover('assets')[0]['tag'], 'Assets')

    def test_accounting_terms_match_whole_query_only(self):
        for query, tag in (
            ('cash flow','NetCashProvidedByUsedInOperatingActivities'),
            ('operating cash flow','NetCashProvidedByUsedInOperatingActivities'),
            ('current assets','AssetsCurrent'), ('net income','NetIncomeLoss'),
            ('eps basic','EarningsPerShareBasic'),
        ):
            self.assertEqual(self.discover(query)[0]['tag'],tag)
        for query in ('cash flow nonsense','eps basic nonsense','net income nonsense'):
            self.assertEqual(self.discover(query),[])

    def test_reviewed_base_prefix_and_bare_alias_default_expansion(self):
        from qlib_yahoo import sec_names
        spec = dict(tag='NetCashProvidedByUsedInOperatingActivities',taxonomy='us-gaap/',unit='USD',qtrs=0,segments='',coreg='')
        with patch.object(sec_names, 'CURATED_NAMES', {spec['tag']:'cashflow_op'}), \
             patch.object(sec_names, 'curated_defs', return_value={'sec_cashflow_op':spec}), \
             patch.object(sec_names, 'spec_for_name', side_effect=lambda name: dict(spec) if name == 'sec_cashflow_op' else None):
            seeds = concepts.discover_concepts(self.catalog,'cashflow_op')
            self.assertEqual(seeds[0]['spec'],spec)
            self.assertTrue(seeds[0]['expand_defaults'])
            self.assertEqual(concepts.discover_concepts(self.catalog,'cashflow')[0]['tag'],spec['tag'])
            self.assertEqual(concepts.discover_concepts(self.catalog,'cashflow nonsense'),[])

    def test_public_registry_uses_concept_seeds_and_preserves_specs(self):
        from qlib_yahoo import sec_registry as registry
        spec = dict(tag='AssetsCurrent',taxonomy='us-gaap/',unit='USD',qtrs=0,segments='',coreg='')
        name = registry._name_for_recipe(spec)
        with closing(sqlite3.connect(self.database)) as db, db:
            db.executescript('''
                CREATE TABLE fields(id INTEGER PRIMARY KEY,name TEXT UNIQUE,spec TEXT UNIQUE,
                    label TEXT,kind TEXT,tag TEXT,description TEXT,taxonomy TEXT,common INTEGER,
                    unit TEXT,qtrs TEXT);
                CREATE INDEX fields_tag ON fields(tag);
                CREATE INDEX fields_kind_name ON fields(kind,common DESC,name);
                CREATE INDEX fields_common_name ON fields(common DESC,name);
                CREATE VIRTUAL TABLE fields_fts USING fts5(name,tag,label,description,
                    content='fields',content_rowid='id');
            ''')
            db.execute('INSERT INTO fields VALUES(1,?,?,?,?,?,?,?,?,?,?)',
                       (name,registry.source._canonical(spec).decode(),'AssetsCurrent','standard',
                        'AssetsCurrent','','us-gaap/',0,'USD','0'))
            db.execute("INSERT INTO fields_fts(fields_fts) VALUES('rebuild')")
        with patch.object(registry, '_ready', return_value=self.catalog), \
             patch.object(registry, 'dictionary_status', return_value={'field_count':1}), \
             patch.object(registry, '_db', side_effect=lambda _: sqlite3.connect(
                 self.database.as_uri()+'?mode=ro&immutable=1',uri=True)), \
             patch.object(registry.source, '_hash', side_effect=AssertionError('bulk hash')):
            out = registry.search_dictionary(self.catalog,'current assets')
        self.assertEqual(out['items'][0]['spec'],spec)
        self.assertEqual(out['items'][0]['kind'],'standard')
        self.assertEqual(out['total'],1)

    def test_empty_query_opens_nothing(self):
        with patch.object(concepts.sqlite3, 'connect', side_effect=AssertionError('opened')):
            self.assertEqual(self.discover('  '), [])

    def test_exact_reviewed_alias_pins_spec(self):
        spec = dict(tag='Assets', taxonomy='us-gaap/', unit='shares', qtrs=4, segments='', coreg='')
        seeds = self.discover('sec_assets_shares_4q', aliases={'sec_assets_shares_4q':spec})
        self.assertEqual(seeds, [dict(tag='Assets', spec=spec, rank=0, preferred=True, expand_defaults=False)])
        self.assertFalse(seeds.truncated)

    def test_alias_guess_uses_actual_source_acronym_without_losing_selectors(self):
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute('INSERT INTO tags VALUES(?,?,?)',
                       ('EPSExample','us-gaap/2024',json.dumps(dict(custom='0'))))
        guess = dict(tag='EpsExample',taxonomy='us-gaap/',unit='USD/shares',qtrs=4,segments='',coreg='')
        seeds = self.discover('sec_eps_example_4q__usd_per_share',
                              aliases={'sec_eps_example_4q__usd_per_share':guess})
        self.assertEqual(seeds[0]['tag'],'EPSExample')
        self.assertEqual(seeds[0]['spec'],dict(guess,tag='EPSExample'))
        self.assertFalse(seeds[0]['expand_defaults'])

    def test_lowercase_generic_parser_guess_uses_source_tag(self):
        from qlib_yahoo import sec_names
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute('INSERT INTO tags VALUES(?,?,?)',
                       ('EPSExample','us-gaap/2024',json.dumps(dict(custom='0'))))
        guess = dict(tag='EpsExample',taxonomy='us-gaap/',unit='USD/shares',qtrs=4,segments='',coreg='')
        query = 'sec_eps_example_4q__usd_per_share'
        with patch.object(sec_names, 'spec_for_name', side_effect=lambda name: guess if name == query else None):
            seeds = concepts.discover_concepts(self.catalog,query)
        self.assertEqual(seeds[0]['spec'],dict(guess,tag='EPSExample'))
        self.assertEqual(len(seeds),1)
        self.assertFalse(seeds[0]['expand_defaults'])

    def test_ordinary_generic_word_does_not_impose_us_gaap_taxonomy(self):
        from qlib_yahoo import sec_names
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute('INSERT INTO tags VALUES(?,?,?)',
                       ('Revenue','ifrs/2024',json.dumps(dict(custom='0'))))
        guess = dict(tag='Revenue',taxonomy='us-gaap/',unit='USD',qtrs=0,segments='',coreg='')
        with patch.object(sec_names,'curated_defs',return_value={}), \
             patch.object(sec_names,'spec_for_name', side_effect=lambda name: guess if name == 'sec_revenue' else None):
            seeds = concepts.discover_concepts(self.catalog,'revenue')
        revenue = [seed for seed in seeds if seed['tag'] == 'Revenue']
        self.assertEqual(revenue[0]['spec']['taxonomy'],'ifrs/')
        self.assertNotIn('preferred',revenue[0])

    def test_custom_unknown_never_receive_invented_family(self):
        for query in ('CustomOnly', 'Unknown'):
            seeds = self.discover(query)
            self.assertEqual(seeds, [dict(tag=query, rank=0)])

    def test_fixed_tag_limit_independent_of_recipe_paging(self):
        seeds = self.discover('earnings', max_tags=1)
        self.assertEqual(len(seeds), 1)
        self.assertTrue(seeds.truncated)

    def test_missing_index_does_not_scan(self):
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute('DROP INDEX tags_key')
        seeds = self.discover('assets')
        self.assertEqual(seeds, [])
        self.assertTrue(seeds.truncated)

    def test_prefix_and_exact_plans_are_index_ranges(self):
        with closing(sqlite3.connect(self.database)) as db, db:
            for sql, params in (
                ('SELECT DISTINCT tag FROM tags INDEXED BY tags_key WHERE tag=? COLLATE NOCASE LIMIT ?', ('Assets',129)),
                ("SELECT DISTINCT tag FROM tags INDEXED BY tags_key WHERE tag LIKE ? ESCAPE '\\' LIMIT ?", ('Earn%',129)),
                ('SELECT version,payload FROM tags INDEXED BY tags_key WHERE tag=? COLLATE NOCASE AND tag=? COLLATE BINARY AND version>=? ORDER BY version DESC LIMIT 129', ('Assets','Assets','A')),
            ):
                plan = str(db.execute('EXPLAIN QUERY PLAN '+sql, params).fetchall())
                self.assertIn('SEARCH tags USING', plan)
                self.assertIn('tags_key', plan)

    def test_metadata_work_interruption_preserves_partial_flag(self):
        with closing(sqlite3.connect(self.database)) as db, db:
            db.executemany('INSERT INTO tags VALUES(?,?,?)',
                           [('Assets',f'us-gaap/{i}',json.dumps(dict(custom='0'))) for i in range(2000)])
        with patch.object(concepts.time, 'monotonic', side_effect=[0]+[1]*10000):
            seeds = self.discover('assets', budget_ms=1)
        self.assertTrue(seeds.truncated)
        self.assertTrue(seeds.interrupted)
        with closing(sqlite3.connect(self.database)) as db, db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM tags').fetchone()[0], 2010)

    def test_invalid_arguments(self):
        for kwargs in ({'query':None}, {'query':'x'*4097}, {'query':'Assets','max_tags':129},
                       {'query':'Assets','budget_ms':0}):
            with self.assertRaises(ValueError):
                concepts.discover_concepts(self.catalog, aliases={}, **kwargs)


if __name__ == '__main__':
    unittest.main()
