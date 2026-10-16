"""Independent metadata-only SEC concept discovery and service contracts."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from qlib_yahoo.sec_catalog import read_catalog, search_fields, search_tags
from qlib_yahoo.webapp.server import AppServer
from qlib_yahoo.webapp.service import ConsoleService, ServiceError


class SecDiscoveryTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.catalog = self.root / 'sec' / 'catalogs' / 'fixture'
        self.catalog.mkdir(parents=True)
        self.database = self.catalog / 'catalog.sqlite'
        self.manifest_path = self.catalog / 'manifest.json'
        self.rows = []
        for year in range(1960, 2027):
            self.add('InventoryNet', f'us-gaap/{year}', '0', doc='needle' if year == 1960 else '')
        for number in range(65):
            self.add(f'InventoryLongPrefix{number:03}', 'us-gaap/2024', '0')
        for number in range(70):
            self.add('Inventory', f'0000123456-24-{number:06}', '1')
        self.add('InventoryNet', 'ifrs-full/2024', '0')
        self.add('InventoryNet', 'us-gaap/2024', '0', doc='duplicate definition')
        self.add('inventoryNet', 'us-gaap/2024', '0')
        self.add('UnlistedFamily', 'new-taxonomy/2023', '0')
        self.add('UnlistedFamily', 'new-taxonomy/2025', '0')
        self.add('OddSyntax', 'not valid/2024', '0')
        self.add('OddSyntax', 'not valid/2025', '0')
        self.add('OddSyntax', 'us-gaap/2024/extra', '0')
        self.add('MissingFlag', 'filing-one', None)
        self.rows[-1].pop('custom')
        self.add('NullFlag', 'filing-two', None)
        self.add('Literal_%', 'us-gaap/2024', '0')
        self.add('MetadataOnly', 'us-gaap/2024', '0', doc='Inventory label evidence')
        with closing(sqlite3.connect(self.database)) as db, db:
            db.executescript("""
                CREATE TABLE tags(tag TEXT,version TEXT,search TEXT,payload TEXT,occurrences INTEGER);
                CREATE INDEX tags_key ON tags(tag COLLATE NOCASE,version);
                CREATE VIRTUAL TABLE tags_fts USING fts5(search,content='tags',content_rowid='rowid');
            """)
            db.executemany('INSERT INTO tags VALUES(?,?,?,?,1)', [
                (row['tag'], row['version'], ' '.join(str(v) for v in row.values()).casefold(), json.dumps(row))
                for row in self.rows])
            db.execute("INSERT INTO tags_fts(tags_fts) VALUES('rebuild')")
        self.seal()
        self.service = ConsoleService(self.root, settings_path=self.root / 'settings.json')

    def add(self, tag, version, custom, doc=''):
        self.rows.append(dict(tag=tag, version=version, custom=custom, tlabel=tag + ' label',
                              doc=doc, datatype='monetary', iord='I'))

    def seal(self, revision=0):
        manifest = dict(schema_version=1, status='ready', tables={}, revision=revision,
                        numeric_materialization_supported=True,
                        artifacts_sha256={'catalog.sqlite': hashlib.sha256(self.database.read_bytes()).hexdigest()})
        canonical = json.dumps(manifest, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
        manifest['catalog_sha256'] = hashlib.sha256(canonical).hexdigest()
        self.manifest_path.write_text(json.dumps(manifest), encoding='utf-8')

    def test_global_grouping_before_pages_and_representative(self):
        first = search_fields(self.catalog, 'Inventory', limit=2)
        self.assertEqual(first['items'][0]['tag'], 'InventoryNet')
        self.assertEqual(first['items'][0]['taxonomy'], 'ifrs-full/')
        item = first['items'][1]
        self.assertEqual(item, dict(tag='InventoryNet', version='us-gaap/2026', custom='0',
                                   tlabel='InventoryNet label', doc='', datatype='monetary', iord='I',
                                   taxonomy='us-gaap/', versions=[f'us-gaap/{y}' for y in range(2026, 1959, -1)],
                                   kind='standard'))
        pages = [search_fields(self.catalog, 'Inventory', limit=2, offset=n)['items']
                 for n in range(0, first['total'], 2)]
        items = [item for page in pages for item in page]
        keys = [(i['tag'], i['taxonomy']) for i in items]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(items), first['total'])
        self.assertTrue(all(i['kind'] == 'standard' for i in items))
        self.assertEqual(items[-1]['tag'], 'MetadataOnly')
        # A match in an old definition still carries all releases and latest metadata.
        needle = search_fields(self.catalog, 'needle')['items'][0]
        self.assertEqual(needle['version'], 'us-gaap/2026')
        self.assertEqual(len(needle['versions']), 67)

    def test_custom_toggle_preserves_all_versions_after_standard_pages(self):
        standard = search_fields(self.catalog, 'Inventory')
        result = search_fields(self.catalog, 'Inventory', include_custom=True, limit=1000)
        self.assertEqual(result['total'], standard['total'] + 70)
        self.assertTrue(all(i['kind'] == 'standard' for i in result['items'][:standard['total']]))
        customs = result['items'][standard['total']:]
        self.assertEqual(len(customs), 70)
        self.assertEqual(len({i['taxonomy'] for i in customs}), 70)
        for item in customs:
            self.assertEqual(item['versions'], [item['version']])
            self.assertEqual(item['taxonomy'], item['version'])
        self.assertEqual(search_fields(self.catalog, 'InventoryNet')['items'][0]['tag'], 'InventoryNet')
        raw = search_tags(self.catalog, 'Inventory', limit=50)
        self.assertEqual(raw['items'][0]['tag'], 'Inventory')
        self.assertNotIn('kind', raw['items'][0])

    def test_unknown_custom_flags_and_unrestricted_standard_families(self):
        for tag in ('MissingFlag', 'NullFlag'):
            self.assertEqual(search_fields(self.catalog, tag)['total'], 0)
            item = search_fields(self.catalog, tag, include_custom=True)['items'][0]
            self.assertEqual(item['kind'], 'custom')
            self.assertEqual(item['versions'], [item['version']])
            self.assertEqual(item['taxonomy'], item['version'])
        item = search_fields(self.catalog, 'UnlistedFamily')['items'][0]
        self.assertEqual(item['taxonomy'], 'new-taxonomy/')
        self.assertEqual(item['versions'], ['new-taxonomy/2025', 'new-taxonomy/2023'])
        odd = search_fields(self.catalog, 'OddSyntax')
        self.assertEqual(odd['total'], 3)
        self.assertTrue(all(i['taxonomy'] == i['version'] for i in odd['items']))

    def test_metadata_only_and_byte_preservation(self):
        before = {p.name: p.read_bytes() for p in self.catalog.iterdir()}
        with patch('qlib_yahoo.sec_catalog._scan', side_effect=AssertionError('NUM scan')), \
             patch('qlib_yahoo.sec_catalog._hash', side_effect=AssertionError('bulk hash')), \
             patch('qlib_yahoo.sec_catalog.field_variants', side_effect=AssertionError('variants')):
            self.assertGreater(self.service.sec_fields(str(self.catalog), 'Inventory')['total'], 50)
            self.assertEqual(self.service.sec_fields(str(self.catalog), '  '), {'items': [], 'total': 0})
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.catalog.iterdir()})
        read_catalog(self.catalog, verify=True)

    def test_safe_parameters_and_literal_search(self):
        for query, offset, flag in [(None, 0, False), ([], 0, False), ('a' * 257, 0, False),
                                    ('x', True, False), ('x', '0', False), ('x', -1, False),
                                    ('x', 10_000_001, False), ('x', 0, 'false'), ('x', 0, 1)]:
            with self.subTest(query=query, offset=offset, flag=flag), self.assertRaises(ServiceError):
                self.service.sec_fields(str(self.catalog), query, offset, flag)
            with self.assertRaises(ValueError):
                search_fields(self.catalog, query, offset=offset, include_custom=flag)
        self.assertEqual(search_fields(self.catalog, 'Literal_%')['total'], 1)
        for query in ('%', '_', "' OR 1=1 --", '" OR *', '\\', '\x00'):
            result = search_fields(self.catalog, query, include_custom=True)
            self.assertIsInstance(result['total'], int)
        self.assertEqual(search_fields(self.catalog, 'Inventory', offset=10_000_000)['items'], [])
        for limit in (True, 0, 1001, '50'):
            with self.assertRaises(ValueError):
                search_fields(self.catalog, 'Inventory', limit=limit)

    def test_legacy_no_fts_keeps_sealed_bytes(self):
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute('DROP TABLE tags_fts')
        self.seal()
        before = self.database.read_bytes()
        result = search_fields(self.catalog, 'needle')
        self.assertEqual(result['items'][0]['version'], 'us-gaap/2026')
        self.assertEqual(search_fields(self.catalog, 'Literal_%')['total'], 1)
        self.assertEqual(before, self.database.read_bytes())

    def test_variant_cache_identity_copies_bounds_and_failures(self):
        with patch('qlib_yahoo.sec_catalog.field_variants', return_value=[{'uom': 'USD', 'nested': ['a']}]) as scan:
            first = self.service.sec_variants(str(self.catalog), 'InventoryNet', 'us-gaap/')
            first[0]['nested'].append('mutated')
            second = self.service.sec_variants(str(self.catalog), 'InventoryNet', 'us-gaap/')
            self.assertEqual(second[0]['nested'], ['a'])
            second.clear()
            self.assertEqual(len(self.service.sec_variants(str(self.catalog), 'InventoryNet', 'us-gaap/')), 1)
            self.assertEqual(scan.call_count, 1)
            self.seal(revision=1)
            self.service.sec_variants(str(self.catalog), 'InventoryNet', 'us-gaap/')
            self.assertEqual(scan.call_count, 2)
            for n in range(9):
                self.service.sec_variants(str(self.catalog), f'Tag{n}', 'us-gaap/')
            self.assertEqual(len(self.service._sec_variant_cache), 8)
            self.manifest_path.write_text('{}', encoding='utf-8')
            with self.assertRaises(ServiceError):
                self.service.sec_variants(str(self.catalog), 'Tag8', 'us-gaap/')
            self.seal(revision=2)
            scan.side_effect = ValueError('failed scan')
            for _ in range(2):
                with self.assertRaisesRegex(ValueError, 'failed scan'):
                    self.service.sec_variants(str(self.catalog), 'Failed', 'us-gaap/')
            scan.side_effect = None
            scan.return_value = [{'uom': 'USD'}] * 20_001
            count = scan.call_count
            for _ in range(2):
                self.service.sec_variants(str(self.catalog), 'Oversized', 'us-gaap/')
            self.assertEqual(scan.call_count, count + 2)
            self.assertFalse(any(key[2] == 'Oversized' for key in self.service._sec_variant_cache))

    def test_variant_scan_does_not_hold_cache_lock(self):
        def scan(*args):
            acquired = self.service._sec_variant_lock.acquire(blocking=False)
            self.assertTrue(acquired)
            if acquired:
                self.service._sec_variant_lock.release()
            return [{'uom': 'USD'}]
        with patch('qlib_yahoo.sec_catalog.field_variants', side_effect=scan):
            self.service.sec_variants(str(self.catalog), 'InventoryNet', 'us-gaap/')

    def test_native_unknown_alias_guidance_without_catalog_reads(self):
        with patch('qlib_yahoo.sec_catalog.read_catalog', side_effect=AssertionError('catalog read')):
            result = self.service.check_alpha('Mean($sec_actual_alias, 2)')
        self.assertFalse(result['ok'])
        for text in ('sec_actual_alias', 'Choose a SEC catalog', 'shared dictionary'):
            self.assertIn(text, result['message'])
        ordinary = self.service.check_alpha('$unknown')['message']
        self.assertNotIn('Add input', ordinary)

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'Lab optional numerical runtime')
    def test_lab_unknown_alias_guidance_without_catalog_reads(self):
        with patch('qlib_yahoo.sec_catalog.read_catalog', side_effect=AssertionError('catalog read')):
            result = self.service.alpha_check('rank(sec_actual_alias)')
        self.assertFalse(result['ok'])
        for text in ('sec_actual_alias', 'Choose a SEC catalog', 'shared dictionary'):
            self.assertIn(text, result['lines'][0]['message'])

    def test_http_strict_boolean_and_page_validation(self):
        server = AppServer(self.service)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def cleanup():
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.addCleanup(cleanup)

        def request(**params):
            values = dict(catalog=str(self.catalog), query='Inventory')
            values.update(params)
            query = urlencode(values)
            req = Request(server.origin + '/api/sec-fields?' + query,
                          headers={'Cookie': 'tqc=' + server.token})
            with urlopen(req, timeout=5) as response:
                return json.load(response)

        default = request()
        self.assertEqual(default, request(include_custom='false'))
        self.assertEqual(request(include_custom='true')['total'], default['total'] + 70)
        self.assertEqual(request(query=''), {'items': [], 'total': 0})
        for params in ({'include_custom': 'False'}, {'include_custom': '0'}, {'include_custom': ''},
                       {'offset': ''}, {'offset': '-1'}, {'offset': '1.0'}, {'offset': 'true'},
                       {'offset': '10000001'}, {'offset': ' 1'}, {'offset': '١'}, {'query': 'a' * 257}):
            with self.subTest(params=params), self.assertRaises(HTTPError) as exc:
                request(**params)
            self.assertEqual(exc.exception.code, 400)
        req = Request(server.origin + '/api/sec-fields?include_custom=true&include_custom=false',
                      headers={'Cookie': 'tqc=' + server.token})
        with self.assertRaises(HTTPError) as exc:
            urlopen(req, timeout=5)
        self.assertEqual(exc.exception.code, 400)


if __name__ == '__main__':
    unittest.main()
