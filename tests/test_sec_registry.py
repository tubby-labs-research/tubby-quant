"""Known-answer dictionary fixtures; no real datasets or daily materialization."""
import builtins
import hashlib
import json
import sqlite3
import subprocess
import sys
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

try:
    from . import test_sec_catalog as fixtures
except ImportError:
    import test_sec_catalog as fixtures

from qlib_yahoo import sec_catalog as catalog
from qlib_yahoo import sec_registry as registry


class DefinitionTests(unittest.TestCase):
    def test_recipe_lru_defaults_keep_128_mib_byte_bound(self):
        self.assertEqual(registry.RECIPE_CACHE_ENTRIES, 262144)
        self.assertEqual(registry.RECIPE_CACHE_BYTES, 128 * 1024 * 1024)

    def test_recipe_lru_hits_misses_and_entry_eviction(self):
        cache = registry._RecipeCache(2, 1024 * 1024)
        key = lambda tag: (tag, 'us-gaap/2012', 'USD', '0', '', '')
        for tag in ('A', 'B', 'A', 'C', 'B'):
            if not cache.contains(key(tag)):
                cache.remember(key(tag))
        self.assertEqual((cache.hits, cache.misses, cache.evictions), (1, 4, 2))
        self.assertEqual(list(cache.entries), [key('C'), key('B')])
        self.assertLessEqual(cache.bytes, cache.max_bytes)

    def test_recipe_lru_byte_eviction_oversized_bypass_and_disabled(self):
        key = lambda tag: (tag, 'us-gaap/2012', 'USD', '0', '', '')
        probe = registry._RecipeCache(10, 1024 * 1024)
        probe.remember(key('A'))
        budget = probe.bytes + 128
        cache = registry._RecipeCache(10, budget)
        cache.remember(key('A'))
        cache.remember(key('B'))
        self.assertFalse(cache.contains(key('A')))
        self.assertTrue(cache.contains(key('B')))
        self.assertEqual(cache.evictions, 1)
        before = dict(cache.entries)
        cache.remember(('Huge', 'us-gaap/2012', 'USD', '0', '\U0001f600' * budget, ''))
        self.assertEqual(dict(cache.entries), before)
        self.assertLessEqual(cache.bytes, budget)
        self.assertLessEqual(cache.peak_bytes, budget)
        for entries, size in ((0, budget), (10, 0)):
            disabled = registry._RecipeCache(entries, size)
            for _ in range(2):
                self.assertFalse(disabled.contains(key('A')))
                disabled.remember(key('A'))
            self.assertEqual((disabled.hits, disabled.misses, disabled.bytes), (0, 2, 0))

    def test_fixed_recipes_and_safe_deterministic_names(self):
        self.assertEqual(registry.COMMON_FIELDS['sec_inventory'], dict(
            tag='InventoryNet', taxonomy='us-gaap/', unit='USD', qtrs=0, segments='', coreg=''))
        self.assertEqual(registry.COMMON_FIELDS['sec_net_income_q']['qtrs'], 1)
        self.assertEqual(registry.COMMON_FIELDS['sec_net_income_4q']['qtrs'], 4)
        spec = dict(registry.COMMON_FIELDS['sec_assets'], unit='EUR')
        name = registry._name_for_recipe(spec)
        self.assertRegex(name, r'^sec_assets_us_gaap_eur_family_q0_[0-9a-f]{20}$')
        self.assertEqual(name, registry._name_for_recipe(dict(reversed(list(spec.items())))))
        self.assertNotEqual(name, registry._name_for_recipe(dict(spec, segments='Axis=One')))

    def test_module_import_has_no_numeric_dependencies(self):
        script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'pyarrow'}:
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from qlib_yahoo import sec_registry
assert sec_registry.COMMON_FIELDS['sec_inventory']['tag'] == 'InventoryNet'
"""
        result = subprocess.run([sys.executable, '-S', '-c', script],
                                cwd=Path(__file__).resolve().parent.parent,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipIf(fixtures.pq is None, 'pyarrow required')
class RegistryTests(unittest.TestCase):
    write_zip = fixtures.CatalogTests.write_zip
    build = fixtures.CatalogTests.build

    def setUp(self):
        fixtures.CatalogTests.setUp(self)
        self.output = self.root / 'catalogs' / 'catalog'

    def dictionary(self, **kwargs):
        self.build()
        return registry.build_dictionary(self.output, **kwargs)

    def items(self, **kwargs):
        return registry.search_dictionary(self.output, include_custom=True, **kwargs)['items']

    def add_numeric(self, rows, tags):
        self.tables['num.txt'] = fixtures.tsv(
            ['adsh', 'tag', 'version', 'qtrs', 'uom', 'segments', 'coreg', 'value'], rows)
        self.tables['tag.txt'] = fixtures.tsv(['tag', 'version', 'custom', 'tlabel', 'doc'], tags)
        self.write_zip()

    def reseal(self):
        manifest = json.loads((self.output / 'manifest.json').read_text())
        for relative in manifest['artifacts_sha256']:
            manifest['artifacts_sha256'][relative] = catalog._hash(self.output / relative)
        manifest.pop('catalog_sha256')
        manifest['catalog_sha256'] = hashlib.sha256(catalog._canonical(manifest)).hexdigest()
        (self.output / 'manifest.json').write_bytes(catalog._canonical(manifest))

    def test_status_hash_provenance_immutable_verify_and_reuse(self):
        self.build()
        missing = registry.dictionary_status(self.output)
        self.assertEqual(missing['status'], 'missing')
        self.assertIsNone(missing['dictionary_sha256'])
        self.assertEqual(Path(missing['path']), self.output.parent.parent / 'dictionaries' /
                         (missing['catalog_sha256'] + '-' + registry.REGISTRY_VERSION))
        raw_before = {p: p.read_bytes() for p in self.output.rglob('*') if p.is_file()}
        result = registry.build_dictionary(self.output)
        path = Path(result['path'])
        self.addCleanup(lambda: self.cleanup_sidecar(path))
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in path.iterdir()}
        self.assertEqual(result, registry.verify_dictionary(self.output))
        self.assertEqual(result, registry.dictionary_status(self.output))
        self.assertEqual(result, registry.build_dictionary(self.output))
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in path.iterdir()})
        self.assertEqual(raw_before, {p: p.read_bytes() for p in self.output.rglob('*') if p.is_file()})
        self.assertEqual(result['dictionary_sha256'], hashlib.sha256((path / 'dictionary.sqlite').read_bytes()).hexdigest())
        self.assertEqual(result['field_count'], 4)  # standard exact+family, two exact custom/unknown

    def cleanup_sidecar(self, path):
        # Default fixture root places dictionaries alongside the temporary root.
        # Delete only the precisely recorded dictionary created by this fixture.
        import shutil
        if path.exists():
            shutil.rmtree(path)

    def test_metadata_calls_never_hash_scan_or_import_numeric_modules(self):
        result = self.dictionary()
        self.addCleanup(lambda: self.cleanup_sidecar(Path(result['path'])))
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'numpy', 'pyarrow'}:
                raise AssertionError(name)
            return original(name, *args, **kwargs)
        with patch.object(builtins, '__import__', side_effect=guarded), \
                patch.object(catalog, '_hash', side_effect=AssertionError('bulk hash')), \
                patch.object(catalog, '_scan', side_effect=AssertionError('NUM scan')):
            self.assertEqual(registry.dictionary_status(self.output), result)
            self.assertEqual(registry.search_dictionary(self.output, 'assets')['total'], 2)
            entry = registry.lookup_fields(self.output, ['sec_assets', 'sec_inventory', 'unknown'])
            self.assertEqual(set(entry), {'sec_assets'})
            self.assertEqual(registry.name_for_spec(self.output, registry.COMMON_FIELDS['sec_assets']), entry['sec_assets'])

    def test_common_only_when_exact_recipe_observed_no_fallback(self):
        self.add_numeric([
            ['a', 'Inventory', 'us-gaap/2024', '0', 'USD', '', '', '1'],
            ['a', 'InventoryNet', 'us-gaap/2024', '0', 'EUR', '', '', '2'],
            ['a', 'Assets', 'us-gaap/2024', '1', 'USD', '', '', '3']], [
            ['Inventory', 'us-gaap/2024', '0', 'Inventories', ''],
            ['InventoryNet', 'us-gaap/2024', '0', 'Net inventories', ''],
            ['Assets', 'us-gaap/2024', '0', 'Assets', '']])
        result = self.dictionary()
        self.addCleanup(lambda: self.cleanup_sidecar(Path(result['path'])))
        self.assertEqual(registry.lookup_fields(self.output, registry.COMMON_FIELDS), {})
        with self.assertRaisesRegex(ValueError, 'not observed'):
            registry.name_for_spec(self.output, registry.COMMON_FIELDS['sec_inventory'])
        self.assertEqual(len(self.items()), 6)

    def test_all_units_durations_contexts_custom_and_exact_editions(self):
        rows = [['a', 'InventoryNet', version, qtrs, unit, segment, coreg, value]
                for version, qtrs, unit, segment, coreg, value in [
                    ('us-gaap/2023', '0', 'USD', '', '', 'bad'),
                    ('us-gaap/2024', '0', 'USD', '', '', ''),
                    ('us-gaap/2024', '1', 'USD', '', '', '5'),
                    ('us-gaap/2024', '4', 'EUR', 'Axis=Thing', 'Sub', '6'),
                    ('0000000001-24-000001', '0', 'USD', '', '', '7')]]
        tags = [['InventoryNet', version, custom, 'Inventory label', 'Merchandise holdings']
                for version, custom in [('us-gaap/2023', '0'), ('us-gaap/2024', '0'),
                                        ('0000000001-24-000001', '1')]]
        self.add_numeric(rows, tags)
        result = self.dictionary()
        self.addCleanup(lambda: self.cleanup_sidecar(Path(result['path'])))
        self.assertEqual(result['field_count'], 8)
        self.assertEqual(registry.search_dictionary(self.output)['total'], 7)
        self.assertEqual(registry.search_dictionary(self.output, 'Merchandise', include_custom=True)['total'], 8)
        self.assertEqual(registry.lookup_fields(self.output, ['sec_inventory'])['sec_inventory']['spec'],
                         registry.COMMON_FIELDS['sec_inventory'])
        custom = registry.name_for_spec(self.output, dict(registry.COMMON_FIELDS['sec_inventory'],
                                                        taxonomy='0000000001-24-000001'))
        self.assertEqual(custom['kind'], 'custom')
        self.assertIn('0000000001_24_000001', custom['name'])
        with self.assertRaises(ValueError):
            registry.name_for_spec(self.output, dict(custom['spec'], taxonomy='0000000001-24-000001/'))

    def test_unknown_conflicting_custom_metadata_blocks_family_namespace(self):
        self.add_numeric([
            ['a', 'Assets', 'us-gaap/2023', '0', 'USD', '', '', '1'],
            ['a', 'Assets', 'us-gaap/2024', '0', 'USD', '', '', '1'],
            ['a', 'Other', 'new-family/2024', '0', 'USD', '', '', '1']], [
            ['Assets', 'us-gaap/2023', '0', 'Assets', ''],
            ['Assets', 'us-gaap/2024', '0', 'Assets', ''],
            ['Assets', 'us-gaap/2024', '1', 'Conflicting custom', '']])
        result = self.dictionary()
        self.addCleanup(lambda: self.cleanup_sidecar(Path(result['path'])))
        self.assertEqual(result['field_count'], 3)
        self.assertEqual(registry.search_dictionary(self.output)['total'], 1)
        self.assertEqual(registry.lookup_fields(self.output, ['sec_assets']), {})
        self.assertFalse(any(row['spec']['taxonomy'].endswith('/') for row in self.items()))

    def test_schema_evolution_missing_context_and_invalid_qtrs_never_blank(self):
        self.build()
        import pyarrow as pa
        import pyarrow.parquet as pq
        manifest = catalog.read_catalog(self.output)
        target = self.output / manifest['tables']['num']['files'][0]
        rows = [dict(tag='Assets', version='us-gaap/2012', uom='USD', qtrs=qtrs,
                     segments=segment, coreg=coreg) for qtrs, segment, coreg in [
                         ('0', None, ''), ('0', '', None), ('0', None, None),
                         ('-1', '', ''), ('1.0', '', ''), ('00', '', ''), ('', '', ''),
                         ('9999999999', '', ''), ('2', 'Axis=Two', '')]]
        pq.write_table(pa.Table.from_pylist(rows), target)
        self.reseal()
        result = registry.build_dictionary(self.output)
        self.addCleanup(lambda: self.cleanup_sidecar(Path(result['path'])))
        self.assertEqual(result['field_count'], 4)
        self.assertEqual(registry.lookup_fields(self.output, ['sec_assets']), {})
        seal = json.loads((Path(result['path']) / 'manifest.json').read_text())
        self.assertEqual(seal['diagnostics']['invalid_context_keys'], 3)
        self.assertEqual(seal['diagnostics']['invalid_selector_keys'], 4)

    def test_missing_columns_in_one_fragment_not_coalesced(self):
        other = self.root / 'other.zip'
        with zipfile.ZipFile(other, 'w') as stream:
            stream.writestr('num.txt', fixtures.tsv(['adsh', 'tag', 'version', 'qtrs', 'uom'],
                            [['a', 'Ghost', 'us-gaap/2012', '0', 'USD']]))
        catalog.build_catalog([self.archive, other], self.output, ticker_map=self.ticker)
        result = registry.build_dictionary(self.output)
        self.addCleanup(lambda: self.cleanup_sidecar(Path(result['path'])))
        self.assertFalse(any(item['spec']['tag'] == 'Ghost' for item in self.items()))

    def test_missing_context_blocks_every_matching_exact_and_family_recipe(self):
        # The valid 2013 edition survives; its all-editions recipe must not.
        # Nonmatching units/durations/tags must remain usable.
        rows = [
            ['a', 'Assets', 'us-gaap/2012', '0', 'USD', '', '', '1'],
            ['a', 'Assets', 'us-gaap/2012', '0', 'USD', 'Axis=A', 'Subsidiary', '2'],
            ['a', 'Assets', 'us-gaap/2013', '0', 'USD', '', '', '3'],
            ['a', 'Assets', 'us-gaap/2012', '0', 'EUR', '', '', '4'],
            ['a', 'Assets', 'us-gaap/2012', '1', 'USD', '', '', '5'],
            ['a', 'InventoryNet', 'us-gaap/2012', '0', 'USD', '', '', '6']]
        tags = [['Assets', 'us-gaap/2012', '0', 'Assets', ''],
                ['Assets', 'us-gaap/2013', '0', 'Assets', ''],
                ['InventoryNet', 'us-gaap/2012', '0', 'Inventory', '']]
        self.add_numeric(rows, tags)
        for missing in ('segments', 'coreg'):
            for reverse in (False, True):
                with self.subTest(missing=missing, reverse=reverse):
                    self.output = self.root / 'catalogs' / f'{missing}-{reverse}'
                    other = self.root / f'{missing}.zip'
                    present = 'coreg' if missing == 'segments' else 'segments'
                    with zipfile.ZipFile(other, 'w') as stream:
                        stream.writestr('num.txt', fixtures.tsv(
                            ['adsh', 'tag', 'version', 'qtrs', 'uom', present, 'value'],
                            [['a', 'Assets', 'us-gaap/2012', '0', 'USD', 'Different context', '7']]))
                    archives = [other, self.archive] if reverse else [self.archive, other]
                    catalog.build_catalog(archives, self.output, ticker_map=self.ticker)
                    blocked = [dict(registry.COMMON_FIELDS['sec_assets'], taxonomy=taxonomy, **context)
                               for taxonomy in ('us-gaap/', 'us-gaap/2012')
                               for context in ({}, {'segments': 'Axis=A', 'coreg': 'Subsidiary'})]
                    for spec in blocked:
                        with self.assertRaisesRegex(ValueError, 'structural context column'):
                            list(catalog.iter_facts(self.output, spec))
                    result = registry.build_dictionary(self.output)
                    self.assertEqual(registry.lookup_fields(self.output, ['sec_assets']), {})
                    for spec in blocked:
                        with self.assertRaisesRegex(ValueError, 'not observed/supported'):
                            registry.name_for_spec(self.output, spec)
                    self.assertEqual(result['field_count'], 7)
                    for item in self.items():
                        self.assertTrue(list(catalog.iter_facts(self.output, item['spec'])))
                    seal = json.loads((Path(result['path']) / 'manifest.json').read_text())
                    self.assertEqual(seal['diagnostics']['blocked_selector_prefixes'], 2)
                    self.assertEqual(seal['diagnostics']['blocked_context_recipes'], 4)

    def test_actual_nulls_block_custom_exact_and_unusual_family_prefix(self):
        self.add_numeric([
            ['a', 'Assets', 'us-gaap/2012', '0', 'USD', '', '', '1'],
            ['a', 'Other', 'a', '0', 'USD', '', '', '2']], [
            ['Assets', 'us-gaap/2012', '0', 'Assets', ''], ['Other', 'a', '1', 'Other', '']])
        self.build()
        import pyarrow as pa
        import pyarrow.parquet as pq
        path = self.output / catalog.read_catalog(self.output)['tables']['num']['files'][0]
        rows = pq.read_table(path).to_pylist()
        rows += [dict(rows[0], version='us-gaap/nested/edition', segments=None, coreg='Unmatched'),
                 dict(rows[1], coreg=None, segments='Unmatched')]
        pq.write_table(pa.Table.from_pylist(rows), path)
        self.reseal()
        for spec in (registry.COMMON_FIELDS['sec_assets'],
                     dict(registry.COMMON_FIELDS['sec_assets'], tag='Other', taxonomy='a')):
            with self.assertRaisesRegex(ValueError, 'structural context column'):
                list(catalog.iter_facts(self.output, spec))
        result = registry.build_dictionary(self.output)
        self.assertEqual(result['field_count'], 1)
        remaining = self.items()[0]['spec']
        self.assertEqual((remaining['tag'], remaining['taxonomy']), ('Assets', 'us-gaap/2012'))
        self.assertTrue(list(catalog.iter_facts(self.output, remaining)))

    def test_grouped_tag_ingestion_consensus_labels_and_readonly_attachment(self):
        self.add_numeric([
            ['a', 'Assets', 'us-gaap/2012', '0', 'USD', '', '', '1'],
            ['a', 'Other', 'a', '0', 'USD', '', '', '2']], [
            ['Assets', 'us-gaap/2012', '0', 'Zulu assets', 'Zebra description'],
            ['Assets', 'us-gaap/2012', '0', 'Alpha assets', 'Aardvark description'],
            ['Other', 'a', '0', 'Z custom', 'Z doc'],
            ['Other', 'a', '1', 'A custom', 'A doc']])
        self.build()
        original_connect = sqlite3.connect
        checked_attachment = []
        statements = []

        class InspectConnection(sqlite3.Connection):
            def execute(self, sql, parameters=(), /):
                if sql.startswith('INSERT INTO tag_metadata'):
                    for pragma, expected in (('main.cache_size', -65536), ('temp.cache_size', -65536),
                                             ('main.journal_mode', 'delete'), ('main.synchronous', 2),
                                             ('temp_store', 1)):
                        self_test.assertEqual(super().execute('PRAGMA ' + pragma).fetchone()[0], expected)
                    indexes = super().execute("PRAGMA main.index_list('fields')").fetchall()
                    self_test.assertEqual(len(indexes), 2)
                    self_test.assertTrue(all(row[2] == 1 for row in indexes))
                    self_test.assertEqual({super(InspectConnection, self).execute(
                        f"PRAGMA main.index_info('{row[1]}')").fetchone()[2] for row in indexes}, {'name', 'spec'})
                    with self_test.assertRaisesRegex(sqlite3.OperationalError, 'readonly'):
                        super().execute("UPDATE source_catalog.tags SET tag='corruption'")
                    checked_attachment.append(True)
                return super().execute(sql, parameters)

        self_test = self
        def connect(*args, **kwargs):
            connection = original_connect(*args, **kwargs, factory=InspectConnection)
            connection.set_trace_callback(statements.append)
            return connection

        raw_before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.output.rglob('*') if p.is_file()}
        with patch.object(registry.sqlite3, 'connect', side_effect=connect):
            result = registry.build_dictionary(self.output)
        self.assertEqual(checked_attachment, [True])
        self.assertEqual(sum(sql.startswith('INSERT INTO tag_metadata') for sql in statements), 1)
        self.assertTrue(any(sql.startswith('ATTACH DATABASE') and 'mode=ro&immutable=1' in sql for sql in statements))
        self.assertEqual(raw_before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.output.rglob('*') if p.is_file()})
        self.assertEqual(registry.lookup_fields(self.output, ['sec_assets'])['sec_assets']['label'], 'Alpha assets')
        custom = registry.name_for_spec(self.output, dict(registry.COMMON_FIELDS['sec_assets'], tag='Other', taxonomy='a'))
        self.assertEqual((custom['kind'], custom['label']), ('custom', 'A custom'))
        self.assertEqual(registry.search_dictionary(self.output, 'aardvark')['total'], 2)
        self.assertEqual(result['field_count'], 3)

    def test_deferred_indexes_exist_before_seal_and_match_previous_build_entries(self):
        import shutil
        self.add_numeric([
            ['a', 'Assets', 'us-gaap/2012', '0', 'USD', '', '', '1'],
            ['a', 'Assets', 'us-gaap/2013', '0', 'USD', '', '', '2'],
            ['a', 'Assets', 'us-gaap/2013', '1', 'EUR', 'Axis=One', 'Subsidiary', '3'],
            ['a', 'Other', 'a', '0', 'USD', '', '', '4']], [
            ['Assets', 'us-gaap/2012', '0', 'Zulu assets', 'Zebra description'],
            ['Assets', 'us-gaap/2013', '0', 'Alpha assets', 'Aardvark description'],
            ['Other', 'a', '1', 'Custom other', 'Custom description']])
        self.build()
        baseline_catalog = self.root / 'baseline' / 'catalogs' / 'catalog'
        shutil.copytree(self.output, baseline_catalog)
        index_sql = (
            'CREATE INDEX fields_kind_name ON fields(kind,common DESC,name);',
            'CREATE INDEX fields_common_name ON fields(common DESC,name);',
            'CREATE INDEX fields_tag ON fields(tag);')
        expected_indexes = {sql.split()[2] for sql in index_sql}
        original_connect = sqlite3.connect
        sealed = []

        def inspect_sealing(message):
            if message != 'Verifying source stability and sealing dictionary':
                return
            parent = Path(registry.dictionary_status(self.output)['path']).parent
            databases = list(parent.glob('sec-dict-*/dictionary.sqlite'))
            self.assertEqual(len(databases), 1)
            with registry.closing(original_connect(databases[0].as_uri() + '?mode=ro', uri=True)) as connection:
                indexes = connection.execute("PRAGMA index_list('fields')").fetchall()
                self.assertEqual({row[1] for row in indexes if not row[2]}, expected_indexes)
                self.assertEqual(sum(row[2] for row in indexes), 2)
                self.assertEqual(connection.execute('PRAGMA integrity_check').fetchall(), [('ok',)])
                self.assertGreater(connection.execute('SELECT COUNT(*) FROM fields_fts').fetchone()[0], 0)
            sealed.append(True)

        optimized = registry.build_dictionary(self.output, progress=inspect_sealing)
        self.assertEqual(sealed, [True])

        class PreviousSettingsConnection(sqlite3.Connection):
            def executescript(self, sql, /):
                # Reproduce the prior cache defaults and early secondary indexes
                # only in this synthetic reference build, without runtime switches.
                if 'CREATE TABLE fields(' in sql:
                    result = super().executescript(sql)
                    super().executescript('PRAGMA main.cache_size=-2000; PRAGMA temp.cache_size=0;'
                                          + ''.join(index_sql))
                    return result
                for statement in index_sql:
                    sql = sql.replace(statement, '')
                return super().executescript(sql)

        def previous_connect(*args, **kwargs):
            return original_connect(*args, **kwargs, factory=PreviousSettingsConnection)

        with patch.object(registry.sqlite3, 'connect', side_effect=previous_connect):
            baseline = registry.build_dictionary(baseline_catalog)
        snapshots = []
        for result in (optimized, baseline):
            self.assertEqual(result['version'], 'sec-dictionary-2')
            with registry.closing(registry._db(Path(result['path']))) as connection:
                snapshots.append((
                    connection.execute('SELECT * FROM fields ORDER BY name').fetchall(),
                    connection.execute('SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name').fetchall()))
        self.assertEqual(snapshots[0], snapshots[1])
        self.assertEqual(registry.search_dictionary(self.output, 'assets', include_custom=True),
                         registry.search_dictionary(baseline_catalog, 'assets', include_custom=True))

    def test_malformed_tag_metadata_fails_clearly_even_with_valid_group_member(self):
        self.build()
        valid = dict(tag='Assets', version='us-gaap/2012', custom='0', tlabel='Assets', doc='')
        malformed = ['{broken', '[]', 'null', json.dumps(dict(valid, tag='Different')),
                     json.dumps(dict(valid, custom={})), json.dumps(dict(valid, tlabel=[])),
                     json.dumps(dict(valid, doc=17)), json.dumps(dict(valid, version=None))]
        database = self.output / 'catalog.sqlite'
        for payload in malformed:
            with self.subTest(payload=payload):
                with registry.closing(sqlite3.connect(database)) as connection:
                    connection.execute('DELETE FROM tags')
                    connection.executemany('INSERT INTO tags VALUES(?,?,?,?,?)', [
                        ('Assets', 'us-gaap/2012', 'assets', json.dumps(valid), 1),
                        ('Assets', 'us-gaap/2012', 'bad', payload, 1)])
                    connection.commit()
                self.reseal()
                path = Path(registry.dictionary_status(self.output)['path'])
                with self.assertRaisesRegex(ValueError, 'Malformed or unsupported TAG metadata'):
                    registry.build_dictionary(self.output)
                self.assertFalse(path.exists())

    def test_prior_staging_and_sidecars_preserved(self):
        self.build()
        destination = Path(registry.dictionary_status(self.output)['path'])
        destination.parent.mkdir(parents=True)
        old_staging = destination.parent / 'sec-dict-previous-attempt'
        old_staging.mkdir()
        (old_staging / 'dictionary.sqlite').write_bytes(b'preserved partial build')
        old_receipt = destination.parent / 'sec-dict-previous-attempt.error.json'
        old_receipt.write_bytes(b'{"status":"failed"}')
        old_sidecar = destination.parent / (catalog.read_catalog(self.output)['catalog_sha256'] + '-sec-dictionary-1')
        old_sidecar.mkdir()
        (old_sidecar / 'manifest.json').write_bytes(b'preserve old version')
        prior = {p: p.read_bytes() for p in destination.parent.rglob('*') if p.is_file()}
        result = registry.build_dictionary(self.output)
        self.assertEqual(result['version'], 'sec-dictionary-2')
        self.assertEqual(prior, {p: p.read_bytes() for p in prior})

    def test_notes_and_wholly_missing_context_are_unsupported(self):
        self.tables['txt.tsv'] = fixtures.tsv(['adsh', 'text'], [['a', 'raw']])
        self.write_zip()
        self.build()
        self.assertEqual(registry.dictionary_status(self.output)['status'], 'unsupported')
        with self.assertRaisesRegex(ValueError, 'numeric adapter'):
            registry.build_dictionary(self.output)

    def test_paging_fts_literal_input_and_parameter_bounds(self):
        result = self.dictionary()
        self.addCleanup(lambda: self.cleanup_sidecar(Path(result['path'])))
        all_items = self.items()
        paged = [self.items(limit=1, offset=i)[0] for i in range(len(all_items))]
        self.assertEqual(all_items, paged)
        self.assertEqual(registry.search_dictionary(self.output, 'balance sheet')['total'], 2)
        self.assertEqual(registry.search_dictionary(self.output, 'sec_assets')['items'][0]['name'], 'sec_assets')
        self.assertEqual(registry.search_dictionary(self.output, "'; DROP TABLE fields; --")['total'], 0)
        self.assertEqual(registry.search_dictionary(self.output, '***')['total'], 0)
        for kwargs in ({'limit': True}, {'limit': 0}, {'offset': -1}, {'include_custom': 1}, {'query': None}):
            with self.assertRaises(ValueError):
                registry.search_dictionary(self.output, **kwargs)
        with self.assertRaises(ValueError):
            registry.lookup_fields(self.output, 'sec_assets')
        for change in ({'qtrs': True}, {'segments': None}, {'unknown': 1}):
            with self.assertRaises(ValueError):
                registry.name_for_spec(self.output, dict(registry.COMMON_FIELDS['sec_assets'], **change))

    def test_batch_group_distinct_and_names_independent_of_order(self):
        self.add_numeric([['a', 'Assets', 'us-gaap/2024', '0', 'EUR', '', '', '1']] * 200,
                         [['Assets', 'us-gaap/2024', '0', 'Total assets', '']])
        with patch.object(registry, 'BATCH_SIZE', 64):
            result = self.dictionary()
        self.addCleanup(lambda: self.cleanup_sidecar(Path(result['path'])))
        seal = json.loads((Path(result['path']) / 'manifest.json').read_text())
        self.assertEqual(seal['diagnostics']['distinct_batch_keys'], 4)
        self.assertEqual(result['field_count'], 2)
        for row in self.items():
            self.assertEqual(row.get('storage_name', row['name']), registry._name_for_recipe(row['spec']))
            self.assertEqual(registry.lookup_fields(self.output, [row['name']])[row['name']]['spec'], row['spec'])

    def test_recipe_cache_matches_disabled_full_entries_across_editions_and_eviction(self):
        import shutil
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.tables['tag.txt'] = fixtures.tsv(['tag', 'version', 'custom', 'tlabel', 'doc'], [
            ['Assets', 'us-gaap/2012', '0', 'Zulu assets', 'Zebra description'],
            ['Assets', 'us-gaap/2013', '0', 'Alpha assets', 'Aardvark description'],
            ['Other', 'new-family/2012', '0', 'Standard other', 'Standard description'],
            ['Other', 'new-family/2013', '1', 'Custom other', 'Custom description']])
        self.write_zip()
        self.build()
        base_catalog = self.output
        target = self.output / catalog.read_catalog(self.output)['tables']['num']['files'][0]
        template = pq.read_table(target).to_pylist()[0]
        base = dict(template, tag='Assets', version='us-gaap/2012', uom='USD', qtrs='0', segments='', coreg='')
        recipes = [base, dict(base, version='us-gaap/2013'), dict(base, qtrs='1'),
                   dict(base, uom='EUR'), dict(base, segments='Axis=One'), dict(base, coreg='Subsidiary'),
                   dict(base, tag='Other', version='new-family/2012'),
                   dict(base, tag='Other', version='new-family/2013'), dict(base, qtrs='3')]
        # Batch size one gives a known stream of nine misses, then twelve hits.
        rows = [row for recipe in recipes for row in (recipe, recipe)] + [recipes[0], recipes[1], recipes[0]]
        rows += [dict(base, qtrs='3', segments=None)] * 2 + [dict(base, qtrs='-1')] * 2
        pq.write_table(pa.Table.from_pylist(rows), target)
        self.reseal()
        snapshots, stats, insert_counts = {}, {}, {}
        for mode, capacity in (('off', 0), ('on', 131072), ('evict', 1)):
            self.output = self.root / mode / 'catalogs' / 'catalog'
            shutil.copytree(base_catalog, self.output)
            progress = []
            with patch.object(registry, 'BATCH_SIZE', 1), \
                    patch.object(registry, 'RECIPE_CACHE_ENTRIES', capacity), \
                    patch.object(registry, '_insert', wraps=registry._insert) as inserts:
                result = registry.build_dictionary(self.output, progress=progress.append)
            path = Path(result['path'])
            with registry.closing(registry._db(path)) as connection:
                snapshots[mode] = connection.execute('SELECT * FROM fields ORDER BY name').fetchall()
            stats[mode] = json.loads((path / 'manifest.json').read_text())['diagnostics']
            insert_counts[mode] = inserts.call_count
            common = registry.lookup_fields(self.output, ['sec_assets'])['sec_assets']
            self.assertEqual(common['label'], 'Alpha assets')
            self.assertEqual(registry.search_dictionary(self.output, 'aardvark')['total'], 2)
            with self.assertRaises(ValueError):
                registry.name_for_spec(self.output, dict(registry.COMMON_FIELDS['sec_assets'], qtrs=3))
            with self.assertRaises(ValueError):
                registry.name_for_spec(self.output, dict(registry.COMMON_FIELDS['sec_assets'],
                                                        tag='Other', taxonomy='new-family/'))
            self.assertTrue(any('recipe cache' in message and 'hits /' in message for message in progress))
            self.assertEqual(stats[mode]['distinct_batch_keys'], 25)
            self.assertEqual(stats[mode]['invalid_context_keys'], 2)
            self.assertEqual(stats[mode]['invalid_selector_keys'], 2)
        self.assertEqual(snapshots['off'], snapshots['on'])
        self.assertEqual(snapshots['off'], snapshots['evict'])
        self.assertEqual((stats['on']['recipe_cache_hits'], stats['on']['recipe_cache_misses']), (12, 9))
        self.assertEqual((stats['evict']['recipe_cache_hits'], stats['evict']['recipe_cache_misses'],
                          stats['evict']['recipe_cache_evictions']), (9, 12, 11))
        self.assertEqual((stats['off']['recipe_cache_hits'], stats['off']['recipe_cache_misses']), (0, 21))
        self.assertLess(insert_counts['on'], insert_counts['off'])
        self.assertLess(insert_counts['evict'], insert_counts['off'])

    def test_hash_collision_is_fatal_retains_receipt_and_cleans_staging(self):
        self.build()
        path = Path(registry.dictionary_status(self.output)['path'])
        with patch.object(registry, '_name_for_recipe', return_value='sec_collision'):
            with self.assertRaisesRegex(ValueError, 'collision'):
                registry.build_dictionary(self.output)
        self.assertFalse(path.exists())
        receipts = list(path.parent.glob('sec-dict-*.error.json'))
        matching = [p for p in receipts if json.loads(p.read_text())['catalog_sha256'] == catalog.read_catalog(self.output)['catalog_sha256']]
        self.assertTrue(matching)
        for receipt in matching:
            self.addCleanup(receipt.unlink)
        self.assertFalse(any(path.parent.glob('sec-dict-*/dictionary.sqlite')))

    def test_cancel_and_source_tamper_leave_no_ready_dictionary(self):
        self.build()
        path = Path(registry.dictionary_status(self.output)['path'])
        def cancel(message):
            if 'Publishing' in message:
                raise KeyboardInterrupt('cancel fixture')
        with self.assertRaises(KeyboardInterrupt):
            registry.build_dictionary(self.output, progress=cancel)
        self.assertFalse(path.exists())
        for receipt in path.parent.glob('sec-dict-*.error.json'):
            if json.loads(receipt.read_text())['catalog_sha256'] == catalog.read_catalog(self.output)['catalog_sha256']:
                self.addCleanup(lambda p=receipt: p.unlink(missing_ok=True))
        def mutate(message):
            if 'NUM file' in message:
                (self.output / 'tickers.json').write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, 'artifact differs'):
            registry.build_dictionary(self.output, progress=mutate)
        self.assertFalse(path.exists())
        for receipt in path.parent.glob('sec-dict-*.error.json'):
            if json.loads(receipt.read_text())['catalog_sha256'] == catalog.read_catalog(self.output)['catalog_sha256']:
                self.addCleanup(lambda p=receipt: p.unlink(missing_ok=True))

    def test_tampered_dictionary_and_incompatible_seal_never_overwritten(self):
        result = self.dictionary()
        path = Path(result['path'])
        self.addCleanup(lambda: self.cleanup_sidecar(path))
        database = path / 'dictionary.sqlite'
        with database.open('ab') as stream:
            stream.write(b'tamper')
        self.assertEqual(registry.dictionary_status(self.output), result)
        for function in (registry.verify_dictionary, registry.build_dictionary):
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                function(self.output)
        seal_path = path / 'manifest.json'
        seal = json.loads(seal_path.read_text())
        seal['version'] = 'unsupported'
        seal_path.write_text(json.dumps(seal))
        self.assertEqual(registry.dictionary_status(self.output)['status'], 'missing')
        with self.assertRaises(ValueError):
            registry.build_dictionary(self.output)

    def test_worker_verifies_dictionary_only_and_detects_mutation_during_hash(self):
        result = self.dictionary()
        path = Path(result['path'])
        self.addCleanup(lambda: self.cleanup_sidecar(path))
        original_hash = catalog._hash
        def guarded_hash(file):
            self.assertEqual(Path(file), path / 'dictionary.sqlite')
            return original_hash(file)
        with patch.object(catalog, '_hash', side_effect=guarded_hash):
            self.assertEqual(registry.verify_dictionary(self.output), result)
        def change_after_hash(file):
            digest = original_hash(file)
            with Path(file).open('ab') as stream:
                stream.write(b'mutated after verification hash')
            return digest
        with patch.object(catalog, '_hash', side_effect=change_after_hash):
            with self.assertRaisesRegex(ValueError, 'changed during verification'):
                registry.verify_dictionary(self.output)

    def test_concurrent_publish_reuses_verified_winner(self):
        self.build()
        winner = []
        def publish_other(message):
            if 'Publishing' in message and not winner:
                winner.append(registry.build_dictionary(self.output))
        result = registry.build_dictionary(self.output, progress=publish_other)
        self.addCleanup(lambda: self.cleanup_sidecar(Path(result['path'])))
        self.assertEqual(result, winner[0])

    def test_indexed_name_lookup_and_nonsearch_paging(self):
        result = self.dictionary()
        path = Path(result['path'])
        self.addCleanup(lambda: self.cleanup_sidecar(path))
        with registry.closing(registry._db(path)) as connection:
            for sql, args in [("SELECT name FROM fields WHERE name=?", ('sec_assets',)),
                              ("SELECT name FROM fields WHERE kind='standard' ORDER BY common DESC,name LIMIT 50", ())]:
                plan = connection.execute('EXPLAIN QUERY PLAN ' + sql, args).fetchall()
                self.assertTrue(any('INDEX' in row[-1] for row in plan))
                self.assertFalse(any('TEMP B-TREE' in row[-1] for row in plan))

    def test_official_name_and_common_recipe_rank_before_editions_paged(self):
        self.add_numeric([
            ['a', 'InventoryNet', 'us-gaap/2023', '0', 'USD', '', '', '1'],
            ['a', 'InventoryNet', 'us-gaap/2024', '0', 'USD', '', '', '2'],
            ['a', 'InventoryNet', 'us-gaap/2024', '0', 'EUR', '', '', '3']], [
            ['InventoryNet', 'us-gaap/2023', '0', 'Inventory', ''],
            ['InventoryNet', 'us-gaap/2024', '0', 'Inventory', '']])
        result = self.dictionary()
        self.addCleanup(lambda: self.cleanup_sidecar(Path(result['path'])))
        page = registry.search_dictionary(self.output, 'inventory', limit=1)
        self.assertEqual(page['items'][0]['name'], 'sec_inventory')
        full = registry.search_dictionary(self.output, 'inventory')['items']
        self.assertEqual(len(full), page['total'])
        self.assertEqual(full, [registry.search_dictionary(self.output, 'inventory', limit=1, offset=i)['items'][0]
                                for i in range(page['total'])])
        qualified = full[-1]['name']
        self.assertEqual(registry.search_dictionary(self.output, qualified)['items'][0]['name'], qualified)
        self.assertEqual(registry.search_dictionary(self.output)['items'][0]['name'], 'sec_inventory')

    def test_uncommon_standard_family_search_precedes_editions_without_hiding_variants(self):
        tag = 'AccruedIncomeTaxesCurrent'
        self.assertNotIn(tag, {spec['tag'] for spec in registry.COMMON_FIELDS.values()})
        versions = ('us-gaap/2012', 'us-gaap/2024')
        self.add_numeric([
            ['a', tag, version, '0', unit, '', '', '1']
            for version in versions for unit in ('USD', 'EUR')], [
            [tag, version, '0', 'Accrued income taxes', 'Tax liability'] for version in versions])
        result = self.dictionary()
        self.addCleanup(lambda: self.cleanup_sidecar(Path(result['path'])))
        full = registry.search_dictionary(self.output, 'accrued income taxes')
        families, editions = full['items'][:2], full['items'][2:]
        self.assertEqual(full['total'], 6)
        self.assertEqual({row['spec']['taxonomy'] for row in families}, {'us-gaap/'})
        self.assertEqual({row['spec']['unit'] for row in families}, {'USD', 'EUR'})
        self.assertEqual({(row['spec']['taxonomy'], row['spec']['unit']) for row in editions},
                         {(version, unit) for version in versions for unit in ('USD', 'EUR')})
        # Indexed default probes lead the concept search; advanced editions
        # retain deterministic storage ordering. Neither order ranks quality.
        self.assertEqual([row['spec']['unit'] for row in families], ['USD', 'EUR'])
        storage_names = [row.get('storage_name', row['name']) for row in editions]
        self.assertEqual(storage_names, sorted(storage_names))
        self.assertEqual(full['items'], [registry.search_dictionary(
            self.output, 'accrued income taxes', limit=1, offset=i)['items'][0] for i in range(6)])
        exact = next(row['name'] for row in editions if row['spec']['taxonomy'] == 'us-gaap/2012')
        self.assertEqual(registry.search_dictionary(self.output, exact)['items'][0]['name'], exact)
        empty = registry.search_dictionary(self.output)
        self.assertEqual(empty['total'], full['total'])
        self.assertEqual({row['name'] for row in empty['items']}, {row['name'] for row in full['items']})
        storage_names = [row.get('storage_name', row['name']) for row in empty['items']]
        self.assertEqual(storage_names, sorted(storage_names))


if __name__ == '__main__':
    unittest.main()
