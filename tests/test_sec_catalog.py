"""Independent synthetic records for lossless SEC catalog boundaries."""
import csv
import errno
import hashlib
import io
import json
import sqlite3
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest.mock import patch

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

from qlib_yahoo import sec_catalog as catalog


def tsv(header, rows):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue()


@unittest.skipIf(pq is None, 'pyarrow required')
class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='sec-catalog-test-')
        self.addCleanup(self.temp.cleanup)
        # macOS's system temporary root may use /var -> /private/var. Resolve
        # that root before creating fixtures; tested child links stay unresolved.
        self.root = Path(self.temp.name).resolve()
        self.ticker = self.root / 'supplied-tickers.json'
        self.ticker.write_bytes(b'{"0":{"cik_str":1,"ticker":"AAA","title":"Issuer"}}\n')
        self.output = self.root / 'catalog'
        self.archive = self.root / '2012q1.zip'
        self.tables = {
            'sub.txt': tsv(['adsh', 'cik', 'accepted', 'filed', 'form', 'name', 'extra'], [
                ['a', '0000000001', '2012-05-01 18:00:00.0', '20120501', '10-Q', 'Issuer', ''],
                ['b', '2', '', 'invalid', '10-Q/A', 'Other', 'opaque']]),
            'tag.txt': tsv(['tag', 'version', 'custom', 'tlabel', 'doc', 'datatype', 'iord', 'extra'], [
                ['Assets', 'us-gaap/2012', '0', 'Total assets', 'Balance sheet amount', 'monetary', 'I', ''],
                ['Other', 'a', '1', 'Custom item', 'Special prose\nwith tab\tand "quotes"', 'string', 'D', 'unpaired']]),
            'num.txt': tsv(['adsh', 'tag', 'version', 'ddate', 'qtrs', 'uom', 'segments', 'coreg', 'value', 'footnote'], [
                ['a', 'Assets', 'us-gaap/2012', '20120331', '0', 'USD', '', '', '9007199254740993.000', ''],
                ['a', 'Other', 'a', '20120331', '1', 'EUR/shares', 'Axis=Thing\tTwo\nThree', 'Subsidiary', '', 'literal "quote"'],
                ['b', 'Assets', 'us-gaap/2012', '20120331', '0', 'USD', '', '', 'bad', ''],
                ['missing', 'Unpaired', 'new-standard/2012', '20120331', '2', 'JPY', '', '', '123', '']]),
            'pre.txt': tsv(['adsh', 'tag', 'version', 'stmt', 'line', 'extra'], [['a', 'Assets', 'us-gaap/2012', 'BS', '1', 'text']]),
            'readme.htm': '<p>source documentation</p>',
        }
        self.write_zip()

    def write_zip(self, extra=None):
        with zipfile.ZipFile(self.archive, 'w', zipfile.ZIP_DEFLATED) as stream:
            for name, text in self.tables.items():
                stream.writestr(name, text)
            if extra:
                stream.writestr(extra, 'opaque')

    def build(self, **kwargs):
        return catalog.build_catalog([self.archive], self.output, ticker_map=self.ticker, **kwargs)

    def spec(self, **kwargs):
        return dict(tag='Assets', taxonomy='us-gaap/', unit='USD', qtrs=0, **kwargs)

    def test_all_columns_rows_blank_strings_decimal_and_provenance(self):
        before = self.archive.read_bytes(), self.ticker.read_bytes()
        manifest = self.build()
        self.assertEqual({k: v['rows'] for k, v in manifest['tables'].items()}, {'sub': 2, 'tag': 2, 'num': 4, 'pre': 1})
        for table in ('sub', 'tag', 'num', 'pre'):
            reader = csv.DictReader(io.StringIO(self.tables[table + '.txt']), delimiter='\t', strict=True)
            expected = list(reader)
            actual = catalog.query_rows(self.output, table)
            actual.sort(key=lambda row: row['source_row'])
            self.assertEqual([{key: row[key] for key in reader.fieldnames} for row in actual], expected)
            self.assertEqual([row['source_row'] for row in actual], list(range(1, len(expected) + 1)))
            self.assertTrue(all(row['source_archive'] == str(self.archive) and row['source_sha256'] == hashlib.sha256(before[0]).hexdigest() for row in actual))
        self.assertEqual(catalog.query_rows(self.output, 'num', {'tag': 'Assets'})[0]['value'], '9007199254740993.000')
        self.assertEqual(self.archive.read_bytes(), before[0])
        self.assertEqual(self.ticker.read_bytes(), before[1])
        self.assertEqual((self.output / 'tickers.json').read_bytes(), before[1])
        self.assertEqual(catalog.read_catalog(self.output, verify=True), manifest)

    def test_tag_search_metadata_prefix_fts_pagination_and_parameters(self):
        self.build()
        self.assertEqual(catalog.search_tags(self.output, 'Ass')['items'][0]['tag'], 'Assets')
        item = catalog.search_tags(self.output, 'Special prose')['items'][0]
        self.assertEqual((item['version'], item['custom'], item['datatype'], item['iord']), ('a', '1', 'string', 'D'))
        self.assertEqual(catalog.search_tags(self.output, limit=1, offset=1)['total'], 2)
        self.assertEqual(catalog.search_tags(self.output, "'; DROP TABLE tags; --")['items'], [])
        self.assertEqual(catalog.query_rows(self.output, 'num', {'tag': "' OR 1=1 --"}), [])
        self.assertEqual(catalog.search_tags(self.output)['total'], 2)

    def test_tag_prefix_does_not_hide_label_matches_and_pages_compact_keys(self):
        self.tables['tag.txt'] = tsv(['tag', 'version', 'tlabel', 'doc'], [
            ['Assets', 'us-gaap/2012', 'Total assets', 'First'],
            ['ZLabelMatch', 'us-gaap/2012', 'Assets alternate disclosure', 'Second'],
            ['aLower', 'us-gaap/2012', 'Unrelated', 'Third']])
        self.write_zip()
        self.build()
        result = catalog.search_tags(self.output, 'Assets', limit=1)
        self.assertEqual(result['total'], 2)
        self.assertEqual(result['items'][0]['tag'], 'Assets')
        self.assertEqual(catalog.search_tags(self.output, 'Assets', limit=1, offset=1)['items'][0]['tag'], 'ZLabelMatch')
        self.assertEqual([row['tag'] for row in catalog.search_tags(self.output)['items']], ['aLower', 'Assets', 'ZLabelMatch'])
        with catalog.closing(catalog._db(self.output)) as connection:
            plan = connection.execute('EXPLAIN QUERY PLAN SELECT rowid,tag,version FROM tags INDEXED BY tags_key ORDER BY tag COLLATE NOCASE,version,rowid LIMIT ? OFFSET ?', (50, 0)).fetchall()
        self.assertTrue(any('COVERING INDEX tags_key' in row[-1] for row in plan))
        self.assertFalse(any('TEMP B-TREE' in row[-1] for row in plan))

    def test_search_exact_standard_releases_before_custom_prefix_and_labels(self):
        self.tables['tag.txt'] = tsv(['tag', 'version', 'custom', 'tlabel', 'doc'], [
            ['NetIncomeLoss', '0000000001-24-000001', '1', 'Profit', 'Custom'],
            ['NetIncomeLoss', 'us-gaap/2012', '0', 'Profit', 'Older standard'],
            ['NetIncomeLoss', 'us-gaap/2025', '0', 'Profit', 'Latest standard'],
            ['NetIncomeLoss', 'another-family/2024', '0', 'Profit', 'No whitelist'],
            ['NetIncomeLoss', 'missing-custom', '', 'Profit', 'Unknown is not standard'],
            ['NetIncomeLossAvailable', 'us-gaap/2025', '0', 'Profit', 'Prefix'],
            ['ALabelOnly', 'us-gaap/2025', '0', 'NetIncomeLoss alternate', 'Label']])
        self.write_zip()
        self.build()
        expected = [('NetIncomeLoss', 'us-gaap/2025'), ('NetIncomeLoss', 'us-gaap/2012'),
                    ('NetIncomeLoss', 'another-family/2024'), ('NetIncomeLoss', 'missing-custom'),
                    ('NetIncomeLoss', '0000000001-24-000001'),
                    ('NetIncomeLossAvailable', 'us-gaap/2025'), ('ALabelOnly', 'us-gaap/2025')]
        result = catalog.search_tags(self.output, 'netincomeloss', limit=3)
        self.assertEqual(result['total'], 7)
        self.assertTrue(all(row['custom'] == '0' for row in result['items']))
        actual = []
        for offset in range(7):
            page = catalog.search_tags(self.output, 'netincomeloss', limit=1, offset=offset)
            self.assertEqual(page['total'], 7)
            actual.extend((row['tag'], row['version']) for row in page['items'])
        self.assertEqual(actual, expected)

    def test_variants_preserve_versions_units_contexts_and_custom(self):
        self.build()
        self.assertEqual(catalog.field_variants(self.output, 'Other', 'a'), [
            {'version': 'a', 'uom': 'EUR/shares', 'qtrs': '1', 'segments': 'Axis=Thing\tTwo\nThree', 'coreg': 'Subsidiary'}])
        self.assertEqual(len(catalog.field_variants(self.output, 'Unpaired', 'new-standard/')), 1)
        with self.assertRaises(ValueError):
            catalog.field_variants(self.output, 'Other', '0000000001-12-000001/')

    def test_pre_fts_catalog_search_is_readonly_and_compatible(self):
        self.build()
        database = self.output / 'catalog.sqlite'
        # Construct the old schema in this synthetic fixture only, including
        # its original case-sensitive tag index. No live catalog is touched.
        connection = sqlite3.connect(database)
        try:
            connection.executescript('DROP TABLE tags_fts; DROP INDEX tags_key; CREATE INDEX tags_key ON tags(tag,version);')
            connection.commit()
        finally:
            connection.close()
        manifest_path = self.output / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['artifacts_sha256']['catalog.sqlite'] = hashlib.sha256(database.read_bytes()).hexdigest()
        manifest.pop('catalog_sha256')
        manifest['catalog_sha256'] = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
        manifest_path.write_bytes(json.dumps(manifest, ensure_ascii=False).encode())
        before = {p.relative_to(self.output): p.read_bytes() for p in self.output.rglob('*') if p.is_file()}
        self.assertEqual(catalog.search_tags(self.output, 'Ass')['items'][0]['tag'], 'Assets')
        self.assertEqual(catalog.search_tags(self.output, 'Special prose')['items'][0]['tag'], 'Other')
        self.assertEqual(catalog.search_tags(self.output, 'Special%', limit=1)['total'], 0)
        self.assertEqual(catalog.search_tags(self.output, "'; DROP TABLE tags; --")['total'], 0)
        self.assertEqual(catalog.search_tags(self.output, limit=1, offset=1)['total'], 2)
        catalog.read_catalog(self.output, verify=True)
        self.assertEqual({p.relative_to(self.output): p.read_bytes() for p in self.output.rglob('*') if p.is_file()}, before)

    def test_iter_facts_join_invalid_clock_unpaired_and_missing(self):
        self.build()
        rows = list(catalog.iter_facts(self.output, self.spec()))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['cik'], '0000000001')
        self.assertEqual(rows[0]['sub']['source_member'], 'sub.txt')
        self.assertEqual(rows[1]['accepted'], '')
        self.assertFalse(any(row['filing_conflict'] for row in rows))
        custom = list(catalog.iter_facts(self.output, dict(tag='Other', taxonomy='a', unit='EUR/shares', qtrs=1,
                                                        segments='Axis=Thing\tTwo\nThree', coreg='Subsidiary')))
        self.assertEqual(custom[0]['value'], '')
        missing = list(catalog.iter_facts(self.output, dict(tag='Unpaired', taxonomy='new-standard/', unit='JPY', qtrs=2)))
        self.assertTrue(missing[0]['filing_missing'])

    def test_conflicting_sub_never_arbitrarily_selects_clock_or_cik(self):
        self.tables['sub.txt'] = tsv(['adsh', 'cik', 'accepted', 'filed'], [
            ['a', '1', '2012-05-01 10:00:00', '20120501'], ['a', '2', '2012-05-02 10:00:00', '20120502']])
        self.write_zip()
        self.build()
        rows = [row for row in catalog.iter_facts(self.output, self.spec()) if row['adsh'] == 'a']
        self.assertEqual({row['cik'] for row in rows}, {'1', '2'})
        self.assertTrue(all(row['filing_conflict'] and row['sub_match_count'] == 2 for row in rows))

    def test_identical_sub_duplicates_retain_provenance_without_conflict(self):
        self.tables['sub.txt'] = tsv(['adsh', 'cik', 'accepted', 'filed'], [['a', '1', '', '']] * 2)
        self.write_zip()
        self.build()
        rows = [row for row in catalog.iter_facts(self.output, self.spec()) if row['adsh'] == 'a']
        self.assertEqual(len(rows), 2)
        self.assertFalse(any(row['filing_conflict'] for row in rows))
        self.assertEqual({row['sub']['source_row'] for row in rows}, {1, 2})

    def test_chunk_sort_and_schema_evolution_without_full_num_python_rows(self):
        other = self.root / '2012q2.zip'
        with zipfile.ZipFile(other, 'w') as stream:
            stream.writestr('num.txt', tsv(['adsh', 'tag', 'version', 'qtrs', 'uom', 'segments', 'coreg', 'newcol'],
                                         [['a', 'Z', 'us-gaap/2013', '0', 'USD', '', '', 'new']]))
        with patch.object(catalog, 'CHUNK_BYTES', 1), patch.object(catalog, 'CSV_BLOCK_BYTES', 256):
            manifest = catalog.build_catalog([self.archive, other], self.output, ticker_map=self.ticker)
        self.assertGreater(len(manifest['tables']['num']['files']), 1)
        rows = catalog.query_rows(self.output, 'num')
        self.assertIsNone(next(row for row in rows if row['tag'] == 'Z')['value'])
        for relative in manifest['tables']['num']['files']:
            block = pq.read_table(self.output / relative)
            keys = list(zip(block['tag'].to_pylist(), block['version'].to_pylist()))
            self.assertEqual(keys, sorted(keys))

    def test_missing_context_columns_remain_null_and_require_explicit_adapter(self):
        other = self.root / '2012q2.zip'
        with zipfile.ZipFile(other, 'w') as stream:
            stream.writestr('num.txt', tsv(['adsh', 'tag', 'version', 'qtrs', 'uom', 'value'],
                                         [['a', 'Assets', 'us-gaap/2012', '0', 'USD', '123']]))
        catalog.build_catalog([self.archive, other], self.output, ticker_map=self.ticker)
        rows = catalog.query_rows(self.output, 'num', {'source_archive': str(other)})
        self.assertIsNone(rows[0]['segments'])
        self.assertIsNone(rows[0]['coreg'])
        variants = catalog.field_variants(self.output, 'Assets', 'us-gaap/')
        self.assertTrue(any(row['segments'] is None for row in variants))
        with self.assertRaisesRegex(ValueError, 'structural context column'):
            list(catalog.iter_facts(self.output, self.spec()))

    def test_notes_directory_all_eight_tables_and_explicit_materialization_rejection(self):
        notes = self.root / 'notes'
        notes.mkdir()
        for name, text in self.tables.items():
            (notes / name.replace('.txt', '.tsv')).write_bytes(text.encode('utf-8'))
        for table in ('cal', 'dim', 'ren', 'txt'):
            (notes / (table + '.tsv')).write_bytes(tsv(['adsh', 'opaque'], [['a', 'note\ttext\n"quotes"']]).encode('utf-8'))
        manifest = catalog.build_catalog([notes], self.output, ticker_map=self.ticker)
        self.assertEqual(set(manifest['tables']), {'sub', 'tag', 'num', 'pre', 'cal', 'dim', 'ren', 'txt'})
        self.assertEqual(catalog.query_rows(self.output, 'txt')[0]['opaque'], 'note\ttext\n"quotes"')
        self.assertEqual(manifest['supported_products'], ['notes'])
        with self.assertRaisesRegex(ValueError, 'schema adapter'):
            list(catalog.iter_facts(self.output, self.spec()))

    def test_mixed_products_rejected(self):
        notes = self.root / 'notes'
        notes.mkdir()
        (notes / 'txt.tsv').write_text('adsh\ttext\na\tnote\n')
        with self.assertRaisesRegex(ValueError, 'separate face and notes'):
            catalog.build_catalog([self.archive, notes], self.output, ticker_map=self.ticker)
        self.assertFalse(self.output.exists())

    def test_arbitrary_safe_table_is_not_silently_skipped(self):
        self.tables['extension.tsv'] = tsv(['one', 'two'], [['', 'text']])
        self.write_zip()
        self.build()
        self.assertEqual(catalog.query_rows(self.output, 'extension')[0]['one'], '')

    def test_empty_table_schema_is_queryable(self):
        self.tables['pre.txt'] = 'adsh\ttag\tversion\n'
        self.write_zip()
        self.build()
        self.assertEqual(catalog.query_rows(self.output, 'pre'), [])

    def test_no_overwrite_and_output_cannot_be_inside_sources(self):
        self.build()
        before = (self.output / 'manifest.json').read_bytes()
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual((self.output / 'manifest.json').read_bytes(), before)
        with self.assertRaises(ValueError):
            catalog.build_catalog([self.root], self.root / 'child', ticker_map=self.ticker)

    def make_source_symlink(self, link, target, *, directory=False):
        try:
            link.symlink_to(target, target_is_directory=directory)
        except OSError as exc:
            if (getattr(exc, 'winerror', None) in {50, 1314}
                    or (getattr(exc, 'winerror', None) is None
                        and exc.errno in {errno.EPERM, errno.EACCES, errno.ENOSYS,
                                          errno.ENOTSUP, errno.EOPNOTSUPP})):
                self.skipTest(f'symlink setup unsupported: {exc}')
            raise

    def assert_source_link_rejected_before_read(self, source, target):
        original_open = Path.open
        target = target.resolve()

        def guarded_open(path, *args, **kwargs):
            if path.resolve() == target:
                raise AssertionError(f'redirected source target opened: {path}')
            return original_open(path, *args, **kwargs)

        # Covers the hash reader and table reader, while allowing ticker reads
        # and failure-receipt writes. Resolve only in the guard, never the input.
        with patch.object(Path, 'open', autospec=True, side_effect=guarded_open):
            with self.assertRaisesRegex(ValueError, 'Redirected path is not allowed'):
                catalog.build_catalog([source], self.output, ticker_map=self.ticker)
        self.assertFalse(self.output.exists())

    def test_symlinked_archive_rejected_before_target_read(self):
        link = self.root / 'linked-archive.zip'
        self.make_source_symlink(link, self.archive)
        self.assert_source_link_rejected_before_read(link, self.archive)

    def test_symlinked_source_directory_rejected_before_target_read(self):
        source = self.root / 'source'
        source.mkdir()
        target = source / 'num.txt'
        target.write_text(self.tables['num.txt'], encoding='utf-8')
        link = self.root / 'linked-source'
        self.make_source_symlink(link, source, directory=True)
        self.assert_source_link_rejected_before_read(link, target)

    def test_symlinked_table_member_rejected_before_target_read(self):
        source = self.root / 'source'
        source.mkdir()
        target = self.root / 'target-num.txt'
        target.write_text(self.tables['num.txt'], encoding='utf-8')
        self.make_source_symlink(source / 'num.txt', target)
        self.assert_source_link_rejected_before_read(source, target)

    def test_tampering_artifacts_inventory_and_manifest_identity(self):
        self.build()
        (self.output / 'tickers.json').write_text('{}')
        catalog.read_catalog(self.output)  # cheap metadata intentionally does not hash bulk bytes
        with self.assertRaisesRegex(ValueError, 'artifact differs'):
            catalog.read_catalog(self.output, verify=True)
        (self.output / 'surprise').write_text('x')
        with self.assertRaisesRegex(ValueError, 'inventory differs'):
            catalog.read_catalog(self.output, verify=True)
        manifest = json.loads((self.output / 'manifest.json').read_text())
        manifest['tables']['num']['rows'] += 1
        (self.output / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            catalog.read_catalog(self.output)

    def test_malformed_headers_and_rows_fail_without_ready_output(self):
        for text in ('adsh\ttag\ttag\n', 'adsh\ttag\tversion\tqtrs\tuom\na\tAssets\tus-gaap/2012\t0\tUSD\textra\n'):
            with self.subTest(text=text):
                self.tables['num.txt'] = text
                self.write_zip()
                with self.assertRaises(Exception):
                    self.build()
                self.assertFalse(self.output.exists())
        for staging in self.root.glob('catalog.building-*'):
            self.assertEqual(json.loads((staging / 'manifest.json').read_text())['status'], 'failed')

    def test_unsafe_duplicate_and_ambiguous_zip_members(self):
        for extra in ('../evil.tsv', '/evil.tsv', 'folder/num.txt', 'NUM.txt', 'num.txt', 'bad-name.tsv'):
            with self.subTest(extra=extra), warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.write_zip(extra)
                with self.assertRaises(ValueError):
                    self.build()
                self.assertFalse(self.output.exists())

    def test_crc_failure_is_fatal(self):
        with zipfile.ZipFile(self.archive, 'w', zipfile.ZIP_STORED) as stream:
            for name, text in self.tables.items():
                stream.writestr(name, text)
        self.archive.write_bytes(self.archive.read_bytes().replace(b'9007199254740993.000', b'9007199254740994.000'))
        with self.assertRaises(zipfile.BadZipFile):
            self.build()
        self.assertFalse(self.output.exists())

    def test_cancel_and_changed_source_keep_nonready_receipts(self):
        def cancel(message):
            if 'Publishing' in message:
                raise KeyboardInterrupt('test cancellation')
        with self.assertRaises(KeyboardInterrupt):
            self.build(progress=cancel)
        staging = next(self.root.glob('catalog.building-*'))
        self.assertEqual(json.loads((staging / 'manifest.json').read_text())['status'], 'cancelled')
        def mutate(message):
            if 'Indexing' in message:
                self.ticker.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            self.build(progress=mutate)
        self.assertFalse(self.output.exists())

    def test_bounds_filters_and_spec_validation(self):
        self.build()
        for limit, offset in ((0, 0), (1001, 0), (1, -1), (True, 0)):
            with self.assertRaises(ValueError):
                catalog.query_rows(self.output, 'num', limit=limit, offset=offset)
        with self.assertRaises(ValueError):
            catalog.query_rows(self.output, 'num', {'unknown': 'x'})
        with self.assertRaises(ValueError):
            list(catalog.iter_facts(self.output, dict(tag='Assets', taxonomy='us-gaap/', unit='USD', qtrs='0')))


if __name__ == '__main__':
    unittest.main()
