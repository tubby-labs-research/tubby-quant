"""Readable names preserve sealed recipes, old spellings and worker checks."""
import sqlite3
import unittest
from contextlib import closing
from unittest.mock import patch

from qlib_yahoo import sec_catalog, sec_execution, sec_registry
from qlib_yahoo.sec_resolution import field_dependencies, resolve_config
try:
    from . import test_sec_execution as fixture
except ImportError:
    import test_sec_execution as fixture


class ReadableResolutionTests(unittest.TestCase):
    write_json = fixture.ExecutionTests.write_json
    seal_parent = fixture.ExecutionTests.seal_parent
    build = fixture.ExecutionTests.build
    shard = fixture.ExecutionTests.shard

    def setUp(self):
        fixture.ExecutionTests.setUp(self)
        self.extra = dict(sec_registry.COMMON_FIELDS['sec_assets'], tag='AssetsCurrent')
        self.old_name = sec_registry._name_for_recipe(self.extra)
        with closing(sqlite3.connect(self.database)) as db:
            db.execute("ALTER TABLE fields ADD COLUMN label TEXT NOT NULL DEFAULT 'Fixture'")
            db.execute("ALTER TABLE fields ADD COLUMN kind TEXT NOT NULL DEFAULT 'standard'")
            db.execute('INSERT INTO fields(name,spec) VALUES(?,?)',
                       (self.old_name, sec_catalog._canonical(self.extra).decode()))
            db.commit()
        self.specs.append(self.extra)
        with closing(sqlite3.connect(self.catalog / 'catalog.sqlite')) as db:
            db.execute('CREATE TABLE tags(tag TEXT,version TEXT,payload TEXT)')
            db.execute('CREATE INDEX tags_key ON tags(tag COLLATE NOCASE,version)')
            db.execute('INSERT INTO tags VALUES(?,?,?)', ('AssetsCurrent', 'us-gaap/2024', '{}'))
            db.commit()
        self.seal_parent()
        self.build()
        self.config = dict(catalog=str(self.catalog), mode='dictionary')

    def test_old_and_readable_names_resolve_same_recipe_and_verify_once(self):
        alias = 'sec_assets_curr'
        entries = sec_registry.lookup_fields(self.catalog, [alias, self.old_name])
        self.assertEqual(entries[alias]['storage_name'], self.old_name)
        self.assertEqual(entries[alias]['spec'], entries[self.old_name]['spec'])
        with patch.object(sec_execution, 'verify_execution_fields',
                          wraps=sec_execution.verify_execution_fields) as verify:
            resolved = resolve_config([alias + '/' + self.old_name], self.config, verify=True)
        self.assertEqual(resolved['bindings'], {alias: self.extra, self.old_name: self.extra})
        self.assertEqual(verify.call_args.args[1], [self.old_name])

    def test_generic_original_tag_alias_works_in_both_expression_dialects(self):
        name = 'sec_assets_current'
        self.assertEqual(field_dependencies([name], dialect='lab'), {name})
        self.assertEqual(field_dependencies(['$' + name], dialect='qlib'), {name})
        self.assertEqual(resolve_config([name], self.config, verify=True)['bindings'], {name: self.extra})

    def test_new_name_never_bypasses_missing_recipe_or_typo_checks(self):
        for name in ('sec_assets_currr', 'sec_totally_invented_accounting_field', 'sec_assets_current_4q'):
            with self.subTest(name=name), patch.object(sec_execution, 'verify_execution_fields') as verify:
                with self.assertRaises(ValueError):
                    resolve_config([name], self.config, verify=True)
                verify.assert_not_called()

    def test_alias_still_rejects_shard_corruption(self):
        self.shard(self.old_name).write_bytes(self.shard(self.old_name).read_bytes() + b'corrupt')
        with self.assertRaises(ValueError):
            resolve_config(['sec_assets_curr'], self.config, verify=True)

    def test_alias_metadata_disagreement_and_saved_bindings_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'different definition'):
            resolve_config(['sec_assets_curr'], {**self.config, 'bindings': {
                'sec_assets_curr': sec_registry.COMMON_FIELDS['sec_assets']}}, verify=True)
        with closing(sqlite3.connect(self.database)) as db:
            db.execute('UPDATE fields SET spec=? WHERE name=?',
                       (sec_catalog._canonical(dict(self.extra, unit='EUR')).decode(), self.old_name))
            db.commit()
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            sec_registry.lookup_fields(self.catalog, ['sec_assets_curr'])

    def test_no_dictionary_or_execution_rebuild_or_bulk_parent_verification(self):
        before = self.database.read_bytes()
        with patch.object(sec_registry, 'verify_dictionary', side_effect=AssertionError('bulk verify')), \
                patch.object(sec_registry, 'build_dictionary', side_effect=AssertionError('rebuild')), \
                patch.object(sec_execution, 'build_execution_index', side_effect=AssertionError('rebuild')):
            self.assertEqual(resolve_config(['sec_assets_curr'], self.config, verify=True)['bindings'],
                             {'sec_assets_curr': self.extra})
        self.assertEqual(before, self.database.read_bytes())

    def test_lowercase_acronym_spelling_resolves_original_tag_and_rejects_collision(self):
        spec = dict(self.extra, tag='EPSFixture')
        first_name = sec_registry._name_for_recipe(spec)
        with closing(sqlite3.connect(self.database)) as db:
            db.execute('INSERT INTO fields(name,spec) VALUES(?,?)',
                       (first_name, sec_catalog._canonical(spec).decode()))
            db.commit()
        with closing(sqlite3.connect(self.catalog / 'catalog.sqlite')) as db:
            db.execute('INSERT INTO tags VALUES(?,?,?)', ('EPSFixture', 'us-gaap/2024', '{}'))
            db.commit()
        found = sec_registry.lookup_fields(self.catalog, ['sec_eps_fixture'])
        self.assertEqual(found['sec_eps_fixture']['spec'], spec)
        other = dict(spec, tag='EpsFixture')
        with closing(sqlite3.connect(self.database)) as db:
            db.execute('INSERT INTO fields(name,spec) VALUES(?,?)',
                       (sec_registry._name_for_recipe(other), sec_catalog._canonical(other).decode()))
            db.commit()
        with closing(sqlite3.connect(self.catalog / 'catalog.sqlite')) as db:
            db.execute('INSERT INTO tags VALUES(?,?,?)', ('EpsFixture', 'us-gaap/2024', '{}'))
            db.commit()
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            sec_registry.lookup_fields(self.catalog, ['sec_eps_fixture'])
        # Discovery must not advertise an ambiguous alias; exact names still work.
        self.assertEqual(sec_registry.name_for_spec(self.catalog, spec)['name'], first_name)
        self.assertEqual(sec_registry.lookup_fields(self.catalog, [first_name])[first_name]['spec'], spec)

    def test_generic_editions_and_ifrs_do_not_use_reviewed_family_shortcut(self):
        for taxonomy, alias in (
                ('us-gaap/2024', 'sec_assets_current__us_gaap_2024'),
                ('ifrs/', 'sec_assets_current__ifrs')):
            with self.subTest(taxonomy=taxonomy):
                spec = dict(self.extra, taxonomy=taxonomy)
                other = dict(spec, tag='ASSETSCurrent')
                with closing(sqlite3.connect(self.database)) as db:
                    for recipe in (spec, other):
                        db.execute('INSERT INTO fields(name,spec) VALUES(?,?)',
                                   (sec_registry._name_for_recipe(recipe),
                                    sec_catalog._canonical(recipe).decode()))
                    db.commit()
                with closing(sqlite3.connect(self.catalog / 'catalog.sqlite')) as db:
                    db.execute('INSERT INTO tags VALUES(?,?,?)', ('ASSETSCurrent', taxonomy, '{}'))
                    db.commit()
                with self.assertRaisesRegex(ValueError, 'Ambiguous'):
                    sec_registry.lookup_fields(self.catalog, [alias])
                original = sec_registry._name_for_recipe(spec)
                self.assertEqual(sec_registry.name_for_spec(self.catalog, spec)['name'], original)
                self.assertEqual(sec_registry.lookup_fields(self.catalog, [original])[original]['spec'], spec)
        # The reviewed US-GAAP family spelling remains an explicit exact mapping.
        self.assertEqual(sec_registry.lookup_fields(self.catalog, ['sec_assets_curr'])
                         ['sec_assets_curr']['spec'], self.extra)


if __name__ == '__main__':
    unittest.main()
