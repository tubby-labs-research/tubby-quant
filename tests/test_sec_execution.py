"""Tiny sealed SQLite fixtures only; no source datasets or full SEC build."""
import hashlib
import importlib
import json
import os
import sqlite3
import sys
import tempfile
import types
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from qlib_yahoo import sec_catalog as source
from qlib_yahoo import sec_execution as execution
from qlib_yahoo import sec_registry as registry


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        # A short writable root also avoids Windows sandbox temp redirection.
        self.temp = tempfile.TemporaryDirectory(prefix=".sec-execution-test-",
                                                 dir=Path(__file__).resolve().parent.parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.catalog = self.root / "sec" / "catalogs" / "fixture"
        self.catalog.mkdir(parents=True)
        metadata = dict(schema_version=source.VERSION, status="ready",
                        numeric_materialization_supported=True,
                        artifacts_sha256={"catalog.sqlite": "0" * 64},
                        tables={"num": dict(columns=list(registry._NUM_KEYS), files=[], rows=0)})
        metadata["catalog_sha256"] = execution._digest(metadata)
        self.write_json(self.catalog / "manifest.json", metadata)
        self.parent = self.root / "sec" / "dictionaries" / (
            metadata["catalog_sha256"] + "-" + registry.REGISTRY_VERSION)
        self.parent.mkdir(parents=True)
        self.database = self.parent / "dictionary.sqlite"
        # Include common, exact edition, custom, non-USD, duration and context.
        self.specs = [registry.COMMON_FIELDS["sec_assets"],
                      dict(registry.COMMON_FIELDS["sec_inventory"], taxonomy="us-gaap/2024"),
                      dict(tag="CustomTag", taxonomy="0001234567-24-000001", unit="EUR",
                           qtrs=4, segments='Axis=Asia\t\n"quoted"', coreg="Subsidiary")]
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute("CREATE TABLE fields(id INTEGER PRIMARY KEY, name TEXT UNIQUE, spec TEXT UNIQUE)")
            connection.executemany("INSERT INTO fields(name,spec) VALUES(?,?)", [
                (registry._name_for_recipe(spec), source._canonical(spec).decode()) for spec in self.specs])
            connection.commit()
        self.seal_parent()
        self.names = [registry._name_for_recipe(spec) for spec in self.specs]
        self.output = self.parent.with_name(metadata["catalog_sha256"] + "-" + execution.EXECUTION_VERSION)

    def write_json(self, path, value):
        path.write_bytes(source._canonical(value))

    def seal_parent(self, **changes):
        metadata = source.read_catalog(self.catalog)
        seal = dict(status="ready", version=registry.REGISTRY_VERSION,
                    catalog_sha256=metadata["catalog_sha256"],
                    definitions_sha256=registry._DEFINITIONS_SHA256,
                    dictionary_sha256=source._hash(self.database), field_count=len(self.specs))
        seal.update(changes)
        seal["manifest_sha256"] = execution._digest(seal)
        self.write_json(self.parent / "manifest.json", seal)

    def build(self, **kwargs):
        return execution.build_execution_index(self.catalog, **kwargs)

    def shard(self, name=None):
        return self.output / execution._SHARDS[execution._bucket(name or self.names[0])]

    def reseal(self, count=None):
        manifest = json.loads((self.output / "manifest.json").read_bytes())
        for name, record in manifest["shards"].items():
            data = (self.output / name).read_bytes()
            record.update(bytes=len(data), count=data.count(b"\n"), sha256=hashlib.sha256(data).hexdigest())
        if count is not None:
            manifest["field_count"] = count
            self.seal_parent(field_count=count)
            manifest["parent"]["manifest_sha256"] = json.loads(
                (self.parent / "manifest.json").read_bytes())["manifest_sha256"]
        manifest["artifact_sha256"] = execution._digest(manifest["shards"])
        manifest.pop("manifest_sha256")
        manifest["manifest_sha256"] = execution._digest(manifest)
        self.write_json(self.output / "manifest.json", manifest)

    def test_api_all_recipes_and_pin_parent_unchanged_idempotence(self):
        parent_before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.parent.iterdir()}
        catalog_before = (self.catalog / "manifest.json").read_bytes()
        with patch.object(execution, "_hash_parent", wraps=execution._hash_parent) as hashes, \
                patch.object(execution, "_recipe", side_effect=AssertionError("bulk recipe parsing")):
            status = self.build()
        self.assertEqual(hashes.call_count, 2)
        self.assertEqual(set(status), {"status", "path", "identity", "message"})
        self.assertEqual(status["status"], "ready")
        self.assertEqual(set(status["identity"]), {"version", "manifest_sha256", "parent"})
        self.assertEqual(execution.execution_status(self.catalog), status)
        result = execution.verify_execution_fields(self.catalog, self.names, status["identity"])
        self.assertEqual(result, dict(fields=dict(zip(self.names, self.specs)), identity=status["identity"]))
        with patch.object(execution, "_hash_parent", side_effect=AssertionError("repeat full hash")):
            self.assertEqual(self.build(), status)
        self.assertEqual(parent_before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.parent.iterdir()})
        self.assertEqual(catalog_before, (self.catalog / "manifest.json").read_bytes())
        manifest = json.loads((self.output / "manifest.json").read_bytes())
        self.assertEqual(manifest["schema_sha256"], execution.SCHEMA_SHA256)
        self.assertEqual(manifest["field_count"], len(self.names))
        self.assertLess((self.output / "manifest.json").stat().st_size, execution.MAX_MANIFEST_BYTES)

    def test_missing_index_fails_fast(self):
        self.assertEqual(execution.execution_status(self.catalog)["status"], "missing")
        with patch.object(registry, "_db", side_effect=AssertionError("database opened")):
            with self.assertRaisesRegex(ValueError, "Prepare run index"):
                execution.verify_execution_fields(self.catalog, self.names)

    def test_status_never_enumerates_stats_opens_or_hashes_shards(self):
        self.build()
        original_open, original_stat, original_lstat = Path.open, Path.stat, Path.lstat
        def guarded(method):
            def call(path, *args, **kwargs):
                if path.suffix == ".tsv":
                    raise AssertionError("status touched shard")
                return method(path, *args, **kwargs)
            return call
        with patch.object(Path, "open", guarded(original_open)), \
                patch.object(Path, "stat", guarded(original_stat)), \
                patch.object(Path, "lstat", guarded(original_lstat)), \
                patch.object(Path, "iterdir", side_effect=AssertionError("status enumerated")), \
                patch.object(source, "_hash", side_effect=AssertionError("hash")), \
                patch.object(registry, "_db", side_effect=AssertionError("SQL")):
            self.assertEqual(execution.execution_status(self.catalog)["status"], "ready")

    def test_worker_reads_only_used_shards_and_only_parses_selected_rows(self):
        self.build()
        selected = self.names[0]
        bucket = execution._bucket(selected)
        # Add a canonical unrequested recipe in the same shard, then reseal.
        other = next(dict(self.specs[0], unit=f"U{i}") for i in range(10000)
                     if execution._bucket(registry._name_for_recipe(dict(self.specs[0], unit=f"U{i}"))) == bucket)
        with self.shard(selected).open("ab") as stream:
            stream.write(registry._name_for_recipe(other).encode() + b"\t" + source._canonical(other) + b"\n")
        self.reseal(count=4)
        original_open, original_stat, original_lstat = Path.open, Path.stat, Path.lstat
        reads = []
        def guarded(method, record=False):
            def call(path, *args, **kwargs):
                if path.suffix == ".tsv":
                    self.assertEqual(path, self.shard(selected))
                    if record:
                        reads.append(path)
                return method(path, *args, **kwargs)
            return call
        with patch.object(Path, "open", guarded(original_open, True)), \
                patch.object(Path, "stat", guarded(original_stat)), \
                patch.object(Path, "lstat", guarded(original_lstat)), \
                patch.object(execution, "_hash_parent", side_effect=AssertionError("full hash")), \
                patch.object(registry, "_db", side_effect=AssertionError("SQL")), \
                patch.object(execution, "_recipe", wraps=execution._recipe) as recipes:
            for _ in range(2):
                self.assertEqual(execution.verify_execution_fields(self.catalog, [selected])["fields"],
                                 {selected: self.specs[0]})
            self.assertEqual(recipes.call_count, 2)
        self.assertEqual(reads, [self.shard(selected), self.shard(selected)])

    def test_tamper_with_same_size_and_restored_mtime_rehashed(self):
        self.build()
        shard = self.shard()
        before = shard.stat()
        data = shard.read_bytes().replace(b"USD", b"EUR")
        shard.write_bytes(data)
        os.utime(shard, ns=(before.st_atime_ns, before.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "hash/count/bytes"):
            execution.verify_execution_fields(self.catalog, [self.names[0]])

    def test_truncated_empty_and_trailing_tamper_fail(self):
        self.build()
        shard = self.shard()
        original = shard.read_bytes()
        for data in (original[:-1], b"", original + b"bad\t{}\n"):
            with self.subTest(data=data[-10:]):
                shard.write_bytes(data)
                with self.assertRaises(ValueError):
                    execution.verify_execution_fields(self.catalog, [self.names[0]])
        shard.write_bytes(original)

    def test_selected_malformed_noncanonical_and_name_mismatch_even_resealed(self):
        self.build()
        shard = self.shard()
        for data in (b"{bad", b"{}", source._canonical(dict(self.specs[0], unit="EUR")),
                     json.dumps(self.specs[0]).encode(),
                     source._canonical(self.specs[0])[:-1] + b',"qtrs":0}'):
            with self.subTest(data=data):
                shard.write_bytes(self.names[0].encode() + b"\t" + data + b"\n")
                self.reseal()
                with self.assertRaises(ValueError):
                    execution.verify_execution_fields(self.catalog, [self.names[0]])

    def test_duplicate_selected_and_missing_names(self):
        self.build()
        shard = self.shard()
        original = shard.read_bytes()
        shard.write_bytes(original + original)
        self.reseal(count=4)
        with self.assertRaisesRegex(ValueError, "Duplicate selected"):
            execution.verify_execution_fields(self.catalog, [self.names[0]])
        with self.assertRaisesRegex(ValueError, "Missing requested"):
            execution.verify_execution_fields(self.catalog, ["sec_not_observed"])

    def test_requested_names_validation_and_empty(self):
        self.build()
        for names in ("sec_assets", ["../sec_assets"], [None], [self.names[0]] * 2,
                      (f"sec_field_{i}" for i in range(execution.MAX_FIELDS + 1))):
            with self.subTest(names=type(names)):
                with self.assertRaises(ValueError):
                    execution.verify_execution_fields(self.catalog, names)
        self.assertEqual(execution.verify_execution_fields(self.catalog, [])["fields"], {})

    def test_manifest_tamper_pin_and_parent_mismatch(self):
        status = self.build()
        manifest_path = self.output / "manifest.json"
        original = manifest_path.read_bytes()
        manifest = json.loads(original)
        manifest["schema_sha256"] = "1" * 64
        self.write_json(manifest_path, manifest)
        self.assertEqual(execution.execution_status(self.catalog)["status"], "invalid")
        with self.assertRaises(ValueError):
            self.build()  # Never overwrite invalid existing index.
        self.assertEqual(json.loads(manifest_path.read_bytes()), manifest)
        manifest_path.write_bytes(original)
        bad_pin = dict(status["identity"], manifest_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "Pinned"):
            execution.verify_execution_fields(self.catalog, self.names, bad_pin)
        self.seal_parent(dictionary_sha256="1" * 64)
        with self.assertRaisesRegex(ValueError, "parent identity"):
            execution.verify_execution_fields(self.catalog, self.names)

    def test_self_seal_counts_artifact_hash_and_oversized_manifest(self):
        self.build()
        path = self.output / "manifest.json"
        original = path.read_bytes()
        for key, value in (("manifest_sha256", "0" * 64), ("artifact_sha256", "0" * 64),
                           ("field_count", True)):
            with self.subTest(key=key):
                manifest = json.loads(original)
                manifest[key] = value
                self.write_json(path, manifest)
                with self.assertRaises(ValueError):
                    execution.verify_execution_fields(self.catalog, self.names)
        path.write_bytes(b" " * (execution.MAX_MANIFEST_BYTES + 1))
        self.assertEqual(execution.execution_status(self.catalog)["status"], "invalid")

    def test_missing_extra_partial_inventory_and_unused_corruption(self):
        self.build()
        unused = next(self.output / name for name in execution._SHARDS
                      if name != self.shard().name and (self.output / name).stat().st_size == 0)
        unused.write_bytes(b"corrupted but unused")
        execution.verify_execution_fields(self.catalog, [self.names[0]])
        unused.unlink()
        with self.assertRaisesRegex(ValueError, "inventory"):
            execution.verify_execution_fields(self.catalog, [self.names[0]])
        unused.touch()
        (self.output / "unsealed").touch()
        with self.assertRaisesRegex(ValueError, "inventory"):
            execution.verify_execution_fields(self.catalog, self.names)

    def test_parent_hash_count_and_inventory_fail_closed(self):
        original = self.database.read_bytes()
        self.database.write_bytes(original + b"tamper")
        with self.assertRaisesRegex(ValueError, "Parent dictionary hash"):
            self.build()
        self.database.write_bytes(original)
        self.seal_parent(field_count=999)
        with self.assertRaisesRegex(ValueError, "count"):
            self.build()
        self.seal_parent()
        (self.parent / "dictionary.sqlite-wal").touch()
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.build()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.output.parent.glob(self.output.name + ".building*")))

    def test_parent_mutation_during_export_and_cancel_cleanup(self):
        def mutate(message):
            if "verifying parent again" in message:
                with self.database.open("ab") as stream:
                    stream.write(b"mutation")
        with self.assertRaisesRegex(ValueError, "changed during export"):
            self.build(progress=mutate)
        self.assertFalse(self.output.exists())
        self.seal_parent()
        def cancel(message):
            if "verifying parent again" in message:
                raise KeyboardInterrupt("cancel fixture")
        with self.assertRaises(KeyboardInterrupt):
            self.build(progress=cancel)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.output.parent.glob(self.output.name + ".building*")))

    def test_selected_artifact_mutation_after_hash_fails(self):
        self.build()
        original = execution._recipe
        def mutate_after_hash(name, serialized, *args):
            spec = original(name, serialized, *args)
            with self.shard(name).open("ab") as stream:
                stream.write(b"tamper-after-hash")
            return spec
        with patch.object(execution, "_recipe", side_effect=mutate_after_hash):
            with self.assertRaisesRegex(ValueError, "changed during verification"):
                execution.verify_execution_fields(self.catalog, [self.names[0]])

    def test_wrong_catalog_and_parent_definitions_rejected(self):
        self.build()
        manifest_path = self.catalog / "manifest.json"
        metadata = json.loads(manifest_path.read_bytes())
        metadata["tables"]["num"]["rows"] = 1
        metadata.pop("catalog_sha256")
        metadata["catalog_sha256"] = execution._digest(metadata)
        self.write_json(manifest_path, metadata)
        with self.assertRaisesRegex(ValueError, "Prepare run index"):
            execution.verify_execution_fields(self.catalog, self.names)
        self.assertTrue(self.output.is_dir())

    def test_parent_without_unique_schema_rejected(self):
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute("ALTER TABLE fields RENAME TO old_fields")
            connection.execute("CREATE TABLE fields(id INTEGER PRIMARY KEY,name TEXT,spec TEXT)")
            connection.execute("INSERT INTO fields SELECT * FROM old_fields")
            connection.commit()
        self.seal_parent()
        with self.assertRaisesRegex(ValueError, "unique name/spec"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_concurrent_builder_lock_and_publication_race(self):
        nested = []
        def competing(message):
            if not nested:
                nested.append(True)
                with self.assertRaises(FileExistsError):
                    self.build()
        self.build(progress=competing)
        self.assertTrue(nested)

    def test_existing_target_created_during_build_never_overwritten(self):
        def publish_other(message):
            if "verifying parent again" in message:
                self.output.mkdir()
                (self.output / "other-owner").write_bytes(b"preserve")
        with self.assertRaises(FileExistsError):
            self.build(progress=publish_other)
        self.assertEqual((self.output / "other-owner").read_bytes(), b"preserve")

    def test_redirects_parent_used_shard_and_ancestor(self):
        self.build()
        shard = self.shard()
        backup = self.root / "backup.tsv"
        shard.rename(backup)
        try:
            shard.symlink_to(backup)
        except OSError as exc:
            backup.rename(shard)
            self.skipTest(f"symlink privilege unavailable: {exc}")
        with self.assertRaisesRegex(ValueError, "Redirected"):
            execution.verify_execution_fields(self.catalog, [self.names[0]])
        shard.unlink()
        backup.rename(shard)
        alias = self.root / "catalog-alias"
        alias.symlink_to(self.catalog, target_is_directory=True)
        with self.assertRaises(ValueError):
            execution.verify_execution_fields(alias, self.names)

    def test_used_shard_reparse_attribute_guard_without_symlink_privilege(self):
        self.build()
        original = Path.lstat
        selected = self.shard()
        def redirected(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            if path == selected:
                return types.SimpleNamespace(st_file_attributes=0x400, st_mode=result.st_mode)
            return result
        with patch.object(Path, "lstat", redirected):
            with self.assertRaisesRegex(ValueError, "Redirected"):
                execution.verify_execution_fields(self.catalog, [self.names[0]])

    def test_oversized_row_fails_without_publishing(self):
        with patch.object(execution, "MAX_ROW_BYTES", 128):
            with self.assertRaises((ValueError, sqlite3.DataError)):
                self.build()
        self.assertFalse(self.output.exists())

    def test_manifest_traversal_shard_keys_and_non_file_used_shard_rejected(self):
        self.build()
        manifest_path = self.output / "manifest.json"
        original = manifest_path.read_bytes()
        manifest = json.loads(original)
        manifest["shards"]["../outside.tsv"] = manifest["shards"].pop("ff.tsv")
        manifest["artifact_sha256"] = execution._digest(manifest["shards"])
        manifest.pop("manifest_sha256")
        manifest["manifest_sha256"] = execution._digest(manifest)
        self.write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "inventory is malformed"):
            execution.verify_execution_fields(self.catalog, self.names)
        manifest_path.write_bytes(original)
        self.shard().unlink()
        self.shard().mkdir()
        with self.assertRaisesRegex(ValueError, "ordinary"):
            execution.verify_execution_fields(self.catalog, [self.names[0]])

    def test_builder_single_open_per_shard_and_no_bulk_recipe_validation(self):
        original = Path.open
        opens = []
        def guarded(path, *args, **kwargs):
            if path.suffix == ".tsv":
                opens.append((path.name, args[0]))
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guarded), \
                patch.object(execution, "_recipe", side_effect=AssertionError("bulk parse")):
            self.build()
        self.assertEqual(len(opens), execution.BUCKETS)
        self.assertEqual(set(opens), {(name, "xb") for name in execution._SHARDS})

    def test_wide_context_supported_beyond_old_one_mib_limit(self):
        self.specs[2] = dict(self.specs[2], segments="X" * (1024 * 1024 + 100))
        self.names[2] = registry._name_for_recipe(self.specs[2])
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute("UPDATE fields SET name=?,spec=? WHERE id=3", (
                self.names[2], source._canonical(self.specs[2]).decode()))
            connection.commit()
        self.seal_parent()
        self.build()
        self.assertEqual(execution.verify_execution_fields(self.catalog, [self.names[2]])["fields"],
                         {self.names[2]: self.specs[2]})

    def test_import_under_registry_stub_does_not_retain_stub(self):
        name = "qlib_yahoo.sec_registry"
        with patch.dict(sys.modules, {name: types.ModuleType(name)}):
            importlib.reload(execution)
        self.assertEqual(execution._registry(), registry)
        self.build()
        execution.verify_execution_fields(self.catalog, self.names)


if __name__ == "__main__":
    unittest.main()
