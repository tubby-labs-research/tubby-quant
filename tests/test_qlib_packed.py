"""Small sealed fixtures for the packed Qlib storage adapter."""

import hashlib
import json
import os
import shutil
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from qlib_yahoo.dataset import DatasetError, tree_sha256
from qlib_yahoo import packed as packed_module
from qlib_yahoo.packed import (feature_provider_config, pack_dataset, verified_entries,
                               verify_packed)


class PackedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "original"
        self.target = self.root / "packed"
        files = {
            "calendars/day.txt": b"2020-01-01\n2020-01-02\n2020-01-03\n",
            "instruments/all.txt": b"AAPL\t2020-01-01\t2020-01-03\n",
            "instruments/top.txt": b"AAPL\t2020-01-02\t2020-01-03\n",
            "features/aapl/close.day.bin": struct.pack("<ffff", 1, 2, float("nan"), 4),
        }
        for name, data in files.items():
            path = self.source / "qlib_data" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        evidence = self.source / "universe" / "top.csv"
        evidence.parent.mkdir()
        evidence.write_bytes(b"date,symbol\n2020-01-02,AAPL\n")
        digest, count = tree_sha256(self.source / "qlib_data")
        self.manifest = {"status": "ready", "qlib_binary_tree_sha256": digest,
                         "qlib_binary_file_count": count}
        self.original = json.dumps(self.manifest).encode()
        (self.source / "manifest.json").write_bytes(self.original)

    def packed(self):
        pack_dataset(self.source, self.target)
        return json.loads((self.target / "manifest.json").read_text())

    def with_universe_hashes(self):
        membership = self.source / "universe" / "top.csv"
        rankings = self.source / "universe" / "rankings.csv"
        rankings.write_bytes(b"date,symbol,rank\n2020-01-02,AAPL,1\n")
        self.manifest["universes"] = {"top": {
            "membership_csv": "universe/top.csv",
            "membership_sha256": hashlib.sha256(membership.read_bytes()).hexdigest(),
            "rankings_csv": "universe/rankings.csv",
            "rankings_sha256": hashlib.sha256(rankings.read_bytes()).hexdigest(),
        }}
        (self.source / "manifest.json").write_text(json.dumps(self.manifest))

    def test_seal_and_preserve_source(self):
        before = {p.relative_to(self.source): p.read_bytes() for p in self.source.rglob("*") if p.is_file()}
        manifest = self.packed()
        self.assertEqual(manifest["storage"]["source_manifest_sha256"], hashlib.sha256(self.original).hexdigest())
        self.assertEqual(verify_packed(self.target, manifest)["qlib_binary_file_count"], 4)
        self.assertEqual((self.target / "universe" / "top.csv").read_bytes(),
                         (self.source / "universe" / "top.csv").read_bytes())
        self.assertEqual(before, {p.relative_to(self.source): p.read_bytes()
                                  for p in self.source.rglob("*") if p.is_file()})
        self.assertEqual(feature_provider_config(self.target, manifest)["kwargs"]["archive_sha256"],
                         manifest["storage"]["archive_sha256"])
        with self.assertRaises(DatasetError):
            pack_dataset(self.source, self.target)

    def test_reject_nested_and_unsealed_source(self):
        with self.assertRaises(DatasetError):
            pack_dataset(self.source, self.source / "child")
        with self.assertRaises(DatasetError):
            pack_dataset(self.source, self.root)
        (self.source / "qlib_data" / "calendars" / "day.txt").write_bytes(b"changed")
        with self.assertRaises(DatasetError):
            pack_dataset(self.source, self.target)
        self.assertFalse((self.target / "manifest.json").exists())

    def test_corruption_and_native_divergence(self):
        manifest = self.packed()
        verify_packed(self.target, manifest)
        native = self.target / "qlib_data" / "instruments" / "all.txt"
        native.write_bytes(b"tampered")
        with self.assertRaises(DatasetError):
            verify_packed(self.target, manifest)
        with self.assertRaises(DatasetError):
            feature_provider_config(self.target, manifest)
        native.write_bytes((self.source / "qlib_data" / "instruments" / "all.txt").read_bytes())
        archive = self.target / "qlib_data.zip"
        archive.write_bytes(archive.read_bytes() + b"X")
        with self.assertRaises(DatasetError):
            verify_packed(self.target, manifest)

    def test_verify_universe_evidence_and_clear_registry_on_failure(self):
        self.with_universe_hashes()
        manifest = self.packed()
        identity = manifest["storage"]["archive_sha256"]
        for filename in ("top.csv", "rankings.csv"):
            evidence = self.target / "universe" / filename
            original = evidence.read_bytes()
            verify_packed(self.target, manifest)
            evidence.write_bytes(b"tampered")
            with self.assertRaises(DatasetError):
                verify_packed(self.target, manifest)
            with self.assertRaises(DatasetError):
                verified_entries(self.target, identity)
            evidence.write_bytes(original)
        verify_packed(self.target, manifest)
        redirected = self.target / "universe" / "rankings.csv"
        with mock.patch.object(Path, "is_junction", autospec=True,
                               side_effect=lambda path: path == redirected):
            with self.assertRaises(DatasetError):
                verify_packed(self.target, manifest)
        with self.assertRaises(DatasetError):
            verified_entries(self.target, identity)
        for unsafe in ("../rankings.csv", "universe/../rankings.csv", "qlib_data/instruments/all.txt"):
            altered = json.loads(json.dumps(manifest))
            altered["universes"]["top"]["rankings_csv"] = unsafe
            with self.assertRaises(DatasetError):
                verify_packed(self.target, altered)

    def test_archive_hash_and_verification_stream_from_one_handle(self):
        archive = self.target / "qlib_data.zip"
        original_read_bytes = Path.read_bytes
        original_open = Path.open
        archive_opens = []

        def guarded_read_bytes(path):
            if path == archive:
                self.fail("archive was read into one byte string")
            return original_read_bytes(path)

        def tracked_open(path, *args, **kwargs):
            if path == archive:
                archive_opens.append(args[0] if args else kwargs.get("mode", "r"))
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "read_bytes", guarded_read_bytes), mock.patch.object(Path, "open", tracked_open):
            manifest = self.packed()
            archive_opens.clear()
            with mock.patch.object(packed_module, "_hash_stream", wraps=packed_module._hash_stream) as hashed:
                verify_packed(self.target, manifest)
            self.assertEqual(archive_opens, ["rb"])
            self.assertEqual(hashed.call_count, 2)

    def test_publication_is_immutable_without_builder_alias(self):
        manifest = self.packed()
        identity = manifest["storage"]["archive_sha256"]
        built = []
        original = packed_module._verify_archive
        def captured(*args, **kwargs):
            result = original(*args, **kwargs)
            built.append(result)
            return result
        with mock.patch.object(packed_module, "_verify_archive", side_effect=captured):
            verify_packed(self.target, manifest)
        entries = verified_entries(self.target, identity)
        name = "features/aapl/close.day.bin"
        expected = entries[name]
        with self.assertRaises(TypeError):
            entries[name] = b"tampered"
        with self.assertRaises(TypeError):
            del entries[name]
        self.assertIsInstance(entries[name], bytes)
        built[0][name] = b"builder alias mutation"
        self.assertEqual(entries[name], expected)
        verify_packed(self.target, manifest)
        self.assertIs(verified_entries(self.target, identity), entries)

    def test_repeated_verification_reuses_bytes_but_streams_payload_and_hashes_again(self):
        manifest = self.packed()
        identity = manifest["storage"]["archive_sha256"]
        verify_packed(self.target, manifest)
        entries = verified_entries(self.target, identity)
        original_open = zipfile.ZipFile.open
        original_read = zipfile.ZipExtFile.read
        opened = []
        reads = []
        def tracked_open(archive, name, *args, **kwargs):
            opened.append(name.filename if isinstance(name, zipfile.ZipInfo) else name)
            return original_open(archive, name, *args, **kwargs)
        def bounded_read(payload, size=-1):
            reads.append(size)
            return original_read(payload, size)
        with mock.patch.object(zipfile.ZipFile, "read", side_effect=AssertionError("reconstructed full payload")), \
                mock.patch.object(zipfile.ZipFile, "open", tracked_open), \
                mock.patch.object(zipfile.ZipExtFile, "read", bounded_read), \
                mock.patch.object(packed_module, "_hash_stream", wraps=packed_module._hash_stream) as hashed:
            verify_packed(self.target, manifest)
        self.assertEqual(hashed.call_count, 2)
        self.assertEqual(set(opened), set(entries))
        self.assertTrue(reads)
        self.assertTrue(all(0 < size <= 1024 * 1024 for size in reads))
        self.assertIs(verified_entries(self.target, identity), entries)
        for name, data in entries.items():
            self.assertIs(verified_entries(self.target, identity)[name], data)

    def test_reused_archive_same_size_restored_mtime_corruption_is_refused(self):
        manifest = self.packed()
        identity = manifest["storage"]["archive_sha256"]
        verify_packed(self.target, manifest)
        archive = self.target / "qlib_data.zip"
        stamp = archive.stat()
        original = archive.read_bytes()
        corrupted = bytearray(original)
        corrupted[40] ^= 1
        archive.write_bytes(corrupted)
        os.utime(archive, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        self.assertEqual(archive.stat().st_size, stamp.st_size)
        with self.assertRaisesRegex(DatasetError, "SHA-256 differs"):
            verify_packed(self.target, manifest)
        with self.assertRaises(DatasetError):
            verified_entries(self.target, identity)

    def test_reused_bytes_do_not_bypass_changed_logical_manifest_or_evidence(self):
        self.with_universe_hashes()
        manifest = self.packed()
        identity = manifest["storage"]["archive_sha256"]
        for key, value, error in (("qlib_binary_tree_sha256", "0" * 64, "logical tree"),
                                   ("qlib_binary_file_count", 5, "entry count")):
            verify_packed(self.target, manifest)
            altered = json.loads(json.dumps(manifest))
            altered[key] = value
            with self.assertRaisesRegex(DatasetError, error):
                verify_packed(self.target, altered)
            with self.assertRaises(DatasetError):
                verified_entries(self.target, identity)
        verify_packed(self.target, manifest)
        altered = json.loads(json.dumps(manifest))
        altered["universes"]["top"]["membership_sha256"] = "0" * 64
        with self.assertRaisesRegex(DatasetError, "universe evidence"):
            verify_packed(self.target, altered)
        with self.assertRaises(DatasetError):
            verified_entries(self.target, identity)

    def test_reject_archive_change_between_hash_passes(self):
        manifest = self.packed()
        identity = manifest["storage"]["archive_sha256"]
        verify_packed(self.target, manifest)
        original_hash = packed_module._hash_stream
        calls = 0

        def changed_second_pass(stream):
            nonlocal calls
            calls += 1
            result = original_hash(stream)
            return result if calls == 1 else "0" * 64

        with mock.patch.object(packed_module, "_hash_stream", side_effect=changed_second_pass):
            with self.assertRaisesRegex(DatasetError, "changed during verification"):
                verify_packed(self.target, manifest)
        self.assertEqual(calls, 2)
        with self.assertRaises(DatasetError):
            verified_entries(self.target, identity)

    def test_reject_local_header_and_payload_tampering_with_matching_hash(self):
        manifest = self.packed()
        archive = self.target / "qlib_data.zip"
        original = archive.read_bytes()
        with zipfile.ZipFile(archive) as inspected:
            first = inspected.infolist()[0]
            offset = first.header_offset
            name_length = len(first.filename.encode("utf-8"))
            data_offset = offset + 30 + name_length + len(first.extra)
        for position in (offset + 6, offset + 14, offset + 30, data_offset):
            altered = bytearray(original)
            altered[position] ^= 1
            archive.write_bytes(altered)
            manifest["storage"]["archive_sha256"] = hashlib.sha256(altered).hexdigest()
            with self.subTest(position=position), self.assertRaises(DatasetError):
                verify_packed(self.target, manifest)

    def test_reject_junctions_in_source_destination_and_runtime(self):
        junction = {self.source / "qlib_data" / "features",
                    self.root / "output_parent", self.target / "qlib_data.zip",
                    self.target / "qlib_data" / "instruments"}
        original = Path.is_junction

        def simulated(path):
            return path in junction or original(path)

        with mock.patch.object(Path, "is_junction", simulated):
            with self.assertRaises(DatasetError):
                pack_dataset(self.source, self.root / "output_parent" / "packed")
            with self.assertRaises(DatasetError):
                pack_dataset(self.source, self.target)
        manifest = self.packed()
        with mock.patch.object(Path, "is_junction", simulated):
            with self.assertRaises(DatasetError):
                verify_packed(self.target, manifest)
        junction.remove(self.target / "qlib_data.zip")
        with mock.patch.object(Path, "is_junction", simulated):
            with self.assertRaises(DatasetError):
                verify_packed(self.target, manifest)

    def test_reject_unsafe_zip_even_with_matching_archive_hash(self):
        manifest = self.packed()
        archive = self.target / "qlib_data.zip"
        with zipfile.ZipFile(archive, "a", compression=zipfile.ZIP_STORED) as out:
            out.writestr("../escape", b"x")
        manifest["storage"]["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
        manifest["storage"]["archive_size"] = archive.stat().st_size
        manifest["qlib_binary_file_count"] += 1
        with self.assertRaises(DatasetError):
            verify_packed(self.target, manifest)

    def test_reject_duplicate_and_malformed_offset(self):
        manifest = self.packed()
        archive = self.target / "qlib_data.zip"
        original = archive.read_bytes()
        with zipfile.ZipFile(archive, "a", compression=zipfile.ZIP_STORED) as out:
            out.writestr("calendars/day.txt", b"duplicate")
        manifest["storage"].update(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                                   archive_size=archive.stat().st_size)
        with self.assertRaises(DatasetError):
            verify_packed(self.target, manifest)
        altered = bytearray(original)
        central = altered.find(b"PK\x01\x02")
        self.assertGreaterEqual(central, 0)
        struct.pack_into("<I", altered, central + 42, len(altered) + 100)
        archive.write_bytes(altered)
        manifest["storage"].update(archive_sha256=hashlib.sha256(altered).hexdigest(),
                                   archive_size=len(altered))
        with self.assertRaises(DatasetError):
            verify_packed(self.target, manifest)

    def test_reinit_requires_matching_verified_snapshot(self):
        manifest = self.packed()
        verify_packed(self.target, manifest)
        self.assertIn("features/aapl/close.day.bin", verified_entries(self.target, manifest["storage"]["archive_sha256"]))
        with self.assertRaises(DatasetError):
            verified_entries(self.source, manifest["storage"]["archive_sha256"])
        second_source = self.root / "second_source"
        second_target = self.root / "second_packed"
        shutil.copytree(self.source, second_source)
        (second_source / "qlib_data" / "features" / "aapl" / "close.day.bin").write_bytes(
            struct.pack("<ffff", 1, 9, 10, 11))
        new_tree, new_count = tree_sha256(second_source / "qlib_data")
        second_manifest = json.loads((second_source / "manifest.json").read_text())
        second_manifest.update(qlib_binary_tree_sha256=new_tree, qlib_binary_file_count=new_count)
        (second_source / "manifest.json").write_text(json.dumps(second_manifest))
        pack_dataset(second_source, second_target)
        loaded = json.loads((second_target / "manifest.json").read_text())
        verify_packed(second_target, loaded)
        with self.assertRaises(DatasetError):
            feature_provider_config(self.target, manifest)
        self.assertEqual(feature_provider_config(second_target, loaded)["kwargs"]["dataset"],
                         str(second_target.resolve()))

    def test_provider_matches_local_feature_storage(self):
        try:
            import numpy as np
            import qlib
            from qlib.data.storage.file_storage import FileFeatureStorage
            from qlib_yahoo.packed_provider import PackedFeatureProvider
        except ImportError:
            self.skipTest("Qlib and numpy are optional on bare hosts")
        manifest = self.packed()
        verify_packed(self.target, manifest)
        qlib.init(provider_uri=str(self.source / "qlib_data"), region="us", kernels=1)
        provider = PackedFeatureProvider(str(self.target), manifest["storage"]["archive_sha256"])
        storage = FileFeatureStorage("aapl", "close", "day", provider_uri={"day": str(self.source / "qlib_data")})
        for start, end in ((0, 3), (1, 1), (2, 2), (3, 8), (8, 10)):
            got = provider.feature("AAPL", "$close", start, end, "day")
            want = storage[start:end + 1]
            self.assertEqual(got.index.tolist(), want.index.tolist())
            np.testing.assert_array_equal(got.to_numpy(), want.to_numpy())
            self.assertEqual(got.dtype, want.dtype)
        self.assertTrue(provider.feature("AAPL", "$missing", 0, 2, "day").empty)
        qlib.init(provider_uri=str(self.target / "qlib_data"), region="us", kernels=1,
                  feature_provider=feature_provider_config(self.target, manifest))
        from qlib.data.data import FeatureD
        self.assertEqual(FeatureD.feature("AAPL", "$close", 1, 3, "day").index.tolist(), [1, 2, 3])

    def test_rolling_expressions_match_native_after_reinit(self):
        try:
            import numpy as np
            import qlib
            from qlib.data.data import ExpressionD
        except ImportError:
            self.skipTest("Qlib and numpy are optional on bare hosts")
        qlib_dir = self.source / "qlib_data"
        (qlib_dir / "calendars" / "day.txt").write_text(
            "".join(f"2020-01-{day:02d}\n" for day in range(1, 9)))
        (qlib_dir / "instruments" / "all.txt").write_text(
            "AAPL\t2020-01-01\t2020-01-08\nNA\t2020-01-01\t2020-01-08\n")
        for symbol in ("aapl", "na"):
            folder = qlib_dir / "features" / symbol
            folder.mkdir(exist_ok=True)
            (folder / "close.day.bin").write_bytes(
                struct.pack("<fffffff", 2, 3, float("nan"), 5, 6, 7, 8))
            (folder / "open.day.bin").write_bytes(
                struct.pack("<fffffff", 2, 3, 4, float("nan"), 6, 7, 8))
        digest, count = tree_sha256(qlib_dir)
        self.manifest.update(qlib_binary_tree_sha256=digest, qlib_binary_file_count=count)
        (self.source / "manifest.json").write_text(json.dumps(self.manifest))
        fields = ("$close", "Mean($close,3)", "Ref($close,1)",
                  "Corr($close,$open,3)", "Rank($close,3)")
        def read():
            return {(symbol, field): ExpressionD.expression(symbol, field, "2020-01-02", "2020-01-08", "day")
                    for symbol in ("AAPL", "NA") for field in fields}
        qlib.init(provider_uri=str(qlib_dir), region="us", kernels=1, expression_cache=None)
        native = read()
        packed_manifest = self.packed()
        verify_packed(self.target, packed_manifest)
        qlib.init(provider_uri=str(self.target / "qlib_data"), region="us", kernels=1,
                  expression_cache=None, feature_provider=feature_provider_config(self.target, packed_manifest))
        packed = read()
        for key in native:
            self.assertEqual(packed[key].index.tolist(), native[key].index.tolist(), key)
            np.testing.assert_allclose(packed[key].to_numpy(), native[key].to_numpy(), equal_nan=True,
                                       err_msg=str(key))
        (qlib_dir / "features" / "na" / "close.day.bin").write_bytes(
            struct.pack("<fffffff", 2, 99, float("nan"), 5, 6, 7, 8))
        qlib.init(provider_uri=str(qlib_dir), region="us", kernels=1, expression_cache=None)
        reset = read()
        self.assertEqual(reset[("NA", "$close")].loc[2], 99)
        self.assertNotEqual(packed[("NA", "$close")].loc[2], 99)


if __name__ == "__main__":
    unittest.main()
