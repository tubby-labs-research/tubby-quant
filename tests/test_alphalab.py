"""Alpha Lab: FASTEXPR parsing, operators against naive references, causality and the simulator's arithmetic."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HAVE_PANDAS = bool(importlib.util.find_spec("pandas") and importlib.util.find_spec("numpy"))

if HAVE_PANDAS:
    import numpy as np


@unittest.skipUnless(HAVE_PANDAS, "numpy and pandas are optional outside the Qlib environment")
class LibraryConnectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "library.sqlite"
        self.connections = []  # Keep handles alive: closure must not rely on garbage collection.
        self.addCleanup(lambda: [db.close() for db in self.connections])
        connect = sqlite3.connect

        def tracked_connect(*args, **kwargs):
            db = connect(*args, **kwargs)
            self.connections.append(db)
            return db

        tracker = patch("qlib_yahoo.alphalab.batch.sqlite3.connect", side_effect=tracked_connect)
        tracker.start()
        self.addCleanup(tracker.stop)

    def assert_connections_closed(self, count):
        self.assertEqual(len(self.connections), count)
        for db in self.connections:
            with self.assertRaisesRegex(sqlite3.ProgrammingError, "closed database"):
                db.execute("SELECT 1")

    def test_every_library_operation_closes_its_connection(self):
        from qlib_yahoo.alphalab.batch import Library
        from qlib_yahoo.alphalab.simulate import Settings

        library = Library(self.path)
        self.assert_connections_closed(1)
        result = {"expression": "rank(close)", "settings": {"universe": "TOP3000", "delay": 1},
                  "daily": {"dates": ["2024-01-02"], "pnl": [1.0]}}
        library.save("alpha", result, "dataset")
        self.assert_connections_closed(2)
        self.assertEqual(library.get("alpha"), result)  # The write committed before closing.
        self.assert_connections_closed(3)
        self.assertEqual([row["id"] for row in library.rows()], ["alpha"])
        self.assert_connections_closed(4)
        self.assertEqual(library.pnl_series("dataset", Settings(), set()),
                         {"alpha": (["2024-01-02"], [1.0])})
        self.assert_connections_closed(5)
        self.assertIsNone(library.get("missing"))
        self.assert_connections_closed(6)
        self.path.unlink()  # Windows must release the file even while handles are retained.

    def test_failed_schema_initialization_closes_its_connection(self):
        from qlib_yahoo.alphalab.batch import Library

        with patch("qlib_yahoo.alphalab.batch.SCHEMA", "invalid SQL"), self.assertRaises(sqlite3.OperationalError):
            Library(self.path)
        self.assert_connections_closed(1)
        self.path.unlink()

    def test_failed_write_rolls_back_and_closes_its_connection(self):
        from qlib_yahoo.alphalab.batch import Library

        library = Library(self.path)
        # FAIL preserves the trigger's earlier insert unless the transaction is rolled back.
        db = library.connect()
        try:
            db.executescript("""
                CREATE TABLE audit (id TEXT);
                CREATE TRIGGER reject_alpha BEFORE INSERT ON alphas BEGIN
                    INSERT INTO audit VALUES (NEW.id);
                    SELECT RAISE(FAIL, 'rejected alpha');
                END;
            """)
        finally:
            db.close()
        with self.assertRaisesRegex(sqlite3.IntegrityError, "rejected alpha"):
            library.save("alpha", {"expression": "rank(close)"}, "dataset")
        self.assert_connections_closed(3)
        db = library.connect()
        try:
            self.assertEqual(db.execute("SELECT * FROM audit").fetchall(), [])
        finally:
            db.close()
        self.assert_connections_closed(4)

    def test_failed_result_decode_closes_read_connections(self):
        from qlib_yahoo.alphalab.batch import Library
        from qlib_yahoo.alphalab.simulate import Settings

        library = Library(self.path)
        library.save("alpha", {"expression": "rank(close)",
                               "settings": {"universe": "TOP3000", "delay": 1}}, "dataset")
        with self.assertRaises(KeyError):
            library.pnl_series("dataset", Settings(), set())
        self.assert_connections_closed(3)


@unittest.skipUnless(HAVE_PANDAS, "numpy and pandas are optional outside the Qlib environment")
class CachedWorkspaceLifetimeTests(unittest.TestCase):
    def setUp(self):
        from qlib_yahoo.alphalab.batch import write_cache

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.maps = []  # Retain arrays and their views so GC cannot release the file locks.
        self.addCleanup(self.close_maps)
        dates = np.arange("2024-01-01", "2024-02-10", dtype="datetime64[D]")
        close = np.column_stack((20 + np.arange(40), 30 + np.arange(40) * 0.5)).astype(np.float32)
        self.source = SimpleNamespace(dataset=self.root / "snapshot", dates=dates, symbols=["AAA", "BBB"],
                                      identity={"dataset": str(self.root / "snapshot"),
                                                "qlib_binary_tree_sha256": "synthetic-cache"},
                                      fields={"close": close, "open": close.copy()}, groups={},
                                      panel=SimpleNamespace(manifest={}),
                                      universe=lambda _name: np.ones(close.shape, dtype=bool))
        self.directory = write_cache(self.source, self.root / "cache", ["TOP2"])
        identity = {**self.source.identity, "source_sha256": "synthetic-source", "symbols": self.source.symbols,
                    "dates": [str(d) for d in dates], "fields": sorted(self.source.fields)}
        source_tracker = patch("qlib_yahoo.alphalab.batch.source_identity", return_value=identity)
        source_tracker.start()
        self.addCleanup(source_tracker.stop)
        load = np.load

        def tracked_load(*args, **kwargs):
            array = load(*args, **kwargs)
            if isinstance(array, np.memmap):
                self.maps.append(array)
            return array

        tracker = patch("qlib_yahoo.alphalab.batch.np.load", side_effect=tracked_load)
        tracker.start()
        self.addCleanup(tracker.stop)

    def close_maps(self):
        for array in self.maps:
            if not array._mmap.closed:
                array._mmap.close()

    def assert_maps_closed(self):
        self.assertTrue(self.maps)
        self.assertTrue(all(array._mmap.closed for array in self.maps))

    def test_close_releases_retained_maps_and_views(self):
        from qlib_yahoo.alphalab.batch import CachedWorkspace

        workspace = CachedWorkspace(self.directory)
        view = workspace.fields["close"][:2]
        self.assertEqual(float(view[0, 0]), 20.0)
        workspace.close()
        workspace.close()  # Closing twice is safe.
        self.assert_maps_closed()
        self.assertTrue(view._mmap.closed)
        for path in self.directory.glob("*.npy"):
            path.unlink()  # Immediate Windows file release, with references still alive.

    def test_partial_load_failure_closes_maps_already_opened(self):
        from qlib_yahoo.alphalab.batch import CachedWorkspace
        from qlib_yahoo.dataset import DatasetError

        (self.directory / "universe_TOP2.npy").write_bytes(b"corrupt")
        with self.assertRaisesRegex(DatasetError, "bytes differ"):
            CachedWorkspace(self.directory)
        self.assertEqual(len(self.maps), 2)
        self.assert_maps_closed()

    def test_missing_array_is_refused_before_mapping(self):
        from qlib_yahoo.alphalab.batch import CachedWorkspace
        from qlib_yahoo.dataset import DatasetError
        (self.directory / "universe_TOP2.npy").unlink()
        with self.assertRaisesRegex(DatasetError, "inventory"):
            CachedWorkspace(self.directory)
        self.assertFalse(self.maps)

    def test_single_worker_batches_close_maps_without_changing_worker_global(self):
        from qlib_yahoo.alphalab import batch
        from qlib_yahoo.alphalab.simulate import Settings

        sentinel = object()
        settings = Settings(universe="TOP2", warmup=5, truncation=0)
        with patch.object(batch, "Workspace", return_value=self.source), patch.object(batch, "_WORKSPACE", sentinel):
            for expression in ("rank(returns)", "rank(nope)"):
                with self.subTest(expression=expression):
                    result = batch.run_batch(self.source.dataset, [expression], settings,
                                             data_root=self.root / "data", workers=1, force=True)
                    self.assertEqual("error" in result[0], expression == "rank(nope)")
                    self.assert_maps_closed()
                    self.assertIs(batch._WORKSPACE, sentinel)
        for path in (self.root / "data" / "cache" / "alphalab").rglob("*.npy"):
            path.unlink()

    def test_progress_failure_closes_single_worker_maps(self):
        from qlib_yahoo.alphalab import batch
        from qlib_yahoo.alphalab.simulate import Settings

        def fail_after_simulation(message):
            if message.startswith("[1/1]"):
                raise RuntimeError("progress failed")

        with patch.object(batch, "Workspace", return_value=self.source):
            with self.assertRaisesRegex(RuntimeError, "progress failed"):
                batch.run_batch(self.source.dataset, ["rank(returns)"], Settings(universe="TOP2", warmup=5),
                                data_root=self.root / "data", workers=1, progress=fail_after_simulation)
        self.assert_maps_closed()

    def test_corrupt_bytes_are_refused_even_when_timestamp_is_preserved(self):
        import os
        from qlib_yahoo.alphalab.batch import CachedWorkspace, write_cache
        from qlib_yahoo.dataset import DatasetError
        path = self.directory / "field_close.npy"
        stamp = path.stat()
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        with self.assertRaisesRegex(DatasetError, "bytes differ"):
            CachedWorkspace(self.directory)
        with self.assertRaisesRegex(DatasetError, "bytes differ"):
            write_cache(self.source, self.root / "cache", ["TOP2"])

    def test_metadata_symbol_order_and_dtype_are_verified(self):
        from qlib_yahoo.alphalab.batch import CachedWorkspace
        from qlib_yahoo.dataset import DatasetError
        path = self.directory / "meta.json"
        original = path.read_text()
        for key, value in (("symbols", ["BBB", "AAA"]), ("version", 0)):
            meta = json.loads(original)
            meta[key] = value
            path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(DatasetError, "metadata"):
                CachedWorkspace(self.directory)
        path.write_text(original)
        np.save(self.directory / "universe_TOP2.npy", np.ones((40, 2), dtype=np.float32))
        with self.assertRaisesRegex(DatasetError, "bytes differ"):
            CachedWorkspace(self.directory)
        self.assert_maps_closed()

    def test_universe_generations_are_immutable_and_isolated(self):
        from qlib_yahoo.alphalab.batch import CachedWorkspace, write_cache
        first_meta = (self.directory / "meta.json").read_bytes()
        second = write_cache(self.source, self.root / "cache", ["TOP1"])
        self.assertNotEqual(second, self.directory)
        self.assertEqual((self.directory / "meta.json").read_bytes(), first_meta)
        from contextlib import closing
        with closing(CachedWorkspace(second)) as ws:
            self.assertEqual(sorted(ws._universes), ["TOP1"])
        self.source.universe = lambda _name: np.zeros((40, 2), dtype=bool)
        changed = write_cache(self.source, self.root / "cache", ["TOP2"])
        self.assertNotEqual(changed, self.directory)
        self.assert_maps_closed()

    def test_nested_research_cache_uses_full_content_digest_and_short_staging(self):
        from contextlib import closing
        from qlib_yahoo.alphalab.batch import CachedWorkspace, cache_dir, write_cache
        from qlib_yahoo.path_utils import cache_io_path

        root = self.root / "runtime" / "backend" / ("a" * 64)
        base = cache_dir(root, self.source)
        mkdtemp = tempfile.mkdtemp
        with patch("qlib_yahoo.alphalab.batch.tempfile.mkdtemp", wraps=mkdtemp) as staging:
            directory = write_cache(self.source, base, ["TOP2"])
        meta = json.loads((directory / "meta.json").read_text())
        self.assertEqual(directory.name, "v2-" + meta["content_sha256"])
        self.assertEqual(len(meta["content_sha256"]), 64)
        self.assertEqual(staging.call_args.kwargs["prefix"], ".tmp-")
        self.assertEqual(staging.call_args.kwargs["dir"], cache_io_path(base.parent))
        legacy = base.with_name(base.name + "-" + meta["content_sha256"])
        self.assertEqual(len(directory.name), 67)
        self.assertLess(len(directory.name), len(legacy.name))
        if os.name == "nt":
            self.assertLess(len(str(directory / "universe_TOP2.npy")), 260)
            self.assertGreater(len(str(legacy / "universe_TOP2.npy")), 260)
        with closing(CachedWorkspace(directory)) as workspace:
            np.testing.assert_array_equal(workspace.fields["close"], self.source.fields["close"])

    def test_compact_generation_separates_source_identity_and_array_contents(self):
        from qlib_yahoo.alphalab.batch import write_cache

        original = (self.directory / "meta.json").read_bytes()
        self.source.identity = {**self.source.identity, "source_sha256": "other-source"}
        other_source = write_cache(self.source, self.root / "cache", ["TOP2"])
        self.source.fields["close"] = self.source.fields["close"].copy()
        self.source.fields["close"][0, 0] += 1
        other_arrays = write_cache(self.source, self.root / "cache", ["TOP2"])
        self.assertEqual(len({self.directory, other_source, other_arrays}), 3)
        self.assertEqual((self.directory / "meta.json").read_bytes(), original)
        self.assertEqual(write_cache(self.source, self.root / "another-base", ["TOP2"]), other_arrays)

    def test_accessible_legacy_generation_is_read_and_reused_with_validation(self):
        from contextlib import closing
        from qlib_yahoo.alphalab.batch import CachedWorkspace, write_cache
        from qlib_yahoo.dataset import DatasetError

        meta = json.loads((self.directory / "meta.json").read_text())
        legacy = self.root / ("cache-" + meta["content_sha256"])
        self.directory.rename(legacy)
        with closing(CachedWorkspace(legacy)) as workspace:
            np.testing.assert_array_equal(workspace.fields["close"], self.source.fields["close"])
        with patch("qlib_yahoo.alphalab.batch.tempfile.mkdtemp", side_effect=AssertionError("rewritten")):
            self.assertEqual(write_cache(self.source, self.root / "cache", ["TOP2"]), legacy)
        self.assertFalse(self.directory.exists())
        (legacy / "field_close.npy").write_bytes(b"corrupt")
        with self.assertRaisesRegex(DatasetError, "bytes differ"):
            write_cache(self.source, self.root / "cache", ["TOP2"])

    def test_inaccessible_legacy_path_does_not_block_compact_generation(self):
        from qlib_yahoo.alphalab.batch import write_cache

        exists = Path.exists
        base = self.root / "cache"
        def legacy_path_error(path):
            if path.name.startswith("cache-"):
                raise OSError("legacy path too long")
            return exists(path)

        with patch.object(Path, "exists", legacy_path_error):
            directory = write_cache(self.source, base, ["TOP1"])
        self.assertTrue((directory / "meta.json").is_file())

    def test_repeated_expressions_run_once_and_library_hit_does_not_allocate(self):
        from qlib_yahoo.alphalab import batch
        from qlib_yahoo.alphalab.simulate import Settings
        settings = Settings(universe="TOP2", warmup=5, truncation=0)
        with patch.object(batch, "Workspace", return_value=self.source), patch.object(batch, "_run_one", wraps=batch._run_one) as run:
            results = batch.run_batch(self.source.dataset, ["rank(returns)", "rank(returns)"], settings,
                                      data_root=self.root / "data", workers=1)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(len(results), 2)
            self.assertIsNot(results[0], results[1])
        with patch.object(batch, "Workspace", side_effect=AssertionError("allocated")):
            results = batch.run_batch(self.source.dataset, ["rank(returns)"] * 2, settings,
                                      data_root=self.root / "data", workers=1)
        self.assertTrue(all(r["cached"] for r in results))

    def test_worker_budget_and_ceiling_bound_parallelism(self):
        from qlib_yahoo.alphalab.batch import worker_count
        with patch("qlib_yahoo.alphalab.batch._available_ram", return_value=1024**3), patch("os.cpu_count", return_value=64):
            self.assertEqual(worker_count(64, 64, self.source, 4), 2)
        with patch("qlib_yahoo.alphalab.batch._available_ram", return_value=128 * 1024**3), patch("os.cpu_count", return_value=64):
            self.assertEqual(worker_count(64, 64, self.source, 3), 3)

    def test_failed_generation_does_not_publish_or_change_existing_cache(self):
        from qlib_yahoo.alphalab.batch import write_cache
        original = (self.directory / "meta.json").read_bytes()
        with patch("qlib_yahoo.alphalab.batch.np.save", side_effect=OSError("disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                write_cache(self.source, self.root / "cache", ["TOP1"])
        self.assertEqual((self.directory / "meta.json").read_bytes(), original)
        self.assertFalse(list(self.root.glob("*.tmp-*")))

    def test_invalid_shapes_and_groups_are_refused_before_publish(self):
        from qlib_yahoo.alphalab.batch import write_cache
        from qlib_yahoo.dataset import DatasetError
        self.source.groups = {"sector": np.array([0.0])}
        with self.assertRaisesRegex(DatasetError, "shapes"):
            write_cache(self.source, self.root / "invalid-cache", ["TOP2"])
        self.assertEqual(list(self.root.glob("v2-*")), [self.directory])


@unittest.skipUnless(HAVE_PANDAS, "numpy and pandas are optional outside the Qlib environment")
class AlphaSourceIdentityTests(unittest.TestCase):
    def setUp(self):
        from qlib_yahoo.dataset import tree_sha256
        import hashlib
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dataset = Path(self.tmp.name)
        calendar = self.dataset / "qlib_data/calendars/day.txt"
        instruments = self.dataset / "qlib_data/instruments/all.txt"
        calendar.parent.mkdir(parents=True)
        instruments.parent.mkdir(parents=True)
        calendar.write_text("2024-01-01\n2024-01-02\n")
        instruments.write_text("AAA\t2024-01-01\t2024-01-02\nBBB\t2024-01-01\t2024-01-02\n")
        self.membership = self.dataset / "universe/membership.csv.gz"
        self.membership.parent.mkdir()
        self.membership.write_bytes(b"frozen-membership-evidence")
        self.groups = self.dataset / "universe/groups.csv"
        self.groups.write_text("symbol,sector,industry,country\nAAA,Tech,Software,US\nBBB,Finance,Bank,US\n")
        self.manifest = {"status": "ready", "provider": "fixture", "fields": ["close", "open"],
                         "qlib_binary_tree_sha256": tree_sha256(self.dataset / "qlib_data")[0],
                         "groups": {"file": "universe/groups.csv", "sha256": hashlib.sha256(self.groups.read_bytes()).hexdigest()},
                         "universes": {"top2": {"size": 2, "membership_csv": "universe/membership.csv.gz",
                                                  "membership_sha256": hashlib.sha256(self.membership.read_bytes()).hexdigest()}}}
        self.save_manifest()

    def save_manifest(self):
        (self.dataset / "manifest.json").write_text(json.dumps(self.manifest))

    def test_provider_field_set_and_auxiliary_evidence_change_result_identity(self):
        import hashlib
        from qlib_yahoo.alphalab.panel import source_identity
        original = source_identity(self.dataset)["source_sha256"]
        for key, value in (("provider", "other-provider"), ("fields", ["close"])):
            old = self.manifest[key]
            self.manifest[key] = value
            self.save_manifest()
            self.assertNotEqual(source_identity(self.dataset)["source_sha256"], original)
            self.manifest[key] = old
        self.membership.write_bytes(b"different-frozen-membership")
        self.manifest["universes"]["top2"]["membership_sha256"] = hashlib.sha256(self.membership.read_bytes()).hexdigest()
        self.save_manifest()
        self.assertNotEqual(source_identity(self.dataset)["source_sha256"], original)
        second = source_identity(self.dataset)["source_sha256"]
        self.groups.write_text(self.groups.read_text().replace("Tech", "Other"))
        self.manifest["groups"]["sha256"] = hashlib.sha256(self.groups.read_bytes()).hexdigest()
        self.save_manifest()
        self.assertNotEqual(source_identity(self.dataset)["source_sha256"], second)

    def test_auxiliary_corruption_is_refused_even_on_library_lookup_path(self):
        from qlib_yahoo.alphalab.panel import source_identity
        from qlib_yahoo.dataset import DatasetError
        self.membership.write_bytes(b"corrupt")
        with self.assertRaisesRegex(DatasetError, "membership"):
            source_identity(self.dataset)

    def test_calendar_and_symbol_order_are_part_of_identity(self):
        from qlib_yahoo.alphalab.panel import source_identity
        from qlib_yahoo.dataset import tree_sha256
        first = source_identity(self.dataset)
        instruments = self.dataset / "qlib_data/instruments/all.txt"
        instruments.write_text("\n".join(reversed(instruments.read_text().splitlines())) + "\n")
        self.manifest["qlib_binary_tree_sha256"] = tree_sha256(self.dataset / "qlib_data")[0]
        self.save_manifest()
        second = source_identity(self.dataset)
        self.assertEqual(second["symbols"], ["BBB", "AAA"])
        self.assertNotEqual(first["source_sha256"], second["source_sha256"])
        (self.dataset / "qlib_data/calendars/day.txt").write_text("2024-01-01\n2024-01-03\n")
        self.manifest["qlib_binary_tree_sha256"] = tree_sha256(self.dataset / "qlib_data")[0]
        self.save_manifest()
        self.assertNotEqual(second["source_sha256"], source_identity(self.dataset)["source_sha256"])


class ParserTests(unittest.TestCase):
    def test_precedence_and_forms(self):
        from qlib_yahoo.alphalab.fastexpr import Binary, Call, Name, Num, Ternary, Unary, parse

        program = parse("-a + b * c ^ 2 ^ 3")
        self.assertEqual(program.result, Binary("+", Unary("-", Name("a")),
                                                Binary("*", Name("b"), Binary("^", Name("c"), Binary("^", Num(2), Num(3))))))
        ternary = parse("close > open && volume > 0 ? 1 : -1").result
        self.assertIsInstance(ternary, Ternary)
        self.assertEqual(ternary.condition.op, "&&")
        call = parse("ts_regression(close, open, 20, lag = 1, rettype=2)").result
        self.assertEqual(call, Call("ts_regression", (Name("close"), Name("open"), Num(20.0)),
                                    (("lag", Num(1.0)), ("rettype", Num(2.0)))))
        many = parse("a = rank(close); b = a * 2;\n a - b")
        self.assertEqual([name for name, _ in many.assignments], ["a", "b"])
        self.assertEqual(parse("x = rank(close);").result, Call("rank", (Name("close"),)))
        self.assertEqual(parse('bucket(rank(close), range="0,1,0.25")').result.kwargs[0][1].value, "0,1,0.25")

    def test_errors_name_a_position(self):
        from qlib_yahoo.alphalab.fastexpr import ExpressionError, parse

        for text, message in (("", "empty"), ("rank(close", "expected ')'"), ("close $ open", "position 7"),
                              ("rank(x=1, close)", "positional argument after keyword"), ("a ? b", "expected ':'"),
                              ("(" * 300 + "1" + ")" * 300, "nested too deeply")):
            with self.subTest(text=text[:20]), self.assertRaisesRegex(ExpressionError, re.escape(message)):
                parse(text)


@unittest.skipUnless(HAVE_PANDAS, "numpy and pandas are optional outside the Qlib environment")
class OperatorTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.x = rng.normal(size=(60, 7)).astype(np.float32)
        self.y = rng.normal(size=(60, 7)).astype(np.float32)
        self.x[5, 2] = np.nan
        self.members = np.ones_like(self.x, dtype=bool)
        self.members[:, 6] = False

    def naive_ts(self, fn, x, d):
        out = np.full(x.shape, np.nan)
        for t in range(d - 1, len(x)):
            for j in range(x.shape[1]):
                window = x[t - d + 1:t + 1, j].astype(np.float64)
                if not np.isnan(window).any():
                    out[t, j] = fn(window)
        return out

    def close(self, got, want):
        np.testing.assert_allclose(np.asarray(got, dtype=np.float64), want, rtol=2e-5, atol=2e-6, equal_nan=True)

    def test_time_series_against_loops(self):
        from qlib_yahoo.alphalab import ops

        d = 5
        self.close(ops.ts_mean(self.x, d), self.naive_ts(np.mean, self.x, d))
        self.close(ops.ts_sum(self.x, d), self.naive_ts(np.sum, self.x, d))
        self.close(ops.ts_std_dev(self.x, d), self.naive_ts(lambda w: np.std(w, ddof=1), self.x, d))
        self.close(ops.ts_max(self.x, d), self.naive_ts(np.max, self.x, d))
        self.close(ops.ts_rank(self.x, d), self.naive_ts(lambda w: (np.sum(w < w[-1]) + (np.sum(w == w[-1]) - 1) / 2) / (d - 1), self.x, d))
        self.close(ops.ts_arg_max(self.x, d), self.naive_ts(lambda w: d - 1 - int(np.argmax(w)), self.x, d))
        weights = np.arange(1, d + 1, dtype=np.float64)
        self.close(ops.ts_decay_linear(self.x, d), self.naive_ts(lambda w: np.dot(w, weights) / weights.sum(), self.x, d))
        delta = self.x.astype(np.float64).copy()
        delta[3:] = self.x[3:] - self.x[:-3]
        delta[:3] = np.nan
        self.close(ops.ts_delta(self.x, 3), delta)
        pair = np.stack([self.x, self.y], axis=2)
        corr = np.full(self.x.shape, np.nan)
        for t in range(d - 1, len(self.x)):
            for j in range(self.x.shape[1]):
                a, b = pair[t - d + 1:t + 1, j, 0], pair[t - d + 1:t + 1, j, 1]
                if not (np.isnan(a).any() or np.isnan(b).any()):
                    corr[t, j] = np.corrcoef(a, b)[0, 1]
        self.close(ops.ts_corr(self.x, self.y, d), corr)
        slope = self.naive_ts(lambda w: 0, self.x, d)
        for t in range(d - 1, len(self.x)):
            for j in range(self.x.shape[1]):
                a, b = self.y[t - d + 1:t + 1, j].astype(float), self.x[t - d + 1:t + 1, j].astype(float)
                if not (np.isnan(a).any() or np.isnan(b).any()):
                    slope[t, j] = np.polyfit(b, a, 1)[0]
        self.close(ops.ts_regression(self.y, self.x, d, rettype=2), slope)

    def test_cross_section_against_loops(self):
        from qlib_yahoo.alphalab import ops

        rank = ops.rank(self.x, self.members)
        for t in (0, 5, 40):
            row = self.x[t, :6].astype(np.float64)
            ok = ~np.isnan(row)
            order = np.argsort(np.argsort(row[ok]))
            want = np.full(7, np.nan)
            want[np.flatnonzero(ok)] = order / (ok.sum() - 1)
            self.close(rank[t], want)
        z = ops.zscore(self.x, self.members)
        row = self.x[9, :6].astype(np.float64)
        self.close(z[9, :6], (row - row.mean()) / row.std())
        self.assertTrue(np.isnan(z[:, 6]).all())  # outside the universe
        scaled = ops.scale(self.x, self.members)
        self.close(np.nansum(np.abs(scaled), axis=1), np.ones(60))
        groups = np.array([0, 0, 1, 1, 1, np.nan, 0])
        neutral = ops.group_neutralize(self.x, groups, self.members)
        self.close(np.nansum(neutral[:, [0, 1]], axis=1), np.zeros(60))
        self.assertTrue(np.isnan(neutral[:, 5]).all())  # unknown group

    def test_turnover_operators(self):
        from qlib_yahoo.alphalab import ops

        x = np.array([[1.0], [1.001], [2.0], [2.0005]], dtype=np.float32)
        self.close(ops.hump(x, hump=0.01)[:, 0], [1.0, 1.0, 2.0, 2.0])
        trigger = np.array([[1], [0], [0], [1]], dtype=np.float32)
        exit = np.array([[0], [0], [1], [0]], dtype=np.float32)
        alpha = np.array([[5], [6], [7], [8]], dtype=np.float32)
        self.close(ops.trade_when(trigger, alpha, exit)[:, 0], [5, 5, np.nan, 8])


@unittest.skipUnless(HAVE_PANDAS, "numpy and pandas are optional outside the Qlib environment")
class CausalityTests(unittest.TestCase):
    """Changing data after day k must leave every alpha value up to day k unchanged."""

    EXPRESSIONS = [
        "-rank(ts_delta(close, 5))", "ts_corr(rank(close), rank(volume), 10)", "ts_rank(returns, 20)",
        "group_neutralize(zscore(ts_mean(returns, 5)), sector)", "ts_decay_linear(rank(vwap), 8)",
        "ts_regression(returns, ts_delay(returns, 1), 15, rettype=0)", "hump(rank(close), hump=0.05)",
        "trade_when(volume > adv20 / close, -returns, returns > 0.1)", "ts_backfill(close, 3) / ts_max(high, 10)",
        "a = ts_arg_max(close, 10); b = ts_product(1 + returns, 5); winsorize(a * b, std=2)",
        "bucket(rank(volume), range=\"0,1,0.25\")", "vector_neut(rank(returns), rank(volume))",
    ]

    def context(self, rng, k=None):
        from qlib_yahoo.alphalab.engine import Context

        shape = (80, 12)
        base = np.exp(np.cumsum(rng.normal(0, 0.02, shape), axis=0)) * 20
        fields = {"close": base, "open": base * 0.99, "high": base * 1.02, "low": base * 0.98,
                  "volume": np.abs(rng.normal(1e6, 2e5, shape))}
        fields = {name: value.astype(np.float32) for name, value in fields.items()}
        if k is not None:
            for value in fields.values():
                value[k + 1:] *= np.float32(1.7)
        members = np.ones(shape, dtype=bool)
        members[:, 11] = False
        return Context(dates=np.arange(shape[0]), symbols=[f"S{i}" for i in range(shape[1])], fields=fields,
                       members=members, groups={"sector": np.array([0, 1, 2] * 4, dtype=float)})

    def test_no_expression_reads_the_future(self):
        from qlib_yahoo.alphalab.engine import evaluate

        k = 50
        for expression in self.EXPRESSIONS:
            with self.subTest(expression=expression):
                full = evaluate(expression, self.context(np.random.default_rng(3)))
                changed = evaluate(expression, self.context(np.random.default_rng(3), k=k))
                np.testing.assert_array_equal(full[:k + 1], changed[:k + 1])

    def test_errors_are_explained(self):
        from qlib_yahoo.alphalab.engine import evaluate
        from qlib_yahoo.alphalab.fastexpr import ExpressionError

        context = self.context(np.random.default_rng(1))
        for expression, message in (("rank(cap)", "unknown data field 'cap'"), ("foo(close)", "unknown operator"),
                                    ("ts_mean(close, 2.5)", "whole number"), ("ts_mean(close, close)", "must be a number"),
                                    ("group_neutralize(close, industry)", "not available"), ("1 + 2", "constant"),
                                    ("rank = 1; rank", "operator name")):
            with self.subTest(expression=expression), self.assertRaisesRegex(ExpressionError, message):
                evaluate(expression, context)


@unittest.skipUnless(HAVE_PANDAS, "numpy and pandas are optional outside the Qlib environment")
class SimulatorArithmeticTests(unittest.TestCase):
    def test_weights_pipeline(self):
        from qlib_yahoo.alphalab.simulate import Settings, weights

        alpha = np.array([[1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 1, np.nan]], dtype=float)
        members = np.ones_like(alpha, dtype=bool)
        tradable = np.ones_like(alpha, dtype=bool)
        w = weights(alpha, members, tradable, Settings(delay=1, truncation=0, neutralization="MARKET"), {})
        self.assertTrue((w[0] == 0).all())  # nothing to hold before the first delayed alpha
        np.testing.assert_allclose(w[1], [-0.375, -0.125, 0.125, 0.375])  # day-0 alpha, demeaned, gross 1
        capped = weights(np.array([[1, 1, 1, 10.0]]), np.ones((1, 4), bool), np.ones((1, 4), bool),
                         Settings(delay=0, neutralization="NONE", truncation=0.4), {})
        self.assertLessEqual(np.abs(capped).max(), 0.4 + 1e-9)
        self.assertAlmostEqual(float(np.abs(capped).sum()), 1.0)
        outside = weights(alpha, np.array([[True, True, False, True]] * 3), tradable,
                          Settings(delay=0, truncation=0, neutralization="NONE"), {})
        self.assertTrue((outside[:, 2] == 0).all())  # pasteurized

    def test_metrics_formulas(self):
        from qlib_yahoo.alphalab.simulate import _stats

        pnl = np.array([100.0, -50.0, 200.0, 0.0, 50.0])
        traded = np.array([0.2, 0.4, 0.2, 0.2, 0.0])
        s = _stats(pnl, traded, 1000.0)
        mean, std = pnl.mean(), pnl.std(ddof=1)
        self.assertAlmostEqual(s["sharpe"], mean / std * math.sqrt(252))
        self.assertAlmostEqual(s["returns"], mean * 252 / 500)
        self.assertAlmostEqual(s["turnover"], 0.2)
        self.assertAlmostEqual(s["fitness"], s["sharpe"] * math.sqrt(abs(s["returns"]) / 0.2))
        self.assertAlmostEqual(s["margin_bps"], 300.0 / (1.0 * 1000.0) * 1e4)
        self.assertAlmostEqual(s["drawdown"], 50.0 / 500)


def _have_importer():
    try:
        from test_qlib_stooq_table import HAVE_ARROW, REAL_DUMPER
    except ImportError:
        return False
    return HAVE_ARROW and bool(importlib.util.find_spec("qlib")) and REAL_DUMPER.is_file()


@unittest.skipUnless(_have_importer(), "Qlib environment and the pinned upstream dump_bin checkout are required")
class WorkspaceAndBatchTests(unittest.TestCase):
    """The simulator on a real imported snapshot: universes, a hand-checked PnL, the shared cache and the library."""

    @classmethod
    def setUpClass(cls):
        from unittest.mock import patch

        from test_qlib_stooq_table import REAL_DUMPER, synthetic_table

        from qlib_yahoo.stooq_table import prepare_stooq_table
        from qlib_yahoo.universe import UniverseRule

        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name).resolve()
        frame, cls.days, _ = synthetic_table()
        table = cls.root / "source" / "us_daily.parquet"
        table.parent.mkdir()
        frame.to_parquet(table, index=False)
        with patch("qlib_yahoo.stooq_table._dump_script", return_value=REAL_DUMPER):
            cls.snapshot = prepare_stooq_table(table, cls.root / "data", "2023-01-01", "2025-01-01",
                                               rule=UniverseRule(name="top4", size=4, adv_sessions=10, min_history_sessions=40),
                                               max_workers=2, progress=lambda _m: None)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_universes_match_the_instrument_file(self):
        from qlib_yahoo.alphalab.simulate import Workspace
        from qlib_yahoo.panel import membership_mask

        workspace = Workspace(self.snapshot)
        np.testing.assert_array_equal(workspace.universe("TOP4"), membership_mask(workspace.panel, "top4"))
        top2 = workspace.universe("TOP2")
        self.assertTrue((top2 <= workspace.universe("TOP4")).all())
        self.assertLessEqual(int(top2.sum(axis=1).max()), 2)

    def test_pnl_is_weights_times_next_returns(self):
        from qlib_yahoo.alphalab.simulate import Settings, Workspace, simulate

        workspace = Workspace(self.snapshot)
        result = simulate("rank(returns)", Settings(universe="TOP4", truncation=0, warmup=20), workspace, keep_weights=True)
        close = workspace.fields["close"].astype(np.float64)
        first = workspace.window(Settings(universe="TOP4", warmup=20).validate())[1]
        returns = np.nan_to_num(close[first:] / close[first - 1:-1] - 1)
        w = result["weights"]
        # PnL of test day t uses the weights set at the close of day t-1 (the last warm-up day for t = start)
        previous = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])
        manual = np.sum(previous[1:] * returns[1:], axis=1) * 20_000_000.0
        np.testing.assert_allclose(result["daily"]["pnl"][1:], manual, rtol=1e-6, atol=1e-3)

    def test_plain_and_gzip_membership_match_and_remain_hash_guarded(self):
        import gzip
        import hashlib
        from qlib_yahoo.alphalab.simulate import Workspace
        from qlib_yahoo.dataset import DatasetError

        workspace = Workspace(self.snapshot)
        original = workspace.panel.manifest["universes"]["top4"]
        compressed = (self.snapshot / original["membership_csv"]).read_bytes()
        plain = gzip.decompress(compressed)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "membership.csv"
            path.write_bytes(plain)
            record = {"membership_csv": str(path), "membership_sha256": hashlib.sha256(plain).hexdigest()}
            import pandas as pd
            pd.testing.assert_frame_equal(workspace._membership("plain", record), workspace._membership("gzip", original))
            path.write_bytes(plain + b"\n")
            with self.assertRaisesRegex(DatasetError, "differs from the dataset manifest"):
                workspace._membership("plain", record)

    def test_historical_universe_uses_spans_and_consecutive_valid_history(self):
        import hashlib
        from types import SimpleNamespace
        from qlib_yahoo.alphalab.simulate import Workspace
        from qlib_yahoo.dataset import DatasetError

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "membership.csv"
            path.write_text("instrument,effective_start,effective_end\nA,2024-01-01,2024-01-06\n")
            workspace = Workspace.__new__(Workspace)
            workspace.dataset = Path(directory)
            workspace.dates = np.arange("2024-01-01", "2024-01-07", dtype="datetime64[D]")
            workspace.symbols = ["A"]
            workspace._universes = {}
            workspace.fields = {name: np.full((6, 1), 10.0) for name in ("open", "high", "low", "close", "volume")}
            workspace.fields["volume"][3] = np.nan
            record = {"mode": "historical_index", "size": 500, "min_history_sessions": 2,
                      "membership_csv": "membership.csv", "membership_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            workspace.panel = SimpleNamespace(manifest={"universes": {"sp500_history": record}})
            np.testing.assert_array_equal(workspace.universe("SP500_HISTORY")[:, 0], [False, True, True, False, False, True])
            with self.assertRaisesRegex(DatasetError, "no liquidity ranking"):
                workspace.universe("TOP200")

    def test_batch_cache_library_and_errors(self):
        from qlib_yahoo.alphalab.batch import Library, run_batch
        from qlib_yahoo.alphalab.simulate import Settings

        settings = Settings(universe="TOP4", truncation=0, warmup=20)
        alphas = ["rank(returns)", "-rank(returns)", "ts_mean(returns, 5)", "rank(nope)"]
        first = run_batch(self.snapshot, alphas, settings, data_root=self.root / "data", workers=2)
        self.assertEqual([("error" in r) for r in first], [False, False, False, True])
        self.assertIn("unknown data field", first[3]["error"])
        self.assertAlmostEqual(first[0]["summary"]["sharpe"], -first[1]["summary"]["sharpe"], places=6)
        again = run_batch(self.snapshot, ["rank(returns)", "0 - rank(returns) + 1"], settings,
                          data_root=self.root / "data", workers=1)
        self.assertTrue(again[0].get("cached"))
        self.assertAlmostEqual(again[1]["self_correlation"], 1.0, places=6)  # same PnL as -rank(returns)
        self.assertEqual(next(c for c in again[1]["checks"] if c["name"] == "SELF_CORRELATION")["result"], "FAIL")
        rows = Library(self.root / "data" / "alphalab" / "library.sqlite").rows(limit=10)
        self.assertEqual(len(rows), 5)
        json.dumps(first[0], allow_nan=False)


if __name__ == "__main__":
    unittest.main()
