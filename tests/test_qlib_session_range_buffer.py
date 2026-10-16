"""Isolated buffer suffix correctness and explicitly requested V6 paired probe.

Never builds, installs, or changes default runtime bindings. The NEW reviewed
QLIB_RANGE_BUFFER_MANIFEST is required; QLIB_SESSION_LOOPS_MANIFEST alone cannot
activate it. Timing additionally requires QLIB_RANGE_BUFFER_RECEIPT and an old
V6 QLIB_SESSION_LOOPS_MANIFEST, and must be separately authorized.
"""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "QLIB_RANGE_BUFFER_MANIFEST"
CHILD = "QLIB_RANGE_BUFFER_CHILD"
RECEIPT = "QLIB_RANGE_BUFFER_RECEIPT"
SOURCE = "qlib_yahoo/session_range_buffer.pyx"


def load_buffer():
    """Use the existing reviewed test loader with a local variable override.

    No environment substitution, default loader modification or sys.modules
    eviction. CPython's path-specific extension load must preserve identity.
    """
    manifest_path = Path(os.environ[MANIFEST])
    receipt = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = receipt["source_sha256"]
    if SOURCE not in sources or "qlib_yahoo/session_loops.pyx" in sources:
        raise RuntimeError("Separate buffer source receipt required; V6 is not a buffer build")
    from tests import test_qlib_session_loops as original
    with patch.object(original, "MANIFEST_ENV", MANIFEST):
        module, seconds, verified = original.load_reviewed_extension()
    artifact = Path(verified["artifact"]["path"]).resolve()
    if Path(module.__file__).resolve() != artifact or Path(module.__spec__.origin).resolve() != artifact:
        raise RuntimeError("Buffer module artifact identity mismatch")
    if module.try_valid_range.__module__ != module.__name__:
        raise RuntimeError("Buffer function module identity mismatch")
    return module, seconds, verified


class SourceTests(unittest.TestCase):
    def test_public_buffer_source_is_unwired_and_releases_both_lease_scopes(self):
        source = (ROOT / SOURCE).read_text(encoding="utf-8")
        self.assertIn("PyBUF_SIMPLE", source)
        self.assertEqual(source.count("PyObject_GetBuffer("), 2)
        self.assertEqual(source.count("PyBuffer_Release(&lease)"), 2)
        self.assertEqual(source.count("finally:"), 2)
        self.assertNotIn("cimport numpy", source)
        self.assertNotIn("except Exception", source)
        self.assertNotIn("lease.itemsize", source)
        self.assertIn("if row == count:", source)

    def test_old_source_receipt_is_rejected_before_extension_loader_import(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            path.write_text(json.dumps({"source_sha256": {"qlib_yahoo/session_loops.pyx": "x"}}), encoding="utf-8")
            with patch.dict(os.environ, {MANIFEST: str(path)}):
                with self.assertRaisesRegex(RuntimeError, "Separate buffer"):
                    load_buffer()


class LauncherTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get(MANIFEST), "separate reviewed buffer build required")
    @unittest.skipIf(os.environ.get(CHILD) == "1", "isolated child")
    def test_verified_native_child(self):
        environment = os.environ.copy()
        environment[CHILD] = "1"
        program = (
            "from qlib_yahoo.runtime import ensure_signed_runtime; ensure_signed_runtime(); "
            "import unittest; r=unittest.TextTestRunner(verbosity=2).run("
            "unittest.defaultTestLoader.loadTestsFromName('tests.test_qlib_session_range_buffer.BufferChild')); "
            "raise SystemExit(not r.wasSuccessful())")
        result = subprocess.run([sys.executable, "-c", program], cwd=ROOT, env=environment,
                                capture_output=True, text=True, timeout=600)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


# Reuse the bounded native corpus: NaT/int64 extremes and all NumPy error
# policies, exact-domain declines, mutable/readonly layouts, prefix callbacks,
# lazy errors/cache, price scalar bits, and live boundary descriptors. Importing
# the corpus does not import Qlib or activate an extension.
from tests import test_qlib_cython_session_row as _row_corpus


class BufferChild(_row_corpus.RowChild):
    __unittest_skip__ = os.environ.get(CHILD) != "1"
    __unittest_skip_why__ = "requires isolated verified buffer child"

    @classmethod
    def setUpClass(cls):
        from tests.test_qlib_fast_signed import SignedChild
        SignedChild.setUpClass()
        cls.fixture = SignedChild()
        cls.extension, cls.load_seconds, cls.manifest = load_buffer()
        cls.native = staticmethod(cls.fixture.FastExchange._session_row)

    def test_leases_release_before_resize_and_next_call_sees_mutation(self):
        np = self.fixture.np
        # Own the datetime allocation: no prior integer view can retain a base.
        index = np.array([0, 1, 2], dtype="datetime64[ns]")
        start, end = index[1], index[1]
        self.assertIs(self.extension.try_valid_range(index, 1, start, end), True)
        index.resize((4,), refcheck=True)
        index.resize((3,), refcheck=True)
        index[0] = start
        self.assertIs(self.extension.try_valid_range(index, 1, start, end), False)
        # Both early rejection branches must also release the index export.
        for row, lower, upper in ((0, start, np.datetime64(0, "ns")),
                                  (1, start, end)):
            self.assertIs(self.extension.try_valid_range(index, row, lower, upper), False)
            index.resize((4,), refcheck=True)
            index.resize((3,), refcheck=True)
        del start, end
        index.resize((5,), refcheck=True)

    def test_live_class_attribute_hook_and_subclass_decline(self):
        f = self.fixture
        for subclass in (False, True):
            observed = []
            for selected in (False, True):
                exchange = f.exchange(True)
                if subclass:
                    class Derived(f.FastExchange):
                        pass
                    exchange.__class__ = Derived
                if selected:
                    exchange._range_validator = self.extension.try_valid_range
                    exchange.reset_session_cache()
                start, end = f.dates[0], f.dates[0] + f.pd.Timedelta(hours=23)
                exchange.begin_session(start, end)
                trace = []
                def attribute(instance, name):
                    if name in ("_session_start_ns", "_session_end_ns"):
                        trace.append(name)
                    return object.__getattribute__(instance, name)
                if subclass:
                    result = self.native(exchange, "AAA", start, end)
                else:
                    with patch.object(f.FastExchange, "__getattribute__", attribute):
                        result = self.native(exchange, "AAA", start, end)
                observed.append((result, trace))
                if selected:
                    self.assertEqual(exchange.range_counts, {"compiled_checks": 0, "native_checks": 1})
            self.exact(observed[1], observed[0])

    @unittest.skipUnless(os.environ.get(RECEIPT), "explicit paired timing not requested")
    def test_complete_row_probe(self):
        from tests.test_qlib_session_loops import load_reviewed_extension
        old, old_seconds, old_manifest = load_reviewed_extension()
        if ("qlib_yahoo/session_loops.pyx" not in old_manifest["source_sha256"]
                or SOURCE in old_manifest["source_sha256"]):
            raise RuntimeError("Old V6 comparison receipt required")
        old_path = Path(old_manifest["artifact"]["path"]).resolve()
        new_path = Path(self.manifest["artifact"]["path"]).resolve()
        if (old_path == new_path or old is self.extension
                or old.try_valid_range is self.extension.try_valid_range
                or Path(old.__file__).resolve() != old_path
                or Path(old.__spec__.origin).resolve() != old_path):
            raise RuntimeError("Independent old/new artifact identity required")
        path = Path(os.environ[RECEIPT]).absolute()
        with path.open("x", encoding="utf-8") as handle:
            handle.write('{"status":"reserved","goal_acceptance":false}\n')
            handle.flush(); os.fsync(handle.fileno())
        receipt = {"status": "running", "goal_acceptance": False,
                   "scope": "actual complete _session_row V6 versus buffer; no backtest/report",
                   "symbol_count": 3000, "index_days": 1255, "batches_per_half": 12,
                   "pairs": [], "manifests": {"v6": old_manifest, "buffer": self.manifest},
                   "load_seconds": {"v6": old_seconds, "buffer": self.load_seconds},
                   "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                     for name in (SOURCE, "qlib_yahoo/fast_signed.py",
                                                  "tests/test_qlib_session_range_buffer.py")},
                   "timed": "session reset/start, complete guarded caller, counters, uncached and cached calls",
                   "excluded": "fixture/loading, checks and receipt writes"}
        def persist():
            temporary = path.with_name(path.name + ".progress.tmp")
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(receipt, handle, indent=2); handle.write("\n")
                handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary, path)
        persist()
        try:
            f, pd = self.fixture, self.fixture.pd
            symbols = [f"S{i:04d}" for i in range(3000)]
            dates = pd.bdate_range("2021-01-01", periods=1255)
            index = pd.MultiIndex.from_product([symbols, dates], names=["instrument", "datetime"])
            frame = pd.DataFrame({"$open": 10., "$close": 11., "$volume": 123.,
                                  "$factor": 1., "limit_buy": False, "limit_sell": False}, index=index)
            exchange = f.exchange(True, frame)
            helpers = {"v6": old.try_valid_range, "buffer": self.extension.try_valid_range}
            for number in range(6):
                pair = {"number": number, "seconds": {}, "counts": {}}
                reference = None
                for name in (("v6", "buffer") if number % 2 == 0 else ("buffer", "v6")):
                    exchange._range_validator = helpers[name]
                    outputs = []
                    started = time.perf_counter()
                    for batch in range(receipt["batches_per_half"]):
                        start, end = dates[600 + batch], dates[600 + batch] + pd.Timedelta(hours=23)
                        exchange.reset_session_cache(); exchange.begin_session(start, end)
                        rows = [self.native(exchange, symbol, start, end) for symbol in symbols]
                        hits = [self.native(exchange, symbol, start, end) for symbol in symbols]
                        outputs.append((rows, hits, list(exchange._session_rows), exchange.range_counts.copy()))
                    pair["seconds"][name] = time.perf_counter() - started
                    for rows, hits, cache_keys, counts in outputs:
                        self.assertTrue(all(a is b for a, b in zip(rows, hits)))
                        self.assertEqual(cache_keys, symbols)
                        self.assertEqual(counts, {"compiled_checks": 3000, "native_checks": 0})
                    comparable = [(rows, keys, counts) for rows, hits, keys, counts in outputs]
                    if reference is not None: self.exact(comparable, reference)
                    reference = comparable
                    pair["counts"][name] = {"compiled_checks": 36000, "native_checks": 0}
                receipt["pairs"].append(pair); persist()
            receipt["median_seconds"] = {name: statistics.median(p["seconds"][name]
                for p in receipt["pairs"]) for name in helpers}
            receipt["buffer_to_v6_ratio"] = receipt["median_seconds"]["buffer"] / receipt["median_seconds"]["v6"]
            receipt["status"] = "completed"; persist()
        except (Exception, KeyboardInterrupt) as exc:
            receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            persist(); raise


if __name__ == "__main__":
    unittest.main()
