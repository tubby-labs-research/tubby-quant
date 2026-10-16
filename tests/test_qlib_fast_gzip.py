"""Bounded isolated codec tests; optional native cases require QLIB_ISAL_PATH.

No full backtest or timing probe. Both codecs use the same stdlib JSON encoder.
"""
import gzip
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import hashlib
import statistics
import tempfile
import time
from datetime import datetime, timezone
from functools import partial
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "qlib_yahoo/fast_gzip.py"
spec = importlib.util.spec_from_file_location("private_fast_gzip_test", SOURCE)
codec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codec)


def encode(factory, records):
    raw = io.BytesIO()
    with factory(fileobj=raw) as packed:
        with io.TextIOWrapper(packed, encoding="utf-8") as text:
            text.write("[")
            for index, record in enumerate(records):
                encoded = json.dumps(record, allow_nan=False)
                text.write((", " if index else "") + encoded)
            text.write("]")
    return raw.getvalue()


class FactoryTests(unittest.TestCase):
    def fallback(self, level=2):
        with patch.object(codec, "_verified_install", return_value=(Path("isolated"), Path("site"), {})), \
                patch.object(codec, "_supported_abi", return_value=False), \
                patch.object(codec, "_load_verified") as load:
            result = codec.activate_gzip("isolated", level)
            load.assert_not_called()
            return result

    def test_unsupported_abi_fallback_before_write(self):
        factory, identity = self.fallback()
        self.assertEqual(identity["requested_level"], 2)
        self.assertEqual(identity["actual_level"], 6)
        self.assertEqual(identity["backend"], "stdlib-zlib")
        self.assertFalse(identity["level_equivalent_to_zlib"])
        self.assertEqual(gzip.decompress(encode(factory, [{"x": "中文", "f": -0.0}])),
                         b'[{"x": "\\u4e2d\\u6587", "f": -0.0}]')

    def test_invalid_level_before_install_or_import(self):
        for value in (True, False, -1, 4, 2.0, "2", None):
            with self.subTest(value=value), patch.object(codec, "_verified_install") as verify:
                with self.assertRaises(ValueError):
                    codec.activate_gzip("irrelevant", value)
                verify.assert_not_called()

    def test_missing_request_does_not_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                codec.activate_gzip(Path(temp) / "absent")

    def test_fixed_header_level_and_mode_before_write(self):
        factory, _ = self.fallback()
        for changes in ({"filename": "named"}, {"mtime": 1}, {"mtime": False},
                        {"mode": "ab"}, {"compresslevel": 2}, {"compresslevel": True}):
            raw = io.BytesIO()
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                factory(fileobj=raw, **changes)
            self.assertEqual(raw.getvalue(), b"")
        with self.assertRaises(ValueError):
            factory()

    def test_nonfinite_native_encoder_rejects_before_separator(self):
        factory, _ = self.fallback()
        raw = io.BytesIO()
        with self.assertRaises(ValueError):
            with factory(fileobj=raw) as packed:
                with io.TextIOWrapper(packed, encoding="utf-8") as text:
                    text.write('[{"valid": 1}')
                    encoded = json.dumps({"bad": float("nan")}, allow_nan=False)
                    text.write(", " + encoded)
        self.assertEqual(gzip.decompress(raw.getvalue()), b'[{"valid": 1}')
        self.assertFalse(raw.closed)

    def test_write_failure_propagates_no_codec_retry(self):
        class Broken(io.BytesIO):
            def write(self, data):
                raise OSError("sink failed")
        factory, _ = self.fallback()
        with self.assertRaisesRegex(OSError, "sink failed"), patch.object(codec, "_load_verified") as load:
            factory(fileobj=Broken())
        load.assert_not_called()


@unittest.skipUnless(os.environ.get("QLIB_ISAL_PATH") and codec._supported_abi(),
                     "native tests require explicit isolated ISA-L path and CP312 Windows x64")
class NativeWheelTests(unittest.TestCase):
    def activate(self, level=2):
        return codec.activate_gzip(os.environ["QLIB_ISAL_PATH"], level)

    def test_default_level_and_exact_decompressed_bytes(self):
        factory, identity = self.activate()
        self.assertEqual(identity["backend"], "ISA-L")
        self.assertEqual(identity["actual_level"], 2)
        self.assertEqual(identity["version"], "1.8.0")
        self.assertEqual(identity["wheel_sha256"], codec.WHEEL_SHA256)
        self.assertIn("isal/isa-l/LICENSE", identity["installed_file_sha256"])
        records = [{"unicode": "中文😀", "float": -0.0, "huge": 2**100,
                    "none": None, "nested": [{"x": 1.2345678901234567}]}] * 2
        native = lambda **kwargs: gzip.GzipFile(filename="", mode="wb", mtime=0, compresslevel=6, **kwargs)
        self.assertEqual(gzip.decompress(encode(factory, records)), gzip.decompress(encode(native, records)))

    def test_all_levels_and_deterministic_header(self):
        for level in range(4):
            factory, identity = self.activate(level)
            result = encode(factory, [])
            self.assertEqual(identity["actual_level"], level)
            self.assertEqual(gzip.decompress(result), b"[]")
            self.assertEqual(result[4:8], b"\0" * 4)
            self.assertFalse(result[3] & 8)  # No filename field.
            self.assertEqual(result, encode(factory, []))

    def test_native_nonfinite_and_exception_cleanup(self):
        factory, _ = self.activate()
        for value in (float("nan"), float("inf"), float("-inf")):
            raw = io.BytesIO()
            with self.assertRaises(ValueError):
                with factory(fileobj=raw) as packed:
                    with io.TextIOWrapper(packed, encoding="utf-8") as text:
                        text.write("[")
                        text.write(json.dumps({"bad": value}, allow_nan=False))
            self.assertEqual(gzip.decompress(raw.getvalue()), b"[")
            self.assertFalse(raw.closed)
        raw = io.BytesIO()
        with self.assertRaises(KeyboardInterrupt):
            with factory(fileobj=raw) as packed:
                packed.write(b"prefix")
                raise KeyboardInterrupt()
        self.assertEqual(gzip.decompress(raw.getvalue()), b"prefix")

    def test_installed_record_wheel_extra_file_tamper_fail_closed(self):
        root = Path(os.environ["QLIB_ISAL_PATH"])
        for relative in (codec.WHEEL_NAME, "site-packages/isal/igzip.py",
                         "site-packages/isal-1.8.0.dist-info/RECORD", "site-packages/extra.py"):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as temp:
                target = Path(temp) / "copy"
                shutil.copytree(root, target)
                item = target / relative
                with item.open("ab") as out:
                    out.write(b"tampered")
                with self.assertRaises(ValueError):
                    codec.activate_gzip(target)

    def test_binding_mutation_rejected_before_output(self):
        factory, _ = self.activate()
        igzip = codec._loaded[1]["isal.igzip"]
        raw = io.BytesIO()
        with patch.object(igzip, "IGzipFile", gzip.GzipFile):
            with self.assertRaises(RuntimeError):
                factory(fileobj=raw)
            with self.assertRaises(RuntimeError):
                self.activate()
        self.assertEqual(raw.getvalue(), b"")


@unittest.skipUnless(os.environ.get("QLIB_ISAL_PATH") and codec._supported_abi(),
                     "writer tests require explicit isolated ISA-L path and verified signed runtime")
class NativeWriterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from qlib_yahoo.runtime import ensure_signed_runtime
        ensure_signed_runtime()
        from tests.test_qlib_decision_compression import native_sessions
        cls.snapshots = native_sessions(stocks=6, sessions=1)

    def writer_session(self, path):
        from qlib_yahoo.streaming import SignedHistoryWriter
        day, original, original_record = self.snapshots[0]
        account = copy.copy(original)
        account.indicator = copy.copy(original.indicator)
        account.indicator.order_indicator_his = dict(original.indicator.order_indicator_his)
        record = copy.deepcopy(original_record)
        strategy = SimpleNamespace(decisions=[record])
        factory, _ = codec.activate_gzip(os.environ["QLIB_ISAL_PATH"], 2)
        writer = SignedHistoryWriter(path, strategy, columnar=True, buffered=True,
                                     decision_gzip_level=2, _gzip_factory=factory)
        return writer, account, strategy, record, day

    def test_native_encoding_write_retention_and_abort(self):
        for failure in ("encoding", "write"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                path = Path(temp)
                writer, account, strategy, record, day = self.writer_session(path)
                writer.positions(account, day)
                if failure == "encoding":
                    record["target_weights"]["bad"] = float("nan")
                    with patch.object(writer._decisions, "write", wraps=writer._decisions.write) as write:
                        with self.assertRaises(ValueError):
                            writer.finish_session(account, day)
                        write.assert_not_called()
                else:
                    with patch.object(writer._decisions, "write", side_effect=OSError("injected write")):
                        with self.assertRaisesRegex(OSError, "injected write"):
                            writer.finish_session(account, day)
                self.assertEqual(writer.sessions, 0)
                self.assertIs(strategy.decisions[0], record)
                self.assertIn(day, account.indicator.order_indicator_his)
                writer.close(completed=False)
                self.assertTrue(writer._raw.closed)
                self.assertEqual(list(path.iterdir()), [])
                self.assertTrue(all(buffer.stats()["pending_bytes"] == 0 for buffer in writer._buffers))

    def test_native_close_failure_never_promotes_and_constructor_cleans(self):
        from qlib_yahoo.streaming import SignedHistoryWriter
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            writer, account, strategy, _, day = self.writer_session(path)
            writer.positions(account, day)
            writer.finish_session(account, day)
            original = writer._gzip.close
            def failed_close():
                original()
                raise OSError("injected close")
            with patch.object(writer._gzip, "close", side_effect=failed_close):
                with self.assertRaisesRegex(OSError, "injected close"):
                    writer.close(completed=True)
            self.assertTrue(writer.closed and writer._raw.closed)
            self.assertEqual(list(path.iterdir()), [])
            self.assertTrue(all(buffer.stats()["pending_bytes"] == 0 for buffer in writer._buffers))
            writer.close(completed=False)
            factory, _ = codec.activate_gzip(os.environ["QLIB_ISAL_PATH"], 2)
            with self.assertRaises(ValueError):
                SignedHistoryWriter(path, strategy, columnar=True, buffered=True,
                                    decision_gzip_level=3, _gzip_factory=factory)
            self.assertEqual(list(path.iterdir()), [])


def _float_bits_equal(left, right):
    import pyarrow as pa
    import pyarrow.parquet as pq
    for name in ("positions.parquet", "qlib_order_indicators.parquet"):
        first, second = pq.read_table(left / name), pq.read_table(right / name)
        if not first.schema.equals(second.schema, check_metadata=True):
            raise AssertionError(f"schema/metadata changed: {name}")
        for field in first.schema:
            if pa.types.is_floating(field.type):
                a = first[field.name].combine_chunks().to_numpy(zero_copy_only=False)
                b = second[field.name].combine_chunks().to_numpy(zero_copy_only=False)
                if a.dtype != b.dtype or a.tobytes() != b.tobytes():
                    raise AssertionError(f"float bits changed: {name}/{field.name}")


def complete_writer_probe(output, *, isolated_path, pairs=5, stocks=1500, sessions=20):
    """Explicit synthetic diagnostic, never ordinary/full-job speed evidence.

    Actual native fixture generation and activation are separate. Each timed
    variant calls the SAME reviewed writer harness, with only its explicitly
    provided gzip factory changed. One continuous stream per variant.
    """
    if type(pairs) is not int or not 1 <= pairs <= 5:
        raise ValueError("bounded probe requires one through five pairs")
    if type(stocks) is not int or not 1 <= stocks <= 1500:
        raise ValueError("bounded fixture requires one through 1500 stocks")
    if type(sessions) is not int or not 1 <= sessions <= 20:
        raise ValueError("bounded fixture requires one through 20 sessions")
    output = Path(output).absolute()
    codec._plain_path(output.parent, directory=True)
    progress = {"schema": 1, "kind": "synthetic_native_complete_writer_codec_boundary",
                "status": "running", "phase": "reserved", "goal_achieved": False,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "stocks": stocks, "sessions": sessions, "pairs": [],
                "timing_boundary": "Same writer construction, positions conversion, JSON conversion/encoding, casts, buffered writes, flush, close and promotion; fixture preparation/activation and parity outside timing",
                "limitations": "Synthetic native snapshots; not full-backtest acceptance or exact ordinary-runtime savings."}
    # Keep ownership of the exclusively created receipt handle throughout.
    with output.open("x", encoding="utf-8") as receipt:
        def persist():
            payload = json.dumps(progress, indent=2, allow_nan=False)
            receipt.seek(0)
            receipt.write(payload)
            receipt.truncate()
            receipt.flush()
            os.fsync(receipt.fileno())
        persist()
        try:
            from qlib_yahoo.runtime import ensure_signed_runtime
            progress["phase"] = "runtime_validation"
            persist()
            progress["engine"] = ensure_signed_runtime()
            from tests.test_qlib_decision_compression import native_sessions, write_sessions, verify_outputs
            import qlib_yahoo.streaming as streaming
            progress["source_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (SOURCE, Path(__file__), Path(streaming.__file__),
                             SOURCE.parent / "signed.py",
                             Path(__file__).with_name("test_qlib_decision_compression.py"))}
            progress["phase"] = "activation_outside_timing"
            persist()
            start = time.perf_counter()
            factory, identity = codec.activate_gzip(isolated_path, 2)
            progress["activation_seconds"] = time.perf_counter() - start
            progress["implementation"] = identity
            if identity["backend"] != "ISA-L":
                raise RuntimeError("diagnostic requires actual ISA-L, not ABI fallback")
            progress["phase"] = "native_fixture_preparation_outside_timing"
            persist()
            start = time.perf_counter()
            snapshots = native_sessions(stocks=stocks, sessions=sessions)
            progress["fixture_seconds"] = time.perf_counter() - start
            progress["orders_per_session"] = [len(record["issued_orders"]) for _, _, record in snapshots]
            actual_writer = streaming.SignedHistoryWriter
            variants = {"stdlib6": (6, None), "stdlib1": (1, None), "isal2": (2, factory)}
            with tempfile.TemporaryDirectory(dir=output.parent, prefix="isal-writer-") as directory:
                for pair in range(pairs):
                    order = (["stdlib6", "stdlib1", "isal2"] if pair % 2 == 0
                             else ["isal2", "stdlib1", "stdlib6"])
                    progress["phase"] = f"pair_{pair}"
                    persist()
                    paths, results = {}, {}
                    for variant in order:
                        level, gzip_factory = variants[variant]
                        path = Path(directory) / f"pair-{pair}-{variant}"
                        path.mkdir()
                        paths[variant] = path
                        progress["phase"] = f"pair_{pair}_{variant}"
                        persist()
                        selected_writer = (actual_writer if gzip_factory is None
                                           else partial(actual_writer, _gzip_factory=gzip_factory))
                        with patch.object(streaming, "SignedHistoryWriter", selected_writer):
                            elapsed, writer = write_sessions(path, level, snapshots)
                        results[variant] = {"seconds": elapsed, "actual_level": level,
                            "gzip_bytes": (path / "signed_decisions.json.gz").stat().st_size,
                            "rows": writer.rows, "sessions": writer.sessions,
                            "closed": writer.closed, "raw_closed": writer._raw.closed,
                            "buffer_stats": writer.buffered_stats()}
                    parity = {}
                    for variant in ("stdlib1", "isal2"):
                        parity[variant] = verify_outputs(paths["stdlib6"], paths[variant])
                        _float_bits_equal(paths["stdlib6"], paths[variant])
                        parity[variant]["exact_table_float_bits"] = True
                    progress["pairs"].append({"pair": pair, "order": order,
                                              "results": results, "parity": parity})
                    persist()
            progress["median_seconds"] = {name: statistics.median(
                pair["results"][name]["seconds"] for pair in progress["pairs"]) for name in variants}
            progress.update(status="completed", phase="completed",
                            completed_utc=datetime.now(timezone.utc).isoformat())
            persist()
            return progress
        except BaseException as exc:
            progress.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                            failed_utc=datetime.now(timezone.utc).isoformat())
            persist()
            raise


if __name__ == "__main__":
    unittest.main()
