"""Owned-buffer bounds/failure cases and optional complete-write microprobe."""

import importlib.util
import unittest
from unittest.mock import Mock


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "optional PyArrow unavailable")
class BufferTests(unittest.TestCase):
    def setUp(self):
        import pyarrow as pa
        from qlib_yahoo.parquet_buffer import BufferedParquetBuffer
        self.pa, self.Buffer = pa, BufferedParquetBuffer
        self.schema = pa.schema([("instrument", pa.string()), ("value", pa.float64())],
                                metadata={b"pandas": b"fixture metadata"})
        self.writer = Mock(schema=self.schema)

    def table(self, count=3):
        return self.pa.Table.from_pydict({"instrument": ["Z", "雪🌍", "A"][:count],
                                         "value": [-0., float("nan"), None][:count]}, schema=self.schema)

    def test_rows_order_nulls_nan_and_metadata(self):
        buffer = self.Buffer(self.writer, target_rows=6)
        table = self.table()
        buffer.write_table(table)
        self.writer.write_table.assert_not_called()
        buffer.write_table(table)
        self.writer.write_table.assert_called_once()
        combined = self.writer.write_table.call_args.args[0]
        self.assertEqual(combined.schema.metadata, self.schema.metadata)
        self.assertEqual(combined["instrument"].to_pylist(), ["Z", "雪🌍", "A"] * 2)
        self.assertEqual(combined["value"].null_count, 2)
        self.assertEqual(self.writer.write_table.call_args.kwargs, {"row_group_size": 6})
        self.assertEqual(buffer.stats()["peak_pending_rows"], 6)
        self.assertEqual(buffer.stats()["pending_bytes"], 0)

    def test_owned_noncanonical_nan_infinity_subnormal_and_signed_zero_bits(self):
        import numpy as np
        bits = np.array([0x8000000000000000, 0x7FF8000000001234,
                         0x0000000000000001, 0xFFF0000000000000], dtype=np.uint64)
        table = self.pa.Table.from_arrays([self.pa.array(["Z", "A", "雪", "Q"]),
                                          self.pa.array(bits.view(np.float64))], schema=self.schema)
        buffer = self.Buffer(self.writer)
        buffer.write_table(table)
        expected = bits.copy()
        bits[:] = 0
        buffer.flush()
        copied = self.writer.write_table.call_args.args[0]["value"]
        self.assertEqual(copied.null_count, 0)
        np.testing.assert_array_equal(copied.to_numpy().view(np.uint64), expected)

    def test_owns_native_numpy_buffers_and_sliced_variable_width_buffers(self):
        import numpy as np
        values = np.array([-0., 1., 2.])
        table = self.pa.Table.from_arrays([self.pa.array(["Z", "雪🌍", "A"]), self.pa.array(values)], schema=self.schema)
        buffer = self.Buffer(self.writer)
        buffer.write_table(table.slice(1))
        values[:] = 999.
        del table
        buffer.flush()
        written = self.writer.write_table.call_args.args[0]
        self.assertEqual(written["value"].to_pylist(), [1., 2.])
        self.assertEqual(written["instrument"].to_pylist(), ["雪🌍", "A"])
        for column in written.columns:
            for chunk in column.chunks:
                self.assertTrue(all(not item.is_mutable for item in chunk.buffers() if item is not None))

    def test_flush_before_byte_or_row_overshoot_and_oversized_direct(self):
        table = self.table()
        size = sum(b.size for c in table.columns for chunk in c.chunks for b in chunk.buffers() if b)
        buffer = self.Buffer(self.writer, target_rows=4, max_bytes=size + 1)
        buffer.write_table(table)
        buffer.write_table(table)
        self.assertEqual(self.writer.write_table.call_count, 1)
        self.assertLessEqual(buffer.stats()["peak_pending_bytes"], size + 1)
        self.assertLessEqual(buffer.stats()["peak_pending_rows"], 4)
        buffer.flush()
        self.assertEqual(self.writer.write_table.call_count, 2)
        self.writer.reset_mock()
        direct = self.Buffer(self.writer, max_bytes=size - 1)
        direct.write_table(table)
        self.writer.write_table.assert_called_once_with(table)
        self.assertEqual(direct.stats()["pending_bytes"], 0)
        self.assertEqual(direct.stats()["direct_tables"], 1)
        self.writer.reset_mock()
        bytes_only = self.Buffer(self.writer, target_rows=100, max_bytes=size + 1)
        bytes_only.write_table(table)
        bytes_only.write_table(table)
        self.assertEqual(self.writer.write_table.call_count, 1)
        self.assertEqual(bytes_only.stats()["pending_rows"], 3)
        self.assertLessEqual(bytes_only.stats()["peak_pending_bytes"], size + 1)

    def test_zero_rows_are_explicit_ordered_writes(self):
        buffer = self.Buffer(self.writer)
        buffer.write_table(self.table())
        buffer.write_table(self.table(0))
        buffer.write_table(self.table(0))
        self.assertEqual([len(call.args[0]) for call in self.writer.write_table.call_args_list], [3, 0, 0])
        self.assertEqual(buffer.stats()["empty_tables"], 2)
        buffer.flush()
        self.assertEqual(self.writer.write_table.call_count, 3)

    def test_partial_write_failure_retains_queue_and_never_retries(self):
        buffer = self.Buffer(self.writer)
        buffer.write_table(self.table())
        before = buffer.stats()
        failure = OSError("partial write")
        self.writer.write_table.side_effect = failure
        with self.assertRaises(OSError) as caught:
            buffer.flush()
        self.assertIs(caught.exception, failure)
        self.assertEqual(buffer.stats()["pending_bytes"], before["pending_bytes"])
        self.assertEqual(buffer.stats()["pending_rows"], 3)
        for call in (buffer.flush, lambda: buffer.write_table(self.table())):
            with self.assertRaisesRegex(RuntimeError, "refusing"):
                call()
        self.assertEqual(self.writer.write_table.call_count, 1)
        buffer.discard()
        self.assertEqual(buffer.stats()["pending_bytes"], 0)
        self.assertTrue(buffer.stats()["failed"])
        self.writer.close.assert_not_called()

    def test_discard_never_writes_or_owns_writer_close(self):
        buffer = self.Buffer(self.writer)
        buffer.write_table(self.table())
        buffer.discard()
        buffer.flush()
        self.writer.write_table.assert_not_called()
        self.writer.close.assert_not_called()

    def test_mutable_string_offsets_and_values_are_owned(self):
        import numpy as np
        characters = bytearray(b"ZA")
        offsets = np.array([0, 1, 2], dtype=np.int32)
        labels = self.pa.Array.from_buffers(self.pa.string(), 2,
            [None, self.pa.py_buffer(offsets), self.pa.py_buffer(characters)])
        table = self.pa.Table.from_arrays([labels, self.pa.array([1., 2.])], schema=self.schema)
        buffer = self.Buffer(self.writer)
        buffer.write_table(table)
        characters[:] = b"QQ"
        offsets[1] = 0
        buffer.flush()
        self.assertEqual(self.writer.write_table.call_args.args[0]["instrument"].to_pylist(), ["Z", "A"])

    def test_direct_write_failure_fails_closed_without_retaining_oversized_ram(self):
        buffer = self.Buffer(self.writer, max_bytes=1)
        self.writer.write_table.side_effect = OSError("partial direct write")
        with self.assertRaises(OSError):
            buffer.write_table(self.table())
        self.assertTrue(buffer.stats()["failed"])
        self.assertEqual(buffer.stats()["pending_bytes"], 0)
        with self.assertRaisesRegex(RuntimeError, "refusing"):
            buffer.flush()
        self.assertEqual(self.writer.write_table.call_count, 1)

    def test_schema_metadata_types_and_configuration_fail_before_queue(self):
        for options in ({"target_rows": True}, {"target_rows": 0}, {"max_bytes": -1}, {"max_bytes": 1.5}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.Buffer(self.writer, **options)
        buffer = self.Buffer(self.writer)
        with self.assertRaises(ValueError):
            buffer.write_table(self.table().replace_schema_metadata({b"changed": b"metadata"}))
        nested = self.pa.table({"nested": [[1], [2]]})
        with self.assertRaises(TypeError):
            self.Buffer(Mock(schema=nested.schema)).write_table(nested)
        self.writer.write_table.assert_not_called()

    def test_real_parquet_decoded_bits_units_and_empty_rows(self):
        import numpy as np
        import pyarrow.parquet as pq
        for unit in ("s", "ms", "us", "ns"):
            schema = self.pa.schema([("datetime", self.pa.timestamp(unit)), *list(self.schema)],
                                    metadata=self.schema.metadata)
            table = self.pa.Table.from_pydict({"datetime": np.array([1, 2, 3], dtype=np.int64).view(f"datetime64[{unit}]"),
                "instrument": ["Z", "雪🌍", "A"], "value": [-0., float("nan"), None]}, schema=schema)
            tables = [table, table, table.slice(0, 0), table]
            decoded = []
            for fast in (False, True):
                sink = self.pa.BufferOutputStream()
                with pq.ParquetWriter(sink, schema, compression="zstd") as writer:
                    buffer = self.Buffer(writer, target_rows=8) if fast else writer
                    for part in tables:
                        buffer.write_table(part)
                    if fast:
                        buffer.flush()
                decoded.append(pq.read_table(self.pa.BufferReader(sink.getvalue())))
            old, new = decoded
            self.assertTrue(old.schema.equals(new.schema, check_metadata=True))
            self.assertEqual(old["datetime"].to_pylist(), new["datetime"].to_pylist())
            self.assertEqual(old["instrument"].to_pylist(), new["instrument"].to_pylist())
            self.assertEqual(old["value"].null_count, new["value"].null_count)
            np.testing.assert_array_equal(old["value"].to_numpy().view(np.uint64), new["value"].to_numpy().view(np.uint64))


def microprobe(output):
    """Verified native tables: conversion/cast/copy/guards/write/flush/close."""
    import json
    import statistics
    import time
    from pathlib import Path
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pandas as pd
    from qlib_yahoo.parquet_buffer import BufferedParquetBuffer
    from qlib_yahoo.streaming_columns import order_indicator_table, position_table
    from tests.test_qlib_streaming_columns import native_updated_fixture, reference, position_schema_and_shared
    from qlib.backtest.signed_position import SignedPosition
    output = Path(output)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("receipt must be new in an existing directory")
    day = pd.Timestamp("2025-01-02").as_unit("ms")
    raw = native_updated_fixture([f"SYM{i:04d}" for i in range(2999, -1, -1)])
    order_schema = reference(raw, day).schema
    pos_schema, shared = position_schema_and_shared()
    position = SignedPosition(cash=1_000_000.)
    for i in range(1499, -1, -1):
        position._init_stock(f"SYM{i:04d}", float(i % 23 + 1) * (-1 if i % 2 else 1), float(i + 1) / 10.)
    result = {"scope": "synthetic 20 native sessions; complete conversion/cast/copy/guards/concat/zstd-write/flush/close; not full worker evidence",
              "pyarrow": pa.__version__, "sessions": 20, "position_rows_per_session": 1500,
              "order_rows_per_session": 3000, "order_metrics": 9, "trials": []}
    for repeat in range(3):
        for buffered in ((False, True) if repeat % 2 == 0 else (True, False)):
            position_path = output.parent / f"{output.stem}-{repeat}-{buffered}-positions.parquet"
            order_path = output.parent / f"{output.stem}-{repeat}-{buffered}-orders.parquet"
            started = time.perf_counter()
            stats = []
            with pq.ParquetWriter(position_path, pos_schema, compression="zstd") as pw, \
                    pq.ParquetWriter(order_path, order_schema, compression="zstd") as ow:
                pb = BufferedParquetBuffer(pw) if buffered else pw
                ob = BufferedParquetBuffer(ow) if buffered else ow
                for i in range(20):
                    current = day + pd.Timedelta(days=i)
                    pb.write_table(position_table(position, {**shared, "datetime": current}, 1_000_000., pos_schema).cast(pos_schema))
                    ob.write_table(order_indicator_table(raw, current, order_schema).cast(order_schema))
                if buffered:
                    pb.flush()
                    ob.flush()
                    stats = [pb.stats(), ob.stats()]
            elapsed = time.perf_counter() - started
            result["trials"].append({"buffered": buffered, "seconds": elapsed, "stats": stats,
                                    "positions": str(position_path), "orders": str(order_path),
                                    "position_row_groups": pq.ParquetFile(position_path).metadata.num_row_groups,
                                    "order_row_groups": pq.ParquetFile(order_path).metadata.num_row_groups})
            if repeat == 0 and buffered:
                import numpy as np
                for name in ("positions", "orders"):
                    old = pq.read_table(result["trials"][0][name])
                    new = pq.read_table(result["trials"][1][name])
                    if not old.schema.equals(new.schema, check_metadata=True):
                        raise AssertionError("decoded schema changed")
                    for field in old.schema:
                        if field.type == pa.float64():
                            np.testing.assert_array_equal(old[field.name].to_numpy().view(np.uint64), new[field.name].to_numpy().view(np.uint64))
                            assert old[field.name].null_count == new[field.name].null_count
                        else:
                            assert old[field.name].to_pylist() == new[field.name].to_pylist()
                result["first_pair_decoded_parity"] = "exact schema/metadata/all rows/non-null and decoded float bits/null counts"
    result["median_control_seconds"] = statistics.median(t["seconds"] for t in result["trials"] if not t["buffered"])
    result["median_buffered_seconds"] = statistics.median(t["seconds"] for t in result["trials"] if t["buffered"])
    result["boundary_improved"] = result["median_buffered_seconds"] < result["median_control_seconds"]
    with output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2, allow_nan=False)
    print(json.dumps({key: result[key] for key in ("median_control_seconds", "median_buffered_seconds", "boundary_improved")}))


if __name__ == "__main__":
    unittest.main()
