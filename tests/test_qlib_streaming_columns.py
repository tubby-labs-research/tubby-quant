"""Native serialization differentials, isolated from ordinary Qlib imports."""

import importlib.util
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from qlib_yahoo.streaming_columns import order_indicator_table, position_table

CHILD = "QUANT_STREAMING_COLUMNS_CHILD"


class FallbackTests(unittest.TestCase):
    def test_no_native_import_or_custom_callback(self):
        class Callback:
            @property
            def data(self):
                raise AssertionError("custom callback invoked")
        with patch.dict(sys.modules, {"qlib.backtest.high_performance_ds": None,
                                     "qlib.utils.index_data": None}):
            self.assertIs(order_indicator_table(Callback(), object(), object()), NotImplemented)
        self.assertIs(order_indicator_table(SimpleNamespace(data={}), None, None), NotImplemented)

    @unittest.skipUnless(importlib.util.find_spec("qlib") and importlib.util.find_spec("pyarrow"),
                         "optional native dependencies unavailable")
    @unittest.skipIf(os.environ.get(CHILD) == "1", "parent launcher")
    def test_signed_child(self):
        from qlib_yahoo.runtime import ROOT, signed_runtime_status
        ready, reason = signed_runtime_status()
        if not ready:
            self.skipTest(reason)
        environment = os.environ.copy()
        environment[CHILD] = "1"
        program = (
            "from qlib_yahoo.runtime import ensure_signed_runtime; ensure_signed_runtime(); "
            "import unittest; result=unittest.TextTestRunner(verbosity=2).run("
            "unittest.defaultTestLoader.loadTestsFromName('tests.test_qlib_streaming_columns.SignedChild')); "
            "raise SystemExit(not result.wasSuccessful())"
        )
        result = subprocess.run([sys.executable, "-c", program], cwd=ROOT, env=environment,
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def reference(raw, day):
    import pandas as pd
    import pyarrow as pa
    frame = pd.DataFrame({name: pd.Series(metric.data, index=metric.index.idx_list)
                          for name, metric in raw.data.items()})
    frame.index.name = "instrument"
    frame = frame.reset_index()
    frame.insert(0, "datetime", day)
    return pa.Table.from_pandas(frame, preserve_index=False)


def fixture(symbols=None, values=None, count=13):
    import numpy as np
    from qlib.backtest.high_performance_ds import NumpyOrderIndicator
    from qlib.utils.index_data import SingleData
    if symbols is None:
        symbols = ["Z", "A", '雪🌍\\"\n', "M"]
    if values is None:
        values = [0., -0., np.nextafter(0., 1.), 1e300]
    raw = NumpyOrderIndicator()
    for i in range(count):
        raw.data[f"metric_{i}"] = SingleData(values, symbols)
    return raw


def native_updated_fixture(symbols):
    """Exercise Qlib's real executed-order update, including calculated ffr."""
    import pandas as pd
    from qlib.backtest.decision import Order
    from qlib.backtest.report import Indicator
    day = pd.Timestamp("2025-01-02").as_unit("ns")
    execute_result = []
    for i, symbol in enumerate(symbols):
        order = Order(symbol, float(i + 1), Order.SELL if i % 2 else Order.BUY, day, day)
        order.deal_amount = float(i + 1) / 2
        order.factor = 1.
        execute_result.append((order, order.deal_amount * 10., .5, 10.))
    indicator = Indicator()
    indicator.update_order_indicators(execute_result)
    return indicator.order_indicator


def position_reference(position, shared, equity, schema):
    import pyarrow as pa
    rows = []
    for symbol in sorted(position.get_stock_list()) or [""]:
        amount = float(position.get_stock_amount(symbol)) if symbol else 0.0
        price = float(position.get_stock_price(symbol)) if symbol else None
        value = amount * price if symbol else 0.0
        rows.append({**shared, "instrument": symbol, "amount": amount, "price": price,
                     "value": value, "weight": value / equity if equity > 0 else float("nan")})
    return pa.Table.from_pylist(rows, schema=schema)


def position_schema_and_shared():
    import tempfile
    import pandas as pd
    from pathlib import Path
    from qlib_yahoo.streaming import SignedHistoryWriter
    with tempfile.TemporaryDirectory() as directory:
        writer = SignedHistoryWriter(Path(directory), SimpleNamespace(decisions=[]))
        schema = writer._position_schema
        writer.close(completed=False)
    # Include metadata to prove the supplied schema survives either conversion.
    schema = schema.with_metadata({b"fixture": b"original position schema"})
    shared = {name: -0.0 for name in schema.names
              if name not in ("instrument", "amount", "price", "value", "weight")}
    shared.update(datetime=pd.Timestamp("2025-01-02").as_unit("ns"),
                  funding_feasible=True, metrics_defined=False, stale_marks=True,
                  stale_mark_symbols='["雪🌍"]')
    return schema, shared


def assert_position_tables(test, old, new):
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    test.assertEqual(old.schema, new.schema)
    test.assertEqual(old.schema.metadata, new.schema.metadata)
    pd.testing.assert_frame_equal(old.to_pandas(), new.to_pandas(), check_exact=True)
    for field in old.schema:
        test.assertEqual(old[field.name].null_count, new[field.name].null_count)
        if field.type == pa.float64():
            np.testing.assert_array_equal(old[field.name].to_numpy().view(np.uint64),
                                          new[field.name].to_numpy().view(np.uint64))
    buffers = []
    for table in (old, new):
        sink = pa.BufferOutputStream()
        with pq.ParquetWriter(sink, table.schema, compression="zstd") as writer:
            writer.write_table(table)
        buffers.append(sink.getvalue().to_pybytes())
    test.assertEqual(buffers[0], buffers[1])


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "optional PyArrow unavailable")
class PositionTests(unittest.TestCase):
    class Position:
        def __init__(self, book, calls):
            self.book, self.calls = book, calls

        def get_stock_list(self):
            self.calls.append(("stock_list",))
            return list(self.book)

        def get_stock_amount(self, symbol):
            self.calls.append(("amount", symbol))
            return self.book[symbol][0]

        def get_stock_price(self, symbol):
            self.calls.append(("price", symbol))
            return self.book[symbol][1]

        def calculate_value(self):
            raise AssertionError("equity recomputed")

    def test_native_schema_generic_callbacks_and_exact_bytes(self):
        schema, shared = position_schema_and_shared()
        for book in ({}, {"Z": (-0.0, 20.), "A": (-3., 11.), "雪🌍": (2., 1e-300)},
                     {"": (123., 456.)}):
            for equity in (100., 0., -1.):
                with self.subTest(book=book, equity=equity):
                    old_calls, new_calls = [], []
                    old = position_reference(self.Position(book, old_calls), shared, equity, schema)
                    new = position_table(self.Position(book, new_calls), shared, equity, schema)
                    self.assertEqual(old_calls, new_calls)
                    self.assertEqual(new_calls, [("stock_list",)] + [call for symbol in sorted(book)
                        if symbol for call in (("amount", symbol), ("price", symbol))])
                    assert_position_tables(self, old, new)
                    if equity <= 0:
                        self.assertEqual(new["weight"].null_count, 0)
                    if not book or "" in book:
                        self.assertEqual(new["price"].null_count, 1)

    def test_getter_exception_propagates_without_retry(self):
        schema, shared = position_schema_and_shared()
        calls = []
        position = self.Position({"Z": (1., 2.), "A": (-2., 4.)}, calls)
        def fail(symbol):
            calls.append(("price", symbol))
            raise ValueError("injected getter failure")
        position.get_stock_price = fail
        with self.assertRaisesRegex(ValueError, "injected getter failure"):
            position_table(position, shared, 100., schema)
        self.assertEqual(calls, [("stock_list",), ("amount", "A"), ("price", "A")])


@unittest.skipUnless(os.environ.get(CHILD) == "1", "requires verified signed child")
class SignedChild(unittest.TestCase):
    def setUp(self):
        import pandas as pd
        self.day = pd.Timestamp("2025-01-02 03:04:05.123456789")

    def compare(self, raw):
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        old = reference(raw, self.day)
        new = order_indicator_table(raw, self.day, old.schema)
        self.assertIsNot(new, NotImplemented)
        self.assertEqual(old.schema, new.schema)
        self.assertEqual(old.schema.metadata, new.schema.metadata)
        self.assertTrue(old.equals(new, check_metadata=True))
        for name in old.column_names[2:]:
            mask = old[name].is_valid().to_numpy()
            a = old[name].to_numpy()[mask].view(np.uint64)
            b = new[name].to_numpy()[mask].view(np.uint64)
            np.testing.assert_array_equal(a, b)
        buffers = []
        for table in (old, new):
            sink = pa.BufferOutputStream()
            with pq.ParquetWriter(sink, table.schema, compression="zstd") as writer:
                writer.write_table(table.cast(old.schema))
            buffers.append(sink.getvalue().to_pybytes())
        self.assertEqual(buffers[0], buffers[1])
        import pandas as pd
        pd.testing.assert_frame_equal(old.to_pandas(), new.to_pandas(), check_exact=True)

    def test_unicode_insertion_order_and_float_bits(self):
        self.compare(fixture())

    def test_null_nan_infinities_extremes_and_negative_zero(self):
        import numpy as np
        symbols = ["Z", "A", "Q", "B", "Y", "M", "X", "N"]
        values = [np.nan, np.inf, -np.inf, -0., 0., 1e-300, -1e300,
                  np.array([0x7FF8000000001234], dtype=np.uint64).view(np.float64)[0]]
        self.compare(fixture(symbols, values))

    def test_duplicate_symbols_exact_aligned_indexes(self):
        self.compare(fixture(["Z", "A", "Z", "B"]))

    def test_noncontiguous_arrays(self):
        import numpy as np
        raw = fixture()
        for metric in raw.data.values():
            metric.data = np.arange(8, dtype=float)[::2]
        self.compare(raw)

    def test_actual_native_update_with_pandas_derived_object_indexes_is_active(self):
        import pandas as pd
        raw = native_updated_fixture(["Z", "A", "雪🌍", "Q"])
        self.compare(raw)
        # The order update currently constructs U indexes. Also exercise native
        # assign's supported Pandas input, the constructor route that creates O.
        raw.assign("pandas_input", pd.Series([1., 2., 3., 4.], index=["Z", "A", "雪🌍", "Q"]))
        self.assertEqual(raw.data["pandas_input"].index.idx_list.dtype.kind, "O")
        self.compare(raw)

    def test_actual_native_update_calendar_resolution_ms_and_seeded_schema_active(self):
        import pandas as pd
        import pyarrow as pa
        self.day = pd.Timestamp("2025-01-02").as_unit("ms")
        raw = native_updated_fixture(["Z", "A", "雪🌍", "Q"])
        old = reference(raw, self.day)
        self.assertEqual(old.schema.field("datetime").type, pa.timestamp("ms"))
        self.assertIn(b"pandas", old.schema.metadata)
        self.compare(raw)

    def test_seeded_timestamp_unit_s_ms_us_ns_preserved_exactly(self):
        import pandas as pd
        import pyarrow as pa
        for unit in ("s", "ms", "us", "ns"):
            with self.subTest(unit=unit):
                self.day = pd.Timestamp("2025-01-02 03:04:05").as_unit(unit)
                raw = fixture()
                self.assertEqual(reference(raw, self.day).schema.field(0).type, pa.timestamp(unit))
                self.compare(raw)

    def test_lossy_ns_to_ms_falls_back_before_arrow_and_native_cast_errors(self):
        import pandas as pd
        import pyarrow as pa
        raw = fixture()
        schema = reference(raw, pd.Timestamp("2025-01-02").as_unit("ms")).schema
        precise = pd.Timestamp("2025-01-02 03:04:05.123456789")
        with patch.object(pa, "array", side_effect=AssertionError("premature allocation")):
            self.assertIs(order_indicator_table(raw, precise, schema), NotImplemented)
        with self.assertRaises(pa.ArrowInvalid):
            reference(raw, precise).cast(schema)

    def test_timestamp_overflow_and_timezone_schema_fall_back_before_arrow(self):
        import pandas as pd
        import pyarrow as pa
        raw = fixture()
        schema = reference(raw, self.day).schema
        with patch.object(pa, "array", side_effect=AssertionError("premature allocation")):
            self.assertIs(order_indicator_table(raw, pd.Timestamp("2500-01-02").as_unit("s"),
                                               schema), NotImplemented)
            timezone = schema.set(0, pa.field("datetime", pa.timestamp("ns", tz="UTC")))
            self.assertIs(order_indicator_table(raw, self.day, timezone), NotImplemented)

    def test_exact_string_object_indexes_and_mixed_unicode_alignment(self):
        raw = fixture()
        for i, metric in enumerate(raw.data.values()):
            if i % 2:
                metric.indices[0].idx_list = metric.indices[0].idx_list.astype(object)
        self.compare(raw)
        for metric in raw.data.values():
            metric.indices[0].idx_list = metric.indices[0].idx_list.astype(object)
        self.compare(raw)

    def test_changed_indexes_or_unsupported_arrays_fall_back_before_arrow(self):
        import numpy as np
        import pyarrow as pa
        mutations = [
            lambda metric: setattr(metric.indices[0], "idx_list", np.array(["A", "Z", "M", "Q"])),
            lambda metric: setattr(metric.indices[0], "idx_list", np.array(["A", "Z", "Q", "M"], dtype=object)),
            lambda metric: setattr(metric, "data", np.arange(len(metric.data), dtype=np.float32)),
            lambda metric: setattr(metric, "data", np.arange(len(metric.data), dtype=np.int64)),
            lambda metric: setattr(metric, "data", metric.data[:2]),
            lambda metric: setattr(metric, "data", metric.data.reshape(2, 2)),
        ]
        for mutation in mutations:
            raw = fixture()
            schema = reference(raw, self.day).schema
            mutation(raw.data["metric_12"])
            with patch.object(pa, "array", side_effect=AssertionError("premature allocation")):
                self.assertIs(order_indicator_table(raw, self.day, schema), NotImplemented)

    def test_object_index_custom_elements_never_invoke_conversion_or_equality(self):
        import numpy as np
        import pyarrow as pa
        class Callback:
            def __eq__(self, other):
                raise AssertionError("custom equality callback")
            def __str__(self):
                raise AssertionError("custom conversion callback")
        class StringSubclass(str):
            def __eq__(self, other):
                raise AssertionError("string subclass equality callback")
        for unsafe in (Callback(), StringSubclass("Z"), None, np.str_("Z")):
            raw = fixture()
            schema = reference(raw, self.day).schema
            raw.data["metric_12"].indices[0].idx_list = np.array([unsafe, "A", "雪🌍", "M"], dtype=object)
            with patch.object(pa, "array", side_effect=AssertionError("premature allocation")):
                self.assertIs(order_indicator_table(raw, self.day, schema), NotImplemented)

    def test_custom_metrics_and_raw_subclass_do_not_invoke_callbacks(self):
        import pyarrow as pa
        raw = fixture()
        schema = reference(raw, self.day).schema
        class Callback:
            @property
            def data(self):
                raise AssertionError("callback")
        raw.data["metric_12"] = Callback()
        with patch.object(pa, "array", side_effect=AssertionError("premature allocation")):
            self.assertIs(order_indicator_table(raw, self.day, schema), NotImplemented)
        class Derived(type(raw)):
            @property
            def data(self):
                raise AssertionError("callback")
        self.assertIs(order_indicator_table(object.__new__(Derived), self.day, schema), NotImplemented)

    def test_schema_empty_and_time_guards(self):
        import pandas as pd
        raw = fixture()
        schema = reference(raw, self.day).schema
        for day in ("2025-01-02", pd.NaT, pd.Timestamp("2025-01-02", tz="UTC")):
            self.assertIs(order_indicator_table(raw, day, schema), NotImplemented)
        self.assertIs(order_indicator_table(raw, self.day, schema.remove(2)), NotImplemented)
        self.assertIs(order_indicator_table(raw, self.day, None), NotImplemented)
        self.assertIs(order_indicator_table(fixture([], []), self.day, schema), NotImplemented)

    def test_realistic_3000_names_13_metrics(self):
        import numpy as np
        symbols = [f"SYM{i:04d}" for i in range(2999, -1, -1)]
        self.compare(fixture(symbols, np.random.default_rng(123).normal(size=3000)))

    def test_actual_native_positions_empty_shorts_insolvent_and_callback_order(self):
        from qlib.backtest.signed_position import SignedPosition
        schema, shared = position_schema_and_shared()
        for book in ({}, {"Z": (-0.0, 20.), "A": (-3., 11.), "雪🌍": (2., 1e-300)}):
            for equity in (100., 0., -1.):
                outputs, calls = [], []
                for builder in (position_reference, position_table):
                    position = SignedPosition(cash=100.)
                    for symbol, (amount, price) in book.items():
                        position._init_stock(symbol, amount, price)
                    observed = []
                    original_list = position.get_stock_list
                    original_amount = position.get_stock_amount
                    original_price = position.get_stock_price
                    def stock_list():
                        observed.append(("stock_list",))
                        return original_list()
                    def amount(symbol):
                        observed.append(("amount", symbol))
                        return original_amount(symbol)
                    def price(symbol):
                        observed.append(("price", symbol))
                        return original_price(symbol)
                    position.get_stock_list = stock_list
                    position.get_stock_amount = amount
                    position.get_stock_price = price
                    outputs.append(builder(position, shared, equity, schema))
                    calls.append(observed)
                self.assertEqual(calls[0], calls[1])
                assert_position_tables(self, *outputs)


def benchmark(sessions=20, repeats=3):
    """Short complete conversion, guards, cast, zstd write and close boundary."""
    import json
    import time
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    symbols = [f"SYM{i:04d}" for i in range(2999, -1, -1)]
    raw = native_updated_fixture(symbols)
    # The real update produces nine metrics. Extend to the full thirteen-column
    # serialization boundary with native Pandas-derived SingleData, outside the
    # timer. Make every index object dtype, so all callback-safe scans are timed.
    from qlib.utils.index_data import SingleData
    for i in range(4):
        raw.data[f"extra_metric_{i}"] = SingleData(pd.Series(
            np.random.default_rng(123 + i).normal(size=3000), index=symbols))
    for metric in raw.data.values():
        metric.indices[0].idx_list = metric.indices[0].idx_list.astype(object)
    # Qlib's actual daily calendar uses nanosecond Timestamp resolution.
    day = pd.Timestamp("2025-01-02").as_unit("ns")
    schema = reference(raw, day).schema
    results = {"pandas_seconds": [], "columnar_seconds": []}
    for repeat in range(repeats):
        for fast in ((False, True) if repeat % 2 == 0 else (True, False)):
            sink = pa.BufferOutputStream()
            start = time.perf_counter()
            with pq.ParquetWriter(sink, schema, compression="zstd") as writer:
                for i in range(sessions):
                    current = day + pd.Timedelta(days=i)
                    table = (order_indicator_table(raw, current, schema) if fast else reference(raw, current))
                    writer.write_table(table.cast(schema))
            elapsed = time.perf_counter() - start
            results["columnar_seconds" if fast else "pandas_seconds"].append(elapsed)
    results.update(names=3000, metrics=13, index_dtype="object_exact_str",
                   native_update_metrics=9, sessions=sessions, repeats=repeats)
    print(json.dumps(results))


def benchmark_positions(sessions=20, repeats=3):
    """Native getters plus complete positions conversion/zstd writing boundary."""
    import json
    import time
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from qlib.backtest.signed_position import SignedPosition
    position = SignedPosition(cash=10_000_000.)
    for i in range(2999, -1, -1):
        position._init_stock(f"SYM{i:04d}", float((i + 1) * (-1 if i % 2 else 1)),
                             float(i + 1) / 100.)
    schema, shared = position_schema_and_shared()
    results = {"pylist_seconds": [], "columnar_seconds": []}
    for repeat in range(repeats):
        for fast in ((False, True) if repeat % 2 == 0 else (True, False)):
            sink = pa.BufferOutputStream()
            start = time.perf_counter()
            with pq.ParquetWriter(sink, schema, compression="zstd") as writer:
                for i in range(sessions):
                    fields = {**shared, "datetime": shared["datetime"] + pd.Timedelta(days=i)}
                    table = (position_table if fast else position_reference)(
                        position, fields, 10_000_000., schema)
                    writer.write_table(table)
            elapsed = time.perf_counter() - start
            results["columnar_seconds" if fast else "pylist_seconds"].append(elapsed)
    results.update(names=3000, sessions=sessions, repeats=repeats)
    print(json.dumps(results))


if __name__ == "__main__":
    unittest.main()
