"""Failure-safe history writer ownership; signed numerical parity is tested separately."""

import importlib.util
import gzip
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "optional PyArrow unavailable")
class StreamOwnershipTests(unittest.TestCase):
    def setUp(self):
        from qlib_yahoo.streaming import SignedHistoryWriter
        self.Writer = SignedHistoryWriter
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.strategy = SimpleNamespace(decisions=[])

    def test_abort_releases_handles_and_temporary_files(self):
        writer = self.Writer(self.path, self.strategy)
        writer.close(completed=False)
        self.assertTrue(writer.closed)
        self.assertTrue(writer._raw.closed)
        self.assertFalse(list(self.path.iterdir()))
        writer.close(completed=False)

    def test_failed_position_close_still_releases_all_other_resources(self):
        writer = self.Writer(self.path, self.strategy)
        original = writer._positions.close
        def failed_close():
            original()
            raise OSError("injected parquet close failure")
        with patch.object(writer._positions, "close", side_effect=failed_close):
            with self.assertRaisesRegex(OSError, "injected"):
                writer.close(completed=True)
        self.assertTrue(writer._raw.closed)
        self.assertTrue(writer._decisions.closed)
        self.assertFalse(list(self.path.iterdir()))
        writer.close(completed=False)

    def test_constructor_failure_closes_already_owned_position_writer(self):
        import pyarrow.parquet as pq
        original = pq.ParquetWriter
        created = []
        def capture(*args, **kwargs):
            writer = original(*args, **kwargs)
            created.append(writer)
            return writer
        with patch("pyarrow.parquet.ParquetWriter", side_effect=capture), patch(
                "qlib_yahoo.streaming.open", side_effect=OSError("injected open failure"), create=True):
            with self.assertRaisesRegex(OSError, "injected open"):
                self.Writer(self.path, self.strategy)
        self.assertEqual(len(created), 1)
        self.assertFalse(created[0].is_open)
        self.assertFalse(list(self.path.iterdir()))

    def session(self, day, *, empty=False):
        symbol = '雪🌍\\"\n\t'
        order = SimpleNamespace(stock_id=symbol, direction=1, BUY=1, amount=3,
                                deal_amount=2, signed_skip_reason=None,
                                signed_partial_reason='budget\\"\n')
        record = {"session": day, "signal_session": None,
                  "target_weights": {} if empty else {symbol: -0.0, "A": 1e-300, "B": 1e300},
                  "target_quantities": {} if empty else {symbol: -2, "A": 0, "B": 10**20},
                  "sizing_equity": None if empty else 100.25,
                  "reasons": [] if empty else ["雪", {"nested": [None, True, 0, 0.5]}],
                  "issued_orders": [] if empty else [order]}
        raw = SimpleNamespace(data={})
        account = SimpleNamespace(indicator=SimpleNamespace(order_indicator_his={day: raw}))
        self.strategy.decisions.append(record)
        return account, record, raw

    def assert_retained(self, writer, account, day, record, raw, sessions=0):
        self.assertEqual(writer.sessions, sessions)
        self.assertIs(self.strategy.decisions[0], record)
        self.assertIs(account.indicator.order_indicator_his[day], raw)

    def test_session_encoding_matches_legacy_bytes_and_uses_one_write(self):
        from qlib_yahoo.signed import signed_decisions
        writer = self.Writer(self.path, self.strategy)
        self.addCleanup(writer.close, completed=False)
        legacy = io.StringIO()
        legacy.write("[")
        expected = []
        for index, day in enumerate(("2025-09-02", "2025-09-03")):
            account, record, raw = self.session(day, empty=bool(index))
            logical = signed_decisions(self.strategy)[0]
            expected.append(logical)
            if index:
                legacy.write(", ")
            json.dump(logical, legacy, allow_nan=False)
            with patch.object(writer._decisions, "write", wraps=writer._decisions.write) as write:
                writer.finish_session(account, day)
            self.assertEqual(write.call_count, 1)
            self.assertFalse(self.strategy.decisions)
            self.assertFalse(account.indicator.order_indicator_his)
        legacy.write("]")
        writer.close(completed=True)
        with gzip.open(self.path / "signed_decisions.json.gz", "rb") as stream:
            actual = stream.read()
        self.assertEqual(actual, legacy.getvalue().encode("utf-8"))
        self.assertEqual(json.loads(actual), expected)
        self.assertEqual(writer.sessions, 2)

    def test_nonfinite_rejected_before_write_and_records_retained(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                writer = self.Writer(self.path, self.strategy)
                account, record, raw = self.session("2025-09-02")
                record["target_weights"]["bad"] = value
                with patch.object(writer._decisions, "write", wraps=writer._decisions.write) as write:
                    with self.assertRaises(ValueError):
                        writer.finish_session(account, "2025-09-02")
                write.assert_not_called()
                self.assert_retained(writer, account, "2025-09-02", record, raw)
                writer.close(completed=False)
                self.assertFalse(list(self.path.iterdir()))
                self.strategy.decisions.clear()

    def test_encoding_failure_on_later_session_writes_no_separator(self):
        writer = self.Writer(self.path, self.strategy)
        self.addCleanup(writer.close, completed=False)
        account, _, _ = self.session("2025-09-02")
        writer.finish_session(account, "2025-09-02")
        account, record, raw = self.session("2025-09-03")
        record["reasons"].append(object())
        with patch.object(writer._decisions, "write", wraps=writer._decisions.write) as write:
            with self.assertRaises(TypeError):
                writer.finish_session(account, "2025-09-03")
        write.assert_not_called()
        self.assert_retained(writer, account, "2025-09-03", record, raw, sessions=1)

    def test_decision_write_failure_keeps_native_records_and_abort_cleans_files(self):
        writer = self.Writer(self.path, self.strategy)
        account, record, raw = self.session("2025-09-02")
        with patch.object(writer._decisions, "write", side_effect=OSError("injected text write")):
            with self.assertRaisesRegex(OSError, "injected text"):
                writer.finish_session(account, "2025-09-02")
        self.assert_retained(writer, account, "2025-09-02", record, raw)
        writer.close(completed=False)
        self.assertTrue(writer._raw.closed)
        self.assertTrue(writer._decisions.closed)
        self.assertFalse(list(self.path.iterdir()))

    def test_partial_decision_write_is_discarded_on_abort(self):
        writer = self.Writer(self.path, self.strategy)
        self.addCleanup(writer.close, completed=False)
        account, record, raw = self.session("2025-09-02")
        original = writer._decisions.write
        def partial_write(value):
            original(value[:20])
            raise OSError("injected partial write")
        with patch.object(writer._decisions, "write", side_effect=partial_write):
            with self.assertRaisesRegex(OSError, "injected partial"):
                writer.finish_session(account, "2025-09-02")
        self.assert_retained(writer, account, "2025-09-02", record, raw)
        writer.close(completed=False)
        self.assertTrue(writer._raw.closed)
        self.assertFalse(list(self.path.iterdir()))

    def test_session_coverage_guard_rejects_mismatch_before_write(self):
        from qlib_yahoo.dataset import DatasetError
        writer = self.Writer(self.path, self.strategy)
        self.addCleanup(writer.close, completed=False)
        account, _, _ = self.session("2025-09-02")
        with patch.object(writer._decisions, "write", wraps=writer._decisions.write) as write:
            with self.assertRaisesRegex(DatasetError, "exactly the closed session"):
                writer.finish_session(account, "2025-09-03")
        write.assert_not_called()
        self.assertEqual(writer.sessions, 0)

    def test_indicator_write_failure_keeps_native_records_and_abort_cleans_files(self):
        writer = self.Writer(self.path, self.strategy)
        account, record, raw = self.session("2025-09-02")
        raw.data = {"amount": SimpleNamespace(data=[2.0], index=SimpleNamespace(idx_list=["A"]))}
        with patch("pyarrow.parquet.ParquetWriter.write_table", side_effect=OSError("injected indicator write")):
            with self.assertRaisesRegex(OSError, "injected indicator"):
                writer.finish_session(account, "2025-09-02")
        self.assert_retained(writer, account, "2025-09-02", record, raw)
        writer.close(completed=False)
        self.assertTrue(writer._raw.closed)
        self.assertFalse(list(self.path.iterdir()))

    def test_native_indicator_values_are_saved_before_records_clear(self):
        import pyarrow.parquet as pq
        writer = self.Writer(self.path, self.strategy)
        self.addCleanup(writer.close, completed=False)
        import pandas as pd
        day = pd.Timestamp("2025-09-02")
        account, _, raw = self.session(day)
        raw.data = {name: SimpleNamespace(data=values, index=SimpleNamespace(idx_list=["B", "A"]))
                    for name, values in {"amount": [-2.0, 3.0], "price": [15.0, 20.0]}.items()}
        writer.finish_session(account, day)
        writer.close(completed=True)
        rows = pq.read_table(self.path / "qlib_order_indicators.parquet").to_pylist()
        self.assertEqual(rows, [{"datetime": day, "instrument": "B", "amount": -2.0, "price": 15.0},
                                {"datetime": day, "instrument": "A", "amount": 3.0, "price": 20.0}])
        self.assertFalse(self.strategy.decisions)
        self.assertFalse(account.indicator.order_indicator_his)


if __name__ == "__main__":
    unittest.main()
