"""Optional history buffer lifecycle and isolated native account reset checks."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

CHILD = "QLIB_STREAMING_OPTIONS_CHILD"


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "optional PyArrow unavailable")
class WriterTests(unittest.TestCase):
    def setUp(self):
        import pandas as pd
        from qlib_yahoo.streaming import SignedHistoryWriter
        self.Writer = SignedHistoryWriter
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.day = pd.Timestamp("2025-01-02").as_unit("ms")
        self.strategy = SimpleNamespace(decisions=[])

    def position_account(self):
        state = {key: 0. for key in ("cash_total", "free_cash", "restricted_cash", "long_value",
            "short_liability", "unfunded_cash_shortfall", "transaction_cost_amount",
            "borrow_cost_amount", "total_cost_amount")}
        state.update(funding_feasible=True, metrics_defined=True, stale_marks=[])
        position = SimpleNamespace(calculate_value=lambda: 100., get_stock_list=lambda: ["雪", "A"],
            get_stock_amount=lambda symbol: -2. if symbol == "雪" else 1., get_stock_price=lambda symbol: 3.)
        return SimpleNamespace(current_position=position, ledger_records={self.day: state},
            portfolio_metrics=SimpleNamespace(get_latest_account_value=lambda: 100.))

    def order_account(self):
        import numpy as np
        raw = SimpleNamespace(data={"amount": SimpleNamespace(data=np.array([-2., 1.]),
            index=SimpleNamespace(idx_list=np.array(["雪", "A"])))})
        self.strategy.decisions.append({"session": self.day, "signal_session": None,
            "target_weights": {"雪": -.5, "A": .5}, "target_quantities": {"雪": -2, "A": 1},
            "sizing_equity": 100., "reasons": {}, "issued_orders": []})
        return SimpleNamespace(indicator=SimpleNamespace(order_indicator_his={self.day: raw}))

    def writer(self, **options):
        writer = self.Writer(self.directory, self.strategy, **options)
        self.addCleanup(writer.close, completed=False)
        return writer

    def test_default_uses_native_writers_without_buffer_creation(self):
        with patch("qlib_yahoo.parquet_buffer.BufferedParquetBuffer", side_effect=AssertionError("default buffered")):
            writer = self.writer()
            writer.positions(self.position_account(), self.day)
            writer.finish_session(self.order_account(), self.day)
            self.assertIs(writer._position_sink, writer._positions)
            self.assertIs(writer._order_sink, writer._orders)
            writer.close(completed=True)
        self.assertTrue((self.directory / "positions.parquet").is_file())
        self.assertTrue((self.directory / "qlib_order_indicators.parquet").is_file())

    def test_buffered_final_flush_precedes_underlying_close_and_promotes_complete_rows(self):
        import pyarrow.parquet as pq
        writer = self.writer(buffered=True)
        writer.positions(self.position_account(), self.day)
        writer.finish_session(self.order_account(), self.day)
        self.assertEqual(writer._position_sink.stats()["pending_rows"], 2)
        self.assertEqual(writer._order_sink.stats()["pending_rows"], 2)
        events = []
        patches = []
        for name, resource in (("positions", writer._positions), ("orders", writer._orders)):
            original_write, original_close = resource.write_table, resource.close
            def write(table, *args, _name=name, _original=original_write, **kwargs):
                events.append((_name, "write", len(table)))
                return _original(table, *args, **kwargs)
            def close(_name=name, _original=original_close):
                events.append((_name, "close"))
                return _original()
            patches.extend((patch.object(resource, "write_table", side_effect=write),
                            patch.object(resource, "close", side_effect=close)))
        from contextlib import ExitStack
        with ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
            writer.close(completed=True)
        for name in ("positions", "orders"):
            self.assertLess(events.index((name, "write", 2)), events.index((name, "close")))
        self.assertLess(max(i for i, event in enumerate(events) if event[1] == "write"),
                        min(i for i, event in enumerate(events) if event[1] == "close"))
        self.assertEqual(pq.read_table(self.directory / "positions.parquet").num_rows, 2)
        self.assertEqual(pq.read_table(self.directory / "qlib_order_indicators.parquet").num_rows, 2)
        self.assertFalse(list(self.directory.glob("*.tmp")))
        stats = writer.buffered_stats()
        self.assertTrue(stats["enabled"])
        for sink in (stats["positions"], stats["order_indicators"]):
            self.assertEqual(sink["target_rows"], 32768)
            self.assertEqual(sink["max_bytes"], 16 * 1024 * 1024)
            self.assertEqual(sink["pending_rows"], 0)
            self.assertEqual(sink["writer_calls"], 1)
        self.assertEqual(writer._position_sink.stats()["pending_rows"], 0)
        self.assertEqual(writer._order_sink.stats()["pending_rows"], 0)
        self.assertEqual(writer._position_sink.stats()["writer_calls"], 1)
        self.assertEqual(writer._order_sink.stats()["writer_calls"], 1)

    def test_failed_flush_closes_all_resources_and_never_promotes_or_retries(self):
        writer = self.writer(buffered=True)
        writer.positions(self.position_account(), self.day)
        writer.finish_session(self.order_account(), self.day)
        failure = OSError("partial buffered write")
        with patch.object(writer._positions, "write_table", side_effect=failure) as write:
            with self.assertRaises(OSError) as caught:
                writer.close(completed=True)
            self.assertIs(caught.exception, failure)
            self.assertEqual(write.call_count, 1)
            writer.close(completed=True)
            self.assertEqual(write.call_count, 1)
        self.assertTrue(writer.closed)
        self.assertTrue(writer._raw.closed)
        self.assertTrue(writer._decisions.closed)
        self.assertFalse(list(self.directory.iterdir()))

    def test_order_flush_failure_after_position_flush_never_promotes_either_file(self):
        writer = self.writer(buffered=True)
        writer.positions(self.position_account(), self.day)
        writer.finish_session(self.order_account(), self.day)
        with patch.object(writer._orders, "write_table", side_effect=OSError("order flush failure")):
            with self.assertRaisesRegex(OSError, "order flush failure"):
                writer.close(completed=True)
        self.assertEqual(writer._position_sink.stats()["writer_calls"], 1)
        self.assertTrue(writer.closed)
        self.assertTrue(writer._raw.closed)
        self.assertFalse(list(self.directory.iterdir()))

    def test_default_native_position_write_failure_propagates_without_buffer(self):
        writer = self.writer()
        failure = OSError("native position write failure")
        with patch.object(writer._positions, "write_table", side_effect=failure) as write:
            with self.assertRaises(OSError) as caught:
                writer.positions(self.position_account(), self.day)
            self.assertIs(caught.exception, failure)
            self.assertEqual(write.call_count, 1)
        self.assertEqual(writer.rows, 0)
        self.assertIs(writer._position_sink, writer._positions)
        writer.close(completed=False)
        self.assertFalse(list(self.directory.iterdir()))

    def test_abort_discards_pending_ram_without_writes_or_promotion(self):
        writer = self.writer(buffered=True)
        writer.positions(self.position_account(), self.day)
        writer.finish_session(self.order_account(), self.day)
        with patch.object(writer._positions, "write_table", side_effect=AssertionError("abort flush")), \
                patch.object(writer._orders, "write_table", side_effect=AssertionError("abort flush")):
            writer.close(completed=False)
        self.assertEqual(writer._position_sink.stats()["pending_bytes"], 0)
        self.assertEqual(writer._order_sink.stats()["pending_bytes"], 0)
        self.assertFalse(list(self.directory.iterdir()))

    def test_buffer_option_requires_exact_bool_before_file_creation(self):
        for value in (1, "true", None):
            with self.subTest(value=value), self.assertRaises(TypeError):
                self.Writer(self.directory, self.strategy, buffered=value)
            self.assertFalse(list(self.directory.iterdir()))


class FactoryTests(unittest.TestCase):
    def test_default_factory_never_imports_flat_helper_and_preserves_native_reset(self):
        from qlib_yahoo.streaming import streaming_account_class
        class Native:
            def reset_report(self, freq, benchmark_config):
                return freq, benchmark_config
        with patch.dict(sys.modules, {"qlib_yahoo.flat_order_indicator": None}):
            cls = streaming_account_class(Native)
            self.assertIs(cls.reset_report, Native.reset_report)
        for value in (1, None, "yes"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                streaming_account_class(Native, flat_indicators=value)

    @unittest.skipUnless(importlib.util.find_spec("qlib"), "optional native Qlib unavailable")
    @unittest.skipIf(os.environ.get(CHILD) == "1", "parent launcher")
    def test_actual_native_reset_once_and_default_indicator_unchanged(self):
        from qlib_yahoo.runtime import ROOT, signed_runtime_status
        ready, reason = signed_runtime_status()
        if not ready:
            self.skipTest(reason)
        environment = os.environ.copy()
        environment[CHILD] = "1"
        program = ("from qlib_yahoo.runtime import ensure_signed_runtime; ensure_signed_runtime(); "
            "import unittest; result=unittest.TextTestRunner(verbosity=2).run("
            "unittest.defaultTestLoader.loadTestsFromName('tests.test_qlib_streaming_options.SignedChild')); "
            "raise SystemExit(not result.wasSuccessful())")
        result = subprocess.run([sys.executable, "-c", program], cwd=ROOT, env=environment,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(os.environ.get(CHILD) == "1", "requires verified signed child")
class SignedChild(unittest.TestCase):
    def test_native_reset_report_once_per_reset_and_default_unchanged(self):
        from qlib.backtest.account import Account
        from qlib.backtest.signed_account import SignedAccount
        from qlib.backtest.report import Indicator
        from qlib_yahoo.flat_order_indicator import FlatIndicator
        from qlib_yahoo.streaming import streaming_account_class
        original = Account.reset_report
        for enabled in (False, True):
            calls = []
            def observed(account, freq, benchmark_config):
                calls.append((account, freq, benchmark_config))
                return original(account, freq, benchmark_config)
            cls = streaming_account_class(SignedAccount, flat_indicators=enabled)
            with patch.object(Account, "reset_report", observed):
                account = cls(history_writer=object(), init_cash=100., benchmark_config={"benchmark": None})
                self.assertEqual(len(calls), 1)
                self.assertIs(type(account.indicator), FlatIndicator if enabled else Indicator)
                portfolio = account.portfolio_metrics
                indicator = account.indicator
                account.reset_report("day", {"benchmark": None})
                self.assertEqual(len(calls), 2)
                self.assertIsNot(account.indicator, indicator)
                self.assertIsNot(account.portfolio_metrics, portfolio)
                self.assertEqual(account.current_position.get_cash(), 100.)
                self.assertEqual(account.ledger_records, {})
            self.assertIs(Account.reset_report, original)


if __name__ == "__main__":
    unittest.main()
