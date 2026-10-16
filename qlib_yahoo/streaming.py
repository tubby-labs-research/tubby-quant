"""Bounded serialization using Qlib's account/history injection hooks.

Trading, borrow charging, marks, metrics and executor validation remain native.
Only historical copies are replaced: one session is written after its fills.
There is one live account throughout the entire requested date window.
"""

from __future__ import annotations

import gzip
import io
import json
import math
import os
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from qlib_yahoo.dataset import DatasetError


DECISION_GZIP_LEVEL = 6


@dataclass(frozen=True)
class EquitySnapshot:
    """The runner's history reconciliation needs only independently checked NAV."""

    value: float

    def calculate_value(self):
        return self.value


class SignedHistoryWriter:
    """Write session-sized position and execution history, then atomically finish."""

    def __init__(self, run_dir: Path, strategy, *, columnar=False, buffered=False,
                 decision_gzip_level=DECISION_GZIP_LEVEL, _gzip_factory=None):
        import pyarrow as pa
        import pyarrow.parquet as pq

        if type(columnar) is not bool:
            raise TypeError("columnar must be a boolean")
        if type(buffered) is not bool:
            raise TypeError("buffered must be a boolean")
        if _gzip_factory is not None and not callable(_gzip_factory):
            raise TypeError("explicit gzip factory must be callable")
        minimum_level = 1 if _gzip_factory is None else 0
        if type(decision_gzip_level) is not int or not minimum_level <= decision_gzip_level <= 9:
            raise ValueError(f"decision_gzip_level must be an integer from {minimum_level} through 9")
        gzip_factory = gzip.GzipFile if _gzip_factory is None else _gzip_factory
        self.run_dir, self.strategy = Path(run_dir), strategy
        self.decision_gzip_level = decision_gzip_level
        self.columnar = columnar
        self.buffered = buffered
        self._buffers = []
        self.columnar_sessions = self.columnar_fallback_sessions = 0
        self.sessions = self.rows = 0
        self.closed = False
        self._position_schema = pa.schema([
            ("datetime", pa.timestamp("ns")), ("instrument", pa.string()),
            *[(name, pa.float64()) for name in (
                "amount", "price", "value", "weight", "cash", "account", "cash_total", "free_cash",
                "restricted_cash", "long_value", "short_liability", "transaction_cost_dollars",
                "borrow_cost_dollars", "total_cost_dollars", "unfunded_cash_shortfall")],
            ("funding_feasible", pa.bool_()), ("metrics_defined", pa.bool_()),
            ("stale_marks", pa.bool_()), ("stale_mark_symbols", pa.string()),
        ])
        self._resources = ExitStack()
        self._order_schema = None
        self._orders = None
        try:
            self._positions = self._resources.enter_context(pq.ParquetWriter(
                self.run_dir / "positions.parquet.tmp", self._position_schema, compression="zstd"))
            self._position_sink = self._buffer(self._positions) if buffered else self._positions
            self._raw = self._resources.enter_context(open(self.run_dir / "signed_decisions.json.gz.tmp", "wb"))
            self._gzip = self._resources.enter_context(gzip_factory(
                filename="", mode="wb", fileobj=self._raw, mtime=0, compresslevel=decision_gzip_level))
            self._decisions = self._resources.enter_context(io.TextIOWrapper(self._gzip, encoding="utf-8"))
            self._decisions.write("[")
        except BaseException:
            try:
                self._resources.close()
            finally:
                self._discard_temporary()
            raise

    def _discard_temporary(self):
        for name in ("positions.parquet", "signed_decisions.json.gz", "qlib_order_indicators.parquet"):
            (self.run_dir / (name + ".tmp")).unlink(missing_ok=True)

    def _buffer(self, writer):
        from .parquet_buffer import BufferedParquetBuffer
        buffer = BufferedParquetBuffer(writer, target_rows=32768, max_bytes=16 * 1024 * 1024)
        self._buffers.append(buffer)
        return buffer

    def buffered_stats(self):
        """Final counters; queued payload is not total worker or writer RAM."""
        return {"enabled": self.buffered,
                "positions": self._position_sink.stats() if self.buffered else None,
                "order_indicators": self._order_sink.stats() if self.buffered and self._orders is not None else None}

    def positions(self, account, day):
        import pyarrow as pa

        position = account.current_position
        state = account.ledger_records[day]
        equity = float(position.calculate_value())
        if not math.isclose(equity, float(account.portfolio_metrics.get_latest_account_value()),
                            rel_tol=1e-7, abs_tol=1e-6):
            raise DatasetError("streamed signed position equity does not reconcile")
        shared = {"datetime": day, "cash": state["free_cash"], "account": equity,
                  **{key: state[key] for key in ("cash_total", "free_cash", "restricted_cash", "long_value",
                      "short_liability", "unfunded_cash_shortfall", "funding_feasible", "metrics_defined")},
                  "transaction_cost_dollars": state["transaction_cost_amount"],
                  "borrow_cost_dollars": state["borrow_cost_amount"], "total_cost_dollars": state["total_cost_amount"],
                  "stale_marks": bool(state["stale_marks"]),
                  "stale_mark_symbols": json.dumps(list(state["stale_marks"]))}
        if self.columnar:
            from .streaming_columns import position_table
            table = position_table(position, shared, equity, self._position_schema)
        else:
            rows = []
            for symbol in sorted(position.get_stock_list()) or [""]:
                amount = float(position.get_stock_amount(symbol)) if symbol else 0.0
                price = float(position.get_stock_price(symbol)) if symbol else None
                value = amount * price if symbol else 0.0
                rows.append({**shared, "instrument": symbol, "amount": amount, "price": price, "value": value,
                             "weight": value / equity if equity > 0 else float("nan")})
            table = pa.Table.from_pylist(rows, schema=self._position_schema)
        self._position_sink.write_table(table)
        self.rows += table.num_rows
        return EquitySnapshot(equity)

    def finish_session(self, account, day):
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        from qlib_yahoo.signed import signed_decisions

        if len(self.strategy.decisions) != 1 or self.strategy.decisions[0]["session"] != day:
            raise DatasetError("streamed signed decision does not cover exactly the closed session")
        # Encode only the closed session. dumps uses the built-in encoder and
        # avoids a text write for every JSON token, with the same dump formatting.
        # Encode before writing even the separator so invalid values leave no
        # partial session in the stream.
        encoded = json.dumps(signed_decisions(self.strategy)[0], allow_nan=False)
        self._decisions.write((", " if self.sessions else "") + encoded)
        # Qlib retains a separate per-order indicator object for every session.
        # Preserve these as an additional table instead of retaining the objects.
        raw = account.indicator.order_indicator_his[day]
        # Native SingleData.to_series passes Qlib's Index wrapper to pandas 2.2,
        # which rejects it. Use its public index values rather than patching Qlib.
        table = NotImplemented
        if self.columnar:
            from .streaming_columns import order_indicator_table
            table = order_indicator_table(raw, day, self._order_schema)
        if table is NotImplemented:
            frame = pd.DataFrame({name: pd.Series(metric.data, index=metric.index.idx_list)
                                  for name, metric in raw.data.items()})
            if not frame.empty:
                frame.index.name = "instrument"
                frame = frame.reset_index()
                frame.insert(0, "datetime", day)
                table = pa.Table.from_pandas(frame, preserve_index=False)
            else:
                table = None
            if self.columnar:
                self.columnar_fallback_sessions += 1
        elif self.columnar:
            self.columnar_sessions += 1
        if table is not None:
            if self._orders is None:
                self._order_schema = table.schema
                self._orders = self._resources.enter_context(pq.ParquetWriter(
                    self.run_dir / "qlib_order_indicators.parquet.tmp", self._order_schema, compression="zstd"))
                self._order_sink = self._buffer(self._orders) if self.buffered else self._orders
            self._order_sink.write_table(table.cast(self._order_schema))
        # Release only after synchronous writes, or successful ownership of
        # exact queued buffers. No file is promoted before final flush/close.
        self.strategy.decisions.clear()  # Orders were encoded after exchange mutation.
        account.indicator.order_indicator_his.pop(day)
        self.sessions += 1

    def close(self, *, completed: bool):
        if self.closed:
            return
        try:
            try:
                if completed:
                    for buffer in self._buffers:
                        buffer.flush()
                    self._decisions.write("]")
            finally:
                # ExitStack attempts every closer even when an earlier one raises.
                self._resources.close()
            if completed:
                for name in ("positions.parquet", "signed_decisions.json.gz", "qlib_order_indicators.parquet"):
                    temporary = self.run_dir / (name + ".tmp")
                    if temporary.exists():
                        os.replace(temporary, self.run_dir / name)
        finally:
            for buffer in self._buffers:
                buffer.discard()
            self.closed = True
            self._discard_temporary()


def streaming_account_class(base, *, flat_indicators=False):
    """Subclass a reviewed native account; do not substitute the executor."""

    if type(flat_indicators) is not bool:
        raise TypeError("flat_indicators must be a boolean")

    class StreamingSignedAccount(base):
        def __init__(self, *, history_writer, **kwargs):
            self.history_writer = history_writer
            super().__init__(**kwargs)

        def update_hist_positions(self, trade_start_time):
            # Match the live-position side effects of Account.update_hist_positions.
            # The only omitted operation is deepcopy of a completed historical book.
            self.current_position.position["now_account_value"] = self.current_position.calculate_value()
            self.current_position.update_weight_all()
            self.hist_positions[trade_start_time] = self.history_writer.positions(self, trade_start_time)

        def update_bar_end(self, trade_start_time, *args, **kwargs):
            super().update_bar_end(trade_start_time, *args, **kwargs)
            self.history_writer.finish_session(self, trade_start_time)

    if not flat_indicators:
        return StreamingSignedAccount

    from .flat_order_indicator import FlatIndicator

    class StreamingFlatIndicatorAccount(StreamingSignedAccount):
        def reset_report(self, freq, benchmark_config):
            # Native portfolio/account setup still runs once. Only the fresh
            # flat-order indicator is replaced; marks, aggregates and recording
            # stay native, including fallback for unsupported indicator inputs.
            result = super().reset_report(freq, benchmark_config)
            self.indicator = FlatIndicator()
            return result

    return StreamingFlatIndicatorAccount
