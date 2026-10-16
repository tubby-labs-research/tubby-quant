# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Session-local quote rows; native signed sizing, orders and fills remain intact.

The strategy boundary derives from the MIT-licensed signed_weight.py at
https://github.com/tubby-labs-research/qlib/blob/da54853c7598599f74ff7776dc979b74106eb1d0/qlib/contrib/strategy/signed_weight.py
Recheck this override when that pin changes. No account or executor is replaced.
"""

from __future__ import annotations

import math
from datetime import time
from numbers import Real
from types import MethodType

import numpy as np
import pandas as pd

from qlib.backtest.decision import Order, TradeDecisionWO
from qlib.backtest.high_performance_ds import NumpyQuote
from qlib.backtest.signed_exchange import SignedExchange
from qlib.backtest.signed_position import SignedPosition
from qlib.contrib.strategy.signed_weight import SignedWeightStrategy

from .daily_quote import DailyQuote


SUPPORTED_COMMIT = "da54853c7598599f74ff7776dc979b74106eb1d0"
SUPPORTED_COMMITS = frozenset((
    SUPPORTED_COMMIT,
    # RT-1 changes only the native executor's immutable timestamp floor cache.
    # Native signed quote/strategy/account implementations are byte-unchanged.
    "d10a5a8ca397ea088da8f5c996e8bd042eea27e0",
    # Optional generic executor hook; the signed ledger/strategy bytes remain unchanged.
    "8ab6249aa6143cb0675259d5ff986d294e918ef4",
))

_NATIVE_GENERATE = SignedWeightStrategy.generate_trade_decision
_NATIVE_WEIGHTS = SignedWeightStrategy._weights
_QUOTE_GET_DATA = {NumpyQuote: NumpyQuote.get_data, DailyQuote: DailyQuote.get_data}


def _reviewed_quote_binding(quote):
    """Recognize native bound queries and DailyQuote's reviewed instance cache.

    DailyQuote records the original bound method and resulting cache identities
    in its reviewed constructor. __wrapped__ alone is writable and can spoof a
    different callback, so replacement wrappers remain unsupported.
    """
    method = quote.get_data
    original = _QUOTE_GET_DATA.get(type(quote))
    if original is None:
        return False
    if (getattr(method, "__self__", None) is quote
            and getattr(method, "__func__", None) is original):
        return True
    if type(quote) is not DailyQuote or method is not getattr(quote, "_native_get_data_cache", None):
        return False
    bound = getattr(quote, "_native_get_data_method", None)
    return (type(bound) is MethodType and bound.__self__ is quote
            and bound.__func__ is original)


def classes_for(commit):
    """Return reviewed (exchange, strategy) hooks, or future native classes."""
    if commit in SUPPORTED_COMMITS:
        return FastSignedExchange, FastSignedWeightStrategy
    return SignedExchange, SignedWeightStrategy


class FastSignedExchange(SignedExchange):
    """Native signed exchange with lazy one-session, one-bar row snapshots.

    The original quote object is retained. Supported default queries share a
    compact scalar tuple per actual symbol; generic queries still call native
    methods. Daily raw-scalar versus weekend aggregation semantics derive from
    da54853c NumpyQuote.get_data and this project's DailyQuote.get_data.
    """

    def __init__(self, *args, _range_validator=None, **kwargs):
        if _range_validator is not None and not callable(_range_validator):
            raise TypeError("date-range validator must be callable or None")
        self._range_validator = _range_validator
        super().__init__(*args, **kwargs)
        self.reset_session_cache()

    def reset_session_cache(self):
        """Clear every row when a prepared exchange starts a fresh account."""
        # Test/injected exchanges can be constructed through __new__.
        self._range_validator = getattr(self, "_range_validator", None)
        self._session_quote = self.quote
        self._fast_quote_supported = (
            type(self.quote) in (NumpyQuote, DailyQuote)
            and self.quote.freq == pd.Timedelta(days=1)
            and self.quote.region in ("cn", "tw", "us")
        )
        self._session_start = self._session_end = None
        self._session_rows = {}
        self.range_counts = ({"compiled_checks": 0, "native_checks": 0}
                             if self._range_validator is not None else None)

    @property
    def supports_session_rows(self):
        return self._fast_quote_supported and self.quote is self._session_quote

    def begin_session(self, start_time, end_time):
        """Validate the session domain once; unrelated queries never switch it."""
        if (not self.supports_session_rows or type(start_time) is not pd.Timestamp
                or type(end_time) is not pd.Timestamp or start_time.tz is not None
                or end_time.tz is not None or start_time > end_time
                or start_time.time() != time.min or start_time.nanosecond != 0):
            self._session_start = self._session_end = None
            self._session_rows = {}
            return False
        try:
            start_ns, end_ns = start_time.as_unit("ns").asm8, end_time.as_unit("ns").asm8
        except (OverflowError, ValueError):
            self._session_start = self._session_end = None
            self._session_rows = {}
            return False
        if start_time != self._session_start or end_time != self._session_end:
            self._session_rows = {}
        self._session_start, self._session_end = start_time, end_time
        self._session_start_ns, self._session_end_ns = start_ns, end_ns
        self._session_raw = start_time.date() == end_time.date()
        return True

    def _session_row(self, stock_id, start_time, end_time):
        if self._session_start is None or type(stock_id) is not str or self.quote is not self._session_quote:
            return None
        # Orders normally retain these exact Timestamp objects; only external
        # queries need equality checks. Guard types before any such comparison.
        if start_time is not self._session_start or end_time is not self._session_end:
            if (type(start_time) is not pd.Timestamp or type(end_time) is not pd.Timestamp
                    or start_time != self._session_start or end_time != self._session_end):
                return None
        cached = self._session_rows.get(stock_id)
        if cached is not None:
            return cached
        panel = self.quote.data.get(stock_id)
        if panel is None or type(panel.data) is not np.ndarray or panel.data.dtype != np.dtype("float64"):
            return None
        index = panel.index.idx_list
        if index.dtype != np.dtype("datetime64[ns]"):
            return None
        row = panel.indices[0].index_map.get(self._session_start_ns)
        if row is None:
            # The native same-day shortcut looks up exactly start_time and
            # returns None when absent; it never selects a later intraday bar.
            if self._session_raw:
                return None
            row = int(index.searchsorted(self._session_start_ns, side="left"))
        # Keep every lazy lookup above and numeric operation below unchanged.
        # Only the comparison suffix has an explicit, verified optional helper.
        valid = NotImplemented
        if self._range_validator is not None:
            # Live namespaces observe descriptors installed by prefix callbacks;
            # unsupported field hooks must retain the original read/error order.
            if (type(self) is FastSignedExchange
                    and type.__getattribute__(FastSignedExchange, "__getattribute__") is object.__getattribute__
                    and type.__getattribute__(FastSignedExchange, "__mro__") is _RANGE_MRO
                    and not any("_session_start_ns" in namespace or "_session_end_ns" in namespace
                                for namespace in _RANGE_NAMESPACES)):
                valid = self._range_validator(index, row, self._session_start_ns, self._session_end_ns)
            if valid is NotImplemented:
                self.range_counts["native_checks"] += 1
            elif type(valid) is bool:
                self.range_counts["compiled_checks"] += 1
            else:
                raise TypeError("date-range validator must return bool or NotImplemented")
        # Prove exactly one observed bar, including duplicated timestamps. A
        # decline resumes this expression only; never repeat lookup callbacks.
        if valid is False:
            return None
        if valid is NotImplemented:
            if (row == len(index) or index[row] > self._session_end_ns
                    or (row > 0 and index[row - 1] >= self._session_start_ns)
                    or (row + 1 < len(index) and index[row + 1] <= self._session_end_ns)):
                return None
        fields = panel.indices[1].index_map
        try:
            price = panel.data[row, fields["$open"]]
            factor = panel.data[row, fields["$factor"]]
            close = panel.data[row, fields["$close"]]
            buy = panel.data[row, fields["limit_buy"]]
            sell = panel.data[row, fields["limit_sell"]]
        except KeyError:
            return None
        if not self._session_raw:
            price = None if np.isnan(price) else price
            factor = None if np.isnan(factor) else factor
            close = None if np.isnan(close) else close
            buy, sell = np.bool_(buy), np.bool_(sell)
        suspended = price is None or not math.isfinite(price) or price <= 0
        cached = (price, factor, close, buy, sell, suspended,
                  not (suspended or bool(buy or sell)),
                  not (suspended or buy), not (suspended or sell))
        self._session_rows[stock_id] = cached
        return cached

    def get_deal_price(self, stock_id, start_time, end_time, direction, method="ts_data_last"):
        # Let native code own errors for unusual directions and changed settings.
        if (type(direction) not in (int, type(Order.BUY)) or direction not in (Order.BUY, Order.SELL)
                or self.buy_price != "$open" or self.sell_price != "$open"
                or type(method) is not str or method != "ts_data_last"):
            return super().get_deal_price(stock_id, start_time, end_time, direction, method)
        row = self._session_row(stock_id, start_time, end_time)
        return row[0] if row is not None else super().get_deal_price(stock_id, start_time, end_time, direction, method)

    def get_factor(self, stock_id, start_time, end_time):
        row = self._session_row(stock_id, start_time, end_time)
        return row[1] if row is not None else super().get_factor(stock_id, start_time, end_time)

    def get_close(self, stock_id, start_time, end_time, method="ts_data_last"):
        if type(method) is not str or method != "ts_data_last":
            return super().get_close(stock_id, start_time, end_time, method)
        row = self._session_row(stock_id, start_time, end_time)
        return row[2] if row is not None else super().get_close(stock_id, start_time, end_time, method)

    def check_stock_suspended(self, stock_id, start_time, end_time):
        if self.buy_price != "$open" or self.sell_price != "$open":
            return super().check_stock_suspended(stock_id, start_time, end_time)
        row = self._session_row(stock_id, start_time, end_time)
        return row[5] if row is not None else super().check_stock_suspended(stock_id, start_time, end_time)

    def check_stock_limit(self, stock_id, start_time, end_time, direction=None):
        if type(direction) not in (type(None), int, type(Order.BUY)) or direction not in (None, Order.BUY, Order.SELL):
            return super().check_stock_limit(stock_id, start_time, end_time, direction)
        row = self._session_row(stock_id, start_time, end_time)
        if row is None:
            return super().check_stock_limit(stock_id, start_time, end_time, direction)
        return bool(row[3] or row[4]) if direction is None else row[3] if direction == Order.BUY else row[4]

    def is_stock_tradable(self, stock_id, start_time, end_time, direction=None):
        if (type(direction) not in (type(None), int, type(Order.BUY)) or direction not in (None, Order.BUY, Order.SELL)
                or self.buy_price != "$open" or self.sell_price != "$open"):
            return super().is_stock_tradable(stock_id, start_time, end_time, direction)
        row = self._session_row(stock_id, start_time, end_time)
        if row is None:
            return super().is_stock_tradable(stock_id, start_time, end_time, direction)
        return row[6] if direction is None else row[7] if direction == Order.BUY else row[8]


class FastSignedWeightStrategy(SignedWeightStrategy):
    """Fuse the three default target queries, retaining native scalar sizing.

    The boundary below derives from MIT-licensed signed_weight.py at the
    supported pin (byte-unchanged in tubby.3). Recheck the complete boundary on
    upgrade. Unsupported bindings use the native strategy; an unavailable row
    uses the original per-symbol query sequence without restarting the record.
    """

    def __init__(self, *, compiled_targets=False, **kwargs):
        if type(compiled_targets) is not bool:
            raise TypeError("compiled_targets must be a boolean")
        self.compiled_targets = compiled_targets
        self.target_session_counts = {"compiled_sessions": 0, "native_fallback_sessions": 0,
                                      "target_symbols": 0, "orders": 0}
        super().__init__(**kwargs)

    def _can_fuse_targets(self, exchange):
        if (type(self) is not FastSignedWeightStrategy
                or type(self.trade_position) is not SignedPosition
                or type(exchange) is not FastSignedExchange
                or SignedWeightStrategy.generate_trade_decision is not _NATIVE_GENERATE
                or self._weights is not _NATIVE_WEIGHTS
                or not exchange.supports_session_rows
                or exchange.quote.freq != pd.Timedelta(days=1)
                or exchange.quote.region not in ("cn", "tw", "us")
                or exchange.buy_price != "$open" or exchange.sell_price != "$open"):
            return False
        for name, original in _FUSED_EXCHANGE_METHODS.items():
            bound = getattr(exchange, name)
            if (getattr(bound, "__self__", None) is not exchange
                    or getattr(bound, "__func__", None) is not original):
                return False
        return _reviewed_quote_binding(exchange.quote)

    def generate_trade_decision(self, execute_result=None):
        position = self.trade_position
        exchange = self.trade_exchange
        if not self._can_fuse_targets(exchange):
            # Fallback must not repeat a potentially stateful calendar call.
            # Clear old rows so altered quote callbacks remain visible to the
            # native query sequence; supported default sessions are set below.
            if type(exchange) is FastSignedExchange:
                exchange._session_start = exchange._session_end = None
                exchange._session_rows = {}
            if self.compiled_targets:
                self.target_session_counts["native_fallback_sessions"] += 1
            return super().generate_trade_decision(execute_result)

        calendar = self.trade_calendar
        step = calendar.get_trade_step()
        start, end = calendar.get_step_time(step)
        record = {
            "session": start,
            "signal_session": None,
            "target_weights": None,
            "target_quantities": None,
            "sizing_equity": None,
            "reasons": {},
            "issued_orders": [],
        }
        self.decisions.append(record)
        if calendar.start_index + step < 1:
            record["reasons"]["session"] = "no_preceding_calendar_session"
            return TradeDecisionWO([], self)
        previous_start, previous_end = calendar.get_step_time(step, shift=1)
        record["signal_session"] = previous_start
        signal = self.signal.get_signal(start_time=previous_start, end_time=previous_end)
        if signal is None:
            record["reasons"]["session"] = "missing_signal"
            return TradeDecisionWO([], self)
        weights = self._weights(signal)
        holdings = position.get_stock_amount_dict()
        target_weights = {symbol: weights.get(symbol, 0.0) for symbol in sorted(weights.keys() | holdings.keys())}
        record["target_weights"] = target_weights
        equity = position.calculate_value()
        if not math.isfinite(equity):
            raise ValueError("sizing equity must be finite")
        record["sizing_equity"] = equity
        if equity <= 0:
            record["reasons"]["session"] = "nonpositive_equity"
            return TradeDecisionWO([], self)

        # Signal/position callbacks may have changed exchange bindings/settings.
        fuse = self._can_fuse_targets(exchange) and exchange.begin_session(start, end)
        if not fuse:
            exchange._session_start = exchange._session_end = None
            exchange._session_rows = {}
        if self.compiled_targets:
            result = NotImplemented
            if fuse:
                from .target_session import try_target_orders
                result = try_target_orders(exchange, start, end, target_weights, holdings, equity)
            if result is not NotImplemented:
                targets, reasons, orders = result
                record["target_quantities"] = targets.copy()
                record["reasons"].update(reasons)
                record["issued_orders"] = orders.copy()
                self.target_session_counts["compiled_sessions"] += 1
                self.target_session_counts["target_symbols"] += len(target_weights)
                self.target_session_counts["orders"] += len(orders)
                return TradeDecisionWO(orders, self)
            self.target_session_counts["native_fallback_sessions"] += 1
        targets = {}
        for symbol, weight in target_weights.items():
            row = exchange._session_row(symbol, start, end) if fuse else None
            tradable = row[6] if row is not None else exchange.is_stock_tradable(symbol, start, end)
            if not tradable:
                record["reasons"][symbol] = "unavailable_for_trade"
                continue
            price = row[0] if row is not None else exchange.get_deal_price(symbol, start, end, Order.BUY)
            if not isinstance(price, Real) or not math.isfinite(price) or price <= 0:
                record["reasons"][symbol] = "missing_or_invalid_open"
                continue
            factor = row[1] if row is not None else exchange.get_factor(symbol, start, end)
            if not isinstance(factor, Real) or not math.isfinite(factor) or factor != 1:
                raise NotImplementedError("SignedWeightStrategy requires factor=1")
            targets[symbol] = (1 if weight >= 0 else -1) * math.floor(abs(weight) * equity / price)
        record["target_quantities"] = targets.copy()

        reductions = []
        increases = []

        def make_order(symbol, amount, direction):
            return Order(symbol, amount, direction, start, end)

        for symbol, target in targets.items():
            current = holdings.get(symbol, 0)
            if not isinstance(current, Real) or not math.isfinite(current) or current != math.floor(current):
                raise ValueError("signed holdings must be finite whole shares")
            current = int(current)
            if current == target:
                continue
            if current and target and (current > 0) != (target > 0):
                reductions.append(make_order(symbol, abs(current), Order.SELL if current > 0 else Order.BUY))
                increases.append(make_order(symbol, abs(target), Order.BUY if target > 0 else Order.SELL))
            elif abs(target) < abs(current):
                reductions.append(make_order(symbol, abs(target - current), Order.BUY if target > current else Order.SELL))
            else:
                increases.append(make_order(symbol, abs(target - current), Order.BUY if target > current else Order.SELL))
        reductions.sort(key=lambda order: (order.direction == Order.BUY, order.stock_id))
        orders = reductions + increases
        record["issued_orders"] = orders.copy()
        return TradeDecisionWO(orders, self)


_FUSED_EXCHANGE_METHODS = {
    name: getattr(FastSignedExchange, name)
    for name in ("is_stock_tradable", "get_deal_price", "get_factor", "get_quote_info",
                 "check_stock_suspended", "check_stock_limit", "_session_row", "begin_session")
}

_RANGE_MRO = FastSignedExchange.__mro__
_RANGE_NAMESPACES = tuple(vars(base) for base in _RANGE_MRO)
