# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
# Derived narrowly from fast_signed.py and its cited signed_weight.py upstream.
# cython: language_level=3, infer_types=False, annotation_typing=False
# cython: binding=True, boundscheck=True, wraparound=True, cdivision=False
"""UNWIRED private sequential prototype: CPython-version-specific generated C.

No public-API-only claim: normal Cython optimisations can use CPython internals.
Economic values stay Python objects; GIL remains held, no speculative quote work,
no C floor/division/arithmetic, no batch targets, no account/executor replacement.
Caller must verify source/build receipt before opt-in loading. Unsupported hosts
and missing artifacts use the original Python implementation before execution.
"""
import math
from numbers import Real
import numpy as np
import pandas as pd
from qlib.backtest.decision import Order, TradeDecisionWO
from .fast_signed import FastSignedExchange, FastSignedWeightStrategy, _NATIVE_GENERATE

REVIEWED_FAST_SIGNED_SHA256 = "5abd4863c48ea38b970790f8d48f6fd59e5c6eb451389baf721eab0cef724c42"
_PY_GENERATE = FastSignedWeightStrategy.generate_trade_decision


cdef object _row(object self, object stock_id, object start_time, object end_time):
    if self._session_start is None or type(stock_id) is not str or self.quote is not self._session_quote:
        return None
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
        if self._session_raw:
            return None
        row = int(index.searchsorted(self._session_start_ns, side="left"))
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


def session_row(self, stock_id, start_time, end_time):
    """Standalone differential entry point; does not replace an exchange binding."""
    return _row(self, stock_id, start_time, end_time)


def generate_trade_decision(self, execute_result=None):
    # Delegate the incompatible batch-target experiment before any callback.
    if self.compiled_targets:
        return _PY_GENERATE(self, execute_result)
    position = self.trade_position
    exchange = self.trade_exchange
    if not self._can_fuse_targets(exchange):
        if type(exchange) is FastSignedExchange:
            exchange._session_start = exchange._session_end = None
            exchange._session_rows = {}
        return _NATIVE_GENERATE(self, execute_result)
    calendar = self.trade_calendar
    step = calendar.get_trade_step()
    start, end = calendar.get_step_time(step)
    record = {"session": start, "signal_session": None, "target_weights": None,
              "target_quantities": None, "sizing_equity": None,
              "reasons": {}, "issued_orders": []}
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
    fuse = self._can_fuse_targets(exchange) and exchange.begin_session(start, end)
    if not fuse:
        exchange._session_start = exchange._session_end = None
        exchange._session_rows = {}
    targets = {}
    for symbol, weight in target_weights.items():
        row = _row(exchange, symbol, start, end) if fuse else None
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
    for symbol, target in targets.items():
        current = holdings.get(symbol, 0)
        if not isinstance(current, Real) or not math.isfinite(current) or current != math.floor(current):
            raise ValueError("signed holdings must be finite whole shares")
        current = int(current)
        if current == target:
            continue
        if current and target and (current > 0) != (target > 0):
            reductions.append(Order(symbol, abs(current), Order.SELL if current > 0 else Order.BUY, start, end))
            increases.append(Order(symbol, abs(target), Order.BUY if target > 0 else Order.SELL, start, end))
        elif abs(target) < abs(current):
            reductions.append(Order(symbol, abs(target - current), Order.BUY if target > current else Order.SELL, start, end))
        else:
            increases.append(Order(symbol, abs(target - current), Order.BUY if target > current else Order.SELL, start, end))
    reductions.sort(key=lambda order: (order.direction == Order.BUY, order.stock_id))
    orders = reductions + increases
    record["issued_orders"] = orders.copy()
    return TradeDecisionWO(orders, self)


# Separate replay-only prototype. The rejected target loop above is unchanged.
# Fixed record layout is ordered_signed_kernel's 14-column SessionResult ABI.
# The caller still owns/proves the complete-session kernel, calendar, account,
# exchange and quote provenance. This function is NOT a native fill replacement.
from qlib.backtest.signed_position import SignedPosition as _ReplayPosition
from qlib.backtest.account import AccumulatedInfo as _ReplayAccum
from . import ordered_signed_kernel as _replay_kernel

_REPLAY_INIT_STOCK = _ReplayPosition._init_stock
_REPLAY_INIT_CODE = _REPLAY_INIT_STOCK.__code__
_REPLAY_DIRECTION = type(Order.BUY)
_REPLAY_FIELDS = {
    Order: ("stock_id", "deal_amount", "factor", "signed_skip_reason", "signed_partial_reason"),
    _ReplayPosition: ("position", "restricted_proceeds", "transaction_cost", "_init_stock"),
    _ReplayAccum: ("cost", "to", "rtn"),
}
_REPLAY_DICT_DESCRIPTORS = {
    cls: tuple((base, vars(base).get("__dict__")) for base in cls.__mro__)
    for cls in _REPLAY_FIELDS
}


def _replay_plain_class(cls):
    # Identity alone does not exclude later monkey-patched descriptors/hooks.
    if (cls.__getattribute__ is not object.__getattribute__
            or cls.__setattr__ is not object.__setattr__
            or cls.__delattr__ is not object.__delattr__):
        return False
    for base in cls.__mro__:
        namespace = vars(base)
        if "__del__" in namespace or "__getattr__" in namespace:
            return False
        for name in _REPLAY_FIELDS[cls]:
            if name == "_init_stock":
                continue
            value = namespace.get(name)
            if value is not None and hasattr(type(value), "__get__"):
                return False
    if any(vars(base).get("__dict__") is not descriptor
           for base, descriptor in _REPLAY_DICT_DESCRIPTORS[cls]):
        return False
    return True


def _replay_inert(value, depth=0):
    # Replaced/deleted values must not trigger custom finalizers, hashing or
    # numeric conversion. Restrict metadata too: removing a native entry can
    # release its last reference. Cycles/deep/custom containers decline.
    if type(value) in (str, int, float, bool, type(None), np.float64, np.bool_, pd.Timestamp):
        return True
    if type(value) is _REPLAY_DIRECTION:
        return (value is Order.BUY or value is Order.SELL) and all(
            "__del__" not in vars(base) for base in _REPLAY_DIRECTION.__mro__)
    if depth >= 8:
        return False
    if type(value) is dict:
        return all(type(key) is str and _replay_inert(item, depth + 1)
                   for key, item in value.items())
    if type(value) in (tuple, list):
        return all(_replay_inert(item, depth + 1) for item in value)
    return False


def _replay_finite_plain(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


cdef object _replay_value(const double[:, ::1] records, Py_ssize_t row, Py_ssize_t column):
    # Binary64 load followed by Python float boxing ONLY. No C economic math.
    return records[row, column]


def try_replay_records(position, accum, orders, quote_rows, records):
    """Unwired replay suffix, or NotImplemented BEFORE mutation on guard failure.

    Input records MUST be the successful verified kernel result for this exact
    unchanged state/orders/quotes, not caller-authored economics. No recomputing
    economics here can prove that provenance. Caller must prohibit concurrent
    native-state mutation and unsupported callbacks over the full boundary.

    GIL remains held. A bytes-owned snapshot prevents later writable NumPy
    aliases from changing the memoryview. Reads box to Python float; comparisons,
    native method calls, dictionary changes and assignments retain Python order.
    Once replay begins, exceptions propagate with their prefix, NEVER fallback.
    MemoryError/interpreter tracing/profiling callbacks are outside equivalence.
    """
    cdef const double[:, ::1] view
    cdef Py_ssize_t i, count
    if (type(position) is not _ReplayPosition or type(accum) is not _ReplayAccum
            or type(orders) is not list or type(quote_rows) is not list
            or type(records) is not np.ndarray or records.dtype != np.dtype("float64")
            or records.ndim != 2 or not records.flags.c_contiguous
            or not records.flags.aligned or records.shape != (len(orders), 14)
            or len(quote_rows) != len(orders)):
        return NotImplemented
    if not all(_replay_plain_class(cls) for cls in (Order, _ReplayPosition, _ReplayAccum)):
        return NotImplemented
    state = position.__dict__
    totals = accum.__dict__
    if type(state) is not dict or type(totals) is not dict:
        return NotImplemented
    if (type(state.get("position")) is not dict
            or type(state.get("restricted_proceeds")) is not dict
            or "_init_stock" in state or _ReplayPosition._init_stock is not _REPLAY_INIT_STOCK
            or _REPLAY_INIT_STOCK.__code__ is not _REPLAY_INIT_CODE
            or not all(type(key) is str for key in state["position"])
            or not all(type(key) is str for key in state["restricted_proceeds"])
            or not _replay_finite_plain(state.get("transaction_cost"))
            or any(not _replay_finite_plain(totals.get(name)) for name in ("cost", "to", "rtn"))):
        return NotImplemented
    seen_orders, seen_entries, seen_symbols = set(), set(), set()
    for order, quote_row in zip(orders, quote_rows):
        if type(order) is not Order or id(order) in seen_orders:
            return NotImplemented
        seen_orders.add(id(order))
        attributes = order.__dict__
        if type(attributes) is not dict or not all(type(key) is str for key in attributes):
            return NotImplemented
        symbol = attributes.get("stock_id")
        if (type(symbol) is not str or not symbol
                or symbol in {"cash", "cash_delay", "now_account_value"}
                or any(not _replay_inert(attributes.get(name)) for name in
                       ("deal_amount", "factor", "signed_skip_reason", "signed_partial_reason"))
                or type(quote_row) is not tuple or len(quote_row) != 9
                or type(quote_row[1]) not in (float, int, np.float64)):
            return NotImplemented
        entry = state["position"].get(symbol)
        if entry is not None:
            if (type(entry) is not dict or id(entry) in seen_entries
                    and symbol not in seen_symbols
                    or not _replay_inert(entry)
                    or type(entry.get("amount")) not in (int, float)
                    or type(entry.get("price")) not in (int, float)):
                return NotImplemented
            seen_entries.add(id(entry))
        if type(state["restricted_proceeds"].get(symbol, 0.0)) not in (int, float):
            return NotImplemented
        # Repeated symbols are valid; distinct symbols sharing one entry are not.
        seen_symbols.add(symbol)
    # Snapshot BEFORE guard calculations. Never expose a const view of mutable
    # caller storage. Immutable bytes own these exact bit patterns, including
    # signed zero and rejected-price NaN payloads.
    owned = np.frombuffer(records.tobytes(order="C"), dtype=np.float64).reshape((len(orders), 14))
    statuses = owned[:, 12]
    if (not np.all(np.isin(statuses, (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)))
            or not np.all(np.isin(owned[:, 13], (0.0, 1.0)))
            or not np.all(np.isin(owned[:, 7], (0.0, 1.0)))
            or not np.all(np.isfinite(owned[:, (0, 1, 2, 4, 5, 6, 8, 9, 10, 11)]))):
        return NotImplemented
    for i in range(len(orders)):
        status = int(owned[i, 12])
        if status in (4, 5):
            if (not np.isnan(owned[i, 3]) or owned[i, 13] != 0
                    or owned[i, 0] != 0 or owned[i, 1] != 0 or owned[i, 2] != 0):
                return NotImplemented
        elif (not math.isfinite(float(owned[i, 3])) or owned[i, 3] <= 0
                or owned[i, 13] != 1 or quote_rows[i][1] != 1):
            return NotImplemented
    count = len(orders)
    if count == 0:
        return []
    view = owned
    execute_result = []
    for i in range(count):
        order, quote_row = orders[i], quote_rows[i]
        status = int(_replay_value(view, i, 12))
        order.deal_amount = _replay_value(view, i, 0)
        if _replay_value(view, i, 13):
            order.factor = quote_row[1]
        if status in (_replay_kernel.INVALID_OPEN, _replay_kernel.NOT_TRADABLE):
            order.signed_skip_reason = "missing_or_invalid_open" if status == _replay_kernel.INVALID_OPEN else "suspended_or_limited"
        else:
            order.signed_skip_reason = {_replay_kernel.ZERO: "zero_amount", _replay_kernel.NO_CASH: "insufficient_free_cash"}.get(status)
            order.signed_partial_reason = "insufficient_free_cash" if status == _replay_kernel.PARTIAL else None
        value, cost, price = (_replay_value(view, i, 1), _replay_value(view, i, 2), _replay_value(view, i, 3))
        if order.deal_amount > 0 and value != 0:
            amount, basis = _replay_value(view, i, 6), _replay_value(view, i, 5)
            if amount == 0:
                position.position.pop(order.stock_id, None)
                position.restricted_proceeds.pop(order.stock_id, None)
            else:
                entry = position.position.get(order.stock_id)
                if entry is None:
                    position._init_stock(order.stock_id, amount, price)
                else:
                    entry["amount"], entry["price"] = amount, price
                if amount < 0:
                    position.restricted_proceeds[order.stock_id] = basis
                else:
                    position.restricted_proceeds.pop(order.stock_id, None)
            position.position["cash"] = _replay_value(view, i, 4)
            position.transaction_cost = _replay_value(view, i, 11)
        execute_result.append((order, value, cost, price))
    accum.cost, accum.to, accum.rtn = (_replay_value(view, count - 1, 8),
                                    _replay_value(view, count - 1, 9),
                                    _replay_value(view, count - 1, 10))
    return execute_result


# Independent date-comparison suffix. Native _session_row MUST execute its
# original lazy quote/map prefix once, and its original numeric suffix unchanged.
# A declined helper resumes only the original comparison expression, never the
# whole row lookup. No map scans, priming, price cache or economic arithmetic.
from libc.stdint cimport int64_t

_RANGE_ARRAY = np.ndarray
_RANGE_DATETIME = np.datetime64
_RANGE_NS_DTYPE = np.dtype("datetime64[ns]")
_RANGE_INT_DTYPE = np.dtype("int64")


def try_valid_range(index, row, start_ns, end_ns):
    """Return original single-bar range validity, or decline without callbacks.

    Invoke AFTER native quote.data.get/index_map.get/searchsorted. Exact ndarray,
    scalar and int types exclude user access/conversion callbacks. NaT makes all
    ordered datetime comparisons false, including when the boundary is NaT.
    Negative/out-of-range rows decline so native indexing owns their exceptions.
    Per-call ndarray view and memoryview creation are part of measured cost.

    GIL stays held; tracing/profiling, concurrent alias mutation and allocation
    failure are outside equivalence, as for the existing sequential prototype.
    No persistent view: later calls observe later native index mutations.
    """
    cdef const int64_t[::1] view
    cdef int64_t left, right, value
    cdef int64_t nat = -9223372036854775807 - 1
    cdef Py_ssize_t location, count
    # Captured immutable native types/dtypes: no mutable NumPy module binding
    # is read or invoked here. Caller keeps all original module use in order.
    if (type(index) is not _RANGE_ARRAY or type(row) is not int
            or type(start_ns) is not _RANGE_DATETIME
            or type(end_ns) is not _RANGE_DATETIME):
        return NotImplemented
    if (index.ndim != 1 or index.dtype != _RANGE_NS_DTYPE
            or not index.dtype.isnative or not index.flags.c_contiguous
            or not index.flags.aligned
            or start_ns.dtype != _RANGE_NS_DTYPE or end_ns.dtype != _RANGE_NS_DTYPE):
        return NotImplemented
    count = len(index)
    if row < 0 or row > count:
        return NotImplemented
    if row == count:
        return False
    location = row
    left = start_ns.view(_RANGE_INT_DTYPE)
    right = end_ns.view(_RANGE_INT_DTYPE)
    view = index.view(_RANGE_INT_DTYPE)
    value = view[location]
    if value != nat and right != nat and value > right:
        return False
    if location > 0:
        value = view[location - 1]
        if value != nat and left != nat and value >= left:
            return False
    if location + 1 < count:
        value = view[location + 1]
        if value != nat and right != nat and value <= right:
            return False
    return True
