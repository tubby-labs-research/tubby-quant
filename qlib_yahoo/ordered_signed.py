"""Experimental complete-session adapter; explicit opt-in, not the default.

The numeric formulas derive from the MIT-licensed tubby.3 signed ledger.
Unsupported state/bindings return NotImplemented before economic mutation.
Native dictionary mutations are replayed in fill order, not replaced by sorted
array reductions; native bar-close marks, funding and indicators remain intact.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from qlib.backtest.account import AccumulatedInfo
from qlib.backtest.decision import Order
from qlib.backtest.decision import TradeDecisionWO
from qlib.backtest.executor import SimulatorExecutor
from qlib.backtest.utils import TradeCalendarManager
from qlib.backtest.signed_account import SignedAccount
from qlib.backtest.signed_exchange import SignedExchange
from qlib.backtest.signed_position import SignedPosition

from . import ordered_signed_kernel as kernel
from .fast_signed import FastSignedExchange, _reviewed_quote_binding


def activate_compiler(path):
    """Explicit isolated dependencies only; never installs or uses a disk cache."""
    from qlib_yahoo.dataset import DatasetError

    folder = Path(path).absolute()
    if folder.is_symlink() or not folder.is_dir():
        raise DatasetError("Ordered-session compiler path is missing or redirected")
    folder = folder.resolve()
    for entry in folder.iterdir():
        if entry.name not in ("numba", "llvmlite", "llvmlite.libs", "bin", "Scripts") and not (
                entry.name.startswith(("numba-", "llvmlite-")) and entry.name.endswith(".dist-info")):
            raise DatasetError("Isolated compiler directory contains an unexpected entry")
    sys.path.insert(0, str(folder))
    import numba
    import llvmlite
    if (numba.__version__ != "0.65.1" or llvmlite.__version__ != "0.47.0"
            or not Path(numba.__file__).resolve().is_relative_to(folder)
            or not Path(llvmlite.__file__).resolve().is_relative_to(folder)):
        raise DatasetError("Ordered-session compiler identity differs from the reviewed optional pins")
    return {"numba": numba.__version__, "llvmlite": llvmlite.__version__,
            "isolated_path": str(folder), "fastmath": False, "parallel": False,
            "disk_cache": False, "first_use_compilation": "included in daily simulation"}


_POSITION_BINDINGS = {
    name: getattr(SignedPosition, name) for name in (
        "update_order", "preview_fill", "_fill_effect", "value_touched_by", "get_cash",
        "get_stock_amount", "get_stock_price", "check_stock", "_init_stock")
}
_ACCOUNT_BINDINGS = {"update_order": SignedAccount.update_order}
_ACCUM_BINDINGS = {
    name: getattr(AccumulatedInfo, name) for name in ("add_cost", "add_turnover", "add_return_value")
}
_EXCHANGE_BINDINGS = {
    name: getattr(FastSignedExchange, name) for name in (
        "deal_order", "_calc_trade_info_by_order", "check_order", "get_deal_price",
        "get_factor", "is_stock_tradable", "check_stock_suspended", "check_stock_limit",
        "_session_row", "begin_session", "_validate_amount", "_require_position")
}
_CALENDAR_BINDINGS = {
    name: getattr(TradeCalendarManager, name) for name in ("get_step_time", "get_trade_step")
}
_EPSILON_CHANGE = TradeCalendarManager.get_step_time.__globals__["epsilon_change"]
_EXECUTOR_ORDER_ITERATOR = SimulatorExecutor._get_order_iterator
_DECISION_GET = TradeDecisionWO.get_decision


def _native_bindings(instance, bindings):
    for name, original in bindings.items():
        bound = getattr(instance, name)
        if bound is original:  # Reviewed static methods.
            continue
        if getattr(bound, "__self__", None) is not instance or getattr(bound, "__func__", None) is not original:
            return False
    return True


def _eligible(exchange, account, orders, account_type=SignedAccount):
    position = account.current_position
    errors = np.geterr()
    if (type(account_type) is not type or not issubclass(account_type, SignedAccount)
            or type(exchange) is not FastSignedExchange or type(account) is not account_type
            or type(position) is not SignedPosition or type(account.accum_info) is not AccumulatedInfo
            or type(orders) is not list or type(position.position) is not dict
            or type(position.restricted_proceeds) is not dict
            or position._settle_type != position.ST_NO or not exchange.supports_session_rows
            or exchange.buy_price != "$open" or exchange.sell_price != "$open"
            or exchange.open_cost != exchange.close_cost
            or type(exchange.open_cost) is not float or not math.isfinite(exchange.open_cost)
            or exchange.open_cost < 0 or not _reviewed_quote_binding(exchange.quote)
            or errors["divide"] != "warn" or errors["over"] != "warn"
            or errors["under"] != "ignore" or errors["invalid"] not in ("warn", "ignore")):
        return False
    if not all(_native_bindings(instance, methods) for instance, methods in (
            (exchange, _EXCHANGE_BINDINGS), (position, _POSITION_BINDINGS),
            (account, _ACCOUNT_BINDINGS), (account.accum_info, _ACCUM_BINDINGS))):
        return False
    active = account._active_session
    if type(active) is not pd.Timestamp or active.tz is not None or active != active.normalize():
        return False
    if not orders:
        return True
    if type(orders[0]) is not Order:
        return False
    start, end = orders[0].start_time, orders[0].end_time
    if (type(start) is not pd.Timestamp or type(end) is not pd.Timestamp
            or start.tz is not None or end.tz is not None or start != active
            or start > end):
        return False
    seen = set()
    for order in orders:
        if (type(order) is not Order or id(order) in seen or type(order.stock_id) is not str
                or not order.stock_id or order.stock_id in {"cash", "cash_delay", "now_account_value"}
                or order.start_time is not start or order.end_time is not end
                or type(order.direction) not in (int, type(Order.BUY))
                or order.direction not in (Order.BUY, Order.SELL)
                or type(order.amount) not in (int, float) or not math.isfinite(order.amount)
                or order.amount < 0 or order.amount > kernel.MAX_QUANTITY
                or order.amount != math.floor(order.amount)):
            return False
        seen.add(id(order))
    return exchange.begin_session(start, end)


def try_execute_orders(exchange, account, orders, *, compiled=False, account_type=SignedAccount):
    """Time guards, packing, optional compilation, fills and replay together.

    Callers own native per-order calendar and executor bookkeeping. The runner
    uses OrderedSessionExecutor through the native generic hook; never call this
    directly to bypass executor validation.
    """
    if not _eligible(exchange, account, orders, account_type):
        return NotImplemented
    if not orders:
        return []
    position, accum = account.current_position, account.accum_info
    names = list(dict.fromkeys(order.stock_id for order in orders))
    lookup = {name: index for index, name in enumerate(names)}
    amounts, marks, bases = (np.zeros(len(names), dtype=np.float64) for _ in range(3))
    exists = np.zeros(len(names), dtype=np.bool_)
    seen_entries = set()
    for index, name in enumerate(names):
        entry = position.position.get(name)
        if entry is not None:
            if (type(entry) is not dict or id(entry) in seen_entries
                    or type(entry.get("amount")) not in (int, float)
                    or type(entry.get("price")) not in (int, float)):
                return NotImplemented
            seen_entries.add(id(entry))
            amounts[index], marks[index], exists[index] = entry["amount"], entry["price"], True
        basis = position.restricted_proceeds.get(name, 0.0)
        if type(basis) not in (int, float):
            return NotImplemented
        bases[index] = basis
    rows = [exchange._session_row(order.stock_id, order.start_time, order.end_time) for order in orders]
    if any(row is None for row in rows):
        return NotImplemented
    result = kernel.run_ordered_session(
        amounts, marks, bases, exists,
        np.array([lookup[order.stock_id] for order in orders], dtype=np.int64),
        np.array([order.amount for order in orders], dtype=np.float64),
        np.array([order.direction for order in orders], dtype=np.int64),
        np.array([row[0] for row in rows], dtype=np.float64),
        np.array([row[1] for row in rows], dtype=np.float64),
        np.array([row[7] if order.direction == Order.BUY else row[8]
                  for order, row in zip(orders, rows)], dtype=np.bool_),
        cash=position.get_cash(), rate=exchange.open_cost, transaction_cost=position.transaction_cost,
        accum_cost=accum.cost, accum_turnover=accum.to, accum_return=accum.rtn, compiled=compiled)
    if result is None:
        return NotImplemented
    # Nothing economic has changed until this complete scratch batch succeeds.
    execute_result = []
    for order, quote_row, record in zip(orders, rows, result.records):
        status = int(record[kernel.STATUS])
        order.deal_amount = float(record[kernel.DEAL])
        if record[kernel.FACTOR_SET]:
            order.factor = quote_row[1]  # Keep the native scalar type.
        if status in (kernel.INVALID_OPEN, kernel.NOT_TRADABLE):
            order.signed_skip_reason = "missing_or_invalid_open" if status == kernel.INVALID_OPEN else "suspended_or_limited"
        else:
            order.signed_skip_reason = {kernel.ZERO: "zero_amount", kernel.NO_CASH: "insufficient_free_cash"}.get(status)
            order.signed_partial_reason = "insufficient_free_cash" if status == kernel.PARTIAL else None
        value, cost, price = (float(record[column]) for column in (kernel.VALUE, kernel.COST, kernel.PRICE))
        if order.deal_amount > 0 and value != 0:
            amount, basis = float(record[kernel.AMOUNT]), float(record[kernel.BASIS])
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
            position.position["cash"] = float(record[kernel.CASH])
            position.transaction_cost = float(record[kernel.TRANSACTION_COST])
        execute_result.append((order, value, cost, price))
    last = result.records[-1]
    accum.cost, accum.to, accum.rtn = (float(last[column]) for column in (
        kernel.ACCUM_COST, kernel.ACCUM_TURNOVER, kernel.ACCUM_RETURN))
    return execute_result


class OrderedSessionExecutor:
    """Opt-in handler for the maintained native executor's generic batch hook.

    The caller supplies the exact reviewed native/streaming account type. Custom
    calendar, exchange, decision, account, methods and state retain the original
    sequential path. No dependent orders or accumulated float sums are reordered.
    """

    def __init__(self, account_type=SignedAccount):
        if type(account_type) is not type or not issubclass(account_type, SignedAccount):
            raise TypeError("ordered sessions require a reviewed signed account type")
        self.account_type = account_type
        self.sessions = self.orders = self.fallbacks = 0

    def try_collect(self, executor, decision):
        def decline():
            self.fallbacks += 1
            return NotImplemented

        if (type(executor) is not SimulatorExecutor or executor.trade_type != SimulatorExecutor.TT_SERIAL
                or executor.time_per_step != "day" or executor.verbose
                or executor._settle_type != SignedPosition.ST_NO
                or not executor.generate_portfolio_metrics
                or type(decision) is not TradeDecisionWO
                or not _native_bindings(executor, {"_get_order_iterator": _EXECUTOR_ORDER_ITERATOR})
                or not _native_bindings(decision, {"get_decision": _DECISION_GET})):
            return decline()
        calendar = executor.trade_calendar
        if (type(calendar) is not TradeCalendarManager or not _native_bindings(calendar, _CALENDAR_BINDINGS)
                or TradeCalendarManager.get_step_time.__globals__["epsilon_change"] is not _EPSILON_CHANGE
                or type(calendar._calendar) is not np.ndarray
                or type(calendar.start_index) is not int or type(calendar.trade_step) is not int
                or type(decision.order_list) is not list
                or (executor.deal_day is not None and (type(executor.deal_day) is not pd.Timestamp
                    or executor.deal_day.tz is not None))
                or type(executor.dealt_order_amount) is not defaultdict
                or executor.dealt_order_amount.default_factory is not float):
            return decline()
        orders = decision.order_list
        # These default calendar functions have no mutation/callbacks. A custom
        # calendar is rejected above before even one call, preserving its effects.
        start, end = calendar.get_step_time()
        if (type(start) is not pd.Timestamp or type(end) is not pd.Timestamp
                or start.tz is not None or end.tz is not None
                or any(type(order) is not Order or order.start_time is not start or order.end_time != end for order in orders)):
            return decline()
        day = start.floor(freq="D") if orders else None
        reset_day = bool(orders) and (executor.deal_day is None or day > executor.deal_day)
        if not reset_day and any(type(executor.dealt_order_amount.get(order.stock_id, 0.0)) is not float
                                 for order in orders):
            return decline()
        result = try_execute_orders(executor.trade_exchange, executor.trade_account, orders,
                                    compiled=True, account_type=self.account_type)
        if result is NotImplemented:
            return decline()
        if orders:
            if reset_day:
                executor.dealt_order_amount = defaultdict(float)
                executor.deal_day = day
            for order in orders:
                executor.dealt_order_amount[order.stock_id] += order.deal_amount
        self.sessions += 1
        self.orders += len(orders)
        return result, {"trade_info": result}

    def snapshot(self):
        return {"mode": "guarded compiled complete ordered sessions", "sessions": self.sessions,
                "orders": self.orders, "native_fallback_sessions": self.fallbacks,
                "fastmath": False, "parallel": False, "disk_jit_cache": False,
                "first_use_compilation_included": True}
