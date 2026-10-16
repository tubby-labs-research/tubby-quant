# Copyright (c) Microsoft Corporation and Tubby Labs contributors.
# Licensed under the MIT License.
"""Experimental private flat-order indicator helper; upstream remains unchanged.

Semantics derive from MIT-licensed Qlib report.py/high_performance_ds.py and
index_data.py at reviewed tubby.4 (8ab6249). Complete guards, packing, native
constructors, FFR and commit belong in its timing boundary. No JIT is used.
Existing account reset_report overrides may install FlatIndicator explicitly;
the default native path is not activated or replaced by this module.
"""
from collections import OrderedDict
import math
from types import MethodType

import numpy as np

from qlib.backtest.decision import Order
from qlib.backtest.high_performance_ds import BaseOrderIndicator, NumpyOrderIndicator
from qlib.backtest.report import Indicator
from qlib.utils.index_data import Index, IndexData, LocIndexer, SingleData

_INDICATOR_UPDATE = Indicator.update_order_indicators
_INDICATOR_TRADE = Indicator._update_order_trade_info
_INDICATOR_FFR = Indicator._update_order_fulfill_rate
_ASSIGN = NumpyOrderIndicator.assign
_TRANSFER = BaseOrderIndicator.transfer
_INDEX_INIT, _SINGLE_INIT, _DATA_INIT = Index.__init__, SingleData.__init__, IndexData.__init__
_ORDER_GETATTRIBUTE = Order.__getattribute__
_INDICATOR_GETATTRIBUTE = Indicator.__getattribute__
_STORAGE_GETATTRIBUTE = NumpyOrderIndicator.__getattribute__
_REINDEX, _REPLACE, _ALIGN = SingleData.reindex, SingleData.replace, SingleData._align_indices
_INDEX_EQUAL = Index.__eq__
_PROC_INDICES = LocIndexer.proc_idx_l
_DIVISION = vars(IndexData)['__truediv__']
_PROPERTIES = {name: vars(Order)[name] for name in ('amount_delta', 'deal_amount_delta', 'sign')}
_ABSENT = object()
_ORDER_FIELDS = {name: vars(Order).get(name, _ABSENT)
                 for name in ('amount', 'deal_amount', 'stock_id', 'direction')}
_STORAGE_FIELDS = {name: vars(NumpyOrderIndicator).get(name, _ABSENT) for name in ('data',)}
_NUMERIC_TYPES = (int, float, np.float64, np.int64)
_DIRECTION_TYPES = (int, type(Order.BUY))
_POLICY = {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'ignore'}
FIELDS = ('amount', 'inner_amount', 'deal_amount', 'trade_price', 'trade_value',
          'trade_cost', 'trade_dir', 'pa', 'ffr')


def _bound(obj, name, method):
    value = getattr(obj, name)
    return type(value) is MethodType and value.__self__ is obj and value.__func__ is method


def try_update_order_indicators(indicator, trade_info):
    """Commit complete scratch metrics and return True, or decline atomically.

    Unsupported/custom state must use native update_order_indicators, preserving
    native callbacks, exceptional value conversions and error-phase mutations.
    Nine native SingleData metrics retain duplicate last-write/first-insertion
    behavior. Ordinary NaN trade prices and zero-amount FFR NaNs are preserved.
    """
    if (type(indicator) not in (Indicator, FlatIndicator) or type(trade_info) is not list
            or Indicator.__getattribute__ is not _INDICATOR_GETATTRIBUTE
            or FlatIndicator.__getattribute__ is not _INDICATOR_GETATTRIBUTE
            or Indicator.update_order_indicators is not _INDICATOR_UPDATE
            or Indicator._update_order_trade_info is not _INDICATOR_TRADE
            or Indicator._update_order_fulfill_rate is not _INDICATOR_FFR
            or NumpyOrderIndicator.__getattribute__ is not _STORAGE_GETATTRIBUTE
            or NumpyOrderIndicator.assign is not _ASSIGN or NumpyOrderIndicator.transfer is not _TRANSFER
            or Order.__getattribute__ is not _ORDER_GETATTRIBUTE
            or any(vars(Order).get(name) is not original for name, original in _PROPERTIES.items())
            or any(vars(Order).get(name, _ABSENT) is not original for name, original in _ORDER_FIELDS.items())
            or any(vars(NumpyOrderIndicator).get(name, _ABSENT) is not original for name, original in _STORAGE_FIELDS.items())
            or FlatIndicator._update_order_trade_info is not _INDICATOR_TRADE
            or FlatIndicator._update_order_fulfill_rate is not _INDICATOR_FFR
            or Index.__init__ is not _INDEX_INIT or SingleData.__init__ is not _SINGLE_INIT
            or IndexData.__init__ is not _DATA_INIT or Index.__eq__ is not _INDEX_EQUAL
            or LocIndexer.proc_idx_l is not _PROC_INDICES
            or SingleData.reindex is not _REINDEX or SingleData.replace is not _REPLACE
            or SingleData._align_indices is not _ALIGN or vars(IndexData)['__truediv__'] is not _DIVISION
            or np.geterr() != _POLICY):
        return False
    raw = object.__getattribute__(indicator, '__dict__')
    storage = raw.get('order_indicator')
    if (raw.get('order_indicator_cls') is not NumpyOrderIndicator or type(storage) is not NumpyOrderIndicator
            or not _bound(indicator, '_update_order_trade_info', _INDICATOR_TRADE)
            or not _bound(indicator, '_update_order_fulfill_rate', _INDICATOR_FFR)
            or not _bound(storage, 'assign', _ASSIGN) or not _bound(storage, 'transfer', _TRANSFER)):
        return False
    metrics = object.__getattribute__(storage, '__dict__').get('data')
    if type(metrics) is not OrderedDict or metrics:
        return False
    # Validate every primitive tuple/order before allocating numeric scratch or
    # evaluating properties. Native fallback alone owns arbitrary callbacks.
    for item in trade_info:
        if type(item) is not tuple or len(item) != 4 or type(item[0]) is not Order:
            return False
        order = object.__getattribute__(item[0], '__dict__')
        symbol, amount, deal, direction = (order.get('stock_id'), order.get('amount'),
                                         order.get('deal_amount'), order.get('direction'))
        if (type(symbol) is not str or type(amount) not in _NUMERIC_TYPES or type(deal) not in _NUMERIC_TYPES
                or type(direction) not in _DIRECTION_TYPES or direction not in (Order.BUY, Order.SELL)
                or not math.isfinite(amount) or not math.isfinite(deal)
                or not 0 <= deal <= amount <= 2 ** 52):
            return False
        value, cost, price = item[1:]
        if (type(value) not in _NUMERIC_TYPES or type(cost) not in _NUMERIC_TYPES
                or not math.isfinite(value) or not math.isfinite(cost)
                or (price is not None and type(price) not in _NUMERIC_TYPES)):
            return False
    slots, rows = {}, []
    for item in trade_info:
        order, value, cost, price = item
        state = object.__getattribute__(order, '__dict__')
        symbol, direction = state['stock_id'], state['direction']
        sign = direction * 2 - 1
        amount, deal = state['amount'] * sign, state['deal_amount'] * sign
        row = (amount, amount, deal, price, value * sign, cost, direction, 0.)
        slot = slots.get(symbol)
        if slot is None:
            slots[symbol] = len(rows)
            rows.append(row)
        else:
            rows[slot] = row
    # Native SingleData({}) routes its empty index through Index(0), preserving
    # integer empty labels, ndarray index_map and _is_sorted=True.
    index = Index(list(slots)) if slots else Index(0)
    columns = np.array(rows, dtype=np.float64).reshape((-1, 8))
    scratch = OrderedDict((name, SingleData(columns[:, i], index)) for i, name in enumerate(FIELDS[:8]))
    # Native deal_amount is finite in this guarded domain; replace({nan:0})
    # changes no entries. Retain exact NumPy division including signed zero and
    # native zero/zero NaNs under the reviewed invalid='ignore' policy.
    scratch['ffr'] = SingleData(columns[:, 2] / columns[:, 0], index)
    metrics.update(scratch)
    return True


class FlatIndicator(Indicator):
    """Explicit opt-in factory for existing private account reset_report hooks."""
    def __init__(self, order_indicator_cls=NumpyOrderIndicator):
        self.flat_update_counts = {'fast': 0, 'fallback': 0}
        super().__init__(order_indicator_cls=order_indicator_cls)

    def update_order_indicators(self, trade_info):
        if try_update_order_indicators(self, trade_info):
            self.flat_update_counts['fast'] += 1
            return
        self.flat_update_counts['fallback'] += 1
        return super().update_order_indicators(trade_info)
