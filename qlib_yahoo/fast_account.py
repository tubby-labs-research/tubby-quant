"""Constant-time per-fill accounting for the pinned signed Qlib account.

Pass an instance through Qlib's existing account-injection hook. The position,
daily marks, sizing and portfolio metrics remain the derivative's own methods.
"""

import math

import pandas as pd

from qlib.backtest.signed_account import SignedAccount


LEGACY_COMMIT = "4c1d321fbcce8eacf49b328b7c2807a6d13ae0cb"


def account_class_for(commit: str):
    """Apply this compatibility adapter only to the derivative actually tested.

    PR #26 / tubby.2 implements local valuation natively. Any newer reviewed pin
    must retain its own account implementation, not inherit a stale override.
    """
    return FastSignedAccount if commit == LEGACY_COMMIT else SignedAccount


class FastSignedAccount(SignedAccount):
    """Avoid whole-book valuation solely for the ancillary accum_info return."""

    def update_order(self, order, trade_val, cost, trade_price):
        if self._active_session is None:
            raise ValueError("call start_bar before signed account fills")
        if pd.Timestamp(order.start_time).normalize() != self._active_session:
            raise ValueError("order date differs from active signed session")

        position = self.current_position
        code = order.stock_id
        before_cash = position.get_cash()
        before_restricted = position.restricted_proceeds.get(code, 0.0)
        before_amount = position.get_stock_amount(code)
        before_mark = before_amount * position.get_stock_price(code) if before_amount else 0.0

        # SignedPosition owns fill validation, cost clamp and all ledger mutations.
        position.update_order(order, trade_val, cost, trade_price)

        after_cash = position.get_cash()
        after_restricted = position.restricted_proceeds.get(code, 0.0)
        after_amount = position.get_stock_amount(code)
        after_mark = after_amount * position.get_stock_price(code) if after_amount else 0.0
        local_return = ((after_cash - before_cash)
                        + (after_restricted - before_restricted)
                        + (after_mark - before_mark) + cost)
        if not math.isfinite(local_return):
            raise ValueError("account return exceeds finite ledger range")
        self.accum_info.add_cost(cost)
        self.accum_info.add_turnover(trade_val)
        self.accum_info.add_return_value(local_return)
