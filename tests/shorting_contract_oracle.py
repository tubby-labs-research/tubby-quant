"""Small independent signed-position accounting oracle for contract fixtures only.

This is an unfunded research account, not a broker-feasible account or cash loan.
It does not model collateral calls, haircuts, liquidation or cash-loan interest.
"""

from decimal import Decimal


def number(value):
    """Convert a finite numeric input without binary-float arithmetic afterward."""
    try:
        result = Decimal(str(value))
    except (ValueError, TypeError, ArithmeticError) as exc:
        raise ValueError(f"invalid number: {value!r}") from exc
    if not result.is_finite():
        raise ValueError(f"nonfinite number: {value!r}")
    return result


def normalize_weights(alphas, gross=1):
    gross = number(gross)
    if not 0 <= gross <= 1:
        raise ValueError("gross must be between zero and one")
    scores = {symbol: number(alpha) for symbol, alpha in alphas.items()}
    denominator = sum((abs(alpha) for alpha in scores.values()), Decimal(0))
    if denominator == 0:
        return {symbol: Decimal(0) for symbol in scores}
    return {symbol: gross * alpha / denominator for symbol, alpha in scores.items()}


class UnaffordableFill(ValueError):
    """A manual order would leave negative free cash; no partial fill was made."""


class SignedLedger:
    def __init__(self, initial_free_cash):
        self.free_cash = number(initial_free_cash)
        if self.free_cash < 0:
            raise ValueError("initial free cash must be nonnegative")
        self.positions = {}
        self.restricted = {}
        self.prices = {}
        self.transaction_cost = Decimal(0)
        self.financing_cost = Decimal(0)

    def fill(self, symbol, delta, price, fee_rate="0.0005"):
        delta, price, fee_rate = number(delta), number(price), number(fee_rate)
        if price <= 0 or fee_rate < 0:
            raise ValueError("price must be positive and fee rate nonnegative")
        if delta == 0:
            return
        old = self.positions.get(symbol, Decimal(0))
        closing = min(abs(old), abs(delta)) if old * delta < 0 else Decimal(0)
        opening = abs(delta) - closing
        free_change = Decimal(0)
        restricted_change = Decimal(0)
        if closing:
            if old < 0:  # cover: release original pooled proceeds, pay current price
                release = self.restricted[symbol] * closing / abs(old)
                restricted_change -= release
                free_change += release - closing * price
            else:  # sell held long
                free_change += closing * price
        if opening:
            if delta < 0:  # short sale proceeds are restricted, never spendable
                restricted_change += opening * price
            else:
                free_change -= opening * price
        fee = abs(delta) * price * fee_rate
        prospective_cash = self.free_cash + free_change - fee
        if prospective_cash < 0:
            raise UnaffordableFill("manual fill would spend unavailable free cash")
        self.free_cash = prospective_cash
        new = old + delta
        if new:
            self.positions[symbol] = new
            self.prices[symbol] = price
        else:
            self.positions.pop(symbol, None)
            self.prices.pop(symbol, None)
        restricted = self.restricted.get(symbol, Decimal(0)) + restricted_change
        if restricted:
            self.restricted[symbol] = restricted
        else:
            self.restricted.pop(symbol, None)
        self.transaction_cost += fee

    def mark(self, prices):
        updates = {symbol: number(price) for symbol, price in prices.items()}
        if any(price <= 0 for price in updates.values()):
            raise ValueError("mark prices must be positive")
        missing = self.positions.keys() - (self.prices.keys() | updates.keys())
        if missing:
            raise ValueError(f"missing marks: {sorted(missing)}")
        self.prices.update(updates)

    def accrue_borrow(self, elapsed_calendar_days, annual_rate="0.004"):
        days, rate = number(elapsed_calendar_days), number(annual_rate)
        if days < 0 or days != days.to_integral_value() or rate < 0:
            raise ValueError("calendar days must be a nonnegative integer and annual rate nonnegative")
        # Caller invokes this before next-open fills, with prior-close marks.
        short_value = sum(
            (-qty * self.prices[symbol] for symbol, qty in self.positions.items() if qty < 0),
            Decimal(0),
        )
        fee = short_value * rate * days / Decimal(365)
        self.free_cash -= fee  # exogenous charge may create an unfunded deficit
        self.financing_cost += fee
        return fee

    def snapshot(self):
        restricted_total = sum(self.restricted.values(), Decimal(0))
        marked_positions = sum(
            (qty * self.prices[symbol] for symbol, qty in self.positions.items()),
            Decimal(0),
        )
        return {
            "free_cash": self.free_cash,
            "restricted_total": restricted_total,
            "equity": self.free_cash + restricted_total + marked_positions,
            "positions": dict(self.positions),
            "financing_cost": self.financing_cost,
            "transaction_cost": self.transaction_cost,
            "funding_shortfall": max(-self.free_cash, Decimal(0)),
        }
