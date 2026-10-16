"""EXPERIMENTAL, unwired ordered-session ledger for tubby.3/d10a5a8.

Only exact native daily-open, factor-one, integer-share, no-settlement accounts
with constructor/method/callback identities verified by the caller are eligible.
This module cannot verify those identities. Caller must also verify symbol/order
metadata, active session, quote provenance and absence of callbacks. Arrays are
float64 holdings/quotes/requests, int64 symbol/direction, bool existence/tradable.
No inputs or native objects are mutated. None means replay the ENTIRE session
natively; successful scratch records must be replayed in order, including native
dictionary deletion/insertion, before native bar-close sums/counts/marks.

Compilation is lazy, serial float64, fastmath=False; no scalar-fill JIT hook.
Returned price is NaN on a rejected check_order, as in the native exchange.
Status distinguishes reason attributes; FACTOR_SET says whether factor is set.
Skipped orders leave existing native reason/partial attributes alone except the
skip reason set by check_order; accepted checks reset both before calculation.
"""

from dataclasses import dataclass
import math

DEAL, VALUE, COST, PRICE, CASH, BASIS, AMOUNT, EXISTS = range(8)
ACCUM_COST, ACCUM_TURNOVER, ACCUM_RETURN, TRANSACTION_COST, STATUS, FACTOR_SET = range(8, 14)
FILLED, PARTIAL, ZERO, NO_CASH, INVALID_OPEN, NOT_TRADABLE = range(6)
MAX_QUANTITY = 2 ** 52


@dataclass(frozen=True)
class SessionResult:
    amounts: object
    marks: object
    bases: object
    exists: object
    records: object


def _ulp(value):
    # Exact binary64 ULP including zero/subnormal; frexp is supported by Numba.
    magnitude = abs(value)
    if magnitude < 2.2250738585072014e-308:
        return 5e-324
    _, exponent = math.frexp(magnitude)
    return math.ldexp(1.0, exponent - 53)


def _effect(old, basis, cash, delta, price, cost):
    if delta > 0:
        covered = min(delta, max(-old, 0.0))
        released = basis * (covered / -old) if old < 0 else 0.0
        new_basis = basis - released
        free = cash + released - delta * price - cost
    else:
        closed_long = min(-delta, max(old, 0.0))
        new_short = -delta - closed_long
        new_basis = basis + new_short * price
        free = cash + closed_long * price - cost
    return free, new_basis, old + delta


def _feasible(quantity, sign, old, basis, cash, price, rate):
    value = quantity * price
    free, post_basis, amount = _effect(old, basis, cash, sign * quantity, price, value * rate)
    if not (math.isfinite(free) and math.isfinite(post_basis) and math.isfinite(amount)):
        return -1
    return 1 if free >= 0.0 else 0


def _session(amounts, marks, bases, exists, symbols, requested, directions,
             opens, factors, tradable, cash, rate, transaction_cost,
             accum_cost, accum_turnover, accum_return, records):
    for i in range(len(symbols)):
        symbol = symbols[i]
        ask = int(requested[i])
        sign = 1 if directions[i] == 1 else -1
        price = float(opens[i])
        old = float(amounts[symbol])
        basis = float(bases[symbol])
        mark = float(marks[symbol]) if exists[symbol] else 0.0
        quantity, value, cost, factor_set = 0, 0.0, 0.0, 0
        status = FILLED
        if not tradable[i]:
            status = INVALID_OPEN if not math.isfinite(price) or price <= 0 else NOT_TRADABLE
            price = math.nan
        else:
            if not math.isfinite(price) or price <= 0 or factors[i] != 1.0:
                return False
            factor_set = 1
            if ask == 0:
                status = ZERO
            else:
                bound = min(ask, max(0, -sign * int(old)))
                for segment in range(2):
                    low = bound + 1 if segment == 0 else 1
                    high = ask if segment == 0 else bound
                    if low > high:
                        continue
                    feasible = _feasible(high, sign, old, basis, cash, price, rate)
                    if feasible < 0:
                        return False
                    if feasible == 1:
                        quantity = high
                        break
                    feasible = _feasible(low, sign, old, basis, cash, price, rate)
                    if feasible < 0:
                        return False
                    if feasible == 0:
                        continue
                    while low < high:
                        mid = (low + high + 1) // 2
                        feasible = _feasible(mid, sign, old, basis, cash, price, rate)
                        if feasible < 0:
                            return False
                        if feasible == 1:
                            low = mid
                        else:
                            high = mid - 1
                    quantity = low
                    break
                if quantity == 0:
                    status = NO_CASH
                else:
                    status = PARTIAL if quantity < ask else FILLED
                    value = quantity * price
                    cost = value * rate
                    if not math.isfinite(value) or not math.isfinite(cost):
                        return False
                    before = cash + basis + old * mark
                    filled = value / price
                    if not math.isfinite(before) or not math.isfinite(filled):
                        return False
                    nearest = round(filled)
                    if abs(filled - nearest) > 8 * _ulp(filled) or nearest > ask:
                        return False
                    if value != 0:
                        if nearest == 0:
                            return False
                        free, new_basis, amount = _effect(old, basis, cash, nearest * sign, price, cost)
                        if not (math.isfinite(free) and math.isfinite(new_basis) and math.isfinite(amount)
                                and math.isfinite(transaction_cost + cost)):
                            return False
                        tolerance = min(1e-8, 8 * max(_ulp(cash), _ulp(value), _ulp(cost)))
                        if free < -tolerance or abs(amount) > MAX_QUANTITY:
                            return False
                        cash = 0.0 if free < 0 else free
                        amounts[symbol] = amount
                        exists[symbol] = amount != 0
                        bases[symbol] = new_basis if amount < 0 else 0.0
                        marks[symbol] = price if amount != 0 else 0.0
                        transaction_cost += cost
                    elif cost != 0:
                        return False
                    after = cash + float(bases[symbol]) + float(amounts[symbol]) * float(marks[symbol])
                    if not math.isfinite(after):
                        return False
                    accum_cost += cost
                    accum_turnover += value
                    accum_return += after - before + cost
                    if not (math.isfinite(accum_cost) and math.isfinite(accum_turnover)
                            and math.isfinite(accum_return)):
                        return False
        records[i, DEAL] = float(quantity)
        records[i, VALUE] = value
        records[i, COST] = cost
        records[i, PRICE] = price
        records[i, CASH] = cash
        records[i, BASIS] = bases[symbol]
        records[i, AMOUNT] = amounts[symbol]
        records[i, EXISTS] = 1.0 if exists[symbol] else 0.0
        records[i, ACCUM_COST] = accum_cost
        records[i, ACCUM_TURNOVER] = accum_turnover
        records[i, ACCUM_RETURN] = accum_return
        records[i, TRANSACTION_COST] = transaction_cost
        records[i, STATUS] = float(status)
        records[i, FACTOR_SET] = float(factor_set)
    return True


_compiled = None


def compiled_session(*, cache=False):
    """Optional lazily compiled whole session, including helpers; no installation."""
    global _compiled
    if _compiled is None:
        from numba import njit
        from numba.extending import register_jitable
        for helper in (_ulp, _effect, _feasible):
            register_jitable(helper)
        _compiled = njit(cache=cache, fastmath=False, parallel=False)(_session)
    return _compiled


def run_ordered_session(amounts, marks, bases, exists, symbols, requested,
                        directions, opens, factors, tradable, *, cash, rate,
                        transaction_cost=0.0, accum_cost=0.0,
                        accum_turnover=0.0, accum_return=0.0, compiled=False):
    """Return immutable-by-convention scratch result, or None for native fallback.

    All guards/conversion/compilation are within this operation's timing boundary.
    The caller must never commit any scratch result until the whole batch succeeds.
    """
    import numpy as np
    holdings = (amounts, marks, bases)
    floats = holdings + (requested, opens, factors)
    if any(type(a) is not np.ndarray or a.dtype != np.float64 or a.ndim != 1 for a in floats):
        return None
    if any(type(a) is not np.ndarray or a.dtype != np.int64 or a.ndim != 1 for a in (symbols, directions)):
        return None
    if any(type(a) is not np.ndarray or a.dtype != np.bool_ or a.ndim != 1 for a in (exists, tradable)):
        return None
    n, m = len(amounts), len(symbols)
    if any(len(a) != n for a in (marks, bases, exists)) or any(len(a) != m for a in (requested, directions, opens, factors, tradable)):
        return None
    scalars = (cash, rate, transaction_cost, accum_cost, accum_turnover, accum_return)
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in scalars) or rate < 0 or transaction_cost < 0:
        return None
    if (np.any(symbols < 0) or np.any(symbols >= n) or np.any((directions != 0) & (directions != 1))
            or not np.all(np.isfinite(requested)) or np.any(requested < 0)
            or np.any(requested > MAX_QUANTITY) or np.any(requested != np.floor(requested))):
        return None
    if (any(not np.all(np.isfinite(a)) for a in holdings) or np.any(np.abs(amounts) > MAX_QUANTITY)
            or np.any(amounts != np.floor(amounts)) or np.any(bases < 0)
            or np.any(exists & (marks <= 0)) or np.any(~exists & ((amounts != 0) | (bases != 0)))
            or np.any((amounts >= 0) & (bases != 0))):
        return None
    scratch = (amounts.copy(), marks.copy(), bases.copy(), exists.copy())
    records = np.empty((m, 14), dtype=np.float64)
    try:
        core = compiled_session() if compiled else _session
        success = core(*scratch, symbols, requested, directions, opens, factors,
                       tradable, *(float(v) for v in scalars), records)
    except (ArithmeticError, ValueError, TypeError, RuntimeError):
        return None
    return SessionResult(*scratch, records) if success else None
