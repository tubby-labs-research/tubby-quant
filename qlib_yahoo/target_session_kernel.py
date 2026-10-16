"""Experimental unwired whole-session signed targets and order classification.

Caller owns native calendar/signal/_weights/equity validation and callbacks.
Inputs follow sorted native target_weights symbols, including held-only zeros.
Quote rows must have reviewed primitive bindings; no speculative custom queries.
None means resume the native target loop on the SAME diagnostic record, before
target/reason/order mutations. Restore any speculative quote-cache changes first.
Native target errors precede holdings errors: both phases remain ordered here.
This intentionally falls back for exceptional/unsafe values instead of replacing
native exception types, warning policies or arbitrary-sized Python integers.
Caller must include packing, diagnostics, Order creation and compilation in timing.
"""
from dataclasses import dataclass
import math

MAX_QUANTITY = 2 ** 52
ACCEPTED, UNAVAILABLE, INVALID_OPEN = range(3)
SYMBOL, AMOUNT, DIRECTION = range(3)


@dataclass(frozen=True)
class TargetSessionResult:
    targets: object
    active: object
    reasons: object
    reductions: object
    increases: object


def _session(weights, current, opens, factors, tradable, equity, targets,
             active, reasons, reductions, increases):
    n = len(weights)
    for i in range(n):
        if not tradable[i]:
            reasons[i] = UNAVAILABLE
            continue
        price = float(opens[i])
        if not math.isfinite(price) or price <= 0:
            reasons[i] = INVALID_OPEN
            continue
        factor = float(factors[i])
        if not math.isfinite(factor) or factor != 1:
            return -1, -1
        weight = float(weights[i])
        # Preserve primitive expression evaluation order; never reassociate.
        product = abs(weight) * equity
        quotient = product / price
        if not math.isfinite(product) or not math.isfinite(quotient) or quotient > MAX_QUANTITY:
            return -1, -1
        target = (1 if weight >= 0 else -1) * math.floor(quotient)
        targets[i] = target
        active[i] = True
    # Native does not inspect current holdings until ALL targets are sized.
    for i in range(n):
        if not active[i]:
            continue
        held = float(current[i])
        if not math.isfinite(held) or abs(held) > MAX_QUANTITY or held != math.floor(held):
            return -1, -1
        if abs(targets[i] - int(held)) > MAX_QUANTITY:
            return -1, -1
    nr, ni = 0, 0
    # Native reductions.sort((direction == BUY, stock_id)). Input symbol indices
    # already follow sorted symbols, so stable SELL/BUY passes match that sort.
    for direction in range(2):
        for i in range(n):
            if not active[i]:
                continue
            held = int(current[i])
            target = targets[i]
            if held == target:
                continue
            reversal = held != 0 and target != 0 and (held > 0) != (target > 0)
            reducing = abs(target) < abs(held)
            if reversal or reducing:
                amount = abs(held) if reversal else abs(target - held)
                side = (0 if held > 0 else 1) if reversal else (1 if target > held else 0)
                if side == direction:
                    reductions[nr, SYMBOL] = i
                    reductions[nr, AMOUNT] = amount
                    reductions[nr, DIRECTION] = side
                    nr += 1
    for i in range(n):
        if not active[i]:
            continue
        held = int(current[i])
        target = targets[i]
        if held == target:
            continue
        reversal = held != 0 and target != 0 and (held > 0) != (target > 0)
        if reversal or abs(target) >= abs(held):
            increases[ni, SYMBOL] = i
            increases[ni, AMOUNT] = abs(target) if reversal else abs(target - held)
            increases[ni, DIRECTION] = (1 if target > 0 else 0) if reversal else (1 if target > held else 0)
            ni += 1
    return nr, ni


_compiled = None


def compiled_session(*, cache=False):
    global _compiled
    if _compiled is None:
        from numba import njit
        _compiled = njit(cache=cache, fastmath=False, parallel=False)(_session)
    return _compiled


def run_target_session(weights, current, opens, factors, tradable, *, equity, compiled=False):
    """Return scratch native-order descriptors, or None for original-loop fallback.

    Arrays: float64 weights/current/open/factor; bool tradable. Original arrays
    are read-only. Symbol-index descriptors and target quantities are int64;
    inactive target entries are zero and must not appear in the native dict.
    This kernel does not validate native gross exposure or symbol metadata.
    """
    import numpy as np
    # Native divides by NumPy quote scalars. Preserve user-configured floating
    # warning/error policies through native fallback, including underflow traps.
    # Other exceptions (overflow, invalid, zero division) cannot succeed under
    # the finite-positive input/intermediate guards. Finite underflow can.
    if np.geterr()['under'] != 'ignore':
        return None
    arrays = (weights, current, opens, factors)
    if any(type(a) is not np.ndarray or a.dtype != np.float64 or a.ndim != 1 for a in arrays):
        return None
    if type(tradable) is not np.ndarray or tradable.dtype != np.bool_ or tradable.ndim != 1:
        return None
    n = len(weights)
    if any(len(a) != n for a in (current, opens, factors, tradable)):
        return None
    if type(equity) not in (float, int) or not math.isfinite(equity) or equity <= 0:
        return None
    if not np.all(np.isfinite(weights)):
        return None
    targets = np.zeros(n, dtype=np.int64)
    active = np.zeros(n, dtype=np.bool_)
    reasons = np.zeros(n, dtype=np.int64)
    reductions = np.empty((n, 3), dtype=np.int64)
    increases = np.empty((n, 3), dtype=np.int64)
    try:
        core = compiled_session() if compiled else _session
        nr, ni = core(*arrays, tradable, float(equity), targets, active, reasons, reductions, increases)
    except (ArithmeticError, ValueError, TypeError, RuntimeError):
        return None
    if nr < 0:
        return None
    return TargetSessionResult(targets, active, reasons, reductions[:nr], increases[:ni])
