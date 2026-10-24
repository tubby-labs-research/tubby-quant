"""Pure S3 allocation and independent USD daily-equity risk checks.

No signal normalization, neutralization, execution, returns or PnL accounting lives
here. Dates are close decision dates for next-session-open execution. Prices passed
to risk must be current reference prices in USD per share, never future closes.
Callers own price freshness, dated group/liquidity selection and signed-engine
cash projections (including fees and restricted-proceeds release).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import hashlib
import json
import math
from numbers import Real
import re
from typing import TYPE_CHECKING, Mapping, Sequence

if TYPE_CHECKING:
    import numpy as np

_ROW_CHUNK = 64
_GROSS_TOLERANCE = 1e-12

__all__ = ["NormalizedTargets", "AllocationLeg", "FixedAllocationPolicy",
           "AllocationResult", "make_fixed_policy", "allocate_fixed",
           "RiskLimits", "RiskResult", "evaluate_risk"]


def _number(value):
    try:
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _label(value):
    return isinstance(value, str) and bool(value) and value.strip() == value


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _owned_weights(values, shape):
    """One owned numeric copy; validation scratch is bounded to 64 rows."""
    import numpy as np

    try:
        if not isinstance(values, np.ndarray):
            # Compatibility for small supplied Python sequences, without building
            # another nested container. Large callers should supply an ndarray.
            if any(not _number(x) for row in values for x in row):
                raise ValueError("target weights must be finite numbers")
            values = np.asarray(values, dtype=np.float64)
        if (values.shape != shape or values.dtype.kind != "f"
                or values.dtype.itemsize not in (4, 8)):
            raise ValueError("target weights require a matching float32/float64 matrix")
        # Canonical little-endian storage makes the binary digest portable.
        owned = np.array(values, dtype=values.dtype.newbyteorder("<"), order="C", copy=True)
    except (TypeError, OverflowError) as error:
        raise ValueError("invalid target matrix") from error
    scratch = np.empty((min(_ROW_CHUNK, shape[0]), shape[1]), dtype=np.float64)
    for start in range(0, shape[0], _ROW_CHUNK):
        block = owned[start:start + _ROW_CHUNK]
        if not np.isfinite(block).all():
            raise ValueError("target weights must be finite, including explicit zeros")
        np.abs(block, out=scratch[:len(block)])
        gross = scratch[:len(block)].sum(axis=1, dtype=np.float64)
        if np.any(gross > 1 + _GROSS_TOLERANCE):
            raise ValueError("supplied target gross exceeds one")
    owned.setflags(write=False)
    return owned


def _matrix_digest(weights, metadata):
    """Hash axes/lineage plus C-order rows directly, without matrix serialization."""
    if (weights.dtype.str not in ("<f4", "<f8") or not weights.flags.c_contiguous
            or weights.ndim != 2):
        raise ValueError("target storage dtype/layout changed")
    header = {**metadata, "digest_format": "portfolio_targets_binary_v1",
              "dtype": weights.dtype.str, "shape": weights.shape, "order": "C"}
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    for row in weights:
        digest.update(memoryview(row).cast("B"))
    return digest.hexdigest()


@dataclass(frozen=True, eq=False)
class NormalizedTargets:
    """Supplied dates-by-symbols weights, already normalized to gross <= 1.

    Label order must match exactly between legs; no sorting, union, missing-value
    filling or date shifting occurs. ``price_identity`` is a nonempty canonical
    source identifier (serialize a structured identity before passing it).
    ``digest`` binds the numeric matrix, axes and supplied lineage; the caller
    must have verified the source candidate result digest against its artifact.

    Storage is an owned, C-contiguous, little-endian float32/float64 copy, exposed
    read-only. Float32 inputs retain their dtype; Python sequences use float64.
    Gross sums accumulate in float64 with absolute tolerance 1e-12 for BOTH
    dtypes (not float32 epsilon). Representation overflow beyond that is rejected,
    never normalized or rounded upward/downward here. Tiny accepted excess is
    preserved. The digest is a cached construction-time seal; policy creation and
    allocation independently verify current storage against it once per leg,
    because ndarray owners can reset their write flag. Concurrent mutation during
    hashing/allocation is unsupported; callers must serialize writes.
    """

    dates: tuple[str, ...]
    symbols: tuple[str, ...]
    weights: np.ndarray = field(repr=False)
    candidate_result_digest: str
    attempt_id: str
    dataset_manifest_sha256: str
    price_identity: str
    _sealed_digest: str = field(init=False, repr=False)

    def __post_init__(self):
        dates, symbols = tuple(self.dates), tuple(self.symbols)
        try:
            valid_dates = all(isinstance(d, str) and date.fromisoformat(d).isoformat() == d for d in dates)
        except (TypeError, ValueError) as error:
            raise ValueError("invalid dates or target matrix") from error
        if not dates or not valid_dates or dates != tuple(sorted(set(dates))):
            raise ValueError("decision dates must be nonempty, ordered unique ISO dates")
        if not symbols or not all(_label(s) for s in symbols) or len(set(symbols)) != len(symbols):
            raise ValueError("symbols must be nonempty unique labels")
        if not (_sha(self.candidate_result_digest) and _sha(self.dataset_manifest_sha256)
                and _label(self.attempt_id) and _label(self.price_identity)):
            raise ValueError("missing or malformed target lineage")
        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "weights", _owned_weights(self.weights, (len(dates), len(symbols))))
        object.__setattr__(self, "_sealed_digest", self._current_digest())

    @property
    def digest(self):
        """Original sealed identity, O(1); use allocation to verify current bytes."""
        return self._sealed_digest

    def _current_digest(self):
        return _matrix_digest(self.weights, {
            "dates": self.dates, "symbols": self.symbols,
            "candidate_result_digest": self.candidate_result_digest,
            "attempt_id": self.attempt_id,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "price_identity": self.price_identity})

    def _verified_digest(self):
        current = self._current_digest()
        if current != self._sealed_digest:
            raise ValueError("target matrix or lineage differs from construction seal")
        return current


@dataclass(frozen=True)
class AllocationLeg:
    target_digest: str
    candidate_result_digest: str
    attempt_id: str
    coefficient: float

    def __post_init__(self):
        if not (_sha(self.target_digest) and _sha(self.candidate_result_digest) and _label(self.attempt_id)):
            raise ValueError("invalid allocation leg lineage")
        if not _number(self.coefficient) or not 0 <= self.coefficient <= 1:
            raise ValueError("fixed coefficients must be finite and nonnegative")
        object.__setattr__(self, "coefficient", float(self.coefficient))


@dataclass(frozen=True)
class FixedAllocationPolicy:
    """One account capital in USD; coefficients are capital fractions, not votes."""

    policy_id: str
    account_id: str
    capital: float
    legs: tuple[AllocationLeg, ...]
    version: str = "portfolio_fixed_v1"

    def __post_init__(self):
        legs = tuple(self.legs)
        if self.version != "portfolio_fixed_v1" or not _label(self.policy_id) or not _label(self.account_id):
            raise ValueError("invalid policy identity/version")
        if not _number(self.capital) or self.capital <= 0:
            raise ValueError("one positive finite account capital is required")
        if len(legs) not in (1, 2) or any(not isinstance(leg, AllocationLeg) for leg in legs):
            raise ValueError("supply one or two allocation legs")
        if len({leg.target_digest for leg in legs}) != len(legs):
            raise ValueError("duplicate target input")
        if math.fsum(leg.coefficient for leg in legs) > 1:
            raise ValueError("fixed coefficients exceed the one-account budget")
        object.__setattr__(self, "legs", legs)
        object.__setattr__(self, "capital", float(self.capital))

    @property
    def digest(self):
        return _digest(self._metadata())

    def _metadata(self):
        return {"policy_id": self.policy_id, "account_id": self.account_id,
                "capital": self.capital, "version": self.version,
                "legs": [{"target_digest": leg.target_digest,
                          "candidate_result_digest": leg.candidate_result_digest,
                          "attempt_id": leg.attempt_id, "coefficient": leg.coefficient}
                         for leg in self.legs]}


@dataclass(frozen=True, eq=False)
class AllocationResult:
    """Read-only float64 output; inputs are retained by reference, never copied.

    Direct construction copies supplied storage. allocate_fixed privately transfers
    its freshly allocated output buffer to avoid a redundant full-matrix copy.
    As with inputs, do not reset array write flags or mutate storage concurrently.
    """
    policy: FixedAllocationPolicy
    inputs: tuple[NormalizedTargets, ...]
    dates: tuple[str, ...]
    symbols: tuple[str, ...]
    weights: np.ndarray = field(repr=False)

    def __post_init__(self):
        object.__setattr__(self, "dates", tuple(self.dates))
        object.__setattr__(self, "symbols", tuple(self.symbols))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "weights", _owned_weights(self.weights, (len(self.dates), len(self.symbols))))

    @classmethod
    def _from_owned(cls, policy, inputs, weights):
        result = object.__new__(cls)
        for key, value in (("policy", policy), ("inputs", inputs),
                           ("dates", inputs[0].dates), ("symbols", inputs[0].symbols),
                           ("weights", weights)):
            object.__setattr__(result, key, value)
        weights.setflags(write=False)
        return result

    @property
    def target_notionals(self):
        """Allocate a new float64 USD-notional ndarray ONLY when requested."""
        import numpy as np
        notionals = np.multiply(self.weights, self.policy.capital, dtype=np.float64)
        notionals.setflags(write=False)
        return notionals

    def to_metadata(self):
        """Detached JSON-compatible lineage suitable for target-bundle metadata."""
        return {"weight_policy": self.policy.version, "allocation_policy": self.policy._metadata(),
                "allocation_policy_digest": self.policy.digest,
                "input_target_digest_format": "portfolio_targets_binary_v1",
                "input_target_digests": [leg.target_digest for leg in self.policy.legs],
                "allocation_dtype": "float64", "input_gross_tolerance": _GROSS_TOLERANCE,
                "candidate_result_digests": [x.candidate_result_digest for x in self.inputs],
                "attempt_ids": [x.attempt_id for x in self.inputs],
                "dataset_manifest_sha256": self.inputs[0].dataset_manifest_sha256,
                "price_identity": self.inputs[0].price_identity,
                "assumptions": ["supplied normalized targets; no renormalization or neutralization",
                                "float64 gross sums; absolute input tolerance 1e-12; no dtype epsilon allowance",
                                "float64 allocation; inputs retain float32/float64 dtype",
                                "decision session close; execution next session open",
                                "target notionals are not fills or returns"]}


def make_fixed_policy(inputs: Sequence[NormalizedTargets], coefficients: Sequence[float], *,
                      account_id: str, capital: float, policy_id: str) -> FixedAllocationPolicy:
    """Pin exact supplied matrices and lineage before allocation."""
    inputs, coefficients = tuple(inputs), tuple(coefficients)
    if len(inputs) != len(coefficients) or any(not isinstance(x, NormalizedTargets) for x in inputs):
        raise ValueError("one coefficient per normalized target input is required")
    return FixedAllocationPolicy(policy_id, account_id, capital, tuple(
        AllocationLeg(x._verified_digest(), x.candidate_result_digest, x.attempt_id, c)
        for x, c in zip(inputs, coefficients)))


def allocate_fixed(policy: FixedAllocationPolicy, inputs: Sequence[NormalizedTargets]) -> AllocationResult:
    """Apply fixed weights in float64, with one output and <=64-row scratch.

    Each leg is binary-hashed exactly once here, even if its coefficient is zero.
    Metadata uses those verified policy pins, not another scan of the matrices.
    Output differs from the ideal sum only by float64 arithmetic; there is no
    gross normalization, clipping, neutralization or dtype-specific gross uplift.
    """
    import numpy as np
    inputs = tuple(inputs)
    if not isinstance(policy, FixedAllocationPolicy) or len(inputs) != len(policy.legs):
        raise ValueError("inputs do not match the fixed policy")
    for source, leg in zip(inputs, policy.legs):
        if not isinstance(source, NormalizedTargets) or (source._verified_digest(), source.candidate_result_digest, source.attempt_id) != (
                leg.target_digest, leg.candidate_result_digest, leg.attempt_id):
            raise ValueError("target matrix or lineage differs from pinned policy")
        if (source.dates, source.symbols, source.dataset_manifest_sha256, source.price_identity) != (
                inputs[0].dates, inputs[0].symbols, inputs[0].dataset_manifest_sha256, inputs[0].price_identity):
            raise ValueError("target axes or price/dataset lineage mismatch")
    shape = inputs[0].weights.shape
    weights = np.empty(shape, dtype=np.float64, order="C")
    scratch = np.empty((min(_ROW_CHUNK, shape[0]), shape[1]), dtype=np.float64) if len(inputs) == 2 else None
    for start in range(0, shape[0], _ROW_CHUNK):
        stop = min(start + _ROW_CHUNK, shape[0])
        block = weights[start:stop]
        np.multiply(inputs[0].weights[start:stop], policy.legs[0].coefficient,
                    out=block, dtype=np.float64)
        if scratch is not None:
            temp = scratch[:stop - start]
            np.multiply(inputs[1].weights[start:stop], policy.legs[1].coefficient,
                        out=temp, dtype=np.float64)
            np.add(block, temp, out=block)
    return AllocationResult._from_owned(policy, inputs, weights)


@dataclass(frozen=True)
class RiskLimits:
    """None disables a limit; zero enables it. All ratios use positive USD equity.

    Gross=sum(abs(weight)), net=abs(sum(weight)), name=max(abs(weight)).
    Group gross sums absolute member weights; group net is absolute signed sum.
    Turnover is FULL traded reference notional/equity (no 1/2 convention).
    Liquidity is absolute traded SHARES / supplied daily reference volume SHARES.
    One group taxonomy and uniform per-name/per-group bounds are supported.
    No-loan funding and positive equity are unconditional checks.
    """

    gross: float | None = None
    net: float | None = None
    name: float | None = None
    group_gross: float | None = None
    group_net: float | None = None
    turnover: float | None = None
    liquidity: float | None = None

    def __post_init__(self):
        for key in ("gross", "net", "name", "group_gross", "group_net", "turnover", "liquidity"):
            value = getattr(self, key)
            if value is not None and (not _number(value) or value < 0):
                raise ValueError(f"{key} limit must be finite and nonnegative, or None")


@dataclass(frozen=True)
class RiskResult:
    status: str
    phase: str
    violations: tuple[str, ...]
    missing_inputs: tuple[str, ...]
    metrics: dict
    assessed_holdings: dict[str, float | None]
    residual_weights: dict[str, float | None]
    limits: RiskLimits
    version: str = "portfolio_risk_v1"


def evaluate_risk(*, phase: str, holdings: Mapping[str, float], trades: Mapping[str, float],
                  target_weights: Mapping[str, float], equity: float,
                  reference_prices: Mapping[str, float], limits: RiskLimits,
                  groups: Mapping[str, str] | None = None,
                  liquidity: Mapping[str, float] | None = None,
                  free_cash: float | None = None,
                  projected_free_cash: float | None = None,
                  traded_quantities: Mapping[str, float] | None = None) -> RiskResult:
    """Independently check ALL enabled limits; never modify targets or trades.

    ``pre_order``: holdings are actual before orders; trades are signed proposed
    shares. Assess holdings+trades AND desired target weights. Require free_cash
    always, plus engine-supplied projected_free_cash for any nonzero proposal. The projection
    must reflect the engine's proposed execution order, fees and restricted cash.
    ``post_fill``: holdings are actual AFTER fills; trades are signed executed
    shares, used ONLY for turnover/liquidity, never added to holdings again. Require
    actual free_cash. Unfilled shares remain in holdings even for a zero target.

    Mappings are sparse: an omitted holding/trade/target means explicitly zero,
    but the union of their symbols requires finite positive current prices.
    Groups are required for every nonzero assessed OR desired exposure when a
    group limit is enabled; unknown labels are missing, never a common bucket.
    Liquidity is required for each nonzero trade when enabled. Missing/nonfinite
    enabled inputs yield missing_inputs, taking precedence over blocked; all
    computed violations are retained. Nonpositive equity is blocked, ratios unavailable.
    Residual weights = desired minus assessed; these are not an order instruction.
    Cash figures are checked, never derived here: this module is not a ledger.
    ``traded_quantities`` optionally supplies absolute gross shares per symbol
    (e.g. both legs of a reversal). It must cover and be at least abs(trades);
    net share changes alone otherwise understate turnover and participation.
    """
    if phase not in ("pre_order", "post_fill") or not isinstance(limits, RiskLimits):
        raise ValueError("use pre_order/post_fill and explicit RiskLimits")
    missing, violations, metrics = set(), set(), {"equity_usd": float(equity) if _number(equity) else None}
    assessed, residual = {}, {}

    def finish():
        # Missing arithmetic remains JSON-safe and deterministic for run records.
        def finite_tree(value):
            if isinstance(value, dict):
                return {k: finite_tree(v) for k, v in value.items()}
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value
        return RiskResult("missing_inputs" if missing else "blocked" if violations else "pass",
                          phase, tuple(sorted(violations)), tuple(sorted(missing)),
                          finite_tree(metrics), finite_tree(assessed), finite_tree(residual), limits)

    def read_mapping(name, values):
        if not isinstance(values, Mapping):
            missing.add(name)
            return {}
        result = {}
        for key, value in values.items():
            if not _label(key) or not _number(value):
                missing.add(f"{name}:{key}")
            else:
                result[key] = float(value)
        return result

    held = read_mapping("holdings", holdings)
    orders = read_mapping("trades", trades)
    gross_trades = ({s: abs(v) for s, v in orders.items()} if traded_quantities is None
                    else read_mapping("traded_quantities", traded_quantities))
    for symbol in set(orders) | set(gross_trades):
        if gross_trades.get(symbol, 0.0) < abs(orders.get(symbol, 0.0)):
            missing.add("traded_quantities:" + symbol)
    desired = read_mapping("target_weights", target_weights)
    prices = read_mapping("reference_prices", reference_prices)
    core_missing = bool(missing)
    symbols = sorted(set(held) | set(orders) | set(desired) | set(gross_trades))
    assessed.update({s: held.get(s, 0.0) + (orders.get(s, 0.0) if phase == "pre_order" else 0.0)
                     for s in symbols})

    def valid_group(symbol):
        group = groups.get(symbol) if isinstance(groups, Mapping) else None
        if not _label(group) or group.lower() in {"unknown", "__unknown__", "none", "null", "nan", "n/a"}:
            missing.add("groups:" + symbol)
            return None
        return group

    if limits.group_gross is not None or limits.group_net is not None:
        for symbol in symbols:
            if assessed[symbol] or desired.get(symbol, 0.0):
                valid_group(symbol)
    # These checks do not depend on equity or marks; do not waive them if another
    # input is missing or the account is already insolvent.
    if limits.liquidity is not None:
        participation = {}
        for symbol in symbols:
            quantity = gross_trades.get(symbol, 0.0)
            if quantity == 0:
                continue
            volume = liquidity.get(symbol) if isinstance(liquidity, Mapping) else None
            if not _number(volume) or volume < 0:
                missing.add("liquidity:" + symbol)
                continue
            participation[symbol] = quantity / volume if volume else None
            if volume and not math.isfinite(participation[symbol]):
                missing.add("finite_liquidity_arithmetic:" + symbol)
            if not volume or participation[symbol] > limits.liquidity:
                violations.add("trades:liquidity:" + symbol)
        metrics["liquidity_participation"] = participation
    for key, value in (("free_cash", free_cash),):
        if not _number(value):
            missing.add(key)
        else:
            metrics[key + "_usd"] = float(value)
            metrics["unfunded_cash_shortfall_usd"] = max(-float(value), 0.0)
            if value < 0:
                violations.add("funding:negative_free_cash")
    if phase == "pre_order" and any(gross_trades.values()):
        if not _number(projected_free_cash):
            missing.add("projected_free_cash")
        else:
            metrics["projected_free_cash_usd"] = float(projected_free_cash)
            if projected_free_cash < 0:
                violations.add("funding:projected_cash_loan")
    if not _number(equity):
        missing.add("equity")
        return finish()
    if equity <= 0:
        violations.add("equity:nonpositive")
    for symbol in symbols:
        if symbol not in prices or prices[symbol] <= 0:
            missing.add("reference_prices:" + symbol)
            core_missing = True
    if core_missing:
        return finish()
    notionals = {s: assessed[s] * prices[s] for s in symbols}
    traded = {s: gross_trades.get(s, 0.0) * prices[s] for s in symbols}
    held_notionals = {s: held.get(s, 0.0) * prices[s] for s in symbols}
    metrics["holdings_notionals_usd"] = held_notionals
    if not all(math.isfinite(x) for x in (*assessed.values(), *notionals.values(),
                                        *traded.values(), *held_notionals.values())):
        missing.add("finite_exposure_arithmetic")
        return finish()
    metrics["assessed_notionals_usd"] = notionals
    if equity <= 0:
        return finish()
    actual = {s: notionals[s] / equity for s in symbols}
    desired = {s: desired.get(s, 0.0) for s in symbols}
    residual.update({s: desired[s] - actual[s] for s in symbols})
    if not all(math.isfinite(x) for x in (*actual.values(), *residual.values())):
        missing.add("finite_weight_arithmetic")
        return finish()

    def check(code, value, bound):
        if bound is not None and value > bound:
            violations.add(code)

    def total(values):
        try:
            value = math.fsum(values)
        except OverflowError:
            value = math.inf
        if not math.isfinite(value):
            missing.add("finite_exposure_arithmetic")
        return value

    for kind, weights in (("assessed", actual), ("target", desired)):
        exposures = {"gross": total(abs(x) for x in weights.values()),
                     "net": total(weights.values()),
                     "name": max((abs(x) for x in weights.values()), default=0.0)}
        metrics[kind] = {**exposures, "weights": dict(weights)}
        for key in ("gross", "net", "name"):
            check(kind + ":" + key, abs(exposures[key]), getattr(limits, key))
        if limits.group_gross is not None or limits.group_net is not None:
            buckets = {}
            for symbol, weight in weights.items():
                if weight == 0:
                    continue
                group = valid_group(symbol)
                if group is None:
                    continue
                buckets.setdefault(group, []).append(weight)
            metrics[kind]["groups"] = {}
            for group, values in sorted(buckets.items()):
                gross, net = total(abs(x) for x in values), total(values)
                metrics[kind]["groups"][group] = {"gross": gross, "net": net}
                check(f"{kind}:group_gross:{group}", gross, limits.group_gross)
                check(f"{kind}:group_net:{group}", abs(net), limits.group_net)
    turnover = total(traded.values()) / equity
    if not math.isfinite(turnover):
        missing.add("finite_exposure_arithmetic")
    metrics["turnover"] = turnover
    check("trades:turnover", turnover, limits.turnover)
    return finish()
