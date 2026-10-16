"""Project policy and serialization for the pinned signed Qlib account."""

from __future__ import annotations

import json
import math

from qlib_yahoo.dataset import DatasetError

# Cross-sectional transforms applied to each session's scores before weights are scaled to the gross.
SIGNAL_TRANSFORMS = ("raw", "rank", "zscore")
WEIGHT_FORMULAS = {
    "raw": "g * raw_score / sum(abs(raw_scores)); zeros give zero targets",
    "rank": "g * centered_rank / sum(abs(centered_ranks)); average ranks for ties, centered on zero",
    "zscore": "g * zscore / sum(abs(zscores)); equal to the demeaned score, since the scale cancels",
}


def transform_section(values, transform):
    """One session's cross-section: raw, centered average ranks, or z-scores (population std).

    Rank and z-score weights sum to zero before gross scaling (dollar-neutral targets). A cross-section
    with fewer than two names, or with no dispersion, has no relative information and gets zero weights.
    """
    import numpy as np
    import pandas as pd

    values = np.asarray(values, dtype=float)
    if transform == "raw":
        return values
    if transform not in SIGNAL_TRANSFORMS:
        raise DatasetError(f"signal transform must be one of {', '.join(SIGNAL_TRANSFORMS)}")
    count = len(values)
    if count < 2:
        return np.zeros(count)
    if transform == "rank":
        ranks = pd.Series(values).rank(method="average").to_numpy(dtype=float)
        return (ranks - (count + 1) / 2) / count
    spread = float(np.std(values))
    if not spread > 0:
        return np.zeros(count)
    return (values - float(np.mean(values))) / spread


def normalize_scores(feature, expression, symbols, signal_days, gross_exposure, *, spans=None, transform="raw"):
    """Normalize scores into signed weights for each signal session.

    Without ``spans`` every fixed eligible symbol needs a finite score each
    session. With dated universe spans, only member observations are used and a
    member without a finite score that session gets no weight (target zero).
    ``transform`` (raw, rank or zscore) is applied across each session's scored
    names first; the returned scores are always the raw alpha values.
    """
    import numpy as np
    import pandas as pd

    if expression not in feature:
        raise DatasetError("alpha expression missing from Qlib features")
    raw = feature[expression]
    if not raw.index.is_unique:
        raise DatasetError("duplicate signed alpha score labels")
    raw = raw.reorder_levels(["datetime", "instrument"])
    if spans is None:
        expected = pd.MultiIndex.from_product(
            [signal_days, symbols], names=["datetime", "instrument"]
        )
        raw = raw.reindex(expected)
        values = raw.to_numpy(dtype=float)
        bad = ~np.isfinite(values)
        if bad.any():
            day, symbol = expected[int(np.flatnonzero(bad)[0])]
            raise DatasetError(f"signed alpha missing or nonfinite for {symbol} on {day}")
    else:
        from qlib_yahoo.universe import mask_to_members

        raw = raw.sort_index()
        raw = mask_to_members(raw.loc[raw.index.get_level_values("datetime").isin(signal_days)], spans)
        raw = raw[np.isfinite(raw.to_numpy(dtype=float))]
        missing = pd.DatetimeIndex(signal_days).difference(raw.index.get_level_values("datetime").unique())
        if len(missing):
            raise DatasetError(f"no universe member has a finite signed score on {missing[0]:%Y-%m-%d}")
        signal_days = pd.DatetimeIndex(signal_days)
    if transform not in SIGNAL_TRANSFORMS:
        raise DatasetError(f"signal transform must be one of {', '.join(SIGNAL_TRANSFORMS)}")
    weights = raw.astype(float).copy()
    for day in signal_days:
        section = transform_section(raw.loc[day].to_numpy(dtype=float), transform)
        scale = float(np.max(np.abs(section)))
        if scale == 0 or gross_exposure == 0:
            normalized = np.zeros(len(section), dtype=np.float32)
        else:
            unit = section / scale
            denominator = math.fsum(abs(float(value)) for value in unit)
            normalized = np.asarray(unit * (gross_exposure / denominator), dtype=np.float32)
            # Qlib's signal adapter stores float32. Enforce the limit on those
            # exact values, rather than on higher-precision intermediates.
            while math.fsum(abs(float(value)) for value in normalized) > gross_exposure + 1e-12:
                largest = int(np.argmax(np.abs(normalized)))
                normalized[largest] = np.nextafter(normalized[largest], np.float32(0))
        weights.loc[day] = normalized.astype(float)
    raw.name = "score"
    weights.name = "weight"
    return raw.sort_index(), weights.sort_index()


def signed_daily(report, positions, initial_cash):
    """Check the derivative's net ledger and preserve every daily observation."""
    import numpy as np
    import pandas as pd

    fields = ("account", "turnover", "net_return", "free_cash", "cash_total",
              "restricted_cash", "long_value", "short_liability", "transaction_cost_amount",
              "borrow_cost_amount", "total_cost_amount", "unfunded_cash_shortfall",
              "funding_feasible", "metrics_defined", "stale_marks")
    if report.empty or any(field not in report for field in fields) or len(report) != len(positions):
        raise DatasetError("signed Qlib ledger or position history incomplete")
    numeric = [x for x in fields if x not in ("net_return", "turnover", "funding_feasible", "metrics_defined", "stale_marks")]
    if not np.isfinite(report[numeric].to_numpy(dtype=float)).all():
        raise DatasetError("signed Qlib ledger has nonfinite balances")
    nonnegative = ["restricted_cash", "long_value", "short_liability", "transaction_cost_amount",
                   "borrow_cost_amount", "total_cost_amount", "unfunded_cash_shortfall"]
    if (report[nonnegative] < -1e-6).any().any():
        raise DatasetError("signed Qlib ledger has negative expense, liability or restricted balance")
    close = lambda left, right: np.allclose(left, right, rtol=1e-7, atol=1e-6)
    if not close(report["account"], report["cash_total"] + report["long_value"] - report["short_liability"]):
        raise DatasetError("signed cash, holdings and liabilities do not reconcile")
    if not close(report["cash_total"], report["free_cash"] + report["restricted_cash"]):
        raise DatasetError("signed free and restricted cash do not reconcile")
    if not close(report["total_cost_amount"], report["transaction_cost_amount"] + report["borrow_cost_amount"]):
        raise DatasetError("signed costs do not reconcile")
    prior = report["account"].shift(1)
    prior.iloc[0] = initial_cash
    defined = (prior > 0) & (report["account"] > 0)
    if not (report["metrics_defined"].astype(bool) == defined).all():
        raise DatasetError("signed metrics-defined flags disagree with equity")
    for field in ("net_return", "turnover", "cost"):
        if field in report:
            if not np.isfinite(report.loc[defined, field].to_numpy(dtype=float)).all():
                raise DatasetError(f"defined signed {field} must be finite")
            if report.loc[~defined, field].notna().any():
                raise DatasetError(f"undefined signed {field} must be NaN")
    if not close(report.loc[defined, "net_return"], (report["account"] / prior - 1).loc[defined]):
        raise DatasetError("signed net returns do not reconcile with equity")
    if "return" in report and "cost" in report and not close(
        report.loc[defined, "return"] - report.loc[defined, "cost"], report.loc[defined, "net_return"]
    ):
        raise DatasetError("signed native net return disagrees with derivative ledger")
    for day, position in positions.items():
        if day not in report.index or not math.isclose(float(position.calculate_value()), float(report.loc[day, "account"]), rel_tol=1e-7, abs_tol=1e-6):
            raise DatasetError("signed position equity does not reconcile")
    daily = pd.DataFrame(index=report.index)
    daily["account"] = report["account"]
    daily["net_return"] = report["net_return"]
    daily["turnover"] = report["turnover"]
    daily["cash"] = report["free_cash"]
    daily["cost_dollars"] = report["total_cost_amount"]
    for field in fields[3:]:
        if field == "stale_marks":
            daily[field] = report[field].map(bool)
            daily["stale_mark_symbols"] = report[field].map(lambda value: json.dumps(list(value)))
        else:
            daily[field] = report[field]
    for native, output in (("transaction_cost_amount", "transaction_cost_dollars"),
                           ("borrow_cost_amount", "borrow_cost_dollars"),
                           ("total_cost_amount", "total_cost_dollars")):
        daily[output] = report[native]
    daily["holdings_value"] = report["long_value"] - report["short_liability"]
    daily["cost"] = report["cost"]
    daily.index.name = "datetime"
    return daily


def signed_positions_frame(positions, daily):
    import pandas as pd

    rows = []
    state_fields = ("cash_total", "free_cash", "restricted_cash", "long_value", "short_liability",
                    "transaction_cost_dollars", "borrow_cost_dollars", "total_cost_dollars",
                    "unfunded_cash_shortfall", "funding_feasible", "metrics_defined", "stale_marks", "stale_mark_symbols")
    for day, position in positions.items():
        state = daily.loc[day]
        for symbol in sorted(position.get_stock_list()) or [""]:
            amount = float(position.get_stock_amount(symbol)) if symbol else 0.0
            price = float(position.get_stock_price(symbol)) if symbol else None
            value = amount * price if symbol else 0.0
            equity = float(state["account"])
            rows.append({"datetime": day, "instrument": symbol, "amount": amount, "price": price,
                         "value": value, "weight": value / equity if equity > 0 else float("nan"),
                         "cash": state["cash"], "account": equity,
                         **{field: state[field] for field in state_fields}})
    return pd.DataFrame(rows)


def signed_decisions(strategy):
    """Serialize strategy intent and mutated orders after the exchange has run."""
    result = []
    for record in strategy.decisions:
        orders = []
        for order in record["issued_orders"]:
            orders.append({"instrument": order.stock_id,
                           "direction": "buy" if order.direction == order.BUY else "sell",
                           "requested_shares": float(order.amount),
                           "filled_shares": float(getattr(order, "deal_amount", 0.0) or 0.0),
                           "skip_reason": getattr(order, "signed_skip_reason", None),
                           "partial_reason": getattr(order, "signed_partial_reason", None)})
        result.append({"session": str(record["session"]),
                       "signal_session": str(record["signal_session"]) if record["signal_session"] is not None else None,
                       "target_weights": record["target_weights"],
                       "target_quantities": record["target_quantities"],
                       "sizing_equity": record["sizing_equity"],
                       "reasons": record["reasons"], "issued_orders": orders})
    return result
