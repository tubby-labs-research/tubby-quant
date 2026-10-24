"""Opt-in independent risk at the pinned signed engine's decision/fill boundary.

Preflight: tubby.4 / 8ab6249 SignedAccount.start_bar charges borrow before
strategy decisions. SignedExchange.deal_order(position=...) projects native
affordability/fees/restricted proceeds on a copied SignedPosition (whose
preview_fill is the native funding oracle). SimulatorExecutor supplies actual
trade_info to account.update_bar_end before daily closing marks. These existing
instance hooks suffice: no Qlib source, runtime pin or ledger implementation changes.

Ordered/compiled execution must be disabled by the caller. The runner records
that explicit fallback; normal runs never install these hooks. Recheck this
adapter against the native hook contracts when upgrading the signed derivative.
"""
from __future__ import annotations

from collections import defaultdict
from copy import copy, deepcopy
import gzip
import io
import json
import math
from pathlib import Path

from .dataset import DatasetError, sha256
from .portfolio import RiskLimits, evaluate_risk

__all__ = ["validate_portfolio_risk", "PortfolioRiskRuntime"]
_LIMITS = ("gross", "net", "name", "group_gross", "group_net", "turnover", "liquidity")


def validate_portfolio_risk(policy):
    """Detach schema-1 policy; all seven limits must be explicit (null disables).

    group_overlay/group_level can be null even with group limits: unavailable
    labels then block affected sessions rather than silently disabling the limit.
    Optional provenance is a finite JSON object, retained without interpretation.
    """
    if (not isinstance(policy, dict) or type(policy.get("schema_version")) is not int
            or policy["schema_version"] != 1
            or set(policy) - {"schema_version", "limits", "group_overlay", "group_level", "provenance"}
            or not isinstance(policy.get("limits"), dict) or set(policy["limits"]) != set(_LIMITS)):
        raise DatasetError("portfolio_risk requires schema_version=1 and supported explicit limits")
    try:
        limits = RiskLimits(**policy["limits"])
        overlay, level = policy.get("group_overlay"), policy.get("group_level")
        if overlay is not None and (not isinstance(overlay, str) or not overlay.strip()):
            raise ValueError("group_overlay must be a path or null")
        if level not in (None, "sector", "industry", "country"):
            raise ValueError("invalid group_level")
        provenance = policy.get("provenance", {})
        if not isinstance(provenance, dict):
            raise ValueError("provenance must be an object")
        return json.loads(json.dumps({"schema_version": 1,
            "limits": {name: getattr(limits, name) for name in _LIMITS},
            "group_overlay": overlay, "group_level": level, "provenance": provenance}, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as error:
        raise DatasetError(f"invalid portfolio_risk: {error}") from error


def _finite(value, *, positive=False):
    try:
        return value is not None and math.isfinite(float(value)) and (not positive or value > 0)
    except (ValueError, TypeError, OverflowError):
        return False


def _date(value):
    import pandas as pd
    return None if value is None else str(pd.Timestamp(value).date())


def _amounts(orders, *, filled=False):
    from qlib.backtest.decision import Order
    net, gross = defaultdict(float), defaultdict(float)
    for order in orders:
        quantity = float(order.deal_amount if filled else order.amount)
        if not _finite(quantity) or quantity < 0:
            raise DatasetError("risk hook requires finite nonnegative order quantities")
        net[order.stock_id] += quantity * (1 if order.direction == Order.BUY else -1)
        gross[order.stock_id] += quantity
    return dict(net), dict(gross)


def _compact(result, *, missing=(), violations=()):
    missing = sorted(set(result.missing_inputs) | set(missing))
    violations = sorted(set(result.violations) | set(violations))
    metrics = {key: value for key, value in result.metrics.items()
               if key not in ("assessed_notionals_usd", "holdings_notionals_usd")}
    for key in ("assessed", "target"):
        if key in metrics:
            metrics[key] = {k: v for k, v in metrics[key].items() if k != "weights"}
    return {"status": "missing_inputs" if missing else "blocked" if violations else "pass",
            "missing_inputs": missing, "violations": violations, "metrics": metrics}


class PortfolioRiskRuntime:
    """Install once on a fresh signed strategy/account; close before run sealing.

    Pre-order exposure uses native executable net quantities and projected
    post-fee equity/cash; turnover/liquidity conservatively use ALL proposed gross
    shares, including native-unaffordable portions. Post-fill uses actual net and
    gross executed shares, actual cash and equity marked at the same current open.
    Native whole-share clipping is retained, with requested/unfilled amounts visible.
    Zero affordable fills for a nonempty proposal is explicitly blocked.

    Group states are frozen at the PRIOR decision-session open, using the existing
    verified overlay's strict activation/expiry convention. This conservative
    decision cutoff excludes later filings (including execution-morning filings).
    Volume is the prior decision session's final volume, never current daily volume.
    Missing current opens never substitute stale marks in risk ratios. Actual
    holdings and stale ledger marks remain intact for native accounting.
    """

    artifact_name = "portfolio_risk.jsonl.gz"

    def __init__(self, policy, *, dataset, symbols, decision_start, decision_end, artifact_path):
        self.policy = validate_portfolio_risk(policy)
        self.limits = RiskLimits(**self.policy["limits"])
        self.overlay = self.volume = None
        self.group_rows, self.group_columns = {}, {}
        self.volume_rows, self.volume_columns = {}, {}
        self.sources = {"dataset_manifest_sha256": sha256(Path(dataset) / "manifest.json"),
                        "reference_prices": "current session open; no stale/future-close substitution",
                        "groups_cutoff": "prior decision-session open; strict dated-overlay activation",
                        "liquidity_cutoff": "prior decision-session final volume in shares",
                        "pretrade_equity": "native projected equity at current open, after borrow and projected transaction fees",
                        "turnover": "full gross shares times current open / phase equity; no half convention"}
        if self.limits.group_gross is not None or self.limits.group_net is not None:
            if self.policy["group_overlay"] and self.policy["group_level"]:
                from .classification import read_classification_overlay
                try:
                    self.overlay = read_classification_overlay(dataset, self.policy["group_overlay"],
                        start=_date(decision_start), end=_date(decision_end), symbols=symbols)
                    self.group_rows = {day: i for i, day in enumerate(self.overlay.calendar)}
                    self.group_columns = {s: i for i, s in enumerate(self.overlay.symbols)}
                    self.sources["group_identity"] = dict(self.overlay.identity)
                except (DatasetError, OSError, ValueError) as error:
                    self.sources["group_error"] = str(error)
            else:
                self.sources["group_error"] = "enabled group limits require a verified overlay and level"
        if self.limits.liquidity is not None:
            from qlib.data import D
            try:
                panel = D.features(list(symbols), ["$volume"], start_time=decision_start,
                                   end_time=decision_end, freq="day", disk_cache=0)
                panel = panel["$volume"].unstack("instrument").reindex(columns=symbols)
                self.volume_rows = {_date(day): i for i, day in enumerate(panel.index)}
                self.volume_columns = {s: i for i, s in enumerate(panel.columns)}
                self.volume = panel.to_numpy(copy=True)
                self.volume.setflags(write=False)
            except (DatasetError, OSError, ValueError, KeyError) as error:
                self.sources["liquidity_error"] = str(error)
        self.counts = {"sessions": 0, "blocked_rebalances": 0, "native_partial_or_unfilled_orders": 0,
                       "pre_order": {key: 0 for key in ("pass", "blocked", "missing_inputs")},
                       "post_fill": {key: 0 for key in ("pass", "blocked", "missing_inputs")}}
        self.pending = None
        self.closed = False
        self.raw = Path(artifact_path).open("xb")
        self.packed = gzip.GzipFile(filename="", mode="wb", fileobj=self.raw, mtime=0)
        self.stream = io.TextIOWrapper(self.packed, encoding="utf-8", newline="\n")

    def snapshot(self):
        return {"schema_version": 1, "policy": deepcopy(self.policy), "sources": deepcopy(self.sources),
                "artifact": self.artifact_name, "counts": deepcopy(self.counts),
                "hooks": "native decision wrapper and pre-close post-fill account hook",
                "execution": "native sequential fills; ordered/compiled fill and target accelerators declined",
                "post_fill_timing": "after actual fills, before daily close marking",
                "model_omissions": ["no live borrow availability, recalls, broker margin or market impact"]}

    def _context(self, day, symbols):
        groups, volume = {}, {}
        if day in self.group_rows:
            row = self.group_rows[day]
            level = self.policy["group_level"]
            for symbol in symbols:
                column = self.group_columns.get(symbol)
                if column is not None:
                    code = int(self.overlay.codes[level][row, column])
                    if code >= 0:
                        groups[symbol] = self.overlay.labels[level][code]
        if day in self.volume_rows:
            row = self.volume_rows[day]
            for symbol in symbols:
                column = self.volume_columns.get(symbol)
                if column is not None:
                    value = float(self.volume[row, column])
                    if _finite(value) and value >= 0:
                        volume[symbol] = value
        return groups, volume

    @staticmethod
    def _marked_copy(position, prices):
        marked = deepcopy(position)
        for symbol in position.get_stock_list():
            if symbol not in prices:
                return marked, None
            marked.update_stock_price(symbol, prices[symbol])
        return marked, marked.calculate_value()

    def _write(self, record):
        self.stream.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        self.counts[record["phase"]][record["status"]] += 1

    def attach(self, strategy, account, exchange):
        """Wrap instance hooks, retaining the native account/exchange/executor types."""
        from qlib.backtest.signed_account import SignedAccount
        from qlib.backtest.signed_exchange import SignedExchange
        from qlib.backtest.signed_position import SignedPosition
        from qlib.contrib.strategy.signed_weight import SignedWeightStrategy
        if (not isinstance(account, SignedAccount) or not isinstance(exchange, SignedExchange)
                or not isinstance(strategy, SignedWeightStrategy)
                or not isinstance(account.current_position, SignedPosition)
                or not callable(getattr(account.current_position, "preview_fill", None))
                or getattr(strategy, "_portfolio_risk_attached", False)):
            raise DatasetError("portfolio risk requires fresh reviewed native signed hooks")
        original_decision, original_end = strategy.generate_trade_decision, account.update_bar_end

        def checked_decision(*args, **kwargs):
            decision = original_decision(*args, **kwargs)
            return self._pre_order(strategy, account, exchange, decision)

        def checked_end(start, end, trade_exchange, atomic, **kwargs):
            if not atomic or trade_exchange is not exchange:
                raise DatasetError("portfolio risk supports only the attached daily atomic exchange")
            self._post_fill(account, kwargs.get("trade_info"))
            return original_end(start, end, trade_exchange, atomic=atomic, **kwargs)

        strategy.generate_trade_decision = checked_decision
        strategy._portfolio_risk_attached = True
        account.update_bar_end = checked_end
        return self

    def _pre_order(self, strategy, account, exchange, decision):
        from qlib.backtest.decision import Order, TradeDecisionWO
        if self.pending is not None:
            raise DatasetError("risk session was not closed before next decision")
        record = strategy.decisions[-1]
        start, end = strategy.trade_calendar.get_step_time()
        if account._active_session is None:
            raise DatasetError("native start_bar must charge borrow before risk checks")
        holdings = dict(account.current_position.get_stock_amount_dict())
        requested, gross_requested = _amounts(decision.get_decision())
        target = record.get("target_weights")
        # Zero, unheld names need no market input; held zero targets remain explicit.
        if isinstance(target, dict):
            target = {s: w for s, w in target.items() if w or s in holdings or s in requested}
        symbols = sorted(set(holdings) | set(target or {}) | set(requested))
        prices = {}
        for symbol in symbols:
            try:
                price = exchange.get_deal_price(symbol, start, end, Order.BUY)
                if _finite(price, positive=True):
                    prices[symbol] = float(price)
            except (KeyError, ValueError, TypeError):
                pass  # unavailable current mark is a missing input, never a stale fallback
        day = _date(record.get("signal_session"))
        groups, volume = self._context(day, symbols)
        projected, equity = self._marked_copy(account.current_position, prices)
        missing, violations, diagnostics = [], [], []
        clones, executable, projected_cash = [], {}, None
        if len(prices) == len(symbols):
            try:
                dealt = defaultdict(float)
                for order in decision.get_decision():
                    clone = copy(order)
                    exchange.deal_order(clone, position=projected, dealt_order_amount=dealt)
                    dealt[clone.stock_id] += clone.deal_amount
                    clones.append(clone)
                    if clone.deal_amount != clone.amount:
                        diagnostics.append({"symbol": clone.stock_id, "requested": float(clone.amount),
                            "executable": float(clone.deal_amount),
                            "reason": getattr(clone, "signed_skip_reason", None)
                                      or getattr(clone, "signed_partial_reason", None)})
                executable, gross_executable = _amounts(clones, filled=True)
                projected_cash = projected.get_cash()
                equity = projected.calculate_value()
                if any(gross_requested.values()) and not any(gross_executable.values()):
                    violations.append("execution:no_affordable_fills")
            except (ValueError, TypeError, KeyError, NotImplementedError) as error:
                missing.append("native_projection")
                diagnostics.append({"projection_error": f"{type(error).__name__}: {error}"})
        else:
            missing.extend("reference_prices:" + s for s in symbols if s not in prices)
        result = evaluate_risk(phase="pre_order", holdings=holdings, trades=executable,
            traded_quantities=gross_requested, target_weights=target, equity=equity,
            reference_prices=prices, limits=self.limits, groups=groups, liquidity=volume,
            free_cash=account.current_position.get_cash(), projected_free_cash=projected_cash)
        check = _compact(result, missing=missing, violations=violations)
        self.pending = {"session": _date(start), "decision_session": day, "target": target,
                        "prices": prices, "groups": groups, "volume": volume,
                        "gross_requested": gross_requested}
        self.counts["sessions"] += 1
        self.counts["native_partial_or_unfilled_orders"] += sum("symbol" in item for item in diagnostics)
        self._write({"schema_version": 1, "session": _date(start), "decision_session": day,
                     "phase": "pre_order", **check, "proposed_trades": requested,
                     "proposed_gross_shares": gross_requested, "executable_trades": executable,
                     "native_projection": diagnostics,
                     "sizing_equity": record.get("sizing_equity")})
        if check["status"] != "pass":
            self.counts["blocked_rebalances"] += 1
            record["reasons"]["portfolio_risk"] = check["status"] + ": " + "; ".join(
                check["missing_inputs"] + check["violations"])
            record["issued_orders"] = []
            return TradeDecisionWO([], strategy)
        return decision

    def _post_fill(self, account, trade_info):
        if self.pending is None:
            raise DatasetError("post-fill risk hook has no matching decision")
        state = self.pending
        missing = []
        if not isinstance(trade_info, list):
            missing.append("actual_trade_info")
            trades, gross = {}, {}
        else:
            trades, gross = _amounts([item[0] for item in trade_info], filled=True)
        holdings = dict(account.current_position.get_stock_amount_dict())
        missing.extend("reference_prices:" + symbol for symbol in
                       sorted(set(holdings) | set(state["target"] or {}))
                       if symbol not in state["prices"])
        _, equity = self._marked_copy(account.current_position, state["prices"])
        result = evaluate_risk(phase="post_fill", holdings=holdings, trades=trades,
            traded_quantities=gross, target_weights=state["target"], equity=equity,
            reference_prices=state["prices"], limits=self.limits, groups=state["groups"],
            liquidity=state["volume"], free_cash=account.current_position.get_cash())
        self._write({"schema_version": 1, "session": state["session"],
                     "decision_session": state["decision_session"], "phase": "post_fill",
                     **_compact(result, missing=missing), "actual_holdings": holdings,
                     "executed_trades": trades, "executed_gross_shares": gross,
                     "unfilled_gross_shares": {s: q - gross.get(s, 0.0)
                         for s, q in state["gross_requested"].items() if q != gross.get(s, 0.0)},
                     "residual_weights": {s: w for s, w in result.residual_weights.items() if w != 0}})
        self.pending = None

    def close(self, *, completed=False):
        if not self.closed:
            try:
                self.stream.close()
            finally:
                self.raw.close()
                self.closed = True
        if completed and self.pending is not None:
            raise DatasetError("cannot seal portfolio risk with an unfinished session")
        return self.snapshot()
