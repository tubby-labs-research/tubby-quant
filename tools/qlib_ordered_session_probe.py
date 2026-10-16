"""Synthetic COMPLETE ordered-session boundary; not full backtest acceptance.

Include guards, input packing, optional JIT first use and native-state replay.
Keep all trials and verify exact native state/metadata after each measurement.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run_probe(compiler_path, *, stocks, trials, record, persist):
    if stocks < 10 or trials < 2:
        raise ValueError("Use at least ten stocks and two completed pairs")
    compiler_path = Path(compiler_path).resolve()
    if not compiler_path.is_dir():
        raise ValueError("Isolated compiler package directory is missing")
    record["stage"] = "runtime_import"
    sys.path.insert(0, str(compiler_path))
    from qlib_yahoo.runtime import ensure_signed_runtime
    identity = ensure_signed_runtime()
    import numpy as np
    import pandas as pd
    import numba
    import llvmlite
    from qlib.backtest.account import AccumulatedInfo
    from qlib.backtest.decision import Order
    from qlib.backtest.signed_account import SignedAccount
    from qlib.backtest.signed_position import SignedPosition
    from qlib_yahoo.daily_quote import DailyQuote
    from qlib_yahoo.fast_signed import FastSignedExchange
    from qlib_yahoo.ordered_signed import try_execute_orders

    record["stage"] = "compiler_validation"
    if numba.__version__ != "0.65.1" or llvmlite.__version__ != "0.47.0":
        raise ValueError("Prototype requires reviewed compiler pins")
    record["stage"] = "fixture_preparation"
    day = pd.Timestamp("2025-01-06")
    end = day + pd.Timedelta(hours=23, minutes=59, seconds=59)
    names = [f"S{index:05d}" for index in range(stocks)]
    prices = [float(5 + index % 100) / 3 for index in range(stocks)]
    index = pd.MultiIndex.from_product([names, [day]], names=["instrument", "datetime"])
    frame = pd.DataFrame({"$open": prices, "$close": prices,
                          "$factor": 1.0, "$volume": 100000.0,
                          "limit_buy": False, "limit_sell": False}, index=index)
    order_spec = []
    for index, symbol in enumerate(names):
        # Repeat symbols across reductions/increases, preserving native order.
        order_spec.append((symbol, 3 + index % 7, Order.BUY if index % 3 else Order.SELL))
    for index, symbol in enumerate(names):
        order_spec.append((symbol, 2 + index % 11, Order.SELL if index % 3 else Order.BUY))

    def fresh():
        position = SignedPosition(cash=100000.0)
        for index, symbol in enumerate(names[::3]):
            price = prices[index * 3]
            amount = -5.0 if index % 2 else 5.0
            position._init_stock(symbol, amount, price * 1.02)
            if amount < 0:
                position.restricted_proceeds[symbol] = -amount * price
        account = SignedAccount.__new__(SignedAccount)
        account.current_position = position
        account.accum_info = AccumulatedInfo()
        account._active_session = day
        exchange = FastSignedExchange.__new__(FastSignedExchange)
        exchange.quote_df = frame
        exchange.quote = DailyQuote(frame, "day", region="us")
        exchange.buy_price = exchange.sell_price = "$open"
        exchange.open_cost = exchange.close_cost = 0.0005
        exchange.reset_session_cache()
        exchange.begin_session(day, end)
        # Both variants inherit the row snapshots target sizing normally creates.
        for symbol in names:
            exchange._session_row(symbol, day, end)
        orders = [Order(symbol, amount, direction, day, end) for symbol, amount, direction in order_spec]
        return exchange, account, orders

    def exact(actual, expected):
        if isinstance(expected, dict):
            assert type(actual) is dict and list(actual) == list(expected), "dictionary membership/order"
            for key in expected:
                exact(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            assert type(actual) is type(expected) and len(actual) == len(expected)
            for left, right in zip(actual, expected):
                exact(left, right)
        elif isinstance(expected, (float, np.floating)):
            assert isinstance(actual, (float, np.floating))
            if np.isnan(expected):
                assert np.isnan(actual)
            else:
                assert np.float64(actual).view(np.uint64) == np.float64(expected).view(np.uint64), (actual, expected)
        else:
            assert actual == expected, (actual, expected)

    def snapshot(account, result):
        return {"position": account.current_position.position,
                "restricted": account.current_position.restricted_proceeds,
                "transaction_cost": account.current_position.transaction_cost,
                "accum": vars(account.accum_info),
                "results": [(vars(order).copy(), value, cost, price) for order, value, cost, price in result]}

    record.update({"schema_version": 1, "record_kind": "synthetic_complete_ordered_session_probe",
              "acceptance_status": "ineligible_synthetic_probe", "goal_achieved": False,
              "status": "running",
              "stocks": stocks, "orders_per_session": len(order_spec), "trials": [],
              "signed_engine_identity": identity, "python": sys.version,
              "compiler": {"numba": numba.__version__, "llvmlite": llvmlite.__version__,
                           "path": str(compiler_path), "fastmath": False, "parallel": False,
                           "disk_cache": False, "first_call_compile_included": True},
              "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in (Path(__file__), ROOT / "qlib_yahoo/ordered_signed.py",
                                              ROOT / "qlib_yahoo/ordered_signed_kernel.py")},
              "scope": "preexisting quote/row/ordered requests; guard+packing+compile+ordered fills+native dict replay included; no full backtest or report"})
    persist()

    def execute(variant, exchange, account, orders):
        if variant == "native":
            result = []
            for order in orders:
                value, cost, price = exchange.deal_order(order, trade_account=account)
                result.append((order, value, cost, price))
            return result
        result = try_execute_orders(exchange, account, orders, compiled=True)
        assert result is not NotImplemented, "whole session unexpectedly fell back"
        return result

    return _run_pairs(fresh, execute, snapshot, exact, trials, record, persist)


def _run_pairs(fresh, execute, snapshot, exact, trials, record, persist):
    for trial in range(trials):
        snapshots, times = {}, {}
        variants = ("native", "compiled") if trial % 2 == 0 else ("compiled", "native")
        for variant in variants:
            record.update(stage="fixture", active_trial=trial, active_variant=variant)
            exchange, account, orders = fresh()
            record["stage"] = "execution"
            started = time.perf_counter()
            result = execute(variant, exchange, account, orders)
            times[variant] = time.perf_counter() - started
            record["stage"] = "snapshot"
            snapshots[variant] = snapshot(account, result)
        record["stage"] = "parity"
        exact(snapshots["compiled"], snapshots["native"])
        record["trials"].append({"trial": trial, "variant_order": variants,
                                 "seconds": times, "parity": "exact float bits/state/dict order/order metadata"})
        record.update(stage="paired_trials", completed_pairs=len(record["trials"]))
        persist()
        print(json.dumps(record["trials"][-1]), flush=True)
    record["stage"] = "summary"
    record["warm_median_seconds"] = {variant: statistics.median(
        trial["seconds"][variant] for trial in record["trials"][1:]) for variant in ("native", "compiled")}
    record.update(status="completed", stage="completed",
                  finished_at_utc=datetime.now(timezone.utc).isoformat())
    persist()
    return record


def probe(output, compiler_path, *, stocks=3000, trials=5):
    """Reserve once; retain preparation, pair progress and failures on that handle."""
    output = Path(output).absolute()
    if output.is_symlink() or output.exists() or not output.parent.is_dir():
        raise ValueError("Use a new receipt in an existing output directory")
    record = {"schema_version": 1, "record_kind": "synthetic_complete_ordered_session_probe",
              "acceptance_status": "ineligible_synthetic_probe", "goal_achieved": False,
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "running", "stage": "preparation", "stocks": stocks,
              "requested_trials": trials, "completed_pairs": 0, "trials": []}
    with output.open("x", encoding="utf-8") as receipt:
        def persist():
            encoded = json.dumps(record, indent=2, allow_nan=False) + "\n"
            receipt.seek(0)
            receipt.write(encoded)
            receipt.truncate()
            receipt.flush()
            os.fsync(receipt.fileno())

        persist()
        try:
            return _run_probe(compiler_path, stocks=stocks, trials=trials,
                              record=record, persist=persist)
        except (Exception, KeyboardInterrupt) as exc:
            record.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                          finished_at_utc=datetime.now(timezone.utc).isoformat())
            persist()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler-path", type=Path, required=True)
    parser.add_argument("--stocks", type=int, default=3000)
    parser.add_argument("--trials", type=int, default=5)
    arguments = parser.parse_args()
    probe(arguments.output, arguments.compiler_path, stocks=arguments.stocks, trials=arguments.trials)


if __name__ == "__main__":
    main()
