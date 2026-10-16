"""Compare the account adapter with the actual pinned derivative in isolation."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tracemalloc
import unittest
from unittest.mock import patch

from qlib_yahoo.dataset import DatasetError
from qlib_yahoo.runtime import PIN, ROOT, platform_pin


CHILD = "QLIB_FAST_ACCOUNT_SIGNED_CHILD"
DAY = "2024-01-05"


def _installed():
    try:
        pin = platform_pin(json.loads(PIN.read_text(encoding="utf-8")))
        return (sys.version_info[:2] == (3, 12)
                and importlib.util.find_spec("pandas") is not None
                and (ROOT / pin["local_runtime"] / "qlib/_tubby_build.json").is_file())
    except (OSError, ValueError, KeyError, DatasetError):
        return False


@unittest.skipIf(os.environ.get(CHILD) == "1", "parent launcher")
class SignedProcess(unittest.TestCase):
    @unittest.skipUnless(_installed(), "pinned derivative unavailable on this host")
    def test_pinned_adapter_in_fresh_process(self):
        program = (
            "from qlib_yahoo.runtime import ensure_signed_runtime; "
            "ensure_signed_runtime(); import unittest; "
            "suite=unittest.defaultTestLoader.discover('tests', pattern='test_qlib_fast_account.py'); "
            "result=unittest.TextTestRunner(verbosity=2).run(suite); "
            "raise SystemExit(not result.wasSuccessful())"
        )
        env = os.environ.copy()
        env[CHILD] = "1"
        result = subprocess.run([sys.executable, "-c", program], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(os.environ.get(CHILD) == "1", "requires fresh verified signed process")
class AdapterAgainstDerivative(unittest.TestCase):
    def setUp(self):
        from qlib.backtest.signed_account import SignedAccount
        from qlib_yahoo.fast_account import FastSignedAccount
        self.original = SignedAccount
        self.adapter = FastSignedAccount

    def accounts(self, cash=1000, rate=0.004):
        return [cls(init_cash=cash, annual_borrow_rate=rate) for cls in
                (self.original, self.adapter)]

    def test_adapter_only_overrides_reviewed_legacy_commit(self):
        from qlib_yahoo.fast_account import LEGACY_COMMIT, account_class_for
        self.assertIs(account_class_for(LEGACY_COMMIT), self.adapter)
        # tubby.2 and future pins retain the installed native account.
        self.assertIs(account_class_for("da54853c7598599f74ff7776dc979b74106eb1d0"), self.original)
        self.assertIs(account_class_for("future-reviewed-pin"), self.original)

    @staticmethod
    def venue(closes=None):
        from qlib.backtest.signed_exchange import SignedExchange
        result = SignedExchange.__new__(SignedExchange)
        closes = closes or {}
        result.get_close = lambda code, start, end: closes.get((str(start)[:10], code))
        return result

    @staticmethod
    def order(code, shares, day=DAY):
        from qlib.backtest.decision import Order
        return Order(code, abs(shares), Order.BUY if shares > 0 else Order.SELL, day, day)

    def fill_both(self, accounts, code, shares, price, fee_rate=0.0005, day=DAY):
        value = abs(shares * price)
        for account in accounts:
            account.update_order(self.order(code, shares, day), value, value * fee_rate, price)
        self.assert_positions_equal(accounts)

    def assert_positions_equal(self, accounts):
        left, right = (account.current_position for account in accounts)
        self.assertEqual(left.position, right.position)
        self.assertEqual(left.restricted_proceeds, right.restricted_proceeds)
        self.assertEqual(left.transaction_cost, right.transaction_cost)
        self.assertEqual(left.borrow_cost, right.borrow_cost)
        self.assertEqual(accounts[0].accum_info.get_cost, accounts[1].accum_info.get_cost)
        self.assertEqual(accounts[0].accum_info.get_turnover, accounts[1].accum_info.get_turnover)
        # Only the ancillary accumulated return changes summation order.
        self.assertAlmostEqual(accounts[0].accum_info.get_return,
                               accounts[1].accum_info.get_return, delta=1e-9)

    def test_literal_fixture_paths_against_original(self):
        fixtures = ROOT / "tests/fixtures/qlib_shorting"
        for name in ("short_partial", "pooled_short_proceeds", "reversals", "weekend_and_deficit"):
            document = json.loads((fixtures / f"{name}.json").read_text(encoding="utf-8"))
            for case in document.get("cases", [document]):
                with self.subTest(name=name, case=case):
                    accounts = self.accounts(float(case["initial_free_cash"]))
                    for account in accounts:
                        account.start_bar(DAY, DAY, self.venue())
                    for event in case["events"]:
                        if "fill" in event:
                            code, signed, price, *fee = event["fill"]
                            self.fill_both(accounts, code, int(signed), float(price),
                                           float(fee[0]) if fee else 0.0005)
                        elif "mark" in event:
                            for account in accounts:
                                for code, price in event["mark"].items():
                                    account.current_position.update_stock_price(code, float(price))
                        elif "borrow" in event:
                            for account in accounts:
                                fee = account.current_position.accrue_borrow(int(event["borrow"]),
                                                                              account.annual_borrow_rate)
                                account.accum_info.add_cost(fee)
                        self.assert_positions_equal(accounts)
                        self.assertEqual(accounts[0].current_position.snapshot(),
                                         accounts[1].current_position.snapshot())

    def test_daily_metrics_and_unchanged_other_stock(self):
        accounts = self.accounts()
        venue = self.venue({(DAY, "L"): 12, (DAY, "S"): 8,
                            ("2024-01-08", "L"): 11, ("2024-01-08", "S"): 9})
        for account in accounts:
            account.start_bar(DAY, DAY, venue)
        self.fill_both(accounts, "L", 5, 10)
        other = [a.current_position.position["L"].copy() for a in accounts]
        self.fill_both(accounts, "S", -10, 10)
        self.fill_both(accounts, "S", 5, 8)
        self.assertEqual(other, [a.current_position.position["L"] for a in accounts])
        for day in (DAY, "2024-01-08"):
            if day != DAY:
                for account in accounts:
                    account.start_bar(day, day, venue)
                self.fill_both(accounts, "S", 5, 9, day=day)
            for account in accounts:
                with patch.object(account, "update_indicator"):
                    account.update_bar_end(day, day, venue, atomic=True,
                                           outer_trade_decision=None, trade_info=[])
            self.assert_positions_equal(accounts)
            left, right = (a.get_portfolio_metrics()[0].loc[day] for a in accounts)
            for field in left.index:
                self.assertEqual(left[field], right[field], field)

    def test_rejections_and_tiny_fee_clamp(self):
        accounts = self.accounts(cash=10)
        for account in accounts:
            with self.assertRaisesRegex(ValueError, "start_bar"):
                account.update_order(self.order("L", 1), 10, 0, 10)
            account.start_bar(DAY, DAY, self.venue())
            with self.assertRaisesRegex(ValueError, "order date"):
                account.update_order(self.order("L", 1, "2024-01-08"), 10, 0, 10)
        for account in accounts:
            before = account.current_position.snapshot()
            with self.assertRaisesRegex(ValueError, "insufficient free cash"):
                account.update_order(self.order("L", 2), 20, 0, 10)
            self.assertEqual(before, account.current_position.snapshot())
        self.fill_both(accounts, "L", 1, 10, fee_rate=0)
        self.fill_both(accounts, "L", -1, 10, fee_rate=0)
        # 0.3 + 0.1 - 0.4 rounds slightly negative and must clamp to zero.
        accounts = self.accounts(cash=1)
        for account in accounts:
            account.start_bar(DAY, DAY, self.venue())
        self.fill_both(accounts, "L", 1, 1, fee_rate=0)
        for account in accounts:
            account.current_position.position["cash"] = 0.3
        self.fill_both(accounts, "L", -1, 0.1, fee_rate=4)
        self.assertEqual(accounts[1].current_position.get_cash(), 0)

        # Borrow can make free cash negative; the next fill still must not spend it.
        accounts = self.accounts(cash=1, rate=365)
        for account in accounts:
            account.start_bar(DAY, DAY, self.venue())
        self.fill_both(accounts, "S", -1, 1, fee_rate=0)
        for account in accounts:
            account.current_position.accrue_borrow(1, account.annual_borrow_rate)
            before = account.current_position.snapshot()
            with self.assertRaisesRegex(ValueError, "insufficient free cash"):
                account.update_order(self.order("L", 1), 1, 0, 1)
            self.assertEqual(before, account.current_position.snapshot())
        self.assert_positions_equal(accounts)

    def test_many_holdings_no_per_fill_full_book_scan(self):
        account = self.adapter(init_cash=10000)
        account.start_bar(DAY, DAY, self.venue())
        for index in range(256):
            account.update_order(self.order(f"S{index}", -1), 1, 0, 1)
        position = account.current_position
        tracemalloc.start()
        started = time.perf_counter()
        with (patch.object(position, "calculate_value", side_effect=AssertionError("full equity scan")),
              patch.object(position, "calculate_stock_value", side_effect=AssertionError("stock scan")),
              patch.object(position, "get_total_cash", side_effect=AssertionError("restricted scan")),
              patch.object(position, "get_stock_list", side_effect=AssertionError("holdings scan"))):
            for index in range(512):
                account.update_order(self.order(f"S{index % 256}", -1), 1, 0, 1)
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.assertEqual(len(position.restricted_proceeds), 256)
        print(f"bounded 256-holding/512-fill probe: {elapsed:.3f}s, peak {peak} bytes")


if __name__ == "__main__":
    unittest.main()
