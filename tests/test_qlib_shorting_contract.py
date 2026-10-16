"""Independent known-answer contract checks; no Qlib or runtime imports."""

import json
from decimal import Decimal
from pathlib import Path
import unittest

from shorting_contract_oracle import SignedLedger, UnaffordableFill, normalize_weights


FIXTURES = Path(__file__).parent / "fixtures" / "qlib_shorting"
REQUIRED_FIXTURES = {
    "short_partial.json": (5,),
    "pooled_short_proceeds.json": (6,),
    "reversals.json": (4, 3),
    "weekend_and_deficit.json": (4, 4),
}


class SignedAlphaTests(unittest.TestCase):
    def test_raw_gross_normalization_preserves_signs_and_ratios(self):
        weights = normalize_weights({"A": 3, "B": 1, "C": -2})
        self.assertEqual(weights, {"A": Decimal("0.5"), "B": Decimal(1) / 6, "C": -Decimal(1) / 3})
        self.assertAlmostEqual(sum(weights.values()), Decimal(1) / 3)
        self.assertEqual(sum(map(abs, weights.values())), 1)
        self.assertEqual(normalize_weights({"A": 3, "B": 1, "C": -2}, "0.6"),
                         {"A": Decimal("0.3"), "B": Decimal("0.1"), "C": Decimal("-0.2")})

    def test_one_sided_and_zero_scores(self):
        self.assertEqual(normalize_weights({"A": 3, "B": 1}),
                         {"A": Decimal("0.75"), "B": Decimal("0.25")})
        self.assertEqual(normalize_weights({"A": -3, "B": -1}),
                         {"A": Decimal("-0.75"), "B": Decimal("-0.25")})
        self.assertEqual(normalize_weights({"A": 0, "B": 0}),
                         {"A": Decimal(0), "B": Decimal(0)})
        self.assertEqual(normalize_weights({}, 0), {})

    def test_scaling_sign_flip_single_asset_and_zero_gross(self):
        base = normalize_weights({"A": 3, "B": 1, "C": -2})
        self.assertEqual(normalize_weights({"A": 6, "B": 2, "C": -4}), base)
        self.assertEqual(normalize_weights({"A": -3, "B": -1, "C": 2}),
                         {symbol: -weight for symbol, weight in base.items()})
        self.assertEqual(normalize_weights({"A": 7}), {"A": Decimal(1)})
        self.assertEqual(normalize_weights({"A": -7}), {"A": Decimal(-1)})
        self.assertEqual(normalize_weights({"A": 3, "B": -2}, 0),
                         {"A": Decimal(0), "B": Decimal(0)})

    def test_bad_scores_and_gross_are_rejected(self):
        for bad in ("NaN", "Infinity", "-Infinity", float("nan")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                normalize_weights({"A": bad})
        for bad in ("NaN", "1.01", "-0.01"):
            with self.subTest(gross=bad), self.assertRaises(ValueError):
                normalize_weights({"A": 1}, bad)


class LedgerFixtureTests(unittest.TestCase):
    def test_literal_event_ledgers(self):
        paths = sorted(FIXTURES.glob("*.json"))
        self.assertEqual({path.name for path in paths}, set(REQUIRED_FIXTURES))
        case_count = event_count = 0
        for path in paths:
            document = json.loads(path.read_text(encoding="utf-8"))
            cases = document.get("cases", [document])
            self.assertEqual(len(cases), len(REQUIRED_FIXTURES[path.name]),
                             f"case count changed in {path.name}")
            for case_index, case in enumerate(cases):
                case_count += 1
                with self.subTest(fixture=path.name, case=case_index):
                    ledger = SignedLedger(case["initial_free_cash"])
                    events = case["events"]
                    self.assertEqual(len(events), REQUIRED_FIXTURES[path.name][case_index],
                                     f"event count changed in {path.name} case {case_index}")
                    for event_index, event in enumerate(events):
                        event_count += 1
                        with self.subTest(event=event_index):
                            if "fill" in event:
                                ledger.fill(*event["fill"])
                            elif "mark" in event:
                                ledger.mark(event["mark"])
                            elif "borrow" in event:
                                ledger.accrue_borrow(event["borrow"])
                            else:
                                self.fail(f"unknown fixture event: {event}")
                            expected = {
                                key: ({symbol: Decimal(value) for symbol, value in raw.items()}
                                      if key == "positions" else Decimal(raw))
                                for key, raw in event["expected"].items()
                            }
                            self.assertEqual(ledger.snapshot(), expected)
        self.assertGreater(case_count, 0)
        self.assertGreater(event_count, 0)

    def test_restricted_proceeds_cannot_fund_a_long_and_rejection_is_atomic(self):
        ledger = SignedLedger("100")
        ledger.fill("S", -100, 10, 0)
        before = ledger.snapshot()
        self.assertEqual(before["free_cash"], 100)
        self.assertEqual(before["restricted_total"], 1000)
        self.assertEqual(before["equity"], 100)
        with self.assertRaises(UnaffordableFill):
            ledger.fill("L", 20, 10, 0)
        self.assertEqual(ledger.snapshot(), before)
        with self.assertRaises(UnaffordableFill):
            ledger.fill("S", 11, 20, 0)
        self.assertEqual(ledger.snapshot(), before)

    def test_fee_is_in_affordability_and_refusal_does_not_book_it(self):
        ledger = SignedLedger("10")
        with self.assertRaises(UnaffordableFill):
            ledger.fill("L", 1, 10)
        self.assertEqual(ledger.snapshot()["transaction_cost"], 0)
        self.assertEqual(ledger.snapshot()["positions"], {})

    def test_flat_price_short_round_trip_without_fees_has_no_upfront_gain(self):
        ledger = SignedLedger("100")
        ledger.fill("S", -10, 5, 0)
        self.assertEqual(ledger.snapshot()["equity"], 100)
        self.assertEqual(ledger.snapshot()["free_cash"], 100)
        ledger.fill("S", 10, 5, 0)
        self.assertEqual(ledger.snapshot()["equity"], 100)
        self.assertEqual(ledger.snapshot()["restricted_total"], 0)

    def test_borrow_on_long_zero_and_flat_is_zero(self):
        ledger = SignedLedger("100")
        self.assertEqual(ledger.accrue_borrow(3), 0)
        ledger.fill("L", 1, 10, 0)
        ledger.mark({"L": 12})
        self.assertEqual(ledger.accrue_borrow(3), 0)
        ledger.fill("L", -1, 12, 0)
        self.assertEqual(ledger.accrue_borrow(3), 0)
        self.assertEqual(ledger.snapshot()["financing_cost"], 0)

    def test_exogenous_deficit_survives_and_manual_fill_remains_refused(self):
        ledger = SignedLedger(1)
        ledger.fill("S", -100, 1, 0)
        ledger.mark({"S": 100})
        ledger.accrue_borrow(365)
        before = ledger.snapshot()
        with self.assertRaises(UnaffordableFill):
            ledger.fill("L", 1, 1, 0)
        self.assertEqual(ledger.snapshot(), before)
        ledger.mark({"S": 50})
        self.assertEqual(ledger.snapshot()["funding_shortfall"], 39)
        self.assertEqual(ledger.snapshot()["equity"], -4939)

    def test_zero_net_exposure_still_has_account_equity(self):
        ledger = SignedLedger(100)
        ledger.fill("L", 5, 10, 0)
        ledger.fill("S", -5, 10, 0)
        self.assertEqual(ledger.snapshot()["equity"], 100)
        self.assertEqual(ledger.snapshot()["free_cash"], 50)
        self.assertEqual(ledger.snapshot()["restricted_total"], 50)
        self.assertEqual(ledger.positions["L"] * 10 + ledger.positions["S"] * 10, 0)

    def test_nonfinite_ledger_inputs_fail_without_mutation(self):
        ledger = SignedLedger(100)
        before = ledger.snapshot()
        for attempted in (
            lambda: ledger.fill("S", "NaN", 10),
            lambda: ledger.fill("S", -1, "Infinity"),
            lambda: ledger.fill("S", -1, 10, "NaN"),
            lambda: ledger.mark({"S": "NaN"}),
            lambda: ledger.accrue_borrow("Infinity"),
        ):
            with self.assertRaises(ValueError):
                attempted()
            self.assertEqual(ledger.snapshot(), before)

    def test_calendar_gap_requires_nonnegative_whole_days(self):
        ledger = SignedLedger(100)
        ledger.fill("S", -1, 10, 0)
        before = ledger.snapshot()
        for days in ("-1", "0.5", "NaN", "Infinity"):
            with self.subTest(days=days), self.assertRaises(ValueError):
                ledger.accrue_borrow(days)
            self.assertEqual(ledger.snapshot(), before)
        self.assertEqual(ledger.accrue_borrow(0), 0)
        self.assertEqual(ledger.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
