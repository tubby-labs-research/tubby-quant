"""Portfolio request and invocation gates; actual signed chain has its own fixture."""
from dataclasses import asdict
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from qlib_yahoo.dataset import DatasetError
from qlib_yahoo.portfolio import RiskLimits, evaluate_risk
from qlib_yahoo.portfolio_workflow import portfolio_options, run_portfolio
from qlib_yahoo.workflow_records import EvidenceStore


def options(**changes):
    return {"validation_ids": ["a" * 32, "b" * 32], "coefficients": [.5, .5],
            "account_id": "fixture-usd", "limits": asdict(RiskLimits()), **changes}


class PortfolioWorkflowTests(unittest.TestCase):
    def test_one_account_fraction_and_explicit_risk_contract(self):
        value = portfolio_options(options(coefficients=[.2, .3]))
        self.assertEqual(value["initial_cash"], 100_000)
        self.assertEqual(value["annual_borrow_rate"], .004)
        self.assertEqual(value["coefficients"], [.2, .3])
        self.assertEqual(set(value["limits"]), set(asdict(RiskLimits())))

    def test_invalid_requests_rejected_without_panels(self):
        changes = [
            {"validation_ids": "x"}, {"validation_ids": 4}, {"validation_ids": ["x", "x"]},
            {"coefficients": [.6, .6]}, {"coefficients": [-1, 1]}, {"coefficients": [True, 0]},
            {"coefficients": [float("nan"), 0]}, {"coefficients": [1]},
            {"limits": {}}, {"limits": {"bogus": 1}}, {"account_id": " "},
            {"initial_cash": 0}, {"annual_borrow_rate": -1}, {"cost_per_side": float("inf")},
            {"limits": asdict(RiskLimits(group_gross=.3))},
            {"group_overlay": "somewhere", "group_level": None}, {"deal_price": "close"},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(DatasetError):
                portfolio_options(options(**change))

    def test_group_controls_are_explicit_not_inferred(self):
        value = portfolio_options(options(
            group_overlay="group-source", group_level="industry",
            limits=asdict(RiskLimits(gross=1, net=.1, name=.1, group_gross=.3))))
        self.assertTrue(Path(value["group_overlay"]).is_absolute())
        self.assertEqual(value["group_level"], "industry")
        self.assertIsNone(value["limits"]["liquidity"])

    def test_invalid_direct_run_retains_submission(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            # run_portfolio's deferred imports must not dispatch the engine.
            with patch("qlib_yahoo.backtest.run_backtest") as engine:
                for request in (None, options(validation_ids=4), options()):
                    with self.assertRaises((DatasetError, ValueError, FileNotFoundError)):
                        run_portfolio(root, request)
                engine.assert_not_called()
            rows = EvidenceStore(root).list_attempts()
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row["status"] == "rejected" for row in rows))

    def test_reversal_gross_shares_not_net_delta_for_participation_and_turnover(self):
        common = dict(phase="pre_order", holdings={"A": 10}, trades={"A": -5},
                      target_weights={"A": .05}, equity=1000,
                      reference_prices={"A": 10}, free_cash=900, projected_free_cash=850,
                      limits=RiskLimits(turnover=.2, liquidity=.2), liquidity={"A": 100})
        # Selling 15 then buying 10 has net -5 but 25 shares traded.
        result = evaluate_risk(**common, traded_quantities={"A": 25})
        self.assertEqual(result.metrics["turnover"], .25)
        self.assertEqual(result.metrics["liquidity_participation"]["A"], .25)
        self.assertEqual(result.status, "blocked")
        self.assertIn("trades:turnover", result.violations)
        self.assertIn("trades:liquidity:A", result.violations)
        incomplete = evaluate_risk(**common, traded_quantities={"A": 1})
        self.assertEqual(incomplete.status, "missing_inputs")
        self.assertIn("traded_quantities:A", incomplete.missing_inputs)

    def test_zero_net_round_trip_still_requires_funding_projection(self):
        result = evaluate_risk(phase="pre_order", holdings={}, trades={"A": 0},
                               traded_quantities={"A": 20}, target_weights={"A": 0},
                               equity=1000, reference_prices={"A": 10}, free_cash=1000,
                               limits=RiskLimits())
        self.assertIn("projected_free_cash", result.missing_inputs)


if __name__ == "__main__":
    unittest.main()
