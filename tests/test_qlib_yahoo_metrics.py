"""Known-answer tests for the project's local daily net-return definitions."""

import json
import math
import unittest

from qlib_yahoo.metrics import METRIC_VERSION, holdings_gate, portfolio_metrics


class PortfolioMetricsTests(unittest.TestCase):
    def test_known_answer_sample_std_and_excess(self):
        metrics = portfolio_metrics([0.01, 0.03, -0.02], [0.0, 0.01, 0.0], periods_per_year=4)
        self.assertEqual(METRIC_VERSION, "daily-simple-net-v1")
        self.assertEqual(set(metrics), {"strategy_net", "benchmark_reference", "excess_net_vs_benchmark"})
        strategy = metrics["strategy_net"]
        excess = metrics["excess_net_vs_benchmark"]
        self.assertEqual(strategy["observations"], 3)
        self.assertAlmostEqual(strategy["mean_daily_return"], 0.02 / 3)
        self.assertAlmostEqual(strategy["annualized_arithmetic_return"], 4 * 0.02 / 3)
        sample_std = math.sqrt((0.01 - 0.02 / 3) ** 2 / 2
                               + (0.03 - 0.02 / 3) ** 2 / 2
                               + (-0.02 - 0.02 / 3) ** 2 / 2)
        self.assertAlmostEqual(strategy["annualized_volatility"], sample_std * 2)
        self.assertAlmostEqual(strategy["sharpe"], (0.02 / 3) / sample_std * 2)
        self.assertAlmostEqual(strategy["cumulative_return"], 1.01 * 1.03 * 0.98 - 1)
        self.assertEqual(set(excess), {
            "observations", "mean_daily_excess_return", "annualized_arithmetic_excess_return",
            "tracking_error", "information_ratio",
        })
        self.assertAlmostEqual(excess["mean_daily_excess_return"], (0.01 + 0.02 - 0.02) / 3)
        excess_std = math.sqrt(((0.01 - 0.01 / 3) ** 2
                                + (0.02 - 0.01 / 3) ** 2
                                + (-0.02 - 0.01 / 3) ** 2) / 2)
        self.assertAlmostEqual(excess["tracking_error"], excess_std * 2)
        self.assertAlmostEqual(excess["information_ratio"], (0.01 / 3) / excess_std * 2)
        json.dumps(metrics, allow_nan=False)

    def test_supplied_net_returns_reflect_costs(self):
        gross = portfolio_metrics([0.02, 0.04], [0.0, 0.0], periods_per_year=2)
        net = portfolio_metrics([0.01, 0.03], [0.0, 0.0], periods_per_year=2)
        self.assertLess(net["strategy_net"]["mean_daily_return"], gross["strategy_net"]["mean_daily_return"])
        self.assertLess(net["strategy_net"]["cumulative_return"], gross["strategy_net"]["cumulative_return"])

    def test_cash_rate_only_changes_primary_sharpe(self):
        zero = portfolio_metrics([0.01, 0.03], [0.0, 0.02], periods_per_year=2)
        cash = portfolio_metrics([0.01, 0.03], [0.0, 0.02], annual_cash_rate=0.21, periods_per_year=2)
        std = math.sqrt(0.0002)
        self.assertAlmostEqual(cash["strategy_net"]["sharpe"], (0.02 - 0.1) / std * math.sqrt(2))
        self.assertEqual(cash["strategy_net"]["sharpe_zero_cash_rate"], zero["strategy_net"]["sharpe"])
        self.assertEqual(cash["excess_net_vs_benchmark"], zero["excess_net_vs_benchmark"])

    def test_initial_loss_is_drawdown_from_starting_nav(self):
        metrics = portfolio_metrics([-0.1, 0.1], [0, 0], periods_per_year=2)["strategy_net"]
        self.assertAlmostEqual(metrics["max_drawdown"], -0.1)
        self.assertAlmostEqual(metrics["cumulative_return"], -0.01)
        self.assertAlmostEqual(metrics["annualized_arithmetic_return"], 0.0)
        self.assertAlmostEqual(metrics["cagr"], -0.01)

    def test_short_and_constant_series_have_undefined_ratios(self):
        # The long constant is one whose fmean rounds, which once left a spurious 1e-19 deviation.
        for returns in ([], [0.02], [0.02, 0.02], [0.0007695759147568863] * 341):
            with self.subTest(returns=returns):
                metrics = portfolio_metrics(returns, [0.0] * len(returns))
                strategy = metrics["strategy_net"]
                excess = metrics["excess_net_vs_benchmark"]
                self.assertIsNone(strategy["sharpe"])
                self.assertIsNone(strategy["sharpe_zero_cash_rate"])
                self.assertIsNone(excess["information_ratio"])
                if not returns:
                    self.assertIsNone(strategy["cagr"])
                    self.assertIsNone(strategy["max_drawdown"])
                json.dumps(metrics, allow_nan=False)

    def test_invalid_inputs_fail_closed(self):
        for invalid in (float("nan"), float("inf"), -1, -1.1, "0.1", True, None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                portfolio_metrics([invalid], [0.0])
        for rate in (float("nan"), float("inf"), -1, -2):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                portfolio_metrics([0.0], [0.0], annual_cash_rate=rate)
        for periods in (0, -1, 2.0, True):
            with self.subTest(periods=periods), self.assertRaises(ValueError):
                portfolio_metrics([0.0], [0.0], periods_per_year=periods)
        with self.assertRaisesRegex(ValueError, "matching lengths"):
            portfolio_metrics([0.0], [])
        with self.assertRaises(ValueError):
            portfolio_metrics([0.0], [float("nan")])

    def test_holdings_gate_strict_finite_and_minimum(self):
        metrics = {"strategy_net": {"observations": 252, "sharpe": 1.3}}
        self.assertFalse(holdings_gate(metrics))
        metrics["strategy_net"]["sharpe"] = 1.30001
        self.assertTrue(holdings_gate(metrics))
        metrics["strategy_net"]["observations"] = 251
        self.assertFalse(holdings_gate(metrics))
        metrics["strategy_net"]["observations"] = 252
        for value in (None, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                metrics["strategy_net"]["sharpe"] = value
                self.assertFalse(holdings_gate(metrics))
        self.assertFalse(holdings_gate({}))


if __name__ == "__main__":
    unittest.main()
