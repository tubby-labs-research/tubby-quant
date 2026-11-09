"""Alpha Lab against an independent reference (tests/alphalab_reference.py) on a planted panel.

The conventions under test are numbered in docs/ALPHA_LAB_REFERENCE.md. The engine
runs through ``run_batch`` (the console's entry) for the headline configuration and
through ``simulate()`` for the cap and missing-close scenarios.
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

HAVE_PANDAS = bool(importlib.util.find_spec("pandas") and importlib.util.find_spec("numpy"))
HAVE_PLANTED = bool(importlib.util.find_spec("autoresearch.domains.alpha.planted"))

if HAVE_PANDAS:
    import numpy as np

    import alphalab_reference as ref

EXPRESSIONS = ("-rank(ts_delta(close, 5))", "rank(log(volume) - ts_mean(log(volume), 20))",
               "-rank(ts_std_dev(returns, 20))")
RAW_SIGNAL = "-ts_delta(close, 5)"
SYMBOLS, SESSIONS, SEED, WARMUP = 60, 400, 11, 252
METRIC_RTOL = 1e-9  # the task asked for 1e-6 (returns, turnover) and 1e-4 (Sharpe); observed <1e-15
PNL_ATOL = 1e-6  # dollars on a 20 M book
TURNOVER_ATOL = 1e-12


@unittest.skipUnless(HAVE_PANDAS, "numpy and pandas are optional outside the Qlib environment")
@unittest.skipUnless(HAVE_PLANTED, "the planted-panel generator ships with the research domains")
class AlphaLabReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from autoresearch.domains.alpha.planted import create_planted_dataset

        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.dataset = create_planted_dataset(cls.root / "datasets" / "planted", symbols=SYMBOLS, days=SESSIONS, seed=SEED)
        snapshot = ref.read_snapshot(cls.dataset)
        cls.dates, cls.symbols, cls.fields = snapshot["dates"], snapshot["symbols"], snapshot["fields"]
        cls.members = ref.membership_mask(cls.dataset, cls.dates, cls.symbols, SYMBOLS)
        cls.start = ref.test_window(cls.members, WARMUP)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def settings(self, truncation):
        from qlib_yahoo.alphalab.simulate import Settings

        return Settings(universe=f"TOP{SYMBOLS}", delay=1, decay=0, neutralization="MARKET", truncation=truncation,
                        pasteurization=True, cost_bps=0.0, warmup=WARMUP)

    def reference(self, expression, truncation, close=None):
        close = self.fields["close"] if close is None else close
        alpha = ref.alpha(expression, {"close": close, "volume": self.fields["volume"]}, self.members)
        return ref.simulate(alpha, close, self.members, start=self.start, truncation=truncation)

    def assert_agreement(self, engine_result, reference_result, label):
        summary = engine_result["summary"]
        for metric in ref.METRICS:
            engine_value, reference_value = summary[metric], reference_result["metrics"][metric]
            self.assertLessEqual(ref.relative_difference(engine_value, reference_value), METRIC_RTOL,
                                 msg=f"{label}: {metric} engine={engine_value!r} reference={reference_value!r}")
        self.assertEqual(summary["days"], reference_result["metrics"]["days"], label)
        np.testing.assert_allclose(engine_result["daily"]["pnl"], reference_result["pnl"], rtol=0, atol=PNL_ATOL,
                                   err_msg=f"{label}: daily PnL")
        np.testing.assert_allclose(engine_result["daily"]["turnover"], reference_result["traded"], rtol=0,
                                   atol=TURNOVER_ATOL, err_msg=f"{label}: daily turnover")
        for key in ("gross_exposure", "max_weight", "long_count", "short_count"):
            self.assertAlmostEqual(summary[key], reference_result[key], places=12, msg=f"{label}: {key}")
        if "weights" in engine_result:
            np.testing.assert_allclose(np.asarray(engine_result["weights"]), reference_result["weights"],
                                       rtol=0, atol=1e-14, err_msg=f"{label}: weights")

    def test_console_entry_matches_the_reference_at_truncation_0_2(self):
        from qlib_yahoo.alphalab.batch import run_batch
        from qlib_yahoo.alphalab.simulate import Workspace, simulate

        settings = self.settings(0.2)
        results = run_batch(self.dataset, list(EXPRESSIONS), settings, data_root=self.root, workers=1)
        workspace = Workspace(self.dataset)
        for expression, result in zip(EXPRESSIONS, results):
            self.assertNotIn("error", result, result.get("error"))
            # Convention 3: the test period is the warm-up row to the last session.
            self.assertEqual(result["summary"]["start"], str(self.dates[self.start]))
            self.assertEqual(result["summary"]["end"], str(self.dates[-1]))
            self.assertEqual(self.start, WARMUP)
            # run_batch (console) and simulate() are the same arithmetic.
            direct = simulate(expression, settings, workspace)
            self.assertEqual(direct["daily"]["pnl"], result["daily"]["pnl"], expression)
            self.assert_agreement(result, self.reference(expression, 0.2), f"{expression} cap 0.2")

    def test_alpha_values_are_bit_identical_to_the_reference_operators(self):
        from qlib_yahoo.alphalab.engine import evaluate
        from qlib_yahoo.alphalab.simulate import Workspace

        workspace = Workspace(self.dataset)
        context = workspace.context(0, len(self.dates) - 1, workspace.universe(f"TOP{SYMBOLS}"))
        for expression in (*EXPRESSIONS, RAW_SIGNAL):
            engine = evaluate(expression, context).astype(np.float64)
            reference = ref.alpha(expression, self.fields, self.members)
            np.testing.assert_array_equal(np.isfinite(engine), np.isfinite(reference), err_msg=expression)
            finite = np.isfinite(engine)
            np.testing.assert_array_equal(engine[finite], reference[finite], err_msg=expression)

    def test_truncation_off_and_binding_caps(self):
        from qlib_yahoo.alphalab.simulate import Workspace, simulate

        workspace = Workspace(self.dataset)
        inert = {}
        for cap in (0.2, 0.0, 0.02, 0.01):
            for expression in EXPRESSIONS:
                result = simulate(expression, self.settings(cap), workspace, keep_weights=True)
                self.assert_agreement(result, self.reference(expression, cap), f"{expression} cap {cap}")
                inert.setdefault(expression, {})[cap] = result["daily"]["pnl"]
                if cap in (0.2, 0.0):  # convention 14: a 60-name rank alpha never reaches 0.2
                    # 0.5 / sum|k/59 - 0.5| up to the float32 storage of the ranks (convention 5, ~4e-11 here)
                    self.assertAlmostEqual(result["summary"]["max_weight"], 0.5 / np.abs(np.arange(60) / 59 - 0.5).sum(), places=9)
                    self.assertAlmostEqual(result["summary"]["gross_exposure"], 1.0, places=12)
                elif cap == 0.02:  # binds names, not the side budget (30 x 0.02 = 0.6 > 0.5)
                    self.assertAlmostEqual(result["summary"]["max_weight"], 0.02, places=12)
                    self.assertAlmostEqual(result["summary"]["gross_exposure"], 1.0, places=12)
                else:  # convention 13: 30 x 0.01 = 0.3 a side, the book shrinks to gross 0.6
                    self.assertAlmostEqual(result["summary"]["max_weight"], 0.01, places=12)
                    self.assertAlmostEqual(result["summary"]["gross_exposure"], 0.6, places=12)
        for expression, by_cap in inert.items():
            self.assertEqual(by_cap[0.2], by_cap[0.0], f"{expression}: cap 0.2 should be inert for rank alphas")
            self.assertNotEqual(by_cap[0.2], by_cap[0.02], expression)
        # A raw signal has heavy-tailed demeaned weights, so 0.2 binds and the water-filling is exercised.
        for cap in (0.2, 0.0):
            result = simulate(RAW_SIGNAL, self.settings(cap), workspace, keep_weights=True)
            self.assert_agreement(result, self.reference(RAW_SIGNAL, cap), f"{RAW_SIGNAL} cap {cap}")
            if cap:
                self.assertAlmostEqual(result["summary"]["max_weight"], cap, places=12)
            else:
                self.assertGreater(result["summary"]["max_weight"], 0.2)

    def test_missing_closes_follow_the_holding_day_masks(self):
        from qlib_yahoo.alphalab.simulate import Workspace, simulate

        workspace = Workspace(self.dataset)
        close = np.array(workspace.fields["close"], copy=True)
        rng = np.random.default_rng(3)
        for row, column in zip(rng.integers(5, SESSIONS - 1, 25), rng.integers(0, SYMBOLS, 25)):
            close[int(row), int(column)] = np.nan
        close[300:304, 7] = np.nan  # a four-day gap inside the test period
        workspace.fields["close"] = close
        for expression in EXPRESSIONS:
            result = simulate(expression, self.settings(0.2), workspace, keep_weights=True)
            reference = self.reference(expression, 0.2, close=close)
            self.assert_agreement(result, reference, f"{expression} with missing closes")
            # Conventions 10 and 15: flat on the day the close is missing, and that exit is traded.
            weights = np.asarray(result["weights"])
            self.assertTrue((weights[300 - self.start:304 - self.start, 7] == 0).all(), expression)
            self.assertGreater(result["daily"]["turnover"][300 - self.start], 0.0)

    def test_metric_formulas_on_a_hand_series(self):
        pnl = np.array([100.0, -50.0, 200.0, 0.0, 50.0])
        traded = np.array([0.2, 0.4, 0.2, 0.2, 0.0])
        m = ref.metrics(pnl, traded, 1000.0)
        self.assertAlmostEqual(m["sharpe"], pnl.mean() / pnl.std(ddof=1) * 252 ** 0.5)
        self.assertAlmostEqual(m["returns"], pnl.mean() * 252 / 500)
        self.assertAlmostEqual(m["turnover"], 0.2)
        self.assertAlmostEqual(m["fitness"], m["sharpe"] * (abs(m["returns"]) / 0.2) ** 0.5)
        self.assertAlmostEqual(m["margin_bps"], 300.0 / 1000.0 * 1e4)
        self.assertAlmostEqual(m["drawdown"], 50.0 / 500)
        # Convention 22: the starting value 0 is a peak.
        self.assertAlmostEqual(ref.metrics(np.array([-30.0, 10.0]), np.ones(2), 100.0)["drawdown"], 30.0 / 50)

    def test_water_filling_closed_form(self):
        # 10/17 x 0.5 and then 5/7 x 0.3 both exceed the cap; the last two names share the remaining 0.1.
        allocation = ref.water_fill(np.array([10.0, 5.0, 1.0, 1.0]), 0.5, 0.2)
        self.assertAlmostEqual(allocation.sum(), 0.5)
        np.testing.assert_allclose(allocation, [0.2, 0.2, 0.05, 0.05])
        # One saturated name, the rest proportional: 8/10.5 x 0.5 > 0.2, then [1.5, 1] share 0.3 at 0.12 each unit.
        np.testing.assert_allclose(ref.water_fill(np.array([8.0, 1.5, 1.0]), 0.5, 0.2), [0.2, 0.18, 0.12])
        np.testing.assert_allclose(ref.water_fill(np.array([3.0, 1.0]), 0.5, 0.0), [0.375, 0.125])  # cap off
        np.testing.assert_allclose(ref.water_fill(np.array([3.0, 1.0]), 0.4, 0.2), [0.2, 0.2])  # budget = n x cap


if __name__ == "__main__":
    unittest.main()
