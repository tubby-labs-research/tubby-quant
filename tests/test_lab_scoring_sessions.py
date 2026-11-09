"""Observation-only Lab scoring on a tiny continuous synthetic price path."""
from dataclasses import asdict, replace
import math
import statistics
import unittest
from unittest.mock import Mock, patch

from qlib_yahoo.alphalab.settings import Settings
from qlib_yahoo.dataset import DatasetError

try:
    import numpy as np
    import qlib_yahoo.alphalab.simulate as simulation
except ImportError:
    np = simulation = None


class ScoringSessionSettingsTests(unittest.TestCase):
    def settings(self, sessions, **changes):
        return Settings(**{"start": "2020-12-01", "end": "2021-01-31",
                           "evaluation_sessions": sessions, **changes})

    def test_none_is_default_and_does_not_require_explicit_bounds(self):
        self.assertIsNone(Settings().validate().evaluation_sessions)
        self.assertEqual(Settings().validate(), Settings(evaluation_sessions=None).validate())

    def test_validation_copies_sessions_and_retains_exact_dates(self):
        dates = ["2020-12-01", "2021-01-31"]
        validated = self.settings(dates).validate()
        self.assertEqual(validated.evaluation_sessions, dates)
        self.assertIsNot(validated.evaluation_sessions, dates)
        dates[0] = "2020-12-02"
        self.assertEqual(validated.evaluation_sessions, ["2020-12-01", "2021-01-31"])
        again = validated.validate()
        self.assertIsNot(again.evaluation_sessions, validated.evaluation_sessions)
        self.assertEqual(asdict(again)["evaluation_sessions"], validated.evaluation_sessions)

    def test_invalid_type_cardinality_order_and_dates_are_rejected(self):
        for sessions in ([], ["2020-12-01"], (), ("2020-12-01", "2021-01-01"),
                         "2020-12-01", {"2020-12-01": True}, True,
                         ["2020-12-01", "2020-12-01"], ["2021-01-01", "2020-12-01"],
                         ["2020-12-01", "2021-02-30"], ["20201201", "2021-01-01"],
                         ["2020-12-1", "2021-01-01"], ["2020-W49-2", "2021-01-01"],
                         ["2020-12-01T00:00:00", "2021-01-01"],
                         [" 2020-12-01", "2021-01-01"], [None, "2021-01-01"],
                         [20201201, "2021-01-01"], [False, "2021-01-01"],
                         [["2020-12-01"], "2021-01-01"]):
            with self.subTest(sessions=sessions), self.assertRaisesRegex(DatasetError, "evaluation_sessions"):
                self.settings(sessions).validate()

    def test_explicit_iso_bounds_are_required_and_enforced(self):
        for changes in ({"start": None}, {"end": None}, {"start": "20201201"},
                        {"end": "2021-01-32"}, {"start": "2021-01-02", "end": "2020-12-30"},
                        {"start": "2020-12-02"}, {"end": "2020-12-31"}):
            with self.subTest(changes=changes), self.assertRaisesRegex(DatasetError, "evaluation_sessions"):
                self.settings(["2020-12-01", "2021-01-01"], **changes).validate()


def oracle_stats(pnl, turnover, booksize=1000.):
    """Scalar reference, independent of the simulator's reductions/stat helper."""
    if len(pnl) < 2:
        return {"days": len(pnl)}
    mean = statistics.mean(pnl)
    std = statistics.stdev(pnl)
    sharpe = mean / std * math.sqrt(252) if std else None
    annual_return = mean * 252 / (booksize / 2)
    mean_turnover = statistics.mean(turnover)
    cumulative = peak = drawdown = 0.
    for value in pnl:
        cumulative += value
        peak = max(peak, cumulative)
        drawdown = max(drawdown, peak - cumulative)
    return {"days": len(pnl), "pnl": sum(pnl), "sharpe": sharpe,
            "returns": annual_return, "turnover": mean_turnover,
            "fitness": sharpe * math.sqrt(abs(annual_return) / max(mean_turnover, .125))
            if sharpe is not None else None,
            "margin_bps": sum(pnl) / (sum(turnover) * booksize) * 10000 if sum(turnover) else None,
            "drawdown": drawdown / (booksize / 2)}


@unittest.skipUnless(simulation is not None, "NumPy/Pandas are required for Lab simulation")
class ScoringSessionSimulationTests(unittest.TestCase):
    def setUp(self):
        # Thirty synthetic weekday sessions spanning two years, with six warmup
        # rows and 24 ordinary scored rows. No market data or filesystem writes.
        ws = simulation.Workspace.__new__(simulation.Workspace)
        days = np.arange("2020-12-01", "2021-01-12", dtype="datetime64[D]")
        ws.dates = days[np.is_busday(days)]
        self.assertEqual(len(ws.dates), 30)
        ws.symbols = ["A", "B"]
        ws.fields = {
            "close": np.asarray([[8, 8], [16, 8], [8, 16], [8, 8]] * 8, dtype=float)[:30],
            "volume": np.asarray([[3, 1], [1, 3], [3, 1], [2, 2]] * 8, dtype=float)[:30],
        }
        ws.identity = {"fixture": "continuous-scoring-sessions"}
        ws.groups = {}
        ws._universes = {"TOP2": np.ones((30, 2), dtype=bool)}
        self.ws = ws
        self.settings = Settings(universe="TOP2", start=str(ws.dates[6]), end=str(ws.dates[-1]),
                                 warmup=6, delay=0, decay=0, neutralization="NONE", truncation=0,
                                 booksize=1000., cost_bps=100., oos_start="2021-01-01")
        self.expression = "volume - 2"
        self.indices = [6, 8, 13, 15, 24, 29]
        self.selected = [str(ws.dates[i]) for i in self.indices]

    def run_simulation(self, selected=None, **kwargs):
        settings = replace(self.settings, evaluation_sessions=selected)
        return simulation.simulate(self.expression, settings, self.ws, **kwargs)

    def assert_stats(self, actual, pnl, turnover):
        for key, expected in oracle_stats(pnl, turnover).items():
            with self.subTest(metric=key):
                if expected is None:
                    self.assertIsNone(actual[key])
                else:
                    self.assertAlmostEqual(actual[key], expected, places=11)

    def test_literal_filtered_pnl_fees_turnover_weights_and_stats(self):
        full = self.run_simulation(keep_weights=True)
        selected = self.run_simulation(self.selected, keep_weights=True)
        # Four-day cycle: holdings +/-[.5,-.5] or zero, gross PnL
        # [0,500,750,250], turnover [1,2,2,1], fees [10,20,20,10].
        pnl = [730., -10., 480., 240., -10., 480.]
        turnover = [2., 1., 2., 1., 1., 2.]
        self.assertEqual(selected["daily"], {"dates": self.selected, "pnl": pnl, "turnover": turnover})
        offsets = [i - 6 for i in self.indices]
        self.assertEqual(pnl, [full["daily"]["pnl"][i] for i in offsets])
        self.assertEqual(turnover, [full["daily"]["turnover"][i] for i in offsets])
        np.testing.assert_array_equal(selected["weights"],
                                      [[.5, -.5], [.5, -.5], [-.5, .5], [0., 0.], [.5, -.5], [-.5, .5]])
        np.testing.assert_array_equal(selected["weights"], full["weights"][offsets])
        self.assertEqual(selected["symbols"], ["A", "B"])
        self.assert_stats(selected["summary"], pnl, turnover)
        self.assertEqual(selected["summary"]["start"], self.selected[0])
        self.assertEqual(selected["summary"]["end"], self.selected[-1])
        self.assertAlmostEqual(selected["summary"]["long_count"], 5 / 6)
        self.assertAlmostEqual(selected["summary"]["short_count"], 5 / 6)
        self.assertAlmostEqual(selected["summary"]["gross_exposure"], 5 / 6)
        self.assertEqual(selected["summary"]["net_exposure"], 0.)
        self.assertEqual(selected["summary"]["max_weight"], .5)
        self.assertAlmostEqual(selected["summary"]["coverage"], 5 / 6)

    def test_gaps_keep_intervening_trades_and_selected_day_fees(self):
        selected = self.run_simulation(self.selected, keep_weights=True)
        # Day 6 and day 8 have identical end-of-day weights, but day 7 closes
        # them. Day 8's reopening fee must survive the observation filter.
        np.testing.assert_array_equal(selected["weights"][0], selected["weights"][1])
        self.assertEqual(selected["daily"]["turnover"][1], 1.)
        self.assertEqual(selected["daily"]["pnl"][1], -10.)
        free = simulation.simulate(self.expression, replace(self.settings, cost_bps=0,
                                   evaluation_sessions=self.selected), self.ws)
        self.assertEqual([a - b for a, b in zip(free["daily"]["pnl"], selected["daily"]["pnl"])],
                         [20., 10., 20., 10., 10., 20.])

    def test_expression_delay_decay_and_warmup_run_over_continuous_path(self):
        self.expression = "ts_delta(volume, 2)"
        self.settings = replace(self.settings, delay=1, decay=2)
        full = self.run_simulation(keep_weights=True)
        with patch.object(simulation, "evaluate", wraps=simulation.evaluate) as evaluate:
            selected = self.run_simulation(self.selected, keep_weights=True)
        context = evaluate.call_args.args[1]
        np.testing.assert_array_equal(context.dates, self.ws.dates)
        np.testing.assert_array_equal(context.fields["volume"], self.ws.fields["volume"])
        # After the two-session delta, delay and two-session decay, the sign is
        # positive on cycle days 0/1 and negative on 2/3: +/- .5 per name.
        self.assertEqual(full["daily"]["pnl"], [-770., -250., -20., 500.] * 6)
        self.assertEqual(full["daily"]["turnover"], [2., 0., 2., 0.] * 6)
        pnl, turnover = [-770., -20., 500., -250., -20., 500.], [2., 2., 0., 0., 2., 0.]
        self.assertEqual(selected["daily"]["pnl"], pnl)
        self.assertEqual(selected["daily"]["turnover"], turnover)
        np.testing.assert_array_equal(selected["weights"], full["weights"][[i - 6 for i in self.indices]])
        self.assert_stats(selected["summary"], pnl, turnover)

    def test_yearly_and_oos_statistics_use_only_retained_rows(self):
        selected = self.run_simulation(self.selected)
        self.assertEqual([row["year"] for row in selected["yearly"]], [2020, 2021])
        self.assert_stats(selected["yearly"][0], [730., -10., 480., 240.], [2., 1., 2., 1.])
        self.assert_stats(selected["yearly"][1], [-10., 480.], [1., 2.])
        self.assertAlmostEqual(selected["yearly"][0]["long_count"], .75)
        self.assertEqual(selected["yearly"][1]["short_count"], 1.)
        self.assert_stats(selected["periods"]["in_sample"], [730., -10., 480., 240.], [2., 1., 2., 1.])
        self.assert_stats(selected["periods"]["out_of_sample"], [-10., 480.], [1., 2.])

    def test_two_rows_single_row_years_and_empty_oos_remain_explicit(self):
        both_years = [self.selected[0], self.selected[-1]]
        selected = self.run_simulation(both_years)
        self.assertEqual([row["days"] for row in selected["yearly"]], [1, 1])
        self.assertNotIn("sharpe", selected["yearly"][0])
        self.assertEqual(selected["periods"], {"in_sample": {"days": 1}, "out_of_sample": {"days": 1}})
        selected = self.run_simulation(self.selected[:2])
        self.assertEqual([row["year"] for row in selected["yearly"]], [2020])
        self.assertEqual(selected["periods"]["out_of_sample"], {"days": 0})

    def test_checks_use_selected_statistics(self):
        self.expression = "ts_delta(volume, 2)"
        self.settings = replace(self.settings, delay=1, decay=2)
        selected = self.run_simulation([str(self.ws.dates[i]) for i in (9, 13, 17)])
        checks = {check["name"]: check for check in selected["checks"]}
        self.assertEqual(checks["LOW_TURNOVER"]["value"], 0.)
        self.assertEqual(checks["LOW_TURNOVER"]["result"], "FAIL")
        self.assertEqual(checks["HIGH_TURNOVER"]["result"], "PASS")
        self.assertEqual(checks["LOW_SHARPE"]["result"], "ERROR")
        self.assertIsNone(checks["LOW_FITNESS"]["value"])
        full_checks = {check["name"]: check for check in self.run_simulation()["checks"]}
        self.assertEqual(full_checks["LOW_TURNOVER"]["result"], "PASS")
        self.assertEqual(full_checks["HIGH_TURNOVER"]["result"], "FAIL")

    def test_selected_weight_summary_uses_aligned_membership(self):
        self.ws._universes["TOP2"][13, 1] = False
        selected = self.run_simulation(self.selected, keep_weights=True)
        np.testing.assert_array_equal(selected["weights"][2], [-1., 0.])
        summary = selected["summary"]
        self.assertEqual(summary["max_weight"], 1.)
        self.assertAlmostEqual(summary["net_exposure"], -1 / 6)
        self.assertAlmostEqual(summary["long_count"], 4 / 6)
        self.assertAlmostEqual(summary["short_count"], 5 / 6)
        self.assertAlmostEqual(summary["coverage"], 5 / 6)

    def test_metadata_pins_observed_calendar_and_conditional_meaning(self):
        selected = self.run_simulation(self.selected)
        evaluation = selected["evaluation"]
        self.assertEqual({k: evaluation[k] for k in
                          ("policy", "observed_start", "observed_end", "observed_count",
                           "selected_sessions", "selected_count")},
                         {"policy": "observation-filter-v1", "observed_start": str(self.ws.dates[6]),
                          "observed_end": str(self.ws.dates[-1]), "observed_count": 24,
                          "selected_sessions": self.selected, "selected_count": 6})
        self.assertIn("Conditional selected-session", evaluation["metrics_meaning"])
        self.assertIn("252", evaluation["metrics_meaning"])
        self.assertIn("compressed selected PnL", evaluation["metrics_meaning"])
        self.assertIn("not continuous investment return", evaluation["metrics_meaning"])
        self.assertIn("full unfiltered observed window", evaluation["diagnostics_scope"])

    def test_drawdown_is_compressed_selected_pnl_and_annualization_is_252(self):
        selected = self.run_simulation([str(self.ws.dates[i]) for i in (8, 12, 16)])
        self.assertEqual(selected["daily"]["pnl"], [-10., -10., -10.])
        self.assertEqual(selected["summary"]["drawdown"], 30 / 500)
        self.assertEqual(self.run_simulation()["summary"]["drawdown"], 10 / 500)
        self.assertEqual(selected["summary"]["returns"], -10 * 252 / 500)

    def test_diagnostics_keep_full_path_scope_even_for_omitted_missing_groups(self):
        self.settings = replace(self.settings, neutralization="INDUSTRY")
        self.ws.groups["industry"] = np.zeros((30, 2))
        self.ws.groups["industry"][10] = np.nan  # omitted session with missing groups
        full, selected = self.run_simulation(), self.run_simulation(self.selected)
        for key in ("classification_coverage", "inactive_neutralized_days", "max_abs_group_net_exposure"):
            self.assertEqual(selected["summary"][key], full["summary"][key])
        self.assertEqual(selected["summary"]["classification_coverage"], 46 / 48)
        self.assertEqual(selected["summary"]["inactive_neutralized_days"], 7)
        self.assertIn("not recomputed on selected sessions", selected["evaluation"]["diagnostics_scope"])

    def test_absent_calendar_date_is_refused_not_rounded(self):
        with self.assertRaisesRegex(DatasetError, "absent from observed calendar: 2020-12-12"):
            self.run_simulation(["2020-12-11", "2020-12-12"])

    def test_decision_targets_refused_before_workspace_or_expression_access(self):
        workspace = Mock()
        with patch.object(simulation, "evaluate") as evaluate:
            with self.assertRaisesRegex(DatasetError, "cannot produce decision targets"):
                simulation.simulate("unused", replace(self.settings, evaluation_sessions=self.selected),
                                    workspace, keep_weights=True, keep_decision_targets=True)
        self.assertEqual(workspace.method_calls, [])
        evaluate.assert_not_called()

    def test_none_preserves_full_numeric_output_and_decision_targets(self):
        default = simulation.simulate(self.expression, self.settings, self.ws,
                                      keep_weights=True, keep_decision_targets=True)
        explicit = self.run_simulation(None, keep_weights=True, keep_decision_targets=True)
        self.assertEqual(simulation.to_json(default), simulation.to_json(explicit))
        self.assertNotIn("evaluation", explicit)
        self.assertNotIn("metrics_meaning", explicit["summary"])
        # Literal full-path baseline, including dates, costs and preceding targets.
        self.assertEqual(explicit["daily"], {"dates": [str(d) for d in self.ws.dates[6:]],
                         "pnl": [730., 240., -10., 480.] * 6, "turnover": [2., 1., 1., 2.] * 6})
        self.assert_stats(explicit["summary"], [730., 240., -10., 480.] * 6, [2., 1., 1., 2.] * 6)
        np.testing.assert_array_equal(explicit["weights"], [[.5, -.5], [0., 0.], [.5, -.5], [-.5, .5]] * 6)
        self.assertEqual(explicit["decision_targets"]["dates"], [str(d) for d in self.ws.dates[5:-1]])
        np.testing.assert_array_equal(explicit["decision_targets"]["weights"],
                                      [[-.5, .5], [.5, -.5], [0., 0.], [.5, -.5]] * 6)

    def test_selecting_every_observed_session_preserves_all_numbers(self):
        full = self.run_simulation(keep_weights=True)
        selected = self.run_simulation(full["daily"]["dates"], keep_weights=True)
        for key in ("summary", "yearly", "periods", "checks", "daily"):
            self.assertEqual(selected[key], full[key])
        np.testing.assert_array_equal(selected["weights"], full["weights"])


if __name__ == "__main__":
    unittest.main()
