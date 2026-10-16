"""Offline schema-4 signed-account replay fixtures; no provider or engine calls."""

import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qlib_yahoo.analysis import _chart, render_report
from qlib_yahoo.metrics import portfolio_metrics
from qlib_yahoo.report_service import ensure_report, validate_run_selection


@unittest.skipUnless(
    importlib.util.find_spec("pandas") and importlib.util.find_spec("plotly"),
    "pandas and plotly are optional outside the Qlib environment",
)
class SignedReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def bundle(self, *, unfunded=False, insolvent=False):
        import pandas as pd

        # Opening short proceeds are restricted; changing the second mark can
        # exhaust free cash or equity without a new fill.
        second_price = 90.0 if unfunded else 110.0
        if insolvent:
            second_price = 230.0
        dates = ["2024-01-02", "2024-01-03"]
        accounts = [99.95, 200.0 - second_price - 0.05]
        free = [99.95, -1.0 if unfunded else 99.95]
        if insolvent:
            free[1] = -0.05
        restricted = [100.0, 100.0]
        # For unfunded case, the second account reflects an accrued expense.
        accounts[1] = free[1] + restricted[1] - second_price
        returns = [accounts[0] / 100.0 - 1, accounts[1] / accounts[0] - 1 if accounts[1] > 0 else float("nan")]
        costs = [0.05, 0.0]
        borrows = [0.0, 100.95 if unfunded else (100.0 if insolvent else 0.0)]
        # Accounting fixtures intentionally place the funding loss in free cash;
        # costs retain separate, internally consistent recorded components.
        frame = pd.DataFrame({
            "datetime": dates, "account": accounts, "net_return": returns,
            "benchmark_return": [0.0, 0.0],
            "cost": [0.0005, float("nan") if insolvent else borrows[1] / accounts[0]],
            "cost_dollars": [0.05, borrows[1]], "turnover": [1.0, float("nan") if insolvent else 0.0],
            "cash": free, "holdings_value": [-100.0, -second_price],
            "free_cash": free, "cash_total": [free[i] + restricted[i] for i in range(2)],
            "restricted_cash": restricted, "long_value": [0.0, 0.0],
            "short_liability": [100.0, second_price],
            "transaction_cost_dollars": costs, "borrow_cost_dollars": borrows,
            "total_cost_dollars": [costs[i] + borrows[i] for i in range(2)],
            "unfunded_cash_shortfall": [max(-value, 0) for value in free],
            "funding_feasible": [value >= 0 for value in free],
            "metrics_defined": [True, accounts[1] > 0], "stale_marks": [False, False],
            "stale_mark_symbols": ["[]", "[]"],
        })
        frame.to_csv(self.root / "daily_report.csv", index=False)
        positions = pd.DataFrame({
            "datetime": dates, "instrument": ["SHORT", "SHORT"],
            "amount": [-1.0, -1.0], "price": [100.0, second_price],
            "value": [-100.0, -second_price],
            "weight": [-100 / accounts[0], -second_price / accounts[1] if accounts[1] > 0 else float("nan")],
            "cash": free, "account": accounts,
            "cash_total": [free[i] + restricted[i] for i in range(2)],
            "free_cash": free, "restricted_cash": restricted,
        })
        for column in ("long_value", "short_liability", "transaction_cost_dollars", "borrow_cost_dollars",
                       "total_cost_dollars", "unfunded_cash_shortfall", "funding_feasible", "metrics_defined",
                       "stale_marks", "stale_mark_symbols"):
            positions[column] = frame[column]
        positions.to_csv(self.root / "positions.csv.gz", index=False, compression="gzip")
        if insolvent:
            metrics = portfolio_metrics([0.0] * len(frame), [0.0] * len(frame))
            for family in ("strategy_net", "excess_net_vs_benchmark"):
                metrics[family] = {name: len(frame) if name == "observations" else None
                                   for name in metrics[family]}
        else:
            metrics = portfolio_metrics(returns, [0.0, 0.0])
        summary = {"metric_contract": {"version": "daily-simple-signed-net-v1", "periods_per_year": 252, "annual_cash_rate": 0.0},
                   "metrics": metrics, "backtest": {"start": dates[0], "end": dates[-1], "cash": 100.0},
                   "metrics_status": {"strategy": "unavailable" if insolvent else "available",
                                      "reason": "nonpositive_equity" if insolvent else None,
                                      "undefined_observations": 1 if insolvent else 0},
                   "funding_state": {"feasible_all_sessions": not (unfunded or insolvent),
                                     "unfunded_sessions": 1 if unfunded or insolvent else 0,
                                     "max_unfunded_cash_shortfall": max(-free[1], 0),
                                     "stale_mark_sessions": 0}}
        (self.root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        self.reseal()

    def reseal(self):
        files = ("summary.json", "daily_report.csv", "positions.csv.gz")
        (self.root / "manifest.json").write_text(json.dumps({
            "schema_version": 4, "status": "completed",
            "artifacts_sha256": {name: hashlib.sha256((self.root / name).read_bytes()).hexdigest() for name in files},
        }), encoding="utf-8")

    def test_healthy_render_and_reuse(self):
        self.bundle()
        self.assertEqual(validate_run_selection(self.root), self.root.resolve())
        index, reused = ensure_report(self.root, "overview")
        self.assertFalse(reused)
        self.assertIn("Restricted short proceeds", (index.parent / "account_components.html").read_text(encoding="utf-8"))
        self.assertIn("Gross position exposure", (index.parent / "exposure.html").read_text(encoding="utf-8"))
        self.assertIn("Stock borrow cost", (index.parent / "returns_cost.html").read_text(encoding="utf-8"))
        self.assertEqual(ensure_report(self.root, "overview"), (index, True))

    def test_unfunded_disables_auto_holdings(self):
        self.bundle(unfunded=True)
        index = render_report(self.root, level="overview", holdings="auto", min_observations=2)
        status = json.loads((index.parent / "report_manifest.json").read_text())
        self.assertFalse(status["holdings_gate_eligible"])
        self.assertFalse(status["holdings_rendered"])
        self.assertIn("1 unfunded sessions", index.read_text(encoding="utf-8"))
        manual = render_report(self.root, level="summary", holdings="always")
        self.assertIn("SHORT", (manual.parent / "holdings.html").read_text(encoding="utf-8"))

    def test_signed_holdings_show_actual_weights_and_insolvent_gaps(self):
        for insolvent in (False, True):
            with self.subTest(insolvent=insolvent):
                self.bundle(insolvent=insolvent)
                captured = []

                def capture(fig, directory, name):
                    if name == "holdings.html":
                        captured.append(fig)
                    return _chart(fig, directory, name)

                with patch("qlib_yahoo.analysis._chart", side_effect=capture):
                    render_report(self.root, level="summary", holdings="always")
                traces = {trace.name: trace for trace in captured[0].data}
                self.assertIn("weights", captured[0].layout.title.text)
                self.assertAlmostEqual(traces["SHORT"].y[0], -100 / 99.95)
                self.assertAlmostEqual(traces["Free cash weight"].y[0], 99.95 / 99.95)
                self.assertAlmostEqual(traces["Restricted cash weight"].y[0], 100 / 99.95)
                if insolvent:
                    for trace in traces.values():
                        self.assertTrue(math.isnan(trace.y[1]))
                else:
                    self.assertLess(traces["SHORT"].y[1], 0)
                    self.assertAlmostEqual(traces["SHORT"].y[1], -110 / 89.95)

    def test_insolvent_retains_full_period_and_undefined_return(self):
        self.bundle(insolvent=True)
        index = render_report(self.root, level="overview", holdings="auto")
        self.assertIn("1 sessions with undefined net returns", index.read_text(encoding="utf-8"))
        self.assertFalse(json.loads((index.parent / "report_manifest.json").read_text())["holdings_gate_eligible"])
        import pandas as pd

        daily = pd.read_csv(self.root / "daily_report.csv")
        self.assertTrue(pd.isna(daily.loc[1, "net_return"]))
        self.assertTrue(pd.isna(daily.loc[1, "cost"]))
        self.assertTrue(pd.isna(daily.loc[1, "turnover"]))
        self.assertEqual(daily.loc[1, "borrow_cost_dollars"], 100.0)
        self.assertEqual(daily.loc[1, "total_cost_dollars"], 100.0)

    def test_resealed_tamper_rejected_before_service_writes(self):
        self.bundle()
        import pandas as pd

        for column, value in (("restricted_cash", 101.0), ("total_cost_dollars", 1.0),
                              ("net_return", 0.2), ("cost", 0.1)):
            with self.subTest(column=column):
                self.bundle()
                daily = pd.read_csv(self.root / "daily_report.csv")
                daily.loc[1, column] = value
                daily.to_csv(self.root / "daily_report.csv", index=False)
                self.reseal()
                with self.assertRaisesRegex(ValueError, "reconcile"):
                    validate_run_selection(self.root)
        self.assertFalse((self.root / "reports").exists())

    def test_position_value_and_weight_tamper_rejected(self):
        import pandas as pd

        for column, value in (("value", -111.0), ("weight", 0.0), ("restricted_cash", 101.0)):
            with self.subTest(column=column):
                self.bundle()
                positions = pd.read_csv(self.root / "positions.csv.gz")
                positions.loc[1, column] = value
                positions.to_csv(self.root / "positions.csv.gz", index=False, compression="gzip")
                self.reseal()
                with self.assertRaisesRegex(ValueError, "reconcile"):
                    validate_run_selection(self.root)

    def test_cash_only_sentinel_amount_and_weight(self):
        import pandas as pd

        for column, value in (("amount", 999.0), ("value", 1.0), ("weight", 0.1)):
            with self.subTest(column=column):
                self.bundle()
                positions = pd.read_csv(self.root / "positions.csv.gz")
                sentinel = positions.iloc[[1]].copy()
                sentinel.loc[:, "instrument"] = ""
                sentinel.loc[:, "amount"] = 0.0
                sentinel.loc[:, "price"] = float("nan")
                sentinel.loc[:, "value"] = 0.0
                sentinel.loc[:, "weight"] = 0.0
                sentinel.loc[:, column] = value
                pd.concat([positions, sentinel], ignore_index=True).to_csv(
                    self.root / "positions.csv.gz", index=False, compression="gzip"
                )
                self.reseal()
                with self.assertRaisesRegex(ValueError, "cash-only position"):
                    validate_run_selection(self.root)


if __name__ == "__main__":
    unittest.main()
