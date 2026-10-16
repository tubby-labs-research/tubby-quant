"""Report replay at universe scale: literal tickers such as NA, and bounded holdings charts."""

from __future__ import annotations

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("pandas") and importlib.util.find_spec("plotly"),
                     "pandas and plotly are optional outside the Qlib environment")
class ReportScaleTests(unittest.TestCase):
    @staticmethod
    def seal_parquet_positions(run):
        import json
        from qlib_yahoo import analysis

        manifest = json.loads((run / "manifest.json").read_text())
        manifest["schema_version"] = 6
        manifest["artifacts_sha256"].pop("positions.csv.gz", None)
        manifest["artifacts_sha256"]["positions.parquet"] = analysis._sha256(run / "positions.parquet")
        (run / "manifest.json").write_text(json.dumps(manifest))

    def test_latest_holdings_projected_batches_keep_only_final_session(self):
        import pandas as pd
        from qlib_yahoo import analysis

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            dates = pd.date_range("2024-01-02", periods=3, freq="B")
            original = pd.DataFrame({"datetime": dates.repeat(2), "instrument": ["NA", ""] * 3,
                                     "weight": [0.5, 0.0] * 3, "unused": ["large"] * 6})
            original.to_parquet(run / "positions.parquet", index=False, row_group_size=2)
            with patch.object(pd, "read_parquet", side_effect=AssertionError("full table read")):
                latest = analysis.latest_positions(run, dates[-1], batch_size=1)
        self.assertEqual(list(latest.columns), ["datetime", "instrument", "weight"])
        self.assertEqual(latest["datetime"].tolist(), [dates[-1]] * 2)
        self.assertEqual(latest["instrument"].iloc[0], "NA")
        self.assertTrue(pd.isna(latest["instrument"].iloc[1]))

    def test_signed_reconciliation_spans_batches_and_rejects_cross_batch_duplicates(self):
        import pandas as pd
        from qlib_yahoo import analysis
        from tests.test_qlib_strategy_only_reports import StrategyOnlyReports

        fixture = StrategyOnlyReports()
        fixture.setUp()
        try:
            fixture.bundle(mode="signed")
            run = fixture.root
            manifest, summary = analysis._bundle(run)
            frame = analysis._daily(run, signed=True, benchmark_enabled=False)
            positions = analysis.read_instrument_table(run / "positions.csv.gz")
            # Unordered legacy records stay supported; each row occupies its own batch.
            positions.iloc[::-1].to_parquet(run / "positions.parquet", index=False, row_group_size=1)
            self.seal_parquet_positions(run)
            original_batches = analysis.instrument_batches

            def tiny_batches(path, columns=None):
                yield from original_batches(path, columns, batch_size=1)

            with (patch.object(analysis, "instrument_batches", tiny_batches),
                  patch.object(pd, "read_parquet", side_effect=AssertionError("full positions allocation"))):
                self.assertEqual(analysis._reconcile_signed(run, frame, summary, benchmark_enabled=False), 100)
                pd.concat([positions, positions.iloc[[0]]]).to_parquet(run / "positions.parquet", index=False, row_group_size=1)
                with self.assertRaisesRegex(ValueError, "duplicate signed position"):
                    analysis._reconcile_signed(run, frame, summary, benchmark_enabled=False)
        finally:
            fixture.doCleanups()

    def test_declared_position_order_is_strict_across_batches(self):
        import pandas as pd
        from qlib_yahoo import analysis
        from tests.test_qlib_strategy_only_reports import StrategyOnlyReports

        fixture = StrategyOnlyReports()
        fixture.setUp()
        try:
            fixture.bundle(mode="signed")
            run = fixture.root
            _, summary = analysis._bundle(run)
            frame = analysis._daily(run, signed=True, benchmark_enabled=False)
            positions = analysis.read_instrument_table(run / "positions.csv.gz")
            original_batches = analysis.instrument_batches

            def tiny_batches(path, columns=None):
                yield from original_batches(path, columns, batch_size=1)

            with (patch.object(analysis, "instrument_batches", tiny_batches),
                  patch("sqlite3.connect", side_effect=AssertionError("ordered replay must not spool"))):
                positions.to_parquet(run / "positions.parquet", index=False, row_group_size=1)
                self.seal_parquet_positions(run)
                self.assertEqual(analysis._reconcile_signed(run, frame, summary, benchmark_enabled=False,
                                                           positions_order="datetime,instrument"), 100)
                for invalid in (positions.iloc[::-1], pd.concat([positions.iloc[[0]], positions])):
                    invalid.to_parquet(run / "positions.parquet", index=False, row_group_size=1)
                    with self.assertRaisesRegex(ValueError, "position order"):
                        analysis._reconcile_signed(run, frame, summary, benchmark_enabled=False,
                                                   positions_order="datetime,instrument")
                positions.to_parquet(run / "positions.parquet", index=False)
                for column in ("amount", "price", "value", "weight", "cash", "account", "restricted_cash"):
                    invalid = positions.copy()
                    invalid.loc[0, column] = float("inf")
                    invalid.to_parquet(run / "positions.parquet", index=False)
                    with self.assertRaises(ValueError):
                        analysis._reconcile_signed(run, frame, summary, benchmark_enabled=False,
                                                   positions_order="datetime,instrument")
        finally:
            fixture.doCleanups()

    def test_bundle_routes_ordered_signed_validation_and_keeps_unavailable_metrics(self):
        import json
        from qlib_yahoo import analysis
        from tests.test_qlib_strategy_only_reports import StrategyOnlyReports

        fixture = StrategyOnlyReports()
        fixture.setUp()
        try:
            for insolvent in (False, True):
                fixture.bundle(mode="signed", insolvent=insolvent)
                run = fixture.root
                positions = analysis.read_instrument_table(run / "positions.csv.gz")
                positions.to_parquet(run / "positions.parquet", index=False, row_group_size=1)
                manifest = json.loads((run / "manifest.json").read_text())
                manifest["schema_version"] = 6
                manifest["history_storage"] = {"positions_order": "datetime,instrument"}
                del manifest["artifacts_sha256"]["positions.csv.gz"]
                manifest["artifacts_sha256"]["positions.parquet"] = analysis._sha256(run / "positions.parquet")
                (run / "manifest.json").write_text(json.dumps(manifest))
                with patch.object(analysis, "_signed_position_totals", side_effect=AssertionError("legacy reducer used")):
                    bundle = analysis.validate_bundle(run)
                self.assertEqual(bundle.cash, 100)
                self.assertEqual(bundle.summary["metrics_status"]["strategy"], "unavailable" if insolvent else "available")
        finally:
            fixture.doCleanups()

    def test_unlisted_positions_cannot_shadow_sealed_legacy_table(self):
        from qlib_yahoo import analysis
        from tests.test_qlib_strategy_only_reports import StrategyOnlyReports

        fixture = StrategyOnlyReports()
        fixture.setUp()
        try:
            fixture.bundle(mode="signed")
            run = fixture.root
            shadow = analysis.read_instrument_table(run / "positions.csv.gz")
            shadow["instrument"] = "UNSEALED"
            shadow.to_parquet(run / "positions.parquet", index=False)
            bundle = analysis.validate_bundle(run)
            self.assertEqual(analysis.table_artifact(run, "positions").name, "positions.csv.gz")
            self.assertEqual(analysis.latest_positions(run, bundle.frame.index[-1])["instrument"].tolist(), ["SHORT"])
            captured = []
            with patch.object(analysis, "_chart", side_effect=lambda fig, directory, name: captured.append(fig) or name):
                analysis._holdings(run, bundle.frame, run, signed=True)
            self.assertIn("SHORT", [trace.name for trace in captured[0].data])
            self.assertNotIn("UNSEALED", [trace.name for trace in captured[0].data])
        finally:
            fixture.doCleanups()

    def test_unlisted_forward_labels_cannot_change_ic_or_score_groups(self):
        import pandas as pd
        from qlib_yahoo import analysis
        from tests.test_qlib_yahoo_reports import ReplayTests

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            fixture = ReplayTests()
            fixture.bundle(run, pairs=True)
            pairs = pd.DataFrame({"datetime": ["2024-01-02"] * 5, "instrument": list("ABCDE"),
                                  "score": [1., 2., 3., 4., 5.], "label": [0.1, 0.2, 0.3, 0.4, 0.5]})
            pairs.to_csv(run / "signal_forward_labels.csv.gz", index=False, compression="gzip")
            fixture.reseal(run)
            shadow = pairs.copy()
            shadow["label"] *= -1
            shadow.to_parquet(run / "signal_forward_labels.parquet", index=False)
            bundle = analysis.validate_bundle(run)
            captured = []
            with patch.object(analysis, "_chart", side_effect=lambda fig, directory, name: captured.append((name, fig)) or name):
                charts, _ = analysis._detailed(bundle.frame, run, run, 0, True)
            self.assertEqual({name for _, name in charts}, {"signal_ic.html", "score_groups.html"})
            self.assertAlmostEqual(dict(captured)["signal_ic.html"].data[0].y[0], 1)
            self.assertGreater(dict(captured)["score_groups.html"].data[0].y[0], 0)

    def test_duplicate_sealed_formats_are_rejected_for_positions_and_labels(self):
        import json
        from qlib_yahoo import analysis
        from tests.test_qlib_yahoo_reports import ReplayTests

        for stem in ("positions", "signal_forward_labels"):
            with tempfile.TemporaryDirectory() as temporary:
                run = Path(temporary)
                ReplayTests().bundle(run, pairs=True)
                analysis.read_instrument_table(run / f"{stem}.csv.gz").to_parquet(run / f"{stem}.parquet", index=False)
                manifest = json.loads((run / "manifest.json").read_text())
                manifest["artifacts_sha256"][f"{stem}.parquet"] = analysis._sha256(run / f"{stem}.parquet")
                (run / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "ambiguous duplicate table formats"):
                    analysis.validate_bundle(run)

    def test_holdings_projection_preserves_every_trace_and_value(self):
        import json
        from plotly.utils import PlotlyJSONEncoder
        from qlib_yahoo import analysis

        original = analysis.read_instrument_table
        for signed in (False, True):
            projected = self.holdings_figure(40, signed=signed)
            with patch.object(analysis, "read_instrument_table", side_effect=lambda path, columns=None: original(path)):
                full = self.holdings_figure(40, signed=signed)
            encode = lambda traces: json.dumps({name: trace.to_plotly_json() for name, trace in traces.items()},
                                              cls=PlotlyJSONEncoder, sort_keys=True)
            self.assertEqual(encode(projected), encode(full))

    def test_ticker_na_is_not_missing(self):
        import pandas as pd
        from qlib_yahoo.analysis import read_instrument_csv

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "positions.csv.gz"
            pd.DataFrame({"datetime": ["2024-01-02"] * 4, "instrument": ["NA", "", "NAN", "NULL"],
                          "value": [1.0, 0.0, None, 2.0]}).to_csv(path, index=False, compression="gzip")
            frame = read_instrument_csv(path)
            pd.DataFrame({"datetime": ["2024-01-02"] * 2, "instrument": ["TRUE", "FALSE"]}).to_csv(path, index=False, compression="gzip")
            booleans = read_instrument_csv(path)
        self.assertEqual(booleans["instrument"].tolist(), ["TRUE", "FALSE"])
        self.assertEqual(frame["instrument"].tolist()[0], "NA")
        self.assertTrue(pd.isna(frame["instrument"].tolist()[1]))
        self.assertEqual(frame["instrument"].tolist()[2:], ["NAN", "NULL"])
        self.assertTrue(math.isnan(frame["value"].tolist()[2]))

    def test_parquet_tables_keep_csv_missing_value_semantics(self):
        import pandas as pd
        from qlib_yahoo.analysis import read_instrument_table, table_artifact

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            self.assertEqual(table_artifact(run, "positions").name, "positions.csv.gz")  # legacy runs
            pd.DataFrame({"datetime": pd.to_datetime(["2024-01-02"] * 3), "instrument": ["NA", "", "TRUE"],
                          "value": [1.0, 0.0, 2.0]}).to_parquet(run / "positions.parquet", index=False)
            self.assertEqual(table_artifact(run, "positions").name, "positions.parquet")
            frame = read_instrument_table(table_artifact(run, "positions"))
        self.assertEqual(frame["instrument"].tolist()[0], "NA")
        self.assertTrue(pd.isna(frame["instrument"].tolist()[1]))  # a cash-only row, as in CSV
        self.assertEqual(frame["instrument"].tolist()[2], "TRUE")
        self.assertEqual(str(frame["datetime"].dtype), "datetime64[ns]")

    def holdings_figure(self, count, *, signed):
        import pandas as pd
        from qlib_yahoo import analysis

        days = pd.date_range("2024-01-02", periods=3, freq="B")
        rows = []
        for i in range(count):
            for day in days:
                weight = (i + 1) / 1000 * (-1 if signed and i % 2 else 1)
                rows.append({"datetime": day, "instrument": f"T{i:03d}", "weight": weight, "value": weight * 100.0})
        frame = pd.DataFrame({"account": [100.0, 100.0, 100.0], "cash": [10.0, 10.0, 10.0],
                              "free_cash": [10.0, 10.0, 10.0], "restricted_cash": [0.0, 0.0, 0.0]}, index=days)
        captured = []
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            pd.DataFrame(rows).to_csv(run / "positions.csv.gz", index=False, compression="gzip")
            with patch("qlib_yahoo.analysis._chart", side_effect=lambda fig, directory, name: captured.append(fig) or name):
                analysis._holdings(run, frame, run, signed=signed)
        return {trace.name: trace for trace in captured[0].data}

    def test_small_portfolios_keep_every_trace(self):
        from qlib_yahoo.analysis import HOLDINGS_TRACES

        traces = self.holdings_figure(HOLDINGS_TRACES, signed=False)
        self.assertEqual(len([n for n in traces if n.startswith("T")]), HOLDINGS_TRACES)
        self.assertFalse(any(n.startswith("Other") for n in traces))

    def test_large_long_portfolio_aggregates_the_rest_exactly(self):
        from qlib_yahoo.analysis import HOLDINGS_TRACES

        traces = self.holdings_figure(HOLDINGS_TRACES + 15, signed=False)
        shown = [n for n in traces if n.startswith("T")]
        self.assertEqual(len(shown), HOLDINGS_TRACES)
        self.assertIn(f"T{HOLDINGS_TRACES + 14:03d}", shown)  # largest weights are kept
        other = traces["Other holdings (15 names)"]
        self.assertAlmostEqual(other.y[0], sum((i + 1) / 1000 for i in range(15)))

    def test_large_signed_portfolio_splits_other_longs_and_shorts(self):
        from qlib_yahoo.analysis import HOLDINGS_TRACES

        traces = self.holdings_figure(HOLDINGS_TRACES + 10, signed=True)
        longs, shorts = traces["Other longs (10 names)"], traces["Other shorts (10 names)"]
        expected_long = sum((i + 1) / 1000 for i in range(10) if i % 2 == 0)
        expected_short = -sum((i + 1) / 1000 for i in range(10) if i % 2)
        self.assertAlmostEqual(longs.y[0], expected_long)
        self.assertAlmostEqual(shorts.y[0], expected_short)
        self.assertIn("Free cash weight", traces)


if __name__ == "__main__":
    unittest.main()
