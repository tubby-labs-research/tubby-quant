"""Exact signed holding-preparation differentials against the frozen report code."""

import statistics
import time
import unittest
import json
from pathlib import Path
from unittest import mock

from qlib_yahoo.holding_series import signed_holding_series

try:
    import numpy as np
    import pandas as pd
except ImportError:
    np = pd = None

try:
    import plotly
except ImportError:
    plotly = None


def native_signed_holding_series(positions, frame):
    """Frozen series-preparation portion of analysis._holdings (2026-10-03)."""
    positive_equity = frame["account"] > 0

    def signed_weights(values):
        weights = pd.Series(float("nan"), index=frame.index, dtype=float)
        weights.loc[positive_equity] = values.loc[positive_equity] / frame.loc[positive_equity, "account"]
        return weights

    series = {}
    for instrument, group in positions.dropna(subset=["instrument"]).groupby("instrument"):
        values = group.set_index("datetime")["value"].reindex(frame.index, fill_value=0.0)
        series[str(instrument)] = signed_weights(values)
    return series


def native_rank_and_other(series, limit):
    """Frozen ranking/Other arithmetic; used unchanged for each paired result."""
    names = sorted(series)
    rest = []
    if len(names) > limit:
        ranked = sorted(names, key=lambda name: (-float(series[name].abs().mean(skipna=True) or 0.0), name))
        names, rest = sorted(ranked[:limit]), ranked[limit:]
    curves = {}
    if rest:
        other = pd.concat([series[name] for name in rest], axis=1)
        undefined = other.isna().all(axis=1)
        curves["longs"] = other.clip(lower=0).sum(axis=1).mask(undefined)
        curves["shorts"] = other.clip(upper=0).sum(axis=1).mask(undefined)
    return names, rest, curves


@unittest.skipIf(pd is None, "optional Pandas/NumPy dependencies required")
class SignedHoldingSeriesTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2021-01-04", periods=8, name="session")
        self.frame = pd.DataFrame({"account": np.array([100., 200., 0., -3., np.nan, 400., 50., 100.])},
                                  index=self.dates)

    def positions(self, rows):
        result = pd.DataFrame(rows, columns=["datetime", "instrument", "value"])
        result["datetime"] = pd.to_datetime(result["datetime"])
        result["instrument"] = result["instrument"].astype(object)
        result["value"] = result["value"].astype(np.float64)
        return result

    def assert_paired(self, positions, frame=None, limit=2):
        frame = self.frame if frame is None else frame
        before_positions, before_frame = positions.copy(deep=True), frame.copy(deep=True)
        expected = native_signed_holding_series(positions, frame)
        actual = signed_holding_series(positions, frame)
        self.assertIsNotNone(actual)
        self.assertEqual(list(actual), list(expected))
        for symbol in expected:
            pd.testing.assert_series_equal(actual[symbol], expected[symbol], check_exact=True)
            np.testing.assert_array_equal(actual[symbol].to_numpy().view(np.uint64),
                                          expected[symbol].to_numpy().view(np.uint64))
            self.assertTrue(actual[symbol].to_numpy().flags.c_contiguous)
            self.assertIs(actual[symbol].index, frame.index)
        native_names, native_rest, native_curves = native_rank_and_other(expected, limit)
        names, rest, curves = native_rank_and_other(actual, limit)
        self.assertEqual((names, rest), (native_names, native_rest))
        self.assertEqual(curves.keys(), native_curves.keys())
        for name in curves:
            pd.testing.assert_series_equal(curves[name], native_curves[name], check_exact=True)
            np.testing.assert_array_equal(curves[name].to_numpy().view(np.uint64),
                                          native_curves[name].to_numpy().view(np.uint64))
        pd.testing.assert_frame_equal(positions, before_positions, check_exact=True)
        pd.testing.assert_frame_equal(frame, before_frame, check_exact=True)
        return actual

    def test_sign_reversals_missing_bars_nan_marks_and_undefined_equity(self):
        rows = [(self.dates[date], symbol, value) for symbol, values in {
            "ZZZ": [(0, 12.), (1, -25.), (2, 9.), (5, np.nan), (6, 7.)],
            "AAA": [(0, -9.), (3, 10.), (4, 18.), (7, 29.)],
            "MMM": [(1, 15.), (5, -100.)],
            "TIE": [(1, 15.), (5, -100.)],
            "ALLNAN": [(0, np.nan), (1, np.nan), (5, np.nan), (6, np.nan), (7, np.nan)],
        }.items() for date, value in values]
        actual = self.assert_paired(self.positions(rows[::-1]))
        self.assertEqual(list(actual), ["AAA", "ALLNAN", "MMM", "TIE", "ZZZ"])
        self.assertEqual(actual["AAA"].iloc[1], 0.)
        self.assertTrue(pd.isna(actual["ZZZ"].iloc[5]))
        self.assertTrue(actual["AAA"].iloc[2:5].isna().all())

    def test_one_contiguous_matrix_with_independent_stock_rows(self):
        positions = self.positions([(self.dates[0], "BBB", 20.), (self.dates[1], "AAA", -10.)])
        actual = self.assert_paired(positions)
        first = actual["AAA"].to_numpy()
        second = actual["BBB"].to_numpy()
        self.assertIs(first.base, second.base)
        self.assertEqual(first.base.shape, (2, len(self.dates)))
        self.assertTrue(first.base.flags.c_contiguous)
        previous = second.copy()
        actual["AAA"].iloc[0] = 999.
        np.testing.assert_array_equal(second.view(np.uint64), previous.view(np.uint64))

    def test_null_and_empty_dummy_symbols(self):
        positions = self.positions([
            (self.dates[0], None, 0.), (self.dates[1], np.nan, 0.),
            (self.dates[5], "", 0.), (self.dates[6], "AAA", -5.),
            (pd.NaT, pd.NA, 0.),
        ])
        actual = self.assert_paired(positions)
        self.assertEqual(list(actual), ["", "AAA"])
        self.assert_paired(self.positions([(self.dates[0], None, 0.)]))

    def test_empty_positions_and_empty_ledger(self):
        positions = self.positions([])
        self.assertEqual(self.assert_paired(positions), {})
        frame = pd.DataFrame({"account": np.array([], dtype=np.float64)},
                             index=pd.DatetimeIndex([], name="datetime"))
        self.assertEqual(self.assert_paired(positions, frame), {})

    def test_arbitrary_position_and_ledger_date_order_and_index_name(self):
        positions = self.positions([(self.dates[date], symbol, value)
                                    for date, symbol, value in ((6, "ZZZ", -1.), (0, "AAA", 3.),
                                                                 (5, "AAA", -9.), (1, "ZZZ", 10.))])
        frame = self.frame.iloc[[6, 2, 0, 7, 1, 4, 5, 3]].copy()
        self.assert_paired(positions, frame)
        self.assert_paired(positions, frame.rename_axis(None))

    def test_float64_bytes_nan_payload_negative_zero_extremes_and_infinities(self):
        values = np.array([0x7FF8000000001234, 0x8000000000000000,
                           0x3FF0000000000001, 0x7FF0000000000000], dtype=np.uint64).view(np.float64)
        frame = self.frame.copy()
        frame["account"] = np.array([3., 7., 10., np.inf, np.nan, -np.inf, 1e-300, 1e300])
        positions = self.positions([(self.dates[i], "AAA", values[i]) for i in range(4)] +
                                   [(self.dates[6], "ZZZ", 1e-200), (self.dates[7], "ZZZ", -1e200)])
        self.assert_paired(positions, frame, limit=1)

    def test_all_undefined_equity_preserves_native_nan_ranking(self):
        frame = self.frame.copy()
        frame["account"] = np.array([0., -1., np.nan, 0., np.nan, -3., 0., -100.])
        positions = self.positions([(self.dates[0], symbol, 1.) for symbol in ("CCC", "AAA", "BBB", "DDD")])
        self.assert_paired(positions, frame, limit=1)

    def test_duplicate_stock_date_returns_none_without_aggregation(self):
        positions = self.positions([(self.dates[0], "AAA", 1.), (self.dates[0], "AAA", 2.)])
        self.assertIsNone(signed_holding_series(positions, self.frame))
        with self.assertRaises(ValueError):
            native_signed_holding_series(positions, self.frame)

    def test_unsupported_dates_and_exotic_layouts_return_none(self):
        base = self.positions([(self.dates[0], "AAA", 1.)])
        cases = []
        cases.append((base, self.frame.iloc[[0, 0, 1]]))
        frame = self.frame.copy()
        frame.index = frame.index.tz_localize("UTC")
        cases.append((base, frame))
        frame = self.frame.copy()
        frame.index = pd.Index(list(frame.index[:-1]) + [pd.NaT])
        cases.append((base, frame))
        for dtype in (np.float32, np.int64, object, "Float64"):
            positions = base.copy()
            positions["value"] = positions["value"].astype(dtype)
            cases.append((positions, self.frame))
            frame = self.frame.copy().fillna(0)
            frame["account"] = frame["account"].astype(dtype)
            cases.append((base, frame))
        for symbol in (1, True, b"AAA", ("AAA",), np.str_("AAA")):
            positions = base.copy()
            positions.loc[0, "instrument"] = symbol
            cases.append((positions, self.frame))
        for dtype in ("category", "string"):
            positions = base.copy()
            positions["instrument"] = positions["instrument"].astype(dtype)
            cases.append((positions, self.frame))
        for date in (pd.NaT, pd.Timestamp("2030-01-01")):
            cases.append((self.positions([(date, "AAA", 1.)]), self.frame))
        positions = base.copy()
        positions["datetime"] = positions["datetime"].astype(object)
        cases.append((positions, self.frame))
        positions = base.copy()
        positions["datetime"] = positions["datetime"].dt.tz_localize("UTC")
        cases.append((positions, self.frame))
        positions = base.copy()
        positions["datetime"] = positions["datetime"].astype("datetime64[us]")
        cases.append((positions, self.frame))
        cases.append((pd.concat([base, base[["value"]]], axis=1), self.frame))
        cases.append((base.drop(columns="value"), self.frame))
        cases.append((base, self.frame.drop(columns="account")))
        for number, (positions, frame) in enumerate(cases):
            with self.subTest(case=number):
                self.assertIsNone(signed_holding_series(positions, frame))

    def test_seeded_sparse_fixtures_preserve_rank_and_other_bits(self):
        rng = np.random.default_rng(717)
        for trial in range(16):
            dates = pd.date_range("2020-01-01", periods=31, name="datetime")
            account = rng.uniform(1., 1e8, len(dates))
            account[rng.choice(len(dates), 5, replace=False)] = [0., -1., np.nan, np.inf, -np.inf]
            frame = pd.DataFrame({"account": account}, index=dates)
            rows = []
            for symbol in ["", *[f"S{i:03}" for i in range(19)]]:
                selected = rng.choice(len(dates), 12, replace=False)
                values = rng.normal(size=len(selected)) * 1e7
                values[0] = np.nan
                rows.extend((dates[day], symbol, value) for day, value in zip(selected, values))
            rng.shuffle(rows)
            with self.subTest(trial=trial):
                self.assert_paired(self.positions(rows), frame, limit=7)

    def render_holdings(self, positions, frame, *, signed=True, force_native=False):
        """Exercise the real renderer while replacing only saved I/O boundaries."""
        from qlib_yahoo import analysis
        figures = []
        def chart(figure, directory, name):
            self.assertEqual(name, "holdings.html")
            figures.append(json.loads(figure.to_json()))
            return "holdings.html"
        provider = (mock.Mock(return_value=None) if force_native
                    else mock.Mock(wraps=signed_holding_series))
        with mock.patch.object(analysis, "table_artifact", return_value=Path("fixture-positions.parquet")), \
             mock.patch.object(analysis, "read_instrument_table", return_value=positions) as reader, \
             mock.patch.object(analysis, "_chart", side_effect=chart), \
             mock.patch.object(analysis, "HOLDINGS_TRACES", 2), \
             mock.patch("qlib_yahoo.holding_series.signed_holding_series", provider):
            result = analysis._holdings(Path("fixture-run"), frame, Path("fixture-report"), signed=signed)
        reader.assert_called_once_with(Path("fixture-positions.parquet"),
                                       columns=["datetime", "instrument", "weight"] + (["value"] if signed else []))
        self.assertEqual(provider.call_count, 1 if signed else 0)
        self.assertEqual(len(figures), 1)
        return result, figures[0]

    @unittest.skipIf(plotly is None, "optional Plotly required")
    def test_actual_signed_holdings_figure_json_matches_native_fallback(self):
        positions = self.positions([(self.dates[day], symbol, value)
                                    for symbol, observations in {
                                        "ZZZ": [(0, 20.), (1, -30.), (5, np.nan), (7, 1.)],
                                        "AAA": [(0, -40.), (5, 30.), (7, -20.)],
                                        "BBB": [(1, 6.), (6, -1.)],
                                        "CCC": [(1, 6.), (6, -1.)],
                                        "DDD": [(0, np.nan), (1, np.nan), (5, np.nan)],
                                        "": [(0, 0.)],
                                    }.items() for day, value in observations])
        positions["weight"] = 0.
        frame = self.frame.copy()
        frame["free_cash"] = np.array([20., -5., 10., np.nan, 10., np.inf, 0., -np.inf])
        frame["restricted_cash"] = np.array([0., 10., np.nan, 1., np.inf, 2., -3., 1.])
        native = self.render_holdings(positions, frame, force_native=True)
        candidate = self.render_holdings(positions, frame)
        self.assertEqual(candidate, native)
        names = [trace["name"] for trace in candidate[1]["data"]]
        self.assertTrue(any(name.startswith("Other longs") for name in names))
        self.assertTrue(any(name.startswith("Other shorts") for name in names))
        self.assertEqual(names[-2:], ["Free cash weight", "Restricted cash weight"])

    @unittest.skipIf(plotly is None, "optional Plotly required")
    def test_actual_holdings_exotic_layout_fallback_matches_original_figure(self):
        positions = self.positions([(self.dates[0], "AAA", 9.), (self.dates[6], "BBB", -4.)])
        positions["value"] = positions["value"].astype(np.float32)
        positions["weight"] = 0.
        frame = self.frame.assign(free_cash=10., restricted_cash=5.)
        self.assertIsNone(signed_holding_series(positions, frame))
        self.assertEqual(self.render_holdings(positions, frame),
                         self.render_holdings(positions, frame, force_native=True))

    @unittest.skipIf(plotly is None, "optional Plotly required")
    def test_actual_long_only_holdings_never_calls_signed_helper(self):
        positions = self.positions([(self.dates[0], symbol, value)
                                    for symbol, value in (("AAA", 1.), ("BBB", 3.), ("CCC", 2.))])
        positions["weight"] = [0.1, 0.3, 0.2]
        frame = self.frame.assign(cash=20.)
        # render_holdings verifies zero calls to the helper for each render.
        with np.errstate(divide="ignore", invalid="ignore"):
            candidate = self.render_holdings(positions, frame, signed=False)
            native = self.render_holdings(positions, frame, signed=False, force_native=True)
        self.assertEqual(candidate, native)
        names = [trace["name"] for trace in candidate[1]["data"]]
        self.assertEqual(names, ["BBB", "CCC", "Other holdings (1 names)", "Cash weight"])

    @unittest.skipIf(plotly is None, "optional Plotly required")
    def test_actual_duplicate_date_fallback_keeps_original_exception(self):
        from qlib_yahoo import analysis
        positions = self.positions([(self.dates[0], "AAA", 1.), (self.dates[0], "AAA", 2.)])
        positions["weight"] = 0.
        frame = self.frame.assign(free_cash=10., restricted_cash=5.)
        messages = []
        for force_native in (True, False):
            provider = (mock.Mock(return_value=None) if force_native
                        else mock.Mock(wraps=signed_holding_series))
            with mock.patch.object(analysis, "table_artifact", return_value=Path("fixture-positions.parquet")), \
                 mock.patch.object(analysis, "read_instrument_table", return_value=positions), \
                 mock.patch.object(analysis, "_chart") as chart, \
                 mock.patch("qlib_yahoo.holding_series.signed_holding_series", provider):
                with self.assertRaisesRegex(ValueError, "duplicate") as error:
                    analysis._holdings(Path("fixture-run"), frame, Path("fixture-report"), signed=True)
                messages.append(str(error.exception))
                provider.assert_called_once_with(positions, frame)
                chart.assert_not_called()
        self.assertEqual(messages[0], messages[1])


def synthetic_microbenchmark(symbol_count=1200, sessions=1255, marks_per_symbol=100, repeats=3):
    """Paired complete series-preparation boundary; excludes fixture construction."""
    if pd is None:
        raise RuntimeError("Pandas and NumPy are required")
    rng = np.random.default_rng(413)
    dates = pd.date_range("2020-01-01", periods=sessions, name="datetime")
    frame = pd.DataFrame({"account": rng.uniform(1e6, 2e6, sessions)}, index=dates)
    rows = []
    for number in range(symbol_count):
        selected = rng.choice(sessions, min(sessions, marks_per_symbol), replace=False)
        values = rng.normal(size=len(selected)) * 1e4
        rows.extend((dates[day], f"S{number:05}", value) for day, value in zip(selected, values))
    positions = pd.DataFrame(rows, columns=["datetime", "instrument", "value"])
    times = {"native": [], "matrix": []}
    for trial in range(repeats):
        pairs = [("native", native_signed_holding_series), ("matrix", signed_holding_series)]
        if trial % 2:
            pairs.reverse()
        outputs = {}
        for name, helper in pairs:
            started = time.perf_counter()
            outputs[name] = helper(positions, frame)
            times[name].append(time.perf_counter() - started)
        assert outputs["matrix"] is not None
        for symbol in outputs["native"]:
            pd.testing.assert_series_equal(outputs["matrix"][symbol], outputs["native"][symbol], check_exact=True)
    return {"symbols": symbol_count, "sessions": sessions, "position_rows": len(positions),
            "seconds": times,
            "median_seconds": {name: statistics.median(samples) for name, samples in times.items()},
            "exact_series_parity": True,
            "boundary": "sparse positions/account through complete dict of Series; excludes fixture; not a full report/backtest"}


if __name__ == "__main__":
    unittest.main()
