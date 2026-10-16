"""Actual Qlib engine smoke test on private, deterministic synthetic bars."""

import csv
import importlib.util
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qlib_yahoo.dataset import sha256, tree_sha256


def _snapshot(root: Path, sessions_count: int = 260, *, include_benchmark: bool = True) -> Path:
    import pandas as pd

    snapshot = root / "data" / "synthetic"
    bins = snapshot / "qlib_data"
    dates = pd.bdate_range("2024-01-02", periods=sessions_count)
    symbols = [f"TST{i}" for i in range(6)]
    benchmark = "TSTB"
    stored_symbols = [*symbols, benchmark] if include_benchmark else symbols
    for folder in (bins / "calendars", bins / "instruments", bins / "features"):
        folder.mkdir(parents=True)
    text = "".join(f"{day:%Y-%m-%d}\n" for day in dates)
    (bins / "calendars" / "day.txt").write_text(text, encoding="utf-8")
    (bins / "calendars" / "day_future.txt").write_text(text, encoding="utf-8")
    (bins / "instruments" / "all.txt").write_text(
        "".join(f"{code}\t{dates[0]:%Y-%m-%d}\t{dates[-1]:%Y-%m-%d}\n" for code in stored_symbols),
        encoding="utf-8",
    )
    for i, code in enumerate(stored_symbols):
        folder = bins / "features" / code.lower()
        folder.mkdir()
        close = [30 + 3 * i + 0.025 * day * (i + 1) + 0.1 * math.sin(day / 5 + i)
                 for day in range(sessions_count)]
        values = {
            "open": [price * (1 + 0.001 * math.sin(day + i)) for day, price in enumerate(close)],
            "close": close,
            "high": [price * 1.01 for price in close],
            "low": [price * 0.99 for price in close],
            "volume": [100000 + 1000 * i + day for day in range(sessions_count)],
            "factor": [1.0] * sessions_count,
        }
        for field, points in values.items():
            payload = (0.0, *points)
            (folder / f"{field}.day.bin").write_bytes(struct.pack(f"<{len(payload)}f", *payload))
    digest, _ = tree_sha256(bins)
    (snapshot / "manifest.json").write_text(json.dumps({
        "symbols": symbols, "benchmark": benchmark,
        "qlib_binary_tree_sha256": digest, "limitations": ["synthetic only"],
    }), encoding="utf-8")
    return snapshot


def _saved_table(run, stem):
    """A saved run table in either format, with dates as the YYYY-MM-DD strings the ledger uses."""
    from qlib_yahoo.analysis import read_instrument_table, table_artifact

    frame = read_instrument_table(table_artifact(run, stem))
    frame["datetime"] = frame["datetime"].dt.strftime("%Y-%m-%d")
    return frame


@unittest.skipUnless(importlib.util.find_spec("qlib"), "optional Qlib environment")
class ExecutionTests(unittest.TestCase):
    def test_actual_engine_ledger_and_replay(self):
        from qlib_yahoo.backtest import run_backtest
        from qlib_yahoo.analysis import render_report

        with tempfile.TemporaryDirectory() as temporary:
            snapshot = _snapshot(Path(temporary))
            run = run_backtest(snapshot, "$close/Ref($close, 1)-1", topk=2, n_drop=1,
                               warmup_sessions=20, report_level="summary")
            manifest = json.loads((run / "manifest.json").read_text())
            summary = json.loads((run / "summary.json").read_text())
            with (run / "daily_report.csv").open(newline="") as handle:
                daily = list(csv.DictReader(handle))
            import pandas as pd
            scores = _saved_table(run, "alpha_scores")
            positions = _saved_table(run, "positions")
            from qlib.data import D
            shorter = D.features(["TST0"], ["$close/Ref($close, 1)-1"],
                                 start_time="2024-01-02", end_time="2024-03-01",
                                 freq="day", disk_cache=0)
            full = D.features(["TST0"], ["$close/Ref($close, 1)-1"],
                              start_time="2024-01-02", end_time="2024-06-03",
                              freq="day", disk_cache=0)
            self.assertTrue(shorter.equals(full.loc[shorter.index]))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["schema_version"], 6)
            self.assertTrue((run / "positions.parquet").is_file() and not (run / "positions.csv.gz").exists())
            self.assertEqual(summary["metric_contract"]["version"], "daily-simple-strategy-net-v1")
            self.assertEqual(set(summary["metrics"]), {"strategy_net"})
            self.assertNotIn("benchmark_return", daily[0])
            self.assertEqual(manifest["backtest"]["execution_contract"]["risk_degree"], 0.95)
            self.assertEqual(manifest["backtest"]["execution_contract"]["n_drop"], 1)
            self.assertEqual(manifest["backtest"]["execution_contract"]["limit_threshold"], None)
            expected_signal_dates = {
                str(day.date()) for day in pd.bdate_range("2024-01-02", periods=260)[19:-2]
            }
            self.assertEqual(set(scores["datetime"]), expected_signal_dates)
            self.assertEqual(manifest["qlib_reviewed_source"]["commit"],
                             "da920b7f954f48ab1bb64117c976710de198373e")
            self.assertGreater(sum(float(row["cost_dollars"]) for row in daily), 0)
            self.assertGreater(sum(float(row["turnover"]) for row in daily), 0)
            first_day = pd.Timestamp(daily[0]["datetime"])
            prior_day = pd.bdate_range("2024-01-02", periods=260)[19]
            prior = scores.loc[scores["datetime"] == str(prior_day.date())]
            expected = set(prior.nlargest(2, "score")["instrument"])
            first_positions = positions.loc[positions["datetime"] == str(first_day.date())]
            self.assertEqual(set(first_positions["instrument"]), expected)
            self.assertNotIn(str(first_day.date()), set(prior["datetime"]))
            cost_basis = 0.0
            expected_cash = 100_000.0
            for row in first_positions.itertuples():
                i = int(row.instrument[3:])
                day = 20
                close = 30 + 3 * i + 0.025 * day * (i + 1) + 0.1 * math.sin(day / 5 + i)
                opening = struct.unpack("<f", struct.pack("<f", close * (1 + 0.001 * math.sin(day + i))))[0]
                expected_amount = math.floor((100_000.0 * 0.95 / 2) / opening + 0.1)
                self.assertEqual(float(row.amount), expected_amount)
                cost_basis += expected_amount * opening
                expected_cash -= expected_amount * opening * 1.0005
            self.assertAlmostEqual(float(daily[0]["cost_dollars"]), cost_basis * 0.0005, places=3)
            self.assertAlmostEqual(float(daily[0]["cash"]), expected_cash, places=3)
            dates = sorted(positions["datetime"].unique())
            holdings_by_day = {
                date: set(positions.loc[positions["datetime"] == date, "instrument"])
                for date in dates
            }
            replacements = []
            for previous_day, current_day in zip(dates, dates[1:]):
                sold = holdings_by_day[previous_day] - holdings_by_day[current_day]
                bought = holdings_by_day[current_day] - holdings_by_day[previous_day]
                self.assertLessEqual(len(sold), 1)
                self.assertEqual(len(sold), len(bought))
                if sold:
                    prior_signal_day = str(pd.bdate_range(end=current_day, periods=2)[0].date())
                    ranks = scores.loc[scores["datetime"] == prior_signal_day].set_index("instrument")["score"]
                    self.assertLessEqual(float(ranks[next(iter(sold))]), float(ranks[next(iter(bought))]))
                    replacements.append(current_day)
            self.assertTrue(replacements, "synthetic scores must exercise rotating sales for n_drop=1")
            previous = 100_000.0
            for row in daily:
                account = float(row["account"])
                self.assertAlmostEqual(account / previous - 1, float(row["net_return"]), places=8)
                self.assertAlmostEqual(account, float(row["cash"]) + float(row["holdings_value"]), places=5)
                previous = account
            for name, digest in manifest["artifacts_sha256"].items():
                self.assertEqual(sha256(run / name), digest)
            original = (run / "manifest.json").read_bytes()
            self.assertEqual(render_report(run, level="summary"), run / "summary.json")
            self.assertEqual((run / "manifest.json").read_bytes(), original)

    def test_zero_dropout_keeps_initial_holdings(self):
        from qlib_yahoo.backtest import run_backtest
        import pandas as pd

        with tempfile.TemporaryDirectory() as temporary:
            run = run_backtest(_snapshot(Path(temporary)), "$close/Ref($close, 1)-1",
                               topk=2, n_drop=0, warmup_sessions=20)
            manifest = json.loads((run / "manifest.json").read_text())
            self.assertEqual(manifest["backtest"]["execution_contract"]["method_sell"], "random")
            self.assertEqual(manifest["backtest"]["execution_contract"]["zero_dropout_policy"],
                             "no sales; vacant topk slots may be filled")
            positions = _saved_table(run, "positions")
            initial = set(positions.loc[positions["datetime"] == positions["datetime"].iloc[0], "instrument"])
            for _, frame in positions.groupby("datetime"):
                self.assertEqual(set(frame["instrument"]), initial)
            daily = pd.read_csv(run / "daily_report.csv")
            self.assertTrue((daily["turnover"].iloc[1:] == 0).all())
            self.assertTrue((daily["cost_dollars"].iloc[1:] == 0).all())

    def test_pre_engine_rejections(self):
        from qlib_yahoo import backtest as module

        for options in (
            {"report_level": "bogus"}, {"holdings": "bogus"},
            {"threshold": float("nan")}, {"threshold": float("inf")},
            {"min_observations": 0}, {"min_observations": 1.5},
        ):
            with self.subTest(options=options), patch.object(module, "validate_alpha") as validate:
                with self.assertRaises(module.DatasetError):
                    module.run_backtest(Path("unused"), "$close", **options)
                validate.assert_not_called()
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = _snapshot(Path(temporary))
            path = snapshot / "qlib_data" / "features" / "tst0" / "close.day.bin"
            with path.open("ab") as handle:
                handle.write(b"tamper")
            with patch("qlib.init") as qlib_init:
                with self.assertRaisesRegex(module.DatasetError, "binary tree differs"):
                    module.run_backtest(snapshot, "$close", warmup_sessions=20)
                qlib_init.assert_not_called()

    def test_label_failure_does_not_fail_simulation_or_detailed_replay(self):
        from qlib_yahoo.backtest import run_backtest
        from qlib_yahoo.analysis import render_report
        from qlib.data import D

        original = D.features

        def features(instruments, fields, *args, **kwargs):
            if fields == ["Ref($open, -2)/Ref($open, -1)-1"]:
                raise RuntimeError("diagnostic fixture fault")
            return original(instruments, fields, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary, patch.object(D, "features", side_effect=features):
            run = run_backtest(_snapshot(Path(temporary)), "$close", topk=2,
                               warmup_sessions=20, report_level="summary")
            manifest = json.loads((run / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["diagnostics"]["forward_labels"], "unavailable")
            self.assertIn("diagnostic fixture fault", manifest["diagnostics"]["reason"])
            self.assertNotIn("signal_forward_labels.parquet", manifest["artifacts_sha256"])
            self.assertFalse((run / "signal_forward_labels.parquet").exists())
            self.assertFalse((run / "signal_forward_labels.parquet.tmp").exists())
            index = render_report(run, level="detailed", holdings="never")
            self.assertTrue(index.is_file())
            self.assertIn("Signal diagnostics unavailable", index.read_text(encoding="utf-8"))

    def test_cli_help_describes_compact_default_and_warmup(self):
        from qlib_yahoo.__main__ import build_parser

        parser = build_parser()
        self.assertIn("compact net ledger and optional HTML", parser.description)
        backtest = next(action for action in parser._actions if action.dest == "command").choices["backtest"]
        start = next(action for action in backtest._actions if action.dest == "start")
        self.assertIn("--warmup-sessions", start.help)

    def test_failed_engine_seals_failure_and_report_failure_keeps_completion(self):
        from qlib_yahoo.backtest import run_backtest
        from qlib_yahoo import backtest as module

        with tempfile.TemporaryDirectory() as temporary:
            snapshot = _snapshot(Path(temporary))
            with patch.object(module, "_check_ledger", side_effect=RuntimeError("synthetic failure")):
                with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                    run_backtest(snapshot, "$close", topk=2, warmup_sessions=20)
            runs = sorted((snapshot.parent.parent / "runs").iterdir())
            failed = json.loads((runs[-1] / "manifest.json").read_text())
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["stage"], "simulation")
            with patch("qlib_yahoo.analysis.render_report", side_effect=RuntimeError("renderer unavailable")):
                completed_run = run_backtest(snapshot, "$close", topk=2, warmup_sessions=20)
            completed = json.loads((completed_run / "manifest.json").read_text())
            self.assertEqual(completed["status"], "completed")


if __name__ == "__main__":
    unittest.main()
