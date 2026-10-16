"""Consolidated Stooq table import: parsing, screens, dated universe and causal universe backtests."""

from __future__ import annotations

import importlib.util
import json
import math
import re
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from qlib_yahoo import DEFAULT_DATA_ROOT
from qlib_yahoo.dataset import DatasetError

REAL_DUMPER = Path(DEFAULT_DATA_ROOT) / "upstream-qlib-v0.9.7" / "scripts" / "dump_bin.py"
HAVE_ARROW = bool(importlib.util.find_spec("pandas") and importlib.util.find_spec("pyarrow"))


def synthetic_table(sessions: int = 320):
    """Eight stocks with staggered histories plus rows every filter must catch."""
    import pandas as pd

    days = pd.bdate_range("2023-01-02", periods=sessions)
    rows = []
    stocks = {f"S{i}": i for i in range(8)}
    for ticker, i in stocks.items():
        first = 0 if i < 6 else 150  # two late listings
        for k, day in enumerate(days[first:], first):
            close = 20 + i * 3 + 0.05 * k + 2 * math.sin(k / (7 + i))
            volume = (1 + i) * 1e5 * (1 + 0.5 * math.sin(k / 11 + i)) + 1e4
            rows.append((ticker, "NYSE", "stock", day, close * 0.995, close * 1.01, close * 0.99, close, volume))
    for ticker, kind in (("ETF1", "etf"), ("PFD_A", "stock"), ("WRT-WS", "stock"), ("UNT-U", "stock"), ("ZVZZT", "stock")):
        for day in days:
            rows.append((ticker, "NYSE", kind, day, 10.0, 10.5, 9.5, 10.0, 1e9))
    frame = pd.DataFrame(rows, columns=("ticker", "exchange", "type", "date", "open", "high", "low", "close", "volume"))
    holiday = days[-1] + pd.Timedelta(days=3)  # a Monday "holiday" quoted by a single stock
    extra = frame[(frame["ticker"] == "S1") & (frame["date"] == days[-1])].assign(date=holiday)
    frame = pd.concat([frame, extra], ignore_index=True)
    frame.loc[(frame["ticker"] == "S2") & (frame["date"] == days[40]), "volume"] = 0.0
    frame.loc[(frame["ticker"] == "S3") & (frame["date"] == days[41]), "high"] = 1.0  # high below close
    frame.loc[(frame["ticker"] == "S4") & (frame["date"] == days[42]), "close"] = -1.0
    frame["date"] = frame["date"].dt.date
    return frame, days, holiday


ASCII_HEADER = "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>\n"
EMPTY_MEMBERS = ["data/daily/us/nasdaq stocks/1/emty.us.txt", "data/daily/us/nyse etfs/hdro.us.txt"]


def ascii_archive(frame, path, members=None):
    """Write a table in Stooq's daily US ASCII zip layout, with empty and header-only members."""
    import zipfile

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for (ticker, exchange, kind), rows in frame.groupby(["ticker", "exchange", "type"], sort=True):
            folder = f"data/daily/us/{exchange.lower()} {kind}s" + ("/1" if kind == "stock" else "")
            lines = [f"{ticker}.US,D,{day:%Y%m%d},000000,{float(o)!r},{float(h)!r},{float(l)!r},{float(c)!r},{float(v)!r},0"
                     for day, o, h, l, c, v in rows.sort_values("date")[["date", *PRICE_COLUMNS]].itertuples(index=False)]
            archive.writestr(f"{folder}/{ticker.lower()}.us.txt", ASCII_HEADER + "\n".join(lines) + "\n")
        archive.writestr(EMPTY_MEMBERS[0], "")
        archive.writestr(EMPTY_MEMBERS[1], ASCII_HEADER)
        archive.writestr("__MACOSX/data/._aaa.us.txt", "resource fork")
        for name, content in (members or {}).items():
            archive.writestr(name, content)


PRICE_COLUMNS = ("open", "high", "low", "close", "volume")


@unittest.skipUnless(HAVE_ARROW, "pandas and pyarrow are optional outside the Qlib environment")
class TableReadingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.frame, self.days, self.holiday = synthetic_table()
        self.parquet = self.root / "source" / "us_daily.parquet"
        self.csv = self.root / "source" / "us_daily.csv"
        self.parquet.parent.mkdir()
        self.frame.to_parquet(self.parquet, index=False)
        self.frame.to_csv(self.csv, index=False)

    def test_parquet_and_csv_read_identically(self):
        from qlib_yahoo.stooq_table import _securities, _stock_frame

        begin, stop = date(2023, 1, 1), date(2025, 1, 1)
        for_parquet = _securities(self.parquet, begin, stop)
        for_csv = _securities(self.csv, begin, stop)
        self.assertEqual(for_parquet[1], for_csv[1])
        self.assertEqual(sorted(map(tuple, for_parquet[0].values.tolist())), sorted(map(tuple, for_csv[0].values.tolist())))
        stocks = [f"S{i}" for i in range(8)]
        a, report_a, dropped_a = _stock_frame(self.parquet, begin, stop, stocks, 0.5)
        b, report_b, dropped_b = _stock_frame(self.csv, begin, stop, stocks, 0.5)
        self.assertTrue(a.reset_index(drop=True).equals(b.reset_index(drop=True)))
        self.assertEqual((report_a, dropped_a), (report_b, dropped_b))
        self.assertEqual([r["session"] for r in report_a], [f"{self.holiday:%Y-%m-%d}"])
        self.assertEqual(dropped_a, 1)
        codes = a["ticker"].cat.codes.to_numpy()
        self.assertTrue((codes[1:] >= codes[:-1]).all(), "rows are grouped by ticker in sorted order")

    def test_ascii_archive_reads_like_the_table(self):
        import pyarrow as pa
        from qlib_yahoo.stooq_table import COLUMNS, _batches, _securities

        archive = self.root / "source" / "d_us_txt.zip"
        ascii_archive(self.frame, archive)
        begin, stop = date(2023, 1, 1), date(2025, 1, 1)

        def rows(source):
            table = pa.Table.from_batches(list(_batches(source, list(COLUMNS), begin, stop)))
            return table.sort_by([("ticker", "ascending"), ("date", "ascending")]).to_pandas()

        from_table, from_archive = rows(self.parquet), rows(archive)
        self.assertEqual(len(from_archive), len(self.frame))
        self.assertTrue(from_table.equals(from_archive))
        empty = []
        self.assertEqual(_securities(archive, begin, stop, empty)[1], _securities(self.parquet, begin, stop)[1])
        self.assertEqual(empty, EMPTY_MEMBERS)

    def test_ascii_archive_layout_is_enforced(self):
        from qlib_yahoo.stooq_table import _securities

        row = ",D,20230103,000000,1,1,1,1,100,0\n"
        cases = {
            "unexpected Stooq archive member": {"data/daily/us/otc stocks/1/abc.us.txt": ASCII_HEADER + "ABC.US" + row},
            "every <TICKER> must be XYZ.US": {"data/daily/us/nyse stocks/1/xyz.us.txt": ASCII_HEADER + "ABC.US" + row},
            "every <PER> must be D": {"data/daily/us/nyse stocks/1/xyz.us.txt": ASCII_HEADER + "XYZ.US" + row.replace(",D,", ",W,")},
            "unexpected header": {"data/daily/us/nyse stocks/1/xyz.us.txt": "Date,Open\n2023-01-03,1\n"},
            "unparseable dates": {"data/daily/us/nyse stocks/1/xyz.us.txt": ASCII_HEADER + "XYZ.US" + row.replace("20230103", "20230230")},
        }
        for index, (message, members) in enumerate(cases.items()):
            archive = self.root / "source" / f"bad{index}.zip"
            ascii_archive(self.frame.head(0), archive, members)
            with self.subTest(message=message), self.assertRaisesRegex(DatasetError, re.escape(message)):
                _securities(archive, date(2023, 1, 1), date(2025, 1, 1))
        impossible = self.root / "source" / "impossible.csv"
        impossible.write_text("ticker,exchange,type,date,open,high,low,close,volume\nXYZ,NYSE,stock,2023-02-30,1,1,1,1,100\n")
        with self.assertRaisesRegex(DatasetError, "unparseable dates"):
            _securities(impossible, date(2023, 1, 1), date(2025, 1, 1))
        broken = self.root / "source" / "broken.zip"
        broken.write_bytes(b"not a zip archive")
        with self.assertRaisesRegex(DatasetError, "cannot read Stooq table"):
            _securities(broken, date(2023, 1, 1), date(2025, 1, 1))

    def test_csv_keeps_tickers_that_look_like_missing_values(self):
        from qlib_yahoo.stooq_table import _securities

        path = self.root / "source" / "na.csv"
        path.write_text("ticker,exchange,type,date,open,high,low,close,volume\n"
                        "NA,NYSE,stock,2023-01-03,1,1,1,1,100\nNULL,NYSE,stock,2023-01-03,1,1,1,1,100\n")
        pairs, rows = _securities(path, date(2023, 1, 1), date(2025, 1, 1))
        self.assertEqual((sorted(pairs["ticker"]), rows), (["NA", "NULL"], 2))
        path.write_text("ticker,exchange,type,date,open,high,low,close,volume\n,NYSE,stock,2023-01-03,1,1,1,1,100\n")
        with self.assertRaisesRegex(DatasetError, "empty ticker"):
            _securities(path, date(2023, 1, 1), date(2025, 1, 1))

    def test_opening_holiday_is_screened(self):
        import pandas as pd
        from qlib_yahoo.stooq_table import _sparse_sessions

        days = pd.bdate_range("2024-01-02", periods=30)
        counts = pd.Series([3] + [500] * 29, index=days)
        counts.iloc[12] = 4
        self.assertEqual(list(_sparse_sessions(counts, 0.5)), [days[0], days[12]])

    def test_interval_is_inclusive_exclusive(self):
        from qlib_yahoo.stooq_table import _securities

        _, rows = _securities(self.parquet, self.days[10].date(), self.days[11].date())
        # One session: six stocks listed so far plus the five non-common lines.
        self.assertEqual(rows, 11)
        self.assertEqual(rows, int((self.frame["date"] == self.days[10].date()).sum()))

    def test_invalid_bars_are_counted_by_reason(self):
        from qlib_yahoo.stooq_table import _invalid_bars, _stock_frame

        frame, _, _ = _stock_frame(self.parquet, date(2023, 1, 1), date(2025, 1, 1), [f"S{i}" for i in range(8)], 0.5)
        invalid, report = _invalid_bars(frame)
        self.assertEqual(report["inconsistent_ohlc"]["rows"], 1)
        self.assertEqual(report["nonpositive_price"]["rows"], 1)
        self.assertEqual(int(invalid.sum()), 2)
        self.assertIn("S3", report["inconsistent_ohlc"]["examples"][0])

    def test_output_inside_the_source_folder_is_refused(self):
        from qlib_yahoo.stooq_table import prepare_stooq_table

        with self.assertRaisesRegex(DatasetError, "overlaps read-only"):
            prepare_stooq_table(self.parquet, self.parquet.parent / "out", "2023-01-01", "2025-01-01")
        with self.assertRaisesRegex(DatasetError, "YYYY-MM-DD"):
            prepare_stooq_table(self.parquet, self.root / "out", "2023-1-1", "2025-01-01")

    def test_duplicates_and_type_changes_fail_explicitly(self):
        import pandas as pd
        from qlib_yahoo.stooq_table import prepare_stooq_table

        output = self.root / "out"
        duplicate = pd.concat([self.frame, self.frame.head(1)], ignore_index=True)
        path = self.root / "source" / "duplicate.parquet"
        duplicate.to_parquet(path, index=False)
        with patch("qlib_yahoo.stooq_table._dump_script", return_value=Path("unused")):
            with self.assertRaisesRegex(DatasetError, "repeats a ticker/date"):
                prepare_stooq_table(path, output, "2023-01-01", "2025-01-01")
        changed = self.frame.copy()
        changed.loc[changed.index[0], "type"] = "etf"
        path = self.root / "source" / "changed.parquet"
        changed.to_parquet(path, index=False)
        with patch("qlib_yahoo.stooq_table._dump_script", return_value=Path("unused")):
            with self.assertRaisesRegex(DatasetError, "changes security type"):
                prepare_stooq_table(path, output, "2023-01-01", "2025-01-01")
        failed = sorted((output / "datasets").iterdir())
        self.assertTrue(all((f / "failed_manifest.json").is_file() for f in failed))


@unittest.skipUnless(HAVE_ARROW and importlib.util.find_spec("qlib") and REAL_DUMPER.is_file(),
                     "Qlib environment and the pinned upstream dump_bin checkout are required")
class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from qlib_yahoo.stooq_table import prepare_stooq_table
        from qlib_yahoo.universe import UniverseRule

        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name).resolve()
        frame, cls.days, cls.holiday = synthetic_table()
        cls.table = cls.root / "source" / "us_daily.parquet"
        cls.table.parent.mkdir()
        frame.to_parquet(cls.table, index=False)
        before = cls.table.read_bytes()
        with patch("qlib_yahoo.stooq_table._dump_script", return_value=REAL_DUMPER):
            cls.snapshot = prepare_stooq_table(cls.table, cls.root / "data", "2023-01-01", "2025-01-01",
                                               rule=UniverseRule(name="top3", size=3, adv_sessions=10, min_history_sessions=40),
                                               make_latest=True, max_workers=2, progress=lambda _message: None)
        cls.source_unchanged = cls.table.read_bytes() == before
        cls.manifest = json.loads((cls.snapshot / "manifest.json").read_text())
        archive = cls.root / "source" / "d_us_txt.zip"
        ascii_archive(frame, archive)
        with patch("qlib_yahoo.stooq_table._dump_script", return_value=REAL_DUMPER):
            from_archive = prepare_stooq_table(archive, cls.root / "data", "2023-01-01", "2025-01-01",
                                               rule=UniverseRule(name="top3", size=3, adv_sessions=10, min_history_sessions=40),
                                               max_workers=2, progress=lambda _message: None)
        cls.archive_manifest = json.loads((from_archive / "manifest.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_snapshot_contract(self):
        m = self.manifest
        self.assertTrue(self.source_unchanged)
        self.assertEqual(m["status"], "ready")
        self.assertEqual(json.loads((self.root / "data" / "latest.json").read_text())["dataset"], str(self.snapshot))
        self.assertEqual(m["security_filter"]["common_stocks"], 8)
        self.assertEqual(set(m["security_filter"]["excluded"]),
                         {"not_stock_type", "preferred_or_series", "non_common_symbol", "warrant_unit_right_or_when_issued",
                          "exchange_test_symbol"})
        self.assertEqual([s["session"] for s in m["session_filter"]["dropped_sessions"]], [f"{self.holiday:%Y-%m-%d}"])
        self.assertEqual(m["invalid_bars"]["inconsistent_ohlc"]["rows"], 1)
        universe = m["universes"]["top3"]
        self.assertEqual(universe["size"], 3)
        self.assertTrue((self.snapshot / "qlib_data" / "instruments" / "top3.txt").is_file())
        self.assertNotIn("ZVZZT", (self.snapshot / "qlib_data" / "instruments" / "top3.txt").read_text())
        # The intermediate CSVs are gone after verification; their hashes remain as provenance.
        self.assertFalse((self.snapshot / "normalized").exists())
        self.assertEqual(m["normalized_csv"]["retained"], False)
        self.assertEqual(len(m["normalized_sha256"]), len(m["symbols"]))
        self.assertTrue((self.snapshot / "universe" / "top3_membership.csv.gz").is_file())
        self.assertNotIn(f"{self.holiday:%Y-%m-%d}", (self.snapshot / "qlib_data" / "calendars" / "day.txt").read_text())
        from qlib_yahoo.dataset_integrity import verify_dataset
        self.assertEqual(verify_dataset(self.snapshot, m)["qlib_binary_tree_sha256"], m["qlib_binary_tree_sha256"])

    def test_ascii_archive_builds_the_same_snapshot(self):
        table, archive = self.manifest, self.archive_manifest
        self.assertEqual((table["source_table"]["format"], archive["source_table"]["format"]),
                         ("parquet", "stooq-daily-us-ascii-zip"))
        self.assertEqual(archive["source_table"]["empty_members"], {"count": 2, "examples": EMPTY_MEMBERS})
        self.assertEqual(archive["source_table"]["rows_in_interval"], table["source_table"]["rows_in_interval"])
        for key in ("qlib_binary_tree_sha256", "qlib_calendar_sha256", "qlib_instruments_sha256", "symbols",
                    "security_filter", "session_filter", "invalid_bars"):
            self.assertEqual(table[key], archive[key], key)
        for key in ("instruments_sha256", "membership_sha256", "rankings_sha256"):
            self.assertEqual(table["universes"]["top3"][key], archive["universes"]["top3"][key], key)

    def test_masked_bars_are_nan_in_the_binaries(self):
        import numpy as np

        sessions = (self.snapshot / "qlib_data" / "calendars" / "day.txt").read_text().split()
        for symbol, day in (("s2", self.days[40]), ("s3", self.days[41]), ("s4", self.days[42])):
            payload = np.fromfile(self.snapshot / "qlib_data" / "features" / symbol / "close.day.bin", dtype="<f4")
            offset = sessions.index(f"{day:%Y-%m-%d}") - int(payload[0])
            self.assertTrue(math.isnan(payload[1 + offset]), symbol)

    def test_universe_backtest_only_buys_members(self):
        import pandas as pd
        from qlib_yahoo.backtest import run_backtest
        from qlib_yahoo.universe import read_instrument_file

        run = run_backtest(self.snapshot, "$close / Ref($close, 5) - 1", topk=2, n_drop=1, warmup_sessions=45,
                           universe="top3", report_level="summary", holdings="never")
        manifest = json.loads((run / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["backtest"]["universe"]["name"], "top3")
        spans = read_instrument_file(self.snapshot / "qlib_data" / "instruments" / "top3.txt")
        from qlib_yahoo.analysis import read_instrument_table, table_artifact

        scores = read_instrument_table(table_artifact(run, "alpha_scores"))
        for row in scores.itertuples():
            self.assertTrue(any(a <= row.datetime <= b for a, b in spans[row.instrument]), (row.datetime, row.instrument))
        positions = read_instrument_table(table_artifact(run, "positions"))
        sessions = pd.DatetimeIndex(pd.read_csv(self.snapshot / "qlib_data" / "calendars" / "day.txt", header=None)[0])
        held = positions.dropna(subset=["instrument"]).groupby("datetime")["instrument"].apply(set)
        previous = set()
        for day, names in held.items():
            signal_day = sessions[sessions.get_loc(day) - 1]
            for name in names - previous:  # every new purchase was a member on its signal session
                self.assertTrue(any(a <= signal_day <= b for a, b in spans[name]), (day, name))
            previous = names

    def test_universe_default_start_waits_for_membership(self):
        import pandas as pd
        from qlib_yahoo.backtest import run_backtest

        first_member = pd.Timestamp(self.manifest["universes"]["top3"]["first_effective"])
        sessions = pd.DatetimeIndex(self.days)
        earliest_start = sessions[sessions.get_loc(first_member) + 1]
        with self.assertRaisesRegex(DatasetError, f"earliest backtest start is {earliest_start:%Y-%m-%d}"):
            run_backtest(self.snapshot, "$close / Ref($close, 5) - 1", start=f"{self.days[20]:%Y-%m-%d}",
                         end=f"{self.days[80]:%Y-%m-%d}", topk=2, warmup_sessions=10, universe="top3")
        run = run_backtest(self.snapshot, "$close / Ref($close, 5) - 1", end=f"{self.days[80]:%Y-%m-%d}",
                           topk=2, warmup_sessions=10, universe="top3", report_level="summary", holdings="never")
        manifest = json.loads((run / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["backtest"]["start"], f"{earliest_start:%Y-%m-%d}")

    def test_alpha_history_must_fit_the_universe_rule(self):
        from qlib_yahoo.backtest import run_backtest

        with self.assertRaisesRegex(DatasetError, "history sessions"):
            run_backtest(self.snapshot, "$close / Ref($close, 60) - 1", warmup_sessions=70, universe="top3")
        with self.assertRaisesRegex(DatasetError, "no 'top9' universe"):
            run_backtest(self.snapshot, "$close / Ref($close, 5) - 1", warmup_sessions=45, universe="top9")


if __name__ == "__main__":
    unittest.main()
