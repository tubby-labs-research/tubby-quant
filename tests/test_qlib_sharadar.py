"""Sharadar tables (delisted stocks included) -> consolidated table -> snapshot, and the bulk download."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path
from unittest.mock import patch

from qlib_yahoo.dataset import DatasetError

HAVE_ARROW = bool(importlib.util.find_spec("pandas") and importlib.util.find_spec("pyarrow"))
KEY = "TestKey_1234567890"
# Headers exactly as Sharadar's API serves them (checked on its public sample, 2026-09).
STOCKS = "ticker,date,open,high,low,close,volume,closeadj,closeunadj,lastupdated"
ACTIONS = "date,action,ticker,name,value,contraticker,contraname"
DAILY = "ticker,date,lastupdated,ev,evebit,evebitda,marketcap,pb,pe,ps"
TICKERS = ("table,permaticker,ticker,name,exchange,isdelisted,category,cusips,siccode,sicsector,sicindustry,figi,"
           "famaindustry,sector,industry,scalemarketcap,scalerevenue,relatedtickers,currency,location,lastupdated,"
           "firstadded,firstpricedate,lastpricedate,firstquarter,lastquarter,secfilings,companysite")
SECURITIES = {  # ticker: (exchange, isdelisted, category, siccode, sector, industry, location)
    "AAA": ("NYSE", "N", "Domestic Common Stock", "3571", "Technology", "Consumer Electronics", "California; U.S.A"),
    "BBB.B": ("NASDAQ", "N", "Domestic Common Stock Secondary Class", "6331", "Financial Services", "Insurance",
              "Nebraska; U.S.A"),
    "DEAD": ("NYSE", "Y", "Domestic Common Stock", "2834", "Healthcare", "Biotechnology", "Texas; U.S.A"),
    "MOVE": ("OTC", "N", "Domestic Common Stock", "7372", "Technology", "Software", "Utah; U.S.A"),
    "OTCX": ("OTC", "N", "Domestic Common Stock", "1000", "Basic Materials", "Gold", "Nevada; U.S.A"),
    "PREF": ("NYSE", "N", "Domestic Preferred Stock", "6798", "Real Estate", "REIT - Mortgage", "New York; U.S.A"),
    "SPAC": ("NASDAQ", "N", "Domestic Common Stock Primary Class", "6770", "Financial Services", "Shell Companies",
             "Delaware; U.S.A"),
    "ADRX": ("NYSE", "N", "ADR Common Stock", "4813", "Communication Services", "Telecom Services", "China"),
    # delisted, venue no longer an exchange and no recorded move: its prices are its listed history
    "GONE": ("OTC", "Y", "Domestic Common Stock", "3674", "Technology", "Semiconductors", "Oregon; U.S.A"),
}
SESSIONS = 320
SPLIT_AT, DELIST_AT, MOVE_AT, SPIKE_AT = 150, 250, 200, 100


def _write(folder: Path, name: str, header: str, lines: list[str], zipped: bool) -> None:
    body = "\n".join([header, *lines]) + "\n"
    if zipped:
        with zipfile.ZipFile(folder / f"{name}.csv.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(f"SHARADAR_{name.upper()}.csv", body)
    else:
        (folder / f"{name}.csv").write_text(body, encoding="utf-8")


def sharadar_folder(folder: Path, zipped: bool = True) -> tuple[Path, list]:
    """Nine securities over SESSIONS business days in Sharadar's own file layout."""
    import numpy as np
    import pandas as pd

    folder.mkdir(parents=True)
    days = [d.date().isoformat() for d in pd.bdate_range("2023-01-02", periods=SESSIONS)]
    rng = np.random.RandomState(7)
    stocks, actions, daily, tickers = [], [], [], []
    for number, (ticker, (exchange, delisted, category, sic, sector, industry, location)) in enumerate(SECURITIES.items()):
        for table in ("stocks", "fundamentals"):  # the real table repeats securities per data table
            tickers.append(f"{table},{100 + number},{ticker},{ticker} CORP,{exchange},{delisted},{category},,{sic},,,,,"
                           f"{sector},{industry},,,,USD,\"{location}\",2026-09-25,2020-01-01,2020-01-01,{days[-1]},,,,")
        close = 40 * np.exp(np.cumsum(rng.normal(0, 0.01, SESSIONS))) * (1 + number / 10)
        volume = rng.randint(200_000, 400_000, SESSIONS).astype(float) * (10 - number)
        adjust = np.ones(SESSIONS)  # closeadj / close: dividends make earlier closes lower
        raw = close.copy()  # traded close
        if ticker == "AAA":
            raw[:SPLIT_AT] *= 2  # 2:1 split: traded at twice the split-adjusted price before it
            actions.append(f"{days[SPLIT_AT]},split,AAA,AAA CORP,2,N/A,N/A")
            for at in (60, 200):  # 0.25 per share on today's basis
                actions.append(f"{days[at]},dividend,AAA,AAA CORP,0.25,N/A,N/A")
                adjust[:at] *= 1 - 0.25 / close[at - 1]
        if ticker == "DEAD":
            actions.append(f"{days[DELIST_AT]},delisted,DEAD,DEAD CORP,,N/A,N/A")
        if ticker == "MOVE":
            actions.append(f"{days[MOVE_AT]},exchangefrom,MOVE,MOVE CORP,,N/A,NASDAQ")
            actions.append(f"{days[MOVE_AT]},exchangeto,MOVE,MOVE CORP,,N/A,OTC")
        if ticker == "ADRX":
            close[SPIKE_AT] *= 3  # an isolated bad print
            raw[SPIKE_AT] *= 3
        for k, day in enumerate(days):
            c = close[k]
            stocks.append(f"{ticker},{day},{c * 0.99:.4f},{c * 1.02:.4f},{c * 0.98:.4f},{c:.4f},{volume[k]:.0f},"
                          f"{c * adjust[k]:.4f},{raw[k]:.4f},2026-09-25")
            if ticker == "AAA":
                daily.append(f"AAA,{day},2026-09-25,,,,{raw[k] * 2.5:.3f},,,")  # 2.5M traded shares, $ millions
    stocks.append(stocks[5])  # one duplicate row
    _write(folder, "stocks", STOCKS, stocks, zipped)
    _write(folder, "actions", ACTIONS, actions, zipped)
    _write(folder, "tickers", TICKERS, tickers, zipped)
    _write(folder, "daily", DAILY, daily, zipped)
    return folder, days


class NamingTests(unittest.TestCase):
    def test_types_symbols_and_file_names(self):
        from qlib_yahoo.sharadar import _table_name, country, security_type, symbol

        self.assertEqual(security_type("Domestic Common Stock Primary Class", "3571"), "stock")
        self.assertEqual(security_type("ADR Common Stock", "4813"), "stock")
        self.assertEqual(security_type("Canadian Common Stock", ""), "stock")
        self.assertEqual(security_type("Domestic Common Stock Secondary Class", "6770"), "spac")
        self.assertEqual(security_type("Domestic Preferred Stock", "6798"), "preferred")
        self.assertEqual(security_type("ETF", ""), "fund")
        self.assertEqual(security_type("Institutional Investor", ""), "other")
        self.assertEqual(symbol("brk.b"), "BRK-B")
        self.assertEqual(country("California; U.S.A"), "United States")
        self.assertEqual(country("China"), "China")
        for name, table in (("stocks.csv.zip", "stocks"), ("SHARADAR_SEP_2_4f3a.zip", "sep"), ("tickers.csv", "tickers"),
                            ("SHARADAR_ACTIONS.csv", "actions"), ("daily-2026.csv.zip", "daily")):
            self.assertEqual(_table_name(name), table)


@unittest.skipUnless(HAVE_ARROW, "pandas and pyarrow are optional outside the Qlib environment")
class BuildTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pandas as pd
        import pyarrow.parquet as pq

        from qlib_yahoo.sharadar import build_table

        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.source, cls.days = sharadar_folder(root / "sharadar")
        cls.table = root / "table.parquet"
        cls.manifest = build_table(cls.source, cls.table)
        cls.frame = pq.read_table(cls.table).to_pandas()
        cls.raw = pd.read_csv(zipfile.ZipFile(cls.source / "stocks.csv.zip").open("SHARADAR_STOCKS.csv"))
        cls.pd = pd

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def rows(self, ticker):
        return self.frame[self.frame["ticker"] == ticker].sort_values("date").reset_index(drop=True)

    def test_types_and_symbols(self):
        types = self.frame.groupby("ticker")["type"].first().to_dict()
        self.assertEqual(types, {"AAA": "stock", "BBB-B": "stock", "DEAD": "stock", "MOVE": "stock", "PREF": "preferred",
                                 "SPAC": "spac", "ADRX": "stock", "GONE": "stock"})
        self.assertNotIn("OTCX", types)  # quoted only off-exchange: no bar kept
        self.assertEqual(len(self.rows("GONE")), SESSIONS)
        self.assertEqual(self.manifest["classification"]["stock"]["count"], 6)
        self.assertEqual(self.manifest["delisted_stocks"], {"count": 2, "examples": ["DEAD", "GONE"]})
        self.assertEqual(self.manifest["duplicate_bars_dropped"], 1)

    def test_prices_volume_raw_close_and_cap(self):
        import numpy as np

        aaa = self.rows("AAA")
        raw = self.raw[self.raw["ticker"] == "AAA"].drop_duplicates("date").reset_index(drop=True)
        factor = raw["closeadj"] / raw["close"]
        np.testing.assert_allclose(aaa["close"], raw["closeadj"])
        np.testing.assert_allclose(aaa["open"], raw["open"] * factor)
        np.testing.assert_allclose(aaa["close"] * aaa["volume"], raw["close"] * raw["volume"])  # traded dollars
        np.testing.assert_allclose(aaa["close_raw"], raw["closeunadj"])
        self.assertAlmostEqual(float(aaa["close_raw"][0] / raw["close"][0]), 2.0, places=3)  # before the 2:1 split
        np.testing.assert_allclose(aaa["cap"], aaa["close_raw"] * 2.5e6, rtol=1e-5)  # fixture: $ millions to 3 decimals
        self.assertTrue(self.rows("BBB-B")["cap"].isna().all())

    def test_dividends_on_traded_share_basis(self):
        aaa = self.rows("AAA")
        paid = aaa[aaa["dividend"] > 0]
        self.assertEqual([d.date().isoformat() for d in paid["date"]], [self.days[60], self.days[200]])
        self.assertAlmostEqual(float(paid["dividend"].iloc[0]), 0.5, places=3)  # 0.25 today = 0.50 before the split
        self.assertAlmostEqual(float(paid["dividend"].iloc[1]), 0.25, places=6)
        self.assertEqual(self.manifest["dividends"], {"actions": 2, "matched_to_bars": 2})

    def test_off_exchange_bars_are_dropped(self):
        dead, move = self.rows("DEAD"), self.rows("MOVE")
        self.assertEqual(dead["date"].iloc[-1].date().isoformat(), self.days[DELIST_AT])  # delisting day kept
        self.assertEqual(move["date"].iloc[-1].date().isoformat(), self.days[MOVE_AT - 1])  # moved to OTC
        self.assertEqual(len(move), MOVE_AT)
        dropped = (SESSIONS - DELIST_AT - 1) + (SESSIONS - MOVE_AT) + SESSIONS
        self.assertEqual(self.manifest["listing_filter"]["bars_dropped"], dropped)

    def test_isolated_spike_is_removed(self):
        adrx = self.rows("ADRX")
        self.assertTrue(self.pd.isna(adrx["close"][SPIKE_AT]))
        self.assertEqual(self.manifest["isolated_spikes"]["rows"], 1)

    def test_existing_output_and_changed_download_are_refused(self):
        from qlib_yahoo.sharadar import SharadarSource, build_table

        with self.assertRaises(DatasetError):
            build_table(self.source, self.table)
        with tempfile.TemporaryDirectory() as tmp:
            folder, _ = sharadar_folder(Path(tmp) / "download")
            manifest = {"files": {"tickers": {"sha256": "0" * 64}}}
            (folder / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(DatasetError, "differs from its download manifest"):
                SharadarSource(folder)

    def test_plain_csv_folder_and_groups(self):
        from qlib_yahoo.reference import group_source

        with tempfile.TemporaryDirectory() as tmp:
            folder, _ = sharadar_folder(Path(tmp) / "csv", zipped=False)
            source = group_source(folder)
            rows, coverage = source.groups(["AAA", "ADRX", "ZZZ"])
        self.assertEqual(rows[0], {"symbol": "AAA", "sector": "Technology", "industry": "Consumer Electronics",
                                   "country": "United States"})
        self.assertEqual(rows[1]["country"], "China")
        self.assertEqual(coverage["not_in_screener"], 1)
        self.assertIn("Sharadar", source.group_limitation)


class FetchTests(unittest.TestCase):
    def zip_body(self, table):
        from test_qlib_sharadar import ACTIONS, DAILY, STOCKS, TICKERS

        header = {"stocks": STOCKS, "actions": ACTIONS, "tickers": TICKERS, "daily": DAILY}[table]
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(f"SHARADAR_{table.upper()}.csv", header + "\n")
        return buffer.getvalue()

    def test_download_records_hashes_and_never_the_key(self):
        from qlib_yahoo.sharadar import fetch_bulk

        requested = []

        def opener(request, timeout):
            requested.append(request.full_url)
            table = request.full_url.split("/data/")[1].split("?")[0]
            return io.BytesIO(self.zip_body(table))

        with tempfile.TemporaryDirectory() as tmp:
            destination = fetch_bulk(Path(tmp) / "sharadar", KEY, opener=opener)
            self.assertEqual(sorted(p.name for p in destination.iterdir()),
                             ["actions.csv.zip", "manifest.json", "stocks.csv.zip", "tickers.csv.zip"])
            text = (destination / "manifest.json").read_text()
            self.assertNotIn(KEY, text)
            manifest = json.loads(text)
            self.assertEqual(manifest["years"], "full")
            self.assertEqual(manifest["files"]["stocks"]["member"], "SHARADAR_STOCKS.csv")
            self.assertTrue(all("years=full" in url and f"api_key={KEY}" in url for url in requested))
            fetch_bulk(destination, KEY, opener=opener)  # resuming skips verified files
            self.assertEqual(len(requested), 3)
            fetch_bulk(destination, KEY, tables=["daily"], opener=opener)
            self.assertIn("daily", json.loads((destination / "manifest.json").read_text())["files"])

    def test_errors_do_not_leak_the_key(self):
        from qlib_yahoo.sharadar import fetch_bulk

        def refused(request, timeout):
            raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, None)

        def offline(request, timeout):
            raise urllib.error.URLError(f"cannot reach {request.full_url}")

        with tempfile.TemporaryDirectory() as tmp:
            messages = []
            for number, opener in enumerate((refused, offline)):
                with self.assertRaises(DatasetError) as caught:
                    fetch_bulk(Path(tmp) / f"s{number}", KEY, opener=opener)
                messages.append(str(caught.exception))
                self.assertNotIn(KEY, messages[-1])
                self.assertIsNone(caught.exception.__cause__)
                self.assertFalse(list((Path(tmp) / f"s{number}").glob("*.part")))
            self.assertIn("HTTP 403", messages[0])
            self.assertIn("<key>", messages[1])

            def vendor_error(request, timeout):  # the API answers some refusals with JSON and status 200
                return io.BytesIO(json.dumps({"error": "Forbidden", "description": f"plan lacks table ({KEY})"}).encode())

            with self.assertRaisesRegex(DatasetError, "not a zip file.*plan lacks table") as caught:
                fetch_bulk(Path(tmp) / "json", KEY, opener=vendor_error)
            self.assertNotIn(KEY, str(caught.exception))
            self.assertFalse(list((Path(tmp) / "json").glob("*.part")))
            with self.assertRaisesRegex(DatasetError, "malformed"):
                fetch_bulk(Path(tmp) / "bad", "short", opener=refused)
            with self.assertRaisesRegex(DatasetError, "tables must be"):
                fetch_bulk(Path(tmp) / "bad", KEY, tables=["sf1"], opener=refused)
            (Path(tmp) / "taken").mkdir()
            with self.assertRaisesRegex(DatasetError, "not a Sharadar download"):
                fetch_bulk(Path(tmp) / "taken", KEY, opener=refused)

    def test_key_comes_from_the_environment_or_a_file(self):
        from qlib_yahoo.sharadar import api_key_from_environment

        with patch.dict(os.environ, {"SHARADAR_API_KEY": KEY}):
            self.assertEqual(api_key_from_environment(), KEY)
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(DatasetError):
            api_key_from_environment()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "key.txt"
            path.write_text(KEY + "\n")
            self.assertEqual(api_key_from_environment(path), KEY)


def _have_importer():
    try:
        from test_qlib_stooq_table import HAVE_ARROW as arrow, REAL_DUMPER
    except ImportError:
        return False
    return arrow and bool(importlib.util.find_spec("qlib")) and REAL_DUMPER.is_file()


@unittest.skipUnless(_have_importer(), "Qlib environment and the pinned upstream dump_bin checkout are required")
class SharadarSnapshotTests(unittest.TestCase):
    """The built table through import-stooq-table: delisted members, market cap field, Sharadar groups."""

    @classmethod
    def setUpClass(cls):
        from test_qlib_stooq_table import REAL_DUMPER

        from qlib_yahoo.sharadar import build_table
        from qlib_yahoo.stooq_table import prepare_stooq_table
        from qlib_yahoo.universe import UniverseRule

        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name).resolve()
        source, cls.days = sharadar_folder(root / "sources" / "sharadar")
        table = root / "sources" / "sharadar-table" / "table.parquet"
        table.parent.mkdir()
        build_table(source, table)
        with patch("qlib_yahoo.stooq_table._dump_script", return_value=REAL_DUMPER):
            cls.snapshot = prepare_stooq_table(
                table, root, "2023-01-01", "2025-01-01",
                rule=UniverseRule(name="top4", size=4, adv_sessions=10, min_history_sessions=40, min_price=1.0),
                max_workers=2, groups=source, progress=lambda _message: None)
        cls.manifest = json.loads((cls.snapshot / "manifest.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_snapshot_records_the_source(self):
        self.assertTrue(self.snapshot.name.startswith("sharadar-table-top4-"))
        self.assertEqual(self.manifest["fields"],
                         ["open", "high", "low", "close", "volume", "factor", "close_raw", "dividend", "cap"])
        self.assertTrue(self.manifest["provider"].startswith("Sharadar"))
        self.assertEqual(self.manifest["table_builder"]["kind"], "sharadar")
        self.assertEqual(self.manifest["table_builder"]["delisted_stocks"], 2)
        self.assertIn("Delisted companies are included", self.manifest["limitations"][0])
        self.assertEqual(self.manifest["groups"]["coverage"]["not_in_screener"], 0)

    def test_delisted_stock_was_a_member_and_counts_as_ended(self):
        import pandas as pd

        from qlib_yahoo.data_quality import scan

        membership = pd.read_csv(self.snapshot / "universe" / "top4_membership.csv.gz")
        self.assertIn("DEAD", set(membership["instrument"]))
        report = scan(self.snapshot)
        self.assertGreaterEqual(report["totals"]["members_that_ended"], 1)


if __name__ == "__main__":
    unittest.main()
