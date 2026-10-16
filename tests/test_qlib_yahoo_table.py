"""Building the consolidated daily table from Yahoo charts plus the Nasdaq screener."""

from __future__ import annotations

import importlib.util
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from test_qlib_corporate_actions import build_archive, chart_payload

from qlib_yahoo.reference import Screener, fetch_screener

HAVE_ARROW = bool(importlib.util.find_spec("pandas") and importlib.util.find_spec("pyarrow"))
EQUITY = {"instrumentType": "EQUITY", "currency": "USD", "exchangeName": "NMS"}


def screener_folder(folder: Path, rows: list[dict]) -> Path:
    body = json.dumps({"data": {"rows": rows}}).encode()
    return fetch_screener(folder, opener=lambda request, timeout: io.BytesIO(body))


def listing(symbol, name, industry="Semiconductors", sector="Technology"):
    return {"symbol": symbol, "name": name, "industry": industry, "sector": sector, "country": "United States"}


class ClassifyTests(unittest.TestCase):
    def test_rules(self):
        from qlib_yahoo.yahoo_table import classify

        universe = {"AGNC", "AGNCL", "AGNCP", "CORZ", "CORZW", "GOOG", "GOOGL"}
        cases = [
            ("AAPL", EQUITY, listing("AAPL", "Apple Inc. Common Stock"), "stock"),
            ("TSM", {**EQUITY, "exchangeName": "NYQ"}, listing("TSM", "Taiwan Semiconductor Manufacturing Company Ltd."), "stock"),
            ("BABA", {**EQUITY, "exchangeName": "NYQ"},
             listing("BABA", "Alibaba Group Holding Limited American Depositary Shares each representing eight Ordinary share"), "stock"),
            ("MSTU", {**EQUITY, "instrumentType": "ETF"}, None, "yahoo_etf"),
            ("AGNCL", EQUITY, listing("AGNCL", "AGNC Investment Corp. Depositary Shares Each Representing a 1/1,000th "
                                               "Interest in a Share of 7.75% Series G Fixed-Rate Reset Cumulative Redeemable Preferred"),
             "preferred"),
            ("AGNCP", EQUITY, None, "preferred"),  # not in the screener: the fifth letter P decides
            ("GOOGL", EQUITY, None, "stock"),  # L is ambiguous (Alphabet class A), so it stays
            ("CORZW", EQUITY, listing("CORZW", "Core Scientific Inc. Tranche 1 Warrants"), "warrant"),
            ("GOOGL", EQUITY, listing("GOOGL", "Alphabet Inc. Class A Common Stock"), "stock"),
            ("NUV", {**EQUITY, "exchangeName": "NYQ"}, listing("NUV", "Nuveen Municipal Value Fund Inc. Common Stock"), "fund"),
            ("BST", {**EQUITY, "exchangeName": "NYQ"},
             listing("BST", "BlackRock Science and Technology Trust Common Shares of Beneficial Interest"), "fund"),
            ("COLD", {**EQUITY, "exchangeName": "NYQ"}, listing("COLD", "Americold Realty Trust Inc. Common Stock",
                                                                   "Real Estate Investment Trusts"), "stock"),
            ("AACO", EQUITY, listing("AACO", "Abony Acquisition Corp. I Class A Ordinary Shares", "Blank Checks", ""), "spac"),
            ("MPLX", EQUITY, listing("MPLX", "MPLX LP Common Units Representing Limited Partner Interests",
                                     "Natural Gas Distribution", "Utilities"), "stock"),
            ("PBT", EQUITY, listing("PBT", "Permian Basin Royalty Trust Common Stock Units of Beneficial Interest",
                                    "Oil & Gas Production", "Energy"), "stock"),
            ("AMH", EQUITY, listing("AMH", "American Homes 4 Rent Common Shares of Beneficial Interest",
                                    "Real Estate Investment Trusts", "Real Estate"), "stock"),
            ("BTSGU", EQUITY, listing("BTSGU", "BrightSpring Health Services Inc. Tangible Equity Unit"), "unit"),
            ("SPACU", EQUITY, listing("SPACU", "Space Acquisition Corp. Units, each consisting of one share and one right"), "unit"),
            ("OTCX", {**EQUITY, "exchangeName": "PNK"}, None, "not_exchange_listed"),
            ("CADX", {**EQUITY, "currency": "CAD"}, None, "non_usd"),
        ]
        for symbol, meta, row, expected in cases:
            with self.subTest(symbol=symbol, expected=expected):
                self.assertEqual(classify(symbol, meta, row, universe), expected)


@unittest.skipUnless(HAVE_ARROW, "pandas and pyarrow are optional outside the Qlib environment")
class SplitReconciliationTests(unittest.TestCase):
    DAYS = ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05", "2026-06-08", "2026-06-09", "2026-06-10"]

    def test_yahoo_series_left_unadjusted_is_repaired_at_the_jump(self):
        from qlib_yahoo.yahoo_chart import parse_chart
        from qlib_yahoo.yahoo_table import _series

        # OPAD, June 2026: the 1:10 reverse split took effect on the 5th, Yahoo dates it the 9th and
        # never adjusted the earlier bars.
        close = [0.77, 0.76, 0.761, 0.74, 6.17, 5.40, 5.09, 5.10]
        history = parse_chart("OPAD", chart_payload("OPAD", self.DAYS, close, splits=[("2026-06-09", 1, 10)]))
        series = _series(history, "2026-07-01")
        self.assertEqual(series["split_events"], [("2026-06-05", 0.1, "unadjusted")])
        adjusted = [round(v, 3) for v in series["close"]]
        self.assertEqual(adjusted[:4], [7.7, 7.6, 7.61, 7.4])  # now on the post-split basis
        self.assertEqual([round(v, 3) for v in series["close_raw"]], close)  # the prices that traded

    def test_unrecorded_reverse_split_restarts_the_series(self):
        import numpy as np

        from qlib_yahoo.yahoo_table import _level_breaks

        close = np.array([0.50, 0.52, 0.51, 4.10, 4.00, 4.20, 3.90, 4.05])       # x8, stays up
        volume = np.array([8e6, 9e6, 8.5e6, 1.0e6, 1.1e6, 0.9e6, 1.2e6, 1.0e6])  # shares /8, same dollars
        self.assertEqual(_level_breaks(close, volume), [3])
        rally = volume.copy()
        rally[3] = 60e6  # a genuine squeeze trades far more dollars
        self.assertEqual(_level_breaks(close, rally), [])
        crash = close[::-1].copy()
        self.assertEqual(_level_breaks(crash, volume[::-1].copy()), [])  # downward shifts are left alone

    def test_adjusted_and_small_splits_are_left_alone(self):
        from qlib_yahoo.yahoo_chart import parse_chart
        from qlib_yahoo.yahoo_table import _series

        smooth = [7.7, 7.6, 7.61, 7.4, 6.17, 5.4, 5.09, 5.1]
        history = parse_chart("X", chart_payload("X", self.DAYS, smooth, splits=[("2026-06-05", 1, 10), ("2026-06-09", 21, 20)]))
        series = _series(history, "2026-07-01")
        self.assertEqual([k for *_, k in series["split_events"]], ["adjusted", "adjusted"])
        self.assertEqual([round(v, 3) for v in series["close"]], smooth)
        # Traded price before the 1:10 split was a tenth, and the 21:20 split (on the 9th) scales earlier bars too.
        self.assertAlmostEqual(series["close_raw"][0], 7.7 / 10 * 21 / 20)


@unittest.skipUnless(HAVE_ARROW, "pandas and pyarrow are optional outside the Qlib environment")
class BuildTableTests(unittest.TestCase):
    DAYS = [f"2024-06-{d:02d}" for d in (3, 4, 5, 6, 7, 10, 11, 12, 13, 14)]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        # SPLT: Yahoo close is split-adjusted (2:1 on the 6th session); a 1.00 dividend is paid on the 3rd
        # session (0.50 per post-split share, as Yahoo reports it); adjclose applies a 5% dividend factor before it.
        close = [50.0] * 10
        adjclose = [47.5, 47.5, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
        spike = [20.0] * 10
        spike[4] = 60.0  # a one-day bad print
        payloads = {
            "SPLT": chart_payload("SPLT", self.DAYS, close, adjclose, [2e6] * 10,
                                  splits=[(self.DAYS[5], 2, 1)], dividends=[(self.DAYS[2], 0.5)]),
            "SPIK": chart_payload("SPIK", self.DAYS, spike),
            "FUNDX": chart_payload("FUNDX", self.DAYS, close, kind="ETF"),
        }
        self.charts = build_archive(root / "charts", payloads)
        self.screener = screener_folder(root / "screener", [listing("SPLT", "Split Corp. Common Stock"),
                                                            listing("SPIK", "Spike Inc. Common Stock")])
        self.output = root / "table.parquet"

    def test_table_values_and_manifest(self):
        import pandas as pd

        from qlib_yahoo.yahoo_table import build_table

        manifest = build_table(self.charts, self.screener, self.output)
        table = pd.read_parquet(self.output)
        splt = table[table["ticker"] == "SPLT"].reset_index(drop=True)
        self.assertEqual(set(table["type"]), {"stock", "yahoo_etf"})
        # Traded close: 100 before the 2:1 split (Yahoo shows 50 split-adjusted), 50 after.
        self.assertEqual(splt["close_raw"].tolist(), [100.0] * 5 + [50.0] * 5)
        # Adjusted close follows adjclose; no factor column (the snapshot keeps Qlib's identity factor).
        self.assertEqual(splt["close"].round(6).tolist(), [47.5, 47.5] + [50.0] * 8)
        self.assertNotIn("factor", table.columns)
        # Dollar volume is the traded one: 2e6 split-adjusted shares at 50 = 1e6 traded shares at 100.
        self.assertAlmostEqual(splt["close"][0] * splt["volume"][0], 100.0 * 1e6)
        self.assertEqual(splt["dividend"].tolist(), [0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0])
        spik = table[table["ticker"] == "SPIK"].reset_index(drop=True)
        self.assertTrue(math.isnan(spik["close"][4]))
        self.assertEqual(manifest["isolated_spikes"]["rows"], 1)
        self.assertEqual(manifest["classification"]["yahoo_etf"]["examples"], ["FUNDX"])
        self.assertEqual(manifest["rows"], len(table))
        self.assertTrue(Path(str(self.output) + ".manifest.json").is_file())

    def test_existing_output_is_refused(self):
        from qlib_yahoo.dataset import DatasetError
        from qlib_yahoo.yahoo_table import build_table

        self.output.write_bytes(b"")
        with self.assertRaisesRegex(DatasetError, "exists"):
            build_table(self.charts, self.screener, self.output)

    def test_screener_is_verified(self):
        import gzip

        from qlib_yahoo.dataset import DatasetError

        self.assertIn("SPLT", Screener(self.screener).rows)
        (self.screener / "screener.json.gz").write_bytes(gzip.compress(b'{"data": {"rows": [{"symbol": "X"}]}}'))
        with self.assertRaisesRegex(DatasetError, "differs from its manifest"):
            Screener(self.screener)


if __name__ == "__main__":
    unittest.main()


def _have_importer():
    try:
        from test_qlib_stooq_table import HAVE_ARROW as arrow, REAL_DUMPER
    except ImportError:
        return False
    return arrow and bool(importlib.util.find_spec("qlib")) and REAL_DUMPER.is_file()


@unittest.skipUnless(_have_importer(), "Qlib environment and the pinned upstream dump_bin checkout are required")
class YahooSnapshotTests(unittest.TestCase):
    """Six synthetic stocks through fetch -> build-yahoo-table -> import: traded close, dividends, identity factor."""

    @classmethod
    def setUpClass(cls):
        from unittest.mock import patch

        import numpy as np
        from test_qlib_stooq_table import REAL_DUMPER, synthetic_table

        from qlib_yahoo.stooq_table import prepare_stooq_table
        from qlib_yahoo.universe import UniverseRule
        from qlib_yahoo.yahoo_table import build_table

        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name).resolve()
        frame, days, _ = synthetic_table()
        payloads, rows = {}, []
        for i in range(6):
            part = frame[(frame["ticker"] == f"S{i}")].sort_values("date")
            part = part[part["date"] <= days[-1].date()]
            dates = [f"{d}" for d in part["date"]]
            close = part["close"].round(4).tolist()
            # S2 split 3:1 on its 200th session; S3 pays dividends (adjclose 2% lower before the ex-date).
            splits = [(dates[200], 3, 1)] if i == 2 else []
            adjclose = [c * (0.98 if i == 3 and k < 150 else 1.0) for k, c in enumerate(close)]
            dividends = [(dates[150], 0.4)] if i == 3 else []
            payloads[f"S{i}"] = chart_payload(f"S{i}", dates, close, adjclose, part["volume"].tolist(),
                                              splits=splits, dividends=dividends)
            rows.append(listing(f"S{i}", f"S{i} Corp. Common Stock"))
        charts = build_archive(root / "charts", payloads)
        screener = screener_folder(root / "screener", rows)
        (root / "sources").mkdir()
        table = root / "sources" / "yahoo.parquet"  # a sources folder under the data root is allowed
        cls.table_manifest = build_table(charts, screener, table)
        with patch("qlib_yahoo.stooq_table._dump_script", return_value=REAL_DUMPER):
            cls.snapshot = prepare_stooq_table(
                table, root, "2023-01-01", "2025-01-01",
                rule=UniverseRule(name="top3", size=3, adv_sessions=10, min_history_sessions=40, min_price=1.0),
                max_workers=2, groups=screener, progress=lambda _message: None)
        cls.manifest = json.loads((cls.snapshot / "manifest.json").read_text())
        cls.np = np

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def bins(self, symbol, field):
        return self.np.fromfile(self.snapshot / "qlib_data" / "features" / symbol / f"{field}.day.bin", dtype="<f4")[1:]

    def test_fields_and_factor(self):
        self.assertEqual(self.manifest["fields"], ["open", "high", "low", "close", "volume", "factor", "close_raw", "dividend"])
        self.assertIn("factor=1 identity", self.manifest["normalization"])
        close, factor, raw = self.bins("s2", "close"), self.bins("s2", "factor"), self.bins("s2", "close_raw")
        ok = self.np.isfinite(close)
        self.assertTrue((self.np.isfinite(raw) == ok).all())
        self.assertTrue((factor[ok] == 1).all())  # the runner's contract
        self.assertAlmostEqual(float(raw[0] / close[0]), 3.0, places=4)  # traded at 3x before the 3:1 split
        self.assertAlmostEqual(float(raw[-1] / close[-1]), 1.0, places=4)
        dividend = self.bins("s3", "dividend")
        self.assertAlmostEqual(float(dividend[self.np.flatnonzero(dividend)[0]]), 0.4, places=6)
        self.assertEqual(self.manifest["groups"]["coverage"]["not_in_screener"], 0)
        self.assertTrue(self.snapshot.name.startswith("yahoo-table-top3-"))
        builder = self.manifest["table_builder"]
        self.assertEqual(builder["table_sha256"], self.table_manifest["table_sha256"])
        self.assertIn("yahoo_chart", builder["sources"])
        self.assertTrue(self.manifest["provider"].startswith("Yahoo Finance"))

    def test_quality_scan_uses_traded_prices(self):
        from qlib_yahoo.data_quality import scan

        report = scan(self.snapshot)
        self.assertEqual(report["price_basis"], "unadjusted close (close_raw)")
        self.assertEqual(report["totals"]["possible_unadjusted_splits"], 0)


@unittest.skipUnless(_have_importer(), "Qlib environment and the pinned upstream dump_bin checkout are required")
class RefreshTests(unittest.TestCase):
    def test_one_command_builds_a_checked_snapshot(self):
        from unittest.mock import patch

        from test_qlib_stooq_table import REAL_DUMPER, synthetic_table

        from qlib_yahoo.refresh import refresh

        frame, days, _ = synthetic_table()
        payloads, rows = {}, []
        for i in range(6):
            part = frame[frame["ticker"] == f"S{i}"].sort_values("date")
            part = part[part["date"] <= days[-1].date()]
            payloads[f"S{i}"] = chart_payload(f"S{i}", [f"{d}" for d in part["date"]], part["close"].round(4).tolist(),
                                              volume=part["volume"].tolist())
            rows.append(listing(f"S{i}", f"S{i} Corp. Common Stock"))
        rows.append(listing("SPYX", "Some Index ETF"))  # the screener's stock list has no ETFs; Yahoo types are checked anyway
        payloads["SPYX"] = chart_payload("SPYX", [f"{d.date()}" for d in days], [100.0] * len(days), kind="ETF")
        body = json.dumps({"data": {"rows": rows}}).encode()
        messages = []
        with tempfile.TemporaryDirectory() as temporary, \
                patch("qlib_yahoo.stooq_table._dump_script", return_value=REAL_DUMPER):
            root = Path(temporary).resolve()
            record = refresh(root, universe_size=3, min_price=1.0, workers=2, progress=messages.append,
                             fetch=lambda symbol, begin, stop: (200, payloads[symbol]) if symbol in payloads else (404, b"{}"),
                             opener=lambda request, timeout: io.BytesIO(body))
            snapshot = Path(record["snapshot"])
            manifest = json.loads((snapshot / "manifest.json").read_text())
            self.assertTrue(snapshot.name.startswith("yahoo-table-top3-"))
            self.assertEqual(record["tickers"], 7)
            self.assertEqual(manifest["table_builder"]["classification"]["yahoo_etf"], 1)
            self.assertEqual(manifest["universes"]["top3"]["min_price"], 1.0)
            self.assertEqual(manifest["groups"]["coverage"]["not_in_screener"], 0)
            self.assertTrue((Path(record["table"]).parent / "refresh.json").is_file())
            self.assertIn("possible_unadjusted_splits", record["quality"])
            self.assertTrue(any(m.startswith("5/5") for m in messages))
