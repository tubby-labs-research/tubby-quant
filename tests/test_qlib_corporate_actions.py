"""Yahoo chart archives and the reconciliation of Stooq bars with split and dividend events."""

from __future__ import annotations

import gzip
import importlib.util
import json
import math
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from qlib_yahoo.dataset import DatasetError
from qlib_yahoo.yahoo_chart import ChartArchive, fetch_archive, parse_chart

HAVE_PANDAS = bool(importlib.util.find_spec("pandas") and importlib.util.find_spec("numpy"))
OFFSET = -4 * 3600  # US Eastern daylight time, as Yahoo reports it


def _stamp(day: str) -> int:
    """Yahoo stamps a daily bar at the 09:30 open in UTC."""
    local = datetime.fromisoformat(day).replace(hour=9, minute=30, tzinfo=timezone.utc)
    return int((local - timedelta(seconds=OFFSET)).timestamp())


def chart_payload(symbol, days, close, adjclose=None, volume=None, splits=(), dividends=(), kind="EQUITY"):
    """A chart response shaped like Yahoo's v8 endpoint."""
    volume = volume or [1e6] * len(days)
    adjclose = adjclose or close
    return json.dumps({"chart": {"result": [{
        "meta": {"symbol": symbol, "currency": "USD", "instrumentType": kind, "gmtoffset": OFFSET, "exchangeName": "NMS",
                 "exchangeTimezoneName": "America/New_York", "longName": f"{symbol} Inc."},
        "timestamp": [_stamp(day) for day in days],
        "events": {
            "splits": {str(_stamp(day)): {"date": _stamp(day), "numerator": n, "denominator": d, "splitRatio": f"{n}:{d}"}
                       for day, n, d in splits},
            "dividends": {str(_stamp(day)): {"date": _stamp(day), "amount": amount} for day, amount in dividends},
        },
        "indicators": {"quote": [{"open": close, "high": close, "low": close, "close": close, "volume": volume}],
                       "adjclose": [{"adjclose": adjclose}]},
    }], "error": None}}).encode()


NOT_FOUND = json.dumps({"chart": {"result": None, "error": {"code": "Not Found", "description": "No data found"}}}).encode()


def build_archive(folder: Path, payloads: dict[str, bytes]) -> Path:
    """Fetch through the real archive code with a fake network."""
    def fake_fetch(symbol, begin, stop):
        return (200, payloads[symbol]) if symbol in payloads else (404, NOT_FOUND)

    return fetch_archive([*payloads, "GONE"], folder, "2023-01-01", "2025-01-01", workers=2, fetch=fake_fetch)


class ChartParsingTests(unittest.TestCase):
    def test_dates_events_and_nulls(self):
        history = parse_chart("ABC", chart_payload(
            "ABC", ["2024-06-03", "2024-06-04"], [10.0, None], [9.5, None],
            splits=[("2024-06-04", 1, 10)], dividends=[("2024-06-03", 0.25)]))
        self.assertEqual(history.dates, ["2024-06-03", "2024-06-04"])
        self.assertEqual(history.close, [10.0, None])
        self.assertEqual(history.splits, [("2024-06-04", 1.0, 10.0)])
        self.assertEqual(history.dividends, [("2024-06-03", 0.25)])
        self.assertEqual(history.meta["instrumentType"], "EQUITY")

    def test_error_responses_are_refused(self):
        for body in (NOT_FOUND, b"not json", json.dumps({"chart": {"result": [{
                "timestamp": [1, 2], "indicators": {"quote": [{"close": [1.0]}]}}]}}).encode()):
            with self.assertRaises(DatasetError):
                parse_chart("ABC", body)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_fetch_resume_and_verification(self):
        payload = chart_payload("ABC", ["2024-06-03"], [10.0])
        path = build_archive(self.root / "charts", {"ABC": payload})
        archive = ChartArchive(path)
        self.assertEqual(archive.symbols(), ["ABC"])
        self.assertEqual(archive.manifest["tickers"]["GONE"]["status"], 404)
        self.assertEqual(gzip.decompress((path / "charts" / "ABC.json.gz").read_bytes()), payload)
        self.assertIsNone(archive.history("GONE"))
        calls = []
        fetch_archive(["ABC"], path, "2023-01-01", "2025-01-01", fetch=lambda *a: calls.append(a) or (200, payload))
        self.assertEqual(calls, [])  # complete archives are not fetched again
        with self.assertRaisesRegex(DatasetError, "different period"):
            fetch_archive(["ABC"], path, "2023-01-02", "2025-01-01")
        (path / "charts" / "ABC.json.gz").write_bytes(gzip.compress(chart_payload("ABC", ["2024-06-03"], [11.0])))
        with self.assertRaisesRegex(DatasetError, "SHA-256"):
            archive.history("ABC")

    def test_failures_leave_a_resumable_archive(self):
        attempts = {"n": 0}

        def flaky(symbol, begin, stop):
            attempts["n"] += 1
            return (503, b"busy") if attempts["n"] == 1 else (200, chart_payload(symbol, ["2024-06-03"], [1.0]))

        with self.assertRaisesRegex(DatasetError, "1 tickers failed"):
            fetch_archive(["ABC"], self.root / "c", "2023-01-01", "2025-01-01", fetch=flaky)
        with self.assertRaisesRegex(DatasetError, "not complete"):
            ChartArchive(self.root / "c")
        fetch_archive(["ABC"], self.root / "c", "2023-01-01", "2025-01-01", fetch=flaky)
        self.assertEqual(ChartArchive(self.root / "c").symbols(), ["ABC"])

    def test_retries_back_off_but_404_is_final(self):
        from qlib_yahoo.yahoo_chart import fetch_chart

        answers = [(429, b""), (500, b""), (200, b"ok")]
        waits = []
        self.assertEqual(fetch_chart("A", date(2024, 1, 1), date(2024, 2, 1), sleep=waits.append,
                                     request=lambda *a: answers.pop(0)), (200, b"ok"))
        self.assertEqual(len(waits), 2)
        self.assertEqual(fetch_chart("A", date(2024, 1, 1), date(2024, 2, 1), sleep=waits.append,
                                     request=lambda *a: (404, b"x"))[0], 404)


@unittest.skipUnless(HAVE_PANDAS, "pandas is optional outside the Qlib environment")
class ReconcileTests(unittest.TestCase):
    """Four tickers over ten sessions; a split takes effect on the sixth."""

    DAYS = [f"2024-06-{d:02d}" for d in (3, 4, 5, 6, 7, 10, 11, 12, 13, 14)]

    def setUp(self):
        import numpy as np
        import pandas as pd

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        split = self.DAYS[5]
        # Yahoo's split-only close is continuous: 10 before and after (1:10 reverse split).
        base = [10.0] * 10
        payloads = {
            # REV: Stooq still shows the raw pre-split 1.00 -> 10.00 jump.
            "REV": chart_payload("REV", self.DAYS, base, splits=[(split, 1, 10)]),
            # FWD: 2:1 split already adjusted by Stooq.
            "FWD": chart_payload("FWD", self.DAYS, base, splits=[(split, 2, 1)],
                                 dividends=[(self.DAYS[2], 0.5)]),
            # ODD: Stooq moves x3 across a 1:10 split: neither adjusted nor unadjusted.
            "ODD": chart_payload("ODD", self.DAYS, base, splits=[(split, 1, 10)]),
            "ETF": chart_payload("ETF", self.DAYS, base, kind="ETF"),
        }
        self.archive = ChartArchive(build_archive(Path(self.tmp.name) / "charts", payloads))
        closes = {"REV": [1.0] * 5 + [10.0] * 5, "FWD": base, "ODD": [3.0] * 5 + [9.0] * 5,
                  "NOYAHOO": [5.0] * 10}
        volumes = {"REV": [1e6] * 5 + [1e5] * 5}
        rows = [(t, pd.Timestamp(day), c, c, c, c, volumes.get(t, [1e6] * 10)[k])
                for t in sorted(closes) for k, (day, c) in enumerate(zip(self.DAYS, closes[t]))]
        frame = pd.DataFrame(rows, columns=["ticker", "date", "open", "high", "low", "close", "volume"])
        frame["ticker"] = frame["ticker"].astype("category")
        self.frame = frame
        self.invalid = np.zeros(len(frame), dtype=bool)

    def rows(self, ticker):
        return (self.frame["ticker"] == ticker).to_numpy()

    def test_repairs_confirms_and_masks(self):
        from qlib_yahoo.corporate_actions import reconcile

        columns, report = reconcile(self.frame, self.invalid, self.archive)
        rev = self.frame[self.rows("REV")]
        self.assertEqual(rev["close"].tolist(), [10.0] * 10)  # repaired to the post-split basis
        self.assertEqual(rev["volume"].tolist(), [1e5] * 10)
        self.assertEqual(report["unadjusted_repaired"]["count"], 1)
        self.assertEqual(report["already_adjusted"]["examples"], [f"FWD {self.DAYS[5]} 2:1"])
        self.assertEqual(report["ambiguous_masked_before_split"]["count"], 1)
        odd = columns["masked"][self.rows("ODD")]
        self.assertEqual(odd.tolist(), [True] * 5 + [False] * 5)
        self.assertEqual(report["tickers_without_yahoo_history"]["examples"], ["NOYAHOO"])
        # Unadjusted close: REV traded at 1.00 before its 1:10 split; FWD at 20 before its 2:1 split.
        raw = columns["close_raw"]
        self.assertEqual(raw[self.rows("REV")].tolist(), [1.0] * 5 + [10.0] * 5)
        self.assertEqual(raw[self.rows("FWD")].tolist(), [20.0] * 5 + [10.0] * 5)
        self.assertEqual(raw[self.rows("NOYAHOO")].tolist(), [5.0] * 10)
        # FWD's 0.50 dividend (split-adjusted) was 1.00 per pre-split share.
        dividend = columns["dividend"][self.rows("FWD")]
        self.assertEqual(dividend.tolist(), [0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(report["price_cross_check"]["days_off"], 0)

    def test_yahoo_series_left_unadjusted_does_not_mislead(self):
        """AMTX, June 2016: Yahoo's own close jumps x4 at a 4:1 split it never adjusted; Stooq is correct."""
        import numpy as np
        import pandas as pd

        from qlib_yahoo.corporate_actions import reconcile

        yahoo = [4.0] * 5 + [16.0] * 5  # the jump sits in Yahoo's split-only close
        payload = chart_payload("AMTX", self.DAYS, yahoo, splits=[(self.DAYS[5], 4, 1)])
        stooq = [4.0] * 10  # already continuous
        frame = pd.DataFrame({"ticker": pd.Categorical(["AMTX"] * 10), "date": pd.to_datetime(self.DAYS),
                              "open": stooq, "high": stooq, "low": stooq, "close": stooq, "volume": [1e6] * 10})
        with patch.object(ChartArchive, "history", lambda _self, t: parse_chart(t, payload)):
            columns, report = reconcile(frame, np.zeros(10, dtype=bool), self.archive)
        self.assertEqual(report["unadjusted_repaired"]["count"], 0)
        self.assertEqual(report["already_adjusted"]["count"], 1)
        self.assertEqual(frame["close"].tolist(), stooq)
        self.assertEqual(columns["close_raw"].tolist(), [16.0] * 5 + [16.0] * 5)  # 4 x 4 traded before the split

    def test_price_disagreement_is_reported(self):
        import numpy as np
        import pandas as pd
        from qlib_yahoo.corporate_actions import reconcile

        # Stooq's close jumps 50% while Yahoo's stays flat: every later day is off.
        long_days = [f"2024-{m:02d}-{d:02d}" for m in (7, 8) for d in range(1, 29) if date(2024, m, d).weekday() < 5]
        stooq = [10.0] * (len(long_days) - 10) + [15.0] * 10
        frame = pd.DataFrame({"ticker": pd.Categorical(["ABC"] * len(long_days)),
                              "date": pd.to_datetime(long_days), "open": stooq, "high": stooq, "low": stooq,
                              "close": stooq, "volume": [1e6] * len(long_days)})
        body = chart_payload("ABC", long_days, [10.0] * len(long_days))
        with patch.object(ChartArchive, "history", lambda _self, t: parse_chart(t, body)):
            _, report = reconcile(frame, np.zeros(len(frame), dtype=bool), self.archive)
        check = report["price_cross_check"]
        self.assertEqual(check["days_off"], 10)
        self.assertEqual(check["tickers_over_5pct_of_days"]["count"], 1)

    def test_non_equity_instruments_are_named(self):
        from qlib_yahoo.corporate_actions import yahoo_type_exclusions

        self.assertEqual(yahoo_type_exclusions(self.archive, ["REV", "ETF", "NOYAHOO"]), {"ETF": "yahoo_type_etf"})


class UniversePriceFloorTests(unittest.TestCase):
    @unittest.skipUnless(HAVE_PANDAS, "pandas is optional outside the Qlib environment")
    def test_cheap_stocks_are_not_ranked(self):
        import numpy as np
        import pandas as pd
        from qlib_yahoo.universe import UniverseRule, liquidity_membership

        sessions = pd.bdate_range("2024-01-01", periods=60)
        tickers = ["CHEAP", "MID", "RICH"]
        dollars = pd.DataFrame(np.tile([3e7, 2e7, 1e7], (60, 1)), index=sessions, columns=tickers)
        valid = pd.DataFrame(True, index=sessions, columns=tickers)
        price = pd.DataFrame(np.tile([0.40, 5.0, 50.0], (60, 1)), index=sessions, columns=tickers)
        rule = UniverseRule(name="top2", size=2, adv_sessions=5, min_history_sessions=5, min_price=1.0)
        membership, _ = liquidity_membership(dollars, valid, rule, price)
        self.assertEqual(sorted(membership["instrument"].unique()), ["MID", "RICH"])
        without_floor, _ = liquidity_membership(dollars, valid, UniverseRule(name="top2", size=2, adv_sessions=5,
                                                                             min_history_sessions=5))
        self.assertEqual(sorted(without_floor["instrument"].unique()), ["CHEAP", "MID"])
        with self.assertRaisesRegex(DatasetError, "unadjusted close panel"):
            liquidity_membership(dollars, valid, rule)

    def test_invalid_floors_are_refused(self):
        from qlib_yahoo.universe import UniverseRule

        for value in (-1, float("nan"), True, "1"):
            with self.subTest(value=value), self.assertRaises(DatasetError):
                UniverseRule(min_price=value).validate()


if __name__ == "__main__":
    unittest.main()


def _have_importer():
    try:
        from test_qlib_stooq_table import HAVE_ARROW, REAL_DUMPER
    except ImportError:
        return False
    return HAVE_ARROW and bool(importlib.util.find_spec("qlib")) and REAL_DUMPER.is_file()


@unittest.skipUnless(_have_importer(), "Qlib environment and the pinned upstream dump_bin checkout are required")
class ImportWithActionsTests(unittest.TestCase):
    """The synthetic Stooq table with an unadjusted 1:10 split (S1), an ETF typed 'stock' and a sub-dollar stock (S5)."""

    @classmethod
    def setUpClass(cls):
        import numpy as np
        from test_qlib_stooq_table import REAL_DUMPER, synthetic_table
        from qlib_yahoo.stooq_table import prepare_stooq_table
        from qlib_yahoo.universe import UniverseRule

        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name).resolve()
        frame, days, _holiday = synthetic_table()
        frame = frame[frame["date"] <= days[-1].date()].copy()  # no single-stock holiday row
        etf = frame[frame["ticker"] == "S0"].assign(ticker="ETFX")
        frame = frame._append(etf, ignore_index=True)
        cls.split_day = days[300].date()
        payloads = {}
        for ticker, rows in frame[frame["type"] == "stock"].groupby("ticker"):
            if ticker not in {f"S{i}" for i in range(8)} | {"ETFX"}:
                continue
            closes = rows.sort_values("date")["close"].tolist()
            split = [(f"{cls.split_day}", 1, 10)] if ticker == "S1" else []
            payloads[ticker] = chart_payload(ticker, [f"{d}" for d in sorted(rows["date"])], closes,
                                             splits=split, kind="ETF" if ticker == "ETFX" else "EQUITY")
        # S1 in Stooq: raw pre-split prices (10x lower, 10x the shares) - the unadjusted state.
        before = (frame["ticker"] == "S1") & (frame["date"] < cls.split_day)
        frame.loc[before, ["open", "high", "low", "close"]] /= 10
        frame.loc[before, "volume"] *= 10
        # S5 trades below a dollar throughout (same dollar volume).
        cheap = frame["ticker"] == "S5"
        frame.loc[cheap, ["open", "high", "low", "close"]] /= 100
        frame.loc[cheap, "volume"] *= 100
        s5 = frame[cheap].sort_values("date")
        payloads["S5"] = chart_payload("S5", [f"{d}" for d in s5["date"]], s5["close"].tolist())
        cls.days = days
        table = root / "source" / "us_daily.parquet"
        table.parent.mkdir()
        frame.to_parquet(table, index=False)
        charts = build_archive(root / "charts", payloads)
        from test_qlib_yahoo_table import listing, screener_folder

        screener = screener_folder(root / "screener", [
            listing(f"S{i}", f"S{i} Corp. Common Stock", sector="Technology" if i % 2 else "Energy") for i in range(7)])
        with patch("qlib_yahoo.stooq_table._dump_script", return_value=REAL_DUMPER):
            cls.snapshot = prepare_stooq_table(
                table, root / "data", "2023-01-01", "2025-01-01",
                rule=UniverseRule(name="top3", size=3, adv_sessions=10, min_history_sessions=40, min_price=1.0),
                max_workers=2, corporate_actions=charts, groups=screener, progress=lambda _message: None)
        cls.manifest = json.loads((cls.snapshot / "manifest.json").read_text())
        cls.calendar = (cls.snapshot / "qlib_data" / "calendars" / "day.txt").read_text().split()
        cls.np = np

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def series(self, symbol, field):
        payload = self.np.fromfile(self.snapshot / "qlib_data" / "features" / symbol / f"{field}.day.bin", dtype="<f4")
        first = int(payload[0])
        return dict(zip(self.calendar[first:], payload[1:].tolist()))

    def test_split_is_repaired_and_raw_close_kept(self):
        actions = self.manifest["corporate_actions"]
        self.assertEqual(actions["unadjusted_repaired"]["count"], 1)
        self.assertEqual(self.manifest["fields"], ["open", "high", "low", "close", "volume", "factor", "close_raw", "dividend"])
        before, after = f"{self.days[299]:%Y-%m-%d}", f"{self.days[300]:%Y-%m-%d}"
        close, raw = self.series("s1", "close"), self.series("s1", "close_raw")
        self.assertLess(abs(math.log(close[after] / close[before])), 0.05)  # no x10 jump left
        self.assertAlmostEqual(raw[before] * 10, close[before], places=3)  # it really traded 10x lower
        self.assertAlmostEqual(raw[after], close[after], places=4)

    def test_etf_and_sub_dollar_stock_leave_the_universe(self):
        excluded = self.manifest["security_filter"]["excluded"]
        self.assertEqual(excluded["yahoo_type_etf"]["examples"], ["ETFX"])
        members = (self.snapshot / "qlib_data" / "instruments" / "top3.txt").read_text()
        self.assertNotIn("S5\t", members)
        self.assertNotIn("ETFX", members)
        self.assertEqual(self.manifest["universes"]["top3"]["min_price"], 1.0)

    def test_quality_scan_sees_the_repaired_prices(self):
        from qlib_yahoo.data_quality import scan, summary_lines

        report = scan(self.snapshot)
        self.assertEqual(report["universe"], "top3")
        self.assertEqual(report["price_basis"], "unadjusted close (close_raw)")
        self.assertEqual(report["totals"]["possible_unadjusted_splits"], 0)  # the x10 jump is gone
        self.assertEqual(report["totals"]["inconsistent_ohlc_bars"], 0)
        self.assertTrue(summary_lines(report)[0].startswith(str(self.snapshot)))

    def test_packed_and_unpacked_panels_are_identical(self):
        from qlib_yahoo.packed import pack_dataset
        from qlib_yahoo.panel import load_panel, membership_mask

        packed = pack_dataset(self.snapshot, self.snapshot.parent / f"{self.snapshot.name}-packed")
        fields = ("close", "volume", "close_raw", "dividend")
        a, b = load_panel(self.snapshot, fields), load_panel(packed, fields)
        self.assertEqual(a.symbols, b.symbols)
        self.assertTrue((a.dates == b.dates).all())
        for name in fields:
            self.assertTrue(self.np.array_equal(a.fields[name], b.fields[name], equal_nan=True), name)
        mask = membership_mask(a, "top3")
        self.assertTrue((mask.sum(axis=1) <= 3).all())
        self.assertNotIn("S5", a.symbols)  # never ranked, so never written to the snapshot
        with self.assertRaisesRegex(DatasetError, "no instrument S5"):
            a.column("S5")
        with self.assertRaisesRegex(DatasetError, "no field"):
            load_panel(self.snapshot, ("vwap",))

    def test_groups_are_recorded_and_verified(self):
        from qlib_yahoo.reference import dataset_groups

        record = self.manifest["groups"]
        self.assertEqual(record["coverage"]["not_in_screener"], 1)  # S7 has no screener row
        groups = dataset_groups(self.snapshot, self.manifest)
        self.assertEqual(groups["S1"]["sector"], "Technology")
        self.assertEqual(groups["S2"]["sector"], "Energy")
        self.assertEqual(groups["S7"]["sector"], "")
        path = self.snapshot / "universe" / "groups.csv"
        original = path.read_bytes()
        try:
            path.write_bytes(original.replace(b"Energy", b"Utilities"))
            with self.assertRaisesRegex(DatasetError, "groups file"):
                dataset_groups(self.snapshot, self.manifest)
        finally:
            path.write_bytes(original)
