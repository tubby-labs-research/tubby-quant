"""HistoricalData.net daily archive (lifecycle files, delisted included) -> consolidated table -> snapshot."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qlib_yahoo.dataset import DatasetError

HAVE_ARROW = bool(importlib.util.find_spec("pandas") and importlib.util.find_spec("pyarrow"))
HEADER = ("date,open,high,low,close,volume,vwap,transactions,adj_open,adj_high,adj_low,adj_close,adj_volume,adj_vwap,"
          "dividend,dividend_type,split")
SESSIONS = 300
# symbol, delisted_at, name, type, exchange, status
LIFECYCLES = [
    ("AAA", "", "Aaa Inc", "CS", "XNAS", "active"),
    ("ANET", "2023-06-30", "Old Anet Corp", "CS", "XNYS", "delisted"),
    ("ANET", "", "Arista Networks", "CS", "XNYS", "active"),
    ("BRK.B", "", "Berkshire Hathaway Class B", "CS", "XNYS", "active"),
    ("CANB", "", "Canadian Bank", "CS", "XNYS", "active"),
    ("FND", "", "Some ETF Trust", "ETF", "ARCX", "active"),
    ("WRNT", "", "Warrant Co Warrants", "WARRANT", "XNAS", "active"),
    ("OTCX", "", "Pink Sheet Co", "CS", "OTCM", "active"),
    ("SPAQ", "", "Spaq Acquisition Corp", "CS", "XNAS", "active"),
    ("GONE", "2023-03-31", "Gone Corp", "CS", "XNYS", "delisted"),
    ("OLDAA", "", "Aaa Predecessor", "CS", "XNAS", "renamed"),
]


def _file_name(symbol, delisted):
    return f"{symbol}_day{'_delisted_' + delisted if delisted else ''}.csv"


def _lifecycle_rows(days, number, events):
    """Raw bars plus the archive's adj_* columns from its formula: each row's factor is the product of later events."""
    import numpy as np

    rng = np.random.RandomState(number)
    close = 20 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days)))) * (1 + number / 5)
    volume = rng.randint(100_000, 900_000, len(days)).astype(float)
    factors = np.ones(len(days))
    for at, kind, value in events:
        if kind == "split":
            left, right = value.split(":")
            factors[:at] *= float(left) / float(right)
            close[:at] *= float(right) / float(left)  # traded prices before a 1:2 split are twice as high
        elif kind == "cash":
            factors[:at] *= (close[at - 1] - value) / close[at - 1]
    rows = []
    blank_adjusted = any(kind == "cur" for _, kind, _ in events)
    last_cur = max((at for at, kind, _ in events if kind == "cur"), default=-1)
    for k, day in enumerate(days):
        c, v, f = close[k], volume[k], factors[k]
        raw = [c * 0.99, c * 1.02, c * 0.98, c, v, c * 1.001, 1000]
        adjusted = [raw[0] * f, raw[1] * f, raw[2] * f, c * f, v / f, raw[5] * f]
        if blank_adjusted and k < last_cur:
            adjusted = [None] * 6
        dividend, kind, split = "", "", ""
        for at, event, value in events:
            if at == k:
                if event == "split":
                    split = value
                elif event == "cash":
                    dividend, kind = f"{value}", "CD"
                elif event == "cur":
                    dividend, kind = f"{value}", "CD+CUR:CAD"
        cells = [day] + [f"{x:.4f}" for x in raw[:4]] + [f"{raw[4]:.0f}", f"{raw[5]:.4f}", "1000"] + \
                [("" if x is None else f"{x:.6f}") for x in adjusted] + [dividend, kind, split]
        rows.append(",".join(cells))
    return rows


def archive(folder: Path, *, with_manifest: bool = True) -> tuple[Path, list[str]]:
    import pandas as pd

    days = [d.date().isoformat() for d in pd.bdate_range("2022-07-01", periods=SESSIONS)]
    root = folder
    (root / "day_by_symbol").mkdir(parents=True)
    listing = ["symbol,symbol_history,name,type,exchange,status,delisted_at,cik,figi"]
    files = []
    for number, (symbol, delisted, name, kind, exchange, status) in enumerate(LIFECYCLES):
        listing.append(f"{symbol},,{name},{kind},{exchange},{status},{delisted},,")
        if status == "renamed":
            continue
        span = days
        if symbol == "ANET" and delisted:
            span = [d for d in days if d <= delisted][:200]
        elif symbol == "ANET":
            span = days[210:]
        elif symbol == "GONE":
            span = [d for d in days if d <= delisted]
        events = {"AAA": [(60, "cash", 0.25), (150, "split", "1:2")], "CANB": [(100, "split", "1:2"), (200, "cur", 1.1)]}.get(symbol, [])
        rows = _lifecycle_rows(span, number, events)
        if symbol == "BRK.B":  # an isolated bad print
            cells = rows[120].split(",")
            for index in (4, 11):  # close and adj_close
                cells[index] = f"{float(cells[index]) * 3:.6f}"
            rows[120] = ",".join(cells)
        if symbol == "AAA":  # an event-only row: a dividend dated on a closed day, no prices
            rows.insert(90, "2022-11-12,,,,,0,,,,,,,,,0.05,CD,")
        path = root / "day_by_symbol" / _file_name(symbol, delisted)
        path.write_bytes(("\r\n".join([HEADER, *rows]) + "\r\n").encode("ascii"))
        files.append(path.name)
    (root / "symbols.csv").write_bytes(("\r\n".join(listing) + "\r\n").encode("utf-8"))
    if with_manifest:
        manifest = {"product": "test", "files": [{"name": f"day_by_symbol/{name}", "bytes": (root / "day_by_symbol" / name).stat().st_size,
                                                  "sha256": hashlib.sha256((root / "day_by_symbol" / name).read_bytes()).hexdigest()}
                                                 for name in files]}
        (root / "day_by_symbol" / "manifest.json").write_text(json.dumps(manifest))
    return root, days


class NamingTests(unittest.TestCase):
    def test_types_and_lifecycle_names(self):
        from qlib_yahoo.historicaldata import lifecycle_names, security_type

        self.assertEqual(security_type("CS", "Apple Inc", "XNAS"), "stock")
        self.assertEqual(security_type("ADRC", "Taiwan Semiconductor ADR", "XNYS"), "stock")
        self.assertEqual(security_type("CS", "Foo Acquisition Corp", "XNAS"), "spac")
        self.assertEqual(security_type("CS", "Pink Co", "OTCM"), "otc")
        self.assertEqual(security_type("ETF", "Trust", "ARCX"), "fund")
        self.assertEqual(security_type("PFD", "Bank Pref", "XNYS"), "preferred")
        self.assertEqual(security_type("WARRANT", "W", "XNAS"), "warrant")
        listing = [{"symbol": s, "delisted_at": d, "status": st} for s, d, _, _, _, st in LIFECYCLES]
        names = lifecycle_names(listing)
        self.assertEqual(names[("ANET", "2023-06-30")], "ANET2306")
        self.assertEqual(names[("ANET", "")], "ANET")
        self.assertEqual(names[("GONE", "2023-03-31")], "GONE")  # the only life of the symbol keeps its name
        self.assertEqual(names[("BRK.B", "")], "BRK-B")
        self.assertNotIn(("OLDAA", ""), names)  # renamed holders live in the current holder's file


@unittest.skipUnless(HAVE_ARROW, "pandas and pyarrow are optional outside the Qlib environment")
class BuildTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pandas as pd
        import pyarrow.parquet as pq

        from qlib_yahoo.historicaldata import build_table

        cls.tmp = tempfile.TemporaryDirectory()
        cls.root, cls.days = archive(Path(cls.tmp.name) / "archive")
        cls.table = Path(cls.tmp.name) / "table.parquet"
        cls.manifest = build_table(cls.root, cls.table)
        cls.frame = pq.read_table(cls.table).to_pandas()
        cls.pd = pd

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def rows(self, ticker):
        return self.frame[self.frame["ticker"] == ticker].sort_values("date").reset_index(drop=True)

    def raw(self, name):
        return self.pd.read_csv(self.root / "day_by_symbol" / name, keep_default_na=True)

    def test_types_names_and_counts(self):
        types = self.frame.groupby("ticker")["type"].first().to_dict()
        self.assertEqual(types, {"AAA": "stock", "ANET": "stock", "ANET2306": "stock", "BRK-B": "stock", "CANB": "stock",
                                 "FND": "fund", "WRNT": "warrant", "OTCX": "otc", "SPAQ": "spac", "GONE": "stock"})
        self.assertEqual(self.manifest["delisted_stocks"]["count"], 2)
        self.assertEqual(self.manifest["reused_symbols"]["count"], 1)
        self.assertTrue(self.manifest["manifest_checked"])
        self.assertEqual(self.manifest["files_without_manifest_hash"], 0)
        self.assertEqual(self.manifest["exchanges"], {"NASDAQ": 3, "NYSE": 5, "NYSEARCA": 1, "OTC": 1})

    def test_adjusted_prices_volume_vwap_and_dividends(self):
        import numpy as np

        aaa, raw = self.rows("AAA"), self.raw("AAA_day.csv").dropna(subset=["close"]).reset_index(drop=True)
        np.testing.assert_allclose(aaa["close"], raw["adj_close"])
        np.testing.assert_allclose(aaa["vwap"], raw["adj_vwap"])
        np.testing.assert_allclose(aaa["close_raw"], raw["close"])
        np.testing.assert_allclose(aaa["close"] * aaa["volume"], raw["close"] * raw["volume"], rtol=1e-5)  # traded dollars (printed decimals)
        paid = aaa[aaa["dividend"] > 0]
        self.assertEqual([d.date().isoformat() for d in paid["date"]], [self.days[60]])
        self.assertAlmostEqual(float(paid["dividend"].iloc[0]), 0.25)
        self.assertEqual(self.manifest["event_only_rows_skipped"], 1)
        self.assertNotIn("2022-11-12", {d.date().isoformat() for d in aaa["date"]})

    def test_foreign_currency_payer_is_split_adjusted_only(self):
        import numpy as np

        canb, raw = self.rows("CANB"), self.raw("CANB_day.csv")
        self.assertEqual(self.manifest["split_only_adjusted"]["examples"], ["CANB"])
        factor = np.where(np.arange(len(raw)) < 100, 0.5, 1.0)
        np.testing.assert_allclose(canb["close"], raw["close"] * factor)
        self.assertEqual(float(canb["dividend"].sum()), 0.0)  # the CAD payment is not a USD cash amount
        self.assertEqual(self.manifest["foreign_currency_dividends_skipped"], 1)

    def test_bad_print_and_changed_file(self):
        from qlib_yahoo.historicaldata import build_table

        self.assertTrue(self.pd.isna(self.rows("BRK-B")["close"][120]))
        self.assertEqual(self.manifest["isolated_spikes"]["rows"], 1)
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = archive(Path(tmp) / "a")
            path = root / "day_by_symbol" / "AAA_day.csv"
            path.write_bytes(path.read_bytes().replace(b"\r\n2022-07-05,", b"\r\n2022-07-05,1"))
            with self.assertRaisesRegex(DatasetError, "differs from the archive manifest"):
                build_table(root, Path(tmp) / "t.parquet")
            with self.assertRaisesRegex(DatasetError, "exists"):
                build_table(self.root, self.table)
            (root / "symbols.csv").unlink()
            with self.assertRaisesRegex(DatasetError, "symbols.csv"):
                build_table(root, Path(tmp) / "t2.parquet")

    def test_start_filter(self):
        from qlib_yahoo.historicaldata import build_table

        with tempfile.TemporaryDirectory() as tmp:
            manifest = build_table(self.root, Path(tmp) / "t.parquet", start=self.days[250])
            import pyarrow.parquet as pq

            frame = pq.read_table(Path(tmp) / "t.parquet").to_pandas()
            self.assertEqual(frame["date"].min().date().isoformat(), self.days[250])
            self.assertNotIn("GONE", set(frame["ticker"]))  # delisted before the start
            self.assertEqual(manifest["start"], self.days[250])


def _have_importer():
    try:
        from test_qlib_stooq_table import HAVE_ARROW as arrow, REAL_DUMPER
    except ImportError:
        return False
    return arrow and bool(importlib.util.find_spec("qlib")) and REAL_DUMPER.is_file()


@unittest.skipUnless(_have_importer(), "Qlib environment and the pinned upstream dump_bin checkout are required")
class ArchiveSnapshotTests(unittest.TestCase):
    def test_snapshot_has_vwap_and_delisted_members(self):
        import numpy as np
        from test_qlib_stooq_table import REAL_DUMPER

        from qlib_yahoo.alphalab.simulate import Workspace
        from qlib_yahoo.historicaldata import build_table
        from qlib_yahoo.stooq_table import prepare_stooq_table
        from qlib_yahoo.universe import UniverseRule

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source, days = archive(root / "sources" / "archive")
            table = root / "sources" / "hd" / "table.parquet"
            table.parent.mkdir()
            build_table(source, table)
            with patch("qlib_yahoo.stooq_table._dump_script", return_value=REAL_DUMPER):
                snapshot = prepare_stooq_table(table, root, "2022-07-01", "2023-12-30",
                                               rule=UniverseRule(name="top4", size=4, adv_sessions=10, min_history_sessions=40),
                                               max_workers=2, progress=lambda _message: None)
            manifest = json.loads((snapshot / "manifest.json").read_text())
            self.assertTrue(snapshot.name.startswith("historicaldata-table-top4-"))
            self.assertEqual(manifest["fields"], ["open", "high", "low", "close", "volume", "factor", "close_raw", "dividend", "vwap"])
            self.assertIn("Delisted companies are included", manifest["limitations"][0])
            self.assertEqual(sorted(manifest["symbols"]), ["AAA", "ANET", "ANET2306", "BRK-B", "CANB", "GONE"])
            workspace = Workspace(snapshot)
            vwap = workspace.panel.fields["vwap"]
            self.assertTrue(np.isfinite(vwap).any())


if __name__ == "__main__":
    unittest.main()
