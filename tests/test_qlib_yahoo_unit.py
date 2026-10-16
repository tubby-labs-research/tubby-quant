"""Offline checks for the explicit-symbol Yahoo/Qlib runner."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qlib_yahoo import BENCHMARK, DEFAULT_DATA_ROOT, DEFAULT_SYMBOLS, default_data_root
from qlib_yahoo.__main__ import build_parser
from qlib_yahoo.analysis import _group_returns, _score_turnover
from qlib_yahoo.dataset import DatasetError, parse_symbols, prepare, resolve_dataset


class YahooRunnerUnitTests(unittest.TestCase):
    def test_default_data_is_off_c_drive_and_explicit(self):
        self.assertEqual(Path(DEFAULT_DATA_ROOT), Path(default_data_root()))
        with patch.dict(os.environ, {"TUBBY_QLIB_DATA_ROOT": ""}):
            expected = r"D:\Qlib" if os.name == "nt" else str(Path.home() / "Qlib")
            self.assertEqual(default_data_root(), expected)
        with patch.dict(os.environ, {"TUBBY_QLIB_DATA_ROOT": "~/elsewhere"}):
            self.assertEqual(default_data_root(), str(Path.home() / "elsewhere"))
        self.assertEqual(BENCHMARK, "SPY")
        self.assertGreaterEqual(len(DEFAULT_SYMBOLS), 6)
        self.assertNotIn(BENCHMARK, DEFAULT_SYMBOLS)

    def test_symbol_parser(self):
        self.assertEqual(
            parse_symbols(" aapl,msft,nvda,amzn,googl,jpm "),
            ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "JPM"),
        )
        self.assertEqual(parse_symbols("AAPL,MSFT,NVDA,AMZN,BRK-B,JPM")[-2], "BRK-B")
        self.assertEqual(parse_symbols("AAPL,MSFT,NVDA,AMZN,JPM,SPY")[-1], "SPY")
        for value in ("AAPL,AAPL", "AAPL,MSFT", "AAPL,../../bad,MSFT", "AAPL,MSFT,NVDA,AMZN,JPM,CON"):
            with self.subTest(value=value), self.assertRaises(DatasetError):
                parse_symbols(value)

    def test_backtest_command_accepts_inline_alpha(self):
        args = build_parser().parse_args(["backtest", "--alpha", "Ref($close, 1)"])
        self.assertEqual(args.alpha, "Ref($close, 1)")

    @unittest.skipUnless(importlib.util.find_spec("pandas"), "pandas is installed only in the Qlib environment")
    def test_group_returns_preserve_score_order_and_spread(self):
        import pandas as pd

        index = pd.MultiIndex.from_product(
            [[f"STOCK{i}" for i in range(10)], [pd.Timestamp("2024-01-02")]],
            names=["instrument", "datetime"],
        )
        frame = pd.DataFrame({"score": range(10), "label": range(10)}, index=index)
        groups = _group_returns(frame)
        self.assertEqual(groups.iloc[0]["group_1"], 8.5)
        self.assertEqual(groups.iloc[0]["group_5"], 0.5)
        self.assertEqual(groups.iloc[0]["long_short_spread"], 8.0)

    @unittest.skipUnless(importlib.util.find_spec("pandas"), "pandas is installed only in the Qlib environment")
    def test_score_turnover_uses_previous_session(self):
        import pandas as pd

        index = pd.MultiIndex.from_product(
            [[f"STOCK{i}" for i in range(10)], pd.to_datetime(["2024-01-02", "2024-01-03"])],
            names=["instrument", "datetime"],
        )
        scores = pd.DataFrame({"score": [value for i in range(10) for value in (i, i)]}, index=index)
        turnover = _score_turnover(scores)
        self.assertEqual(len(turnover), 1)
        self.assertEqual(turnover.iloc[0]["top_score_bucket_turnover"], 0.0)
        self.assertEqual(turnover.iloc[0]["bottom_score_bucket_turnover"], 0.0)

    def test_no_latest_dataset_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(DatasetError, "run 'prepare' first"):
                resolve_dataset(Path(directory), None)

    def test_completed_snapshot_resumes_only_matching_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            snapshot = root / "datasets" / "example"
            snapshot.mkdir(parents=True)
            symbols = DEFAULT_SYMBOLS
            (snapshot / "request.json").write_text(json.dumps({
                "start_inclusive": "2022-01-01", "end_exclusive": "2025-01-01",
                "symbols": list(symbols), "benchmark": None,
            }), encoding="utf-8")
            (snapshot / "manifest.json").write_text("{}", encoding="utf-8")
            with patch("qlib_yahoo.dataset.ensure_upstream", return_value=root / "dump_bin.py"):
                self.assertEqual(prepare(root, "2022-01-01", "2025-01-01", symbols, resume_snapshot=snapshot), snapshot)
                with self.assertRaisesRegex(DatasetError, "does not match"):
                    prepare(root, "2020-01-01", "2025-01-01", symbols, resume_snapshot=snapshot)

    def test_legacy_spy_request_cannot_resume_under_new_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "datasets" / "legacy-incomplete"
            snapshot.mkdir(parents=True)
            original = {
                "start_inclusive": "2022-01-01", "end_exclusive": "2025-01-01",
                "symbols": list(DEFAULT_SYMBOLS), "benchmark": BENCHMARK,
            }
            request_path = snapshot / "request.json"
            request_path.write_text(json.dumps(original), encoding="utf-8")
            with patch("qlib_yahoo.dataset.ensure_upstream") as upstream, \
                 patch("qlib_yahoo.dataset.fetch_yahoo") as fetch:
                with self.assertRaisesRegex(DatasetError, "does not match"):
                    prepare(root, "2022-01-01", "2025-01-01", DEFAULT_SYMBOLS, resume_snapshot=snapshot)
            upstream.assert_not_called()
            fetch.assert_not_called()
            self.assertEqual(json.loads(request_path.read_text(encoding="utf-8")), original)

    @unittest.skipUnless(importlib.util.find_spec("pandas"), "pandas is installed only in the Qlib environment")
    def test_adjclose_normalization_has_known_answer(self):
        import pandas as pd
        from qlib_yahoo.dataset import normalize_frame

        frame = pd.DataFrame([
            {"date": "2024-01-02", "symbol": "AAPL", "open": 10., "high": 11., "low": 9., "close": 10., "volume": 100., "adjclose": 5.},
            {"date": "2024-01-03", "symbol": "AAPL", "open": 12., "high": 13., "low": 11., "close": 12., "volume": 120., "adjclose": 6.},
        ])
        normalized = normalize_frame(frame)
        self.assertEqual(normalized["close"].tolist(), [1.0, 1.2])
        self.assertEqual(normalized["volume"].tolist(), [1000.0, 1200.0])
        self.assertEqual(normalized["factor"].tolist(), [0.1, 0.1])


if __name__ == "__main__":
    unittest.main()
