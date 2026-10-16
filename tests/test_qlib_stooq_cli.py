"""Command-line contract for the Stooq importer without touching source data."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qlib_yahoo.__main__ import main
from qlib_yahoo.dataset import DatasetError


class StooqCliTests(unittest.TestCase):
    def test_requires_dates_and_explicit_symbol_selection(self):
        cases = (
            ["import-stooq", "--symbols", "AAA"],
            ["import-stooq", "--start", "2024-01-01", "--end", "2025-01-01"],
            ["import-stooq", "--end", "2025-01-01", "--symbols", "AAA"],
            ["import-stooq", "--start", "2024-01-01", "--symbols", "AAA"],
        )
        for args in cases:
            with self.subTest(args=args), patch("qlib_yahoo.stooq.prepare_stooq") as importer:
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as result:
                        main(args)
                self.assertEqual(result.exception.code, 2)
                importer.assert_not_called()

    def test_symbols_file_ignores_blank_lines_and_comments(self):
        with tempfile.TemporaryDirectory() as temporary:
            symbols_file = Path(temporary) / "symbols.txt"
            symbols_file.write_text("# chosen names\n\n AAA \n  # omitted\nMSFT\n", encoding="utf-8")
            with patch("qlib_yahoo.stooq.prepare_stooq") as importer:
                result = main(["import-stooq", "--start", "2024-01-01", "--end", "2025-01-01",
                               "--symbols-file", str(symbols_file)])
            self.assertEqual(result, 0)
            self.assertEqual(importer.call_args.args[2], ("AAA", "MSFT"))

    def test_default_does_not_make_latest_and_passes_paths_and_worker_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            data_root = Path(temporary) / "output"
            base = ["--data-root", str(data_root), "import-stooq", "--source", str(source),
                    "--start", "2024-01-01", "--end", "2025-01-01", "--symbols", "AAA,MSFT",
                    "--max-workers", "4"]
            with patch("qlib_yahoo.stooq.prepare_stooq") as importer:
                self.assertEqual(main(base), 0)
                importer.assert_called_once_with(source, data_root, ("AAA", "MSFT"),
                                                 "2024-01-01", "2025-01-01",
                                                 make_latest=False, max_workers=4)
                importer.reset_mock()
                self.assertEqual(main([*base, "--make-latest"]), 0)
                importer.assert_called_once_with(source, data_root, ("AAA", "MSFT"),
                                                 "2024-01-01", "2025-01-01",
                                                 make_latest=True, max_workers=4)

    def test_explicit_spy_is_passed_as_a_trade_symbol(self):
        with patch("qlib_yahoo.stooq.prepare_stooq") as importer:
            self.assertEqual(main(["import-stooq", "--start", "2024-01-01", "--end", "2025-01-01",
                                   "--symbols", "AAA,SPY"]), 0)
        self.assertEqual(importer.call_args.args[2], ("AAA", "SPY"))

    def test_importer_dataset_error_returns_friendly_failure(self):
        with patch("qlib_yahoo.stooq.prepare_stooq", side_effect=DatasetError("invalid Stooq input")):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                result = main(["import-stooq", "--start", "2024-01-01", "--end", "2025-01-01",
                               "--symbols", "AAA"])
        self.assertEqual(result, 1)
        self.assertEqual(stderr.getvalue().strip(), "Qlib workflow failed: invalid Stooq input")


if __name__ == "__main__":
    unittest.main()
