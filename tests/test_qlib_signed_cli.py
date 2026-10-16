"""Signed CLI selection retained independently of desktop frontends."""

from pathlib import Path
import unittest
from unittest.mock import patch

from qlib_yahoo.__main__ import build_parser, main


class SignedCliTests(unittest.TestCase):
    def test_cli_options_and_default(self):
        parser = build_parser()
        defaults = parser.parse_args(["backtest", "--alpha", "$close"])
        self.assertEqual((defaults.strategy_mode, defaults.gross_exposure, defaults.borrow_rate),
                         ("long-only", 1.0, 0.004))
        selected = parser.parse_args(["backtest", "--alpha", "$close", "--strategy-mode", "signed",
                                      "--gross-exposure", "0.5", "--borrow-rate", "0.006"])
        self.assertEqual((selected.strategy_mode, selected.gross_exposure, selected.borrow_rate),
                         ("signed", 0.5, 0.006))

    def test_cli_passes_signed_options_to_runner(self):
        with patch("qlib_yahoo.__main__.resolve_dataset", return_value=Path("C:/prepared")), \
                patch("qlib_yahoo.__main__.run_backtest") as runner:
            self.assertEqual(main(["backtest", "--alpha", "$close", "--strategy-mode", "signed",
                                   "--gross-exposure", "0.5"]), 0)
        self.assertEqual(runner.call_args.kwargs["strategy_mode"], "signed")
        self.assertEqual(runner.call_args.kwargs["gross_exposure"], 0.5)
        self.assertEqual(runner.call_args.kwargs["annual_borrow_rate"], 0.004)


if __name__ == "__main__":
    unittest.main()
