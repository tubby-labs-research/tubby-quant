"""Production selection precedes Qlib imports; reference/probes stay explicit."""
import inspect
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from qlib_yahoo.backtest import run_backtest
from qlib_yahoo.dataset import DatasetError


class ProfileOptionsTests(unittest.TestCase):
    def test_default_is_auto(self):
        self.assertEqual(inspect.signature(run_backtest).parameters["runtime_profile"].default, "auto")

    def test_default_signed_selects_profile_before_engine_import(self):
        options = {"_columnar_history": True, "_flat_indicators": True,
                   "_buffered_history": True, "_index_reuse": True}
        with patch("qlib_yahoo.fast_runtime.resolve_fast_profile", return_value=(options, {"name": "accepted-v1"})) as select, \
                patch("qlib_yahoo.runtime.ensure_signed_runtime", side_effect=LookupError("engine boundary")) as engine, \
                patch.object(sys, "dont_write_bytecode", False):
            with self.assertRaisesRegex(LookupError, "engine boundary"):
                run_backtest(Path("unused"), "$close", strategy_mode="signed")
            select.assert_called_once_with()
            engine.assert_called_once_with()
            self.assertTrue(sys.dont_write_bytecode)

    def test_native_reference_and_memory_do_not_select_optional_dependencies(self):
        variants = ({"runtime_profile": "native"}, {"execution_profile": "reference"},
                    {"history_mode": "memory"}, {"_signed_pin_path": "pin.json"})
        with patch("qlib_yahoo.fast_runtime.resolve_fast_profile", side_effect=AssertionError("optional activation")) as select, \
                patch("qlib_yahoo.runtime.ensure_signed_runtime", side_effect=LookupError("engine boundary")):
            for kwargs in variants:
                with self.subTest(kwargs=kwargs), self.assertRaisesRegex(LookupError, "engine boundary"):
                    run_backtest(Path("unused"), "$close", strategy_mode="signed", **kwargs)
            select.assert_not_called()

    def test_explicit_experiment_is_not_combined_with_production_recipe(self):
        variants = ({"_columnar_history": True}, {"_flat_indicators": True},
                    {"_buffered_history": True}, {"_index_reuse": True},
                    {"_row_range_manifest": "old-receipt.json"},
                    {"_installed_range_root": "installed-helper"},
                    {"_decision_gzip_level": 6}, {"_gzip_path": "codec"},
                    {"_ordered_session": True, "_compiler_path": "compiler"})
        with patch("qlib_yahoo.fast_runtime.resolve_fast_profile", side_effect=AssertionError("optional activation")) as select, \
                patch("qlib_yahoo.runtime.ensure_signed_runtime", side_effect=LookupError("engine boundary")):
            for kwargs in variants:
                with self.subTest(kwargs=kwargs), self.assertRaisesRegex(LookupError, "engine boundary"):
                    run_backtest(Path("unused"), "$close", strategy_mode="signed", **kwargs)
            select.assert_not_called()

    def test_invalid_profile_and_installed_helper_reject_before_import_or_data(self):
        invalid = [{"runtime_profile": value} for value in (None, "", "fast", True, [], {})]
        invalid += [{"_installed_range_root": value, "strategy_mode": "signed"}
                    for value in ("", False, 3, [], Path("helper"))]
        invalid += [{"_installed_range_root": "helper"},
                    {"_installed_range_root": "helper", "strategy_mode": "signed", "execution_profile": "reference"},
                    {"_installed_range_root": "helper", "strategy_mode": "signed", "_row_range_manifest": "prototype"}]
        with patch("qlib_yahoo.fast_runtime.resolve_fast_profile", side_effect=AssertionError("optional activation")) as select, \
                patch("qlib_yahoo.runtime.ensure_signed_runtime", side_effect=AssertionError("engine activation")) as engine, \
                patch("qlib_yahoo.backtest.verify_dataset", side_effect=AssertionError("dataset access")) as verify:
            for kwargs in invalid:
                with self.subTest(kwargs=kwargs), self.assertRaises(DatasetError):
                    run_backtest(Path("unused"), "$close", **kwargs)
            select.assert_not_called()
            engine.assert_not_called()
            verify.assert_not_called()

    def test_long_only_preserves_normal_alpha_validation(self):
        with patch("qlib_yahoo.fast_runtime.resolve_fast_profile", side_effect=AssertionError("optional activation")) as select, \
                patch("qlib_yahoo.backtest.validate_alpha", side_effect=LookupError("normal validation")):
            with self.assertRaisesRegex(LookupError, "normal validation"):
                run_backtest(Path("unused"), "$close")
            select.assert_not_called()


if __name__ == "__main__":
    unittest.main()
