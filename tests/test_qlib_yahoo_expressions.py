"""Expression timing checks use Qlib's own extended-window API."""

import importlib.util
import sys
import unittest
from unittest.mock import patch

from qlib_yahoo import DEFAULT_ALPHA
from qlib_yahoo.dataset import DatasetError
from qlib_yahoo.expressions import check_alpha_syntax


class FastSyntaxTests(unittest.TestCase):
    def test_valid_syntax_without_qlib_or_numeric_eval(self):
        with patch.dict(sys.modules, {"qlib": None, "qlib.data": None}), \
             patch("builtins.eval", side_effect=AssertionError("fast check must not eval")):
            self.assertIsNone(check_alpha_syntax(DEFAULT_ALPHA))
            self.assertIsNone(check_alpha_syntax(" \n  " + DEFAULT_ALPHA + "\n  "))
            self.assertIsNone(check_alpha_syntax("If(Greater($close, Ref($open, 1)), $close, 0)"))
            self.assertIsNone(check_alpha_syntax("Corr($close, $volume, 20)"))

    def test_sign_before_a_feature_term_is_rejected_early(self):
        # Qlib expression objects have no unary operators; the run would otherwise fail in the worker.
        for alpha in ("-$close", "-Ref($close, 5)", "+Mean($volume, 20)", "$open - -$close"):
            with self.subTest(alpha=alpha), self.assertRaisesRegex(DatasetError, r"write -1 \* \(\.\.\.\)"):
                check_alpha_syntax(alpha)
        for alpha in ("-1 * $close", "$close * -1", "Ref($close, 5) / -2", "-1 * (-2 + $open)"):
            with self.subTest(alpha=alpha):
                self.assertIsNone(check_alpha_syntax(alpha))

    def test_unsupported_feature_uses_user_spelling(self):
        with patch.dict(sys.modules, {"qlib": None, "qlib.data": None}):
            with self.assertRaisesRegex(DatasetError, r"\$NotAField") as caught:
                check_alpha_syntax(" $NotAField ")
        self.assertNotIn("_feature_", str(caught.exception))

    def test_rejects_gibberish_names_unsafe_and_nonmarket_expressions(self):
        invalid = (
            "gibberish", "_feature_close", "$close + _feature_open", "$vwap",
            "Foo($close, 5)", "__import__('os')", "$close.__class__",
            "If($close, Foo($open), 0)",
            "42", "1 + 2", "Power(2, 3) + $close", "2**1000000000 * $close",
            "$close[0]", "$close if True else $open", "$$close", "",
        )
        with patch.dict(sys.modules, {"qlib": None, "qlib.data": None}):
            for alpha in invalid:
                with self.subTest(alpha=alpha), self.assertRaises(DatasetError):
                    check_alpha_syntax(alpha)

    def test_operator_signatures_windows_and_literals(self):
        invalid = (
            "Abs($close, $open)", "Power($close)", "If($close, $open)",
            "Corr($close, $volume)", "Mean($close)", "Ref($close, 1, 2)",
            "Mean($close, 0)", "Ref($close, -1)", "Ref(Ref($close, -2), 2)",
            "Mean($close, 1.5)", "Mean($close, True)", "Mean($close, 4097)",
            "Mean($close, 1 + 1)", "$close + 1e309", "$close + 1e20",
            "$close + True", "Abs(1) + $close",
        )
        with patch.dict(sys.modules, {"qlib": None, "qlib.data": None}):
            for alpha in invalid:
                with self.subTest(alpha=alpha), self.assertRaises(DatasetError):
                    check_alpha_syntax(alpha)

    def test_input_size_bounds(self):
        with patch.dict(sys.modules, {"qlib": None, "qlib.data": None}):
            with self.assertRaisesRegex(DatasetError, "characters"):
                check_alpha_syntax("$close" + " " * 4091)
            with self.assertRaisesRegex(DatasetError, "syntax nodes"):
                check_alpha_syntax("$close" + "+1" * 130)


@unittest.skipUnless(importlib.util.find_spec("qlib"), "optional Qlib environment")
class ExpressionTests(unittest.TestCase):
    def test_finite_history(self):
        from qlib_yahoo.expressions import validate_alpha

        self.assertEqual(validate_alpha("$close/$open-1"), 0)
        self.assertEqual(validate_alpha("\n  $close/$open-1 \n"), 0)
        self.assertEqual(validate_alpha("Mean($close, 20)/$close-1"), 19)
        self.assertEqual(validate_alpha("Ref(Mean($close, 5), 2)"), 6)

    def test_nested_future_and_two_offsets(self):
        from qlib_yahoo.expressions import validate_alpha

        for expression in (
            "Ref($close, -1)",
            "Mean(Ref($close, -2), 5)",
            "Ref(Ref($close, -2), 1)",
            "Ref($close, -2) / Ref($open, -1)",
            "Ref(Ref($close, -2), 2)",
        ):
            with self.subTest(expression=expression), self.assertRaises(DatasetError):
                validate_alpha(expression)

    def test_unsafe_and_unbounded(self):
        from qlib_yahoo.expressions import validate_alpha

        for expression in (
            "Ref($close, 0)", "Mean($close, 0)", "EMA($close, 0.2)",
            "__import__('os')", "$close.__class__", "Foo($close, 5)",
            "Ref($close, -1 if 1 else 2)", "$$close", "Rank($close, -5)", "$vwap",
        ):
            with self.subTest(expression=expression), self.assertRaises(DatasetError):
                validate_alpha(expression)


if __name__ == "__main__":
    unittest.main()
