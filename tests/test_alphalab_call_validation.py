"""Public FASTEXPR call signatures, checked without arrays or evaluation."""
import importlib.util
import unittest
from unittest.mock import Mock, patch


@unittest.skipUnless(importlib.util.find_spec("numpy"), "optional Lab runtime")
class CallValidationTests(unittest.TestCase):
    def setUp(self):
        from qlib_yahoo.alphalab import engine

        self.engine = engine

    def test_missing_and_excess_arguments_and_unknown_keywords(self):
        invalid = {
            "ts_mean(sec_assets)": "missing argument 2",
            "rank(sec_assets,bogus=2)": "no parameter 'bogus'",
            "rank()": "missing argument 1",
            "rank(close,1,2)": "at most 2 arguments",
            "ts_corr(close,open)": "missing argument 3",
            "ts_mean(close,3,d=4)": "no parameter 'd'",
            "ts_mean(close,D=3)": "missing argument 2",
            "ts_mean(x=close,d=3)": "missing argument 1",
            "group_rank(close)": "missing argument 2",
            "group_rank(close,market,members=1)": "no parameter 'members'",
            "normalize(close,1,2,3)": "at most 3 arguments",
            "add(close)": "two or more series",
            "divide(close,open,bogus=1)": "no keywords",
            "not(close,open)": "one series",
            "not(close,bogus=1)": "no keywords",
            "min(close)": "two or more series",
            "max(close,open,bogus=1)": "no keywords",
        }
        for expression, message in invalid.items():
            with self.subTest(expression=expression):
                self.assertIn(message, "; ".join(self.engine.check_expression(expression, fields={"sec_assets"})))

    def test_public_numeric_keywords_options_aliases_and_implicit_masks(self):
        valid = (
            "ts_mean(close,d=3)", "ts_corr(close,open,d=3)",
            "ts_regression(close,open,d=3,LAG=1,rettype=0)",
            "delay(close,d=3)", "correlation(close,open,d=3)",
            "rank(close)", "rank(close,1)", "rank(close,RATE=1)",
            "normalize(close,useStd=true,limit=1)", "demean(close,USESTD=true)",
            "scale(close,1,2,3)", "bucket(rank(close),range='0,1,0.1')",
            "group_rank(close,market)", "group_mean(close,bucket(rank(close)))",
            "clamp(close,lower=-1,upper=1)", "clip(close,-1,1)",
            "min(close,open,1)", "max(close,open)", "add(close,open,1)",
            "not(close)", "ts_mean(close,3)+rank(volume)",
            # Preserve the evaluator's existing option-override behavior.
            "rank(close,1,rate=2)", "rank(close,rate=1,rate=2)",
            "a=ts_mean(close,d=3);rank(a)",
        )
        for expression in valid:
            with self.subTest(expression=expression):
                self.assertEqual(self.engine.check_expression(expression), [])

    def test_every_registry_operator_and_alias_uses_public_metadata(self):
        for name in (*self.engine.REGISTRY, *self.engine.ALIASES):
            canonical = self.engine.ALIASES.get(name, name)
            kinds = self.engine.REGISTRY[canonical][1]
            args = ["close", "open"] if kinds == ("*",) else [
                {"x": "close", "n": "3", "g": "market", "s": "'text'"}[kind] for kind in kinds]
            with self.subTest(name=name):
                self.assertEqual(self.engine.check_expression(f"{name}({','.join(args)})"), [])

    def test_nested_and_assignment_calls_are_checked_without_evaluation(self):
        function = Mock(side_effect=AssertionError("operator evaluated"))
        _, kinds, defaults = self.engine.REGISTRY["ts_mean"]
        with patch.dict(self.engine.REGISTRY, {"ts_mean": (function, kinds, defaults)}), \
                patch.object(self.engine.Evaluator, "run", side_effect=AssertionError("program evaluated")), \
                patch.object(self.engine.np, "full", side_effect=AssertionError("array allocated")):
            self.assertEqual(self.engine.check_expression("ts_mean(sec_assets,d=3)", fields={"sec_assets"}), [])
            problems = self.engine.check_expression("a=ts_mean(sec_assets);rank(a,bogus=2)", fields={"sec_assets"})
        self.assertTrue(any("missing argument" in p for p in problems))
        self.assertTrue(any("no parameter" in p for p in problems))
        function.assert_not_called()


if __name__ == "__main__":
    unittest.main()
