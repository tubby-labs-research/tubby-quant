"""Sequential FASTEXPR bindings and semantic result-version isolation."""

import importlib.util
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock, patch

HAVE_DEPS = bool(importlib.util.find_spec("numpy") and importlib.util.find_spec("pandas"))
if HAVE_DEPS:
    import numpy as np
    from qlib_yahoo.alphalab.engine import Context, REGISTRY, evaluate


@unittest.skipUnless(HAVE_DEPS, "numpy and pandas are optional outside the Qlib environment")
class AssignmentTests(unittest.TestCase):
    def setUp(self):
        self.context = Context(
            None, ["A", "B"],
            {"close": np.array([[10, 20], [30, 40]], dtype=np.float32),
             "volume": np.array([[100, 200], [300, 400]], dtype=np.float32)},
            np.ones((2, 2), dtype=bool))

    def test_exact_reproducer(self):
        context = Context(None, ["A"],
                          {"close": np.array([[10]], dtype=np.float32),
                           "volume": np.array([[100]], dtype=np.float32)},
                          np.ones((1, 1), dtype=bool))
        np.testing.assert_array_equal(evaluate("a=close;b=a;a=volume;a-b", context), [[90]])

    def test_sequential_bindings_against_independent_expected_arrays(self):
        cases = [
            ("a=close;b=a;a=volume;a-b", [[90, 180], [270, 360]]),
            ("a=close;b=a+1;a=volume;(a+1)-b", [[90, 180], [270, 360]]),
            ("a=close;b=abs(a);a=volume;abs(a)-b", [[90, 180], [270, 360]]),
            ("n=1;b=close*n;n=2;close*n-b", [[10, 20], [30, 40]]),
            ("d=1;b=ts_delay(close,d);d=0;ts_delay(close,d)-b",
             [[np.nan, np.nan], [20, 20]]),
            ("b=close;close=volume;close-b", [[90, 180], [270, 360]]),
            ("close=close+1;close=close+1;close", [[12, 22], [32, 42]]),
            ("a=close;a=a+1;a=a+1;a", [[12, 22], [32, 42]]),
            ("n=1;n=n+1;close*n", [[20, 40], [60, 80]]),
            ("a=close;b=a;a=a+1;b", [[10, 20], [30, 40]]),
            ("a=close+volume;b=a*2;b-a", [[110, 220], [330, 440]]),
            ("(close+1)+(close+1)", [[22, 42], [62, 82]]),
        ]
        for expression, expected in cases:
            with self.subTest(expression=expression):
                np.testing.assert_allclose(evaluate(expression, self.context), expected, rtol=0, atol=0,
                                           equal_nan=True)
        np.testing.assert_array_equal(self.context.fields["close"], [[10, 20], [30, 40]])

    def test_common_subexpressions_reused_within_rhs_and_result(self):
        function, kinds, defaults = REGISTRY["abs"]
        for expression in ("abs(close)+abs(close)", "a=abs(close)+abs(close);a"):
            with self.subTest(expression=expression), patch.dict(REGISTRY):
                counted = Mock(wraps=function)
                REGISTRY["abs"] = (counted, kinds, defaults)
                np.testing.assert_array_equal(evaluate(expression, self.context), [[20, 40], [60, 80]])
                self.assertEqual(counted.call_count, 1)

    def test_repeated_operator_recomputed_after_assignment(self):
        function, kinds, defaults = REGISTRY["abs"]
        with patch.dict(REGISTRY):
            counted = Mock(wraps=function)
            REGISTRY["abs"] = (counted, kinds, defaults)
            result = evaluate("a=close;b=abs(a)+abs(a);a=volume;abs(a)+abs(a)-b", self.context)
            np.testing.assert_array_equal(result, [[180, 360], [540, 720]])
            self.assertEqual(counted.call_count, 2)


@unittest.skipUnless(HAVE_DEPS, "numpy and pandas are optional outside the Qlib environment")
class AssignmentVersionTests(unittest.TestCase):
    def test_old_results_remain_readable_but_are_not_automatically_reused(self):
        from qlib_yahoo.alphalab import batch
        from qlib_yahoo.alphalab.simulate import ENGINE_VERSION, Settings

        self.assertEqual(ENGINE_VERSION, "alphalab-3")
        settings = Settings().validate()
        expression = "a=close;b=a;a=volume;a-b"
        source_hash = "synthetic-source"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library_path = root / "library.sqlite"
            library = batch.Library(library_path)
            old_result = {"engine": "alphalab-1", "expression": expression,
                          "settings": {"universe": settings.universe, "delay": settings.delay},
                          "daily": {"dates": ["2024-01-02"], "pnl": [0.0]}}
            with patch.object(batch, "ENGINE_VERSION", "alphalab-1"):
                old_id = batch.alpha_id(expression, settings, source_hash)
                library.save(old_id, old_result, source_hash)
            new_id = batch.alpha_id(expression, settings, source_hash)
            self.assertNotEqual(old_id, new_id)
            self.assertIsNone(library.get(new_id))
            self.assertEqual(library.get(old_id), old_result)
            self.assertEqual([r["id"] for r in library.rows()], [old_id])
            self.assertEqual(library.pnl_series(source_hash, settings, set()), {})
            with closing(library.connect()) as db:
                before = tuple(db.execute("SELECT * FROM alphas WHERE id=?", (old_id,)).fetchone())
            # A miss must proceed to loading. Stop there so this test never reads market data.
            with patch.object(batch, "source_identity", return_value={"source_sha256": source_hash}), \
                    patch.object(batch, "Workspace", side_effect=RuntimeError("new simulation required")) as load:
                with self.assertRaisesRegex(RuntimeError, "new simulation required"):
                    batch.run_batch(root / "synthetic", [expression], settings,
                                    data_root=root, library=library_path)
                load.assert_called_once_with(root / "synthetic")
            new_result = {**old_result, "engine": ENGINE_VERSION}
            library.save(new_id, new_result, source_hash)
            with closing(library.connect()) as db:
                new_before = tuple(db.execute("SELECT * FROM alphas WHERE id=?", (new_id,)).fetchone())
            expected_engines = {old_id: "alphalab-1", new_id: "alphalab-3"}
            for dataset_filter in (None, source_hash):
                with self.subTest(dataset_filter=dataset_filter):
                    self.assertEqual({row["id"]: row["engine"] for row in library.rows(dataset_filter)},
                                     expected_engines)
            with closing(library.connect()) as db:
                after = tuple(db.execute("SELECT * FROM alphas WHERE id=?", (old_id,)).fetchone())
                new_after = tuple(db.execute("SELECT * FROM alphas WHERE id=?", (new_id,)).fetchone())
            self.assertEqual(after, before)
            self.assertEqual(new_after, new_before)
            self.assertEqual(library.get(old_id), old_result)
            self.assertEqual(library.get(new_id), new_result)


if __name__ == "__main__":
    unittest.main()
