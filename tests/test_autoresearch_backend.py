"""Offline adapter integration against real sealed Qlib-format artificial data."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from autoresearch.backend import AlphaLabBackend, create_demo_dataset

HAVE_NUMERIC = bool(importlib.util.find_spec("numpy") and importlib.util.find_spec("pandas"))


@unittest.skipUnless(HAVE_NUMERIC, "requires local numpy and pandas")
class BackendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = create_demo_dataset(self.root / "dataset")
        self.config = {"dataset": str(self.dataset), "universe": "TOP8", "cost_bps": 5,
                       "development_start": "2023-06-01", "development_end": "2023-12-29",
                       "validation_start": "2024-01-01", "validation_end": "2024-09-30",
                       "holdout_start": "2024-10-01", "warmup": 60, "truncation": 0}
        self.backend = AlphaLabBackend(self.config, self.root / "runtime")

    def test_bounds_cost_delay_and_setting_validation(self):
        invalid = [{"cost_bps": None}, {"cost_bps": float("nan")}, {"delay": 0}, {"delay": True},
                   {"validation_start": "2023-12-29"}, {"holdout_start": "2024-09-30"},
                   {"development_start": "20230601"}, {"validation_end": "2024-02-30"},
                   {"warmup": 0}, {"warmup": 2001}, {"pasteurization": "false"}, {"decay": 513}]
        for change in invalid:
            with self.subTest(change=change), self.assertRaises((ValueError, RuntimeError)):
                AlphaLabBackend({**self.config, **change}, self.root / "runtime")
        costless = dict(self.config)
        del costless["cost_bps"]
        with self.assertRaisesRegex(ValueError, "cost_bps"):
            AlphaLabBackend(costless, self.root / "runtime")

    def test_source_identity_is_verified_portable_and_setting_sensitive(self):
        identity = self.backend.identity()
        self.assertNotIn(str(self.root), json.dumps(identity))
        moved = self.root / "moved"
        shutil.copytree(self.dataset, moved)
        other = AlphaLabBackend({**self.config, "dataset": str(moved)}, self.root / "elsewhere")
        self.assertEqual(identity, other.identity())
        changed = AlphaLabBackend({**self.config, "cost_bps": 6}, self.root / "runtime")
        self.assertNotEqual(identity, changed.identity())
        changed = AlphaLabBackend({**self.config, "holdout_start": "2024-11-01"}, self.root / "runtime")
        self.assertEqual(identity, changed.identity())  # Untouched holdout is not scored.
        feature = moved / "qlib_data/features/syn00/close.day.bin"
        feature.write_bytes(feature.read_bytes()[:-4])
        with self.assertRaisesRegex(RuntimeError, "differs"):
            other.identity()

    def test_real_validation_rejects_syntax_and_missing_runtime_fields(self):
        self.assertEqual(self.backend.validate("rank(ts_delta(close,5))"), [])
        self.assertEqual(self.backend.validate("rank(returns) + rank(adv20)"), [])
        self.assertTrue(self.backend.validate("rank(close"))
        self.assertTrue(self.backend.validate("rank(cap)"))
        self.assertTrue(self.backend.validate("group_rank(close, sector)"))
        self.assertEqual(self.backend.evaluate("rank(cap)")["status"], "error")

    def test_actual_engine_costs_caching_and_holdout_separation(self):
        from qlib_yahoo.alphalab import batch
        from qlib_yahoo.alphalab.simulate import Workspace

        seen = []
        original = Workspace.context
        def tracked(workspace, first, last, members):
            context = original(workspace, first, last, members)
            seen.append(str(context.dates[-1]))
            return context

        with patch.object(Workspace, "context", tracked):
            first = self.backend.evaluate("rank(ts_delta(close,5))")
        self.assertEqual(first["status"], "ok", first)
        self.assertFalse(first["cached"])
        self.assertEqual(seen, ["2023-12-29", "2024-09-30"])
        self.assertEqual(first["settings"]["cost_bps"], 5)
        self.assertFalse(first["settings"]["holdout_evaluated"])
        for period in ("development", "validation"):
            self.assertGreater(first["periods"][period]["days"], 100)
            self.assertLessEqual(first["periods"][period]["end"], self.config[period + "_end"])
        artifact = self.root / "runtime" / first["artifacts"][0]
        raw = json.loads(artifact.read_text())
        self.assertLessEqual(max(raw["validation"]["daily"]["dates"]), "2024-09-30")
        self.assertNotIn("2024-10", artifact.read_text())
        self.assertNotIn(str(self.root), json.dumps(first))
        json.dumps(first, allow_nan=False)
        with patch.object(batch, "_run_one", side_effect=AssertionError("cache should avoid evaluation")):
            second = self.backend.evaluate("rank(ts_delta(close,5))")
        self.assertEqual(second["status"], "ok", second)
        self.assertTrue(second["cached"])
        self.assertEqual(second["metrics"], first["metrics"])
        gross = AlphaLabBackend({**self.config, "cost_bps": 0}, self.root / "gross").evaluate("rank(ts_delta(close,5))")
        self.assertEqual(gross["status"], "ok", gross)
        for period in ("development", "validation"):
            self.assertLess(first["metrics"][period]["pnl"], gross["metrics"][period]["pnl"])
        self.assertFalse((self.root / "alphalab").exists())

    def test_native_failed_cache_retries_while_preserving_successful_period(self):
        from qlib_yahoo.alphalab import batch
        real_simulate = batch.simulate
        calls = []
        fail_validation = True
        def transient(expression, settings, workspace):
            nonlocal fail_validation
            calls.append(settings.start)
            if settings.start == self.config["validation_start"] and fail_validation:
                fail_validation = False
                raise RuntimeError("temporary engine failure")
            return real_simulate(expression, settings, workspace)
        with patch.object(batch, "simulate", side_effect=transient):
            first = self.backend.evaluate("rank(ts_delta(close,5))")
            self.assertEqual(first["status"], "error", first)
            self.assertNotEqual(first.get("error_kind"), "expression")
            self.assertEqual(len(calls), 2)  # A fresh failure is not retried here.
            second = self.backend.evaluate("rank(ts_delta(close,5))")
            self.assertEqual(second["status"], "ok", second)
            self.assertEqual(calls, ["2023-06-01", "2024-01-01", "2024-01-01"])
            third = self.backend.evaluate("rank(ts_delta(close,5))")
            self.assertEqual(third["status"], "ok", third)
            self.assertTrue(third["cached"])
            self.assertEqual(len(calls), 3)

    def test_runtime_expression_error_is_available_for_model_repair(self):
        expression = "rank(ts_delay(close,-1))"
        self.assertTrue(self.backend.validate(expression))
        result = self.backend.evaluate(expression)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_kind"], "expression")
        self.assertIn("window", result["error"])

    def test_changed_holdout_prices_cannot_change_research_metrics(self):
        import numpy as np
        from qlib_yahoo.dataset import tree_sha256

        baseline = self.backend.evaluate("rank(ts_delta(close,5))")
        dates = (self.dataset / "qlib_data/calendars/day.txt").read_text().splitlines()
        first_holdout = dates.index("2024-10-01")
        for feature in (self.dataset / "qlib_data/features").glob("*/close.day.bin"):
            values = np.fromfile(feature, dtype="<f4")
            values[first_holdout + 1:] *= 100
            values.tofile(feature)
        manifest_path = self.dataset / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["qlib_binary_tree_sha256"] = tree_sha256(self.dataset / "qlib_data")[0]
        manifest_path.write_text(json.dumps(manifest))
        changed = self.backend.evaluate("rank(ts_delta(close,5))")
        self.assertEqual(changed["status"], "ok", changed)
        self.assertNotEqual(baseline["data_identity"], changed["data_identity"])
        self.assertEqual(baseline["metrics"], changed["metrics"])
        self.assertFalse(changed["cached"])

    def test_errors_remain_json_and_suppress_absolute_paths(self):
        self.dataset.joinpath("manifest.json").unlink()
        result = self.backend.evaluate("rank(close)")
        self.assertEqual(result["status"], "error")
        self.assertNotIn(str(self.root), json.dumps(result))
        json.dumps(result, allow_nan=False)

    def test_demo_is_deterministic_idempotent_and_does_not_overwrite(self):
        second = create_demo_dataset(self.root / "second")
        other = AlphaLabBackend({**self.config, "dataset": str(second)}, self.root / "runtime")
        self.assertEqual(self.backend.identity()["data_identity"], other.identity()["data_identity"])
        self.assertEqual(create_demo_dataset(self.dataset), self.dataset)
        (self.root / "occupied").mkdir()
        (self.root / "occupied/user.txt").write_text("keep")
        with self.assertRaisesRegex(ValueError, "empty"):
            create_demo_dataset(self.root / "occupied")


if __name__ == "__main__":
    unittest.main()
