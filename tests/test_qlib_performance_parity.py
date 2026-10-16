"""End-to-end parity of the bounded Qlib execution and storage paths."""

import importlib.util
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from qlib_yahoo.dataset import DatasetError, tree_sha256
from qlib_yahoo.packed import pack_dataset
from qlib_yahoo.runtime import PIN, ROOT, platform_pin


CHILD = "QLIB_PERFORMANCE_PARITY_SIGNED_CHILD"


def _signed_available():
    try:
        pin = platform_pin(json.loads(PIN.read_text(encoding="utf-8")))
        return (sys.version_info[:2] == (3, 12)
                and (ROOT / pin["local_runtime"] / "qlib/_tubby_build.json").is_file())
    except (OSError, ValueError, KeyError, DatasetError):
        return False


@unittest.skipUnless(importlib.util.find_spec("qlib"), "Qlib environment unavailable")
class ParityTests(unittest.TestCase):
    def setUp(self):
        from tests.test_qlib_yahoo_execution import _snapshot

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name).resolve()
        self.unpacked = _snapshot(root, sessions_count=35, include_benchmark=False)
        manifest_path = self.unpacked / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        digest, count = tree_sha256(self.unpacked / "qlib_data")
        manifest.update(status="ready", qlib_binary_tree_sha256=digest,
                        qlib_binary_file_count=count)
        manifest.pop("benchmark")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.packed = pack_dataset(self.unpacked, root / "data" / "packed")
        sealed = json.loads((self.packed / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual((sealed["status"], sealed["qlib_binary_tree_sha256"],
                          sealed["qlib_binary_file_count"]), ("ready", digest, count))

    def run_variants(self, *, mode, transform="raw"):
        from qlib_yahoo.backtest import run_backtest
        import pandas as pd

        results = {}
        for storage, dataset in (("unpacked", self.unpacked), ("packed", self.packed)):
            for profile in ("reference", "fast"):
                with self.subTest(mode=mode, transform=transform, storage=storage, profile=profile):
                    run = run_backtest(dataset, "$close-42" if mode == "signed" else
                                       "$close/Ref($close,1)-1", warmup_sessions=20,
                                       strategy_mode=mode, signal_transform=transform,
                                       execution_profile=profile, topk=2, n_drop=1,
                                       holdings="never")
                    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
                    self.assertEqual(manifest["status"], "completed")
                    self.assertEqual(manifest["qlib_binary_file_count"],
                                     json.loads((dataset / "manifest.json").read_text())["qlib_binary_file_count"])
                    self.assertEqual(manifest["backtest"]["execution_profile"], profile)
                    tables = {name: pd.read_parquet(run / f"{name}.parquet")
                              for name in ("alpha_scores", "positions")}
                    if mode == "signed":
                        tables["signed_weights"] = pd.read_parquet(run / "signed_weights.parquet")
                        with gzip.open(run / "signed_decisions.json.gz", "rt", encoding="utf-8") as handle:
                            decisions = json.load(handle)
                    else:
                        decisions = None
                    results[(storage, profile)] = (
                        pd.read_csv(run / "daily_report.csv"), tables,
                        json.loads((run / "summary.json").read_text(encoding="utf-8")), decisions)

        baseline = results[("unpacked", "reference")]
        daily, tables, summary, reference_decisions = baseline
        self.assertGreater(len(daily), 5)
        self.assertGreater(daily["turnover"].sum(), 0)
        if mode == "signed":
            self.assertGreater(daily["borrow_cost_dollars"].sum(), 0)
            self.assertGreater(daily["transaction_cost_dollars"].sum(), 0)
            self.assertTrue((tables["positions"].amount < 0).any())
            self.assertTrue((tables["positions"].amount > 0).any())
            self.assertEqual(summary["backtest"]["execution_contract"]["signal_transform"], transform)
            if transform != "raw":
                daily_net_target = tables["signed_weights"].groupby("datetime").weight.sum()
                self.assertTrue((daily_net_target.abs() < 1e-6).all())

        for variant, (other_daily, other_tables, other_summary, decisions) in results.items():
            with self.subTest(mode=mode, transform=transform, variant=variant):
                pd.testing.assert_frame_equal(daily, other_daily, check_exact=False,
                                              rtol=1e-9, atol=1e-7)
                for name, table in tables.items():
                    pd.testing.assert_frame_equal(table, other_tables[name], check_exact=False,
                                                  rtol=1e-9, atol=1e-7)
                self.assertEqual(summary["metrics"], other_summary["metrics"])
                if mode == "signed":
                    self.assertEqual(summary["funding_state"], other_summary["funding_state"])
                    self.assertEqual(reference_decisions, decisions)

    @unittest.skipIf(os.environ.get(CHILD) == "1", "normal Qlib is not the signed derivative")
    def test_long_only_packed_and_fast_parity(self):
        self.run_variants(mode="long-only")

    @unittest.skipIf(os.environ.get(CHILD) == "1", "parent runner check")
    def test_runner_and_selector_reject_redirect_before_resolving(self):
        from qlib_yahoo.backtest import run_backtest
        from qlib_yahoo.dataset import resolve_dataset

        # A simulated Windows junction lets the test run without link privileges.
        original = Path.is_junction
        def redirected(path):
            return path == self.packed or original(path)
        with patch.object(Path, "is_junction", redirected):
            with self.assertRaisesRegex(DatasetError, "redirected dataset path"):
                run_backtest(self.packed, "$close", warmup_sessions=20, topk=2)
            with self.assertRaisesRegex(DatasetError, "redirected dataset path"):
                resolve_dataset(self.packed.parent.parent, self.packed)

    @unittest.skipUnless(_signed_available(), "pinned signed derivative unavailable")
    @unittest.skipIf(os.environ.get(CHILD) == "1", "parent launcher")
    def test_signed_packed_and_fast_parity(self):
        program = (
            "from qlib_yahoo.runtime import ensure_signed_runtime; "
            "ensure_signed_runtime(); import unittest; "
            "suite=unittest.defaultTestLoader.loadTestsFromName("
            "'tests.test_qlib_performance_parity.SignedChild'); "
            "result=unittest.TextTestRunner(verbosity=2).run(suite); "
            "raise SystemExit(not result.wasSuccessful())"
        )
        env = os.environ.copy()
        env[CHILD] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run([sys.executable, "-c", program], cwd=ROOT, env=env,
                                capture_output=True, text=True, encoding="utf-8", timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(os.environ.get(CHILD) == "1", "requires verified signed child")
class SignedChild(ParityTests):
    def test_failed_stream_records_original_error_even_if_cleanup_raises(self):
        from qlib_yahoo.backtest import run_backtest
        from qlib_yahoo.streaming import SignedHistoryWriter

        data_root = self.packed.parent.parent
        original_close = SignedHistoryWriter.close

        def close_then_fail(writer, *, completed):
            original_close(writer, completed=completed)
            raise OSError("secondary cleanup failure")

        with patch.object(SignedHistoryWriter, "positions", side_effect=DatasetError("primary stream failure")), \
                patch.object(SignedHistoryWriter, "close", close_then_fail):
            with self.assertRaisesRegex(DatasetError, "primary stream failure"):
                run_backtest(self.packed, "$close-42", warmup_sessions=20,
                             strategy_mode="signed", history_mode="stream", holdings="never")
        runs = list((data_root / "runs").iterdir())
        self.assertEqual(len(runs), 1)
        manifest = json.loads((runs[0] / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["error"], {"type": "DatasetError", "message": "primary stream failure"})
        self.assertFalse(list(runs[0].glob("*.tmp")))

    def test_streamed_year_boundary_missing_marks_and_fills_match_native(self):
        from tests.test_qlib_yahoo_execution import _snapshot
        from qlib_yahoo.backtest import run_backtest
        import numpy as np
        import pandas as pd

        # tools/__init__.py belongs to the legacy Brain workflow and imports
        # unrelated optional dependencies. Load this standalone checker directly.
        spec = importlib.util.spec_from_file_location(
            "qlib_runtime_check", ROOT / "tools/qlib_runtime_check.py")
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)

        with tempfile.TemporaryDirectory() as temporary:
            dataset = _snapshot(Path(temporary), sessions_count=350, include_benchmark=False)
            # The score remains finite, while a held name lacks one close and
            # another cannot fill at one open. Calendar gaps and the year boundary
            # exercise borrow/account continuity on the actual pinned engine.
            for symbol, field, offset in (("tst0", "close", 230), ("tst5", "open", 255)):
                path = dataset / "qlib_data/features" / symbol / (field + ".day.bin")
                values = np.fromfile(path, dtype="<f4")
                values[offset + 1] = np.nan
                values.tofile(path)
            manifest_path = dataset / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["qlib_binary_tree_sha256"] = tree_sha256(dataset / "qlib_data")[0]
            manifest_path.write_text(json.dumps(manifest))
            options = dict(warmup_sessions=20, strategy_mode="signed", holdings="never")
            original = run_backtest(dataset, "$volume-102500", execution_profile="reference", history_mode="memory", **options)
            streamed = run_backtest(dataset, "$volume-102500", execution_profile="fast", history_mode="stream", **options)
            comparison = checker.compare_runs(original, streamed)
            self.assertEqual(comparison["decision_sessions"], 329)
            ledger = pd.read_csv(streamed / "daily_report.csv")
            self.assertTrue(ledger["stale_marks"].any())
            self.assertEqual(set(pd.to_datetime(ledger["datetime"]).dt.year), {2024, 2025})

    def test_signed_transforms(self):
        for transform in ("raw", "rank", "zscore"):
            with self.subTest(transform=transform):
                self.run_variants(mode="signed", transform=transform)


if __name__ == "__main__":
    unittest.main()
