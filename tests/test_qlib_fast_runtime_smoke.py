"""Offline checks for the bounded real-worker CI smoke harness."""

import importlib.util
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("runtime_smoke", ROOT / "scripts/smoke_qlib_fast_runtime.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SmokeTests(unittest.TestCase):
    def test_redirected_windows_output_keeps_unicode_progress(self):
        for encoding in ("cp1252", "ascii"):
            with self.subTest(encoding=encoding):
                raw_out, raw_err = io.BytesIO(), io.BytesIO()
                out = io.TextIOWrapper(raw_out, encoding=encoding)
                err = io.TextIOWrapper(raw_err, encoding=encoding)
                def fake_run(root):
                    print("progress ▍ 测试")
                    print("diagnostic ▍", file=smoke.sys.stderr)
                    return {"status": "passed"}
                with patch.object(smoke.sys, "stdout", out), patch.object(smoke.sys, "stderr", err), \
                        patch.object(smoke.sys, "argv", ["smoke"]), \
                        patch.object(smoke, "run_smoke", side_effect=fake_run):
                    self.assertEqual(smoke.main(), 0)
                out.flush()
                err.flush()
                self.assertIn("progress ▍ 测试", raw_out.getvalue().decode("utf-8"))
                self.assertIn("diagnostic ▍", raw_err.getvalue().decode("utf-8"))
                out.close()
                err.close()

    def test_mac_profile_records_compatible_fallback(self):
        smoke.check_default_profile({"name": "accepted-v1", "platform": "macos", "engine": "native-signed",
                                     "ordered_session": False, "gzip": "stdlib-zlib", "date_range": "python",
                                     "fallback_reason": "Windows acceleration ABI unavailable"}, "darwin")

    def test_mac_profile_rejects_windows_acceleration(self):
        profile = {"name": "accepted-v1", "platform": "macos", "engine": "ordered-signed",
                   "ordered_session": True, "gzip": "ISA-L", "date_range": "public_buffer_leases"}
        with self.assertRaisesRegex(AssertionError, "Windows ordered"):
            smoke.check_default_profile(profile, "darwin")

    def test_profile_requires_actual_default_selection(self):
        with self.assertRaisesRegex(AssertionError, "accepted profile"):
            smoke.check_default_profile({"selection": "native"}, "darwin")

    def test_synthetic_bins_have_both_alpha_signs_and_valid_hash(self):
        from qlib_yahoo.dataset import tree_sha256
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = smoke.synthetic_snapshot(Path(temporary))
            manifest = json.loads((snapshot / "manifest.json").read_text())
            bins = snapshot / "qlib_data"
            self.assertEqual(tree_sha256(bins)[0], manifest["qlib_binary_tree_sha256"])
            self.assertEqual(len(manifest["symbols"]), 6)
            calendar = (bins / "calendars/day.txt").read_text().splitlines()
            self.assertEqual(len(calendar), 45)
            close = [struct.unpack("<46f", (bins / f"features/{code.lower()}/close.day.bin").read_bytes())
                     for code in manifest["symbols"]]
            self.assertTrue(all(values[0] == 0 for values in close))
            self.assertLess(close[0][21] - 40, 0)
            self.assertGreater(close[-1][21] - 40, 0)
            self.assertNotIn("benchmark", manifest)

    def test_short_calendar_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "25 sessions"):
                smoke.synthetic_snapshot(Path(temporary), 24)

    def report_fixture(self, root, *, status="completed", chart=True, linked=True):
        folder = root / "reports" / "one"
        folder.mkdir(parents=True)
        (folder / "report_manifest.json").write_text(json.dumps({
            "status": status, "charts": ["nav.html"] if chart else []}))
        link = '<a href="nav.html">NAV</a>' if linked else ""
        (folder / "report_index.html").write_text(f"<html><body><table></table>{link}</body></html>")
        (folder / "nav.html").write_text("<html><body><script>Plotly.newPlot('nav', [])</script></body></html>")

    def test_html_chart_links_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.report_fixture(root)
            self.assertEqual(smoke.check_html_report(root), 1)

    def test_report_warning_cannot_pass_as_simulation_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(AssertionError, "without an HTML"):
                smoke.check_html_report(Path(temporary))

    def test_failed_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.report_fixture(root, status="failed")
            with self.assertRaisesRegex(AssertionError, "rendering failed"):
                smoke.check_html_report(root)

    def test_empty_overview_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.report_fixture(root, chart=False)
            with self.assertRaisesRegex(AssertionError, "no charts"):
                smoke.check_html_report(root)

    def test_unlinked_chart_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.report_fixture(root, linked=False)
            with self.assertRaisesRegex(AssertionError, "absent"):
                smoke.check_html_report(root)

    def test_actual_worker_requests_preserve_default_and_batch_fresh_accounts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for name in ("single", "first"):
                (root / name).mkdir()
                (root / name / "daily_report.csv").write_text("same deterministic ledger")
            receipt = root / "batch.json"
            receipt.write_text(json.dumps({"status": "completed", "errors": [],
                                          "runs": [str(root / "first"), str(root / "second")],
                                          "preparation": {"fresh_account_per_run": True}}))
            with patch("qlib_yahoo.jobs.Job") as job, patch.object(smoke, "validate_run", return_value={}) as validate:
                job.return_value.run.side_effect = [root / "single", receipt]
                evidence = smoke.run_smoke(root)
                self.assertEqual(evidence["worker_runs"], 3)
                self.assertEqual(validate.call_count, 3)
                requests = [call.args[0] for call in job.call_args_list]
                self.assertEqual([r["kind"] for r in requests], ["backtest", "backtest-batch"])
                self.assertEqual(len(requests[1]["alphas"]), 2)
                for request in requests:
                    self.assertNotIn("runtime_profile", request)
                    self.assertEqual(request["strategy_mode"], "signed")
                    self.assertEqual(request["report_level"], "overview")


if __name__ == "__main__":
    unittest.main()
