"""Synthetic saved-run checks for the GUI report reuse boundary."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qlib_yahoo import report_service as service
from tests import test_qlib_yahoo_reports as report_tests


@unittest.skipUnless(
    importlib.util.find_spec("pandas") and importlib.util.find_spec("plotly"),
    "pandas and plotly are optional outside the Qlib environment",
)
class ReportServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        # Resolve: macOS temp folders live behind the /var -> /private/var symlink.
        self.run = Path(self.temp.name).resolve()
        report_tests.ReplayTests().bundle(self.run)
        self.render_count = 0

    def fake_render(self, run_dir, *, level, holdings, threshold, min_observations, _validated=None):
        self.render_count += 1
        directory = run_dir / "reports" / f"report-{self.render_count:02d}"
        directory.mkdir(parents=True)
        charts = ["holdings.html"] if level == "summary" else [
            "account.html", "drawdown.html", "returns_cost.html", "turnover.html"
        ]
        for name in [*charts, "plotly.min.js", "report_index.html"]:
            (directory / name).write_text(name, encoding="utf-8")
        status = {
            "status": "completed", "level": level, "holdings": holdings,
            "threshold": threshold, "min_observations": min_observations,
            "charts": charts, **service.analysis._provenance(run_dir),
        }
        (directory / "report_manifest.json").write_text(json.dumps(status), encoding="utf-8")
        return directory / "report_index.html"

    def test_selection_and_invalid_paths_do_not_write(self):
        for selected in [self.run, *(self.run / name for name in service._SELECTIONS)]:
            self.assertEqual(service.validate_run_selection(selected), self.run.resolve())
        for selected in [self.run / "positions.csv.gz", self.run / "missing.json"]:
            with self.assertRaises(ValueError):
                service.validate_run_selection(selected)
        self.assertFalse((self.run / "reports").exists())
        self.assertFalse((self.run / ".report_reuse.lock").exists())

    def test_reuse_and_profile_separation(self):
        with patch.object(service.analysis, "render_report", side_effect=self.fake_render):
            holdings, reused = service.ensure_report(self.run, "holdings")
            self.assertFalse(reused)
            self.assertEqual(service.ensure_report(self.run, "holdings"), (holdings, True))
            overview, reused = service.ensure_report(self.run, "overview")
            self.assertFalse(reused)
            self.assertNotEqual(overview, holdings)
            detailed, reused = service.ensure_report(self.run, "detailed")
            self.assertFalse(reused)
            self.assertEqual(service.ensure_report(self.run, "detailed"), (detailed, True))
        self.assertEqual(self.render_count, 3)
        self.assertFalse((self.run / ".report_reuse.lock").exists())

    def test_real_renderer_reused_without_second_report(self):
        first, reused = service.ensure_report(self.run, "holdings")
        self.assertFalse(reused)
        self.assertTrue((first.parent / "holdings.html").is_file())
        self.assertEqual(service.ensure_report(self.run, "holdings"), (first, True))
        self.assertEqual(len(list((self.run / "reports").iterdir())), 1)

    def test_one_hash_and_reconciliation_pass_per_request(self):
        analysis = service.analysis
        for reused_expected in (False, True):
            with (patch.object(analysis, "_bundle", wraps=analysis._bundle) as bundle,
                  patch.object(analysis, "_reconcile_ledger", wraps=analysis._reconcile_ledger) as ledger,
                  patch.object(analysis, "_sha256", wraps=analysis._sha256) as hashes):
                _, reused = service.ensure_report(self.run, "overview")
                self.assertEqual(reused, reused_expected)
                self.assertEqual(bundle.call_count, 1)
                self.assertEqual(ledger.call_count, 1)
                files = [call.args[0].name for call in hashes.call_args_list]
                self.assertEqual(files.count("positions.csv.gz"), 1)
                self.assertEqual(files.count("daily_report.csv"), 1)

    def test_mutation_after_lock_acquisition_is_rejected(self):
        from contextlib import contextmanager

        original_lock = service._run_lock

        @contextmanager
        def mutate_under_lock(run):
            with original_lock(run):
                (run / "daily_report.csv").write_text("tampered")
                yield

        with patch.object(service, "_run_lock", mutate_under_lock):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                service.ensure_report(self.run, "overview")
        self.assertFalse((self.run / "reports").exists())
        self.assertFalse((self.run / ".report_reuse.lock").exists())

    def test_mutation_during_render_never_publishes_completed_report(self):
        original_chart = service.analysis._chart

        def mutate_during_chart(*args, **kwargs):
            result = original_chart(*args, **kwargs)
            with (self.run / "daily_report.csv").open("a") as target:
                target.write("tampered")
            return result

        with patch.object(service.analysis, "_chart", mutate_during_chart):
            with self.assertRaisesRegex(ValueError, "changed during report request"):
                service.ensure_report(self.run, "overview")
        statuses = [json.loads(path.read_text())["status"] for path in (self.run / "reports").glob("*/report_manifest.json")]
        self.assertEqual(statuses, ["failed"])
        self.assertFalse((self.run / ".report_reuse.lock").exists())

    def test_large_artifact_hashing_never_uses_read_bytes(self):
        with patch.object(Path, "read_bytes", side_effect=AssertionError("whole-file allocation")):
            service.ensure_report(self.run, "overview")

    def test_same_size_mutation_with_restored_mtime_is_rejected(self):
        import os

        original_chart = service.analysis._chart
        changed = False

        def mutate(*args, **kwargs):
            nonlocal changed
            result = original_chart(*args, **kwargs)
            if not changed:
                target = self.run / "daily_report.csv"
                original = target.stat()
                text = target.read_text()
                target.write_text(text.replace("100.3002", "101.3002"))
                os.utime(target, ns=(original.st_atime_ns, original.st_mtime_ns))
                changed = True
            return result

        with patch.object(service.analysis, "_chart", mutate):
            with self.assertRaisesRegex(ValueError, "changed during report request"):
                service.ensure_report(self.run, "overview")

    def test_reuse_checks_for_mutation_before_returning_existing_report(self):
        service.ensure_report(self.run, "overview")
        original_reusable = service._reusable

        def mutate(*args, **kwargs):
            result = original_reusable(*args, **kwargs)
            (self.run / "summary.json").write_text("tampered")
            return result

        with patch.object(service, "_reusable", mutate):
            with self.assertRaisesRegex(ValueError, "changed during report request"):
                service.ensure_report(self.run, "overview")

    def test_qualified_auto_summary_satisfies_holdings_button(self):
        import pandas as pd

        daily = pd.read_csv(self.run / "daily_report.csv").iloc[[0]].copy()
        dates = pd.bdate_range("2024-01-02", periods=252)
        daily = pd.concat([daily] * len(dates), ignore_index=True)
        daily["datetime"] = dates
        daily["net_return"] = [0.001 if index % 2 else 0.002 for index in range(len(dates))]
        daily["account"] = 100.0 * (1 + daily["net_return"]).cumprod()
        daily["cash"] = daily["account"] * 0.2
        daily["holdings_value"] = daily["account"] * 0.8
        daily.to_csv(self.run / "daily_report.csv", index=False)
        report_tests.ReplayTests().reseal(self.run)

        auto_index = service.analysis.render_report(
            self.run, level="summary", holdings="auto",
            threshold=1.3, min_observations=252,
        )
        status = json.loads((auto_index.parent / "report_manifest.json").read_text())
        self.assertTrue(status["holdings_gate_eligible"])
        self.assertTrue(status["holdings_rendered"])
        self.assertEqual(service.ensure_report(self.run, "holdings"), (auto_index, True))
        self.assertEqual(len(list((self.run / "reports").iterdir())), 1)

    def test_auto_summary_without_rendered_holdings_is_not_equivalent(self):
        with patch.object(service.analysis, "render_report", side_effect=self.fake_render):
            auto_index = self.fake_render(
                self.run, level="summary", holdings="auto",
                threshold=1.3, min_observations=252,
            )
            status_path = auto_index.parent / "report_manifest.json"
            status = json.loads(status_path.read_text())
            status["holdings_rendered"] = False
            status_path.write_text(json.dumps(status))
            index, reused = service.ensure_report(self.run, "holdings")
            self.assertFalse(reused)
            self.assertNotEqual(index, auto_index)

    def test_real_renderer_all_profiles_have_complete_outputs(self):
        expected = {
            "holdings": ("summary", "always", {"holdings.html"}),
            "overview": ("overview", "auto", {"account.html", "drawdown.html", "returns_cost.html", "turnover.html"}),
            "detailed": ("detailed", "auto", {"account.html", "drawdown.html", "returns_cost.html", "turnover.html"}),
        }
        for profile, (level, holdings, required) in expected.items():
            with self.subTest(profile=profile):
                index, reused = service.ensure_report(self.run, profile)
                self.assertFalse(reused)
                manifest = json.loads((index.parent / "report_manifest.json").read_text())
                self.assertEqual((manifest["level"], manifest["holdings"]), (level, holdings))
                self.assertEqual((manifest["threshold"], manifest["min_observations"]), (1.3, 252))
                self.assertTrue(required.issubset(manifest["charts"]))
                for name in ["report_index.html", "plotly.min.js", *manifest["charts"]]:
                    self.assertGreater((index.parent / name).stat().st_size, 0)
                self.assertEqual(service.ensure_report(self.run, profile), (index, True))
        self.assertEqual(len(list((self.run / "reports").iterdir())), 3)

    def test_older_match_beyond_256_report_directories_is_reused(self):
        with patch.object(service.analysis, "render_report", side_effect=self.fake_render):
            first, _ = service.ensure_report(self.run, "holdings")
            for index in range(260):
                (self.run / "reports" / f"newer-{index:03d}").mkdir()
            self.assertEqual(service.ensure_report(self.run, "holdings"), (first, True))
        self.assertEqual(self.render_count, 1)

    def test_reuse_never_inspects_an_unrelated_run(self):
        with tempfile.TemporaryDirectory() as other_temp:
            other = Path(other_temp)
            (other / "reports").mkdir()
            original_iterdir = Path.iterdir
            original_read_text = Path.read_text

            def guarded_iterdir(path):
                if path == other or path.is_relative_to(other):
                    raise AssertionError("inspected unrelated run")
                return original_iterdir(path)

            def guarded_read_text(path, *args, **kwargs):
                if path == other or path.is_relative_to(other):
                    raise AssertionError("read unrelated run")
                return original_read_text(path, *args, **kwargs)

            with (patch.object(Path, "iterdir", guarded_iterdir),
                  patch.object(Path, "read_text", guarded_read_text),
                  patch.object(service.analysis, "render_report", side_effect=self.fake_render)):
                first, _ = service.ensure_report(self.run, "holdings")
                self.assertEqual(service.ensure_report(self.run, "holdings"), (first, True))

    def test_core_and_renderer_provenance_invalidation(self):
        with patch.object(service.analysis, "render_report", side_effect=self.fake_render):
            first, _ = service.ensure_report(self.run, "overview")
            manifest = self.run / "manifest.json"
            manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            second, reused = service.ensure_report(self.run, "overview")
            self.assertFalse(reused)
            self.assertNotEqual(first, second)
            original = service.analysis._provenance
            with patch.object(service.analysis, "_provenance", side_effect=lambda run: {
                **original(run), "renderer_source_sha256": "0" * 64,
            }):
                third, reused = service.ensure_report(self.run, "overview")
                self.assertFalse(reused)
                self.assertNotEqual(second, third)
        self.assertEqual(self.render_count, 3)

    def test_plotly_version_invalidation(self):
        with patch.object(service.analysis, "render_report", side_effect=self.fake_render):
            first, _ = service.ensure_report(self.run, "holdings")
            original = service.analysis._provenance
            with patch.object(service.analysis, "_provenance", side_effect=lambda run: {
                **original(run), "plotly_version": "future-version",
            }):
                second, reused = service.ensure_report(self.run, "holdings")
                self.assertFalse(reused)
                self.assertNotEqual(first, second)

    def test_renderer_helper_provenance_invalidation(self):
        with patch.object(service.analysis, "render_report", side_effect=self.fake_render):
            first, _ = service.ensure_report(self.run, "holdings")
            original = service.analysis._provenance
            with patch.object(service.analysis, "_provenance", side_effect=lambda run: {
                **original(run), "renderer_dependencies_sha256": {"holding_series.py": "changed"},
            }):
                second, reused = service.ensure_report(self.run, "holdings")
                self.assertFalse(reused)
                self.assertNotEqual(first, second)

    def test_missing_failed_and_unsafe_report_are_not_reused(self):
        with patch.object(service.analysis, "render_report", side_effect=self.fake_render):
            first, _ = service.ensure_report(self.run, "overview")
            (first.parent / "account.html").unlink()
            second, reused = service.ensure_report(self.run, "overview")
            self.assertFalse(reused)
            self.assertNotEqual(first, second)
            status_path = second.parent / "report_manifest.json"
            status = json.loads(status_path.read_text())
            status["status"] = "failed"
            status_path.write_text(json.dumps(status))
            third, reused = service.ensure_report(self.run, "overview")
            self.assertFalse(reused)
            status_path = third.parent / "report_manifest.json"
            status = json.loads(status_path.read_text())
            status["charts"] = ["../outside.html"]
            status_path.write_text(json.dumps(status))
            fourth, reused = service.ensure_report(self.run, "overview")
            self.assertFalse(reused)
            status_path = fourth.parent / "report_manifest.json"
            status = json.loads(status_path.read_text())
            status["charts"] = [status["charts"], "account.html"]
            status_path.write_text(json.dumps(status))
            malformed, reused = service.ensure_report(self.run, "overview")
            self.assertFalse(reused)
            (malformed.parent / "plotly.min.js").unlink()
            fifth, reused = service.ensure_report(self.run, "overview")
            self.assertFalse(reused)
        self.assertEqual(self.render_count, 6)

    def test_empty_index_chart_or_shared_js_is_not_reused(self):
        with patch.object(service.analysis, "render_report", side_effect=self.fake_render):
            for name in ("report_index.html", "holdings.html", "plotly.min.js"):
                index, reused = service.ensure_report(self.run, "holdings")
                self.assertFalse(reused)
                (index.parent / name).write_bytes(b"")
            _, reused = service.ensure_report(self.run, "holdings")
            self.assertFalse(reused)
        self.assertEqual(self.render_count, 4)

    def test_junction_selection_reports_and_child_are_rejected_or_skipped(self):
        original = Path.is_junction if hasattr(Path, "is_junction") else lambda path: False
        redirects = {self.run}

        def marked_junction(path):
            return path in redirects or original(path)

        with patch.object(Path, "is_junction", marked_junction, create=True):
            with self.assertRaisesRegex(ValueError, "junction"):
                service.validate_run_selection(self.run)
            with self.assertRaisesRegex(ValueError, "junction"):
                service.validate_run_selection(self.run / "manifest.json")
            redirects.clear()
            redirects.add(self.run / "reports")
            with patch.object(service.analysis, "render_report", side_effect=self.fake_render):
                (self.run / "reports").mkdir()
                with self.assertRaisesRegex(ValueError, "Reports directory"):
                    service.ensure_report(self.run, "holdings")
            self.assertEqual(self.render_count, 0)
            redirects.clear()
            with patch.object(service.analysis, "render_report", side_effect=self.fake_render):
                first, _ = service.ensure_report(self.run, "holdings")
                redirects.add(first.parent)
                second, reused = service.ensure_report(self.run, "holdings")
                self.assertFalse(reused)
                self.assertNotEqual(first, second)

    def test_invalid_run_rejected_before_any_write(self):
        with self.assertRaises(ValueError):
            service.ensure_report(self.run, "unknown")
        manifest_path = self.run / "manifest.json"
        original = json.loads(manifest_path.read_text())
        for update in ({"status": "running"}, {"schema_version": 2}):
            manifest_path.write_text(json.dumps({**original, **update}))
            with self.assertRaises(ValueError):
                service.ensure_report(self.run, "holdings")
        manifest_path.write_text(json.dumps(original))
        (self.run / "daily_report.csv").write_text("tampered")
        with self.assertRaises(ValueError):
            service.ensure_report(self.run, "overview")
        self.assertFalse((self.run / "reports").exists())
        self.assertFalse((self.run / ".report_reuse.lock").exists())

    def test_resealed_inconsistent_ledger_is_rejected(self):
        ledger = self.run / "daily_report.csv"
        ledger.write_text(ledger.read_text().replace("100.3002", "101.3002"))
        manifest_path = self.run / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts_sha256"]["daily_report.csv"] = hashlib.sha256(ledger.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            service.validate_run_selection(self.run)
        self.assertFalse((self.run / "reports").exists())

    def test_lock_left_by_an_exited_process_is_recovered(self):
        import os
        import subprocess
        import sys

        ended = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True, check=True)
        lock = self.run / ".report_reuse.lock"
        lock.write_text(ended.stdout.strip())  # e.g. the app was force-quit while rendering
        with patch.object(service.analysis, "render_report", side_effect=RuntimeError("renderer ran")):
            with self.assertRaisesRegex(RuntimeError, "renderer ran"):
                service.ensure_report(self.run, "overview")
        self.assertFalse(lock.exists())
        lock.write_text(str(os.getpid()))  # a live holder still blocks
        with self.assertRaisesRegex(RuntimeError, "already in progress"):
            service.ensure_report(self.run, "overview")
        lock.unlink()

    def test_existing_lock_blocks_and_render_failure_releases(self):
        lock = self.run / ".report_reuse.lock"
        lock.write_text("other window")
        with self.assertRaisesRegex(RuntimeError, "already in progress"):
            service.ensure_report(self.run, "overview")
        self.assertEqual(lock.read_text(), "other window")
        lock.unlink()
        with patch.object(service.analysis, "render_report", side_effect=RuntimeError("renderer failed")):
            with self.assertRaisesRegex(RuntimeError, "renderer failed"):
                service.ensure_report(self.run, "overview")
        self.assertFalse(lock.exists())


if __name__ == "__main__":
    unittest.main()
