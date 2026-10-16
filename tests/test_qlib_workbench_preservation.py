"""Backend contracts retained when the classic Tk frontend was retired."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from qlib_yahoo.jobs import Job
from qlib_yahoo.webapp.service import ConsoleService, ServiceError
from qlib_yahoo.workspace import STAGE_PREFIXES
from tests.test_qlib_webapp import make_dataset, make_run


class WorkbenchPreservationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.dataset = make_dataset(self.root)
        self.run = make_run(self.root)
        (self.root / "latest.json").write_text(json.dumps({"dataset": str(self.dataset)}), encoding="utf-8")
        self.requests = []
        owner = self

        class Worker:
            def __init__(self, request):
                owner.requests.append(request)

            def cancel(self):
                return False

            def run(self, on_line):
                on_line("Started alpha evaluation")
                return owner.run

        self.service = ConsoleService(self.root, job_factory=Worker, settings_path=self.root / "webapp.json")
        self.addCleanup(self.service.research.close)

    def finish(self, job):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = self.service.job(job["id"])
            if result["state"] != "running":
                self.assertEqual(result["state"], "completed", result)
                return result
            time.sleep(0.005)
        self.fail("service task did not finish")

    def test_invalid_inputs_never_start_worker(self):
        cases = [("gibberish here", {}, "alpha syntax"),
                 ("$close", {"cash_rate": "nan"}, "Cash rate")]
        for gross in ("nan", "inf", "-0.01", "1.01", "garbage"):
            cases.append(("$close", {"strategy_mode": "signed", "gross_exposure": gross}, "could not convert" if gross == "garbage" else "Gross exposure"))
        cases.append(("$close", {"strategy_mode": "signed", "signal_transform": "softmax"}, "Weighting"))
        for alpha, options, message in cases:
            with self.subTest(options=options), self.assertRaisesRegex(ServiceError, message):
                self.service.start_backtest({"alpha": alpha, **options})
        self.assertEqual(self.requests, [])
        self.assertFalse(self.service.busy())

    def test_signed_options_reach_isolated_worker_without_parent_qlib(self):
        with patch.dict(sys.modules, {"qlib": None}), \
                patch("qlib_yahoo.expressions.validate_alpha", side_effect=AssertionError("parent imported Qlib")):
            done = self.finish(self.service.start_backtest({
                "alpha": "$close", "strategy_mode": "signed", "cash_rate": "0.02",
                "gross_exposure": "0.75", "signal_transform": "rank"}))
        request, = self.requests
        self.assertEqual(request["dataset"], str(self.dataset))
        self.assertEqual(request["strategy_mode"], "signed")
        self.assertEqual(request["annual_cash_rate"], 0.02)
        self.assertEqual(request["gross_exposure"], 0.75)
        self.assertEqual(request["annual_borrow_rate"], 0.004)
        self.assertEqual(request["signal_transform"], "rank")
        self.assertEqual(done["result"], str(self.run))

    def test_long_only_ignores_signed_gross_setting(self):
        self.finish(self.service.start_backtest({"alpha": "$close", "cash_rate": "0.04", "gross_exposure": "invalid"}))
        request, = self.requests
        self.assertEqual(request["annual_cash_rate"], 0.04)
        self.assertNotIn("gross_exposure", request)
        self.assertNotIn("annual_borrow_rate", request)

    def test_report_reuse_still_goes_through_report_service(self):
        index = self.run / "reports" / "r1" / "report_index.html"
        with patch("qlib_yahoo.report_service.ensure_report", return_value=(index, True)) as report:
            done = self.finish(self.service.start_report(str(self.run), "holdings"))
        report.assert_called_once_with(self.run, "holdings")
        self.assertEqual(done["result"], str(index))

    def test_worker_performance_lines_are_recognized_as_stages(self):
        for line in ("Backtest worker PID 123", "Started alpha evaluation",
                     "Completed alpha evaluation in 1.00s, peak RSS 100.0 MiB",
                     "Failed daily simulation in 2.00s"):
            self.assertTrue(line.startswith(STAGE_PREFIXES), line)

    def test_active_entrypoints_import_without_tk_or_classic_console(self):
        code = """import sys
sys.modules['tkinter'] = None
sys.modules['qlib_yahoo.console'] = None
import qlib_yahoo.__main__
import qlib_yahoo.webapp.__main__
import qlib_yahoo.desktop
assert 'qlib' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(sys.platform == "win32", "Windows pythonw fallback")
    def test_browser_fallback_stop_dialog_needs_no_tk(self):
        import ctypes
        from qlib_yahoo.webapp.__main__ import _serve_in_browser

        server = SimpleNamespace(origin="http://127.0.0.1:1234")
        url = server.origin + "/start?token=fixture"
        with patch.dict(sys.modules, {"tkinter": None}), \
                patch.object(sys, "executable", "C:/Python312/pythonw.exe"), \
                patch("webbrowser.open") as opened, \
                patch.object(ctypes.windll.user32, "MessageBoxW", return_value=1) as dialog, \
                patch("qlib_yahoo.webapp.__main__.time.sleep", side_effect=AssertionError("did not stop")):
            _serve_in_browser(server, url)
        opened.assert_called_once_with(url)
        self.assertIn("press OK to stop", dialog.call_args.args[1])

    def test_signed_validation_script_rejects_retired_tk_flags(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "validate_qlib_signed.py"
        help_result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--dataset", help_result.stdout)
        self.assertIn("--evidence", help_result.stdout)
        self.assertNotIn("--console", help_result.stdout)
        for flag in ("--console", "--native-console"):
            rejected = subprocess.run([sys.executable, str(script), "--dataset", str(self.dataset),
                                       "--evidence", str(self.root / "evidence.json"), flag],
                                      capture_output=True, text=True)
            self.assertEqual(rejected.returncode, 2, rejected.stderr)
            self.assertIn("unrecognized arguments", rejected.stderr)
        self.assertFalse((self.root / "evidence.json").exists())

    @unittest.skipUnless(importlib.util.find_spec("qlib") and sys.version_info < (3, 13),
                         "optional Qlib runtime unavailable")
    def test_real_synthetic_worker_with_pythonw_streams(self):
        from tests.test_qlib_yahoo_execution import _snapshot

        with tempfile.TemporaryDirectory() as temporary:
            snapshot = _snapshot(Path(temporary).resolve(), sessions_count=260)
            lines = []
            with patch.object(sys, "stdout", None), patch.object(sys, "stderr", None):
                run = Job({"kind": "backtest", "dataset": str(snapshot),
                           "alpha": "$close/Ref($close, 1)-1", "annual_cash_rate": 0.0,
                           "topk": 2, "n_drop": 1, "warmup_sessions": 20}).run(lines.append)
            self.assertEqual(json.loads((run / "manifest.json").read_text())["status"], "completed")
            self.assertTrue((run / "summary.json").is_file())
            self.assertTrue(lines)


if __name__ == "__main__":
    unittest.main()
