"""Actual new-workbench worker acceptance, not a FakeJob or classic Tk run."""

import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest

from qlib_yahoo.dataset import DatasetError
from qlib_yahoo.runtime import PIN, ROOT, platform_pin


def available():
    try:
        pin = platform_pin(json.loads(PIN.read_text(encoding="utf-8")))
        return (sys.version_info[:2] == (3, 12) and importlib.util.find_spec("qlib") is not None
                and (ROOT / pin["local_runtime"] / "qlib/_tubby_build.json").is_file())
    except (OSError, ValueError, KeyError, DatasetError):
        return False


@unittest.skipUnless(available(), "actual pinned signed worker unavailable")
class WorkbenchRuntimeTests(unittest.TestCase):
    def test_different_alpha_batch_uses_actual_accelerated_workers(self):
        from tests.test_qlib_yahoo_execution import _snapshot
        from qlib_yahoo.webapp.service import ConsoleService

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            original = _snapshot(root / "fixture", sessions_count=45, include_benchmark=False)
            dataset = root / "datasets" / "synthetic"
            shutil.copytree(original, dataset)
            # The console accepts ready imports, not the runner's deliberately
            # minimal synthetic manifest. Preserve its sealed data bytes.
            manifest_path = dataset / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = "ready"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            service = ConsoleService(root)  # default REAL Job factory and subprocess
            job = service.start_backtest({"dataset": str(dataset), "alpha": "$close-42\n42-$close",
                                          "strategy_mode": "signed", "warmup_sessions": "20"})
            deadline = time.monotonic() + 120
            try:
                while service.busy() and time.monotonic() < deadline:
                    time.sleep(0.05)
                result = service.job(job["id"])
                self.assertEqual(result["state"], "completed", result)
                receipt = json.loads(Path(result["result"]).read_text(encoding="utf-8"))
                self.assertEqual(receipt["status"], "completed")
                self.assertEqual(receipt["alphas"], ["$close-42", "42-$close"])
                self.assertEqual(receipt["errors"], [])
                self.assertEqual(len(set(receipt["runs"])), 2)
                self.assertTrue(receipt["preparation"]["fresh_account_per_run"])
                self.assertGreater(receipt["preparation"]["hits"], 0)
                for alpha, path in zip(receipt["alphas"], receipt["runs"]):
                    run = Path(path)
                    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
                    self.assertEqual(manifest["alpha"], alpha)
                    self.assertEqual(manifest["status"], "completed")
                    self.assertIn("fast_signed.py", manifest["runner_source_sha256"])
                    self.assertFalse(manifest["execution_acceleration"]["fastmath"])
                    self.assertEqual(manifest["history_storage"]["mode"], "stream")
                    profile = manifest["runtime_profile"]
                    self.assertEqual(profile["name"], "accepted-v1")
                    self.assertEqual(profile["selection"], "accepted_windows" if sys.platform == "win32"
                                     else "accepted_portable_macos")
                    compression = manifest["history_storage"]["decision_compression"]
                    self.assertEqual(compression["codec"], "gzip")
                    # ISA-L's level 2 is not equivalent to zlib's level 6.
                    # Each platform must select its exact accepted recipe.
                    self.assertEqual(compression["level"], 2 if sys.platform == "win32" else 6)
                    self.assertEqual(profile["gzip"], "ISA-L" if sys.platform == "win32" else "stdlib-zlib")
                    self.assertFalse(manifest["history_storage"]["account_reset_at_boundaries"])
                    self.assertTrue((run / "qlib_trade_indicators.csv").is_file())
                    self.assertTrue((run / "signed_decisions.json.gz").is_file())
            finally:
                if service.busy():
                    service.cancel()
                    deadline = time.monotonic() + 12
                    while service.busy() and time.monotonic() < deadline:
                        time.sleep(0.05)
