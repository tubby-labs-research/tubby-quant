import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from qlib_yahoo.workflow_records import EvidenceStore
from qlib_yahoo.workflow_recovery import owner_alive, recover


class WorkflowRecoveryTests(unittest.TestCase):
    def test_remote_or_unverifiable_owner_is_not_dead(self):
        self.assertIsNone(owner_alive({"host": "not-" + socket.gethostname(), "pid": os.getpid()}))
        with patch.dict(sys.modules, {"psutil": None}):
            self.assertIsNone(owner_alive({"host": socket.gethostname(), "pid": os.getpid()}))

    @unittest.skipUnless(importlib.util.find_spec("psutil"), "optional process inspector unavailable")
    def test_live_owner_and_exited_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            store = EvidenceStore(root)
            live = store.begin("still-live", {})
            self.assertTrue(owner_alive(live["owner"]))
            altered = {**live["owner"], "create_time": live["owner"]["create_time"] - 10}
            self.assertFalse(owner_alive(altered))
            script = ("import sys,json; from qlib_yahoo.workflow_records import EvidenceStore; "
                      "print(EvidenceStore(sys.argv[1]).begin('exited-worker', {})['id'])")
            result = subprocess.run(
                [sys.executable, "-B", "-c", script, str(root)],
                capture_output=True, text=True, check=True, timeout=30,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            dead_id = result.stdout.strip()
            self.assertEqual(store.get(dead_id)["status"], "pending")
            report = recover(root)
            self.assertEqual(report["recovered"], [dead_id])
            self.assertEqual(store.get(dead_id)["status"], "interrupted_unknown")
            self.assertEqual(store.get(live["id"])["status"], "pending")
            self.assertEqual(recover(root)["recovered"], [])


if __name__ == "__main__":
    unittest.main()
