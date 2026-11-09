"""scripts/check_static.py: a syntax error fails, a missing tool is an error unless lenient."""
from __future__ import annotations

import importlib.util
import io
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_static", ROOT / "scripts" / "check_static.py")
check_static = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_static)


class NodeGateTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "node is optional")
    def test_a_syntax_error_fails_and_a_clean_script_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "good.js").write_text("const a = 1;\n", encoding="utf-8")
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(check_static.run_node(False, folder), 0)
            (folder / "bad.js").write_text("const a = ;\n", encoding="utf-8")
            err = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(err):
                self.assertEqual(check_static.run_node(False, folder), 1)
            self.assertIn("bad.js", err.getvalue())

    def test_a_missing_node_is_an_error_unless_lenient(self):
        with mock.patch.object(check_static.shutil, "which", return_value=None), redirect_stdout(io.StringIO()):
            self.assertEqual(check_static.run_node(False), 1)
            self.assertEqual(check_static.run_node(True), 0)

    def test_a_missing_ruff_is_an_error_unless_lenient(self):
        def missing(*args, **kwargs):
            raise FileNotFoundError("python")
        with mock.patch.object(check_static.subprocess, "run", side_effect=missing), redirect_stdout(io.StringIO()):
            self.assertEqual(check_static.run_ruff(False), 1)
            self.assertEqual(check_static.run_ruff(True), 0)


if __name__ == "__main__":
    unittest.main()
