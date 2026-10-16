"""Package examples must work without the private research tree."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from qlib_yahoo.alphalab.fastexpr import parse
from qlib_yahoo.webapp import service


class PublicExamplesTests(unittest.TestCase):
    def test_package_examples_ignore_private_research(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private = root / "research/alphas/alpha101_price_volume.txt"
            private.parent.mkdir(parents=True)
            private.write_text("PRIVATE_SENTINEL", encoding="utf-8")
            with patch.object(service, "ROOT", root):
                text = service._lab_examples()
            self.assertNotIn("PRIVATE_SENTINEL", text)
            expressions = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
            self.assertGreaterEqual(len(expressions), 3)
            for expression in expressions:
                parse(expression)

    def test_missing_resource_keeps_safe_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(service, "__file__", str(Path(temporary) / "webapp/service.py")):
                self.assertEqual(service._lab_examples(), "")

    def test_export_does_not_discover_parent_checkout_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            (parent / ".git").mkdir()
            exported = parent / "framework"
            exported.mkdir()
            instance = SimpleNamespace(data_root=parent / "data", current=None)
            with patch.object(service, "ROOT", exported), \
                    patch("qlib_yahoo.runtime.signed_runtime_status", return_value=(False, "fixture")), \
                    patch.object(service, "dataset_summary", return_value={}), \
                    patch.object(service.subprocess, "run") as run:
                result = service.ConsoleService.overview(instance)
            self.assertEqual(result["version"], "local")
            run.assert_not_called()

    def test_checkout_with_own_git_marker_can_show_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # Worktrees use a .git file rather than a directory.
            (root / ".git").write_text("gitdir: fixture", encoding="utf-8")
            instance = SimpleNamespace(data_root=root / "data", current=None)
            with patch.object(service, "ROOT", root), \
                    patch("qlib_yahoo.runtime.signed_runtime_status", return_value=(False, "fixture")), \
                    patch.object(service, "dataset_summary", return_value={}), \
                    patch.object(service.subprocess, "run", return_value=SimpleNamespace(stdout="abcdef0\n")) as run:
                result = service.ConsoleService.overview(instance)
            self.assertEqual(result["version"], "abcdef0")
            self.assertEqual(run.call_args.args[0], ["git", "-C", str(root), "rev-parse", "--short", "HEAD"])
