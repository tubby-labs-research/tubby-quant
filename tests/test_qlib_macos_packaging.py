"""macOS packaging and platform helpers: scripts, app bundle, icon, signed pins and settings."""

from __future__ import annotations

import importlib.util
import json
import plistlib
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [ROOT / "scripts" / "macos" / name for name in ("setup.sh", "build_app.sh", "app_launcher.sh")] + [ROOT / "launch-qlib-console.command"]


class ScriptTests(unittest.TestCase):
    @unittest.skipIf(sys.platform == "win32", "Windows bash may launch WSL rather than parse scripts locally")
    @unittest.skipUnless(shutil.which("bash"), "bash unavailable")
    def test_shell_scripts_parse(self):
        for script in SCRIPTS:
            with self.subTest(script=script.name):
                self.assertEqual(subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode, 0)

    def test_scripts_are_executable_in_git(self):
        listed = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-s", "--", *map(str, SCRIPTS)],
                                capture_output=True, text=True)
        if listed.returncode or not listed.stdout.strip():
            self.skipTest("scripts not tracked yet")
        for line in listed.stdout.splitlines():
            with self.subTest(line=line):
                self.assertTrue(line.startswith("100755"), line)

    def test_launchers_open_the_workbench(self):
        self.assertIn("-m qlib_yahoo.webapp", (ROOT / "scripts" / "macos" / "app_launcher.sh").read_text())
        self.assertIn("-m qlib_yahoo.webapp", (ROOT / "launch-qlib-console.command").read_text())
        self.assertIn("pywebview==", (ROOT / "requirements-qlib-macos.txt").read_text())

    def test_setup_requires_python_312_without_tk(self):
        setup = SCRIPTS[0].read_text()
        self.assertIn("sys.version_info[:2] == (3, 12)", setup)
        self.assertIn("import qlib, pandas, pyarrow, plotly, webview", setup)
        self.assertNotIn("tkinter", setup)
        self.assertNotIn("python-tk", setup)
        self.assertNotIn("with Tk", setup)
        self.assertNotIn("qlib_yahoo.console", setup)

    def test_setup_runs_app_builder_without_requiring_archive_execute_bits(self):
        setup = SCRIPTS[0].read_text()
        self.assertIn('bash "$ROOT/scripts/macos/build_app.sh"', setup)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is optional outside the Qlib environment")
class IconTests(unittest.TestCase):
    def test_png_encoding(self):
        from qlib_yahoo.icon import png_bytes, render_icon

        for size in (16, 64, 256):
            data = png_bytes(size)
            self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(struct.unpack(">II", data[16:24]), (size, size))
        image = render_icon(64)
        self.assertEqual(image.shape, (64, 64, 4))
        self.assertEqual(image[0, 0, 3], 0)  # transparent corner
        self.assertEqual(image[32, 32, 3], 255)  # opaque tile
        with self.assertRaises(ValueError):
            render_icon(8)


@unittest.skipUnless(sys.platform == "darwin" and shutil.which("iconutil") and (ROOT / "env" / "qlib" / "bin" / "python").exists(),
                     "macOS with the project environment is required to build the app bundle")
class AppBundleTests(unittest.TestCase):
    def test_build_app_into_a_temporary_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            dest = Path(temporary)
            result = subprocess.run([str(ROOT / "scripts" / "macos" / "build_app.sh"), "--dest", str(dest), "--data-root", str(dest / "data")],
                                    capture_output=True, text=True, timeout=300)
            self.assertEqual(result.returncode, 0, result.stderr)
            app = dest / "Tubby Qlib Console.app"
            info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
            self.assertEqual(info["CFBundleExecutable"], "TubbyQlibConsole")
            self.assertEqual(info["CFBundleIdentifier"], "org.tubbylabs.qlib-console")
            self.assertFalse(info["NSRequiresAquaSystemAppearance"])
            launcher = app / "Contents" / "MacOS" / "TubbyQlibConsole"
            self.assertTrue(launcher.stat().st_mode & 0o111)
            self.assertEqual((app / "Contents" / "Resources" / "repo-path").read_text(), str(ROOT))
            self.assertEqual((app / "Contents" / "Resources" / "data-root").read_text(), str(dest / "data"))
            self.assertGreater((app / "Contents" / "Resources" / "AppIcon.icns").stat().st_size, 10_000)
            # A foreign bundle with the same name is never replaced.
            (app / "Contents" / "Resources" / "repo-path").unlink(missing_ok=True)
            refused = subprocess.run([str(ROOT / "scripts" / "macos" / "build_app.sh"), "--dest", str(dest)], capture_output=True, text=True)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("Refusing to replace", refused.stderr)


class SignedPinTests(unittest.TestCase):
    def test_platform_selection(self):
        from qlib_yahoo.dataset import DatasetError
        from qlib_yahoo.runtime import PIN, platform_pin

        pin = json.loads(PIN.read_text(encoding="utf-8"))
        self.assertIs(platform_pin(pin, "win32-amd64"), pin)
        mac = platform_pin(pin, "darwin-universal2")
        self.assertEqual(mac["commit"], "d10a5a8ca397ea088da8f5c996e8bd042eea27e0")
        self.assertEqual(mac["version"], "0.9.7+tubby.3")
        self.assertEqual(mac["branch"], "codex/executor-day-cache")
        self.assertNotEqual(mac["wheel_sha256"], pin["wheel_sha256"])
        self.assertTrue(mac["local_wheel"].endswith("macosx_10_13_universal2.whl"))
        with self.assertRaisesRegex(DatasetError, "No reviewed signed-engine build"):
            platform_pin(pin, "linux-x86_64")

    def test_status_is_informational(self):
        from qlib_yahoo.runtime import signed_runtime_status

        available, detail = signed_runtime_status()
        self.assertIsInstance(available, bool)
        self.assertTrue(detail)
        with patch("qlib_yahoo.runtime.platform_key", return_value="linux-x86_64"):
            available, detail = signed_runtime_status()
        self.assertFalse(available)


class DesktopSettingsTests(unittest.TestCase):
    def test_damaged_settings_fall_back_to_defaults(self):
        from qlib_yahoo.desktop import load_settings, save_settings

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "settings.json"
            self.assertEqual(load_settings(path), {})
            save_settings(path, {"alpha": "$close"})
            self.assertEqual(load_settings(path), {"alpha": "$close"})
            path.write_text("{not json", encoding="utf-8")
            self.assertEqual(load_settings(path), {})
            path.write_text("[1, 2]", encoding="utf-8")
            self.assertEqual(load_settings(path), {})

    def test_default_data_root_override(self):
        import os
        from qlib_yahoo import default_data_root

        with patch.dict(os.environ, {"TUBBY_QLIB_DATA_ROOT": "/tmp/qlib-data"}):
            self.assertEqual(default_data_root(), str(Path("/tmp/qlib-data")))


if __name__ == "__main__":
    unittest.main()
