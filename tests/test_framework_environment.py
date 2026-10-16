"""Synthetic installed-distribution fixtures; never mutate real packages."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import verify_framework_environment as verifier


class FrameworkEnvironmentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.prefix = Path(temporary.name) / "venv"
        self.site = self.prefix / "lib" / "python3.12" / "site-packages"
        self.site.mkdir(parents=True)
        self.dist = self.site / "demo-1.0.dist-info"
        self.dist.mkdir()
        self.write("demo-1.0.dist-info/METADATA", "Name: demo\nVersion: 1.0\n")
        self.write("demo/__init__.py", "# synthetic\n")
        self.record("demo/__init__.py", "demo-1.0.dist-info/METADATA", "demo-1.0.dist-info/RECORD")

    def write(self, relative, text):
        target = self.site / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    def record(self, *entries):
        self.write("demo-1.0.dist-info/RECORD", "".join(f"{entry},,\n" for entry in entries))

    def check(self):
        return verifier.verify_distributions(self.prefix, [self.site, self.site])

    def test_complete_distribution_and_duplicate_roots(self):
        self.assertEqual(self.check(), (1, []))

    def test_partial_dist_info_missing_metadata(self):
        (self.dist / "METADATA").unlink()
        count, issues = self.check()
        self.assertEqual(count, 1)
        self.assertEqual(issues, ["distribution: missing or invalid Name"])

    def test_missing_or_invalid_metadata_fields(self):
        for text, reason in (
            ("Version: 1.0\n", "Name"),
            ("Name: demo\n", "Version"),
            ("Name: /private/credential\nVersion: 1.0\n", "Name"),
            ("Name: demo\nVersion: \n", "Version"),
        ):
            with self.subTest(reason=reason, text=text):
                self.write("demo-1.0.dist-info/METADATA", text)
                issues = self.check()[1]
                self.assertIn(reason, issues[0])
                self.assertNotIn("credential", " ".join(issues))

    def test_missing_and_empty_record(self):
        (self.dist / "RECORD").unlink()
        self.assertEqual(self.check()[1], ["demo: missing or empty RECORD"])
        self.write("demo-1.0.dist-info/RECORD", "\n")
        self.assertEqual(self.check()[1], ["demo: missing or empty RECORD"])

    def test_missing_yaml_tokens_is_detected_without_importing(self):
        self.write("demo-1.0.dist-info/METADATA", "Name: PyYAML\nVersion: 6.0.2\n")
        self.record("yaml/tokens.py")
        self.assertEqual(self.check()[1], ["PyYAML: declared file missing"])

    def test_bytecode_is_optional(self):
        self.record("demo/__init__.py", "demo/__pycache__/missing.pyc", "demo/missing.pyo")
        self.assertEqual(self.check(), (1, []))

    def test_scripts_within_virtualenv_are_checked(self):
        self.write("../../../bin/demo", "# synthetic entrypoint\n")
        self.record("../../../bin/demo")
        self.assertEqual(self.check(), (1, []))
        (self.prefix / "bin" / "demo").unlink()
        self.assertEqual(self.check()[1], ["demo: declared file missing"])

    def test_absolute_and_escaping_paths_fail_without_path_disclosure(self):
        for entry in ("../../../../private/credential", "/private/credential", "C:/private/credential", "C:credential", "\\\\server\\private\\credential"):
            with self.subTest(entry=entry):
                self.record(entry)
                self.assertEqual(self.check()[1], ["demo: RECORD path outside virtualenv"])

    def test_external_symlink_fails(self):
        outside = self.prefix.parent / "external.py"
        outside.write_text("# synthetic\n", encoding="utf-8")
        link = self.site / "demo" / "linked.py"
        try:
            link.symlink_to(outside)
        except OSError:
            self.skipTest("Creating symlinks requires platform permission")
        self.record("demo/linked.py")
        self.assertEqual(self.check()[1], ["demo: RECORD path outside virtualenv"])

    def test_editable_record_can_be_local(self):
        self.write("__editable__.demo.pth", "/external/source\n")
        self.record("__editable__.demo.pth", "demo-1.0.dist-info/METADATA", "demo-1.0.dist-info/RECORD")
        self.assertEqual(self.check(), (1, []))  # Does not follow .pth contents.

    def test_legacy_egg_info_without_record_fails_clearly(self):
        egg = self.site / "legacy.egg-info"
        egg.mkdir()
        (egg / "PKG-INFO").write_text("Name: legacy\nVersion: 1.0\n", encoding="utf-8")
        self.assertEqual(self.check()[1], ["legacy: missing or empty RECORD"])

    def test_system_site_packages_are_not_enumerated(self):
        system = self.prefix.parent / "system-site-packages"
        system.mkdir()
        (system / "broken.dist-info").mkdir()
        with patch.object(verifier.metadata, "PathDistribution", wraps=verifier.metadata.PathDistribution) as discover:
            self.assertEqual(self.check(), (1, []))
        discover.assert_called_once_with(self.dist.resolve())
        self.assertEqual(verifier.verify_distributions(self.prefix, [system])[1], ["site-packages: outside virtualenv or missing"])

    def test_metadata_escape_is_rejected_before_reading(self):
        original_resolve = Path.resolve
        outside = self.prefix.parent / "external-metadata"

        def resolve(path, *args, **kwargs):
            if path == self.dist / "METADATA":
                return outside
            return original_resolve(path, *args, **kwargs)

        with patch.object(Path, "resolve", resolve), patch.object(verifier.metadata, "PathDistribution", side_effect=AssertionError("external metadata read")):
            self.assertEqual(self.check()[1], ["distribution: metadata outside virtualenv"])

    def test_success_exit(self):
        with patch.object(verifier.sys, "prefix", str(self.prefix)), patch.object(verifier.sys, "base_prefix", str(self.prefix.parent)), patch.object(verifier.sysconfig, "get_path", return_value=str(self.site)), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(verifier.main(), 0)
        self.assertIn("1 local distributions", output.getvalue())

    def test_record_hashes_are_not_checked_and_package_content_not_read(self):
        self.write("demo-1.0.dist-info/RECORD", "demo/__init__.py,sha256=deliberately-wrong,99999\n")
        with patch.object(Path, "read_bytes", side_effect=AssertionError("package bytes read")):
            self.assertEqual(self.check(), (1, []))

    def test_invalid_record(self):
        for record in ("demo/__init__.py\n", ",,\n", '"unterminated,,\n'):
            with self.subTest(record=record):
                self.write("demo-1.0.dist-info/RECORD", record)
                self.assertTrue(self.check()[1])

    def test_empty_site_packages(self):
        empty = self.prefix / "empty-site-packages"
        empty.mkdir()
        self.assertEqual(verifier.verify_distributions(self.prefix, [empty]), (0, ["site-packages: no installed distributions"]))

    def test_failure_exit_and_recovery_instruction(self):
        output = io.StringIO()
        with patch.object(verifier.sys, "prefix", str(self.prefix)), patch.object(verifier.sys, "base_prefix", str(self.prefix.parent)), patch.object(verifier.sysconfig, "get_path", return_value=str(self.site)), contextlib.redirect_stderr(output):
            self.record("yaml/tokens.py")
            self.assertEqual(verifier.main(), 1)
        self.assertIn("-Recreate", output.getvalue())
        self.assertIn("--recreate", output.getvalue())
        self.assertIn("moves the old environment aside", output.getvalue())
        self.assertNotIn(str(self.prefix), output.getvalue())

    def test_non_virtualenv_rejected(self):
        with patch.object(verifier.sys, "prefix", "same"), patch.object(verifier.sys, "base_prefix", "same"), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(verifier.main(), 1)

    def test_setup_call_order_and_windows_format(self):
        root = Path(__file__).resolve().parents[1]
        windows = (root / "scripts/windows/setup.ps1").read_bytes()
        self.assertTrue(windows.startswith(b"\xef\xbb\xbf"))
        self.assertNotIn(b"\n", windows.replace(b"\r\n", b""))
        for path in ("scripts/windows/setup.ps1", "scripts/macos/setup.sh"):
            text = (root / path).read_text(encoding="utf-8-sig")
            check = '"check")' if path.endswith(".ps1") else "-m pip check"
            self.assertLess(text.index(check), text.index("verify_framework_environment.py"))
            self.assertLess(text.index("verify_framework_environment.py"), text.index("import qlib, pandas"))


if __name__ == "__main__":
    unittest.main()
