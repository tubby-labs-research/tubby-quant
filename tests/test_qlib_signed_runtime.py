"""Runtime pin tests use tiny synthetic archives, not a local Qlib install."""

import hashlib
import io
import importlib.util
import json
from pathlib import Path
import subprocess
import py_compile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from qlib_yahoo.dataset import DatasetError
from qlib_yahoo.runtime import verify_signed_install, run_signed_backtest, _SourceOnlyLoader
from qlib_yahoo import runtime as runtime_module


class SignedRuntimeTests(unittest.TestCase):
    def fixture(self, root, *, dirty=False):
        package = root / "runtime" / "qlib"
        package.mkdir(parents=True)
        build = {"commit": "our-commit", "upstream_commit": "upstream",
                 "base_wheel_sha256": "base", "dirty": dirty}
        files = {"qlib/__init__.py": b"__version__ = 'test'\n",
                 "qlib/_tubby_build.json": json.dumps(build).encode()}
        wheel = root / "test.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            for name, data in files.items():
                archive.writestr(name, data)
                (root / "runtime" / name).write_bytes(data)
        return {**build, "local_wheel": "test.whl", "local_runtime": "runtime",
                "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}

    def test_verified_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            pin = self.fixture(root)
            runtime, build = verify_signed_install(pin, root)
            self.assertEqual(runtime, root / "runtime")
            self.assertEqual(build["commit"], "our-commit")

    def test_modified_installed_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pin = self.fixture(root)
            (root / "runtime/qlib/__init__.py").write_text("changed")
            with self.assertRaisesRegex(DatasetError, "differs"):
                verify_signed_install(pin, root)

    def test_shadowing_package_at_runtime_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pin = self.fixture(root)
            (root / "runtime/bin").mkdir()
            (root / "runtime/bin/qlib-cli").write_text("#!/bin/sh")  # console scripts are not importable
            verify_signed_install(pin, root)
            (root / "runtime/pandas").mkdir()
            (root / "runtime/pandas/__init__.py").write_text("raise SystemExit(7)")
            with self.assertRaisesRegex(DatasetError, "importable entry: pandas"):
                verify_signed_install(pin, root)

    def test_unreviewed_module(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pin = self.fixture(root)
            (root / "runtime/qlib/extra.py").write_text("changed")
            with self.assertRaisesRegex(DatasetError, "unreviewed"):
                verify_signed_install(pin, root)

    def test_wheel_and_identity_guards(self):
        for change in ({"wheel_sha256": "bad"}, {"commit": "bad"},
                       {"local_wheel": "../test.whl"}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                pin = self.fixture(root)
                pin.update(change)
                with self.assertRaises(DatasetError):
                    verify_signed_install(pin, root)

    def test_dirty_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pin = self.fixture(root, dirty=True)
            with self.assertRaisesRegex(DatasetError, "clean build"):
                verify_signed_install(pin, root)

    def test_timestamp_valid_bytecode_cannot_replace_verified_source(self):
        import os
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "example.py"
            source.write_text("VALUE = 'wrong'\n")
            stamp = source.stat().st_mtime_ns
            py_compile.compile(str(source), doraise=True)
            source.write_text("VALUE = 'right'\n")  # Same length and timestamp.
            os.utime(source, ns=(stamp, stamp))
            self.assertTrue(Path(importlib.util.cache_from_source(str(source))).is_file())
            namespace = {}
            exec(_SourceOnlyLoader("example", str(source)).get_code("example"), namespace)
            self.assertEqual(namespace["VALUE"], "right")

    def test_console_request_is_json_not_code(self):
        expression = '"; arbitrary text $close'
        result = subprocess.CompletedProcess([], 0, '__QUANT_SIGNED_RUN__="run-path"\n', '')
        with patch("qlib_yahoo.runtime.subprocess.run", return_value=result) as run:
            self.assertEqual(run_signed_backtest(Path("dataset"), expression, gross_exposure=.8), Path("run-path"))
        command = run.call_args.args[0]
        self.assertNotIn(expression, command[2])
        self.assertEqual(json.loads(command[3])["alpha"], expression)
        self.assertEqual(json.loads(command[3])["strategy_mode"], "signed")

    def test_console_failure_not_completed(self):
        result = subprocess.CompletedProcess([], 1, '', 'failure')
        with patch("qlib_yahoo.runtime.subprocess.run", return_value=result):
            with self.assertRaisesRegex(DatasetError, "failure"):
                run_signed_backtest(Path("dataset"), "$close")

    def test_console_logging_supports_legacy_windows_encoding(self):
        result = subprocess.CompletedProcess([], 0, '__QUANT_SIGNED_RUN__="run-path"\n', '\u2588\u2588 progress')
        buffer = io.BytesIO()
        stream = io.TextIOWrapper(buffer, encoding="cp1252")
        with patch("qlib_yahoo.runtime.subprocess.run", return_value=result), patch("sys.stdout", stream):
            self.assertEqual(run_signed_backtest(Path("dataset"), "$close"), Path("run-path"))
        stream.flush()
        self.assertIn(b"?? progress", buffer.getvalue())

    def pin_fixture(self, root):
        pin = self.fixture(root)
        pin.update(repository="fixture-repository", version="fixture-version", build_method="fixture-build")
        docs = root / "docs"
        docs.mkdir()
        default = docs / "qlib-derivative-pin.json"
        candidate = docs / "qlib-derivative-pin.fixture-candidate.json"
        for path in (default, candidate):
            path.write_text(json.dumps(pin), encoding="utf-8")
        return pin, default, candidate

    def activate_fixture(self, root, default, candidate=None):
        fake_qlib = SimpleNamespace(__file__=str(root / "runtime/qlib/__init__.py"), __version__="fixture-version")
        with patch.object(runtime_module, "ROOT", root), patch.object(runtime_module, "PIN", default), \
                patch.object(runtime_module.sys, "version_info", (3, 12)), \
                patch.object(runtime_module, "platform_key", return_value="win32-amd64"), \
                patch.object(runtime_module, "verify_signed_install", side_effect=lambda pin: verify_signed_install(pin, root)) as verify, \
                patch.object(runtime_module.sys, "path", list(runtime_module.sys.path)), \
                patch.object(runtime_module.sys, "meta_path", list(runtime_module.sys.meta_path)), \
                patch.dict(runtime_module.sys.modules, {"qlib": None}), \
                patch.object(runtime_module.importlib, "import_module", return_value=fake_qlib):
            identity = runtime_module.ensure_signed_runtime(pin_path=candidate)
            self.assertEqual(verify.call_count, 1)
            self.assertIsInstance(runtime_module.sys.meta_path[0], runtime_module._SignedSourceFinder)
        return identity

    def test_explicit_candidate_retains_default_pin_and_full_install_verification(self):
        original_default_bytes = runtime_module.PIN.read_bytes()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            pin, default, candidate = self.pin_fixture(root)
            default_bytes = default.read_bytes()
            identity = self.activate_fixture(root, default, candidate)
            self.assertTrue(identity["installed_package_verified"])
            self.assertEqual(identity["commit"], pin["commit"])
            self.assertEqual(default.read_bytes(), default_bytes)
            default_identity = self.activate_fixture(root, default)
            self.assertEqual(default_identity, identity)
        self.assertEqual(runtime_module.PIN.read_bytes(), original_default_bytes)

    def test_candidate_does_not_relax_installed_bytes_wheel_or_extra_source_guards(self):
        for corruption in ("installed", "wheel", "extra_source"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temp:
                root = Path(temp).resolve()
                _pin, default, candidate = self.pin_fixture(root)
                if corruption == "installed":
                    (root / "runtime/qlib/__init__.py").write_text("UNREVIEWED = True\n")
                elif corruption == "wheel":
                    (root / "test.whl").write_bytes(b"changed wheel")
                else:
                    (root / "runtime/qlib/extra.py").write_text("UNREVIEWED = True\n")
                with self.assertRaises(DatasetError):
                    self.activate_fixture(root, default, candidate)

    def test_missing_outside_nonversioned_and_symlink_candidate_paths_rejected_before_install(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            _pin, default, candidate = self.pin_fixture(root)
            outside = root / candidate.name
            outside.write_bytes(candidate.read_bytes())
            (root / "docs/qlib-derivative-pin..json").write_bytes(candidate.read_bytes())
            invalid = (default, root / "docs/qlib-derivative-pin..json", outside,
                       root / "docs/qlib-derivative-pin.missing.json", root / "docs/other.json")
            with patch.object(runtime_module, "ROOT", root), patch.object(runtime_module, "PIN", default), \
                    patch.object(runtime_module.sys, "version_info", (3, 12)), \
                    patch.object(runtime_module, "platform_key", return_value="win32-amd64"), \
                    patch.object(runtime_module, "verify_signed_install") as verify:
                for path in invalid:
                    with self.subTest(path=path), self.assertRaises(DatasetError):
                        runtime_module.ensure_signed_runtime(pin_path=path)
                    verify.assert_not_called()
                with patch.object(Path, "is_symlink", autospec=True, side_effect=lambda path: path == candidate), \
                        self.assertRaisesRegex(DatasetError, "explicit versioned"):
                    runtime_module.ensure_signed_runtime(pin_path=candidate)
                verify.assert_not_called()

    def test_candidate_still_rejects_preimported_or_unverified_source_qlib(self):
        for wrong_path in (True, False):
            with self.subTest(wrong_path=wrong_path), tempfile.TemporaryDirectory() as temp:
                root = Path(temp).resolve()
                _pin, default, candidate = self.pin_fixture(root)
                package = root / ("other/qlib/__init__.py" if wrong_path else "runtime/qlib/__init__.py")
                loaded = SimpleNamespace(__file__=str(package), __version__="fixture-version")
                with patch.object(runtime_module, "ROOT", root), patch.object(runtime_module, "PIN", default), \
                        patch.object(runtime_module.sys, "version_info", (3, 12)), \
                        patch.object(runtime_module, "platform_key", return_value="win32-amd64"), \
                        patch.object(runtime_module, "verify_signed_install", side_effect=lambda pin: verify_signed_install(pin, root)), \
                        patch.object(runtime_module.sys, "meta_path", []), \
                        patch.dict(runtime_module.sys.modules, {"qlib": loaded}), \
                        patch.object(runtime_module.importlib, "import_module") as imported:
                    with self.assertRaisesRegex(DatasetError, "already imported|source-only verification"):
                        runtime_module.ensure_signed_runtime(pin_path=candidate)
                    imported.assert_not_called()


if __name__ == "__main__":
    unittest.main()
