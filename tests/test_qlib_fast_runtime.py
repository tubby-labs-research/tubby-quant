"""Dependency-free accepted-profile and archive integrity checks."""
import base64
import csv
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from qlib_yahoo import fast_runtime as runtime
from qlib_yahoo.dataset import DatasetError

INSTALLER_PATH = Path(__file__).absolute().parents[1] / "scripts/install_fast_runtime.py"
loader = importlib.util.spec_from_file_location("fast_runtime_installer_test", INSTALLER_PATH)
installer = importlib.util.module_from_spec(loader)
loader.loader.exec_module(installer)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def wheel(path, package="toy", extra=None):
    contents = {f"{package}/__init__.py": b"raise RuntimeError('must never import')\n",
                f"{package}-1.dist-info/METADATA": f"Name: {package}\nVersion: 1\n".encode()}
    contents.update(extra or {})
    record = f"{package}-1.dist-info/RECORD"
    rows = [[name, "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode(),
             str(len(raw))] for name, raw in contents.items()]
    rows.append([record, "", ""])
    stream = io.StringIO(newline="")
    csv.writer(stream).writerows(rows)
    contents[record] = stream.getvalue().encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, raw in contents.items():
            archive.writestr(name, raw)
    spec = {"filename": path.name, "sha256": runtime.sha256(path), "size": path.stat().st_size,
            "url": "https://files.pythonhosted.org/packages/test/" + path.name}
    return spec, contents


class FastRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).absolute()

    def test_wheel_record_and_full_bytes_are_checked(self):
        path = self.root / "toy.whl"
        spec, contents = wheel(path)
        self.assertEqual(runtime.wheel_contents(path, spec), contents)
        changed = dict(spec, sha256="0" * 64)
        with self.assertRaisesRegex(DatasetError, "SHA256"):
            runtime.wheel_contents(path, changed)
        with self.assertRaisesRegex(DatasetError, "size"):
            runtime.wheel_contents(path, dict(spec, size=1))

    def test_self_consistent_pin_cannot_hide_invalid_record(self):
        path = self.root / "toy.whl"
        spec, content = wheel(path)
        content["toy/__init__.py"] = b"tampered"
        with zipfile.ZipFile(path, "w") as archive:
            for name, raw in content.items():
                archive.writestr(name, raw)
        spec.update(sha256=runtime.sha256(path), size=path.stat().st_size)
        with self.assertRaisesRegex(DatasetError, "RECORD content"):
            runtime.wheel_contents(path, spec)

    def test_empty_directory_entries_in_official_wheels_are_supported(self):
        path = self.root / "directories.whl"
        spec, contents = wheel(path)
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("toy/", b"")
        spec.update(sha256=runtime.sha256(path), size=path.stat().st_size)
        self.assertEqual(runtime.wheel_contents(path, spec), contents)

    def test_directory_entries_cannot_hide_links_or_payloads(self):
        for name, mode, raw in (("../bad/", 0o040755, b""),
                                ("toy/", 0o120777, b""),
                                ("toy/", 0o040755, b"payload")):
            with self.subTest(name=name, mode=mode):
                path = self.root / "bad-directory.whl"
                spec, _ = wheel(path)
                item = zipfile.ZipInfo(name)
                item.external_attr = mode << 16
                with zipfile.ZipFile(path, "a") as archive:
                    archive.writestr(item, raw)
                spec.update(sha256=runtime.sha256(path), size=path.stat().st_size)
                with self.assertRaises(DatasetError):
                    runtime.wheel_contents(path, spec)

    def test_unsafe_wheel_paths_rejected(self):
        for name in ("../evil.py", "/evil.py", "a\\evil.py", "C:evil.py", "a/./evil.py"):
            with self.subTest(name=name):
                path = self.root / "unsafe.whl"
                spec, _ = wheel(path, extra={name: b"bad"})
                with self.assertRaises(DatasetError):
                    runtime.wheel_contents(path, spec)

    def test_link_archive_rejected(self):
        path = self.root / "link.whl"
        item = zipfile.ZipInfo("toy/__init__.py")
        item.external_attr = 0o120777 << 16
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(item, "elsewhere")
        spec = {"sha256": runtime.sha256(path), "size": path.stat().st_size}
        with self.assertRaisesRegex(DatasetError, "Unsafe"):
            runtime.wheel_contents(path, spec)

    def test_case_collision_rejected(self):
        path = self.root / "collision.whl"
        spec, _ = wheel(path, extra={"TOY/__init__.py": b"other"})
        with self.assertRaisesRegex(DatasetError, "duplicate"):
            runtime.wheel_contents(path, spec)

    def test_compiler_script_mapping_preserves_bytes(self):
        source = {"numba/__init__.py": b"module", "numba-1.data/scripts/numba": b"#!original\n"}
        self.assertEqual(runtime.compiler_contents(source),
                         {"numba/__init__.py": b"module", "bin/numba": b"#!original\n"})
        with self.assertRaises(DatasetError):
            runtime.compiler_contents({"numba-1.data/purelib/evil.py": b"bad"})

    def test_unrecorded_bytecode_is_rejected(self):
        target = self.root / "install"
        contents = {"toy/__init__.py": b"source"}
        installer.install_tree(target, contents)
        cache = target / "toy/__pycache__"
        cache.mkdir()
        (cache / "__init__.cpython-312.pyc").write_bytes(b"poison")
        with self.assertRaisesRegex(DatasetError, "Unrecorded"):
            runtime.verify_tree(target, contents)

    def test_tree_inventory_errors_precede_changed_bytes(self):
        target = self.root / "install"
        contents = {"first.py": b"first", "second.py": b"second"}
        installer.install_tree(target, contents)
        (target / "first.py").write_bytes(b"changed")
        (target / "unknown.py").write_bytes(b"unknown")
        with self.assertRaisesRegex(DatasetError, "Unrecorded or missing"):
            runtime.verify_tree(target, contents)
        (target / "unknown.py").unlink()
        (target / "second.py").unlink()
        with self.assertRaisesRegex(DatasetError, "Unrecorded or missing"):
            runtime.verify_tree(target, contents)

    def test_tree_byte_errors_follow_expected_order_not_traversal(self):
        target = self.root / "install"
        contents = {"first.py": b"first", "second.py": b"second"}
        installer.install_tree(target, contents)
        for name in contents:
            (target / name).write_bytes(b"changed")
        with patch.object(Path, "rglob", return_value=iter((target / "second.py", target / "first.py"))):
            with self.assertRaisesRegex(DatasetError, "archive: first.py"):
                runtime.verify_tree(target, contents)

    def test_tree_extra_files_are_inventoried_and_path_checked(self):
        target = self.root / "install"
        contents = {"module.py": b"module"}
        installer.install_tree(target, contents)
        extra = target / "receipt.json"
        extra.write_bytes(b"{}")
        with patch.object(runtime, "checked_path", wraps=runtime.checked_path) as checked:
            runtime.verify_tree(target, contents, extra=(extra.name,))
        self.assertIn(extra, [call.args[0] for call in checked.call_args_list])
        extra.unlink()
        with self.assertRaisesRegex(DatasetError, "Unrecorded or missing"):
            runtime.verify_tree(target, contents, extra=(extra.name,))

    def test_tree_rejects_redirected_directory_and_file_ancestor(self):
        target = self.root / "install"
        contents = {"toy/module.py": b"module"}
        installer.install_tree(target, contents)
        ancestor = target / "toy"
        original_lstat = Path.lstat
        def redirected(path, *args, **kwargs):
            info = original_lstat(path, *args, **kwargs)
            if path == ancestor:
                return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
            return info
        # The directory itself and a known file's ancestor must both be checked,
        # even when traversal has yielded the file before its parent directory.
        for entries in ((ancestor,), (ancestor / "module.py",)):
            with self.subTest(entries=entries), patch.object(Path, "lstat", redirected), \
                    patch.object(Path, "rglob", return_value=iter(entries)), \
                    patch.object(Path, "read_bytes", side_effect=AssertionError("redirected read")):
                with self.assertRaisesRegex(DatasetError, "Redirected"):
                    runtime.verify_tree(target, contents)

    def test_tree_validates_expected_member_and_checks_path_before_read(self):
        target = self.root / "install"
        contents = {"module.py": b"module"}
        installer.install_tree(target, contents)
        events = []
        original_check, original_read = runtime.checked_path, Path.read_bytes
        def checked(path, **kwargs):
            result = original_check(path, **kwargs)
            events.append(("check", path))
            return result
        def read(path):
            self.assertEqual(events[-1], ("check", path))
            events.append(("read", path))
            return original_read(path)
        with patch.object(runtime, "checked_path", checked), patch.object(Path, "read_bytes", read), \
                patch.object(runtime, "member_name", wraps=runtime.member_name) as names:
            runtime.verify_tree(target, contents)
        names.assert_called_once_with("module.py")
        self.assertEqual(events.count(("check", target / "module.py")), 1)

    def test_tree_unsafe_expected_alias_cannot_escape_inventory(self):
        target = self.root / "install"
        installer.install_tree(target, {"module.py": b"module"})
        with patch.object(Path, "read_bytes", side_effect=AssertionError("unsafe read")):
            with self.assertRaisesRegex(DatasetError, "Unrecorded or missing"):
                runtime.verify_tree(target, {"../module.py": b"module"})

    def test_status_mac_does_not_access_windows_files(self):
        with patch.object(runtime.sys, "platform", "darwin"), \
                patch.object(runtime, "read_json", side_effect=AssertionError("filesystem")):
            ready, message = runtime.fast_runtime_status()
        self.assertTrue(ready)
        self.assertIn("portable", message)

    def test_tree_reuse_tamper_and_extra_importables(self):
        target = self.root / "install"
        contents = {"toy/__init__.py": b"original"}
        installer.install_tree(target, contents)
        installer.install_tree(target, contents)
        (target / "evil.py").write_bytes(b"other")
        with self.assertRaisesRegex(DatasetError, "Unrecorded"):
            installer.install_tree(target, contents)
        self.assertEqual((target / "evil.py").read_bytes(), b"other")
        (target / "evil.py").unlink()
        (target / "toy/__init__.py").write_bytes(b"changed")
        with self.assertRaisesRegex(DatasetError, "differs"):
            installer.install_tree(target, contents)
        self.assertEqual((target / "toy/__init__.py").read_bytes(), b"changed")

    def test_missing_windows_config_is_actionable(self):
        with patch.object(runtime, "supported_windows", return_value=True), \
                patch.object(runtime, "CONFIG", self.root / "absent.json"), \
                patch.object(runtime.sys, "platform", "win32"):
            with self.assertRaisesRegex(DatasetError, "install_fast_runtime.py"):
                runtime.resolve_fast_profile()

    def test_complete_profile_verifies_without_importing_packages(self):
        deps = self.root / "runtime-deps"
        deps.mkdir()
        (deps / "wheels").mkdir()
        compiler_spec, compiler = wheel(deps / "wheels/compiler.whl", "compiler_toy")
        installer.install_tree(deps / "compiler", compiler)
        gzip = deps / "gzip"
        gzip.mkdir()
        gzip_spec, gzip_contents = wheel(gzip / "gzip.whl", "gzip_toy")
        installer.install_tree(gzip / "site-packages", gzip_contents)
        source = self.root / "source.pyx"
        source.write_bytes(b"reviewed source")
        native = deps / "native"
        installer.install_tree(native, {"helper.pyd": b"native bytes", "provenance.json": b"{}"})
        pin = {"schema_version": 1, "kind": "accepted_signed_runtime_profile", "name": "accepted-v1",
               "compiler": {"folder": "compiler", "wheels": [compiler_spec]},
               "gzip": {"folder": "gzip", "wheels": [gzip_spec]},
               "date_range": {"folder": "native", "artifact": "helper.pyd",
                              "artifact_sha256": digest(b"native bytes"), "provenance_sha256": digest(b"{}"),
                              "source_sha256": {"source.pyx": digest(b"reviewed source")}}}
        pin_path = self.root / "pin.json"
        pin_path.write_text(json.dumps(pin))
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps({"schema_version": 1, "profile": "accepted-v1",
                                          "dependency_root": str(deps),
                                          "profile_pin_sha256": runtime.sha256(pin_path)}))
        before = {name for name in sys.modules if name == "qlib" or name.startswith("qlib.")}
        with patch.object(runtime, "ROOT", self.root), patch.object(runtime, "PIN", pin_path), \
                patch.object(runtime, "CONFIG", config_path), \
                patch.object(runtime, "supported_windows", return_value=True), \
                patch.object(runtime.sys, "platform", "win32"):
            options, identity = runtime.resolve_fast_profile()
            self.assertEqual(options["_compiler_path"], str(deps / "compiler"))
            self.assertTrue(options["_ordered_session"])
            self.assertEqual(identity["selection"], "accepted_windows")
            self.assertTrue(runtime.fast_runtime_status()[0])
            source.write_bytes(b"changed source")
            with self.assertRaisesRegex(DatasetError, "SHA256"):
                runtime.resolve_fast_profile()
        after = {name for name in sys.modules if name == "qlib" or name.startswith("qlib.")}
        self.assertEqual(before, after)
        self.assertNotIn("compiler_toy", sys.modules)
        self.assertNotIn("gzip_toy", sys.modules)

    def test_mac_portable_profile_never_reads_windows_paths(self):
        with patch.object(runtime.sys, "platform", "darwin"), \
                patch.object(runtime, "read_json", side_effect=AssertionError("filesystem read")), \
                patch.object(runtime, "supported_windows", side_effect=AssertionError("Windows ABI check")):
            options, identity = runtime.resolve_fast_profile()
        self.assertEqual(set(options), {"_columnar_history", "_flat_indicators", "_buffered_history", "_index_reuse"})
        self.assertTrue(all(options.values()))
        self.assertFalse(identity["ordered_session"])
        self.assertEqual(identity["gzip"], "stdlib-zlib")
        self.assertEqual(identity["date_range"], "python")

    def test_unsupported_profile_has_no_filesystem_access(self):
        with patch.object(runtime.sys, "platform", "linux"), \
                patch.object(runtime, "supported_windows", return_value=False), \
                patch.object(runtime, "read_json", side_effect=AssertionError("filesystem read")):
            options, identity = runtime.resolve_fast_profile()
        self.assertEqual(options, {})
        self.assertIn("Unsupported", identity["fallback_reason"])

    def test_installer_refuses_unsupported_host_before_engine(self):
        with patch.object(installer.sys, "platform", "linux"), \
                patch.object(runtime, "supported_windows", return_value=False), \
                patch.object(installer.subprocess, "run", side_effect=AssertionError("engine invoked")):
            with self.assertRaisesRegex(DatasetError, "CPython"):
                installer.configure(self.root)

    def test_official_download_urls_only(self):
        for url in ("http://files.pythonhosted.org/packages/x", "https://evil.test/packages/x",
                    "https://files.pythonhosted.org/other/x", "https://files.pythonhosted.org/packages/x?q=1"):
            with self.subTest(url=url), self.assertRaises(DatasetError):
                installer.download_wheel({"filename": "toy.whl", "url": url}, self.root / "toy.whl")

    def test_download_uses_pinned_url_bounded_pip_and_verifies_before_promotion(self):
        path = self.root / "source.whl"
        spec, _ = wheel(path)
        destination = self.root / "downloads" / path.name
        def downloaded(command, **kwargs):
            import shutil
            self.assertEqual(kwargs["timeout"], 900)
            self.assertIn("--no-deps", command)
            self.assertIn("--no-index", command)
            self.assertIn("--no-cache-dir", command)
            self.assertEqual(command[-1], spec["url"])
            shutil.copyfile(path, Path(command[command.index("--dest") + 1]) / path.name)
            return type("Result", (), {"returncode": 0})()
        with patch.object(installer.subprocess, "run", side_effect=downloaded), \
                patch.object(installer.sys, "platform", "linux"):
            installer.download_wheel(spec, destination)
        self.assertEqual(runtime.sha256(destination), spec["sha256"])

    def test_download_timeout_and_wrong_bytes_never_promote(self):
        spec = {"filename": "toy.whl", "url": "https://files.pythonhosted.org/packages/test/toy.whl",
                "sha256": "0" * 64, "size": 3}
        destination = self.root / "downloads/toy.whl"
        with patch.object(installer.subprocess, "run", side_effect=installer.subprocess.TimeoutExpired("pip", 240)):
            with self.assertRaisesRegex(DatasetError, "timed out"):
                installer.download_wheel(spec, destination)
        self.assertFalse(destination.exists())

        def wrong(command, **kwargs):
            (Path(command[command.index("--dest") + 1]) / "toy.whl").write_bytes(b"bad")
            return type("Result", (), {"returncode": 0})()
        with patch.object(installer.subprocess, "run", side_effect=wrong), \
                patch.object(installer.sys, "platform", "linux"):
            with self.assertRaisesRegex(DatasetError, "SHA256"):
                installer.download_wheel(spec, destination)
        self.assertFalse(destination.exists())

    @unittest.skipUnless(sys.platform == "win32", "Windows OS curl path")
    def test_windows_download_uses_os_curl_and_keeps_scratch_on_data_drive(self):
        path = self.root / "source.whl"
        spec, _ = wheel(path)
        destination = self.root / "downloads" / path.name
        def downloaded(command, **kwargs):
            import os
            import shutil
            self.assertEqual(Path(command[0]), Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/curl.exe")
            self.assertEqual(kwargs["timeout"], 900)
            self.assertEqual(command[-1], spec["url"])
            candidate = Path(command[command.index("--output") + 1])
            self.assertTrue(candidate.is_relative_to(destination.parent))
            self.assertEqual(kwargs["env"]["TEMP"], str(candidate.parent))
            shutil.copyfile(path, candidate)
            return type("Result", (), {"returncode": 0})()
        with patch.object(installer.subprocess, "run", side_effect=downloaded):
            installer.download_wheel(spec, destination)
        self.assertEqual(runtime.sha256(destination), spec["sha256"])

    def test_native_archive_digest_checked_before_extraction(self):
        artifact = b"not executed"
        provenance = b"{}"
        asset = "native.zip"
        source = self.root / asset
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("helper.pyd", artifact)
            archive.writestr("provenance.json", provenance)
        spec = {"artifact": "helper.pyd", "artifact_sha256": digest(artifact),
                "provenance_sha256": digest(provenance), "release_repository": "tubby-labs-research/Quant",
                "release_tag": "accepted-v1", "asset": asset, "asset_sha256": "0" * 64}
        def downloaded(command, **kwargs):
            import shutil
            shutil.copyfile(source, Path(command[-1]) / asset)
            return type("Result", (), {"returncode": 0})()
        with patch.object(installer.subprocess, "run", side_effect=downloaded):
            with self.assertRaisesRegex(DatasetError, "SHA256"):
                installer.install_native(spec, self.root / "native")
            self.assertFalse((self.root / "native").exists())
            spec["asset_sha256"] = runtime.sha256(source)
            installer.install_native(spec, self.root / "native")
        self.assertEqual((self.root / "native/helper.pyd").read_bytes(), artifact)

    def test_duplicate_json_keys_and_relative_paths_rejected(self):
        path = self.root / "config.json"
        path.write_text('{"schema":1,"schema":2}')
        with self.assertRaisesRegex(DatasetError, "Duplicate"):
            runtime.read_json(path)
        with self.assertRaisesRegex(DatasetError, "absolute"):
            runtime.checked_path(Path("relative"), missing=True)

    def test_setup_delegates_engine_once_to_profile_installer(self):
        root = INSTALLER_PATH.parents[1]
        for name in ("scripts/windows/setup.ps1", "scripts/macos/setup.sh"):
            source = (root / name).read_text()
            self.assertIn("install_fast_runtime.py", source)
            self.assertNotIn("install_signed_runtime.py", source)


if __name__ == "__main__":
    unittest.main()
