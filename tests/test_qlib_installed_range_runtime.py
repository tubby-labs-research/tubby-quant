"""Portable offline deployment checks; native initialization is always faked."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

from qlib_yahoo import range_runtime as runtime
from qlib_yahoo.dataset import DatasetError
from scripts import package_qlib_fast_runtime as packager


SUPPORTED = {"implementation": "cpython", "python_abi": "cpython-312",
             "python_version": [3, 12], "platform": "win32", "build_platform": "win-amd64",
             "machine": "AMD64", "extension_suffix": runtime.SUFFIX}


class InstalledRangeRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pin_path = self.root / "docs/qlib-fast-runtime.v1.json"
        self.pin_path.parent.mkdir()
        self.installation = self.root / "data/runtime-deps" / runtime.INSTALLED_FOLDER
        self.installation.mkdir(parents=True)
        self.artifact = self.installation / ("session_loops" + runtime.SUFFIX)
        self.artifact_raw = b"fake reviewed buffer extension"
        self.artifact.write_bytes(self.artifact_raw)
        self.artifact_sha = runtime._sha(self.artifact_raw)
        self.sources = {}
        for name in runtime.BUFFER_SOURCES:
            source = self.root / name
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("reviewed source " + name, encoding="utf-8")
            self.sources[name] = runtime._sha(source.read_bytes())
        self.provenance = {"schema_version": 1, "kind": "sanitized_date_range_build_provenance",
                           "implementation": "public_buffer_leases", "source_sha256": self.sources,
                           "artifact_sha256": self.artifact_sha, "numpy_c_api_used": False,
                           "gil_held": True, "public_api_only_certified": False,
                           "installation_healthy": False, "allow_incomplete_toolchain": True,
                           "private_import_tokens": ["_Py_Dealloc"],
                           "api_audit_limitation": "Private imports; no public-only claim"}
        self.pin = {"schema_version": 1, "kind": "accepted_signed_runtime_profile", "name": "accepted-v1",
                    "date_range": {"folder": runtime.INSTALLED_FOLDER, "artifact": self.artifact.name,
                                   "artifact_sha256": self.artifact_sha, "source_sha256": self.sources}}
        self.persist()
        self.buffer_pin = self.root / "docs/buffer-pin.json"
        for name, value in (("ROOT", self.root), ("BUFFER_PIN", self.buffer_pin),
                            ("INSTALLED_ARTIFACT_SHA256", self.artifact_sha),
                            ("INSTALLED_SOURCES", self.sources)):
            context = patch.object(runtime, name, value)
            context.start()
            self.addCleanup(context.stop)
        context = patch.object(runtime, "_host", return_value=SUPPORTED)
        context.start()
        self.addCleanup(context.stop)
        context = patch.dict(runtime._OWNED, {}, clear=True)
        context.start()
        self.addCleanup(context.stop)
        context = patch.dict(sys.modules)
        context.start()
        self.addCleanup(context.stop)
        sys.modules.pop(runtime.MODULE, None)

    def persist(self):
        provenance_path = self.installation / "provenance.json"
        provenance_path.write_text(json.dumps(self.provenance), encoding="utf-8")
        self.pin["date_range"]["provenance_sha256"] = runtime._sha(provenance_path.read_bytes())
        self.pin_path.write_text(json.dumps(self.pin), encoding="utf-8")

    def reject(self, installation=None):
        with patch.object(runtime.importlib.machinery, "ExtensionFileLoader") as factory:
            with self.assertRaises(DatasetError):
                runtime.activate_installed_range(self.installation if installation is None else installation)
            factory.assert_not_called()

    def fake_loader(self):
        fake = types.ModuleType(runtime.MODULE)
        fake.__file__ = str(self.artifact)
        fake.try_valid_range = lambda *args: (True, True)
        return fake

    def test_verified_owned_reuse_and_truthful_identity(self):
        fake = self.fake_loader()
        with patch.object(runtime.importlib.machinery, "ExtensionFileLoader") as factory, \
                patch.object(runtime.importlib.util, "spec_from_file_location", return_value=object()), \
                patch.object(runtime.importlib.util, "module_from_spec", return_value=fake):
            function, identity = runtime.activate_installed_range(self.installation)
            self.assertIs(function, fake.try_valid_range)
            self.assertEqual(identity["selection"], "accepted_cython_date_range")
            self.assertEqual(identity["implementation"], "public_buffer_leases")
            self.assertEqual(identity["trusted_pin_path"], str(self.pin_path))
            self.assertEqual(identity["source_sha256"], self.sources)
            self.assertEqual(identity["installation_root"], str(self.installation))
            self.assertEqual(identity["artifact_sha256"], self.artifact_sha)
            self.assertFalse(identity["public_api_only_certified"])
            self.assertFalse(identity["installation_healthy"])
            self.assertTrue(identity["gil_held"])
            self.assertFalse(identity["numpy_c_api_used"])
            self.assertIs(runtime.activate_installed_range(self.installation)[0], function)
            factory.assert_called_once()
            fake.try_valid_range = lambda *args: False
            self.reject()

    def test_public_rebuild_requires_its_own_code_owned_identity(self):
        public_root = self.installation.with_name(runtime.PUBLIC_INSTALLED_FOLDER)
        self.installation.rename(public_root)
        self.installation = public_root
        self.artifact = public_root / self.artifact.name
        self.pin["date_range"]["folder"] = runtime.PUBLIC_INSTALLED_FOLDER
        self.persist()
        # A caller-edited descriptor alone cannot authorize another artifact.
        self.reject()
        with patch.object(runtime, "PUBLIC_INSTALLED_ARTIFACT_SHA256", self.artifact_sha):
            artifact, identity = runtime._verify_installed(public_root)
            self.assertEqual(artifact, self.artifact)
            self.assertEqual(identity["artifact_sha256"], self.artifact_sha)
            # The original private profile cannot point at the new folder.
            self.pin["date_range"]["folder"] = runtime.INSTALLED_FOLDER
            self.persist()
            self.reject()

    def test_unreviewed_public_folder_is_not_authorized_by_matching_hashes(self):
        self.pin["date_range"]["folder"] = "date-range-buffer-unreviewed"
        self.persist()
        self.reject()

    def test_unsupported_hosts_never_access_files(self):
        for key, value in (("platform", "darwin"), ("python_abi", "cpython-313"),
                           ("build_platform", "win-arm64"), ("implementation", "pypy")):
            with self.subTest(key=key), patch.object(runtime, "_host", return_value={**SUPPORTED, key: value}), \
                    patch.object(runtime, "_read", side_effect=AssertionError("no file read")), \
                    patch.object(runtime, "_checked", side_effect=AssertionError("no filesystem")):
                function, identity = runtime.activate_installed_range("nonexistent")
                self.assertIsNone(function)
                self.assertEqual(identity["fallback"], "unsupported_platform_or_abi")

    def test_tampered_source_artifact_provenance_and_missing_pin(self):
        for path in (self.root / runtime.BUFFER_SOURCES[0], self.artifact,
                     self.installation / "provenance.json"):
            with self.subTest(path=path):
                before = path.read_bytes()
                path.write_bytes(before + b"tamper")
                self.reject()
                path.write_bytes(before)
        self.pin_path.unlink()
        self.reject()

    def test_profile_cannot_authorize_other_sources_or_artifacts(self):
        original = copy.deepcopy(self.pin)
        for field, value in (("folder", "other"), ("artifact", "../other.pyd"),
                             ("artifact_sha256", "0" * 64), ("source_sha256", {}),
                             ("provenance_sha256", "bad")):
            self.pin = copy.deepcopy(original)
            self.pin["date_range"][field] = value
            self.pin_path.write_text(json.dumps(self.pin), encoding="utf-8")
            self.reject()
        self.pin = original
        for field, value in (("schema_version", True), ("name", "candidate"), ("kind", "caller_pin")):
            altered = {**original, field: value}
            self.pin_path.write_text(json.dumps(altered), encoding="utf-8")
            self.reject()

    def test_unowned_preload_rejected_even_if_path_matches(self):
        fake = self.fake_loader()
        sys.modules[runtime.MODULE] = fake
        self.reject()
        self.assertIs(sys.modules[runtime.MODULE], fake)

    def test_reparse_paths_and_parent_traversal_rejected(self):
        original = Path.lstat
        for redirected in (self.pin_path.parent, self.installation, self.artifact,
                           self.root / "qlib_yahoo"):
            def details(path):
                result = original(path)
                if path == redirected:
                    return types.SimpleNamespace(st_mode=result.st_mode, st_file_attributes=0x400)
                return result
            with self.subTest(path=redirected), patch.object(Path, "lstat", details):
                self.reject()
        self.reject(self.installation / ".." / self.installation.name)
        self.reject(Path("relative") / runtime.INSTALLED_FOLDER)
        wrong_folder = self.root / runtime.INSTALLED_FOLDER
        wrong_folder.mkdir()
        self.reject(wrong_folder)

    def test_failed_native_initialization_restores_only_own_entry(self):
        dependency = types.ModuleType("installed_range_side_effect")
        def fail(spec):
            sys.modules[runtime.MODULE] = self.fake_loader()
            sys.modules[dependency.__name__] = dependency
            raise ImportError("mock initialization failure")
        with patch.object(runtime.importlib.machinery, "ExtensionFileLoader"), \
                patch.object(runtime.importlib.util, "spec_from_file_location", return_value=object()), \
                patch.object(runtime.importlib.util, "module_from_spec", side_effect=fail):
            with self.assertRaisesRegex(DatasetError, "initialization failure"):
                runtime.activate_installed_range(self.installation)
        self.assertNotIn(runtime.MODULE, sys.modules)
        self.assertNotIn(runtime.MODULE, runtime._OWNED)
        self.assertIs(sys.modules[dependency.__name__], dependency)

    def build_receipt(self):
        workspace = self.root / "prototype"
        artifact = workspace / "artifacts/qlib_yahoo" / self.artifact.name
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(self.artifact_raw)
        self.receipt = {"schema_version": 1, "kind": "private_sequential_cython_build",
                        "status": "completed", "stage": "completed", "source_sha256": self.sources,
                        "cython_version": "3.1.8", "wheel": {"sha256": runtime.WHEEL_SHA256},
                        "directives": runtime.DIRECTIVES, "compiler_flags": ["/MD", "/fp:strict"],
                        "macros": [["CYTHON_USE_DICT_VERSIONS", "0"]], "numpy_c_api_used": False,
                        "public_api_only_certified": False, "allow_incomplete_toolchain": True,
                        "private_import_tokens": ["_Py_Dealloc"], "workspace": str(workspace),
                        "artifact": {"path": str(artifact), "sha256": self.artifact_sha},
                        "builder": {"installation_healthy": False, "platform": "win-amd64",
                                    "extension_suffix": runtime.SUFFIX, "compiler_banner": "MSVC Compiler 19.44.35222 for x64",
                                    "msvc_tools_version": "14.44.35207", "windows_sdk_version": "10.0.26100.0"},
                        "build_stdout": "DO NOT EXPORT: personal strategy and environment",
                        "saved_run": "DO NOT EXPORT", "compiler_environment": {"DO NOT EXPORT": "private"}}
        self.manifest = self.root / "original-receipt.json"
        self.manifest.write_text(json.dumps(self.receipt), encoding="utf-8")
        pin = {"schema_version": 1, "kind": "candidate_date_range_build_pin",
               "cython_version": "3.1.8", "source_sha256": self.sources,
               "builds": {"win32-amd64": {"python_abi": "cp312", "extension_suffix": runtime.SUFFIX,
                                          "artifact_sha256": self.artifact_sha,
                                          "manifest_sha256": runtime._sha(self.manifest.read_bytes())}}}
        self.buffer_pin.write_text(json.dumps(pin), encoding="utf-8")

    def test_package_is_deterministic_sanitized_and_loadable(self):
        self.build_receipt()
        first = self.root / "first.zip"
        second = self.root / "second.zip"
        a = packager.package(self.manifest, first)
        b = packager.package(self.manifest, second)
        self.assertEqual(a, b)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        with zipfile.ZipFile(first) as archive:
            self.assertEqual(archive.namelist(), ["provenance.json", self.artifact.name])
            provenance_raw = archive.read("provenance.json")
            self.assertNotIn(str(self.root).encode(), provenance_raw)
            self.assertNotIn(b"DO NOT EXPORT", provenance_raw)
            self.assertEqual(runtime._sha(provenance_raw), a["provenance_sha256"])
            self.assertTrue(all(entry.date_time == (1980, 1, 1, 0, 0, 0) for entry in archive.infolist()))
        (self.installation / "provenance.json").write_bytes(provenance_raw)
        self.pin["date_range"]["provenance_sha256"] = a["provenance_sha256"]
        self.pin_path.write_text(json.dumps(self.pin), encoding="utf-8")
        fake = self.fake_loader()
        with patch.object(runtime.importlib.machinery, "ExtensionFileLoader"), \
                patch.object(runtime.importlib.util, "spec_from_file_location", return_value=object()), \
                patch.object(runtime.importlib.util, "module_from_spec", return_value=fake):
            self.assertIs(runtime.activate_installed_range(self.installation)[0], fake.try_valid_range)
        with self.assertRaises(DatasetError):
            packager.package(self.manifest, first)

    def test_packager_rejects_untrusted_receipt_before_parsing_and_tamper(self):
        self.build_receipt()
        before = self.manifest.read_bytes()
        self.manifest.write_bytes(b"not JSON")
        with patch.object(runtime, "_json", wraps=runtime._json) as parser:
            with self.assertRaises(DatasetError):
                packager.package(self.manifest, self.root / "rejected.zip")
            self.assertEqual(parser.call_count, 1)  # Trusted original pin only.
        self.manifest.write_bytes(before)
        for path in (self.root / runtime.BUFFER_SOURCES[0], Path(self.receipt["artifact"]["path"])):
            raw = path.read_bytes()
            path.write_bytes(raw + b"tamper")
            with self.assertRaises(DatasetError):
                packager.package(self.manifest, self.root / "rejected.zip")
            path.write_bytes(raw)
        self.assertFalse((self.root / "rejected.zip").exists())


if __name__ == "__main__":
    unittest.main()
