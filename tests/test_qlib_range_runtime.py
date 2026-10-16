"""Offline loader controls; fake extension loading never executes native code."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from qlib_yahoo import range_runtime as runtime
from qlib_yahoo.dataset import DatasetError


SUPPORTED = {"implementation": "cpython", "python_abi": "cpython-312",
             "python_version": [3, 12], "platform": "win32", "build_platform": "win-amd64",
             "machine": "AMD64", "extension_suffix": runtime.SUFFIX}


class RangeRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.workspace = self.data / "build-v6"
        self.artifact = self.workspace / "artifacts/qlib_yahoo" / ("session_loops" + runtime.SUFFIX)
        self.artifact.parent.mkdir(parents=True)
        self.artifact.write_bytes(b"fake extension bytes; no native import")
        self.manifest = self.data / "receipt.json"
        self.pin_path = self.root / "docs/pin.json"
        self.buffer_pin_path = self.root / "docs/buffer-pin.json"
        self.pin_path.parent.mkdir()
        self.sources = {}
        for name in runtime.SOURCES:
            target = self.root / name
            target.parent.mkdir(exist_ok=True, parents=True)
            target.write_text("reviewed source " + name, encoding="utf-8")
            self.sources[name] = hashlib.sha256(target.read_bytes()).hexdigest()
        self.receipt = {"schema_version": 1, "kind": "private_sequential_cython_build",
                        "status": "completed", "stage": "completed", "source_sha256": self.sources,
                        "cython_version": "3.1.8", "wheel": {"sha256": runtime.WHEEL_SHA256},
                        "directives": copy.deepcopy(runtime.DIRECTIVES), "compiler_flags": ["/MD", "/fp:strict"],
                        "macros": [["CYTHON_USE_DICT_VERSIONS", "0"]], "numpy_c_api_used": False,
                        "builder": {"python": "3.12.10 (reviewed builder)", "executable": sys.executable,
                                    "platform": "win-amd64", "extension_suffix": runtime.SUFFIX,
                                    "installation_healthy": False}, "allow_incomplete_toolchain": True,
                        "public_api_only_certified": False, "private_import_tokens": ["_Py_Dealloc"],
                        "api_audit_limitation": "Private symbols retained; no public-only certificate",
                        "workspace": str(self.workspace),
                        "artifact": {"path": str(self.artifact),
                                     "sha256": hashlib.sha256(self.artifact.read_bytes()).hexdigest()}}
        self.pin = {"schema_version": 1, "kind": "candidate_date_range_build_pin",
                    "cython_version": "3.1.8", "source_sha256": self.sources,
                    "builds": {"win32-amd64": {"python_abi": "cp312", "extension_suffix": runtime.SUFFIX,
                                               "artifact_sha256": self.receipt["artifact"]["sha256"]}}}
        self.persist()
        self.buffer_sources = {}
        for name in runtime.BUFFER_SOURCES:
            target = self.root / name
            if not target.exists():
                target.write_text("reviewed source " + name, encoding="utf-8")
            self.buffer_sources[name] = hashlib.sha256(target.read_bytes()).hexdigest()
        self.buffer_workspace = self.data / "build-buffer"
        self.buffer_artifact = self.buffer_workspace / "artifacts/qlib_yahoo" / ("session_loops" + runtime.SUFFIX)
        self.buffer_artifact.parent.mkdir(parents=True)
        self.buffer_artifact.write_bytes(b"fake buffer extension bytes; no native import")
        self.buffer_manifest = self.data / "buffer-receipt.json"
        self.buffer_receipt = copy.deepcopy(self.receipt)
        self.buffer_receipt.update(source_sha256=self.buffer_sources, workspace=str(self.buffer_workspace),
                                   artifact={"path": str(self.buffer_artifact),
                                             "sha256": hashlib.sha256(self.buffer_artifact.read_bytes()).hexdigest()})
        self.buffer_pin = copy.deepcopy(self.pin)
        self.buffer_pin["source_sha256"] = self.buffer_sources
        self.buffer_pin["builds"]["win32-amd64"]["artifact_sha256"] = self.buffer_receipt["artifact"]["sha256"]
        self.persist_buffer()
        for name, value in (("ROOT", self.root), ("PIN", self.pin_path),
                            ("BUFFER_PIN", self.buffer_pin_path), ("DATA_ROOT", self.data)):
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
        self.manifest.write_text(json.dumps(self.receipt), encoding="utf-8")
        self.pin["builds"]["win32-amd64"]["manifest_sha256"] = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        self.pin_path.write_text(json.dumps(self.pin), encoding="utf-8")

    def persist_buffer(self):
        self.buffer_manifest.write_text(json.dumps(self.buffer_receipt), encoding="utf-8")
        self.buffer_pin["builds"]["win32-amd64"]["manifest_sha256"] = hashlib.sha256(self.buffer_manifest.read_bytes()).hexdigest()
        self.buffer_pin_path.write_text(json.dumps(self.buffer_pin), encoding="utf-8")

    def reject_before_import(self, manifest=None):
        with patch.object(runtime.importlib.machinery, "ExtensionFileLoader") as loader:
            with self.assertRaises(DatasetError):
                runtime.activate_range(self.manifest if manifest is None else manifest)
            loader.assert_not_called()

    def test_verified_load_and_owned_reuse(self):
        function = lambda *args: (True, True)
        fake = types.ModuleType(runtime.MODULE)
        fake.try_valid_range = function
        fake.__file__ = str(self.artifact)
        with patch.object(runtime.importlib.machinery, "ExtensionFileLoader") as loader, \
                patch.object(runtime.importlib.util, "spec_from_file_location", return_value=object()), \
                patch.object(runtime.importlib.util, "module_from_spec", return_value=fake):
            loaded, identity = runtime.activate_range(self.manifest)
            self.assertIs(loaded, function)
            self.assertEqual(identity["selection"], "candidate_cython_date_range")
            self.assertEqual(identity["implementation"], "scalar_views")
            self.assertEqual(identity["trusted_pin_path"], str(self.pin_path))
            self.assertEqual(identity["source_sha256"], self.sources)
            self.assertEqual(identity["artifact_sha256"], self.receipt["artifact"]["sha256"])
            self.assertTrue(identity["gil_held"])
            self.assertFalse(identity["public_api_only_certified"])
            reused, _ = runtime.activate_range(self.manifest)
            self.assertIs(reused, function)
            loader.assert_called_once()
            fake.try_valid_range = lambda *args: (False, False)
            with self.assertRaisesRegex(DatasetError, "conflicting preloaded"):
                runtime.activate_range(self.manifest)
            fake.try_valid_range = function
            fake.__file__ = str(self.artifact.parent / "other.pyd")
            with self.assertRaisesRegex(DatasetError, "conflicting preloaded"):
                runtime.activate_range(self.manifest)

    def test_unsupported_abi_falls_back_without_file_access(self):
        for name, value in (("implementation", "pypy"), ("python_abi", "cpython-313"),
                            ("python_version", [3, 13]), ("platform", "darwin"),
                            ("build_platform", "win-arm64"), ("machine", "ARM64"),
                            ("extension_suffix", ".wrong.pyd")):
            with self.subTest(name=name), patch.object(runtime, "_host", return_value={**SUPPORTED, name: value}), \
                    patch.object(runtime, "_read", side_effect=AssertionError("no read")), \
                    patch.object(runtime, "_checked", side_effect=AssertionError("no file access")), \
                    patch.object(runtime, "_json", side_effect=AssertionError("no parsing")), \
                    patch.object(runtime.importlib.machinery, "ExtensionFileLoader") as loader:
                function, identity = runtime.activate_range("missing")
                self.assertIsNone(function)
                self.assertEqual(identity["fallback"], "unsupported_platform_or_abi")
                loader.assert_not_called()

    def test_missing_and_tampered_manifest(self):
        self.manifest.unlink()
        self.reject_before_import()
        self.persist()
        self.manifest.write_bytes(b"tampered, invalid JSON")
        with patch.object(runtime, "_json", wraps=runtime._json) as parser:
            self.reject_before_import()
            self.assertEqual(parser.call_count, 2)  # Both trusted pins only.

    def test_missing_pin(self):
        self.pin_path.unlink()
        self.reject_before_import()

    def test_tampered_source_and_artifact(self):
        for target in (self.root / runtime.SOURCES[0], self.artifact):
            with self.subTest(target=target):
                original = target.read_bytes()
                target.write_bytes(original + b"tampered")
                self.reject_before_import()
                target.write_bytes(original)

    def test_invalid_reviewed_receipts(self):
        original = copy.deepcopy(self.receipt)
        mutations = [("compiler_flags", ["/MD", "/fp:fast"]), ("numpy_c_api_used", True),
                     ("macros", []), ("allow_incomplete_toolchain", False),
                     ("status", "running"), ("public_api_only_certified", True),
                     ("source_sha256", {}), ("wheel", {"sha256": "0" * 64}),
                     ("cython_version", "3.2.0"), ("schema_version", True)]
        for name, value in mutations:
            with self.subTest(name=name):
                self.receipt = copy.deepcopy(original)
                self.receipt[name] = value
                self.persist()
                self.reject_before_import()
        for name, value in (("infer_types", True), ("boundscheck", False), ("wraparound", False),
                            ("annotation_typing", True), ("cdivision", True), ("binding", 1)):
            with self.subTest(directive=name):
                self.receipt = copy.deepcopy(original)
                self.receipt["directives"][name] = value
                self.persist()
                self.reject_before_import()

    def test_wrong_pin_abi_digest_and_source_map(self):
        for name, value in (("python_abi", "cp313"), ("artifact_sha256", "0" * 64),
                            ("manifest_sha256", "not-a-digest")):
            with self.subTest(name=name):
                self.persist()
                self.pin["builds"]["win32-amd64"][name] = value
                self.pin_path.write_text(json.dumps(self.pin), encoding="utf-8")
                self.reject_before_import()
                self.pin["builds"]["win32-amd64"][name] = (
                    "cp312" if name == "python_abi" else self.receipt["artifact"]["sha256"])
        self.pin["source_sha256"] = {}
        self.persist()
        self.reject_before_import()

    def test_unowned_preloaded_module_rejected_even_with_matching_path(self):
        fake = types.ModuleType(runtime.MODULE)
        fake.__file__ = str(self.artifact)
        fake.try_valid_range = lambda *args: True
        sys.modules[runtime.MODULE] = fake
        self.reject_before_import()
        self.assertIs(sys.modules[runtime.MODULE], fake)

    def test_failed_native_initialization_preserves_unrelated_import_changes(self):
        new_dependency = types.ModuleType("range_test_import_side_effect")
        replacement = types.ModuleType("range_test_replaced_dependency")
        sys.modules["range_test_replaced_dependency"] = object()
        def fail(spec):
            sys.modules[runtime.MODULE] = types.ModuleType(runtime.MODULE)
            sys.modules["range_test_import_side_effect"] = new_dependency
            sys.modules["range_test_replaced_dependency"] = replacement
            raise ImportError("mock native initialization failure")
        with patch.object(runtime.importlib.machinery, "ExtensionFileLoader"), \
                patch.object(runtime.importlib.util, "spec_from_file_location", return_value=object()), \
                patch.object(runtime.importlib.util, "module_from_spec", side_effect=fail):
            with self.assertRaisesRegex(DatasetError, "initialization failure"):
                runtime.activate_range(self.manifest)
        self.assertNotIn(runtime.MODULE, sys.modules)
        self.assertIs(sys.modules["range_test_import_side_effect"], new_dependency)
        self.assertIs(sys.modules["range_test_replaced_dependency"], replacement)

    def test_reparse_source_parent_rejected(self):
        original = Path.lstat
        source_parent = self.root / "qlib_yahoo"
        def details(path):
            result = original(path)
            if path == source_parent:
                return types.SimpleNamespace(st_mode=result.st_mode, st_file_attributes=0x400)
            return result
        with patch.object(Path, "lstat", details):
            self.reject_before_import()

    def test_failed_exec_and_missing_callable_restore_modules(self):
        for failure in (RuntimeError("mock exec failure"), None):
            with self.subTest(failure=failure):
                fake = types.ModuleType(runtime.MODULE)
                before = dict(sys.modules)
                with patch.object(runtime.importlib.machinery, "ExtensionFileLoader") as factory, \
                        patch.object(runtime.importlib.util, "spec_from_file_location", return_value=object()), \
                        patch.object(runtime.importlib.util, "module_from_spec", return_value=fake):
                    factory.return_value.exec_module.side_effect = failure
                    with self.assertRaises(DatasetError):
                        runtime.activate_range(self.manifest)
                self.assertEqual(set(sys.modules), set(before))
                self.assertTrue(all(sys.modules[name] is value for name, value in before.items()))
                self.assertNotIn(runtime.MODULE, runtime._OWNED)

    def test_builder_identity_and_reparse_manifest_artifact_rejected(self):
        original_receipt = copy.deepcopy(self.receipt)
        for name, value in (("platform", "win-arm64"), ("python", "3.13.1"),
                            ("extension_suffix", ".cp313-win_amd64.pyd"), ("executable", "python.exe")):
            with self.subTest(builder=name):
                self.receipt = copy.deepcopy(original_receipt)
                self.receipt["builder"][name] = value
                self.persist()
                self.reject_before_import()
        self.receipt = original_receipt
        self.persist()
        original = Path.lstat
        for redirected in (self.manifest.parent, self.artifact.parent, self.pin_path.parent):
            def details(path):
                result = original(path)
                if path == redirected:
                    return types.SimpleNamespace(st_mode=result.st_mode, st_file_attributes=0x400)
                return result
            with self.subTest(redirected=redirected), patch.object(Path, "lstat", details):
                self.reject_before_import()

    def test_receipt_and_artifact_must_stay_inside_recorded_roots(self):
        self.receipt["artifact"]["path"] = str(self.root / runtime.SOURCES[0])
        self.persist()
        self.reject_before_import()
        self.receipt["artifact"]["path"] = str(self.artifact)
        self.receipt["workspace"] = str(self.root)
        self.persist()
        self.reject_before_import()

    def test_normal_and_no_site_import_do_not_activate_extension(self):
        program = ("import sys; from qlib_yahoo import range_runtime; "
                   "assert 'qlib_yahoo.session_loops' not in sys.modules; "
                   "assert not range_runtime._OWNED")
        for options in ([], ["-S"]):
            with self.subTest(options=options):
                result = subprocess.run([sys.executable, *options, "-B", "-c", program],
                                        cwd=Path(__file__).absolute().parents[1], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_buffer_verified_load_and_owned_reuse(self):
        function = lambda *args: True
        fake = types.ModuleType(runtime.MODULE)
        fake.try_valid_range = function
        fake.__file__ = str(self.buffer_artifact)
        with patch.object(runtime.importlib.machinery, "ExtensionFileLoader") as loader, \
                patch.object(runtime.importlib.util, "spec_from_file_location", return_value=object()), \
                patch.object(runtime.importlib.util, "module_from_spec", return_value=fake):
            loaded, identity = runtime.activate_range(self.buffer_manifest)
            self.assertIs(loaded, function)
            self.assertEqual(identity["implementation"], "public_buffer_leases")
            self.assertEqual(identity["trusted_pin_path"], str(self.buffer_pin_path))
            self.assertEqual(identity["source_sha256"], self.buffer_sources)
            self.assertEqual(identity["artifact_sha256"], self.buffer_receipt["artifact"]["sha256"])
            self.assertFalse(identity["public_api_only_certified"])
            self.assertEqual(identity["private_import_tokens"], ["_Py_Dealloc"])
            reused, _ = runtime.activate_range(self.buffer_manifest)
            self.assertIs(reused, function)
            loader.assert_called_once()

    def test_buffer_unknown_digest_never_parses_receipt(self):
        self.buffer_manifest.write_bytes(b"untrusted and invalid JSON")
        with patch.object(runtime, "_json", wraps=runtime._json) as parser:
            self.reject_before_import(self.buffer_manifest)
        self.assertEqual(parser.call_count, 2)
        self.assertEqual([call.args[0] for call in parser.call_args_list],
                         [self.pin_path.read_bytes(), self.buffer_pin_path.read_bytes()])

    def test_buffer_tampered_sources_artifact_and_missing_pin_fail_closed(self):
        for target in (self.root / runtime.BUFFER_SOURCES[0],
                       self.root / runtime.BUFFER_SOURCES[1], self.buffer_artifact):
            with self.subTest(target=target):
                original = target.read_bytes()
                target.write_bytes(original + b"tampered")
                self.reject_before_import(self.buffer_manifest)
                target.write_bytes(original)
        self.buffer_pin_path.unlink()
        self.reject_before_import(self.buffer_manifest)

    def test_buffer_policy_guards_and_exact_source_inventory(self):
        original = copy.deepcopy(self.buffer_receipt)
        for field, value in (("compiler_flags", ["/MD", "/fp:fast"]),
                             ("allow_incomplete_toolchain", False),
                             ("public_api_only_certified", True),
                             ("numpy_c_api_used", True), ("macros", []),
                             ("wheel", {"sha256": "0" * 64})):
            with self.subTest(field=field):
                self.buffer_receipt = copy.deepcopy(original)
                self.buffer_receipt[field] = value
                self.persist_buffer()
                self.reject_before_import(self.buffer_manifest)
        self.buffer_receipt = original
        self.buffer_receipt["directives"]["boundscheck"] = False
        self.persist_buffer()
        self.reject_before_import(self.buffer_manifest)
        self.buffer_pin["source_sha256"] = {**self.buffer_sources, "unreviewed.pyx": "0" * 64}
        self.persist_buffer()
        self.reject_before_import(self.buffer_manifest)

    def test_cross_variant_source_receipts_fail_before_import(self):
        # Even when a trusted pin names the whole receipt digest, its variant's
        # exact source inventory still determines which implementation it owns.
        self.receipt["source_sha256"] = self.buffer_sources
        self.persist()
        self.reject_before_import()
        self.buffer_receipt["source_sha256"] = self.sources
        self.persist_buffer()
        self.reject_before_import(self.buffer_manifest)

    def test_ambiguous_digest_fails_before_receipt_parsing(self):
        self.buffer_pin["builds"]["win32-amd64"]["manifest_sha256"] = self.pin["builds"]["win32-amd64"]["manifest_sha256"]
        self.buffer_pin_path.write_text(json.dumps(self.buffer_pin), encoding="utf-8")
        with patch.object(runtime, "_json", wraps=runtime._json) as parser, \
                patch.object(runtime.importlib.machinery, "ExtensionFileLoader") as loader:
            with self.assertRaisesRegex(DatasetError, "ambiguous"):
                runtime.activate_range(self.manifest)
        self.assertEqual(parser.call_count, 2)
        loader.assert_not_called()

    def test_owned_variant_switch_conflicts_without_module_eviction(self):
        for first_manifest, first_artifact, second_manifest in (
                (self.manifest, self.artifact, self.buffer_manifest),
                (self.buffer_manifest, self.buffer_artifact, self.manifest)):
            with self.subTest(first=first_manifest):
                sys.modules.pop(runtime.MODULE, None)
                runtime._OWNED.clear()
                fake = types.ModuleType(runtime.MODULE)
                fake.try_valid_range = lambda *args: True
                fake.__file__ = str(first_artifact)
                with patch.object(runtime.importlib.machinery, "ExtensionFileLoader") as loader, \
                        patch.object(runtime.importlib.util, "spec_from_file_location", return_value=object()), \
                        patch.object(runtime.importlib.util, "module_from_spec", return_value=fake):
                    runtime.activate_range(first_manifest)
                    owned = runtime._OWNED[runtime.MODULE]
                    with self.assertRaisesRegex(DatasetError, "conflicting preloaded"):
                        runtime.activate_range(second_manifest)
                    self.assertIs(sys.modules[runtime.MODULE], fake)
                    self.assertIs(runtime._OWNED[runtime.MODULE], owned)
                    loader.assert_called_once()


if __name__ == "__main__":
    unittest.main()
