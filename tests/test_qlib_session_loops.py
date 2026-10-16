"""Unwired extension differential tests and explicit complete-decision diagnostic.

No build or installation is triggered by tests. Supply a reviewed successful
build manifest through QLIB_SESSION_LOOPS_MANIFEST to enable an isolated signed
child. Optional QLIB_SESSION_LOOPS_MEASURE_RECEIPT enables a new synthetic 3000
stock paired receipt, NOT worker/acceptance evidence. Do not run while another
full-worker timing measurement is active.
"""
import contextlib
from fractions import Fraction
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PureWindowsPath
import statistics
import subprocess
import sys
import tempfile
import time
from types import MethodType
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("private_session_builder", ROOT / "scripts/build_qlib_session_loops.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)
MANIFEST_ENV = "QLIB_SESSION_LOOPS_MANIFEST"
CHILD_ENV = "QLIB_SESSION_LOOPS_CHILD"


class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        metadata = mock.patch.object(builder.importlib.metadata, "version", return_value="mock-version")
        metadata.start()
        self.addCleanup(metadata.stop)

    def wheel(self, *, version="3.1.8", name="Cython", member="Cython/__init__.py"):
        wheel = self.root / "cython-3.1.8-cp312-cp312-win_amd64.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("cython-3.1.8.dist-info/METADATA", f"Name: {name}\nVersion: {version}\n")
            archive.writestr(member, "# synthetic wheel; never executed\n")
        return wheel, builder.sha256(wheel)

    def test_reviewed_wheel_identity_and_digest(self):
        wheel, digest = self.wheel()
        self.assertEqual(builder.verify_wheel(wheel, digest), wheel)
        for bad in ("bad", "0" * 64):
            with self.assertRaises(ValueError):
                builder.verify_wheel(wheel, bad)

    def test_wrong_distribution_version_or_traversal_is_rejected(self):
        for kwargs in ({"version": "3.3.0"}, {"name": "Other"}, {"member": "../escape.py"}):
            with self.subTest(kwargs=kwargs):
                wheel, digest = self.wheel(**kwargs)
                with self.assertRaises(ValueError):
                    builder.verify_wheel(wheel, digest)

    def test_wrong_binary_tag_is_rejected(self):
        wheel, digest = self.wheel()
        renamed = self.root / "cython-3.1.8-cp311-cp311-win_amd64.whl"
        wheel.rename(renamed)
        with self.assertRaises(ValueError):
            builder.verify_wheel(renamed, digest)

    def test_non_windows_preflight_is_fail_closed(self):
        with mock.patch.object(builder.sys, "platform", "darwin"), mock.patch.object(builder.subprocess, "run") as run:
            with self.assertRaises(RuntimeError):
                builder.compiler_preflight()
        run.assert_not_called()

    def test_compiler_gate_failure_is_saved_before_wheel_or_build(self):
        manifest = self.root / "manifest.json"
        workspace = self.root / "workspace"
        original_checked = builder.checked_path
        def checked(path, **kwargs):
            return self.root if str(path) == r"D:\Qlib" else original_checked(path, **kwargs)
        with mock.patch.object(builder, "checked_path", side_effect=checked), \
                mock.patch.object(builder, "compiler_preflight", side_effect=RuntimeError("no compiler")), \
                mock.patch.object(builder, "verify_wheel") as wheel, \
                mock.patch.object(builder.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "no compiler"):
                builder.build(workspace, manifest, self.root / "unused.whl", "0" * 64)
        receipt = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(receipt["stage"], "compiler_preflight")
        self.assertEqual(receipt["status"], "failed")
        self.assertFalse(receipt["public_api_only_certified"])
        self.assertFalse(workspace.exists())
        wheel.assert_not_called(); run.assert_not_called()

    def test_no_clobber_or_outside_data_drive(self):
        manifest = self.root / "manifest.json"
        manifest.write_text("preserved", encoding="utf-8")
        original_checked = builder.checked_path
        def checked(path, **kwargs):
            return self.root if str(path) == r"D:\Qlib" else original_checked(path, **kwargs)
        with mock.patch.object(builder, "checked_path", side_effect=checked), \
                mock.patch.object(builder, "compiler_preflight") as gate:
            with self.assertRaises(ValueError):
                builder.build(self.root / "workspace", manifest, "unused", "0" * 64)
            with self.assertRaises(ValueError):
                builder.build(self.root.parent / "outside", self.root / "new.json", "unused", "0" * 64)
        self.assertEqual(manifest.read_text(encoding="utf-8"), "preserved")
        gate.assert_not_called()

    def test_source_and_builder_declared_controls(self):
        source = (ROOT / "qlib_yahoo/session_loops.pyx").read_text(encoding="utf-8")
        self.assertIn(builder.REVIEWED_SHA256, source)
        self.assertEqual(builder.sha256(builder.REVIEWED), builder.REVIEWED_SHA256)
        self.assertFalse(builder.DIRECTIVES["infer_types"])
        self.assertFalse(builder.DIRECTIVES["annotation_typing"])
        self.assertEqual(builder.MACROS, [("CYTHON_USE_DICT_VERSIONS", "0")])
        self.assertEqual(builder.FLAGS, ["/MD", "/fp:strict"])
        self.assertNotIn("nogil", source.replace("no C floor/division/arithmetic", ""))
        self.assertNotIn("target_session import", source)

    def mocked_data_paths(self):
        original = builder.checked_path
        return mock.patch.object(builder, "checked_path", side_effect=lambda p, **kw:
                                 self.root if str(p) == r"D:\Qlib" else original(p, **kw))

    def test_partial_discovery_retains_failed_default_query_and_raw_packages(self):
        partial = {"installationPath": "installed", "state": 13, "isComplete": False,
                   "isLaunchable": False, "properties": {"canceled": "1"},
                   "packages": [{"id": "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"}]}
        with mock.patch.object(builder.subprocess, "run", side_effect=[
                subprocess.CompletedProcess([], 0, stdout="", stderr=""),
                subprocess.CompletedProcess([], 0, stdout=json.dumps([partial]), stderr="")]) as run:
            result = builder.discover_tools("vswhere")
        self.assertEqual(result["selected_instance"], partial)
        self.assertEqual(result["default_query_stdout"], "")
        self.assertIn("incomplete", result["warning"])
        self.assertIn("-all", run.call_args_list[1].args[0])
        self.assertIn("packages", run.call_args_list[1].args[0])

    def test_discovery_prefers_healthy_instance_over_newer_partial(self):
        partial = {"installationPath": "new-partial", "isComplete": False, "isLaunchable": False}
        healthy = {"installationPath": "old-healthy", "isComplete": True, "isLaunchable": True}
        with mock.patch.object(builder.subprocess, "run", side_effect=[
                subprocess.CompletedProcess([], 0, stdout="old-healthy", stderr=""),
                subprocess.CompletedProcess([], 0, stdout=json.dumps([partial, healthy]), stderr="")]):
            result = builder.discover_tools("vswhere")
        self.assertEqual(result["selected_instance"], healthy)
        self.assertIsNone(result["warning"])

    def test_empty_discovery_is_fail_closed(self):
        with mock.patch.object(builder.subprocess, "run", side_effect=[
                subprocess.CompletedProcess([], 0, stdout="", stderr=""),
                subprocess.CompletedProcess([], 0, stdout="[]", stderr="")]):
            with self.assertRaisesRegex(RuntimeError, "not installed"):
                builder.discover_tools("vswhere")

    def test_partial_installation_default_fails_before_environment_or_wheel(self):
        identity = {"installation_healthy": False, "discovery": {"selected_instance": {"state": 13}}}
        manifest = self.root / "failed.json"
        with self.mocked_data_paths(), mock.patch.object(builder, "compiler_preflight", return_value=identity), \
                mock.patch.object(builder, "build_environment") as environment, \
                mock.patch.object(builder, "verify_wheel") as wheel:
            with self.assertRaisesRegex(RuntimeError, "Incomplete Visual Studio"):
                builder.build(self.root / "workspace", manifest, "unused", "0" * 64)
        receipt = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertFalse(receipt["allow_incomplete_toolchain"])
        self.assertEqual(receipt["builder"], identity)
        self.assertEqual(receipt["stage"], "installation_state_acknowledgement")
        environment.assert_not_called(); wheel.assert_not_called()

    def test_partial_acknowledgement_is_recorded_and_does_not_mark_it_healthy(self):
        identity = {"installation_healthy": False, "discovery": {"selected_instance": {"state": 13}}}
        manifest = self.root / "failed.json"
        with self.mocked_data_paths(), mock.patch.object(builder, "compiler_preflight", return_value=identity), \
                mock.patch.object(builder, "build_environment", return_value={"VCTOOLSVERSION": "14.51"}) as environment, \
                mock.patch.object(builder, "verify_wheel", side_effect=ValueError("unreviewed wheel")) as wheel:
            with self.assertRaisesRegex(ValueError, "unreviewed wheel"):
                builder.build(self.root / "workspace", manifest, "unused", "0" * 64,
                              allow_incomplete_toolchain=True)
        receipt = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertTrue(receipt["allow_incomplete_toolchain"])
        self.assertFalse(receipt["builder"]["installation_healthy"])
        self.assertEqual(receipt["stage"], "wheel_verification")
        self.assertFalse((self.root / "workspace").exists())
        environment.assert_called_once_with(identity); wheel.assert_called_once()

    def test_acknowledgement_type_and_explicit_cli_forwarding(self):
        with self.assertRaises(TypeError):
            builder.build("unused", "unused", "unused", "unused", allow_incomplete_toolchain="yes")
        arguments = ["builder", "--allow-build", "--workspace", "workspace", "--manifest", "manifest",
                     "--cython-wheel", "wheel", "--cython-wheel-sha256", "0" * 64]
        for flag in ([], ["--allow-incomplete-toolchain"]):
            with self.subTest(flag=flag), mock.patch.object(builder.sys, "argv", arguments + flag), \
                    mock.patch.object(builder, "build", return_value={}) as build, \
                    contextlib.redirect_stdout(io.StringIO()):
                builder.main()
            self.assertIs(build.call_args.kwargs["allow_incomplete_toolchain"], bool(flag))

    def test_failed_compile_retains_output_sources_and_syncs_no_clobber(self):
        wheel, digest = self.wheel()
        manifest = self.root / "manifest.json"
        workspace = self.root / "workspace"
        identity = {"installation_healthy": True, "extension_suffix": ".cp312-win_amd64.pyd"}
        with self.mocked_data_paths(), mock.patch.object(builder, "compiler_preflight", return_value=identity), \
                mock.patch.object(builder, "build_environment", return_value={}), \
                mock.patch.object(builder.subprocess, "run", return_value=subprocess.CompletedProcess(
                    [], 1, stdout=b"retained compiler stdout\x90", stderr=b"retained compiler error")), \
                mock.patch.object(builder.os, "fsync", wraps=builder.os.fsync) as sync:
            with self.assertRaisesRegex(RuntimeError, "Cython build failed"):
                builder.build(workspace, manifest, wheel, digest)
        receipt = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["stage"], "compile_and_link")
        self.assertEqual(receipt["build_stderr"], "retained compiler error")
        self.assertEqual(receipt["source_sha256"]["qlib_yahoo/fast_signed.py"], builder.REVIEWED_SHA256)
        self.assertEqual(receipt["wheel"]["sha256"], digest)
        self.assertEqual(sync.call_count, 5)
        self.assertEqual(Path(receipt["raw_build_logs"]["stdout"]["path"]).read_bytes(),
                         b"retained compiler stdout\x90")
        self.assertEqual(receipt["raw_build_logs"]["stdout"]["sha256"],
                         hashlib.sha256(b"retained compiler stdout\x90").hexdigest())
        preserved = manifest.read_bytes()
        with self.mocked_data_paths(), mock.patch.object(builder, "compiler_preflight") as gate:
            with self.assertRaises(ValueError):
                builder.build(workspace, manifest, wheel, digest)
        gate.assert_not_called()
        self.assertEqual(manifest.read_bytes(), preserved)

    def test_interrupt_after_compiler_gate_retains_partial_identity(self):
        identity = {"installation_healthy": False}
        manifest = self.root / "interrupt.json"
        with self.mocked_data_paths(), mock.patch.object(builder, "compiler_preflight", return_value=identity), \
                mock.patch.object(builder, "build_environment", side_effect=KeyboardInterrupt("cancelled")):
            with self.assertRaises(KeyboardInterrupt):
                builder.build(self.root / "workspace", manifest, "unused", "0" * 64,
                              allow_incomplete_toolchain=True)
        receipt = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(receipt["stage"], "compiler_environment")
        self.assertEqual(receipt["error_type"], "KeyboardInterrupt")
        self.assertFalse(receipt["builder"]["installation_healthy"])

    def test_mock_success_preserves_artifact_generated_c_and_private_api_audit(self):
        wheel, digest = self.wheel()
        manifest, workspace = self.root / "success.json", self.root / "workspace"
        identity = {"installation_healthy": False, "extension_suffix": ".cp312-win_amd64.pyd",
                    "dumpbin": "mock-dumpbin", "compiler": "reviewed-cl", "tool_sha256": {"reviewed-cl": "hash"}}
        def command(arguments, **kwargs):
            if arguments[0] == sys.executable:
                artifact = workspace / "artifacts/qlib_yahoo/session_loops.cp312-win_amd64.pyd"
                artifact.parent.mkdir(parents=True)
                artifact.write_bytes(b"MOCK PE; not a loadable extension")
                generated = workspace / "generated/session_loops.c"
                generated.parent.mkdir()
                generated.write_text("/* mock generated C */", encoding="utf-8")
                return subprocess.CompletedProcess(arguments, 0, stdout="mock compile", stderr="")
            text = {"/imports": "python312.dll _PyPrivateSymbol", "/exports": "PyInit_session_loops", "/headers": "x64"}
            return subprocess.CompletedProcess(arguments, 0, stdout=text[arguments[1]], stderr="")
        with self.mocked_data_paths(), mock.patch.object(builder, "compiler_preflight", return_value=identity), \
                mock.patch.object(builder, "build_environment", return_value={"UNRELATED_SECRET": "example_must_not_persist"}), \
                mock.patch.object(builder.subprocess, "run", side_effect=command):
            result = builder.build(workspace, manifest, wheel, digest, allow_incomplete_toolchain=True)
        receipt = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(result, receipt)
        self.assertEqual(receipt["status"], "completed")
        self.assertFalse(receipt["public_api_only_certified"])
        self.assertFalse(receipt["builder"]["installation_healthy"])
        self.assertTrue(receipt["allow_incomplete_toolchain"])
        self.assertEqual(receipt["private_import_tokens"], ["_PyPrivateSymbol"])
        self.assertEqual(receipt["artifact"]["sha256"], builder.sha256(receipt["artifact"]["path"]))
        self.assertEqual(receipt["generated_c_sha256"]["generated/session_loops.c"],
                         builder.sha256(workspace / "generated/session_loops.c"))
        self.assertNotIn("example_must_not_persist", manifest.read_text(encoding="utf-8"))

    def test_child_environment_requires_verified_compiler_and_sdk_versions(self):
        script = self.root / "vcvars64.bat"
        script.write_text("@echo off\n", encoding="utf-8")
        identity = {"vcvars64": str(script), "vcvars64_sha256": builder.sha256(script),
                    "compiler": str(self.root / "bin/cl.exe"), "msvc_tools_version": "14.51.36231",
                    "windows_sdk_version": "10.0.26100.0"}
        def response(version):
            return f"Path={self.root / 'bin'}\nVCToolsVersion={version}\nWindowsSDKVersion=10.0.26100.0\\\nINCLUDE=verified\nLIB=verified\n"
        command_processor = r"C:\Windows\System32\cmd.exe"
        original_checked_path = builder.checked_path
        def checked_path(path, **kwargs):
            if path == command_processor:
                self.assertEqual(kwargs, {"kind": "file"})
                return PureWindowsPath(command_processor)
            return original_checked_path(path, **kwargs)
        with mock.patch.object(builder, "checked_path", side_effect=checked_path) as checked, \
                mock.patch.object(builder.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 0, stdout=response("14.51.36231"), stderr="")) as initialized:
            environment = builder.build_environment(identity)
        checked.assert_called_once_with(command_processor, kind="file")
        self.assertEqual(environment["DISTUTILS_USE_SDK"], "1")
        self.assertEqual(environment["MSSDK"], "1")
        self.assertEqual(environment["PATH"], str(self.root / "bin"))
        self.assertNotIn("Path", environment)
        self.assertIsInstance(initialized.call_args.args[0], str)
        self.assertNotIn('\\"', initialized.call_args.args[0])
        self.assertTrue(all(k == k.upper() for k in initialized.call_args.kwargs["env"]))
        self.assertTrue(initialized.call_args.args[0].startswith(f'"{PureWindowsPath(command_processor)}" /d /s /c '))
        with mock.patch.object(builder, "checked_path", side_effect=checked_path), \
                mock.patch.object(builder.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 0, stdout=response("14.44.0"), stderr="")):
            with self.assertRaisesRegex(RuntimeError, "differs from preflight"):
                builder.build_environment(identity)
        with mock.patch.object(builder, "checked_path", side_effect=checked_path), \
                mock.patch.object(builder.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 0, stdout=response("14.51.36231").replace("10.0.26100.0", "10.0.22621.0"), stderr="")):
            with self.assertRaisesRegex(RuntimeError, "differs from preflight"):
                builder.build_environment(identity)
        script.write_text("@echo changed\n", encoding="utf-8")
        with mock.patch.object(builder, "checked_path", side_effect=checked_path) as checked, \
                mock.patch.object(builder.subprocess, "run") as initialized:
            with self.assertRaisesRegex(RuntimeError, "changed compiler environment script"):
                builder.build_environment(identity)
        checked.assert_not_called()
        initialized.assert_not_called()

    def test_exclusive_manifest_reservation_race_does_not_overwrite(self):
        manifest = self.root / "race.json"
        original_open = Path.open
        def race(path, *args, **kwargs):
            if path == manifest and args and args[0] == "x":
                with original_open(path, "w", encoding="utf-8") as competing:
                    competing.write("other contributor")
            return original_open(path, *args, **kwargs)
        with self.mocked_data_paths(), mock.patch.object(Path, "open", race), \
                mock.patch.object(builder, "compiler_preflight") as gate:
            with self.assertRaises(FileExistsError):
                builder.build(self.root / "workspace", manifest, "unused", "0" * 64)
        self.assertEqual(manifest.read_text(encoding="utf-8"), "other contributor")
        gate.assert_not_called()


def load_reviewed_extension():
    manifest = json.loads(Path(os.environ[MANIFEST_ENV]).read_text(encoding="utf-8"))
    if manifest["status"] != "completed" or manifest["kind"] != "private_sequential_cython_build":
        raise RuntimeError("Successful private build receipt required")
    if sys.version_info[:2] != (3, 12) or sys.platform != "win32":
        raise RuntimeError("Extension ABI unavailable; Python fallback required")
    for relative, digest in manifest["source_sha256"].items():
        if builder.sha256(ROOT / relative) != digest:
            raise RuntimeError("Reviewed source changed after extension build")
    artifact = builder.checked_path(manifest["artifact"]["path"], kind="file")
    if builder.sha256(artifact) != manifest["artifact"]["sha256"]:
        raise RuntimeError("Extension artifact hash mismatch")
    started = time.perf_counter()
    specification = importlib.util.spec_from_file_location("qlib_yahoo.session_loops", artifact)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module, time.perf_counter() - started, manifest


class LauncherTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get(MANIFEST_ENV), "private extension build not requested/available")
    @unittest.skipIf(os.environ.get(CHILD_ENV) == "1", "isolated child")
    def test_verified_native_child(self):
        environment = os.environ.copy()
        environment[CHILD_ENV] = "1"
        program = ("from qlib_yahoo.runtime import ensure_signed_runtime; ensure_signed_runtime(); "
                   "import unittest; suite=unittest.defaultTestLoader.loadTestsFromName("
                   "'tests.test_qlib_session_loops.NativeChild'); "
                   "r=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(not r.wasSuccessful())")
        result = subprocess.run([sys.executable, "-c", program], cwd=ROOT, env=environment,
                                capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(os.environ.get(CHILD_ENV) == "1", "requires explicit verified extension child")
class NativeChild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_qlib_fast_signed import SignedChild
        SignedChild.setUpClass()
        cls.fixture = SignedChild()
        cls.extension, cls.load_seconds, cls.manifest = load_reviewed_extension()

    def exact(self, actual, expected):
        np = self.fixture.np
        if isinstance(expected, dict):
            self.assertIs(type(actual), type(expected)); self.assertEqual(list(actual), list(expected))
            for key in expected:
                self.exact(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            self.assertIs(type(actual), type(expected)); self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected): self.exact(a, b)
        elif isinstance(expected, (float, np.floating)):
            self.assertIs(type(actual), type(expected))
            self.assertEqual(np.float64(actual).view(np.uint64), np.float64(expected).view(np.uint64))
        else:
            self.assertEqual(actual, expected)

    def pair(self, weights, *, configure=None, **kwargs):
        results = []
        for compiled in (False, True):
            strategy, position = self.fixture.strategy(True, weights, **kwargs)
            if configure: configure(strategy, position)
            if compiled:
                strategy.generate_trade_decision = MethodType(self.extension.generate_trade_decision, strategy)
            results.append((self.fixture.outcome(strategy), position.position.copy(),
                            position.restricted_proceeds.copy()))
        self.exact(results[1], results[0])
        return results[0]

    def test_whole_decisions_reversals_closes_and_float_bits(self):
        f = self.fixture
        for quote in (f.NumpyQuote, f.DailyQuote):
            for price in (0.1, 3.0, f.np.nextafter(10., 0), f.np.nextafter(10., f.np.inf)):
                frame = f.frame({"AAA": price, "BBB": 20., "CCC": 30.})
                def configure(strategy, position):
                    strategy.trade_exchange.quote = quote(frame, "day", region="us")
                    strategy.trade_exchange.reset_session_cache()
                self.pair(f.pd.Series({"AAA": -.5, "BBB": .5}), frame=frame,
                          holdings={"AAA": 4, "BBB": -3, "CCC": 2}, configure=configure)

    def test_missing_signals_first_day_and_nonpositive_equity(self):
        f = self.fixture
        self.pair(None)
        self.pair(f.pd.Series({"AAA": .5}), step=0)
        self.pair(f.pd.Series({"AAA": .5}), cash=0.)
        self.pair(f.pd.Series({"AAA": .5, "MISSING": -.5}))

    def test_late_bad_factor_and_earlier_overflow_keep_record_prefix(self):
        f = self.fixture
        for factor in (2., f.np.nan, f.np.inf, 0.):
            outcome = self.pair(f.pd.Series({"AAA": .5, "BBB": -.5}),
                                frame=f.frame({"AAA": 10., "BBB": 20.}, {"BBB": factor}))
            self.assertIs(outcome[0][0][1], NotImplementedError)
            self.assertIsNone(outcome[0][1][0]["target_quantities"])
        with f.np.errstate(all="raise"):
            self.pair(f.pd.Series({"AAA": .5, "BBB": -.5}), cash=1e100,
                      frame=f.frame({"AAA": 1e-300, "BBB": 20.}, {"BBB": 2.}))

    def test_custom_numeric_query_and_invalid_holdings_fallback(self):
        f = self.fixture
        for price in (f.np.float32(3), Fraction(3, 7), 3):
            def configure(strategy, position):
                strategy.trade_exchange.get_deal_price = mock.Mock(return_value=price)
            self.pair(f.pd.Series({"AAA": -.4, "BBB": .5}), configure=configure)
        for holding in (.5, f.np.inf, f.np.nan, "bad"):
            def configure(strategy, position):
                position.get_stock_amount_dict = mock.Mock(return_value={"AAA": holding})
            self.pair(f.pd.Series({"AAA": .5}), configure=configure)

    def test_signal_callback_changes_binding_without_repeat_or_restart(self):
        f = self.fixture
        traces = []
        for compiled in (False, True):
            strategy, _ = f.strategy(True, f.pd.Series({"AAA": .5, "BBB": -.5}))
            trace = []
            original = strategy.signal.get_signal
            def signal(*args, **kwargs):
                trace.append("signal")
                strategy.trade_exchange.get_factor = lambda *args: trace.append(("factor", args[0])) or 1.
                return original(*args, **kwargs)
            strategy.signal.get_signal = signal
            if compiled: strategy.generate_trade_decision = MethodType(self.extension.generate_trade_decision, strategy)
            traces.append((trace, f.outcome(strategy)))
        self.exact(traces[1], traces[0])
        self.assertEqual(traces[0][0].count("signal"), 1)

    def test_mutable_quote_map_is_sequential_and_error_prefix_matches(self):
        f = self.fixture
        for fail in (False, True):
            outcomes = []
            for compiled in (False, True):
                strategy, _ = f.strategy(True, f.pd.Series({"AAA": .5, "BBB": -.5}))
                exchange, calls = strategy.trade_exchange, []
                class Mutable(dict):
                    def get(self, key, default=None):
                        calls.append(key)
                        if key == "AAA": self.pop("BBB", None)
                        if key == "BBB" and fail: raise LookupError("second symbol")
                        return super().get(key, default)
                exchange.quote.data = Mutable(exchange.quote.data)
                if compiled: strategy.generate_trade_decision = MethodType(self.extension.generate_trade_decision, strategy)
                outcomes.append((f.outcome(strategy), calls, list(exchange._session_rows)))
            self.exact(outcomes[1], outcomes[0])

    def test_row_same_day_weekend_missing_duplicate_and_mutable_cache(self):
        f = self.fixture
        for quote in (f.NumpyQuote, f.DailyQuote):
            for duplicate in (False, True):
                frame = f.frame({"AAA": 10., "BBB": f.np.nan})
                if duplicate: frame = f.pd.concat([frame, frame.loc[[("AAA", f.dates[0])]]]).sort_index()
                for end in (f.dates[0] + f.pd.Timedelta(hours=23), f.dates[0] + f.pd.Timedelta(days=2, hours=23)):
                    exchanges = [f.exchange(True, frame, quote) for _ in range(2)]
                    for e in exchanges: e.begin_session(f.dates[0], end)
                    for symbol in ("AAA", "BBB", "MISSING", "AAA"):
                        expected = exchanges[0]._session_row(symbol, f.dates[0], end)
                        actual = self.extension.session_row(exchanges[1], symbol, f.dates[0], end)
                        self.exact(actual, expected)
                        self.exact(exchanges[1]._session_rows, exchanges[0]._session_rows)

    def test_compiled_targets_delegates_before_callbacks(self):
        strategy = type("Sentinel", (), {"compiled_targets": True})()
        events = []
        with mock.patch.object(self.extension, "_PY_GENERATE", side_effect=lambda *a: events.append(a) or "result"):
            self.assertEqual(self.extension.generate_trade_decision(strategy, "execute"), "result")
        self.assertEqual(events, [(strategy, "execute")])

    @unittest.skipUnless(os.environ.get("QLIB_SESSION_LOOPS_MEASURE_RECEIPT"), "complete diagnostic explicitly disabled")
    def test_complete_3000_stock_alternating_diagnostic(self):
        receipt = complete_decision_pairs(self, os.environ["QLIB_SESSION_LOOPS_MEASURE_RECEIPT"])
        self.assertEqual(receipt["completed_pairs"], 5)


def complete_decision_pairs(testcase, output, *, stocks=3000, trials=5):
    """Complete synthetic boundary, fixture preparation/verification outside timing."""
    if stocks != 3000 or trials < 2:
        raise ValueError("Diagnostic requires 3000 stocks and at least two pairs")
    output = builder.checked_path(output)
    builder.checked_path(output.parent, kind="directory")
    record = {"schema_version": 1, "kind": "synthetic_complete_cython_decision_boundary",
              "status": "running", "stage": "fixture_preparation", "stocks": stocks,
              "acceptance_status": "ineligible_synthetic_probe", "goal_achieved": False,
              "extension_load_seconds": testcase.load_seconds, "build_identity": testcase.manifest,
              "completed_pairs": 0, "pairs": [],
              "scope": "complete guards+sequential row lookup+Python scalar sizing+native Orders; no backtest/report"}
    with output.open("x", encoding="utf-8") as handle:
        def persist():
            encoded = json.dumps(record, indent=2, allow_nan=False) + "\n"
            handle.seek(0); handle.write(encoded); handle.truncate(); handle.flush(); os.fsync(handle.fileno())
        persist()
        try:
            f = testcase.fixture
            prices = {f"S{i:05d}": float(5 + i % 100) / 3 for i in range(stocks)}
            weights = f.pd.Series({symbol: (-1 if i % 2 else 1) / (2 * stocks)
                                   for i, symbol in enumerate(prices)})
            frame = f.frame(prices)
            for trial in range(trials):
                outcomes, seconds = {}, {}
                variants = ("python", "cython") if trial % 2 == 0 else ("cython", "python")
                for variant in variants:
                    record.update(stage="fixture", active_trial=trial, active_variant=variant)
                    strategy, _ = f.strategy(True, weights, cash=1e6, frame=frame)
                    call = (f.FastStrategy.generate_trade_decision if variant == "python"
                            else testcase.extension.generate_trade_decision)
                    record["stage"] = "execution"
                    started = time.perf_counter()
                    decision = call(strategy)
                    seconds[variant] = time.perf_counter() - started
                    outcomes[variant] = (f.orders(decision.get_decision()),
                                         [{**r, "issued_orders": f.orders(r["issued_orders"])} for r in strategy.decisions])
                record["stage"] = "parity"
                testcase.exact(outcomes["cython"], outcomes["python"])
                record["pairs"].append({"trial": trial, "variant_order": variants,
                                         "seconds": seconds, "complete_decisions_exact": True})
                record.update(stage="paired_trials", completed_pairs=len(record["pairs"]))
                persist()
            record["median_seconds"] = {v: statistics.median(p["seconds"][v] for p in record["pairs"])
                                         for v in ("python", "cython")}
            record.update(status="completed", stage="completed")
            persist(); return record
        except (Exception, KeyboardInterrupt) as exc:
            record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            persist(); raise


if __name__ == "__main__":
    unittest.main()
