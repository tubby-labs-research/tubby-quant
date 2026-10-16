"""Offline encoding regressions using real, bounded Python child processes."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UNICODE = "\u6d4b\u8bd5-\u03bb-\U0001f680"


class SignedInstallerEncodingTests(unittest.TestCase):
    def probe(self, source):
        """Start an installer parent whose pipes really are CP1252 at startup."""
        for utf8 in (None, "0"):
            with self.subTest(PYTHONUTF8=utf8), tempfile.TemporaryDirectory() as temporary:
                cwd = Path(temporary) / UNICODE
                cwd.mkdir()
                env = os.environ.copy()
                env.pop("PYTHONUTF8", None)
                if utf8 is not None:
                    env["PYTHONUTF8"] = utf8
                env.update(PYTHONIOENCODING="cp1252", PYTHONDONTWRITEBYTECODE="1")
                prefix = (
                    "import os, sys, json\n"
                    "from pathlib import Path\n"
                    "from unittest.mock import patch\n"
                    f"sys.path.insert(0, {str(ROOT)!r})\n"
                    "from scripts import install_signed_runtime as installer\n"
                    "from qlib_yahoo.dataset import DatasetError\n"
                    "assert sys.stdout.encoding == 'cp1252', sys.stdout.encoding\n"
                    "assert sys.stderr.encoding == 'cp1252', sys.stderr.encoding\n"
                    "before = dict(os.environ)\n"
                )
                result = subprocess.run(
                    [sys.executable, "-B", "-c", prefix + textwrap.dedent(source)],
                    cwd=cwd, env=env, capture_output=True, timeout=30,
                )
                stdout = result.stdout.decode("utf-8")
                stderr = result.stderr.decode("utf-8")
                self.assertEqual(result.returncode, 0, (stdout, stderr))
                yield cwd, stdout, stderr

    def test_run_unicode_stdout_stderr_and_cwd_preserves_environment(self):
        child = (
            "import json, os, sys; "
            "assert sys.stdout.encoding == sys.stderr.encoding == 'utf-8'; "
            "assert os.environ['PYTHONUTF8'] == '1'; "
            "assert os.environ['PYTHONIOENCODING'] == 'utf-8'; "
            f"print({UNICODE!r}, file=sys.stderr); "
            "print(json.dumps({'cwd': os.getcwd()}, ensure_ascii=False))"
        )
        source = f"""
            output = installer._run(sys.executable, '-B', '-c', {child!r}, cwd=Path.cwd())
            assert json.loads(output)['cwd'] == str(Path.cwd()), output
            assert dict(os.environ) == before
            assert sys.stdout.encoding == 'cp1252'
            print(json.dumps(json.loads(output)))
        """
        for cwd, stdout, stderr in self.probe(source):
            self.assertEqual(json.loads(stdout), {"cwd": str(cwd)})
            self.assertEqual(stderr, "")

    def test_run_unicode_stderr_failure_is_dataset_error(self):
        child = (
            "import os, sys; "
            f"print({UNICODE!r} + ' ' + os.getcwd(), file=sys.stderr); sys.exit(7)"
        )
        source = f"""
            try:
                installer._run(sys.executable, '-B', '-c', {child!r}, cwd=Path.cwd())
            except DatasetError as error:
                print(json.dumps(str(error)))
            else:
                raise AssertionError('failed command succeeded')
            assert dict(os.environ) == before
        """
        for cwd, stdout, stderr in self.probe(source):
            message = json.loads(stdout)
            self.assertIn(UNICODE + " " + str(cwd), message)
            self.assertIn("failed:", message)
            self.assertEqual(stderr, "")

    def test_run_unicode_stdout_failure_fallback_is_dataset_error(self):
        child = f"import sys; print({UNICODE!r}); sys.exit(9)"
        source = f"""
            try:
                installer._run(sys.executable, '-B', '-c', {child!r})
            except DatasetError as error:
                print(json.dumps(str(error)))
            else:
                raise AssertionError('failed command succeeded')
            assert dict(os.environ) == before
        """
        for _, stdout, _ in self.probe(source):
            self.assertTrue(json.loads(stdout).endswith("failed: " + UNICODE))

    def test_run_legacy_diagnostic_bytes_remain_dataset_error(self):
        # Raw bytes model non-Python tools which ignore PYTHONIOENCODING.
        child = "import os, sys; os.write(2, b'legacy \\x81\\xe9 error'); sys.exit(3)"
        source = f"""
            try:
                installer._run(sys.executable, '-B', '-c', {child!r})
            except DatasetError as error:
                print(json.dumps(str(error)))
            else:
                raise AssertionError('failed command succeeded')
            assert dict(os.environ) == before
        """
        for _, stdout, _ in self.probe(source):
            self.assertTrue(json.loads(stdout).endswith("failed: legacy \ufffd\ufffd error"))

    def test_derivative_wheel_build_and_move_in_unicode_cwd(self):
        raw = b"synthetic pinned wheel\x00\xff"
        builder = textwrap.dedent(f"""
            import argparse
            import hashlib
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument('--base-wheel', type=Path, required=True)
            parser.add_argument('--base-sha256', required=True)
            parser.add_argument('--output-dir', type=Path, required=True)
            args = parser.parse_args()
            assert Path.cwd().name == {UNICODE!r}
            assert hashlib.sha256(args.base_wheel.read_bytes()).hexdigest() == args.base_sha256
            output = args.output_dir / 'synthetic.whl'
            output.write_bytes({raw!r})
            print(output, flush=True)
        """)
        source = f"""
            import hashlib
            root = Path.cwd()
            base = root / 'base.whl'
            base.write_bytes(b'synthetic base')
            builder = root / 'synthetic-source/scripts/build_signed_wheel.py'
            builder.parent.mkdir(parents=True)
            builder.write_text({builder!r}, encoding='utf-8')
            wheel = root / 'wheels/synthetic.whl'
            pin = dict(local_wheel=str(wheel.relative_to(root)),
                       base_wheel_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
                       wheel_sha256=hashlib.sha256({raw!r}).hexdigest())
            assert not wheel.exists()
            # Use the real subprocess, digest check, and move; only redirect ROOT.
            with patch.object(installer, 'ROOT', root):
                result = installer.derivative_wheel(pin, base, builder.parents[1])
            assert result == wheel
            assert result.read_bytes() == {raw!r}
            assert list(wheel.parent.iterdir()) == [wheel]
            assert base.read_bytes() == b'synthetic base'
            assert dict(os.environ) == before
        """
        for _, stdout, stderr in self.probe(source):
            self.assertEqual(stdout.strip(), "Derivative wheel built and verified: synthetic.whl")
            self.assertEqual(stderr, "")

    def test_main_unicode_progress_in_captured_parent(self):
        # Existing synthetic wheels/runtime: exercise real main and progress
        # prints without downloading, building, or installing any package.
        source = """
            import hashlib
            root = Path.cwd()
            raw = b'synthetic wheel'
            digest = hashlib.sha256(raw).hexdigest()
            base = root / 'env/qlib-signed-base/pyqlib-0.9.7-test.whl'
            base.parent.mkdir(parents=True)
            base.write_bytes(raw)
            wheel = root / ('derivative-' + root.name + '.whl')
            wheel.write_bytes(raw)
            runtime = root / ('runtime-' + root.name)
            runtime.mkdir()
            pin = dict(platform_tag='test', version='test', base_wheel_sha256=digest,
                       wheel_sha256=digest, local_wheel=wheel.name, local_runtime=runtime.name)
            pin_path = root / 'pin.json'
            pin_path.write_text(json.dumps(pin), encoding='utf-8')
            with patch.object(installer, 'ROOT', root), \\
                 patch.object(installer, 'PIN', pin_path), \\
                 patch.object(installer, 'platform_pin', side_effect=lambda p: p), \\
                 patch.object(installer, 'platform_key', return_value=root.name), \\
                 patch.object(installer, 'fork_checkout', return_value=root), \\
                 patch.object(installer, 'verify_signed_install') as verify, \\
                 patch.object(installer, '_run', side_effect=AssertionError('unexpected command')), \\
                 patch.object(sys, 'version_info', (3, 12)), \\
                 patch.object(sys, 'argv', ['installer']):
                assert installer.main() == 0
                verify.assert_called_once_with(pin, root)
            assert dict(os.environ) == before
        """
        for cwd, stdout, stderr in self.probe(source):
            self.assertIn("Platform " + UNICODE, stdout)
            self.assertIn("Derivative wheel verified: derivative-" + UNICODE, stdout)
            self.assertIn("Existing runtime verified: " + str(cwd), stdout)
            self.assertIn("Signed runtime ready.", stdout)
            self.assertEqual(stderr, "")

    def test_main_unicode_error_in_captured_parent(self):
        source = """
            error = DatasetError('rejected ' + str(Path.cwd()))
            with patch.object(installer, 'PIN', Path.cwd() / 'absent-pin'), \\
                 patch.object(Path, 'read_text', side_effect=error), \\
                 patch.object(sys, 'version_info', (3, 12)), \\
                 patch.object(sys, 'argv', ['installer']):
                assert installer.main() == 1
            assert dict(os.environ) == before
        """
        for cwd, stdout, stderr in self.probe(source):
            self.assertEqual(stdout, "")
            self.assertEqual(stderr.strip(), "error: rejected " + str(cwd))

    def test_unicode_wheel_paths_still_require_exact_hashes(self):
        source = """
            root = Path.cwd()
            base = root / 'env/qlib-signed-base/pyqlib-0.9.7-test.whl'
            base.parent.mkdir(parents=True)
            base.write_bytes(b'corrupt base')
            wheel = root / (root.name + '.whl')
            wheel.write_bytes(b'corrupt derivative')
            pin = dict(platform_tag='test', base_wheel_sha256='0' * 64,
                       wheel_sha256='0' * 64, local_wheel=wheel.name)
            with patch.object(installer, 'ROOT', root), \\
                 patch.object(installer, '_run', side_effect=AssertionError('unexpected command')):
                for operation in (lambda: installer.base_wheel(pin),
                                  lambda: installer.derivative_wheel(pin, base, root)):
                    try:
                        operation()
                    except DatasetError as error:
                        assert 'differs from the pin' in str(error), error
                    else:
                        raise AssertionError('accepted corrupt wheel')
            assert base.read_bytes() == b'corrupt base'
            assert wheel.read_bytes() == b'corrupt derivative'
            assert dict(os.environ) == before
        """
        for _, stdout, stderr in self.probe(source):
            self.assertEqual((stdout, stderr), ("", ""))


if __name__ == "__main__":
    unittest.main()
