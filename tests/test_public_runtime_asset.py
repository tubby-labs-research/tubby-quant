"""Offline public build safeguards; never compile or install from tests."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("public_builder", ROOT / "scripts/build_public_runtime_asset.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class PublicBuilderTests(unittest.TestCase):
    def test_relative_cython_input_and_accepted_controls(self):
        program = builder.build_program()
        self.assertIn("[sys.argv[2]]", program)
        self.assertIn("build_dir=sys.argv[3]", program)
        self.assertIn("extra_link_args=['/Brepro']", program)
        self.assertIn("'/MD','/fp:strict','/Brepro','/experimental:deterministic',sys.argv[6]", program)
        self.assertIn("CYTHON_USE_DICT_VERSIONS", program)
        self.assertIn("'boundscheck':True,'wraparound':True,'cdivision':False", program)
        self.assertIn("'infer_types':False,'annotation_typing':False", program)
        self.assertIn("'/gl','/fp:fast','/fp:contract'", program)

    def test_recipe_drift_fails_closed(self):
        with patch.object(builder.accepted, "BUILD_PROGRAM", "changed upstream"):
            with self.assertRaises(RuntimeError):
                builder.build_program()

    def test_ascii_and_wide_privacy_detection(self):
        value = "Z:/Users/example/my-first-agent/source.pyx"
        for raw in (value.encode(), value.encode("utf-16le")):
            counts = builder.privacy_counts(raw)
            self.assertGreater(counts["absolute_windows_paths"], 0)
            self.assertGreater(counts["personal_roots"], 0)
            self.assertGreater(counts["checkout_name"], 0)
            with self.assertRaises(RuntimeError):
                builder.require_private_free(raw)
        self.assertFalse(any(builder.privacy_counts(b"qlib_yahoo/session_range_buffer.pyx").values()))

    def test_output_scope_new_only_and_redirects(self):
        (ROOT / "release-candidates").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "release-candidates", prefix="public-native-tests-") as directory:
            root = Path(directory)
            (root / "release-candidates").mkdir()
            with patch.object(builder, "ROOT", root):
                good = root / "release-candidates/public-native-a"
                self.assertEqual(builder.output_workspace(good), good)
                for bad in (root / "outside", root / "release-candidates/staging",
                            root / "release-candidates/public-native-a/nested",
                            root / "release-candidates/../public-native-a"):
                    with self.assertRaises(ValueError):
                        builder.output_workspace(bad)
                good.mkdir()
                with self.assertRaises(ValueError):
                    builder.output_workspace(good)

    def test_deterministic_two_entry_zip(self):
        (ROOT / "release-candidates").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "release-candidates", prefix="public-native-tests-") as directory:
            root = Path(directory)
            members = {builder.ARTIFACT: b"synthetic test bytes", "provenance.json": builder.encoded({"test": True})}
            a, b = root / "a.zip", root / "b.zip"
            builder.deterministic_zip(a, members)
            builder.deterministic_zip(b, dict(reversed(list(members.items()))))
            self.assertEqual(a.read_bytes(), b.read_bytes())
            with zipfile.ZipFile(a) as archive:
                self.assertEqual(sorted(archive.namelist()), sorted(members))
                for info in archive.infolist():
                    self.assertEqual(info.date_time, (1980, 1, 1, 0, 0, 0))
                    self.assertEqual(info.compress_type, zipfile.ZIP_STORED)
            with self.assertRaises(FileExistsError):
                builder.deterministic_zip(a, members)

    def test_current_three_inputs_match_descriptor(self):
        profile = json.loads((ROOT / "docs/qlib-fast-runtime.v1.json").read_bytes())
        sources = profile["date_range"]["source_sha256"]
        self.assertEqual(set(sources), set(builder.SOURCE_NAMES))
        for name, expected in sources.items():
            self.assertEqual(builder.accepted.sha256(ROOT / name), expected)

    def test_explicit_incomplete_acknowledgement_type(self):
        with self.assertRaises(TypeError):
            builder.build("unused", "unused", allow_incomplete_toolchain="yes")


if __name__ == "__main__":
    unittest.main()
