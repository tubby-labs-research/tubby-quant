"""Offline cache namespace checks and ordinary-path Windows demo acceptance."""
from contextlib import closing
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from qlib_yahoo import path_utils

ROOT = Path(__file__).resolve().parents[1]


def padded_cache_root(base, filename):
    """Reach MAX_PATH at the target filename, independent of checkout depth."""
    base = Path(base)
    while min(len(str(base / filename)), len(os.fsencode(base / filename))) < 260:
        needed = 260 - min(len(str(base / filename)), len(os.fsencode(base / filename)))
        base /= "p" * min(100, max(1, needed - 1))
    return base


class CachePathTests(unittest.TestCase):
    def test_drive_absolute_and_dot_segments(self):
        self.assertEqual(path_utils.windows_extended_path(r"C:\cache\parent\..\arrays"),
                         "\\\\?\\C:\\cache\\arrays")

    def test_unc_and_existing_namespace(self):
        unc = "\\\\server\\share\\cache"
        extended = "\\\\?\\UNC\\server\\share\\cache"
        self.assertEqual(path_utils.windows_extended_path(unc), extended)
        self.assertEqual(path_utils.windows_extended_path(extended), extended)
        drive = "\\\\?\\C:\\cache"
        self.assertEqual(path_utils.windows_extended_path(drive), drive)

    def test_relative_path_becomes_absolute_without_resolving_links(self):
        with patch.object(path_utils.ntpath, "abspath", return_value=r"C:\work\cache") as absolute, \
                patch.object(Path, "resolve", side_effect=AssertionError("no symlink resolution")):
            self.assertEqual(path_utils.windows_extended_path("cache"), "\\\\?\\C:\\work\\cache")
        absolute.assert_called_once_with("cache")

    def test_device_namespace_is_not_a_cache(self):
        with self.assertRaises(ValueError):
            path_utils.windows_extended_path("\\\\.\\PhysicalDrive0")

    def test_extended_device_and_malformed_namespaces_are_rejected(self):
        for tail in ("GLOBALROOT\\Device\\HarddiskVolume1\\cache", "Volume{fixture}\\cache",
                     "PhysicalDrive0", "", "C:", "C:relative", "1:\\cache",
                     "UNC", "UNC\\", "UNC\\server", "UNC\\server\\",
                     "UNC\\\\share", "UNC\\.\\share", "UNC\\server\\..",
                     "UNC\\server\\bad:share", "C:/cache", "C:\\cache\x00"):
            with self.subTest(tail=tail), self.assertRaises(ValueError):
                path_utils.windows_extended_path("\\\\?\\" + tail)
        for value in ("//?/C:/cache", "\\\\?/C:\\cache"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                path_utils.windows_extended_path(value)

    def test_extended_drive_and_unc_roots_are_accepted(self):
        for value in ("\\\\?\\C:\\", "\\\\?\\z:\\cache",
                      "\\\\?\\UNC\\server\\share", "\\\\?\\unc\\server\\share\\cache"):
            with self.subTest(value=value):
                self.assertEqual(path_utils.windows_extended_path(value), path_utils.ntpath.abspath(value))

    def test_non_windows_retains_relative_path(self):
        original = Path("relative/cache")
        with patch.object(path_utils, "os", SimpleNamespace(name="posix")), \
                patch.object(path_utils, "windows_extended_path", side_effect=AssertionError("Windows only")):
            self.assertEqual(path_utils.cache_io_path(original), original)


@unittest.skipUnless(os.name == "nt", "actual Windows filesystem acceptance")
class WindowsCacheTests(unittest.TestCase):
    def setUp(self):
        # Honor the caller's ordinary TMP/TEMP, independent of the source root.
        self.root = Path(tempfile.mkdtemp(prefix="win-cache-"))
        self.assertFalse(str(self.root).startswith("\\\\?\\"))
        self.addCleanup(shutil.rmtree, path_utils.cache_io_path(self.root))

    def assert_long_filename(self, path):
        ordinary = str(path).removeprefix("\\\\?\\")
        self.assertGreaterEqual(len(ordinary), 260)
        self.assertGreaterEqual(len(os.fsencode(ordinary)), 260)

    def source(self):
        import numpy as np
        dates = np.arange("2024-01-01", "2024-01-05", dtype="datetime64[D]")
        values = np.arange(8, dtype=np.float32).reshape(4, 2) + 20
        return SimpleNamespace(
            identity={"dataset": str(self.root / "synthetic"), "source": "fixture"},
            dates=dates, symbols=["SYN00", "SYN01"], fields={"close": values},
            groups={}, panel=SimpleNamespace(manifest={}),
            universe=lambda _: np.ones(values.shape, dtype=bool))

    def test_numpy_writer_reader_legacy_reuse_and_symlink_guard(self):
        import numpy as np
        from qlib_yahoo.alphalab.batch import CachedWorkspace, write_cache
        from qlib_yahoo.dataset import DatasetError
        source = self.source()
        base = padded_cache_root(self.root / "cache",
                                 Path("v2-" + "c" * 64) / "field_close.npy") / "source"
        directory = write_cache(source, base, ["TOP2"])
        self.assert_long_filename(directory / "field_close.npy")
        self.assert_long_filename(directory / "universe_TOP2.npy")
        self.assertFalse(str(directory).startswith("\\\\?\\"))
        with closing(CachedWorkspace(directory)) as cached:
            np.testing.assert_array_equal(cached.fields["close"], source.fields["close"])
            np.testing.assert_array_equal(cached.universe("TOP2"), source.universe("TOP2"))
        meta = json.loads((path_utils.cache_io_path(directory) / "meta.json").read_text())
        self.assertEqual(meta["identity"], source.identity)
        self.assertEqual(directory.name, "v2-" + meta["content_sha256"])
        self.assertEqual(len(meta["content_sha256"]), 64)
        self.assertEqual(write_cache(source, base, ["TOP2"]), directory)
        legacy = base.with_name(base.name + "-" + meta["content_sha256"])
        path_utils.cache_io_path(directory).rename(path_utils.cache_io_path(legacy))
        self.assertEqual(write_cache(source, base, ["TOP2"]), legacy)
        original = Path.is_symlink
        def linked(path):
            return path == path_utils.cache_io_path(legacy) or original(path)
        with patch.object(Path, "is_symlink", linked), self.assertRaises(DatasetError):
            CachedWorkspace(legacy)
        (path_utils.cache_io_path(legacy) / "field_close.npy").write_bytes(b"corrupt")
        with self.assertRaises(DatasetError):
            write_cache(source, base, ["TOP2"])

    def test_child_demo_succeeds_and_resumes_with_ordinary_long_cache_paths(self):
        from autoresearch.records import ResearchStore
        from autoresearch.demo import create_demo
        demo = self.root / "demo"
        # The real worker uses two 64-character cache digests, without prefixing
        # its managed data, records, runtime or SQLite paths. Pad only the runtime
        # branch enough for its cache leaf; keep the demo/records root short.
        cache_leaf = Path("engine/backend") / ("b" * 64) / "cache/alphalab" / ("v2-" + "c" * 64) / "universe_TOP8.npy"
        runtime = padded_cache_root(self.root / "runtime", cache_leaf)
        self.assert_long_filename(runtime / cache_leaf)
        config_path = create_demo(demo)
        config = json.loads(config_path.read_text())
        config["runtime_root"] = str(runtime)
        config_path.write_text(json.dumps(config))
        command = [sys.executable, "-m", "autoresearch", "demo", "--root", str(demo)]
        for attempt in range(2):
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            updates = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual(updates[-1]["evaluations"], 3)
            self.assertEqual(updates[-1]["queued"], 0)
            experiments = ResearchStore(demo / "research").list("experiments")
            self.assertEqual(len(experiments), 3)
            self.assertTrue(all(e["result"]["status"] == "ok" for e in experiments))
            for experiment in experiments:
                for artifact in experiment["result"]["artifacts"]:
                    self.assertLess(len(str(runtime / "engine" / artifact)), 260)
            for record in (demo / "research").rglob("*"):
                self.assertLess(len(str(record)), 260)
            if attempt == 0:
                ids = {e["id"] for e in experiments}
            else:
                self.assertEqual({e["id"] for e in experiments}, ids)
        cache_root = runtime / "engine/backend"
        files = list(path_utils.cache_io_path(cache_root).rglob("universe_TOP8.npy"))
        self.assertTrue(files)
        for filename in files:
            self.assert_long_filename(filename)


if __name__ == "__main__":
    unittest.main()
