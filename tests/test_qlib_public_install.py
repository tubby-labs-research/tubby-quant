"""Offline public-bundle installation checks; all writes use disposable fixtures."""
import hashlib
import json
import shutil
import stat
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import install_fast_runtime as installer
from qlib_yahoo import fast_runtime as runtime
from qlib_yahoo.dataset import DatasetError


ASSET = "qlib-date-range-buffer-cp312-win_amd64-v1.zip"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class PublicInstallTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).absolute()
        self.archive = self.root / "runtime-artifacts" / ASSET
        self.archive.parent.mkdir()
        self.destination = self.root / "data/runtime-deps/native"
        self.contents = {"helper.pyd": b"synthetic native bytes; never imported", "provenance.json": b"{}"}
        self.spec = {
            "artifact": "helper.pyd", "artifact_sha256": digest(self.contents["helper.pyd"]),
            "provenance_sha256": digest(self.contents["provenance.json"]),
            "asset": ASSET, "bundled_asset": "runtime-artifacts/" + ASSET,
        }
        self.write_archive(list(self.contents.items()))
        root_patch = patch.object(installer, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def write_archive(self, members):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # Intentional duplicate ZIP fixtures.
            with zipfile.ZipFile(self.archive, "w") as archive:
                for name, raw in members:
                    archive.writestr(name, raw)
        self.spec["asset_sha256"] = runtime.sha256(self.archive)

    def install_without_transport(self, spec=None):
        with patch.object(installer.subprocess, "run", side_effect=AssertionError("transport invoked")) as run:
            installer.install_native(self.spec if spec is None else spec, self.destination)
        run.assert_not_called()

    def test_public_bundle_without_private_coordinates_or_gh(self):
        self.install_without_transport()
        runtime.verify_tree(self.destination, self.contents)
        self.assertEqual(self.archive.read_bytes()[:2], b"PK")
        self.install_without_transport()  # Verified, idempotent setup.

    def test_bundle_takes_precedence_over_private_coordinates(self):
        self.spec.update(release_repository="tubby-labs-research/Quant", release_tag="accepted-v1")
        self.install_without_transport()

    def test_missing_bundle_fails_even_with_private_coordinates_or_existing_install(self):
        self.spec.update(release_repository="tubby-labs-research/Quant", release_tag="accepted-v1")
        for existing in (False, True):
            with self.subTest(existing=existing):
                if existing:
                    self.install_without_transport()
                original = self.archive.read_bytes()
                self.archive.unlink()
                with self.assertRaisesRegex(DatasetError, "missing"):
                    self.install_without_transport()
                self.archive.write_bytes(original)
        runtime.verify_tree(self.destination, self.contents)

    def test_corrupt_bundle_fails_even_with_private_coordinates_or_existing_install(self):
        self.spec.update(release_repository="tubby-labs-research/Quant", release_tag="accepted-v1")
        for existing in (False, True):
            with self.subTest(existing=existing):
                if existing:
                    self.install_without_transport()
                original = self.archive.read_bytes()
                self.archive.write_bytes(original + b"tamper")
                with self.assertRaisesRegex(DatasetError, "SHA256"):
                    self.install_without_transport()
                self.assertEqual(self.destination.exists(), existing)
                self.archive.write_bytes(original)
        runtime.verify_tree(self.destination, self.contents)

    def test_invalid_bundle_paths_and_types_fail_closed(self):
        paths = (None, False, 1, [], {}, "", ".", "../outside.zip", "runtime-artifacts/../../outside.zip",
                 "/outside.zip", "C:/outside.zip", "C:outside.zip", "//server/share/native.zip",
                 "runtime-artifacts\\native.zip", "runtime-artifacts/./native.zip",
                 "runtime-artifacts//native.zip", "runtime-artifacts/NUL.zip",
                 "runtime-artifacts/native.zip.", "runtime-artifacts/native.zip ")
        for name in paths:
            with self.subTest(name=name), self.assertRaises(DatasetError):
                self.install_without_transport(dict(self.spec, bundled_asset=name))
        self.assertFalse(self.destination.exists())

    def test_bundle_directory_is_not_an_archive(self):
        self.archive.unlink()
        self.archive.mkdir()
        with self.assertRaisesRegex(DatasetError, "wrong type"):
            self.install_without_transport()
        self.assertFalse(self.destination.exists())

    def test_links_and_reparse_points_on_archive_and_parent_are_rejected(self):
        original = Path.lstat
        for redirected in (self.archive, self.archive.parent):
            for link in (False, True):
                def lstat(path, *args, **kwargs):
                    info = original(path, *args, **kwargs)
                    if path == redirected:
                        return SimpleNamespace(st_mode=stat.S_IFLNK if link else info.st_mode,
                                               st_file_attributes=0 if link else 0x400)
                    return info
                with self.subTest(path=redirected, link=link), patch.object(Path, "lstat", lstat):
                    with self.assertRaisesRegex(DatasetError, "Redirected"):
                        self.install_without_transport()
                self.assertFalse(self.destination.exists())

    def test_real_symlink_bundle_cannot_redirect_to_another_file(self):
        outside = self.root / "outside.zip"
        self.archive.rename(outside)
        try:
            self.archive.symlink_to(outside)
        except OSError as error:
            self.skipTest(f"Host does not permit symlink creation: {error}")
        with self.assertRaisesRegex(DatasetError, "Redirected"):
            self.install_without_transport()
        self.assertFalse(self.destination.exists())

    def test_destination_reparse_point_is_rejected_before_writing(self):
        self.destination.mkdir(parents=True)
        original = Path.lstat
        def lstat(path, *args, **kwargs):
            info = original(path, *args, **kwargs)
            if path == self.destination:
                return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
            return info
        with patch.object(Path, "lstat", lstat), self.assertRaisesRegex(DatasetError, "Redirected"):
            self.install_without_transport()
        self.assertEqual(list(self.destination.iterdir()), [])

    def test_member_hashes_checked_independently_of_archive_hash(self):
        for changed in self.contents:
            with self.subTest(member=changed):
                members = dict(self.contents, **{changed: b"changed member"})
                self.write_archive(list(members.items()))
                with self.assertRaisesRegex(DatasetError, "member digest"):
                    self.install_without_transport()
                self.assertFalse(self.destination.exists())

    def test_archive_inventory_and_member_paths_remain_strict(self):
        members = list(self.contents.items())
        variants = (members[:1], members + [members[0]], members + [("extra.txt", b"extra")],
                    members + [("../escape", b"escape")], members + [("/absolute", b"escape")],
                    members + [("folder/", b"")], members + [("HELPER.PYD", b"duplicate")])
        for entries in variants:
            with self.subTest(entries=entries):
                self.write_archive(entries)
                with self.assertRaises(DatasetError):
                    self.install_without_transport()
                self.assertFalse(self.destination.exists())

    def test_nonregular_archive_member_is_rejected(self):
        for mode in (stat.S_IFLNK, stat.S_IFIFO, stat.S_IFDIR):
            item = zipfile.ZipInfo("helper.pyd")
            item.create_system = 3
            item.external_attr = (mode | 0o644) << 16
            self.write_archive([(item, self.contents["helper.pyd"]),
                                ("provenance.json", self.contents["provenance.json"])])
            with self.subTest(mode=mode), self.assertRaisesRegex(DatasetError, "Unsafe"):
                self.install_without_transport()
            self.assertFalse(self.destination.exists())

    def test_malformed_zip_with_matching_outer_hash_is_rejected(self):
        self.archive.write_bytes(b"not a ZIP")
        self.spec["asset_sha256"] = runtime.sha256(self.archive)
        with self.assertRaises(zipfile.BadZipFile):
            self.install_without_transport()
        self.assertFalse(self.destination.exists())

    def test_malformed_pin_digest_is_rejected(self):
        self.spec["asset_sha256"] = "not a digest"
        with self.assertRaisesRegex(DatasetError, "Malformed"):
            self.install_without_transport()

    def test_existing_corrupt_or_extra_files_are_not_repaired(self):
        self.install_without_transport()
        helper = self.destination / "helper.pyd"
        helper.write_bytes(b"corrupt installed bytes")
        with self.assertRaisesRegex(DatasetError, "differs"):
            self.install_without_transport()
        self.assertEqual(helper.read_bytes(), b"corrupt installed bytes")
        helper.write_bytes(self.contents["helper.pyd"])
        (self.destination / "extra").write_bytes(b"unrecorded")
        with self.assertRaisesRegex(DatasetError, "Unrecorded"):
            self.install_without_transport()

    def test_private_coordinates_remain_compatible_when_bundle_key_is_absent(self):
        spec = dict(self.spec, release_repository="tubby-labs-research/Quant", release_tag="accepted-v1")
        del spec["bundled_asset"]
        def downloaded(command, **kwargs):
            self.assertEqual(command[:4], ["gh", "release", "download", "accepted-v1"])
            self.assertEqual(command[4:8], ["--repo", "tubby-labs-research/Quant", "--pattern", ASSET])
            self.assertEqual(kwargs["timeout"], 120)
            shutil.copyfile(self.archive, Path(command[-1]) / ASSET)
            return SimpleNamespace(returncode=0)
        with patch.object(installer.subprocess, "run", side_effect=downloaded) as run:
            installer.install_native(spec, self.destination)
        run.assert_called_once()
        runtime.verify_tree(self.destination, self.contents)
        self.archive.unlink()  # Private repeat setup still uses the verified install.
        self.install_without_transport(spec)

    def test_private_download_errors_remain_closed(self):
        spec = dict(self.spec, release_repository="tubby-labs-research/Quant", release_tag="accepted-v1")
        del spec["bundled_asset"]
        with patch.object(installer.subprocess, "run", side_effect=FileNotFoundError):
            with self.assertRaisesRegex(DatasetError, "GitHub CLI"):
                installer.install_native(spec, self.destination)
        self.assertFalse(self.destination.exists())

    def test_public_configure_still_verifies_sources_before_writing_config(self):
        # Exercise the normal configure/verification boundary with empty synthetic
        # wheel inventories, while retaining the real native/source hash checks.
        source = self.root / "builder.py"
        source.write_bytes(b"pinned fixture source")
        self.spec.update(folder="native", source_sha256={"builder.py": runtime.sha256(source)})
        pin = {"name": "accepted-v1", "compiler": {"folder": "compiler", "wheels": []},
               "gzip": {"folder": "gzip", "wheels": []}, "date_range": self.spec}
        pin_path = self.root / "pin.json"
        pin_path.write_text(json.dumps(pin), encoding="utf-8")
        config = self.root / "config.json"
        with patch.object(runtime, "ROOT", self.root), patch.object(runtime, "PIN", pin_path), \
                patch.object(runtime, "CONFIG", config), patch.object(runtime, "load_pin", return_value=pin), \
                patch.object(installer.sys, "version_info", (3, 12)), \
                patch.object(installer.sys, "platform", "win32"), \
                patch.object(runtime, "supported_windows", return_value=True), \
                patch.object(installer.subprocess, "run", side_effect=AssertionError("transport invoked")):
            installer.configure(self.root / "data", install_engine=False)
            runtime.verify_dependencies(pin, self.root / "data/runtime-deps")
            config.unlink()
            source.write_bytes(b"changed fixture source")
            with self.assertRaisesRegex(DatasetError, "SHA256"):
                installer.configure(self.root / "data", install_engine=False)
            self.assertFalse(config.exists())


if __name__ == "__main__":
    unittest.main()
