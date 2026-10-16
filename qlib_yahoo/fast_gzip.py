"""Opt-in, verified isolated ISA-L gzip factory; no JSON or engine changes.

Selection is before opening output. Runtime compression errors propagate: this
module never retries or switches codecs after a write. ISA-L levels 0..3 are
not equivalent to zlib levels. No optional package is imported by default.
"""
from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import importlib.util
import io
import platform
import stat
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

WHEEL_NAME = "isal-1.8.0-cp312-cp312-win_amd64.whl"
WHEEL_SHA256 = "1d376b7644434d50fedfb670483150ece64082212b6e1f23976f92a91fa1b99b"
WHEEL_SIZE = 203025
VERSION = "1.8.0"
_loaded = None


def _plain_path(path, *, directory=False):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"redirected path: {item}")
    if directory and not path.is_dir():
        raise ValueError(f"not a directory: {path}")
    if not directory and not path.is_file():
        raise ValueError(f"not a regular file: {path}")
    return path.resolve(strict=True)


def _supported_abi():
    return (sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 12)
            and sys.platform == "win32" and platform.machine().lower() in ("amd64", "x86_64")
            and sys.maxsize > 2**32)


def _verified_install(path):
    root = _plain_path(path, directory=True)
    wheel = _plain_path(root / WHEEL_NAME)
    packed = wheel.read_bytes()
    if len(packed) != WHEEL_SIZE or hashlib.sha256(packed).hexdigest() != WHEEL_SHA256:
        raise ValueError("approved wheel size/SHA256 mismatch")
    site = _plain_path(root / "site-packages", directory=True)
    dist = "isal-1.8.0.dist-info/"
    hashes = {}
    with zipfile.ZipFile(io.BytesIO(packed)) as archive:
        names = set()
        for item in archive.infolist():
            name = item.filename
            parts = PurePosixPath(name).parts
            if (not parts or name.startswith("/") or "\\" in name or ":" in name
                    or ".." in parts or name in names or item.is_dir()
                    or stat.S_ISLNK(item.external_attr >> 16)):
                raise ValueError("unsafe/duplicate wheel member")
            names.add(name)
        record_name = dist + "RECORD"
        record = archive.read(record_name)
        records = list(csv.reader(io.StringIO(record.decode("utf-8"))))
        if any(len(row) != 3 for row in records) or len({row[0] for row in records}) != len(records):
            raise ValueError("invalid wheel RECORD")
        if {row[0] for row in records} != names:
            raise ValueError("wheel RECORD inventory mismatch")
        for name, digest, size in records:
            original = archive.read(name)
            if name != record_name:
                expected = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(original).digest()).rstrip(b"=").decode()
                if digest != expected or size != str(len(original)):
                    raise ValueError("wheel RECORD content mismatch")
            elif digest or size:
                raise ValueError("unexpected RECORD self-hash")
            installed = _plain_path(site.joinpath(*PurePosixPath(name).parts))
            content = installed.read_bytes()
            if content != original:
                raise ValueError(f"installed file differs from approved wheel: {name}")
            hashes[name] = hashlib.sha256(content).hexdigest()
        # No unrecorded modules, native libraries or modified metadata.
        actual = set()
        for installed in site.rglob("*"):
            _plain_path(installed, directory=installed.is_dir())
            if installed.is_file():
                actual.add(installed.relative_to(site).as_posix())
        if actual != names:
            raise ValueError("unrecorded installed files")
        metadata = BytesParser().parsebytes(archive.read(dist + "METADATA"))
        if (metadata["Name"] != "isal" or metadata["Version"] != VERSION
                or metadata.get_all("Requires-Dist")):
            raise ValueError("unexpected package identity/dependencies")
        wheel_metadata = BytesParser().parsebytes(archive.read(dist + "WHEEL"))
        if wheel_metadata.get_all("Tag") != ["cp312-cp312-win_amd64"]:
            raise ValueError("unexpected wheel ABI")
    return root, site, hashes


def _load_verified(site):
    global _loaded
    present = {key: value for key, value in sys.modules.items() if key == "isal" or key.startswith("isal.")}
    if _loaded is not None:
        old_site, modules, bindings = _loaded
        if (old_site != site or present.keys() != modules.keys()
                or any(present[key] is not module for key, module in modules.items())):
            raise RuntimeError("ISA-L module ownership changed")
        if any(getattr(owner, name, None) is not expected for owner, name, expected in bindings):
            raise RuntimeError("ISA-L binding changed")
        return modules["isal"], modules["isal.igzip"]
    if present:
        raise RuntimeError("preexisting ISA-L modules are not owned by this loader")
    old_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("isal", site / "isal/__init__.py",
                                                    submodule_search_locations=[str(site / "isal")])
        package = importlib.util.module_from_spec(spec)
        sys.modules["isal"] = package
        spec.loader.exec_module(package)
        from isal import igzip
        if package.__version__ != VERSION or igzip.GzipFile is not igzip.IGzipFile:
            raise RuntimeError("ISA-L imported identity mismatch")
        if igzip.IGzipFile.__bases__ != (gzip.GzipFile,) or igzip.gzip is not gzip:
            raise RuntimeError("unexpected gzip binding")
        modules = {key: value for key, value in sys.modules.items() if key == "isal" or key.startswith("isal.")}
        for module in modules.values():
            source = _plain_path(module.__file__)
            if not source.is_relative_to(site):
                raise RuntimeError("ISA-L module outside verified install")
        bindings = [(package, "__version__", package.__version__),
                    (package, "ISAL_VERSION", package.ISAL_VERSION),
                    (igzip, "IGzipFile", igzip.IGzipFile), (igzip, "GzipFile", igzip.GzipFile),
                    (igzip, "gzip", gzip), (igzip, "isal_zlib", igzip.isal_zlib),
                    (igzip, "igzip_lib", igzip.igzip_lib)]
        for owner in (igzip.IGzipFile, gzip.GzipFile, igzip.isal_zlib, igzip.igzip_lib):
            bindings.extend((owner, name, value) for name, value in vars(owner).items()
                            if callable(value) or name.isupper())
        bindings.extend((igzip, name, value) for name, value in vars(igzip).items()
                        if name.isupper())
        _loaded = (site, modules, bindings)
        return package, igzip
    except BaseException:
        for key in list(sys.modules):
            if key == "isal" or key.startswith("isal."):
                del sys.modules[key]
        raise
    finally:
        sys.dont_write_bytecode = old_bytecode


def activate_gzip(path, level=2):
    """Return (write-only factory, provenance), without opening any output.

    ``path`` is the isolated root containing the approved wheel and its exact
    ``site-packages`` extraction. Requested missing/tampered installs fail;
    a verified install on an unsupported ABI selects stdlib gzip level 6 before
    any write. The identity explicitly distinguishes requested and actual level.
    """
    if type(level) is not int or not 0 <= level <= 3:
        raise ValueError("ISA-L level must be an integer from 0 through 3")
    root, site, hashes = _verified_install(path)
    identity = {"format": "gzip", "requested_backend": "ISA-L", "requested_level": level,
                "path": str(root), "wheel": WHEEL_NAME, "wheel_sha256": WHEEL_SHA256,
                "version": VERSION, "installed_file_sha256": hashes,
                "level_equivalent_to_zlib": False}
    use_isal = _supported_abi()
    if use_isal:
        package, igzip = _load_verified(site)
        implementation, actual_level = igzip.IGzipFile, level
        identity.update(backend="ISA-L", actual_level=actual_level, fallback_reason=None,
                        isal_version=package.ISAL_VERSION)
    else:
        implementation, actual_level = gzip.GzipFile, 6
        identity.update(backend="stdlib-zlib", actual_level=actual_level,
                        fallback_reason="unsupported CP312 Windows x64 ABI")

    def factory(filename="", mode="wb", compresslevel=actual_level, fileobj=None, mtime=0):
        if (type(filename) is not str or filename != "" or type(mode) is not str or mode != "wb"
                or type(mtime) is not int or mtime != 0
                or type(compresslevel) is not int or compresslevel != actual_level or fileobj is None):
            raise ValueError("factory requires anonymous deterministic write-only gzip and selected level")
        if use_isal:
            _load_verified(site)  # Detect binding mutation before output is opened/written.
        return implementation(filename="", mode="wb", compresslevel=actual_level,
                              fileobj=fileobj, mtime=0)

    return factory, identity
