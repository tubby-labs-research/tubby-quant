"""Accepted signed runtime selection and dependency verification, before Qlib imports."""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import platform
import re
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath

from .dataset import DatasetError

ROOT = Path(__file__).absolute().parents[1]
PIN = ROOT / "docs/qlib-fast-runtime.v1.json"
CONFIG = ROOT / "env/qlib-fast-runtime.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_path(path, *, directory=False, missing=False):
    """Reject links/reparse points in every existing ancestor, even before mkdir."""
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise DatasetError("Runtime paths must be absolute without parent traversal")
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            if missing:
                continue
            raise DatasetError(f"Runtime path is missing: {part}") from None
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise DatasetError(f"Redirected runtime path is forbidden: {part}")
    if not missing and not (path.is_dir() if directory else path.is_file()):
        raise DatasetError(f"Runtime path has the wrong type: {path}")
    return path


def member_name(name):
    parts = PurePosixPath(name).parts
    if (not parts or name.startswith("/") or "\\" in name or ":" in name
            or ".." in parts or "." in name.split("/") or "" in name.split("/")
            or any(part.rstrip(". ") != part for part in parts)
            or any(part.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL",
                                                   *(f"COM{i}" for i in range(1, 10)),
                                                   *(f"LPT{i}" for i in range(1, 10))} for part in parts)):
        raise DatasetError(f"Unsafe runtime archive member: {name}")
    return parts


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise DatasetError("Duplicate runtime JSON key")
            result[key] = value
        return result
    try:
        value = json.loads(checked_path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)
    except (ValueError, OSError) as error:
        raise DatasetError(f"Cannot read runtime configuration: {error}") from error
    if type(value) is not dict:
        raise DatasetError("Runtime JSON must be an object")
    return value


def load_pin(pin_path=None):
    pin = read_json(PIN if pin_path is None else pin_path)
    if (type(pin.get("schema_version")) is not int or pin["schema_version"] != 1
            or pin.get("kind") != "accepted_signed_runtime_profile" or pin.get("name") != "accepted-v1"):
        raise DatasetError("Invalid accepted runtime profile pin")
    for key in ("compiler", "gzip", "date_range"):
        member_name(pin[key]["folder"])
        if len(PurePosixPath(pin[key]["folder"]).parts) != 1:
            raise DatasetError("Runtime dependency folder must be one component")
    return pin


def verify_blob(path, digest, size=None):
    checked_path(path)
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise DatasetError("Malformed runtime digest")
    if size is not None and (type(size) is not int or size < 0 or Path(path).stat().st_size != size):
        raise DatasetError(f"Runtime artifact size mismatch: {path}")
    if sha256(path) != digest:
        raise DatasetError(f"Runtime artifact SHA256 mismatch: {path}")


def wheel_contents(path, spec):
    """Verify whole pinned wheel and each RECORD entry; return exact extraction bytes."""
    verify_blob(path, spec["sha256"], spec["size"])
    try:
        with zipfile.ZipFile(path) as archive:
            content = {}
            folded = set()
            for item in archive.infolist():
                # Official wheels may contain explicit, empty directory entries.
                # Validate their canonical name but exclude them from RECORD's
                # file inventory and the extracted byte inventory.
                name = item.filename[:-1] if item.is_dir() else item.filename
                member_name(name)
                mode = item.external_attr >> 16
                allowed_modes = (0, stat.S_IFDIR) if item.is_dir() else (0, stat.S_IFREG)
                if (stat.S_ISLNK(mode) or stat.S_IFMT(mode) not in allowed_modes
                        or name.casefold() in folded or (item.is_dir() and item.file_size != 0)):
                    raise DatasetError("Unsafe or duplicate wheel member")
                folded.add(name.casefold())
                if item.is_dir():
                    continue
                content[item.filename] = archive.read(item)
            records = [name for name in content if name.endswith(".dist-info/RECORD")]
            if len(records) != 1:
                raise DatasetError("Wheel must have exactly one RECORD")
            record = records[0]
            rows = list(csv.reader(io.StringIO(content[record].decode("utf-8"))))
            if (any(len(row) != 3 for row in rows) or len({row[0] for row in rows}) != len(rows)
                    or {row[0] for row in rows} != set(content)):
                raise DatasetError("Wheel RECORD inventory mismatch")
            for name, digest, size in rows:
                raw = content[name]
                expected = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode()
                if ((name == record and (digest or size))
                        or (name != record and (digest != expected or size != str(len(raw))))):
                    raise DatasetError("Wheel RECORD content mismatch")
            return content
    except (zipfile.BadZipFile, UnicodeError, KeyError) as error:
        raise DatasetError(f"Invalid runtime wheel: {error}") from error


def verify_tree(root, contents, *, extra=()):
    checked_path(root, directory=True)
    actual = set()
    mismatches = set()
    comparison_errors = {}
    for path in root.rglob("*"):
        name = path.relative_to(root).as_posix()
        if name in contents:
            try:
                member_name(name)
            except DatasetError as error:
                comparison_errors[name] = error
        directory = path.is_dir()
        checked_path(path, directory=directory)
        if not directory:
            actual.add(name)
            if name in contents and name not in comparison_errors:
                try:
                    if path.read_bytes() != contents[name]:
                        mismatches.add(name)
                except OSError as error:
                    comparison_errors[name] = error
    if actual != set(contents) | set(extra):
        raise DatasetError(f"Unrecorded or missing runtime files: {root}")
    # Keep inventory errors first and content errors in the expected inventory's
    # order, while checking paths and bytes together in the single tree walk.
    for name in contents:
        if name in comparison_errors:
            raise comparison_errors[name]
        if name in mismatches:
            raise DatasetError(f"Installed runtime file differs from pinned archive: {name}")


def compiler_contents(contents):
    """Relocate only wheel scripts, retaining original bytes and verified RECORD."""
    mapped = {}
    folded = set()
    for name, raw in contents.items():
        parts = member_name(name)
        if parts[0].endswith(".data"):
            if len(parts) != 3 or parts[1] != "scripts":
                raise DatasetError("Unsupported compiler wheel data relocation")
            name = "bin/" + parts[2]
        if name.casefold() in folded:
            raise DatasetError("Compiler wheel relocation collision")
        folded.add(name.casefold())
        mapped[name] = raw
    return mapped


def verify_dependencies(pin, dependency_root):
    root = checked_path(dependency_root, directory=True)
    compiler = root / pin["compiler"]["folder"]
    combined = {}
    for spec in pin["compiler"]["wheels"]:
        contents = compiler_contents(wheel_contents(root / "wheels" / spec["filename"], spec))
        if {name.casefold() for name in combined} & {name.casefold() for name in contents}:
            raise DatasetError("Compiler wheels have overlapping members")
        combined.update(contents)
    verify_tree(compiler, combined)
    gzip = root / pin["gzip"]["folder"]
    gzip_contents = {}
    for spec in pin["gzip"]["wheels"]:
        gzip_contents.update(wheel_contents(gzip / spec["filename"], spec))
    verify_tree(gzip / "site-packages", gzip_contents)
    expected_gzip = {"site-packages", *(spec["filename"] for spec in pin["gzip"]["wheels"])}
    if {item.name for item in gzip.iterdir()} != expected_gzip:
        raise DatasetError("Unexpected gzip runtime entry")
    native = root / pin["date_range"]["folder"]
    spec = pin["date_range"]
    member_name(spec["artifact"])
    verify_blob(native / spec["artifact"], spec["artifact_sha256"])
    verify_blob(native / "provenance.json", spec["provenance_sha256"])
    verify_tree(native, {spec["artifact"]: (native / spec["artifact"]).read_bytes(),
                         "provenance.json": (native / "provenance.json").read_bytes()})
    for name, digest in spec["source_sha256"].items():
        verify_blob(ROOT.joinpath(*member_name(name)), digest)
    return compiler, gzip, native


def supported_windows():
    return (sys.platform == "win32" and sys.implementation.name == "cpython"
            and sys.version_info[:2] == (3, 12) and sys.maxsize > 2**32
            and platform.machine().lower() in ("amd64", "x86_64"))


def resolve_fast_profile():
    """Return private run_backtest kwargs plus truthful platform provenance."""
    portable = {"_columnar_history": True, "_flat_indicators": True,
                "_buffered_history": True, "_index_reuse": True}
    fallback = {"profile": "accepted-v1", "name": "accepted-v1", "requested": "auto", "engine": "native-signed",
                "gzip": "stdlib-zlib", "date_range": "python", "ordered_session": False}
    if sys.platform == "darwin":
        return portable, dict(fallback, platform="macos", selection="accepted_portable_macos",
                              fallbacks=["ordered_session:native", "gzip:stdlib-zlib", "date_range:python"],
                              fallback_reason="Windows acceleration ABI unavailable")
    if not supported_windows():
        return {}, dict(fallback, platform=sys.platform, selection="unsupported_host_fallback",
                        fallbacks=["acceleration:unavailable"], fallback_reason="Unsupported acceleration host")
    if not CONFIG.exists():
        raise DatasetError("Accepted signed runtime is not configured. Run env\\qlib\\Scripts\\python.exe "
                           "scripts\\install_fast_runtime.py --data-root <your-Qlib-data-folder>")
    try:
        config = read_json(CONFIG)
        if (type(config.get("schema_version")) is not int or config["schema_version"] != 1
                or config.get("profile") != "accepted-v1"):
            raise DatasetError("Invalid local fast runtime configuration")
        pin = load_pin()
        verify_blob(PIN, config["profile_pin_sha256"])
        compiler, gzip, native = verify_dependencies(pin, Path(config["dependency_root"]))
    except (OSError, KeyError, ValueError) as error:
        raise DatasetError(f"Accepted signed runtime verification failed: {error}") from error
    kwargs = dict(portable, _ordered_session=True, _compiler_path=str(compiler),
                  _gzip_path=str(gzip), _isal_level=2, _installed_range_root=str(native))
    identity = {"profile": pin["name"], "name": pin["name"], "requested": "auto",
                "selection": "accepted_windows", "fallbacks": [], "profile_pin_sha256": sha256(PIN),
                "dependency_root": config["dependency_root"], "platform": "win32-amd64",
                "engine": "ordered-signed", "ordered_session": True,
                "gzip": "ISA-L", "isal_level": 2, "date_range": "public_buffer_leases"}
    return kwargs, identity


def fast_runtime_status():
    """Cheap UI status; full byte verification remains mandatory before a run."""
    if sys.platform == "darwin":
        return True, "macOS portable signed profile; native execution and stdlib gzip"
    if not supported_windows():
        return False, "Accepted Windows acceleration is unavailable on this host"
    try:
        config = read_json(CONFIG)
        pin = load_pin()
        if config.get("schema_version") != 1 or config.get("profile") != pin["name"]:
            raise DatasetError("Profile configuration differs")
        verify_blob(PIN, config["profile_pin_sha256"])
        root = checked_path(Path(config["dependency_root"]), directory=True)
        for key in ("compiler", "gzip", "date_range"):
            checked_path(root / pin[key]["folder"], directory=True)
        return True, "Accepted signed profile configured; files are verified before each run"
    except (DatasetError, OSError, ValueError, KeyError):
        return False, "Run scripts/install_fast_runtime.py --data-root <your-Qlib-data-folder>"
