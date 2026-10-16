"""Immutable ZIP_STORED Qlib feature snapshots; no Qlib dependency at import time."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import stat
import struct
import zipfile
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from qlib_yahoo.dataset import DatasetError, is_os_metadata


FORMAT = "qlib-zip-store-v1"
ARCHIVE = "qlib_data.zip"
_verified: tuple[Path, str, Mapping[str, bytes]] | None = None


def _reject_redirects(path: Path) -> None:
    """Reject links and Windows junctions in the path or any existing parent."""
    absolute = Path(path).absolute()
    for component in (absolute, *absolute.parents):
        if component.is_symlink() or component.is_junction():
            raise DatasetError(f"redirected dataset path: {component}")


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_stream(stream) -> str:
    stream.seek(0)
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _hash_file(path: Path) -> str:
    with path.open("rb") as stream:
        return _hash_stream(stream)


def _expected(manifest: dict) -> tuple[str, int]:
    if not isinstance(manifest, dict) or manifest.get("status") != "ready":
        raise DatasetError("packed source must have a ready manifest")
    digest, count = manifest.get("qlib_binary_tree_sha256"), manifest.get("qlib_binary_file_count")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise DatasetError("packed source requires a recorded binary tree SHA-256")
    if type(count) is not int or count < 1:
        raise DatasetError("packed source requires a recorded binary file count")
    return digest, count


def _name(name: str) -> str:
    if (not name or "\\" in name or name.startswith("/") or ":" in name
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise DatasetError(f"unsafe Qlib archive entry: {name!r}")
    return name


def _native(name: str) -> bool:
    return name.startswith("calendars/") or name.startswith("instruments/")


def _tree_update(digest, name: str, content_digest: bytes) -> None:
    encoded = name.encode("utf-8")
    digest.update(len(encoded).to_bytes(4, "big"))
    digest.update(encoded)
    digest.update(content_digest)


def _source_files(root: Path) -> list[tuple[str, Path]]:
    _reject_redirects(root)
    if not root.is_dir():
        raise DatasetError("source Qlib data directory is missing or redirected")
    result = []
    for path in root.rglob("*"):
        _reject_redirects(path)
        if path.is_file() and not is_os_metadata(path):
            result.append((_name(path.relative_to(root).as_posix()), path))
    return sorted(result, key=lambda item: item[1])


def pack_dataset(source: Path, destination: Path) -> Path:
    """Create a new sealed packed snapshot without modifying source or latest.json."""
    source, destination = Path(source), Path(destination)
    _reject_redirects(source)
    _reject_redirects(destination)
    if not source.is_dir():
        raise DatasetError("source snapshot is missing or redirected")
    src, dst = source.resolve(), destination.resolve()
    if destination.exists() or dst == src or src in dst.parents:
        raise DatasetError("destination must be new and outside source")
    # A source nested inside the destination would disappear into its own output.
    if dst in src.parents:
        raise DatasetError("destination cannot contain source")
    manifest_path = source / "manifest.json"
    _reject_redirects(manifest_path)
    original = manifest_path.read_bytes()
    manifest = json.loads(original)
    expected, expected_count = _expected(manifest)
    if "storage" in manifest:
        raise DatasetError("source is already a storage variant")
    files = _source_files(source / "qlib_data")
    if not files:
        raise DatasetError("source has no Qlib files")
    destination.mkdir(parents=True, exist_ok=False)
    try:
        digest = hashlib.sha256()
        native: dict[str, bytes] = {}
        with zipfile.ZipFile(destination / ARCHIVE, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for number, (name, path) in enumerate(files, 1):
                info = zipfile.ZipInfo(name)
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o100644 << 16
                per_file = hashlib.sha256()
                with path.open("rb") as inp, archive.open(info, "w") as out:
                    chunks = [] if _native(name) else None
                    for chunk in iter(lambda: inp.read(1024 * 1024), b""):
                        per_file.update(chunk)
                        out.write(chunk)
                        if chunks is not None:
                            chunks.append(chunk)
                _tree_update(digest, name, per_file.digest())
                if chunks is not None:
                    native[name] = b"".join(chunks)
                if number % 2000 == 0 or number == len(files):
                    print(f"Packed {number}/{len(files)} Qlib files", flush=True)
        if digest.hexdigest() != expected or len(files) != expected_count:
            raise DatasetError("source Qlib tree differs from sealed manifest")
        for name, data in native.items():
            target = destination / "qlib_data" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        evidence = source / "universe"
        if evidence.exists():
            for name, path in _source_files(evidence):
                target = destination / "universe" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with path.open("rb") as inp, target.open("wb") as out:
                    shutil.copyfileobj(inp, out, length=1024 * 1024)
        for record in manifest.get("universes", {}).values():
            for file_key, hash_key in (("membership_csv", "membership_sha256"),
                                       ("rankings_csv", "rankings_sha256")):
                if file_key in record:
                    relative = _name(record[file_key])
                    if not relative.startswith("universe/"):
                        raise DatasetError("unsafe universe evidence path")
                    copied = destination / relative
                    if not copied.is_file() or _hash(copied.read_bytes()) != record.get(hash_key):
                        raise DatasetError("copied universe evidence differs from manifest")
        if manifest_path.read_bytes() != original:
            raise DatasetError("source manifest changed during packing")
        packed = copy.deepcopy(manifest)
        archive_path = destination / ARCHIVE
        packed["storage"] = {"format": FORMAT, "archive": ARCHIVE,
                             "archive_sha256": _hash_file(archive_path),
                             "archive_size": archive_path.stat().st_size,
                             "source_manifest_sha256": _hash(original)}
        # The archive itself is validated before declaring the destination ready.
        provisional = copy.deepcopy(packed)
        provisional["status"] = "building"
        (destination / "manifest.json").write_text(json.dumps(provisional, indent=2) + "\n", encoding="utf-8")
        _verify_archive(destination, packed)
        if manifest_path.read_bytes() != original:
            raise DatasetError("source manifest changed during packing")
        (destination / "manifest.json").write_text(json.dumps(packed, indent=2) + "\n", encoding="utf-8")
        return destination
    except BaseException:
        # A partial destination is intentionally retained for diagnosis, never ready.
        raise


def _verify_archive(dataset: Path, manifest: dict, *, reusable: Mapping[str, bytes] | None = None) -> Mapping[str, bytes]:
    expected, expected_count = _expected(manifest)
    storage = manifest.get("storage")
    if (not isinstance(storage, dict) or storage.get("format") != FORMAT
            or storage.get("archive") != ARCHIVE):
        raise DatasetError("invalid packed storage record")
    path = dataset / ARCHIVE
    _reject_redirects(path)
    if not path.is_file():
        raise DatasetError("packed archive is missing or redirected")
    source_identity = storage.get("source_manifest_sha256")
    if (not isinstance(source_identity, str) or len(source_identity) != 64
            or any(c not in "0123456789abcdef" for c in source_identity)):
        raise DatasetError("missing original manifest identity")
    entries = {} if reusable is None else reusable
    seen = set()
    spans = []
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            archive_size = stream.tell()
            if type(storage.get("archive_size")) is not int or archive_size != storage["archive_size"]:
                raise DatasetError("packed archive size differs")
            if _hash_stream(stream) != storage.get("archive_sha256"):
                raise DatasetError("packed archive SHA-256 differs")
            with zipfile.ZipFile(stream) as archive:
                for info in archive.infolist():
                    name = _name(info.filename)
                    mode = info.external_attr >> 16
                    if (name in seen or info.is_dir() or info.compress_type != zipfile.ZIP_STORED
                            or info.flag_bits != 0 or stat.S_ISLNK(mode)
                            or (stat.S_IFMT(mode) not in (0, stat.S_IFREG))):
                        raise DatasetError(f"invalid packed entry: {name}")
                    offset = info.header_offset
                    if offset < 0 or offset + 30 > archive_size:
                        raise DatasetError("invalid ZIP local offset")
                    stream.seek(offset)
                    local = stream.read(30)
                    if len(local) != 30 or local[:4] != b"PK\x03\x04":
                        raise DatasetError("invalid ZIP local offset")
                    _, _, flags, method, _, _, crc, compressed, size, name_len, extra_len = struct.unpack(
                        "<IHHHHHIIIHH", local)
                    begin = offset + 30 + name_len + extra_len
                    end = begin + info.compress_size
                    local_name = stream.read(name_len)
                    if (flags or method != 0 or compressed != info.compress_size or size != info.file_size
                            or crc != info.CRC or end > archive_size or len(local_name) != name_len
                            or local_name != name.encode("utf-8")):
                        raise DatasetError("ZIP local header differs from directory")
                    spans.append((offset, end))
                    seen.add(name)
                    if reusable is None:
                        entries[name] = archive.read(info)
                    else:
                        if name not in reusable or len(reusable[name]) != info.file_size:
                            raise DatasetError("packed immutable entry inventory differs")
                        # ZipExtFile checks CRC at EOF. Stream the actual payload again,
                        # retaining only a bounded chunk alongside the provider's bytes.
                        with archive.open(info) as payload:
                            while payload.read(1024 * 1024):
                                pass
                if len(seen) != expected_count or seen != set(entries):
                    raise DatasetError("packed entry count differs")
            # A file changed during entry reads must not inherit the first hash's identity.
            stream.seek(0, 2)
            if stream.tell() != archive_size or _hash_stream(stream) != storage["archive_sha256"]:
                raise DatasetError("packed archive changed during verification")
        spans.sort()
        if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
            raise DatasetError("overlapping ZIP entry offsets")
    except (OSError, ValueError, struct.error, zipfile.BadZipFile, EOFError, UnicodeError) as error:
        raise DatasetError(f"malformed packed archive: {error}") from error
    digest = hashlib.sha256()
    for name, data in sorted(entries.items(), key=lambda item: Path(item[0])):
        _tree_update(digest, name, hashlib.sha256(data).digest())
    if digest.hexdigest() != expected:
        raise DatasetError("packed logical tree differs from manifest")
    native = {name: data for name, data in entries.items() if _native(name)}
    extracted = _source_files(dataset / "qlib_data")
    if {name for name, _ in extracted} != set(native):
        raise DatasetError("extracted Qlib native file set differs")
    for name, path in extracted:
        if path.read_bytes() != native[name]:
            raise DatasetError(f"extracted Qlib native file differs: {name}")
    return entries


def _verify_universe_evidence(dataset: Path, manifest: dict) -> None:
    universes = manifest.get("universes", {})
    if not isinstance(universes, dict):
        raise DatasetError("invalid universe records")
    for record in universes.values():
        if not isinstance(record, dict):
            raise DatasetError("invalid universe record")
        for file_key, hash_key in (("membership_csv", "membership_sha256"),
                                   ("rankings_csv", "rankings_sha256")):
            if file_key not in record:
                continue
            relative = record[file_key]
            if not isinstance(relative, str):
                raise DatasetError("unsafe universe evidence path")
            relative = _name(relative)
            if not relative.startswith("universe/"):
                raise DatasetError("unsafe universe evidence path")
            path = dataset / relative
            _reject_redirects(path)
            if not path.is_file() or _hash_file(path) != record.get(hash_key):
                raise DatasetError(f"universe evidence differs from manifest: {relative}")


def verify_packed(dataset: Path, manifest: dict) -> dict:
    """Verify actual archive bytes on every call and retain one exact verified snapshot."""
    global _verified
    previous = _verified
    _verified = None
    _reject_redirects(dataset)
    dataset = Path(dataset).resolve()
    storage = manifest.get("storage") if isinstance(manifest, dict) else None
    identity = storage.get("archive_sha256") if isinstance(storage, dict) else None
    reusable = previous[2] if previous is not None and previous[:2] == (dataset, identity) else None
    entries = _verify_archive(dataset, manifest, reusable=reusable)
    _verify_universe_evidence(dataset, manifest)
    # Never publish the builder dict itself, even through a read-only proxy:
    # copy its small index so no retained writable alias can alter the provider.
    published = reusable if reusable is not None else MappingProxyType(dict(entries))
    _verified = (dataset, identity, published)
    return {"mode": FORMAT, "qlib_binary_tree_sha256": manifest["qlib_binary_tree_sha256"],
            "qlib_binary_file_count": len(entries)}


def verified_entries(dataset: Path, archive_sha256: str) -> Mapping[str, bytes]:
    """Only hand a provider the bytes validated by the latest verification."""
    _reject_redirects(dataset)
    if _verified is None or _verified[0] != Path(dataset).resolve() or _verified[1] != archive_sha256:
        raise DatasetError("packed dataset has not been verified in this process")
    return _verified[2]


def feature_provider_config(dataset: Path, manifest: dict) -> dict:
    """Small qlib.init(feature_provider=...) config after verify_packed."""
    storage = manifest.get("storage", {})
    identity = storage.get("archive_sha256")
    verified_entries(dataset, identity)
    return {"class": "PackedFeatureProvider", "module_path": "qlib_yahoo.packed_provider",
            "kwargs": {"dataset": str(Path(dataset).resolve()), "archive_sha256": identity}}
