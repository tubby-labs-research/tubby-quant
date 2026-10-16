"""Install and verify the accepted signed profile: --data-root <Qlib folder>."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import urllib.parse
import zipfile
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

from qlib_yahoo.dataset import DatasetError  # noqa: E402
from qlib_yahoo import fast_runtime as runtime  # noqa: E402


def download_wheel(spec, destination):
    runtime.member_name(spec["filename"])
    url = urllib.parse.urlparse(spec["url"])
    if (url.scheme != "https" or url.hostname != "files.pythonhosted.org" or url.username
            or url.password or url.query or url.fragment or not url.path.startswith("/packages/")):
        raise DatasetError("Runtime wheel URL must be an immutable official PyPI HTTPS artifact")
    runtime.checked_path(destination, missing=True)
    if destination.exists():
        runtime.verify_blob(destination, spec["sha256"], spec["size"])
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading verified runtime dependency: {spec['filename']}", flush=True)
    # A failed download never occupies the canonical path.
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        candidate = Path(temporary) / spec["filename"]
        # Use a standard package transport, with both socket and
        # whole-process deadlines. A trickling response cannot hang setup
        # indefinitely. Direct pinned URL only: no index, dependency resolution,
        # package execution, shared cache or change to the normal environment.
        try:
            command = [
                sys.executable, "-m", "pip", "--isolated", "download", "--no-index", "--no-deps",
                "--only-binary=:all:", "--no-cache-dir", "--disable-pip-version-check",
                "--timeout", "30", "--retries", "2", "--dest", temporary, spec["url"],
            ]
            # Native Windows TLS avoids the observed stalled Python transport.
            # Use only the OS-owned absolute curl path, never a PATH replacement.
            curl = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/curl.exe"
            if sys.platform == "win32" and curl.is_file():
                runtime.checked_path(curl)
                command = [str(curl), "--fail", "--show-error", "--silent", "--location",
                           "--proto", "=https", "--proto-redir", "=https",
                           "--connect-timeout", "20", "--max-time", "600",
                           "--max-filesize", str(spec["size"]), "--output", str(candidate), spec["url"]]
            environment = os.environ.copy()
            environment.update(TMP=temporary, TEMP=temporary, TMPDIR=temporary)
            result = subprocess.run(command, capture_output=True, text=True, timeout=900, env=environment)
        except subprocess.TimeoutExpired as error:
            raise DatasetError(f"Runtime wheel download timed out: {spec['filename']}; retry setup") from error
        if result.returncode:
            raise DatasetError(f"Runtime wheel download failed: {spec['filename']}; check connectivity and retry")
        runtime.wheel_contents(candidate, spec)
        candidate.rename(destination)


def install_tree(destination, content):
    runtime.checked_path(destination, missing=True)
    if destination.exists():
        runtime.verify_tree(destination, content)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        candidate = Path(temporary) / "installed"
        candidate.mkdir()
        for name, raw in content.items():
            target = candidate.joinpath(*runtime.member_name(name))
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(raw)
        runtime.verify_tree(candidate, content)
        candidate.rename(destination)


def native_archive_contents(spec, archive_path):
    """Check the pinned archive and exact member inventory before extraction."""
    expected = {spec["artifact"]: spec["artifact_sha256"],
                "provenance.json": spec["provenance_sha256"]}
    runtime.verify_blob(archive_path, spec["asset_sha256"])
    content = {}
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            runtime.member_name(item.filename)
            mode = item.external_attr >> 16
            if (item.filename not in expected or item.filename in content or item.is_dir()
                    or stat.S_ISLNK(mode) or stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
                raise DatasetError("Unsafe or unexpected native release archive member")
            raw = archive.read(item)
            if hashlib.sha256(raw).hexdigest() != expected[item.filename]:
                raise DatasetError("Native release member digest mismatch")
            content[item.filename] = raw
    if set(content) != set(expected):
        raise DatasetError("Native release archive inventory mismatch")
    return content


def install_native(spec, destination):
    runtime.checked_path(destination, missing=True)
    if "bundled_asset" in spec:
        # Presence opts into a required local bundle, even on repeat setup. Never
        # fall back to private downloads or resolve away links/reparse points.
        name = spec["bundled_asset"]
        if not isinstance(name, str):
            raise DatasetError("Bundled native asset must be a root-relative path")
        archive_path = ROOT.joinpath(*runtime.member_name(name))
        if not archive_path.is_relative_to(ROOT):
            raise DatasetError("Bundled native asset must stay within the project root")
        install_tree(destination, native_archive_contents(spec, archive_path))
        return
    expected = {spec["artifact"]: spec["artifact_sha256"],
                "provenance.json": spec["provenance_sha256"]}
    if destination.exists():
        for name, digest in expected.items():
            runtime.verify_blob(destination / name, digest)
        runtime.verify_tree(destination, {name: (destination / name).read_bytes() for name in expected})
        return
    if (spec["release_repository"] != "tubby-labs-research/Quant"
            or not spec["release_tag"] or spec["release_tag"].startswith("-")
            or len(runtime.member_name(spec["asset"])) != 1
            or any(character in spec["asset"] for character in "*?[]")):
        raise DatasetError("Invalid private native release coordinates")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        try:
            result = subprocess.run(["gh", "release", "download", spec["release_tag"], "--repo",
                                     spec["release_repository"], "--pattern", spec["asset"], "--dir", temporary],
                                    capture_output=True, text=True, timeout=120)
        except FileNotFoundError as error:
            raise DatasetError("GitHub CLI (gh) is required for the private helper. Install gh, sign in with "
                               "your account's Quant read access, and retry setup.") from error
        except subprocess.TimeoutExpired as error:
            raise DatasetError("Private helper download timed out; retry setup. No alternate binary is used.") from error
        if result.returncode:
            raise DatasetError("Private native download failed. Authenticate gh with access to the private "
                               "Quant repository and retry; no alternate binary is used.")
        archive_path = Path(temporary) / spec["asset"]
        install_tree(destination, native_archive_contents(spec, archive_path))


def configure(data_root, *, install_engine=True):
    if (sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12)
            or (sys.platform != "darwin" and not runtime.supported_windows())):
        raise DatasetError("Accepted signed setup requires CPython 3.12 on Windows x64 or macOS")
    data_root = Path(data_root).absolute()
    runtime.checked_path(data_root, missing=True)
    if install_engine:
        result = subprocess.run([sys.executable, str(ROOT / "scripts/install_signed_runtime.py")])
        if result.returncode:
            raise DatasetError("Pinned signed engine installation failed")
    if sys.platform == "darwin":
        print("macOS signed engine ready; portable history/indicators are selected, with native execution "
              "and stdlib gzip. Windows acceleration is not installed.")
        return
    pin = runtime.load_pin()
    root = data_root / "runtime-deps"
    runtime.checked_path(root, missing=True)
    root.mkdir(parents=True, exist_ok=True)
    content = {}
    for spec in pin["compiler"]["wheels"]:
        wheel = root / "wheels" / spec["filename"]
        download_wheel(spec, wheel)
        members = runtime.compiler_contents(runtime.wheel_contents(wheel, spec))
        if {name.casefold() for name in content} & {name.casefold() for name in members}:
            raise DatasetError("Compiler wheels overlap")
        content.update(members)
    install_tree(root / pin["compiler"]["folder"], content)
    gzip_root = root / pin["gzip"]["folder"]
    runtime.checked_path(gzip_root, missing=True)
    if gzip_root.exists():
        # Do not repair an incomplete existing install implicitly.
        if not (gzip_root / "site-packages").is_dir():
            raise DatasetError(f"Incomplete existing gzip install: {gzip_root}; move it aside explicitly")
    else:
        # Stage wheel and extraction together so an interrupted download cannot poison the install.
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            candidate = Path(temporary) / "gzip"
            candidate.mkdir()
            members = {}
            for spec in pin["gzip"]["wheels"]:
                download_wheel(spec, candidate / spec["filename"])
                members.update(runtime.wheel_contents(candidate / spec["filename"], spec))
            install_tree(candidate / "site-packages", members)
            candidate.rename(gzip_root)
    install_native(pin["date_range"], root / pin["date_range"]["folder"])
    runtime.verify_dependencies(pin, root)
    config = {"schema_version": 1, "profile": pin["name"], "dependency_root": str(root),
              "profile_pin_sha256": runtime.sha256(runtime.PIN)}
    runtime.checked_path(runtime.CONFIG, missing=True)
    runtime.CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=runtime.CONFIG.parent) as temporary:
        candidate = Path(temporary) / "config.json"
        candidate.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        os.replace(candidate, runtime.CONFIG)
    print(f"Accepted signed runtime installed, verified and configured: {root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--skip-engine", action="store_true", help="Verify/install acceleration only; engine already installed.")
    args = parser.parse_args()
    try:
        configure(args.data_root, install_engine=not args.skip_engine)
    except (DatasetError, OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
