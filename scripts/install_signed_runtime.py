"""Build and install the pinned signed-Qlib derivative for this platform.

Every artifact is checked against docs/qlib-derivative-pin.json: the official
base wheel digest, the exact fork commit, the reproducible derivative wheel
digest and, finally, each installed package file. Network access is used only
when the base wheel or fork checkout is missing. An existing runtime is never
overwritten; ``--reinstall`` moves it aside first.

Run with the project environment, e.g. ``env/qlib/bin/python scripts/install_signed_runtime.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qlib_yahoo.dataset import DatasetError  # noqa: E402
from qlib_yahoo.runtime import PIN, platform_key, platform_pin, verify_signed_install  # noqa: E402

# Shallow history must reach the upstream base commit, which the wheel builder checks; the derivative
# is a few commits above it, and this leaves room for later versions without cloning all of Qlib.
FORK_HISTORY_DEPTH = 32


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(*command: str, cwd: Path | None = None) -> str:
    # Match Python children's pipe encoding to our decoder, even when the caller
    # explicitly inherited a legacy PYTHONIOENCODING (which overrides UTF8 mode).
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", env=env)
    # Non-Python tools (notably Git) can emit legacy bytes in diagnostics. Decode
    # those lossily so a command failure still reaches the DatasetError boundary.
    if result.returncode:
        raise DatasetError(f"{' '.join(command[:3])} … failed: {(result.stderr or result.stdout).strip()[-800:]}")
    return result.stdout.strip()


def base_wheel(pin: dict) -> Path:
    folder = ROOT / "env" / "qlib-signed-base"
    wheel = folder / f"pyqlib-0.9.7-{pin['platform_tag']}.whl"
    if not wheel.is_file():
        folder.mkdir(parents=True, exist_ok=True)
        platform_tag = pin["platform_tag"].split("-", 2)[2]
        print(f"Downloading the official pyqlib 0.9.7 {platform_tag} wheel from PyPI …", flush=True)
        _run(sys.executable, "-m", "pip", "download", "pyqlib==0.9.7", "--no-deps", "--only-binary=:all:",
             "--platform", platform_tag, "--python-version", "3.12", "-d", str(folder))
    if _sha256(wheel) != pin["base_wheel_sha256"]:
        raise DatasetError(f"base wheel digest differs from the pin: {wheel}")
    print(f"Base wheel verified: {wheel.name}", flush=True)
    return wheel


def fork_checkout(pin: dict) -> Path:
    source = ROOT / "env" / "qlib-signed-source"
    if not source.is_dir():
        print(f"Cloning {pin['repository']} ({pin['branch']}) …", flush=True)
        try:
            _run("git", "clone", "--quiet", "--depth", str(FORK_HISTORY_DEPTH), "--branch", pin["branch"],
                 pin["repository"], str(source))
        except DatasetError:
            # The branch may be deleted once its pull request merges; the pinned commit stays fetchable.
            print(f"Branch {pin['branch']} unavailable; fetching the pinned commit directly …", flush=True)
            shutil.rmtree(source, ignore_errors=True)
            _run("git", "init", "--quiet", str(source))
            _run("git", "-C", str(source), "remote", "add", "origin", pin["repository"])
            _run("git", "-C", str(source), "fetch", "--quiet", "--depth", str(FORK_HISTORY_DEPTH), "origin", pin["commit"])
            _run("git", "-C", str(source), "checkout", "--quiet", "--detach", pin["commit"])
    if _run("git", "-C", str(source), "rev-parse", "HEAD") != pin["commit"]:
        _run("git", "-C", str(source), "fetch", "--quiet", "--depth", str(FORK_HISTORY_DEPTH), "origin", pin["commit"])
        _run("git", "-C", str(source), "checkout", "--quiet", "--detach", pin["commit"])
    try:
        _run("git", "-C", str(source), "merge-base", "--is-ancestor", pin["upstream_commit"], "HEAD")
    except DatasetError:  # an older shallow checkout: deepen it rather than failing inside the builder
        _run("git", "-C", str(source), "fetch", "--quiet", "--depth", str(FORK_HISTORY_DEPTH), "origin", pin["commit"])
        _run("git", "-C", str(source), "merge-base", "--is-ancestor", pin["upstream_commit"], "HEAD")
    if _run("git", "-C", str(source), "rev-parse", "HEAD") != pin["commit"]:
        raise DatasetError("fork checkout is not at the pinned commit")
    if _run("git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"):
        raise DatasetError(f"fork checkout has local changes: {source}")
    print(f"Fork verified at {pin['commit'][:12]}", flush=True)
    return source


def derivative_wheel(pin: dict, base: Path, source: Path) -> Path:
    wheel = ROOT / pin["local_wheel"]
    if wheel.is_file():
        if _sha256(wheel) != pin["wheel_sha256"]:
            raise DatasetError(f"existing derivative wheel differs from the pin: {wheel}")
        print(f"Derivative wheel verified: {wheel.name}", flush=True)
        return wheel
    wheel.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=wheel.parent) as temporary:
        _run(sys.executable, str(source / "scripts" / "build_signed_wheel.py"), "--base-wheel", str(base),
             "--base-sha256", pin["base_wheel_sha256"], "--output-dir", temporary)
        built = Path(temporary) / wheel.name
        digest = _sha256(built)
        if digest != pin["wheel_sha256"]:
            raise DatasetError(f"rebuilt wheel {digest} differs from pinned {pin['wheel_sha256']}; "
                               "the build is not reproducible here — do not use it")
        shutil.move(str(built), wheel)
    print(f"Derivative wheel built and verified: {wheel.name}", flush=True)
    return wheel


def install(pin: dict, wheel: Path, reinstall: bool) -> Path:
    runtime = ROOT / pin["local_runtime"]
    if runtime.exists():
        try:
            verify_signed_install(pin, ROOT)
            print(f"Existing runtime verified: {runtime}", flush=True)
            return runtime
        except DatasetError as error:
            if not reinstall:
                raise DatasetError(f"{error} Re-run with --reinstall to move it aside and install fresh.") from error
            aside = runtime.with_name(f"{runtime.name}.old-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
            runtime.rename(aside)
            print(f"Moved the unverified runtime aside: {aside}", flush=True)
    _run(sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--no-compile",
         "--target", str(runtime), str(wheel))
    verify_signed_install(pin, ROOT)
    print(f"Installed and verified: {runtime}", flush=True)
    return runtime


def main() -> int:
    # Setup may capture this script's output through pipes under a legacy code
    # page. Configure only this entrypoint's streams, not the process environment.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reinstall", action="store_true", help="Move an unverified runtime aside and reinstall.")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        print("error: run this with the project's Python 3.12 environment (env/qlib).", file=sys.stderr)
        return 1
    try:
        pin = platform_pin(json.loads(PIN.read_text(encoding="utf-8")))
        print(f"Platform {platform_key()}: pinned build {pin['version']} ({pin['platform_tag']})", flush=True)
        wheel = derivative_wheel(pin, base_wheel(pin), fork_checkout(pin))
        install(pin, wheel, args.reinstall)
    except (DatasetError, OSError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("Signed runtime ready. The console and CLI verify it again before every signed run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
