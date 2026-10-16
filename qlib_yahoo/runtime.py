"""Explicit, verified signed-engine activation; never patch site-packages."""

from __future__ import annotations

import hashlib
import importlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

from qlib_yahoo.dataset import DatasetError

ROOT = Path(__file__).resolve().parents[1]
PIN = ROOT / "docs" / "qlib-derivative-pin.json"
_RESULT = "__QUANT_SIGNED_RUN__="


class _SourceOnlyLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        # -B only prevents writes: it does not stop loading existing bytecode.
        # Compile verified source instead of accepting timestamp-valid .pyc.
        return self.source_to_code(self.get_data(self.path), self.path)


class _SignedSourceFinder(importlib.abc.MetaPathFinder):
    def __init__(self, runtime):
        self.package = (runtime / "qlib").resolve()

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "qlib" and not fullname.startswith("qlib."):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None:
            return None
        if not spec.origin or not Path(spec.origin).resolve().is_relative_to(self.package):
            raise ImportError(f"Signed Qlib module is outside the verified package: {fullname}")
        if isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            spec.loader = _SourceOnlyLoader(fullname, spec.origin)
        elif isinstance(spec.loader, importlib.machinery.SourcelessFileLoader):
            raise ImportError(f"Unreviewed source-less Qlib bytecode: {fullname}")
        return spec


def _local_path(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise DatasetError("Signed runtime pin must use repository-relative paths")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise DatasetError("Signed runtime path escapes the checkout")
    return path


def verify_signed_install(pin: dict, root: Path = ROOT) -> tuple[Path, dict]:
    """Compare installed package bytes against the independently pinned wheel.

    Checking a version or a self-reported commit alone would accept modified code.
    The wheel digest anchors both its embedded provenance and installed files.
    """
    try:
        wheel = _local_path(root, pin["local_wheel"])
        runtime = _local_path(root, pin["local_runtime"])
        if hashlib.sha256(wheel.read_bytes()).hexdigest() != pin["wheel_sha256"]:
            raise DatasetError("Signed Qlib wheel hash differs from the reviewed pin")
        with zipfile.ZipFile(wheel) as archive:
            names = {name for name in archive.namelist() if name.startswith("qlib/") and not name.endswith("/")}
            if "qlib/_tubby_build.json" not in names:
                raise DatasetError("Signed wheel lacks build provenance")
            build = json.loads(archive.read("qlib/_tubby_build.json"))
            if (build.get("commit") != pin["commit"]
                    or build.get("upstream_commit") != pin["upstream_commit"]
                    or build.get("base_wheel_sha256") != pin["base_wheel_sha256"]
                    or build.get("dirty") is not False):
                raise DatasetError("Signed Qlib build identity differs from the reviewed clean build")
            for name in names:
                installed = _local_path(runtime, name)
                if not installed.is_file() or installed.read_bytes() != archive.read(name):
                    raise DatasetError(f"Signed Qlib installed file differs from its pinned wheel: {name}")
            for installed in (runtime / "qlib").rglob("*"):
                if installed.is_file() and installed.suffix in (".py", ".pyd", ".so"):
                    if installed.relative_to(runtime).as_posix() not in names:
                        raise DatasetError("Signed runtime contains an unreviewed executable package file")
            # The runtime root goes first on sys.path: any other importable entry there would shadow env/qlib.
            modules = (".py", ".pyc", ".pyd", ".so")
            for entry in runtime.iterdir():
                importable = (entry.suffix in modules if entry.is_file()
                              else any((entry / f"__init__{suffix}").exists() for suffix in modules))
                if importable and entry.name != "qlib":
                    raise DatasetError(f"Signed runtime contains an unreviewed importable entry: {entry.name}")
        return runtime, build
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise DatasetError(f"Signed Qlib runtime unavailable or invalid: {exc}. See docs/QLIB_SETUP.md.") from exc


def platform_key() -> str:
    machine = platform.machine().lower()
    if sys.platform == "win32":
        return "win32-amd64" if machine in ("amd64", "x86_64") else f"win32-{machine}"
    if sys.platform == "darwin":
        return "darwin-universal2"  # the pinned macOS wheel is universal2 (arm64 and x86_64)
    return f"{sys.platform}-{machine}"


def platform_pin(pin: dict, key: str | None = None) -> dict:
    """Select the reviewed build for this platform.

    The top-level fields are the original Windows x64 pin; other platforms add a
    reviewed entry under ``platforms`` that overrides only build-specific fields.
    """
    key = key or platform_key()
    if key == "win32-amd64":
        return pin
    entry = (pin.get("platforms") or {}).get(key)
    if not isinstance(entry, dict):
        raise DatasetError(f"No reviewed signed-engine build is pinned for {key}. See docs/QLIB_SETUP.md.")
    return {**pin, **entry}


def signed_runtime_status() -> tuple[bool, str]:
    """Cheap availability check for the UI; the full byte-level verification runs per signed run."""
    if sys.version_info[:2] != (3, 12):
        return False, "requires Python 3.12"
    try:
        pin = platform_pin(json.loads(PIN.read_text(encoding="utf-8")))
        wheel = _local_path(ROOT, pin["local_wheel"])
        runtime = _local_path(ROOT, pin["local_runtime"])
    except (OSError, ValueError, KeyError, DatasetError) as exc:
        return False, str(exc)
    if not wheel.is_file() or not (runtime / "qlib" / "__init__.py").is_file():
        hint = ("scripts/macos/setup.sh --signed" if sys.platform == "darwin" else
                "env/qlib/Scripts/python scripts/install_fast_runtime.py --data-root <data folder>")
        return False, f"pinned build {pin['version']} is not installed ({hint})"
    return True, f"pinned build {pin['version']} ({pin['platform_tag']})"


def ensure_signed_runtime(*, pin_path: Path | None = None) -> dict:
    """Activate the reviewed build before anything imports Qlib in this process."""
    if sys.version_info[:2] != (3, 12):
        raise DatasetError("The signed runtime pin requires Python 3.12; use env/qlib. See docs/QLIB_SETUP.md.")
    selected_pin = PIN
    if pin_path is not None:
        selected_pin = Path(pin_path).absolute()
        prefix = "qlib-derivative-pin."
        if (selected_pin.is_symlink() or selected_pin.parent.resolve() != (ROOT / "docs").resolve()
                or not selected_pin.name.startswith(prefix) or selected_pin.suffix != ".json"
                or not selected_pin.name[len(prefix):-5]):
            raise DatasetError("Experimental signed pin must be an explicit versioned JSON in repository docs")
    try:
        pin = json.loads(selected_pin.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DatasetError(f"Cannot read signed-engine pin: {exc}") from exc
    pin = platform_pin(pin)
    runtime, build = verify_signed_install(pin)
    expected = runtime / "qlib" / "__init__.py"
    loaded = sys.modules.get("qlib")
    if loaded is not None and Path(loaded.__file__).resolve() != expected:
        raise DatasetError("Another Qlib is already imported. Start a fresh signed-mode process (the console does this automatically).")
    if loaded is not None and not any(isinstance(finder, _SignedSourceFinder)
                                     and finder.package == expected.parent for finder in sys.meta_path):
        raise DatasetError("Qlib was imported before source-only verification. Start a fresh signed-mode process.")
    if loaded is None:
        sys.path.insert(0, str(runtime))
        sys.meta_path.insert(0, _SignedSourceFinder(runtime))
        importlib.invalidate_caches()
    qlib = importlib.import_module("qlib")
    if Path(qlib.__file__).resolve() != expected or qlib.__version__ != pin["version"]:
        raise DatasetError("Imported Qlib identity does not match the signed-engine pin")
    return {
        "repository": pin["repository"], "commit": build["commit"],
        "upstream_commit": build["upstream_commit"], "version": qlib.__version__,
        "wheel_sha256": pin["wheel_sha256"], "base_wheel_sha256": build["base_wheel_sha256"],
        "build_method": pin["build_method"], "installed_package_verified": True,
        "build_dirty": False, "platform": platform_key(), "platform_tag": pin.get("platform_tag"),
    }


def run_signed_backtest(dataset: Path, alpha: str, **settings) -> Path:
    """Console boundary: keep signed and legacy Qlib imports in separate processes."""
    executable = Path(sys.executable)
    if executable.name.lower() == "pythonw.exe":
        executable = executable.with_name("python.exe")
    request = {"dataset": str(dataset), "alpha": alpha, **settings, "strategy_mode": "signed"}
    # Request values are JSON data, never interpolated into Python source.
    program = (
        "import json,sys; from pathlib import Path; "
        "from qlib_yahoo.backtest import run_backtest; "
        "r=json.loads(sys.argv[1]); r['dataset']=Path(r['dataset']); "
        "p=run_backtest(**r); print('__QUANT_SIGNED_RUN__='+json.dumps(str(p)))"
    )
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        [str(executable), "-c", program, json.dumps(request, allow_nan=False)],
        cwd=ROOT, env=environment, capture_output=True, text=True, encoding="utf-8", errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    # Keep the UI log bounded; full accounting and errors are in the saved run.
    for output in (completed.stdout, completed.stderr):
        if output:
            message = output[-12000:]
            encoding = getattr(sys.stdout, "encoding", None)
            if encoding:
                message = message.encode(encoding, errors="replace").decode(encoding)
            print(message)
    if completed.returncode:
        tail = (completed.stderr or completed.stdout).strip()[-2000:]
        raise DatasetError(f"Signed backtest failed: {tail}")
    markers = [line[len(_RESULT):] for line in completed.stdout.splitlines() if line.startswith(_RESULT)]
    if len(markers) != 1:
        raise DatasetError("Signed worker did not return a unique completed run path")
    return Path(json.loads(markers[0]))
