"""Explicit private CPython 3.12 Windows build; never downloads or changes Qlib.

Run --preflight-only BEFORE acquiring Cython. A later authorised build requires
an existing released binary wheel and its separately reviewed SHA256. Extraction,
generated C, objects and extension stay in a NEW D:\\Qlib workspace. Normal Cython
may use private CPython internals: audit metadata is not a public-API certificate.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import sysconfig
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "qlib_yahoo/session_loops.pyx"
REVIEWED = ROOT / "qlib_yahoo/fast_signed.py"
CYTHON_VERSION = "3.1.8"
REVIEWED_SHA256 = "5abd4863c48ea38b970790f8d48f6fd59e5c6eb451389baf721eab0cef724c42"
FLAGS = ["/MD", "/fp:strict"]
MACROS = [("CYTHON_USE_DICT_VERSIONS", "0")]
DIRECTIVES = {"language_level": 3, "infer_types": False, "annotation_typing": False,
              "binding": True, "boundscheck": True, "wraparound": True, "cdivision": False}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_path(path, *, kind=None):
    path = Path(path).absolute()
    for parent in (path, *path.parents):
        if parent.is_symlink() or (hasattr(parent, "is_junction") and parent.is_junction()):
            raise ValueError(f"Redirected path is forbidden: {parent}")
        if parent.exists() and getattr(parent.stat(), "st_file_attributes", 0) & 0x400:
            raise ValueError(f"Reparse point is forbidden: {parent}")
    if kind == "file" and not path.is_file():
        raise ValueError(f"Missing regular file: {path}")
    if kind == "directory" and not path.is_dir():
        raise ValueError(f"Missing directory: {path}")
    return path


def compiler_preflight():
    """No build/import/download; establish the actual local ABI prerequisites."""
    if sys.platform != "win32" or sys.version_info[:2] != (3, 12) or sysconfig.get_platform() != "win-amd64":
        raise RuntimeError("Builder supports CPython 3.12 Windows x64 only; use Python fallback elsewhere")
    vswhere = checked_path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe", kind="file")
    discovery = discover_tools(vswhere)
    selected = discovery["selected_instance"]
    found = selected["installationPath"]
    installation = checked_path(found, kind="directory")
    version_file = checked_path(installation / "VC/Auxiliary/Build/Microsoft.VCToolsVersion.default.txt", kind="file")
    version = version_file.read_text(encoding="utf-8-sig").strip()
    if not re.fullmatch(r"14\.\d+\.\d+", version):
        raise RuntimeError("Unrecognised installed MSVC version")
    binaries = installation / "VC/Tools/MSVC" / version / "bin/Hostx64/x64"
    compiler = checked_path(binaries / "cl.exe", kind="file")
    linker = checked_path(binaries / "link.exe", kind="file")
    dumpbin = checked_path(binaries / "dumpbin.exe", kind="file")
    result = subprocess.run([str(compiler)], capture_output=True)
    banner = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    if not re.search(r"19\.\d+\.\d+", banner):
        raise RuntimeError("Compiler version banner not recognised")
    result = subprocess.run([str(linker), "/?"], capture_output=True)
    linker_banner = (result.stdout + result.stderr).decode("utf-8", errors="replace").splitlines()[:4]
    if not re.search(r"14\.\d+\.\d+", "\n".join(linker_banner)):
        raise RuntimeError("Linker version banner not recognised")
    include = checked_path(sysconfig.get_path("include"), kind="directory")
    header = checked_path(include / "Python.h", kind="file")
    import_lib = checked_path(Path(sys.base_prefix) / "libs/python312.lib", kind="file")
    sdk_root = checked_path(r"C:\Program Files (x86)\Windows Kits\10", kind="directory")
    sdks = sorted((p.name for p in (sdk_root / "Include").iterdir()
                   if p.is_dir() and re.fullmatch(r"10\.\d+\.\d+\.\d+", p.name)),
                  key=lambda value: tuple(map(int, value.split("."))))
    if not sdks:
        raise RuntimeError("Windows SDK unavailable")
    sdk_version = sdks[-1]
    sdk_files = [checked_path(sdk_root / f"Include/{sdk_version}/ucrt/stdlib.h", kind="file"),
                 checked_path(sdk_root / f"Include/{sdk_version}/ucrt/stdio.h", kind="file"),
                 checked_path(sdk_root / f"Include/{sdk_version}/um/Windows.h", kind="file"),
                 checked_path(sdk_root / f"Lib/{sdk_version}/ucrt/x64/ucrt.lib", kind="file"),
                 checked_path(sdk_root / f"Lib/{sdk_version}/um/x64/kernel32.lib", kind="file")]
    vc_root = installation / "VC/Tools/MSVC" / version
    vc_files = [checked_path(vc_root / "include/vcruntime.h", kind="file"),
                checked_path(vc_root / "lib/x64/vcruntime.lib", kind="file"),
                checked_path(vc_root / "lib/x64/msvcrt.lib", kind="file")]
    vcvars = checked_path(installation / "VC/Auxiliary/Build/vcvars64.bat", kind="file")
    return {"python": sys.version, "executable": sys.executable, "platform": sysconfig.get_platform(),
            "extension_suffix": sysconfig.get_config_var("EXT_SUFFIX"), "installation": str(installation),
            "msvc_tools_version": version, "compiler_banner": banner, "compiler": str(compiler),
            "linker_banner": linker_banner, "discovery": discovery,
            "installation_healthy": selected.get("isComplete") is True and selected.get("isLaunchable") is True,
            "linker": str(linker), "dumpbin": str(dumpbin), "include": str(include),
            "python_header_sha256": sha256(header), "python_import_library": str(import_lib),
            "python_import_library_sha256": sha256(import_lib),
            "windows_sdk_version": sdk_version,
            "sdk_file_sha256": {str(p): sha256(p) for p in sdk_files},
            "vc_file_sha256": {str(p): sha256(p) for p in vc_files},
            "vcvars64": str(vcvars), "vcvars64_sha256": sha256(vcvars),
            "tool_sha256": {str(p): sha256(p) for p in (compiler, linker, dumpbin)}}


def discover_tools(vswhere):
    """Preserve default-query failure and partial installer state, never hide it."""
    ordinary = [str(vswhere), "-latest", "-products", "*", "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"]
    first = subprocess.run(ordinary, capture_output=True, text=True, check=True)
    inventory = [str(vswhere), "-all", "-sort", "-products", "*", "-requires",
                 "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-format", "json", "-include", "packages", "-utf8"]
    result = subprocess.run(inventory, capture_output=True, encoding="utf-8-sig", check=True)
    instances = json.loads(result.stdout)
    if not instances or not isinstance(instances, list):
        raise RuntimeError("Reviewed x64 MSVC tools are not installed")
    complete = [i for i in instances if i.get("isComplete") is True and i.get("isLaunchable") is True]
    selected = (complete or instances)[0]
    if not selected.get("installationPath"):
        raise RuntimeError("Visual Studio inventory lacks installation path")
    return {"default_query": ordinary, "default_query_stdout": first.stdout,
            "default_query_stderr": first.stderr, "default_query_returncode": first.returncode,
            "all_query": inventory, "all_instances": instances, "selected_instance": selected,
            "warning": None if complete else "Only incomplete/unlaunchable installation; explicit prototype acknowledgement required"}


def build_environment(identity):
    """Use verified existing VC scripts in a child; never repair installer state."""
    script = identity["vcvars64"]
    if any(c in script for c in '&%!"\r\n') or sha256(script) != identity["vcvars64_sha256"]:
        raise RuntimeError("Unsafe or changed compiler environment script")
    command = f'call "{script}" >nul && set'
    command_processor = checked_path(r"C:\Windows\System32\cmd.exe", kind="file")
    # Pass the raw Windows command line: list2cmdline would insert backslashes
    # before the batch-path quotes, which cmd.exe does NOT decode like a C argv.
    invocation = f'"{command_processor}" /d /s /c {command}'
    environment = {name.upper(): value for name, value in os.environ.items()}
    # Some host launchers supply both Path and PATH; normalise before cmd too,
    # otherwise `set` can emit both and overwrite the newly selected VC path.
    initialized = subprocess.run(invocation, env=environment, capture_output=True, text=True, check=True)
    for line in initialized.stdout.splitlines():
        if "=" in line and not line.startswith("="):
            name, value = line.split("=", 1)
            environment[name.upper()] = value
    environment.update(DISTUTILS_USE_SDK="1", MSSDK="1")
    # Do not persist the complete environment: it can contain unrelated secrets.
    compiler = Path(identity["compiler"])
    search = environment.get("PATH", environment.get("Path", "")).split(os.pathsep)
    if not any(Path(p).absolute() == compiler.parent for p in search if p):
        raise RuntimeError("VC environment does not select the verified x64 compiler")
    if (environment.get("VCTOOLSVERSION", "").rstrip("\\/") != identity["msvc_tools_version"]
            or environment.get("WINDOWSSDKVERSION", "").rstrip("\\/") != identity["windows_sdk_version"]
            or not environment.get("INCLUDE") or not environment.get("LIB")):
        raise RuntimeError("VC environment version/header/library selection differs from preflight")
    return environment


def verify_wheel(wheel, expected_sha256):
    wheel = checked_path(wheel, kind="file")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256) or sha256(wheel) != expected_sha256.lower():
        raise ValueError("Cython wheel SHA256 mismatch or malformed reviewed digest")
    if wheel.name.lower() != f"cython-{CYTHON_VERSION}-cp312-cp312-win_amd64.whl":
        raise ValueError("Requires the explicit released Cython 3.1.8 cp312 Windows x64 binary wheel")
    with zipfile.ZipFile(wheel) as archive:
        for entry in archive.infolist():
            name = entry.filename.replace("\\", "/")
            if name.startswith("/") or ":" in name or ".." in name.split("/") or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe wheel member")
        metadata = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise ValueError("Ambiguous wheel metadata")
        content = archive.read(metadata[0]).decode("utf-8")
        if not re.search(r"^Name: Cython\s*$", content, re.M | re.I) or not re.search(
                rf"^Version: {re.escape(CYTHON_VERSION)}\s*$", content, re.M):
            raise ValueError("Wheel declared distribution/version mismatch")
    return wheel


BUILD_PROGRAM = r'''
import sys
sys.path.insert(0, sys.argv[1])
import Cython
if Cython.__version__ != '3.1.8': raise RuntimeError('Cython version mismatch')
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
class StrictBuild(build_ext):
    def build_extensions(self):
        # Keep actual compile command in captured output; disable inherited LTCG.
        if self.compiler.compiler_type != 'msvc': raise RuntimeError('Reviewed MSVC compiler required')
        if not self.compiler.initialized: self.compiler.initialize()
        for name in ('compile_options', 'compile_options_debug'):
            options = getattr(self.compiler, name, None)
            if options is not None:
                options[:] = [o for o in options if o.lower() not in ('/gl','/fp:fast','/fp:contract')]
        for options in self.compiler._ldflags.values():
            options[:] = [o for o in options if not o.lower().startswith('/ltcg')]
        super().build_extensions()
extension = Extension('qlib_yahoo.session_loops', [sys.argv[2]],
    define_macros=[('CYTHON_USE_DICT_VERSIONS','0')], extra_compile_args=['/MD','/fp:strict'])
extensions = cythonize([extension], build_dir=sys.argv[3], force=True,
    compiler_directives={'language_level':3,'infer_types':False,'annotation_typing':False,
                        'binding':True,'boundscheck':True,'wraparound':True,'cdivision':False})
setup(name='private-qlib-session-loops', ext_modules=extensions,
      cmdclass={'build_ext':StrictBuild},
      script_args=['build_ext','--build-lib',sys.argv[4],'--build-temp',sys.argv[5]])
'''


def build(workspace, manifest, wheel, wheel_sha256, *, allow_incomplete_toolchain=False):
    if type(allow_incomplete_toolchain) is not bool:
        raise TypeError("allow_incomplete_toolchain must be an explicit boolean")
    workspace = checked_path(workspace)
    manifest = checked_path(manifest)
    data_root = checked_path(r"D:\Qlib", kind="directory")
    if not workspace.is_relative_to(data_root) or not manifest.is_relative_to(data_root):
        raise ValueError("Build outputs must be private D:\\Qlib paths")
    checked_path(workspace.parent, kind="directory")
    checked_path(manifest.parent, kind="directory")
    if workspace.exists() or manifest.exists() or manifest.is_relative_to(workspace):
        raise ValueError("Use a new build workspace and separate new manifest")
    record = {"schema_version": 1, "kind": "private_sequential_cython_build", "status": "running",
              "stage": "compiler_preflight", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "experimental": True, "acceptance_status": "not_runtime_evidence", "goal_achieved": False,
              "public_api_only_certified": False, "crosshost_policy": "separate verified build or original Python fallback",
              "compiler_flags": FLAGS.copy(), "macros": [list(pair) for pair in MACROS], "directives": DIRECTIVES.copy(),
              "cython_version": CYTHON_VERSION, "workspace": str(workspace)}
    record["numpy_c_api_used"] = False
    record["allow_incomplete_toolchain"] = allow_incomplete_toolchain
    with manifest.open("x", encoding="utf-8") as handle:
        def persist():
            encoded = json.dumps(record, indent=2) + "\n"
            handle.seek(0); handle.write(encoded); handle.truncate(); handle.flush(); os.fsync(handle.fileno())
        persist()
        try:
            record["builder"] = compiler_preflight()  # Gate before touching compiler wheel.
            record["build_dependency_versions"] = {name: importlib.metadata.version(name)
                                                   for name in ("setuptools", "wheel", "numpy", "pandas")}
            record["stage"] = "installation_state_acknowledgement"
            if not record["builder"]["installation_healthy"] and not allow_incomplete_toolchain:
                raise RuntimeError("Incomplete Visual Studio installation; use explicit --allow-incomplete-toolchain only for reviewed prototype")
            record["stage"] = "compiler_environment"
            environment = build_environment(record["builder"])
            record["compiler_environment"] = {key: environment.get(key.upper()) for key in
                ("VCToolsVersion", "WindowsSDKVersion", "VSCMD_ARG_HOST_ARCH", "VSCMD_ARG_TGT_ARCH", "DISTUTILS_USE_SDK", "MSSdk")}
            record["stage"] = "wheel_verification"
            wheel = verify_wheel(wheel, wheel_sha256)
            record["wheel"] = {"path": str(wheel), "sha256": wheel_sha256.lower()}
            if sha256(REVIEWED) != REVIEWED_SHA256:
                raise RuntimeError("Reviewed Python boundary changed; re-review prototype before building")
            record["source_sha256"] = {p.relative_to(ROOT).as_posix(): sha256(p) for p in (SOURCE, REVIEWED, Path(__file__))}
            workspace.mkdir(exist_ok=False)
            compiler_packages = workspace / "compiler-packages"
            compiler_packages.mkdir()
            with zipfile.ZipFile(wheel) as archive:
                archive.extractall(compiler_packages)
            record["stage"] = "compile_and_link"
            command = [sys.executable, "-c", BUILD_PROGRAM, str(compiler_packages), str(SOURCE),
                       str(workspace / "generated"), str(workspace / "artifacts"), str(workspace / "objects")]
            record["command"] = command
            persist()
            started = datetime.now(timezone.utc)
            completed = subprocess.run(command, cwd=workspace, env=environment, capture_output=True)
            raw_logs = {}
            for name, value in (("stdout", completed.stdout), ("stderr", completed.stderr)):
                if value is None:
                    raise RuntimeError("Compiler log capture failed; do not certify this build")
                raw = value if isinstance(value, bytes) else value.encode("utf-8")
                path = workspace / f"build.{name}.bin"
                with path.open("xb") as log:
                    log.write(raw); log.flush(); os.fsync(log.fileno())
                raw_logs[name] = {"path": str(path), "sha256": sha256(path), "bytes": len(raw)}
            record["raw_build_logs"] = raw_logs
            def decoded(value):
                return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
            record.update(build_stdout=decoded(completed.stdout), build_stderr=decoded(completed.stderr),
                          build_returncode=completed.returncode,
                          build_elapsed_seconds=(datetime.now(timezone.utc) - started).total_seconds())
            if completed.returncode:
                raise RuntimeError("Cython build failed; see retained compiler output")
            artifacts = list((workspace / "artifacts").rglob("*.pyd"))
            if len(artifacts) != 1 or not artifacts[0].name.endswith(record["builder"]["extension_suffix"]):
                raise RuntimeError("Missing or ambiguous CPython-tagged extension artifact")
            record["stage"] = "artifact_audit"
            artifact = artifacts[0]
            record["artifact"] = {"path": str(artifact), "sha256": sha256(artifact)}
            record["generated_c_sha256"] = {p.relative_to(workspace).as_posix(): sha256(p) for p in workspace.rglob("*.c")}
            audits = {}
            for option in ("/imports", "/exports", "/headers"):
                audit = subprocess.run([record["builder"]["dumpbin"], option, str(artifact)],
                                       capture_output=True, text=True, check=True)
                audits[option] = audit.stdout
            if "PyInit_session_loops" not in audits["/exports"]:
                raise RuntimeError("Extension initialization export absent")
            record["pe_audit"] = audits
            record["private_import_tokens"] = sorted(set(re.findall(r"\b_Py\w+", audits["/imports"])))
            record["api_audit_limitation"] = "Generated C/disabled branches/internal access require human review; no public-only claim"
            for relative, digest in record["source_sha256"].items():
                if sha256(ROOT / relative) != digest:
                    raise RuntimeError("Source changed during build")
            record.update(status="completed", stage="completed", finished_at_utc=datetime.now(timezone.utc).isoformat())
            persist()
            return record
        except (Exception, KeyboardInterrupt) as exc:
            record.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                          finished_at_utc=datetime.now(timezone.utc).isoformat())
            if isinstance(exc, subprocess.CalledProcessError) and record["stage"] == "compiler_environment":
                record["environment_failure_stderr"] = exc.stderr
                record["environment_failure_returncode"] = exc.returncode
            persist()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-build", action="store_true", help="Requires separate human/root build authorisation")
    parser.add_argument("--allow-incomplete-toolchain", action="store_true",
                        help="Acknowledge verified tools from a partial installation; does NOT repair/mark it healthy")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cython-wheel", type=Path)
    parser.add_argument("--cython-wheel-sha256")
    args = parser.parse_args()
    if args.preflight_only:
        print(json.dumps(compiler_preflight(), indent=2)); return
    if not args.allow_build or not all((args.workspace, args.manifest, args.cython_wheel, args.cython_wheel_sha256)):
        parser.error("Explicit --allow-build and all four build/artifact arguments are required")
    result = build(args.workspace, args.manifest, args.cython_wheel, args.cython_wheel_sha256,
                   allow_incomplete_toolchain=args.allow_incomplete_toolchain)
    print(json.dumps({key: result.get(key) for key in ("status", "stage", "artifact", "build_elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
