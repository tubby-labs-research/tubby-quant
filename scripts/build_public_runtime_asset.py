"""Rebuild the public date buffer locally; no downloads, installs or pin edits.

Use CPython 3.12 Windows x64 and the existing reviewed Cython wheel. Output is
NEW release-candidates/public-native* only. The source copy is byte-for-byte.
Portable receipt/provenance omit workspace paths; raw build logs remain local.
MSVC /pathmap has a workspace-specific operand represented explicitly as a
template in portable metadata. /Brepro removes PE timestamp nondeterminism.
This preserves private CPython imports and incomplete-toolchain limitations.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("accepted_helper_builder", ROOT / "scripts/build_qlib_session_loops.py")
accepted = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(accepted)
WHEEL_SHA256 = "cae61e8e730bfc7de917204c8b7c1ca8e3e0d134dcaa55a2e2ef98de741590fb"
SOURCE = "qlib_yahoo/session_range_buffer.pyx"
ARTIFACT = "session_loops.cp312-win_amd64.pyd"
ASSET = "qlib-date-range-buffer-cp312-win_amd64-public-v1.zip"
FOLDER = "date-range-buffer-cp312-win_amd64-public-v1"
SOURCE_NAMES = (SOURCE, "qlib_yahoo/fast_signed.py", "scripts/build_qlib_session_loops.py")


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def privacy_counts(raw):
    """Counts only; scan ASCII and UTF-16LE without printing matched values."""
    texts = [raw.decode("latin1"), raw.decode("utf-16le", errors="ignore")]
    return {
        "absolute_windows_paths": sum(len(re.findall(r"[A-Za-z]:[\\/]", t)) for t in texts),
        "personal_roots": sum(len(re.findall(r"(?:[\\/]Users[\\/]|[\\/]home[\\/])", t, re.I)) for t in texts),
        "checkout_name": sum(t.lower().count("my-first-agent") for t in texts),
    }


def require_private_free(raw):
    counts = privacy_counts(raw)
    if any(counts.values()):
        raise RuntimeError("Public content failed path privacy scan: " + json.dumps(counts))
    return counts


def output_workspace(value):
    path = Path(value).absolute()
    if ".." in path.parts:
        raise ValueError("Parent traversal forbidden")
    accepted.checked_path(path)
    parent = ROOT / "release-candidates"
    if path.parent != parent or not path.name.startswith("public-native") or path.exists():
        raise ValueError("Use a NEW release-candidates/public-native* workspace")
    accepted.checked_path(parent, kind="directory")
    return path


def build_program():
    # Retain accepted StrictBuild and every Cython directive/macro/math control.
    # Relative source and build directories are supplied from the copied tree.
    program = accepted.BUILD_PROGRAM.replace(
        "extra_compile_args=['/MD','/fp:strict']",
        "extra_compile_args=['/MD','/fp:strict','/Brepro','/experimental:deterministic',sys.argv[6]], "
        "extra_link_args=['/Brepro']")
    if program == accepted.BUILD_PROGRAM:
        raise RuntimeError("Accepted build recipe no longer matches reviewed adapter")
    return program.replace("name='private-qlib-session-loops'", "name='public-qlib-date-buffer'")


def deterministic_zip(path, members):
    with path.open("xb") as stream:
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, raw in sorted(members.items()):
                info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, raw)


def build(workspace, wheel, *, allow_incomplete_toolchain=False):
    if type(allow_incomplete_toolchain) is not bool:
        raise TypeError("Explicit boolean acknowledgement required")
    workspace = output_workspace(workspace)
    public_builder_sha = accepted.sha256(Path(__file__))
    # Use the public export's current descriptor, not historical private pins.
    profile = json.loads((ROOT / "docs/qlib-fast-runtime.v1.json").read_bytes())
    sources = profile["date_range"]["source_sha256"]
    if set(sources) != set(SOURCE_NAMES):
        raise RuntimeError("Descriptor must pin exactly three implementation inputs")
    for name, expected in sources.items():
        if accepted.sha256(ROOT / name) != expected:
            raise RuntimeError("Pinned implementation input changed: " + name)
    identity = accepted.compiler_preflight()
    if not identity["installation_healthy"] and not allow_incomplete_toolchain:
        raise RuntimeError("Incomplete MSVC requires explicit --allow-incomplete-toolchain")
    environment = accepted.build_environment(identity)
    wheel = accepted.verify_wheel(wheel, WHEEL_SHA256)
    program = build_program()
    workspace.mkdir()
    copied = workspace / SOURCE
    copied.parent.mkdir()
    copied.write_bytes((ROOT / SOURCE).read_bytes())
    packages = workspace / "compiler-packages"
    packages.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(packages)
    pathmap = "/pathmap:" + str(workspace) + "=/framework"
    command = [sys.executable, "-c", program, "compiler-packages", SOURCE,
               "generated", "artifacts", "objects", pathmap]
    result = subprocess.run(command, cwd=workspace, env=environment, capture_output=True)
    # These are local-only evidence, not archive members or public provenance.
    for name, raw in (("stdout", result.stdout), ("stderr", result.stderr)):
        (workspace / ("build." + name + ".bin")).write_bytes(raw)
    (workspace / "local-evidence.json").write_bytes(encoded({
        "builder": identity, "command": command, "returncode": result.returncode}))
    if result.returncode:
        raise RuntimeError("Build failed; inspect retained local logs (not publication inputs)")
    if re.search(rb"\bD900[27]\b", result.stdout + result.stderr):
        raise RuntimeError("Compiler rejected or ignored a requested flag; inspect local logs")
    artifacts = list((workspace / "artifacts").rglob("*.pyd"))
    if len(artifacts) != 1 or artifacts[0].name != ARTIFACT:
        raise RuntimeError("Missing or ambiguous CP312 artifact")
    artifact = artifacts[0]
    raw = artifact.read_bytes()
    counts = require_private_free(raw)
    audits = {}
    for option in ("/imports", "/exports", "/headers"):
        audit = subprocess.run([identity["dumpbin"], option, str(artifact)],
                               capture_output=True, text=True, check=True)
        audits[option] = audit.stdout
    (workspace / "local-pe-audit.json").write_bytes(encoded(audits))
    if "PyInit_session_loops" not in audits["/exports"] or "python312.dll" not in audits["/imports"].lower():
        raise RuntimeError("Native ABI export/import audit failed")
    tokens = sorted(set(re.findall(r"\b_Py\w+", audits["/imports"])))
    for name, expected in sources.items():
        if accepted.sha256(ROOT / name) != expected:
            raise RuntimeError("Pinned implementation changed during build")
    if digest(copied.read_bytes()) != sources[SOURCE]:
        raise RuntimeError("Build source copy changed")
    if accepted.sha256(Path(__file__)) != public_builder_sha:
        raise RuntimeError("Public builder changed during build")
    generated_hashes = {p.relative_to(workspace).as_posix(): accepted.sha256(p)
                        for p in (workspace / "generated").rglob("*.c")}
    (workspace / "local-generated-c.json").write_bytes(encoded(generated_hashes))
    # Deterministic, truthful portable receipt. No timestamps, paths or logs.
    receipt = {
        "schema_version": 1, "kind": "public_date_range_cython_build",
        "status": "completed", "stage": "completed", "source_sha256": sources,
        "public_builder_sha256": public_builder_sha,
        "build_program_sha256": digest(program.encode()),
        "artifact": {"name": ARTIFACT, "sha256": digest(raw)},
        "cython_version": accepted.CYTHON_VERSION, "cython_wheel_sha256": WHEEL_SHA256,
        "compiler_flags": accepted.FLAGS + ["/Brepro", "/experimental:deterministic", "/pathmap:<build-root>=/framework"],
        "compiler_flags_representation": "pathmap build-root is the actual absolute workspace; all other flags are literal",
        "linker_flags": ["/Brepro"], "directives": accepted.DIRECTIVES,
        "macros": [list(pair) for pair in accepted.MACROS],
        "generated_c_identity_location": "local-generated-c.json (workspace-dependent Cython metadata; local evidence only)",
        "tool_sha256": {Path(p).name: sha for p, sha in identity["tool_sha256"].items()},
        "python_header_sha256": identity["python_header_sha256"],
        "python_import_library_sha256": identity["python_import_library_sha256"],
        "build_dependency_versions": {n: accepted.importlib.metadata.version(n)
                                      for n in ("setuptools", "wheel", "numpy", "pandas")},
        "python_version": list(sys.version_info[:3]), "platform": "win-amd64",
        "extension_suffix": ".cp312-win_amd64.pyd", "private_import_tokens": tokens,
        "compiler_banner": identity["compiler_banner"],
        "msvc_tools_version": identity["msvc_tools_version"],
        "windows_sdk_version": identity["windows_sdk_version"],
        "installation_healthy": identity["installation_healthy"],
        "allow_incomplete_toolchain": allow_incomplete_toolchain,
        "numpy_c_api_used": False, "gil_held": True, "public_api_only_certified": False,
        "path_privacy_counts": counts,
    }
    receipt_raw = encoded(receipt)
    require_private_free(receipt_raw)
    (workspace / "public-build-receipt.json").write_bytes(receipt_raw)
    provenance = {key: receipt[key] for key in (
        "schema_version", "source_sha256", "cython_version", "compiler_flags", "directives",
        "macros", "private_import_tokens", "compiler_banner", "msvc_tools_version",
        "windows_sdk_version", "installation_healthy", "allow_incomplete_toolchain",
        "numpy_c_api_used", "gil_held", "public_api_only_certified", "linker_flags",
        "compiler_flags_representation", "public_builder_sha256", "build_program_sha256")}
    provenance.update(kind="sanitized_date_range_build_provenance", implementation="public_buffer_leases",
                      original_manifest_sha256=digest(receipt_raw), artifact_sha256=digest(raw),
                      cython_wheel_sha256=WHEEL_SHA256,
                      original_manifest_description="SHA256 of new portable public-build-receipt.json; not the private accepted receipt",
                      api_audit_limitation="Private CPython imports retained; generated C/internal access is not a public-API-only certificate.")
    provenance_raw = encoded(provenance)
    require_private_free(provenance_raw)
    destination = workspace / ASSET
    deterministic_zip(destination, {ARTIFACT: raw, "provenance.json": provenance_raw})
    summary = {"asset": ASSET, "folder": FOLDER, "artifact": ARTIFACT,
               "asset_sha256": accepted.sha256(destination), "artifact_sha256": digest(raw),
               "provenance_sha256": digest(provenance_raw),
               "original_manifest_sha256": digest(receipt_raw), "bytes": destination.stat().st_size,
               "path_privacy_counts": counts}
    (workspace / "summary.json").write_bytes(encoded(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--cython-wheel", required=True, type=Path)
    parser.add_argument("--allow-incomplete-toolchain", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(build(args.workspace, args.cython_wheel,
                               allow_incomplete_toolchain=args.allow_incomplete_toolchain), indent=2))
    except Exception as exc:
        # Underlying preflight errors can contain personal paths; keep them local.
        print("Public build failed (" + type(exc).__name__ + "); check local evidence or run scoped diagnostics.", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
