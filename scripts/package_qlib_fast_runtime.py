"""Package only the reviewed helper and sanitized provenance; never build/install.

The receipt's complete digest is authorized by the original code-owned buffer
pin, independently of the accepted deployment profile populated afterwards.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import zipfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from qlib_yahoo import range_runtime as runtime


def _safe_banner(value):
    # Compiler banners are static identity text, not command output/logs. Reject
    # path-looking content rather than attempting to redact unknown personal data.
    runtime._require(isinstance(value, str) and bool(value.strip())
                     and not re.search(r"[A-Za-z]:[\\/]|[\\/](?:Users|home)[\\/]", value),
                     "Compiler banner is absent or contains a personal path")
    return value


def _provenance(manifest):
    pin_raw = runtime._read(runtime.BUFFER_PIN)
    pin = runtime._json(pin_raw)
    runtime._require(type(pin.get("schema_version")) is int and pin["schema_version"] == 1
                     and pin.get("kind") == "candidate_date_range_build_pin"
                     and pin.get("cython_version") == "3.1.8"
                     and pin.get("source_sha256") == runtime.INSTALLED_SOURCES,
                     "Invalid code-owned buffer build pin")
    selected = pin["builds"]["win32-amd64"]
    runtime._require(selected.get("python_abi") == "cp312"
                     and selected.get("extension_suffix") == runtime.SUFFIX
                     and selected.get("artifact_sha256") == runtime.INSTALLED_ARTIFACT_SHA256,
                     "Original helper pin differs from the accepted artifact")
    raw = runtime._read(manifest)
    manifest_sha = runtime._sha(raw)
    runtime._require(manifest_sha == runtime._digest(selected["manifest_sha256"]),
                     "Build receipt differs from its complete original pinned digest")
    receipt = runtime._json(raw)  # Only after full-byte trust is established.
    sources = {name: runtime._sha(runtime._read(runtime.ROOT / name))
               for name in runtime.INSTALLED_SOURCES}
    runtime._require(sources == runtime.INSTALLED_SOURCES
                     and receipt.get("source_sha256") == sources,
                     "Reviewed helper source bytes changed")
    runtime._require(type(receipt.get("schema_version")) is int and receipt["schema_version"] == 1
                     and receipt.get("kind") == "private_sequential_cython_build"
                     and receipt.get("status") == "completed" and receipt.get("stage") == "completed"
                     and receipt.get("cython_version") == "3.1.8"
                     and receipt.get("wheel", {}).get("sha256") == runtime.WHEEL_SHA256
                     and receipt.get("directives") == runtime.DIRECTIVES
                     and receipt.get("compiler_flags") == ["/MD", "/fp:strict"]
                     and receipt.get("macros") == [["CYTHON_USE_DICT_VERSIONS", "0"]]
                     and receipt.get("numpy_c_api_used") is False
                     and receipt.get("public_api_only_certified") is False
                     and receipt.get("allow_incomplete_toolchain") is True,
                     "Receipt lacks reviewed build policy and limitations")
    builder = receipt["builder"]
    runtime._require(builder.get("installation_healthy") is False
                     and builder.get("platform") == "win-amd64"
                     and builder.get("extension_suffix") == runtime.SUFFIX,
                     "Receipt builder identity or incomplete-install acknowledgement differs")
    workspace = runtime._checked(receipt["workspace"], directory=True)
    artifact = runtime._checked(receipt["artifact"]["path"])
    runtime._require(artifact.parent == workspace / "artifacts/qlib_yahoo"
                     and artifact.name == "session_loops" + runtime.SUFFIX,
                     "Receipt artifact is outside its recorded build workspace")
    artifact_raw = runtime._read(artifact)
    artifact_sha = runtime._sha(artifact_raw)
    runtime._require(artifact_sha == runtime.INSTALLED_ARTIFACT_SHA256
                     == receipt["artifact"]["sha256"], "Reviewed artifact bytes changed")
    tokens = receipt.get("private_import_tokens")
    runtime._require(isinstance(tokens, list) and all(isinstance(t, str)
                     and re.fullmatch(r"_Py\w+", t) for t in tokens),
                     "Malformed private-CPython import inventory")
    versions = {name: builder.get(name) for name in ("msvc_tools_version", "windows_sdk_version")}
    runtime._require(all(isinstance(value, str) and re.fullmatch(r"\d+(?:\.\d+)+", value)
                         for value in versions.values()), "Missing compiler/SDK version identity")
    provenance = {
        "schema_version": 1, "kind": "sanitized_date_range_build_provenance",
        "implementation": "public_buffer_leases", "source_sha256": sources,
        "original_manifest_sha256": manifest_sha, "artifact_sha256": artifact_sha,
        "cython_version": "3.1.8", "cython_wheel_sha256": runtime.WHEEL_SHA256,
        "compiler_flags": receipt["compiler_flags"], "directives": receipt["directives"],
        "macros": receipt["macros"], "private_import_tokens": tokens,
        "compiler_banner": _safe_banner(builder.get("compiler_banner")), **versions,
        "installation_healthy": False, "allow_incomplete_toolchain": True,
        "numpy_c_api_used": False, "gil_held": True, "public_api_only_certified": False,
        "api_audit_limitation": "Private CPython imports retained; generated C/internal access is not a public-API-only certificate.",
    }
    return artifact.name, artifact_raw, (json.dumps(provenance, sort_keys=True,
                                                  indent=2) + "\n").encode("utf-8")


def package(manifest, output):
    """Create a new deterministic two-entry zip and return portable digests."""
    artifact_name, artifact_raw, provenance_raw = _provenance(manifest)
    output = Path(output)
    runtime._require(output.is_absolute() and ".." not in output.parts,
                     "Package output must be absolute without parent traversal")
    runtime._checked(output.parent, directory=True)
    runtime._require(not output.exists() and not output.is_symlink(),
                     "Package output must be a new file")
    with output.open("xb") as handle:
        with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, raw in sorted(((artifact_name, artifact_raw), ("provenance.json", provenance_raw))):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, raw)
    return {"artifact": artifact_name, "artifact_sha256": runtime._sha(artifact_raw),
            "provenance_sha256": runtime._sha(provenance_raw),
            "asset_sha256": runtime._sha(runtime._read(output))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(package(args.manifest, args.output), indent=2))


if __name__ == "__main__":
    main()
