"""Explicit activation of reviewed date-range builds and installed helpers.

No build, compiler discovery, environment edits or path insertion happens here.
The reviewed native build retains private CPython symbols; this is not a
public-API-only certificate. Unsupported hosts keep the original Python path.
"""
from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import platform
import re
import stat
import sys
import sysconfig

from qlib_yahoo.dataset import DatasetError

ROOT = Path(__file__).absolute().parents[1]
PIN = ROOT / "docs/qlib-row-range-build.candidate.json"
BUFFER_PIN = ROOT / "docs/qlib-row-range-buffer-build.candidate.json"
INSTALLED_FOLDER = "date-range-buffer-cp312-win_amd64-v1"
INSTALLED_ARTIFACT_SHA256 = "0c01419fd7cd6e782a50a57210628147d29d439009f9bee58224958b873d0ca8"
# Same reviewed implementation, rebuilt with relative source names and
# reproducible path mapping for the clean framework distribution. The original
# private installation remains a separate, byte-pinned identity.
PUBLIC_INSTALLED_FOLDER = "date-range-buffer-cp312-win_amd64-public-v1"
PUBLIC_INSTALLED_ARTIFACT_SHA256 = "20ef8147b397d47686146c22bd40bdacc5e1a152b3da2915ced4ab97684b3bd6"
INSTALLED_SOURCES = {
    "qlib_yahoo/session_range_buffer.pyx": "48c641d31e55280f6a1087fbb0eba2ae6eae0abcd58b3fb798f72447690e55bd",
    "qlib_yahoo/fast_signed.py": "5abd4863c48ea38b970790f8d48f6fd59e5c6eb451389baf721eab0cef724c42",
    "scripts/build_qlib_session_loops.py": "bc92de10aa0ffd1a73e4796ac0a3ee0c600877599d8c0cc5003f050db3c6b07a",
}
DATA_ROOT = Path(r"D:\Qlib")
MODULE = "qlib_yahoo.session_loops"
SUFFIX = ".cp312-win_amd64.pyd"
SOURCES = ("qlib_yahoo/session_loops.pyx", "qlib_yahoo/fast_signed.py",
           "scripts/build_qlib_session_loops.py")
BUFFER_SOURCES = ("qlib_yahoo/session_range_buffer.pyx", "qlib_yahoo/fast_signed.py",
                  "scripts/build_qlib_session_loops.py")
WHEEL_SHA256 = "cae61e8e730bfc7de917204c8b7c1ca8e3e0d134dcaa55a2e2ef98de741590fb"
DIRECTIVES = {"language_level": 3, "infer_types": False, "annotation_typing": False,
              "binding": True, "boundscheck": True, "wraparound": True, "cdivision": False}
_OWNED = {}  # Modules created by this loader, never inferred from __file__ alone.


def _host():
    return {"implementation": sys.implementation.name,
            "python_abi": sys.implementation.cache_tag,
            "python_version": list(sys.version_info[:2]), "platform": sys.platform,
            "build_platform": sysconfig.get_platform(), "machine": platform.machine(),
            "extension_suffix": sysconfig.get_config_var("EXT_SUFFIX")}


def _supported(host):
    return (host["implementation"] == "cpython" and host["python_abi"] == "cpython-312"
            and host["python_version"] == [3, 12] and host["platform"] == "win32"
            and host["build_platform"] == "win-amd64"
            and host["machine"].lower() in ("amd64", "x86_64")
            and host["extension_suffix"] == SUFFIX)


def _checked(path, *, directory=False):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise DatasetError("Date-range paths must be absolute without parent traversal")
    for parent in (path, *path.parents):
        details = parent.lstat()
        if stat.S_ISLNK(details.st_mode) or getattr(details, "st_file_attributes", 0) & 0x400:
            raise DatasetError(f"Redirected date-range path is forbidden: {parent}")
    mode = path.stat().st_mode
    if not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)):
        raise DatasetError(f"Date-range path is not a regular {'directory' if directory else 'file'}: {path}")
    return path


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise DatasetError("Date-range pin contains a malformed SHA256")
    return value


def _read(path):
    return _checked(path).read_bytes()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    def unique(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise DatasetError("Duplicate date-range JSON key")
            result[name] = value
        return result
    value = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise DatasetError("Date-range JSON must be an object")
    return value


def _require(condition, message):
    if not condition:
        raise DatasetError(message)


def _verify(manifest_path):
    # Only these code-owned pins can authorize a receipt. Parse their metadata,
    # never the caller's receipt, until its complete digest selects one variant.
    candidates = []
    for pin_path, source_names, implementation in (
            (PIN, SOURCES, "scalar_views"),
            (BUFFER_PIN, BUFFER_SOURCES, "public_buffer_leases")):
        pin_raw = _read(pin_path)
        pin = _json(pin_raw)
        _require(type(pin.get("schema_version")) is int and pin["schema_version"] == 1
                 and pin.get("kind") == "candidate_date_range_build_pin"
                 and pin.get("cython_version") == "3.1.8", "Invalid trusted date-range pin")
        sources = pin["source_sha256"]
        _require(isinstance(sources, dict) and set(sources) == set(source_names),
                 "Trusted date-range pin must cover exactly the three reviewed sources")
        selected = pin["builds"]["win32-amd64"]
        _require(selected["python_abi"] == "cp312" and selected["extension_suffix"] == SUFFIX,
                 "Trusted date-range pin has an unsupported ABI")
        candidates.append((pin_path, pin_raw, source_names, sources, selected,
                           implementation, _digest(selected["manifest_sha256"])))
    manifest_path = _checked(manifest_path)
    root = _checked(DATA_ROOT, directory=True)
    _require(manifest_path.is_relative_to(root), "Date-range receipt must be under D:\\Qlib")
    raw = _read(manifest_path)
    manifest_sha = _sha(raw)
    matches = [candidate for candidate in candidates if candidate[-1] == manifest_sha]
    _require(bool(matches),
             "Date-range receipt differs from the trusted build pin")
    _require(len(matches) == 1, "Date-range receipt has an ambiguous trusted build digest")
    pin_path, pin_raw, source_names, sources, selected, implementation, _ = matches[0]
    # Only parse a receipt after checking its complete bytes against the pin.
    receipt = _json(raw)
    actual_sources = {name: _sha(_read(ROOT / name)) for name in source_names}
    _require(actual_sources == {name: _digest(sources[name]) for name in source_names},
             "Date-range reviewed source bytes changed")
    _require(type(receipt.get("schema_version")) is int and receipt["schema_version"] == 1
             and receipt.get("kind") == "private_sequential_cython_build"
             and receipt.get("status") == "completed" and receipt.get("stage") == "completed",
             "Date-range receipt is not a completed private build")
    _require(receipt.get("source_sha256") == actual_sources
             and receipt.get("cython_version") == "3.1.8"
             and receipt.get("wheel", {}).get("sha256") == WHEEL_SHA256,
             "Date-range source or Cython build identity differs")
    directives = receipt.get("directives")
    _require(isinstance(directives, dict) and set(directives) == set(DIRECTIVES)
             and all(type(directives[k]) is type(v) and directives[k] == v
                     for k, v in DIRECTIVES.items())
             and receipt.get("compiler_flags") == ["/MD", "/fp:strict"]
             and receipt.get("macros") == [["CYTHON_USE_DICT_VERSIONS", "0"]]
             and receipt.get("numpy_c_api_used") is False,
             "Date-range build flags or directives differ from the reviewed policy")
    builder = receipt["builder"]
    _require(builder.get("platform") == "win-amd64" and builder.get("extension_suffix") == SUFFIX
             and isinstance(builder.get("python"), str) and re.match(r"3\.12\.\d+\b", builder["python"])
             and isinstance(builder.get("executable"), str) and Path(builder["executable"]).is_absolute(),
             "Date-range builder ABI or executable identity is unsupported")
    _require(builder.get("installation_healthy") is True
             or (builder.get("installation_healthy") is False
                 and receipt.get("allow_incomplete_toolchain") is True),
             "Date-range build lacks explicit incomplete-toolchain acknowledgement")
    _require(receipt.get("public_api_only_certified") is False
             and isinstance(receipt.get("private_import_tokens"), list)
             and isinstance(receipt.get("api_audit_limitation"), str),
             "Date-range build lacks its private-CPython audit limitation")
    workspace = _checked(receipt["workspace"], directory=True)
    _require(workspace.is_relative_to(root) and not manifest_path.is_relative_to(workspace),
             "Date-range build workspace is outside the permitted root or contains its receipt")
    artifact = _checked(receipt["artifact"]["path"])
    _require(artifact.parent == workspace / "artifacts/qlib_yahoo"
             and artifact.name == "session_loops" + SUFFIX,
             "Date-range artifact is outside its recorded build workspace")
    artifact_sha = _sha(_read(artifact))
    _require(artifact_sha == _digest(selected["artifact_sha256"])
             == receipt["artifact"]["sha256"], "Date-range extension bytes changed")
    return artifact, {"trusted_pin_path": str(pin_path), "trusted_pin_sha256": _sha(pin_raw),
                      "implementation": implementation,
                      "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha,
                      "source_sha256": actual_sources, "artifact_path": str(artifact),
                      "artifact_sha256": artifact_sha, "cython_version": "3.1.8",
                      "builder": builder, "compiler_flags": receipt["compiler_flags"],
                      "directives": directives, "numpy_c_api_used": False, "gil_held": True,
                      "public_api_only_certified": False,
                      "private_import_tokens": receipt["private_import_tokens"],
                      "api_audit_limitation": receipt["api_audit_limitation"]}


def _load_verified(artifact, build):
    """Shared process-local ownership policy for prototype and installed bytes."""
    previous = sys.modules.get(MODULE)
    if MODULE in sys.modules:
        owned = _OWNED.get(MODULE)
        _require(previous is not None and owned is not None
                 and owned[0] is previous and owned[1] == str(artifact)
                 and owned[2] == build["artifact_sha256"]
                 and owned[4] == build["implementation"]
                 and getattr(previous, "__file__", None) == str(artifact)
                 and getattr(previous, "try_valid_range", None) is owned[3],
                 "Unowned or conflicting preloaded date-range extension")
        function = previous.try_valid_range
    else:
        loader = importlib.machinery.ExtensionFileLoader(MODULE, str(artifact))
        spec = importlib.util.spec_from_file_location(MODULE, artifact, loader=loader)
        _require(spec is not None, "Unable to create date-range extension spec")
        # create_module can execute native initialization. Restore only our
        # own named entry on failure; ordinary dependency imports and other
        # threads' import changes remain, as with normal Python imports.
        absent = object()
        prior_entry = sys.modules.get(MODULE, absent)
        try:
            module = importlib.util.module_from_spec(spec)
            sys.modules[MODULE] = module
            loader.exec_module(module)
            function = module.try_valid_range
            _require(callable(function), "Date-range extension has no callable range boundary")
            _require(getattr(module, "__file__", None) == str(artifact),
                     "Date-range extension module path differs from its pinned artifact")
        except BaseException:
            if prior_entry is absent:
                sys.modules.pop(MODULE, None)
            else:
                sys.modules[MODULE] = prior_entry
            raise
        _OWNED[MODULE] = (module, str(artifact), build["artifact_sha256"], function,
                          build["implementation"])
    _require(callable(function), "Date-range extension has no callable range boundary")
    return function


def activate_range(manifest_path):
    """Return ``(try_valid_range, identity)`` only for an explicitly pinned build.

    Unsupported ABI selection is a recorded original-Python fallback, before
    touching any files. Missing or invalid supported-host requests fail closed.
    Native loading is process-local; default callers need not import this module.
    """
    host = _host()
    identity = {"schema_version": 1, "kind": "candidate_date_range_runtime",
                "host": host, "selection": "original_python", "fallback": None}
    if not _supported(host):
        identity["fallback"] = "unsupported_platform_or_abi"
        return None, identity
    try:
        artifact, build = _verify(manifest_path)
        identity.update(build)
        function = _load_verified(artifact, build)
        identity.update(selection="candidate_cython_date_range", fallback=None)
        return function, identity
    except DatasetError:
        raise
    except Exception as exc:
        raise DatasetError(f"Date-range candidate activation failed: {exc}") from exc


def _verify_installed(installation_root):
    # There is no caller-selected receipt or pin in this deployment interface.
    pin_path = ROOT / "docs/qlib-fast-runtime.v1.json"
    pin_raw = _read(pin_path)
    pin = _json(pin_raw)
    _require(type(pin.get("schema_version")) is int and pin["schema_version"] == 1
             and pin.get("kind") == "accepted_signed_runtime_profile"
             and pin.get("name") == "accepted-v1", "Invalid accepted date-range runtime profile")
    selected = pin.get("date_range")
    _require(isinstance(selected, dict), "Accepted date-range profile is missing")
    folder = selected.get("folder")
    reviewed = {INSTALLED_FOLDER: INSTALLED_ARTIFACT_SHA256,
                PUBLIC_INSTALLED_FOLDER: PUBLIC_INSTALLED_ARTIFACT_SHA256}
    _require(isinstance(folder, str) and folder in reviewed,
             "Accepted date-range folder differs from the reviewed helpers")
    expected_artifact = reviewed[folder]
    _require(selected.get("artifact") == "session_loops" + SUFFIX
             and selected.get("artifact_sha256") == expected_artifact
             and selected.get("source_sha256") == INSTALLED_SOURCES,
             "Accepted date-range profile differs from the reviewed helper")
    sources = {name: _sha(_read(ROOT / name)) for name in INSTALLED_SOURCES}
    _require(sources == INSTALLED_SOURCES, "Date-range reviewed source bytes changed")
    installation = _checked(installation_root, directory=True)
    _require(installation.name == folder
             and installation.parent.name == "runtime-deps",
             "Date-range installation must use the configured runtime-deps folder")
    artifact = installation / selected["artifact"]
    artifact_sha = _sha(_read(artifact))
    _require(artifact_sha == expected_artifact, "Date-range extension bytes changed")
    provenance_path = installation / "provenance.json"
    provenance_raw = _read(provenance_path)
    provenance_sha = _sha(provenance_raw)
    _require(provenance_sha == _digest(selected.get("provenance_sha256")),
             "Installed date-range provenance differs from the accepted profile")
    provenance = _json(provenance_raw)
    _require(type(provenance.get("schema_version")) is int and provenance["schema_version"] == 1
             and provenance.get("kind") == "sanitized_date_range_build_provenance"
             and provenance.get("source_sha256") == sources
             and provenance.get("artifact_sha256") == artifact_sha
             and provenance.get("implementation") == "public_buffer_leases"
             and provenance.get("numpy_c_api_used") is False
             and provenance.get("gil_held") is True
             and provenance.get("public_api_only_certified") is False
             and provenance.get("installation_healthy") is False
             and provenance.get("allow_incomplete_toolchain") is True
             and isinstance(provenance.get("private_import_tokens"), list)
             and isinstance(provenance.get("api_audit_limitation"), str),
             "Installed date-range provenance lacks the reviewed limitations")
    return artifact, {"trusted_pin_path": str(pin_path), "trusted_pin_sha256": _sha(pin_raw),
                      "profile_name": "accepted-v1", "implementation": "public_buffer_leases",
                      "installation_root": str(installation), "source_sha256": sources,
                      "artifact_path": str(artifact), "artifact_sha256": artifact_sha,
                      "provenance_path": str(provenance_path), "provenance_sha256": provenance_sha,
                      "provenance": provenance, "numpy_c_api_used": False, "gil_held": True,
                      "public_api_only_certified": False,
                      "installation_healthy": False, "allow_incomplete_toolchain": True,
                      "private_import_tokens": provenance["private_import_tokens"],
                      "api_audit_limitation": provenance["api_audit_limitation"]}


def activate_installed_range(installation_root):
    """Activate the accepted helper from an absolute configured installation.

    Unsupported hosts return the original Python path before filesystem access;
    a supported host with missing or mismatched installation fails closed.
    """
    host = _host()
    identity = {"schema_version": 1, "kind": "accepted_date_range_runtime",
                "host": host, "selection": "original_python", "fallback": None}
    if not _supported(host):
        identity["fallback"] = "unsupported_platform_or_abi"
        return None, identity
    try:
        artifact, build = _verify_installed(installation_root)
        identity.update(build)
        function = _load_verified(artifact, build)
        identity.update(selection="accepted_cython_date_range", fallback=None)
        return function, identity
    except DatasetError:
        raise
    except Exception as exc:
        raise DatasetError(f"Installed date-range activation failed: {exc}") from exc
