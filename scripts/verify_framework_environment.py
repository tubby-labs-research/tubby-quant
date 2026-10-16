"""Read-only installed-file check for setup's isolated virtual environment.

Only the selected interpreter's own site-packages are enumerated. RECORD paths
may refer to scripts elsewhere in that virtualenv, but never outside it. This
does not validate hashes, imports, dependency compatibility, or external editable
source trees; setup's pip check and fixed import checks remain separate.
"""

from __future__ import annotations

import csv
import importlib.metadata as metadata
import io
from pathlib import Path, PureWindowsPath
import re
import sys
import sysconfig


def verify_distributions(prefix: Path, site_packages: list[Path]) -> tuple[int, list[str]]:
    """Check local distribution metadata and declared files without modifying them."""
    prefix = prefix.resolve()
    roots = list(dict.fromkeys(path.resolve() for path in site_packages))
    if not roots or any(not root.is_relative_to(prefix) or not root.is_dir() for root in roots):
        return 0, ["site-packages: outside virtualenv or missing"]
    issues: list[str] = []
    count = 0
    try:
        # Inspect only immediate metadata entries, never recurse through package
        # trees or follow metadata symlinks into another installation.
        entries = (
            entry for root in roots for entry in root.iterdir()
            if entry.name.lower().endswith((".dist-info", ".egg-info"))
        )
        for entry in entries:
            count += 1
            label = "distribution"
            try:
                if not entry.resolve().is_relative_to(prefix) or not entry.is_dir():
                    issues.append(f"{label}: metadata outside virtualenv or unsupported legacy installation")
                    continue
                metadata_name = "METADATA" if entry.name.lower().endswith(".dist-info") else "PKG-INFO"
                if any(not (entry / filename).resolve().is_relative_to(prefix) for filename in (metadata_name, "RECORD")):
                    issues.append(f"{label}: metadata outside virtualenv")
                    continue
                dist = metadata.PathDistribution(entry)
                info = dist.metadata
                name = info.get("Name", "")
                if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,198}[A-Za-z0-9])?", name):
                    issues.append(f"{label}: missing or invalid Name")
                    continue
                label = name
                version = info.get("Version", "")
                if not version or not re.fullmatch(r"[A-Za-z0-9.!+_-]+", version):
                    issues.append(f"{label}: missing or invalid Version")
                    continue
                record = dist.read_text("RECORD")
                if not record or not record.strip():
                    issues.append(f"{label}: missing or empty RECORD")
                    continue
                base = Path(dist.locate_file("")).resolve()
                if base not in roots:
                    issues.append(f"{label}: distribution outside site-packages")
                    continue
                for row in csv.reader(io.StringIO(record), strict=True):
                    if len(row) != 3 or not row[0]:
                        issues.append(f"{label}: invalid RECORD")
                        break
                    entry = row[0]
                    # Reject absolute paths in either platform's syntax, even
                    # when inspecting synthetic records on the other platform.
                    relative = Path(entry.replace("\\", "/"))
                    if relative.is_absolute() or PureWindowsPath(entry).drive or "\x00" in entry:
                        issues.append(f"{label}: RECORD path outside virtualenv")
                        break
                    target = (base / relative).resolve()
                    if not target.is_relative_to(prefix):
                        issues.append(f"{label}: RECORD path outside virtualenv")
                        break
                    if target.suffix.lower() in {".pyc", ".pyo"}:
                        continue
                    if not target.is_file():
                        issues.append(f"{label}: declared file missing")
                        break
            except (OSError, ValueError, TypeError, RuntimeError, csv.Error):
                # Never echo exception messages: they can contain private paths
                # or untrusted metadata contents.
                issues.append(f"{label}: unreadable installed metadata or file")
    except (OSError, ValueError):
        issues.append("site-packages: unable to enumerate distributions")
    if not count:
        issues.append("site-packages: no installed distributions")
    return count, issues


def main() -> int:
    if sys.prefix == sys.base_prefix:
        issues = ["interpreter: an isolated virtualenv is required"]
        count = 0
    else:
        count, issues = verify_distributions(
            Path(sys.prefix),
            [Path(sysconfig.get_path(key)) for key in ("purelib", "platlib")],
        )
    if issues:
        print("Environment verification failed:", file=sys.stderr)
        for issue in issues[:10]:
            print(f"  {issue}", file=sys.stderr)
        if len(issues) > 10:
            print(f"  and {len(issues) - 10} more distribution problems", file=sys.stderr)
        print(
            "Rerun setup with -Recreate (Windows) or --recreate (macOS); "
            "this moves the old environment aside.",
            file=sys.stderr,
        )
        return 1
    print(f"Environment verification passed: {count} local distributions; metadata and declared files present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
