"""Type-check the research-lab harness and its neighbours with a fixed mypy configuration.

Targets: the lab harness (``autoresearch/lab``), the research domains, the record
store the harness writes, the console's lab service and the demonstration market.
Flags, always the same so a local run means what the CI run means:

    --ignore-missing-imports   a third-party package without stubs is not an error
    --check-untyped-defs       the bodies of unannotated functions are checked too
    --warn-unused-ignores      a ``# type: ignore`` that suppresses nothing is an error
    --warn-redundant-casts     a ``cast`` to the type mypy already inferred is an error
    --follow-imports=silent    modules the targets import are analysed for their types,
                               but their own errors are not reported by this gate

``autoresearch.lab`` is additionally held to mypy's ``--strict`` set through the
``[[tool.mypy.overrides]]`` entry in ``pyproject.toml`` (every function annotated,
no implicit ``Any`` generics, no untyped calls, no ``Any`` returns). Exit status 1
on any error; the error count is printed either way.

    python scripts/check_types.py            # mypy required
    python scripts/check_types.py --lenient  # skip when mypy is not installed

mypy is pinned in requirements-dev.txt.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    "autoresearch/lab",
    "autoresearch/domains",
    "autoresearch/records.py",
    "qlib_yahoo/webapp/lab_service.py",
    "qlib_yahoo/demo_market.py",
]
FLAGS = ["--ignore-missing-imports", "--check-untyped-defs", "--warn-unused-ignores", "--warn-redundant-casts",
         "--follow-imports=silent", "--config-file", "pyproject.toml"]
_FOUND = re.compile(r"^Found (\d+) errors? in \d+ files? \(checked (\d+) source files?\)", re.MULTILINE)
_CLEAN = re.compile(r"^Success: no issues found in (\d+) source files?", re.MULTILINE)


def run_mypy(lenient: bool) -> int:
    command = [sys.executable, "-m", "mypy", *FLAGS, *[t for t in TARGETS if (ROOT / t).exists()]]
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    except FileNotFoundError:
        result = None
    if result is None or (result.returncode != 0 and "No module named mypy" in (result.stderr or "")):
        message = "mypy is not installed (pip install -r requirements-dev.txt)"
        print(("skipped: " if lenient else "error: ") + message)
        return 0 if lenient else 1
    sys.stdout.write(result.stdout)
    if result.returncode:
        sys.stderr.write(result.stderr)
    found, clean = _FOUND.search(result.stdout), _CLEAN.search(result.stdout)
    if found:
        errors, checked = int(found.group(1)), int(found.group(2))
    elif clean:
        errors, checked = 0, int(clean.group(1))
    else:  # mypy did not finish (bad flag, crash): report that as a failure
        errors, checked = -1, 0
    status = 1 if result.returncode or errors != 0 else 0
    label = f"{errors} error{'s' if errors != 1 else ''}" if errors >= 0 else "did not complete"
    print(f"mypy ({' '.join(FLAGS[:5])}; strict for autoresearch.lab): {label}, {checked} source files checked")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lenient", action="store_true", help="skip the check when mypy is missing instead of failing")
    args = parser.parse_args(argv)
    status = run_mypy(args.lenient)
    print("Type checks " + ("passed." if status == 0 else "FAILED."))
    return status


if __name__ == "__main__":
    sys.exit(main())
