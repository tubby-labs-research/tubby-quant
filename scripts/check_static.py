"""Syntax and undefined-name gate: the mistakes a test suite can miss.

Runs a strict, low-noise ruff selection over the Python packages (syntax errors,
undefined names, redefinitions, mutable or call defaults) and a
syntax parse of every console script with Node. Exit status 1 on any finding.

    python scripts/check_static.py            # ruff and node required
    python scripts/check_static.py --lenient  # skip a check whose tool is missing

ruff is pinned in requirements-dev.txt.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_TARGETS = ["autoresearch", "qlib_yahoo", "scripts", "tests", "tools"]
RUFF_SELECT = "E9,F63,F7,F82,F821,F811,F823,B006,B008"
EXCLUDE = "env,worktrees,release-candidates,node_modules,data,upstream-qlib-v0.9.7"
STATIC_JS = ROOT / "qlib_yahoo" / "webapp" / "static"


def run_ruff(lenient: bool) -> int:
    command = [sys.executable, "-m", "ruff", "check", "--select", RUFF_SELECT, "--exclude", EXCLUDE,
               *[t for t in PYTHON_TARGETS if (ROOT / t).exists()]]
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    except FileNotFoundError:
        result = None
    if result is None or (result.returncode != 0 and "No module named ruff" in (result.stderr or "")):
        message = "ruff is not installed (pip install -r requirements-dev.txt)"
        print(("skipped: " if lenient else "error: ") + message)
        return 0 if lenient else 1
    sys.stdout.write(result.stdout)
    if result.returncode:
        sys.stderr.write(result.stderr)
    print(f"ruff ({RUFF_SELECT}): {'ok' if result.returncode == 0 else 'findings above'}")
    return 1 if result.returncode else 0


def run_node(lenient: bool, folder: Path = STATIC_JS) -> int:
    node = shutil.which("node")
    if not node:
        print(("skipped: " if lenient else "error: ") + "node is not installed; console scripts not parsed")
        return 0 if lenient else 1
    failures = 0
    for script in sorted(folder.glob("*.js")):
        check = ("const vm = require('vm'), fs = require('fs'); const f = process.argv[1];"
                 "try { new vm.Script(fs.readFileSync(f, 'utf8'), { filename: f }); }"
                 "catch (e) { console.error(f + ': ' + e.message); process.exit(1); }")
        result = subprocess.run([node, "-e", check, str(script)], cwd=ROOT, text=True, capture_output=True, check=False)
        if result.returncode:
            failures += 1
            sys.stderr.write(result.stderr)
    print(f"node syntax ({len(list(folder.glob('*.js')))} scripts): {'ok' if not failures else f'{failures} failed'}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lenient", action="store_true", help="skip a check whose tool is missing instead of failing")
    args = parser.parse_args(argv)
    status = run_ruff(args.lenient) | run_node(args.lenient)
    print("Static checks " + ("passed." if status == 0 else "FAILED."))
    return status


if __name__ == "__main__":
    sys.exit(main())
