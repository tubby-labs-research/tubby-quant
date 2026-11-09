"""Run the offline unit-test suite from a canonical temporary root.

``python -m unittest discover -s tests -p "test_*.py"`` is unchanged and keeps
working on Windows and Linux. On macOS the default temporary directory is
reached through a symbolic link (``/var`` -> ``/private/var``), and several
project checks deliberately refuse paths that pass through links or reparse
points (runtime installs, date-range buffers, evidence stores, release export).
Under ``python -m unittest`` those checks then reject every ``TemporaryDirectory``
before the behaviour under test runs, so the suite reports hundreds of spurious
failures on a Mac.

This runner points :mod:`tempfile` (and ``TMPDIR`` for subprocesses started by
tests) at the canonical temporary root before discovery. It adds no fixtures,
changes no test semantics, and discovers tests exactly like the unittest
command above: start directory ``tests``, top-level directory ``tests``.

Usage::

    python scripts/run_tests.py                      # whole offline suite
    python scripts/run_tests.py -p "test_alphalab*.py"
    python scripts/run_tests.py -k neutralization -v
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"


def canonical_tempdir() -> str:
    """Make the process and its children create temporary files below a link-free root."""
    path = os.path.realpath(tempfile.gettempdir())
    tempfile.tempdir = path
    os.environ["TMPDIR"] = path
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-p", "--pattern", default="test_*.py",
                        help="test file pattern passed to discovery (default: test_*.py)")
    parser.add_argument("-k", dest="keywords", action="append", default=[],
                        help="only run tests whose name contains this text (repeatable)")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose test output")
    parser.add_argument("-f", "--failfast", action="store_true", help="stop at the first failure or error")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    temp_root = canonical_tempdir()
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    loader = unittest.TestLoader()
    if args.keywords:
        loader.testNamePatterns = [f"*{keyword}*" for keyword in args.keywords]
    suite = loader.discover(start_dir=str(TESTS), pattern=args.pattern, top_level_dir=str(TESTS))
    print(f"temporary root: {temp_root}", file=sys.stderr)
    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1, failfast=args.failfast,
                                     warnings="default")
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
