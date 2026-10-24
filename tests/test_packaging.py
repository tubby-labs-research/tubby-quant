"""Packaging metadata stays consistent with the tree it describes.

Pure standard library: these checks run in the dependency-light CI job. They do
not build or install anything; an installed distribution is checked only when
one is present.
"""

from __future__ import annotations

import ast
import fnmatch
import importlib.metadata
import re
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
PACKAGES = ("qlib_yahoo", "autoresearch")
VERSION = re.compile(r"^\d+(\.\d+){1,2}((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?$")
SKIPPED_SUFFIXES = {".py", ".pyc", ".pyo"}


def load_pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def requirement_pins(path: Path) -> set[str]:
    """Exact ``name==version`` lines of a requirements file; include lines are not followed."""
    pins = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            pins.add(line.replace(" ", ""))
    return pins


def module_version() -> str:
    """Read ``qlib_yahoo.__version__`` from source without importing the package."""
    tree = ast.parse((ROOT / "qlib_yahoo" / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets):
            assert isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
            return node.value.value
    raise AssertionError("qlib_yahoo/__init__.py defines no literal __version__")


class PyprojectTests(unittest.TestCase):
    def setUp(self):
        self.data = load_pyproject()
        self.project = self.data["project"]

    def test_identity_and_python_requirement(self):
        self.assertEqual(self.project["name"], "tubby-quant")
        self.assertEqual(self.project["requires-python"], ">=3.12")
        self.assertIn("version", self.project["dynamic"])
        self.assertEqual(self.data["tool"]["setuptools"]["dynamic"]["version"], {"attr": "qlib_yahoo.__version__"})
        self.assertRegex(module_version(), VERSION)
        self.assertEqual(self.project["license"], "Apache-2.0")

    def test_only_framework_packages_are_distributed(self):
        include = self.data["tool"]["setuptools"]["packages"]["find"]["include"]
        self.assertEqual(sorted(include), sorted(f"{name}*" for name in PACKAGES))
        for name in PACKAGES:
            self.assertTrue((ROOT / name / "__init__.py").is_file(), name)
        for excluded in ("tests", "tools", "scripts", "release"):
            self.assertFalse(any(fnmatch.fnmatch(excluded, pattern) for pattern in include), excluded)

    def test_console_scripts_point_at_existing_entry_functions(self):
        scripts = self.project["scripts"]
        self.assertEqual(set(scripts), {"tubby-quant", "tubby-qlib", "tubby-research", "tubby-alpha-mcp"})
        for name, target in scripts.items():
            module, function = target.split(":")
            path = ROOT.joinpath(*module.split(".")).with_suffix(".py")
            self.assertTrue(path.is_file(), f"{name}: {module} is not a module file")
            tree = ast.parse(path.read_text(encoding="utf-8"))
            defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
            self.assertIn(function, defined, f"{name}: {module} defines no {function}()")

    def test_package_data_covers_every_non_python_file(self):
        patterns = self.data["tool"]["setuptools"]["package-data"]
        uncovered = []
        for package in PACKAGES:
            globs = patterns.get(package, [])
            for path in sorted((ROOT / package).rglob("*")):
                if not path.is_file() or "__pycache__" in path.parts or path.suffix in SKIPPED_SUFFIXES:
                    continue
                relative = path.relative_to(ROOT / package).as_posix()
                if not any(fnmatch.fnmatch(relative, glob) for glob in globs):
                    uncovered.append(f"{package}/{relative}")
        self.assertEqual(uncovered, [], "add these to [tool.setuptools.package-data] or exclude them")
        self.assertFalse(self.data["tool"]["setuptools"].get("include-package-data", True),
                         "package data is listed explicitly; do not rely on SCM discovery")

    def test_core_dependencies_are_ranges_and_extras_mirror_pinned_requirements(self):
        for requirement in self.project["dependencies"]:
            self.assertNotIn("==", requirement, "core dependencies are ranges; exact pins belong to extras")
        extras = self.project["optional-dependencies"]
        self.assertEqual(set(extras["qlib"]), requirement_pins(ROOT / "requirements-qlib-yahoo.txt"))
        self.assertEqual(set(extras["table"]), requirement_pins(ROOT / "requirements-qlib-table.txt"))
        desktop_pins = {pin for pin in requirement_pins(ROOT / "requirements-qlib-macos.txt")
                        | requirement_pins(ROOT / "requirements-qlib-windows.txt") if pin.startswith("pywebview")}
        self.assertEqual(set(extras["desktop"]), desktop_pins)
        self.assertTrue(any(item.startswith("anthropic") for item in extras["research"]))

    def test_installed_distribution_if_present_exposes_the_entry_points(self):
        try:
            distribution = importlib.metadata.distribution("tubby-quant")
        except importlib.metadata.PackageNotFoundError:
            self.skipTest("tubby-quant is not installed in this interpreter")
        self.assertEqual(distribution.version, module_version())
        names = {entry.name for entry in distribution.entry_points if entry.group == "console_scripts"}
        self.assertEqual(names, set(self.project["scripts"]),
                         "installed metadata is stale: run `pip install -e .` again after changing [project.scripts]")


if __name__ == "__main__":
    unittest.main()
