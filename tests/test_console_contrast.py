"""Every text and semantic token clears WCAG AA contrast on every surface, in both appearances.

Parsed from app.css so a colour change that breaks readability fails the build:
primary, secondary and tertiary text at 4.5:1; faint text at 3:1; the accent and
the direction/state colours at 4.5:1 as text; white on the primary button at 4.5:1.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

CSS = Path(__file__).resolve().parents[1] / "qlib_yahoo" / "webapp" / "static" / "app.css"
SURFACES = ("--canvas", "--panel", "--center", "--raised", "--overlay")
TEXT = {"--t1": 4.5, "--t2": 4.5, "--t3": 4.5, "--t4": 3.0, "--accent": 4.5, "--pos": 4.5, "--neg": 4.5, "--warn": 4.5}


def luminance(hex_colour: str) -> float:
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lin = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4  # noqa: E731
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def contrast(a: str, b: str) -> float:
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def tokens(selector_pattern: str) -> dict[str, str]:
    css = CSS.read_text(encoding="utf-8")
    match = re.search(selector_pattern + r"\s*\{(.*?)\n\}", css, re.S)
    assert match, selector_pattern
    return dict(re.findall(r"(--[a-z0-9-]+):\s*(#[0-9a-fA-F]{6})", match.group(1)))


class ConsoleContrastTests(unittest.TestCase):
    def check(self, theme: dict[str, str], label: str):
        for name, need in TEXT.items():
            for surface in SURFACES:
                ratio = contrast(theme[name], theme[surface])
                self.assertGreaterEqual(ratio, need, f"{label}: {name} on {surface} is {ratio:.2f}:1, needs {need}:1")
        ratio = contrast("#ffffff", theme["--accent-strong"])
        self.assertGreaterEqual(ratio, 4.5, f"{label}: white on --accent-strong is {ratio:.2f}:1")

    def test_dark_tokens(self):
        self.check(tokens(r"\n:root"), "dark")

    def test_pinned_light_tokens(self):
        self.check(tokens(r"\n:root\[data-theme=light\]"), "light")

    def test_system_light_matches_pinned_light(self):
        self.assertEqual(tokens(r"\n  :root:not\(\[data-theme=dark\]\)"), tokens(r"\n:root\[data-theme=light\]"))


if __name__ == "__main__":
    unittest.main()
