"""A campaign is reproducible from its seed, and only from its seed.

Two runs of the alpha demonstration campaign in separate interpreters (different
``PYTHONHASHSEED`` values) must record identical content-addressed hypotheses,
experiments, reviews and finals; a run with a different campaign seed must not.
Both ledgers must verify, and their entries may differ only in ``created_at``,
``prev`` and ``sha256`` (the chain covers timestamps).
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAVE_NUMERIC = bool(importlib.util.find_spec("numpy") and importlib.util.find_spec("pandas"))

# Fields that legitimately differ between two runs of the same science: clocks and locations.
VOLATILE_KEYS = {"created_at", "updated_at", "started_at", "finished_at", "elapsed_seconds",
                 "research_root", "runtime_root", "dataset", "path", "root", "config_path"}


def _scrub(value):
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items() if k not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value


def _records(root: Path, collection: str) -> dict[str, dict]:
    folder = root / "records" / collection
    return {p.name: _scrub(json.loads(p.read_text(encoding="utf-8"))) for p in sorted(folder.glob("*.json"))}


def _ledger_path(root: Path) -> Path:
    (path,) = list((root / "records" / "ledgers").glob("*.jsonl"))
    return path


def _demo_in_subprocess(root: Path, hash_seed: str) -> None:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    subprocess.run([sys.executable, "-m", "autoresearch.lab.cli", "demo", "alpha", "--root", str(root), "--rounds", "2"],
                   cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


@unittest.skipUnless(HAVE_NUMERIC, "numpy and pandas are optional outside the Qlib environment")
class LabReproducibilityTests(unittest.TestCase):
    def test_same_seed_in_separate_processes_records_the_same_science(self):
        from autoresearch.lab.ledger import TrialLedger

        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "first", Path(tmp) / "second"
            _demo_in_subprocess(first, "0")
            _demo_in_subprocess(second, "1")
            for root in (first, second):
                config = json.loads((root / "campaign.json").read_text(encoding="utf-8"))
                self.assertEqual(config["seed"], 3, "the demonstration campaign pins its seed")
            for collection in ("campaigns", "hypotheses", "experiments", "reviews", "finals"):
                a, b = _records(first, collection), _records(second, collection)
                self.assertTrue(a, f"{collection}: nothing recorded")
                self.assertEqual(set(a), set(b), f"{collection}: identifiers differ")
                for key in a:
                    self.assertEqual(a[key], b[key], f"{collection}/{key}: content differs")
            la, lb = TrialLedger(_ledger_path(first)), TrialLedger(_ledger_path(second))
            for ledger in (la, lb):
                verified = ledger.verify()
                self.assertTrue(verified["ok"], verified)
                self.assertEqual(verified["torn_tail_bytes"], 0)
            ea, eb = la.entries(), lb.entries()
            self.assertEqual(len(ea), len(eb))
            self.assertGreater(len(ea), 0)
            for x, y in zip(ea, eb):
                differing = {k for k in set(x) | set(y) if x.get(k) != y.get(k)}
                self.assertTrue(differing <= {"created_at", "prev", "sha256"}, f"ledger entry differs in {sorted(differing)}")
                canonical_x = {k: v for k, v in x.items() if k not in ("created_at", "prev", "sha256")}
                canonical_y = {k: v for k, v in y.items() if k not in ("created_at", "prev", "sha256")}
                self.assertEqual(canonical_x, canonical_y)

    def test_a_different_seed_changes_the_search(self):
        from autoresearch.lab.cli import create_demo, main

        with tempfile.TemporaryDirectory() as tmp:
            base, other = Path(tmp) / "base", Path(tmp) / "other"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["demo", "alpha", "--root", str(base), "--rounds", "2"]), 0)
                path = create_demo("alpha", other, rounds=2)
                config = json.loads(path.read_text(encoding="utf-8"))
                config["seed"] = config["seed"] + 1
                path.write_text(json.dumps(config, indent=2), encoding="utf-8")
                self.assertEqual(main(["run", str(path), "--rounds", "2"]), 0)
            a, b = _records(base, "hypotheses"), _records(other, "hypotheses")
            self.assertTrue(a and b)
            self.assertNotEqual(set(a), set(b), "a different seed must propose a different set of hypotheses")


if __name__ == "__main__":
    unittest.main()
