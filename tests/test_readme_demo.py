"""Offline contracts for screenshot preparation; no simulation or downloads."""

import contextlib
import csv
import gzip
import hashlib
import importlib.util
import io
import json
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import struct
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, call, patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_readme_demo.py"
spec = importlib.util.spec_from_file_location("readme_demo_under_test", SCRIPT)
demo = importlib.util.module_from_spec(spec)
# Loading the script must not change the test runner's import search order.
with patch.object(sys, "path", list(sys.path)):
    spec.loader.exec_module(demo)


def module(name, **attributes):
    result = ModuleType(name)
    result.__dict__.update(attributes)
    return result


@dataclass(frozen=True)
class StubSettings:
    universe: str
    delay: int
    decay: int
    neutralization: str
    truncation: float
    cost_bps: float
    warmup: int
    start: str
    end: str

    def validate(self):
        return self


class ReadmeDemoTests(unittest.TestCase):
    def test_existing_output_is_refused_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            sentinel = output / "keep.txt"
            sentinel.write_bytes(b"existing user data")
            with patch.object(demo, "create_gallery_dataset") as create:
                with self.assertRaises(FileExistsError):
                    demo.prepare(output)
            create.assert_not_called()
            self.assertEqual(sentinel.read_bytes(), b"existing user data")
            self.assertEqual(list(output.iterdir()), [sentinel])
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileExistsError):
                demo.prepare(Path(temporary))  # Empty existing folders are refused too.

    def test_source_and_descendant_are_refused_before_any_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary).resolve()
            sentinel = source / "keep.txt"
            sentinel.write_bytes(b"source content")
            with patch.object(demo, "ROOT", source), \
                    patch.object(demo, "create_gallery_dataset") as create:
                for output in (source, source / "new" / "nested"):
                    with self.subTest(output=output), self.assertRaises(ValueError):
                        demo.prepare(output)
            create.assert_not_called()
            self.assertEqual(sentinel.read_bytes(), b"source content")
            self.assertEqual(list(source.iterdir()), [sentinel])

    def test_main_requires_output_and_does_not_prepare_on_help(self):
        for arguments, status in (([], 2), (["--help"], 0)):
            with self.subTest(arguments=arguments), \
                    patch.object(sys, "argv", [str(SCRIPT), *arguments]), \
                    patch.object(demo, "prepare") as prepare, \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    demo.main()
                self.assertEqual(raised.exception.code, status)
                prepare.assert_not_called()

    def test_main_passes_output_and_prints_receipt(self):
        receipt = {"synthetic": True}
        stream = io.StringIO()
        with patch.object(sys, "argv", [str(SCRIPT), "--output", "fresh-gallery"]), \
                patch.object(demo, "prepare", return_value=receipt) as prepare, \
                contextlib.redirect_stdout(stream):
            self.assertEqual(demo.main(), 0)
        prepare.assert_called_once_with(Path("fresh-gallery"))
        self.assertEqual(json.loads(stream.getvalue()), receipt)

    def test_fixture_replicas_metadata_and_membership(self):
        fields = ("open", "high", "low", "close", "volume")
        days = []
        day = date(2023, 1, 2)
        while day < date(2025, 1, 1):
            if day.weekday() < 5:
                days.append(str(day))
            day += timedelta(days=1)

        def template(directory):
            qlib = directory / "qlib_data"
            (qlib / "calendars").mkdir(parents=True)
            (qlib / "calendars/day.txt").write_text("\n".join(days) + "\n", encoding="utf-8")
            for i in range(8):
                folder = qlib / "features" / f"syn{i:02d}"
                folder.mkdir(parents=True)
                for j, field in enumerate(fields):
                    (folder / f"{field}.day.bin").write_bytes(struct.pack("<ff", 0, i * 10 + j))
            return directory

        create = Mock(side_effect=template)
        seal = Mock(return_value=("fixture-tree-hash", 1002))
        verify = Mock()
        imports = {
            "autoresearch.backend": module("autoresearch.backend", create_demo_dataset=create),
            "qlib_yahoo.dataset": module("qlib_yahoo.dataset", tree_sha256=seal),
            "qlib_yahoo.alphalab.panel": module("qlib_yahoo.alphalab.panel", source_identity=verify),
        }
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, imports):
            output = Path(temporary)
            dataset = demo.create_gallery_dataset(output)
            manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(dataset, output / "datasets/SYNTHETIC-DEMO-not-market-data")
            create.assert_called_once_with(output / "template")
            seal.assert_called_once_with(dataset / "qlib_data")
            verify.assert_called_once_with(dataset)
            symbols = [f"SYN{i:03d}" for i in range(200)]
            self.assertEqual(manifest["symbols"], symbols)
            self.assertEqual(manifest["sessions"], len(days))
            self.assertEqual(manifest["date_range"], {
                "start_inclusive": "2023-01-02", "end_exclusive": "2025-01-01"})
            self.assertEqual(manifest["qlib_binary_tree_sha256"], "fixture-tree-hash")
            self.assertEqual(manifest["fields"], list(fields))
            self.assertEqual(manifest["status"], "ready")
            self.assertIn("SYNTHETIC ONLY", manifest["note"])
            self.assertIn("No added diversity", manifest["note"])
            replication = manifest["replication"]
            self.assertEqual((replication["seed"], replication["original_paths"],
                              replication["replicas_per_path"]), (7403, 8, 25))
            self.assertIs(replication["adds_diversity"], False)
            self.assertEqual(Counter(replication["source_symbols"].values()),
                             Counter({f"SYN{i:02d}": 25 for i in range(8)}))
            for symbol, original in replication["source_symbols"].items():
                for field in fields:
                    relative = Path("qlib_data/features")
                    self.assertEqual(
                        (dataset / relative / symbol.lower() / f"{field}.day.bin").read_bytes(),
                        (output / "template" / relative / original.lower() / f"{field}.day.bin").read_bytes())
            instruments = (dataset / "qlib_data/instruments/all.txt").read_text().splitlines()
            self.assertEqual(instruments, [f"{symbol}\t{days[0]}\t{days[-1]}" for symbol in symbols])
            self.assertEqual((dataset / "qlib_data/calendars/day.txt").read_text().split(), days)
            universe = manifest["universes"]["top200"]
            self.assertEqual(set(manifest["universes"]), {"top200"})
            self.assertEqual((universe["size"], universe["unique_members"]), (200, 200))
            self.assertEqual(universe["first_effective"], days[0])
            self.assertEqual(universe["full_size_from"], days[0])
            self.assertEqual(universe["last_effective"], days[-1])
            membership = (dataset / universe["membership_csv"]).read_bytes()
            self.assertEqual(universe["membership_sha256"], hashlib.sha256(membership).hexdigest())
            rows = list(csv.DictReader(io.StringIO(gzip.decompress(membership).decode("utf-8"))))
            self.assertEqual(rows, [{"instrument": symbol, "effective_start": days[0],
                                     "effective_end": days[-1], "rank": str(i + 1)}
                                    for i, symbol in enumerate(symbols)])

    def prepare_with_mocks(self, output, results):
        run = Mock(return_value=results)
        service = Mock()
        constructor = Mock(return_value=service)
        imports = {
            "qlib_yahoo.alphalab.batch": module("qlib_yahoo.alphalab.batch", run_batch=run),
            "qlib_yahoo.alphalab.settings": module("qlib_yahoo.alphalab.settings", Settings=StubSettings),
            "qlib_yahoo.webapp.service": module("qlib_yahoo.webapp.service", ConsoleService=constructor),
        }
        dataset = output / "datasets/SYNTHETIC-DEMO-not-market-data"
        with patch.dict(sys.modules, imports), \
                patch.object(demo, "create_gallery_dataset", return_value=dataset) as create:
            receipt = demo.prepare(output)
        create.assert_called_once_with(output)
        return receipt, run, constructor, service, dataset

    def test_prepare_run_arguments_labels_default_and_receipt(self):
        expressions = ["rank(ts_delta(close, 5))", "-rank(ts_delta(close, 5))"]
        results = [{"id": str(i), "expression": expression, "summary": {"pnl": pnl},
                    "dataset": {"source_sha256": "fixture-identity"}}
                   for i, (expression, pnl) in enumerate(zip(expressions, (100, -200)))]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve() / "fresh"
            receipt, run, constructor, service, dataset = self.prepare_with_mocks(output, results)
            settings = StubSettings("TOP200", 1, 4, "MARKET", 0.01, 5.0, 252,
                                    "2024-01-02", "2024-12-31")
            run.assert_called_once_with(dataset, expressions, settings, data_root=output,
                                        library=output / "alphalab/library.sqlite",
                                        workers=1, worker_ceiling=1)
            constructor.assert_called_once_with(output, settings_path=output / "console-settings.json")
            self.assertEqual(service.annotate.call_args_list, [
                call({"kind": "alphas", "key": "0", "label": "SYNTHETIC / positive result"}),
                call({"kind": "alphas", "key": "1", "label": "SYNTHETIC / negative result"})])
            service.make_default.assert_called_once_with(str(dataset))
            self.assertIs(receipt["synthetic"], True)
            self.assertEqual(receipt["dataset"], results[0]["dataset"])
            self.assertIn("no added diversity", receipt["replication"])
            self.assertEqual(json.loads((output / "screenshot-demo.json").read_text()), receipt)

    def test_failed_or_noncontrasting_results_do_not_publish_receipt(self):
        cases = ([{"error": "simulation failed"}],
                 [{"id": "a", "expression": "fixture", "summary": {"pnl": 1}},
                  {"id": "b", "expression": "fixture", "summary": {"pnl": 0}}])
        for results in cases:
            with self.subTest(results=results), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary).resolve() / "fresh"
                with self.assertRaises(RuntimeError):
                    self.prepare_with_mocks(output, results)
                self.assertFalse((output / "screenshot-demo.json").exists())


if __name__ == "__main__":
    unittest.main()
