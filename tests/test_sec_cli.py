"""CLI SEC plumbing checks; no data, Qlib initialization or optional dependencies."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from qlib_yahoo import __main__ as cli
from qlib_yahoo.alphalab.cli import load_sec_config
from qlib_yahoo.alphalab.settings import Settings
from qlib_yahoo.workflow_records import EvidenceStore


class SecCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / "selection.json"
        self.selection = {"catalog": "catalog", "bindings": {"sec_assets": {
            "tag": "Assets", "taxonomy": "us-gaap/", "unit": "USD",
            "qtrs": 0, "segments": "", "coreg": ""}}}
        self.config.write_text(json.dumps(self.selection), encoding="utf-8")

    def invoke(self, arguments):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return cli.main(["--data-root", str(self.root), *arguments])

    def test_parser_optional_paths(self):
        for args in (["backtest"], ["alpha", "run", "close"], ["alpha", "batch", "alphas.txt"]):
            with self.subTest(args=args):
                self.assertIsNone(cli.build_parser().parse_args(args).sec_config)
                parsed = cli.build_parser().parse_args([*args, "--sec-config", str(self.config)])
                self.assertEqual(parsed.sec_config, self.config)

    def test_relative_catalog_and_read_only_config(self):
        original = self.config.read_bytes()
        loaded = load_sec_config(self.config)
        self.assertEqual(loaded["catalog"], str((self.root / "catalog").resolve()))
        self.assertEqual(loaded["bindings"], self.selection["bindings"])
        self.assertEqual(self.config.read_bytes(), original)
        self.selection["catalog"] = str(self.root.resolve())
        self.config.write_text(json.dumps(self.selection), encoding="utf-8")
        self.assertEqual(load_sec_config(self.config)["catalog"], str(self.root.resolve()))

    def test_invalid_configs(self):
        bad = [[], None, {}, {"catalog": 3, "bindings": {}},
               {"catalog": " ", "bindings": {}}, {"catalog": "c", "bindings": []},
               {"catalog": "c", "bindings": {"unsafe": self.selection["bindings"]["sec_assets"]}},
               {"catalog": "c", "bindings": {"sec_assets": {"tag": "Assets"}}}]
        for value in bad:
            with self.subTest(value=value):
                self.config.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Invalid SEC config"):
                    load_sec_config(self.config)
        self.config.write_text("{broken", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid SEC config"):
            load_sec_config(self.config)
        with self.assertRaisesRegex(ValueError, "Invalid SEC config"):
            load_sec_config(self.root / "missing.json")

    def test_native_forwarding_and_price_only(self):
        for requested in (False, True):
            with self.subTest(requested=requested), patch.object(cli, "resolve_dataset", return_value=self.root), \
                    patch.object(cli, "run_backtest") as run:
                args = ["backtest", "--alpha", "$close"]
                if requested:
                    args += ["--sec-config", str(self.config)]
                self.assertEqual(self.invoke(args), 0)
                self.assertEqual("sec" in run.call_args.kwargs, requested)
                if requested:
                    self.assertEqual(run.call_args.kwargs["sec"], load_sec_config(self.config))

    def test_bad_config_reports_error_before_native_run(self):
        self.config.write_text("[]", encoding="utf-8")
        with patch.object(cli, "run_backtest") as run, contextlib.redirect_stderr(io.StringIO()) as error:
            self.assertEqual(cli.main(["--data-root", str(self.root), "backtest", "--alpha", "$close",
                                       "--sec-config", str(self.config)]), 1)
            self.assertIn("Invalid SEC config", error.getvalue())
            run.assert_not_called()

    def test_alpha_forwarding_including_unsaved_single(self):
        batch = types.ModuleType("qlib_yahoo.alphalab.batch")
        batch.Library = Mock()
        batch.read_expressions = Mock(return_value=["sec_assets"])
        batch.batch_correlations = Mock(return_value=([], []))
        batch.run_batch = Mock(return_value=[{"id": "fixture", "expression": "sec_assets", "summary": {}}])
        batch.save_batch = Mock(return_value=self.root / "batch.json")
        for command in (["run", "sec_assets"], ["run", "sec_assets", "--save"], ["batch", "alphas.txt"]):
            with self.subTest(command=command), patch.dict("sys.modules", {batch.__name__: batch}), \
                    patch("qlib_yahoo.dataset.resolve_dataset", return_value=self.root), \
                    patch("qlib_yahoo.alphalab.cli.print_result") as print_result:
                batch.run_batch.reset_mock()
                batch.save_batch.reset_mock()
                batch.read_expressions.reset_mock()
                self.assertEqual(self.invoke(["alpha", *command, "--sec-config", str(self.config)]), 0)
                batch.run_batch.assert_called_once()
                self.assertEqual(batch.run_batch.call_args.args[0], self.root)
                self.assertEqual(batch.run_batch.call_args.args[2], Settings())
                self.assertEqual(batch.run_batch.call_args.kwargs["data_root"], self.root)
                self.assertEqual(batch.run_batch.call_args.kwargs["sec"], load_sec_config(self.config))
                if command[0] == "run":
                    self.assertEqual(batch.run_batch.call_args.args[1], ["sec_assets"])
                    self.assertEqual(batch.run_batch.call_args.kwargs["workers"], 1)
                    print_result.assert_called_once_with(batch.run_batch.return_value[0])
                    batch.read_expressions.assert_not_called()
                    batch.save_batch.assert_not_called()
                else:
                    batch.read_expressions.assert_called_once_with(Path("alphas.txt"))
                    self.assertEqual(batch.run_batch.call_args.args[1], batch.read_expressions.return_value)
                    self.assertEqual(batch.run_batch.call_args.kwargs["workers"], 4)
                    self.assertFalse(batch.run_batch.call_args.kwargs["force"])
                    self.assertTrue(callable(batch.run_batch.call_args.kwargs["progress"]))
                    batch.save_batch.assert_called_once_with(
                        self.root, batch.run_batch.return_value, batch.run_batch.call_args.args[2], str(self.root))
                    print_result.assert_not_called()
        for command in (["run", "close", "--save"], ["batch", "alphas.txt"]):
            with self.subTest(price_only=command), patch.dict("sys.modules", {batch.__name__: batch}), \
                    patch("qlib_yahoo.dataset.resolve_dataset", return_value=self.root), \
                    patch("qlib_yahoo.alphalab.cli.print_result"):
                batch.run_batch.reset_mock()
                batch.save_batch.reset_mock()
                batch.read_expressions.return_value = ["close"]
                batch.run_batch.return_value = [{"id": "fixture", "expression": "close", "summary": {}}]
                self.assertEqual(self.invoke(["alpha", *command]), 0)
                batch.run_batch.assert_called_once()
                self.assertEqual(batch.run_batch.call_args.args, (self.root, ["close"], Settings()))
                self.assertEqual(batch.run_batch.call_args.kwargs["data_root"], self.root)
                self.assertNotIn("sec", batch.run_batch.call_args.kwargs)
                self.assertNotIn("group_overlay", batch.run_batch.call_args.kwargs)
                if command[0] == "batch":
                    batch.save_batch.assert_called_once_with(
                        self.root, batch.run_batch.return_value, batch.run_batch.call_args.args[2], str(self.root))
                else:
                    batch.save_batch.assert_not_called()

    def test_price_only_single_uses_shared_batch_path(self):
        batch = types.ModuleType("qlib_yahoo.alphalab.batch")
        for name in ("Library", "batch_correlations", "read_expressions", "run_batch"):
            setattr(batch, name, Mock())
        result = {"id": "fixture", "expression": "close", "summary": {}}
        batch.run_batch.return_value = [result]
        output = self.root / "result.json"
        with patch.dict("sys.modules", {batch.__name__: batch}), \
                patch("qlib_yahoo.dataset.resolve_dataset", return_value=self.root), \
                patch("qlib_yahoo.alphalab.cli.print_result") as print_result:
            self.assertEqual(self.invoke(["alpha", "run", "close", "--json", str(output)]), 0)
            batch.run_batch.assert_called_once_with(
                self.root, ["close"], Settings(), data_root=self.root, workers=1)
            print_result.assert_called_once_with(result)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), result)
            batch.Library.assert_not_called()
            batch.read_expressions.assert_not_called()
        store = EvidenceStore(self.root)
        attempts = store.list_attempts()
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["kind"], "cli-lab")
        self.assertEqual(attempts[0]["status"], "completed")
        self.assertEqual(store.get(attempts[0]["id"])["request"]["data_root"], str(self.root))


if __name__ == "__main__":
    unittest.main()
