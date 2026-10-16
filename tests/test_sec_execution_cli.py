"""Mocked SEC execution CLI regression checks; no catalogs, builds or jobs."""
import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import ANY, Mock, call, patch

from qlib_yahoo import __main__ as cli
from qlib_yahoo import sec_catalog, sec_execution, sec_registry


class SecExecutionCliTests(unittest.TestCase):
    def setUp(self):
        self.catalog = Path("mock-sec-catalog")
        self.ready = {"status": "ready", "version": "sec-dictionary-2"}
        self.execution = {"status": "ready", "version": "sec-execution-1",
                          "manifest_sha256": "mock-execution-manifest"}
        self.calls = Mock()
        for module, name in (
                (sec_registry, "build_dictionary"),
                (sec_registry, "dictionary_status"),
                (sec_registry, "search_dictionary"),
                (sec_execution, "build_execution_index"),
                (sec_execution, "execution_status")):
            mocked = self.enterContext(patch.object(module, name))
            setattr(self, name, mocked)
            self.calls.attach_mock(mocked, name)
        # Fresh metadata reflects real reads and prevents mutation between calls.
        self.dictionary_status.side_effect = lambda catalog: dict(self.ready)
        self.execution_status.return_value = self.execution
        self.enterContext(patch.object(
            sec_catalog, "read_catalog", side_effect=AssertionError("Real catalog read")))

    def invoke(self, command, *arguments):
        output, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            result = cli.main(["sec-dictionary", command, "--catalog", str(self.catalog),
                               *arguments])
        return result, output.getvalue(), error.getvalue()

    def assert_metadata(self, result):
        code, output, error = result
        self.assertEqual(code, 0)
        self.assertEqual(error, "")
        self.assertEqual(json.loads(output), dict(self.ready, execution=self.execution))

    def assert_progress_callback(self, builder):
        self.assertTrue(callable(builder.call_args.kwargs["progress"]))

    def test_prepare_execution_dispatches_only_execution_builder(self):
        self.assert_metadata(self.invoke("prepare-execution"))
        self.assertEqual(self.calls.mock_calls, [
            call.build_execution_index(self.catalog, progress=ANY),
            call.dictionary_status(self.catalog),
            call.execution_status(self.catalog)])
        self.assert_progress_callback(self.build_execution_index)
        self.build_dictionary.assert_not_called()
        self.search_dictionary.assert_not_called()

    def test_build_missing_dictionary_builds_dictionary_then_execution(self):
        self.dictionary_status.side_effect = [{"status": "missing"}, dict(self.ready)]
        self.assert_metadata(self.invoke("build"))
        self.assertEqual(self.calls.mock_calls, [
            call.dictionary_status(self.catalog),
            call.build_dictionary(self.catalog, progress=ANY),
            call.build_execution_index(self.catalog, progress=ANY),
            call.dictionary_status(self.catalog),
            call.execution_status(self.catalog)])
        self.assert_progress_callback(self.build_dictionary)
        self.assert_progress_callback(self.build_execution_index)
        self.search_dictionary.assert_not_called()

    def test_build_ready_dictionary_prepares_execution_without_rebuild(self):
        self.assert_metadata(self.invoke("build"))
        self.assertEqual(self.calls.mock_calls, [
            call.dictionary_status(self.catalog),
            call.build_execution_index(self.catalog, progress=ANY),
            call.dictionary_status(self.catalog),
            call.execution_status(self.catalog)])
        self.assert_progress_callback(self.build_execution_index)
        self.build_dictionary.assert_not_called()
        self.search_dictionary.assert_not_called()

    def test_status_reports_execution_metadata_without_building(self):
        for status in ("ready", "missing"):
            with self.subTest(execution_status=status):
                self.calls.reset_mock()
                self.execution_status.return_value = dict(self.execution, status=status)
                code, output, error = self.invoke("status")
                self.assertEqual(code, 0)
                self.assertEqual(error, "")
                self.assertEqual(json.loads(output), dict(
                    self.ready, execution=dict(self.execution, status=status)))
                self.assertEqual(self.calls.mock_calls, [
                    call.dictionary_status(self.catalog),
                    call.execution_status(self.catalog)])
                self.build_dictionary.assert_not_called()
                self.build_execution_index.assert_not_called()
                self.search_dictionary.assert_not_called()

    def test_invalid_arguments_keep_argparse_exit_style_without_dispatch(self):
        invalid = [
            (["sec-dictionary"], "required"),
            (["sec-dictionary", "unknown", "--catalog", str(self.catalog)], "invalid choice"),
        ]
        for command in ("prepare-execution", "build", "status"):
            invalid.append((["sec-dictionary", command], "--catalog"))
        for option in ("--offset", "--limit"):
            invalid.append((["sec-dictionary", "prepare-execution", "--catalog",
                             str(self.catalog), option, "not-an-integer"], "invalid int value"))
        invalid.append((["sec-dictionary", "prepare-execution", "--catalog",
                         str(self.catalog), "--unknown"], "unrecognized arguments"))
        for arguments, message in invalid:
            with self.subTest(arguments=arguments):
                output, error = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
                    with self.assertRaises(SystemExit) as raised:
                        cli.main(arguments)
                self.assertEqual(raised.exception.code, 2)
                self.assertEqual(output.getvalue(), "")
                self.assertIn("usage:", error.getvalue())
                self.assertIn("error:", error.getvalue())
                self.assertIn(message, error.getvalue())
                self.assertEqual(self.calls.mock_calls, [])

    def test_execution_failure_keeps_workflow_error_style(self):
        self.build_execution_index.side_effect = ValueError("Invalid execution parent")
        code, output, error = self.invoke("prepare-execution")
        self.assertEqual(code, 1)
        self.assertEqual(output, "")
        self.assertEqual(error, "Qlib workflow failed: Invalid execution parent\n")
        self.assertEqual(self.calls.mock_calls, [
            call.build_execution_index(self.catalog, progress=ANY)])

    def test_dictionary_build_failure_stops_before_execution(self):
        self.dictionary_status.side_effect = [{"status": "missing"}]
        self.build_dictionary.side_effect = RuntimeError("Dictionary preparation failed")
        code, output, error = self.invoke("build")
        self.assertEqual(code, 1)
        self.assertEqual(output, "")
        self.assertEqual(error, "Qlib workflow failed: Dictionary preparation failed\n")
        self.assertEqual(self.calls.mock_calls, [
            call.dictionary_status(self.catalog),
            call.build_dictionary(self.catalog, progress=ANY)])
        self.build_execution_index.assert_not_called()


if __name__ == "__main__":
    unittest.main()
