"""Metadata-only recovery guards; no market data, worker runs or downloads."""

from __future__ import annotations

import builtins
import http.client
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from qlib_yahoo.webapp.server import start_server
from qlib_yahoo.webapp.service import ConsoleService, ServiceError
from qlib_yahoo.workspace import dataset_summary, list_datasets, list_runs, set_default_dataset


HAVE_NUMPY = importlib.util.find_spec("numpy") is not None


class RecoveryServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.dataset = self.root / "datasets" / "stooq-ordinary"
        self.dataset.mkdir(parents=True)
        self.manifest = {"schema_version": 1, "status": "ready", "provider": "Stooq",
                         "symbols": ["AAA"], "sessions": 5}
        self.save_manifest()
        (self.root / "latest.json").write_text(json.dumps({"dataset": str(self.dataset)}), encoding="utf-8")
        self.factory = Mock(side_effect=AssertionError("no real worker in a recovery test"))
        self.service = ConsoleService(self.root, job_factory=self.factory)

    def save_manifest(self):
        (self.dataset / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def run_fixture(self, changes=None):
        run = self.root / "runs" / "saved-run"
        (run / "reports" / "existing").mkdir(parents=True)
        manifest = {"schema_version": 6, "status": "completed", "alpha": "$close",
                    "dataset": str(self.dataset), "backtest": {"execution_contract": {}}}
        manifest.update(changes or {})
        (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (run / "summary.json").write_text(json.dumps({"metrics": {"strategy_net": {"sharpe": 1.5}}}), encoding="utf-8")
        (run / "reports" / "existing" / "report_index.html").write_text("saved", encoding="utf-8")
        (run / "daily_report.csv").write_text("large detail must not be read", encoding="utf-8")
        return run

    def assert_no_jobs(self):
        self.factory.assert_not_called()
        self.assertIsNone(self.service.current)
        self.assertEqual(self.service.jobs, {})

    def test_ordinary_missing_fields_infers_contract_without_heavy_imports(self):
        original_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("numpy", "qlib") or name in (
                    "qlib_yahoo.dataset_integrity", "qlib_yahoo.alphalab.panel"):
                raise AssertionError(f"metadata imported heavy dependency {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded):
            summary = dataset_summary(self.root, self.dataset)
        self.assertTrue(summary["price_eligible"])
        self.assertEqual(summary["fields"], ["open", "high", "low", "close", "volume", "factor"])
        self.assertEqual(summary["data_role"], "prices")
        self.assertEqual(summary["storage_kind"], "qlib-binary-tree")
        self.assertEqual(summary["compatibility_note"], "")
        self.assertEqual(self.service.dataset(str(self.dataset))["fields"], summary["fields"])

    def test_packed_and_explicit_optional_fields_are_supported(self):
        self.manifest.update(storage={"format": "qlib-zip-store-v1"},
                             fields=["open", "high", "low", "close", "volume", "factor", "cap"])
        self.save_manifest()
        summary = dataset_summary(self.root, self.dataset)
        self.assertTrue(summary["price_eligible"])
        self.assertEqual(summary["storage_kind"], "qlib-zip-store-v1")
        self.assertIn("cap", summary["fields"])

    def test_schema_less_offline_fixture_remains_supported(self):
        self.manifest.pop("schema_version")
        self.manifest["provider"] = "synthetic"
        self.save_manifest()
        request, _, path = self.service._backtest_request({"alpha": "$close"})
        self.assertEqual(path, self.dataset)
        self.assertEqual(request["dataset"], str(self.dataset))

    def test_pilot_and_known_unsupported_metadata_are_ineligible_and_preserved(self):
        variants = ({"fundamentals": {}}, {"fundamentals": None}, {"data_role": "alpha-inputs"},
                    {"dataset_role": "alpha-inputs"}, {"data_role": "combined"},
                    {"schema_version": 2}, {"schema_version": True},
                    {"alpha_input_storage": {"format": "pilot"}}, {"storage": {"format": "pilot"}},
                    {"fields": ["assets"]}, {"fields": "close"})
        baseline = dict(self.manifest)
        pointer = (self.root / "latest.json").read_bytes()
        for changes in variants:
            with self.subTest(changes=changes):
                self.manifest = {**baseline, **changes}
                self.save_manifest()
                before = (self.dataset / "manifest.json").read_bytes()
                summary = dataset_summary(self.root, self.dataset)
                self.assertFalse(summary["price_eligible"])
                self.assertIn("preserved, not deleted", summary["compatibility_note"])
                self.assertIn("rebuild-pending", summary["compatibility_note"])
                self.assertFalse(list_datasets(self.root)[0]["price_eligible"])
                for action in (lambda: self.service.start_backtest({"alpha": "$close"}),
                               lambda: self.service.start_alpha({"expressions": "rank(close)"}),
                               lambda: self.service.make_default(str(self.dataset))):
                    with self.assertRaisesRegex(ServiceError, "unsupported"):
                        action()
                self.assertEqual((self.dataset / "manifest.json").read_bytes(), before)
                self.assertEqual((self.root / "latest.json").read_bytes(), pointer)
                self.assert_no_jobs()

    def test_missing_unready_and_outside_default_do_not_queue_jobs(self):
        for status in ("running", "failed", "incomplete"):
            self.manifest["status"] = status
            self.save_manifest()
            with self.assertRaisesRegex(ServiceError, "ready"):
                self.service.start_backtest({"alpha": "$close"})
            with self.assertRaisesRegex(ServiceError, "ready"):
                self.service.start_alpha({"expressions": "rank(close)"})
        (self.dataset / "manifest.json").unlink()
        row = list_datasets(self.root)[0]
        self.assertFalse(row["price_eligible"])
        for key in ("fields", "data_role", "storage_kind", "compatibility_note"):
            self.assertIn(key, row)
        with self.assertRaisesRegex(ServiceError, "readable prepared"):
            self.service.start_alpha({"expressions": "rank(close)"})
        (self.root / "latest.json").write_text(json.dumps({"dataset": str(self.root / "outside")}), encoding="utf-8")
        with self.assertRaisesRegex(ServiceError, "outside"):
            self.service.start_backtest({"alpha": "$close"})
        (self.root / "latest.json").unlink()
        with self.assertRaisesRegex(ServiceError, "No prepared dataset"):
            self.service.start_alpha({"expressions": "rank(close)"})
        self.assert_no_jobs()

    def test_default_requires_eligible_prices(self):
        set_default_dataset(self.root, self.dataset)
        self.assertEqual(json.loads((self.root / "latest.json").read_text())["dataset"], str(self.dataset))
        self.manifest["data_role"] = "alpha-inputs"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "unsupported"):
            set_default_dataset(self.root, self.dataset)

    def test_stale_execution_and_classification_options_are_rejected(self):
        for key, value in (("classification", ""), ("classification_path", "pilot"), ("price_dataset", str(self.dataset)),
                           ("alpha_dialect", "fastexpr"), ("fast_delay", 0), ("fast_settings", {}),
                           ("alpha_settings", {}), ("execution_phase", "close"), ("reuse_run", "saved-run")):
            with self.subTest(key=key):
                for action in (lambda: self.service.start_backtest({"alpha": "$close", key: value}),
                               lambda: self.service.start_alpha({"expressions": "rank(close)", key: value})):
                    with self.assertRaisesRegex(ServiceError, "Unsupported PR45"):
                        action()
        self.assert_no_jobs()

    def test_actual_backtest_constraints_and_zero_settings(self):
        request, _, _ = self.service._backtest_request({"alpha": "$close", "strategy_mode": "signed",
                                                       "gross_exposure": 0, "borrow_rate": 0, "cost_bps": 0})
        self.assertEqual(request["gross_exposure"], 0)
        self.assertEqual(request["annual_borrow_rate"], 0)
        self.assertEqual(request["cost_per_side"], 0)
        request, _, _ = self.service._backtest_request({"alpha": "$close", "n_drop": 0})
        self.assertEqual(request["n_drop"], 0)
        for settings in ({"topk": 0}, {"warmup_sessions": 0}, {"topk": 2, "n_drop": 3},
                         {"cash_rate": "nan"}, {"cash_rate": -1}, {"cost_bps": "inf"},
                         {"strategy_mode": "signed", "gross_exposure": 1.1},
                         {"strategy_mode": "signed", "signal_transform": "bad"},
                         {"start": "2024-01-01", "end": "2023-01-01"}):
            with self.subTest(settings=settings), self.assertRaises(ServiceError):
                self.service.start_backtest({"alpha": "$close", **settings})
        self.assert_no_jobs()

    def test_newer_runs_stay_listed_but_reports_and_detail_reads_are_guarded(self):
        run = self.run_fixture({"schema_version": 7})
        before = (run / "manifest.json").read_bytes()
        with patch.object(self.service, "_series", side_effect=AssertionError("detailed ledger read")), \
                patch.object(self.service, "_latest_holdings", side_effect=AssertionError("position read")):
            row = self.service.runs()[0]
            self.assertTrue(row["read_only"])
            self.assertFalse(row["reports_supported"])
            self.assertFalse(row["reuse_supported"])
            self.assertEqual(row["sharpe"], 1.5)
            detail = self.service.run_detail(str(run))
            self.assertTrue(detail["read_only"])
            self.assertEqual(detail["reports"], [])
            self.assertEqual(detail["settings"], {})
        for action in (lambda: self.service.start_report(str(run), "overview"),
                       lambda: self.service.run_signal(str(run)),
                       lambda: self.service.report_file("saved-run/reports/existing/report_index.html")):
            with self.assertRaisesRegex(ServiceError, "listed read-only"):
                action()
        self.assertEqual((run / "manifest.json").read_bytes(), before)
        self.assert_no_jobs()

    def test_pr45_schema6_markers_are_read_only_too(self):
        run = self.run_fixture({"data_sources": {"version": "alpha-price-roles-v1"}})
        self.assertTrue(list_runs(self.root)[0]["read_only"])
        with self.assertRaisesRegex(ServiceError, "PR45"):
            self.service.start_report(str(run), "holdings")
        self.assert_no_jobs()

    def test_boot_lists_never_open_saved_detail_artifacts(self):
        self.run_fixture()
        original_read = Path.read_text
        original_open = Path.open

        def read(path, *args, **kwargs):
            self.assertNotIn(path.name, ("daily_report.csv", "positions.csv.gz", "positions.parquet", "signed_decisions.json.gz"))
            return original_read(path, *args, **kwargs)

        def open_file(path, *args, **kwargs):
            self.assertNotIn(path.name, ("daily_report.csv", "positions.csv.gz", "positions.parquet", "signed_decisions.json.gz"))
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "read_text", new=read), patch.object(Path, "open", new=open_file):
            self.service.datasets()
            self.service.runs()


    @unittest.skipUnless(HAVE_NUMPY, "numpy is optional outside the Qlib environment")
    def test_optional_dataset_guard_and_cheap_syntax_path(self):
        with patch("qlib_yahoo.dataset_integrity.verify_dataset", side_effect=AssertionError("full verification")), \
                patch("qlib_yahoo.dataset.sha256", side_effect=AssertionError("file hashing")), \
                patch("qlib_yahoo.panel.load_panel", side_effect=AssertionError("matrix loading")):
            self.assertTrue(self.service.alpha_check("rank(close)")["ok"])
            self.assertTrue(self.service.alpha_check("rank(close) + ts_mean(returns, 5)", str(self.dataset))["ok"])
            self.assertFalse(self.service.alpha_check("rank(cap)", str(self.dataset))["ok"])
            self.assertFalse(self.service.alpha_check("group_rank(close, sector)", str(self.dataset))["ok"])
            self.assertTrue(self.service.alpha_check("group_rank(close, market)", str(self.dataset))["ok"])
            self.assertFalse(self.service.alpha_check("unknown(close)")["ok"])
            self.assertFalse(self.service.alpha_check("# comment only")["ok"])
        self.assert_no_jobs()

    @unittest.skipUnless(HAVE_NUMPY, "numpy is optional outside the Qlib environment")
    def test_sequential_bindings_do_not_hide_earlier_field_reads(self):
        for expression in ("a = cap; cap = close; rank(a)", "a = sector; sector = close; rank(a)"):
            with self.subTest(expression=expression):
                checked = self.service.alpha_check(expression, str(self.dataset))
                self.assertFalse(checked["ok"])
        self.assertTrue(self.service.alpha_check("cap = close; rank(cap)", str(self.dataset))["ok"])
        self.assertTrue(self.service.alpha_check("a = close; a = rank(a); a", str(self.dataset))["ok"])

    @unittest.skipUnless(HAVE_NUMPY, "numpy is optional outside the Qlib environment")
    def test_current_and_older_static_group_headers(self):
        for group_record in ({"file": "universe/groups.csv", "fields": ["sector", "industry", "country"]},
                             {"file": "universe/groups.csv"},
                             {"file": "universe/groups.csv", "columns": ["symbol", "sector", "industry", "country"]}):
            with self.subTest(record=group_record):
                self.manifest["groups"] = group_record
                self.save_manifest()
                self.assertTrue(self.service.alpha_check("group_rank(close, sector)", str(self.dataset))["ok"])
                self.assertFalse(self.service.alpha_check("group_rank(close, 'sector')", str(self.dataset))["ok"])

    def test_neutralization_rejects_missing_group_headers_without_loading_group_data(self):
        with patch("qlib_yahoo.reference.dataset_groups", side_effect=AssertionError("group data verification")), \
                patch.object(self.service, "_start", return_value={"ok": True}) as start:
            for neutralization in ("SECTOR", "INDUSTRY", "COUNTRY"):
                with self.subTest(neutralization=neutralization), self.assertRaisesRegex(ServiceError, "needs groups"):
                    self.service.start_alpha({"expressions": "rank(close)", "neutralization": neutralization})
            start.assert_not_called()
        self.assert_no_jobs()

    @unittest.skipUnless(HAVE_NUMPY, "numpy is optional outside the Qlib environment")
    def test_neutralization_accepts_group_headers_without_loading_group_data(self):
        with patch("qlib_yahoo.reference.dataset_groups", side_effect=AssertionError("group data verification")), \
                patch.object(self.service, "_start", return_value={"ok": True}):
            for neutralization in ("MARKET", "NONE"):
                self.service.start_alpha({"expressions": "rank(close)", "neutralization": neutralization})
            self.manifest["groups"] = {"file": "universe/groups.csv"}
            self.save_manifest()
            self.service.start_alpha({"expressions": "rank(close)", "neutralization": "SECTOR"})
        self.assert_no_jobs()

    @unittest.skipUnless(HAVE_NUMPY, "numpy is optional outside the Qlib environment")
    def test_invalid_expressions_never_queue_verification(self):
        with patch("qlib_yahoo.dataset_integrity.verify_dataset", side_effect=AssertionError("full verification")):
            for expression in ("foo(close)", "rank(cap)", "group_rank(close, sector)", "rank($close)"):
                with self.subTest(expression=expression), self.assertRaises(ServiceError):
                    self.service.start_alpha({"expressions": expression})
        self.assert_no_jobs()

    def test_invalid_settings_never_queue_verification(self):
        with patch("qlib_yahoo.dataset_integrity.verify_dataset", side_effect=AssertionError("full verification")):
            for settings in ({"delay": 2}, {"workers": 0}, {"workers": 65}, {"decay": -1},
                             {"truncation": 1.1}, {"cost_bps": -1}, {"warmup": 2001},
                             {"pasteurization": "false"}, {"delay": True}, {"neutralization": "bad"}):
                with self.subTest(settings=settings), self.assertRaises(ServiceError):
                    self.service.start_alpha({"expressions": "rank(close)", **settings})
        self.assert_no_jobs()

    @unittest.skipUnless(HAVE_NUMPY, "numpy is optional outside the Qlib environment")
    def test_valid_lab_settings_are_forwarded_and_worker_remains_responsible(self):
        with patch.object(self.service, "_start", return_value={"ok": True}) as start:
            self.service.start_alpha({"expressions": "rank(close)", "delay": 0, "decay": 4,
                                      "warmup": 0, "cost_bps": 0, "truncation": 0,
                                      "pasteurization": False, "workers": 3})
        request = start.call_args.args[2]
        self.assertEqual(request["kind"], "alpha-batch")
        self.assertEqual(request["dataset"], str(self.dataset))
        self.assertEqual(request["workers"], 3)
        self.assertEqual(request["settings"]["delay"], 0)
        self.assertFalse(request["settings"]["pasteurization"])
        self.assertEqual(request["settings"]["warmup"], 0)
        self.factory.assert_not_called()

    @unittest.skipUnless(HAVE_NUMPY, "numpy is optional outside the Qlib environment")
    def test_batch_response_preserves_stored_old_new_and_missing_engine(self):
        library = self.service._alpha_library()
        expected = {"old": "alphalab-1", "new": "alphalab-2", "unversioned": None, "missing": None}
        for identifier in ("old", "new", "unversioned"):
            result = {"expression": "rank(close)", "settings": {}, "summary": {}}
            if expected[identifier] is not None:
                result["engine"] = expected[identifier]
            library.save(identifier, result, "synthetic-dataset")
        batch = self.root / "alphalab" / "batch.json"
        batch.write_text(json.dumps({"ids": list(expected), "settings": {},
                                     "created_at_utc": "2026-10-09T00:00:00Z"}), encoding="utf-8")

        response = self.service.alpha_batch(str(batch))

        self.assertEqual({row["id"]: row["engine"] for row in response["rows"]}, expected)
        self.assertEqual(response["rows"][-1]["error"], "not in the library")
        self.assert_no_jobs()

    def http_connection(self):
        server, thread = start_server(self.service)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1])
        self.addCleanup(connection.close)
        headers = {"Cookie": f"tqc={server.token}", "X-Token": server.token, "Origin": server.origin,
                   "Content-Type": "application/json"}
        return connection, headers

    @unittest.skipUnless(HAVE_NUMPY, "numpy is optional outside the Qlib environment")
    def test_http_alpha_check_passes_dataset(self):
        connection, headers = self.http_connection()
        self.manifest["fundamentals"] = {}
        self.save_manifest()
        with patch.object(self.service, "alpha_check", wraps=self.service.alpha_check) as check:
            connection.request("POST", "/api/alpha-check", json.dumps({"text": "rank(close)", "dataset": str(self.dataset)}), headers)
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            self.assertIn("unsupported", response.read().decode())
            check.assert_called_once_with("rank(close)", str(self.dataset))
        self.assert_no_jobs()

    def test_http_alpha_check_refuses_stale_options(self):
        connection, headers = self.http_connection()
        for endpoint, body in (("alpha-check", {"text": "rank(close)", "classification": ""}),
                               ("check-alpha", {"alpha": "$close", "alpha_dialect": "fastexpr"})):
            connection.request("POST", "/api/" + endpoint, json.dumps(body), headers)
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            self.assertIn("Unsupported PR45", response.read().decode())
        self.assert_no_jobs()


if __name__ == "__main__":
    unittest.main()
