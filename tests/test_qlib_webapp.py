"""Desktop workbench service and loopback HTTP server: validation, jobs, safety and security."""

from __future__ import annotations

import csv
import gzip
import http.client
import importlib.util
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from qlib_yahoo.webapp.server import start_server
from qlib_yahoo.webapp.service import ConsoleService, ServiceError, snap_dates
from qlib_yahoo.workspace import ALL_SYMBOLS, list_runs, parse_backtest_options, parse_import_options


class FakeJob:
    """Stands in for the worker process; records the request and replays scripted output."""

    created: list = []
    release = threading.Event()  # cleared by a test that needs the job to stay running

    def __init__(self, request):
        self.request = request
        self.cancelled = False
        self.constituents = []
        FakeJob.created.append(self)

    def cancel(self):
        self.cancelled = True
        return True

    def run(self, on_line):
        if self.request.get("kind") == "backtest-batch":
            from qlib_yahoo.preparation import run_backtest_batch
            request = dict(self.request)
            request.pop("kind")
            request.pop("alpha", None)
            dataset, alphas = Path(request.pop("dataset")), request.pop("alphas")
            with patch("qlib_yahoo.backtest.run_backtest", side_effect=self.simulate_constituent):
                return run_backtest_batch(dataset, alphas, **request)
        on_line("Verifying dataset example ...")
        on_line("backtest loop:  50%|█████     | 5/10")
        on_line("Backtest completed: somewhere")
        self.release.wait(5)
        if self.request.get("alpha") == "$open":
            from qlib_yahoo.jobs import JobCancelled

            while not self.cancelled:
                time.sleep(0.01)
            raise JobCancelled("Cancelled.")
        return Path(self.request["dataset"]).parent.parent / "runs" / "run-1"

    def simulate_constituent(self, dataset, alpha, **settings):
        self.constituents.append({"dataset": dataset, "alpha": alpha, **settings})
        return Path(dataset).parent.parent / "runs" / f"run-{len(self.constituents)}"


def make_dataset(root: Path, name: str = "stooq-table-top2-1", *, universe: bool = True) -> Path:
    dataset = root / "datasets" / name
    (dataset / "qlib_data" / "calendars").mkdir(parents=True)
    (dataset / "universe").mkdir()
    (dataset / "qlib_data" / "calendars" / "day.txt").write_text(
        "2024-01-02\n2024-01-03\n2024-01-05\n2024-01-08\n2024-01-09\n", encoding="utf-8")
    manifest = {"status": "ready", "provider": "synthetic", "symbols": ["AAA", "BBB"], "sessions": 5,
                "date_range": {"start_inclusive": "2024-01-02", "end_exclusive": "2024-01-10"},
                "security_filter": {"common_stocks": 2, "excluded": {}}, "limitations": ["synthetic"]}
    if universe:
        manifest["universes"] = {"top2": {"size": 2, "adv_sessions": 5, "min_history_sessions": 2,
                                          "full_size_from": "2024-01-03", "unique_members": 2}}
        (dataset / "universe" / "top2_rankings.csv").write_text(
            "ranked_on,effective_start,effective_end,eligible,members,cutoff_adv\n2024-01-03,2024-01-05,2024-01-09,2,2,1.0\n",
            encoding="utf-8")
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return dataset


def make_run(root: Path, name: str = "20260926T000000Z-abc") -> Path:
    run = root / "runs" / name
    (run / "reports" / "r1").mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({
        "status": "completed", "alpha": "$close", "strategy_mode": "long-only", "created_at_utc": "2026-09-26T00:00:00",
        "dataset": str(root / "datasets" / "x"), "backtest": {"start": "2024-01-03", "end": "2024-01-08",
        "universe": {"name": "top2"}, "execution_contract": {"topk": 1, "n_drop": 1, "open_cost": 0.0005},
        "warmup_sessions": 2, "cash": 100000, "trade_symbols": ["AAA", "BBB"]}}), encoding="utf-8")
    (run / "summary.json").write_text(json.dumps({"metrics": {"strategy_net": {"sharpe": 1.2, "cagr": 0.1, "max_drawdown": -0.05}}}), encoding="utf-8")
    with (run / "daily_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["datetime", "account", "net_return", "cost", "cost_dollars", "turnover", "cash", "holdings_value"])
        for day, account in (("2024-01-03", 100000), ("2024-01-05", 110000), ("2024-01-08", 99000)):
            writer.writerow([day, account, 0, 0, 5, 0.1, 0, account])
    with gzip.open(run / "positions.csv.gz", "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["datetime", "instrument", "amount", "price", "value", "weight", "cash", "account"])
        writer.writerow(["2024-01-08", "AAA", 1, 1, 1, 0.6, 0, 1])
        writer.writerow(["2024-01-08", "BBB", 1, 1, 1, -0.3, 0, 1])
    (run / "reports" / "r1" / "report_index.html").write_text("<html>report</html>", encoding="utf-8")
    (run / "reports" / "r1" / "report_manifest.json").write_text(json.dumps({"status": "completed", "level": "overview"}), encoding="utf-8")
    return run


class WorkspaceParsingTests(unittest.TestCase):
    def test_defaults_add_nothing(self):
        self.assertEqual(parse_backtest_options({}, "long-only"), {})
        self.assertEqual(parse_backtest_options({"universe": ALL_SYMBOLS, "topk": "5", "n_drop": "1", "cost_bps": "5",
                                                 "warmup_sessions": "252"}, "long-only"), {})
        self.assertEqual(parse_backtest_options({"borrow_rate": "0.004"}, "signed"), {})

    def test_changed_settings_and_errors(self):
        self.assertEqual(parse_backtest_options({"universe": "top3000", "topk": "50", "n_drop": "5", "cost_bps": "10",
                                                 "start": "2018-01-02", "end": "2026-01-02", "warmup_sessions": "60"}, "long-only"),
                         {"universe": "top3000", "topk": 50, "n_drop": 5, "cost_per_side": 0.001, "start": "2018-01-02",
                          "end": "2026-01-02", "warmup_sessions": 60})
        self.assertEqual(parse_backtest_options({"topk": "50", "borrow_rate": "0.01"}, "signed"), {"annual_borrow_rate": 0.01})
        for options, message in (({"topk": "0"}, "Top-k"), ({"n_drop": "9", "topk": "5"}, "exceed"), ({"start": "2018-1-2"}, "Start"),
                                 ({"start": "2020-01-02", "end": "2019-01-02"}, "before"), ({"cost_bps": "nan"}, "Cost"),
                                 ({"universe": "../etc"}, "universe"), ({"universe": "all"}, "universe")):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                parse_backtest_options(options, "long-only")

    def test_import_options(self):
        with tempfile.TemporaryDirectory() as temporary:
            table = Path(temporary) / "t.parquet"
            table.write_bytes(b"x")
            request = parse_import_options(str(table), "2010-01-01", "2026-09-26", "3000")
            self.assertEqual(request["rule"], {"name": "top3000", "size": 3000, "adv_sessions": 20, "min_history_sessions": 252,
                                              "min_price": 0.0})
            with self.assertRaisesRegex(ValueError, "parquet"):
                parse_import_options(str(Path(temporary) / "missing.parquet"), "2010-01-01", "2026-09-26", "3000")
            with self.assertRaisesRegex(ValueError, "before"):
                parse_import_options(str(table), "2026-01-01", "2010-01-01", "3000")
            with self.assertRaisesRegex(ValueError, "Universe name"):
                parse_import_options(str(table), "2010-01-01", "2026-09-26", "3000", name="Top 3000")


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.dataset = make_dataset(self.root)
        self.run = make_run(self.root)
        (self.root / "latest.json").write_text(json.dumps({"dataset": str(self.dataset)}), encoding="utf-8")
        FakeJob.created = []
        FakeJob.release.set()
        self.service = ConsoleService(self.root, job_factory=FakeJob, settings_path=self.root / "settings.json")

    def wait(self, job_id):
        deadline = time.time() + 5
        while time.time() < deadline:
            job = self.service.job(job_id)
            if job["state"] != "running":
                return job
            time.sleep(0.01)
        self.fail("job did not finish")

    def test_backtest_job_lifecycle_and_date_snapping(self):
        FakeJob.release.clear()
        job = self.service.start_backtest({"alpha": "$close/Ref($close,1)-1", "strategy_mode": "long-only", "universe": "top2",
                                           "start": "2024-01-04", "end": "2024-01-09", "topk": "1", "n_drop": "1"})
        with self.assertRaisesRegex(ServiceError, "Another task"):
            self.service.start_backtest({"alpha": "$close", "strategy_mode": "long-only"})
        FakeJob.release.set()
        done = self.wait(job["id"])
        self.assertEqual(done["state"], "completed")
        self.assertEqual(done["progress"], 0.5)
        request = FakeJob.created[0].request
        self.assertEqual((request["start"], request["end"], request["universe"], request["topk"]), ("2024-01-05", "2024-01-08", "top2", 1))
        self.assertTrue(any("not a session" in line for line in done["log"]))
        self.assertEqual(self.service.job(job["id"], since=done["log_index"])["log"], [])

    def test_signed_request_shape_and_cancel(self):
        job = self.service.start_backtest({"alpha": "$open", "strategy_mode": "signed", "gross_exposure": "0.5", "borrow_rate": "0.01"})
        time.sleep(0.05)
        self.assertTrue(self.service.cancel())
        self.assertEqual(self.wait(job["id"])["state"], "cancelled")
        request = FakeJob.created[0].request
        self.assertEqual((request["strategy_mode"], request["gross_exposure"], request["annual_borrow_rate"]), ("signed", 0.5, 0.01))
        self.assertEqual(request["signal_transform"], "raw")
        FakeJob.release.set()
        job = self.service.start_backtest({"alpha": "$close", "strategy_mode": "signed", "signal_transform": "zscore"})
        self.wait(job["id"])
        self.assertEqual(FakeJob.created[-1].request["signal_transform"], "zscore")
        with self.assertRaisesRegex(ServiceError, "Weighting"):
            self.service.start_backtest({"alpha": "$close", "strategy_mode": "signed", "signal_transform": "softmax"})

    def test_validation_errors_do_not_start_jobs(self):
        for params, message in (({"alpha": "gibberish here"}, "alpha syntax"),
                                ({"alpha": "$close", "strategy_mode": "short"}, "Choose"),
                                ({"alpha": "$close", "strategy_mode": "signed", "gross_exposure": "2"}, "Gross"),
                                ({"alpha": "$close", "dataset": "/etc"}, "outside the data folder")):
            with self.subTest(params=params), self.assertRaisesRegex(ServiceError, message):
                self.service.start_backtest(params)
        self.assertEqual(FakeJob.created, [])

    def test_paths_stay_inside_the_data_folder(self):
        for bad in ("/etc/passwd", "../../etc", str(self.root.parent)):
            with self.subTest(bad=bad), self.assertRaises(ServiceError):
                self.service.run_detail(bad)
        with self.assertRaises(ServiceError):
            self.service.report_file("../../latest.json")
        (self.run / "alpha_scores.csv.gz").write_bytes(b"scores")
        with self.assertRaises(ServiceError):
            self.service.report_file(f"{self.run.name}/alpha_scores.csv.gz")  # not inside reports/, not linked
        # Saved-data files that reports link to ("../../summary.json") are served; nothing else in the run is.
        self.assertEqual(self.service.report_file(f"{self.run.name}/manifest.json").name, "manifest.json")
        self.assertEqual(self.service.report_file(f"{self.run.name}/reports/r1/report_index.html").name, "report_index.html")

    def test_a_new_stage_clears_the_finished_progress_bar(self):
        class StagedJob(FakeJob):
            def run(self, on_line):
                on_line("Simulating 2024-01-05 to 2024-01-08 (long-only) ...")
                on_line("backtest loop: 100%|██████████| 10/10")
                on_line("Rendering the summary report ...")
                progress.append(service.current.progress)
                return Path(self.request["dataset"]).parent.parent / "runs" / "run-1"

        progress, service = [], self.service
        service.job_factory = StagedJob
        self.wait(service.start_backtest({"alpha": "$close", "strategy_mode": "long-only"})["id"])
        self.assertEqual(progress, [None])

    def test_symlinked_data_root_behaves_like_the_real_folder(self):
        link = self.root.parent / f"{self.root.name}-link"
        try:
            link.symlink_to(self.root, target_is_directory=True)
        except OSError as error:
            if sys.platform == "win32" and error.winerror == 1314:
                self.skipTest("Windows symlink privilege is unavailable")
            raise
        self.addCleanup(link.unlink)
        service = ConsoleService(link, job_factory=FakeJob, settings_path=self.root / "linked.json")
        self.assertEqual(service.overview()["data_root"], str(self.root))
        report = service.run_detail(str(link / "runs" / self.run.name))["reports"][0]
        self.assertTrue(str(report["path"]).startswith(str(self.root / "runs")))
        (self.root / "latest.json").write_text(json.dumps({"dataset": str(link / "datasets" / self.dataset.name)}), encoding="utf-8")
        self.assertTrue(any(row["is_default"] for row in service.datasets()))

    def test_read_models(self):
        overview = self.service.overview()
        self.assertEqual(overview["default_dataset"]["name"], self.dataset.name)
        self.assertTrue(overview["examples"])
        dataset = self.service.dataset(None)
        self.assertEqual(dataset["ranking_series"]["top2"][0]["members"], 2)
        self.assertEqual(dataset["audit"]["limitations"], ["synthetic"])
        detail = self.service.run_detail(str(self.run))
        self.assertEqual(detail["universe"], "top2")
        self.assertEqual(detail["series"]["account"], [100000.0, 110000.0, 99000.0])
        self.assertAlmostEqual(detail["series"]["drawdown"][-1], 99000 / 110000 - 1)
        self.assertEqual([h["instrument"] for h in detail["holdings"]["top"]], ["AAA", "BBB"])
        self.assertEqual(detail["reports"][0]["url"], f"/files/{self.run.name}/reports/r1/report_index.html")
        self.assertEqual(list_runs(self.root)[0]["sharpe"], 1.2)
        self.assertEqual(self.service.check_alpha("$close")["ok"], True)
        self.assertEqual(self.service.check_alpha("os.system('x')")["ok"], False)

    def test_short_windows_are_flagged(self):
        from qlib_yahoo.report_service import MIN_OBSERVATIONS
        from qlib_yahoo.workspace import SHORT_WINDOW_SESSIONS

        self.assertEqual(SHORT_WINDOW_SESSIONS, MIN_OBSERVATIONS)
        row = list_runs(self.root)[0]  # an older summary without a session count: shown, not flagged
        self.assertEqual((row["sessions"], row["short_window"]), (None, False))
        self.assertFalse(self.service.run_detail(str(self.run))["short_window"])
        for sessions, short in ((20, True), (251, True), (252, False), (2300, False)):
            with self.subTest(sessions=sessions):
                (self.run / "summary.json").write_text(json.dumps({"metrics": {"strategy_net": {
                    "sharpe": 5.42, "cagr": 3.65, "max_drawdown": -0.038, "observations": sessions}}}), encoding="utf-8")
                row = list_runs(self.root)[0]
                self.assertEqual((row["sessions"], row["short_window"]), (sessions, short))
                self.assertEqual(self.service.run_detail(str(self.run))["short_window"], short)

    def test_market_data_refresh_and_import_options(self):
        self.assertEqual(self.service.latest_sources(), {"table": None, "screener": None})
        job = self.service.start_refresh({"min_price": "2", "make_default": False})
        self.wait(job["id"])
        request = FakeJob.created[-1].request
        self.assertEqual((request["kind"], request["min_price"], request["make_latest"], request["previous_tables"]),
                         ("refresh-data", 2.0, False, []))
        with self.assertRaisesRegex(ServiceError, "Minimum price"):
            self.service.start_refresh({"min_price": "-1"})
        table = self.root / "sources" / "yahoo-table-20260928T000000Z" / "table.parquet"
        table.parent.mkdir(parents=True)
        table.write_bytes(b"PAR1")
        screener = self.root / "sources" / "nasdaq-screener-20260928T000000Z"
        screener.mkdir()
        (screener / "manifest.json").write_text("{}", encoding="utf-8")
        (screener / "screener.json.gz").write_bytes(b"")
        self.assertEqual(self.service.latest_sources(), {"table": str(table), "screener": str(screener)})
        job = self.service.start_refresh({})
        self.wait(job["id"])
        self.assertEqual(FakeJob.created[-1].request["previous_tables"], [str(table)])
        options = parse_import_options(str(table), "2020-01-01", "2021-01-01", "3000", min_price="1.5", groups=str(screener))
        self.assertEqual((options["rule"]["min_price"], options["groups"]), (1.5, str(screener)))
        self.assertEqual(parse_import_options(str(table), "2020-01-01", "2021-01-01", "3000")["rule"]["min_price"], 0.0)
        with self.assertRaisesRegex(ValueError, "fetch-nasdaq-screener"):
            parse_import_options(str(table), "2020-01-01", "2021-01-01", "3000", groups=str(self.root))
        with self.assertRaisesRegex(ValueError, "Minimum price"):
            parse_import_options(str(table), "2020-01-01", "2021-01-01", "3000", min_price="-3")

    @unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("pandas"),
                         "numpy and pandas are optional outside the Qlib environment")
    def test_alpha_lab_endpoints(self):
        from qlib_yahoo.alphalab.batch import Library

        # This request explicitly exercises SECTOR. Declare the static-group
        # header, as a real imported dataset does; FakeJob does no data loading.
        manifest_path = self.dataset / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["groups"] = {"file": "universe/groups.csv", "fields": ["sector"]}
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        checked = self.service.alpha_check("rank(close)\n# note\nfoo(close)\nrank(nope)")
        self.assertEqual([(line["line"], line["ok"]) for line in checked["lines"]], [(1, True), (3, False), (4, False)])
        self.assertIn("unknown operator foo()", checked["lines"][1]["message"])
        self.assertIn("unknown field 'nope'", checked["lines"][2]["message"])
        with self.assertRaisesRegex(ServiceError, "at least one alpha"):
            self.service.start_alpha({"expressions": "# only a comment"})
        with self.assertRaisesRegex(ServiceError, "delay"):
            self.service.start_alpha({"expressions": "rank(close)", "delay": "2"})
        job = self.service.start_alpha({"expressions": "rank(close)\n-rank(close)", "universe": "TOP1000", "decay": "4",
                                        "neutralization": "sector", "workers": "3", "pasteurization": False})
        self.wait(job["id"])
        request = FakeJob.created[-1].request
        self.assertEqual(request["kind"], "alpha-batch")
        self.assertEqual(request["expressions"], ["rank(close)", "-rank(close)"])
        self.assertEqual((request["settings"]["universe"], request["settings"]["decay"], request["settings"]["neutralization"],
                          request["settings"]["pasteurization"], request["workers"]), ("TOP1000", 4, "SECTOR", False, 3))
        self.assertEqual(request["dataset"], str(self.dataset))
        # A saved result and a batch record are read back for the workbench.
        library = Library(self.root / "alphalab" / "library.sqlite")
        days = [f"2024-01-{d:02d}" for d in range(2, 31)] * 3
        daily = {"dates": [f"{i:04d}" for i in range(len(days))], "pnl": [float((i % 7) - 3) for i in range(len(days))]}
        for identifier, sign in (("a1", 1.0), ("a2", -1.0)):
            library.save(identifier, {"expression": f"{sign} * rank(close)", "settings": {"universe": "TOP3000", "delay": 1},
                                      "summary": {"sharpe": sign, "fitness": sign}, "checks": [], "yearly": [],
                                      "daily": {"dates": daily["dates"], "pnl": [sign * v for v in daily["pnl"]]}}, "hash")
        detail = self.service.alpha_result("a1")
        self.assertEqual(detail["series"]["cumulative"][:3], [-3.0, -5.0, -6.0])
        self.assertNotIn("daily", detail)
        batch = self.root / "alphalab" / "batches" / "b.json"
        batch.parent.mkdir(parents=True)
        batch.write_text(json.dumps({"created_at_utc": "now", "settings": {"universe": "TOP3000"}, "ids": ["a1", "a2"],
                                     "errors": {}}), encoding="utf-8")
        board = self.service.alpha_batch(str(batch))
        self.assertEqual([row["id"] for row in board["rows"]], ["a1", "a2"])
        self.assertAlmostEqual(board["top_pairs"][0]["correlation"], -1.0)
        self.assertEqual([row["id"] for row in self.service.alpha_library("fitness")], ["a1", "a2"])
        with self.assertRaisesRegex(ServiceError, "outside the data folder"):
            self.service.alpha_batch(str(self.root / "runs" / "x.json"))
        with self.assertRaisesRegex(ServiceError, "Unknown alpha"):
            self.service.alpha_result("missing")
        self.assertTrue(self.service.overview()["lab_examples"].lstrip().startswith("#"))


    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "pyarrow is optional outside the Qlib environment")
    def test_holdings_read_identically_from_parquet(self):
        import pandas as pd

        from_csv = self.service._latest_holdings(self.run)
        frame = pd.read_csv(self.run / "positions.csv.gz", keep_default_na=False, parse_dates=["datetime"])
        frame.to_parquet(self.run / "positions.parquet", index=False)
        from_parquet = self.service._latest_holdings(self.run)
        self.assertEqual(from_parquet, from_csv)
        self.assertEqual([h["instrument"] for h in from_parquet["top"]], ["AAA", "BBB"])

    def test_settings_roundtrip_and_default_dataset(self):
        self.service.save_settings({"alpha": "$close", "nested": {"x": 1}})
        self.assertEqual(self.service.settings(), {"alpha": "$close"})
        second = make_dataset(self.root, "stooq-table-top2-2")
        self.service.make_default(str(second))
        self.assertEqual(json.loads((self.root / "latest.json").read_text())["dataset"], str(second))

    def test_snap_dates(self):
        request = {"start": "2024-01-04", "end": "2024-01-20"}
        notes = snap_dates(self.dataset, request)
        # Start moves forward to a session; end moves back and keeps one later session for the runner.
        self.assertEqual((request["start"], request["end"]), ("2024-01-05", "2024-01-08"))
        self.assertEqual(len(notes), 2)
        unchanged = {"start": "2024-01-03"}
        self.assertEqual(snap_dates(self.dataset, unchanged), [])
        with self.assertRaises(ServiceError):
            snap_dates(self.dataset, {"start": "2024-01-09"})
        with self.assertRaises(ServiceError):
            snap_dates(self.dataset, {"start": "2024-01-08", "end": "2024-01-08"})


class OrganizeTests(unittest.TestCase):
    """Stars, labels, the recoverable run trash, hidden alphas, full daily series and signal diagnostics."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.dataset = make_dataset(self.root)
        self.run = make_run(self.root)
        self.other = make_run(self.root, "20260926T000001Z-def")
        (self.root / "latest.json").write_text(json.dumps({"dataset": str(self.dataset)}), encoding="utf-8")
        self.service = ConsoleService(self.root, job_factory=FakeJob)

    def row(self, name):
        return next(r for r in self.service.runs() if r["name"] == name)

    def test_stars_labels_and_factor_notes(self):
        self.service.annotate({"kind": "runs", "key": self.run.name, "starred": True, "label": "  first   try "})
        row = self.row(self.run.name)
        self.assertEqual((row["starred"], row["label"]), (True, "first try"))
        self.assertEqual(row["factor"], self.row(self.other.name)["factor"])  # same alpha, same factor
        self.service.annotate({"kind": "factors", "key": row["factor"], "starred": True, "label": "close level"})
        other = self.row(self.other.name)
        self.assertEqual((other["starred"], other["factor_starred"], other["factor_label"]), (False, True, "close level"))
        detail = self.service.run_detail(str(self.run))
        self.assertTrue(detail["starred"] and detail["factor_starred"])
        self.service.annotate({"kind": "runs", "key": self.run.name, "starred": False, "label": ""})
        notes = json.loads((self.root / "workbench" / "annotations.json").read_text())
        self.assertNotIn(self.run.name, notes["runs"])  # empty entries are removed
        for bad, message in (({"kind": "runs", "key": "../x", "starred": True}, "Unknown run"),
                             ({"kind": "factors", "key": "nothex", "starred": True}, "Invalid annotation key"),
                             ({"kind": "runs", "key": self.run.name, "starred": "yes"}, "true or false"),
                             ({"kind": "runs", "key": self.run.name, "label": "x" * 121}, "limited"),
                             ({"kind": "other", "key": "x"}, "Unknown annotation kind")):
            with self.subTest(bad=bad), self.assertRaisesRegex(ServiceError, message):
                self.service.annotate(bad)

    def test_trash_restore_and_system_trash(self):
        from unittest.mock import patch

        moved = self.service.trash_runs([str(self.run)])
        self.assertEqual(moved, {"trashed": [self.run.name]})
        self.assertFalse(self.run.exists())
        self.assertNotIn(self.run.name, [r["name"] for r in self.service.runs()])
        trash = self.service.trash()
        self.assertEqual([r["name"] for r in trash["runs"]], [self.run.name])
        self.assertTrue(trash["runs"][0]["trashed_at"])
        with self.assertRaisesRegex(ServiceError, "outside the data folder"):
            self.service.run_detail(str(self.root / "trash" / "runs" / self.run.name))
        self.assertEqual(self.service.restore_runs([self.run.name]), {"restored": [self.run.name]})
        self.assertTrue((self.run / "manifest.json").is_file())
        # Emptying moves folders to the operating system's trash, never deletes them here.
        self.service.trash_runs([str(self.run), str(self.other)])
        system = self.root / "system-trash"
        system.mkdir()
        with patch("qlib_yahoo.desktop.move_to_system_trash", side_effect=lambda path: path.rename(system / path.name)):
            self.assertEqual(sorted(self.service.empty_trash([self.run.name])["emptied"]), [self.run.name])
            self.assertTrue((system / self.run.name / "manifest.json").is_file())
            self.assertEqual(self.service.empty_trash(None)["emptied"], [self.other.name])
        self.assertEqual(self.service.trash()["runs"], [])
        for bad in ([], [str(self.root / "datasets" / "x")], [str(self.root / "runs")], "not a list"):
            with self.subTest(bad=bad), self.assertRaises(ServiceError):
                self.service.trash_runs(bad)

    def test_a_run_being_written_is_not_trashed(self):
        manifest = json.loads((self.run / "manifest.json").read_text())
        manifest["status"] = "running"
        (self.run / "manifest.json").write_text(json.dumps(manifest))
        FakeJob.release.clear()
        self.addCleanup(FakeJob.release.set)
        job = self.service.start_backtest({"alpha": "$close", "strategy_mode": "long-only"})
        with self.assertRaisesRegex(ServiceError, "still being written"):
            self.service.trash_runs([str(self.run)])
        FakeJob.release.set()
        deadline = time.time() + 5
        while self.service.job(job["id"])["state"] == "running" and time.time() < deadline:
            time.sleep(0.02)
        self.assertEqual(self.service.trash_runs([str(self.run)]), {"trashed": [self.run.name]})  # stale once idle

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "the alpha library needs numpy (Qlib environment)")
    def test_hidden_alphas_and_restore(self):
        from qlib_yahoo.alphalab.batch import Library

        library = Library(self.root / "alphalab" / "library.sqlite")
        ids = ["0123456789abcdef", "fedcba9876543210"]
        for identifier in ids:
            library.save(identifier, {"expression": f"rank(close) # {identifier}", "settings": {"universe": "TOP3000", "delay": 1},
                                      "summary": {"sharpe": 1.0, "fitness": 1.0}, "checks": [], "yearly": [],
                                      "daily": {"dates": ["2024-01-02"], "pnl": [1.0]}}, "hash")
        self.service.annotate({"kind": "alphas", "key": ids[0], "starred": True, "label": "mine"})
        rows = {r["id"]: r for r in self.service.alpha_library()}
        self.assertEqual((rows[ids[0]]["starred"], rows[ids[0]]["label"]), (True, "mine"))
        self.service.alpha_delete([ids[1]])
        self.assertEqual([r["id"] for r in self.service.alpha_library()], [ids[0]])
        self.assertEqual([r["id"] for r in self.service.trash()["alphas"]], [ids[1]])
        self.assertIsNotNone(self.service.alpha_result(ids[1]))  # hidden, never removed from the library
        self.service.alpha_delete([ids[1]], deleted=False)
        self.assertEqual(len(self.service.alpha_library()), 2)
        with self.assertRaisesRegex(ServiceError, "Unknown alpha"):
            self.service.alpha_delete(["0000000000000000"])

    def test_backtest_batch_shares_preparation_worker_with_independent_alphas(self):
        FakeJob.release.set()
        FakeJob.created.clear()
        checked = self.service.check_alpha("$close\n# a comment\n-1 * $close\n\n$open / $close")
        self.assertEqual((checked["ok"], checked["count"]), (True, 3))
        self.assertIn("Line 2", self.service.check_alpha("$close\nbad(")["message"])
        with self.assertRaisesRegex(ServiceError, "Line 3"):
            self.service.start_backtest({"alpha": "$close\n\nnope(", "strategy_mode": "long-only"})
        job = self.service.start_backtest({"alpha": "$close\n-1 * $close\n$high / $low", "strategy_mode": "long-only",
                                           "start": "2024-01-04"})
        deadline = time.time() + 10
        while self.service.job(job["id"])["state"] == "running" and time.time() < deadline:
            time.sleep(0.02)
        done = self.service.job(job["id"])
        self.assertEqual(done["state"], "completed")
        self.assertEqual(len(FakeJob.created), 1)
        worker = FakeJob.created[0]
        self.assertEqual(worker.request["kind"], "backtest-batch")
        self.assertEqual(worker.request["alphas"], ["$close", "-1 * $close", "$high / $low"])
        self.assertEqual([call["alpha"] for call in worker.constituents], worker.request["alphas"])
        self.assertTrue(all(call["start"] == "2024-01-05" for call in worker.constituents))
        self.assertTrue(all(call.get("strategy_mode", "long-only") == "long-only" for call in worker.constituents))
        self.assertEqual(len({id(call["_preparation"]) for call in worker.constituents}), 1)
        batch = self.service.backtest_batch(done["result"])
        self.assertEqual(len(batch["runs"]), 3)
        self.assertEqual(batch["errors"], [])
        self.assertTrue(Path(done["result"]).is_relative_to(self.root / "workbench" / "batches"))

    def test_a_failing_alpha_does_not_stop_the_batch(self):
        class Flaky(FakeJob):
            def simulate_constituent(self, dataset, alpha, **settings):
                result = super().simulate_constituent(dataset, alpha, **settings)
                if alpha == "$open":
                    raise DatasetError("warmup too short")
                return result

        from qlib_yahoo.dataset import DatasetError

        FakeJob.release.set()
        service = ConsoleService(self.root, job_factory=Flaky)
        job = service.start_backtest({"alpha": "$close\n$open\n$high", "strategy_mode": "long-only"})
        deadline = time.time() + 10
        while service.job(job["id"])["state"] == "running" and time.time() < deadline:
            time.sleep(0.02)
        batch = service.backtest_batch(service.job(job["id"])["result"])
        self.assertEqual((len(batch["runs"]), [e["alpha"] for e in batch["errors"]]), (2, ["$open"]))
        worker = FakeJob.created[-1]
        self.assertEqual([call["alpha"] for call in worker.constituents], ["$close", "$open", "$high"])
        self.assertEqual(worker.request["kind"], "backtest-batch")

    def test_full_daily_series_for_the_result_view(self):
        series = self.service.run_detail(str(self.run))["series"]
        self.assertEqual(series["dates"], ["2024-01-03", "2024-01-05", "2024-01-08"])
        self.assertEqual(series["net_return"], [0.0, 0.0, 0.0])
        self.assertEqual(series["holdings"], [100000.0, 110000.0, 99000.0])
        self.assertAlmostEqual(series["drawdown"][-1], 99000 / 110000 - 1)
        self.assertNotIn("long", series)

    @unittest.skipIf(sys.platform.startswith("win"), "the Recycle Bin call is exercised on Windows machines")
    def test_system_trash_keeps_the_folder(self):
        import os
        from unittest.mock import patch

        from qlib_yahoo.desktop import move_to_system_trash

        home = self.root / "home"
        (home / ".Trash").mkdir(parents=True)
        with patch("pathlib.Path.home", return_value=home), patch.dict(os.environ, {"XDG_DATA_HOME": str(home / "share")}):
            move_to_system_trash(self.other)
            clash = make_run(self.root, self.other.name)  # the same name again: kept side by side
            move_to_system_trash(clash)
        where = home / ".Trash" if sys.platform == "darwin" else home / "share" / "Trash" / "files"
        kept = sorted(p.name for p in where.iterdir())
        self.assertEqual(len(kept), 2)
        self.assertTrue(all((where / name / "manifest.json").is_file() for name in kept))
        self.assertFalse(self.other.exists())

    @unittest.skipUnless(importlib.util.find_spec("pyarrow") and importlib.util.find_spec("pandas"), "pandas and pyarrow are optional")
    def test_signal_diagnostics_are_computed_and_cached(self):
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(3)
        days = pd.bdate_range("2024-01-02", periods=60)
        rows = []
        for day in days:
            score = rng.normal(size=40)
            label = 0.002 * score + rng.normal(scale=0.01, size=40)
            rows += [{"datetime": day, "instrument": f"S{i}", "score": score[i], "label": label[i]} for i in range(40)]
        pd.DataFrame(rows).to_parquet(self.run / "signal_forward_labels.parquet", index=False)
        self.assertTrue(self.service.run_detail(str(self.run))["signal_available"])
        result = self.service.run_signal(str(self.run))
        summary = result["summary"]
        self.assertEqual(len(result["dates"]), 60)
        self.assertGreater(summary["rank_ic"]["mean"], 0.1)
        self.assertGreater(result["quantiles"]["mean"][-1], result["quantiles"]["mean"][0])
        self.assertEqual(set(result["daily"]), {"q1", "q2", "q3", "q4", "q5"})
        cached = list((self.root / "workbench" / "cache").glob("signal-*.json"))
        self.assertEqual(len(cached), 1)
        self.assertEqual(self.service.run_signal(str(self.run)), json.loads(cached[0].read_text()))
        with self.assertRaisesRegex(ServiceError, "no saved forward labels"):
            self.service.run_signal(str(self.other))


class ServerSecurityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name).resolve()
        make_dataset(root)
        self.run = make_run(root)
        self.server, thread = start_server(ConsoleService(root, job_factory=FakeJob))
        self.addCleanup(self.server.server_close)
        self.addCleanup(thread.join, 5)
        self.addCleanup(self.server.shutdown)
        self.port = self.server.server_address[1]

    def request(self, method, path, *, headers=None, body=None, host=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        all_headers = {"Host": host or f"127.0.0.1:{self.port}", **(headers or {})}
        try:
            # Keep stdlib framing/encoding, but send these small requests together.
            # An early HTTP/1.0 rejection can close before a separate body send
            # reaches it, hiding the response behind a Windows connection abort.
            outgoing = []
            with patch.object(connection, "send", side_effect=outgoing.append):
                connection.request(method, path, body=body, headers=all_headers)
            connection.send(b"".join(outgoing))
            with connection.getresponse() as response:
                data = response.read()
            return response, data
        finally:
            connection.close()

    def cookie(self):
        response, _ = self.request("GET", f"/?token={self.server.token}")
        self.assertEqual(response.status, 303)
        cookie = response.getheader("Set-Cookie")
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Strict", cookie)
        return {"Cookie": cookie.split(";")[0]}

    def test_token_cookie_and_static(self):
        self.assertEqual(self.request("GET", "/")[0].status, 403)
        self.assertEqual(self.request("GET", "/?token=wrong")[0].status, 403)
        self.assertEqual(self.request("GET", "/?token=%C3%A9")[0].status, 403)  # non-ASCII is refused, not a crash
        cookie = self.cookie()
        response, data = self.request("GET", "/", headers=cookie)
        self.assertEqual(response.status, 200)
        self.assertIn("script-src 'self'", response.getheader("Content-Security-Policy"))
        self.assertIn(b"Tubby Qlib", data)
        self.assertEqual(self.request("GET", "/static/app.js", headers=cookie)[0].status, 200)
        self.assertEqual(self.request("GET", "/static/../server.py", headers=cookie)[0].status, 404)
        self.assertEqual(self.request("GET", "/static/%2e%2e/server.py", headers=cookie)[0].status, 404)

    def test_host_origin_and_request_token(self):
        cookie = self.cookie()
        self.assertEqual(self.request("GET", "/", headers=cookie, host="evil.example")[0].status, 421)
        body = json.dumps({"alpha": "$close"})
        self.assertEqual(self.request("POST", "/api/check-alpha", headers=cookie, body=body)[0].status, 403)
        headers = {**cookie, "X-Token": self.server.token, "Content-Type": "application/json"}
        self.assertEqual(self.request("POST", "/api/check-alpha", headers={**headers, "Origin": "http://evil.example"}, body=body)[0].status, 403)
        response, data = self.request("POST", "/api/check-alpha", headers={**headers, "Origin": self.server.origin}, body=body)
        self.assertEqual(response.status, 200)
        self.assertTrue(json.loads(data)["ok"])
        self.assertEqual(self.request("POST", "/api/run-anything", headers=headers, body=body)[0].status, 404)

    def test_rejected_post_sends_headers_and_body_together(self):
        cookie = self.cookie()
        body = b'{"alpha": "$close"}'
        original_send = http.client.HTTPConnection.send
        with patch.object(http.client.HTTPConnection, "send", autospec=True, side_effect=original_send) as send:
            response, data = self.request("POST", "/api/check-alpha", headers=cookie, body=body)
        self.assertEqual(response.status, 403)
        self.assertEqual(json.loads(data), {"error": "Missing request token"})
        send.assert_called_once()
        request = send.call_args.args[1]
        head, sent_body = request.split(b"\r\n\r\n", 1)
        self.assertTrue(head.startswith(b"POST /api/check-alpha HTTP/1.1\r\n"))
        self.assertIn(f"Content-Length: {len(body)}".encode("ascii"), head)
        self.assertEqual(sent_body, body)

    def test_api_and_report_files(self):
        cookie = self.cookie()
        response, data = self.request("GET", "/api/overview", headers=cookie)
        self.assertEqual(response.status, 200)
        self.assertIn("examples", json.loads(data))
        self.assertEqual(self.request("GET", "/api/overview")[0].status, 403)
        self.assertEqual(self.request("GET", f"/files/{self.run.name}/reports/r1/report_index.html", headers=cookie)[0].status, 200)
        self.assertEqual(self.request("GET", f"/files/{self.run.name}/manifest.json", headers=cookie)[0].status, 200)
        self.assertEqual(self.request("GET", f"/files/{self.run.name}/settings.json", headers=cookie)[0].status, 404)
        self.assertEqual(self.request("GET", "/files/../../latest.json", headers=cookie)[0].status, 404)
        response, data = self.request("GET", "/api/run?path=/etc", headers=cookie)
        self.assertEqual(response.status, 400)
        self.assertIn("outside", json.loads(data)["error"])


if __name__ == "__main__":
    unittest.main()
