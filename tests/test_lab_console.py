"""Discover: the console service for research-domain campaigns (alpha discovery, machine learning).

Offline: the worker process is replaced by a scripted fake that prints what
``tubby-lab run`` prints, so no dataset is evaluated. One optional test runs the
real worker on a small artificial panel when TUBBY_LAB_INTEGRATION=1.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

from qlib_yahoo.webapp.lab_service import DOMAINS, PRESETS, LabService


def make_prices(root: Path, name: str = "demo-prices", sessions: int = 800) -> Path:
    """A ready price snapshot with a calendar and a dated TOP-N universe; no bars are read here."""
    import datetime as dt

    dataset = root / "datasets" / name
    (dataset / "qlib_data" / "calendars").mkdir(parents=True)
    (dataset / "qlib_data" / "instruments").mkdir()
    (dataset / "universe").mkdir()
    day, days = dt.date(2016, 1, 4), []
    while len(days) < sessions:
        if day.weekday() < 5:
            days.append(day.isoformat())
        day += dt.timedelta(days=1)
    (dataset / "qlib_data" / "calendars" / "day.txt").write_text("\n".join(days) + "\n", encoding="utf-8")
    symbols = [f"S{i:02d}" for i in range(12)]
    members = "".join(f"{s}\t{days[0]}\t{days[-1]}\n" for s in symbols)
    (dataset / "qlib_data" / "instruments" / "all.txt").write_text(members, encoding="utf-8")
    (dataset / "qlib_data" / "instruments" / "top12.txt").write_text(members, encoding="utf-8")
    csv = gzip.compress(("instrument,effective_start,effective_end,rank\n"
                         + "".join(f"{s},{days[0]},{days[-1]},{i + 1}\n" for i, s in enumerate(symbols))).encode(), mtime=0)
    (dataset / "universe" / "membership.csv.gz").write_bytes(csv)
    manifest = {"status": "ready", "provider": "fixture", "symbols": symbols, "fields": ["open", "high", "low", "close", "volume"],
                "date_range": {"start_inclusive": days[0], "end_exclusive": days[-1]}, "sessions": sessions,
                "universes": {"top12": {"size": 12, "membership_csv": "universe/membership.csv.gz",
                                        "membership_sha256": hashlib.sha256(csv).hexdigest(),
                                        "instruments_file": "instruments/top12.txt", "min_history_sessions": 60,
                                        "instruments_sha256": hashlib.sha256(members.encode()).hexdigest()}}}
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return dataset


class FakeProcess:
    """Prints scripted lines like the lab CLI, then exits; records the command it was given."""

    def __init__(self, command, lines, code=0, delay=0.0):
        self.command, self.returncode, self._delay = command, None, delay
        self._code = code
        self.stdout = io.StringIO("".join(line + "\n" for line in lines))
        self._terminated = False

    def poll(self):
        if self.returncode is None and self.stdout.tell() >= len(self.stdout.getvalue()) and not self._delay:
            self.returncode = self._code
        return self.returncode

    def wait(self, timeout=None):
        if self._delay:
            time.sleep(self._delay)
            self._delay = 0.0
        self.returncode = self._code
        return self.returncode

    def terminate(self):
        self._terminated = True
        self._delay = 0.0

    def kill(self):
        self.terminate()


class LabServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dataset = make_prices(self.root)
        self.spawned = []
        self.script = []

        def spawn(command):
            process = FakeProcess(command, list(self.script))
            self.spawned.append(process)
            return process

        self.service = LabService(self.root, spawn=spawn)

    def tearDown(self):
        self.service.close()
        self.tmp.cleanup()

    def wait_idle(self, limit=5.0):
        deadline = time.time() + limit
        while time.time() < deadline:
            if not self.service.worker()["running"]:
                return
            time.sleep(0.02)
        self.fail("worker did not finish")

    def test_catalog_lists_domains_presets_tasks_and_the_factor_library(self):
        catalog = self.service.catalog()
        self.assertTrue(catalog["available"])
        self.assertEqual([d["name"] for d in catalog["domains"]], ["alpha", "ml"])
        self.assertEqual(set(catalog["presets"]), set(DOMAINS))
        self.assertEqual(set(catalog["presets"]["alpha"]), {"quick", "standard", "thorough"})
        self.assertIn("panel", [t["id"] for t in catalog["ml_tasks"]])
        self.assertGreaterEqual(len(catalog["ml_features"]), 20)
        self.assertEqual(catalog["ml_features"][0]["name"], "ret_1")

    def test_alpha_plan_cuts_sealed_periods_from_the_calendar_and_validates(self):
        plan = self.service.plan({"domain": "alpha", "name": "My search", "preset": "standard", "dataset": "demo-prices",
                                  "universe": "TOP12", "cost_bps": 7.5, "min_sharpe": 1.25, "seeds": "-rank(ts_delta(close, 5))\n\n# comment"})
        config, summary = plan["config"], plan["summary"]
        self.assertEqual(config["domain"], "alpha")
        self.assertEqual(config["limits"]["max_rounds"], PRESETS["alpha"]["standard"]["limits"]["max_rounds"])
        self.assertEqual(config["settings"]["universe"], "TOP12")
        self.assertEqual(config["settings"]["search"]["seeds"], ["-rank(ts_delta(close, 5))"])
        self.assertEqual(config["policy"], {"min_sharpe": 1.25})
        periods = summary["periods"]
        self.assertLess(periods["development"]["end"], periods["validation"]["start"])
        self.assertLess(periods["validation"]["end"], periods["holdout"]["start"])
        self.assertEqual(config["settings"]["holdout_start"], periods["holdout"]["start"])
        self.assertEqual(sum(p["sessions"] for p in periods.values()), 800 - summary["warmup"])
        self.assertTrue(config["research_root"].endswith(os.path.join("lab", "alpha", "records")))
        # A smaller TOP-N cut of the dataset's universe is allowed; an unknown one is not.
        self.assertEqual(self.service.plan({"domain": "alpha", "dataset": "demo-prices", "universe": "top5"})["summary"]["universe"], "TOP5")
        with self.assertRaisesRegex(ValueError, "not available"):
            self.service.plan({"domain": "alpha", "dataset": "demo-prices", "universe": "TOP500"})
        with self.assertRaisesRegex(ValueError, "sessions"):
            make_prices(self.root, "short", sessions=200)
            self.service.plan({"domain": "alpha", "dataset": "short"})
        with self.assertRaisesRegex(ValueError, "preset"):
            self.service.plan({"domain": "alpha", "dataset": "demo-prices", "preset": "huge"})

    def test_ml_plans_cover_builtin_csv_and_panel_factor_analysis(self):
        builtin = self.service.plan({"domain": "ml", "task": "builtin:friedman1", "rows": 1000, "families": ["ridge", "gbdt"]})
        self.assertEqual(builtin["config"]["settings"]["table"], "builtin:friedman1")
        self.assertEqual(builtin["config"]["settings"]["search"]["families"], ["ridge", "gbdt"])
        table = self.root / "table.csv"
        table.write_text("a,b,y\n1,2,3\n", encoding="utf-8")
        csv = self.service.plan({"domain": "ml", "task": "csv", "table": str(table), "target": "y", "order_by": "a"})
        self.assertEqual(csv["config"]["settings"]["target"], "y")
        self.assertEqual(csv["config"]["settings"]["order_by"], "a")
        with self.assertRaisesRegex(ValueError, "target"):
            self.service.plan({"domain": "ml", "task": "csv", "table": str(table)})
        panel = self.service.plan({"domain": "ml", "task": "panel", "dataset": "demo-prices", "features": ["ret_5", "vol_20"],
                                   "extra_features": "rank(ts_delta(close, 3))"})
        settings = panel["config"]["settings"]
        self.assertEqual(settings["task"], "panel")
        self.assertEqual(settings["features"], ["ret_5", "vol_20"])
        self.assertEqual(settings["extra_features"], ["rank(ts_delta(close, 3))"])
        self.assertIn("development_start", settings)
        self.assertEqual(panel["summary"]["features"], 2)
        every = self.service.plan({"domain": "ml", "task": "panel", "dataset": "demo-prices"})
        self.assertEqual(every["config"]["settings"]["features"], "default")
        with self.assertRaisesRegex(ValueError, "unknown operator"):
            self.service.plan({"domain": "ml", "task": "panel", "dataset": "demo-prices", "extra_features": "nosuchop(close)"})
        with self.assertRaisesRegex(ValueError, "library"):
            self.service.plan({"domain": "ml", "task": "panel", "dataset": "demo-prices", "features": ["not_a_factor"]})

    def test_create_lists_runs_records_the_campaign_id_and_reports(self):
        row = self.service.create({"domain": "alpha", "name": "Smoke Test!", "preset": "quick", "dataset": "demo-prices"})
        self.assertTrue(row["id"].startswith("alpha/smoke-test-"))
        self.assertEqual(row["status"], "not started")
        self.assertIsNone(row["campaign_id"])
        path = self.root / "autoresearch" / "lab" / "alpha" / "campaigns" / (row["id"].split("/")[1] + ".json")
        self.assertTrue(path.is_file())
        from autoresearch.lab.campaign import load_lab_config
        self.assertEqual(load_lab_config(path)["name"], "Smoke Test!")
        self.assertEqual([r["id"] for r in self.service.campaigns()], [row["id"]])
        with self.assertRaisesRegex(ValueError, "Run the campaign once"):
            self.service.submit({"id": row["id"], "expression": "rank(close)"})
        self.assertIsNone(self.service.report(row["id"])["report"])
        with self.assertRaisesRegex(ValueError, "Run the campaign first"):
            self.service.verify(row["id"])

        self.script = [
            json.dumps({"campaign_id": "campaign_abc123", "status": "round", "round": 1, "proposed": 24, "screened": 20, "screen_ok": 19,
                        "submitted": 0, "validated": 2, "promoted": 1, "best_fitness": 1.5, "decisions": [{"decision": "promote", "score": 0.4}]}),
            "a stray log line with /private/path",
            json.dumps({"campaign_id": "campaign_abc123", "status": "stopped", "round": 1, "screen_trials": 20, "validation_trials": 2,
                        "promotions": 1, "budget": None}),
        ]
        worker = self.service.start({"id": row["id"], "rounds": 1})
        self.assertTrue(worker["running"])
        self.assertEqual(worker["action"], "run")
        self.wait_idle()
        command = self.spawned[-1].command
        self.assertEqual(command[1:4], ["-m", "autoresearch.lab.cli", "run"])
        self.assertEqual(command[-2:], ["--rounds", "1"])
        worker = self.service.worker()
        self.assertEqual(worker["status"], "finished")
        self.assertEqual((worker["round"], worker["screened"], worker["validated"], worker["promoted"]), (1, 20, 2, 1))
        self.assertEqual(worker["result"]["screen_trials"], 20)
        self.assertTrue(any(line.startswith("Round 1:") for line in worker["log"]))
        self.assertFalse(any("/private/path" in line for line in worker["log"]))
        self.assertEqual(self.service.campaign(row["id"])["campaign_id"], "campaign_abc123")
        # With a checkpoint the row reads the campaign's own counts and budget.
        runtime = self.root / "autoresearch" / "lab" / "alpha" / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        (runtime / "campaign_abc123.json").write_text(json.dumps({"campaign_id": "campaign_abc123", "status": "running", "round": 6,
                                                                   "screen_trials": 120, "validation_trials": 12, "promotions": 2,
                                                                   "population": [{"fitness": 1.9}], "holdout_accessed": False}))
        listed = self.service.campaigns()[0]
        self.assertEqual((listed["round"], listed["validation_trials"], listed["best_fitness"]), (6, 12, 1.9))
        self.assertEqual(listed["status"], "complete")
        self.assertEqual(listed["budget"], "round budget reached")
        with self.assertRaisesRegex(ValueError, "used its budget"):
            self.service.start({"id": row["id"]})
        extended = self.service.extend({"id": row["id"]})
        self.assertEqual(extended["limits"]["max_rounds"], 12)
        self.assertEqual(extended["status"], "in progress")
        self.assertIsNone(extended["budget"])
        # Submissions go to the campaign's inbox once its identity is known.
        queued = self.service.submit({"id": row["id"], "expression": "-rank(ts_std_dev(returns, 20))", "rationale": "low vol"})
        self.assertTrue(queued["queued"])
        inbox = (runtime / "campaign_abc123.inbox.jsonl").read_text(encoding="utf-8")
        self.assertIn("ts_std_dev", inbox)
        self.assertIn('"submitted_by": "console"', inbox)
        # A queued idea is checked like a seed: syntax first, then the fields this campaign's dataset can supply.
        with self.assertRaisesRegex(ValueError, r"Idea \('rank\(cap\)'\): .*field 'cap' is not in this dataset"):
            self.service.submit({"id": row["id"], "expression": "rank(cap)"})
        with self.assertRaisesRegex(ValueError, r"Idea \('rank\(close'\): "):
            self.service.submit({"id": row["id"], "expression": "rank(close"})
        self.assertEqual((runtime / "campaign_abc123.inbox.jsonl").read_text(encoding="utf-8"), inbox, "nothing was written for a refused idea")
        self.assertIn("no campaign with that identifier", self.service.report(row["id"])["note"])

    def test_finalize_can_read_the_holdout_as_non_confirmatory_evidence(self):
        row = self.service.create({"domain": "alpha", "name": "Evidence", "preset": "quick", "dataset": "demo-prices"})
        slug = row["id"].split("/")[1]
        side = self.root / "autoresearch" / "lab" / "alpha" / "campaigns" / f"{slug}.console.json"
        side.write_text(json.dumps({**json.loads(side.read_text()), "campaign_id": "campaign_evd"}), encoding="utf-8")
        runtime = self.root / "autoresearch" / "lab" / "alpha" / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        (runtime / "campaign_evd.json").write_text(json.dumps({"campaign_id": "campaign_evd", "status": "running", "round": 2,
                                                               "screen_trials": 10, "validation_trials": 3, "promotions": 1,
                                                               "population": [], "holdout_accessed": True}))
        with self.assertRaisesRegex(ValueError, "non_confirmatory must be true or false"):
            self.service.finalize({"id": row["id"], "confirm": True, "non_confirmatory": "yes"})
        self.script = [json.dumps({"campaign_id": "campaign_evd", "status": "finalized",
                                   "finals": [{"key": "k1", "decision": "not_confirmed", "non_confirmatory": True}]})]
        self.service.finalize({"id": row["id"], "confirm": True, "non_confirmatory": True})
        command = self.spawned[-1].command
        self.assertIn("finalize", command)
        self.assertIn("--non-confirmatory", command)
        self.assertNotIn("--allow-reuse", command)
        self.wait_idle()
        self.script = [json.dumps({"campaign_id": "campaign_evd", "status": "finalized", "finals": []})]
        self.service.finalize({"id": row["id"], "confirm": True})
        self.assertNotIn("--non-confirmatory", self.spawned[-1].command)

    def test_finalize_needs_confirmation_promotions_and_parses_the_result(self):
        row = self.service.create({"domain": "alpha", "name": "Final", "preset": "quick", "dataset": "demo-prices"})
        with self.assertRaisesRegex(ValueError, "explicit confirmation"):
            self.service.finalize({"id": row["id"]})
        with self.assertRaisesRegex(ValueError, "Run the campaign"):
            self.service.finalize({"id": row["id"], "confirm": True})
        slug = row["id"].split("/")[1]
        side = self.root / "autoresearch" / "lab" / "alpha" / "campaigns" / f"{slug}.console.json"
        side.write_text(json.dumps({**json.loads(side.read_text()), "campaign_id": "campaign_fin"}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Nothing was promoted"):
            self.service.finalize({"id": row["id"], "confirm": True})
        runtime = self.root / "autoresearch" / "lab" / "alpha" / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        (runtime / "campaign_fin.json").write_text(json.dumps({"campaign_id": "campaign_fin", "status": "running", "round": 2,
                                                               "screen_trials": 10, "validation_trials": 3, "promotions": 2,
                                                               "population": [], "holdout_accessed": False}))
        self.script = [json.dumps({"campaign_id": "campaign_fin", "status": "finalized",
                                   "finals": [{"key": "k1", "decision": "confirmed"}, {"key": "k2", "decision": "not_confirmed"}]}, indent=2)]
        self.service.finalize({"id": row["id"], "confirm": True, "keys": ["k1", "k2"]})
        self.wait_idle()
        self.assertEqual(self.spawned[-1].command[3], "finalize")
        self.assertEqual(self.spawned[-1].command.count("--key"), 2)
        worker = self.service.worker()
        self.assertEqual(len(worker["result"]["finals"]), 2)
        self.assertIn("1 confirmed", worker["log"][-1])
        with self.assertRaisesRegex(ValueError, "identifiers"):
            self.service.finalize({"id": row["id"], "confirm": True, "keys": ["bad key!"]})

    def test_one_worker_at_a_time_stop_and_error_reporting(self):
        row = self.service.create({"domain": "ml", "name": "Err", "preset": "quick", "task": "builtin:friedman1"})
        self.script = ["tubby-lab: campaign domain must be one of: alpha, ml"]

        def failing(command):
            process = FakeProcess(command, list(self.script), code=2)
            self.spawned.append(process)
            return process
        self.service._spawn = failing
        self.service.start({"id": row["id"]})
        self.wait_idle()
        worker = self.service.worker()
        self.assertEqual(worker["status"], "error")
        self.assertIn("campaign domain", worker["last_error"])

        def slow(command):
            process = FakeProcess(command, ['{"status": "round", "round": 1, "campaign_id": "campaign_slow", "screened": 1, "validated": 0, "promoted": 0}'], delay=2.0)
            self.spawned.append(process)
            return process
        self.service._spawn = slow
        self.service.start({"id": row["id"]})
        with self.assertRaisesRegex(ValueError, "already running"):
            self.service.start({"id": row["id"]})
        with self.assertRaisesRegex(ValueError, "Stop the campaign"):
            self.service.extend({"id": row["id"]})
        stopped = self.service.stop(_grace=1)
        self.assertFalse(stopped["running"])
        self.assertEqual(stopped["status"], "stopped")

    def test_identifiers_and_paths_are_checked(self):
        for bad in (None, "alpha", "alpha/../x", "nope/slug", "alpha/Bad Slug", 7):
            with self.assertRaises(ValueError):
                self.service.campaign(bad)
        with self.assertRaisesRegex(ValueError, "no longer exists"):
            self.service.campaign("alpha/missing-campaign")
        with self.assertRaisesRegex(ValueError, "directly under"):
            self.service.plan({"domain": "alpha", "dataset": str(self.root / "elsewhere")})
        with self.assertRaisesRegex(ValueError, "Choose a demonstration"):
            self.service.demo({"domain": "other"})


@unittest.skipUnless(os.environ.get("TUBBY_LAB_INTEGRATION") == "1", "set TUBBY_LAB_INTEGRATION=1 to run the real worker")
class LabServiceIntegrationTests(unittest.TestCase):
    def test_real_worker_runs_a_round_reports_and_finalizes(self):
        from qlib_yahoo.demo_market import create_demo_market

        with tempfile.TemporaryDirectory(dir="/private/tmp" if Path("/private/tmp").is_dir() else None) as tmp:
            root = Path(tmp)
            dataset = create_demo_market(root / "datasets" / "demo", symbols=40, days=700)
            service = LabService(root)
            try:
                row = service.create({"domain": "alpha", "name": "Real", "preset": "quick", "dataset": str(dataset)})
                service.start({"id": row["id"], "rounds": 1})
                deadline = time.time() + 600
                while service.worker()["running"] and time.time() < deadline:
                    time.sleep(1)
                worker = service.worker()
                self.assertEqual(worker["status"], "finished", worker)
                report = service.report(row["id"])["report"]
                self.assertTrue(report["ledger"]["ok"])
                self.assertEqual(len(report["rounds"]), 1)
                verified = service.verify(row["id"])
                self.assertTrue(verified["checkpoint_matches_ledger"])
                self.assertIn("registry", verified)
                self.assertNotIn(verified["registry"].get("status"), ("registry_reset", "registry_shrunk"))
                self.assertIn("ok", verified)
                if report["leaderboard"]:
                    detail = service.candidate({"id": row["id"], "hypothesis": report["leaderboard"][0]["hypothesis_id"]})
                    self.assertIn("validation", detail["experiment"]["metrics"])
            finally:
                service.close()


if __name__ == "__main__":
    unittest.main()
