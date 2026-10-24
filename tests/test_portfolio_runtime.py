"""S3 risk hooks through the installed signed engine in a fresh subprocess."""
import gzip
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CHILD = "--portfolio-runtime-child" in sys.argv

from qlib_yahoo.backtest import run_backtest
from qlib_yahoo.dataset import DatasetError, sha256
from qlib_yahoo.portfolio_runtime import validate_portfolio_risk


def policy(**limits):
    return {"schema_version": 1,
            "limits": {**dict.fromkeys(("gross", "net", "name", "group_gross", "group_net", "turnover", "liquidity")),
                       **limits}, "group_overlay": None, "group_level": None}


def installed():
    try:
        from qlib_yahoo.runtime import platform_pin
        pin = platform_pin(json.loads((ROOT / "docs/qlib-derivative-pin.json").read_text()))
        return (sys.version_info[:2] == (3, 12)
                and all(importlib.util.find_spec(x) for x in ("numpy", "pandas", "pyarrow"))
                and (ROOT / pin["local_runtime"] / "qlib/_tubby_build.json").is_file())
    except (DatasetError, OSError, KeyError, ValueError):
        return False


@unittest.skipIf(CHILD, "parent boundary checks")
class BoundaryTests(unittest.TestCase):
    def test_schema_detaches_and_rejects_invalid_enabled_limits(self):
        supplied = policy(gross=0, name=.5)
        detached = validate_portfolio_risk(supplied)
        supplied["limits"]["name"] = 99
        self.assertEqual(detached["limits"]["name"], .5)
        self.assertEqual(detached["limits"]["gross"], 0)
        for bad in ({}, {**policy(), "limits": {}},
                    {**policy(), "schema_version": True}, {**policy(), "extra": 1},
                    policy(gross=float("nan")), policy(liquidity=-1), policy(unknown=1),
                    {**policy(), "group_level": "beta"}, {**policy(), "group_overlay": []},
                    {**policy(), "provenance": {"bad": float("inf")}}):
            with self.subTest(bad=bad), self.assertRaises(DatasetError):
                validate_portfolio_risk(bad)

    def test_opt_in_requires_signed_and_valid_policy_before_runtime(self):
        with patch("qlib_yahoo.runtime.ensure_signed_runtime", side_effect=AssertionError("runtime loaded")):
            for options in ({"portfolio_risk": policy()},
                            {"portfolio_risk": {}, "strategy_mode": "signed"}):
                with self.assertRaises(DatasetError):
                    run_backtest(Path("absent"), "$close", **options)

    @unittest.skipUnless(installed(), "verified signed runtime/dependencies unavailable")
    def test_actual_installed_engine_subprocess(self):
        result = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                                 "--portfolio-runtime-child"], cwd=ROOT,
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"},
                                capture_output=True, text=True, encoding="utf-8", timeout=180)
        self.assertEqual(result.returncode, 0, (result.stdout + result.stderr)[-25000:])
        self.assertIn("PORTFOLIO_RUNTIME_CHILD_OK", result.stdout)


@unittest.skipUnless(CHILD, "isolated installed-engine cases")
class InstalledRiskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from qlib_yahoo.runtime import ensure_signed_runtime
        cls.identity = ensure_signed_runtime()

    def setUp(self):
        import numpy as np
        import pandas as pd
        from test_qlib_yahoo_execution import _snapshot
        self.np, self.pd = np, pd
        self.temp = tempfile.TemporaryDirectory(prefix="portfolio-risk-test-", dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = _snapshot(self.root, sessions_count=12, include_benchmark=False)
        self.days = pd.bdate_range("2024-01-02", periods=12)
        self.symbols = [f"TST{i}" for i in range(6)]
        self.values = np.zeros((9, 6), dtype=np.float32)
        self.values[0:2, :2] = [.25, -.125]
        self.values[2:4, :2] = [-.125, .25]

    def change_bar(self, symbol, field, index, value):
        from qlib_yahoo.dataset_integrity import tree_sha256
        path = self.dataset / "qlib_data/features" / symbol.lower() / f"{field}.day.bin"
        values = list(struct.unpack("<13f", path.read_bytes()))
        if index is None:
            values[1:] = [value] * 12
        else:
            values[index + 1] = value
        path.write_bytes(struct.pack("<13f", *values))
        manifest_path = self.dataset / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["qlib_binary_tree_sha256"] = tree_sha256(self.dataset / "qlib_data")[0]
        manifest_path.write_text(json.dumps(manifest))

    def run_case(self, risk, **options):
        from qlib_yahoo.target_bundle import write_bundle
        from test_target_bundle import metadata
        envelope = write_bundle(self.root / ("targets-" + str(len(list(self.root.glob("targets-*"))))),
            dates=[str(d.date()) for d in self.days[1:-2]], symbols=self.symbols, weights=self.values,
            metadata=metadata(sha256(self.dataset / "manifest.json")))
        run = run_backtest(self.dataset, "SUPPLIED TARGETS", target_bundle=envelope,
            strategy_mode="signed", warmup_sessions=2, runtime_profile=options.pop("runtime_profile", "native"),
            execution_profile=options.pop("execution_profile", "reference"),
            history_mode=options.pop("history_mode", "memory"), holdings="never", portfolio_risk=risk, **options)
        manifest = json.loads((run / "manifest.json").read_text())
        events = ([json.loads(line) for line in gzip.decompress((run / "portfolio_risk.jsonl.gz").read_bytes()).decode().splitlines()]
                  if risk is not None else [])
        return run, manifest, events

    def test_native_projection_shorts_reversals_actual_fills_and_unchanged_baseline(self):
        # Runtime hooks execute genuine exchange/position/account methods, not mocks.
        baseline, _, _ = self.run_case(None)
        run, manifest, events = self.run_case(policy(gross=1, net=1, name=.6, turnover=2))
        expected = self.pd.read_csv(baseline / "daily_report.csv")
        actual = self.pd.read_csv(run / "daily_report.csv")
        self.pd.testing.assert_frame_equal(expected, actual)
        self.assertEqual(len(events), 18)
        self.assertTrue(all(e["status"] == "pass" for e in events))
        post = [e for e in events if e["phase"] == "post_fill"]
        self.assertGreater(post[0]["actual_holdings"]["TST0"], 0)
        self.assertLess(post[0]["actual_holdings"]["TST1"], 0)
        self.assertLess(post[2]["actual_holdings"]["TST0"], 0)
        for index, record in enumerate(post):
            self.assertAlmostEqual(record["metrics"]["free_cash_usd"], actual.free_cash.iloc[index])
        self.assertEqual(manifest["portfolio_risk"]["counts"]["blocked_rebalances"], 0)
        self.assertEqual(manifest["artifacts_sha256"]["portfolio_risk.jsonl.gz"], sha256(run / "portfolio_risk.jsonl.gz"))
        summary = json.loads((run / "summary.json").read_text())
        self.assertEqual(summary["portfolio_risk"], manifest["portfolio_risk"])
        self.assertIn("portfolio_runtime.py", manifest["runner_source_sha256"])

    def test_concentration_blocks_entire_rebalance(self):
        _, manifest, events = self.run_case(policy(name=.1))
        self.assertGreater(manifest["portfolio_risk"]["counts"]["blocked_rebalances"], 0)
        pre, post = events[0:2]
        self.assertEqual(pre["status"], "blocked")
        self.assertIn("target:name", pre["violations"])
        self.assertEqual(post["actual_holdings"], {})
        self.assertEqual(post["executed_trades"], {})
        self.assertTrue(post["unfilled_gross_shares"])

    def test_runner_source_drift_prevents_successful_seal(self):
        original = sha256
        reads = 0

        def drift(path):
            nonlocal reads
            result = original(path)
            if Path(path).resolve() == (ROOT / "qlib_yahoo/backtest.py").resolve():
                reads += 1
                if reads > 1:
                    return "0" * 64
            return result

        with patch("qlib_yahoo.backtest.sha256", side_effect=drift):
            with self.assertRaisesRegex(DatasetError, "implementation changed"):
                self.run_case(policy())
        manifests = [json.loads(p.read_text()) for p in (self.root / "runs").glob("*/manifest.json")]
        self.assertEqual(len(manifests), 1)
        self.assertEqual(manifests[0]["status"], "failed")

    def test_missing_open_zero_target_retains_held_shares(self):
        self.values[:] = 0
        self.values[0, 0] = -.25
        self.change_bar("TST0", "open", 3, float("nan"))
        run, _, events = self.run_case(policy(gross=1))
        pre, post = events[2:4]
        self.assertEqual(pre["status"], "missing_inputs")
        self.assertIn("reference_prices:TST0", pre["missing_inputs"])
        self.assertIn("reference_prices:TST0", post["missing_inputs"])
        self.assertEqual(post["actual_holdings"], events[1]["actual_holdings"])
        self.assertEqual(post["executed_trades"], {})
        self.assertEqual(events[5]["actual_holdings"], {})  # later valid open allows liquidation
        decisions = json.loads(gzip.decompress((run / "signed_decisions.json.gz").read_bytes()))
        self.assertIn("portfolio_risk", decisions[1]["reasons"])

    def test_exhausted_funding_after_borrow_blocks_and_keeps_short(self):
        # Deliberate high-rate native stress fixture; not a replacement borrow default.
        self.values[:] = 0
        self.values[0, 0] = -.5
        run, _, events = self.run_case(policy(gross=1), annual_borrow_rate=5000)
        self.assertEqual(events[2]["status"], "blocked")
        self.assertIn("funding:negative_free_cash", events[2]["violations"])
        self.assertIn("equity:nonpositive", events[2]["violations"])
        self.assertEqual(events[3]["actual_holdings"], events[1]["actual_holdings"])
        daily = self.pd.read_csv(run / "daily_report.csv")
        self.assertLess(daily.free_cash.iloc[1], 0)
        self.assertGreater(daily.borrow_cost_dollars.iloc[1], 0)

    def test_native_fee_affordability_clips_and_gross_requested_differs_from_actual(self):
        self.values[:] = 0
        self.values[0, 0] = 1
        for field in ("open", "close"):
            self.change_bar("TST0", field, None, 100)
        _, _, events = self.run_case(policy(gross=1, turnover=2), initial_cash=1000)
        pre, post = events[:2]
        self.assertEqual(pre["status"], "pass")
        self.assertEqual(pre["proposed_gross_shares"], {"TST0": 10})
        self.assertEqual(pre["executable_trades"], {"TST0": 9})
        self.assertEqual(post["executed_gross_shares"], {"TST0": 9})
        self.assertEqual(post["unfilled_gross_shares"], {"TST0": 1})
        self.assertAlmostEqual(pre["metrics"]["turnover"], 1000 / 999.55)
        self.assertAlmostEqual(post["metrics"]["turnover"], 900 / 999.55)
        self.assertAlmostEqual(post["metrics"]["free_cash_usd"], 99.55)

    def test_native_cash_projection_and_turnover_limit_block_jointly(self):
        self.values[:] = 0
        self.values[0, 0] = 1
        for field in ("open", "close"):
            self.change_bar("TST0", field, None, 100)
        _, _, events = self.run_case(policy(turnover=1), initial_cash=1000)
        self.assertEqual(events[0]["status"], "blocked")
        self.assertIn("trades:turnover", events[0]["violations"])
        self.assertEqual(events[1]["actual_holdings"], {})
        self.assertEqual(events[0]["native_projection"][0]["executable"], 9)

    def test_no_affordable_fill_is_explicitly_blocked_without_injecting_cash(self):
        self.values[:] = 0
        self.values[0, 0] = 1
        self.change_bar("TST0", "open", None, 100)
        _, _, events = self.run_case(policy(), initial_cash=100)
        self.assertIn("execution:no_affordable_fills", events[0]["violations"])
        self.assertEqual(events[0]["status"], "blocked")
        self.assertEqual(events[1]["actual_holdings"], {})
        self.assertEqual(events[1]["metrics"]["free_cash_usd"], 100)

    def test_missing_group_source_is_never_waived(self):
        _, _, events = self.run_case(policy(group_gross=.8))
        self.assertEqual(events[0]["status"], "missing_inputs")
        self.assertIn("groups:TST0", events[0]["missing_inputs"])
        self.assertEqual(events[1]["actual_holdings"], {})

    def test_late_group_information_waits_for_decision_session_open(self):
        from qlib_yahoo.classification import create_classification_overlay
        self.values[:] = 0
        self.values[:, 0] = .25
        overlay = create_classification_overlay(self.dataset, self.root / "groups", [{
            "instrument": "TST0", "available_at": f"{self.days[2]:%Y-%m-%d}T15:00:00Z",
            "sector": "G", "industry": "I", "country": "US"}])
        configured = policy(group_gross=.8)
        configured.update(group_overlay=str(overlay), group_level="sector")
        _, manifest, events = self.run_case(configured)
        self.assertEqual(events[0]["status"], "missing_inputs")
        self.assertEqual(events[2]["status"], "missing_inputs")
        self.assertEqual(events[4]["status"], "pass")
        self.assertIn("group_identity", manifest["portfolio_risk"]["sources"])

    def test_prior_volume_missing_current_future_volume_cannot_fill_gap(self):
        self.values[:] = 0
        self.values[:, 0] = .25
        self.change_bar("TST0", "volume", 1, float("nan"))
        self.change_bar("TST0", "volume", 2, 1e9)
        _, _, events = self.run_case(policy(liquidity=.05))
        self.assertEqual(events[0]["status"], "missing_inputs")
        self.assertIn("liquidity:TST0", events[0]["missing_inputs"])
        self.assertEqual(events[1]["actual_holdings"], {})
        self.assertEqual(events[2]["status"], "pass")

    def test_zero_prior_volume_blocks_even_with_large_current_volume(self):
        self.change_bar("TST0", "volume", 1, 0)
        self.change_bar("TST0", "volume", 2, 1e9)
        _, _, events = self.run_case(policy(liquidity=.05))
        self.assertEqual(events[0]["status"], "blocked")
        self.assertIn("trades:liquidity:TST0", events[0]["violations"])
        self.assertEqual(events[1]["actual_holdings"], {})

    def test_all_zero_and_explicit_accelerator_fallback_keep_cancellation(self):
        from qlib_yahoo.preparation import check_cancel
        self.values[:] = 0
        with patch("qlib_yahoo.fast_runtime.resolve_fast_profile", side_effect=AssertionError("automatic compiler recipe")), \
                patch("qlib_yahoo.preparation.check_cancel", wraps=check_cancel) as cancellation:
            run, manifest, events = self.run_case(policy(gross=0, turnover=0, liquidity=0, group_gross=0),
                execution_profile="fast", history_mode="stream", runtime_profile="auto",
                _ordered_session=True, _compiler_path="must-not-load", _compiled_targets=True,
                _cancel_file=str(self.root / "cancel"))
        self.assertEqual(cancellation.call_count, 9)
        self.assertTrue(all(e["status"] == "pass" for e in events))
        self.assertTrue(manifest["runtime_profile"]["declined"]["ordered_session"])
        self.assertEqual(manifest["execution_acceleration"]["selection"], "portfolio_risk_native_fills")
        daily = self.pd.read_csv(run / "daily_report.csv")
        self.assertTrue((daily.account == 100000).all())
        self.assertTrue((daily.cost_dollars == 0).all())


if __name__ == "__main__":
    if CHILD:
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(InstalledRiskTests))
        if result.wasSuccessful():
            print("PORTFOLIO_RUNTIME_CHILD_OK")
        sys.exit(0 if result.wasSuccessful() else 1)
    unittest.main()
