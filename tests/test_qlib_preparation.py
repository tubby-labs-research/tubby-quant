"""Bounded preparation reuse preserves complete independent Qlib trials."""
import contextlib
import gzip
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from qlib_yahoo.dataset import DatasetError
from qlib_yahoo.preparation import BacktestPreparation, run_backtest_batch

HAVE_PANDAS = bool(importlib.util.find_spec("pandas"))
CHILD = "QLIB_PREPARATION_SIGNED_CHILD"


class PreparationTests(unittest.TestCase):
    def test_identity_change_clears_all_reusable_state(self):
        prep = BacktestPreparation()
        original = {"dataset": "data", "window": ["start", "end"], "universe": "top2", "runtime": "pin"}
        prep.bind(original)
        prep.initialized = prep.factor_validated = True
        prep.exchange = object()
        prep.eligible = ("AAA",)
        identity = prep.identity
        prep.bind(dict(reversed(list(original.items()))))
        self.assertEqual(prep.identity, identity)
        self.assertTrue(prep.initialized)
        for key, value in (("dataset", "other"), ("window", ["start", "new"]),
                           ("universe", "top1"), ("runtime", "other-pin")):
            changed = {**original, key: value}
            prep.bind(changed)
            self.assertNotEqual(prep.identity, identity)
            self.assertFalse(prep.initialized or prep.factor_validated)
            self.assertIsNone(prep.exchange)
            self.assertIsNone(prep.eligible)
            prep.bind(original)
            prep.initialized = prep.factor_validated = True
            prep.exchange = object()
            prep.eligible = ("AAA",)
        for cap in (-1, 1.5, True):
            with self.assertRaises(DatasetError):
                BacktestPreparation(cap)

    @unittest.skipUnless(HAVE_PANDAS, "pandas unavailable")
    def test_frame_cache_reuses_copies_keys_and_evicts_within_cap(self):
        import pandas as pd
        frame = pd.DataFrame({"score": [1.0, 2.0]})
        size = int(frame.memory_usage(index=True, deep=True).sum())
        prep = BacktestPreparation(cache_bytes=size)
        prep.bind({"source": "same"})
        perf = Mock()
        perf.call.side_effect = lambda stage, fn, *args, **kwargs: fn(*args, **kwargs)
        provider = Mock(side_effect=lambda *args, **kwargs: frame.copy(deep=True))
        first = prep.features(perf, "scores", provider, ["AAA"], ["$close"], start_time="s")
        first.iloc[0, 0] = 99
        again = prep.features(perf, "scores", provider, ["AAA"], ["$close"], start_time="s")
        self.assertEqual(again.iloc[0, 0], 1)
        again.iloc[0, 0] = 88
        self.assertEqual(prep.features(perf, "scores", provider, ["AAA"], ["$close"], start_time="s").iloc[0, 0], 1)
        self.assertEqual(provider.call_count, 1)
        for symbols, fields, start in ((["BBB"], ["$close"], "s"), (["BBB"], ["$open"], "s"),
                                       (["BBB"], ["$open"], "new")):
            prep.features(perf, "scores", provider, symbols, fields, start_time=start)
            self.assertLessEqual(prep.snapshot()["cached_frame_bytes"], size)
        self.assertEqual(len(prep._frames), 1)
        prep.bind({"source": "changed"})
        self.assertEqual(prep.snapshot()["cached_frame_bytes"], 0)
        prep.features(perf, "scores", provider, ["AAA"], ["$close"], start_time="s")
        self.assertEqual(provider.call_count, 5)
        prep = BacktestPreparation(cache_bytes=size - 1)
        for _ in range(2):
            prep.features(perf, "scores", provider, ["AAA"], ["$close"])
        self.assertFalse(prep._frames)
        self.assertEqual(provider.call_count, 7)

    def test_duplicate_trials_are_run_independently_and_failures_are_isolated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calls = []
            def run(dataset, alpha, **settings):
                calls.append((alpha, settings))
                if alpha == "$open":
                    raise DatasetError("fixture failure")
                return root / "runs" / str(len(calls))
            with patch("qlib_yahoo.backtest.run_backtest", side_effect=run):
                receipt = run_backtest_batch(root / "dataset", ["$close", "$open", "$close"], data_root=root,
                                             strategy_mode="signed", start="2024-01-03")
            record = json.loads(receipt.read_text())
            self.assertEqual([a for a, _ in calls], ["$close", "$open", "$close"])
            self.assertEqual(record["runs"], [str(root / "runs/1"), str(root / "runs/3")])
            self.assertEqual(record["errors"][0]["alpha"], "$open")
            self.assertEqual(record["status"], "completed")
            self.assertTrue(all(options["start"] == "2024-01-03" for _, options in calls))
            self.assertEqual(len({id(options["_preparation"]) for _, options in calls}), 1)
            self.assertIsNone(calls[0][1]["_preparation"].identity)

    def test_interrupt_keeps_receipt_cancelled_not_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch("qlib_yahoo.backtest.run_backtest", side_effect=[root / "runs/1", KeyboardInterrupt("cancel")]):
                with self.assertRaises(KeyboardInterrupt):
                    run_backtest_batch(root / "dataset", ["$close", "$open"], data_root=root)
            receipt = next((root / "workbench/batches").glob("*.json"))
            record = json.loads(receipt.read_text())
            self.assertEqual(record["status"], "cancelled")
            self.assertEqual(record["runs"], [str(root / "runs/1")])

    def test_cancel_marker_after_first_trial_skips_later_trials(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            marker = root / "cancel"
            def completed(dataset, alpha, **settings):
                marker.write_text("cancel")
                return root / "runs/1"
            with patch("qlib_yahoo.backtest.run_backtest", side_effect=completed) as run:
                with self.assertRaises(KeyboardInterrupt):
                    run_backtest_batch(root / "dataset", ["$close", "$open"], data_root=root, _cancel_file=str(marker))
            self.assertEqual(run.call_count, 1)
            record = json.loads(next((root / "workbench/batches").glob("*.json")).read_text())
            self.assertEqual(record["status"], "cancelled")
            self.assertEqual(record["runs"], [str(root / "runs/1")])

    def test_cli_alphas_file_keeps_duplicates_and_shared_settings(self):
        from qlib_yahoo.__main__ import main
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            alphas = root / "alphas.txt"
            alphas.write_text("# comment\n$close\n\n$close\n$open\n")
            with patch("qlib_yahoo.__main__.resolve_dataset", return_value=root / "dataset"), \
                    patch("qlib_yahoo.preparation.run_backtest_batch", return_value=root / "receipt") as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--data-root", str(root), "backtest", "--alphas-file", str(alphas),
                                       "--strategy-mode", "signed", "--history-mode", "stream", "--warmup-sessions", "20"]), 0)
            args, settings = run.call_args
            self.assertEqual(args, (root / "dataset", ["$close", "$close", "$open"]))
            self.assertEqual(settings["strategy_mode"], "signed")
            self.assertEqual(settings["history_mode"], "stream")
            self.assertEqual(settings["warmup_sessions"], 20)
            self.assertNotIn("alpha", settings)


@unittest.skipUnless(importlib.util.find_spec("qlib"), "optional Qlib runtime unavailable")
class PreparationRunnerTests(unittest.TestCase):
    def make_snapshot(self, root):
        from tests.test_qlib_yahoo_execution import _snapshot
        return _snapshot(root, sessions_count=35, include_benchmark=False)

    @unittest.skipIf(os.environ.get(CHILD) == "1", "normal-runtime runner")
    def test_changed_actual_bytes_refused_before_preparation_reuse(self):
        from qlib_yahoo.backtest import run_backtest
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            dataset = self.make_snapshot(root)
            prep = BacktestPreparation()
            run_backtest(dataset, "$close", warmup_sessions=20, topk=2, holdings="never", _preparation=prep)
            self.assertTrue(prep.initialized)
            identity, exchange = prep.identity, prep.exchange
            feature = dataset / "qlib_data/features/tst0/close.day.bin"
            stat = feature.stat()
            data = bytearray(feature.read_bytes())
            data[-1] ^= 1
            feature.write_bytes(data)
            os.utime(feature, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            with self.assertRaisesRegex(DatasetError, "tree differs"):
                run_backtest(dataset, "$close", warmup_sessions=20, topk=2, holdings="never", _preparation=prep)
            # A legitimately resealed new snapshot must invalidate the in-memory cache too.
            from qlib_yahoo.dataset import tree_sha256
            manifest_path = dataset / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["qlib_binary_tree_sha256"] = tree_sha256(dataset / "qlib_data")[0]
            manifest_path.write_text(json.dumps(manifest))
            changed = run_backtest(dataset, "$close", warmup_sessions=20, topk=2, holdings="never", _preparation=prep)
            self.assertNotEqual(prep.identity, identity)
            self.assertIsNot(prep.exchange, exchange)
            fresh = run_backtest(dataset, "$close", warmup_sessions=20, topk=2, holdings="never")
            import pandas as pd
            pd.testing.assert_frame_equal(pd.read_parquet(changed / "alpha_scores.parquet"),
                                          pd.read_parquet(fresh / "alpha_scores.parquet"))
            pd.testing.assert_frame_equal(pd.read_csv(changed / "daily_report.csv"), pd.read_csv(fresh / "daily_report.csv"))

    def assert_independent_parity(self, mode):
        from qlib_yahoo.backtest import run_backtest
        import pandas as pd
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            dataset = self.make_snapshot(root)
            expression = "$close-42" if mode == "signed" else "$close/Ref($close,1)-1"
            options = dict(warmup_sessions=20, strategy_mode=mode, holdings="never", topk=2,
                           history_mode="memory", execution_profile="fast")
            reference = run_backtest(dataset, expression, **options)
            preparations = []
            original = BacktestPreparation.quote_exchange
            def observed(prep, performance, factory, **options):
                wrapped_factory = Mock(wraps=factory)
                result = original(prep, performance, wrapped_factory, **options)
                preparations.extend(wrapped_factory.call_args_list)
                return result
            with patch.object(BacktestPreparation, "quote_exchange", observed):
                receipt = run_backtest_batch(dataset, [expression, expression], data_root=root, **options)
            self.assertEqual(len(preparations), 1, "prepare quotes once for two independent accounts")
            record = json.loads(receipt.read_text())
            self.assertEqual(record["status"], "completed")
            self.assertFalse(record["errors"])
            self.assertEqual(len(set(record["runs"])), 2)
            self.assertGreater(record["preparation"]["hits"], 0)
            manifests = [json.loads((Path(p) / "manifest.json").read_text()) for p in record["runs"]]
            self.assertTrue(all(m["preparation_reuse"]["fresh_account_per_run"] for m in manifests))
            for other in map(Path, record["runs"]):
                pd.testing.assert_frame_equal(pd.read_csv(reference / "daily_report.csv"), pd.read_csv(other / "daily_report.csv"))
                for table in ("alpha_scores", "positions", *(("signed_weights",) if mode == "signed" else ())):
                    pd.testing.assert_frame_equal(pd.read_parquet(reference / f"{table}.parquet"), pd.read_parquet(other / f"{table}.parquet"))
                reference_summary = json.loads((reference / "summary.json").read_text())
                other_summary = json.loads((other / "summary.json").read_text())
                self.assertEqual(reference_summary["metrics"], other_summary["metrics"])
                if mode == "signed":
                    self.assertEqual(reference_summary["funding_state"], other_summary["funding_state"])
                    with gzip.open(reference / "signed_decisions.json.gz", "rt") as left, gzip.open(other / "signed_decisions.json.gz", "rt") as right:
                        self.assertEqual(json.load(left), json.load(right))

    @unittest.skipIf(os.environ.get(CHILD) == "1", "normal-runtime runner")
    def test_long_only_duplicate_full_runs_reset_state(self):
        self.assert_independent_parity("long-only")

    @unittest.skipIf(os.environ.get(CHILD) == "1", "parent launcher")
    def test_signed_duplicate_full_runs_in_verified_pinned_child(self):
        from tests.test_qlib_performance_parity import _signed_available
        from qlib_yahoo.runtime import ROOT
        if not _signed_available():
            self.skipTest("pinned signed derivative unavailable")
        program = ("from qlib_yahoo.runtime import ensure_signed_runtime; ensure_signed_runtime(); import unittest; "
                   "suite=unittest.defaultTestLoader.loadTestsFromName('tests.test_qlib_preparation.SignedPreparationChild'); "
                   "result=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(not result.wasSuccessful())")
        env = {**os.environ, CHILD: "1"}
        result = subprocess.run([sys.executable, "-c", program], cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(os.environ.get(CHILD) == "1", "requires pinned signed child")
class SignedPreparationChild(PreparationRunnerTests):
    def test_signed_full_run_parity(self):
        self.assert_independent_parity("signed")

    def test_different_signed_expressions_match_independent_fresh_runs(self):
        from qlib_yahoo.backtest import run_backtest
        from qlib.backtest.signed_account import SignedAccount
        import pandas as pd

        expressions = ["$close-42", "42-$close"]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            dataset = self.make_snapshot(root)
            settings = dict(warmup_sessions=20, strategy_mode="signed", initial_cash=100_000.0,
                            holdings="never", history_mode="memory", execution_profile="fast")
            references = [run_backtest(dataset, alpha, **settings) for alpha in expressions]
            accounts, quote_builds = [], []
            original_init = SignedAccount.__init__
            original_quote = BacktestPreparation.quote_exchange
            def initialized(account, *args, **kwargs):
                original_init(account, *args, **kwargs)
                accounts.append((account, tuple(account.current_position.get_stock_list()),
                                 account.current_position.get_cash()))
            def quoted(preparation, performance, factory, **options):
                observed_factory = Mock(wraps=factory)
                result = original_quote(preparation, performance, observed_factory, **options)
                quote_builds.extend(observed_factory.call_args_list)
                return result
            with patch.object(SignedAccount, "__init__", initialized), \
                    patch.object(BacktestPreparation, "quote_exchange", quoted):
                receipt = run_backtest_batch(dataset, expressions, data_root=root, **settings)
            record = json.loads(receipt.read_text())
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["errors"], [])
            self.assertEqual(record["alphas"], expressions)
            self.assertEqual(len(set(record["runs"])), 2)
            self.assertEqual(len(quote_builds), 1)
            self.assertEqual(len(accounts), 2)
            self.assertIsNot(accounts[0][0], accounts[1][0])
            self.assertEqual([(stocks, cash) for _, stocks, cash in accounts], [((), 100_000.0)] * 2)
            self.assertGreater(record["preparation"]["hits"], 0)
            self.assertFalse(pd.read_parquet(references[0] / "alpha_scores.parquet").equals(
                pd.read_parquet(references[1] / "alpha_scores.parquet")))
            for alpha, reference, run_path in zip(expressions, references, record["runs"]):
                with self.subTest(alpha=alpha):
                    run = Path(run_path)
                    manifest = json.loads((run / "manifest.json").read_text())
                    self.assertEqual(manifest["alpha"], alpha)
                    self.assertTrue(manifest["preparation_reuse"]["fresh_account_per_run"])
                    for name in ("daily_report.csv", "qlib_trade_indicators.csv"):
                        pd.testing.assert_frame_equal(pd.read_csv(reference / name), pd.read_csv(run / name),
                                                      check_exact=True)
                    for name in ("alpha_scores", "positions", "signed_weights", "signal_forward_labels"):
                        pd.testing.assert_frame_equal(pd.read_parquet(reference / f"{name}.parquet"),
                                                      pd.read_parquet(run / f"{name}.parquet"), check_exact=True)
                    with gzip.open(reference / "signed_decisions.json.gz", "rt") as left, \
                            gzip.open(run / "signed_decisions.json.gz", "rt") as right:
                        self.assertEqual(json.load(left), json.load(right))
                    expected_summary = json.loads((reference / "summary.json").read_text())
                    actual_summary = json.loads((run / "summary.json").read_text())
                    self.assertEqual(expected_summary["metrics"], actual_summary["metrics"])
                    self.assertEqual(expected_summary["funding_state"], actual_summary["funding_state"])
