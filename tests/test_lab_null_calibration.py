"""Null calibration of the alpha pipeline: exact binomial intervals, a tiny replicated study, generator options."""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

HAVE_NUMERIC = bool(importlib.util.find_spec("numpy") and importlib.util.find_spec("pandas"))
HAVE_SCIPY = bool(importlib.util.find_spec("scipy"))


class IntervalTests(unittest.TestCase):
    def test_closed_forms_at_zero_and_at_every_success(self):
        from autoresearch.domains.alpha.null_calibration import clopper_pearson

        low, high = clopper_pearson(0, 5)
        self.assertEqual(low, 0.0)
        self.assertAlmostEqual(high, 1 - 0.025 ** (1 / 5), places=12)      # 0.5218: five clean panels bound little
        low, high = clopper_pearson(0, 200)
        self.assertAlmostEqual(high, 1 - 0.025 ** (1 / 200), places=12)    # 0.0183
        low, high = clopper_pearson(200, 200)
        self.assertEqual(high, 1.0)
        self.assertAlmostEqual(low, 0.025 ** (1 / 200), places=12)

    def test_interior_interval_matches_the_tabulated_value_and_inverts_the_binomial_tails(self):
        from autoresearch.domains.alpha.null_calibration import binomial_cdf, binomial_tail, clopper_pearson

        low, high = clopper_pearson(3, 10)
        self.assertAlmostEqual(low, 0.0667, places=4)
        self.assertAlmostEqual(high, 0.6525, places=4)
        self.assertAlmostEqual(binomial_tail(3, 10, low), 0.025, places=9)
        self.assertAlmostEqual(binomial_cdf(3, 10, high), 0.025, places=9)
        low, high = clopper_pearson(7, 200, confidence=0.9)
        self.assertAlmostEqual(binomial_tail(7, 200, low), 0.05, places=9)
        self.assertAlmostEqual(binomial_cdf(7, 200, high), 0.05, places=9)

    def test_interval_is_symmetric_monotone_and_covers_the_point_estimate(self):
        from autoresearch.domains.alpha.null_calibration import clopper_pearson

        for k in range(0, 21):
            low, high = clopper_pearson(k, 20)
            mirror_low, mirror_high = clopper_pearson(20 - k, 20)
            self.assertAlmostEqual(low, 1 - mirror_high, places=10)
            self.assertAlmostEqual(high, 1 - mirror_low, places=10)
            self.assertLessEqual(low, k / 20)
            self.assertGreaterEqual(high, k / 20)
        self.assertLess(clopper_pearson(0, 400)[1], clopper_pearson(0, 200)[1])
        wide = clopper_pearson(5, 50, confidence=0.99)
        narrow = clopper_pearson(5, 50, confidence=0.9)
        self.assertLess(wide[0], narrow[0])
        self.assertGreater(wide[1], narrow[1])

    @unittest.skipUnless(HAVE_SCIPY, "scipy cross-check is optional")
    def test_matches_the_beta_quantile_form(self):
        from scipy.stats import beta

        from autoresearch.domains.alpha.null_calibration import clopper_pearson

        for k, n in ((1, 200), (4, 200), (17, 60), (50, 100), (99, 100)):
            low, high = clopper_pearson(k, n)
            self.assertAlmostEqual(low, float(beta.ppf(0.025, k, n - k + 1)), places=9, msg=(k, n))
            self.assertAlmostEqual(high, float(beta.ppf(0.975, k + 1, n - k)), places=9, msg=(k, n))

    def test_binomial_sums_and_argument_checks(self):
        from autoresearch.domains.alpha.null_calibration import binomial_cdf, binomial_tail, clopper_pearson

        self.assertAlmostEqual(binomial_cdf(2, 4, 0.5), 11 / 16, places=12)
        self.assertAlmostEqual(binomial_tail(3, 4, 0.5), 5 / 16, places=12)
        self.assertEqual(binomial_cdf(4, 4, 0.3), 1.0)
        self.assertEqual(binomial_tail(0, 4, 0.3), 1.0)
        self.assertEqual(binomial_cdf(-1, 4, 0.3), 0.0)
        for bad in ((1, 0), (-1, 5), (6, 5)):
            with self.assertRaises(ValueError):
                clopper_pearson(*bad)
        with self.assertRaises(ValueError):
            clopper_pearson(1, 5, confidence=1.0)


class SeedAndSummaryTests(unittest.TestCase):
    def test_replicate_seeds_are_deterministic_distinct_and_nonnegative(self):
        from autoresearch.domains.alpha.null_calibration import replicate_seeds

        seeds = [replicate_seeds(2026, i) for i in range(500)]
        self.assertEqual(seeds, [replicate_seeds(2026, i) for i in range(500)])
        self.assertEqual(len({panel for panel, _ in seeds}), 500)
        self.assertTrue(all(0 <= panel < 2 ** 31 and 0 <= search < 2 ** 31 for panel, search in seeds))
        self.assertNotEqual(replicate_seeds(2026, 0), replicate_seeds(2027, 0))
        with self.assertRaises(ValueError):
            replicate_seeds(-1, 0)

    def test_summary_counts_campaigns_not_trials(self):
        from autoresearch.domains.alpha.null_calibration import summarize

        def replicate(index, trials, promotions, finalists):
            return {"index": index, "panel_seed": 1, "search_seed": 2, "campaign_id": "c", "status": "complete",
                    "budget": "round budget reached", "rounds": 8, "screen_trials": 100 + index, "screen_errors": 0,
                    "validation_trials": trials, "promotions": promotions, "finalists": finalists,
                    "holdout_accessed": False, "ledger_ok": True, "promoted": [], "seconds": 1.0}

        results = [replicate(0, 0, 0, 0), replicate(1, 4, 2, 1), replicate(2, 6, 0, 0), replicate(3, 1, 1, 0)]
        summary = summarize(results, 0.05)
        self.assertEqual(summary["promotion_at_review"]["events"], 2)
        self.assertEqual(summary["promotion_at_review"]["of"], 4)
        self.assertEqual(summary["holdout_reach"]["events"], 1)
        self.assertEqual(summary["holdout_read"]["events"], 0)
        self.assertEqual(summary["validated_any"]["events"], 3)
        self.assertEqual(summary["promotion_at_review_given_validation"], {**summary["promotion_at_review_given_validation"],
                                                                           "events": 2, "of": 3})
        self.assertEqual(summary["per_trial"]["events"], 3)
        self.assertEqual(summary["per_trial"]["of"], 11)
        self.assertEqual(summary["validation_trials"]["histogram"], {"0": 1, "1": 1, "4": 1, "6": 1})
        self.assertEqual(summary["validation_trials"]["median"], 2.5)
        self.assertFalse(summary["promotion_upper_bound_below_nominal"])
        self.assertTrue(summary["all_complete"] and summary["all_ledgers_ok"])
        clean = summarize([replicate(i, 3, 0, 0) for i in range(200)], 0.05)
        self.assertTrue(clean["promotion_upper_bound_below_nominal"])
        self.assertAlmostEqual(clean["promotion_at_review"]["ci95"][1], 1 - 0.025 ** (1 / 200), places=5)
        with self.assertRaises(ValueError):
            summarize([], 0.05)


@unittest.skipUnless(HAVE_NUMERIC, "numpy and pandas are optional outside the Qlib environment")
class GeneratorOptionTests(unittest.TestCase):
    @unittest.skipUnless(HAVE_NUMERIC and __import__("numpy").__version__.startswith("1.26."),
                         "the byte-identical digest was recorded under the pinned numpy 1.26; other versions reseed the stream")
    def test_default_generator_output_is_unchanged(self):
        # Checksum of generate(40, 320, seed=7) recorded before the signal-profile and dated-universe options existed.
        import numpy as np

        from autoresearch.domains.alpha.planted import generate

        data = generate(40, 320, seed=7)
        digest = hashlib.sha256()
        for field in ("open", "high", "low", "close", "volume"):
            digest.update(np.ascontiguousarray(data["fields"][field]).tobytes())
        for name in ("reversal", "volume", "lowvol"):
            digest.update(np.ascontiguousarray(data["planted"][name]).tobytes())
        self.assertEqual(digest.hexdigest(), "0b557555b2dd6cdd0494c0ea98ec99b5324163fe7b828918cdd8c0b4726a7c43")
        self.assertEqual(data["signal_profile"], "stationary")

    def test_signal_profiles_keep_the_average_coefficient(self):
        import numpy as np

        from autoresearch.domains.alpha.planted import signal_profile

        self.assertTrue(np.all(signal_profile("stationary", 500) == 1.0))
        decay = signal_profile("decay", 500)
        self.assertAlmostEqual(float(decay.mean()), 1.0, places=9)
        self.assertEqual((decay[0], decay[-1]), (2.0, 0.0))
        self.assertTrue(np.all(np.diff(decay) < 0))
        blocks = signal_profile("intermittent", 400, block_sessions=100)
        self.assertAlmostEqual(float(blocks.mean()), 1.0, places=9)
        self.assertTrue(np.all(blocks[:100] == 2.0) and np.all(blocks[100:200] == 0.0) and np.all(blocks[200:300] == 2.0))
        with self.assertRaises(ValueError):
            signal_profile("seasonal", 500)
        with self.assertRaises(ValueError):
            signal_profile("intermittent", 500, block_sessions=0)

    def _information_coefficient(self, data, name, first, last):
        import numpy as np

        close = data["fields"]["close"]
        returns = np.diff(np.log(close), axis=0)            # returns[t - 1] is the close-to-close return of session t
        z = data["planted"][name]
        values = []
        for t in range(max(first, 22), last):
            predictor, outcome = z[t - 2], returns[t - 1]
            if predictor.std() > 0:
                values.append(float(np.corrcoef(predictor, outcome)[0, 1]))
        return float(np.mean(values))

    def test_decayed_and_intermittent_signals_live_where_the_profile_says(self):
        from autoresearch.domains.alpha.planted import generate

        strong = {"reversal": 0.15}
        decayed = generate(60, 600, seed=3, strength=strong, signal_profile_mode="decay")
        early = self._information_coefficient(decayed, "reversal", 0, 240)
        late = self._information_coefficient(decayed, "reversal", 360, 600)
        self.assertGreater(early, late + 0.08)
        self.assertGreater(early, 0.1)
        switched = generate(60, 600, seed=3, strength=strong, signal_profile_mode="intermittent", block_sessions=150)
        on = self._information_coefficient(switched, "reversal", 300, 450)
        off = self._information_coefficient(switched, "reversal", 150, 300)
        self.assertGreater(on, off + 0.1)
        self.assertLess(abs(off), 0.06)
        stationary = generate(60, 600, seed=3, strength=strong)
        self.assertAlmostEqual(self._information_coefficient(stationary, "reversal", 0, 300),
                               self._information_coefficient(stationary, "reversal", 300, 600), delta=0.06)

    def test_dated_universe_ranks_by_trailing_dollar_volume_without_looking_ahead(self):
        import numpy as np

        from autoresearch.domains.alpha.planted import generate, membership_rows, month_end_sessions

        data = generate(40, 320, seed=5)
        dates, names = data["dates"], data["symbols"]
        dollar = data["fields"]["close"] * data["fields"]["volume"]
        static = membership_rows(dates, names, dollar, universe_size=30, dated=False)
        self.assertEqual(len(static), 30)
        self.assertEqual({(row[1], row[2]) for row in static}, {(str(dates[0]), str(dates[-1]))})
        dated = membership_rows(dates, names, dollar, universe_size=30, dated=True, adv_sessions=20)
        ranking = month_end_sessions(dates)
        self.assertEqual(len(dated), 30 * len(ranking))
        periods = sorted({(row[1], row[2]) for row in dated})
        self.assertEqual(periods[0][0], str(dates[ranking[0] + 1]))            # nothing before the first month end
        self.assertEqual(periods[-1][1], str(dates[-1]))
        for number, session in enumerate(ranking):
            start = str(dates[session + 1])
            members = {row[0] for row in dated if row[1] == start}
            ranks = sorted(row[3] for row in dated if row[1] == start)
            self.assertEqual(ranks, list(range(1, 31)))
            trailing = dollar[max(0, session - 19):session + 1].mean(axis=0)
            expected = {names[i] for i in np.argsort(-trailing, kind="stable")[:30]}
            self.assertEqual(members, expected, f"period {number} must be ranked on data through its ranking session")
        with self.assertRaises(ValueError):
            membership_rows(dates, names, dollar, universe_size=41, dated=True)
        with self.assertRaises(ValueError):
            membership_rows(dates[:15], names, dollar[:15], universe_size=30, dated=True)

    def test_dated_dataset_loads_as_a_time_varying_universe(self):
        from autoresearch.domains.alpha.planted import create_planted_dataset
        from qlib_yahoo.alphalab.simulate import Workspace

        with tempfile.TemporaryDirectory() as tmp:
            dataset = create_planted_dataset(Path(tmp) / "dated", symbols=40, days=320, seed=5, dated_universe=True,
                                             universe_size=30, signal_profile_mode="intermittent", block_sessions=60)
            manifest = json.loads((dataset / "manifest.json").read_text())
            record = manifest["universes"]["top30"]
            self.assertEqual(record["size"], 30)
            self.assertEqual(record["ranking"]["mode"], "trailing_dollar_volume")
            self.assertEqual(manifest["planted"]["signal_profile"], "intermittent")
            self.assertEqual(manifest["planted"]["block_sessions"], 60)
            header = gzip.decompress((dataset / "universe/membership.csv.gz").read_bytes()).decode().splitlines()[0]
            self.assertEqual(header, "instrument,effective_start,effective_end,rank")
            mask = Workspace(dataset).universe("TOP30")
            members = mask.sum(axis=1)
            self.assertEqual(int(members[0]), 0)
            self.assertEqual(int(members.max()), 30)
            self.assertGreater(len({tuple(row) for row in mask.tolist()}), 5)
            static = create_planted_dataset(Path(tmp) / "static", symbols=40, days=320, seed=5, universe_size=30)
            self.assertEqual(json.loads((static / "manifest.json").read_text())["universes"]["top30"]["ranking"],
                             {"mode": "full_sample_dollar_volume"})
            with self.assertRaises(ValueError):
                create_planted_dataset(Path(tmp) / "bad", symbols=40, days=320, universe_size=0)


@unittest.skipUnless(HAVE_NUMERIC, "numpy and pandas are optional outside the Qlib environment")
class NullStudyTests(unittest.TestCase):
    def test_two_replicates_are_recorded_deterministic_and_never_read_the_holdout(self):
        from autoresearch.domains.alpha.null_calibration import REPORT_JSON, REPORT_MARKDOWN, null_replicate, run_null_study

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "study"
            report = run_null_study(root, replicates=2, names=40, sessions=320, rounds=1, proposals=8)
            self.assertEqual(report["kind"], "alpha_null_calibration")
            self.assertEqual(len(report["replicates"]), 2)
            summary = report["summary"]
            self.assertEqual(summary["replicates"], 2)
            self.assertEqual(summary["holdout_read"]["events"], 0)
            self.assertEqual(summary["holdout_read"]["ci95"], [0.0, round(1 - 0.025 ** 0.5, 6)])
            self.assertTrue(summary["all_ledgers_ok"])
            self.assertIn("--replicates 2", report["study"]["command"])
            for replicate in report["replicates"]:
                self.assertFalse(replicate["holdout_accessed"])
                self.assertEqual(replicate["status"], "complete")
                self.assertGreater(replicate["screen_trials"], 0)
                self.assertEqual(len(replicate["promoted"]), replicate["promotions"])
                self.assertLessEqual(replicate["finalists"], replicate["promotions"])
                self.assertTrue((root / f"replicate-{replicate['index']:03d}" / "result.json").exists())
                self.assertFalse(any((root / f"replicate-{replicate['index']:03d}" / "records").glob("finals/*")))
            self.assertNotEqual(report["replicates"][0]["panel_seed"], report["replicates"][1]["panel_seed"])
            written = json.loads((root / REPORT_JSON).read_text())
            self.assertEqual(written["summary"], summary)
            markdown = (root / REPORT_MARKDOWN).read_text()
            self.assertIn("95 % Clopper-Pearson", markdown)
            self.assertIn("2 independent null panels", markdown)
            # The same (study seed, index) elsewhere reproduces the campaign; a finished replicate is reused.
            again = null_replicate(Path(tmp) / "elsewhere", 1, names=40, sessions=320, rounds=1, proposals=8)
            first = report["replicates"][1]
            for key in ("campaign_id", "panel_seed", "search_seed", "screen_trials", "validation_trials", "promotions",
                        "finalists"):
                self.assertEqual(again[key], first[key], key)
            resumed = run_null_study(root, replicates=2, names=40, sessions=320, rounds=1, proposals=8)
            self.assertEqual([r["seconds"] for r in resumed["replicates"]], [r["seconds"] for r in report["replicates"]])
            with self.assertRaises(ValueError):
                run_null_study(root, replicates=0)
            with self.assertRaises(ValueError):
                run_null_study(root, replicates=1, names=10)

    def test_the_command_line_requires_a_replicate_count(self):
        from autoresearch.lab.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            errors = io.StringIO()
            with redirect_stderr(errors):
                code = main(["benchmark", "null", "--root", tmp])
            self.assertEqual(code, 2)
            self.assertIn("--replicates", errors.getvalue())

    def test_planted_benchmark_runs_alone_on_a_dated_universe(self):
        from autoresearch.domains.alpha.benchmark import run_benchmark

        with tempfile.TemporaryDirectory() as tmp:
            report = run_benchmark(Path(tmp), symbols=40, days=320, rounds=1, proposals=8, null_twin=False,
                                   extra_null_seeds=(), dated_universe=True, universe_size=30, signal_profile="decay")
            self.assertNotIn("null", report)
            self.assertEqual(report["null_total"]["panels"], 0)
            self.assertEqual(report["panel"]["universe"], "TOP30")
            self.assertTrue(report["panel"]["dated_universe"])
            self.assertEqual(set(report["planted"]["recall"]), {"reversal", "volume", "lowvol"})
            markdown = (Path(tmp) / "benchmark.md").read_text()
            self.assertIn("decay planted signals", markdown)
            self.assertIn("dated TOP30 universe", markdown)
            self.assertNotIn("null twin", markdown)
            manifest = json.loads((Path(tmp) / "planted-data" / "manifest.json").read_text())
            self.assertEqual(manifest["universes"]["top30"]["ranking"]["mode"], "trailing_dollar_volume")


if __name__ == "__main__":
    unittest.main()
