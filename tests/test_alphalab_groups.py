"""Classification transport, cache identity and dated-workspace regression tests."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

NUMERICAL = bool(importlib.util.find_spec("numpy") and importlib.util.find_spec("pandas"))


@unittest.skipUnless(NUMERICAL, "optional numerical environment")
class ClassificationBatchTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from qlib_yahoo.alphalab.simulate import Workspace, Settings

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = self.root / "prices"
        self.dates = np.arange("2024-01-01", "2024-03-01", dtype="datetime64[D]")
        self.symbols = ["A", "B", "C", "D"]
        t = np.arange(len(self.dates), dtype=float)
        close = np.column_stack([10 + i + t * (.2 + i * .1) + np.sin(t + i) for i in range(4)]).astype("float32")
        self.panel = SimpleNamespace(dataset=self.dataset, dates=self.dates, symbols=self.symbols,
                                     fields={"close": close}, manifest={})

        def workspace(_dataset):
            result = Workspace(self.dataset, panel=self.panel)
            result._universes["TOP4"] = np.ones(close.shape, dtype=bool)
            return result

        self.workspace = workspace
        self.price_identity = {"dataset": str(self.dataset), "source_sha256": "price-source",
                               "symbols": self.symbols, "dates": [str(d) for d in self.dates],
                               "fields": ["close"]}
        self.source = {"path": str(self.root / "groups"), "events_sha256": "a" * 64,
                       "manifest_sha256": "b" * 64, "groups": ["industry", "sector", "country"]}
        codes = np.tile(np.array([0, 0, 1, 1], dtype="int32"), (len(t), 1))
        codes[:6] = -1
        self.overlay = SimpleNamespace(calendar=tuple(str(d) for d in self.dates), symbols=tuple(self.symbols),
            codes={"industry": codes, "sector": codes.copy(), "country": codes.copy()},
            coverage={"industry": {"labeled_cells": int((codes >= 0).sum())}}, identity=self.source)
        self.settings = Settings(universe="TOP4", warmup=8, truncation=.3)
        for target, value in (("qlib_yahoo.alphalab.batch.source_identity", self.price_identity),
                              ("qlib_yahoo.classification.classification_identity", self.source),
                              ("qlib_yahoo.classification.load_classification_groups", self.overlay)):
            stub = patch(target, return_value=value)
            setattr(self, target.rsplit(".", 1)[-1], stub.start())
            self.addCleanup(stub.stop)
        stub = patch("qlib_yahoo.alphalab.batch.Workspace", side_effect=workspace)
        stub.start()
        self.addCleanup(stub.stop)

    def run_alphas(self, expressions, **kwargs):
        from qlib_yahoo.alphalab.batch import run_batch
        return run_batch(self.dataset, expressions, self.settings, data_root=self.root / "output",
                         workers=1, group_overlay=self.source["path"], **kwargs)

    def test_mixed_batch_consumes_overlay_only_where_used_and_reuses_exact_ids(self):
        expressions = ["rank(close)", "group_neutralize(rank(close), industry)"]
        first = self.run_alphas(expressions)
        self.assertTrue(all("error" not in r for r in first), first)
        self.assertNotIn("classification", first[0]["dataset"])
        self.assertNotIn("classification", first[0])
        self.assertEqual(first[1]["classification"]["source_identity"], self.source)
        self.assertEqual(first[1]["dataset"]["classification"], self.source)
        second = self.run_alphas(expressions)
        self.assertTrue(all(r["cached"] for r in second))
        self.assertEqual([r["id"] for r in first], [r["id"] for r in second])
        self.assertEqual(self.classification_identity.call_count, 2)
        self.assertEqual(self.load_classification_groups.call_count, 1)
        plain = self.run_alphas([expressions[0]])[0]
        self.assertEqual(plain["id"], first[0]["id"])
        self.assertEqual(self.classification_identity.call_count, 2)

    def test_changed_classification_changes_only_group_consumer_id(self):
        expressions = ["rank(close)", "group_neutralize(rank(close), industry)"]
        first = self.run_alphas(expressions)
        changed = {**self.source, "events_sha256": "c" * 64}
        self.classification_identity.return_value = changed
        second = self.run_alphas(expressions)
        self.assertEqual(first[0]["id"], second[0]["id"])
        self.assertNotEqual(first[1]["id"], second[1]["id"])

    def test_warm_result_does_not_bypass_source_verification(self):
        from qlib_yahoo.dataset import DatasetError
        expr = ["group_neutralize(close, industry)"]
        self.run_alphas(expr)
        self.classification_identity.side_effect = DatasetError("classification events SHA-256 mismatch")
        with self.assertRaisesRegex(DatasetError, "SHA-256 mismatch"):
            self.run_alphas(expr)

    def test_portfolio_neutralization_consumes_groups_without_expression_reference(self):
        from dataclasses import replace
        self.settings = replace(self.settings, neutralization="INDUSTRY")
        result = self.run_alphas(["rank(close)"])[0]
        self.assertNotIn("error", result)
        self.assertIn("classification", result)
        self.assertLess(result["summary"]["max_abs_group_net_exposure"], 1e-10)

    def test_misaligned_dated_codes_are_rejected(self):
        from qlib_yahoo.dataset import DatasetError
        self.overlay.symbols = ("A", "B", "C", "OTHER")
        with self.assertRaisesRegex(DatasetError, "calendar or symbols"):
            self.run_alphas(["group_neutralize(close, sector)"])

    def test_labels_follow_explicit_symbols_when_inventory_sort_order_differs(self):
        import numpy as np
        from qlib_yahoo.alphalab.batch import _aligned_groups
        names = ["AGM-A", "AGM", "BH-A", "BH"]
        canonical = sorted(names)
        codes = np.tile(np.array([1, 2, 3, 4], dtype="int32"), (len(self.dates), 1))
        overlay = SimpleNamespace(symbols=canonical, calendar=self.overlay.calendar, codes={"industry": codes})
        workspace = SimpleNamespace(symbols=names, dates=self.dates)
        result = _aligned_groups(overlay, workspace)
        np.testing.assert_array_equal(result["industry"][0], [2, 1, 4, 3])
        np.testing.assert_array_equal(codes[0], [1, 2, 3, 4])

    def test_local_variable_named_industry_does_not_consume_source(self):
        result = self.run_alphas(["industry = rank(close); industry"])[0]
        self.assertNotIn("error", result)
        self.classification_identity.assert_not_called()

    def test_dated_codes_are_mapped_sliced_and_released(self):
        import numpy as np
        from qlib_yahoo.alphalab.batch import write_cache, CachedWorkspace
        source = self.workspace(self.dataset)
        source.identity = self.price_identity
        source.groups = self.overlay.codes
        folder = write_cache(source, self.root / "cache", ["TOP4"])
        with closing(CachedWorkspace(folder)) as cached:
            matrix = cached.groups["industry"]
            self.assertIsInstance(matrix, np.memmap)
            context = cached.context(4, 9, cached.universe("TOP4"))
            np.testing.assert_array_equal(context.groups["industry"], self.overlay.codes["industry"][4:10])
        self.assertTrue(matrix._mmap.closed)

    def test_sec_and_classifications_keep_both_provenances_and_stable_individual_ids(self):
        import numpy as np
        from test_sec_runner_integration import config, sec_api, patched_sec_api

        api = sec_api()
        api.materialize.return_value = {"identity": "selected-sec", "path": "fixture"}
        api.load_arrays.return_value = {"sec_assets": self.panel.fields["close"] * np.float32(3)}
        with patched_sec_api(api):
            expr = "group_neutralize(sec_assets / close, industry)"
            first = self.run_alphas([expr], sec=config())[0]
            together = self.run_alphas(["rank(close)", expr], sec=config(), force=True)
        self.assertNotIn("error", first)
        self.assertEqual(first["id"], together[1]["id"])
        self.assertIn("sec", first["dataset"])
        self.assertIn("classification", first["dataset"])
        self.assertIn("sec", first)
        self.assertIn("classification", first)
        self.assertNotIn("sec", together[0]["dataset"])
        self.assertNotIn("classification", together[0]["dataset"])

    def test_console_worker_forwards_classification_source(self):
        from dataclasses import asdict
        from qlib_yahoo.jobs import _run_request
        request = {"kind": "alpha-batch", "dataset": str(self.dataset), "data_root": str(self.root),
                   "expressions": ["rank(close)"], "settings": asdict(self.settings),
                   "group_overlay": self.source["path"], "workers": 1}
        with patch("qlib_yahoo.alphalab.batch.run_batch", return_value=[]) as run, \
                patch("qlib_yahoo.alphalab.batch.save_batch", return_value=self.root / "batch.json"):
            _run_request(request)
        self.assertEqual(run.call_args.kwargs["group_overlay"], self.source["path"])

    def test_cli_single_without_save_still_uses_verified_group_batch_path(self):
        from qlib_yahoo.__main__ import build_parser
        from qlib_yahoo.alphalab.cli import main
        args = build_parser().parse_args(["--data-root", str(self.root), "alpha", "run", "rank(close)",
            "--dataset", str(self.dataset), "--group-overlay", self.source["path"],
            "--neutralization", "INDUSTRY"])
        with patch("qlib_yahoo.dataset.resolve_dataset", return_value=self.dataset), \
                patch("qlib_yahoo.alphalab.batch.run_batch", return_value=[{}]) as run, \
                patch("qlib_yahoo.alphalab.cli.print_result"):
            main(args)
        self.assertEqual(run.call_args.kwargs["group_overlay"], Path(self.source["path"]))
        self.assertEqual(run.call_args.kwargs["data_root"], self.root)
        self.assertEqual(run.call_args.args[2].neutralization, "INDUSTRY")


if __name__ == "__main__":
    unittest.main()
