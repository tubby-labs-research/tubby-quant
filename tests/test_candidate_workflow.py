"""Selected immutable Lab results to targets, with real tiny price/Lab numerics."""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

NUMERICAL = bool(importlib.util.find_spec("numpy") and importlib.util.find_spec("pandas"))


@unittest.skipUnless(NUMERICAL, "optional numpy/pandas environment")
class CandidateWorkflowTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from qlib_yahoo.alphalab.settings import Settings
        from qlib_yahoo.dataset import tree_sha256

        self.tmp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.dataset = self.root / "prices"
        self.dates = np.arange("2024-01-01", "2024-03-01", dtype="datetime64[D]")
        self.symbols = ["BBB", "AAA", "CCC", "DDD"]
        qlib = self.dataset / "qlib_data"
        (qlib / "calendars").mkdir(parents=True)
        (qlib / "instruments").mkdir()
        (qlib / "calendars/day.txt").write_text("\n".join(map(str, self.dates)) + "\n")
        (qlib / "instruments/all.txt").write_text("".join(
            f"{symbol}\t{self.dates[0]}\t{self.dates[-1]}\n" for symbol in self.symbols))
        fields = ["open", "high", "low", "close", "volume", "factor"]
        t = np.arange(len(self.dates), dtype=float)
        for i, symbol in enumerate(self.symbols):
            close = 10 + i + .07 * (i + 1) * t + np.sin(t + i)
            values = {"open": close * .99, "high": close * 1.03, "low": close * .97,
                      "close": close, "volume": np.full(len(t), 1000.), "factor": np.ones(len(t))}
            folder = qlib / "features" / symbol.lower()
            folder.mkdir(parents=True)
            for field in fields:
                np.concatenate(([0], values[field])).astype("<f4").tofile(folder / f"{field}.day.bin")
        membership = self.dataset / "membership.csv"
        membership.write_text("instrument,rank,effective_start,effective_end\n" + "".join(
            f"{symbol},{i+1},{self.dates[0]},{self.dates[-1]}\n" for i, symbol in enumerate(self.symbols)))
        native_membership = qlib / "instruments" / "top4.txt"
        native_membership.write_text("".join(
            f"{symbol}\t{self.dates[0]}\t{self.dates[-1]}\n" for symbol in self.symbols))
        tree, count = tree_sha256(qlib)
        self.manifest = {"status": "ready", "provider": "synthetic", "fields": fields, "symbols": self.symbols,
                         "qlib_binary_tree_sha256": tree, "qlib_binary_file_count": count,
                         "limitations": ["synthetic prices; no economic PIT evidence"],
                         "universes": {"top4": {"size": 4, "min_history_sessions": 1,
                                                "membership_csv": "membership.csv",
                                                "instruments_file": "instruments/top4.txt",
                                                "instruments_sha256": hashlib.sha256(native_membership.read_bytes()).hexdigest(),
                                                "membership_sha256": hashlib.sha256(membership.read_bytes()).hexdigest()}}}
        self.save_manifest()
        self.settings = Settings(universe="TOP4", start=str(self.dates[10]), end=str(self.dates[-3]),
                                 warmup=8, truncation=.1, decay=2, cost_bps=5)

    def save_manifest(self):
        (self.dataset / "manifest.json").write_text(json.dumps(self.manifest, indent=2) + "\n")

    def record(self, expression="rank(close)", settings=None, **kwargs):
        from qlib_yahoo.alphalab.batch import run_batch
        result = run_batch(self.dataset, [expression], settings or self.settings,
                           data_root=self.root, workers=1, **kwargs)[0]
        self.assertNotIn("error", result, result)
        return result

    def changed_record(self, original, mutate, *, status="completed"):
        from qlib_yahoo.workflow_records import EvidenceStore
        store = EvidenceStore(self.root)
        old = store.get(original["attempt_id"])
        result = copy.deepcopy(store.get_result(old["result_digest"]))
        mutate(result)
        request = copy.deepcopy(old["request"])
        request["settings"] = result["settings"]
        entry = store.begin("lab-evaluation", request, candidate_id=old["candidate_id"],
                            source=old["source"])
        store.finish(entry["id"], status, result=result)
        return entry["id"]

    def test_preview_service_contract_is_cheap_and_uses_attempt_not_library(self):
        from qlib_yahoo.candidate_workflow import preview_candidate
        from qlib_yahoo.dataset import DatasetError
        result = self.record()
        with patch("qlib_yahoo.alphalab.batch.source_identity", side_effect=AssertionError("no source hash")), \
                patch("qlib_yahoo.alphalab.batch.Workspace", side_effect=AssertionError("no panels")), \
                patch("qlib_yahoo.sec_resolution.resolve_config", side_effect=AssertionError("no catalog")), \
                patch("qlib_yahoo.alphalab.batch.Library", side_effect=AssertionError("no mutable library")):
            preview = preview_candidate(self.root, result["attempt_id"])
        self.assertEqual(preview["source_attempt_id"], result["attempt_id"])
        self.assertEqual(preview["result_digest"], result["result_digest"])
        self.assertEqual(preview["dataset"], str(self.dataset))
        self.assertEqual(preview["settings"], asdict(self.settings))
        self.assertEqual(preview["period"], {key: result["summary"][key] for key in ("start", "end")})
        self.assertEqual(preview["qualification"]["status"], "not_verified")
        json.dumps(preview, allow_nan=False)
        with self.assertRaises(DatasetError):
            preview_candidate(self.root, result["id"])
        reused = self.record()
        reused_preview = preview_candidate(self.root, reused["attempt_id"])
        self.assertEqual(reused_preview["source_status"], "reused")
        self.assertEqual(reused_preview["result_digest"], preview["result_digest"])

    def test_real_result_delay_zero_one_exact_decision_alignment_and_no_double_transform(self):
        import numpy as np
        from qlib_yahoo.alphalab import batch
        from qlib_yahoo.alphalab.simulate import simulate
        from qlib_yahoo.candidate_workflow import prepare_candidate_targets
        from qlib_yahoo.target_bundle import load_bundle

        for delay in (0, 1):
            with self.subTest(delay=delay):
                settings = replace(self.settings, delay=delay)
                result = self.record("rank(ts_delta(close, 1))", settings=settings)
                prepared = prepare_candidate_targets(self.root, result["attempt_id"], self.root / f"targets{delay}")
                bundle = load_bundle(prepared["bundle"])
                workspace = batch.Workspace(self.dataset)
                expected = simulate(result["expression"], settings, workspace,
                                    keep_weights=True, keep_decision_targets=True)
                targets = expected["decision_targets"]
                self.assertEqual(bundle["dates"], targets["dates"])
                self.assertEqual(bundle["dates"][0], str(self.dates[9]))
                self.assertEqual(bundle["dates"][-1], str(self.dates[-4]))
                self.assertEqual(len(bundle["dates"]), result["summary"]["days"])
                np.testing.assert_allclose(bundle["weights"], targets["weights"], atol=1e-8, rtol=1e-6)
                self.assertTrue(np.any(np.abs(targets["weights"] - expected["weights"]) > 1e-7))
                self.assertLess(np.abs(bundle["weights"]).sum(axis=1).max(), .5)
                self.assertEqual(prepared["reproduction"]["status"], "matched")
                self.assertEqual(prepared["qualification"]["status"], "exploratory_only")
                raw = (self.dataset / "manifest.json").read_bytes()
                self.assertEqual(bundle["metadata"]["dataset_manifest_sha256"], hashlib.sha256(raw).hexdigest())
                self.assertNotEqual(bundle["metadata"]["dataset_manifest_sha256"],
                                    bundle["metadata"]["price_identity"]["manifest_sha256"])
                self.assertEqual(bundle["metadata"]["attempt_ids"], [result["attempt_id"]])
                ordinary = simulate(result["expression"], settings, workspace)
                self.assertNotIn("decision_targets", ordinary)

    def test_legacy_changed_code_unrecognized_settings_and_missing_bindings_refuse(self):
        from qlib_yahoo.candidate_workflow import preview_candidate
        from qlib_yahoo.dataset import DatasetError
        from qlib_yahoo.workflow_provenance import lab_implementation
        result = self.record()
        for mutation in (lambda r: r.pop("engine"), lambda r: r.pop("implementation"),
                         lambda r: r["settings"].update(mystery=True),
                         lambda r: r["dataset"].pop("verified")):
            attempt_id = self.changed_record(result, mutation)
            with self.assertRaises(DatasetError):
                preview_candidate(self.root, attempt_id)
        changed = lab_implementation()
        changed["sha256"] = "0" * 64
        with patch("qlib_yahoo.workflow_provenance.lab_implementation", return_value=changed):
            with self.assertRaisesRegex(DatasetError, "code or runtime"):
                preview_candidate(self.root, result["attempt_id"])
        for status in ("failed", "rejected", "cancelled", "interrupted_unknown"):
            attempt_id = self.changed_record(result, lambda r: None, status=status)
            with self.assertRaisesRegex(DatasetError, "successful or reused"):
                preview_candidate(self.root, attempt_id)

    def test_literal_close_targets_apply_delay_once(self):
        import numpy as np
        from qlib_yahoo.candidate_workflow import prepare_candidate_targets
        from qlib_yahoo.target_bundle import load_bundle
        for delay in (0, 1):
            settings = replace(self.settings, delay=delay, decay=0, neutralization="NONE", truncation=0)
            result = self.record("close", settings=settings)
            bundle = load_bundle(prepare_candidate_targets(
                self.root, result["attempt_id"], self.root / f"literal{delay}")["bundle"])
            # Read the literal source rows, independent of evaluate/weights/simulate.
            close = np.column_stack([np.fromfile(
                self.dataset / f"qlib_data/features/{symbol.lower()}/close.day.bin", dtype="<f4")[1:]
                for symbol in self.symbols]).astype(float)
            raw = close[9-delay:len(self.dates)-3-delay]
            expected = raw / np.abs(raw).sum(axis=1, keepdims=True)
            np.testing.assert_allclose(bundle["weights"], expected, atol=1e-7, rtol=1e-6)

    def test_immutable_result_byte_tampering_refuses(self):
        from qlib_yahoo.candidate_workflow import preview_candidate
        from qlib_yahoo.dataset import DatasetError
        result = self.record()
        path = self.root / "workflow/results" / (result["result_digest"] + ".json")
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(DatasetError, "digest mismatch"):
            preview_candidate(self.root, result["attempt_id"])

    def test_nonfinite_record_and_nonfinite_reconstruction_never_publish(self):
        import numpy as np
        from qlib_yahoo.candidate_workflow import prepare_candidate_targets, preview_candidate
        from qlib_yahoo.alphalab.simulate import simulate
        from qlib_yahoo.dataset import DatasetError
        from qlib_yahoo.workflow_records import EvidenceStore
        result = self.record()
        store = EvidenceStore(self.root)
        bad = store.get_result(result["result_digest"])
        bad["summary"]["pnl"] = float("inf")
        with patch.object(EvidenceStore, "get_result", return_value=bad), self.assertRaises(DatasetError):
            preview_candidate(self.root, result["attempt_id"])
        def nonfinite(*args, **kwargs):
            value = simulate(*args, **kwargs)
            value["decision_targets"]["weights"][0, 0] = np.nan
            return value
        with patch("qlib_yahoo.alphalab.simulate.simulate", side_effect=nonfinite):
            with self.assertRaisesRegex(DatasetError, "nonfinite"):
                prepare_candidate_targets(self.root, result["attempt_id"], self.root / "bad")
        self.assertFalse((self.root / "bad").exists())

    def test_numerical_mismatch_is_not_reproduction(self):
        from qlib_yahoo.candidate_workflow import prepare_candidate_targets
        from qlib_yahoo.dataset import DatasetError
        result = self.record()
        attempt_id = self.changed_record(result, lambda r: r["daily"]["pnl"].__setitem__(0, 123456.))
        with self.assertRaisesRegex(DatasetError, "reproduction"):
            prepare_candidate_targets(self.root, attempt_id, self.root / "mismatch")
        self.assertFalse((self.root / "mismatch").exists())

    def test_price_and_membership_checksum_failures_refuse(self):
        from qlib_yahoo.candidate_workflow import prepare_candidate_targets
        from qlib_yahoo.dataset import DatasetError
        result = self.record()
        path = self.dataset / "qlib_data/features/aaa/close.day.bin"
        original = path.read_bytes()
        path.write_bytes(original + b"changed")
        with self.assertRaises(DatasetError):
            prepare_candidate_targets(self.root, result["attempt_id"], self.root / "bad-price")
        path.write_bytes(original)
        (self.dataset / "membership.csv").write_text("changed")
        with self.assertRaisesRegex(DatasetError, "membership"):
            prepare_candidate_targets(self.root, result["attempt_id"], self.root / "bad-membership")
        self.assertFalse((self.root / "bad-price").exists())
        self.assertFalse((self.root / "bad-membership").exists())

    def test_load_time_source_change_is_refused(self):
        from qlib_yahoo.alphalab import batch
        from qlib_yahoo.candidate_workflow import prepare_candidate_targets
        from qlib_yahoo.dataset import DatasetError
        result = self.record()
        identity = batch.source_identity(self.dataset)
        changed = {**identity, "source_sha256": "changed"}
        with patch.object(batch, "source_identity", side_effect=[identity, changed]):
            with self.assertRaisesRegex(DatasetError, "price source"):
                prepare_candidate_targets(self.root, result["attempt_id"], self.root / "raced")
        self.assertFalse((self.root / "raced").exists())

    def test_adapter_source_drift_refuses_before_target_publication(self):
        from qlib_yahoo.candidate_workflow import prepare_candidate_targets
        from qlib_yahoo.dataset import DatasetError
        result = self.record()
        destination = self.root / "adapter-drift"
        with patch("qlib_yahoo.candidate_workflow._adapter_sources",
                   side_effect=[{"adapter": "before"}, {"adapter": "after"}]):
            with self.assertRaisesRegex(DatasetError, "adapter implementation"):
                prepare_candidate_targets(self.root, result["attempt_id"], destination)
        self.assertFalse(destination.exists())

    def test_supported_subset_refuses_zero_warmup_and_unmasked_targets_cheaply(self):
        from qlib_yahoo.candidate_workflow import preview_candidate
        from qlib_yahoo.dataset import DatasetError
        result = self.record()
        for key, value, message in (("warmup", 0, "warmup<1"), ("pasteurization", False, "pasteurization=False")):
            attempt_id = self.changed_record(result, lambda r: r["settings"].update({key: value}))
            with patch("qlib_yahoo.alphalab.batch.source_identity", side_effect=AssertionError("cheap refusal")):
                with self.assertRaisesRegex(DatasetError, message):
                    preview_candidate(self.root, attempt_id)

    def test_dated_groups_actual_source_alignment_no_current_label_fallback(self):
        import numpy as np
        from qlib_yahoo.classification import create_classification_overlay
        from qlib_yahoo.candidate_workflow import prepare_candidate_targets, preview_candidate
        from qlib_yahoo.target_bundle import load_bundle
        from qlib_yahoo.dataset import DatasetError
        overlay = create_classification_overlay(self.dataset, self.root / "groups", [
            {"instrument": symbol, "available_at": "2023-12-31T00:00:00Z",
             "industry": "SIC", "sector": "Division", "country": "US"} for symbol in self.symbols],
            taxonomy_metadata={"industry": "synthetic SIC"})
        result = self.record("group_neutralize(rank(close), industry)", group_overlay=overlay,
                             settings=replace(self.settings, neutralization="INDUSTRY"))
        prepared = prepare_candidate_targets(self.root, result["attempt_id"], self.root / "group-targets")
        bundle = load_bundle(prepared["bundle"])
        self.assertLess(np.abs(bundle["weights"].sum(axis=1)).max(), 1e-7)
        self.assertEqual(prepared["qualification"]["groups"]["source"]["taxonomy_metadata"]["industry"], "synthetic SIC")
        self.assertEqual(bundle["symbols"], self.symbols)  # overlay sorts; price inventory does not
        attempt_id = self.changed_record(result, lambda r: r["dataset"].pop("classification"))
        with self.assertRaisesRegex(DatasetError, "dated classification"):
            preview_candidate(self.root, attempt_id)
        (overlay / "events.jsonl").write_bytes(b"changed")
        with self.assertRaisesRegex(DatasetError, "SHA-256"):
            prepare_candidate_targets(self.root, result["attempt_id"], self.root / "corrupt-groups")

    def test_native_calendar_boundaries_refuse_cheaply_without_clipping_settings(self):
        from qlib_yahoo.candidate_workflow import preview_candidate
        from qlib_yahoo.dataset import DatasetError
        # Lab permits clipping the requested warmup at the start of the panel;
        # Main passes the exact warmup to native validation, which cannot do so.
        clipped = self.record(settings=replace(self.settings, warmup=20))
        last = self.record(settings=replace(self.settings, end=str(self.dates[-1])))
        with patch("qlib_yahoo.alphalab.batch.source_identity", side_effect=AssertionError("cheap refusal")):
            with self.assertRaisesRegex(DatasetError, "preceding sessions"):
                preview_candidate(self.root, clipped["attempt_id"])
            with self.assertRaisesRegex(DatasetError, "boundary session"):
                preview_candidate(self.root, last["attempt_id"])

    def test_combined_sec_and_groups_rebuild_batch_identity_order(self):
        import numpy as np
        from test_sec_runner_integration import config, sec_api, patched_sec_api
        from qlib_yahoo.classification import create_classification_overlay
        from qlib_yahoo.candidate_workflow import prepare_candidate_targets
        api = sec_api()
        api.materialize.return_value = {"identity": "selected-sec", "path": "fixture"}
        api.load_arrays.side_effect = lambda handle: {
            "sec_assets": np.full((len(self.dates), len(self.symbols)), 50., dtype="float32")}
        sec_config = config()
        sec_config["catalog"] = str(self.root / "catalog")
        overlay = create_classification_overlay(self.dataset, self.root / "combined-groups", [
            {"instrument": symbol, "available_at": "2023-12-31T00:00:00Z", "industry": "SIC"}
            for symbol in self.symbols])
        with patched_sec_api(api):
            result = self.record("group_neutralize(sec_assets / close, industry)",
                                 sec=sec_config, group_overlay=overlay)
            prepared = prepare_candidate_targets(self.root, result["attempt_id"], self.root / "combined-targets")
        self.assertEqual(prepared["reproduction"]["status"], "matched")
        self.assertIsNotNone(prepared["qualification"]["groups"])
        self.assertIsNotNone(prepared["qualification"]["sec"])

    def test_selected_sec_source_recipe_and_registry_pins(self):
        import numpy as np
        from test_sec_runner_integration import config, sec_api, patched_sec_api
        from qlib_yahoo.candidate_workflow import prepare_candidate_targets, preview_candidate
        from qlib_yahoo.dataset import DatasetError
        api = sec_api()
        api.materialize.return_value = {"identity": "selected-sec", "path": "fixture"}
        api.load_arrays.side_effect = lambda handle: {
            "sec_assets": np.full((len(self.dates), len(self.symbols)), 50., dtype="float32")}
        sec_config = config()
        sec_config["catalog"] = str(self.root / "catalog")
        with patched_sec_api(api):
            result = self.record("sec_assets / close", sec=sec_config)
            api.materialize.reset_mock()
            api.feature_identity.reset_mock()
            prepared = prepare_candidate_targets(self.root, result["attempt_id"], self.root / "sec-targets")
            self.assertEqual(prepared["reproduction"]["status"], "matched")
            self.assertEqual(set(api.materialize.call_args.args[1]), {"sec_assets"})
            self.assertFalse(api.materialize.call_args.kwargs["force_verify"])
            self.assertTrue(all(set(call.args[1]) == {"sec_assets"} for call in api.feature_identity.call_args_list))
            api.feature_identity.side_effect = lambda catalog, bindings: {"changed": True}
            with self.assertRaisesRegex(DatasetError, "source identities"):
                prepare_candidate_targets(self.root, result["attempt_id"], self.root / "changed-sec")
            attempt_id = self.changed_record(result, lambda r: r["dataset"]["sec"].pop("bindings"))
            with self.assertRaisesRegex(DatasetError, "SEC"):
                preview_candidate(self.root, attempt_id)

    def test_dictionary_selected_recipes_are_verified_and_cannot_be_reinterpreted(self):
        import numpy as np
        from test_sec_runner_integration import config, sec_api, patched_sec_api
        from qlib_yahoo.candidate_workflow import prepare_candidate_targets
        from qlib_yahoo.dataset import DatasetError
        api = sec_api()
        api.materialize.return_value = {"identity": "selected-sec", "path": "fixture"}
        api.load_arrays.side_effect = lambda handle: {
            "sec_assets": np.full((len(self.dates), len(self.symbols)), 50., dtype="float32")}
        catalog = str(self.root / "catalog")
        registry = {"version": 1, "catalog_sha256": "a" * 64, "dictionary_sha256": "b" * 64,
                    "execution": {"sha256": "c" * 64}}
        resolved = {"catalog": catalog, "mode": "dictionary",
                    "bindings": {"sec_assets": config()["bindings"]["sec_assets"]}, "registry": registry}
        # This bounded fixture replaces the source API, not the Lab calculation.
        # Dedicated SEC tests verify real catalog/index formats; here the boundary
        # must forward its recorded pin and freshly verify only the used recipe.
        with patched_sec_api(api), patch("qlib_yahoo.sec_resolution.resolve_config", return_value=resolved) as resolve:
            result = self.record("sec_assets / close", sec={"catalog": catalog, "mode": "dictionary"})
            resolve.reset_mock()
            prepared = prepare_candidate_targets(self.root, result["attempt_id"], self.root / "dictionary-targets")
            self.assertEqual(prepared["qualification"]["sec"]["source"]["registry"], registry)
            self.assertEqual(resolve.call_count, 2)  # before and after loading
            for call in resolve.call_args_list:
                self.assertTrue(call.kwargs["verify"])
                self.assertEqual(call.args[1]["registry"], registry)
                self.assertEqual(set(call.args[1]["bindings"]), {"sec_assets"})
            changed = copy.deepcopy(resolved)
            changed["bindings"]["sec_assets"]["tag"] = "Liabilities"
            resolve.return_value = changed
            with self.assertRaisesRegex(DatasetError, "recipes"):
                prepare_candidate_targets(self.root, result["attempt_id"], self.root / "wrong-recipe")
            changed = copy.deepcopy(resolved)
            changed["registry"]["dictionary_sha256"] = "d" * 64
            resolve.return_value = changed
            with self.assertRaisesRegex(DatasetError, "registry"):
                prepare_candidate_targets(self.root, result["attempt_id"], self.root / "wrong-registry")

    def test_missing_factor_or_nonidentity_factor_refuses(self):
        import numpy as np
        from qlib_yahoo.candidate_workflow import prepare_candidate_targets
        from qlib_yahoo.dataset import DatasetError, tree_sha256
        self.manifest["fields"].remove("factor")
        self.save_manifest()
        result = self.record()
        with self.assertRaisesRegex(DatasetError, "factor"):
            prepare_candidate_targets(self.root, result["attempt_id"], self.root / "missing-factor")
        self.manifest["fields"].append("factor")
        factor = self.dataset / "qlib_data/features/aaa/factor.day.bin"
        values = np.fromfile(factor, dtype="<f4")
        values[15] = 2
        values.tofile(factor)
        self.manifest["qlib_binary_tree_sha256"] = tree_sha256(self.dataset / "qlib_data")[0]
        self.save_manifest()
        result = self.record()
        with self.assertRaisesRegex(DatasetError, "factor=1"):
            prepare_candidate_targets(self.root, result["attempt_id"], self.root / "nonidentity-factor")

    def test_numerical_summary_mismatch_and_existing_destination_refuse(self):
        from qlib_yahoo.candidate_workflow import prepare_candidate_targets
        from qlib_yahoo.dataset import DatasetError
        result = self.record()
        attempt_id = self.changed_record(result, lambda r: r["summary"].update(gross_exposure=.99))
        with self.assertRaisesRegex(DatasetError, "summary"):
            prepare_candidate_targets(self.root, attempt_id, self.root / "wrong-summary")
        directory = self.root / "existing"
        directory.mkdir()
        sentinel = directory / "preserve.txt"
        sentinel.write_text("keep")
        with self.assertRaisesRegex(DatasetError, "new directory"):
            prepare_candidate_targets(self.root, result["attempt_id"], directory)
        self.assertEqual(sentinel.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
