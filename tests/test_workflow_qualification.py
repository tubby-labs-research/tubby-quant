"""Real tiny SEC import/dictionary/materialization + dated-group handoff tests.

These are synthetic causal/integrity checks, not economic PIT certification or
installed-Qlib acceptance. No SEC resolution/materialization APIs are stubbed.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HAVE_INPUTS = all(importlib.util.find_spec(name) for name in ("numpy", "pandas", "pyarrow"))

try:
    from .workflow_fixture_helpers import build_sec_group_fixture
except ImportError:
    from workflow_fixture_helpers import build_sec_group_fixture


@unittest.skipUnless(HAVE_INPUTS, "optional numpy/pandas/pyarrow input environment")
class WorkflowQualificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix=".workflow-qualification-", dir=Path.cwd())
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()

    def build(self, name="base", **changes):
        return build_sec_group_fixture(self.root / name, **changes)

    def prepare(self, fixture):
        from qlib_yahoo.target_bundle import load_bundle
        result = fixture.record()
        self.assertNotIn("error", result, result)
        prepared = fixture.prepare(result)
        return result, prepared, load_bundle(prepared["bundle"])

    def test_real_selected_sec_and_groups_reproduce_recorded_lab_result(self):
        import numpy as np
        from qlib_yahoo.sec_resolution import resolve_config
        from qlib_yahoo.workflow_records import EvidenceStore
        fixture = self.build()
        inputs, handle = fixture.selected_panel()
        np.testing.assert_array_equal(inputs[0], fixture.initial_values)
        self.assertEqual(set(handle["field_objects"]), {fixture.names["base"]})
        resolved = resolve_config([fixture.expression], fixture.sec, dialect="lab", verify=True)
        self.assertEqual(resolved["bindings"], {fixture.names["base"]: fixture.recipes["base"]})
        self.assertIn("execution", resolved["registry"])
        result, prepared, bundle = self.prepare(fixture)
        self.assertEqual(prepared["reproduction"]["status"], "matched")
        self.assertEqual(prepared["preview"]["execution_universe"], "top6")
        self.assertTrue(np.isfinite(bundle["weights"]).all())
        self.assertGreater(np.count_nonzero(bundle["weights"]), 0)
        self.assertLessEqual(np.abs(bundle["weights"]).sum(axis=1).max(), 1 + 1e-7)
        self.assertEqual(set(result["dataset"]["sec"]["bindings"]), {fixture.names["base"]})
        self.assertEqual(result["dataset"]["classification"]["path"], str(fixture.overlay))
        frozen = EvidenceStore(fixture.root).get_result(result["result_digest"])
        self.assertEqual(frozen["daily"], result["daily"])
        qualification = prepared["qualification"]
        self.assertEqual(qualification["status"], "exploratory_only")
        self.assertEqual(qualification["checks"]["source_bytes"], "verified")
        self.assertIn("inferred", qualification["sec"]["limitations"])
        self.assertIn("current ticker", qualification["sec"]["limitations"])
        self.assertIn("synthetic", qualification["groups"]["source"]["taxonomy_metadata"]["industry"])
        self.assertIn("share-adjustment", " ".join(qualification["limitations"]))
        # Five observed recipe contexts do not trigger five daily panels.
        objects = fixture.root / "cache/sec/fields-v1"
        self.assertEqual(len([p for p in objects.iterdir() if p.is_dir()]), 1)

    def assert_future_fact_causal(self, change, filing_index, symbol_index, old_value, new_value):
        import numpy as np
        import pandas as pd
        baseline = self.build("baseline")
        altered = self.build("altered", **change)
        left, _ = baseline.selected_panel()
        right, _ = altered.selected_panel()
        # Independent calendar-bound oracle: later filed/accepted day + 2
        # calendar days, then first supplied price session on/after that bound.
        bound = baseline.days[filing_index] + pd.Timedelta(days=2)
        eligible = int(baseline.days.searchsorted(bound))
        np.testing.assert_array_equal(left[:eligible], right[:eligible])
        self.assertEqual(left[eligible, symbol_index], old_value)
        self.assertEqual(right[eligible, symbol_index], new_value)
        _, left_prepared, left_bundle = self.prepare(baseline)
        _, right_prepared, right_bundle = self.prepare(altered)
        self.assertEqual(left_bundle["dates"], right_bundle["dates"])
        dates = np.asarray(left_bundle["dates"], dtype="datetime64[D]")
        # Delay 1 is already applied: decisions ON the eligible session still
        # depend on yesterday's input. Future values cannot affect that row.
        earlier = dates <= np.datetime64(baseline.days[eligible].date())
        np.testing.assert_array_equal(left_bundle["weights"][earlier], right_bundle["weights"][earlier])
        self.assertTrue(np.any(np.abs(left_bundle["weights"][~earlier] -
                                      right_bundle["weights"][~earlier]) > 1e-6))
        self.assertNotEqual(left_prepared["preview"]["source_bindings"]["sec"]["source_identity"],
                            right_prepared["preview"]["source_bindings"]["sec"]["source_identity"])

    def test_future_newer_filing_does_not_rewrite_earlier_inputs_or_targets(self):
        self.assert_future_fact_causal({"future_filing": 7000.}, 30, 0, 700., 7000.)

    def test_future_revision_does_not_rewrite_earlier_inputs_or_targets(self):
        self.assert_future_fact_causal({"future_revision": 4500.}, 34, 1, 450., 4500.)

    def test_future_classification_clear_keeps_earlier_groups_and_targets(self):
        import numpy as np
        baseline = self.build("baseline")
        altered = self.build("altered", future_industry=None)
        left_inputs, _ = baseline.selected_panel()
        right_inputs, _ = altered.selected_panel()
        np.testing.assert_array_equal(left_inputs, right_inputs)
        left_groups, _ = baseline.groups()
        right_groups, _ = altered.groups()
        for name in ("industry", "sector", "country"):
            np.testing.assert_array_equal(left_groups[name][:39], right_groups[name][:39])
        self.assertGreaterEqual(right_groups["industry"][38, 2], 0)
        self.assertEqual(right_groups["industry"][39, 2], -1)  # equal-open event activates next session
        _, _, left = self.prepare(baseline)
        _, _, right = self.prepare(altered)
        before = np.asarray(left["dates"], dtype="datetime64[D]") < np.datetime64(baseline.days[39].date())
        np.testing.assert_array_equal(left["weights"][before], right["weights"][before])
        self.assertTrue(np.any(np.abs(left["weights"][~before] - right["weights"][~before]) > 1e-6))
        self.assertTrue(np.all(right["weights"][~before, 2] == 0))

    def test_currency_quarter_segment_and_coreg_recipes_never_alias_base(self):
        import numpy as np
        from qlib_yahoo.sec_features import feature_identity
        from qlib_yahoo.sec_resolution import resolve_config
        fixture = self.build()
        base_result = fixture.record()
        self.assertNotIn("error", base_result, base_result)
        identities = []
        for offset, name in enumerate(("currency", "quarter", "segment", "coreg"), 1):
            with self.subTest(recipe=name):
                public = fixture.names[name]
                inputs, _ = fixture.selected_panel(name)
                np.testing.assert_array_equal(inputs[0], np.asarray(fixture.initial_values) + 1000 * offset)
                resolved = resolve_config([public], fixture.sec, dialect="lab", verify=True)
                self.assertEqual(resolved["bindings"][public], fixture.recipes[name])
                identity = feature_identity(fixture.catalog, resolved["bindings"])
                identities.append(json.dumps(identity, sort_keys=True))
                result = fixture.record(expression=f"{public} / close")
                self.assertNotIn("error", result, result)
                self.assertNotEqual(result["id"], base_result["id"])
                self.assertNotEqual(result["dataset"]["source_sha256"], base_result["dataset"]["source_sha256"])
                self.assertEqual(fixture.prepare(result, name=f"targets-{name}")["reproduction"]["status"], "matched")
                # Same dictionary name with a different supplied definition is
                # refused; callers cannot relabel EUR/q1/context facts as USD q0.
                conflicting = {**fixture.sec, "bindings": {fixture.names["base"]: fixture.recipes[name]}}
                with self.assertRaisesRegex(ValueError, "different definition"):
                    resolve_config([fixture.names["base"]], conflicting, dialect="lab", verify=True)
        self.assertEqual(len(set(identities)), 4)

    def test_late_initial_filing_stays_missing_until_eligible_and_is_qualified(self):
        import numpy as np
        import pandas as pd
        fixture = self.build(late_initial=True)
        values, _ = fixture.selected_panel()
        eligible = int(fixture.days.searchsorted(fixture.days[20] + pd.Timedelta(days=2)))
        self.assertTrue(np.isnan(values[:eligible]).all())
        np.testing.assert_array_equal(values[eligible], fixture.initial_values)
        result, prepared, bundle = self.prepare(fixture)
        dates = np.asarray(bundle["dates"], dtype="datetime64[D]")
        earlier = dates <= np.datetime64(fixture.days[eligible].date())
        self.assertTrue((bundle["weights"][earlier] == 0).all())
        self.assertTrue(np.any(bundle["weights"][~earlier] != 0))
        self.assertGreater(result["summary"]["inactive_neutralized_days"], 0)
        self.assertEqual(prepared["qualification"]["status"], "exploratory_only")
        self.assertIn("missing coverage stays missing", prepared["qualification"]["sec"]["limitations"])

    def test_unknown_group_is_excluded_and_partial_coverage_visible(self):
        import numpy as np
        fixture = self.build(unknown_symbols=("TST2",))
        result, prepared, bundle = self.prepare(fixture)
        self.assertTrue(np.all(bundle["weights"][:, 2] == 0))
        coverage = prepared["qualification"]["groups"]["coverage"]["industry"]
        self.assertEqual(coverage["unknown_cells"], len(fixture.days))
        self.assertLess(result["summary"]["classification_coverage"], 1)
        self.assertIn("coverage", prepared["qualification"]["groups"]["limitations"])

    def test_no_available_groups_or_unobserved_base_recipe_refuses(self):
        from qlib_yahoo.candidate_workflow import prepare_candidate_targets
        from qlib_yahoo.dataset import DatasetError
        from qlib_yahoo.sec_resolution import resolve_config
        from qlib_yahoo.workflow_records import EvidenceStore
        fixture = self.build("unknown", all_unknown=True)
        result = fixture.record()
        self.assertIn("error", result)
        self.assertIn("classification", result["error"])
        self.assertEqual(EvidenceStore(fixture.root).get(result["attempt_id"])["status"], "failed")
        with self.assertRaises(DatasetError):
            prepare_candidate_targets(fixture.root, result["attempt_id"], fixture.root / "targets")
        missing = self.build("missing-recipe", include_base=False)
        with self.assertRaisesRegex(ValueError, "unavailable|not observed"):
            resolve_config(["sec_assets / close"], missing.sec, dialect="lab", verify=True)
        with self.assertRaisesRegex(ValueError, "unavailable|not observed"):
            missing.record(expression="sec_assets / close")
        self.assertFalse((missing.root / "targets").exists())

    def test_real_selected_source_checksum_failure_refuses_even_with_warm_cache(self):
        from qlib_yahoo.candidate_workflow import prepare_candidate_targets
        from qlib_yahoo.dataset import DatasetError
        fixture = self.build()
        result = fixture.record()
        self.assertNotIn("error", result, result)
        # The original evaluation has populated a selected daily-panel cache.
        # Damage only this disposable catalog's immutable NUM object. A warm
        # selected panel must not bypass genuine source verification.
        manifest = json.loads((fixture.catalog / "manifest.json").read_text())
        relative = manifest["tables"]["num"]["files"][0]
        path = fixture.catalog / relative
        path.write_bytes(path.read_bytes() + b"checksum corruption")
        with self.assertRaises((DatasetError, ValueError)):
            prepare_candidate_targets(fixture.root, result["attempt_id"], fixture.root / "corrupt-targets")
        self.assertFalse((fixture.root / "corrupt-targets").exists())


if __name__ == "__main__":
    unittest.main()
