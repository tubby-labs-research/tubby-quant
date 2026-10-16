"""Saved batch cache badges: real frontend functions and bounded service receipts."""

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import test_qlib_recovery_console as frontend
from qlib_yahoo.webapp.service import ConsoleService


SETUP = """
  const rows = [0, 1].map(i => ({id: String(i), expression: 'rank(close)',
    engine: 'alphalab-3', summary: {}, checks: []}));
  function badge(batch) {
    state.lab.batch = batch;
    renderLabBatch();
    return $('#lab-body').children[0].children[1].children.map(tag => tag.text);
  }
  function render(diagnostics, selectedRows = rows) {
    return badge({rows: selectedRows, settings: {}, top_pairs: [], diagnostics});
  }
  function receipt(reused, computed) {
    return {version: 1, alphalab: {cache_hit_count: reused, new_computation_count: computed}};
  }
"""


@unittest.skipUnless(shutil.which("node"), "Node is needed for frontend behavior")
class BatchCacheFrontendTests(unittest.TestCase):
    js = frontend.RecoveryHelperTests.js

    def render_js(self, code):
        return self.js(SETUP + code)

    def test_cached_fresh_and_mixed_receipts(self):
        result = self.render_js("""
          return [[2, 0], [0, 2], [1, 1]].map(([hits, computed]) => render(receipt(hits, computed)));
        """)
        for tags, expected in zip(result, (
            "2 results · 2 reused · 0 computed",
            "2 results · 0 reused · 2 computed",
            "2 results · 1 reused · 1 computed",
        )):
            self.assertIn(expected, tags)
            self.assertFalse(any("simulated" in tag for tag in tags))

    def test_legacy_missing_and_unknown_version_are_neutral(self):
        result = self.render_js("""
          return [undefined, null, {}, {version: 1}, {version: 2, alphalab: receipt(2, 0).alphalab},
            {version: '1', alphalab: receipt(2, 0).alphalab}].map(d => render(d));
        """)
        for tags in result:
            self.assertEqual(tags[-1], "2 results")

    def test_malformed_counts_fall_back_without_coercion_or_out_of_bounds(self):
        result = self.render_js("""
          const invalid = [undefined, null, '2', true, -1, 0.5, NaN, Infinity, -Infinity,
            3, Number.MAX_SAFE_INTEGER + 1, {}, []];
          return [...invalid.flatMap(value => [receipt(value, 0), receipt(0, value)]),
            receipt(2, 1), {version: 1, alphalab: null}, {version: 1, alphalab: []}]
            .map(d => render(d));
        """)
        for tags in result:
            self.assertEqual(tags[-1], "2 results")

    def test_distinct_computations_need_not_partition_results(self):
        # Two duplicate inputs may require only one computation.
        tags = self.render_js("return render(receipt(0, 1));")
        self.assertIn("2 results · 0 reused · 1 computed", tags)

    def test_failed_computation_and_empty_batch(self):
        result = self.render_js("""
          return {failed: render(receipt(0, 2), [rows[0], {id: 'bad', expression: 'bad', error: 'failed'}]),
            empty: render(receipt(0, 0), [])};
        """)
        self.assertIn("1 results · 0 reused · 2 computed", result["failed"])
        self.assertIn("1 failed", result["failed"])
        self.assertIn("0 results · 0 reused · 0 computed", result["empty"])

    def test_ui_version_is_preserved(self):
        self.assertEqual(self.js("return UI_VERSION;"), "workbench-cleanup-2026-10-14")


@unittest.skipUnless(importlib.util.find_spec("numpy"), "optional Lab runtime")
class BatchCacheServiceTests(unittest.TestCase):
    def response(self, diagnostics=None):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root = Path(tmp).resolve()
            service = ConsoleService(root, settings_path=root / "settings.json")
            library = service._alpha_library()
            for identifier in ("one", "two"):
                library.save(identifier, {"expression": "rank(close)", "summary": {},
                                         "settings": {}, "engine": "alphalab-3"}, "fixture")
            record = {"ids": ["one", "two"], "settings": {}, "created_at_utc": "2026-10-13T00:00:00Z"}
            if diagnostics is not None:
                record["diagnostics"] = diagnostics
            path = root / "alphalab/batch.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            response = service.alpha_batch(str(path))
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), record)
            self.assertEqual([row["id"] for row in response["rows"]], record["ids"])
            return response

    def test_saved_diagnostics_forwarded_in_real_batch_response(self):
        diagnostics = {"version": 1, "alphalab": {"cache_hit_count": 2, "new_computation_count": 0,
                                                "stages_seconds": {"cache_lookup": 0.1}}}
        response = self.response(diagnostics)
        self.assertEqual(response["diagnostics"], diagnostics)
        if shutil.which("node"):
            tags = frontend.RecoveryHelperTests.js(self, SETUP + "return badge(" + json.dumps(response) + ");")
            self.assertIn("2 results · 2 reused · 0 computed", tags)

    def test_old_receipt_has_no_invented_counts(self):
        response = self.response()
        self.assertIsNone(response.get("diagnostics"))
        if shutil.which("node"):
            tags = frontend.RecoveryHelperTests.js(self, SETUP + "return badge(" + json.dumps(response) + ");")
            self.assertEqual(tags[-1], "2 results")


if __name__ == "__main__":
    unittest.main()
