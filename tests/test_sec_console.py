"""SEC browser bounds, lightweight editor checks and worker forwarding."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from qlib_yahoo.webapp.service import ConsoleService, ServiceError


class SecConsoleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # Match the service's canonical paths, including macOS's /var alias.
        self.root = Path(self.tmp.name).resolve()
        self.catalog = self.root / "sec" / "catalogs" / "ready"
        self.catalog.mkdir(parents=True)
        (self.catalog / "manifest.json").write_text("{}", encoding="utf-8")
        self.manifest = {"status": "ready", "numeric_materialization_supported": True, "tables": {key: {"columns": ["tag", "version"]} for key in ("sub", "num", "tag", "pre", "txt")}}
        self.header = Mock(return_value=self.manifest)
        self.search = Mock(return_value={"items": [{"tag": "Revenue", "taxonomy": "us-gaap/2024"}], "total": 51})
        self.rows = Mock(return_value=[{"tag": "Revenue", "value": "12.00"}])
        self.variants = Mock(return_value=[{"unit": "USD", "qtrs": 1, "segments": "", "coreg": ""}])
        # Patch individual functions: restoring sys.modules wholesale can remove
        # NumPy modules imported by another test and cause a later reload.
        for name, replacement in (("read_catalog", self.header), ("search_tags", self.search),
                                  ("query_rows", self.rows), ("field_variants", self.variants)):
            mocked = patch(f"qlib_yahoo.sec_catalog.{name}", replacement)
            mocked.start()
            self.addCleanup(mocked.stop)
        self.service = ConsoleService(self.root, settings_path=self.root / "settings.json")
        self.sec = {"catalog": str(self.catalog), "bindings": {"sec_revenue": {"tag": "Revenue", "taxonomy": "us-gaap/2024", "unit": "USD", "qtrs": 1, "segments": "", "coreg": ""}}}
        self.dataset = self.root / "datasets" / "prices"
        self.dataset.mkdir(parents=True)
        (self.dataset / "manifest.json").write_text("{}", encoding="utf-8")
        self.price = patch.object(self.service, "_price_dataset", return_value=(self.dataset, {"fields": ["open", "high", "low", "close", "volume"]}))
        self.price.start()
        self.addCleanup(self.price.stop)

    def test_catalog_header_only_and_direct_ready_children(self):
        self.assertEqual(self.service.sec_catalogs()[0]["path"], str(self.catalog.resolve()))
        self.header.assert_called_with(self.catalog.resolve(), verify=False)
        self.rows.assert_not_called()
        self.search.assert_not_called()
        for path in (self.root / "datasets", self.catalog / "nested", self.root / "sec" / "catalogs"):
            with self.assertRaises(ServiceError):
                self.service._sec_catalog(path)
        self.header.side_effect = ValueError("incomplete")
        self.assertEqual(self.service.sec_catalogs(), [])

    def test_search_variants_and_all_supplied_tables_are_bounded(self):
        self.service.sec_search(str(self.catalog), "Revenue", 50)
        self.search.assert_called_once_with(self.catalog.resolve(), query="Revenue", limit=50, offset=50)
        self.service.sec_variants(str(self.catalog), "Revenue", "us-gaap/2024")
        self.service.sec_rows(str(self.catalog), "TXT", '{"tag":"Revenue"}', 100)
        self.rows.assert_called_once_with(self.catalog.resolve(), "txt", filters={"tag": "Revenue"}, limit=100, offset=100)
        for table, filters, offset in (("other", {}, 0), ("num", "[]", 0), ("num", "broken", 0), ("num", {}, -1)):
            with self.assertRaises(ServiceError):
                self.service.sec_rows(str(self.catalog), table, filters, offset)

    def test_qlib_editor_uses_aliases_without_fact_or_hash_queries(self):
        self.assertTrue(self.service.check_alpha("Mean($sec_revenue, 2)", sec=self.sec)["ok"])
        missing = self.service.check_alpha("Mean($sec_missing, 2)", sec=self.sec)
        self.assertFalse(missing["ok"])
        self.assertIn("sec_missing", missing["message"])
        self.rows.assert_not_called()
        self.variants.assert_not_called()
        self.search.assert_not_called()
        self.header.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "Lab optional numerical runtime")
    def test_lab_editor_uses_aliases_without_fact_or_hash_queries(self):
        result = self.service.alpha_check("rank(sec_revenue) + rank(close)", str(self.dataset), sec=self.sec)
        self.assertTrue(result["ok"], result)
        result = self.service.alpha_check("rank(sec_missing)", str(self.dataset), sec=self.sec)
        self.assertFalse(result["ok"])
        self.rows.assert_not_called()
        self.search.assert_not_called()
        self.variants.assert_not_called()
        self.header.assert_not_called()

    def test_backtest_and_batch_forward_common_config(self):
        params = {"dataset": str(self.dataset), "alpha": "$sec_revenue", "sec": self.sec}
        request, _, _ = self.service._backtest_request(params)
        self.assertEqual(request["sec"], {**self.sec, "mode": "legacy"})
        captured = {}
        def start(kind, label, request, work):
            captured.update(request)
            return request
        with patch.object(self.service, "_start", side_effect=start):
            self.service.start_backtest({**params, "alpha": "$sec_revenue\n-1 * $sec_revenue"})
        self.assertEqual(captured["sec"], {**self.sec, "mode": "legacy"})
        self.assertEqual(captured["alphas"], ["$sec_revenue", "-1 * $sec_revenue"])

    def test_notes_catalog_browsable_but_not_numeric(self):
        self.manifest["numeric_materialization_supported"] = False
        self.assertTrue(self.service.sec_catalogs())
        self.service.sec_rows(str(self.catalog), "txt")
        with self.assertRaisesRegex(ServiceError, "browsing only"):
            self.service._sec_config(self.sec)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "Lab optional numerical runtime")
    def test_alpha_service_forwards_common_config(self):
        with patch.object(self.service, "_start", side_effect=lambda kind, label, request, work: request):
            request = self.service.start_alpha({"dataset": str(self.dataset), "expressions": "rank(sec_revenue)", "sec": self.sec})
        self.assertEqual(request["sec"], {**self.sec, "mode": "legacy"})
        self.assertEqual(request["kind"], "alpha-batch")

    def test_binding_validation_is_real_and_does_not_scan_facts(self):
        for spec in ({**self.sec["bindings"]["sec_revenue"], "qtrs": "1"},
                     {**self.sec["bindings"]["sec_revenue"], "unit": ""}):
            with self.assertRaisesRegex(ServiceError, "Invalid SEC bindings"):
                self.service._sec_config({**self.sec, "bindings": {"sec_revenue": spec}})
        self.rows.assert_not_called()
        self.search.assert_not_called()

    def test_run_details_preserve_sec_provenance_even_read_only(self):
        run = self.root / "runs" / "saved"
        run.mkdir(parents=True)
        for version in (6, 7):
            manifest = {"schema_version": version, "status": "failed", "alpha": "$sec_revenue", "dataset": str(self.dataset),
                        "sec": {**self.sec, "source_identity": {"catalog_sha256": "a" * 64}}, "backtest": {}}
            (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            detail = self.service.run_detail(str(run))
            self.assertEqual(detail["sec"], manifest["sec"])
            self.assertEqual(detail["dataset_path"], str(self.dataset))
            self.assertEqual(detail["read_only"], version == 7)

    def test_alpha_result_preserves_saved_sec_provenance(self):
        library = Mock()
        library.get.return_value = {"expression": "sec_revenue", "dataset": {"dataset": str(self.dataset)}, "sec": self.sec}
        with patch.object(self.service, "_alpha_library", return_value=library):
            self.assertEqual(self.service.alpha_result("saved")["sec"], self.sec)

    def test_catalog_response_omits_bulk_inventory(self):
        self.manifest.update(sources=["large"], artifacts_sha256={"x": "y"})
        self.manifest["tables"]["num"]["files"] = ["large"]
        manifest = self.service.sec_catalogs()[0]["manifest"]
        self.assertNotIn("sources", manifest)
        self.assertNotIn("artifacts_sha256", manifest)
        self.assertNotIn("files", manifest["tables"]["num"])

    def test_settings_partial_save_preserves_prices_and_other_preferences(self):
        self.service.save_settings({"alpha": "$close", "dataset": "existing", "lab": '{"dataset":"lab-prices"}'})
        self.service.save_settings({"sec": json.dumps(self.sec)})
        self.assertEqual(self.service.settings()["alpha"], "$close")
        self.assertEqual(self.service.settings()["dataset"], "existing")
        self.assertEqual(json.loads(self.service.settings()["lab"])["dataset"], "lab-prices")

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "Lab optional numerical runtime")
    def test_alpha_worker_forwards_sec_only_when_present(self):
        from qlib_yahoo.jobs import _run_request
        request = {"kind": "alpha-batch", "dataset": str(self.dataset), "data_root": str(self.root), "expressions": ["sec_revenue"], "settings": {}, "sec": self.sec}
        with patch("qlib_yahoo.alphalab.batch.run_batch", return_value=[]) as run_batch, \
                patch("qlib_yahoo.alphalab.batch.save_batch", return_value=self.root / "batch.json"):
            _run_request(dict(request))
            self.assertEqual(run_batch.call_args.kwargs["sec"], self.sec)
            request.pop("sec")
            _run_request(dict(request))
            self.assertNotIn("sec", run_batch.call_args.kwargs)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "Qlib optional numerical runtime")
    def test_native_workers_forward_sec_single_and_batch(self):
        from qlib_yahoo.jobs import _run_request
        with patch("qlib_yahoo.backtest.run_backtest", return_value=self.root / "run") as single, \
                patch("qlib_yahoo.preparation.run_backtest_batch", return_value=self.root / "batch") as batch:
            _run_request({"kind": "backtest", "dataset": str(self.dataset), "alpha": "$sec_revenue", "sec": self.sec})
            self.assertEqual(single.call_args.kwargs["sec"], self.sec)
            _run_request({"kind": "backtest-batch", "dataset": str(self.dataset), "data_root": str(self.root), "alphas": ["$sec_revenue"], "sec": self.sec})
            self.assertEqual(batch.call_args.kwargs["sec"], self.sec)


@unittest.skipUnless(shutil.which("node"), "Node needed for frontend behavior")
class SecFrontendTests(unittest.TestCase):
    def test_sec_controls_keep_selector_height_and_dictionary_items_sized(self):
        css = (Path(__file__).resolve().parents[1] / "qlib_yahoo/webapp/static/app.css").read_text(encoding="utf-8")
        control_rule = css.split(".sec-dialog select, .sec-dialog input:not([type=checkbox]) {", 1)[1].split("}", 1)[0]
        item_rule = css.split(".sec-dictionary-item {", 1)[1].split("}", 1)[0]
        self.assertIn("height: 36px", control_rule)
        self.assertIn("#sec-dictionary-items { height: 300px; }", css)
        self.assertIn("flex: 0 0 auto", item_rule)

    def js(self, code, *, mode="legacy"):
        # Existing DOM harness executes actual packaged JS and checks HTML IDs.
        # Saved-binding tests preserve exact historical reuse; no new manual binding UI.
        # Dictionary cases opt in explicitly; broader discovery is covered
        # in test_sec_dictionary_frontend.py.
        from tests.test_qlib_recovery_console import HARNESS, ROOT
        # Extend only the tested nodes with event capture; keep the shared harness
        # and its actual-markup ID checks intact.
        setup = """
          $('#code-editor').style = {};
          const selectNode = document.querySelector;
          document.querySelector = selector => {
            const node = selectNode(selector);
            if (!node.handlers) {
              node.handlers = {};
              node.firstChild = {textContent: ''};
              node.addEventListener = (type, handler) => {node.handlers[type] = handler;};
              node.append = (...children) => {node.children = [...(node.children || []), ...children];};
            }
            return node;
          };
        """
        run = subprocess.run([shutil.which("node"), "-e", HARNESS],
                             input=json.dumps({"code": setup + "state.sec.mode = " + json.dumps(mode) + ";\n" + code}), text=True, encoding="utf-8",
                             capture_output=True, cwd=ROOT, timeout=15)
        self.assertEqual(run.returncode, 0, run.stderr)
        return json.loads(run.stdout)

    def test_typing_checks_do_not_trigger_catalog_searches(self):
        result = self.js("""
          state.sec = {...state.sec, catalog:'ready', bindings:{sec_revenue:{tag:'Revenue'}}};
          $('#alpha').value = '$sec_revenue'; $('#lab-text').value = 'sec_revenue';
          api = async (name, data) => {calls.push({name,data}); return {ok:true,count:1,lines:[{line:1,ok:true}]};};
          await checkAlpha(); await checkLab();
          return calls;
        """)
        self.assertEqual([call["name"] for call in result], ["check-alpha", "alpha-check"])
        self.assertTrue(all("sec" in call["data"] for call in result))

    def test_raw_row_requests_ignore_stale_responses(self):
        result = self.js("""
          state.sec.catalog = 'ready';
          const pending = [];
          api = (name, data) => new Promise(resolve => pending.push({name, data, resolve}));
          $('#sec-table').value = 'sub'; $('#sec-filters').value = '{}';
          const oldRows = inspectSecRows(true);
          $('#sec-table').value = 'num'; const newRows = inspectSecRows(true);
          pending[1].resolve({rows:[{tag:'New'}],metadata:{}}); await newRows;
          pending[0].resolve({rows:[{tag:'Old'}],metadata:{}}); await oldRows;
          return {rows:$('#sec-rows').textContent, requests:pending.map(p=>p.data)};
        """)
        self.assertEqual([item["table"] for item in result["requests"]], ["sub", "num"])
        self.assertIn('"tag": "New"', result["rows"])
        self.assertNotIn("Old", result["rows"])

    def test_removed_manual_selection_markup_and_apis_stay_absent(self):
        root = Path(__file__).resolve().parents[1] / "qlib_yahoo/webapp/static"
        html = (root / "index.html").read_text(encoding="utf-8")
        js = (root / "app.js").read_text(encoding="utf-8")
        for removed_id in ("sec-query", "sec-tags", "sec-taxonomy", "sec-context", "sec-alias", "sec-add"):
            self.assertNotIn(f'id="{removed_id}"', html)
        for removed_api in ("searchSecTags", "chooseSecTag", "loadSecVariants", "addSecBinding", "renderSecChoices"):
            self.assertNotIn(removed_api, js)
        self.assertIn('id="sec-dictionary-items"', html)
        self.assertIn('id="sec-table"', html)


    def test_restore_original_inputs_and_clear_for_price_only(self):
        result = self.js("""
          checkAlpha = checkLab = async () => true;
          api = async (name,data) => name === 'sec-bindings' ? data.sec : [{path:'original',manifest:{catalog_sha256:'original-hash'}}];
          state.sec.catalog = 'current'; state.sec.bindings = {sec_current:{tag:'Current'}};
          const saved = {sec:{catalog:'original',bindings:{sec_old:{tag:'Old'}},source_identity:{catalog_sha256:'original-hash'}}};
          const ok = await restoreSavedSec(saved);
          const restored = secRequest().sec;
          await restoreSavedSec({alpha:'$close'});
          return {ok,restored,cleared:secRequest(),catalog:state.sec.catalog};
        """)
        self.assertTrue(result["ok"])
        self.assertEqual(result["restored"]["catalog"], "original")
        self.assertEqual(list(result["restored"]["bindings"]), ["sec_old"])
        self.assertEqual(result["cleared"], {})
        self.assertEqual(result["catalog"], "")

    def test_missing_or_replaced_saved_catalog_stops_validation(self):
        for missing in (True, False):
            code = """
              checkLab = async () => true;
              api = async (name,data) => {
                if (name === 'sec-bindings') { MISSING return data.sec; }
                return [{path:'original',manifest:{catalog_sha256:'different'}}];
              };
              const ok = await restoreSavedSec({sec:{catalog:'original',bindings:{sec_old:{tag:'Old'}},source_identity:{catalog_sha256:'original-hash'}}});
              $('#alpha').value = '$sec_old';
              const valid = await checkAlpha();
              return {ok,valid,error:state.sec.restoreError,catalog:state.sec.catalog};
            """.replace("MISSING", "throw new Error('Catalog missing');" if missing else "")
            result = self.js(code)
            self.assertFalse(result["ok"])
            self.assertFalse(result["valid"])
            self.assertEqual(result["catalog"], "original")
            self.assertIn("cannot be reused", result["error"])

    def test_run_window_reuse_restores_sec_before_copying_dates(self):
        result = self.js("""
          checkAlpha = checkLab = async () => true;
          renderPeriod = renderUniversePanel = () => {};
          setMode = () => {}; setAlpha = text => {$('#alpha').value=text;};
          api = async (name,data) => data.sec;
          state.runDetail = {path:'saved',status:'completed',read_only:false,settings:{},period:{start:'2020-01-01',end:'2020-12-31'},mode:'long-only',alpha:'$sec_old',universe:'all',sec:{catalog:'original',bindings:{sec_old:{tag:'Old'}}}};
          await useWindowForBacktest({dates:['2021-01-01','2021-12-31']});
          return {sec:secRequest().sec,start:$('#start').value,end:$('#end').value,alpha:$('#alpha').value};
        """)
        self.assertEqual(result["sec"]["catalog"], "original")
        self.assertEqual(result["start"], "2021-01-01")
        self.assertEqual(result["alpha"], "$sec_old")

    def test_lab_saved_settings_restore_embedded_sec_identity(self):
        result = self.js("""
          checkAlpha = checkLab = async () => true; renderLabPeriod = () => {};
          api = async (name,data) => data.sec;
          state.lab.detail = {expression:'rank(sec_old)',settings:{universe:'TOP1000',delay:0,decay:2,cost_bps:5},dataset:{sec:{catalog:'original',bindings:{sec_old:{tag:'Old'}}}}};
          const ok = await useLabSettings();
          return {ok,sec:secRequest().sec,text:$('#lab-text').value,cost:$('#lab-cost').value,universe:$('#lab-universe').value};
        """)
        self.assertTrue(result["ok"])
        self.assertEqual(result["sec"]["catalog"], "original")
        self.assertEqual(result["text"], "rank(sec_old)")
        self.assertEqual(result["cost"], "5")
        self.assertEqual(result["universe"], "TOP1000")

    def test_single_catalog_fixed_raw_only_catalog_and_binding_navigation_safeguard(self):
        result = self.js("""
          state.sec.catalog = 'notes'; bindSecBrowser();
          api = async name => name === 'sec-dictionary'
            ? {status:'unsupported',items:[],message:'Notes support raw inspection only'}
            : [{path:'notes',name:'Notes source',manifest:{numeric_materialization_supported:false,tables:{tag:{},dim:{}}}}];
          await loadSecCatalogs();
          const notes = {catalog:state.sec.catalog,hidden:$('#sec-catalog').hidden,text:$('#sec-catalog-fixed').textContent,tables:$('#sec-table').children,disabled:$('#sec-dictionary-build').hidden,message:$('#sec-message').textContent};
          state.sec.bindings = {sec_Existing:{tag:'Keep'}};
          $('#sec-catalog').value = 'other'; await $('#sec-catalog').handlers.change();
          return {notes,catalog:state.sec.catalog,bindings:state.sec.bindings,message:$('#sec-message').textContent};
        """)
        self.assertTrue(result["notes"]["hidden"])
        self.assertEqual(result["notes"]["text"], "Notes source")
        self.assertEqual([o["value"] for o in result["notes"]["tables"]], ["tag", "dim"])
        self.assertTrue(result["notes"]["disabled"])
        self.assertIn("raw inspection", result["notes"]["message"])
        self.assertEqual(result["catalog"], "notes")
        self.assertIn("sec_Existing", result["bindings"])
        self.assertIn("preserved", result["message"])

    def test_restore_race_preserves_newer_saved_inputs(self):
        result = self.js("""
          checkAlpha = checkLab = async () => true;
          const pending=[]; api = (name,data) => new Promise(resolve=>pending.push({data,resolve}));
          const old=restoreSavedSec({sec:{catalog:'old',bindings:{sec_Old:{tag:'Old'}}}});
          const newer=restoreSavedSec({sec:{catalog:'new',bindings:{sec_New:{tag:'New'}}}});
          pending[1].resolve(pending[1].data.sec); const ok=await newer;
          pending[0].resolve(pending[0].data.sec); const stale=await old;
          return {ok,stale,sec:secRequest().sec};
        """)
        self.assertTrue(result["ok"])
        self.assertFalse(result["stale"])
        self.assertEqual(result["sec"]["catalog"], "new")
        self.assertEqual(list(result["sec"]["bindings"]), ["sec_New"])

    def test_unknown_dictionary_name_editor_error_explains_search_without_worker(self):
        result = self.js("""
          updateRunLabel = () => {};
          $('#alpha').value='$sec_inventory'; $('#lab-text').value='sec_inventory';
          api = async (name,data) => {
            calls.push({name,data});
            return name==='check-alpha' ? {ok:false,message:'unknown field sec_inventory'} : {lines:[{line:1,ok:false,message:'unknown field sec_inventory'}]};
          };
          await checkAlpha(); await checkLab();
          return {calls,qlib:$('#alpha-status').textContent,lab:$('#lab-status').textContent};
        """, mode="dictionary")
        self.assertEqual([c["name"] for c in result["calls"]], ["check-alpha", "alpha-check"])
        self.assertIn("shared SEC dictionary", result["qlib"])
        self.assertNotIn("Add input", result["qlib"])
        self.assertIn("cannot be created", result["lab"])
        self.assertIn("case-sensitive", result["lab"])

    def test_backend_editor_guidance_is_not_duplicated(self):
        result = self.js("""
          const message='Unknown SEC input sec_inventory. Open SEC fields and Add input with this field name.';
          return {message,formatted:secEditorError(message)};
        """)
        self.assertEqual(result["formatted"], result["message"])

    def test_dictionary_resolver_guidance_is_not_duplicated(self):
        message = "Unknown SEC field: sec_nonsense. Search the SEC dictionary for an official name."
        result = self.js("return secEditorError(" + json.dumps(message) + ");")
        self.assertEqual(result, message)

if __name__ == "__main__":
    unittest.main()
