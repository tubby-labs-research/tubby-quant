"""Execute the packaged SEC dictionary frontend; no browser, catalog or worker."""

import json
import shutil
import subprocess
import unittest
from pathlib import Path

try:
    from .test_qlib_recovery_console import HARNESS, ROOT
except ImportError:
    from test_qlib_recovery_console import HARNESS, ROOT


SETUP = r"""
  const selectNode = document.querySelector;
  document.querySelector = selector => {
    const node = selectNode(selector);
    if (!node.handlers) {
      node.handlers = {}; node.style = {}; node.firstChild = {textContent:''};
      node.addEventListener = (type, handler) => {node.handlers[type] = handler;};
      node.append = (...children) => {node.children = [...(node.children || []), ...children];};
      node.close = () => {node.closed = true;}; node.showModal = () => {};
      node.focus = () => {node.focused = true;};
      node.setRangeText = (text, start, end) => {
        node.value = node.value.slice(0, start) + text + node.value.slice(end);
        node.selectionStart = node.selectionEnd = start + text.length;
      };
    }
    return node;
  };
  state.sec.catalog = 'catalog'; state.sec.numeric = true;
  state.sec.selectionEpoch = 1;
  const spec = {tag:'InventoryNet',taxonomy:'us-gaap/',unit:'USD',qtrs:0,segments:'',coreg:''};
  const item = {name:'sec_inventory',label:'Inventory',kind:'standard',spec};
  bindSecBrowser();
"""


@unittest.skipUnless(shutil.which("node"), "Node is needed for frontend behavior")
class SecDictionaryFrontendTests(unittest.TestCase):
    def js(self, code):
        result = subprocess.run(
            [shutil.which("node"), "-e", HARNESS],
            input=json.dumps({"code": SETUP + code}), text=True, encoding="utf-8",
            capture_output=True, cwd=ROOT, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_catalog_only_request_is_explicit_dictionary_for_both_engines(self):
        result = self.js("""
          return {lab:labValues().sec,qlib:formValues().sec,
            inferred:[secMode({catalog:'c'}),secMode({bindings:{Sec_saved:spec}})]};
        """)
        self.assertEqual(result["lab"], {"catalog": "catalog", "mode": "dictionary", "bindings": {}})
        self.assertEqual(result["qlib"], result["lab"])
        self.assertEqual(result["inferred"], ["dictionary", "legacy"])

    def test_dictionary_search_metadata_pagination_and_custom_flag(self):
        result = self.js("""
          state.sec.dictionaryView = 'recipes';
          $('#sec-dictionary-query').value = 'inventory definition';
          $('#sec-dictionary-custom').checked = true; state.sec.dictionaryOffset = 50;
          api = async (name,data) => {calls.push({name,data}); return {
            status:'ready',version:'v1',catalog_sha256:'hash',path:'sidecar',field_count:120,
            total:120,offset:50,items:[item]};};
          await searchSecDictionary();
          const row=$('#sec-dictionary-items').children[0];
          return {calls,page:$('#sec-dictionary-page').textContent,
            prev:$('#sec-dictionary-prev').disabled,next:$('#sec-dictionary-next').disabled,
            name:row.children[0].text,
            definition:row.children[3].children.find(node=>node?.tag==='pre').text,
            details:row.children[3].tag,open:!!row.children[3].open,build:$('#sec-dictionary-build').hidden};
        """)
        self.assertEqual(result["calls"][0], {"name": "sec-dictionary", "data": {
            "catalog": "catalog", "query": "inventory definition", "offset": 50, "include_custom": True, "view": "recipes"}})
        self.assertEqual(result["page"], "51–51 of 120 definition matches")
        self.assertFalse(result["prev"])
        self.assertFalse(result["next"])
        self.assertEqual(result["name"], "sec_inventory")
        self.assertIn("InventoryNet", result["definition"])
        self.assertEqual(json.loads(result["definition"]), {
            "tag": "InventoryNet", "taxonomy": "us-gaap/", "unit": "USD", "qtrs": 0,
            "segments": "", "coreg": ""})
        self.assertEqual(result["details"], "details")
        self.assertFalse(result["open"])
        self.assertTrue(result["build"])

    def test_typing_invalidates_search_without_requests_or_builds(self):
        result = self.js("""
          let finish; api=()=>new Promise(resolve=>finish=resolve);
          const pending=searchSecDictionary(true);
          $('#sec-dictionary-query').value='changed'; $('#sec-dictionary-query').handlers.input();
          finish({status:'ready',items:[item],total:1}); await pending;
          return {calls,items:$('#sec-dictionary-items').children};
        """)
        self.assertEqual(result, {"calls": [], "items": []})

    def test_dictionary_badges_distinguish_family_exact_edition_and_company_definition(self):
        result = self.js("""
          state.sec.dictionaryView = 'recipes';
          const items=[item,{...item,name:'sec_exact',spec:{...spec,taxonomy:'us-gaap/2012'}},
            {...item,name:'sec_custom',kind:'custom',spec:{...spec,taxonomy:'0000123456-24-000007'}}];
          api=async()=>({status:'ready',items,total:3,field_count:3,version:'v1'});
          await searchSecDictionary(true);
          return $('#sec-dictionary-items').children.map(row=>({badge:row.children[2].text,
            spec:JSON.parse(row.children[3].children.find(node=>node?.tag==='pre').text),
            context:row.children[3].children[1].text,insert:row.children[4].text}));
        """)
        self.assertEqual([row['badge'] for row in result], [
            'Standard taxonomy · Instant (point in time) · Unit: USD · No segment or co-registrant (blank context) · All editions · us-gaap/',
            'Standard taxonomy · Instant (point in time) · Unit: USD · No segment or co-registrant (blank context) · Special recipe: exact edition · us-gaap/2012',
            'Company-defined or unconfirmed · Instant (point in time) · Unit: USD · No segment or co-registrant (blank context) · Exact definition (see details)'])
        self.assertEqual([row['spec']['taxonomy'] for row in result], [
            'us-gaap/', 'us-gaap/2012', '0000123456-24-000007'])
        self.assertTrue(all(row['insert'] == 'Insert field' for row in result))
        for row in result:
            self.assertEqual(row['spec'], {
                'tag': 'InventoryNet', 'taxonomy': row['spec']['taxonomy'], 'unit': 'USD',
                'qtrs': 0, 'segments': '', 'coreg': ''})
            self.assertEqual(row['context'], 'Segments: none (blank) · Co-registrant: none (blank)')

    def test_missing_requires_explicit_background_build_and_unsupported_stays_raw(self):
        result = self.js("""
          api=async()=>({status:'missing',items:[],message:'Prepare once'});
          await searchSecDictionary(true);
          const missing={hidden:$('#sec-dictionary-build').hidden,calls:[...calls]};
          api=async(name,data)=>{calls.push({name,data});return {kind:'build-dictionary',id:'job'};};
          startPolling=job=>calls.push({name:'poll',job});
          await buildSecDictionary(); await buildSecDictionary();
          const built=[...calls];
          api=async()=>({status:'unsupported',items:[],message:'Notes are raw only'});
          await searchSecDictionary(true);
          return {missing,built,unsupported:$('#sec-dictionary-build').hidden};
        """)
        self.assertEqual(result["missing"], {"hidden": False, "calls": []})
        self.assertEqual([c["name"] for c in result["built"]], ["sec-dictionary-build", "poll"])
        self.assertTrue(result["unsupported"])

    def test_insert_replaces_selection_preserves_case_and_does_not_bind(self):
        result = self.js("""
          const outcomes=[];
          for(const id of ['#lab-text','#alpha']) {
            state.sec.editor=id; const editor=$(id); editor.value='rank(old / close)';
            editor.selectionStart=5;editor.selectionEnd=8;
            insertSecField({...item,name:'sec_ExactCase'});
            outcomes.push({text:editor.value,caret:editor.selectionStart,focus:editor.focused});
          }
          return {outcomes,bindings:state.sec.bindings,calls};
        """)
        self.assertEqual(result["outcomes"][0]["text"], "rank(sec_ExactCase / close)")
        self.assertEqual(result["outcomes"][1]["text"], "rank($sec_ExactCase / close)")
        self.assertTrue(all(row["focus"] for row in result["outcomes"]))
        self.assertEqual(result["outcomes"][0]["caret"], 18)
        self.assertEqual(result["bindings"], {})
        self.assertEqual(result["calls"], [])

    def test_checks_display_server_inputs_and_disable_unknown_fields(self):
        result = self.js("""
          $('#lab-text').value='rank(sec_inventory / sec_assets)';
          api=async()=>({lines:[{ok:true,line:1}],sec_inputs:[item],registry:{version:'v1'}});
          const valid=await checkLab(); const resolved=state.sec.resolved.lab;
          const enabled=!$('#lab-run').disabled;
          api=async()=>({lines:[{ok:false,line:1,message:'Unknown sec_missing'}]});
          $('#lab-text').value='rank(sec_missing)'; onLabInput(); const unknown=await checkLab();
          return {valid,resolved,enabled,unknown,disabled:$('#lab-run').disabled,
            error:$('#lab-status').textContent,bindings:state.sec.bindings};
        """)
        self.assertTrue(result["valid"])
        self.assertTrue(result["enabled"])
        self.assertEqual(result["resolved"]["items"][0]["name"], "sec_inventory")
        self.assertEqual(result["resolved"]["registry"], {"version": "v1"})
        self.assertFalse(result["unknown"])
        self.assertTrue(result["disabled"])
        self.assertIn("cannot be created", result["error"])
        self.assertNotIn("Add input", result["error"])
        self.assertEqual(result["bindings"], {})

    def test_stale_success_and_error_cannot_enable_run_or_render_other_catalog(self):
        for engine in ("lab", "qlib"):
            with self.subTest(engine=engine):
                result = self.js("""
                  const engine='ENGINE', editor=engine==='lab'?'#lab-text':'#alpha';
                  $(editor).value='sec_inventory'; const pending=[];
                  api=()=>new Promise((resolve,reject)=>pending.push({resolve,reject}));
                  const check=engine==='lab'?checkLab:checkAlpha;
                  const old=check(); state.sec.catalog='other'; invalidateSecChecks();
                  const fresh=check();
                  pending[1].resolve({ok:false,lines:[{ok:false,line:1,message:'Unknown sec_inventory'}],message:'Unknown sec_inventory'});
                  await fresh;
                  pending[0].resolve({ok:true,count:1,lines:[{ok:true,line:1}],sec_inputs:[item]});await old;
                  const lateError=check();state.sec.catalog='catalog';invalidateSecChecks();
                  pending[2].reject(new Error('wrong old catalog'));await lateError;
                  return {valid:engine==='lab'?state.labValid:state.alphaValid,
                    disabled:$(engine==='lab'?'#lab-run':'#run-2').disabled,
                    resolved:state.sec.resolved,error:$(engine==='lab'?'#lab-status':'#alpha-status').textContent};
                """.replace("ENGINE", engine))
                self.assertFalse(result["valid"])
                self.assertTrue(result["disabled"])
                self.assertEqual(result["resolved"], {})
                self.assertNotIn("wrong old catalog", result["error"])

    def test_legacy_restore_and_explicit_switch_preserve_then_clear_exact_mapping(self):
        result = self.js("""
          const bindings={sec_MyInventory:{...spec,taxonomy:'us-gaap/2024',unit:'EUR'}};
          const saved={sec:{catalog:'catalog',bindings,registry:{version:'saved-pin'}}};
          const historical=JSON.stringify(saved);
          $('#alpha').value='$sec_MyInventory';$('#lab-text').value='rank(sec_MyInventory)';
          window.confirm=()=>{throw new Error('Native confirmation must not be used');};
          api=async(name,data)=>{calls.push({name,data});if(name==='sec-dictionary')return {status:'missing',items:[]};if(name==='sec-bindings')return data.sec;
            if(name==='alpha-check')return {lines:[]};return {ok:true,count:1};};
          await restoreSavedSec(saved);
          const before={config:secRequest().sec,banner:$('#sec-legacy').hidden};
          const requestCount=calls.length;
          await $('#sec-switch-dictionary').handlers.click();
          const shown={config:secRequest().sec,hidden:$('#sec-switch-confirmation').hidden,
            expanded:$('#sec-switch-dictionary').ariaExpanded,focused:$('#sec-switch-confirmation').focused};
          await $('#sec-switch-cancel').handlers.click();
          const cancelled=secRequest().sec;
          const dismissed={hidden:$('#sec-switch-confirmation').hidden,expanded:$('#sec-switch-dictionary').ariaExpanded,
            focused:$('#sec-switch-dictionary').focused,requests:calls.length-requestCount};
          await $('#sec-switch-confirm').handlers.click();
          const afterCancelledConfirm=secRequest().sec;
          await $('#sec-switch-dictionary').handlers.click();
          await $('#sec-switch-confirm').handlers.click();
          return {before,shown,cancelled,dismissed,afterCancelledConfirm,after:secRequest().sec,
            hidden:$('#sec-switch-confirmation').hidden,expanded:$('#sec-switch-dictionary').ariaExpanded,
            searchFocused:$('#sec-dictionary-query').focused,historicalUnchanged:historical===JSON.stringify(saved),
            actions:calls.slice(requestCount).map(call=>call.name),formulas:[$('#alpha').value,$('#lab-text').value]};
        """)
        self.assertEqual(result["before"]["config"]["mode"], "legacy")
        self.assertFalse(result["before"]["banner"])
        self.assertEqual(result["shown"]["config"], result["before"]["config"])
        self.assertFalse(result["shown"]["hidden"])
        self.assertEqual(result["shown"]["expanded"], "true")
        self.assertTrue(result["shown"]["focused"])
        self.assertEqual(result["cancelled"], result["before"]["config"])
        self.assertEqual(result["afterCancelledConfirm"], result["cancelled"])
        self.assertEqual(result["dismissed"], {"hidden": True, "expanded": "false", "focused": True, "requests": 0})
        self.assertEqual(result["before"]["config"]["bindings"]["sec_MyInventory"]["unit"], "EUR")
        self.assertEqual(result["after"], {"catalog": "catalog", "mode": "dictionary", "bindings": {}})
        self.assertTrue(result["hidden"])
        self.assertEqual(result["expanded"], "false")
        self.assertTrue(result["searchFocused"])
        self.assertTrue(result["historicalUnchanged"])
        self.assertEqual(result["actions"], ["check-alpha", "alpha-check", "sec-dictionary"])
        self.assertEqual(result["formulas"], ["$sec_MyInventory", "rank(sec_MyInventory)"])

    def test_inline_switch_is_invalidated_by_saved_restore_even_with_identical_inputs(self):
        result = self.js("""
          state.sec.mode='legacy'; state.sec.bindings={sec_saved:spec}; state.sec.registry={version:'pin'};
          const saved={sec:JSON.parse(JSON.stringify(secConfig()))};
          checkAlpha=checkLab=async()=>true;
          await $('#sec-switch-dictionary').handlers.click();
          let finish;api=(name,data)=>{calls.push({name,data});return new Promise(resolve=>finish=()=>resolve(data.sec));};
          const pending=restoreSavedSec(saved);
          const hiddenWhileRestoring=$('#sec-switch-confirmation').hidden;
          await $('#sec-switch-dictionary').handlers.click();
          const blocked={hidden:$('#sec-switch-confirmation').hidden,message:$('#sec-message').textContent};
          await $('#sec-switch-confirm').handlers.click();
          const during=secConfig(); finish();await pending;
          await $('#sec-switch-confirm').handlers.click();
          return {hiddenWhileRestoring,blocked,during,after:secConfig(),saved:saved.sec,calls};
        """)
        self.assertTrue(result["hiddenWhileRestoring"])
        self.assertTrue(result["blocked"]["hidden"])
        self.assertIn("Restore valid legacy inputs", result["blocked"]["message"])
        self.assertEqual(result["during"], result["saved"])
        self.assertEqual(result["after"], result["saved"])
        self.assertEqual([call["name"] for call in result["calls"]], ["sec-bindings"])

    def test_inline_switch_rechecks_current_catalog_mode_bindings_and_pins(self):
        changes = (
            "state.sec.catalog='other'",
            "state.sec.bindings={sec_replaced:{...spec,unit:'EUR'}}",
            "state.sec.registry={version:'new-pin'}",
            "state.sec.mode='dictionary'",
            "state.sec.restoreError='Saved catalog unavailable'",
        )
        for change in changes:
            with self.subTest(change=change):
                result = self.js("""
                  state.sec.mode='legacy';state.sec.bindings={sec_saved:spec};state.sec.registry={version:'pin'};
                  await $('#sec-switch-dictionary').handlers.click();
                  CHANGE;
                  const current=JSON.stringify(secConfig());
                  await $('#sec-switch-confirm').handlers.click();
                  return {unchanged:JSON.stringify(secConfig())===current,hidden:$('#sec-switch-confirmation').hidden,
                    pending:state.sec.switchConfirmation,error:$('#sec-message').textContent,calls};
                """.replace("CHANGE", change))
                self.assertTrue(result["unchanged"])
                self.assertTrue(result["hidden"])
                self.assertIsNone(result["pending"])
                self.assertIn("SEC inputs changed", result["error"])
                self.assertEqual(result["calls"], [])

    def test_inline_switch_hides_on_catalog_and_binding_resets_or_dialog_close(self):
        result = self.js("""
          checkAlpha=checkLab=async()=>true;
          const results=[];
          for(const reset of [()=>{state.sec.catalog='other';resetSecSelection();},
            ()=>{state.sec.bindings={sec_new:spec};secChanged();},
            ()=>$('#sec-close').handlers.click(),()=>$('#sec-dialog').handlers.close()]) {
            state.sec.mode='legacy';state.sec.catalog='catalog';state.sec.bindings={sec_saved:spec};
            await $('#sec-switch-dictionary').handlers.click();reset();
            const current=JSON.stringify(secConfig());
            await $('#sec-switch-confirm').handlers.click();
            results.push({hidden:$('#sec-switch-confirmation').hidden,pending:state.sec.switchConfirmation,
              unchanged:current===JSON.stringify(secConfig())});
          }
          return {results,calls};
        """)
        self.assertEqual(result["results"], [{"hidden": True, "pending": None, "unchanged": True}] * 4)
        self.assertEqual(result["calls"], [])

    def test_inline_switch_markup_has_explicit_controls_and_accessible_warning(self):
        html = (ROOT / "qlib_yahoo/webapp/static/index.html").read_text(encoding="utf-8")
        self.assertIn('aria-controls="sec-switch-confirmation" aria-expanded="false"', html)
        self.assertIn('id="sec-switch-confirmation" role="group" aria-labelledby="sec-switch-title" '
                      'aria-describedby="sec-switch-warning" tabindex="-1" hidden', html)
        self.assertIn('id="sec-switch-confirm" class="btn">Confirm switch</button>', html)
        self.assertIn('id="sec-switch-cancel" class="btn">Cancel</button>', html)
        self.assertIn("saved aliases, definitions and registry pins", html)
        self.assertIn("Historical runs and their original definitions remain unchanged", html)

    def test_restored_dictionary_pins_survive_failed_validation_and_preferences_save(self):
        result = self.js("""
          const saved={catalog:'catalog',mode:'dictionary',bindings:{sec_inventory:spec},registry:{version:'v1',catalog_sha256:'hash'}};
          let finish;api=(name,data)=>{
            calls.push({name,data});if(name==='sec-bindings')return new Promise(resolve=>finish=resolve);
            return Promise.resolve({});};
          const pending=restoreSavedSec({sec:saved});
          const during=secConfig();finish(saved);await pending;
          const restored=secConfig();
          api=async()=>{throw new Error('Unknown field');};
          $('#alpha').value='$sec_inventory';await checkAlpha();
          state.restoring=false;persist();timers.pop()();
          return {during,restored,after:secConfig(),persisted:state.settings.sec};
        """)
        for key in ("during", "restored", "after"):
            self.assertEqual(result[key]["mode"], "dictionary")
            self.assertEqual(result[key]["registry"]["version"], "v1")
            self.assertIn("sec_inventory", result[key]["bindings"])
        self.assertEqual(result["persisted"]["bindings"], result["restored"]["bindings"])

    def test_edit_away_and_back_invalidates_both_editor_responses(self):
        result = self.js("""
          const results=[];
          for (const engine of ['lab','qlib']) {
            const id=engine==='lab'?'#lab-text':'#alpha',check=engine==='lab'?checkLab:checkAlpha;
            const changed=engine==='lab'?onLabInput:onAlphaInput;
            $(id).value='sec_inventory';let finish;api=()=>new Promise(resolve=>finish=resolve);
            const pending=check();$(id).value='sec_other';changed();$(id).value='sec_inventory';changed();
            finish({ok:true,count:1,lines:[{ok:true,line:1}],sec_inputs:[item]});
            results.push({accepted:await pending,disabled:$(engine==='lab'?'#lab-run':'#run-2').disabled});
          }
          return results;
        """)
        self.assertEqual(result, [{"accepted": False, "disabled": True}] * 2)

    def test_insert_rejects_stale_catalog_and_legacy_cannot_create_dictionary_name(self):
        result = self.js("""
          const errors=[];
          try {insertSecField(item,'old-catalog',1);}catch(e){errors.push(e.message);}
          state.sec.mode='legacy';
          try {insertSecField(item);}catch(e){errors.push(e.message);}
          return {errors,calls};
        """)
        self.assertEqual(len(result["errors"]), 2)
        self.assertEqual(result["calls"], [])

    def test_markup_keeps_advanced_definitions_raw_inspection_and_readonly_legacy_inputs(self):
        html = (ROOT / "qlib_yahoo/webapp/static/index.html").read_text(encoding="utf-8")
        for removed_id in ("sec-query", "sec-tags", "sec-taxonomy", "sec-context", "sec-alias", "sec-add"):
            self.assertNotIn(f'id="{removed_id}"', html)
        for control in ("sec-dictionary-view", "sec-dictionary-items", "sec-bindings",
                        "sec-clear", "sec-table", "sec-filters", "sec-rows"):
            self.assertIn(f'id="{control}"', html)
        self.assertIn("Definitions are read only", html)
        self.assertIn("Earlier saved inputs (read only)", html)
        self.assertIn("Text and notes catalogs support raw inspection only", html)
        js = (ROOT / "qlib_yahoo/webapp/static/app.js").read_text(encoding="utf-8")
        for removed_api in ("searchSecTags", "chooseSecTag", "loadSecVariants", "addSecBinding", "renderSecChoices"):
            self.assertNotIn(removed_api, js)

    def test_legacy_bindings_are_readonly_clearable_and_history_is_preserved(self):
        result = self.js("""
          const saved={catalog:'catalog',mode:'legacy',bindings:{sec_MyInventory:{...spec,unit:'EUR'}},registry:{version:'saved-pin'}};
          const historical=JSON.stringify(saved);state.sec.mode='legacy';
          state.sec.catalog=saved.catalog;state.sec.bindings=JSON.parse(JSON.stringify(saved.bindings));
          state.sec.registry=saved.registry;state.sec.resolved={};renderSecBindings();
          const before={legacyHidden:$('#sec-legacy').hidden,clearHidden:$('#sec-clear').hidden,
            bindings:state.sec.bindings,request:secRequest().sec,
            row:$('#sec-bindings').children[0],text:$('#sec-bindings').children[0].children.map(n=>n.text||n.textContent||'').join(' ')};
          await $('#sec-clear').handlers.click();
          return {before,after:secRequest(),mode:state.sec.mode,bindings:state.sec.bindings,
            registry:state.sec.registry,legacyHidden:$('#sec-legacy').hidden,
            clearHidden:$('#sec-clear').hidden,historicalUnchanged:historical===JSON.stringify(saved)};
        """)
        self.assertFalse(result["before"]["legacyHidden"])
        self.assertFalse(result["before"]["clearHidden"])
        self.assertEqual(result["before"]["bindings"]["sec_MyInventory"]["unit"], "EUR")
        self.assertEqual(result["before"]["request"]["bindings"], result["before"]["bindings"])
        row_tags = [node["tag"] for node in result["before"]["row"]["children"]]
        self.assertEqual(row_tags, ["code", "small"])
        self.assertNotIn("input", row_tags)
        self.assertNotIn("Remove", result["before"]["text"])
        self.assertEqual(result["after"]["sec"], {"catalog": "catalog", "mode": "dictionary", "bindings": {}})
        self.assertEqual(result["mode"], "dictionary")
        self.assertEqual(result["bindings"], {})
        self.assertIsNone(result["registry"])
        self.assertTrue(result["legacyHidden"])
        self.assertTrue(result["clearHidden"])
        self.assertTrue(result["historicalUnchanged"])


if __name__ == "__main__":
    unittest.main()
