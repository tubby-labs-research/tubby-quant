"""Readable SEC recipe results, using the packaged JavaScript and existing harness."""

import shutil
import subprocess
import unittest

import test_sec_dictionary_frontend as frontend


@unittest.skipUnless(shutil.which("node"), "Node is needed for frontend behavior")
class ReadableSecFrontendTests(unittest.TestCase):
    js = frontend.SecDictionaryFrontendTests.js

    def test_full_accounting_tag_is_primary_even_with_short_source_label(self):
        result = self.js("""
          const row=renderSecDictionaryItem({...item,label:'Basic',
            spec:{...spec,tag:'EarningsPerShareBasic'}},'catalog',1);
          return row.children[1].children.filter(Boolean).map(n=>n.text);
        """)
        self.assertEqual(result, ["Earnings Per Share Basic", "SEC tag: EarningsPerShareBasic", "Source label: Basic"])

    def test_long_tag_acronyms_and_generic_formula_name_stay_readable(self):
        result = self.js("""
          const tag='USGAAPOperatingIncomeLossBeforeIncomeTaxesAndNoncontrollingInterest';
          const row=renderSecDictionaryItem({...item,label:'',
            name:'sec_usgaap_operating_income_loss_before_income_taxes_and_noncontrolling_interest',
            spec:{...spec,tag}},'catalog',1);
          return {title:row.children[1].children[0].text,name:row.children[0].text,
            wrap:row.children[0].style.overflowWrap};
        """)
        self.assertEqual(result["title"], "USGAAP Operating Income Loss Before Income Taxes And Noncontrolling Interest")
        self.assertEqual(result["name"], "sec_usgaap_operating_income_loss_before_income_taxes_and_noncontrolling_interest")
        self.assertEqual(result["wrap"], "anywhere")

    def test_summary_explains_period_unit_blank_context_and_scope(self):
        result = self.js("""
          return [0,1,2,3,4].map(qtrs=>secRecipeSummary({...item,spec:{...spec,qtrs}}));
        """)
        for summary, period in zip(result, ["Instant (point in time)", "1 quarter", "2 quarters", "3 quarters", "4 quarters (annual)"]):
            self.assertIn(period, summary)
            self.assertIn("Unit: USD", summary)
            self.assertIn("No segment or co-registrant (blank context)", summary)
            self.assertIn("Standard taxonomy", summary)
            self.assertIn("All editions · us-gaap/", summary)
            self.assertNotIn("year-to-date", summary)
            self.assertNotIn("rolling", summary)

    def test_special_recipes_are_flagged_and_exact_context_remains_in_details(self):
        result = self.js("""
          const row=renderSecDictionaryItem({...item,spec:{...spec,taxonomy:'us-gaap/2024',
            segments:'Division A',coreg:'Subsidiary B'}},'catalog',1);
          return {summary:row.children[2].text,details:row.children[3].children.filter(Boolean)};
        """)
        for flag in ["Special recipe: segment", "Special recipe: co-registrant", "Special recipe: exact edition · us-gaap/2024"]:
            self.assertIn(flag, result["summary"])
        self.assertIn("Division A", result["details"][1]["text"])
        self.assertIn("Subsidiary B", result["details"][1]["text"])

    def test_json_and_storage_name_are_only_inside_closed_details(self):
        result = self.js("""
          const row=renderSecDictionaryItem({...item,storage_name:'sec_old_canonical'},'catalog',1);
          const visible=(node)=>!node || node.tag==='details' ? '' :
            typeof node==='string' ? node : (node.text||'')+(node.children||[]).map(visible).join(' ');
          const details=row.children[3];
          return {visible:visible(row),tag:details.tag,open:!!details.open,
            storage:details.children[2].children[1].text,
            raw:JSON.parse(details.children[3].text)};
        """)
        self.assertEqual(result["tag"], "details")
        self.assertFalse(result["open"])
        self.assertNotIn("sec_old_canonical", result["visible"])
        self.assertNotIn('"taxonomy"', result["visible"])
        self.assertEqual(result["storage"], "sec_old_canonical")
        self.assertEqual(result["raw"]["tag"], "InventoryNet")

    def test_custom_unknown_conflict_and_absent_kind_are_uncertain(self):
        result = self.js("""
          return ['custom','unknown','conflict',undefined].map(kind=>
            secRecipeSummary({...item,kind,spec:{...spec,taxonomy:'us-gaap/2024'}}));
        """)
        for summary in result:
            self.assertIn("Company-defined or unconfirmed", summary)
            self.assertNotIn("Standard taxonomy", summary)
            self.assertNotIn("Special recipe: exact edition", summary)

    def test_null_or_missing_context_is_not_described_as_blank(self):
        result = self.js("""
          return secRecipeSummary({name:'sec_test',spec:{tag:'Test',segments:null}});
        """)
        for text in ["Period unavailable", "Unit: unavailable", "Segment context unavailable", "Co-registrant context unavailable"]:
            self.assertIn(text, result)
        self.assertNotIn("blank context", result)

    def test_injection_is_text_and_insert_uses_actual_name_in_both_editors(self):
        result = self.js("""
          const hostile='<img src=x onerror=alert(1)>';
          const recipe={...item,label:hostile,storage_name:hostile,
            spec:{...spec,tag:hostile,segments:hostile,coreg:hostile}};
          const row=renderSecDictionaryItem(recipe,'catalog',1);
          const nodes=[];const walk=n=>{if(n&&typeof n==='object'){nodes.push(n);(n.children||[]).forEach(walk);}};walk(row);
          const texts=nodes.map(n=>n.text||'');
          onLabInput=onAlphaInput=()=>{};persist=()=>{};
          const inserted=[];
          for(const id of ['#lab-text','#alpha']){
            state.sec.editor=id;$(id).value='';$(id).selectionStart=$(id).selectionEnd=0;
            await row.children[4].onclick();inserted.push($(id).value);
          }
          return {unsafe:nodes.some(n=>Object.hasOwn(n,'html')||n.tag==='img'),texts,inserted};
        """)
        self.assertFalse(result["unsafe"])
        self.assertTrue(any("<img" in text for text in result["texts"]))
        self.assertEqual(result["inserted"], ["sec_inventory", "$sec_inventory"])

    def test_redundant_source_label_is_omitted(self):
        result = self.js("""
          return renderSecDictionaryItem({...item,label:'Inventory Net'},'catalog',1)
            .children[1].children.filter(Boolean).map(n=>n.text);
        """)
        self.assertEqual(result, ["Inventory Net", "SEC tag: InventoryNet"])

    def test_real_dom_helper_never_interprets_recipe_text_as_html(self):
        core = (frontend.ROOT / "qlib_yahoo/webapp/static/core.js").read_text(encoding="utf-8")
        helper = core[core.index("function el("):core.index("/* Replace a node's children")]
        result = self.js(helper.replace("function el(", "el = function(", 1) + """
          document.createElement=tag=>({tag,style:{},children:[],
            set innerHTML(value){throw new Error('HTML injection');},
            setAttribute(key,value){this[key]=value;},
            addEventListener(){},append(child){this.children.push(child);}});
          const hostile='<script>alert(1)</script>';
          const row=renderSecDictionaryItem({...item,label:'Short '+hostile,storage_name:hostile,
            spec:{...spec,tag:hostile,segments:hostile}},'catalog',1);
          return {title:row.children[1].children[0].textContent,
            label:row.children[1].children[2].textContent,
            storage:row.children[3].children[2].children[1].textContent};
        """)
        self.assertEqual(result["title"], "<script>alert(1)</script>")
        self.assertEqual(result["label"], "Source label: Short <script>alert(1)</script>")
        self.assertEqual(result["storage"], "<script>alert(1)</script>")

    def test_generic_and_original_canonical_names_insert_without_renaming(self):
        result = self.js("""
          onLabInput=()=>{};persist=()=>{};state.sec.editor='#lab-text';
          const inserted=[];
          for(const name of ['sec_earnings_per_share_basic','sec_original_canonical_2024']){
            $('#lab-text').value='';$('#lab-text').selectionStart=$('#lab-text').selectionEnd=0;
            const row=renderSecDictionaryItem({...item,name,label:'Basic',storage_name:'sec_storage',
              spec:{...spec,tag:'EarningsPerShareBasic'}},'catalog',1);
            await row.children[4].onclick();inserted.push($('#lab-text').value);
          }
          return inserted;
        """)
        self.assertEqual(result, ["sec_earnings_per_share_basic", "sec_original_canonical_2024"])

    def test_insert_buttons_retain_restore_and_legacy_guards(self):
        result = self.js("""
          const results=[];
          for(const guard of ['legacy','restoreError','restoreLoading']){
            state.sec.mode=guard==='legacy'?'legacy':'dictionary';
            state.sec.restoreError=guard==='restoreError'?'Unavailable':null;
            state.sec.restoreLoading=guard==='restoreLoading';
            const row=renderSecDictionaryItem(item,'catalog',1);
            await row.children[4].onclick();
            results.push({disabled:row.children[4].disabled,error:$('#sec-message').textContent});
          }
          return results;
        """)
        for row in result:
            self.assertTrue(row["disabled"])
            self.assertIn("SEC selection changed", row["error"])

    def test_help_describes_search_inputs_and_partial_results(self):
        html = (frontend.ROOT / "qlib_yahoo/webapp/static/index.html").read_text(encoding="utf-8")
        self.assertIn('for="sec-dictionary-query">Accounting words, SEC tag or formula name', html)
        self.assertIn('placeholder="inventory, InventoryNet or sec_inventory"', html)
        self.assertIn("partial results within the search budget", html)
        self.assertIn("Earlier canonical names remain usable", html)
        self.assertIn("Formula names use lowercase snake_case", html)
        self.assertIn("sec_earnings_per_share_basic", html)
        self.assertIn("Include company-defined or unconfirmed fields", html)

    def test_packaged_javascript_syntax(self):
        result = subprocess.run([shutil.which("node"), "--check", "qlib_yahoo/webapp/static/app.js"],
                                cwd=frontend.ROOT, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
