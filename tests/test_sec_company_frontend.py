"""Company concepts and advanced recipes in the packaged frontend; synthetic Node fixtures."""

import shutil
import unittest

try:
    from . import test_sec_dictionary_frontend as frontend
except ImportError:
    import test_sec_dictionary_frontend as frontend


SETUP = r"""
  const usd={...item,name:'sec_inventory_usd',storage_name:'sec_sealed_usd'};
  const eur={...item,name:'sec_inventory_eur_4q',spec:{...spec,unit:'EUR',qtrs:4}};
  const concept={tag:'InventoryNet',taxonomy:'us-gaap/',label:'Inventory',variants:[usd,eur]};
  const ready=(items=[concept],extra={})=>({status:'ready',version:'v1',field_count:29600000,
    items,total:items.length,has_more:false,search_scope:'company_starter',...extra});
  onLabInput=onAlphaInput=()=>{};persist=()=>{};
  const flatten=n=>!n ? [] : typeof n==='string' ? [n] :
    [n.textContent||n.text||'',...(n.children||[]).flatMap(flatten)];
  const find=(row,tag)=>row.children.find(n=>n?.tag===tag);
  const button=row=>find(row,'button');
"""


@unittest.skipUnless(shutil.which("node"), "Node is needed for frontend behavior")
class CompanySecFrontendTests(unittest.TestCase):
    def js(self, code):
        return frontend.SecDictionaryFrontendTests.js(self, SETUP + code)

    def test_default_explicit_company_request_hides_custom_and_counts_starter_concepts(self):
        result = self.js("""
          $('#sec-dictionary-custom').checked=true;
          api=async(name,data)=>{calls.push({name,data});return ready();};
          await searchSecDictionary(true);
          return {request:calls[0],custom:$('#sec-dictionary-custom-control').hidden,
            view:$('#sec-dictionary-view').value,status:$('#sec-dictionary-status').textContent,
            page:$('#sec-dictionary-page').textContent,rows:$('#sec-dictionary-items').children.length};
        """)
        self.assertEqual(result['request']['data'], {'catalog': 'catalog', 'query': '',
            'offset': 0, 'include_custom': False, 'view': 'company'})
        self.assertTrue(result['custom'])
        self.assertEqual(result['view'], 'company')
        self.assertEqual(result['rows'], 1)
        self.assertIn('29600000 stored definitions', result['status'])
        self.assertIn('not the exhaustive', result['status'])
        self.assertEqual(result['page'], '1–1 of 1 starter concepts')

    def test_one_card_full_title_and_exact_observed_variant_options(self):
        result = self.js("""
          api=async()=>ready([{...concept,tag:'EarningsPerShareBasic',label:'Basic'}]);
          await searchSecDictionary(true);const row=$('#sec-dictionary-items').children[0];
          return {title:row.children[0].text,options:find(row,'select').children,
            disabled:button(row).disabled,text:flatten(row)};
        """)
        self.assertEqual(result['title'], 'Earnings Per Share Basic')
        self.assertEqual([o['value'] for o in result['options']], ['', '0', '1'])
        self.assertEqual([o['text'] for o in result['options'][1:]], [
            'USD · Instant (point in time) · sec_inventory_usd',
            'EUR · 4 quarters (annual) · sec_inventory_eur_4q'])
        self.assertTrue(result['disabled'])
        self.assertIn('SEC tag: EarningsPerShareBasic · Taxonomy family: us-gaap/', result['text'])

    def test_choice_required_then_only_selected_name_inserts_in_both_editors(self):
        result = self.js("""
          const results=[];
          for(const id of ['#lab-text','#alpha']) {
            state.sec.editor=id;$(id).value='';$(id).selectionStart=$(id).selectionEnd=0;
            api=async()=>ready();await searchSecDictionary(true);
            const row=$('#sec-dictionary-items').children[0];
            await button(row).onclick();const before=$(id).value;
            find(row,'select').onchange({target:{value:'1'}});
            const enabled=!button(row).disabled;await button(row).onclick();
            results.push({before,enabled,text:$(id).value});
          }
          return {results,bindings:state.sec.bindings};
        """)
        self.assertEqual(result['results'], [
            {'before': '', 'enabled': True, 'text': 'sec_inventory_eur_4q'},
            {'before': '', 'enabled': True, 'text': '$sec_inventory_eur_4q'}])
        self.assertEqual(result['bindings'], {})

    def test_single_choice_fixed_text_and_storage_only_in_closed_details(self):
        result = self.js("""
          api=async()=>ready([{...concept,variants:[usd]}]);await searchSecDictionary(true);
          const row=$('#sec-dictionary-items').children[0],details=find(row,'details');
          state.sec.editor='#lab-text';$('#lab-text').value='';
          $('#lab-text').selectionStart=$('#lab-text').selectionEnd=0;
          await button(row).onclick();
          return {select:!!find(row,'select'),fixed:find(row,'p').text,
            disabled:button(row).disabled,inserted:$('#lab-text').value,
            open:!!details.open,details:flatten(details),
            visible:row.children.filter(n=>n?.tag!=='details').flatMap(flatten)};
        """)
        self.assertFalse(result['select'])
        self.assertFalse(result['disabled'])
        self.assertEqual(result['inserted'], 'sec_inventory_usd')
        self.assertIn('USD', result['fixed'])
        self.assertFalse(result['open'])
        self.assertIn('sec_sealed_usd', result['details'])
        self.assertNotIn('sec_sealed_usd', result['visible'])

    def test_no_variant_and_invalid_name_cannot_insert(self):
        result = self.js("""
          state.sec.editor='#lab-text';$('#lab-text').value='';
          const results=[];
          for(const variants of [[],[{...usd,name:'invented name'}]]) {
            api=async()=>ready([{...concept,variants}]);await searchSecDictionary(true);
            const row=$('#sec-dictionary-items').children[0];await button(row).onclick();
            results.push({disabled:button(row).disabled,text:$('#lab-text').value,error:$('#sec-message').textContent});
          }
          return results;
        """)
        self.assertTrue(result[0]['disabled'])
        self.assertTrue(all(r['text'] == '' and 'No official numeric' in r['error'] for r in result))

    def test_partial_concepts_and_variant_lists_never_claim_global_counts(self):
        result = self.js("""
          $('#sec-dictionary-query').value='inventory';
          api=async()=>ready([concept],{truncated:true,total:null,has_more:true,search_scope:'company_concepts'});
          await searchSecDictionary(true);
          return {page:$('#sec-dictionary-page').textContent,status:$('#sec-dictionary-status').textContent,
            text:flatten($('#sec-dictionary-items').children[0]),next:$('#sec-dictionary-next').disabled};
        """)
        self.assertIn('total unknown concept matches', result['page'])
        self.assertIn('not global counts', result['page'])
        self.assertIn('Partial group / observed variant list; not a global count.', result['text'])
        self.assertNotIn('starter', result['page'])
        self.assertFalse(result['next'])

    def test_view_change_resets_paging_and_restores_advanced_custom_cards(self):
        result = self.js("""
          state.sec.dictionaryOffset=100;$('#sec-dictionary-custom').checked=true;
          api=async(name,data)=>{calls.push({name,data});return ready([item]);};
          $('#sec-dictionary-view').value='recipes';await $('#sec-dictionary-view').handlers.change();
          const advanced={request:calls[0].data,hidden:$('#sec-dictionary-custom-control').hidden,
            name:$('#sec-dictionary-items').children[0].children[0].text};
          api=async(name,data)=>{calls.push({name,data});return ready();};
          $('#sec-dictionary-view').value='company';await $('#sec-dictionary-view').handlers.change();
          return {advanced,company:calls[1].data,hidden:$('#sec-dictionary-custom-control').hidden};
        """)
        self.assertEqual(result['advanced']['request']['view'], 'recipes')
        self.assertEqual(result['advanced']['request']['offset'], 0)
        self.assertTrue(result['advanced']['request']['include_custom'])
        self.assertFalse(result['advanced']['hidden'])
        self.assertEqual(result['advanced']['name'], 'sec_inventory')
        self.assertEqual(result['company']['view'], 'company')
        self.assertFalse(result['company']['include_custom'])
        self.assertTrue(result['hidden'])

    def test_stale_advanced_success_and_error_after_view_change_are_ignored(self):
        for reject in (False, True):
            with self.subTest(reject=reject):
                result = self.js("""
                  state.sec.dictionaryView='recipes';let finish,fail;
                  api=()=>new Promise((resolve,reject)=>{finish=resolve;fail=reject;});
                  const pending=searchSecDictionary(true);
                  api=async()=>ready();$('#sec-dictionary-view').value='company';
                  await $('#sec-dictionary-view').handlers.change();
                  REPLY;await pending;
                  return {title:$('#sec-dictionary-items').children[0].children[0].text,
                    status:$('#sec-dictionary-status').textContent};
                """.replace('REPLY', "fail(new Error('stale advanced'))" if reject else 'finish(ready([item]))'))
                self.assertEqual(result['title'], 'Inventory Net')
                self.assertNotIn('stale advanced', result['status'])

    def test_old_advanced_button_rejected_after_company_and_back(self):
        result = self.js("""
          state.sec.dictionaryView='recipes';api=async()=>ready([item]);await searchSecDictionary(true);
          const old=$('#sec-dictionary-items').children[0];
          state.sec.editor='#lab-text';$('#lab-text').value='';
          api=async()=>ready();$('#sec-dictionary-view').value='company';
          await $('#sec-dictionary-view').handlers.change();await button(old).onclick();
          const first=$('#sec-message').textContent;
          api=async()=>ready([item]);$('#sec-dictionary-view').value='recipes';
          await $('#sec-dictionary-view').handlers.change();await button(old).onclick();
          return {first,second:$('#sec-message').textContent,text:$('#lab-text').value};
        """)
        self.assertEqual(result['text'], '')
        self.assertIn('SEC selection changed', result['first'])
        self.assertIn('SEC selection changed', result['second'])

    def test_company_buttons_preserve_catalog_restore_and_legacy_guards(self):
        result = self.js("""
          const results=[];state.sec.editor='#lab-text';$('#lab-text').value='';
          for(const guard of ['legacy','restoreError','restoreLoading','catalog']) {
            state.sec.mode='dictionary';state.sec.restoreError=null;state.sec.restoreLoading=false;
            state.sec.catalog='catalog';api=async()=>ready([{...concept,variants:[usd]}]);
            await searchSecDictionary(true);const row=$('#sec-dictionary-items').children[0];
            if(guard==='legacy')state.sec.mode='legacy';
            if(guard==='restoreError')state.sec.restoreError='Missing';
            if(guard==='restoreLoading')state.sec.restoreLoading=true;
            if(guard==='catalog')state.sec.catalog='other';
            await button(row).onclick();results.push($('#sec-message').textContent);
          }
          return {results,text:$('#lab-text').value};
        """)
        self.assertEqual(result['text'], '')
        self.assertTrue(all('SEC selection changed' in text for text in result['results']))

    def test_text_only_rendering_and_taxonomy_groups_remain_distinct(self):
        result = self.js("""
          const hostile='<img src=x onerror=alert(1)>';
          api=async()=>ready([concept,{...concept,taxonomy:'ifrs-full/',label:hostile,
            variants:[{...usd,name:'sec_verified_descriptive_alias',storage_name:hostile}]}]);
          await searchSecDictionary(true);const rows=$('#sec-dictionary-items').children;
          const walk=n=>n&&typeof n==='object'?[n,...(n.children||[]).flatMap(walk)]:[];
          return {rows:rows.length,text:rows.flatMap(flatten),
            unsafe:rows.flatMap(walk).some(n=>n.tag==='img'||Object.hasOwn(n,'html'))};
        """)
        self.assertEqual(result['rows'], 2)
        self.assertFalse(result['unsafe'])
        self.assertIn('Source label: <img src=x onerror=alert(1)>', result['text'])
        self.assertIn('SEC tag: InventoryNet · Taxonomy family: ifrs-full/', result['text'])

    def test_descriptive_advanced_alias_inserts_unchanged_not_storage_name(self):
        result = self.js("""
          const entry={...item,name:'sec_inventory_usd_instant_us_gaap_2024',storage_name:'sec_old_sealed_name',
            spec:{...spec,taxonomy:'us-gaap/2024'}};
          state.sec.dictionaryView='recipes';api=async()=>ready([entry]);await searchSecDictionary(true);
          const row=$('#sec-dictionary-items').children[0];
          state.sec.editor='#lab-text';$('#lab-text').value='';
          $('#lab-text').selectionStart=$('#lab-text').selectionEnd=0;
          await button(row).onclick();
          return {text:$('#lab-text').value,details:flatten(find(row,'details'))};
        """)
        self.assertEqual(result['text'], 'sec_inventory_usd_instant_us_gaap_2024')
        self.assertIn('sec_old_sealed_name', result['details'])

    def test_old_company_card_rejected_after_query_or_paging_invalidation(self):
        for action in ('typing', 'paging'):
            with self.subTest(action=action):
                result = self.js("""
                  api=async()=>ready([{...concept,variants:[usd]}]);await searchSecDictionary(true);
                  const old=$('#sec-dictionary-items').children[0];
                  state.sec.editor='#lab-text';$('#lab-text').value='';
                  INVALIDATE;
                  await button(old).onclick();
                  return {text:$('#lab-text').value,error:$('#sec-message').textContent};
                """.replace('INVALIDATE', "$('#sec-dictionary-query').value='new';$('#sec-dictionary-query').handlers.input()"
                            if action == 'typing' else "await $('#sec-dictionary-next').handlers.click()"))
                self.assertEqual(result['text'], '')
                self.assertIn('SEC selection changed', result['error'])

    def test_compact_help_raw_browser_and_consistent_version(self):
        html = (frontend.ROOT / 'qlib_yahoo/webapp/static/index.html').read_text(encoding='utf-8')
        self.assertIn('<option value="company" selected>Company fields</option>', html)
        self.assertIn('<option value="recipes">Advanced definitions</option>', html)
        self.assertIn('<details><summary>Naming and search help</summary>', html)
        self.assertIn('<details><summary>Advanced: inspect raw source tables and notes</summary>', html)
        self.assertIn('id="sec-dictionary-items"', html)
        for text in ('no automatic FX conversion or computed TTM', 'not an exhaustive universe',
                     'not a promise of company coverage or data quality', 'including special saved inputs'):
            self.assertIn(text, html)
        version = self.js('return UI_VERSION;')
        self.assertEqual(version, 'workbench-cleanup-2026-10-14')
        self.assertEqual(html.count(version), 3)

    def test_alias_error_company_single_and_multiple_choices_hide_old_names_and_block_insert(self):
        for reason in ('Ambiguous descriptive alias', 'Unsupported context', 'Search budget exhausted', ''):
            with self.subTest(reason=reason):
                result = self.js("""
                  const unavailable={...usd,name:'sec_old_hash_abcdef',alias_error:REASON};
                  const results=[];
                  state.sec.editor='#lab-text';$('#lab-text').value='';
                  for(const variants of [[unavailable],[eur,unavailable]]) {
                    api=async()=>ready([{...concept,variants}]);await searchSecDictionary(true);
                    const row=$('#sec-dictionary-items').children[0],select=find(row,'select');
                    if(select)select.onchange({target:{value:'1'}});
                    await button(row).onclick();
                    results.push({disabled:button(row).disabled,
                      visible:row.children.filter(n=>n?.tag!=='details').flatMap(flatten).join(' '),
                      details:flatten(find(row,'details')).join(' '),
                      error:$('#sec-message').textContent,text:$('#lab-text').value});
                  }
                  return results;
                """.replace('REASON', repr(reason)))
                for row in result:
                    self.assertTrue(row['disabled'])
                    self.assertEqual(row['text'], '')
                    self.assertNotIn('sec_old_hash_abcdef', row['visible'])
                    self.assertIn('Descriptive name unavailable', row['visible'])
                    self.assertIn('sec_old_hash_abcdef', row['details'])
                    self.assertIn('Descriptive name unavailable', row['error'])

    def test_alias_error_advanced_card_hides_original_and_blocks_insert(self):
        result = self.js("""
          state.sec.dictionaryView='recipes';state.sec.editor='#lab-text';$('#lab-text').value='';
          const unavailable={...item,name:'sec_old_hash_abcdef',storage_name:'sec_sealed_hash',
            alias_error:'Ambiguous descriptive name for sec_old_hash_abcdef'};
          api=async()=>ready([unavailable]);await searchSecDictionary(true);
          const row=$('#sec-dictionary-items').children[0];await button(row).onclick();
          return {disabled:button(row).disabled,text:$('#lab-text').value,
            visible:row.children.filter(n=>n?.tag!=='details').flatMap(flatten).join(' '),
            details:flatten(find(row,'details')).join(' '),error:$('#sec-message').textContent};
        """)
        self.assertTrue(result['disabled'])
        self.assertEqual(result['text'], '')
        self.assertIn('Descriptive name unavailable', result['visible'])
        self.assertNotIn('sec_old_hash_abcdef', result['visible'])
        self.assertNotIn('sec_sealed_hash', result['visible'])
        self.assertIn('sec_old_hash_abcdef', result['details'])
        self.assertIn('sec_sealed_hash', result['details'])
        self.assertIn('Descriptive name unavailable', result['error'])

    def test_switching_from_unavailable_variant_to_verified_variant_enables_exact_insert(self):
        result = self.js("""
          const unavailable={...usd,name:'sec_old_hash',alias_error:'Ambiguous'};
          api=async()=>ready([{...concept,variants:[unavailable,eur]}]);await searchSecDictionary(true);
          const row=$('#sec-dictionary-items').children[0],select=find(row,'select');
          select.onchange({target:{value:'0'}});const blocked=button(row).disabled;
          select.onchange({target:{value:'1'}});const enabled=!button(row).disabled;
          state.sec.editor='#lab-text';$('#lab-text').value='';
          $('#lab-text').selectionStart=$('#lab-text').selectionEnd=0;
          await button(row).onclick();select.onchange({target:{value:'0'}});
          return {blocked,enabled,blockedAgain:button(row).disabled,text:$('#lab-text').value};
        """)
        self.assertEqual(result, {'blocked': True, 'enabled': True, 'blockedAgain': True,
                                  'text': 'sec_inventory_eur_4q'})

    def test_alias_error_reason_is_short_text_and_saved_formula_checks_keep_old_names(self):
        result = self.js("""
          const old={...item,name:'sec_original_hash',alias_error:'<script>alert(1)</script> '+ 'x'.repeat(500)};
          const display=secFormulaDisplay(old);
          // Compatibility checks receive the exact original name; browsing guards
          // must not reinterpret a server-validated existing expression.
          $('#lab-text').value='rank(sec_original_hash)';
          api=async()=>({lines:[{ok:true,line:1}],sec_inputs:[old],registry:{version:'v1'}});
          const ok=await checkLab();
          const resolved=state.sec.resolved.lab.items[0].name;
          return {display,ok,resolved,text:$('#lab-text').value};
        """)
        self.assertLessEqual(len(result['display']), len('Descriptive name unavailable: ') + 180)
        self.assertIn('<script>', result['display'])
        self.assertTrue(result['ok'])
        self.assertEqual(result['resolved'], 'sec_original_hash')
        self.assertEqual(result['text'], 'rank(sec_original_hash)')

    def test_pending_advanced_verifies_exact_metadata_once_before_insert(self):
        result = self.js("""
          state.sec.dictionaryView='recipes';state.sec.editor='#lab-text';$('#lab-text').value='';
          $('#lab-text').selectionStart=$('#lab-text').selectionEnd=0;
          const pendingItem={...item,name:'sec_candidate',storage_name:'sec_sealed',alias_pending:true};
          const row=renderSecDictionaryItem(pendingItem,'catalog',1);
          const before={text:flatten(row).join(' '),label:button(row).text};
          let finish;api=(name,data,method)=>{calls.push({name,data,method});return new Promise(resolve=>finish=resolve);};
          const first=button(row).onclick();await button(row).onclick();
          const during={disabled:button(row).disabled,text:$('#lab-text').value};
          finish({...item,name:'sec_verified'});await first;
          return {before,during,calls,text:$('#lab-text').value,disabled:button(row).disabled};
        """)
        self.assertEqual(result['before']['label'], 'Verify & insert')
        self.assertIn('Verified when inserted', result['before']['text'])
        self.assertEqual(result['during'], {'disabled': True, 'text': ''})
        self.assertEqual(result['calls'], [{'name': 'sec-name', 'data': {'catalog': 'catalog',
            'spec': {'tag': 'InventoryNet', 'taxonomy': 'us-gaap/', 'unit': 'USD', 'qtrs': 0,
                     'segments': '', 'coreg': ''}}, 'method': 'POST'}])
        self.assertEqual(result['text'], 'sec_verified')
        self.assertFalse(result['disabled'])

    def test_pending_response_ignores_stale_view_request_epoch_and_catalog(self):
        for change in ("state.sec.dictionaryView='company'", "state.sec.dictionaryLoad=2",
                       "state.sec.selectionEpoch++", "state.sec.catalog='other'"):
            with self.subTest(change=change):
                result = self.js("""
                  state.sec.dictionaryView='recipes';state.sec.editor='#lab-text';$('#lab-text').value='';
                  const row=renderSecDictionaryItem({...item,alias_pending:true},'catalog',1);
                  let finish;api=()=>new Promise(resolve=>finish=resolve);
                  const pending=button(row).onclick();CHANGE;
                  secMessage('Current view');finish({...item,name:'sec_wrong'});await pending;
                  return {text:$('#lab-text').value,message:$('#sec-message').textContent};
                """.replace('CHANGE', change))
                self.assertEqual(result, {'text': '', 'message': 'Current view'})

    def test_pending_ambiguity_hides_hash_and_disables_retry_without_insertion(self):
        result = self.js("""
          state.sec.dictionaryView='recipes';state.sec.editor='#lab-text';$('#lab-text').value='';
          const row=renderSecDictionaryItem({...item,name:'sec_candidate',alias_pending:true},'catalog',1);
          api=async()=>({...item,name:'sec_hash_original',alias_error:'Ambiguous descriptive alias'});
          await button(row).onclick();
          return {visible:row.children.filter(n=>n?.tag!=='details').flatMap(flatten).join(' '),
            details:flatten(find(row,'details')).join(' '),disabled:button(row).disabled,
            text:$('#lab-text').value,message:$('#sec-message').textContent};
        """)
        self.assertIn('Descriptive name unavailable', result['visible'])
        self.assertNotIn('sec_hash_original', result['visible'])
        self.assertIn('sec_hash_original', result['details'])
        self.assertTrue(result['disabled'])
        self.assertEqual(result['text'], '')
        self.assertIn('Ambiguous', result['message'])

    def test_pending_spec_mismatch_and_still_pending_response_cannot_insert(self):
        for reply in ("{...item,spec:{...spec,unit:'EUR'}}", "{...item,alias_pending:true}"):
            with self.subTest(reply=reply):
                result = self.js("""
                  state.sec.dictionaryView='recipes';state.sec.editor='#lab-text';$('#lab-text').value='';
                  const row=renderSecDictionaryItem({...item,alias_pending:true},'catalog',1);
                  api=async()=>(REPLY);await button(row).onclick();
                  return {text:$('#lab-text').value,error:$('#sec-message').textContent};
                """.replace('REPLY', reply))
                self.assertEqual(result['text'], '')
                self.assertTrue('differs' in result['error'] or 'still pending' in result['error'])

    def test_pending_insert_cannot_bypass_verification_and_verified_cards_need_no_request(self):
        result = self.js("""
          let error;try{insertSecField({...item,alias_pending:true});}catch(e){error=e.message;}
          state.sec.dictionaryView='recipes';state.sec.editor='#lab-text';$('#lab-text').value='';
          $('#lab-text').selectionStart=$('#lab-text').selectionEnd=0;
          const row=renderSecDictionaryItem(item,'catalog',1);await button(row).onclick();
          return {error,calls,text:$('#lab-text').value};
        """)
        self.assertIn('verified before insertion', result['error'])
        self.assertEqual(result['calls'], [])
        self.assertEqual(result['text'], 'sec_inventory')

    def test_view_invalidation_clears_catalog_loading_message_before_stale_reply(self):
        for failure in (False, True):
            with self.subTest(failure=failure):
                result = self.js("""
                  let finish,fail;api=()=>new Promise((resolve,reject)=>{finish=resolve;fail=reject;});
                  const catalogs=loadSecCatalogs();const busy=$('#sec-message').textContent;
                  api=async()=>ready([item]);$('#sec-dictionary-view').value='recipes';
                  await $('#sec-dictionary-view').handlers.change();
                  const after=$('#sec-message').textContent;
                  REPLY;await catalogs;
                  return {busy,after,final:$('#sec-message').textContent,
                    dictionary:$('#sec-dictionary-items').children[0].children[0].text};
                """.replace('REPLY', "fail(new Error('obsolete catalog failure'))" if failure else 'finish([])'))
                self.assertEqual(result['busy'], 'Loading source catalogs…')
                self.assertNotEqual(result['after'], result['busy'])
                self.assertEqual(result['final'], result['after'])
                self.assertEqual(result['dictionary'], 'sec_inventory')

    def test_boot_restores_open_sec_catalog_without_eager_closed_dialog_fetch(self):
        for opened in (False, True):
            with self.subTest(opened=opened):
                result = self.js("""
                  bind=initializeResearch=restoreSettings=setTransform=renderEditor=setResultTab=setView=consoleLine=()=>{};
                  restorePriceSelections=refreshRuns=checkAlpha=checkLab=async()=>{};
                  state.sec.catalog='';$('#sec-dialog').open=OPENED;
                  const pending=[];let settings;
                  api=(name,data)=>{
                    calls.push({name,data});
                    if(name==='session')return Promise.resolve({token:'fixture'});
                    if(name==='overview')return Promise.resolve({data_root:'fixture',signed:{available:false,detail:''},examples:[]});
                    if(name==='settings')return new Promise(resolve=>{settings=resolve;});
                    if(name==='sec-catalogs')return new Promise(resolve=>pending.push(resolve));
                    if(name==='sec-dictionary')return Promise.resolve(ready());
                    return Promise.resolve([]);
                  };
                  let bootFinished=false;
                  const booting=boot().then(()=>{bootFinished=true;});
                  for(let i=0;i<10&&!settings;i++)await Promise.resolve();
                  if(!settings)throw new Error('Boot did not request saved settings');
                  const early=OPENED?loadSecCatalogs():null;
                  settings({sec:{catalog:'catalog',mode:'dictionary',bindings:{}}});
                  for(let i=0;i<20;i++)await Promise.resolve();
                  const loads=pending.length;
                  if(OPENED&&loads!==2)throw new Error('Open modal was not refreshed after saved catalog restoration');
                  if(!OPENED&&loads)throw new Error('Closed modal eagerly fetched catalogs');
                  if(!bootFinished)throw new Error('Catalog refresh blocked the rest of workbench startup');
                  if(OPENED)pending[1]([{path:'catalog',name:'Current source',manifest:{numeric_materialization_supported:true,tables:{num:{},dim:{}}}}]);
                  await booting;
                  for(let i=0;i<20;i++)await Promise.resolve();
                  // The response started before settings must not overwrite the fresh view.
                  if(early){pending[0]([{path:'catalog',name:'Stale source',manifest:{tables:{txt:{}}}}]);await early;}
                  return {loads,catalog:state.sec.catalog,selector:$('#sec-catalog').value,
                    fixed:$('#sec-catalog-fixed').textContent,tables:($('#sec-table').children||[]).map(n=>n.value),
                    cards:($('#sec-dictionary-items').children||[]).length,
                    dictionaryCalls:calls.filter(c=>c.name==='sec-dictionary').length};
                """.replace('OPENED', 'true' if opened else 'false'))
                self.assertEqual(result['catalog'], 'catalog')
                self.assertEqual(result['loads'], 2 if opened else 0)
                self.assertEqual(result['dictionaryCalls'], 1 if opened else 0)
                if opened:
                    self.assertEqual(result['selector'], 'catalog')
                    self.assertEqual(result['fixed'], 'Current source')
                    self.assertEqual(result['tables'], ['num', 'dim'])
                    self.assertEqual(result['cards'], 1)

    def test_stale_catalog_completion_cannot_clear_new_load_or_current_warning(self):
        result = self.js("""
          const pending=[];api=()=>new Promise(resolve=>pending.push(resolve));
          const old=loadSecCatalogs();resetSecSelection();state.sec.catalog='';const fresh=loadSecCatalogs();
          pending[0]([]);await old;const newer=$('#sec-message').textContent;
          secMessage('Current warning',true);pending[1]([]);await fresh;
          // Successful current loads intentionally replace the loading status.
          const current=$('#sec-message').textContent;
          const late=loadSecCatalogs();secMessage('Preserve this warning',true);
          resetSecSelection();pending[2]([]);await late;
          return {newer,current,warning:$('#sec-message').textContent};
        """)
        self.assertEqual(result['newer'], 'Loading source catalogs…')
        self.assertIn('No ready SEC catalog', result['current'])
        self.assertEqual(result['warning'], 'Preserve this warning')


if __name__ == '__main__':
    unittest.main()
