"""Offline Evidence UI checks: no server, data, packages or Qlib worker needed."""
import pathlib
import shutil
import subprocess
import unittest

STATIC = pathlib.Path(__file__).resolve().parents[1] / "qlib_yahoo/webapp/static"


class WorkflowStructureTests(unittest.TestCase):
    def test_existing_console_hooks_and_local_transport(self):
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        app = (STATIC / "app.js").read_text(encoding="utf-8")
        js = (STATIC / "workflow.js").read_text(encoding="utf-8")
        self.assertIn('data-view="evidence"', html)
        self.assertLess(html.index('/static/workflow.js'), html.index('/static/app.js'))
        self.assertIn('initializeWorkflow();', app)
        finished = app[app.index('async function finished(job)'):app.index('// ================================================================= Alpha Lab')]
        self.assertIn('"candidate-validation"', finished)
        self.assertIn('"portfolio-validation"', finished)
        self.assertIn('await openRun(job.result)', finished)
        self.assertIn('startPolling(job)', js)
        self.assertIn('api("cancel", {})', app)
        for forbidden in ('innerHTML', 'fetch(', 'XMLHttpRequest', 'import(', 'setInterval', 'setTimeout'):
            self.assertNotIn(forbidden, js)
        self.assertIn('Older library entries lack immutable attempt history', js)
        self.assertIn('source-byte checks and numerical reproduction occur in the worker', js)
        self.assertIn('workflowChronologyJobFinished(job)', finished)
        self.assertLess(finished.index('workflowChronologyJobFinished(job)'), finished.index('job.state === "cancelled"'))
        self.assertIn('"chronology-trial"', finished)
        self.assertIn('"chronology-final"', finished)
        self.assertIn('await refreshWorkflowChronology()', finished)


@unittest.skipUnless(shutil.which("node"), "Node needed for focused JS behavior")
class WorkflowBehaviorTests(unittest.TestCase):
    def js(self, body):
        # A minimal DOM double exercises real workflow functions and event handlers.
        harness = r'''
const assert = require('node:assert/strict');
const nodes = new Map();
function el(tag, attrs = {}, ...children) {
  const n = {tag, ...attrs, textContent: String(attrs.text || ''), children: children.flat(Infinity).filter(v=>v!=null),
    value: String(attrs.value ?? ''), disabled: !!attrs.disabled,
    append(...v){this.children.push(...v)}, prepend(...v){this.children.unshift(...v)},
    checkValidity(){return this.value.trim() !== '' && Number.isFinite(Number(this.value)) && Number(this.value)>=Number(this.min||0)}};
  if(attrs.id) nodes.set('#'+attrs.id,n);
  return n;
}
function $(id){ if(!nodes.has(id)) nodes.set(id,el('div')); return nodes.get(id); }
function setChildren(n,...v){n.children=v.flat(Infinity).filter(x=>x!=null);return n;}
function text(n){return typeof n==='object' ? n.textContent+' '+n.children.map(text).join(' ') : String(n);}
const state={view:'evidence',job:null}, researchUI={overview:null};
const calls=[]; let polled=null;
let get=async(name,params)=>{calls.push([name,params]);return []};
let api=async(name,params)=>{calls.push([name,params]);return {id:'job',state:'running',attempt_id:'validation'}};
function startPolling(job){polled=job;state.job=job;workflowUpdateButtons();}
function setView(){} function openRun(){}
'''
        setup = r'''
initializeWorkflow();
function attempt(id='a',status='completed',kind='lab-evaluation') {
 return {id,status,kind,candidate_id:'candidate',result_digest:'digest',parents:[],request:{},source:{}};
}
function select(a=attempt()) {workflowUI.selected=a.id;workflowUI.detail={attempt:a,result:{expression:'rank(close)',settings:{delay:1}}};}
function preview(params) {return {source_attempt_id:params.attempt_id,candidate_id:'candidate',result_digest:'digest',
 expression:'rank(close)',dataset:'local',period:{start:'2020-01-01',end:'2021-01-01'},settings:{delay:1},
 differences:['next-open'],qualification:{status:'not_verified'},execution:{...params,annual_cash_rate:0,omissions:['market impact']}};}
function portfolioSetup(two=false) {
 workflowUI.rows=[{...attempt('v1','completed','candidate-validation'),result_digest:'sealed-v1'},
  {...attempt('v2','completed','candidate-validation'),result_digest:'sealed-v2'},attempt('failed','failed','candidate-validation')];
 workflowPortfolioOptions();
 $('#wf-p-leg1').value='v1'; $('#wf-p-leg2').value=two?'v2':'';
 $('#wf-p-weight1').value=two?'.6':'1'; $('#wf-p-weight2').value=two?'.3':'0';
 $('#wf-p-account').value='research-account';
}
function portfolioPreview(request) {return {schema_version:1,request,legs:workflowPortfolioPins(request).map((pin,i)=>({
 ...pin,result_digest:'candidate-digest-'+i,candidate_attempt_id:'source-'+i,expression:'rank(close)',qualification:{status:'exploratory_only'}})),
 period:{start:'2020-01-01',end:'2021-01-01'},dataset:'local',execution:{initial_cash:request.initial_cash,cost_per_side:request.cost_per_side,annual_borrow_rate:request.annual_borrow_rate,omissions:['impact']},
 risk_contract:{group_missing:'refuse'},qualification:{status:'exploratory_only'},policy:{version:'fixed-v1'}};}
function chronologySetup() {
 select(attempt('source'));
 workflowUI.detail.result={expression:'rank(close)',summary:{start:'2019-01-01',end:'2019-12-31'},
  settings:{delay:1,universe:'TOP500',neutralization:'MARKET',truncation:.05,cost_bps:5},dataset:{dataset:'local'}};
 workflowChronologyUseSource();
 for(const [phase,start,end] of [['development','2020-01-02','2020-03-31'],
   ['validation','2020-04-01','2020-06-30'],['final','2020-07-01','2020-09-30']]) {
   $('#wf-c-'+phase+'-start').value=start; $('#wf-c-'+phase+'-end').value=end;
 }
 $('#wf-c-rule').value='Review net returns and drawdown, retain two signals';
 $('#wf-c-phase').value='development';
}
function chronologyPreview(request) {
 const binding={source_attempt_id:request.source_attempt_id,source_result_digest:'digest',
  settings:workflowChronology.source.result.settings,source_identity:{dates:Array(20000).fill('2020-01-02'),symbols:Array(30000).fill('ABC')}};
 return {schema_version:1,protocol:{...request,schema_version:1,sessions:Array(20000).fill('2020-01-02'),
  metadata:{chronology_workflow:binding,...(request.samples?{samples:request.samples}:{})}},
  sample_split:{assignments:{development:[]},excluded:[],overlap_check:'not_available'},
  qualification:{status:'exploratory_only',limitations:['source bytes pending until execution']}};
}
function chronologyView(population=[],selected=[],selection_id=null,accesses=[]) {
 return {schema_version:1,protocol_id:'protocol',source_attempt_id:'source',source_result_digest:'digest',
  intervals:{development:{start:'2020-01-02',end:'2020-03-31'}},population,selected,selection_id,accesses,
  sample_split:{excluded:[],overlap_check:'not_available'},qualification:{status:'exploratory_only'}};
}
function chronologyDeclared(view=chronologyView()) {
 workflowChronology.protocolId='protocol'; workflowChronology.view=view; workflowChronology.stale=false;
 workflowUpdateButtons();
}
function chronologySelection(selected,exposure_reasons=[]) {
 return {...attempt('selection','completed','chronology.selection'),
  request:{protocol_id:'protocol',selected,decision_reason:'Explicit comparison',exposure_reasons}};
}
function chronologyRef(id,status='completed') {return {id,status,result_digest:'digest-'+id,parents:['protocol']};}
function chronologyCoverage(status='complete') {
 return {execution_policy:status==='not_supplied'?'calendar-window-v1':'observation-filter-v1',
  excluded:status==='not_supplied'?[]:[{id:'excluded-middle',phase:'development'}],overlap_check:'explicit_intervals',
  coverage:{status,unit:'daily portfolio return observations',phases:Object.fromEntries(workflowPhases.map(phase=>[
   phase,{observed_count:3,checked_count:status==='complete'?3:0,
    unchecked_sessions:status==='complete'?[]:['2020-01-02','2020-01-03','2020-01-06'],
    excluded_sessions:phase==='development' && status!=='not_supplied'?['2020-01-03']:[],
    evaluation_sessions:status==='not_supplied'?null:phase==='development'?['2020-01-02','2020-01-06']:['2020-01-02','2020-01-03','2020-01-06'],
    ready:status!=='partial',refusal:status==='partial'?'Explicit intervals need complete coverage across all phases':null}]))}};
}
'''
        script = harness + (STATIC / "workflow.js").read_text(encoding="utf-8")
        script += '\n(async()=>{\n' + setup + body + '\n})().catch(e=>{console.error(e);process.exitCode=1});'
        result = subprocess.run([shutil.which("node")], input=script, capture_output=True, text=True, encoding="utf-8", timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_explicit_fetch_all_statuses_and_selection_never_previews(self):
        self.js(r'''
assert.equal(calls.length,0);
get=async(name,p)=>{calls.push([name,p]);return name==='workflow-attempts'
 ? ['completed','reused','failed','rejected','cancelled','interrupted_unknown','pending'].map((s,i)=>attempt(String(i),s))
 : {attempt:attempt(p.id),result:{expression:'<img src=x onerror=evil()>',settings:{delay:1}}};};
await refreshWorkflow();
assert.equal(workflowUI.rows.length,7);
assert.match(text($('#wf-attempts')),/reused \(cache\)/);
assert.equal($('#wf-message').textContent,'History refreshed. Select an attempt to inspect.');
await selectWorkflowAttempt('0');
assert.deepEqual(calls.map(c=>c[0]),['workflow-attempts','workflow-detail']);
assert.equal($('#wf-run').disabled,true);
assert.match(text($('#wf-detail')),/<img src=x onerror=evil\(\)>/);
assert.equal($('#wf-detail').children.some(n=>n.tag==='img'),false);
''')

    def test_units_explicit_preview_and_job_flow(self):
        self.js(r'''
select();
api=async(name,p)=>{calls.push([name,p]);return name==='workflow-preview'?preview(p):{id:'j',state:'running',attempt_id:'validation'};};
await previewWorkflow();
assert.equal(calls[0][1].initial_cash,100000);
assert.equal(calls[0][1].cost_per_side,.0005);
assert.equal(calls[0][1].annual_borrow_rate,.004);
assert.equal($('#wf-run').disabled,false);
assert.match(text($('#wf-preview')),/following session open/);
assert.match(text($('#wf-preview')),/not_verified/);
assert.match(text($('#wf-preview')),/market impact/);
const pending=runWorkflowValidation();
assert.equal($('#wf-run').disabled,true);
await runWorkflowValidation(); await pending;
assert.equal(calls.filter(c=>c[0]==='workflow-validate').length,1);
assert.equal(polled.id,'j');
assert.equal(workflowUI.preview,null);
assert.equal($('#wf-run').disabled,true);
''')

    def test_selection_race_and_cost_change_invalidate_preview(self):
        self.js(r'''
select(); let release;
api=(name,p)=>new Promise(resolve=>release=()=>resolve(preview(p)));
const old=previewWorkflow();
get=async(name,p)=>({attempt:attempt(p.id),result:{}});
await selectWorkflowAttempt('b'); release(); await old;
assert.equal(workflowUI.preview,null);
assert.equal($('#wf-run').disabled,true);
api=async(name,p)=>preview(p); await previewWorkflow();
assert.equal(workflowCanRun(),true);
$('#wf-cost').value='10'; $('#wf-cost').oninput();
assert.equal(workflowUI.preview,null); assert.equal(workflowCanRun(),false);
assert.match($('#wf-message').textContent,/settings changed/);
''')

    def test_exact_refusal_and_mismatched_identity_fail_closed(self):
        self.js(r'''
select(); api=async()=>{throw Error('exact refusal: unsupported saved delay');};
await previewWorkflow();
assert.equal($('#wf-message').textContent,'exact refusal: unsupported saved delay');
assert.equal(workflowCanRun(),false);
api=async(name,p)=>({...preview(p),result_digest:'different'});
await previewWorkflow();
assert.match($('#wf-message').textContent,/identity differs/);
assert.equal(workflowCanRun(),false);
api=async(name,p)=>({...preview(p),execution:{...preview(p).execution,cost_per_side:.1}});
await previewWorkflow(); assert.match($('#wf-message').textContent,/settings differ/);
assert.equal(workflowCanRun(),false);
''')

    def test_busy_invalid_numeric_and_ineligible_attempts_cannot_submit(self):
        self.js(r'''
select(); api=async(name,p)=>{calls.push([name,p]);return preview(p);};
await previewWorkflow(); state.job={state:'running'};
await runWorkflowValidation(); assert.equal(calls.length,1);
state.job=null; $('#wf-cash').value=''; await previewWorkflow();
assert.equal(workflowCanRun(),false); assert.equal(calls.length,1);
$('#wf-cash').value='100000';
for(const s of ['failed','cancelled','rejected','pending','interrupted_unknown']) {
 select(attempt('a',s)); await previewWorkflow(); assert.equal(workflowEligible(),false);
}
select(attempt('a','completed','candidate-validation')); await previewWorkflow();
assert.equal(calls.length,1);
select(attempt('a','reused')); assert.equal(workflowEligible(),true);
''')

    def test_overlapping_detail_requests_and_stale_error_are_ignored(self):
        self.js(r'''
let rejectOld,resolveNew;
get=(name,p)=>new Promise((resolve,reject)=>{if(p.id==='a')rejectOld=reject;else resolveNew=resolve;});
const a=selectWorkflowAttempt('a'), b=selectWorkflowAttempt('b');
resolveNew({attempt:attempt('b'),result:{}}); await b;
rejectOld(Error('stale failure')); await a;
assert.equal(workflowUI.detail.attempt.id,'b');
assert.doesNotMatch($('#wf-message').textContent,/stale failure/);
''')

    def test_real_completion_hook_opens_validation_run(self):
        app = (STATIC / "app.js").read_text(encoding="utf-8")
        finished = app[app.index('async function finished(job)'):app.index('// ================================================================= Alpha Lab')]
        self.js(finished + r'''
const events=[];
function renderJob(){} function toast(){}
async function refreshRuns(){events.push('refreshRuns');}
function afterSelection(){} function setRunFilter(){}
async function openRun(path){events.push(['openRun',path]);}
function setView(view){events.push(['setView',view]);}
function setResultTab(){} function home(v){return v;}
const document={querySelector:()=>null}, CSS={escape:v=>v};
await finished({state:'completed',kind:'candidate-validation',result:'local/run'});
assert.deepEqual(events,['refreshRuns',['openRun','local/run'],['setView','research']]);
events.length=0;
await finished({state:'completed',kind:'portfolio-validation',result:'local/portfolio'});
assert.deepEqual(events,['refreshRuns',['openRun','local/portfolio'],['setView','research']]);
''')

    def test_real_stop_hook_uses_existing_cancel_endpoint(self):
        app = (STATIC / "app.js").read_text(encoding="utf-8")
        cancel = app[app.index('async function cancelJob()'):app.index('function startPolling(job)')]
        self.js(cancel + r'''
function toast(){}
api=async(name,p)=>{calls.push([name,p]);return {cancelled:true};};
for (const kind of ['candidate-validation','portfolio-validation','chronology-trial','chronology-final']) {
 state.job={state:'running',kind};
 await cancelJob();
}
assert.deepEqual(calls,Array.from({length:4},()=>['cancel',{}]));
''')

    def test_real_chronology_completion_hook_preserves_protocol_and_never_opens_run(self):
        app = (STATIC / "app.js").read_text(encoding="utf-8")
        finished = app[app.index('async function finished(job)'):app.index('// ================================================================= Alpha Lab')]
        self.js(finished + r'''
function renderJob(job){state.job=job;} function toast(){} function setBottomTab(){}
function setView(view){assert.equal(view,'evidence');}
function openRun(){throw Error('chronology must not open a Qlib run');}
chronologySetup(); chronologyDeclared();
get=async(name,p)=>{calls.push([name,p]);assert.equal(name,'workflow-attempts');return [];};
api=async(name,p)=>{calls.push([name,p]);assert.equal(name,'workflow-chronology-view');return chronologyView();};
for(const kind of ['chronology-trial','chronology-final']) {
 await finished({state:'completed',kind,result:'local/chronology-receipt',receipt:{access_label:'reused_holdout'}});
 assert.equal(workflowChronology.protocolId,'protocol'); assert.equal(workflowChronology.source.attempt.id,'source');
 assert.equal(workflowChronologyReady(),true);
}
assert.deepEqual(calls.map(c=>c[0]),['workflow-attempts','workflow-chronology-view','workflow-attempts','workflow-chronology-view']);
for(const state of ['failed','cancelled']) {
 await finished({state,kind:'chronology-final'});
 assert.equal(workflowChronology.stale,true); assert.match($('#wf-c-message').textContent,new RegExp(state));
}
assert.equal(calls.length,4); // Early failure/cancel returns don't scan history or read final results.
''')

    def test_selects_have_explicit_stable_label_targets_after_option_refresh(self):
        self.js(r'''
function descendants(n) {
 return typeof n==='object' ? [n,...n.children.flatMap(descendants)] : [];
}
const labels=descendants($('#center')).filter(n=>n.tag==='label');
const expected=new Map([
 ['wf-p-leg1','Validated candidate 1'], ['wf-p-leg2','Validated candidate 2'],
 ['wf-p-group-level','Group level (required for group limits)'], ['wf-c-phase','Trial phase'],
]);
function assertLabels() {
 for(const [id,caption] of expected) {
   const select=$('#'+id), matching=labels.filter(n=>n.for===id);
   assert.equal(matching.length,1);
   assert.ok(matching[0].children.includes(select));
   assert.equal(select['aria-labelledby'],id+'-label');
   const labelSpan=$('#'+select['aria-labelledby']);
   assert.ok(matching[0].children.includes(labelSpan));
   assert.equal(labelSpan.textContent,caption); // Exact label excludes every select option.
 }
}
assertLabels();
portfolioSetup(true);
workflowUI.rows[0].display_label='rank(close) · completed · a very long source label';
workflowPortfolioOptions();
assertLabels();
assert.match(text(labels.find(n=>n.for==='wf-p-leg1')),/rank\(close\)/);
assert.equal($('#wf-p-leg1-label').textContent,'Validated candidate 1');
$('#wf-p-leg1').value='v2'; $('#wf-p-leg1').onchange(); assertLabels();
workflowUI.rows=[]; workflowPortfolioOptions(); assertLabels();
assert.equal(calls.length,0);
''')

    def test_risk_limit_label_is_stable_and_state_is_a_separate_description(self):
        self.js(r'''
for(const key of workflowLimitKeys) {
 const input=$('#wf-p-limit-'+key);
 assert.equal(input['aria-labelledby'],'wf-p-limit-'+key+'-label');
 assert.equal(input['aria-describedby'],'wf-p-limit-state-'+key);
 const caption=$('#'+input['aria-labelledby']).textContent;
 assert.ok(caption); assert.doesNotMatch(caption,/Disabled|Enabled/);
 assert.equal($('#'+input['aria-describedby']).textContent,'Disabled');
 input.value='0'; input.oninput();
 assert.equal($('#'+input['aria-labelledby']).textContent,caption);
 assert.match($('#'+input['aria-describedby']).textContent,/Enabled.*0/);
 input.value=''; input.oninput();
 assert.equal($('#'+input['aria-labelledby']).textContent,caption);
 assert.equal($('#'+input['aria-describedby']).textContent,'Disabled');
}
assert.equal(calls.length,0);
''')

    def test_portfolio_payload_disabled_limits_units_and_one_two_legs(self):
        self.js(r'''
portfolioSetup();
const single=workflowPortfolioParams();
assert.deepEqual(Object.keys(single).sort(),['validation_ids','coefficients','account_id','initial_cash','cost_per_side','annual_borrow_rate','limits','group_overlay','group_level'].sort());
assert.deepEqual(single.validation_ids,['v1']); assert.deepEqual(single.coefficients,[1]);
assert.equal(single.initial_cash,100000); assert.equal(single.cost_per_side,.0005); assert.equal(single.annual_borrow_rate,.004);
assert.deepEqual(single.limits,{gross:null,net:null,name:null,group_gross:null,group_net:null,turnover:null,liquidity:null});
assert.equal(single.group_overlay,null); assert.equal(single.group_level,null);
assert.doesNotMatch(text($('#wf-p-leg1')),/failed/);
portfolioSetup(true); const two=workflowPortfolioParams();
assert.deepEqual(two.validation_ids,['v1','v2']); assert.deepEqual(two.coefficients,[.6,.3]);
assert.equal(calls.length,0); assert.equal($('#wf-p-run').disabled,true);
''')

    def test_portfolio_group_requirements_zero_limits_and_invalid_allocations(self):
        self.js(r'''
portfolioSetup(true);
$('#wf-p-weight2').value='.5'; assert.throws(workflowPortfolioParams,/sum to at most 1/);
$('#wf-p-weight2').value='.3'; $('#wf-p-leg2').value='v1'; assert.throws(workflowPortfolioParams,/distinct/);
$('#wf-p-leg2').value='v2'; $('#wf-p-limit-group_gross').value='0';
assert.throws(workflowPortfolioParams,/explicit classification/);
$('#wf-p-group-path').value='D:/prepared/groups'; $('#wf-p-group-level').value='sector';
$('#wf-p-limit-net').value='0'; $('#wf-p-limit-turnover').value='.25'; $('#wf-p-limit-liquidity').value='.01';
const p=workflowPortfolioParams(); assert.equal(p.limits.group_gross,0); assert.equal(p.limits.net,0);
assert.equal(p.group_overlay,'D:/prepared/groups'); assert.equal(p.group_level,'sector');
workflowPortfolioButtons(); assert.match($('#wf-p-limit-state-group_gross').textContent,/Enabled/);
$('#wf-p-group-level').value='unknown'; assert.throws(workflowPortfolioParams,/Unknown groups/);
$('#wf-p-limit-group_gross').value=''; assert.equal(workflowPortfolioParams().group_overlay,null);
$('#wf-p-limit-net').value='-1'; assert.throws(workflowPortfolioParams,/nonnegative/);
$('#wf-p-limit-net').value='NaN'; assert.throws(workflowPortfolioParams,/finite/);
$('#wf-p-limit-net').value=''; $('#wf-p-cash').value='0'; assert.throws(workflowPortfolioParams,/positive/);
''')

    def test_portfolio_preview_and_run_share_payload_and_existing_jobs(self):
        self.js(r'''
portfolioSetup(true);
api=async(name,p)=>{calls.push([name,p]);return name==='workflow-portfolio-preview'?portfolioPreview(p):{id:'portfolio-job',state:'running',kind:'portfolio-validation',attempt_id:'portfolio-attempt'};};
await previewWorkflowPortfolio();
assert.equal(workflowPortfolioCanRun(),true);
assert.doesNotMatch(text($('#wf-p-preview')),/sealed-v1/); assert.match(text($('#wf-p-preview')),/exploratory_only/);
assert.match(text($('#wf-p-preview')),/Disabled/);
const provenance=$('#wf-p-preview').children[0].children.find(n=>n.tag==='details' && text(n).includes('full bindings'));
provenance.open=true; provenance.ontoggle({currentTarget:provenance});
assert.match(text(provenance),/sealed-v1/);
const running=runWorkflowPortfolio(); assert.equal($('#wf-p-run').disabled,true);
await runWorkflowPortfolio(); await running;
assert.deepEqual(calls.map(c=>c[0]),['workflow-portfolio-preview','workflow-portfolio-run']);
assert.deepEqual(calls[0][1],calls[1][1]); assert.equal(polled.kind,'portfolio-validation');
assert.equal(workflowPortfolio.preview,null); assert.equal($('#wf-p-run').disabled,true);
''')

    def test_portfolio_refuses_canonical_or_immutable_mismatch_and_exact_errors(self):
        self.js(r'''
portfolioSetup();
api=async(name,p)=>{const result=portfolioPreview(p);result.request={...p,initial_cash:2};return result;};
await previewWorkflowPortfolio(); assert.match($('#wf-p-message').textContent,/Canonical portfolio request differs/);
assert.equal(workflowPortfolioCanRun(),false);
api=async(name,p)=>{const result=portfolioPreview(p);result.legs[0].validation_result_digest='another-result';return result;};
await previewWorkflowPortfolio(); assert.match($('#wf-p-message').textContent,/immutable validation results differ/);
assert.equal(workflowPortfolioCanRun(),false);
api=async(name,p)=>{const result=portfolioPreview(p);result.execution.cost_per_side=.1;return result;};
await previewWorkflowPortfolio(); assert.match($('#wf-p-message').textContent,/execution settings differ/);
assert.equal(workflowPortfolioCanRun(),false);
api=async()=>{throw Error('exact refusal: missing dated group for XYZ');};
await previewWorkflowPortfolio(); assert.equal($('#wf-p-message').textContent,'exact refusal: missing dated group for XYZ');
assert.equal(workflowPortfolioCanRun(),false);
''')

    def test_portfolio_preview_selection_and_input_races_fail_closed(self):
        self.js(r'''
portfolioSetup(); let release;
api=(name,p)=>new Promise(resolve=>release=()=>resolve(portfolioPreview(p)));
const old=previewWorkflowPortfolio();
$('#wf-p-leg1').value='v2'; $('#wf-p-leg1').onchange();
release(); await old; assert.equal(workflowPortfolio.preview,null);
api=async(name,p)=>portfolioPreview(p); await previewWorkflowPortfolio(); assert.equal(workflowPortfolioCanRun(),true);
$('#wf-p-cost').value='8'; $('#wf-p-cost').oninput(); assert.equal(workflowPortfolioCanRun(),false);
await previewWorkflowPortfolio(); assert.equal(workflowPortfolioCanRun(),true);
workflowUI.rows.find(a=>a.id==='v2').result_digest='changed';
assert.equal(workflowPortfolioCanRun(),false);
await runWorkflowPortfolio(); assert.equal(polled,null);
''')

    def test_portfolio_refresh_and_recovery_invalidate_pending_previews(self):
        self.js(r'''
portfolioSetup(); let release;
api=(name,p)=>new Promise(resolve=>release=()=>resolve(portfolioPreview(p)));
const pending=previewWorkflowPortfolio(); get=async()=>workflowUI.rows;
await refreshWorkflow(); release(); await pending; assert.equal(workflowPortfolio.preview,null);
portfolioSetup(); api=async(name,p)=>{calls.push([name,p]);return name==='workflow-portfolio-preview'?portfolioPreview(p):{recovered:['old'],remaining_pending:1};};
await previewWorkflowPortfolio(); assert.equal(workflowPortfolioCanRun(),true);
const recovery=recoverWorkflow(); assert.equal($('#wf-p-run').disabled,true);
await recoverWorkflow(); await recovery;
assert.equal(calls.filter(c=>c[0]==='workflow-recover').length,1);
assert.equal(workflowPortfolio.preview,null);
assert.match(text($('#wf-recovery')),/Refresh history explicitly/);
assert.deepEqual(calls.map(c=>c[0]),['workflow-portfolio-preview','workflow-recover']);
''')

    def test_portfolio_invalid_or_busy_never_dispatches(self):
        self.js(r'''
portfolioSetup(); $('#wf-p-account').value='';
await previewWorkflowPortfolio(); assert.match($('#wf-p-message').textContent,/account ID/); assert.equal(calls.length,0);
$('#wf-p-account').value='one-account'; state.job={state:'running'};
await previewWorkflowPortfolio(); await recoverWorkflow(); await runWorkflowPortfolio(); assert.equal(calls.length,0);
state.job=null; workflowUI.rows[0].status='failed';
await previewWorkflowPortfolio(); assert.equal(calls.length,0); assert.match($('#wf-p-message').textContent,/incomplete/);
''')

    def test_portfolio_canonical_path_is_reviewed_and_sent_without_other_changes(self):
        self.js(r'''
portfolioSetup(); $('#wf-p-limit-group_net').value='.4';
$('#wf-p-group-path').value='D:/prepared/groups/'; $('#wf-p-group-level').value='industry';
api=async(name,p)=>{calls.push([name,p]);if(name==='workflow-portfolio-run')return {id:'j',kind:'portfolio-validation',state:'running'};
 const result=portfolioPreview(p);result.request={...p,group_overlay:'D:\\prepared\\groups'};return result;};
await previewWorkflowPortfolio(); assert.equal(workflowPortfolioCanRun(),true);
await runWorkflowPortfolio(); assert.equal(calls[1][1].group_overlay,'D:\\prepared\\groups');
assert.equal(calls[1][1].group_level,'industry'); assert.equal(calls[1][1].limits.group_net,.4);
state.job=null; portfolioSetup(); $('#wf-p-group-path').value='D:/prepared/groups';
api=async(name,p)=>({...portfolioPreview(p),request:{...p,group_overlay:'D:/another/groups'}});
await previewWorkflowPortfolio(); assert.equal(workflowPortfolioCanRun(),false);
''')

    def test_old_portfolio_error_cannot_replace_new_preview(self):
        self.js(r'''
portfolioSetup(); let rejectOld;
api=()=>new Promise((resolve,reject)=>rejectOld=reject);
const old=previewWorkflowPortfolio(); $('#wf-p-borrow').value='.8'; $('#wf-p-borrow').oninput();
api=async(name,p)=>portfolioPreview(p); await previewWorkflowPortfolio();
rejectOld(Error('old refusal')); await old;
assert.equal(workflowPortfolioCanRun(),true); assert.doesNotMatch($('#wf-p-message').textContent,/old refusal/);
assert.equal(workflowPortfolio.preview.request.annual_borrow_rate,.008);
''')

    def test_recovery_error_does_not_keep_old_previews_or_refresh_automatically(self):
        self.js(r'''
select(); api=async(name,p)=>preview(p); await previewWorkflow();
portfolioSetup(); api=async(name,p)=>portfolioPreview(p); await previewWorkflowPortfolio();
api=async(name,p)=>{calls.push([name,p]);throw Error('exact recovery refusal');};
await recoverWorkflow();
assert.equal(workflowUI.preview,null); assert.equal(workflowPortfolio.preview,null);
assert.equal(workflowUI.recovering,false); assert.deepEqual(calls,[['workflow-recover',{}]]);
assert.match(text($('#wf-recovery')),/exact recovery refusal/);
''')

    def test_saved_portfolio_details_reuse_run_and_lazy_result_view(self):
        self.js(r'''
select(attempt('p','completed','portfolio-validation'));
workflowUI.detail.result={run_path:'local/portfolio',portfolio:{period:{start:'2020',end:'2021'},policy:{account_id:'one-account'},
 risk_contract:{missing_groups:'refuse'}},summary:{},qualification:{status:'exploratory_simulation'}};
renderWorkflowDetail();
assert.match(text($('#wf-detail')),/Open saved portfolio run/);
assert.match(text($('#wf-detail')),/Portfolio allocation, risk and reconciliation/);
const details=$('#wf-detail').children.find(n=>n.tag==='details' && text(n).includes('Portfolio allocation'));
details.open=true; details.ontoggle({currentTarget:details});
assert.match(text(details),/one-account/); assert.match(text(details),/refuse/);
''')

    def test_bounded_lazy_provenance_and_safe_text(self):
        self.js(r'''
let reads=0;
const source={get calendar(){reads++;return Array.from({length:100000},(_,i)=>String(i));},symbols:Array(30000).fill('<script>evil()</script>')};
const disclosure=workflowDisclosure('Source details',source);
assert.equal(reads,0); assert.equal(disclosure.children[1].children.length,0);
disclosure.open=true; disclosure.ontoggle({currentTarget:disclosure});
assert.equal(reads,1);
const output=text(disclosure); assert.match(output,/100000 items/); assert.match(output,/showing first 3/);
assert.ok(output.length<3000); assert.match(output,/<script>evil/);
disclosure.ontoggle({currentTarget:disclosure}); assert.equal(reads,1);
let deep={}; for(let i=0;i<30;i++) deep={a:deep,b:deep};
assert.ok(text(workflowValue(deep)).length<6000);
assert.ok(workflowText('x'.repeat(100000)).length<1000);
''')

    def test_display_labels_short_ids_and_completed_empty_refresh_message(self):
        self.js(r'''
const id='12345678-full-validation-id', candidate='abcdefgh-full-candidate-id';
workflowUI.rows=[{...attempt(id,'completed','candidate-validation'),candidate_id:candidate,
 display_label:'rank(close)',created_at:'2026-10-20'}, {...attempt('portfolio-full-id','completed','portfolio-validation'),display_label:'Portfolio: myaccount'}];
renderWorkflowList(); workflowPortfolioOptions();
const row=$('#wf-attempts').children[0], option=$('#wf-p-leg1').children[1];
assert.equal(row.children[0].textContent,'rank(close)');
assert.match(text(row),/candidate-validation · completed · 2026-10-20/);
assert.match(text(row),/12345678…/); assert.doesNotMatch(text(row),/full-validation-id|full-candidate-id/);
assert.ok(option.textContent.startsWith('rank(close)')); assert.equal(option.value,id);
assert.doesNotMatch(option.textContent,/full-validation-id/);
assert.match(text($('#wf-attempts')),/Portfolio: myaccount/);
assert.equal(workflowAttemptLabel(attempt()),'Lab evaluation');
workflowUI.rows[0].display_label='<img src=x onerror=evil()>'; renderWorkflowList();
assert.equal($('#wf-attempts').children[0].children[0].tag,'strong');
assert.match($('#wf-attempts').children[0].children[0].textContent,/<img/);
get=async()=>[]; await refreshWorkflow();
assert.equal($('#wf-message').textContent,'History refreshed. No recorded attempts.');
''')

    def test_rejection_has_error_without_empty_metrics_or_qualification_prompt(self):
        self.js(r'''
get=async()=>({attempt:{...attempt('bad','rejected'),result_digest:null,
 request:{expression:'bad(formula)'},error:{message:'unsupported formula',phase:'preview'}},result:{summary:{}}});
await selectWorkflowAttempt('bad');
const content=text($('#wf-detail'));
assert.match(content,/unsupported formula/); assert.doesNotMatch(content,/Lab screening metrics|Not recorded|preview for current checks/);
assert.equal($('#wf-detail').children[1].role,'alert');
assert.equal($('#wf-message').textContent,'Attempt rejected.');
assert.equal($('#wf-preview-button').disabled,true); assert.equal($('#wf-run').disabled,true);
''')

    def test_compact_candidate_preview_and_detail_keep_full_provenance_on_demand(self):
        self.js(r'''
select({...attempt('attempt-full-identity'),candidate_id:'candidate-full-identity',result_digest:'digest-full-identity'});
workflowUI.detail.result.summary={sharpe:1.23456,returns:.123456,turnover:0};
renderWorkflowDetail();
let content=text($('#wf-detail'));
assert.match(content,/1\.23/); assert.match(content,/12\.35%/); assert.match(content,/0%/);
assert.doesNotMatch(content,/attempt-full-identity|candidate-full-identity|digest-full-identity|Not recorded/);
const provenance=$('#wf-detail').children.find(n=>n.tag==='details' && text(n).includes('full identities'));
provenance.open=true; provenance.ontoggle({currentTarget:provenance});
assert.match(text(provenance),/attempt-full-identity/);
const p={...preview(workflowParams()),source_attempt_id:'attempt-full-identity',result_digest:'digest-full-identity',candidate_id:'candidate-full-identity'};
renderWorkflowPreview(p); content=text($('#wf-preview'));
assert.match(content,/rank\(close\)|following session open/); assert.match(content,/Not modeled:.*market impact/);
assert.match(content,/not_verified/); assert.doesNotMatch(content,/attempt-full-identity|digest-full-identity|candidate-full-identity/);
const details=$('#wf-preview').children[0].children.find(n=>n.tag==='details' && text(n).includes('full identities'));
details.open=true; details.ontoggle({currentTarget:details});
assert.match(text(details),/digest-full-identity/);
''')

    def test_filtered_detail_labels_observations_and_safely_displays_evaluation_metadata(self):
        self.js(r'''
select();
workflowUI.detail.result={expression:'rank(close)',settings:{evaluation_sessions:['2020-01-02','2020-01-06']},
 summary:{sharpe:1.2,returns:.1,drawdown:.03,days:2},evaluation:{
  policy:'observation-filter-v1 <img src=x onerror=evil()>',
  metrics_meaning:'Selected PnL <script>evil()</script>',
  observed_count:3,selected_count:2,selected_sessions:['2020-01-02','2020-01-06'],
  diagnostics_scope:'Full continuous signal path'}};
renderWorkflowDetail();
const detail=$('#wf-detail'), output=text(detail);
assert.match(output,/Observation-filter warning/);
assert.match(output,/not a continuous investment return or a masked trading policy/);
assert.match(output,/Qlib validation is unavailable; chronology selection remains available/);
assert.match(output,/Selected-observation metrics · saved result/);
assert.match(output,/Selected observations\s+2/); assert.doesNotMatch(output,/Lab screening metrics/);
assert.match(output,/observation-filter-v1 <img src=x onerror=evil\(\)>/);
assert.match(output,/Selected PnL <script>evil\(\)<\/script>/);
const tags=n=>typeof n==='object'?[n.tag,...n.children.flatMap(tags)]:[];
assert.equal(tags(detail).includes('img'),false); assert.equal(tags(detail).includes('script'),false);
const saved=detail.children.find(n=>n.tag==='details' && text(n).includes('Saved settings and metrics'));
assert.doesNotMatch(text(saved),/Full continuous signal path/);
saved.open=true; saved.ontoggle({currentTarget:saved});
assert.match(text(saved),/evaluation/); assert.match(text(saved),/Full continuous signal path/);
assert.match(text(saved),/drawdown follows the compressed selected-observation PnL/);
// Older filtered results still get a warning even if evaluation metadata is absent.
delete workflowUI.detail.result.evaluation; renderWorkflowDetail();
assert.match(text($('#wf-detail')),/Selected-observation metrics/);
assert.match(text($('#wf-detail')),/Observation-filter warning/);
''')

    def test_filtered_results_cannot_preview_qlib_or_pin_a_new_protocol_source(self):
        self.js(r'''
for(const status of ['completed','reused']) {
 for(const sessions of [[],['2020-01-02','2020-01-06']]) {
  select(attempt('filtered',status)); workflowUI.detail.result.settings.evaluation_sessions=sessions;
  workflowUI.preview=preview(workflowParams()); workflowUI.previewKey=workflowKey();
  workflowUpdateButtons();
  assert.equal(workflowEligible(),false); assert.equal(workflowCanRun(),false);
  assert.equal($('#wf-preview-button').disabled,true); assert.equal($('#wf-run').disabled,true);
  assert.equal($('#run').disabled,true); assert.equal($('#wf-c-use').disabled,true);
  await previewWorkflow(); await runWorkflowValidation(); assert.equal(calls.length,0);
  workflowChronologyUseSource(); assert.equal(workflowChronology.source,null);
 }
}
select(); workflowUI.detail.result.evaluation={policy:'observation-filter-v1'};
workflowUpdateButtons(); assert.equal($('#wf-preview-button').disabled,true);
assert.equal($('#wf-c-use').disabled,true);
const filtered=workflowUI.detail;
get=async()=>filtered; await selectWorkflowAttempt('a');
assert.match($('#wf-message').textContent,/Selected-observation result.*Qlib validation is unavailable/);
for(const settings of [{delay:1},{delay:1,evaluation_sessions:null}]) {
 select(); workflowUI.detail.result.settings=settings; workflowUpdateButtons();
 assert.equal(workflowEligible(),true); assert.equal($('#wf-preview-button').disabled,false);
 assert.equal($('#wf-c-use').disabled,false);
 renderWorkflowDetail(); assert.doesNotMatch(text($('#wf-detail')),/Observation-filter warning/);
}
''')

    def test_filtered_chronology_cards_label_observations_and_remain_selectable(self):
        self.js(r'''
chronologySetup(); const ref=chronologyRef('filtered');
chronologyDeclared(chronologyView([ref]));
get=async(name,p)=>({attempt:{...attempt(p.id),result_digest:ref.result_digest},
 result:{expression:'rank(close)',settings:{evaluation_sessions:['2020-01-02','2020-01-06']},
  summary:{sharpe:1.2,returns:.1,days:2},evaluation:{policy:'observation-filter-v1',
   metrics_meaning:'Selected observations retain full-path costs'}}});
await loadWorkflowChronologyCandidates();
const output=text($('#wf-c-candidates'));
assert.match(output,/Selected-observation metrics/); assert.match(output,/Selected observations\s+2/);
assert.match(output,/observation-filter-v1/); assert.match(output,/Selected observations retain full-path costs/);
assert.match(output,/not a continuous investment return/);
const input=workflowChronology.candidateInputs[0].input;
assert.equal(input.disabled,false); input.checked=true; input.onchange({currentTarget:input});
$('#wf-c-reason').value='Reviewed selected-observation evidence'; workflowUpdateButtons();
assert.equal(workflowChronologyCanFreeze(),true); assert.equal($('#wf-c-freeze').disabled,false);
assert.deepEqual([...workflowChronology.chosen],['filtered']);
''')

    def test_chronology_explicit_preview_declare_and_source_binding(self):
        self.js(r'''
chronologySetup(); assert.equal(calls.length,0);
assert.equal($('#wf-c-declare').disabled,true);
api=async(name,p)=>{calls.push([name,p]);return {...chronologyPreview(p),...(name.endsWith('declare')?{protocol_id:'protocol'}:{})};};
await previewWorkflowChronology();
assert.equal($('#wf-c-declare').disabled,false);
const params=calls[0][1];
assert.deepEqual(Object.keys(params).sort(),['source_attempt_id','development','validation','final','selection_rule'].sort());
assert.deepEqual(params.selection_rule,{rule:'Review net returns and drawdown, retain two signals'});
assert.match(text($('#wf-c-source')),/TOP500|MARKET/);
assert.match(text($('#wf-c-preview')),/no purge verification|No purge verification/);
assert.match(text($('#wf-c-preview')),/omits stock borrow/);
assert.doesNotMatch(text($('#wf-c-preview')),/20000|30000/);
assert.ok(text($('#wf-c-preview')).length<2200);
select(attempt('another')); // Explicitly pinned source survives explorer selection.
assert.equal(workflowChronologyParams().source_attempt_id,'source');
await declareWorkflowChronology();
assert.equal(workflowChronology.protocolId,'protocol');
assert.equal($('#wf-c-trial').disabled,true);
assert.deepEqual(calls.map(c=>c[0]),['workflow-chronology-preview','workflow-chronology-declare']);
assert.deepEqual(calls[1][1],params); assert.equal(polled,null);
''')

    def test_chronology_cheap_inputs_and_samples_are_not_overrides(self):
        self.js(r'''
chronologySetup();
$('#wf-c-development-end').value='2020-04-01';
assert.throws(workflowChronologyParams,/non-overlapping/);
$('#wf-c-development-end').value='2020-02-30';
assert.throws(workflowChronologyParams,/valid start/);
$('#wf-c-development-end').value='2020-03-31'; $('#wf-c-rule').value='';
await previewWorkflowChronology(); assert.equal(calls.length,0); assert.match($('#wf-c-message').textContent,/selection rule/);
$('#wf-c-rule').value='Review';
$('#wf-c-samples').value='{"mask":[1]}'; assert.throws(workflowChronologyParams,/sample intervals/);
$('#wf-c-samples').value='[{"id":"x","session":"2020-01-02","information_start":"2020-01-01","information_end":"2020-01-03","mask":true}]';
assert.throws(workflowChronologyParams,/not a mask or source override/);
$('#wf-c-samples').value='[{"id":"x","session":"2020-01-02","information_start":"2020-01-01","information_end":"2020-01-03"}]';
assert.equal(workflowChronologyParams().samples.length,1);
$('#wf-c-samples').value='['; assert.throws(workflowChronologyParams,/JSON list/);
$('#wf-c-samples').value=JSON.stringify(Array(1001).fill({}));
assert.throws(workflowChronologyParams,/1,000/);
''')

    def test_chronology_old_preview_or_error_cannot_enable_new_declaration(self):
        self.js(r'''
chronologySetup(); let release;
api=(name,p)=>new Promise(resolve=>release=()=>resolve(chronologyPreview(p)));
const old=previewWorkflowChronology();
$('#wf-c-final-end').value='2020-10-30'; $('#wf-c-final-end').oninput();
release(); await old;
assert.equal(workflowChronology.preview,null); assert.equal($('#wf-c-declare').disabled,true);
let reject;
api=()=>new Promise((resolve,rej)=>reject=rej);
const obsolete=previewWorkflowChronology(); workflowChronologyUseSource();
api=async(name,p)=>chronologyPreview(p); await previewWorkflowChronology();
reject(Error('old error')); await obsolete;
assert.equal($('#wf-c-declare').disabled,false); assert.doesNotMatch($('#wf-c-message').textContent,/old error/);
$('#wf-c-rule').value='Changed without event'; await declareWorkflowChronology();
assert.equal(workflowChronology.protocolId,null);
''')

    def test_chronology_preview_mismatch_and_exact_refusal_fail_closed(self):
        self.js(r'''
chronologySetup();
api=async(name,p)=>{const v=chronologyPreview(p);v.protocol.metadata.chronology_workflow.source_result_digest='wrong';return v;};
await previewWorkflowChronology(); assert.equal($('#wf-c-declare').disabled,true);
assert.match($('#wf-c-message').textContent,/differs/);
api=async(name,p)=>{const v=chronologyPreview(p);v.protocol.final={start:'2021',end:'2022'};return v;};
await previewWorkflowChronology(); assert.equal(workflowChronology.preview,null);
api=async()=>{throw Error('exact refusal: final window needs 21 sessions');};
await previewWorkflowChronology(); assert.equal($('#wf-c-message').textContent,'exact refusal: final window needs 21 sessions');
select(attempt('rejected','rejected')); workflowChronologyUseSource();
assert.equal(workflowChronology.source.attempt.id,'source');
''')

    def test_chronology_trial_job_payload_busy_and_required_refresh(self):
        self.js(r'''
chronologySetup(); chronologyDeclared();
$('#wf-c-expressions').value=' rank(close)\n\nrank(close)\n<bad>'; $('#wf-c-phase').value='validation';
api=async(name,p)=>{calls.push([name,p]);return {id:'trial-job',state:'running',kind:'chronology-trial'};};
const run=runWorkflowChronologyTrial(); await runWorkflowChronologyTrial(); await run;
assert.deepEqual(calls,[['workflow-chronology-trial',{protocol_id:'protocol',phase:'validation',expressions:['rank(close)','rank(close)','<bad>']}]]);
assert.equal(polled.kind,'chronology-trial'); assert.equal(workflowChronology.stale,true);
assert.equal($('#wf-c-trial').disabled,true); assert.equal($('#wf-p-run').disabled,true);
state.job={state:'completed'}; workflowChronologyJobFinished({...polled,state:'completed'});
assert.equal(workflowChronology.protocolId,'protocol'); assert.equal(workflowChronologyReady(),false);
await runWorkflowChronologyTrial(); assert.equal(calls.length,1);
api=async(name,p)=>{calls.push([name,p]);return chronologyView();};
await refreshWorkflowChronology(); assert.equal(workflowChronologyReady(),true);
assert.equal(calls[1][0],'workflow-chronology-view'); assert.equal(calls.length,2);
''')

    def test_chronology_population_pages_selection_and_freeze_contract(self):
        self.js(r'''
chronologySetup();
const population=Array.from({length:25},(_,i)=>chronologyRef('r'+i,i===1?'failed':'completed'));
chronologyDeclared(chronologyView(population));
get=async(name,p)=>{calls.push([name,p]);const ref=population.find(r=>r.id===p.id);
 return {attempt:{...attempt(p.id,ref.status,p.id==='r0'?'chronology.trial':'lab-evaluation'),result_digest:ref.result_digest},
  result:{expression:'rank(close)',summary:{sharpe:1.23456,start:'2020-01-02',end:'2020-03-31'}}};};
await loadWorkflowChronologyCandidates();
assert.equal(calls.length,12); assert.equal(workflowChronology.candidateInputs.length,12);
assert.equal(workflowChronology.candidateInputs[0].input.disabled,true); assert.equal(workflowChronology.candidateInputs[1].input.disabled,true);
const chosen=workflowChronology.candidateInputs[2].input; chosen.checked=true; chosen.onchange({currentTarget:chosen});
$('#wf-c-reason').value='Reviewed development evidence'; workflowUpdateButtons();
assert.equal(workflowChronologyCanFreeze(),true);
await loadWorkflowChronologyCandidates(); assert.equal(calls.length,24); assert.equal(workflowChronology.candidateInputs.length,12);
assert.equal(workflowChronology.page,1); assert.equal(workflowChronology.chosen.size,1);
$('#wf-c-previous').onclick(); assert.equal(workflowChronology.page,0); assert.equal(calls.length,24);
await loadWorkflowChronologyCandidates(); assert.equal(workflowChronology.page,1); assert.equal(calls.length,24);
await loadWorkflowChronologyCandidates(); assert.equal(workflowChronology.candidateInputs.length,1); assert.equal(calls.length,25);
api=async(name,p)=>{calls.push([name,p]);return {protocol_id:'protocol',selection_id:'selection'};};
await freezeWorkflowChronology();
assert.deepEqual(calls.at(-1),['workflow-chronology-freeze',{protocol_id:'protocol',selected_ids:['r2'],decision_reason:'Reviewed development evidence'}]);
assert.equal(workflowChronology.stale,true); assert.equal($('#wf-c-final').disabled,true);
''')

    def test_chronology_freeze_requires_current_successful_immutable_population(self):
        self.js(r'''
chronologySetup(); const ref=chronologyRef('trial');
chronologyDeclared(chronologyView([ref]));
workflowChronology.records.set('trial',{attempt:{...attempt('trial'),result_digest:ref.result_digest}});
workflowChronology.chosen.add('trial'); $('#wf-c-reason').value='Reviewed';
assert.equal(workflowChronologyCanFreeze(),true);
workflowChronology.view.population.push(chronologyRef('pending','pending'));
assert.equal(workflowChronologyCanFreeze(),false); await freezeWorkflowChronology(); assert.equal(calls.length,0);
workflowChronology.view.population.pop(); workflowChronology.records.get('trial').attempt.result_digest='changed';
assert.equal(workflowChronologyCanFreeze(),false);
workflowChronology.records.get('trial').attempt.result_digest=ref.result_digest;
workflowChronology.records.get('trial').attempt.kind='lab-batch'; assert.equal(workflowChronologyCanFreeze(),false);
''')

    def test_chronology_final_acknowledgment_default_repeat_labels_and_no_numeric_reads(self):
        self.js(r'''
chronologySetup(); const selected=[chronologyRef('evaluated')];
chronologyDeclared();
let exposed=[];
api=async(name,p)=>{calls.push([name,p]);return name==='workflow-chronology-view'
 ? chronologyView(selected,selected,'selection'):{id:'final-job',kind:'chronology-final',state:'running'};};
get=async(name,p)=>{calls.push([name,p]);assert.equal(p.id,'selection');return {attempt:chronologySelection(selected,exposed)};};
await refreshWorkflowChronology();
assert.equal(workflowChronologyCanFinal(),true); assert.equal($('#wf-c-exposed').checked,false);
await runWorkflowChronologyFinal();
assert.deepEqual(calls.at(-1),['workflow-chronology-final',{protocol_id:'protocol',selection_id:'selection',allow_exposed:false}]);
state.job=null; exposed=[{attempt_id:'source',reason:'source_evidence_reaches_final_or_unknown'}];
await refreshWorkflowChronology(); assert.equal(workflowChronologyCanFinal(),false);
assert.match(text($('#wf-c-selection')),/source_evidence_reaches_final_or_unknown/);
const count=calls.length; await runWorkflowChronologyFinal(); assert.equal(calls.length,count);
$('#wf-c-exposed').checked=true; assert.equal(workflowChronologyCanFinal(),true);
await runWorkflowChronologyFinal(); assert.equal(calls.at(-1)[1].allow_exposed,true);
state.job=null; exposed=[];
api=async(name,p)=>{calls.push([name,p]);return chronologyView(selected,selected,'selection',
 [{id:'prior-final',status:'failed',access_label:'first_recorded_access',reused_holdout:false},
 {id:'repeat',status:'cancelled',access_label:'reused_holdout',reused_holdout:true}]);};
await refreshWorkflowChronology();
assert.equal($('#wf-c-exposed').checked,false); assert.equal(workflowChronologyCanFinal(),false);
assert.match(text($('#wf-c-outcomes')),/first_recorded_access.*failed/);
assert.match(text($('#wf-c-outcomes')),/reused_holdout.*cancelled/);
assert.match($('#wf-c-final').textContent,/repeat access/);
assert.equal(calls.filter(([name,p])=>name==='workflow-detail' && p.id!=='selection').length,0);
workflowChronology.view.accesses.push({id:'unfinished',status:'pending'});
$('#wf-c-exposed').checked=true; assert.equal(workflowChronologyCanFinal(),false);
''')

    def test_chronology_masks_block_affected_phases_not_fabricate_purged_runs(self):
        self.js(r'''
chronologySetup(); chronologyDeclared();
workflowChronology.view.sample_split={overlap_check:'explicit_intervals',excluded:[{id:'x',phase:'development'}]};
$('#wf-c-expressions').value='rank(close)';
workflowUpdateButtons(); assert.equal($('#wf-c-trial').disabled,true);
await runWorkflowChronologyTrial(); assert.equal(calls.length,0); assert.match($('#wf-c-message').textContent,/Unsupported mask/);
$('#wf-c-phase').value='validation'; workflowUpdateButtons(); assert.equal($('#wf-c-trial').disabled,false);
const selected=[chronologyRef('trial')];
workflowChronology.selection=chronologySelection(selected); workflowChronology.view.selection_id='selection';
workflowChronology.view.sample_split.excluded.push({id:'y',phase:'final'}); $('#wf-c-exposed').checked=true;
assert.equal(workflowChronologyCanFinal(),false);
await runWorkflowChronologyFinal(); assert.equal(calls.length,0);
renderWorkflowChronologyView();
assert.match(text($('#wf-c-outcomes')),/development: Unsupported mask/);
assert.match(text($('#wf-c-outcomes')),/final: Unsupported mask/);
''')

    def test_chronology_complete_noncontiguous_observations_enable_trials_final_and_selection(self):
        self.js(r'''
chronologySetup(); const selected=[chronologyRef('trial')];
const view=chronologyView(selected); view.sample_split=chronologyCoverage();
chronologyDeclared(view); $('#wf-c-expressions').value='rank(close)';
workflowChronology.records.set('trial',{attempt:{...attempt('trial'),result_digest:'digest-trial'}});
workflowChronology.chosen.add('trial'); $('#wf-c-reason').value='Reviewed selected observations';
workflowUpdateButtons(); assert.equal($('#wf-c-trial').disabled,false);
assert.equal(workflowChronologyCanFreeze(),true);
renderWorkflowChronologyView();
const output=text($('#wf-c-outcomes'));
assert.match(output,/Observation coverage: complete/); assert.match(output,/3\/3 checked.*1 excluded.*2 retained.*ready/);
assert.match(output,/Full-window signal and warmup/);
assert.match(output,/not a continuous investment return or a masked trading policy/);
assert.match(output,/Qlib validation refuses filtered results/);
assert.doesNotMatch(output,/Unsupported mask/);
const p=chronologyPreview(workflowChronologyParams()); p.sample_split=view.sample_split;
renderWorkflowChronologyPreview(p); assert.match(text($('#wf-c-preview')),/observation-filter-v1/);
assert.doesNotMatch(text($('#wf-c-preview')),/Unsupported mask/);
api=async(name,p)=>{calls.push([name,p]);return {id:'trial-job',state:'running',kind:'chronology-trial'};};
await runWorkflowChronologyTrial(); assert.equal(calls.length,1);
assert.equal(calls[0][0],'workflow-chronology-trial');
state.job=null; workflowChronology.stale=false;
workflowChronology.selection=chronologySelection(selected); view.selection_id='selection'; view.selected=selected;
view.sample_split.excluded.push({id:'final-middle',phase:'final'});
view.sample_split.coverage.phases.final.excluded_sessions=['2020-07-02'];
view.sample_split.coverage.phases.final.evaluation_sessions=['2020-07-01','2020-07-03'];
workflowUpdateButtons(); assert.equal($('#wf-c-final').disabled,false);
api=async(name,p)=>{calls.push([name,p]);return {id:'final-job',state:'running',kind:'chronology-final'};};
await runWorkflowChronologyFinal(); assert.equal(calls.at(-1)[0],'workflow-chronology-final');
''')

    def test_chronology_partial_empty_and_not_ready_coverage_refuse(self):
        self.js(r'''
chronologySetup(); const selected=[chronologyRef('trial')];
chronologyDeclared(chronologyView(selected,selected,'selection'));
workflowChronology.selection=chronologySelection(selected); $('#wf-c-expressions').value='rank(close)';
for(const empty of [false,true]) {
 const split=chronologyCoverage('partial');
 if(empty) {split.excluded=[]; for(const row of Object.values(split.coverage.phases)) row.evaluation_sessions=[];}
 workflowChronology.view.sample_split=split;
 for(const phase of ['development','validation']) {
  $('#wf-c-phase').value=phase; assert.throws(workflowChronologyTrialParams,/complete coverage across all phases/);
 }
 assert.equal(workflowChronologyCanFinal(),false);
 await runWorkflowChronologyTrial(); await runWorkflowChronologyFinal();
 assert.equal(calls.length,0); renderWorkflowChronologyView();
 assert.match(text($('#wf-c-outcomes')),/Observation coverage: partial/);
 assert.match(text($('#wf-c-outcomes')),/blocked/);
}
const split=chronologyCoverage(); workflowChronology.view.sample_split=split;
split.coverage.phases.validation.ready=false;
split.coverage.phases.validation.refusal='At least two scored observations must remain';
assert.throws(workflowChronologyTrialParams,/At least two scored observations/);
$('#wf-c-phase').value='development'; assert.doesNotThrow(workflowChronologyTrialParams);
split.coverage.phases.final.ready=false; assert.equal(workflowChronologyCanFinal(),false);
// A ready flag cannot override partial all-phase coverage.
split.coverage.status='partial'; split.coverage.phases.development.ready=true;
assert.throws(workflowChronologyTrialParams,/complete coverage/);
''')

    def test_chronology_no_samples_calendar_and_old_mocks_remain_usable(self):
        self.js(r'''
chronologySetup(); const selected=[chronologyRef('trial')];
chronologyDeclared(chronologyView(selected,selected,'selection'));
workflowChronology.selection=chronologySelection(selected); $('#wf-c-expressions').value='rank(close)';
for(const split of [{excluded:[],overlap_check:'not_available'},chronologyCoverage('not_supplied')]) {
 workflowChronology.view.sample_split=split;
 assert.doesNotThrow(workflowChronologyTrialParams); assert.equal(workflowChronologyCanFinal(),true);
 renderWorkflowChronologyView(); assert.match(text($('#wf-c-outcomes')),/Calendar windows only/);
 assert.doesNotMatch(text($('#wf-c-outcomes')),/Unsupported mask|blocked/);
}
// Explicit policies without coverage cannot be mistaken for ordinary calendar mocks.
workflowChronology.view.sample_split={excluded:[],execution_policy:'observation-filter-v1'};
assert.throws(workflowChronologyTrialParams,/complete coverage/);
assert.equal(workflowChronologyCanFinal(),false);
''')

    def test_chronology_new_external_exposure_requires_fresh_acknowledgment(self):
        self.js(r'''
chronologySetup(); const selected=[chronologyRef('trial')]; chronologyDeclared();
let external=[];
api=async(name,p)=>{calls.push([name,p]); return {...chronologyView(selected,selected,'selection'),external_exposure_reasons:external};};
get=async(name,p)=>{calls.push([name,p]);return {attempt:chronologySelection(selected)};};
await refreshWorkflowChronology(); assert.equal(workflowChronologyCanFinal(),true);
$('#wf-c-exposed').checked=true;
external=[{attempt_id:'outside-trial',reason:'external_evaluation_reaches_final'}];
await refreshWorkflowChronology();
assert.equal($('#wf-c-exposed').checked,false); assert.equal($('#wf-c-final').disabled,true);
assert.match(text($('#wf-c-selection')),/external_evaluation_reaches_final.*Explicit acknowledgment/);
assert.match($('#wf-c-final').textContent,/exposed holdout/);
const count=calls.length; await runWorkflowChronologyFinal(); assert.equal(calls.length,count);
workflowChronology.selection.request.exposure_reasons=[{reason:'frozen_source_exposure'}];
workflowChronology.view.accesses=[{id:'old-access',status:'failed'}];
renderWorkflowChronologyView();
assert.match(text($('#wf-c-selection')),/frozen_source_exposure.*external_evaluation_reaches_final.*prior final access/);
$('#wf-c-exposed').checked=true; workflowUpdateButtons(); assert.equal($('#wf-c-final').disabled,false);
api=async(name,p)=>{calls.push([name,p]);return {id:'final-job',kind:'chronology-final',state:'running'};};
await runWorkflowChronologyFinal(); assert.equal(calls.at(-1)[1].allow_exposed,true);
assert.equal(calls.filter(([name,p])=>name==='workflow-detail' && p.id!=='selection').length,0);
''')

    def test_chronology_frozen_candidate_review_reads_training_results_not_final(self):
        self.js(r'''
chronologySetup(); const selected=[chronologyRef('evaluated')];
chronologyDeclared(chronologyView([...selected,chronologyRef('not-selected')],selected,'selection',
 [{id:'final-access',status:'completed',access_label:'reused_holdout',reused_holdout:true}]));
workflowChronology.selection=chronologySelection(selected);
workflowUI.rows=[{...attempt('evaluated'),display_label:'rank(volume)'}];
renderWorkflowChronologyView(); workflowUpdateButtons();
assert.match(text($('#wf-c-candidates')),/rank\(volume\)/);
assert.equal($('#wf-c-reason-field').hidden,true); assert.equal($('#wf-c-freeze').hidden,true);
assert.equal($('#wf-c-load').disabled,false);
get=async(name,p)=>{calls.push([name,p]);assert.equal(p.id,'evaluated');
 return {attempt:{...attempt('evaluated'),result_digest:'digest-evaluated'},result:{expression:'rank(volume)'}};};
await loadWorkflowChronologyCandidates();
assert.match(text($('#wf-c-candidates')),/rank\(volume\)/);
assert.equal(workflowChronology.candidateInputs[0].input.disabled,true);
assert.equal(workflowChronology.candidateInputs[0].input.checked,true);
assert.equal($('#wf-c-load').disabled,true); assert.equal(calls.length,1);
''')

    def test_chronology_async_view_population_and_source_races_fail_closed(self):
        self.js(r'''
chronologySetup(); chronologyDeclared(); let release;
api=()=>new Promise(resolve=>release=()=>resolve(chronologyView()));
const old=refreshWorkflowChronology(); workflowChronologyUseSource(); release(); await old;
assert.equal(workflowChronology.protocolId,null); assert.equal(workflowChronology.view,null);
chronologyDeclared(chronologyView([chronologyRef('trial')]));
get=(name,p)=>new Promise(resolve=>release=()=>resolve({attempt:{...attempt('trial'),result_digest:'digest-trial'},result:{}}));
const pending=loadWorkflowChronologyCandidates(); workflowChronologyUseSource(); release(); await pending;
assert.equal(workflowChronology.records.size,0); assert.equal(workflowChronology.loading,false);
chronologyDeclared(chronologyView([chronologyRef('trial')]));
get=async()=>({attempt:{...attempt('trial'),result_digest:'different'},result:{}});
await loadWorkflowChronologyCandidates(); assert.equal(workflowChronology.stale,true);
assert.match($('#wf-c-message').textContent,/Population changed/); assert.equal(workflowChronologyCanFreeze(),false);
''')

    def test_chronology_resume_and_main_refresh_keep_protocol_and_pinned_source(self):
        self.js(r'''
chronologySetup(); const plan=chronologyPreview(workflowChronologyParams()).protocol;
const source=workflowChronology.source;
select({...attempt('protocol','completed','chronology.protocol'),request:plan});
get=async(name,p)=>{calls.push([name,p]);return name==='workflow-attempts'?[]:
 p.id==='protocol'?{attempt:{...attempt('protocol','completed','chronology.protocol'),request:plan}}:source;};
await openWorkflowChronology();
assert.equal(workflowChronology.protocolId,'protocol'); assert.equal(workflowChronology.source.attempt.id,'source');
workflowChronologyJobFinished({kind:'chronology-final',state:'failed',receipt:{access_label:'already_exposed_holdout'}});
assert.match($('#wf-c-message').textContent,/failed.*already_exposed_holdout/);
api=async(name,p)=>{calls.push([name,p]);return chronologyView();};
await refreshWorkflow(); await refreshWorkflowChronology();
assert.equal(workflowUI.selected,null); assert.equal(workflowChronology.protocolId,'protocol');
assert.equal(workflowChronology.source.attempt.id,'source'); assert.equal(workflowChronologyReady(),true);
assert.deepEqual(calls.map(c=>c[0]),['workflow-detail','workflow-detail','workflow-attempts','workflow-chronology-view']);
''')

    def test_chronology_failure_and_recovery_force_explicit_refresh(self):
        self.js(r'''
chronologySetup(); chronologyDeclared(); $('#wf-c-expressions').value='rank(close)';
api=async(name,p)=>{calls.push([name,p]);throw Error('exact refusal: source bytes changed');};
await runWorkflowChronologyTrial();
assert.match($('#wf-c-message').textContent,/^exact refusal: source bytes changed/);
assert.equal(workflowChronology.stale,true); assert.equal(workflowChronology.submitting,false);
assert.equal($('#wf-c-trial').disabled,true); assert.equal(calls.length,1);
workflowChronology.stale=false;
await recoverWorkflow(); assert.equal(workflowChronology.stale,true);
assert.equal(workflowChronology.protocolId,'protocol');
''')

    def test_portfolio_result_shows_risk_counts_and_omissions_without_disclosure(self):
        self.js(r'''
select(attempt('portfolio','completed','portfolio-validation'));
workflowUI.detail.result={summary:{portfolio_risk:{counts:{sessions:80,blocked_rebalances:4,native_partial_or_unfilled_orders:3,
 pre_order:{pass:76,blocked:2,missing_inputs:2},post_fill:{pass:77,blocked:1,missing_inputs:2}},
 model_omissions:['no live borrow availability, recalls, broker margin or market impact'],sources:{digest:'secret-long-risk-provenance'}}},
 qualification:{status:'verified_target_reuse_and_reconciliation'}};
renderWorkflowDetail();
const output=text($('#wf-detail'));
assert.match(output,/does not mean limits passed/); assert.match(output,/Blocked rebalances\s+4/);
assert.match(output,/Post-fill violations\s+1/); assert.match(output,/Post-fill missing inputs\s+2/);
assert.match(output,/Pre-order missing inputs\s+2/); assert.match(output,/no live borrow availability/);
assert.match(output,/verified_target_reuse_and_reconciliation/);
assert.doesNotMatch(output,/secret-long-risk-provenance/);
''')


if __name__ == "__main__":
    unittest.main()
