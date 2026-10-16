"""Bounded recovery contracts: real JS helpers in Node, no browser or price data."""

import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "qlib_yahoo/webapp/static"
NODE = shutil.which("node")
HARNESS = r"""
const fs = require('node:fs'), vm = require('node:vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const html = fs.readFileSync('qlib_yahoo/webapp/static/index.html', 'utf8');
const ids = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]));
const timers = [], calls = [], nodes = new Map();
const classes = () => {const s = new Set(); return {
  contains: k => s.has(k), add: k => s.add(k),
  toggle(k, on) {on = on === undefined ? !s.has(k) : on; on ? s.add(k) : s.delete(k);}
};};
const context = vm.createContext({input, timers, calls, nodes,
  navigator: {platform: 'Win32'}, MutationObserver: class {observe() {}},
  document: {addEventListener() {}}, window: {addEventListener() {}},
  clearTimeout() {}, setTimeout(fn) {timers.push(fn); return timers.length;},
  requestAnimationFrame(fn) {fn();}, getComputedStyle: () => ({getPropertyValue: () => ''})
});
for (const name of ['core.js', 'app.js']) {
  let source = fs.readFileSync('qlib_yahoo/webapp/static/' + name, 'utf8');
  if (name === 'app.js') {
    const footer = /\nboot\(\)\.catch\(showBootFailure\);\s*$/;
    if (!footer.test(source)) throw new Error('Unexpected boot footer');
    source = source.replace(footer, '\n');
  }
  vm.runInContext(source, context, {filename: name});
}
context.document.body = {dataset: {}, classList: classes()};
context.document.documentElement = {style: {setProperty() {}}};
context.document.querySelectorAll = () => [];
context.document.querySelector = selector => {
  const id = selector.match(/^#([\w-]+)/);
  if (id && !ids.has(id[1])) throw new Error('Missing HTML ID: ' + id[1]);
  if (!nodes.has(selector)) nodes.set(selector, {value: '', checked: false,
    textContent: '', hidden: false, options: [], classList: classes(),
    replaceChildren(...children) {this.children = children;}});
  return nodes.get(selector);
};
vm.runInContext(`
  el = (tag, attrs = {}, ...children) => ({tag, ...attrs, children: children.flat(Infinity)});
  toast = (...args) => calls.push({name: 'toast', args});
  api = async (name, data) => {
    calls.push({name, data});
    if (name === 'dataset') {
      if (data.path === input.failPath) throw new Error('Snapshot disappeared');
      return {path: data.path, calendar: [], universes: {}};
    }
    if (name === 'alpha-check') return {lines: [{line: 1, ok: !input.badSyntax, message: 'Invalid FASTEXPR'}]};
    return {};
  };
`, context);
vm.runInContext('(async () => {' + input.code + '})()', context)
  .then(result => process.stdout.write(JSON.stringify(result)))
  .catch(error => {console.error(error.stack); process.exitCode = 1;});
"""


class RecoveryMarkupTests(unittest.TestCase):
    def test_notices_share_bounded_row_above_flexible_active_view(self):
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        self.assertRegex(html, r'<main class="center" id="center">\s*'
                         r'<div class="console-notices" id="console-notices">\s*'
                         r'<div[^>]+id="boot-error"[^>]*></div>\s*'
                         r'<div[^>]+id="recovery-note"[^>]*></div>\s*</div>\s*'
                         r'<div class="cview"')
        css = (STATIC / "app.css").read_text(encoding="utf-8")

        def rule(selector):
            match = re.search(r'(?m)^' + re.escape(selector) + r'\s*\{([^}]+)\}', css)
            self.assertIsNotNone(match, selector)
            return match.group(1)

        self.assertIn('grid-template-rows: auto minmax(0, 1fr) 1px var(--bottom-h)', rule('.center'))
        self.assertIn('grid-template-rows: auto minmax(0, 1fr) 0 0', rule('.center.no-bottom'))
        notices = rule('.console-notices')
        self.assertIn('max-height: 25vh', notices)
        self.assertIn('overflow-y: auto', notices)
        self.assertIn('min-height: 0', notices)
        self.assertNotRegex(notices, r'(?:display:\s*none|visibility:\s*hidden|overflow(?:-y)?:\s*hidden)')
        self.assertIn('display: flex', rule('.cview.active'))
        self.assertIn('min-height: 0', rule('.cview.active'))

    def test_required_ids_and_toolbar_selector_outside_inspector(self):
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        ids = re.findall(r'\bid="([^"]+)"', html)
        required = {"tb-lab-dataset", "tb-lab-dataset-name", "tb-dataset", "tb-dataset-name",
                    "lab-text", "alpha", "lab-run", "run-2", "recovery-note", "boot-error",
                    "rh-compatibility", "use-settings", "report-btn", "sb-version"}
        self.assertTrue(required.issubset(ids))
        self.assertEqual(len(ids), len(set(ids)))
        toolbar = html[html.index('<header class="toolbar">'):html.index('</header>')]
        self.assertIn('id="tb-lab-dataset"', toolbar)
        self.assertIn('Trading prices', toolbar)
        css = (STATIC / "app.css").read_text(encoding="utf-8")
        self.assertIn('body:not([data-view=lab]) .lab-only', css)
        for rule in re.findall(r'([^{}]+)\{[^{}]*\}', css):
            self.assertFalse('no-inspector' in rule and ('lab-only' in rule or 'tb-lab-dataset' in rule))

    def test_lab_boot_and_no_eager_run_detail_or_chart_calls(self):
        source = (STATIC / "app.js").read_text(encoding="utf-8")
        boot = source.split('async function boot() {', 1)[1].split('function showBootFailure', 1)[0]
        self.assertIn('setView("lab")', boot)
        self.assertIn('state.run = remembered ? remembered.path : null', boot)
        self.assertNotRegex(boot, r'\b(openRun|renderRun|renderOverview|renderLabCharts)\s*\(')
        self.assertNotRegex(boot, r'get\("(?:run|signal|alpha-result)"')
        self.assertRegex(source, r'\nboot\(\)\.catch\(showBootFailure\);\s*$')
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        self.assertIn('<body data-view="lab">', html)
        for group in ('xview', 'cview', 'iview'):
            self.assertIn(f'class="{group} active" data-view="lab"', html)


@unittest.skipUnless(NODE, "Node.js is needed to execute the frontend helpers")
class RecoveryHelperTests(unittest.TestCase):
    def js(self, code, **options):
        result = subprocess.run([NODE, "-e", HARNESS], input=json.dumps({"code": code, **options}),
                                text=True, encoding="utf-8", capture_output=True, cwd=ROOT, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_malformed_preferences_and_original_expression_preserved(self):
        result = self.js("""
          const malformed = [null, [], 3, true, 'null', '[]', '{broken', '"text"'];
          const parsed = malformed.map(preferenceObject);
          state.settings = {alpha: 'rank(ts_mean(close, 5))', lab: 'null', layout: '[]'};
          restoreSettings(state.settings);
          return {parsed, valid: preferenceObject('{"dataset":"saved"}'),
                  alpha: $('#alpha').value, lab: state.settings.lab};
        """)
        self.assertEqual(result["parsed"], [{}] * 8)
        self.assertEqual(result["valid"], {"dataset": "saved"})
        self.assertEqual(result["alpha"], "rank(ts_mean(close, 5))")
        self.assertEqual(result["lab"], {})

    def test_eligibility_default_and_packed_labels_use_metadata(self):
        result = self.js("""
          const ordinary = {path: 'plain', name: 'stooq-top3000', status: 'ready', price_eligible: true,
                            storage_kind: 'qlib-binary-tree', data_role: 'prices', fields: ['open', 'close']};
          const packed = {...ordinary, path: 'no-suffix', storage_kind: 'qlib-zip-store-v1', is_default: true};
          state.datasets = [{...ordinary, path: 'sec-packed', price_eligible: false, is_default: true}, ordinary, packed];
          return {eligible: [null, {...ordinary, price_eligible: 'true'}, {...ordinary, status: 'failed'}, packed].map(priceEligible),
                  selected: defaultPriceRow().path, titles: [datasetTitle(ordinary), datasetTitle(packed)], metadata: datasetMetadata(packed)};
        """)
        self.assertEqual(result["eligible"], [False, False, False, True])
        self.assertEqual(result["selected"], "no-suffix")
        self.assertNotEqual(*result["titles"])
        self.assertIn("recommended", result["metadata"])

    def test_saved_selections_switch_and_persist_independently(self):
        result = self.js("""
          state.datasets = ['a', 'b', 'c'].map(path => ({path, name: path, status: 'ready', price_eligible: true, is_default: path === 'c'}));
          state.settings = {dataset: 'a', lab: {dataset: 'b'}};
          await restorePriceSelections();
          const saved = [state.datasetPath, state.lab.datasetPath];
          await loadDataset('c', true, 'lab');
          const afterLab = [formValues().dataset, labValues().dataset];
          await loadDataset('b', true);
          state.restoring = false; persist(); timers.pop()();
          const settings = calls.find(c => c.name === 'settings').data;
          return {saved, afterLab, requests: [formValues().dataset, labValues().dataset],
                  persisted: [settings.dataset, JSON.parse(settings.lab).dataset]};
        """)
        self.assertEqual(result["saved"], ["a", "b"])
        self.assertEqual(result["afterLab"], ["a", "c"])
        self.assertEqual(result["requests"], ["b", "c"])
        self.assertEqual(result["persisted"], ["b", "c"])

    def test_sec_recovery_offers_saved_prices_without_translation(self):
        result = self.js("""
          state.datasets = [{path:'sec', name:'sec', status:'ready', price_eligible:false, compatibility_note:'Unsupported SEC pilot'},
                            {path:'prices', name:'prices', status:'ready', price_eligible:true}];
          state.settings = {dataset:'sec', price_dataset:'prices', lab:{dataset:'prices'}};
          $('#alpha').value = 'rank(close)';
          await restorePriceSelections();
          const before = [state.datasetPath, state.lab.datasetPath];
          const button = $('#recovery-note').children.find(n => n.tag === 'button');
          await button.onclick();
          return {before, after:[state.datasetPath, state.lab.datasetPath], alpha:$('#alpha').value, notice:state.recovery.research};
        """)
        self.assertEqual(result["before"], [None, "prices"])
        self.assertEqual(result["after"], ["prices", "prices"])
        self.assertEqual(result["alpha"], "rank(close)")
        self.assertIn("Switched backtest prices", result["notice"])

    def test_unrelated_layout_save_preserves_unresolved_validation_choice(self):
        for eligible in (False, True):
            with self.subTest(saved_dataset_load_fails=eligible):
                result = self.js("""
                  state.datasets = ['invalid', 'prices', 'gone'].map(path => ({path, name:path,
                    status:'ready', price_eligible:path !== 'invalid' || input.eligible, is_default:path === 'prices'}));
                  state.settings = {dataset:'invalid', price_dataset:'prices', alpha:'rank(close)', lab:{dataset:'prices'}};
                  restoreSettings(state.settings);
                  await restorePriceSelections();
                  const unselected = state.datasetPath;
                  state.restoring = false;
                  const saved = () => calls.filter(c => c.name === 'settings').at(-1).data;
                  toggleInspector(false); timers.pop()();
                  const unresolved = saved();
                  await loadDataset('prices', true); timers.pop()();
                  const accepted = saved();
                  input.failPath = 'gone';
                  await loadDataset('gone', true);
                  toggleInspector(true); timers.pop()();
                  return {unselected, unresolved, accepted, failed:saved(), selected:state.datasetPath};
                """, eligible=eligible, failPath="invalid")
                self.assertIsNone(result["unselected"])
                self.assertEqual(result["unresolved"]["dataset"], "invalid")
                self.assertTrue(json.loads(result["unresolved"]["layout"])["inspectorHidden"])
                self.assertEqual(result["accepted"]["dataset"], "prices")
                self.assertIsNone(result["selected"])
                self.assertEqual(result["failed"]["dataset"], "prices")
                for phase in ("unresolved", "accepted", "failed"):
                    self.assertEqual(result[phase]["alpha"], "rank(close)")
                    self.assertEqual(json.loads(result[phase]["lab"])["dataset"], "prices")

    def test_stale_lab_selection_uses_only_explicit_eligible_default(self):
        result = self.js("""
          state.settings = {dataset:'stale', lab:{dataset:'stale'}};
          state.datasets = [{path:'looks-packed', name:'looks-packed', status:'ready', price_eligible:true}];
          await restorePriceSelections();
          const none = state.lab.datasetPath, message = state.recovery.lab;
          state.datasets[0].is_default = true;
          await restorePriceSelections();
          return {none, message, explicit:state.lab.datasetPath};
        """)
        self.assertIsNone(result["none"])
        self.assertIn("No eligible default", result["message"])
        self.assertEqual(result["explicit"], "looks-packed")

    def test_disappeared_snapshot_does_not_load_backend_default(self):
        result = self.js("""
          state.datasets = [{path:'gone', name:'gone', status:'ready', price_eligible:true}];
          await loadDataset('gone', true, 'lab');
          return {selected:state.lab.datasetPath, notice:state.recovery.lab, calls};
        """, failPath="gone")
        self.assertIsNone(result["selected"])
        self.assertIn("no replacement", result["notice"])
        self.assertEqual([c["data"] for c in result["calls"] if c["name"] == "dataset"], [{"path": "gone"}])

    def test_unsupported_runs_and_legacy_engine_are_identified(self):
        result = self.js("""
          return {reuse:[{}, {read_only:true}, {reports_supported:false}, null].map(runReusable),
                  old:labEngineNote({engine:'alphalab-1'}), corrected:labEngineNote({engine:'alphalab-2'})};
        """)
        self.assertEqual(result["reuse"], [True, False, False, False])
        self.assertIn("not been recomputed", result["old"])
        self.assertIn("Corrected engine", result["corrected"])

    def test_price_check_dataset_is_optional_and_syntax_rejection_visible(self):
        result = self.js("""
          $('#lab-text').value = 'Ref($close, 5)';
          const ok = await checkLab();
          state.lab.datasetPath = 'lab-prices'; await checkLab();
          return {ok, status:$('#lab-status').textContent, calls};
        """, badSyntax=True)
        self.assertFalse(result["ok"])
        self.assertIn("Invalid FASTEXPR", result["status"])
        checks = [c["data"] for c in result["calls"] if c["name"] == "alpha-check"]
        self.assertNotIn("dataset", checks[0])
        self.assertEqual(checks[1]["dataset"], "lab-prices")
