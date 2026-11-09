"""Start and Discover views: page structure, script discipline and pure helpers (offline, no server)."""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import unittest

STATIC = pathlib.Path(__file__).resolve().parents[1] / "qlib_yahoo/webapp/static"
HTML = (STATIC / "index.html").read_text(encoding="utf-8")
APP = (STATIC / "app.js").read_text(encoding="utf-8")
HOME = (STATIC / "home.js").read_text(encoding="utf-8")
DISCOVER = (STATIC / "discover.js").read_text(encoding="utf-8")


class StructureTests(unittest.TestCase):
    def test_scripts_styles_and_views_are_wired(self):
        for name in ("home.js", "discover.js", "home.css", "discover.css"):
            self.assertIn(f"/static/{name}?v=", HTML)
        self.assertLess(HTML.index("/static/workflow.js"), HTML.index("/static/home.js"))
        self.assertLess(HTML.index("/static/discover.js"), HTML.index("/static/app.js"))
        self.assertIn('"home", "discover"', APP)
        self.assertIn('initializeHome()', APP)
        self.assertIn('initializeDiscover()', APP)
        boot = APP.split("async function boot() {", 1)[1].split("function showBootFailure", 1)[0]
        self.assertIn('setView("lab")', boot)
        self.assertIn('setView("home")', boot)
        finished = APP[APP.index("async function finished(job)"):APP.index("// ================================================================= Alpha Lab")]
        self.assertIn('"demo-data"', finished)
        self.assertIn('"5": () => setView("discover")', APP)
        self.assertIn("fillLabUniverses()", APP)
        self.assertIn("Rewrite as -1 * (…)", APP)
        for view in ("home", "discover"):
            self.assertNotIn(f'data-view="{view}"', HTML, "the new views are built by their scripts, not the page")

    def test_new_scripts_keep_the_console_transport_discipline(self):
        for name, source in (("home.js", HOME), ("discover.js", DISCOVER)):
            for forbidden in ("innerHTML", "fetch(", "XMLHttpRequest", "import(", "eval(", "localStorage"):
                self.assertNotIn(forbidden, source, f"{name} must use the shared api()/el() helpers; found {forbidden}")
            self.assertTrue(source.startswith('"use strict";'), name)
        self.assertIn("api(\"lab-finalize\", { id: row.id, confirm: true })", DISCOVER)
        self.assertIn("danger: true", DISCOVER, "finalize asks with a destructive-style confirmation")
        self.assertNotIn("allow_reuse", DISCOVER, "the console never requests a labelled holdout reuse")

    def test_server_routes_exist_for_every_call_the_scripts_make(self):
        import re
        server = (STATIC.parents[0] / "server.py").read_text(encoding="utf-8")
        routes = set(re.findall(r'\("(?:GET|POST)", "([a-z-]+)"\)', server))
        calls = set(re.findall(r'(?:api|get)\("([a-z-]+)"', HOME + DISCOVER))
        missing = sorted(call for call in calls if call not in routes)
        self.assertEqual(missing, [], "scripts call endpoints the server does not serve")


@unittest.skipUnless(shutil.which("node"), "Node needed for the JS helpers")
class HelperTests(unittest.TestCase):
    def js(self, body: str) -> dict:
        harness = r'''
const assert = require('node:assert/strict');
const nodes = new Map();
function el(tag, attrs = {}, ...children) {
  const n = { tag, ...attrs, textContent: String(attrs.text || ''), children: children.flat(Infinity).filter((v) => v != null),
    value: String(attrs.value ?? ''), append(...v) { this.children.push(...v); }, prepend(...v) { this.children.unshift(...v); },
    addEventListener() {}, classList: { toggle() {}, contains() { return false; } }, insertBefore() {}, parentElement: { insertBefore() {} } };
  if (attrs.id) nodes.set('#' + attrs.id, n);
  return n;
}
function $(id) { if (!nodes.has(id)) nodes.set(id, el('div')); return nodes.get(id); }
function $$() { return []; }
function setChildren(n, ...v) { n.children = v.flat(Infinity).filter((x) => x != null); return n; }
function svgEl(tag) { return { tag, append() {} }; }
function text(n) { return typeof n === 'object' && n ? n.textContent + ' ' + (n.children || []).map(text).join(' ') : String(n ?? ''); }
const finite = (v) => typeof v === 'number' && isFinite(v);
const fmt = { int: (v) => finite(v) ? String(Math.round(v)) : '—', num: (v, d = 2) => finite(v) ? Number(v).toFixed(d) : '—', pct: (v, d = 1) => finite(v) ? (v * 100).toFixed(d) + '%' : '—' };
const home = (p) => p; const shortTime = (s) => s || ''; const css = () => '#000';
const state = { view: 'discover', datasets: [], lab: { rows: [], dataset: null, datasetPath: null }, dataset: null, job: null, overview: {} };
const priceEligible = (r) => !!r && r.status === 'ready' && r.price_eligible === true;
const defaultPriceRow = () => state.datasets.find((r) => priceEligible(r) && r.is_default);
const datasetTitle = (d) => d.name;
let get = async () => ({}); let api = async () => ({}); const toasts = [];
function toast(title, message) { toasts.push([title, message]); }
function setView() {} function persist() {} function dialog() { return Promise.resolve(true); } function barChart() {} function drawWhenSized(c, draw) { draw(); }
function loadDataset() {} function onLabInput() {} function startPolling() {} function runLab() {} function startRefresh() {} function showRef() {}
const document = { body: {} };
'''
        source = HOME + "\n" + DISCOVER
        script = harness + source + "\n" + body
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout.strip().splitlines()[-1])

    def test_plan_payload_follows_the_domain_and_task(self):
        result = self.js("""
          discoverUI.catalog = { ml_features: [{name: 'ret_5', expression: 'x'}, {name: 'vol_20', expression: 'y'}], presets: {alpha: {}, ml: {}}, defaults: {cost_bps: 5, min_sharpe: 1} };
          state.datasets = [{ path: '/d/prices', name: 'prices', status: 'ready', price_eligible: true, is_default: true, universes: { top12: { size: 12 } } }];
          discoverUI.domain = 'alpha'; discoverUI.form = { domain: 'alpha', ...discoverFormDefaults('alpha'), name: 'A', seeds: '-rank(close)' };
          const alpha = discoverFormPayload();
          discoverUI.domain = 'ml'; discoverUI.form = { domain: 'ml', ...discoverFormDefaults('ml') };
          const panel = discoverFormPayload();
          discoverUI.form.task = 'builtin:friedman1'; discoverUI.form.rows = 1234;
          const builtin = discoverFormPayload();
          discoverUI.form.task = 'csv'; discoverUI.form.table = '/t.csv'; discoverUI.form.target = 'y';
          const csv = discoverFormPayload();
          console.log(JSON.stringify({ alpha, panel, builtin, csv }));
        """)
        self.assertEqual(result["alpha"]["seeds"], "-rank(close)")
        self.assertEqual(result["alpha"]["dataset"], "/d/prices")
        self.assertNotIn("task", result["alpha"])
        self.assertEqual(result["panel"]["task"], "panel", "with prices available, factor analysis is the ML default")
        self.assertEqual(result["panel"]["features"], "default")
        self.assertEqual(result["builtin"]["rows"], 1234)
        self.assertNotIn("dataset", result["builtin"])
        self.assertEqual(result["csv"]["target"], "y")

    def test_stage_and_decision_labels(self):
        result = self.js("""
          const ml = discoverStages('ml').map((s) => s[2]);
          const alpha = discoverStages('alpha').map((s) => s[2]);
          const tag = discoverTag('promote');
          const pending = discoverTag(null);
          console.log(JSON.stringify({ ml, alpha, tag: tag.textContent, cls: tag.class, pending: pending.textContent,
            stage: discoverStage({ round: 3, promotions: 0, holdout_accessed: false }), done: discoverStage({ round: 3, promotions: 2, holdout_accessed: true }) }));
        """)
        self.assertEqual(len(result["ml"]), 7)
        self.assertNotEqual(result["ml"][0], result["alpha"][0])
        self.assertEqual(result["tag"], "Promoted")
        self.assertIn("good", result["cls"])
        self.assertEqual(result["pending"], "Pending")
        self.assertEqual(result["stage"], "review")
        self.assertEqual(result["done"], "finalize")

    def test_start_checklist_reflects_data_library_and_campaigns(self):
        result = self.js("""
          state.view = 'home';
          state.datasets = [{ path: '/d/p', name: 'p', status: 'ready', price_eligible: true, is_default: true, universes: {} }];
          state.lab.rows = [{}, {}];
          homeUI.campaigns = [];
          initializeHome();
          renderHomeChecklist(homeStatus());
          const rows = $('#home-checklist').children.map((c) => [c.class, text(c).trim()]);
          console.log(JSON.stringify(rows));
        """)
        self.assertEqual(len(result), 3)
        self.assertIn("done", result[0][0])
        self.assertIn("done", result[1][0])
        self.assertNotIn("done", result[2][0])
        self.assertIn("2 in the library", result[1][1])


if __name__ == "__main__":
    unittest.main()
