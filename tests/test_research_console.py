"""Research view front-end logic in the packaged JS: finals, finalize, hand-off and the openai provider.

Runs the actual static files in the shared Node harness (core.js, app.js and
research.js) with stubbed DOM nodes and API calls; no browser or server.
"""

import json
import shutil
import subprocess
import unittest

from tests.test_qlib_recovery_console import HARNESS, ROOT

NODE = shutil.which("node")
LOAD_RESEARCH = HARNESS.replace("for (const name of ['core.js', 'app.js'])", "for (const name of ['core.js', 'app.js', 'research.js'])")
assert LOAD_RESEARCH != HARNESS, "harness file list changed; update the research console test"

SETUP = """
  const fakeNode = () => ({value: '', checked: false, textContent: '', hidden: false, disabled: false, options: [],
    dataset: {}, style: {}, parentElement: {hidden: false}, children: [],
    classList: {contains() {return false;}, add() {}, toggle() {}},
    replaceChildren(...children) {this.children = children;},
    append(...children) {this.children = [...(this.children || []), ...children];},
    reset() {}, reportValidity() {return true;}});
  const selectNode = document.querySelector;
  document.querySelector = (selector) => {
    if (/^#ar-/.test(selector)) {
      if (!nodes.has(selector)) nodes.set(selector, fakeNode());
      return nodes.get(selector);
    }
    const node = selectNode(selector);
    if (!node.append) node.append = (...children) => {node.children = [...(node.children || []), ...children];};
    return node;
  };
  window.confirm = (message) => {calls.push({name: 'confirm', message}); return input.confirm !== false;};
  setView = (view) => calls.push({name: 'setView', view});
  onLabInput = () => calls.push({name: 'onLabInput'});
  refreshResearch = async () => calls.push({name: 'refreshResearch'});
  renderResearch = () => calls.push({name: 'renderResearch'});
"""


@unittest.skipUnless(NODE, "Node.js is required for the console JS harness")
class ResearchConsoleTests(unittest.TestCase):
    def js(self, code, **options):
        result = subprocess.run([NODE, "-e", LOAD_RESEARCH], input=json.dumps({"code": SETUP + code, **options}),
                                text=True, encoding="utf-8", capture_output=True, cwd=ROOT, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_finalize_button_reflects_holdout_runtime_worker_and_promotion(self):
        result = self.js("""
          const base = {holdout: {reserved: true, accessed: false, runtime_available: true, start: '2024-10-01'},
                        hypotheses: [{latest_review: {decision: 'promote'}}, {latest_review: {decision: 'reject'}}]};
          const idle = {running: false};
          const cases = {
            ready: researchFinalizeButton(base, idle),
            noHoldout: researchFinalizeButton({...base, holdout: {...base.holdout, reserved: false}}, idle),
            noRuntime: researchFinalizeButton({...base, holdout: {...base.holdout, runtime_available: false}}, idle),
            running: researchFinalizeButton(base, {running: true}),
            nothingPromoted: researchFinalizeButton({...base, hypotheses: [{latest_review: {decision: 'refine'}}]}, idle),
            accessed: researchFinalizeButton({...base, holdout: {...base.holdout, accessed: true}}, idle),
          };
          return Object.fromEntries(Object.entries(cases).map(([k, b]) => [k, {text: b.text, disabled: b.disabled, title: b.title}]));
        """)
        self.assertEqual(result["ready"], {"text": "Finalize on holdout", "disabled": False,
                                          "title": "Scores every promoted idea once on the reserved holdout."})
        self.assertTrue(result["noHoldout"]["disabled"])
        self.assertIn("reserved no holdout", result["noHoldout"]["title"])
        self.assertTrue(result["noRuntime"]["disabled"])
        self.assertIn("not on this machine", result["noRuntime"]["title"])
        self.assertTrue(result["running"]["disabled"])
        self.assertTrue(result["nothingPromoted"]["disabled"])
        self.assertIn("No promoted ideas", result["nothingPromoted"]["title"])
        self.assertEqual(result["accessed"]["text"], "Finalize again (reuse)")
        self.assertFalse(result["accessed"]["disabled"])
        self.assertIn("never confirmed", result["accessed"]["title"])

    def test_finalize_confirms_records_reuse_explicitly_and_respects_cancel(self):
        code = """
          researchUI.detail = {campaign: {id: 'campaign_x'}, hypotheses: [],
                               holdout: {reserved: true, accessed: false, runtime_available: true, start: '2024-10-01'}};
          researchUI.overview = {worker: {running: false}};
          api = async (name, data) => {calls.push({name, data}); return {};};
          await finalizeResearch();
          researchUI.detail.holdout.accessed = true;
          await finalizeResearch();
          return {calls, follow: researchUI.followWorker};
        """
        result = self.js(code)
        confirms = [c["message"] for c in result["calls"] if c["name"] == "confirm"]
        self.assertEqual(len(confirms), 2)
        self.assertIn("cannot be undone", confirms[0])
        self.assertIn("2024-10-01", confirms[0])
        self.assertIn("labelled reused", confirms[1])
        self.assertIn("never be confirmed", confirms[1])
        posts = [c["data"] for c in result["calls"] if c["name"] == "research-finalize"]
        self.assertEqual(posts, [{"id": "campaign_x", "allow_reuse": False}, {"id": "campaign_x", "allow_reuse": True}])
        self.assertTrue(result["follow"])
        self.assertEqual(sum(1 for c in result["calls"] if c["name"] == "refreshResearch"), 2)
        cancelled = self.js(code, confirm=False)
        self.assertFalse(any(c["name"] == "research-finalize" for c in cancelled["calls"]))

    def test_handoff_fills_the_lab_editor_or_reports_nothing(self):
        code = """
          researchUI.detail = {campaign: {id: 'campaign_x'}, hypotheses: [], holdout: {}};
          api = async (name, data) => {calls.push({name, data}); return input.export;};
          $('#lab-text').value = 'untouched';
          await exportResearchBatch(input.decision || 'promote');
          return {calls, lab: $('#lab-text').value};
        """
        empty = self.js(code, export={"count": 0})
        self.assertEqual(empty["lab"], "untouched")
        self.assertEqual([c["data"] for c in empty["calls"] if c["name"] == "research-export"],
                         [{"id": "campaign_x", "decision": "promote"}])
        self.assertTrue(any(c["name"] == "toast" and c["args"][0] == "Nothing to hand off" for c in empty["calls"]))
        self.assertFalse(any(c["name"] == "setView" for c in empty["calls"]))
        filled = self.js(code, export={"count": 2, "text": "# hand-off\nrank(close)\n-rank(close)\n"}, decision="confirmed")
        self.assertEqual(filled["lab"], "# hand-off\nrank(close)\n-rank(close)\n")
        names = [c["name"] for c in filled["calls"]]
        self.assertIn("onLabInput", names)
        self.assertEqual([c["view"] for c in filled["calls"] if c["name"] == "setView"], ["lab"])
        self.assertEqual([c["data"]["decision"] for c in filled["calls"] if c["name"] == "research-export"], ["confirmed"])
        self.assertTrue(any(c["name"] == "toast" and c["args"][0] == "2 expressions opened in Alpha Lab" for c in filled["calls"]))

    def test_ideas_table_and_detail_show_the_holdout_final(self):
        result = self.js("""
          const final = {id: 'final_1', decision: 'confirmed', reused: false, access_sequence: 1,
                         reason: 'Reserved holdout agreed with validation after costs; evidence for deeper research only.',
                         result: {metrics: {holdout: {days: 60, sharpe: 0.9, returns: 0.05, drawdown: 0.02, turnover: 0.3}}}};
          researchUI.detail = {campaign: {id: 'c', demo: false, counts: {}, limits: {}}, experiments: [], reviews: [], sources: [], events: [],
            finals: [final], checkpoint: {}, holdout: {reserved: true, accessed: true, runtime_available: true},
            hypotheses: [{id: 'h1', title: 'Idea one', expression: 'rank(close)', created_at: '2026-01-01', experiments: [], evidence: [],
                          result: {status: 'ok', metrics: {development: {sharpe: 1.1}, validation: {sharpe: 1.2, returns: 0.1, days: 100, drawdown: 0.05, turnover: 0.2}}},
                          latest_review: {decision: 'promote', reason: 'All gates passed'}, final},
                         {id: 'h2', title: 'Idea two', expression: '-rank(close)', created_at: '2026-01-02', experiments: [], evidence: [],
                          result: {status: 'ok', metrics: {development: {sharpe: -1}, validation: {sharpe: -1.2, returns: -0.1, days: 100}}},
                          latest_review: {decision: 'reject', reason: 'Nonpositive'}, final: null}]};
          researchUI.overview = {worker: {running: false}};
          researchUI.tab = 'ideas';
          renderResearchContent();
          const texts = [];
          const walk = (n) => {if (!n || typeof n !== 'object') return; if (typeof n.text === 'string') texts.push(n.text); (n.children || []).forEach(walk);};
          ($('#ar-content').children || []).forEach(walk);
          return texts;
        """)
        self.assertIn("Holdout", result)  # table header
        self.assertIn("Confirmed on holdout", result)  # the promoted idea's final cell
        self.assertIn("—", result)  # the rejected idea has no final
        self.assertTrue(any(t.startswith("Holdout final · Confirmed on holdout: Reserved holdout agreed") for t in result))
        self.assertIn("holdout", result)  # metrics row label in the idea detail

    def test_openai_provider_round_trips_through_campaign_save(self):
        result = self.js("""
          const fields = {name: 'OpenAI campaign', dataset: '/data/demo', universe: 'top8', 'dev-start': '2023-06-01', 'dev-end': '2023-12-29',
            'val-start': '2024-01-01', 'val-end': '2024-09-30', holdout: '2024-10-01', cost: '5', warmup: '100', sharpe: '0.5',
            provider: 'openai', model: 'gpt-fixture', queries: 'all:momentum', days: '120', drawdown: '0.3', turnover: '0.7',
            improvement: '0.05', evaluations: '30', family: '6', depth: '0', calls: '30', daily: '10', focus: 'Why?'};
          for (const [key, value] of Object.entries(fields)) $(`#ar-${key}`).value = value;
          api = async (name, data) => {calls.push({name, data}); return {id: 'config_new', ...data};};
          const saved = await saveResearchCampaign();
          return {saved, calls: calls.filter((c) => c.name === 'research-save'), configId: researchUI.configId};
        """)
        self.assertEqual(result["configId"], "config_new")
        payload = result["calls"][0]["data"]
        self.assertEqual(payload["model"], {"provider": "openai", "model": "gpt-fixture", "research_focus": "Why?"})
        self.assertEqual(payload["backend"]["holdout_start"], "2024-10-01")
        self.assertEqual(payload["policy"]["min_sharpe"], 0.5)
        self.assertNotIn("base_url", json.dumps(payload))  # endpoints are never supplied by the browser


if __name__ == "__main__":
    unittest.main()
