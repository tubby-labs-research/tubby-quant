"""Execute the workbench JavaScript with Node; no browser, installs or saved-run writes."""

import json
import shutil
import subprocess
import unittest
from pathlib import Path

from qlib_yahoo.metrics import portfolio_metrics


ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")

# Load the production scripts, suppress only application boot, and replace DOM
# drawing with collectors. Calculations, window slicing, render callbacks,
# formatting and caption generation are the real workbench functions.
HARNESS = r"""
const fs = require('node:fs'), vm = require('node:vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const context = vm.createContext({
  navigator: {platform: 'Win32'},
  MutationObserver: class {observe() {}},
  document: {addEventListener() {}}, window: {addEventListener() {}},
  input,
});
for (const name of ['core.js', 'analytics.js', 'app.js']) {
  let source = fs.readFileSync('qlib_yahoo/webapp/static/' + name, 'utf8');
  if (name === 'app.js') {
    const footer = /\nboot\(\)\.catch\(showBootFailure\);\s*$/;
    if (!footer.test(source)) throw new Error('Application boot footer changed; update the fixture');
    source = source.replace(footer, '\n');
  }
  vm.runInContext(source, context, {filename: name});
}
const result = vm.runInContext(`
  const nodes = new Map(), charts = {}, kpis = {};
  document.querySelector = (selector) => {
    if (!nodes.has(selector)) nodes.set(selector, {selector, classList: {toggle() {}}, replaceChildren(...children) {this.children = children;}});
    return nodes.get(selector);
  };
  el = (tag, attrs = {}, ...children) => ({tag, ...attrs, children: children.flat(Infinity)});
  icon = () => null;
  kpiTile = (title, value) => {kpis[title] = value; return {title, value};};
  drawWhenSized = (_, draw) => draw();
  crosshairGroup = () => ({add() {}});
  lineChart = (node, cfg) => {charts[node.selector] = cfg; return null;};
  rangeNavigator = () => {};
  monthlyHeatmap = () => {};
  barChart = () => {};
  state.runDetail = {status: 'completed', mode: 'signed', series: input.series};
  state.window = input.window || null;
  let w = windowSlice(state.runDetail);
  renderOverview();
  const dd = charts['#dd-chart'].values;
  const worst = Math.min(...dd), worstAt = dd.indexOf(worst);
  const equity = charts['#equity-chart'];
  const caption = nodes.get('#equity-note').textContent;
  const tooltip = equity.tooltip(worstAt);
  renderReturns();
  const table = nodes.get('#returns-body').children;
  const episodes = drawdownEpisodes(w.dates, w.account, 5, w.openingNAV, w.openingDate);
  const stats = statsOf(w.returns);
  const out = {openingNAV: w.openingNAV, openingDate: w.openingDate,
    dates: w.dates, dd, worst, worstDate: w.dates[worstAt], caption, tooltip,
    kpis, baseline: equity.baseline, episodes, table, stats,
    nav: equity.values, emptyDD: drawdownOf([], 100), emptyEpisodes: drawdownEpisodes([], [], 5, 100)};
  // Exercise the production brush callback and its absolute session mapping.
  if (input.brush) {equity.onBrush(...input.brush); out.brushedWindow = state.window; out.brushedOpening = windowSlice(state.runDetail).openingNAV;}
  JSON.stringify(out);
`, context);
process.stdout.write(result);
"""


@unittest.skipUnless(NODE, "Node.js is needed to execute the chart JavaScript")
class ChartDrawdownTests(unittest.TestCase):
    def chart(self, returns, *, window=None, opening=100000, dates=None, brush=None):
        accounts = []
        value = opening
        for daily in returns:
            value *= 1 + daily
            accounts.append(value)
        dates = dates or [f"2025-09-{i + 2:02d}" for i in range(len(returns))]
        payload = {"series": {"dates": dates, "net_return": returns, "account": accounts,
                              "turnover": [0] * len(returns), "cost": [0] * len(returns)},
                   "window": window, "brush": brush}
        result = subprocess.run([NODE, "-e", HARNESS], input=json.dumps(payload),
                                text=True, encoding="utf-8", capture_output=True, cwd=ROOT, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        out = json.loads(result.stdout)
        selected = returns if window is None else returns[window[0]:window[1] + 1]
        expected = portfolio_metrics(selected)["strategy_net"]
        self.assertAlmostEqual(out["worst"], expected["max_drawdown"], places=12)
        self.assertAlmostEqual(out["stats"]["maxDD"], expected["max_drawdown"], places=12)
        self.assertEqual(out["baseline"], out["openingNAV"])
        self.assertEqual(len(out["dd"]), len(selected))
        self.assertEqual(len(out["dates"]), len(selected))
        self.assertEqual(out["emptyDD"], [])
        self.assertEqual(out["emptyEpisodes"], [])
        return out

    def test_observed_opening_loss_matches_metric_caption_tooltip_and_table(self):
        # First three saved net returns from the reported signed run. Later
        # recovery makes this the deepest episode, just as in the full run.
        out = self.chart([-0.0090186043027298, -0.018019866866187464,
                          -0.020796053859311783, 0.013701412581969717,
                          0.013242782320469626, 0.08822387873023102])
        self.assertAlmostEqual(out["openingNAV"], 100000)
        self.assertAlmostEqual(out["worst"], -0.04711309712758565)
        self.assertEqual(out["worstDate"], "2025-09-04")
        self.assertIn("worst -4.7% on 2025-09-04", out["caption"])
        self.assertEqual(out["kpis"]["Max drawdown"], "-4.7%")
        self.assertEqual(out["tooltip"][-1][1], "-4.7%")
        episode = out["episodes"][0]
        self.assertEqual(episode["start"], "Opening NAV")
        self.assertEqual(episode["trough"], "2025-09-04")
        self.assertEqual(episode["length"], 6)
        self.assertTrue(episode["recovered"])
        self.assertAlmostEqual(episode["depth"], out["worst"])
        self.assertIn('"text": "Opening NAV"', json.dumps(out["table"]))
        self.assertIn('"text": "-4.7%"', json.dumps(out["table"]))

    def test_subwindow_includes_first_selected_return_but_resets_historical_peak(self):
        out = self.chart([0.2, -0.25, -0.1, 0.05], window=[2, 3])
        self.assertAlmostEqual(out["openingNAV"], 90000)
        self.assertAlmostEqual(out["dd"][0], -0.1)
        self.assertAlmostEqual(out["worst"], -0.1)
        self.assertEqual(out["openingDate"], "2025-09-03")
        self.assertEqual(out["episodes"][0]["start"], "2025-09-03")
        self.assertEqual(out["episodes"][0]["length"], 2)
        self.assertFalse(out["episodes"][0]["recovered"])
        self.assertIn("worst -10.0% on 2025-09-04", out["caption"])

    def test_first_session_only_loss_and_open_episode(self):
        out = self.chart([-0.05, 0.1], window=[0, 0])
        self.assertAlmostEqual(out["worst"], -0.05)
        self.assertEqual(out["episodes"][0]["length"], 1)
        self.assertFalse(out["episodes"][0]["recovered"])
        self.assertEqual(out["episodes"][0]["trough"], "2025-09-02")

    def test_later_peak_and_recovery_keep_session_dates_and_duration(self):
        out = self.chart([0.1, -0.1, -0.1, 0.3])
        episode = out["episodes"][0]
        self.assertEqual(episode["start"], "2025-09-02")
        self.assertEqual(episode["trough"], "2025-09-04")
        self.assertEqual(episode["end"], "2025-09-05")
        self.assertEqual(episode["length"], 3)
        self.assertAlmostEqual(episode["depth"], -0.19)

    def test_gains_and_flat_sessions_have_no_drawdown_episode(self):
        out = self.chart([0, 0.1, 0, 0.05])
        self.assertEqual(out["dd"], [0, 0, 0, 0])
        self.assertEqual(out["episodes"], [])

    def test_brushing_selected_window_maps_to_original_sessions(self):
        out = self.chart([0.2, -0.25, -0.1, 0.05], window=[1, 3], brush=[1, 2])
        self.assertEqual(out["brushedWindow"], [2, 3])
        self.assertAlmostEqual(out["brushedOpening"], 90000)


if __name__ == "__main__":
    unittest.main()
