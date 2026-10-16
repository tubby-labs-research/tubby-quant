"use strict";
/* Research workspace: observable campaigns and immutable evidence, using the existing UI primitives. */
const researchUI = { overview: null, detail: null, selected: null, configId: null, tab: "ideas", idea: null,
  timer: null, refreshing: false, editing: false, datasets: [], submitting: false, filter: "all", followWorker: false, revision: 0, refreshAgain: false,
  test: null, series: new Map(), seriesPending: new Set(), seriesBinding: null, checkingAgent: false };
const researchLabels = { promote: "Explore further", refine: "Refine", reject: "Rejected", park: "Parked", repair: "Repair", pending: "Pending" };
function researchTag(value) {
  return el("span", { class: `tag ar-decision ar-${value || "pending"}`, text: researchLabels[value] || value || "Pending" });
}
function researchButton(text, action, cls = "") { return el("button", { class: `btn ${cls}`, type: "button", text, onclick: action }); }
function researchNote(text, cls = "") { return el("div", { class: `ar-note ${cls}`, text }); }
function researchField(label, id, attrs = {}) {
  const input = el(attrs.rows ? "textarea" : "input", { id: `ar-${id}`, ...attrs });
  if (attrs.rows && attrs.value !== undefined) input.textContent = attrs.value;
  return el("label", { class: "ar-field" }, el("span", { text: label }), input);
}
function researchSetupIdentity() {
  return JSON.stringify([...$$("#ar-form input, #ar-form select, #ar-form textarea")]
    .filter((field) => !["ar-consent", "ar-continuous"].includes(field.id))
    .map((field) => [field.id, field.value]).concat([["focus", $("#ar-focus").value]]));
}

function initializeResearch() {
  $("#explorer").append(el("div", { class: "xview", "data-view": "auto" },
    el("div", { class: "pane-head" }, "Campaigns", researchButton("↻", () => refreshResearch(), "ghost small")),
    el("div", { class: "ar-explorer-actions" }, researchButton("New campaign", () => newResearchCampaign(), "primary"),
      researchButton("Try demo", runResearchDemo)),
    el("div", { class: "search" }, el("input", { id: "ar-campaign-search", type: "search", placeholder: "Find a campaign…", "aria-label": "Find campaigns", oninput: renderResearchList })),
    el("div", { class: "list", id: "ar-campaigns" })));
  $("#center").prepend(el("div", { class: "cview ar-workspace", "data-view": "auto" },
    el("div", { id: "ar-header", class: "ar-header" }),
    el("div", { id: "ar-workflow", class: "ar-workflow", "aria-label": "Research stages" }),
    el("div", { id: "ar-budget", class: "ar-budget" }),
    el("div", { class: "tabs", id: "ar-tabs", role: "tablist", "aria-label": "Campaign evidence" },
      ...[["ideas", "Ideas & results"], ["sources", "Sources"], ["activity", "Activity"]].map(([id, name]) => el("button", { class: `tab${id === "ideas" ? " active" : ""}`, role: "tab", "aria-selected": id === "ideas", text: name,
        onclick: () => { researchUI.tab = id; renderResearchContent(); } }))),
    el("div", { id: "ar-content", class: "ar-content" })));
  $("#inspector").append(el("div", { class: "iview", "data-view": "auto" },
    el("div", { class: "pane-head" }, "Campaign setup", el("span", { class: "count", text: "Daily · Price / volume" })),
    el("form", { id: "ar-form", class: "ar-form", onsubmit: (e) => { e.preventDefault(); saveResearchCampaign(); } },
      el("div", { id: "ar-worker-note", class: "ar-note", hidden: true, text: "Research is running. You can edit and save a new setup, then start it when this run finishes." }),
      researchField("Campaign name", "name", { required: true, maxlength: 200, placeholder: "Daily alpha research" }),
      el("label", { class: "ar-field" }, el("span", { text: "Dataset" }), el("select", { id: "ar-dataset", required: true, onchange: researchDatasetChanged })),
      el("label", { class: "ar-field" }, el("span", { text: "Universe" }), el("select", { id: "ar-universe", required: true })),
      el("div", { id: "ar-data-note", class: "ar-note" }),
      researchButton("Manage data", () => setView("data"), "ghost small"),
      researchNote("Development guides ideas. Validation chooses candidates. The holdout stays untouched."),
      el("div", { class: "ar-section-title", text: "Development period" }),
      el("div", { class: "ar-field-pair" }, researchField("From", "dev-start", { type: "date", required: true }), researchField("Through", "dev-end", { type: "date", required: true })),
      el("div", { class: "ar-section-title", text: "Validation period" }),
      el("div", { class: "ar-field-pair" }, researchField("From", "val-start", { type: "date", required: true }), researchField("Through", "val-end", { type: "date", required: true })),
      researchField("Reserved holdout starts", "holdout", { type: "date" }),
      el("div", { class: "ar-field-pair" }, researchField("Cost (basis points)", "cost", { type: "number", min: 0, step: 0.1, value: 5, required: true }), researchField("Minimum Sharpe", "sharpe", { type: "number", step: 0.05, value: 1.75, required: true })),
      el("label", { class: "ar-field" }, el("span", { text: "Research agent" }), el("select", { id: "ar-provider", onchange: () => { $("#ar-consent").checked = false; researchProviderChanged(); } },
        el("option", { value: "reviewed", text: "Reviewed source-backed ideas" }), el("option", { value: "anthropic", text: "Anthropic · direct model" }))),
      researchField("Model ID", "model", { maxlength: 200, placeholder: "Your configured model ID" }),
      el("div", { class: "ar-connection" }, el("div", { id: "ar-agent-identity", class: "ar-agent-identity" }),
        el("div", { id: "ar-credential", class: "ar-note", role: "status" }),
        el("button", { id: "ar-agent-check", class: "btn ghost small", type: "button", hidden: true, text: "Check connection", onclick: checkResearchAgent })),
      el("details", { class: "ar-advanced" }, el("summary", { text: "Connect an agent" }),
        researchNote("Your server administrator can connect any provider through the Research Agent API. Select its name here once configured. Keys stay on the server."),
        el("a", { href: "/research-agent-guide", target: "_blank", rel: "noopener noreferrer", text: "Agent API setup guide" })),
      el("div", { id: "ar-live-fields" }, researchField("Paper searches · one per line", "queries", { rows: 3, maxlength: 10000, value: 'cat:q-fin.TR AND (ti:momentum OR ti:reversal OR abs:volume)' })),
      el("details", { id: "ar-reviewed-fields", class: "ar-advanced", hidden: true }, el("summary", { text: "Source-backed research plan" }), researchField("Reviewed research plan (JSON)", "reviewed", { rows: 7, spellcheck: false, placeholder: '{"sources": […], "hypotheses": […], "refinements": […]}' }),
        researchNote("Cited ideas prepared by a researcher. Tests run automatically; new ideas require an updated plan.")),
      el("details", { class: "ar-advanced" }, el("summary", { text: "Budgets & review rules" }),
        el("div", { class: "ar-field-pair" }, researchField("Total evaluations", "evaluations", { type: "number", min: 1, max: 1000, value: 30, required: true }), researchField("Refinement depth", "depth", { type: "number", min: 0, max: 10, value: 2, required: true })),
        el("div", { class: "ar-field-pair" }, researchField("Total model calls", "calls", { type: "number", min: 1, max: 1000, value: 30, required: true }), researchField("Calls / UTC day", "daily", { type: "number", min: 1, max: 1000, value: 10, required: true })),
        el("div", { class: "ar-field-pair" }, researchField("Validation sessions", "days", { type: "number", min: 2, value: 120, required: true }), researchField("Max drawdown", "drawdown", { type: "number", step: 0.01, min: 0, max: 1, value: 0.3, required: true })),
        el("div", { class: "ar-field-pair" }, researchField("Max turnover", "turnover", { type: "number", step: 0.05, min: 0, value: 0.7, required: true }), researchField("Warm-up sessions", "warmup", { type: "number", min: 2, value: 252, required: true })),
        el("div", { class: "ar-field-pair" }, researchField("Ideas / family", "family", { type: "number", min: 1, value: 6, required: true }), researchField("Min improvement", "improvement", { type: "number", min: 0, step: 0.05, value: 0.05, required: true }))),
      el("label", { class: "ar-checkbox" }, el("input", { id: "ar-continuous", type: "checkbox" }), el("span", { text: "Keep researching while the app is open" })),
      researchNote("Continuous mode still stops at the total budget. Closing the app stops its worker."),
      el("label", { class: "ar-checkbox", id: "ar-consent-row" }, el("input", { id: "ar-consent", type: "checkbox" }), el("span", { text: "Allow calls to this configured AI model" })),
      el("div", { id: "ar-form-error", class: "ar-form-error", role: "alert" }),
      el("div", { class: "ar-form-actions" }, el("button", { class: "btn primary", type: "submit", id: "ar-save", text: "Save campaign" }), el("button", { class: "btn", type: "button", id: "ar-start", text: "Start", onclick: startResearchCampaign })))),
  );
  newResearchCampaign(false);
  refreshResearch();
}

function researchProviderChanged() {
  const reviewed = $("#ar-provider").value === "reviewed";
  const agent = $("#ar-provider").value.startsWith("agent:");
  $("#ar-live-fields").hidden = reviewed; $("#ar-reviewed-fields").hidden = !reviewed;
  $("#ar-consent-row").hidden = reviewed; $("#ar-model").required = !reviewed;
  if (reviewed) $("#ar-reviewed-fields").open = true;
  $("#ar-model").parentElement.hidden = reviewed || agent;
  $("#ar-model").required = !reviewed && !agent;
  const connection = researchConnection();
  if (agent && connection) $("#ar-model").value = connection.model;
  renderResearchConnection();
}
function researchConnection() {
  return ((researchUI.overview && researchUI.overview.connections || {}).agents || [])
    .find((row) => row.id === $("#ar-provider").value);
}
function updateResearchAgents(selected = $("#ar-provider").value) {
  const rows = ((researchUI.overview && researchUI.overview.connections || {}).agents || []);
  if (!rows.length) return;
  setChildren($("#ar-provider"), ...rows.map((row) => el("option", { value: row.id, text: row.name })));
  if (!rows.some((row) => row.id === selected)) $("#ar-provider").append(el("option", { value: selected, text: "Saved agent · unavailable on this server" }));
  $("#ar-provider").value = selected;
  renderResearchConnection();
}
function renderResearchConnection() {
  const row = researchConnection(), selected = $("#ar-provider").value;
  $("#ar-agent-identity").textContent = row ? `${row.provider_name}${row.model ? " · " + row.model : ""}` : selected === "reviewed" ? "Reviewed plan" : "Agent configuration unavailable";
  const issues = ((researchUI.overview && researchUI.overview.connections || {}).issues || []).join(" ");
  $("#ar-credential").textContent = (row ? row.message : "Refresh or ask the server administrator to configure this agent.") + (issues ? " " + issues : "");
  $("#ar-credential").dataset.state = row ? row.state : "unavailable";
  $("#ar-agent-check").hidden = !selected.startsWith("agent:");
  $("#ar-agent-check").disabled = researchUI.checkingAgent || !row || !row.ready;
  $("#ar-agent-check").textContent = researchUI.checkingAgent ? "Checking…" : "Check connection";
}
async function checkResearchAgent() {
  const row = researchConnection();
  if (!row || researchUI.checkingAgent) return;
  researchUI.checkingAgent = true; renderResearchConnection();
  try { await api("research-agent-check", { agent_id: row.agent_id }); await refreshResearch(); }
  catch (e) { toast("Connection check failed", e.message, "bad"); }
  finally { researchUI.checkingAgent = false; renderResearchConnection(); }
}
function researchDatasetChanged(preserve = false) {
  const d = researchUI.datasets.find((row) => row.path === $("#ar-dataset").value);
  const current = preserve ? $("#ar-universe").value : "";
  const keys = d ? Object.keys(d.universes || {}) : [];
  setChildren($("#ar-universe"), el("option", { value: "", text: "Choose universe…" }), ...keys.map((key) => el("option", { value: key, text: `${key} · ${d.universes[key].size || ""} stocks` })));
  if (keys.includes(current)) $("#ar-universe").value = current;
  else if (keys.length === 1) $("#ar-universe").value = keys[0];
  $("#ar-data-note").textContent = d ? `${d.provider || "Local data"} · ${d.start || "Start not recorded"} to ${d.end || "end not recorded"} (end exclusive). Choose your research split explicitly.` :
    researchUI.datasets.some((row) => row.status === "ready") ? "Choose a dataset available on this machine." : "No prepared datasets found. Import daily data in Data, then create a campaign.";
}
async function newResearchCampaign(show = true) {
  const revision = ++researchUI.revision;
  researchUI.savedSetup = null;
  researchUI.configId = null; researchUI.selected = null; researchUI.detail = null; researchUI.idea = null; researchUI.editing = true; researchUI.followWorker = false;
  if ($("#ar-form")) { $("#ar-form").reset(); $("#ar-form-error").textContent = ""; researchProviderChanged(); }
  if ($("#ar-focus")) $("#ar-focus").value = "";
  if (show) { setView("auto"); document.body.classList.remove("no-inspector"); }
  try {
    const datasets = await get("datasets");
    if (revision !== researchUI.revision) return;
    researchUI.datasets = datasets;
    setChildren($("#ar-dataset"), el("option", { value: "", text: "Choose prepared data…" }), ...researchUI.datasets.filter((d) => d.status === "ready").map((d) => el("option", { value: d.path, text: `${d.name} · ${d.symbols} stocks` })));
    researchDatasetChanged();
  } catch (e) { $("#ar-form-error").textContent = e.message; }
  renderResearch();
}
async function openResearchConfig(id) {
  const revision = ++researchUI.revision;
  try {
    const cfg = await get("research-config", { id });
    if (revision !== researchUI.revision) return;
    researchUI.configId = id; researchUI.editing = true;
    await populateResearchConfig(cfg, revision);
    if (revision === researchUI.revision) renderResearch();
  } catch (e) { toast("Cannot open campaign setup", e.message, "bad"); }
}
async function populateResearchConfig(cfg, revision = researchUI.revision) {
  const datasets = await get("datasets");
  if (revision !== researchUI.revision) return;
  $("#ar-form").reset();
  const b = cfg.backend || {}, m = cfg.model || {}, p = { min_validation_days: 60, min_sharpe: 1, max_drawdown: .3, max_turnover: .7, min_refinement_improvement: .05, ...cfg.policy }, l = cfg.limits || {};
  const mapping = { name: cfg.name, "dev-start": b.development_start, "dev-end": b.development_end, "val-start": b.validation_start,
    "val-end": b.validation_end, holdout: b.holdout_start, cost: b.cost_bps, sharpe: p.min_sharpe, model: m.model,
    queries: (cfg.discovery.arxiv_queries || []).join("\n"), evaluations: l.max_experiments, depth: l.max_refinement_depth,
    calls: l.max_model_calls, daily: l.daily_model_calls, days: p.min_validation_days, drawdown: p.max_drawdown,
    turnover: p.max_turnover, warmup: b.warmup, family: l.max_per_family, improvement: p.min_refinement_improvement };
  for (const [key, value] of Object.entries(mapping)) if (value !== undefined) $(`#ar-${key}`).value = value;
  researchUI.datasets = datasets;
  setChildren($("#ar-dataset"), el("option", { value: "", text: "Choose prepared data…" }), ...researchUI.datasets.filter((d) => d.status === "ready").map((d) => el("option", { value: d.path, text: d.name })));
  const d = researchUI.datasets.find((row) => row.path === b.dataset || row.name === b.dataset);
  $("#ar-dataset").value = d ? d.path : ""; researchDatasetChanged(); $("#ar-universe").value = b.universe || "";
  $("#ar-focus").value = m.research_focus || "";
  updateResearchAgents(m.provider === "agent" ? `agent:${m.agent_id}` : m.provider === "fixture" || m.provider === "reviewed" ? "reviewed" : "anthropic");
  if ($("#ar-provider").value === "reviewed") $("#ar-reviewed").value = JSON.stringify({ sources: cfg.discovery.fixture_sources || [], hypotheses: m.hypotheses || [], refinements: m.refinements || [] }, null, 2);
  $("#ar-consent").checked = false; researchProviderChanged(); $("#ar-form-error").textContent = "";
  researchUI.savedSetup = researchSetupIdentity();
}
async function saveResearchCampaign() {
  if (researchUI.submitting || !$("#ar-form").reportValidity()) return null;
  const read = (key) => $(`#ar-${key}`).value.trim(), num = (key) => Number(read(key));
  const reviewed = read("provider") === "reviewed";
  const connection = researchConnection();
  const agent = read("provider").startsWith("agent:");
  try {
    researchUI.submitting = true; $("#ar-save").disabled = true;
    const cfg = { name: read("name"), backend: { dataset: read("dataset"), universe: read("universe"), development_start: read("dev-start"), development_end: read("dev-end"),
      validation_start: read("val-start"), validation_end: read("val-end"), holdout_start: read("holdout"), cost_bps: num("cost"), warmup: num("warmup") },
      model: { provider: reviewed ? "reviewed" : agent ? "agent" : "anthropic", model: reviewed ? "session-reviewed" : agent && connection ? connection.model : read("model"), research_focus: $("#ar-focus").value.trim() },
      discovery: { arxiv_queries: read("queries").split("\n").filter(Boolean), max_results: 6 },
      policy: { min_validation_days: num("days"), min_sharpe: num("sharpe"), max_drawdown: num("drawdown"), max_turnover: num("turnover"), min_refinement_improvement: num("improvement") },
      limits: { max_experiments: num("evaluations"), max_hypotheses: num("evaluations"), max_per_family: num("family"), max_refinement_depth: num("depth"), max_model_calls: num("calls"), daily_model_calls: num("daily"), max_job_errors: 2 } };
    if (reviewed) {
      const plan = JSON.parse(read("reviewed"));
      cfg.model.hypotheses = plan.hypotheses; cfg.model.refinements = plan.refinements || [];
      cfg.discovery = { fixture_sources: plan.sources, arxiv_queries: [], max_results: Math.min(10, (plan.sources || []).length) };
    }
    if (agent) cfg.model.agent_id = read("provider").slice(6);
    // Saving a changed plan creates a distinct config, preserving prior campaign identities and evidence.
    const saved = await api("research-save", cfg);
    researchUI.revision++;
    researchUI.configId = saved.id; researchUI.selected = null; researchUI.detail = null; researchUI.idea = null;
    researchUI.followWorker = false; researchUI.editing = true; $("#ar-form-error").textContent = "";
    researchUI.savedSetup = researchSetupIdentity();
    toast("Campaign saved", "Start it when the setup is ready."); await refreshResearch(); return saved;
  } catch (e) { $("#ar-form-error").textContent = e.message; return null; }
  finally { researchUI.submitting = false; renderResearch(); }
}
async function startResearchCampaign() {
  if (researchUI.submitting || (researchUI.overview && researchUI.overview.worker.running)) return;
  if (researchUI.detail && researchUI.detail.campaign.demo) { await runResearchDemo(); return; }
  // Start the setup currently shown, including any unsaved edits.
  if (researchUI.savedSetup !== researchSetupIdentity() && !await saveResearchCampaign()) return;
  const cfg = researchUI.configId ? await get("research-config", { id: researchUI.configId }) : null;
  if (!cfg) { toast("Save the campaign first", "Choose data, periods and research inputs in Campaign setup."); return; }
  const provider = cfg.provider || (cfg.model && cfg.model.provider);
  const reviewed = provider === "fixture" || provider === "reviewed";
  if (!reviewed && !$("#ar-consent").checked) { toast("Enable model calls in setup", "Starting research sends sources and saved research history to the configured model."); return; }
  try {
    researchUI.revision++;
    await api("research-start", { id: researchUI.configId, continuous: $("#ar-continuous").checked, allow_model: !reviewed && $("#ar-consent").checked });
    researchUI.revision++;
    researchUI.editing = false; researchUI.selected = null; researchUI.detail = null; researchUI.followWorker = true;
    $("#ar-reviewed-fields").open = false; await refreshResearch();
  } catch (e) { $("#ar-form-error").textContent = e.message; toast("Research could not start", e.message, "bad"); }
}
async function runResearchDemo() {
  try { researchUI.revision++; await api("research-demo", {}); researchUI.revision++; researchUI.configId = null; researchUI.editing = false; researchUI.selected = null; researchUI.detail = null; researchUI.followWorker = true; setView("auto"); await refreshResearch(); }
  catch (e) { toast("Demo could not start", e.message, "bad"); }
}
async function stopResearch() {
  try { toast("Stopping research", "The current step is given time to save its checkpoint."); await api("research-stop", {}); await refreshResearch(); }
  catch (e) { toast("Could not stop research", e.message, "bad"); }
}
async function selectResearchCampaign(id) {
  const revision = ++researchUI.revision;
  researchUI.selected = id; researchUI.idea = null; researchUI.editing = false; researchUI.followWorker = false;
  try {
    const detail = await get("research-detail", { id });
    if (revision !== researchUI.revision) return;
    researchUI.detail = detail;
    researchUI.configId = researchUI.detail.campaign.config_id;
    renderResearch();
    if (researchUI.configId && researchUI.configId !== "demo") {
      const cfg = await get("research-config", { id: researchUI.configId });
      if (revision !== researchUI.revision) return;
      await populateResearchConfig(cfg, revision);
    }
    if (revision === researchUI.revision) renderResearch();
  } catch (e) { toast("Cannot open research", e.message, "bad"); }
}
async function refreshResearch() {
  if (researchUI.refreshing) { researchUI.refreshAgain = true; return; }
  researchUI.refreshing = true; researchUI.refreshAgain = false; clearTimeout(researchUI.timer);
  const revision = researchUI.revision;
  try {
    const overview = await get("research-overview");
    if (revision !== researchUI.revision) return;
    researchUI.overview = overview;
    const w = overview.worker;
    // A poll started before Start or a manual selection must not overwrite that action.
    const matchingWorker = w.config_id === researchUI.configId || (w.demo && !researchUI.configId);
    if (!researchUI.editing && w.campaign_id && researchUI.followWorker && matchingWorker && researchUI.selected !== w.campaign_id) { researchUI.selected = w.campaign_id; researchUI.idea = null; }
    if (!researchUI.editing && !researchUI.selected && !researchUI.followWorker && overview.campaigns.length) researchUI.selected = overview.campaigns[overview.campaigns.length - 1].id;
    const selected = researchUI.selected;
    if (selected) {
      const detail = await get("research-detail", { id: selected });
      if (revision !== researchUI.revision || selected !== researchUI.selected) return;
      researchUI.detail = detail;
    }
    if (researchUI.detail && researchUI.followWorker && matchingWorker && researchUI.selected === w.campaign_id && !w.running) researchUI.followWorker = false;
    renderResearch();
  } catch (e) { if (revision === researchUI.revision && state.view === "auto") $("#ar-content").replaceChildren(researchNote(`Cannot refresh research: ${e.message}`, "ar-error")); }
  finally { researchUI.refreshing = false; researchUI.timer = setTimeout(refreshResearch, researchUI.refreshAgain || revision !== researchUI.revision ? 0 : researchUI.overview && researchUI.overview.worker.running ? 1200 : 5000); }
}
function renderResearchList() {
  if (!researchUI.overview) return;
  const q = $("#ar-campaign-search").value.toLowerCase();
  const campaigns = researchUI.overview.campaigns.filter((c) => `${c.name} ${c.id}`.toLowerCase().includes(q))
    .sort((a, b) => (b.created_at || "").localeCompare(a.created_at || ""));
  const items = campaigns.map((c) => el("button", { class: `ar-campaign${researchUI.selected === c.id ? " selected" : ""}`, onclick: () => selectResearchCampaign(c.id) },
    el("span", { class: "ar-campaign-name", text: c.name }), el("span", { class: "ar-campaign-meta", text: `${c.demo ? "Artificial demo · " : ""}${c.counts.experiments} tests · ${c.status}${c.queued ? " · " + c.queued + " queued" : ""}` }),
    el("span", { class: "ar-campaign-date", text: shortTime(c.created_at) })));
  const configs = researchUI.overview.configs.filter((c) => !campaigns.some((row) => row.config_id === c.id) && `${c.name} ${c.id}`.toLowerCase().includes(q));
  items.push(...configs.map((c) => el("button", { class: `ar-campaign${researchUI.configId === c.id ? " selected" : ""}`, onclick: async () => { researchUI.selected = null; researchUI.detail = null; await openResearchConfig(c.id); } },
    el("span", { class: "ar-campaign-name", text: c.name }), el("span", { class: "ar-campaign-meta", text: c.ready ? "Saved · ready to start" : "Saved · setup needed" }))));
  setChildren($("#ar-campaigns"), items.length ? items : el("div", { class: "list-empty", text: q ? "No matching campaigns." : "Your research history will appear here. Failed ideas are retained too." }));
}
function renderResearchToolbar() {
  if (!researchUI.overview || state.view !== "auto") return;
  const w = researchUI.overview.worker;
  const cfg = researchUI.overview.configs.find((c) => c.id === researchUI.configId);
  $("#run-label").textContent = researchUI.detail && researchUI.detail.campaign.demo ? "Resume demo" : "Start research";
  $("#run").disabled = w.running || researchUI.submitting || (state.job && state.job.state === "running") || (!cfg && !(researchUI.detail && researchUI.detail.campaign.demo));
  $("#cancel").disabled = !w.running;
  $("#sb-text").textContent = w.running ? `Research · ${w.phase || w.status} · ${w.evaluations || 0} evaluations` : w.last_error ? `Research · ${w.last_error}` : "Research ready";
}
function renderResearch() {
  if (!$("#ar-header")) return;
  renderResearchList();
  const d = researchUI.detail, c = d && d.campaign, w = researchUI.overview && researchUI.overview.worker;
  const focus = $("#ar-focus") ? $("#ar-focus").value : "";
  updateResearchAgents();
  if (document.activeElement !== $("#ar-focus")) setChildren($("#ar-header"),
    el("div", { class: "ar-heading" }, el("span", { class: "ar-eyebrow", text: "Daily alpha research" }), el("h1", { text: c ? c.name : "From evidence to expression" }),
      el("p", { text: c && c.demo ? "Artificial demo · prepared ideas and synthetic prices, real test engine" : c && c.reviewed ? "Reviewed source-backed ideas · local market data · automatic testing and review" : "Research a mechanism. Test its expression. Keep the evidence." }),
      c ? el("div", { class: "ar-detail-meta", text: `${c.provider_name || c.provider || "Reviewed plan"} · ${c.model_id || "prepared ideas"}${c.agent_id ? " · " + c.agent_id : ""}` }) : null),
    el("div", { class: "ar-head-actions" }, w && w.running ? researchTag(w.phase || w.status) : null,
      c ? researchButton("Export snapshot", exportResearch, "small") : researchButton("Try demo", runResearchDemo, "small")),
    el("div", { class: "ar-composer" }, el("label", { for: "ar-focus", text: "Research question" }),
      el("textarea", { id: "ar-focus", rows: 2, maxlength: 2000, placeholder: "Which daily price and volume mechanism should we investigate?", disabled: researchUI.submitting, value: focus,
        oninput: () => { researchUI.editing = true; } }, focus),
      researchNote(c ? "The saved question belongs to this campaign. Save changes as a new campaign before starting." : "Ideas must cite retrieved evidence; the question alone is not evidence.")));
  if (w && w.status === "error" && (!c || c.id === w.campaign_id)) $("#ar-header").append(researchNote(`Research stopped with an error: ${w.last_error || "Check Activity for worker messages."}`, "ar-error ar-stop-note"));
  else if (c && c.status === "interrupted") $("#ar-header").append(researchNote(`Research is interrupted with ${c.queued || 0} tests queued. Saved results are retained; this campaign is not complete.`, "ar-stop-note"));
  const phases = ["Discover", "Build expression", "Test", "Review", "Refine / switch"];
  const phase = w && w.running && (!c || c.id === w.campaign_id) ? w.phase || w.status : "";
  const active = /discover/.test(phase) ? 0 : /evaluat/.test(phase) ? 2 : /review/.test(phase) ? 3 : /refin/.test(phase) ? 4 : phase === "starting" ? 0 : -1;
  setChildren($("#ar-workflow"), ...phases.map((name, i) => el("div", { class: `ar-stage${i === active ? " current" : ""}` }, el("span", { class: "ar-stage-number", text: i + 1 }), el("span", { text: name }))));
  const counts = c ? c.counts : {}, checkpoint = d ? d.checkpoint : {}, limits = c ? c.limits : {};
  setChildren($("#ar-budget"), ...[["Sources", counts.sources || 0], ["Ideas", counts.hypotheses || 0], ["Evaluations", `${checkpoint.evaluations || counts.experiments || 0} / ${limits.max_experiments || "—"}`], [c && (c.reviewed || c.demo) ? "Plan steps" : "Model calls", `${checkpoint.model_calls || 0} / ${limits.max_model_calls || "—"}`]].map(([name, value]) => el("div", { class: "ar-stat" }, el("span", { text: name }), el("strong", { text: value }))));
  updateResearchSetupAvailability(w);
  renderResearchConnection();
  renderResearchContent(); renderResearchToolbar();
}
function updateResearchSetupAvailability(worker) {
  // Saving creates a separate configuration; the active worker owns its saved snapshot.
  $$("#ar-form input, #ar-form select, #ar-form textarea, #ar-form button").forEach((field) => { field.disabled = researchUI.submitting; });
  $("#ar-start").disabled = researchUI.submitting || !!(worker && worker.running) || !!(state.job && state.job.state === "running");
  $("#ar-worker-note").hidden = !(worker && worker.running);
}
function renderResearchContent() {
  const d = researchUI.detail, node = $("#ar-content"); if (!node) return;
  researchUI.seriesBinding = null;
  $$("#ar-tabs .tab").forEach((b, i) => { const active = ["ideas", "sources", "activity"][i] === researchUI.tab; b.classList.toggle("active", active); b.setAttribute("aria-selected", active); });
  if (!d) {
    setChildren(node, el("div", { class: "ar-empty" }, icon("compare"), el("h2", { text: "A research trail you can inspect" }),
      el("p", { text: "Start with a daily price and volume question. Each idea keeps its sources, expression, test results and decision — including failures." }),
      el("div", { class: "ar-empty-actions" }, researchButton("Save campaign", saveResearchCampaign, "primary"), researchButton("Try offline demo", runResearchDemo)),
      researchNote("Set data, chronological periods and a budget in the right panel. Nothing runs until you start.")));
    return;
  }
  if (researchUI.tab === "sources") {
    setChildren(node, ...(d.sources.length ? d.sources.map((s) => el("article", { class: "ar-card" },
      el("div", { class: "ar-card-head" }, el("h3", { text: s.title }), researchTag(s.evidence_level || "source")),
      /^https?:\/\//.test(s.url) ? el("a", { href: s.url, target: "_blank", rel: "noopener noreferrer", text: s.url }) : null,
      el("p", { text: s.excerpt || s.text || "Source metadata retained. Supporting quotes appear with each idea." }),
      el("small", { text: `Retrieved ${shortTime(s.retrieved_at)} · ${s.content_sha256 ? s.content_sha256.slice(0, 12) : s.id}` }))) : [researchNote("Sources will appear after discovery.")])); return;
  }
  if (researchUI.tab === "activity") {
    const rows = [...d.experiments.map((e) => ({ created_at: e.created_at, title: "Evaluated expression", text: `${e.expression} · ${e.result.status}` })),
      ...d.reviews.map((r) => ({ created_at: r.created_at, title: researchLabels[r.decision] || r.decision, text: r.reason })),
      ...d.events.map((e) => ({ created_at: e.created_at, title: (e.kind || "Event").replaceAll("_", " "), text: e.reason || e.message || e.hypothesis_id || "Saved to research history" }))].sort((a, b) => (b.created_at || "").localeCompare(a.created_at || ""));
    const w = researchUI.overview.worker;
    setChildren(node, el("div", { class: "ar-card" }, el("h3", { text: "Execution history" }), ...rows.map((r) => el("div", { class: "ar-activity" }, el("span", { class: "ar-activity-dot" }),
      el("div", {}, el("b", { text: r.title }), el("p", { text: r.text }), el("small", { text: shortTime(r.created_at) }))))),
      w.campaign_id === d.campaign.id && w.last_error ? researchNote(w.last_error, "ar-error") : null,
      w.campaign_id === d.campaign.id ? el("details", { class: "ar-card" }, el("summary", { text: "Worker messages" }), el("pre", { class: "ar-log", text: (w.log || []).join("\n") })) : null); return;
  }
  const ideas = [...d.hypotheses].sort((a, b) => {
    const score = (h) => h.result && h.result.status === "ok" ? h.result.metrics.validation.sharpe : null;
    const x = score(a), y = score(b);
    return Number.isFinite(x) && Number.isFinite(y) ? y - x : Number.isFinite(x) ? -1 : Number.isFinite(y) ? 1 : (b.created_at || "").localeCompare(a.created_at || "");
  });
  const visible = ideas.filter((h) => researchUI.filter === "all" || (h.latest_review || {}).decision === researchUI.filter);
  const toolbar = el("div", { class: "ar-table-toolbar" }, el("b", { text: `${ideas.length} research ideas` }), el("select", { "aria-label": "Filter research decisions", value: researchUI.filter, onchange: (e) => { researchUI.filter = e.target.value; renderResearchContent(); } },
    ...[["all", "All decisions"], ["promote", "Explore further"], ["refine", "Refine"], ["reject", "Rejected"], ["park", "Parked"], ["repair", "Repair"]].map(([value, text]) => el("option", { value, selected: value === researchUI.filter, text }))));
  const table = el("table", { class: "ar-table" }, el("thead", {}, el("tr", {}, ...["Idea / expression", "Dev. Sharpe", "Val. Sharpe", "Val. annual return", "Decision"].map((text) => el("th", { text })))),
    el("tbody", {}, ...visible.map((h) => {
      const metrics = (h.result || {}).metrics || {}, val = metrics.validation || {};
      const button = el("button", { class: "ar-idea-link", onclick: () => { researchUI.idea = h.id; renderResearchContent(); } }, el("b", { text: h.title }), el("code", { text: h.expression }));
      return el("tr", { class: h.id === researchUI.idea ? "selected" : "" }, el("td", {}, button), el("td", { class: "mono", text: fmt.num((metrics.development || {}).sharpe) }),
        el("td", { class: "mono", text: fmt.num(val.sharpe) }), el("td", { class: "mono", text: fmt.signedPct(val.returns) }), el("td", {}, researchTag((h.latest_review || {}).decision)));
    })));
  const h = visible.find((row) => row.id === researchUI.idea) || visible[0];
  setChildren(node, toolbar, visible.length ? el("div", { class: "ar-table-wrap" }, table) : researchNote(ideas.length ? "No ideas match this decision." : "Discovery and expression generation are in progress."),
    h ? renderResearchIdea(h) : null,
    researchNote("Validation is used for selection across many trials. A high score here needs an independent holdout check; costs exclude market impact and borrow fees."));
}
function renderResearchIdea(h) {
  const tests = h.experiments || [];
  const test = tests.find((row) => row.id === researchUI.test) || tests[tests.length - 1];
  const r = test ? (researchUI.detail.reviews || []).find((row) => row.experiment_id === test.id && row.hypothesis_id === h.id) || {} : h.latest_review || {};
  const result = test ? test.result || {} : h.result || {}, metrics = result.metrics || {};
  const detail = el("article", { class: "ar-card ar-idea-detail" },
    el("div", { class: "ar-card-head" }, el("h3", { text: h.title }), researchTag(r.decision)),
    el("pre", { class: "ar-expression", text: h.expression }),
    el("div", { class: "ar-inline-actions" }, researchButton("Copy expression", () => copyText(h.expression, "Expression copied"), "small"),
      researchButton("Open in Alpha Lab", () => { $("#lab-text").value = h.expression; onLabInput(); setView("lab"); toast("Expression opened", "Match the dataset, dates and costs to this research before comparing results."); }, "small")),
    el("p", { text: h.mechanism }), el("div", { class: "ar-detail-meta", text: `${h.family} · ${h.expected_horizon} · ${(h.required_fields || []).join(", ")}` }),
    r.reason ? researchNote(r.reason, `ar-review ar-${r.decision}`) : null,
    h.question ? el("p", {}, el("b", { text: "Refinement question: " }), h.question) : null,
    h.expected_change ? el("p", {}, el("b", { text: "Expected change: " }), h.expected_change) : null,
    h.lineage && h.lineage.length ? el("div", { class: "ar-lineage" }, el("b", { text: "Refines" }), ...h.lineage.map((p) => el("code", { text: p.expression }))) : null,
    result.status === "error" ? researchNote(result.error || result.message || "Evaluation failed; this is an execution error, not evidence against the idea.", "ar-error") : null,
    tests.length ? el("label", { class: "ar-field ar-test-picker" }, el("span", { text: "Retained test" }),
      el("select", { "aria-label": "Retained test", onchange: (e) => { researchUI.test = e.target.value; renderResearchContent(); } },
        ...tests.map((row, i) => el("option", { value: row.id, selected: row.id === test.id,
          text: `Test ${i + 1} · ${(row.result || {}).status || "pending"} · ${row.id.slice(-12)}` })))) : null,
    el("table", { class: "ar-table ar-metrics" }, el("thead", {}, el("tr", {}, ...["Period", "Sessions", "Sharpe", "Annual return*", "Drawdown*", "Turnover"].map((text) => el("th", { text })))),
      el("tbody", {}, ...["development", "validation"].map((p) => { const m = metrics[p] || {}; return el("tr", {}, el("td", { text: p }), el("td", { text: m.days || "—" }), el("td", { text: fmt.num(m.sharpe) }), el("td", { text: fmt.signedPct(m.returns) }), el("td", { text: fmt.pct(m.drawdown) }), el("td", { text: fmt.pct(m.turnover) })); }))),
    researchNote("* Annualised net PnL and peak-to-trough PnL loss divided by half the fixed book size."),
    renderResearchSeries(h, test),
    el("div", { class: "ar-section-title", text: "Evidence behind the idea" }),
    ...(h.evidence || []).map((e) => { const s = (h.sources || []).find((row) => row.id === e.source_id); return el("blockquote", {}, el("p", { text: e.quote }), el("cite", { text: s ? s.title : e.source_id })); }),
    el("div", { class: "ar-detail-meta", text: result.settings ? `Delay ${result.settings.delay} · ${result.settings.cost_bps} bp costs · ${result.settings.neutralization} neutralization · ${result.settings.warmup} warm-up sessions · holdout ${result.settings.holdout_evaluated ? "evaluated" : "reserved"}` : "" }));
  return detail;
}
function renderResearchSeries(h, test) {
  if (!test) return researchNote("No completed test yet. PnL appears after a test saves its daily series.");
  const campaignId = researchUI.detail.campaign.id, key = `${campaignId}/${test.id}`;
  researchUI.seriesBinding = key;
  const data = researchUI.series.get(key), box = el("div", { class: "ar-pnl", "data-test-id": test.id });
  if (!data) {
    box.append(researchNote("Loading retained daily PnL…"));
    if (!researchUI.seriesPending.has(key)) loadResearchSeries(campaignId, test.id, key);
    return box;
  }
  if (data.status !== "ready") {
    box.append(researchNote(data.message || "Daily series unavailable.", data.status === "invalid" ? "ar-error" : ""));
    box.append(researchButton("Retry series", () => { researchUI.series.delete(key); renderResearchContent(); }, "ghost small"));
    return box;
  }
  box.append(el("div", { class: "ar-section-title", text: "Cumulative net PnL" }),
    researchNote(`${data.semantics} Book size ${fmt.money(data.booksize)}.${data.reused ? " Reused evaluation · original saved series." : ""}`));
  for (const period of ["development", "validation"]) {
    const series = data.periods[period], chart = el("div", { class: "chart ar-pnl-chart" }), dd = el("div", { class: "chart short ar-dd-chart" });
    box.append(el("div", { class: "chart-panel ar-period" }, el("div", { class: "chart-head" }, el("span", { text: period === "development" ? "Development" : "Validation" }),
      el("span", { class: "chart-note", text: `${series.dates[0]} → ${series.dates[series.dates.length - 1]} · ${fmt.money(series.cumulative[series.cumulative.length - 1])} net` })), chart, dd));
    drawWhenSized(chart, () => {
      if (!chart.isConnected || researchUI.seriesBinding !== key) return;
      const group = crosshairGroup();
      group.add(lineChart(chart, { dates: series.dates, values: series.cumulative, baseline: 0, format: fmt.money, label: `${period} cumulative net PnL`, group,
        tooltip: (i) => [["Daily net PnL", fmt.money(series.pnl[i]), tone(series.pnl[i])], ["Cumulative net PnL", fmt.money(series.cumulative[i]), tone(series.cumulative[i])], ["Turnover", fmt.pct(series.turnover[i]), ""]] }));
      group.add(lineChart(dd, { dates: series.dates, values: series.drawdown, kind: "dd", format: fmt.money, axisX: false, label: `${period} drawdown`, group,
        tooltip: (i) => [["Drawdown", fmt.money(series.drawdown[i]), series.drawdown[i] < 0 ? "neg" : ""]] }));
    });
  }
  return box;
}
async function loadResearchSeries(campaignId, experimentId, key) {
  researchUI.seriesPending.add(key);
  try {
    const data = await get("research-series", { campaign_id: campaignId, experiment_id: experimentId });
    if (data.campaign_id !== campaignId || data.experiment_id !== experimentId) throw new Error("Series did not match the selected test.");
    researchUI.series.set(key, data);
    while (researchUI.series.size > 24) researchUI.series.delete(researchUI.series.keys().next().value);
  } catch (e) { researchUI.series.set(key, { status: "unavailable", message: e.message }); }
  finally {
    researchUI.seriesPending.delete(key);
    if (researchUI.selected === campaignId && researchUI.seriesBinding === key && researchUI.tab === "ideas") renderResearchContent();
  }
}
function exportResearch() {
  if (!researchUI.detail) return;
  const blob = new Blob([JSON.stringify(researchUI.detail, null, 2) + "\n"], { type: "application/json" });
  const url = URL.createObjectURL(blob), a = el("a", { href: url, download: `${researchUI.detail.campaign.id}-snapshot.json` });
  a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
