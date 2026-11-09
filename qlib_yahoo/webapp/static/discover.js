"use strict";
/* Discover: the research-domain lab (autonomous alpha discovery, machine learning) as a guided view.
   A short form becomes a complete campaign file; the campaign runs in its own worker process; this view
   shows progress, the leaderboard, the ledger check and the single-use holdout step. */
const discoverUI = { catalog: null, campaigns: [], selected: null, detail: null, worker: null, timer: null, refreshing: false,
  domain: "alpha", form: {}, plan: null, planError: "", planTimer: null, planRevision: 0, composer: false, rounds: "3",
  wasRunning: false, verify: null, lastExport: null, busy: false, candidate: null, candidateId: null };
const DISCOVER_DECISIONS = { promote: ["Promoted", "good"], reject: ["Rejected", "bad"], park: ["Parked", "warn"], repair: ["Repair", "warn"],
  redundant: ["Redundant", ""], error: ["Error", "bad"], confirmed: ["Confirmed", "good"], not_confirmed: ["Not confirmed", "bad"], insufficient: ["Holdout too short", "warn"] };
const DISCOVER_STAGES = [["propose", "Propose", "templates, random trees, mutation, your seeds"], ["screen", "Screen", "development data only"],
  ["shortlist", "Shortlist", "fitness, gates, diversity"], ["validate", "Validate", "one counted trial each"], ["review", "Review", "gates, deflated Sharpe, robustness"],
  ["promote", "Promote", "eligible for deeper research"], ["finalize", "Finalize", "reserved holdout, once"]];
const DISCOVER_STAGES_ML = { propose: "default, random and mutated recipes, your recipes", screen: "subsample of development rows",
  shortlist: "fitness, gates, prediction diversity", review: "bootstrap p-value or deflated Sharpe", promote: "eligible for deeper research" };
function discoverStages(domain) { return DISCOVER_STAGES.map(([id, title, sub]) => [id, title, domain === "ml" && DISCOVER_STAGES_ML[id] ? DISCOVER_STAGES_ML[id] : sub]); }

function discoverIcon() {
  const svg = svgEl("svg", { viewBox: "0 0 20 20" });
  svg.append(svgEl("circle", { cx: "8.5", cy: "8.5", r: "5.5" }), svgEl("path", { d: "M12.5 12.5L17 17" }), svgEl("path", { d: "M6 8.5h5M8.5 6v5" }));
  return svg;
}
function discoverButton(text, action, cls = "", attrs = {}) { return el("button", { class: `btn ${cls}`, type: "button", text, onclick: action, ...attrs }); }
function discoverTag(decision) {
  const [label, tone] = DISCOVER_DECISIONS[decision] || [decision ? String(decision) : "Pending", ""];
  return el("span", { class: `tag ${tone}`, text: label });
}
function discoverNumber(value, digits = 2, percent = false) {
  if (!finite(value)) return "—";
  return percent ? `${(value * 100).toFixed(1)}%` : Number(value).toFixed(digits);
}
function discoverDomainTitle(domain) {
  const found = ((discoverUI.catalog || {}).domains || []).find((d) => d.name === domain);
  return found ? found.title : domain === "ml" ? "Machine learning" : "Autonomous alpha discovery";
}
function discoverCurrentRow() { return discoverUI.campaigns.find((c) => c.id === discoverUI.selected) || null; }
function discoverRunningHere() { const w = discoverUI.worker; return !!(w && w.running && discoverUI.selected && w.id === discoverUI.selected); }

function initializeDiscover() {
  const button = el("button", { class: "act", "data-view": "discover", "aria-label": "Discover",
    title: "Discover · autonomous alpha discovery and machine learning (⌘5)", onclick: () => { setView("discover"); persist(); } },
    discoverIcon(), el("span", { class: "act-label", text: "Discover" }));
  const lab = $('.act[data-view="lab"]');
  lab.parentElement.insertBefore(button, lab.nextSibling);
  $("#explorer").append(el("div", { class: "xview", "data-view": "discover" },
    el("div", { class: "pane-head" }, "Campaigns", el("div", { class: "pane-actions" }, el("span", { class: "count", id: "dv-count" }),
      discoverButton("↻", () => discoverRefresh(), "ghost small", { title: "Refresh" }))),
    el("div", { class: "seg dv-domain", id: "dv-domain", role: "tablist", "aria-label": "Research domain" },
      ...[["alpha", "Alphas"], ["ml", "Models"]].map(([id, text]) => el("button", { type: "button", "data-domain": id, text, role: "tab",
        class: id === discoverUI.domain ? "active" : "", onclick: () => discoverSetDomain(id) }))),
    el("div", { class: "dv-explorer-actions" }, discoverButton("New campaign", () => discoverOpenNew(discoverUI.domain), "primary"),
      discoverButton("Demo", () => discoverRunDemo(discoverUI.domain), "", { title: "A short campaign on artificial data with a known answer" })),
    el("div", { class: "list", id: "dv-list", "aria-label": "Lab campaigns" })));
  $("#center").prepend(el("div", { class: "cview dv-workspace", "data-view": "discover" },
    el("div", { id: "dv-header", class: "dv-header" }), el("div", { id: "dv-stages", class: "dv-stages", "aria-label": "Campaign stages" }),
    el("div", { id: "dv-stats", class: "dv-stats" }), el("div", { id: "dv-actions", class: "dv-actions" }),
    el("div", { id: "dv-progress", class: "dv-progress" }), el("div", { id: "dv-content", class: "dv-content" })));
  $("#inspector").append(el("div", { class: "iview", "data-view": "discover" },
    el("div", { class: "pane-head" }, el("span", { id: "dv-setup-title", text: "New campaign" })),
    el("form", { id: "dv-form", class: "dv-form", onsubmit: (e) => { e.preventDefault(); discoverCreate(false); } }),
    el("div", { class: "inspector-foot dv-foot" },
      discoverButton("Create & run", () => discoverCreate(true), "primary block", { id: "dv-create-run" }),
      discoverButton("Create only", () => discoverCreate(false), "block small", { id: "dv-create" }))));
}

// ------------------------------------------------------------------ data
async function discoverRefresh(selectId) {
  if (discoverUI.refreshing) return;
  discoverUI.refreshing = true;
  try {
    // The catalog lists ready datasets, so it is read again on every refresh: demonstration data created on Start
    // or an import must reach the campaign form without a reload.
    const [catalog, campaigns, worker] = await Promise.all([get("lab-catalog"), get("lab-campaigns"), get("lab-worker")]);
    discoverUI.catalog = catalog; discoverUI.campaigns = campaigns; discoverUI.worker = worker;
    if (typeof homeUI !== "undefined") homeUI.campaigns = campaigns;
    if (selectId && selectId !== discoverUI.selected) { discoverUI.selected = selectId; discoverUI.candidate = null; discoverUI.candidateId = null; discoverUI.verify = null; discoverUI.composer = false; }
    if (discoverUI.selected && !campaigns.some((c) => c.id === discoverUI.selected)) discoverUI.selected = null;
    if (discoverUI.selected) {
      const detail = await get("lab-report", { id: discoverUI.selected });
      if (detail.id === discoverUI.selected) discoverUI.detail = detail;
    } else discoverUI.detail = null;
  } catch (e) { toast("Cannot load Discover", e.message, "bad"); }
  finally { discoverUI.refreshing = false; }
  discoverRender();
  discoverPoll();
}
function discoverPoll() {
  clearTimeout(discoverUI.timer);
  const w = discoverUI.worker;
  const running = !!(w && w.running);
  if (discoverUI.wasRunning && !running) {
    discoverUI.wasRunning = false;
    if (w.last_error && /already evaluated/i.test(w.last_error)) toast("Holdout already read for this candidate", `${w.last_error} The console never repeats a read; a repeated read is requested from the command line with --allow-reuse and is labelled reused.`, "bad");
    else if (w.last_error) toast("Campaign stopped with an error", w.last_error, "bad");
    else if (w.action === "finalize") {
      const finals = (w.result && w.result.finals) || [];
      toast("Holdout scored", `${finals.length} candidate${finals.length === 1 ? "" : "s"} finalized · ${finals.filter((f) => f.decision === "confirmed").length} confirmed`, "good");
    } else {
      const r = w.result || {};
      toast(w.status === "complete" ? "Campaign complete" : "Campaign paused", `${fmt.int(r.screen_trials)} screened · ${fmt.int(r.validation_trials)} validated · ${fmt.int(r.promotions)} promoted${r.budget ? " · " + r.budget : ""}`, "good");
    }
    discoverRefresh();
    return;
  }
  discoverUI.wasRunning = running;
  if (!running) return;
  discoverUI.timer = setTimeout(async () => {
    try {
      const update = await get("lab-worker");
      discoverUI.worker = update;
      if (state.view === "discover") { discoverRenderProgress(); discoverRenderStats(); discoverRenderActions(); discoverRenderList(); }
      if (typeof homeRender === "function" && state.view === "home") homeRender();
    } catch (e) { /* retry on the next tick */ }
    discoverPoll();
  }, 1000);
}
function discoverSetDomain(domain) {
  discoverUI.domain = domain;
  $$("#dv-domain button").forEach((b) => b.classList.toggle("active", b.dataset.domain === domain));
  const current = discoverCurrentRow();
  if (current && current.domain !== domain) { discoverUI.selected = null; discoverUI.detail = null; }
  discoverRenderForm(); discoverRender(); discoverSchedulePlan();
}
async function discoverSelect(id) {
  discoverUI.selected = id; discoverUI.composer = false; discoverUI.verify = null; discoverUI.candidate = null; discoverUI.candidateId = null;
  const row = discoverUI.campaigns.find((c) => c.id === id);
  if (row && row.domain !== discoverUI.domain) discoverSetDomain(row.domain);
  discoverRenderList();
  try { discoverUI.detail = await get("lab-report", { id }); } catch (e) { toast("Cannot read the campaign", e.message, "bad"); return; }
  if (discoverUI.selected === id) discoverRender();
}
function discoverOpenNew(domain) {
  if (domain && domain !== discoverUI.domain) { discoverUI.domain = domain; $$("#dv-domain button").forEach((b) => b.classList.toggle("active", b.dataset.domain === domain)); }
  discoverUI.selected = null; discoverUI.detail = null;
  setView("discover"); persist();
  discoverRenderForm(); discoverRender(); discoverSchedulePlan();
  const name = $("#dv-f-name"); if (name) name.focus();
}

// ------------------------------------------------------------------ actions
async function discoverRunDemo(domain) {
  if (discoverUI.busy) return;
  discoverUI.busy = true;
  try {
    setView("discover"); persist();
    const result = await api("lab-demo", { domain: domain || discoverUI.domain });
    toast("Demo campaign started", "Artificial data with a known answer; a quick preset. Watch the rounds arrive.", "good");
    discoverUI.worker = result.worker; discoverUI.wasRunning = true;
    await discoverRefresh(result.campaign.id);
  } catch (e) { toast("Cannot start the demo", e.message, "bad"); }
  finally { discoverUI.busy = false; }
}
async function discoverRunSelected() {
  const row = discoverCurrentRow();
  if (!row) { toast("Choose a campaign", "Select one on the left or create a new one on the right."); return; }
  if (discoverUI.worker && discoverUI.worker.running) { toast("A campaign is already running", discoverUI.worker.id || ""); return; }
  const rounds = discoverUI.rounds === "" ? undefined : Number(discoverUI.rounds);
  try {
    discoverUI.worker = await api("lab-start", { id: row.id, ...(rounds ? { rounds } : {}) });
    discoverUI.wasRunning = true;
    toast("Campaign running", rounds ? `${rounds} round${rounds === 1 ? "" : "s"} requested; budgets still apply.` : "Until its budget is used.", "good");
    discoverRender(); discoverPoll();
  } catch (e) { toast("Cannot run the campaign", e.message, "bad"); }
}
async function discoverStop() {
  try { discoverUI.worker = await api("lab-stop", {}); toast("Stopping", "The current round finishes its reserved trials, then the campaign pauses."); discoverPoll(); }
  catch (e) { toast("Cannot stop", e.message, "bad"); }
}
function discoverLineage() { const report = discoverUI.detail && discoverUI.detail.report; return (report && report.lineage) || null; }
/* Candidates the service left out of a finalize because their holdout was already read (here or in another campaign). */
function discoverSkippedToast(worker) {
  const skipped = (worker && worker.skipped_already_read) || [];
  if (skipped.length) toast("Some candidates skipped", `${skipped.length} promoted candidate${skipped.length === 1 ? " was" : "s were"} left out because the holdout was already read for ${skipped.length === 1 ? "it" : "them"} (${skipped.slice(0, 4).join(", ")}${skipped.length > 4 ? "…" : ""}); a repeated read is command-line only (--allow-reuse) and is labelled reused.`, "warn");
}
async function discoverFinalizeNonConfirmatory(row) {
  const ok = await dialog({ title: "Read the holdout as evidence only?", confirm: "Read as evidence only", danger: true,
    message: "This data's reserved holdout can no longer serve as an untouched test: it was read for confirmation or as evidence (by this campaign or another on the same lineage), or this campaign's lineage registry does not match its records. The candidates will be scored and recorded as non-confirmatory evidence for deeper research; nothing will be labelled confirmed." });
  if (!ok) return;
  try {
    discoverUI.worker = await api("lab-finalize", { id: row.id, confirm: true, non_confirmatory: true });
    discoverSkippedToast(discoverUI.worker);
    discoverUI.wasRunning = true; discoverRender(); discoverPoll();
  } catch (e) { toast("Cannot read the holdout", e.message, "bad"); }
}
/* Whether the harness would refuse a confirmatory read: retired or exposed holdout (unless this campaign is completing the
   batch it declared before the exposure), a record store that disagrees with the registry, or a registry that no longer
   matches this campaign's last snapshot of it. */
function discoverLineageBlocksConfirmation(lineage, row) {
  if (!lineage || lineage.error) return false;
  const completing = lineage.open_batch && row && lineage.open_batch.campaign_id === row.campaign_id;
  return Boolean(lineage.holdout_retired || (lineage.holdout_exposed && !completing)
    || (lineage.trials && lineage.trials.agree === false)
    || /registry_reset|registry_shrunk|ledger_broken/.test(lineage.registry_status || ""));
}
async function discoverFinalize() {
  const row = discoverCurrentRow();
  if (!row || !row.promotions) return;
  const lineage = discoverLineage();
  if (discoverLineageBlocksConfirmation(lineage, row)) { await discoverFinalizeNonConfirmatory(row); return; }
  const ok = await dialog({ title: "Score the promoted candidates on the reserved holdout?", confirm: "Finalize once", danger: true,
    message: `${row.promotions} promoted candidate${row.promotions === 1 ? "" : "s"} will be scored on data no trial has seen. Each candidate's holdout can be read once; a repeat is refused here and must be labelled as a reuse from the command line. Candidates selected after any holdout read of this data cannot be confirmed. Promotion and confirmation are evidence for deeper research, not a trading recommendation.` });
  if (!ok) return;
  try {
    discoverUI.worker = await api("lab-finalize", { id: row.id, confirm: true });
    discoverSkippedToast(discoverUI.worker);
    discoverUI.wasRunning = true; discoverRender(); discoverPoll();
  } catch (e) {
    if (/retired|non-confirmatory|registry|exposed|outside a confirmatory batch/i.test(e.message)) toast("Holdout cannot confirm for this data", e.message, "bad", { label: "Read as evidence only", run: () => discoverFinalizeNonConfirmatory(row) });
    else toast("Cannot finalize", e.message, "bad");
  }
}
async function discoverVerify() {
  const row = discoverCurrentRow();
  if (!row) return;
  try {
    const result = await get("lab-verify", { id: row.id });
    discoverUI.verify = result;
    const registry = result.registry || {};
    const registryBad = /registry_reset|registry_shrunk/.test(registry.status || "");
    const ok = result.ledger && result.ledger.ok && result.checkpoint_matches_ledger && !registryBad;
    if (registryBad) toast("Registry problem", `${registry.status === "registry_reset" ? "The lineage registry was replaced or deleted since this campaign's last snapshot" : "The lineage registry lost trials or accesses since this campaign's last snapshot"}${(registry.problems || []).length ? ": " + registry.problems.join("; ") : "."} Confirmations stay blocked until it is restored.`, "bad");
    else toast(ok ? "Ledger intact" : "Ledger problem", ok ? `${fmt.int(result.ledger.entries)} chained entries match the checkpoint${registry.status === "ok" ? " and the lineage registry matches its last snapshot" : ""}.` : (result.ledger && result.ledger.problem) || "Checkpoint and ledger counts differ.", ok ? "good" : "bad");
    discoverRenderContent();
  } catch (e) { toast("Cannot verify", e.message, "bad"); }
}
async function discoverExport(decision) {
  const row = discoverCurrentRow();
  if (!row) return;
  try {
    const result = await api("lab-export", { id: row.id, decision });
    discoverUI.lastExport = result;
    toast(`Exported ${fmt.int(result.count)} candidate${result.count === 1 ? "" : "s"}`, home(result.file), "good", { label: "Copy path", run: () => copyText(result.file, "Path copied") });
    discoverRenderContent();
  } catch (e) { toast("Cannot export", e.message, "bad"); }
}
async function discoverSubmit() {
  const row = discoverCurrentRow();
  if (!row) return;
  const payload = { id: row.id };
  if (row.domain === "alpha") { payload.expression = $("#dv-idea").value; payload.rationale = $("#dv-idea-why").value; }
  else payload.recipe = $("#dv-idea").value;
  try {
    const result = await api("lab-submit", payload);
    toast("Idea queued", result.note, "good");
    $("#dv-idea").value = "";
  } catch (e) { toast("Cannot queue the idea", e.message, "bad"); }
}

// ------------------------------------------------------------------ setup form
function discoverFormDefaults(domain) {
  const defaults = (discoverUI.catalog || {}).defaults || { cost_bps: 5, min_sharpe: 1 };
  const base = { name: "", preset: "quick", dataset: "", universe: "", cost_bps: defaults.cost_bps, min_sharpe: defaults.min_sharpe, seeds: "",
    task: "builtin:friedman1", rows: 4000, table: "", target: "", order_by: "", families: ["ridge", "gbdt", "random_forest", "mlp"],
    features: null, extra_features: "" };
  const chosen = state.lab.dataset || state.dataset || defaultPriceRow();
  if (chosen) { base.dataset = chosen.path; base.task = "panel"; }
  return domain === "ml" ? { ...base, name: "" } : base;
}
function discoverFormPayload() {
  const f = discoverUI.form, domain = discoverUI.domain;
  const payload = { domain, name: f.name, preset: f.preset };
  if (domain === "alpha") Object.assign(payload, { dataset: f.dataset, universe: f.universe, cost_bps: f.cost_bps, min_sharpe: f.min_sharpe, seeds: f.seeds });
  else {
    payload.task = f.task; payload.families = f.families;
    if (f.task.startsWith("builtin:")) payload.rows = f.rows;
    else if (f.task === "csv") Object.assign(payload, { table: f.table, target: f.target, order_by: f.order_by });
    else Object.assign(payload, { dataset: f.dataset, universe: f.universe, cost_bps: f.cost_bps, features: f.features || "default", extra_features: f.extra_features });
  }
  return payload;
}
function discoverField(label, input, hint) {
  return el("label", { class: "dv-field" }, el("span", { text: label }), input, hint ? el("small", { text: hint }) : null);
}
function discoverInput(key, attrs = {}) {
  const f = discoverUI.form;
  const input = el(attrs.rows ? "textarea" : "input", { id: `dv-f-${key}`, ...attrs });
  input.value = f[key] === undefined || f[key] === null ? "" : String(f[key]);
  input.addEventListener("input", () => { f[key] = attrs.type === "number" ? input.value : input.value; discoverSchedulePlan(); });
  return input;
}
function discoverDatasetFields() {
  const f = discoverUI.form;
  const ready = state.datasets.filter(priceEligible);
  const dataset = el("select", { id: "dv-f-dataset", onchange: () => { f.dataset = dataset.value; f.universe = ""; discoverRenderForm(); discoverSchedulePlan(); } },
    ...(ready.length ? ready.map((d) => el("option", { value: d.path, text: datasetTitle(d) })) : [el("option", { value: "", text: "No ready prices — see Start" })]));
  if (ready.some((d) => d.path === f.dataset)) dataset.value = f.dataset; else f.dataset = dataset.value;
  const row = ready.find((d) => d.path === f.dataset);
  const names = Object.keys((row && row.universes) || {}).map((n) => n.toUpperCase()).sort((a, b) => (+b.slice(3) || 0) - (+a.slice(3) || 0));
  const universe = el("select", { id: "dv-f-universe", onchange: () => { f.universe = universe.value; discoverSchedulePlan(); } },
    ...(names.length ? names.map((n) => el("option", { value: n, text: n })) : [el("option", { value: "", text: "No dated universe" })]));
  if (names.includes(f.universe)) universe.value = f.universe; else f.universe = universe.value;
  return [discoverField("Prices", dataset, ready.length ? "A ready snapshot; development, validation and holdout are cut from its calendar." : "Create demonstration data or import prices first."),
    discoverField("Universe", universe), discoverField("Cost per side", discoverInput("cost_bps", { type: "number", min: 0, max: 200, step: 0.5 }), "basis points of traded notional")];
}
function discoverFactorFields() {
  const f = discoverUI.form, library = (discoverUI.catalog || {}).ml_features || [];
  const chosen = new Set(f.features || library.map((x) => x.name));
  const summary = el("summary", { text: `Factor library · ${chosen.size} of ${library.length} selected` });
  const boxes = library.map((feature) => {
    const box = el("input", { type: "checkbox", id: `dv-feat-${feature.name}` });
    box.checked = chosen.has(feature.name);
    box.addEventListener("change", () => {
      const picked = library.filter((x) => $(`#dv-feat-${x.name}`).checked).map((x) => x.name);
      f.features = picked.length === library.length ? null : picked;
      summary.textContent = `Factor library · ${picked.length} of ${library.length} selected`;
      discoverSchedulePlan();
    });
    return el("label", { class: "dv-family dv-feature", title: feature.expression }, box, el("span", {}, el("b", { text: feature.name }), el("code", { text: feature.expression })));
  });
  const all = (on) => { library.forEach((x) => { $(`#dv-feat-${x.name}`).checked = on; }); f.features = on ? null : [];
    summary.textContent = `Factor library · ${on ? library.length : 0} of ${library.length} selected`; discoverSchedulePlan(); };
  return [el("div", { class: "dv-section", text: "Factors" }),
    el("div", { class: "dv-field" }, el("small", { text: "Each factor is rank-normalised per day; the target is the next session's cross-sectional return rank. Predictions are traded by the Alpha Lab simulator, so a recipe is judged like an alpha: Sharpe after costs, robustness, deflated Sharpe." })),
    el("details", { class: "dv-features" }, summary, el("div", { class: "dv-actions-inline" }, discoverButton("All", () => all(true), "ghost small"), discoverButton("None", () => all(false), "ghost small")), ...boxes),
    discoverField("Your own factors", discoverInput("extra_features", { rows: 3, placeholder: "rank(ts_corr(close, volume, 10))\none FASTEXPR per line", spellcheck: "false" }),
      "Added to the library as extra features; checked for syntax before the campaign is created.")];
}
function discoverRenderForm() {
  const form = $("#dv-form");
  if (!form) return;
  const domain = discoverUI.domain, catalog = discoverUI.catalog || {};
  if (!discoverUI.form.domain || discoverUI.form.domain !== domain) discoverUI.form = { domain, ...discoverFormDefaults(domain) };
  const f = discoverUI.form;
  $("#dv-setup-title").textContent = domain === "ml" ? "New model search" : "New alpha discovery";
  const presets = (catalog.presets || {})[domain] || {};
  const presetCards = el("div", { class: "dv-presets", role: "radiogroup", "aria-label": "Preset" }, ...Object.entries(presets).map(([id, p]) =>
    el("button", { type: "button", class: `dv-preset${f.preset === id ? " active" : ""}`, role: "radio", "aria-checked": f.preset === id,
      onclick: () => { f.preset = id; discoverRenderForm(); discoverSchedulePlan(); } }, el("b", { text: p.label }), el("small", { text: p.note }))));
  const children = [discoverField("Name", discoverInput("name", { type: "text", maxlength: 120, placeholder: domain === "ml" ? "Model search" : "Alpha discovery" })),
    el("div", { class: "dv-section", text: "Budget" }), presetCards];
  if (domain === "alpha") {
    children.push(el("div", { class: "dv-section", text: "Data" }), ...discoverDatasetFields(),
      el("div", { class: "dv-section", text: "Gates" }),
      discoverField("Minimum Sharpe", discoverInput("min_sharpe", { type: "number", min: 0, max: 10, step: 0.05 }), "on validation, after costs; the deflated Sharpe ratio and robustness vetoes apply too"),
      el("div", { class: "dv-section", text: "Your ideas (optional)" }),
      discoverField("Seed expressions", discoverInput("seeds", { rows: 3, placeholder: "-rank(ts_delta(close, 5))\none FASTEXPR per line", spellcheck: "false" }),
        "Screened in round one alongside the generators; later ideas can be queued while the campaign runs."));
  } else {
    const tasks = catalog.ml_tasks || [];
    const task = el("select", { id: "dv-f-task", onchange: () => { f.task = task.value; discoverRenderForm(); discoverSchedulePlan(); } },
      ...tasks.map((t) => el("option", { value: t.id, text: t.title })));
    task.value = f.task;
    const current = tasks.find((t) => t.id === f.task);
    children.push(el("div", { class: "dv-section", text: "Task" }), discoverField("Task", task, current && current.note));
    if (f.task.startsWith("builtin:")) children.push(discoverField("Rows", discoverInput("rows", { type: "number", min: 300, max: 200000, step: 100 }), "generated deterministically from the task's seed"));
    else if (f.task === "csv") children.push(discoverField("CSV file", discoverInput("table", { type: "text", placeholder: "/full/path/to/table.csv" })),
      discoverField("Target column", discoverInput("target", { type: "text", placeholder: "y" })),
      discoverField("Order column (optional)", discoverInput("order_by", { type: "text", placeholder: "date" }), "chronological development → validation → holdout split when given"));
    else children.push(...discoverDatasetFields(), ...discoverFactorFields());
    const families = catalog.ml_families || [];
    children.push(el("div", { class: "dv-section", text: "Model families" }), el("div", { class: "dv-families" }, ...families.map((fam) => {
      const box = el("input", { type: "checkbox", id: `dv-fam-${fam.id}` });
      box.checked = f.families.includes(fam.id);
      box.addEventListener("change", () => { f.families = families.filter((x) => $(`#dv-fam-${x.id}`).checked).map((x) => x.id); discoverSchedulePlan(); });
      return el("label", { class: "dv-family" }, box, el("span", { text: fam.title }));
    })));
  }
  children.push(el("div", { class: "dv-section", text: "Plan" }), el("div", { id: "dv-plan", class: "dv-plan" }));
  setChildren(form, ...children);
  discoverRenderPlan();
}
function discoverSchedulePlan() {
  clearTimeout(discoverUI.planTimer);
  const revision = ++discoverUI.planRevision;
  const payload = discoverFormPayload();
  if (!payload.dataset && (discoverUI.domain === "alpha" || payload.task === "panel")) {
    // Nothing to plan without prices; say so here instead of asking the server to refuse.
    discoverUI.plan = null; discoverUI.planError = "Choose a dataset: create demonstration data on Start or import prices first.";
    discoverRenderPlan();
    return;
  }
  discoverUI.planTimer = setTimeout(async () => {
    try {
      const planned = await api("lab-plan", payload);
      if (revision !== discoverUI.planRevision) return;
      discoverUI.plan = planned; discoverUI.planError = "";
    } catch (e) { if (revision !== discoverUI.planRevision) return; discoverUI.plan = null; discoverUI.planError = e.message; }
    discoverRenderPlan();
  }, 320);
}
function discoverRenderPlan() {
  const box = $("#dv-plan");
  if (!box) return;
  if (discoverUI.planError) { setChildren(box, el("div", { class: "dv-plan-error", text: discoverUI.planError })); $("#dv-create-run").disabled = $("#dv-create").disabled = true; return; }
  $("#dv-create-run").disabled = $("#dv-create").disabled = !discoverUI.plan;
  if (!discoverUI.plan) { setChildren(box, el("div", { class: "muted", text: "Checking the setup…" })); return; }
  const s = discoverUI.plan.summary, lim = s.limits || {};
  const rows = [[`${lim.max_rounds} rounds · ${lim.proposals_per_round} ideas each`, `${lim.validations_per_round} validations per round, ${lim.max_validation_trials} in all`]];
  if (s.periods) {
    const p = s.periods;
    rows.push(["Development", `${p.development.start} → ${p.development.end} · ${fmt.int(p.development.sessions)} sessions`],
      ["Validation", `${p.validation.start} → ${p.validation.end} · ${fmt.int(p.validation.sessions)} sessions`],
      ["Holdout (reserved)", `${p.holdout.start} → ${p.holdout.end} · ${fmt.int(p.holdout.sessions)} sessions`],
      [`Warm-up ${s.warmup} sessions`, `${s.universe} · ${s.cost_bps} bp per side`]);
  }
  if (s.rows) rows.push(["Rows", `${fmt.int(s.rows)} · 60% development, 20% validation, 20% holdout`]);
  if (s.table) rows.push(["Table", `${s.table} → ${s.target}${s.order_by ? " · ordered by " + s.order_by : ""}`]);
  if (s.families) rows.push(["Families", s.families.join(", ")]);
  if (s.features !== undefined && s.task === "panel") rows.push(["Factors", `${fmt.int(s.features)} library${s.extra_features && s.extra_features.length ? ` + ${s.extra_features.length} yours` : ""}`]);
  if (s.seeds && s.seeds.length) rows.push(["Seeds", `${s.seeds.length} expression${s.seeds.length === 1 ? "" : "s"}`]);
  setChildren(box, ...rows.map(([k, v]) => el("div", { class: "dv-plan-row" }, el("span", { text: k }), el("b", { text: v }))),
    el("details", { class: "dv-json" }, el("summary", { text: "Campaign file (what tubby-lab will run)" }), el("pre", { text: JSON.stringify(discoverUI.plan.config, null, 1) })));
}
async function discoverCreate(run) {
  if (discoverUI.busy) return;
  discoverUI.busy = true;
  try {
    const row = await api("lab-create", discoverFormPayload());
    toast("Campaign created", row.name, "good");
    discoverUI.form = {};
    await discoverRefresh(row.id);
    discoverRenderForm(); discoverSchedulePlan();
    if (run) await discoverRunSelected();
  } catch (e) { toast("Cannot create the campaign", e.message, "bad"); }
  finally { discoverUI.busy = false; }
}

// ------------------------------------------------------------------ rendering
function discoverRender() {
  if (state.view !== "discover") return;
  discoverRenderList(); discoverRenderHeader(); discoverRenderStages(); discoverRenderStats(); discoverRenderActions(); discoverRenderProgress(); discoverRenderContent();
  if (!$("#dv-form").children.length) { discoverRenderForm(); discoverSchedulePlan(); }
  else if (discoverUI.form && !discoverUI.form.dataset && state.datasets.some(priceEligible)
    && (discoverUI.domain === "alpha" || discoverUI.form.task === "panel")) { discoverRenderForm(); discoverSchedulePlan(); }  // prices arrived since the form was drawn
  const running = !!(discoverUI.worker && discoverUI.worker.running);
  $("#run").disabled = running || !discoverUI.selected || (state.job && state.job.state === "running");
  $("#cancel").disabled = !running;
  $("#sb-text").textContent = running ? `Discover · ${discoverUI.worker.action === "finalize" ? "scoring the holdout" : "round " + (discoverUI.worker.round || "…")} · ${fmt.int(discoverUI.worker.screened)} screened` : "Discover ready";
}
function discoverRenderList() {
  const list = $("#dv-list");
  if (!list) return;
  const rows = discoverUI.campaigns.filter((c) => c.domain === discoverUI.domain);
  $("#dv-count").textContent = rows.length || "";
  const w = discoverUI.worker;
  setChildren(list, ...(rows.length ? rows.map((c) => {
    const live = w && w.running && w.id === c.id;
    return el("button", { class: `dv-campaign${discoverUI.selected === c.id ? " selected" : ""}`, type: "button", onclick: () => discoverSelect(c.id) },
      el("span", { class: "dv-campaign-name", text: c.name }),
      el("span", { class: "dv-campaign-meta", text: `${live ? "Running · " : ""}${c.status} · round ${c.round}${c.limits && c.limits.max_rounds ? "/" + c.limits.max_rounds : ""} · ${fmt.int(c.promotions)} promoted` }),
      el("span", { class: "dv-campaign-date", text: `${c.preset || ""} · ${c.dataset || c.task_title || ""} · ${shortTime(c.created_at)}` }));
  }) : [el("div", { class: "list-empty", text: discoverUI.domain === "ml" ? "No model searches yet. Try the demo or set one up on the right." : "No alpha campaigns yet. Try the demo or set one up on the right." })]));
}
function discoverRenderHeader() {
  const box = $("#dv-header"), row = discoverCurrentRow(), d = discoverUI.detail;
  if (!row) {
    setChildren(box, el("div", { class: "dv-heading" }, el("div", { class: "dv-eyebrow", text: discoverDomainTitle(discoverUI.domain) }), el("h1", { text: "Discover" }),
      el("p", { text: discoverUI.domain === "ml" ? "Search training recipes under a budget with honest selection control." : "Let the machine propose, test and promote alphas under a budget, with a ledger of every trial." })));
    return;
  }
  const s = (d && d.summary) || {};
  const meta = [];
  if (s.dataset) meta.push(`${s.dataset} · ${s.universe}`);
  if (s.task_title) meta.push(s.task_title);
  if (s.periods) meta.push(`development ${s.periods.development.start} → ${s.periods.development.end} · validation → ${s.periods.validation.end} · holdout from ${s.periods.holdout.start}`);
  if (s.rows) meta.push(`${fmt.int(s.rows)} rows`);
  setChildren(box, el("div", { class: "dv-heading" }, el("div", { class: "dv-eyebrow", text: discoverDomainTitle(row.domain) }), el("h1", { text: row.name }),
    el("p", { text: meta.join(" · ") || "" }), discoverLineageLine()),
    el("div", { class: "dv-head-tags" }, el("span", { class: "tag accent", text: row.preset || "custom" }), el("span", { class: `tag ${row.running ? "good" : ""}`, text: row.running ? "running" : row.status }),
      row.holdout_accessed ? el("span", { class: "tag warn", text: "holdout read" }) : null,
      ...discoverLineageTags()));
}
/* How a final's holdout p-value was obtained and what it assumes: the harness records it per final (statistic_note);
   older finals carry the statistic name only. */
function discoverStatisticNote(confirmation) {
  if (confirmation.statistic_note) return confirmation.statistic_note;
  const notes = { "psr": "PSR; sessions treated as independent", "psr-gaussian": "Gaussian PSR; sessions treated as independent",
    "iid-bootstrap": "paired bootstrap; rows resampled independently", "moving-block-bootstrap": "moving-block bootstrap on ordered rows",
    "bootstrap": "paired bootstrap", "unavailable": "no usable statistic; entered as p = 1" };
  return notes[confirmation.statistic] || confirmation.statistic || "psr";
}
/* Lineage: one registry beside the data counts every campaign's validation trials and the single confirmatory holdout read. */
function discoverLineageTags() {
  const lineage = discoverLineage();
  if (!lineage || lineage.error) return [];
  const tags = [];
  if (lineage.holdout_retired) tags.push(el("span", { class: "tag warn", title: lineage.holdout_retired.method === "voided-after-access" ? `A holdout batch of ${lineage.holdout_retired.campaign_id || "another campaign"} was voided after a key had been read; further reads are evidence only.` : `The holdout of this data was read for confirmation by ${lineage.holdout_retired.campaign_id || "another campaign"}; further reads are evidence only.`, text: "holdout retired" }));
  if (lineage.holdout_exposed && !lineage.holdout_retired) tags.push(el("span", { class: "tag warn", title: `The holdout of this data was read outside a confirmatory batch by ${lineage.holdout_exposed.campaign_id || "another campaign"} (${lineage.holdout_exposed.key || "a candidate"}); a confirmatory batch declared after that read is refused, so reads are evidence only.`, text: "holdout exposed" }));
  if (/registry_reset|registry_shrunk/.test(lineage.registry_status || "")) tags.push(el("span", { class: "tag bad", title: lineage.registry_status === "registry_reset" ? "The lineage registry was replaced or deleted since this campaign's last snapshot of it; confirmatory finalize is refused until it is restored." : "The lineage registry lost trials, accesses, a retirement or an exposure since this campaign's last snapshot of it; confirmatory finalize is refused.", text: lineage.registry_status === "registry_reset" ? "registry reset" : "registry shrunk" }));
  if (lineage.seeded) tags.push(el("span", { class: "tag", title: `The registry's entry for this data was seeded from a record store by ${lineage.seeded.campaign_id || "a campaign"} (${fmt.int(lineage.seeded.trials)} trials, ${fmt.int(lineage.seeded.accesses)} holdout reads carried over): its baseline is that store's word.`, text: "registry seeded" }));
  if (lineage.trials && lineage.trials.agree === false) tags.push(el("span", { class: "tag bad", title: `Record store counts ${fmt.int(lineage.trials.store)} validation trials, the lineage registry ${fmt.int(lineage.trials.registry)}; confirmatory finalize is refused until the registry is used.`, text: "lineage disagrees" }));
  if (lineage.non_confirmatory) tags.push(el("span", { class: "tag", title: "This campaign reads the holdout as evidence only.", text: "non-confirmatory" }));
  if (lineage.canonical === false) tags.push(el("span", { class: "tag warn", title: `This campaign uses a registry other than this machine's canonical one (${lineage.registry}); its confirmations are marked registry_canonical: false.`, text: "custom registry" }));
  return tags;
}
function discoverLineageLine() {
  const lineage = discoverLineage();
  if (!lineage) return null;
  if (lineage.error) return el("p", { class: "muted", text: `Lineage registry unavailable: ${lineage.error}` });
  const trials = lineage.trials || {};
  const parts = [`${fmt.int(trials.registry)} validation trial${trials.registry === 1 ? "" : "s"} by other campaigns in this registry`,
    `holdout confirmation ${lineage.confirmation || "batch"} at level ${fmt.num(lineage.level, 3)} (nominal p-values)`,
    lineage.canonical === false ? "custom registry" : "this machine's canonical registry"];
  if ((lineage.foreign_campaigns || []).length) parts.push(`${fmt.int(lineage.foreign_campaigns.length)} other campaign${lineage.foreign_campaigns.length === 1 ? "" : "s"} counted from the registry`);
  if (lineage.registry_status && !/^(ok|no_snapshot)$/.test(lineage.registry_status)) parts.push(`registry check: ${lineage.registry_status}`);
  return el("p", { class: "muted dv-lineage", title: `Registry: ${lineage.registry}`, text: `Lineage · ${parts.join(" · ")}` });
}
function discoverStage(row) {
  if (!row) return "propose";
  if (discoverRunningHere()) return discoverUI.worker.action === "finalize" ? "finalize" : "validate";
  if (row.holdout_accessed) return "finalize";
  if (row.promotions > 0) return "promote";
  if (row.round > 0) return "review";
  return "propose";
}
function discoverRenderStages() {
  const row = discoverCurrentRow(), current = discoverStage(row);
  const stages = discoverStages(row ? row.domain : discoverUI.domain), currentIndex = stages.findIndex(([id]) => id === current);
  setChildren($("#dv-stages"), ...stages.map(([id, title, sub], i) => el("div", { class: `dv-stage${id === current ? " current" : i < currentIndex ? " done" : ""}`, title: sub,
      "aria-current": id === current ? "step" : null, "aria-label": `${i + 1}. ${title}${id === current ? " (current stage)" : i < currentIndex ? " (completed)" : ""}` },
    el("span", { class: "dv-stage-number", text: String(i + 1) }), el("span", {}, el("b", { text: title }), el("small", { text: sub })))));
}
function discoverStat(label, value, note) { return el("div", { class: "dv-stat" }, el("span", { text: label }), el("strong", { text: value }), note ? el("small", { text: note }) : null); }
function discoverRenderStats() {
  const box = $("#dv-stats"), row = discoverCurrentRow();
  if (!row) { setChildren(box); box.hidden = true; return; }
  box.hidden = false;
  const live = discoverRunningHere() ? discoverUI.worker : null;
  const report = discoverUI.detail && discoverUI.detail.report, funnel = (report && report.funnel) || {};
  const lim = row.limits || {};
  const round = live && live.round ? live.round : row.round;
  const ledger = report && report.ledger;
  setChildren(box,
    discoverStat("Round", `${fmt.int(round)}${lim.max_rounds ? " / " + lim.max_rounds : ""}`, row.budget || (row.status === "complete" ? "budget used" : "")),
    discoverStat("Screened", fmt.int(live ? row.screen_trials + live.screened : row.screen_trials), "development data only"),
    discoverStat("Validated", `${fmt.int(live ? row.validation_trials + live.validated : row.validation_trials)}${lim.max_validation_trials ? " / " + lim.max_validation_trials : ""}`, "counted trials"),
    discoverStat("Promoted", fmt.int(live ? row.promotions + live.promoted : row.promotions), funnel.redundant ? `${fmt.int(funnel.redundant)} redundant skipped` : ""),
    discoverStat("Finalized", `${fmt.int(funnel.finalized || 0)}`, funnel.finalized ? `${fmt.int(funnel.confirmed || 0)} confirmed on holdout` : "holdout untouched"),
    discoverStat("Ledger", ledger ? (ledger.ok ? "intact" : "broken") : "—", ledger && ledger.ok ? `${fmt.int(ledger.entries)} chained entries` : (ledger && ledger.problem) || ""));
}
function discoverRenderActions() {
  const box = $("#dv-actions"), row = discoverCurrentRow();
  if (!row) { setChildren(box); box.hidden = true; return; }
  box.hidden = false;
  const running = !!(discoverUI.worker && discoverUI.worker.running), here = discoverRunningHere();
  const rounds = el("select", { id: "dv-rounds", "aria-label": "Rounds to run", onchange: () => { discoverUI.rounds = rounds.value; } },
    el("option", { value: "1", text: "1 round" }), el("option", { value: "3", text: "3 rounds" }), el("option", { value: "", text: "Until the budget is used" }));
  rounds.value = discoverUI.rounds;
  const spent = row.status === "complete";
  setChildren(box,
    el("div", { class: "dv-run-group" }, discoverButton(spent ? "Budget used" : "Run", discoverRunSelected, "primary", { disabled: running || spent }), rounds),
    discoverButton("Stop", discoverStop, "", { disabled: !here }),
    discoverButton("Finalize on holdout…", discoverFinalize, "", { disabled: running || !row.promotions, title: "Scores promoted candidates once on the reserved period" }),
    discoverButton("Verify ledger", discoverVerify, "", { disabled: !row.campaign_id }),
    discoverButton(row.domain === "alpha" ? "Export promoted to Alpha Lab" : "Export promoted recipes", () => discoverExport("promote"), "", { disabled: !row.campaign_id }),
    discoverButton(discoverUI.composer ? "Hide idea box" : "Queue your own idea", () => { discoverUI.composer = !discoverUI.composer; discoverRenderContent(); }, "ghost", { disabled: !row.campaign_id }),
    spent ? discoverButton("Add budget", discoverExtend, "", { disabled: running, title: "Raise the round, validation and promotion limits by one preset step; gates stay the same" }) : null,
    row.domain === "alpha" && row.promotions ? discoverButton("Open promoted in Alpha Lab", discoverOpenInLab, "", { title: "Paste the promoted expressions into Alpha Lab on this campaign's prices" }) : null);
}
async function discoverExtend() {
  const row = discoverCurrentRow();
  if (!row) return;
  try {
    const detail = await api("lab-extend", { id: row.id });
    const lim = detail.limits || {};
    toast("Budget added", `Now ${lim.max_rounds} rounds and ${lim.max_validation_trials} validation trials. Gates and identity are unchanged.`, "good");
    await discoverRefresh(row.id);
  } catch (e) { toast("Cannot add budget", e.message, "bad"); }
}
async function discoverOpenInLab() {
  const row = discoverCurrentRow(), report = discoverUI.detail && discoverUI.detail.report;
  if (!row || !report) return;
  const promoted = report.leaderboard.filter((r) => r.decision === "promote" && r.expression).map((r) => r.expression);
  if (!promoted.length) { toast("Nothing promoted yet", "Run more rounds first."); return; }
  const path = discoverUI.detail.summary && discoverUI.detail.summary.dataset_path;
  const dataset = path && state.datasets.find((d) => d.path === path && priceEligible(d));
  if (dataset && state.lab.datasetPath !== dataset.path) await loadDataset(dataset.path, true, "lab");
  $("#lab-text").value = promoted.join("\n"); onLabInput();
  setView("lab"); persist();
  toast(`${promoted.length} promoted alpha${promoted.length === 1 ? "" : "s"} in Alpha Lab`, "Press ⌘R to simulate them with your own settings; the campaign's validation used its sealed periods.", "good");
}
async function discoverOpenCandidate(hypothesisId) {
  const row = discoverCurrentRow();
  if (!row) return;
  if (discoverUI.candidateId === hypothesisId) { discoverUI.candidate = null; discoverUI.candidateId = null; discoverRenderContent(); return; }
  discoverUI.candidateId = hypothesisId;
  try { discoverUI.candidate = await get("lab-candidate", { id: row.id, hypothesis: hypothesisId }); }
  catch (e) { toast("Cannot read the candidate", e.message, "bad"); discoverUI.candidateId = null; return; }
  discoverRenderContent();
  const drawer = $("#dv-candidate"); if (drawer) drawer.scrollIntoView({ block: "nearest", behavior: "smooth" });
}
function discoverRenderProgress() {
  const box = $("#dv-progress");
  if (!discoverRunningHere()) { setChildren(box); box.hidden = true; return; }
  const w = discoverUI.worker;
  box.hidden = false;
  setChildren(box, el("div", { class: "dv-progress-head" }, el("span", { class: "sb-dot busy" }),
    el("b", { text: w.action === "finalize" ? "Scoring the reserved holdout…" : w.status === "starting" ? "Starting the worker and loading prices…" : `Round ${w.round || "…"} in progress` }),
    el("span", { class: "muted", text: w.action === "finalize" ? "one holdout access per promoted candidate" : w.rounds_requested ? `${w.rounds_requested} round${w.rounds_requested === 1 ? "" : "s"} requested` : "until the budget is used" })),
    el("pre", { class: "dv-log", text: (w.log || []).slice(-12).join("\n") || (w.action === "finalize" ? "Refitting on development and validation, then scoring the reserved period…" : "Waiting for the first round…") }));
}
function discoverRenderContent() {
  const box = $("#dv-content"), row = discoverCurrentRow();
  if (!row) {
    setChildren(box, el("div", { class: "dv-empty" },
      el("h2", { text: "How a campaign works" }),
      el("ol", {}, ...discoverStages(discoverUI.domain).map(([, title, sub]) => el("li", {}, el("b", { text: title }), ` — ${sub}`))),
      el("h3", { text: "Why you can trust what it promotes" }),
      el("ul", {},
        el("li", { text: "Screening sees development data only; validation is a separate, counted trial; the holdout is read once, by you." }),
        el("li", { text: "Every trial is appended to a hash-chained ledger. Deleting or editing a line is detectable." }),
        el("li", { text: "Selection is priced: the deflated Sharpe ratio (or a Bonferroni-adjusted bootstrap for models) rises with every trial on the same data, in this and earlier campaigns." }),
        el("li", { text: "Changing a threshold starts a new campaign; it never rescues an old candidate." })),
      el("div", { class: "dv-empty-actions" }, discoverButton("Run a demo", () => discoverRunDemo(discoverUI.domain), "primary"), discoverButton("Set up a campaign", () => discoverOpenNew(discoverUI.domain)))));
    return;
  }
  const d = discoverUI.detail, report = d && d.report;
  const children = [];
  if (discoverUI.composer) children.push(discoverComposer(row));
  if (discoverUI.verify && discoverUI.verify.id === row.id) {
    const v = discoverUI.verify, registryBad = /registry_reset|registry_shrunk/.test((v.registry && v.registry.status) || ""), ok = v.ledger.ok && v.checkpoint_matches_ledger && !registryBad;
    if (registryBad) children.push(el("div", { class: "dv-note bad", text: `Lineage registry ${v.registry.status === "registry_reset" ? "replaced or deleted" : "shrunk"} since this campaign's last snapshot${(v.registry.problems || []).length ? ": " + v.registry.problems.join("; ") : ""}. Confirmatory finalize is refused until it is restored; evidence-only reads remain possible.` }));
    children.push(el("div", { class: `dv-note ${ok ? "good" : "bad"}`, text: ok ? `Ledger intact: ${fmt.int(v.ledger.entries)} entries, head ${String(v.ledger.head || "").slice(0, 12)}…, screen and validation counts match the checkpoint.` : `Ledger problem: ${v.ledger.problem || "counts differ from the checkpoint"}` }));
  }
  if (discoverUI.lastExport && discoverUI.lastExport.file.includes(row.id.split("/")[1])) children.push(el("div", { class: "dv-note", text: `Exported ${fmt.int(discoverUI.lastExport.count)} to ${home(discoverUI.lastExport.file)}. ${discoverUI.lastExport.note}` }));
  if (!report) children.push(el("div", { class: "dv-empty small" }, el("p", { text: (d && d.note) || "Run the campaign to see its candidates." })));
  else {
    children.push(discoverAnalysis(row, report));
    if (discoverUI.candidate && discoverUI.candidateId && discoverUI.candidate.id === row.id) children.push(discoverCandidateDrawer(row, discoverUI.candidate));
    children.push(discoverLeaderboard(row, report));
  }
  setChildren(box, ...children);
}
function discoverChart(container, draw) { drawWhenSized(container, () => draw(container)); return container; }
function discoverAnalysis(row, report) {
  const cards = [];
  const rounds = report.rounds || [];
  if (rounds.length) {
    const chart = el("div", { class: "dv-chart" });
    cards.push(el("div", { class: "dv-card" }, el("div", { class: "dv-card-head" }, el("b", { text: "Progress by round" }),
      el("span", { class: "muted", text: row.domain === "ml" && row.task === "panel" ? "best screen fitness (mean daily rank IC)" : row.domain === "ml" ? "best screen fitness (negative inner loss)" : "best screen fitness (weakest development fold Sharpe, minus complexity)" })),
      discoverChart(chart, (c) => barChart(c, { labels: rounds.map((r) => `R${r.round}`), values: rounds.map((r) => r.best_fitness), format: (v) => fmt.num(v, Math.abs(v) < 0.1 ? 3 : 2), valueLabels: rounds.length <= 16,
        color: () => css("--cat-1"), label: "Best fitness by round",
        tooltip: (i) => `Round ${rounds[i].round}: ${rounds[i].screened} screened, ${rounds[i].validated} validated, ${rounds[i].promoted} promoted${rounds[i].submitted ? `, ${rounds[i].submitted} of yours` : ""}` })),
      el("div", { class: "dv-round-strip" }, ...rounds.map((r) => el("span", { class: `dv-round${r.promoted ? " hit" : ""}`, title: `Round ${r.round}`, text: `${r.round}: ${r.screened}/${r.validated}/${r.promoted}` })),
        el("small", { class: "muted", text: "round: screened / validated / promoted" }))));
  }
  const factors = report.factor_importance;
  if (factors && factors.rows && factors.rows.length) {
    const top = factors.rows.slice(0, 15);
    const chart = el("div", { class: "dv-chart auto" });
    cards.push(el("div", { class: "dv-card" }, el("div", { class: "dv-card-head" }, el("b", { text: "Which factors the promoted trees used" }),
      el("span", { class: "muted", text: `training-period split-gain share · ${factors.source} · ${fmt.int(factors.models)} tree model${factors.models === 1 ? "" : "s"}${factors.rows.length > top.length ? ` · top ${top.length} of ${factors.rows.length}` : ""}` })),
      discoverChart(chart, (c) => hbarChart(c, { labels: top.map((r) => r.name), values: top.map((r) => r.weight), format: (v) => fmt.pct(v, 1),
        color: () => css("--cat-3"), label: "Share of split gain", tooltip: (i) => `${top[i].name}: ${fmt.pct(top[i].weight, 1)} of split gain` })),
      el("small", { class: "muted", text: factors.note })));
  }
  return el("div", { class: `dv-analysis${cards.length === 2 ? " two" : ""}` }, ...cards);
}
function discoverMetricRows(sections, keys) {
  const names = Object.keys(sections).filter((k) => sections[k] && typeof sections[k] === "object");
  if (!names.length) return null;
  const present = keys.filter(([key]) => names.some((n) => sections[n][key] !== undefined && sections[n][key] !== null));
  return el("table", { class: "dv-table compact" }, el("thead", {}, el("tr", {}, el("th", { text: "Metric" }), ...names.map((n) => el("th", { text: n })))),
    el("tbody", {}, ...present.map(([key, label, kind]) => el("tr", {}, el("td", { text: label }), ...names.map((n) => {
      const v = sections[n][key];
      return el("td", { class: "mono", text: typeof v === "number" ? (kind === "pct" ? fmt.pct(v, 1) : kind === "int" ? fmt.int(v) : fmt.num(v, kind === "p" ? 4 : 2)) : v == null ? "—" : String(v) });
    })))));
}
function discoverCandidateDrawer(row, c) {
  const h = c.hypothesis || {}, x = c.experiment || {}, review = c.review || {}, final = c.final;
  const metrics = x.metrics || {};
  const sections = { development: metrics.development, validation: metrics.validation, holdout: final && final.holdout };
  const signalKeys = [["days", "Sessions", "int"], ["sharpe", "Sharpe"], ["returns", "Return", "pct"], ["turnover", "Turnover", "pct"], ["drawdown", "Max drawdown", "pct"], ["fitness", "Fitness"], ["ic_mean", "Mean rank IC", "p"], ["ic_ir", "IC IR"], ["ic_in_sample", "In-sample IC", "p"], ["margin_bps", "Margin (bp)"], ["coverage", "Coverage", "pct"]];
  const tabularKeys = [["rows", "Rows", "int"], ["score", "Score", "p"], ["baseline", "Constant baseline", "p"], ["improvement", "Improvement", "p"], ["relative_improvement", "Relative improvement", "pct"], ["p_value", "p-value", "p"], ["ci_low", "CI low", "p"], ["ci_high", "CI high", "p"], ["noise_floor", "Noise floor", "p"], ["metric", "Metric"]];
  const keys = sections.validation && "score" in (sections.validation || {}) ? tabularKeys : signalKeys;
  const robust = x.robustness || {}, checks = review.checks || {}, dsr = checks.deflated_sharpe || {};
  const facts = [];
  if (robust.half_sharpes) facts.push(["Validation halves", robust.half_sharpes.map((v) => fmt.num(v)).join(" / ")]);
  if (robust.cost_stress_sharpe !== undefined) facts.push([`Sharpe at ${robust.cost_stress_multiplier || 2}× costs`, fmt.num(robust.cost_stress_sharpe)]);
  if (robust.perturbations && robust.perturbations.length) facts.push(["Window ×0.75 / ×1.33", robust.perturbations.map((p) => fmt.num(p.sharpe)).join(" / ")]);
  if (dsr.dsr !== undefined) facts.push(["Deflated Sharpe", `${fmt.num(dsr.dsr, 3)} · benchmark daily Sharpe ${fmt.num(dsr.benchmark_sharpe, 3)} over ${fmt.int(dsr.n_trials)} trials, ${fmt.int(dsr.observations)} sessions`]);
  if (checks.p_value !== undefined) facts.push(["Bonferroni", `p ${fmt.num(checks.p_value, 4)} vs level ${fmt.num(checks.bonferroni_level, 5)} for ${fmt.int(checks.validation_trials)} trials`]);
  if (checks.perturbation_retention != null) facts.push(["Perturbation retention", fmt.pct(checks.perturbation_retention, 0)]);
  const confirmation = final && final.confirmation;
  if (confirmation && confirmation.method === "batch-holm") facts.push(["Holdout confirmation", `Holm across a batch of ${fmt.int(confirmation.batch)} · nominal p ${fmt.num(confirmation.p_value, 5)} (${discoverStatisticNote(confirmation)}) vs step-down level ${fmt.num(confirmation.holm_level, 5)}, family level ${fmt.num(confirmation.level, 3)} · ${confirmation.holm_rejected ? "below its Holm level" : "not below its Holm level"}`]);
  else if (confirmation && confirmation.method === "per-candidate") facts.push(["Holdout confirmation", "per-candidate: the domain's own holdout verdict, no familywise control across finalists"]);
  else if (confirmation && confirmation.method) facts.push(["Holdout confirmation", String(confirmation.method)]);
  else if (final) facts.push(["Holdout confirmation", "domain verdict only: this final predates batch confirmation and was not Holm-tested"]);
  if (final && final.non_confirmatory) facts.push(["Holdout read", "non-confirmatory: evidence only, never confirmed"]);
  if (final && final.lineage_exposed) facts.push(["Holdout exposure", "this data's holdout had been read outside a confirmatory batch before this read"]);
  if (final && final.registry_status && !/^(ok|no_snapshot)$/.test(final.registry_status)) facts.push(["Registry at finalize", String(final.registry_status)]);
  if (final && final.lineage_seeded) facts.push(["Registry baseline", "seeded from a record store"]);
  const model = x.model || {};
  const importance = Array.isArray(model.feature_importance) ? model.feature_importance : null;
  let chart = null;
  if (importance && importance.length) {
    const total = importance.reduce((a, b) => a + (finite(b) && b > 0 ? b : 0), 0) || 1;
    const pairs = importance.map((w, i) => [c.feature_names[i] || `feature ${i}`, w / total]).sort((a, b) => b[1] - a[1]).slice(0, 15);
    chart = el("div", { class: "dv-chart auto" });
    discoverChart(chart, (box) => hbarChart(box, { labels: pairs.map((p) => p[0]), values: pairs.map((p) => p[1]), format: (v) => fmt.pct(v, 1), color: () => css("--cat-3"), label: "Feature importance", tooltip: (i) => `${pairs[i][0]}: ${fmt.pct(pairs[i][1], 1)}` }));
  }
  const params = h.recipe && h.recipe.params ? Object.entries(h.recipe.params).map(([k, v]) => `${k} = ${JSON.stringify(v)}`).join(" · ") : "";
  return el("section", { class: "dv-candidate", id: "dv-candidate" },
    el("div", { class: "dv-card-head" }, el("div", {}, el("b", { text: h.title || h.expression || "Candidate" }), el("div", { class: "muted", text: `${h.generator || ""} · round ${h.round ?? "—"} · screen fitness ${fmt.num(h.screen_fitness, 3)}${h.post_holdout ? " · proposed after a holdout read" : ""}` })),
      el("div", { class: "dv-actions-inline" }, discoverTag(review.decision), final ? discoverTag(final.decision) : null, discoverButton("Close", () => { discoverUI.candidate = null; discoverUI.candidateId = null; discoverRenderContent(); }, "ghost small"))),
    h.expression ? el("code", { class: "dv-expression", text: h.expression }) : null,
    params ? el("div", { class: "dv-params", text: params }) : null,
    review.reason ? el("p", { class: "dv-reason", text: review.reason }) : null,
    final && final.reason ? el("p", { class: "dv-reason", text: `Holdout: ${final.reason}${final.reused ? " (reused access)" : ""}` }) : null,
    el("div", { class: "dv-candidate-grid" },
      el("div", {}, discoverMetricRows(sections, keys) || el("p", { class: "muted", text: x.error ? `Evaluation error: ${x.error}` : "No metrics recorded." }),
        facts.length ? el("dl", { class: "dv-facts" }, ...facts.map(([k, v]) => el("div", {}, el("dt", { text: k }), el("dd", { text: v })))) : null),
      chart ? el("div", {}, el("b", { class: "dv-chart-title", text: "Feature importance of this recipe" }), chart) : null),
    x.settings && Object.keys(x.settings).length ? el("small", { class: "muted", text: Object.entries(x.settings).map(([k, v]) => `${k} ${v}`).join(" · ") }) : null);
}
function discoverComposer(row) {
  const alpha = row.domain === "alpha";
  return el("section", { class: "dv-composer" }, el("b", { text: alpha ? "Queue your own expression" : "Queue your own recipe" }),
    el("p", { text: alpha ? "Screened at the start of the next round like any generated idea, with the same gates and the same budget." : 'JSON such as {"family": "gbdt", "params": {"depth": 4, "learning_rate": 0.05}}; unknown params fall back to the family defaults.' }),
    el("textarea", { id: "dv-idea", rows: 2, spellcheck: "false", placeholder: alpha ? "-rank(ts_corr(close, volume, 10))" : '{"family": "gbdt", "params": {}}' }),
    alpha ? el("input", { id: "dv-idea-why", type: "text", maxlength: 500, placeholder: "Why might this work? (kept with the record)" }) : null,
    el("div", { class: "dv-actions-inline" }, discoverButton("Queue for the next round", discoverSubmit, "primary small")));
}
function discoverLeaderboard(row, report) {
  const rows = report.leaderboard || [];
  const alpha = row.domain === "alpha";
  // Panel (factor-analysis) models are scored by the Alpha Lab simulator, so their rows carry signal metrics like alphas do.
  const signal = alpha || rows.some((r) => r.validation && r.validation.sharpe !== undefined);
  const head = signal ? ["Decision", alpha ? "Candidate" : "Recipe", "Round", "From", "Sharpe", "Returns", "Turnover", "DSR", "Holdout"]
    : ["Decision", "Recipe", "Round", "From", "Score", "Baseline", "Improvement", "p-value", "Holdout"];
  const body = rows.map((r) => {
    const v = r.validation || {};
    const holdout = r.final ? `${r.final.non_confirmatory ? "Evidence only" : (DISCOVER_DECISIONS[r.final.decision] || [r.final.decision])[0]}${r.final.reused ? " · reused" : ""}` : "—";
    const cells = signal
      ? [discoverNumber(v.sharpe), discoverNumber(v.returns, 1, true), discoverNumber(v.turnover, 1, true), discoverNumber(r.deflated_sharpe, 3)]
      : [discoverNumber(v.score, 4), discoverNumber(v.baseline, 4), discoverNumber(v.relative_improvement, 1, true), discoverNumber(v.p_value, 4)];
    return el("tr", { class: `${r.decision === "promote" ? "promoted" : ""}${discoverUI.candidateId === r.hypothesis_id ? " selected" : ""}`, title: "Show this candidate's full evidence",
      onclick: () => discoverOpenCandidate(r.hypothesis_id) }, el("td", {}, discoverTag(r.decision)),
      el("td", { class: "dv-expr" }, el("code", { text: r.expression || r.title || "" }), r.reason ? el("small", { text: r.reason }) : null),
      el("td", { class: "mono", text: r.round == null ? "—" : String(r.round) }), el("td", { text: r.generator || "" }),
      ...cells.map((c) => el("td", { class: "mono", text: c })), el("td", { text: holdout }));
  });
  return el("section", { class: "dv-board" },
    el("div", { class: "dv-board-head" }, el("b", { text: `${fmt.int(rows.length)} validated candidate${rows.length === 1 ? "" : "s"} · click one for its full evidence` }),
      el("span", { class: "muted", text: signal ? `Validation-period ${alpha ? "metrics" : "simulator metrics of each model's predictions"} after costs; DSR is the deflated Sharpe ratio over every trial on this data.` : "Validation score versus the constant baseline; p-values are paired-bootstrap, Bonferroni-adjusted at review." })),
    rows.length ? el("div", { class: "dv-table-wrap" }, el("table", { class: "dv-table" }, el("thead", {}, el("tr", {}, ...head.map((h) => el("th", { text: h })))), el("tbody", {}, ...body)))
      : el("p", { class: "muted", text: "No candidate reached validation yet." }),
    el("p", { class: "dv-foot", text: "Promotion and confirmation are evidence for deeper research, not a trading recommendation." }));
}
