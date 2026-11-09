"use strict";
/* Bounded immutable evidence browser. Preview and execution require explicit actions. */
const workflowUI = { rows: [], selected: null, detail: null, preview: null, previewKey: null,
  revision: 0, refreshing: false, previewing: false, submitting: false, recovering: false };
const workflowPortfolio = { revision: 0, preview: null, key: null, previewing: false, submitting: false };
const workflowChronology = { revision: 0, source: null, preview: null, key: null, protocolId: null,
  plan: null, view: null, selection: null, records: new Map(), chosen: new Set(), page: 0, candidateInputs: [],
  loading: false, submitting: false, stale: false };
const workflowPhases = ["development", "validation", "final"];
const workflowLimitKeys = ["gross", "net", "name", "group_gross", "group_net", "turnover", "liquidity"];

function workflowButton(text, action, id) {
  return el("button", { class: "btn small", type: "button", text, id, onclick: action });
}
function workflowShortId(id) {
  return id ? String(id).slice(0, 8) + (String(id).length > 8 ? "…" : "") : "not assigned";
}
function workflowAttemptLabel(a) {
  return typeof a.display_label === "string" && a.display_label.trim() ? workflowText(a.display_label.trim()) :
    ({ "lab-evaluation": "Lab evaluation", "candidate-validation": "Candidate validation", "portfolio-validation": "Portfolio validation" })[a.kind] || a.kind;
}
function workflowAttemptMeta(a) {
  return `${a.kind} · ${a.status === "reused" ? "reused (cache)" : a.status} · ${a.created_at || "Date not recorded"}`;
}
function workflowRecorded(record) {
  return Object.fromEntries(Object.entries(record).filter(([, value]) => value !== undefined && value !== null && value !== ""));
}
function workflowNumber(value, percent = false) {
  return typeof value === "number" && Number.isFinite(value)
    ? (percent ? value * 100 : value).toLocaleString("en-US", { maximumFractionDigits: 2 }) + (percent ? "%" : "") : null;
}
function workflowPeriod(period) {
  return period?.start && period?.end ? `${period.start} → ${period.end}` : period?.start || period?.end;
}
function workflowSummary(record) {
  return el("div", { class: "wf-summary" }, ...Object.entries(workflowRecorded(record)).map(([key, value]) =>
    el("div", {}, el("small", { text: key.replaceAll("_", " ") }), el("strong", { text: workflowText(value) }))));
}
function workflowBrief(values) {
  const items = Array.isArray(values) ? values : values ? [values] : [];
  return items.slice(0, 8).map(workflowText).join("; ") +
    (items.length > 8 ? `; ${items.length - 8} more items in the bounded details / structured CLI` : "");
}
function workflowOmissions(values) {
  return el("p", { class: "wf-omissions" }, el("strong", { text: "Not modeled: " }),
    workflowBrief(values) || "Omitted costs are not recorded. Review the execution details before running.");
}
function workflowText(value) {
  if (value == null) return "Not recorded";
  if (typeof value === "object") return Array.isArray(value) ? `${value.length} items` : `${Object.keys(value).length} fields`;
  const text = String(value);
  return text.length > 800 ? text.slice(0, 800) + "… (text shortened; full record available through the structured CLI)" : text;
}
function workflowFacts(record, budget = { left: 80 }, depth = 0) {
  const keys = Object.keys(record || {}), shown = keys.slice(0, Math.max(0, Math.min(16, budget.left)));
  budget.left -= shown.length;
  return el("dl", { class: "kv wf-facts" }, ...shown.map(key => [
    el("dt", { text: workflowText(key.replaceAll("_", " ")) }), el("dd", {}, workflowValue(record[key], budget, depth + 1))]),
    keys.length > shown.length ? [el("dt", { text: "Additional fields" }),
      el("dd", { text: `${keys.length - shown.length} omitted here; full record available through the structured CLI.` })] : []);
}
function workflowValue(value, budget = { left: 80 }, depth = 0) {
  if (value && typeof value === "object" && (depth >= 3 || budget.left <= 0))
    return workflowDisclosure(`${workflowText(value)} · inspect bounded details`, value);
  if (Array.isArray(value)) {
    if (!value.length) return "None recorded";
    const count = Math.min(value.length <= 8 ? value.length : 3, budget.left, value.length);
    budget.left -= count;
    return el("div", {}, el("small", { text: `${value.length} items${value.length > count ? ` · showing first ${count}; full record available through the structured CLI` : ""}` }),
      el("ul", {}, ...value.slice(0, count).map(v => el("li", {}, workflowValue(v, budget, depth + 1)))));
  }
  if (value && typeof value === "object") return workflowFacts(value, budget, depth);
  return workflowText(value);
}
function workflowDisclosure(title, record, render = workflowValue) {
  const content = el("div");
  let loaded = false;
  // Closed disclosures do not traverse calendars, symbol arrays or nested evidence.
  return el("details", { class: "wf-card", ontoggle: event => {
    if (!event.currentTarget.open || loaded) return;
    loaded = true; setChildren(content, render(record));
  } }, el("summary", { text: title }), content);
}
function workflowMessage(text) { $("#wf-message").textContent = text; }
function workflowInvalidate(message = "Preview required for the selected attempt and execution settings.") {
  workflowUI.revision++;
  workflowUI.preview = null; workflowUI.previewKey = null; workflowUI.previewing = false;
  setChildren($("#wf-preview")); workflowMessage(message); workflowUpdateButtons();
}
function workflowEligible() {
  return workflowChronologyEligible() && !workflowObservationFiltered(workflowUI.detail?.result);
}
function workflowChronologyEligible() {
  const a = workflowUI.detail?.attempt;
  return a?.id === workflowUI.selected && a.kind === "lab-evaluation" &&
    ["completed", "reused"].includes(a.status) && !!a.result_digest;
}
function workflowObservationFiltered(result) {
  return result?.settings?.evaluation_sessions != null || result?.evaluation?.policy === "observation-filter-v1";
}
function workflowEvaluationWarning(result) {
  if (!workflowObservationFiltered(result)) return null;
  return el("section", { class: "wf-omissions", role: "note" },
    el("h4", { text: "Observation-filter warning · selected observations" }),
    el("p", { text: "Metrics describe selected daily portfolio return observations, not a continuous investment return or a masked trading policy. Qlib validation is unavailable; chronology selection remains available." }),
    result.evaluation?.policy ? el("p", { text: `Evaluation policy: ${workflowText(result.evaluation.policy)}` }) : null,
    result.evaluation?.metrics_meaning ? el("p", { text: workflowText(result.evaluation.metrics_meaning) }) : null);
}
function workflowParams() {
  const number = id => {
    const input = $(`#wf-${id}`), value = input.value.trim();
    if (!value || !input.checkValidity() || !Number.isFinite(Number(value))) throw new Error("Enter valid cash, per-side cost and annual borrow values.");
    return Number(value);
  };
  return { attempt_id: workflowUI.selected, initial_cash: number("cash"),
    cost_per_side: number("cost") / 10000, annual_borrow_rate: number("borrow") / 100 };
}
function workflowKey() { try { return JSON.stringify(workflowParams()); } catch { return null; } }
function workflowBusy() {
  return workflowUI.submitting || workflowPortfolio.submitting || workflowChronology.submitting || workflowUI.recovering || workflowUI.refreshing ||
    state.job?.state === "running" || !!researchUI.overview?.worker?.running;
}
function workflowCanRun() {
  return workflowEligible() && !workflowBusy() && !workflowUI.previewing &&
    !!workflowUI.preview && workflowUI.previewKey === workflowKey();
}
function workflowUpdateButtons() {
  if (!$("#wf-run")) return;
  $("#wf-run").disabled = !workflowCanRun();
  $("#wf-preview-button").disabled = !workflowEligible() || workflowBusy() || workflowUI.previewing;
  $("#wf-refresh").disabled = workflowUI.refreshing || workflowUI.submitting || workflowPortfolio.submitting || workflowChronology.submitting || workflowUI.recovering;
  for (const id of ["cash", "cost", "borrow"]) $(`#wf-${id}`).disabled = workflowUI.submitting || workflowPortfolio.submitting || workflowChronology.submitting || workflowUI.recovering;
  workflowPortfolioButtons();
  workflowChronologyButtons();
  if (state.view === "evidence") {
    $("#run-label").textContent = "Run selected validation";
    $("#run").disabled = !workflowCanRun();
  }
}
function initializeWorkflow() {
  $("#explorer").append(el("div", { class: "xview", "data-view": "evidence" },
    el("div", { class: "pane-head" }, "Submitted attempts", workflowButton("Refresh", refreshWorkflow, "wf-refresh")),
    el("p", { class: "wf-note", id: "wf-count", text: "Refresh to load the latest 100 attempts, including failures and cancellations." }),
    el("div", { class: "list", id: "wf-attempts", "aria-label": "Submitted attempt history" })));
  const field = (label, id, value, min, step) => el("label", {}, el("span", { text: label }),
    el("input", { id: `wf-${id}`, type: "number", value, min, step, required: true,
      oninput: () => workflowInvalidate("Execution settings changed. Preview again before running.") }));
  $("#center").prepend(el("section", { class: "cview wf-workspace", "data-view": "evidence", "aria-label": "Evidence" },
    el("h2", { text: "Evidence · validation and portfolios" }),
    el("p", { class: "wf-note", text: "Each submission retains its outcome, including cache reuse and failure. Qlib validation produces a separate account result. Refresh history explicitly." }),
    el("p", { class: "wf-note", text: "Older library entries lack immutable attempt history. Rerun through Alpha Lab to record an attempt." }),
    el("div", { id: "wf-detail" }, el("p", { text: "Select an attempt to inspect its evidence and parents." })),
    el("section", { class: "wf-card" }, el("h3", { text: "Selected validation settings" }),
      el("p", { text: "Saved candidate settings are read only. These values apply to the new signed Qlib validation." }),
      el("div", { class: "wf-fields" }, field("Initial account cash (USD)", "cash", 100000, .01, "any"),
        field("Trading cost per side (basis points)", "cost", 5, 0, "any"),
        field("Annual stock borrow (%)", "borrow", .4, 0, "any")),
      el("div", { class: "wf-actions" }, workflowButton("Preview selected validation", previewWorkflow, "wf-preview-button"),
        workflowButton("Run selected validation", runWorkflowValidation, "wf-run"))),
    el("p", { id: "wf-message", role: "status", "aria-live": "polite" }), el("div", { id: "wf-preview" }),
    workflowPortfolioSection(), workflowChronologySection(),
    el("section", { class: "wf-card" }, el("h3", { text: "Unfinished attempts" }),
      el("p", { text: "Check recorded process ownership and mark interrupted attempts when the owner is confirmed gone. Live or unverifiable owners remain pending. This does not retry simulations." }),
      workflowButton("Recover unfinished attempts · check ownership", recoverWorkflow, "wf-recover"),
      el("div", { id: "wf-recovery", role: "status", "aria-live": "polite" }))));
  workflowUpdateButtons();
}
async function refreshWorkflow() {
  if (workflowUI.refreshing || workflowUI.submitting || workflowPortfolio.submitting || workflowChronology.submitting || workflowUI.recovering) return;
  workflowUI.refreshing = true;
  workflowUI.detail = null;
  workflowInvalidate("Refreshing history. Select an attempt again to preview.");
  workflowPortfolioInvalidate("History refresh invalidated the portfolio preview. Review the selections and preview again.");
  try {
    workflowUI.rows = await get("workflow-attempts", { limit: 100 });
    workflowUI.selected = null;
    setChildren($("#wf-detail"), el("p", { text: "Select an attempt to inspect." }));
    renderWorkflowList();
    workflowPortfolioOptions();
    $("#wf-count").textContent = `${workflowUI.rows.length} recent attempts (limit 100). Every status is included. Refresh explicitly for new submissions.`;
    workflowMessage(workflowUI.rows.length ? "History refreshed. Select an attempt to inspect." : "History refreshed. No recorded attempts.");
  } catch (e) { workflowMessage(e.message); }
  finally { workflowUI.refreshing = false; workflowUpdateButtons(); }
}
function renderWorkflowList() {
  setChildren($("#wf-attempts"), ...workflowUI.rows.map(a => el("button", {
    class: `wf-attempt${a.id === workflowUI.selected ? " active" : ""}`, type: "button",
    "aria-pressed": String(a.id === workflowUI.selected), onclick: () => selectWorkflowAttempt(a.id)
  }, el("strong", { text: workflowAttemptLabel(a) }),
  el("small", { text: workflowAttemptMeta(a) }),
  el("small", { text: `Attempt ${workflowShortId(a.id)}${a.candidate_id ? " · Candidate " + workflowShortId(a.candidate_id) : ""}` }))));
  if (!workflowUI.rows.length) setChildren($("#wf-attempts"), el("p", { class: "wf-note", text: "No recorded attempts. Submit through Lab, then refresh." }));
}
async function selectWorkflowAttempt(id) {
  if (workflowUI.submitting || workflowPortfolio.submitting || workflowChronology.submitting || workflowUI.refreshing || workflowUI.recovering) return;
  workflowUI.selected = id; workflowUI.detail = null;
  workflowInvalidate("Loading selected attempt…"); renderWorkflowList();
  const revision = workflowUI.revision;
  setChildren($("#wf-detail"));
  try {
    const detail = await get("workflow-detail", { id });
    if (revision !== workflowUI.revision) return;
    if (detail.attempt?.id !== id) throw new Error("Returned attempt does not match the selection. Refresh history.");
    workflowUI.detail = detail; renderWorkflowDetail();
    workflowMessage(workflowEligible() ? "Preview the selected Lab result before validation."
      : workflowChronologyEligible() && workflowObservationFiltered(detail.result)
        ? "Selected-observation result: Qlib validation is unavailable. Select it within its existing chronology protocol; a new protocol needs an unfiltered source." : `Attempt ${detail.attempt.status}.`);
  } catch (e) { if (revision === workflowUI.revision) workflowMessage(e.message); }
  finally { workflowUpdateButtons(); }
}
function renderWorkflowDetail() {
  const { attempt: a, result: r } = workflowUI.detail;
  const candidate = r?.candidate || r?.portfolio || r || {};
  const filtered = workflowObservationFiltered(r);
  const parents = a.parents || [];
  const summary = r?.summary || {};
  const metrics = workflowRecorded({
    Sharpe: workflowNumber(summary.sharpe), "Annual return (half-book)": workflowNumber(summary.returns, true),
    "Mean daily turnover": workflowNumber(summary.turnover, true), "Drawdown (half-book)": workflowNumber(summary.drawdown, true),
    Fitness: workflowNumber(summary.fitness),
  });
  const facts = workflowRecorded({ expression: candidate.expression || a.request?.expression,
    period: workflowPeriod(candidate.period || summary),
    qualification: r?.qualification?.status || candidate.qualification?.status, decision: r?.decision });
  const saved = workflowRecorded({ settings: candidate.settings, period: candidate.period,
    summary: Object.keys(summary).length ? summary : null, definitions: r?.metric_definitions || r?.definitions,
    evaluation: r?.evaluation, qualification: r?.qualification || candidate.qualification });
  setChildren($("#wf-detail"), el("h3", { text: `${a.kind} · ${a.status}` }),
    a.error ? el("div", { class: "wf-error", role: "alert" }, workflowValue(a.error.message || a.error)) : null,
    Object.keys(facts).length ? workflowFacts(facts) : null,
    workflowEvaluationWarning(r),
    a.kind === "lab-evaluation" && Object.keys(metrics).length ? el("section", { class: "wf-card" },
      el("h4", { text: filtered ? "Selected-observation metrics · saved result" : "Lab screening metrics · saved result" }),
      workflowSummary({ ...metrics, [filtered ? "Selected observations" : "Sessions"]: workflowNumber(summary.days) }),
      el("p", { class: "wf-note", text: "Configured Lab turnover costs included; stock borrow excluded. Metrics are rounded." })) : null,
    r?.run_path ? workflowButton(a.kind === "portfolio-validation" ? "Open saved portfolio run" : "Open saved Qlib run",
      () => { setView("research"); openRun(r.run_path); }) : null,
    a.kind === "portfolio-validation" && summary.portfolio_risk ? workflowPortfolioRisk(summary.portfolio_risk) : null,
    a.kind === "portfolio-validation" && r?.portfolio ? workflowDisclosure("Portfolio allocation, risk and reconciliation",
      { request: r?.portfolio?.request || a.request, policy: r?.portfolio?.policy, risk: r?.risk || r?.portfolio?.risk_contract,
        qualification: r?.qualification, reconciliation: r?.reconciliation, summary: r?.summary }) : null,
    Object.keys(saved).length ? workflowDisclosure("Saved settings and metrics · read only", {
      ...saved, ...(Object.keys(metrics).length ? { metric_meaning: filtered
        ? "Metrics use selected daily observations. Annualization uses 252 times their mean daily result; drawdown follows the compressed selected-observation PnL series. Annual return and drawdown use half-book capital. This is not a continuous investment return."
        : "Lab annual return and drawdown use half-book capital. Sharpe uses daily PnL and 252 sessions/year. Qlib uses net account returns." } : {}),
    }) : null,
    workflowDisclosure("Provenance · full identities and submitted record", { attempt_id: a.id, candidate_id: a.candidate_id,
      result_digest: a.result_digest, created_at: a.created_at, error: a.error, parents, request: a.request, source: a.source,
      artifacts: a.artifacts, result: r }),
    parents.length ? workflowDisclosure(`Parent attempts · ${parents.length}`, parents, rows =>
      el("div", { class: "wf-actions" }, ...rows.slice(0, 8).map(id => workflowButton(id, () => selectWorkflowAttempt(id))),
        rows.length > 8 ? el("span", { text: `${rows.length - 8} more parents in the structured CLI.` }) : null)) : null);
}
async function previewWorkflow() {
  if (!workflowEligible() || workflowBusy() || workflowUI.previewing) return;
  workflowInvalidate("Checking recorded candidate compatibility…");
  const revision = workflowUI.revision;
  try {
    const params = workflowParams(), key = JSON.stringify(params);
    workflowUI.previewing = true; workflowUpdateButtons();
    const preview = await api("workflow-preview", params);
    if (revision !== workflowUI.revision) return;
    if (key !== workflowKey()) throw new Error("Selection or execution settings changed. Preview again before running.");
    if (preview.source_attempt_id !== params.attempt_id || preview.result_digest !== workflowUI.detail.attempt.result_digest ||
        preview.candidate_id !== workflowUI.detail.attempt.candidate_id) throw new Error("Preview identity differs from the selected evidence. Refresh history and select again.");
    const execution = preview.execution;
    if (!execution || ["initial_cash", "cost_per_side", "annual_borrow_rate"].some(k => execution[k] !== params[k]))
      throw new Error("Preview execution settings differ from the requested values. Preview again.");
    workflowUI.preview = preview; workflowUI.previewKey = key;
    renderWorkflowPreview(preview);
    workflowMessage("Preview ready. Review differences, qualification and omitted costs before running.");
  } catch (e) { if (revision === workflowUI.revision) { workflowUI.preview = null; workflowUI.previewKey = null; workflowMessage(e.message); } }
  finally { if (revision === workflowUI.revision) workflowUI.previewing = false; workflowUpdateButtons(); }
}
function renderWorkflowPreview(p) {
  setChildren($("#wf-preview"), el("section", { class: "wf-card" }, el("h3", { text: "Lab → Qlib validation preview" }),
    workflowFacts({ expression: p.expression, period: workflowPeriod(p.period), qualification: p.qualification?.status || "Not recorded" }),
    el("p", { text: "Lab: adjusted close-to-close, fixed book. Qlib: the saved close targets fill at the following session open in a cash/share account; transforms are already applied." }),
    workflowSummary({ "Initial cash (USD)": workflowNumber(p.execution.initial_cash), "Cost per side (bp)": workflowNumber(p.execution.cost_per_side * 10000),
      "Annual borrow": workflowNumber(p.execution.annual_borrow_rate, true), "Annual cash rate": workflowNumber(p.execution.annual_cash_rate, true) }),
    workflowOmissions(p.execution.omissions),
    el("p", { class: "wf-note", text: p.qualification?.note || "Recorded evidence only: source-byte checks and numerical reproduction occur in the worker." }),
    workflowDisclosure("Qualification checks and limitations", p.qualification),
    workflowDisclosure("Execution, saved settings and differences", { execution: p.execution, settings: p.settings, differences: p.differences }),
    workflowDisclosure("Provenance · full identities and sources", { source_attempt_id: p.source_attempt_id, result_digest: p.result_digest,
      candidate_id: p.candidate_id, dataset: p.dataset, source_bindings: p.source_bindings })));
}
async function runWorkflowValidation() {
  if (!workflowCanRun()) { workflowMessage("A current preview is required, and no other task may be running."); return; }
  const params = workflowParams();
  workflowUI.submitting = true; workflowUpdateButtons();
  workflowPortfolioInvalidate("A candidate validation was submitted. Preview the portfolio again before running.");
  try {
    const job = await api("workflow-validate", params);
    workflowInvalidate(`Validation submitted${job.attempt_id ? " · attempt " + job.attempt_id : ""}. Stop is available in the toolbar while the worker is cancellable.`);
    startPolling(job);
  } catch (e) { workflowInvalidate(e.message); }
  finally { workflowUI.submitting = false; workflowUpdateButtons(); }
}

function workflowPortfolioSection() {
  const changed = () => workflowPortfolioInvalidate("Portfolio inputs changed. Preview again before running.");
  const field = (label, id, value = "", attrs = {}) => el("label", {}, el("span", { text: label }),
    el("input", { id: `wf-p-${id}`, type: "number", min: 0, step: "any", value, oninput: changed, ...attrs }));
  // Name selects from the visible caption, not the wrapping label's option text.
  const leg = number => el("div", { class: "wf-fields wf-leg" },
    el("label", { for: `wf-p-leg${number}` }, el("span", { id: `wf-p-leg${number}-label`, text: `Validated candidate ${number}` }),
      el("select", { id: `wf-p-leg${number}`, "aria-labelledby": `wf-p-leg${number}-label`, onchange: changed },
        el("option", { value: "", text: number === 1 ? "Refresh history, then choose a validation…" : "No second candidate" }))),
    field(`Capital allocation ${number} (fraction)`, `weight${number}`, number === 1 ? 1 : 0, { max: 1 }));
  return el("section", { class: "wf-card", id: "wf-portfolio" }, el("h3", { text: "Portfolio · one shared account" }),
    el("p", { text: "Choose one or two completed candidate validations from the refreshed history. Fractions must sum to at most 1; the remainder stays unallocated. Each run uses one fresh historical simulation account with the saved targets." }),
    leg(1), leg(2),
    el("div", { class: "wf-fields" }, field("Account ID", "account", "", { type: "text", placeholder: "Enter an account identity" }),
      field("Initial account cash (USD)", "cash", 100000, { min: .01 }),
      field("Trading cost per side (basis points)", "cost", 5), field("Annual stock borrow (%)", "borrow", .4)),
    el("details", { class: "wf-card" }, el("summary", { text: "Optional independent limits" }),
      el("p", { text: "All limits are fractions: 1 = 100%. Empty = disabled; 0 = enabled with a zero limit. Limits check joint account exposure and trades. Missing required group or liquidity inputs must be refused." }),
      el("div", { class: "wf-fields" }, ...workflowLimitKeys.map(key => el("label", { for: `wf-p-limit-${key}` },
        el("span", { id: `wf-p-limit-${key}-label`, text: ({ gross: "Gross exposure / equity", net: "Absolute net exposure / equity", name: "Each name: absolute exposure / equity",
          group_gross: "Each group: gross exposure / equity", group_net: "Each group: absolute net / equity",
          turnover: "Full traded notional / equity (no half factor)", liquidity: "Traded shares / daily reference volume shares" })[key] }),
        el("input", { id: `wf-p-limit-${key}`, "aria-labelledby": `wf-p-limit-${key}-label`,
          "aria-describedby": `wf-p-limit-state-${key}`, type: "number", min: 0, step: "any", placeholder: "Disabled (empty)", oninput: changed }),
        el("small", { id: `wf-p-limit-state-${key}`, text: "Disabled" })))),
      el("div", { class: "wf-fields" }, field("Classification overlay path (required for group limits)", "group-path", "", { type: "text" }),
        el("label", { for: "wf-p-group-level" }, el("span", { id: "wf-p-group-level-label", text: "Group level (required for group limits)" }),
          el("select", { id: "wf-p-group-level", "aria-labelledby": "wf-p-group-level-label", onchange: changed }, el("option", { value: "", text: "Choose a group level…" }),
            ...["sector", "industry", "country"].map(value => el("option", { value, text: value }))))),
      workflowButton("Use Lab classification path", () => {
        if (workflowBusy() || !state.lab?.groupOverlay) return;
        $("#wf-p-group-path").value = state.lab.groupOverlay; changed();
      }, "wf-p-use-group"),
      el("p", { id: "wf-p-group-note", text: "Group limits are disabled. No classification source is assumed." })),
    el("p", { id: "wf-p-limits-summary", class: "wf-note" }),
    el("div", { class: "wf-actions" }, workflowButton("Preview portfolio", previewWorkflowPortfolio, "wf-p-preview-button"),
      workflowButton("Run portfolio validation", runWorkflowPortfolio, "wf-p-run")),
    el("p", { id: "wf-p-message", role: "status", "aria-live": "polite", text: "Choose a completed validation and enter an account ID, then preview." }),
    el("div", { id: "wf-p-preview" }));
}
function workflowPortfolioOptions() {
  const rows = workflowUI.rows.filter(a => a.kind === "candidate-validation" && a.status === "completed" && a.result_digest);
  for (const number of [1, 2]) {
    const node = $(`#wf-p-leg${number}`), saved = node.value;
    setChildren(node, el("option", { value: "", text: number === 1 ? "Choose a completed validation…" : "No second candidate" }),
      ...rows.map(a => el("option", { value: a.id,
        text: `${workflowAttemptLabel(a)} · ${workflowAttemptMeta(a)} · ${workflowShortId(a.id)}` })));
    node.value = rows.some(a => a.id === saved) ? saved : "";
  }
}
function workflowPortfolioParams() {
  const read = key => $(`#wf-p-${key}`).value.trim();
  const number = key => {
    const text = read(key), value = Number(text);
    if (!text || !Number.isFinite(value) || value < 0) throw new Error(`Enter a finite nonnegative value for ${key}.`);
    return value;
  };
  const ids = [read("leg1"), read("leg2")];
  if (!ids[0] || ids[0] === ids[1]) throw new Error("Choose one or two distinct completed candidate validations.");
  const validation_ids = ids.filter(Boolean);
  for (const id of validation_ids) {
    if (!workflowUI.rows.some(a => a.id === id && a.kind === "candidate-validation" && a.status === "completed" && a.result_digest))
      throw new Error("Selected validation is unavailable or incomplete. Refresh history and select again.");
  }
  const coefficients = validation_ids.map((id, i) => number(`weight${i + 1}`));
  if (coefficients.some(v => v > 1) || coefficients.reduce((sum, v) => sum + v, 0) > 1)
    throw new Error("Capital allocation fractions must sum to at most 1.");
  const account_id = read("account");
  if (!account_id) throw new Error("Enter one account ID for this portfolio.");
  const initial_cash = number("cash");
  if (initial_cash <= 0) throw new Error("Initial account cash must be positive.");
  const limits = Object.fromEntries(workflowLimitKeys.map(key => [key, read(`limit-${key}`) === "" ? null : number(`limit-${key}`)]));
  const group = limits.group_gross !== null || limits.group_net !== null;
  const group_overlay = group ? read("group-path") : null, group_level = group ? read("group-level") : null;
  if (group && (!group_overlay || !["sector", "industry", "country"].includes(group_level)))
    throw new Error("Group limits require an explicit classification overlay path and sector, industry or country level. Unknown groups are not assumed.");
  return { validation_ids, coefficients, account_id, initial_cash, cost_per_side: number("cost") / 10000,
    annual_borrow_rate: number("borrow") / 100, limits, group_overlay, group_level };
}
function workflowPortfolioPins(params) {
  return params.validation_ids.map(id => ({ validation_id: id,
    validation_result_digest: workflowUI.rows.find(a => a.id === id)?.result_digest }));
}
function workflowCanonical(value) {
  if (Array.isArray(value)) return value.map(workflowCanonical);
  if (value && typeof value === "object") return Object.fromEntries(Object.keys(value).sort().map(k => [k, workflowCanonical(value[k])]));
  return value;
}
function workflowPortfolioKey() {
  try { const request = workflowPortfolioParams(); return JSON.stringify(workflowCanonical({ request, pins: workflowPortfolioPins(request) })); }
  catch { return null; }
}
function workflowPortfolioRequestMatches(canonical, requested) {
  if (!canonical || typeof canonical !== "object") return false;
  // Accept only separator/trailing-slash normalization, never a changed source.
  const path = value => typeof value === "string" ? value.replaceAll("\\", "/").replace(/\/+$/, "") : value;
  return path(canonical.group_overlay) === path(requested.group_overlay) &&
    JSON.stringify(workflowCanonical({ ...canonical, group_overlay: requested.group_overlay })) ===
      JSON.stringify(workflowCanonical(requested));
}
function workflowPortfolioInvalidate(message) {
  workflowPortfolio.revision++; workflowPortfolio.preview = null; workflowPortfolio.key = null; workflowPortfolio.previewing = false;
  setChildren($("#wf-p-preview")); $("#wf-p-message").textContent = message;
  workflowUpdateButtons();
}
function workflowPortfolioCanRun() {
  return !workflowBusy() && !workflowPortfolio.previewing && !!workflowPortfolio.preview &&
    workflowPortfolio.key === workflowPortfolioKey();
}
function workflowPortfolioButtons() {
  if (!$("#wf-p-run")) return;
  const busy = workflowBusy(), locked = workflowUI.submitting || workflowPortfolio.submitting || workflowChronology.submitting || workflowUI.recovering;
  $("#wf-p-run").disabled = !workflowPortfolioCanRun();
  // Leave preview clickable for local input validation and its precise explanation.
  $("#wf-p-preview-button").disabled = busy || workflowPortfolio.previewing;
  $("#wf-recover").disabled = busy;
  for (const key of ["leg1", "leg2", "weight1", "weight2", "account", "cash", "cost", "borrow", ...workflowLimitKeys.map(k => `limit-${k}`)])
    $(`#wf-p-${key}`).disabled = locked;
  $("#wf-p-weight2").disabled = locked || !$("#wf-p-leg2").value;
  const enabled = workflowLimitKeys.filter(k => $(`#wf-p-limit-${k}`).value.trim() !== "");
  for (const key of workflowLimitKeys) $(`#wf-p-limit-state-${key}`).textContent =
    enabled.includes(key) ? `Enabled · ${$(`#wf-p-limit-${key}`).value} (fraction)` : "Disabled";
  $("#wf-p-limits-summary").textContent = `Independent limits: ${enabled.length ? enabled.join(", ") + " enabled" : "all seven disabled"}. Empty fields disable only that limit.`;
  const group = enabled.includes("group_gross") || enabled.includes("group_net");
  $("#wf-p-group-path").disabled = $("#wf-p-group-level").disabled = locked || !group;
  $("#wf-p-use-group").hidden = !state.lab?.groupOverlay;
  $("#wf-p-use-group").disabled = busy || !group;
  $("#wf-p-group-note").textContent = group
    ? "Group limits require this explicit path and level; missing or unknown groups must be refused. Preview checks compatibility."
    : "Group limits are disabled. The classification path and level are not used.";
}
async function previewWorkflowPortfolio() {
  if (workflowBusy() || workflowPortfolio.previewing) return;
  workflowPortfolioInvalidate("Checking selected immutable validations and portfolio settings…");
  const revision = workflowPortfolio.revision;
  try {
    const request = workflowPortfolioParams(), key = workflowPortfolioKey(), pins = workflowPortfolioPins(request);
    workflowPortfolio.previewing = true; workflowUpdateButtons();
    const preview = await api("workflow-portfolio-preview", request);
    if (revision !== workflowPortfolio.revision) return;
    if (key !== workflowPortfolioKey()) throw new Error("Portfolio selection or settings changed. Preview again.");
    if (preview.schema_version !== 1 || !workflowPortfolioRequestMatches(preview.request, request))
      throw new Error("Canonical portfolio request differs from the selected inputs. Check the settings and use an absolute classification path, then preview again.");
    if (!Array.isArray(preview.legs) || preview.legs.length !== pins.length ||
        preview.legs.some((leg, i) => leg.validation_id !== pins[i].validation_id ||
          leg.validation_result_digest !== pins[i].validation_result_digest || !leg.result_digest || !leg.candidate_attempt_id))
      throw new Error("Portfolio preview immutable validation results differ from the selection. Refresh history and select again.");
    for (const field of ["execution", "risk_contract", "qualification", "policy", "period"])
      if (!preview[field] || typeof preview[field] !== "object" || Array.isArray(preview[field]))
        throw new Error(`Portfolio preview is missing ${field}. Preview again when the service is ready.`);
    if (["initial_cash", "cost_per_side", "annual_borrow_rate"].some(k => preview.execution[k] !== request[k]))
      throw new Error("Portfolio execution settings differ from the canonical request. Preview again.");
    workflowPortfolio.preview = preview; workflowPortfolio.key = key;
    renderWorkflowPortfolioPreview(preview);
    $("#wf-p-message").textContent = "Preview ready. Review allocation, qualification, execution and independent risk checks before running.";
  } catch (e) {
    if (revision === workflowPortfolio.revision) { workflowPortfolio.preview = null; workflowPortfolio.key = null; $("#wf-p-message").textContent = e.message; }
  } finally {
    if (revision === workflowPortfolio.revision) workflowPortfolio.previewing = false;
    workflowUpdateButtons();
  }
}
function renderWorkflowPortfolioPreview(p) {
  const enabled = workflowLimitKeys.filter(key => p.request.limits[key] !== null);
  const disabled = workflowLimitKeys.filter(key => p.request.limits[key] === null);
  const omissions = [...new Set([...(p.risk_contract.omissions || []), ...(p.execution.omissions || [])])];
  setChildren($("#wf-p-preview"), el("section", { class: "wf-card" }, el("h4", { text: `Portfolio preview · ${p.request.account_id}` }),
    workflowFacts({ period: workflowPeriod(p.period), qualification: p.qualification.status || "Not recorded" }),
    workflowSummary({ "Initial cash (USD)": workflowNumber(p.request.initial_cash),
      "Cost per side (bp)": workflowNumber(p.request.cost_per_side * 10000), "Annual borrow": workflowNumber(p.request.annual_borrow_rate, true),
      "Unallocated capital": workflowNumber(Math.max(0, 1 - p.request.coefficients.reduce((sum, v) => sum + v, 0)), true) }),
    ...p.legs.map((leg, i) => el("div", { class: "wf-portfolio-leg" },
      el("strong", { text: `${leg.expression} · allocation ${workflowNumber(p.request.coefficients[i], true)}` }),
      el("small", { text: `Source ${workflowShortId(leg.candidate_attempt_id)} · Qualification: ${leg.qualification?.status || "Not recorded"}` }))),
    el("p", { text: `Limits (fractions): ${enabled.length ? enabled.map(key => `${key} ≤ ${p.request.limits[key]}`).join("; ") : "none enabled"}. Disabled: ${disabled.length ? disabled.join(", ") : "none"}.` }),
    el("p", { class: "wf-note", text: p.risk_contract.violation_policy || p.risk_contract.missing_inputs || "Review the independent risk contract before running." }),
    workflowOmissions(omissions),
    el("p", { class: "wf-note", text: p.qualification.large_source_and_target_artifacts
      ? `Source and target artifacts: ${p.qualification.large_source_and_target_artifacts}.`
      : "Allocation retains the underlying data qualification; preview is not a completed simulation." }),
    workflowDisclosure("Qualification and independent risk details", { qualification: p.qualification, risk_contract: p.risk_contract }),
    workflowDisclosure("Execution and allocation policy", { execution: p.execution, policy: p.policy }),
    workflowDisclosure("Provenance · full bindings and request", { legs: p.legs, request: p.request, dataset: p.dataset })));
}
async function runWorkflowPortfolio() {
  if (!workflowPortfolioCanRun()) { $("#wf-p-message").textContent = "A current matching portfolio preview is required, and no other task may be running."; return; }
  const request = JSON.parse(JSON.stringify(workflowPortfolio.preview.request));
  workflowPortfolio.submitting = true; workflowUpdateButtons();
  workflowInvalidate("A portfolio validation was submitted. Preview the candidate again before running.");
  try {
    const job = await api("workflow-portfolio-run", request);
    workflowPortfolioInvalidate(`Portfolio validation submitted${job.attempt_id ? " · attempt " + job.attempt_id : ""}. Use the existing Stop control while cancellable.`);
    startPolling(job);
  } catch (e) { workflowPortfolioInvalidate(e.message); }
  finally { workflowPortfolio.submitting = false; workflowUpdateButtons(); }
}
async function recoverWorkflow() {
  if (workflowBusy()) return;
  workflowUI.recovering = true;
  workflowChronologyStale("Recovery requested. Refresh the protocol before further chronology actions.");
  workflowInvalidate("Recovery requested. Refresh history after it finishes.");
  workflowPortfolioInvalidate("Recovery invalidated the portfolio preview. Refresh history and preview again.");
  setChildren($("#wf-recovery"), "Checking unfinished attempt owners…");
  try {
    const result = await api("workflow-recover", {});
    setChildren($("#wf-recovery"), workflowValue(result), el("p", { text: "Recovery finished. Refresh history explicitly to see updated attempts." }));
  } catch (e) { setChildren($("#wf-recovery"), el("p", { class: "wf-error", text: e.message })); }
  finally { workflowUI.recovering = false; workflowUpdateButtons(); }
}

function workflowPortfolioRisk(risk) {
  const counts = risk.counts || {};
  return el("section", { class: "wf-card" }, el("h4", { text: "Portfolio risk outcomes" }),
    el("p", { text: "Completed means the simulation finished; it does not mean limits passed." }),
    workflowSummary({
      Sessions: workflowNumber(counts.sessions), "Blocked rebalances": workflowNumber(counts.blocked_rebalances),
      "Partial / unfilled native orders": workflowNumber(counts.native_partial_or_unfilled_orders),
      "Pre-order blocked": workflowNumber(counts.pre_order?.blocked), "Pre-order missing inputs": workflowNumber(counts.pre_order?.missing_inputs),
      "Post-fill violations": workflowNumber(counts.post_fill?.blocked), "Post-fill missing inputs": workflowNumber(counts.post_fill?.missing_inputs),
    }), workflowOmissions(risk.model_omissions),
    workflowDisclosure("Risk checks · counts and provenance", risk));
}

// Chronology is explicitly operated. Reading a protocol never reads final numerical results.
function workflowChronologySection() {
  const changed = () => workflowChronologyInvalidate("Declaration changed. Preview again before declaring.");
  const input = (label, id, type = "text") => el("label", {}, el("span", { text: label }),
    el("input", { id: `wf-c-${id}`, type, oninput: changed }));
  return el("details", { class: "wf-card", id: "wf-chronology" },
    el("summary", { text: "Chronological evaluation · declare → trial → freeze → final" }),
    el("p", { class: "wf-note", text: "Lab screening only, using sealed source/settings. No automatic search or final evaluation. No claim of economic point-in-time readiness." }),
    el("div", { class: "wf-actions" },
      workflowButton("Use selected Lab candidate · new declaration", workflowChronologyUseSource, "wf-c-use"),
      workflowButton("Open selected protocol", openWorkflowChronology, "wf-c-open")),
    el("div", { id: "wf-c-source" }, "Select a completed or reused Lab attempt, then pin it here."),
    el("div", { id: "wf-c-plan-fields" },
      el("p", { class: "wf-note", text: "Choose non-overlapping trading-session windows in order. Each phase needs at least 21 saved sessions and sufficient warmup; preview checks the saved calendar." }),
      el("div", { class: "wf-fields wf-windows" }, ...workflowPhases.map(phase =>
        el("fieldset", {}, el("legend", { text: phase }), input("Start", `${phase}-start`, "date"), input("End", `${phase}-end`, "date")))),
      el("label", { class: "wf-c-text" }, el("span", { text: "Selection rule · state the criterion before trials" }),
        el("textarea", { id: "wf-c-rule", rows: 2, oninput: changed, placeholder: "Describe the metric, comparison and selection criterion." })),
      el("details", { class: "wf-card" }, el("summary", { text: "Advanced · explicit sample information intervals (optional)" }),
        el("p", { text: "Leave empty for calendar windows: information-overlap checks are unavailable. Optional JSON list fields: id, session, information_start, information_end; optional feature_available_at. Maximum 1,000 samples here; use the structured CLI for larger records." }),
        el("p", { class: "wf-omissions", text: "Explicit intervals require complete coverage across all phases. Signals and warmup use the full window; only retained daily portfolio return observations are scored. Filtered results are not a continuous investment return or a masked trading policy. Qlib validation refuses filtered results; chronological candidate selection remains available." }),
        el("label", { class: "wf-c-text" }, el("span", { text: "Sample intervals (JSON list, not a source/settings override)" }),
          el("textarea", { id: "wf-c-samples", rows: 3, oninput: changed }))),
      el("div", { class: "wf-actions" },
        workflowButton("Preview windows", previewWorkflowChronology, "wf-c-preview-button"),
        workflowButton("Declare reviewed protocol", declareWorkflowChronology, "wf-c-declare"))),
    el("p", { id: "wf-c-message", role: "status", "aria-live": "polite" }),
    el("div", { id: "wf-c-preview" }),
    el("section", { id: "wf-c-protocol", hidden: true },
      el("div", { class: "wf-actions" }, workflowButton("Refresh protocol · outcomes and access labels", refreshWorkflowChronology, "wf-c-refresh")),
      el("div", { id: "wf-c-outcomes" }),
      el("section", { id: "wf-c-trial-controls" }, el("h4", { text: "Submit supplied expressions" }),
        el("div", { class: "wf-fields" }, el("label", { for: "wf-c-phase" }, el("span", { id: "wf-c-phase-label", text: "Trial phase" }),
          el("select", { id: "wf-c-phase", "aria-labelledby": "wf-c-phase-label", onchange: workflowUpdateButtons },
            el("option", { value: "development", text: "Development" }), el("option", { value: "validation", text: "Validation" })))),
        el("label", { class: "wf-c-text" }, el("span", { text: "Expressions · one per line, up to 50; duplicates remain recorded trials" }),
          el("textarea", { id: "wf-c-expressions", rows: 3, oninput: workflowUpdateButtons })),
        workflowButton("Run supplied phase trials", runWorkflowChronologyTrial, "wf-c-trial")),
      el("section", { id: "wf-c-freeze-controls" }, el("h4", { id: "wf-c-candidates-title", text: "Freeze evaluated candidates" }),
        el("p", { class: "wf-note", text: "Choose successful numerical Lab results below, not grouping records. The server retains the complete population, including failed and reused trials; the visible page is not the population." }),
        el("div", { id: "wf-c-candidates" }),
        el("div", { class: "wf-actions" },
          workflowButton("Previous page", () => {
            if (!workflowChronologyReady() || workflowChronology.page === 0) return;
            workflowChronology.page--; renderWorkflowChronologyCandidates(); workflowUpdateButtons();
          }, "wf-c-previous"),
          workflowButton("Next 12 population records", loadWorkflowChronologyCandidates, "wf-c-load")),
        el("label", { class: "wf-c-text", id: "wf-c-reason-field" }, el("span", { text: "Decision reason · why these evaluated candidates?" }),
          el("textarea", { id: "wf-c-reason", rows: 2, oninput: workflowUpdateButtons })),
        workflowButton("Freeze selected candidates and reason", freezeWorkflowChronology, "wf-c-freeze")),
      el("section", { id: "wf-c-final-controls", hidden: true },
        el("h4", { text: "Final access · frozen expressions only" }),
        el("div", { id: "wf-c-selection" }),
        el("p", { class: "wf-omissions", text: "Every final access is recorded, including failure and cancellation. first_recorded_access means first in this registry, not proof of an untouched holdout. Repeated access stays labelled." }),
        el("label", { class: "wf-check" }, el("input", { id: "wf-c-exposed", type: "checkbox", onchange: workflowUpdateButtons }),
          el("span", { text: "I acknowledge frozen selection reasons, current external exposure warnings and prior final access, including unknown holdout evidence, and permit a labelled final access (off by default)." })),
        workflowButton("Request final evaluation", runWorkflowChronologyFinal, "wf-c-final"))));
}

function workflowChronologyMessage(message) { $("#wf-c-message").textContent = message; }
function workflowChronologyInvalidate(message) {
  const c = workflowChronology;
  c.revision++; c.preview = null; c.key = null; c.loading = false;
  setChildren($("#wf-c-preview")); workflowChronologyMessage(message); workflowUpdateButtons();
}
function workflowChronologyReset() {
  const c = workflowChronology;
  c.protocolId = null; c.plan = null; c.view = null; c.selection = null; c.records.clear(); c.chosen.clear(); c.page = 0; c.candidateInputs = []; c.stale = false;
  $("#wf-c-exposed").checked = false;
  $("#wf-c-reason").value = ""; $("#wf-c-expressions").value = "";
  setChildren($("#wf-c-outcomes")); setChildren($("#wf-c-candidates")); setChildren($("#wf-c-selection"));
  workflowChronologyInvalidate("Source pinned. Enter chronological windows and a selection rule, then preview.");
}
function workflowChronologyUseSource() {
  if (workflowBusy() || !workflowEligible()) return;
  workflowChronologyReset();
  workflowChronology.source = workflowUI.detail;
  renderWorkflowChronologySource(); workflowUpdateButtons();
}
function workflowChronologyProtocolFromSelection() {
  const a = workflowUI.detail?.attempt;
  return a?.kind === "chronology.protocol" ? a.id : a?.kind?.startsWith("chronology.") ? a.request?.protocol_id : null;
}
function workflowChronologyParams() {
  const source = workflowChronology.source?.attempt;
  if (!source?.result_digest) throw new Error("Pin a successful saved Lab source first.");
  const request = { source_attempt_id: source.id };
  const date = value => /^\d{4}-\d{2}-\d{2}$/.test(value) && Number.isFinite(Date.parse(value)) &&
    new Date(value).toISOString().slice(0, 10) === value;
  for (const phase of workflowPhases) {
    const start = $(`#wf-c-${phase}-start`).value, end = $(`#wf-c-${phase}-end`).value;
    if (!date(start) || !date(end) || start > end) throw new Error("Enter valid start/end dates for all three windows.");
    request[phase] = { start, end };
  }
  if (request.development.end >= request.validation.start || request.validation.end >= request.final.start)
    throw new Error("Windows must be chronological and non-overlapping.");
  const rule = $("#wf-c-rule").value.trim();
  if (!rule) throw new Error("State a selection rule before declaring.");
  request.selection_rule = { rule };
  const raw = $("#wf-c-samples").value.trim();
  if (raw) {
    if (raw.length > 200000) throw new Error("Sample input is too large for this bounded UI; use the structured CLI.");
    let samples;
    try { samples = JSON.parse(raw); } catch { throw new Error("Sample intervals must be a JSON list."); }
    if (!Array.isArray(samples) || samples.length > 1000) throw new Error("Supply at most 1,000 explicit sample intervals here.");
    const required = ["id", "session", "information_start", "information_end"];
    if (samples.some(s => !s || typeof s !== "object" || Array.isArray(s) ||
      required.some(k => typeof s[k] !== "string" || !s[k].trim()) ||
      Object.keys(s).some(k => ![...required, "feature_available_at"].includes(k)) ||
      required.slice(1).some(k => !date(s[k])) || ("feature_available_at" in s && !date(s.feature_available_at))))
      throw new Error("Invalid sample fields. Supply id and dated information intervals, not a mask or source override.");
    request.samples = samples;
  }
  return request;
}
function workflowChronologyKey() {
  try { return JSON.stringify(workflowCanonical({ request: workflowChronologyParams(),
    result_digest: workflowChronology.source.attempt.result_digest })); } catch { return null; }
}
function workflowChronologyReady() {
  const c = workflowChronology;
  return !!c.protocolId && c.view?.protocol_id === c.protocolId && !c.stale && !c.loading && !workflowBusy();
}
function workflowChronologyPhaseRefusal(phase, split = workflowChronology.view?.sample_split || workflowChronology.preview?.sample_split) {
  const c = workflowChronology, coverage = split?.coverage;
  const explicit = split?.execution_policy === "observation-filter-v1" ||
    coverage?.status === "complete" || coverage?.status === "partial" ||
    c.plan?.metadata?.samples != null || c.preview?.protocol?.metadata?.samples != null;
  if (explicit && coverage?.status !== "complete")
    return coverage?.phases?.[phase]?.refusal || "Explicit sample intervals require complete coverage across all phases.";
  if (coverage) {
    if (!["complete", "not_supplied"].includes(coverage.status) ||
        (explicit && workflowPhases.some(p => !coverage.phases?.[p])))
      return "Observation coverage is missing or incomplete. Refresh the protocol.";
    const row = coverage.phases?.[phase];
    return row?.ready === true ? null : row?.refusal || "Phase observation coverage is not ready.";
  }
  return (split?.excluded || []).some(row => row.phase === phase)
    ? "Unsupported mask: observation coverage is unavailable for this legacy split with exclusions. Refresh the protocol." : null;
}
function workflowChronologyExposureReasons() {
  const c = workflowChronology;
  return [...(c.selection?.request.exposure_reasons || []), ...(c.view?.external_exposure_reasons || [])];
}
function workflowChronologyCoverage(split) {
  const coverage = split.coverage;
  return el("section", {},
    el("p", { text: `Observation coverage: ${coverage?.status || "not recorded"} · ${coverage?.unit || "daily portfolio return observations"}. Execution policy: ${split.execution_policy || "legacy calendar windows"}.` }),
    coverage ? workflowSummary(Object.fromEntries(workflowPhases.map(phase => {
      const row = coverage.phases?.[phase];
      return [phase, row ? `${row.checked_count}/${row.observed_count} checked · ${row.excluded_sessions?.length || 0} excluded · ${row.evaluation_sessions === null ? "calendar window" : `${row.evaluation_sessions?.length || 0} retained`} · ${row.ready ? "ready" : "blocked"}` : "coverage missing"];
    }))) : null,
    ...workflowPhases.map(phase => {
      const refusal = workflowChronologyPhaseRefusal(phase, split);
      return refusal ? el("p", { class: "wf-error", text: `${phase}: ${refusal}` }) : null;
    }),
    split.execution_policy === "observation-filter-v1" || ["complete", "partial"].includes(coverage?.status)
      ? el("p", { class: "wf-omissions", text: "Full-window signal and warmup are maintained; only retained daily portfolio return observations are scored. This is not a continuous investment return or a masked trading policy. Qlib validation refuses filtered results; chronological candidate selection remains available." })
      : el("p", { class: "wf-note", text: "Calendar windows only; no sample coverage or purge verification is implied." }));
}
function workflowChronologyCandidateRefs() {
  const v = workflowChronology.view;
  return v ? v.selection_id ? v.selected : v.population : [];
}
function workflowChronologyTrialParams() {
  const phase = $("#wf-c-phase").value, expressions = $("#wf-c-expressions").value.split(/\r?\n/).map(v => v.trim()).filter(Boolean);
  if (!["development", "validation"].includes(phase)) throw new Error("Choose development or validation; final requires frozen selection.");
  if (!expressions.length || expressions.length > 50) throw new Error("Supply 1–50 expressions, one per line.");
  const refusal = workflowChronologyPhaseRefusal(phase);
  if (refusal) throw new Error(refusal);
  return { protocol_id: workflowChronology.protocolId, phase, expressions };
}
function workflowChronologyCanFreeze() {
  const c = workflowChronology;
  return workflowChronologyReady() && !c.view.selection_id && c.chosen.size > 0 &&
    !c.view.population.some(r => r.status === "pending") && !!$("#wf-c-reason").value.trim() &&
    [...c.chosen].every(id => {
      const record = c.records.get(id)?.attempt, ref = c.view.population.find(r => r.id === id);
      return record?.kind === "lab-evaluation" && ["completed", "reused"].includes(record.status) &&
        !!record.result_digest && record.result_digest === ref?.result_digest && record.status === ref.status;
    });
}
function workflowChronologyCanFinal() {
  const c = workflowChronology;
  return workflowChronologyReady() && !!c.selection && c.view.selection_id === c.selection.id &&
    c.selection.request.selected.length > 0 && !c.view.accesses.some(a => a.status === "pending") &&
    !workflowChronologyPhaseRefusal("final") &&
    (!(workflowChronologyExposureReasons().length || c.view.accesses.length) || $("#wf-c-exposed").checked);
}
function workflowChronologyButtons() {
  if (!$("#wf-c-use")) return;
  const c = workflowChronology, busy = workflowBusy(), ready = workflowChronologyReady();
  $("#wf-c-use").disabled = busy || !workflowEligible();
  $("#wf-c-open").disabled = busy || !workflowChronologyProtocolFromSelection();
  $("#wf-c-plan-fields").hidden = !!c.protocolId;
  $("#wf-c-protocol").hidden = !c.protocolId;
  $("#wf-c-preview-button").disabled = busy || c.loading || !c.source || !!c.protocolId;
  $("#wf-c-declare").disabled = busy || c.loading || !!c.protocolId || !c.preview || c.key !== workflowChronologyKey();
  $("#wf-c-refresh").disabled = busy || c.loading || !c.protocolId;
  const refs = workflowChronologyCandidateRefs();
  $("#wf-c-load").disabled = !ready || !refs.length || (c.records.size > 0 && (c.page + 1) * 12 >= refs.length);
  $("#wf-c-load").textContent = c.view?.selection_id ? "Review next 12 frozen candidates" : "Next 12 population records";
  $("#wf-c-previous").disabled = !ready || c.page === 0;
  for (const { input, eligible } of c.candidateInputs) input.disabled = !ready || !!c.view?.selection_id || !eligible;
  let trialValid = false;
  try { workflowChronologyTrialParams(); trialValid = true; } catch { /* Local validation message below. */ }
  $("#wf-c-trial").disabled = !ready || !!c.view?.selection_id || !trialValid;
  $("#wf-c-freeze").disabled = !workflowChronologyCanFreeze();
  $("#wf-c-final").disabled = !workflowChronologyCanFinal();
  $("#wf-c-exposed").disabled = !ready || !c.selection;
  $("#wf-c-final").textContent = c.view?.accesses?.length ? "Request final again · repeat access recorded"
    : workflowChronologyExposureReasons().length ? "Request final evaluation · exposed holdout" : "Request final evaluation";
  $("#wf-c-final-controls").hidden = !c.view?.selection_id;
  $("#wf-c-trial-controls").hidden = !!c.view?.selection_id;
  $("#wf-c-reason-field").hidden = $("#wf-c-freeze").hidden = !!c.view?.selection_id;
  $("#wf-c-candidates-title").textContent = c.view?.selection_id ? "Frozen evaluated candidates · read only" : "Freeze evaluated candidates";
  for (const id of [...workflowPhases.flatMap(p => [`${p}-start`, `${p}-end`]), "rule", "samples", "phase", "expressions", "reason"])
    $(`#wf-c-${id}`).disabled = busy || (["phase", "expressions", "reason"].includes(id) ? c.loading : !!c.protocolId);
}
function renderWorkflowChronologySource(binding = null) {
  const detail = workflowChronology.source, result = detail?.result;
  const settings = binding?.settings || result?.settings || {}, summary = result?.summary || {};
  setChildren($("#wf-c-source"), workflowSummary({
    "Pinned source": result?.expression || detail?.attempt.request?.expression,
    "Saved evidence period": workflowPeriod(summary) || workflowPeriod(settings),
    "Source status": detail?.attempt.status, Delay: settings.delay, Universe: settings.universe,
    Neutralization: settings.neutralization, "Truncation (fraction)": settings.truncation,
    "Lab cost per side (bp)": settings.cost_bps,
  }), el("small", { text: `Source ${workflowShortId(detail?.attempt.id)} · selection elsewhere does not change this pinned source.` }),
    workflowDisclosure("Actual saved settings and source · read only", { settings, source: binding?.source_identity || result?.dataset,
      qualification: binding?.qualification || result?.qualification, source_attempt_id: detail?.attempt.id,
      source_result_digest: detail?.attempt.result_digest, binding }));
}
function renderWorkflowChronologyPreview(p) {
  const plan = p.protocol, binding = plan.metadata.chronology_workflow, split = p.sample_split;
  const sourceEnd = workflowChronology.source?.result?.summary?.end || workflowChronology.source?.result?.settings?.end;
  renderWorkflowChronologySource(binding);
  setChildren($("#wf-c-preview"), workflowSummary(Object.fromEntries(workflowPhases.map(phase => [phase, workflowPeriod(plan[phase])]))),
    el("p", { text: `Selection rule: ${workflowText(plan.selection_rule.rule || workflowText(plan.selection_rule))}` }),
    el("p", { class: "wf-omissions", text: !sourceEnd || sourceEnd >= plan.final.start
      ? "Exposure warning: the saved source reaches the final window, or its observed end is unknown. A new declaration cannot erase prior exposure."
      : "Exposure remains subject to frozen-population checks. Declaring new windows does not certify an untouched holdout." }),
    el("p", { text: `Qualification: ${p.qualification?.status || "not recorded"} · ${workflowBrief(p.qualification?.limitations)}` }),
    el("p", { class: "wf-omissions", text: "Lab fixed-book screening omits stock borrow and realistic cash/share execution. No Qlib/account validation is implied." }),
    el("p", { text: `Information-overlap check: ${split.overlap_check}. ${split.excluded.length} sample exclusions.${split.overlap_check === "not_available" ? " Calendar split only; no purge verification." : ""}` }),
    workflowChronologyCoverage(split),
    workflowDisclosure("Declaration, sample split and qualification · bounded details", p));
}
function workflowChronologyCheckPreview(p, request) {
  const plan = p?.protocol, binding = plan?.metadata?.chronology_workflow, source = workflowChronology.source?.attempt;
  const equal = (a, b) => JSON.stringify(workflowCanonical(a)) === JSON.stringify(workflowCanonical(b));
  if (p?.schema_version !== 1 || !binding || binding.source_attempt_id !== request.source_attempt_id ||
      binding.source_result_digest !== source?.result_digest || !equal(binding.settings, workflowChronology.source?.result?.settings) ||
      !p.sample_split || !Array.isArray(p.sample_split.excluded) ||
      workflowPhases.some(phase => !equal(plan[phase], request[phase])) || !equal(plan.selection_rule, request.selection_rule) ||
      !equal(plan.metadata.samples, request.samples))
    throw new Error("Chronology preview differs from the pinned source or declared inputs. Preview again.");
}
async function previewWorkflowChronology() {
  const c = workflowChronology;
  if (workflowBusy() || c.loading || !c.source || c.protocolId) return;
  workflowChronologyInvalidate("Checking calendar windows and saved source…");
  const revision = c.revision;
  try {
    const request = workflowChronologyParams(), key = workflowChronologyKey();
    c.loading = true; workflowUpdateButtons();
    const p = await api("workflow-chronology-preview", request);
    if (revision !== c.revision) return;
    if (key !== workflowChronologyKey()) throw new Error("Declaration changed. Preview again.");
    workflowChronologyCheckPreview(p, request);
    c.preview = p; c.key = key; renderWorkflowChronologyPreview(p);
    workflowChronologyMessage("Preview only; no trials ran. Review windows, qualifications and exposure before declaring.");
  } catch (e) { if (revision === c.revision) workflowChronologyMessage(e.message); }
  finally { if (revision === c.revision) c.loading = false; workflowUpdateButtons(); }
}
async function declareWorkflowChronology() {
  const c = workflowChronology;
  if (workflowBusy() || c.loading || c.protocolId || !c.preview || c.key !== workflowChronologyKey()) return;
  const revision = c.revision, request = workflowChronologyParams();
  c.submitting = true; workflowUpdateButtons();
  try {
    const p = await api("workflow-chronology-declare", request);
    if (revision !== c.revision) return;
    workflowChronologyCheckPreview(p, request);
    if (!p.protocol_id) throw new Error("Declaration response is missing its protocol identity. Inspect history before retrying.");
    c.protocolId = p.protocol_id; c.plan = p.protocol; c.preview = p; c.stale = true;
    renderWorkflowChronologyPreview(p);
    workflowChronologyMessage("Protocol declared. Refresh protocol explicitly before submitting trials.");
  } catch (e) { if (revision === c.revision) { c.preview = null; c.key = null; workflowChronologyMessage(e.message); } }
  finally { c.submitting = false; workflowUpdateButtons(); }
}
async function openWorkflowChronology() {
  const c = workflowChronology, id = workflowChronologyProtocolFromSelection();
  if (workflowBusy() || !id) return;
  workflowChronologyReset(); c.source = null; c.loading = true;
  setChildren($("#wf-c-source"));
  const revision = c.revision; workflowUpdateButtons();
  try {
    const detail = await get("workflow-detail", { id });
    if (revision !== c.revision) return;
    const plan = detail.attempt?.request, binding = plan?.metadata?.chronology_workflow;
    if (detail.attempt?.id !== id || detail.attempt.kind !== "chronology.protocol" || !binding)
      throw new Error("Selected record is not a saved chronology workflow protocol.");
    const source = await get("workflow-detail", { id: binding.source_attempt_id });
    if (revision !== c.revision) return;
    if (source.attempt?.id !== binding.source_attempt_id || source.attempt.result_digest !== binding.source_result_digest)
      throw new Error("Protocol source identity differs from saved evidence.");
    c.source = source; c.protocolId = id; c.plan = plan; c.stale = true;
    renderWorkflowChronologySource(binding);
    setChildren($("#wf-c-preview"), workflowSummary(Object.fromEntries(workflowPhases.map(p => [p, workflowPeriod(plan[p])]))),
      el("p", { text: `Selection rule: ${workflowText(plan.selection_rule.rule || workflowText(plan.selection_rule))}` }),
      el("p", { class: "wf-omissions", text: "Saved source inspection may already expose the final window. Refresh protocol to review recorded qualification and exposure; opening a declaration does not reset access history." }),
      workflowDisclosure("Saved declaration · windows and selection rule", plan));
    workflowChronologyMessage("Saved protocol opened. Refresh protocol to load population and access labels.");
  } catch (e) { if (revision === c.revision) workflowChronologyMessage(e.message); }
  finally { if (revision === c.revision) c.loading = false; workflowUpdateButtons(); }
}
function workflowChronologyStale(message) {
  const c = workflowChronology;
  c.revision++; c.loading = false; c.stale = true; c.chosen.clear();
  $("#wf-c-exposed").checked = false;
  if (c.protocolId) workflowChronologyMessage(message);
  workflowUpdateButtons();
}
// Optional Main completion hook: no fetches, scans or automatic numerical reads.
function workflowChronologyJobFinished(job) {
  if (!["chronology-trial", "chronology-final"].includes(job.kind)) return;
  workflowChronologyStale(`${job.kind} ${job.state}${job.receipt?.access_label ? " · " + job.receipt.access_label : ""}. Refresh protocol explicitly for recorded outcomes and access history.`);
}
async function refreshWorkflowChronology() {
  const c = workflowChronology;
  if (workflowBusy() || c.loading || !c.protocolId) return;
  workflowChronologyStale("Refreshing protocol metadata; final numerical results are not read.");
  const revision = c.revision, id = c.protocolId;
  c.loading = true; c.selection = null; c.records.clear(); c.page = 0; c.candidateInputs = [];
  setChildren($("#wf-c-candidates")); workflowUpdateButtons();
  try {
    const view = await api("workflow-chronology-view", { protocol_id: id });
    if (revision !== c.revision) return;
    if (view.schema_version !== 1 || view.protocol_id !== id || view.source_attempt_id !== c.source?.attempt.id ||
        view.source_result_digest !== c.source?.attempt.result_digest || !Array.isArray(view.population) ||
        !Array.isArray(view.accesses) || !Array.isArray(view.selected) || !Array.isArray(view.sample_split?.excluded))
      throw new Error("Protocol view does not match the pinned evidence.");
    let selection = null;
    if (view.selection_id) {
      const detail = await get("workflow-detail", { id: view.selection_id });
      if (revision !== c.revision) return;
      selection = detail.attempt;
      if (selection?.id !== view.selection_id || selection.kind !== "chronology.selection" ||
          selection.status !== "completed" || !selection.result_digest ||
          selection.request?.protocol_id !== id || !Array.isArray(selection.request.exposure_reasons) ||
          JSON.stringify(workflowCanonical(selection.request.selected)) !== JSON.stringify(workflowCanonical(view.selected)))
        throw new Error("Frozen selection or exposure evidence differs from the protocol view.");
    }
    c.view = view; c.selection = selection; c.stale = false;
    if (c.plan) renderWorkflowChronologyPreview({ schema_version: 1, protocol: c.plan,
      sample_split: view.sample_split, qualification: view.qualification });
    renderWorkflowChronologyView();
    workflowChronologyMessage("Protocol refreshed. Actions use this recorded population; refresh again after any job.");
  } catch (e) { if (revision === c.revision) workflowChronologyMessage(e.message); }
  finally { if (revision === c.revision) c.loading = false; workflowUpdateButtons(); }
}
function renderWorkflowChronologyView() {
  const c = workflowChronology, v = c.view, split = v.sample_split;
  setChildren($("#wf-c-outcomes"), workflowSummary({ Protocol: workflowShortId(c.protocolId),
    "Recorded population": v.population.length, "Frozen candidates": v.selected.length, "Final accesses": v.accesses.length }),
    el("p", { text: `Qualification: ${v.qualification?.status || "not recorded"} · Information-overlap check: ${split.overlap_check}.` }),
    el("p", { class: "wf-note", text: workflowBrief(v.qualification?.limitations) }),
    workflowChronologyCoverage(split),
    ...v.accesses.slice(-8).map(a => el("p", { text: `${a.access_label} · ${a.status} · ${a.reused_holdout ? "reused holdout" : "first recorded reservation"} · ${workflowShortId(a.id)}` })),
    workflowDisclosure(`Protocol provenance · ${v.accesses.length} access records, full population and qualification`, v));
  renderWorkflowChronologyCandidates();
  const s = c.selection?.request, reasons = workflowChronologyExposureReasons();
  const exposureText = rows => workflowBrief(rows.map(r => r.reason || r));
  setChildren($("#wf-c-selection"), s ? [
    el("p", { text: `${s.selected.length} frozen candidates · ${workflowText(s.decision_reason)}` }),
    el("p", { class: "wf-omissions", text: reasons.length || v.accesses.length
      ? `Exposure warning: frozen selection: ${exposureText(s.exposure_reasons) || "none recorded"}; current external audit: ${exposureText(v.external_exposure_reasons || []) || "none recorded"}${v.accesses.length ? "; prior final access recorded" : ""}. Explicit acknowledgment is required.`
      : "No exposure reasons recorded by the freeze check or current audit; this is not a guarantee of an untouched holdout." }),
    workflowDisclosure("Frozen candidates, reason and exposure provenance", c.selection),
  ] : []);
}
async function loadWorkflowChronologyCandidates() {
  const c = workflowChronology;
  if (!workflowChronologyReady()) return;
  if ((c.page + 1) * 12 < c.records.size) {
    c.page++; renderWorkflowChronologyCandidates(); workflowUpdateButtons(); return;
  }
  const refs = workflowChronologyCandidateRefs();
  const rows = refs.filter(r => !c.records.has(r.id)).slice(0, 12), revision = c.revision;
  if (!rows.length) return;
  c.loading = true; workflowUpdateButtons();
  try {
    // Bounded metadata/compact-result reads, only for training population references.
    const details = await Promise.all(rows.map(r => get("workflow-detail", { id: r.id })));
    if (revision !== c.revision) return;
    details.forEach((d, i) => {
      const r = rows[i], a = d.attempt;
      if (a?.id !== r.id || a.result_digest !== r.result_digest || a.status !== r.status ||
          a.kind === "chronology.final") throw new Error("Population changed or returned an unexpected result. Refresh protocol.");
    });
    c.page = Math.floor(c.records.size / 12);
    details.forEach(d => c.records.set(d.attempt.id, d));
    renderWorkflowChronologyCandidates();
    workflowChronologyMessage(`Loaded ${c.records.size} of ${refs.length} ${c.view.selection_id ? "frozen candidates" : "population records"}. Full population is retained by the server.`);
  } catch (e) { if (revision === c.revision) { c.stale = true; workflowChronologyMessage(e.message); } }
  finally { if (revision === c.revision) c.loading = false; workflowUpdateButtons(); }
}
function renderWorkflowChronologyCandidates() {
  const c = workflowChronology;
  const frozen = !!c.view.selection_id, refs = workflowChronologyCandidateRefs();
  c.candidateInputs = [];
  setChildren($("#wf-c-candidates"), el("p", { class: "wf-note", text: frozen
    ? `${refs.length} frozen candidates · ${c.records.size} loaded. Review their expressions below; they may differ from the original source expression.`
    : `${c.chosen.size} selected · ${c.records.size} of ${refs.length} records loaded · page ${c.page + 1}. Only successful numerical Lab attempts are selectable.` }),
    frozen && !c.records.size ? el("div", {}, ...refs.slice(0, 8).map(r =>
      el("p", { text: `${workflowText(workflowUI.rows.find(a => a.id === r.id)?.display_label || "Saved evaluated candidate")} · ${r.status} · ${workflowShortId(r.id)}` }))) : null,
    ...[...c.records.values()].slice(c.page * 12, (c.page + 1) * 12).map(d => {
    const a = d.attempt, r = d.result || {}, eligible = a.kind === "lab-evaluation" &&
      ["completed", "reused"].includes(a.status) && !!a.result_digest;
    const input = el("input", { type: "checkbox", checked: frozen || c.chosen.has(a.id),
      disabled: !eligible || !!c.view?.selection_id, onchange: event => {
        if (!workflowChronologyReady() || c.view.selection_id || !eligible) { event.currentTarget.checked = c.chosen.has(a.id); return; }
        if (event.currentTarget.checked) c.chosen.add(a.id); else c.chosen.delete(a.id);
        renderWorkflowChronologyCandidates(); workflowUpdateButtons();
      } });
    c.candidateInputs.push({ input, eligible });
    return el("div", { class: "wf-chronology-row" },
      el("label", { class: "wf-check" }, input, el("span", { text: r.expression || a.request?.expression || workflowAttemptLabel(a) })),
      el("small", { text: `${a.kind} · ${a.status} · ${workflowPeriod(r.summary || r.settings) || "grouping / no numerical period"} · ${workflowShortId(a.id)}` }),
      workflowEvaluationWarning(r),
      eligible && workflowObservationFiltered(r) ? el("h4", { text: "Selected-observation metrics" }) : null,
      eligible ? workflowSummary(workflowRecorded({ Sharpe: workflowNumber(r.summary?.sharpe),
        "Annual return (half-book)": workflowNumber(r.summary?.returns, true),
        ...(workflowObservationFiltered(r) ? { "Selected observations": workflowNumber(r.summary?.days) } : {}),
        Qualification: r.qualification?.status || "exploratory Lab evidence" })) : null,
      a.error ? el("p", { class: "wf-error", text: workflowText(a.error.message || a.error) }) : null,
      workflowDisclosure("Trial details and immutable result identity", d));
  }));
}
async function workflowChronologySubmit(endpoint, request, isJob) {
  const c = workflowChronology, revision = c.revision;
  c.submitting = true; workflowUpdateButtons();
  try {
    const result = await api(endpoint, request);
    if (revision !== c.revision) return;
    if (isJob) {
      workflowChronologyStale("Job submitted. Use Stop while cancellable, then explicitly refresh protocol for outcomes and final-access labels.");
      workflowInvalidate("Chronology job submitted. Preview candidate validation again before running.");
      workflowPortfolioInvalidate("Chronology job submitted. Preview the portfolio again before running.");
      startPolling(result);
    } else {
      if (result.protocol_id !== c.protocolId || !result.selection_id)
        throw new Error("Freeze response identity differs. Refresh protocol before further actions.");
      workflowChronologyStale("Selection frozen. Refresh protocol to review recorded exposure before requesting final.");
    }
  } catch (e) {
    if (revision === c.revision) workflowChronologyStale(`${e.message}\nRefresh protocol before retrying; a submission or access may already be recorded.`);
  } finally { c.submitting = false; workflowUpdateButtons(); }
}
async function runWorkflowChronologyTrial() {
  const c = workflowChronology;
  if (!workflowChronologyReady() || c.view.selection_id) return;
  try { await workflowChronologySubmit("workflow-chronology-trial", workflowChronologyTrialParams(), true); }
  catch (e) { workflowChronologyMessage(e.message); }
}
async function freezeWorkflowChronology() {
  if (!workflowChronologyCanFreeze()) return;
  await workflowChronologySubmit("workflow-chronology-freeze", { protocol_id: workflowChronology.protocolId,
    selected_ids: [...workflowChronology.chosen], decision_reason: $("#wf-c-reason").value.trim() }, false);
}
async function runWorkflowChronologyFinal() {
  if (!workflowChronologyCanFinal()) return;
  await workflowChronologySubmit("workflow-chronology-final", { protocol_id: workflowChronology.protocolId,
    selection_id: workflowChronology.selection.id, allow_exposed: !!$("#wf-c-exposed").checked }, true);
}
