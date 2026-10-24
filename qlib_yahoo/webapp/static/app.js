"use strict";
/* Tubby Qlib Console — research workbench.
   Talks only to the loopback service that served this page (cookie + request token).
   core.js (DOM, API, menus, dialogs), charts.js and analytics.js load first. */

const FEATURES = new Set(["open", "high", "low", "close", "volume", "factor"]);
const OPERATORS = new Set(["Abs", "Sign", "Log", "Power", "Greater", "Less", "If", "Ref", "Mean", "Sum", "Std", "Max", "Min",
  "Rank", "Delta", "Slope", "WMA", "EMA", "Corr", "Cov"]);
const DEFAULTS = { gross: "1", borrow: "0.4", topk: "50", ndrop: "5", start: "", end: "", warmup: "252", cost: "5", cash: "0" };
const MAX_COMPARE = 6;
const VIEWS = ["research", "lab", "data", "auto", "evidence"];
const UI_VERSION = "workbench-cleanup-2026-10-14";

const state = {
  overview: null, view: "lab", mode: "signed", transform: "raw", restoring: true,
  dataset: null, datasetPath: null, datasets: [], detailDataset: null,
  datasetLoad: 0, recovery: { research: "", lab: "" }, recoveryPrice: null,
  runs: [], runSort: "created", runFilter: "all", factorFilter: null, trash: { runs: [], alphas: [] },
  run: null, runDetail: null, selected: [], anchor: null,
  resultTab: "overview", window: null, moreOpen: false, signal: {}, compareCache: new Map(),
  job: null, jobSince: 0, poll: null, settings: {}, bottomTab: "console", scale: "linear",
  lab: { rows: [], sort: "created_at", filter: "all", selected: null, detail: null, batch: null, delay: "1", chosen: [],
    dataset: null, datasetPath: null, datasetLoad: 0, groupOverlay: "", groupExplicit: false,
    groupMetadata: null, groupEpoch: 0, groupLoading: false, groupIdentity: null, groupRestoreError: "" },
  sec: { catalog: "", mode: "dictionary", dictionaryView: "company", registry: null, bindings: {}, resolved: {}, generation: 0, rowsOffset: 0, rowFilters: "{}", rowTable: "SUB", numeric: false },
};

// SEC names/recipes are shared across both engines. Editor checks read only
// dictionary metadata; source scans and dictionary builds are explicit actions.
function secMode(config) {
  return config?.mode || (config && Object.hasOwn(config, "bindings") ? "legacy" : "dictionary");
}
function secConfig() {
  return { catalog: state.sec.catalog, mode: state.sec.mode, bindings: state.sec.bindings,
    ...(state.sec.registry ? { registry: state.sec.registry } : {}) };
}
function secSwitchState() {
  if (state.sec.mode !== "legacy" || !state.sec.catalog || state.sec.restoreLoading || state.sec.restoreError) return null;
  return JSON.stringify([secConfig(), state.sec.restoreLoad, state.sec.generation]);
}
function resetSecSwitchConfirmation() {
  state.sec.switchConfirmation = null;
  $("#sec-switch-confirmation").hidden = true;
  $("#sec-switch-dictionary").ariaExpanded = "false";
}
function secRequest() {
  if (state.sec.restoreLoading) throw new Error("Restoring saved SEC inputs; wait for validation to finish.");
  if (state.sec.restoreError) throw new Error(state.sec.restoreError);
  return state.sec.catalog || Object.keys(state.sec.bindings).length ? { sec: secConfig() } : {};
}
function secMessage(text, bad = false) {
  $("#sec-message").textContent = text;
  $("#sec-message").className = bad ? "warn" : "";
}
function secAction(action) {
  return async (...args) => { try { await action(...args); } catch (e) { secMessage(e.message, true); } };
}
function secEditorError(message) {
  const text = String(message || "Invalid expression");
  const name = /\bsec_[A-Za-z][A-Za-z0-9_]*\b/.exec(text)?.[0];
  return name && /unknown|unavailable|unbound/i.test(text) && !/open SEC fields|add input|search (?:the )?(?:shared )?SEC dictionary/i.test(text) && !Object.hasOwn(state.sec.bindings, name)
    ? text + (state.sec.mode === "legacy"
      ? ". Legacy inputs keep their saved names. Choose Switch to dictionary to use official names. Names are case-sensitive."
      : ". Search the shared SEC dictionary for an existing official name and definition. Names are case-sensitive; unknown names cannot be created here.")
    : text;
}
function secPeriodLabel(qtrs) {
  return ({0: "Instant (point in time)", 1: "1 quarter", 2: "2 quarters", 3: "3 quarters", 4: "4 quarters (annual)"})[qtrs] || `${qtrs} quarters`;
}
function secFamily(taxonomy) {
  return /^[A-Za-z][A-Za-z0-9_.-]*\/$/.test(taxonomy || "") ? taxonomy : null;
}
function secContextLabel(v) {
  return `Segments: ${v.segments === "" ? "none (blank)" : v.segments ?? "unavailable (null)"} · Co-registrant: ${v.coreg === "" ? "none (blank)" : v.coreg ?? "unavailable (null)"}`;
}
function secReadableTitle(item) {
  return String(item.spec?.tag || item.label || "Unavailable field")
    .replace(/([A-Z]+)([A-Z][a-z])/g, "$1 $2")
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/_/g, " ");
}
function secRecipeSummary(item) {
  const spec = item.spec || {}, family = secFamily(spec.taxonomy);
  const scope = item.kind === "standard" ? "Standard taxonomy" : "Company-defined or unconfirmed";
  const period = spec.qtrs == null ? "Period unavailable" : secPeriodLabel(spec.qtrs);
  const context = spec.segments === "" && spec.coreg === ""
    ? "No segment or co-registrant (blank context)"
    : [spec.segments == null ? "Segment context unavailable" : spec.segments !== "" ? "Special recipe: segment" : "",
      spec.coreg == null ? "Co-registrant context unavailable" : spec.coreg !== "" ? "Special recipe: co-registrant" : ""].filter(Boolean).join(" · ");
  const edition = family ? `All editions · ${spec.taxonomy}`
    : item.kind === "standard" && spec.taxonomy ? `Special recipe: exact edition · ${spec.taxonomy}`
    : "Exact definition (see details)";
  return `${scope} · ${period} · Unit: ${spec.unit || "unavailable"} · ${context} · ${edition}`;
}
function secDictionaryView() { return state.sec.dictionaryView === "recipes" ? "recipes" : "company"; }
function secAliasUnavailable(item) { return typeof item?.alias_error === "string"; }
function secFormulaDisplay(item) {
  if (!secAliasUnavailable(item)) return item?.name || "Choose a variant to see its formula name";
  let reason = item.alias_error.replace(/\s+/g, " ").trim();
  // Some resolver diagnostics quote the storage identifier. Keep it in details.
  for (const name of [item.name, item.storage_name]) if (name) reason = reason.replaceAll(name, "stored definition");
  if (reason.length > 180) reason = reason.slice(0, 177) + "…";
  return "Descriptive name unavailable" + (reason ? ": " + reason : "");
}
function renderSecDictionaryView() {
  const advanced = secDictionaryView() === "recipes";
  $("#sec-dictionary-view").value = secDictionaryView();
  $("#sec-dictionary-custom-control").hidden = !advanced;
  $("#sec-dictionary-view-help").textContent = advanced
    ? "Advanced definitions include special dimensions, entities, custom tags and exact editions. Review each recipe."
    : "Company fields group standard taxonomy families with blank segment and co-registrant context. Choose an observed unit and period before inserting.";
}
function renderSecCompanyItem(concept, catalog, epoch, scope, partial = false) {
  const variants = concept.variants || [];
  let selected = variants.length === 1 ? variants[0] : null;
  const title = secReadableTitle({spec: {tag: concept.tag}, label: concept.label});
  const name = el("code", {text: secFormulaDisplay(selected), style: {overflowWrap: "anywhere"}});
  const blocked = () => !selected || secAliasUnavailable(selected) || state.sec.mode !== "dictionary" || !!state.sec.restoreError || !!state.sec.restoreLoading;
  const button = el("button", {class: "btn small", text: "Insert field", disabled: blocked(),
    onclick: secAction(() => insertSecField(selected, catalog, epoch, scope))});
  const variantText = entry => `${entry.spec?.unit || "Unit unavailable"} · ${entry.spec?.qtrs == null ? "Period unavailable" : secPeriodLabel(entry.spec.qtrs)} · ${secFormulaDisplay(entry)}`;
  const choice = variants.length === 1
    ? el("p", {text: variantText(selected)})
    : el("select", {value: "", "aria-label": `Unit and period for ${title}`, onchange: event => {
      const value = event.target.value;
      selected = /^\d+$/.test(value) ? variants[Number(value)] || null : null;
      name.textContent = secFormulaDisplay(selected);
      button.disabled = blocked();
    }}, el("option", {value: "", text: "Choose unit and period…"}),
      ...variants.map((entry, index) => el("option", {value: String(index), text: variantText(entry)})));
  return el("div", {class: "sec-dictionary-item"},
    el("strong", {text: title, style: {overflowWrap: "anywhere"}}),
    el("small", {text: `SEC tag: ${concept.tag} · Taxonomy family: ${concept.taxonomy}`, style: {overflowWrap: "anywhere"}}),
    concept.label ? el("small", {text: `Source label: ${concept.label}`}) : null,
    choice, name,
    el("details", {}, el("summary", {text: "Recipe details"}),
      ...variants.map(entry => el("div", {}, el("code", {text: entry.name}),
        entry.storage_name ? el("p", {}, "Storage name: ", el("code", {text: entry.storage_name, style: {overflowWrap: "anywhere"}})) : null,
        el("pre", {text: JSON.stringify(entry.spec, null, 2)})))),
    el("small", {text: partial || concept.truncated || concept.variants_truncated
      ? "Partial group / observed variant list; not a global count."
      : "Observed unit/period choices on this page; not a global count."}), button);
}
function renderSecDictionaryItem(item, catalog, epoch, scope = {view: secDictionaryView(), request: state.sec.dictionaryLoad}) {
  const spec = item.spec || {};
  const title = secReadableTitle(item), label = String(item.label || "").trim();
  const compact = text => text.replace(/[^a-z0-9]/gi, "").toLowerCase();
  const formula = el("code", {text: secFormulaDisplay(item), style: {overflowWrap: "anywhere"}, "aria-label": "Formula name"});
  const verification = el("small", {text: item.alias_pending && !secAliasUnavailable(item) ? "Verified when inserted" : ""});
  const original = el("code", {text: item.alias_pending ? item.storage_name || item.name : item.name, style: {overflowWrap: "anywhere"}});
  const blocked = () => secAliasUnavailable(item) || state.sec.mode !== "dictionary" || !!state.sec.restoreError || !!state.sec.restoreLoading;
  let verifying = false;
  const button = el("button", {class: "btn small", text: item.alias_pending ? "Verify & insert" : "Insert field", disabled: blocked(),
    onclick: secAction(async () => {
      if (!secInsertionCurrent(catalog, epoch, scope)) throw new Error("The SEC selection changed. Review it before inserting.");
      if (!item.alias_pending || secAliasUnavailable(item)) return insertSecField(item, catalog, epoch, scope);
      if (verifying) return;
      verifying = true; button.disabled = true;
      try {
        const resolved = await api("sec-name", {catalog, spec}, "POST");
        if (!secInsertionCurrent(catalog, epoch, scope)) return;
        if (!resolved?.spec || Object.keys(spec).length !== Object.keys(resolved.spec).length ||
            !Object.keys(spec).every(key => Object.hasOwn(resolved.spec, key) && spec[key] === resolved.spec[key]))
          throw new Error("Verified SEC definition differs from the selected recipe. Search again.");
        if (resolved.alias_pending) throw new Error("Descriptive name verification is still pending. Try again.");
        item = resolved;
        formula.textContent = secFormulaDisplay(item);
        verification.textContent = "";
        original.textContent = item.name;
        button.textContent = "Insert field";
        insertSecField(item, catalog, epoch, scope);
      } catch (error) {
        if (!secInsertionCurrent(catalog, epoch, scope)) return;
        throw error;
      } finally {
        verifying = false;
        if (secInsertionCurrent(catalog, epoch, scope)) button.disabled = blocked();
      }
    })});
  return el("div", {class: "sec-dictionary-item"},
    formula,
    el("span", {style: {overflowWrap: "anywhere"}},
      el("strong", {text: title}),
      el("small", {text: `SEC tag: ${spec.tag || "unavailable"}`}),
      label && compact(label) !== compact(title) ? el("small", {text: `Source label: ${label}`}) : null,
      ...(item.alias_pending ? [verification] : [])),
    el("small", {class: "sec-badge", text: secRecipeSummary(item)}),
    el("details", {}, el("summary", {text: "Recipe details"}),
      el("p", {text: secContextLabel(spec), style: {overflowWrap: "anywhere"}}),
      ...(secAliasUnavailable(item) || item.alias_pending ? [el("p", {}, "Original name: ", original)] : []),
      item.storage_name ? el("p", {}, "Storage name: ", el("code", {text: item.storage_name, style: {overflowWrap: "anywhere"}})) : null,
      el("pre", {text: JSON.stringify(spec, null, 2), style: {whiteSpace: "pre-wrap", overflowWrap: "anywhere"}})),
    button);
}
function renderSecBindings() {
  if (state.sec.switchConfirmation && state.sec.switchConfirmation !== secSwitchState()) resetSecSwitchConfirmation();
  const legacy = state.sec.mode === "legacy";
  $("#sec-legacy").hidden = !legacy;
  if (!legacy) {
    const resolved = state.sec.resolved[state.view === "lab" ? "lab" : "qlib"];
    const entries = resolved?.items || [];
    $("#sec-count").textContent = entries.length ? `(${entries.length})` : "";
    $("#sec-clear").hidden = !Object.keys(state.sec.bindings).length && !state.sec.restoreError;
    setChildren($("#sec-bindings"), ...entries.map(input => el("div", {},
      el("code", {text: input.name}), el("pre", {text: JSON.stringify(input.spec, null, 2)}))));
    if (!entries.length) setChildren($("#sec-bindings"), el("p", {text: state.sec.catalog
      ? "Formula inputs resolve automatically from the shared dictionary. Check the active editor to see their definitions."
      : "Choose a source catalog to use SEC dictionary names."}));
    return;
  }
  const entries = Object.entries(state.sec.bindings);
  $("#sec-count").textContent = entries.length ? `(${entries.length})` : "";
  $("#sec-clear").hidden = !entries.length && !state.sec.restoreError;
  setChildren($("#sec-bindings"), ...entries.map(([alias, saved]) => {
    const spec = saved && typeof saved === "object" ? saved : {};
    return el("div", {},
    el("code", { text: `Lab: ${alias} · Qlib: $${alias}` }),
    el("small", { text: `${spec.tag || "Unavailable field"} · ${spec.taxonomy || "Unavailable edition"} · ${spec.unit || "Unavailable unit"} · ${spec.qtrs == null ? "Unavailable period" : secPeriodLabel(spec.qtrs)} · ${secContextLabel(spec)}` }));
  }));
  if (!entries.length) setChildren($("#sec-bindings"), el("p", { text: "No SEC inputs selected. Price-only execution." }));
}
function secChanged() {
  resetSecSwitchConfirmation();
  state.sec.restoreLoad = (state.sec.restoreLoad || 0) + 1;
  state.sec.restoreLoading = false;
  state.sec.restoreError = null;
  invalidateSecChecks();
  renderSecBindings(); renderEditor(); persist(); checkAlpha(); checkLab();
}
function invalidateSecChecks() {
  state.sec.generation++;
  state.sec.resolved = {};
  state.alphaValid = state.labValid = false;
  updateSecRunButtons();
}
function updateSecRunButtons() {
  const running = state.job?.state === "running";
  $("#run-2").disabled = running || !state.alphaValid;
  $("#lab-run").disabled = running || !state.labValid;
  if (state.view !== "auto") $("#run").disabled = running || !(state.view === "lab" ? state.labValid : state.alphaValid);
  if (state.view === "evidence") workflowUpdateButtons();
}
function acceptSecInputs(result, engine) {
  const inputs = result.sec_inputs || result.lines?.flatMap(line => line.sec_inputs || []) || [];
  const items = [...new Map(inputs.map(input => [input.name, input])).values()];
  state.sec.resolved[engine] = {items, registry: result.registry || result.sec_registry};
  renderSecBindings();
}
function savedSec(record) {
  return record?.sec || (record?.dataset && typeof record.dataset === "object" ? record.dataset.sec : null) || null;
}
function secProvenance(record) {
  const sec = savedSec(record);
  return el("details", { class: "fact-card" }, el("summary", { text: "Saved SEC provenance (read only)" }),
    el("pre", { class: "selectable", style: { maxHeight: "240px", overflow: "auto", whiteSpace: "pre-wrap" },
      text: sec ? JSON.stringify({ catalog: sec.catalog, mode: secMode(sec), registry: sec.registry, bindings: sec.bindings, source_identity: sec.source_identity,
        materialized_identity: sec.materialized?.identity || sec.materialized_identity }, null, 2) : "Price-only result — no SEC inputs." }));
}
async function restoreSavedSec(record) {
  const request = state.sec.restoreLoad = (state.sec.restoreLoad || 0) + 1;
  state.sec.restoreLoading = true;
  invalidateSecChecks();
  const saved = savedSec(record);
  resetSecSelection();
  state.sec.restoreError = null;
  state.sec.catalog = saved?.catalog || "";
  state.sec.bindings = saved?.bindings && typeof saved.bindings === "object" ? JSON.parse(JSON.stringify(saved.bindings)) : {};
  state.sec.mode = secMode(saved);
  state.sec.registry = saved?.registry ? JSON.parse(JSON.stringify(saved.registry)) : null;
  renderSecBindings(); renderEditor();
  if (!saved) { secChanged(); return true; }
  try {
    const config = await api("sec-bindings", { sec: secConfig() });
    const expected = saved.source_identity?.catalog_sha256;
    if (expected) {
      const catalogs = await get("sec-catalogs");
      const current = catalogs.find(c => c.path === config.catalog);
      if (!current || current.manifest?.catalog_sha256 !== expected) throw new Error("Saved SEC catalog identity differs from the available catalog.");
    }
    if (request !== state.sec.restoreLoad) return false;
    state.sec.catalog = config.catalog; state.sec.bindings = config.bindings || state.sec.bindings;
    state.sec.mode = config.mode || state.sec.mode;
    state.sec.registry = config.registry || state.sec.registry;
    secChanged(); return true;
  } catch (error) {
    if (request !== state.sec.restoreLoad) return false;
    state.sec.restoreLoading = false;
    state.sec.restoreError = `Saved SEC inputs cannot be reused: ${error.message}. Restore the original catalog or explicitly clear/reselect SEC fields.`;
    renderSecBindings();
    persist(); toast("Saved SEC inputs unavailable", state.sec.restoreError, "bad"); return false;
  }
}
async function loadRunAlpha(row) {
  try {
    const d = state.runDetail?.path === row.path ? state.runDetail : await get("run", { path: row.path });
    if (!runReusable(d)) throw new Error(d.compatibility_note || "Saved run settings are read only.");
    if (!await restoreSavedSec(d)) return;
    setAlpha(d.alpha || ""); persist();
  } catch (error) { toast("Cannot reuse saved alpha", error.message, "bad"); }
}
// Selection epochs invalidate browser requests even when a catalog/scope is
// changed away and back. Saved input restoration keeps its separate guard.
function resetSecSelection() {
  resetSecSwitchConfirmation();
  clearSecCatalogLoading();
  for (const key of ["catalogLoad", "rowsLoad", "selectionEpoch", "dictionaryLoad"])
    state.sec[key] = (state.sec[key] || 0) + 1;
  state.sec.dictionaryOffset = 0;
  $("#sec-dictionary-build").hidden = true;
  setChildren($("#sec-dictionary-items"));
  $("#sec-dictionary-prev").disabled = $("#sec-dictionary-next").disabled = true;
  $("#sec-dictionary-page").textContent = ""; $("#sec-rows-page").textContent = "";
  $("#sec-rows").textContent = "Choose a table and press Read rows.";
  $("#sec-rows-prev").disabled = $("#sec-rows-next").disabled = true;
}
function clearSecCatalogLoading(request = state.sec.catalogMessageRequest) {
  if (request == null || request !== state.sec.catalogMessageRequest) return;
  state.sec.catalogMessageRequest = null;
  if ($("#sec-message").textContent === "Loading source catalogs…")
    secMessage(state.sec.restoreError || (state.sec.catalog ? "Review the current SEC view and search." : "Choose a source catalog first."), !!state.sec.restoreError);
}
async function loadSecCatalogs() {
  const request = state.sec.catalogLoad = (state.sec.catalogLoad || 0) + 1;
  const catalog = state.sec.catalog, restore = state.sec.restoreLoad;
  state.sec.catalogMessageRequest = request;
  secMessage("Loading source catalogs…");
  let catalogs;
  try { catalogs = await get("sec-catalogs"); }
  catch (error) { if (request !== state.sec.catalogLoad || catalog !== state.sec.catalog || restore !== state.sec.restoreLoad) return; throw error; }
  finally { clearSecCatalogLoading(request); }
  if (request !== state.sec.catalogLoad || catalog !== state.sec.catalog || restore !== state.sec.restoreLoad) return;
  setChildren($("#sec-catalog"), el("option", { value: "", text: "Choose source catalog…" }),
    ...catalogs.map(c => el("option", { value: c.path, text: c.name })));
  if (state.sec.catalog && !catalogs.some(c => c.path === state.sec.catalog)) {
    $("#sec-catalog").append(el("option", { value: state.sec.catalog, text: "Saved catalog unavailable — restore it or clear active inputs" }));
  }
  $("#sec-catalog").value = state.sec.catalog;
  const current = catalogs.find(c => c.path === state.sec.catalog);
  const fixed = catalogs.length === 1 && !!current;
  $("#sec-catalog").hidden = fixed; $("#sec-catalog-fixed").hidden = !fixed;
  $("#sec-catalog-fixed").textContent = fixed ? current.name : "";
  state.sec.numeric = current?.manifest?.numeric_materialization_supported === true;
  const tables = Object.keys(current?.manifest?.tables || {});
  setChildren($("#sec-table"), ...tables.map(t => el("option", { value: t, text: t.toUpperCase() })));
  if (tables.length) $("#sec-table").value = tables.includes(state.sec.rowTable) ? state.sec.rowTable : tables[0];
  secMessage(state.sec.restoreError || (catalogs.length
    ? (!current ? "Choose a source catalog to search the shared dictionary."
      : state.sec.numeric ? "Search by accounting words, SEC tag or formula name. Company-defined or unconfirmed fields are optional."
      : "This source supports raw inspection only. Open Advanced source tables and notes to browse its fields.")
    : "No ready SEC catalog found in the data folder. Refresh after a catalog is prepared."), !!state.sec.restoreError);
  if (state.sec.catalog) await searchSecDictionary(true);
}
async function searchSecDictionary(reset = false) {
  const catalog = state.sec.catalog;
  if (!catalog) { $("#sec-dictionary-status").textContent = "Choose a source catalog first."; return; }
  if (reset) state.sec.dictionaryOffset = 0;
  const query = $("#sec-dictionary-query").value.trim(), offset = state.sec.dictionaryOffset || 0;
  const view = secDictionaryView();
  const include_custom = view === "recipes" && $("#sec-dictionary-custom").checked;
  const request = state.sec.dictionaryLoad = (state.sec.dictionaryLoad || 0) + 1;
  const epoch = state.sec.selectionEpoch;
  const current = () => request === state.sec.dictionaryLoad && catalog === state.sec.catalog && epoch === state.sec.selectionEpoch &&
    view === secDictionaryView() && query === $("#sec-dictionary-query").value.trim() &&
    include_custom === (view === "recipes" && $("#sec-dictionary-custom").checked);
  $("#sec-dictionary-prev").disabled = $("#sec-dictionary-next").disabled = true;
  $("#sec-dictionary-build").hidden = true;
  $("#sec-dictionary-status").textContent = "Reading dictionary metadata…";
  setChildren($("#sec-dictionary-items"));
  let result;
  try { result = await get("sec-dictionary", {catalog, query, offset, include_custom, view}); }
  catch (error) { if (!current()) return; $("#sec-dictionary-status").textContent = error.message; return; }
  if (!current()) return;
  state.sec.dictionaryStatus = result.status;
  state.sec.executionStatus = result.execution?.status;
  const executionMissing = result.status === "ready" && state.sec.executionStatus && state.sec.executionStatus !== "ready";
  $("#sec-dictionary-build").hidden = result.status !== "missing" && !executionMissing;
  $("#sec-dictionary-build").textContent = executionMissing ? "Prepare run index" : "Prepare shared dictionary";
  $("#sec-dictionary-build").disabled = state.sec.dictionaryBuildCatalog === catalog && state.job?.state === "running";
  $("#sec-dictionary-status").textContent = result.message || (result.status === "ready"
    ? `${result.field_count ?? "Unknown number of"} stored definitions · dictionary ${result.version}`
    : result.status === "missing" ? "Prepare the shared dictionary once before searching it."
    : "This source supports raw inspection only.");
  if (executionMissing) $("#sec-dictionary-status").textContent += " " + (result.execution.message || "Prepare the run index before running formulas.");
  if (result.truncated && !/partial/i.test($("#sec-dictionary-status").textContent))
    $("#sec-dictionary-status").textContent += " Partial search results and variant lists, not global counts. Narrow the query or use an exact official name.";
  if (view === "company" && result.status === "ready")
    $("#sec-dictionary-status").textContent += !query || result.search_scope === "company_starter"
      ? " Starter concepts only, not the exhaustive company-field universe. Search to find other concepts."
      : " Company concept matches; observed variants do not promise coverage or data quality.";
  $("#sec-dictionary-status").title = JSON.stringify({version: result.version, catalog_sha256: result.catalog_sha256, path: result.path});
  const items = result.items || [];
  const scope = {view, request};
  setChildren($("#sec-dictionary-items"), ...items.map(item => view === "company"
    ? renderSecCompanyItem(item, catalog, epoch, scope, result.truncated)
    : renderSecDictionaryItem(item, catalog, epoch, scope)));
  $("#sec-dictionary-page").textContent = result.status === "ready"
    ? `${offset + (items.length ? 1 : 0)}–${offset + items.length}` +
      (Number.isInteger(result.total) ? ` of ${result.total}` : " · total unknown") +
      (view === "company" ? (!query || result.search_scope === "company_starter" ? " starter concepts" : " concept matches") : " definition matches") +
      (result.truncated ? " · partial matches, not global counts" : "") : "";
  $("#sec-dictionary-prev").disabled = result.status !== "ready" || offset === 0;
  const hasMore = typeof result.has_more === "boolean" ? result.has_more :
    Number.isInteger(result.total) && offset + items.length < result.total;
  $("#sec-dictionary-next").disabled = result.status !== "ready" || !hasMore;
}
async function buildSecDictionary() {
  const executionMissing = state.sec.dictionaryStatus === "ready" && state.sec.executionStatus && state.sec.executionStatus !== "ready";
  if (!state.sec.catalog || (state.sec.dictionaryStatus !== "missing" && !executionMissing) || state.sec.dictionaryBuilding) return;
  const catalog = state.sec.catalog;
  state.sec.dictionaryBuilding = true; $("#sec-dictionary-build").disabled = true;
  try {
    const job = await api("sec-dictionary-build", {catalog});
    if (catalog !== state.sec.catalog) return;
    state.sec.dictionaryStatus = "building"; $("#sec-dictionary-build").hidden = true;
    state.sec.dictionaryBuildCatalog = catalog;
    $("#sec-dictionary-status").textContent = (executionMissing ? "Run index" : "Dictionary") + " preparation started in the background. Press Search after it finishes.";
    startPolling(job.job || job);
  } finally { state.sec.dictionaryBuilding = false; $("#sec-dictionary-build").disabled = false; }
}
function secInsertionCurrent(catalog, epoch, scope) {
  return state.sec.mode === "dictionary" && !state.sec.restoreLoading && !state.sec.restoreError && catalog === state.sec.catalog && epoch === state.sec.selectionEpoch &&
    (!scope || (scope.view === secDictionaryView() && scope.request === state.sec.dictionaryLoad));
}
function insertSecField(item, catalog = state.sec.catalog, epoch = state.sec.selectionEpoch, scope = null) {
  if (!secInsertionCurrent(catalog, epoch, scope))
    throw new Error("The SEC selection changed. Review it before inserting.");
  if (secAliasUnavailable(item)) throw new Error(secFormulaDisplay(item));
  if (item?.alias_pending) throw new Error("Descriptive name must be verified before insertion.");
  if (!/^sec_[A-Za-z][A-Za-z0-9_]*$/.test(item?.name)) throw new Error("No official numeric field name is available.");
  const id = state.sec.editor || (state.view === "lab" ? "#lab-text" : "#alpha"), editor = $(id);
  const text = (id === "#alpha" ? "$" : "") + item.name;
  editor.setRangeText(text, editor.selectionStart, editor.selectionEnd, "end");
  $("#sec-dialog").close(); editor.focus();
  if (id === "#alpha") onAlphaInput(); else onLabInput();
  persist();
}
async function inspectSecRows(reset = false) {
  if (!state.sec.catalog) throw new Error("Choose a ready SEC source catalog first.");
  if (reset) { state.sec.rowsOffset = 0; state.sec.rowFilters = $("#sec-filters").value; state.sec.rowTable = $("#sec-table").value; }
  const catalog = state.sec.catalog, table = state.sec.rowTable, filters = state.sec.rowFilters, offset = state.sec.rowsOffset;
  const epoch = state.sec.selectionEpoch;
  const request = state.sec.rowsLoad = (state.sec.rowsLoad || 0) + 1;
  const current = () => request === state.sec.rowsLoad && catalog === state.sec.catalog && epoch === state.sec.selectionEpoch &&
    table === state.sec.rowTable && filters === state.sec.rowFilters && offset === state.sec.rowsOffset;
  $("#sec-rows").textContent = "Loading source rows…"; $("#sec-rows-prev").disabled = $("#sec-rows-next").disabled = true;
  let result;
  try { result = await get("sec-rows", { catalog, table, filters, offset }); }
  catch (error) { if (!current()) return; $("#sec-rows").textContent = "Could not read rows. Check filters and press Read rows to retry."; throw error; }
  if (!current()) return;
  const rows = Array.isArray(result.rows) ? result.rows : result.rows.rows || [];
  $("#sec-rows").textContent = JSON.stringify({ metadata: result.metadata, rows }, null, 2);
  $("#sec-rows-page").textContent = rows.length ? "Rows " + (offset + 1) + "–" + (offset + rows.length) : "No source rows match these filters.";
  $("#sec-rows-prev").disabled = offset === 0; $("#sec-rows-next").disabled = rows.length < 100;
}
function bindSecBrowser() {
  for (const id of ["#alpha", "#lab-text"]) $(id).addEventListener("focus", () => { state.sec.editor = id; });
  $("#sec-open").addEventListener("click", secAction(async () => {
    state.sec.editor = state.view === "lab" ? "#lab-text" : "#alpha";
    $("#sec-dialog").showModal(); renderSecBindings(); await loadSecCatalogs();
  }));
  $("#sec-switch-dictionary").addEventListener("click", secAction(() => {
    resetSecSwitchConfirmation();
    const current = secSwitchState();
    if (!current) throw new Error("Restore valid legacy inputs before switching to the shared dictionary.");
    state.sec.switchConfirmation = current;
    $("#sec-switch-confirmation").hidden = false;
    $("#sec-switch-dictionary").ariaExpanded = "true";
    $("#sec-switch-confirmation").focus();
  }));
  $("#sec-switch-cancel").addEventListener("click", () => {
    resetSecSwitchConfirmation(); $("#sec-switch-dictionary").focus();
  });
  $("#sec-switch-confirm").addEventListener("click", secAction(async () => {
    const expected = state.sec.switchConfirmation, current = secSwitchState();
    resetSecSwitchConfirmation();
    if (!expected) return;
    if (expected !== current) throw new Error("SEC inputs changed. Review the current inputs and choose Switch to dictionary again.");
    state.sec.mode = "dictionary"; state.sec.bindings = {}; state.sec.registry = null;
    resetSecSelection(); secChanged(); $("#sec-dictionary-query").focus(); await searchSecDictionary(true);
  }));
  $("#sec-dictionary-search").addEventListener("click", secAction(() => searchSecDictionary(true)));
  renderSecDictionaryView();
  $("#sec-dictionary-view").addEventListener("change", secAction(async () => {
    state.sec.dictionaryView = $("#sec-dictionary-view").value === "recipes" ? "recipes" : "company";
    resetSecSelection();
    $("#sec-dictionary-page").textContent = "";
    renderSecDictionaryView();
    await searchSecDictionary(true);
  }));
  $("#sec-dictionary-query").addEventListener("keydown", secAction(event => {
    if (event.key === "Enter") { event.preventDefault(); return searchSecDictionary(true); }
  }));
  $("#sec-dictionary-query").addEventListener("input", () => {
    state.sec.dictionaryLoad = (state.sec.dictionaryLoad || 0) + 1;
    $("#sec-dictionary-prev").disabled = $("#sec-dictionary-next").disabled = true;
  });
  $("#sec-dictionary-custom").addEventListener("change", secAction(() => searchSecDictionary(true)));
  $("#sec-dictionary-prev").addEventListener("click", secAction(() => { state.sec.dictionaryOffset = Math.max(0, (state.sec.dictionaryOffset || 0) - 50); return searchSecDictionary(); }));
  $("#sec-dictionary-next").addEventListener("click", secAction(() => { state.sec.dictionaryOffset = (state.sec.dictionaryOffset || 0) + 50; return searchSecDictionary(); }));
  $("#sec-dictionary-build").addEventListener("click", secAction(buildSecDictionary));
  $("#sec-close").addEventListener("click", () => { resetSecSwitchConfirmation(); $("#sec-dialog").close(); });
  $("#sec-dialog").addEventListener("close", resetSecSwitchConfirmation);
  $("#sec-reload").addEventListener("click", secAction(async () => { resetSecSelection(); await loadSecCatalogs(); }));
  $("#sec-catalog").addEventListener("change", secAction(async () => {
    if (Object.keys(state.sec.bindings).length || state.sec.restoreLoading || state.sec.restoreError) {
      $("#sec-catalog").value = state.sec.catalog;
      throw new Error("Restore the original source or explicitly clear active inputs before changing catalog. Existing inputs are preserved.");
    }
    state.sec.catalog = $("#sec-catalog").value; state.sec.numeric = false; resetSecSelection();
    state.sec.registry = null; secChanged(); await loadSecCatalogs();
  }));
  $("#sec-clear").addEventListener("click", () => {
    state.sec.bindings = {}; state.sec.registry = null; state.sec.mode = "dictionary";
    resetSecSelection(); secChanged();
  });
  $("#sec-inspect").addEventListener("click", secAction(() => inspectSecRows(true)));
  $("#sec-rows-prev").addEventListener("click", secAction(() => { state.sec.rowsOffset = Math.max(0, state.sec.rowsOffset - 100); return inspectSecRows(); }));
  $("#sec-rows-next").addEventListener("click", secAction(() => { state.sec.rowsOffset += 100; return inspectSecRows(); }));
}
// ================================================================= editor
function tokenize(text) {
  const out = [];
  const pattern = /(\$[A-Za-z_][A-Za-z0-9_]*)|([A-Za-z_][A-Za-z0-9_]*)|(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|([+\-*/<>=!&|]+)|([(),])|(\s+)|(.)/g;
  let match;
  while ((match = pattern.exec(text))) {
    const [token, feature, word, number, op, punct, space] = match;
    let cls = "tok-bad";
    if (feature) cls = FEATURES.has(feature.slice(1)) || Object.hasOwn(state.sec.bindings, feature.slice(1)) || state.sec.resolved.qlib?.items.some(input => input.name === feature.slice(1)) ? "tok-feature" : "tok-bad";
    else if (word) cls = OPERATORS.has(word) ? "tok-func" : "tok-bad";
    else if (number) cls = "tok-num";
    else if (op) cls = "tok-op";
    else if (punct) cls = "tok-punct";
    else if (space) cls = "";
    out.push(cls ? `<span class="${cls}">${escapeHtml(token)}</span>` : escapeHtml(token));
  }
  return out.join("");
}
function renderEditor() {
  const text = $("#alpha").value;
  $("#highlight").innerHTML = text.split("\n").map((line) => line.trimStart().startsWith("#")
    ? `<span class="tok-comment">${escapeHtml(line)}</span>` : tokenize(line)).join("\n") + "\n";
  const lines = Math.max(1, text.split("\n").length);
  $("#gutter").textContent = Array.from({ length: lines }, (_, i) => i + 1).join("\n");
  $("#code-editor").style.height = `${Math.max(84, Math.min(224, lines * 20 + 24))}px`;  // grows for a batch, up to ten lines
  syncEditorScroll();
}
function syncEditorScroll() {
  const area = $("#alpha");
  $("#highlight").scrollTop = area.scrollTop;
  $("#gutter").scrollTop = area.scrollTop;
}
let alphaTimer = null;
let alphaCheckEpoch = 0;
function onAlphaInput() {
  ++alphaCheckEpoch; state.alphaValid = false; delete state.sec.resolved.qlib;
  updateSecRunButtons(); renderSecBindings(); renderEditor(); clearTimeout(alphaTimer); alphaTimer = setTimeout(checkAlpha, 280);
}
async function checkAlpha() {
  const box = $("#alpha-status");
  const text = $("#alpha").value;
  const epoch = ++alphaCheckEpoch, generation = state.sec.generation;
  const current = () => epoch === alphaCheckEpoch && generation === state.sec.generation && text === $("#alpha").value;
  state.alphaValid = false; updateSecRunButtons();
  try {
    const inputs = JSON.stringify(secRequest());
    const result = await api("check-alpha", { alpha: text, ...secRequest() });
    if (!current() || inputs !== JSON.stringify(secRequest())) return false;
    state.alphaValid = result.ok === true;
    acceptSecInputs(result, "qlib"); renderEditor(); updateSecRunButtons();
    state.alphaCount = result.count || 0;
    box.className = `syntax ${result.ok ? "ok" : "bad"}`;
    box.textContent = `${result.ok ? "✓" : "✕"} ${result.ok ? (result.count > 1 ? `${result.count} alphas · batch` : "Valid expression") : secEditorError(result.message)}`;
    box.title = result.ok ? result.message : secEditorError(result.message);
    updateRunLabel();
    return result.ok === true;
  } catch (error) {
    if (!current()) return false;
    state.alphaValid = false; delete state.sec.resolved.qlib; renderSecBindings(); updateSecRunButtons();
    box.className = "syntax bad"; box.textContent = secEditorError(error.message); return false;
  }
}
function setAlpha(text) { $("#alpha").value = text; onAlphaInput(); }
function updateRunLabel() {
  updateSecRunButtons(); renderSecBindings();
  const many = state.view === "research" && state.alphaCount > 1;
  $("#run-label").textContent = state.view === "auto" ? "Start research" : state.view === "lab" ? "Simulate alphas" : many ? `Validate ${state.alphaCount} alphas` : "Run validation";
  $("#run-2").firstChild.textContent = many ? `Validate ${state.alphaCount} alphas` : "Run validation";
}

// ================================================================= views & layout
function setView(view) {
  if (!VIEWS.includes(view)) view = "lab";
  state.view = view;
  document.body.dataset.view = view;
  $$(".act[data-view]").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  for (const group of [".xview", ".cview", ".iview"]) $$(group).forEach((n) => n.classList.toggle("active", n.dataset.view === view));
  updateRunLabel();
  updateStatusRight();
  if (view === "data") { refreshDatasets(); showSources(); }
  else if (view === "evidence") { workflowUpdateButtons(); }
  else if (view === "lab") { refreshLab(); requestAnimationFrame(() => { renderLabPeriod(); renderLabCharts(); }); }
  else if (view === "auto") { refreshResearch(); renderResearchToolbar(); }
  else {
    // A remembered run is only a path until the validation view is opened.
    const path = state.run || (state.runs[0] && state.runs[0].path);
    if (path && (!state.runDetail || state.runDetail.path !== path)) openRun(path);
    else requestAnimationFrame(() => { renderPeriod(); renderResultTab(); });
  }
  if (view !== "auto") renderJob(state.job);
}
function setResultTab(tab) {
  state.resultTab = tab;
  $$("#main-tabs .tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === tab));
  $$(".cview[data-view=research] .tab-body").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  persist();
  requestAnimationFrame(renderResultTab);
}
function setBottomTab(tab) {
  state.bottomTab = tab;
  $$("#bottom-tabs .tab").forEach((t) => t.classList.toggle("active", t.dataset.ptab === tab));
  $$(".ptab").forEach((p) => p.classList.toggle("active", p.dataset.ptab === tab));
  if (tab === "console") $("#console-badge").hidden = true;
  if (tab === "universe") renderUniversePanel();
}
function toggleBottom(force) {
  const center = $("#center");
  const hide = force === undefined ? !center.classList.contains("no-bottom") : !force;
  center.classList.toggle("no-bottom", hide);
  saveLayout();
  requestAnimationFrame(redrawAll);
}
function redrawAll() {
  if (state.view === "research") { renderPeriod(); renderResultTab(); }
  else if (state.view === "lab") { renderLabPeriod(); renderLabCharts(); }
  else if (state.view === "auto") renderResearchContent();
  renderUniversePanel();
}

function bindResizers() {
  const root = document.documentElement;
  const limits = { explorer: [220, 480], inspector: [250, 460], bottom: [110, 640] };
  $$(".resizer").forEach((handle) => {
    handle.addEventListener("mousedown", (start) => {
      start.preventDefault();
      const kind = handle.dataset.resize, [min, max] = limits[kind];
      const initial = parseFloat(getComputedStyle(root).getPropertyValue(`--${kind === "bottom" ? "bottom-h" : kind + "-w"}`));
      handle.classList.add("dragging");
      document.body.style.cursor = kind === "bottom" ? "row-resize" : "col-resize";
      const move = (event) => {
        const delta = kind === "explorer" ? event.clientX - start.clientX : kind === "inspector" ? start.clientX - event.clientX : start.clientY - event.clientY;
        const limit = kind === "bottom" ? Math.min(max, window.innerHeight * 0.6) : max;
        root.style.setProperty(`--${kind === "bottom" ? "bottom-h" : kind + "-w"}`, `${Math.round(Math.max(min, Math.min(limit, initial + delta)))}px`);
      };
      const up = () => {
        handle.classList.remove("dragging"); document.body.style.cursor = "";
        window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up);
        saveLayout(); redrawAll();
      };
      window.addEventListener("mousemove", move); window.addEventListener("mouseup", up);
    });
  });
}
function saveLayout() {
  const style = getComputedStyle(document.documentElement);
  state.settings.layout = {
    explorer: style.getPropertyValue("--explorer-w").trim(), inspector: style.getPropertyValue("--inspector-w").trim(),
    bottom: style.getPropertyValue("--bottom-h").trim(), bottomHidden: $("#center").classList.contains("no-bottom"),
    inspectorHidden: document.body.classList.contains("no-inspector"),
  };
  persist();
}
function toggleInspector(force) {
  const hide = force === undefined ? !document.body.classList.contains("no-inspector") : !force;
  document.body.classList.toggle("no-inspector", hide);
  saveLayout();
  requestAnimationFrame(redrawAll);
}
function restoreLayout(layout) {
  if (!layout) { $("#center").classList.add("no-bottom"); return; }  // first launch: the panel opens with the first task
  document.body.classList.toggle("no-inspector", !!layout.inspectorHidden);
  const root = document.documentElement;
  for (const [key, prop] of [["explorer", "--explorer-w"], ["inspector", "--inspector-w"], ["bottom", "--bottom-h"]]) {
    if (/^\d+px$/.test(layout[key] || "")) root.style.setProperty(prop, layout[key]);
  }
  $("#center").classList.toggle("no-bottom", !!layout.bottomHidden);
}

// ================================================================= parameters
function setMode(mode) {
  state.mode = mode === "long-only" ? "long-only" : "signed";
  $$("#mode button").forEach((b) => b.classList.toggle("active", b.dataset.mode === state.mode));
  $$(".prop[data-for]").forEach((p) => { p.hidden = p.dataset.for !== state.mode; });
  const signed = state.overview && state.overview.signed;
  const weighting = {
    raw: "Raw scores become weights scaled to the gross exposure; no neutrality target",
    rank: "Each session's scores are ranked and centered: dollar-neutral, outliers cannot dominate",
    zscore: "Each session's scores are z-scored (the demeaned alpha): dollar-neutral",
  }[state.transform];
  $("#assumptions").textContent = state.mode === "signed"
    ? `${weighting}; no cash borrowing, no forced liquidation.${signed && !signed.available ? " Signed engine unavailable: " + signed.detail : ""}`
    : "Buys the top-k scores with 95% of equity and replaces up to the daily count. Names leaving the universe rank last and are sold first.";
}
function setTransform(transform) {
  state.transform = transform === "rank" || transform === "zscore" ? transform : "raw";
  $$("#transform button").forEach((b) => {
    const on = b.dataset.transform === state.transform;
    b.classList.toggle("active", on);
    b.setAttribute("aria-checked", String(on));
  });
  setMode(state.mode);
}
const TRANSFORM_LABELS = { raw: "Raw alpha", rank: "Cross-sectional rank", zscore: "Cross-sectional z-score" };
// The service flags runs under one year of sessions (workspace.SHORT_WINDOW_SESSIONS).
const SHORT_WINDOW_NOTE = (n) => `Only ${fmt.int(n)} sessions, under one year: annualized Sharpe and CAGR extrapolate a short window.`;
const FIELD_IDS = { gross: "#gross", borrow: "#borrow", topk: "#topk", ndrop: "#ndrop", start: "#start", end: "#end", warmup: "#warmup", cost: "#cost", cash: "#cash" };
function resetParams() {
  for (const [key, id] of Object.entries(FIELD_IDS)) $(id).value = DEFAULTS[key];
  $("#gross-range").value = DEFAULTS.gross;
  setTransform("raw");
  renderPeriod();
  persist();
}
function formValues() {
  return {
    alpha: $("#alpha").value, strategy_mode: state.mode, dataset: state.datasetPath || "", ...secRequest(),
    universe: $("#universe").value, start: $("#start").value, end: $("#end").value, warmup_sessions: $("#warmup").value,
    cost_bps: $("#cost").value, cash_rate: String((parseFloat($("#cash").value) || 0) / 100),
    topk: $("#topk").value, n_drop: $("#ndrop").value, gross_exposure: $("#gross").value,
    borrow_rate: String((parseFloat($("#borrow").value) || 0) / 100), signal_transform: state.transform,
  };
}
let persistTimer = null;
function persist() {
  if (state.restoring) return;
  clearTimeout(persistTimer);
  persistTimer = setTimeout(() => {
    const s = state.settings;
    s.sec = { ...secConfig(), restore_error: state.sec.restoreError || null };
    for (const [key, id] of Object.entries(FIELD_IDS)) s[key] = $(id).value;
    s.lab = { text: $("#lab-text").value, universe: $("#lab-universe").value, delay: state.lab.delay, neutralization: $("#lab-neutralization").value,
      decay: $("#lab-decay").value, truncation: $("#lab-truncation").value, pasteurize: $("#lab-pasteurize").checked, cost: $("#lab-cost").value,
      start: $("#lab-start").value, end: $("#lab-end").value, oos: $("#lab-oos").value, workers: $("#lab-workers").value, filter: state.lab.filter,
      dataset: state.lab.datasetPath || "",
      ...(state.lab.groupExplicit ? { group_overlay: state.lab.groupOverlay, group_identity: state.lab.groupIdentity,
        group_restore_error: state.lab.groupRestoreError } : {}) };
    // An unresolved or failed Validation selection must survive unrelated saves.
    const validationDataset = state.datasetPath || (typeof s.dataset === "string" ? s.dataset : "");
    Object.assign(s, { alpha: $("#alpha").value, mode: state.mode, transform: state.transform, universe: $("#universe").value, dataset: validationDataset,
      table: $("#table").value, imp_start: $("#imp-start").value, imp_size: $("#imp-size").value, imp_adv: $("#imp-adv").value,
      imp_history: $("#imp-history").value, imp_workers: $("#imp-workers").value, imp_minprice: $("#imp-minprice").value,
      imp_groups: $("#imp-groups").value, scale: state.scale, run: state.run || "", view: state.view,
      run_filter: state.runFilter, run_sort: state.runSort, result_tab: state.resultTab, more_open: state.moreOpen ? "1" : "" });
    const flat = {};
    for (const [k, v] of Object.entries(s)) flat[k] = typeof v === "object" && v !== null ? JSON.stringify(v) : v;
    api("settings", flat).catch(() => {});
  }, 400);
}
function restoreSettings(s) {
  for (const [key, id] of Object.entries(FIELD_IDS)) if (typeof s[key] === "string") $(id).value = s[key];
  for (const [key, id] of [["table", "#table"], ["imp_start", "#imp-start"], ["imp_size", "#imp-size"], ["imp_adv", "#imp-adv"],
    ["imp_history", "#imp-history"], ["imp_workers", "#imp-workers"], ["imp_minprice", "#imp-minprice"], ["imp_groups", "#imp-groups"]])
    if (typeof s[key] === "string" && s[key]) $(id).value = s[key];
  $("#gross-range").value = $("#gross").value;
  if (typeof s.transform === "string") setTransform(s.transform);
  if (typeof s.alpha === "string") $("#alpha").value = s.alpha;
  if (s.scale === "log" || s.scale === "linear") state.scale = s.scale;
  if (["all", "starred", "factors", "trash"].includes(s.run_filter)) state.runFilter = s.run_filter === "trash" ? "all" : s.run_filter;
  if (["created", "sharpe", "cagr", "max_drawdown"].includes(s.run_sort)) state.runSort = s.run_sort;
  if (["overview", "returns", "signal", "details"].includes(s.result_tab)) state.resultTab = s.result_tab;
  state.moreOpen = s.more_open === "1";
  s.layout = preferenceObject(s.layout);
  s.lab = preferenceObject(s.lab);
  restoreLab(s.lab);
  restoreLayout(s.layout);
}
function preferenceObject(value) {
  try { if (typeof value === "string") value = JSON.parse(value); } catch (e) { return {}; }
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

// ================================================================= datasets
function priceEligible(row) { return !!row && row.status === "ready" && row.price_eligible === true; }
function priceRow(path) { return typeof path === "string" ? state.datasets.find((r) => r.path === path && priceEligible(r)) : null; }
function defaultPriceRow() { return state.datasets.find((r) => priceEligible(r) && r.is_default === true); }
function datasetMetadata(row) {
  const fields = Array.isArray(row.fields) ? row.fields.join(", ") : "fields unavailable";
  const packed = row.storage_kind === "qlib-zip-store-v1" ? "packed · recommended for faster loading" : (row.storage_kind || "storage unavailable");
  return `${row.data_role || "role unavailable"} · ${packed} · ${fields}${row.compatibility_note ? " · " + row.compatibility_note : ""}`;
}
function renderRecovery() {
  const note = $("#recovery-note");
  const messages = [state.recovery.lab && `Alpha Lab: ${state.recovery.lab}`, state.recovery.research && `Qlib Validation: ${state.recovery.research}`].filter(Boolean);
  note.hidden = !messages.length;
  setChildren(note, ...messages.map((text) => el("div", { text })), state.recoveryPrice ? el("button", {
    class: "btn small", text: `Use saved backtest prices: ${datasetTitle(state.recoveryPrice)}`,
    onclick: async () => {
      const row = state.recoveryPrice;
      if (row && await loadDataset(row.path)) {
        state.recoveryPrice = null;
        state.recovery.research = `Switched backtest prices to ${datasetTitle(row)}. Your saved expression is unchanged; native Qlib syntax is required.`;
        renderRecovery();
        toast("Backtest prices switched", `${datasetTitle(row)}. Saved expression unchanged.`, "good");
      }
    },
  }) : null);
}
async function restorePriceSelections() {
  const s = state.settings, saved = priceRow(s.dataset), fallback = priceRow(s.price_dataset);
  if (saved) await loadDataset(saved.path, true);
  else if (s.dataset) {
    const old = state.datasets.find((r) => r.path === s.dataset);
    state.recovery.research = `Saved dataset is unavailable or cannot supply trading prices.${old && old.compatibility_note ? " " + old.compatibility_note : ""} Your saved expression is unchanged. Choose ordinary OHLCV prices; native Qlib syntax is required.`;
    state.recoveryPrice = fallback || null;
  } else {
    const row = fallback || defaultPriceRow();
    if (row) await loadDataset(row.path, true);
    else state.recovery.research = "Choose an eligible trading-price dataset in the toolbar. No eligible default is configured.";
  }
  const labSaved = priceRow(s.lab.dataset), labDefault = defaultPriceRow();
  if (labSaved) await loadDataset(labSaved.path, true, "lab");
  else if (labDefault) {
    await loadDataset(labDefault.path, true, "lab");
    if (state.lab.dataset) state.recovery.lab = `Selected the configured trading-price default: ${datasetTitle(labDefault)}.${s.lab.dataset ? " The saved Lab dataset is unavailable or incompatible." : ""}`;
  } else state.recovery.lab = "Choose Trading prices in the toolbar. No eligible default is configured; saved expressions have been kept.";
  renderRecovery();
}
async function loadDataset(path, quiet = false, target = "research") {
  const owner = target === "lab" ? state.lab : state;
  const row = path ? priceRow(path) : defaultPriceRow();
  const request = ++owner.datasetLoad;
  owner.dataset = null; owner.datasetPath = null;
  if (target === "lab") { ++state.lab.groupEpoch; state.lab.groupMetadata = null; renderLabGroups(); }
  if (!row) {
    state.recovery[target] = "No eligible trading prices selected. Choose ready ordinary OHLCV data in the toolbar.";
    updateDatasetSelection(target); renderRecovery();
    return false;
  }
  try {
    const detail = await get("dataset", { path: row.path });
    if (request !== owner.datasetLoad) return false;
    owner.dataset = { ...detail, ...row };
    owner.datasetPath = row.path;
  } catch (error) {
    if (request !== owner.datasetLoad) return false;
    state.recovery[target] = `Cannot load the selected trading prices: ${error.message}. Choose another eligible dataset; no replacement was selected.`;
    updateDatasetSelection(target); renderRecovery();
    if (!quiet) toast("Cannot load trading prices", error.message, "bad");
    return false;
  }
  state.recovery[target] = "";
  if (target === "research") {
    state.recoveryPrice = null;
    state.settings.dataset = row.path;
  }
  updateDatasetSelection(target); renderRecovery();
  if (target === "lab") await loadLabGroups();
  persist();
  return true;
}
function updateDatasetSelection(target) {
  const owner = target === "lab" ? state.lab : state;
  const id = target === "lab" ? "#tb-lab-dataset" : "#tb-dataset", d = owner.dataset;
  $(id + "-name").textContent = d ? datasetTitle(d) : "Choose prices…";
  $(id).title = d ? `${d.path}\n${datasetMetadata(d)}` : "Choose eligible trading prices (ordinary OHLCV)";
  if (target === "lab") renderLabPeriod();
  else { fillUniverses(); renderPeriod(); }
  updateStatusRight();
}
function datasetTitle(d) {
  d = { ...d, name: typeof d.name === "string" ? d.name : "Dataset" };
  const source = /^sharadar/.test(d.name) ? "Sharadar" : /^historicaldata/.test(d.name) ? "HistoricalData" : /^yahoo/.test(d.name) ? "Yahoo" : /^stooq/.test(d.name) ? "Stooq" : "";
  const stamp = (d.name.match(/(\d{4})(\d{2})(\d{2})T/) || []).slice(1).join("-");
  const universe = Object.keys(d.universes || {})[0];
  const title = [source, universe ? universe.toUpperCase() : "", stamp].filter(Boolean).join(" · ") || d.name;
  return d.storage_kind === "qlib-zip-store-v1" ? `${title} · Packed` : d.storage_kind === "qlib-binary-tree" ? `${title} · Unpacked` : title;
}
function fillUniverses() {
  const select = $("#universe"), wanted = state.settings.universe;
  const universes = (state.dataset && state.dataset.universes) || {}, names = Object.keys(universes).sort();
  setChildren(select, ...names.map((n) => el("option", { value: n, text: n })), el("option", { value: "", text: "all symbols" }));
  select.value = names.includes(wanted) || wanted === "" ? wanted : (names[0] || "");
  renderUniversePanel();
  updateStatusRight();
}
function renderUniversePanel() {
  const panel = $("#universe-panel"), d = state.dataset, name = $("#universe").value;
  const u = d && (d.universes || {})[name];
  if (!u) { setChildren(panel, el("div", { class: "muted pad", text: "The selected universe is the dataset's fixed symbol list." })); return; }
  const chartBox = el("div", { class: "bars" });
  setChildren(panel, 
    el("div", { class: "u-head" }, el("span", { text: `${name} · top ${fmt.int(u.size)} by ${u.adv_sessions}-session dollar volume · re-ranked monthly` }),
      el("span", { class: "mono muted", text: `${fmt.int(u.unique_members)} names · full from ${u.full_size_from || "—"}` })),
    chartBox);
  if ($(".ptab[data-ptab=universe]").classList.contains("active")) drawWhenSized(chartBox, () => membershipBars(chartBox, (d.ranking_series || {})[name], u.size));
}
function membershipBars(container, series, size) {
  if (!series || !series.length) { setChildren(container, el("div", { class: "muted", text: "No rankings." })); return; }
  barChart(container, { labels: series.map((p) => p.date), values: series.map((p) => p.members), dense: true, format: fmt.int,
    color: (v) => v >= size ? css("--cat-1") : css("--series-dim"), tooltip: (i) => `${series[i].date}: ${fmt.int(series[i].members)} members of ${fmt.int(series[i].eligible)} eligible`,
    label: "Universe members per ranking" });
}
function showDatasetPopover(target = state.view === "lab" ? "lab" : "research") {
  if (target !== "lab") target = "research";
  const pop = $("#dataset-popover");
  if (!pop.hidden && pop.dataset.target === target) { pop.hidden = true; return; }
  pop.dataset.target = target;
  get("datasets").then((rows) => {
    if (pop.dataset.target !== target) return;
    state.datasets = rows;
    const owner = target === "lab" ? state.lab : state;
    const box = $(target === "lab" ? "#tb-lab-dataset" : "#tb-dataset").getBoundingClientRect();
    pop.style.left = `${box.left}px`; pop.style.top = `${box.bottom + 4}px`;
    const ready = rows.filter((r) => r.status === "ready");
    setChildren(pop, ...(ready.length ? ready.map((r) => el("button", {
      class: owner.datasetPath === r.path ? "current" : "", disabled: !priceEligible(r),
      onclick: () => { if (!priceEligible(r)) return; pop.hidden = true; loadDataset(r.path, false, target); } },
      `${r.is_default ? "★ " : ""}${datasetTitle(r)}`, el("small", { text: `${r.start} → ${r.end} · ${fmt.int(r.symbols)} symbols` }),
      el("small", { text: `${priceEligible(r) ? "Trading prices" : "Not eligible for trading prices"} · ${datasetMetadata(r)}` })))
      : [el("div", { class: "list-empty", text: "No ready datasets. Import one in the Data view." })]),
      el("div", { class: "sep" }), el("button", { onclick: () => { pop.hidden = true; setView("data"); } }, "Manage datasets…"));
    pop.hidden = false;
  }).catch((e) => toast("Cannot list datasets", e.message, "bad"));
}
async function refreshDatasets() {
  try { state.datasets = await get("datasets"); } catch (e) { toast("Cannot list datasets", e.message, "bad"); return; }
  $("#datasets-count").textContent = state.datasets.length || "";
  const list = $("#datasets-list");
  if (!state.datasets.length) { setChildren(list, el("div", { class: "list-empty", text: "No datasets yet. Refresh market data →" })); renderDatasetDetail(null); return; }
  const selected = state.detailDataset || (state.dataset && state.dataset.path) || state.datasets[0].path;
  setChildren(list, ...state.datasets.map((d) => el("div", {
    class: `dataset-item${d.path === selected ? " selected" : ""}`, onclick: () => { state.detailDataset = d.path; refreshDatasets(); },
    ondblclick: () => { if (priceEligible(d)) { loadDataset(d.path); setView("research"); } } },
    el("div", { class: "name" }, d.is_default ? el("span", { class: "star", text: "★", title: "Default dataset" }) : null, el("span", { text: datasetTitle(d) })),
    el("div", { class: "meta" }, el("span", { class: `tag ${d.status === "ready" ? "good" : d.status === "failed" ? "bad" : "warn"}`, text: d.status }),
      d.start ? el("span", { class: "mono", text: `${d.start.slice(0, 4)}–${(d.end || "").slice(0, 4)}` }) : null,
      d.symbols ? el("span", { class: "mono", text: `${fmt.int(d.symbols)} stocks` }) : null),
    el("div", { class: "ds-name", text: d.name, title: d.path }))));
  renderDatasetDetail(selected);
}
function kpiTile(label, value, note, cls = "") {
  return el("div", { class: "kpi" }, el("div", { class: "label", text: label }), el("div", { class: `value ${cls}`, text: value }),
    note ? el("div", { class: "note", text: note, title: note }) : null);
}
async function renderDatasetDetail(path) {
  const view = $("#dataset-view");
  if (!path) { setChildren(view, el("div", { class: "empty-state" }, el("div", { class: "empty-title", text: "No dataset yet" }), el("div", { class: "empty-sub", text: "Refresh market data with the panel on the right." }))); return; }
  let d;
  try { d = await get("dataset", { path }); } catch (e) { setChildren(view, el("div", { class: "empty-state", text: e.message })); return; }
  const audit = d.audit || {}, filter = audit.security_filter || {}, sessions = audit.session_filter || {}, invalid = audit.invalid_bars || {};
  const uname = Object.keys(d.universes || {})[0], u = uname ? d.universes[uname] : null;
  const builder = audit.table_builder || {};
  const fields = d.fields || [];
  const survivorship = ["sharadar", "historicaldata"].includes(builder.kind) ? ["good", "Includes delisted stocks"] : ["warn", "Survivors only (no delisted stocks)"];
  const bars = el("div", { class: "bars", style: { height: "150px" } });
  const excluded = Object.entries(filter.excluded || {});
  const limitations = audit.limitations || [];
  setChildren(view, 
    el("div", { class: "ds-head" },
      el("div", {}, el("div", { class: "ds-title", text: datasetTitle(d) }), el("div", { class: "ds-sub", text: `${d.name} · ${d.provider}` })),
      el("div", { class: "ds-actions" },
        el("button", { class: "btn small primary", text: "Use for Qlib Validation", disabled: !priceEligible(state.datasets.find((r) => r.path === path)), onclick: () => { loadDataset(d.path); setView("research"); } }),
        el("button", { class: "btn small", text: "Use for Alpha Lab", disabled: !priceEligible(state.datasets.find((r) => r.path === path)), onclick: () => { loadDataset(d.path, false, "lab"); setView("lab"); } }),
        el("button", { class: "btn small", text: d.is_default ? "Default ★" : "Set default", disabled: !priceEligible(state.datasets.find((r) => r.path === path)) || d.is_default,
          onclick: async () => { try { await api("default-dataset", { path: d.path }); toast("Default dataset updated", datasetTitle(d), "good"); refreshDatasets(); } catch (e) { toast("Cannot set default", e.message, "bad"); } } }),
        el("button", { class: "btn small", text: "Reveal", onclick: () => api("reveal", { path: d.path }).catch((e) => toast("Cannot reveal", e.message, "bad")) }))),
    el("div", { class: "health" },
      el("span", { class: `tag ${survivorship[0]}`, text: survivorship[1] }),
      el("span", { class: `tag ${fields.includes("close_raw") ? "good" : "warn"}`, text: fields.includes("close_raw") ? "Traded price" : "No traded price" }),
      el("span", { class: `tag ${fields.includes("dividend") ? "good" : ""}`, text: fields.includes("dividend") ? "Dividends" : "No dividends" }),
      el("span", { class: `tag ${fields.includes("cap") ? "good" : ""}`, text: fields.includes("cap") ? "Market cap" : "No market cap" }),
      el("span", { class: `tag ${fields.includes("vwap") ? "good" : ""}`, text: fields.includes("vwap") ? "VWAP" : "No VWAP" }),
      el("span", { class: `tag ${audit.groups ? "good" : ""}`, text: audit.groups ? "Sector groups" : "No sector groups" })),
    el("div", { class: "kpis six" },
      kpiTile("Period", `${(d.start || "").slice(0, 4)}–${(d.end || "").slice(0, 4)}`, `${d.start} → ${d.end}`),
      kpiTile("Sessions", fmt.int(d.sessions), "trading days"),
      kpiTile("Stocks", fmt.int(d.symbols), "ever in the universe"),
      kpiTile("Universe", u ? `top ${fmt.int(u.size)}` : "fixed", u ? `${fmt.int(u.unique_members)} names overall` : "explicit list"),
      kpiTile("Full from", u && u.full_size_from ? u.full_size_from : "—", "first complete ranking"),
      kpiTile("Excluded", fmt.int(excluded.reduce((a, [, v]) => a + (v.count || 0), 0)), "non-common securities")),
    u ? el("div", { class: "chart-card", style: { marginTop: "12px" } }, el("div", { class: "chart-head" }, el("span", { class: "chart-title", text: `${uname} members per monthly ranking` }),
      el("span", { class: "chart-note", text: `${u.adv_sessions}-session ADV · ≥${u.min_history_sessions} sessions of history · effective next session` })), bars) : null,
    limitations.length ? el("div", { class: "fact-card", style: { marginTop: "12px" } }, el("h3", { text: "Know before you trust a result" }),
      el("ul", { class: "limits" }, limitations.slice(0, 2).map((t) => el("li", { text: t })))) : null,
    el("details", { class: "adv audit" }, el("summary", { text: "Audit details" }),
      el("div", { class: "facts" },
        el("div", { class: "fact-card" }, el("h3", { text: "Security filter" }),
          el("table", { class: "table" }, el("thead", {}, el("tr", {}, el("th", { text: "Excluded" }), el("th", { class: "num", text: "Tickers" }), el("th", { text: "Examples" }))),
            el("tbody", {}, excluded.map(([reason, info]) => el("tr", {}, el("td", { text: reason.replace(/_/g, " ") }), el("td", { class: "num", text: fmt.int(info.count) }),
              el("td", { class: "ex", text: (info.examples || []).slice(0, 3).join(" ") }))),
              el("tr", {}, el("td", { text: "never ranked (not dumped)" }), el("td", { class: "num", text: fmt.int((filter.never_ranked_not_dumped || {}).count) }), el("td", { class: "ex", text: ((filter.never_ranked_not_dumped || {}).examples || []).slice(0, 3).join(" ") })))),
          el("p", { class: "prop-note", text: `${fmt.int(filter.common_stocks)} common stocks considered. ${filter.rule || ""}` })),
        el("div", { class: "fact-card" }, el("h3", { text: "Data checks" }),
          el("dl", { class: "kv" },
            el("dt", { text: "Rows in window" }), el("dd", { text: fmt.int((audit.source || {}).rows) }),
            el("dt", { text: "Source size" }), el("dd", { text: fmt.bytes((audit.source || {}).size_bytes) }),
            el("dt", { text: "Zero-volume bars" }), el("dd", { text: fmt.int(audit.zero_volume_rows) }),
            ...Object.entries(invalid).filter(([k]) => k !== "policy").flatMap(([k, v]) => [el("dt", { text: k.replace(/_/g, " ") }), el("dd", { text: fmt.int((v || {}).rows) })]),
            ...(builder.kind ? [el("dt", { text: "Isolated bad prints" }), el("dd", { text: fmt.int(builder.isolated_spikes) })] : []),
            ...(builder.delisted_stocks !== undefined ? [el("dt", { text: "Delisted stocks" }), el("dd", { text: fmt.int(builder.delisted_stocks) })] : []),
            el("dt", { text: "Dropped sessions" }), el("dd", {}, (sessions.dropped_sessions || []).length
              ? sessions.dropped_sessions.map((s) => el("div", { text: `${s.session} · ${fmt.int(s.bars)} bars` })) : "none")),
          el("p", { class: "prop-note", text: invalid.policy ? `Invalid bars: ${invalid.policy}.` : "" })),
        el("div", { class: "fact-card" }, el("h3", { text: "All limitations" }), el("ul", { class: "limits" }, limitations.map((t) => el("li", { text: t })))))));
  if (u) drawWhenSized(bars, () => membershipBars(bars, (d.ranking_series || {})[uname], u.size));
}

// ================================================================= backtest window (period bar)
function sessionIndex(calendar, date, mode) {
  // mode "start": first session >= date; "end": last session <= date
  let lo = 0, hi = calendar.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (calendar[mid] < date) lo = mid + 1; else hi = mid; }
  if (mode === "start") return lo;
  return calendar[lo] === date ? lo : lo - 1;
}
function coverageSeries(d, universe) {
  const calendar = d.calendar || [], series = ((d.ranking_series || {})[universe]) || [];
  if (!series.length) return null;
  const out = new Array(calendar.length).fill(null);
  let k = 0, value = null;
  for (let i = 0; i < calendar.length; i++) {
    while (k < series.length && series[k].date < calendar[i]) { value = series[k].members; k++; }
    out[i] = value;
  }
  return out;
}
function windowBounds(calendar, warmup) {
  const n = calendar.length;
  return { min: Math.min(Math.max(0, warmup), Math.max(0, n - 3)), max: Math.max(1, n - 2) };
}
function currentRange(calendar, bounds, startValue, endValue) {
  const i0 = startValue ? Math.max(bounds.min, sessionIndex(calendar, startValue, "start")) : bounds.min;
  const i1 = endValue ? Math.min(bounds.max, sessionIndex(calendar, endValue, "end")) : bounds.max;
  return [Math.min(i0, bounds.max - 1), Math.max(i1, Math.min(i0, bounds.max - 1) + 1)];
}
function periodSummary(calendar, i0, i1, defaults) {
  const n = i1 - i0 + 1;
  return el("span", {}, el("b", { text: `${calendar[i0]} → ${calendar[i1]}` }), ` · ${fmt.years(n)} · ${fmt.int(n)} sessions`,
    defaults ? el("span", { class: "muted", text: " · dataset default" }) : null);
}
function renderPeriod() {
  const box = $("#period-nav"), d = state.dataset;
  if (!box || state.view !== "research") return;
  const calendar = (d && d.calendar) || [];
  if (calendar.length < 10) { setChildren(box, el("div", { class: "chart-empty", text: "Choose a dataset to set the window." })); $("#period-summary").textContent = ""; return; }
  const warmup = parseInt($("#warmup").value, 10) || 252;
  const bounds = windowBounds(calendar, warmup);
  const [i0, i1] = currentRange(calendar, bounds, $("#start").value, $("#end").value);
  const universe = $("#universe").value, u = (d.universes || {})[universe];
  const markers = [];
  if (u && u.full_size_from) { const i = sessionIndex(calendar, u.full_size_from, "start"); if (i > 0 && i < calendar.length) markers.push({ i, label: `full ${universe}` }); }
  setChildren($("#period-summary"), periodSummary(calendar, i0, i1, !$("#start").value && !$("#end").value));
  drawWhenSized(box, () => rangeNavigator(box, {
    dates: calendar, area: coverageSeries(d, universe), minIndex: bounds.min, maxIndex: bounds.max, markers, range: [i0, i1], label: "Backtest window",
    onChange: (a, b, final) => {
      setChildren($("#period-summary"), periodSummary(calendar, a, b, false));
      if (final) {
        $("#start").value = a === bounds.min ? "" : calendar[a];
        $("#end").value = b === bounds.max ? "" : calendar[b];
        setChildren($("#period-summary"), periodSummary(calendar, a, b, a === bounds.min && b === bounds.max));
        persist();
      }
    },
  }));
}
function presetRange(kind, calendar, bounds, universeInfo) {
  const lastDate = calendar[bounds.max];
  if (kind === "all") return ["", ""];
  if (kind === "universe") {
    if (!universeInfo || !universeInfo.full_size_from) { toast("No full-size universe", "This dataset universe never reaches its full size."); return null; }
    const i = sessionIndex(calendar, universeInfo.full_size_from, "start") + 1;
    return [calendar[Math.max(bounds.min, Math.min(i, bounds.max - 1))], ""];
  }
  if (kind === "ytd") return [`${lastDate.slice(0, 4)}-01-01`, ""];
  const years = { "10y": 10, "5y": 5, "3y": 3, "1y": 1 }[kind];
  const start = new Date(`${lastDate}T00:00:00Z`);
  start.setUTCFullYear(start.getUTCFullYear() - years);
  const iso = start.toISOString().slice(0, 10);
  return [iso < calendar[bounds.min] ? "" : iso, ""];
}
function applyPreset(kind) {
  const d = state.dataset, calendar = (d && d.calendar) || [];
  if (calendar.length < 10) return;
  const bounds = windowBounds(calendar, parseInt($("#warmup").value, 10) || 252);
  const range = presetRange(kind, calendar, bounds, (d.universes || {})[$("#universe").value]);
  if (!range) return;
  [$("#start").value, $("#end").value] = range;
  renderPeriod(); persist();
}

// ================================================================= runs: list, filters, selection
async function refreshRuns(selectFirst = false) {
  try { state.runs = await get("runs"); } catch (e) { toast("Cannot list runs", e.message, "bad"); return; }
  if (state.runFilter === "trash") await refreshTrash();
  const known = new Set(state.runs.map((r) => r.path));
  state.selected = state.selected.filter((p) => known.has(p));
  renderRunList();
  if (selectFirst && state.view === "research" && !state.run && state.runs.length) openRun(state.runs[0].path);
}
async function refreshTrash() {
  try { state.trash = await get("trash"); } catch (e) { toast("Cannot read the trash", e.message, "bad"); }
}
function setRunFilter(filter) {
  state.runFilter = filter;
  $$("#run-filter button").forEach((b) => b.classList.toggle("active", b.dataset.filter === filter));
  if (filter !== "all" && filter !== "starred") state.factorFilter = null;
  if (filter === "trash") refreshTrash().then(renderRunList); else renderRunList();
  persist();
}
const runTitle = (r) => r.label || r.alpha || r.name;
function runMatches(r, query) {
  return !query || `${r.alpha} ${r.label || ""} ${r.factor_label || ""} ${r.mode} ${r.universe} ${r.status} ${r.created} ${r.dataset}`.toLowerCase().includes(query);
}
function sortRuns(rows, key) {
  if (key === "created") return rows;
  return rows.slice().sort((a, b) => {
    const x = a[key], y = b[key];
    if (!finite(x)) return 1; if (!finite(y)) return -1;
    return y - x;
  });
}
function visibleRuns() {
  const query = $("#run-search").value.trim().toLowerCase();
  let rows = state.runs.filter((r) => runMatches(r, query));
  if (state.runFilter === "starred") rows = rows.filter((r) => r.starred || r.factor_starred);
  if (state.factorFilter) rows = rows.filter((r) => r.factor === state.factorFilter);
  return sortRuns(rows, state.runSort);
}
function factorGroups() {
  const query = $("#run-search").value.trim().toLowerCase();
  const groups = new Map();
  for (const r of state.runs) {
    if (!runMatches(r, query)) continue;
    const g = groups.get(r.factor) || { factor: r.factor, alpha: r.alpha, label: r.factor_label, starred: r.factor_starred, runs: [] };
    g.runs.push(r);
    groups.set(r.factor, g);
  }
  const rows = Array.from(groups.values()).map((g) => {
    const done = g.runs.filter((r) => finite(r.sharpe));
    const full = done.filter((r) => !r.short_window);  // a few-week window's annualized Sharpe is not a fair best
    const best = (full.length ? full : done).reduce((m, r) => (m && m.sharpe >= r.sharpe ? m : r), null);
    return { ...g, best, bestSharpe: best ? best.sharpe : null, latest: g.runs[0].created, count: g.runs.length };
  });
  if (state.runSort === "sharpe") rows.sort((a, b) => (b.bestSharpe ?? -1e9) - (a.bestSharpe ?? -1e9));
  else if (state.runSort === "cagr") rows.sort((a, b) => ((b.best || {}).cagr ?? -1e9) - ((a.best || {}).cagr ?? -1e9));
  else if (state.runSort === "max_drawdown") rows.sort((a, b) => ((b.best || {}).max_drawdown ?? -1e9) - ((a.best || {}).max_drawdown ?? -1e9));
  return rows;
}
function yearsLabel(r) {
  if (!r.start || !r.end) return "";
  const a = r.start.slice(0, 4), b = r.end.slice(0, 4);
  return a === b ? a : `${a}–${b.slice(2)}`;
}
function starButton(on, title, onclick) {
  return el("button", { class: `ri-star${on ? " on" : ""}`, title, "aria-label": title, "aria-pressed": String(!!on),
    onclick: (e) => { e.stopPropagation(); onclick(); } }, icon("star", on ? "filled" : ""));
}
function runItem(r) {
  const selected = state.selected.includes(r.path) || (!state.selected.length && r.path === state.run);
  const node = el("div", { class: `run-item${selected ? " selected" : ""}${r.path === state.run ? " current" : ""}`, "data-path": r.path, role: "option", "aria-selected": String(selected),
    onclick: (e) => onRunClick(e, r), oncontextmenu: (e) => { e.preventDefault(); onRunContext(e, r); }, ondblclick: () => openRun(r.path) },
    el("div", { class: "ri-top" },
      starButton(r.starred, r.starred ? "Unstar run" : "Star run", () => starRuns([r], !r.starred)),
      el("span", { class: `ri-title${r.label ? "" : " mono"}`, text: runTitle(r), title: r.alpha }),
      r.status === "completed" ? el("span", { class: `ri-metric ${tone(r.sharpe)}`, title: "Sharpe (annualized, net)", text: fmt.num(r.sharpe) })
        : el("span", { class: `tag ${r.status === "failed" ? "bad" : "warn"}`, text: r.status })),
    r.label ? el("div", { class: "ri-alpha", text: r.alpha, title: r.alpha }) : null,
    el("div", { class: "ri-sub" },
      el("span", { class: "ri-chip", text: r.mode === "signed" ? (r.transform && r.transform !== "raw" ? `signed·${r.transform}` : "signed") : "long" }),
      el("span", { class: "ri-chip", text: r.universe }),
      el("span", { text: yearsLabel(r) }),
      finite(r.cagr) ? el("span", { class: tone(r.cagr), text: `CAGR ${fmt.pct(r.cagr)}` }) : null,
      finite(r.max_drawdown) ? el("span", { text: `DD ${fmt.pct(r.max_drawdown, 0)}` }) : null,
      r.short_window ? el("span", { class: "warn", text: "short", title: SHORT_WINDOW_NOTE(r.sessions) }) : null,
      r.factor_starred ? el("span", { class: "fstar", title: "Starred factor", text: "★ factor" }) : null,
      r.note ? el("span", { class: "has-note", title: r.note, text: "note" }) : null,
      !runReusable(r) ? el("span", { class: "warn", text: "preserved · read only", title: r.compatibility_note || "Reports and reuse of settings are unavailable." }) : null));
  return node;
}
function renderRunList() {
  const list = $("#runs-list");
  $("#run-sort").hidden = state.runFilter === "trash";
  renderRunChips();
  if (state.runFilter === "trash") return renderTrashList();
  if (state.runFilter === "factors") return renderFactorList();
  const rows = visibleRuns();
  $("#runs-count").textContent = state.runs.length ? `${rows.length}/${state.runs.length}` : "";
  $("#runs-foot").hidden = true;
  if (!rows.length) {
    setChildren(list, el("div", { class: "list-empty", text: state.runs.length
      ? state.runFilter === "starred" ? "No starred runs yet. Right-click a run (or press S) to star it." : "No runs match the filter."
      : "No runs yet — write an alpha and press ⌘R." }));
    return;
  }
  setChildren(list, ...rows.map(runItem));
}
function renderRunChips() {
  const chips = $("#run-chips"), parts = [];
  if (state.factorFilter && state.runFilter !== "factors" && state.runFilter !== "trash") {
    const r = state.runs.find((x) => x.factor === state.factorFilter);
    parts.push(el("span", { class: "chip" }, el("span", { text: `Factor: ${r ? (r.factor_label || r.alpha) : state.factorFilter}` }),
      el("button", { class: "chip-x", title: "Show all runs", "aria-label": "Clear factor filter", onclick: () => { state.factorFilter = null; renderRunList(); } }, icon("close"))));
  }
  if (state.selected.length > 1) {
    parts.push(el("span", { class: "chip accent" }, el("span", { text: `${state.selected.length} selected` }),
      el("button", { class: "chip-x", title: "Clear selection", "aria-label": "Clear selection", onclick: () => clearSelection() }, icon("close"))));
  }
  chips.hidden = !parts.length;
  setChildren(chips, ...parts);
}
function renderFactorList() {
  const list = $("#runs-list"), groups = factorGroups();
  $("#runs-count").textContent = groups.length ? `${groups.length} factors` : "";
  $("#runs-foot").hidden = true;
  if (!groups.length) { setChildren(list, el("div", { class: "list-empty", text: "No factors yet." })); return; }
  setChildren(list, ...groups.map((g) => el("div", { class: "run-item factor-item", "data-factor": g.factor,
    onclick: () => { state.factorFilter = g.factor; setRunFilter("all"); if (g.runs[0]) openRun(g.runs[0].path); },
    oncontextmenu: (e) => { e.preventDefault(); onFactorContext(e, g); } },
    el("div", { class: "ri-top" },
      starButton(g.starred, g.starred ? "Unstar factor" : "Star factor", () => annotateFactor(g, { starred: !g.starred })),
      el("span", { class: `ri-title${g.label ? "" : " mono"}`, text: g.label || g.alpha, title: g.alpha }),
      el("span", { class: `ri-metric ${tone(g.bestSharpe)}`, title: "Best Sharpe among its runs", text: fmt.num(g.bestSharpe) })),
    g.label ? el("div", { class: "ri-alpha", text: g.alpha }) : null,
    el("div", { class: "ri-sub" }, el("span", { class: "ri-chip", text: `${g.count} run${g.count === 1 ? "" : "s"}` }),
      el("span", { text: `last ${shortTime(g.latest).slice(0, 5)}` }),
      g.best ? el("span", { class: tone(g.best.cagr), text: `best CAGR ${fmt.pct(g.best.cagr)}` }) : null))));
}
function renderTrashList() {
  const list = $("#runs-list"), rows = state.trash.runs || [];
  $("#runs-count").textContent = rows.length ? `${rows.length} in trash` : "";
  const foot = $("#runs-foot");
  foot.hidden = !rows.length;
  setChildren(foot, el("span", { class: "muted", text: "Deleted runs stay here until you empty the trash." }),
    el("button", { class: "btn small danger-ghost", text: "Empty trash…", onclick: () => emptyTrash(rows.map((r) => r.name)) }));
  if (!rows.length) { setChildren(list, el("div", { class: "list-empty", text: "The trash is empty. Deleted runs appear here and can be restored." })); return; }
  setChildren(list, ...rows.map((r) => el("div", { class: "run-item trashed", oncontextmenu: (e) => {
    e.preventDefault();
    contextMenu(e.clientX, e.clientY, [{ label: "Restore", icon: "restore", action: () => restoreRuns([r.name]) },
      { label: "Copy alpha", icon: "copy", action: () => copyText(r.alpha, "Alpha copied") }, "sep",
      { label: "Move to system trash…", icon: "trash", danger: true, action: () => emptyTrash([r.name]) }]);
  } },
    el("div", { class: "ri-top" }, el("span", { class: `ri-title${r.label ? "" : " mono"}`, text: runTitle(r), title: r.alpha }),
      el("button", { class: "btn small", text: "Restore", onclick: () => restoreRuns([r.name]) })),
    el("div", { class: "ri-sub" }, el("span", { text: `deleted ${shortTime(r.trashed_at)}` }), el("span", { class: tone(r.sharpe), text: `Sharpe ${fmt.num(r.sharpe)}` }),
      el("span", { text: yearsLabel(r) })))));
}
function onRunClick(e, r) {
  const rows = visibleRuns().map((x) => x.path);
  if (e.metaKey || e.ctrlKey) {
    const base = state.selected.length ? state.selected : (state.run ? [state.run] : []);
    state.selected = base.includes(r.path) ? base.filter((p) => p !== r.path) : base.concat(r.path);
    state.anchor = r.path;
    afterSelection();
    return;
  }
  if (e.shiftKey && (state.anchor || state.run)) {
    const a = rows.indexOf(state.anchor || state.run), b = rows.indexOf(r.path);
    if (a >= 0 && b >= 0) { state.selected = rows.slice(Math.min(a, b), Math.max(a, b) + 1); afterSelection(); return; }
  }
  state.selected = []; state.anchor = r.path;
  openRun(r.path);
}
function clearSelection() { state.selected = []; afterSelection(); }
function afterSelection() {
  renderRunList();
  const n = state.selected.length;
  $("#compare-tab").hidden = n < 2;
  $("#compare-count").textContent = n >= 2 ? String(Math.min(n, MAX_COMPARE)) : "";
  if (n >= 2) setResultTab("compare");
  else if (state.resultTab === "compare") setResultTab("overview");
}
function contextTargets(r) {
  if (state.selected.length > 1 && state.selected.includes(r.path)) return state.runs.filter((x) => state.selected.includes(x.path));
  return [r];
}
function onRunContext(e, r) {
  const targets = contextTargets(r);
  if (targets.length === 1 && r.path !== state.run) { state.selected = []; openRun(r.path); }
  const many = targets.length > 1;
  const allStarred = targets.every((t) => t.starred);
  const items = many ? [
    { header: `${targets.length} runs selected` },
    { label: `Compare ${Math.min(targets.length, MAX_COMPARE)} runs`, icon: "compare", action: () => { state.selected = targets.map((t) => t.path); afterSelection(); } },
    { label: allStarred ? "Unstar all" : "Star all", icon: "star", checked: allStarred, action: () => starRuns(targets, !allStarred) },
    { label: "Copy alphas", icon: "copy", action: () => copyText(Array.from(new Set(targets.map((t) => t.alpha))).join("\n"), "Alphas copied") },
    "sep",
    { label: `Move ${targets.length} runs to Trash…`, icon: "trash", hint: "⌫", danger: true, action: () => trashRuns(targets) },
  ] : [
    { label: "Open", icon: "open", action: () => openRun(r.path) },
    { label: r.starred ? "Unstar run" : "Star run", icon: "star", checked: r.starred, hint: "S", action: () => starRuns([r], !r.starred) },
    { label: "Rename…", icon: "edit", hint: "F2", action: () => renameRun(r) },
    { label: r.note ? "Edit note…" : "Add note…", icon: "edit", action: () => editNote(r) },
    "sep",
    { label: r.factor_starred ? "Unstar this factor" : "Star this factor", icon: "star", checked: r.factor_starred, action: () => annotateFactor({ factor: r.factor, alpha: r.alpha }, { starred: !r.factor_starred }) },
    { label: "Show all runs of this factor", icon: "filter", action: () => { state.factorFilter = r.factor; setRunFilter("all"); } },
    { label: "Load alpha into the editor", icon: "load", action: () => loadRunAlpha(r) },
    { label: "Use all settings", icon: "settings", disabled: !runReusable(r), action: () => { if (!runReusable(r)) return; if (state.runDetail && state.runDetail.path === r.path) useRunSettings(); else openRun(r.path).then(useRunSettings); } },
    { label: "Copy alpha", icon: "copy", action: () => copyText(r.alpha, "Alpha copied") },
    { label: "Add to comparison", icon: "compare", disabled: !state.run || state.run === r.path, action: () => { state.selected = Array.from(new Set([state.run, ...state.selected, r.path])); afterSelection(); } },
    "sep",
    { label: revealLabel(), icon: "folder", action: () => api("reveal", { path: r.path }).catch((err) => toast("Cannot reveal", err.message, "bad")) },
    { label: "Move to Trash…", icon: "trash", hint: "⌫", danger: true, action: () => trashRuns([r]) },
  ];
  contextMenu(e.clientX, e.clientY, items);
}
function onFactorContext(e, g) {
  contextMenu(e.clientX, e.clientY, [
    { header: `${g.count} run${g.count === 1 ? "" : "s"}` },
    { label: "Show runs", icon: "filter", action: () => { state.factorFilter = g.factor; setRunFilter("all"); } },
    { label: g.starred ? "Unstar factor" : "Star factor", icon: "star", checked: g.starred, action: () => annotateFactor(g, { starred: !g.starred }) },
    { label: "Rename factor…", icon: "edit", action: () => renameFactor(g) },
    { label: "Load newest run's alpha and SEC inputs", icon: "load", action: () => loadRunAlpha(g.runs[0]) },
    { label: "Copy alpha", icon: "copy", action: () => copyText(g.alpha, "Alpha copied") },
    { label: `Compare its runs`, icon: "compare", disabled: g.count < 2, action: () => { state.selected = g.runs.slice(0, MAX_COMPARE).map((r) => r.path); state.factorFilter = g.factor; setRunFilter("all"); afterSelection(); } },
    "sep",
    { label: `Move ${g.count} run${g.count === 1 ? "" : "s"} to Trash…`, icon: "trash", danger: true, action: () => trashRuns(g.runs) },
  ]);
}
const revealLabel = () => IS_MAC ? "Reveal in Finder" : IS_WINDOWS ? "Show in Explorer" : "Show in folder";

// ================================================================= runs: actions
async function starRuns(rows, starred) {
  try {
    for (const r of rows) await api("annotate", { kind: "runs", key: r.name, starred });
    rows.forEach((r) => { r.starred = starred; });
    if (state.runDetail && rows.some((r) => r.path === state.runDetail.path)) { state.runDetail.starred = starred; renderRunHeaderActions(); }
    renderRunList();
  } catch (e) { toast("Cannot update the star", e.message, "bad"); }
}
async function renameRun(r) {
  const value = await dialog({ title: "Rename run", message: "A short label shown instead of the alpha. Leave empty to show the alpha.",
    input: { value: r.label || "", placeholder: r.alpha.slice(0, 60) }, confirm: "Save" });
  if (value === null) return;
  try {
    await api("annotate", { kind: "runs", key: r.name, label: value });
    r.label = value.trim();
    if (state.runDetail && state.runDetail.path === r.path) { state.runDetail.label = r.label; renderRunHeader(); }
    renderRunList();
  } catch (e) { toast("Cannot rename", e.message, "bad"); }
}
async function editNote(r) {
  const value = await dialog({ title: "Note", message: "What you tried, what you learned — shown under the run's title.",
    input: { value: r.note || "", placeholder: "e.g. works only before 2022; try SECTOR neutral", maxlength: 2000 }, confirm: "Save" });
  if (value === null) return;
  try {
    await api("annotate", { kind: "runs", key: r.name, note: value });
    r.note = value.trim();
    if (state.runDetail && state.runDetail.path === r.path) { state.runDetail.note = r.note; renderRunHeader(); }
    renderRunList();
  } catch (e) { toast("Cannot save the note", e.message, "bad"); }
}
async function annotateFactor(g, fields) {
  try {
    await api("annotate", { kind: "factors", key: g.factor, ...fields });
    for (const r of state.runs) if (r.factor === g.factor) {
      if ("starred" in fields) r.factor_starred = fields.starred;
      if ("label" in fields) r.factor_label = fields.label.trim();
    }
    renderRunList();
  } catch (e) { toast("Cannot update the factor", e.message, "bad"); }
}
async function renameFactor(g) {
  const value = await dialog({ title: "Rename factor", message: "A name for this alpha expression, shown in the Factors list.",
    input: { value: g.label || "", placeholder: g.alpha.slice(0, 60) }, confirm: "Save" });
  if (value !== null) annotateFactor(g, { label: value });
}
async function trashRuns(rows) {
  if (!rows.length) return;
  const n = rows.length;
  const ok = await dialog({ title: n === 1 ? "Move this run to the Trash?" : `Move ${n} runs to the Trash?`,
    message: n === 1 ? "It leaves the run list but stays in the app's Trash, where you can restore it."
      : "They leave the run list but stay in the app's Trash, where you can restore them.",
    detail: el("div", { class: "dlg-list" }, rows.slice(0, 5).map((r) => el("div", { class: "mono", text: runTitle(r) })), n > 5 ? el("div", { class: "muted", text: `… and ${n - 5} more` }) : null),
    confirm: "Move to Trash", danger: true });
  if (!ok) return;
  try {
    const result = await api("trash-runs", { paths: rows.map((r) => r.path) });
    const moved = new Set(result.trashed);
    const wasOpen = state.run && rows.some((r) => r.path === state.run);
    state.selected = state.selected.filter((p) => !rows.some((r) => r.path === p));
    await refreshRuns();
    if (wasOpen) { state.run = null; state.runDetail = null; const next = visibleRuns()[0]; if (next) openRun(next.path); else renderRun(); }
    afterSelection();
    toast(`Moved ${moved.size} run${moved.size === 1 ? "" : "s"} to the Trash`, "Undo here, or restore later from the Trash filter.", "", { label: "Undo", run: () => restoreRuns(result.trashed) });
  } catch (e) { toast("Cannot move to the Trash", e.message, "bad"); }
}
async function restoreRuns(names) {
  try {
    const result = await api("restore-runs", { names });
    await refreshTrash(); await refreshRuns();
    toast(`Restored ${result.restored.length} run${result.restored.length === 1 ? "" : "s"}`, "", "good");
  } catch (e) { toast("Cannot restore", e.message, "bad"); }
}
async function emptyTrash(names) {
  const ok = await dialog({ title: names.length === 1 ? "Move this run to the system Trash?" : `Move ${names.length} runs to the system Trash?`,
    message: "They leave the app completely. Your operating system's Trash still keeps them until you empty it there.", confirm: "Move", danger: true });
  if (!ok) return;
  try {
    const result = await api("empty-trash", { names });
    await refreshTrash(); renderRunList();
    toast(`Moved ${result.emptied.length} run${result.emptied.length === 1 ? "" : "s"} to the system Trash`);
  } catch (e) { toast("Cannot empty the trash", e.message, "bad"); }
}
function moveRunSelection(step) {
  const items = $$("#runs-list .run-item[data-path]");
  if (!items.length) return;
  const index = items.findIndex((i) => i.dataset.path === state.run);
  const next = items[Math.max(0, Math.min(items.length - 1, index + step))];
  state.selected = [];
  openRun(next.dataset.path);
  next.scrollIntoView({ block: "nearest" });
}
function listTargets() {
  const paths = state.selected.length ? state.selected : state.run ? [state.run] : [];
  return state.runs.filter((r) => paths.includes(r.path));
}

// ================================================================= run detail
async function openRun(path) {
  if (state.runLoad && state.runLoad.path === path) return state.runLoad.promise;
  state.run = path;
  state.runDetail = null;
  renderRun();
  renderRunList();
  const promise = (async () => {
    let d;
    try { d = await get("run", { path }); } catch (e) { toast("Cannot read the run", e.message, "bad"); return; }
    if (state.run !== path) return;
    state.runDetail = d;
    state.window = null;
    persist();
    renderRun();
  })();
  state.runLoad = { path, promise };
  try { await promise; } finally { if (state.runLoad && state.runLoad.promise === promise) state.runLoad = null; }
}
function runReturns(d) {
  const s = d.series || {};
  if (!s.dates) return { dates: [], returns: [], account: [] };
  return { dates: s.dates, returns: s.net_return || [], account: s.account };
}
function windowSlice(d) {
  const all = runReturns(d), n = all.dates.length;
  const [i0, i1] = state.window || [0, n - 1];
  const pick = (arr) => (arr || []).slice(i0, i1 + 1);
  const s = d.series || {};
  // A selection measures only its own returns, starting at the preceding close.
  // For the first session infer opening capital from its net return (also works
  // when the service has capped a very long ledger to its last daily records).
  const openingNAV = i0 > 0 ? all.account[i0 - 1]
    : all.account[0] / (1 + (finite(all.returns[0]) ? all.returns[0] : 0));
  const openingDate = i0 > 0 ? all.dates[i0 - 1] : "Opening NAV";
  return { i0, i1, n, openingNAV, openingDate, dates: pick(all.dates), returns: pick(all.returns), account: pick(all.account), turnover: pick(s.turnover), cost: pick(s.cost),
    long: s.long ? pick(s.long) : null, short: s.short ? pick(s.short) : null, holdings: s.holdings ? pick(s.holdings) : null };
}
const cashDaily = (d) => Math.expm1(Math.log1p(+d.annual_cash_rate || 0) / 252);
function renderRun() {
  const d = state.runDetail;
  const compatibility = $("#rh-compatibility");
  compatibility.hidden = !d || !(d.compatibility_note || !runReusable(d));
  compatibility.textContent = d ? (d.compatibility_note || (!runReusable(d) ? "Preserved result from an unsupported contract. Reports and reuse of settings are disabled." : "")) : "";
  $("#result-empty").hidden = !!d; $("#result-body").hidden = !d; $("#result-actions").hidden = !d;
  if (!d) { ["#returns-body", "#signal-body", "#details-body"].forEach((id) => $(id).replaceChildren()); return; }
  renderRunHeader();
  renderRunHeaderActions();
  renderReportMenu();
  renderHoldings();
  renderResultTab();
}
function renderRunHeader() {
  const d = state.runDetail;
  if (!d) return;
  $("#rh-status").className = `dot ${d.status}`;
  setChildren($("#rh-title"), el("span", { class: d.label ? "" : "mono", text: d.label || d.alpha || "", title: d.alpha || "" }),
    d.label ? el("span", { class: "rh-alpha mono", text: d.alpha }) : null);
  const cfg = d.settings || {};
  setChildren($("#rh-meta"), 
    el("span", { class: "tag accent", text: d.mode === "signed" ? `signed · ${TRANSFORM_LABELS[cfg.signal_transform || "raw"] || cfg.signal_transform}` : "long-only" }),
    el("span", { class: "tag", text: d.universe }), el("span", { class: "tag", text: `${d.period.start || "—"} → ${d.period.end || "—"}` }),
    el("span", { class: "tag", title: d.dataset, text: datasetTitle({ name: d.dataset || "", universes: {} }) || d.dataset }),
    d.short_window ? el("span", { class: "tag warn", text: `short window · ${fmt.int((d.metrics || {}).observations)} sessions`, title: SHORT_WINDOW_NOTE((d.metrics || {}).observations) }) : null);
  const note = $("#rh-note");
  note.hidden = !d.note;
  note.textContent = d.note || "";
}
function renderRunHeaderActions() {
  const d = state.runDetail, button = $("#star-run");
  if (!d) return;
  setChildren(button, icon("star", d.starred ? "filled" : ""));
  button.classList.toggle("on", !!d.starred);
  button.title = d.starred ? "Unstar this run (S)" : "Star this run (S)";
  $("#use-settings").disabled = !runReusable(d);
  $("#use-settings").title = runReusable(d) ? "Use this run's settings" : d.compatibility_note || "Preserved unsupported run: settings cannot be reused";
  $("#report-btn").disabled = !reportsSupported(d);
  $("#report-btn").title = reportsSupported(d) ? "Reports" : d.compatibility_note || "Reports are unavailable for this preserved run";
}
function renderResultTab() {
  if (state.view !== "research") return;
  const tab = state.resultTab;
  if (tab === "compare") return renderCompare();
  if (tab === "report") return;
  const d = state.runDetail;
  if (!d) return;
  if (tab === "overview") renderOverview();
  else if (tab === "returns") renderReturns();
  else if (tab === "signal") renderSignal();
  else if (tab === "details") renderDetails();
}
function renderWindowBar(d, w) {
  const bar = $("#window-bar");
  if (!state.window) { bar.hidden = true; return; }
  bar.hidden = false;
  setChildren(bar, icon("window"),
    el("span", {}, "Analysing ", el("b", { text: `${w.dates[0]} → ${w.dates[w.dates.length - 1]}` }), ` · ${fmt.int(w.dates.length)} of ${fmt.int(w.n)} sessions`),
    el("div", { class: "tabs-spacer" }),
    el("button", { class: "btn small", text: "Backtest this window", title: "Copy these dates into the backtest window", onclick: () => useWindowForBacktest(w) }),
    el("button", { class: "btn small ghost", text: "Show whole run", onclick: () => { state.window = null; renderResultTab(); } }));
}
async function useWindowForBacktest(w) {
  if (!await useRunSettings()) return;
  $("#start").value = w.dates[0]; $("#end").value = w.dates[w.dates.length - 1];
  renderPeriod(); persist();
  toast("Window set for the next backtest", `${w.dates[0]} → ${w.dates[w.dates.length - 1]} · press ⌘R to run`);
}
function metricGrid(rows) {
  return el("div", { class: "metric-grid" }, rows.map(([label, value, cls, title]) => el("div", { class: "mg-item", title: title || "" },
    el("span", { class: "mg-label", text: label }), el("span", { class: `mg-value ${cls || ""}`, text: value }))));
}
function renderOverview() {
  const d = state.runDetail;
  const completed = d.status === "completed" && d.series && d.series.dates;
  $(".charts").hidden = !completed;
  if (!completed) {
    $("#window-bar").hidden = true; setChildren($("#more-metrics"));
    $("#kpis").className = "kpis";
    setChildren($("#kpis"), kpiTile("Status", d.status, (d.error && d.error.message) || "", "neg"));
    return;
  }
  const w = windowSlice(d), st = statsOf(w.returns, cashDaily(d));
  renderWindowBar(d, w);
  const short = st.observations < 252;
  const turnover = w.turnover.length ? w.turnover.reduce((a, b) => a + (b || 0), 0) / w.turnover.length : null;
  $("#kpis").className = "kpis";
  setChildren($("#kpis"), 
    kpiTile("Sharpe", fmt.num(st.sharpe), short ? `only ${fmt.int(st.observations)} sessions` : "annualized, net of costs", tone(st.sharpe)),
    kpiTile("CAGR", fmt.signedPct(st.cagr), short ? "extrapolated" : "compound annual", tone(st.cagr)),
    kpiTile("Max drawdown", fmt.pct(st.maxDD), "peak to trough", "neg"),
    kpiTile("Volatility", fmt.pct(st.vol), "annualized"),
    kpiTile("Total return", fmt.signedPct(st.total), `${fmt.years(st.observations)}`, tone(st.total)),
    kpiTile("Turnover", fmt.pct(turnover), "traded per session"));
  const signed = d.mode === "signed";
  const avg = (arr) => { const v = (arr || []).filter(finite); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
  const gross = signed && w.long && w.short ? avg(w.long.map((l, i) => finite(l) && finite(w.short[i]) && w.account[i] ? (l + w.short[i]) / w.account[i] : null)) : null;
  const net = signed && w.long && w.short ? avg(w.long.map((l, i) => finite(l) && finite(w.short[i]) && w.account[i] ? (l - w.short[i]) / w.account[i] : null)) : null;
  const invested = !signed && w.holdings ? avg(w.holdings.map((h, i) => finite(h) && w.account[i] ? h / w.account[i] : null)) : null;
  const costs = w.cost.reduce((a, b) => a + (b || 0), 0);
  const meanEquity = avg(w.account);
  const more = $("#more-metrics");
  setChildren(more, 
    el("button", { class: `more-toggle${state.moreOpen ? " open" : ""}`, "aria-expanded": String(state.moreOpen),
      onclick: () => { state.moreOpen = !state.moreOpen; persist(); renderOverview(); } }, icon("chevron"), state.moreOpen ? "Fewer metrics" : "More metrics"),
    state.moreOpen ? metricGrid([
      ["Sortino", fmt.num(st.sortino), tone(st.sortino), "mean over downside deviation, annualized"],
      ["Calmar", fmt.num(st.calmar), tone(st.calmar), "CAGR over max drawdown"],
      ["Hit rate", fmt.pct(st.hitRate), "", "share of sessions with a gain"],
      ["Best day", fmt.signedPct(st.best, 2), "pos"], ["Worst day", fmt.signedPct(st.worst, 2), "neg"],
      ["Avg gain day", fmt.signedPct(st.avgUp, 2), "pos"], ["Avg loss day", fmt.signedPct(st.avgDown, 2), "neg"],
      ["Skew", fmt.num(st.skew), ""], ["Excess kurtosis", fmt.num(st.kurtosis, 1), ""],
      ["Longest drawdown", `${fmt.int(st.longestDD)} sessions`, "", "sessions below the previous peak"],
      ["Sessions", fmt.int(st.observations), ""], ["Total costs", fmt.money(costs), "", "trading and borrow costs in the window"],
      ["Cost drag", meanEquity ? fmt.pct((costs / meanEquity) * (252 / Math.max(1, w.dates.length))) : "—", "", "costs per year over mean equity"],
      ...(signed ? [["Avg gross exposure", fmt.pct(gross, 0), ""], ["Avg net exposure", fmt.signedPct(net, 1), ""]] : [["Avg invested", fmt.pct(invested, 0), ""]]),
    ]) : null);
  const group = crosshairGroup();
  const startValue = w.openingNAV;
  const nav = w.account;
  const dd = drawdownOf(nav, startValue);
  drawWhenSized($("#equity-chart"), () => {
    group.add(lineChart($("#equity-chart"), {
      dates: w.dates, values: nav, baseline: startValue, format: fmt.money, log: state.scale === "log", label: "Net equity", group,
      tooltip: (i) => [["Equity", fmt.money(nav[i])], ["Since start", fmt.signedPct(nav[i] / startValue - 1), tone(nav[i] - startValue)],
        ["Day", fmt.signedPct(w.returns[i], 2), tone(w.returns[i])], ["Drawdown", fmt.pct(dd[i]), dd[i] < 0 ? "neg" : ""]],
      onBrush: (a, b) => { state.window = [w.i0 + a, w.i0 + b]; renderOverview(); },
      onReset: () => { state.window = null; renderOverview(); },
    }));
    group.add(lineChart($("#dd-chart"), { dates: w.dates, values: dd, kind: "dd", format: (v) => fmt.pct(v, 0), axisX: false, label: "Drawdown", group,
      tooltip: (i) => [["Drawdown", fmt.pct(dd[i]), dd[i] < 0 ? "neg" : ""]] }));
    const all = runReturns(d);
    rangeNavigator($("#analysis-nav"), { dates: all.dates, values: all.account, range: [w.i0, w.i1], height: 42, label: "Analysis window",
      onChange: (a, b, final) => { if (final) { state.window = a === 0 && b === all.dates.length - 1 ? null : [a, b]; renderOverview(); } } });
  });
  const worst = Math.min(...dd), worstAt = dd.indexOf(worst);
  $("#equity-note").textContent = `${fmt.money(startValue)} → ${fmt.money(nav[nav.length - 1])} · ${fmt.signedPct(nav[nav.length - 1] / startValue - 1)} · worst ${fmt.pct(worst)} on ${w.dates[worstAt]}`;
}

// ------------------------------------------------------------------ Returns tab
function card(title, note, ...body) {
  return el("div", { class: "chart-card" }, el("div", { class: "chart-head" }, el("span", { class: "chart-title", text: title }), note ? el("span", { class: "chart-note", text: note }) : null), ...body);
}
function renderReturns() {
  const d = state.runDetail, box = $("#returns-body");
  if (!(d.status === "completed" && d.series && d.series.dates)) { setChildren(box, el("div", { class: "empty-state" }, el("div", { class: "empty-title", text: "No returns for this run" }))); return; }
  const w = windowSlice(d);
  const table = monthlyTable(w.dates, w.returns);
  const years = annualStats(w.dates, w.returns, cashDaily(d));
  const nav = w.account, episodes = drawdownEpisodes(w.dates, nav, 5, w.openingNAV, w.openingDate);
  const rolling = rollingSharpe(w.returns, 126);
  const heat = el("div", { class: "heatmap-wrap" });
  const annual = el("div", { class: "chart", style: { height: "190px" } });
  const roll = el("div", { class: "chart", style: { height: "170px" } });
  setChildren(box, el("div", { class: "result-body" },
    state.window ? el("div", { class: "window-bar" }, icon("window"), el("span", {}, "Returns for ", el("b", { text: `${w.dates[0]} → ${w.dates[w.dates.length - 1]}` })),
      el("div", { class: "tabs-spacer" }), el("button", { class: "btn small ghost", text: "Show whole run", onclick: () => { state.window = null; renderReturns(); } })) : null,
    el("div", { class: "grid-2" },
      card("Monthly returns", "compounded, % · blue up, red down", heat),
      el("div", { class: "stack" },
        card("Calendar-year returns", "", annual),
        card("Rolling Sharpe", "126 sessions (about 6 months)", roll))),
    el("div", { class: "grid-2" },
      el("div", { class: "fact-card" }, el("h3", { text: "By year" }),
        el("table", { class: "table" }, el("thead", {}, el("tr", {}, ...["Year", "Return", "Sharpe", "Volatility", "Max DD", "Sessions"].map((h, i) => el("th", { class: i ? "num" : "", text: h })))),
          el("tbody", {}, ...years.slice().reverse().map((y) => el("tr", {}, el("td", { text: y.year }), el("td", { class: `num ${tone(y.total)}`, text: fmt.signedPct(y.total) }),
            el("td", { class: `num ${tone(y.sharpe)}`, text: fmt.num(y.sharpe) }), el("td", { class: "num", text: fmt.pct(y.vol) }),
            el("td", { class: "num", text: fmt.pct(y.maxDD) }), el("td", { class: "num", text: fmt.int(y.observations) })))))),
      el("div", { class: "fact-card" }, el("h3", { text: "Deepest drawdowns" }),
        el("table", { class: "table" }, el("thead", {}, el("tr", {}, ...["Peak", "Trough", "Recovered", "Depth", "Length"].map((h, i) => el("th", { class: i > 2 ? "num" : "", text: h })))),
          el("tbody", {}, ...episodes.map((e) => el("tr", {}, el("td", { class: "mono", text: e.start }), el("td", { class: "mono", text: e.trough }),
            el("td", { class: "mono", text: e.recovered ? e.end : "not yet" }), el("td", { class: "num neg", text: fmt.pct(e.depth) }),
            el("td", { class: "num", text: `${fmt.int(e.length)} sess` })))))))));
  monthlyHeatmap(heat, table);
  drawWhenSized(annual, () => barChart(annual, { labels: years.map((y) => y.year), values: years.map((y) => y.total), format: (v) => fmt.pct(v, 0), valueLabels: years.length <= 10,
    color: (v) => v >= 0 ? css("--div-pos-4") : css("--div-neg-4"), tooltip: (i) => `${years[i].year}: ${fmt.signedPct(years[i].total)}`, label: "Calendar-year returns" }));
  drawWhenSized(roll, () => lineChart(roll, { dates: w.dates, values: rolling, baseline: 0, format: (v) => fmt.num(v, 1), label: "Rolling Sharpe", wash: false, empty: "Needs at least 126 sessions",
    tooltip: (i) => [["Rolling Sharpe", fmt.num(rolling[i]), tone(rolling[i])]] }));
}

// ------------------------------------------------------------------ Signal tab
async function renderSignal() {
  const d = state.runDetail, box = $("#signal-body");
  if (!d.signal_available) {
    setChildren(box, el("div", { class: "empty-state" }, el("div", { class: "empty-title", text: "No signal diagnostics for this run" }),
      el("div", { class: "empty-sub", text: "Runs made since September 2026 save each score with the next open-to-open return; older or failed runs do not." })));
    return;
  }
  const cached = state.signal[d.path];
  if (!cached) {
    setChildren(box, el("div", { class: "empty-state" }, el("div", { class: "spinner" }), el("div", { class: "empty-title", text: "Measuring the signal…" }),
      el("div", { class: "empty-sub", text: "Correlating every score with the next session's return (a few seconds for TOP3000)." })));
    state.signal[d.path] = "loading";
    try { state.signal[d.path] = await get("run-signal", { path: d.path }); }
    catch (e) { state.signal[d.path] = { error: e.message }; }
    if (state.runDetail === d && state.resultTab === "signal") renderSignal();
    return;
  }
  if (cached === "loading") return;
  if (cached.error) {
    // a failed request is not remembered for good: the loopback call can fail while the machine sleeps
    setChildren(box, el("div", { class: "empty-state" }, el("div", { class: "empty-title", text: "Cannot measure the signal" }), el("div", { class: "empty-sub", text: cached.error }),
      el("button", { class: "btn small", text: "Try again", onclick: () => { delete state.signal[d.path]; renderSignal(); } })));
    return;
  }
  const sig = windowSignal(cached, d);
  const s = sig.summary;
  const icChart = el("div", { class: "chart", style: { height: "190px" } });
  const qChart = el("div", { class: "chart", style: { height: "230px" } });
  const qBars = el("div", { class: "chart", style: { height: "190px" } });
  const spread = el("div", { class: "chart", style: { height: "170px" } });
  const monthly = monthlyMeans(sig.dates, sig.rank_ic);
  const trend = trailingMean(monthly.means, 12);
  const ramp = [1, 2, 3, 4, 5].map((k) => css(`--seq-${k}`));
  setChildren(box, el("div", { class: "result-body" },
    state.window ? el("div", { class: "window-bar" }, icon("window"), el("span", {}, "Signal for ", el("b", { text: `${sig.dates[0]} → ${sig.dates[sig.dates.length - 1]}` })),
      el("div", { class: "tabs-spacer" }), el("button", { class: "btn small ghost", text: "Show whole run", onclick: () => { state.window = null; renderSignal(); } })) : null,
    el("div", { class: "kpis" },
      kpiTile("Rank IC", fmt.num(s.rank_ic.mean, 4), "mean daily Spearman", tone(s.rank_ic.mean)),
      kpiTile("Rank ICIR", fmt.num(s.rank_ic.annualized_ir), "annualized mean / std", tone(s.rank_ic.annualized_ir)),
      kpiTile("IC > 0", fmt.pct(s.rank_ic.positive_share, 0), "share of sessions"),
      kpiTile("t-stat", fmt.num(s.rank_ic.t_stat, 1), `${fmt.int(s.rank_ic.days)} sessions`, Math.abs(s.rank_ic.t_stat || 0) >= 2 ? "pos" : ""),
      kpiTile(`Q${s.groups}−Q1 spread`, fmt.signedPct(s.spread_annualized), "per year, gross", tone(s.spread_annualized)),
      kpiTile("Monotonicity", fmt.num(s.monotonicity), "quantile order, −1 to 1", tone(s.monotonicity))),
    el("div", { class: "grid-2" },
      card("Rank IC by month", "bars: monthly mean · line: 12-month mean", icChart),
      card("Mean next-day return by quantile", `Q1 = lowest scores · ${fmt.int(s.median_names)} names a day (median)`, qBars)),
    card("Growth of $1 by score quantile", "equal weight, rebalanced every session, before costs", qChart),
    card(`Long–short: Q${s.groups} minus Q1`, "cumulative daily spread, before costs", spread),
    el("p", { class: "prop-note", text: `Label: ${cached.label}. IC is measured on names with both a score and a label; quantiles hold near-equal counts each session. Costs and borrow are not included here — compare with the backtest's net results.` })));
  drawWhenSized(icChart, () => barChart(icChart, { labels: monthly.months, values: monthly.means, dense: true, format: (v) => fmt.num(v, 3), line: trend,
    color: (v) => (v >= 0 ? css("--div-pos-4") : css("--div-neg-4")), tooltip: (i) => `${monthly.months[i]}: ${fmt.num(monthly.means[i], 4)}`, label: "Monthly rank IC" }));
  drawWhenSized(qBars, () => barChart(qBars, { labels: sig.qmean.map((_, i) => `Q${i + 1}`), values: sig.qmean.map((v) => (finite(v) ? v * 1e4 : null)), valueLabels: true,
    format: (v) => `${fmt.num(v, 1)}`, color: (v, i) => ramp[i] || ramp[4], tooltip: (i) => `Q${i + 1}: ${fmt.num(sig.qmean[i] * 1e4, 2)} bp a day`, label: "Quantile mean returns (bp)" }));
  drawWhenSized(qChart, () => multiLineChart(qChart, { format: (v) => `${fmt.num(v, 2)}×`, log: true, label: "Quantile growth",
    series: sig.paths.map((values, i) => ({ name: `Q${i + 1}${i === 0 ? " (lowest)" : i === sig.paths.length - 1 ? " (highest)" : ""}`, short: `Q${i + 1}`, dates: sig.dates, values, color: ramp[i] })) }));
  drawWhenSized(spread, () => lineChart(spread, { dates: sig.dates, values: sig.spreadPath, baseline: 0, format: (v) => fmt.pct(v, 0), label: "Long-short spread",
    tooltip: (i) => [["Cumulative spread", fmt.signedPct(sig.spreadPath[i]), tone(sig.spreadPath[i])]] }));
}
/* The signal restricted to the analysis window (the whole run when none is set). */
function windowSignal(sig, d) {
  const full = !state.window;
  const s = d.series || {};
  let a = 0, b = sig.dates.length - 1;
  if (!full) {
    const from = s.dates[state.window[0]], to = s.dates[state.window[1]];
    a = sessionIndex(sig.dates, from, "start"); b = sessionIndex(sig.dates, to, "end");
  }
  const pick = (arr) => (arr || []).slice(a, b + 1);
  const dates = pick(sig.dates), rank = pick(sig.rank_ic), ic = pick(sig.ic);
  const groups = sig.summary.groups, names = Object.keys(sig.daily || {});
  const daily = names.map((k) => pick(sig.daily[k]));
  const paths = daily.map((r) => navOf(r, 1));
  const spreadDaily = daily.length ? daily[daily.length - 1].map((v, i) => (finite(v) && finite(daily[0][i]) ? v - daily[0][i] : 0)) : [];
  let cum = 0;
  const spreadPath = spreadDaily.map((v) => (cum += v));
  const qmean = daily.map((r) => { const v = r.filter(finite); return v.length ? v.reduce((x, y) => x + y, 0) / v.length : null; });
  const describe = (vals) => {
    const v = vals.filter(finite), n = v.length;
    if (n < 2) return { mean: null, annualized_ir: null, t_stat: null, positive_share: null, days: n };
    const mean = v.reduce((x, y) => x + y, 0) / n, std = Math.sqrt(v.reduce((x, y) => x + (y - mean) ** 2, 0) / (n - 1));
    return { mean, annualized_ir: std ? (mean / std) * Math.sqrt(252) : null, t_stat: std ? (mean / std) * Math.sqrt(n) : null, positive_share: v.filter((x) => x > 0).length / n, days: n };
  };
  const spreadMean = spreadDaily.length ? spreadDaily.reduce((x, y) => x + y, 0) / spreadDaily.length : null;
  let monotonicity = null;
  if (qmean.every(finite)) {
    const order = qmean.map((v, i) => [v, i]).sort((x, y) => x[0] - y[0]).map((x) => x[1]);
    const ranks = new Array(qmean.length); order.forEach((q, r) => { ranks[q] = r; });
    const m = (qmean.length - 1) / 2;
    let sxy = 0, sxx = 0;
    ranks.forEach((r, i) => { sxy += (i - m) * (r - m); sxx += (i - m) ** 2; });
    monotonicity = sxx ? sxy / sxx : null;
  }
  return { dates, rank_ic: rank, ic, paths, spreadPath, qmean,
    summary: { rank_ic: describe(rank), ic: describe(ic), groups, median_names: sig.summary.median_names,
      spread_annualized: finite(spreadMean) ? spreadMean * 252 : null, monotonicity } };
}

// ------------------------------------------------------------------ Details tab
function renderDetails() {
  const d = state.runDetail, box = $("#details-body");
  const m = d.metrics || {}, s = d.series || {}, cfg = d.settings || {};
  const facts = [
    ["Configuration", [["Mode", d.mode], ["Universe", d.universe], ["Dataset", d.dataset], ["Warm-up", `${fmt.int(d.warmup_sessions)} sessions`],
      ["Initial capital", fmt.money(d.cash)], ["Run folder", d.name]]],
    ["Trading", [["Mean turnover", `${fmt.pct(s.mean_turnover)} per session`], ["Total costs", fmt.money(s.total_cost)],
      ["Cost per side", `${fmt.num((cfg.open_cost || 0) * 1e4, 1)} bps`], ["Sessions", fmt.int(s.sessions)]]],
    ["Portfolio", d.mode === "signed" ? [["Weighting", TRANSFORM_LABELS[cfg.signal_transform || "raw"] || String(cfg.signal_transform)], ["Gross exposure", fmt.num(cfg.gross_exposure, 2)], ["Borrow rate", fmt.pct(cfg.annual_borrow_rate, 2)],
      ["Names held (mean)", d.holdings ? fmt.num(d.holdings.mean_names, 1) : "—"], ["Instruments in window", fmt.int(d.trade_symbols)]]
      : [["Top-k", fmt.int(cfg.topk)], ["Replace per day", fmt.int(cfg.n_drop)], ["Invested share", fmt.pct(cfg.risk_degree, 0)],
        ["Names held (mean)", d.holdings ? fmt.num(d.holdings.mean_names, 1) : "—"], ["Instruments in window", fmt.int(d.trade_symbols)]]],
    ["Saved summary", [["Sharpe", fmt.num(m.sharpe)], ["CAGR", fmt.signedPct(m.cagr)], ["Max drawdown", fmt.pct(m.max_drawdown)],
      ["Volatility", fmt.pct(m.annualized_volatility)], ["Observations", fmt.int(m.observations)]]],
  ];
  const provenance = secProvenance(d);
  if (d.universe_scoring) {
    const u = d.universe_scoring;
    facts.push(["Universe coverage", [["Member observations", fmt.int(u.member_observations)], ["Unscored", fmt.int(u.unscored_member_observations)],
      ["Scored per day", `${fmt.int(u.min_scored_members)}–${fmt.int(u.max_scored_members)} (median ${fmt.int(u.median_scored_members)})`]]]);
  }
  if (d.funding_state) {
    const f = d.funding_state;
    facts.push(["Funding (signed)", [["Feasible every session", f.feasible_all_sessions ? "yes" : "no"], ["Unfunded sessions", fmt.int(f.unfunded_sessions)],
      ["Max cash shortfall", fmt.money(f.max_unfunded_cash_shortfall)], ["Stale marks", fmt.int(f.stale_mark_sessions)]]]);
  }
  const exposure = el("div", { class: "chart", style: { height: "170px" } });
  const turnover = el("div", { class: "chart", style: { height: "150px" } });
  const w = windowSlice(d);
  setChildren(box, el("div", { class: "result-body" }, provenance,
    s.dates ? el("div", { class: "grid-2" }, card(d.mode === "signed" ? "Exposure" : "Invested share", d.mode === "signed" ? "gross and net, as a share of equity" : "holdings over equity", exposure),
      card("Turnover", "per session · line: 21-session mean", turnover)) : null,
    el("div", { class: "facts" }, facts.map(([title, rows]) => el("div", { class: "fact-card" }, el("h3", { text: title }),
      el("dl", { class: "kv" }, rows.flatMap(([k, v]) => [el("dt", { text: k }), el("dd", { class: "selectable", text: v ?? "—" })])))))));
  if (!s.dates) return;
  drawWhenSized(exposure, () => {
    if (d.mode === "signed" && w.long && w.short) {
      multiLineChart(exposure, { format: (v) => fmt.pct(v, 0), baseline: 0, label: "Exposure", endLabels: false, series: [
        { name: "Gross", dates: w.dates, values: w.long.map((l, i) => finite(l) && finite(w.short[i]) && w.account[i] ? (l + w.short[i]) / w.account[i] : null), color: css("--cat-1") },
        { name: "Net", dates: w.dates, values: w.long.map((l, i) => finite(l) && finite(w.short[i]) && w.account[i] ? (l - w.short[i]) / w.account[i] : null), color: css("--cat-2") }] });
    } else if (w.holdings) {
      lineChart(exposure, { dates: w.dates, values: w.holdings.map((h, i) => (finite(h) && w.account[i] ? h / w.account[i] : null)), format: (v) => fmt.pct(v, 0), label: "Invested share",
        tooltip: (i) => [["Invested", fmt.pct(w.holdings[i] / w.account[i])]] });
    } else setChildren(exposure, el("div", { class: "chart-empty", text: "No exposure recorded" }));
  });
  drawWhenSized(turnover, () => barChart(turnover, { labels: w.dates, values: w.turnover, dense: true, format: (v) => fmt.pct(v, 0), line: trailingMean(w.turnover, 21),
    color: () => css("--series-dim"), tooltip: (i) => `${w.dates[i]}: ${fmt.pct(w.turnover[i])}`, label: "Turnover" }));
}

// ------------------------------------------------------------------ Compare tab
async function renderCompare() {
  const box = $("#compare-body");
  const paths = state.selected.slice(0, MAX_COMPARE);
  if (paths.length < 2) { setChildren(box, el("div", { class: "empty-state" }, el("div", { class: "empty-title", text: "Select two or more runs" }),
    el("div", { class: "empty-sub", text: "⌘-click or Shift-click runs in the list, or right-click → Add to comparison." }))); return; }
  const missing = paths.filter((p) => !state.compareCache.has(p));
  if (missing.length) {
    setChildren(box, el("div", { class: "empty-state" }, el("div", { class: "spinner" }), el("div", { class: "empty-title", text: "Loading runs…" })));
    await Promise.all(missing.map((p) => get("run", { path: p }).then((d) => state.compareCache.set(p, d)).catch(() => state.compareCache.set(p, null))));
    if (state.resultTab === "compare") renderCompare();
    return;
  }
  const runs = paths.map((p) => state.compareCache.get(p)).filter((d) => d && d.series && d.series.dates);
  const unreadable = paths.length - runs.length;
  if (runs.length < 2) {
    setChildren(box, el("div", { class: "empty-state" }, el("div", { class: "empty-title", text: "Not enough runs to compare" }),
      el("div", { class: "empty-sub", text: `${unreadable} of the ${paths.length} selected runs could not be read: they may be incomplete, moved to the Trash, or outside this data folder.` })));
    return;
  }
  const rows = runs.map((d, k) => {
    const st = statsOf(d.series.net_return || [], cashDaily(d));
    const setup = `${d.mode === "signed" ? `signed${(d.settings || {}).signal_transform && d.settings.signal_transform !== "raw" ? "·" + d.settings.signal_transform : ""}` : "long"} · ${d.universe} · ${(d.period.start || "").slice(0, 4)}–${(d.period.end || "").slice(2, 4)}`;
    return { d, k, st, setup, color: css(`--cat-${k + 1}`), name: d.label || d.alpha, turnover: d.series.mean_turnover };
  });
  const cols = [["Sharpe", (r) => r.st.sharpe, (v) => fmt.num(v), "max"], ["CAGR", (r) => r.st.cagr, (v) => fmt.signedPct(v), "max"],
    ["Max DD", (r) => r.st.maxDD, (v) => fmt.pct(v), "max"], ["Volatility", (r) => r.st.vol, (v) => fmt.pct(v), "min"],
    ["Sortino", (r) => r.st.sortino, (v) => fmt.num(v), "max"], ["Calmar", (r) => r.st.calmar, (v) => fmt.num(v), "max"],
    ["Turnover", (r) => r.turnover, (v) => fmt.pct(v), "min"], ["Sessions", (r) => r.st.observations, (v) => fmt.int(v), null]];
  const best = cols.map(([, get, , rule]) => {
    if (!rule) return null;
    const vals = rows.map(get).filter(finite);
    return vals.length ? (rule === "max" ? Math.max(...vals) : Math.min(...vals)) : null;
  });
  const overlay = el("div", { class: "chart", style: { height: "300px" } });
  const corr = rows.map((a) => rows.map((b) => a === b ? 1 : correlationOf({ dates: a.d.series.dates, values: a.d.series.net_return }, { dates: b.d.series.dates, values: b.d.series.net_return })));
  setChildren(box, el("div", { class: "result-body" },
    el("div", { class: "cmp-head" }, el("div", { class: "run-title" }, el("span", { text: `Comparing ${rows.length} runs` })),
      state.selected.length > MAX_COMPARE ? el("span", { class: "tag warn", text: `first ${MAX_COMPARE} of ${state.selected.length}` }) : null,
      unreadable ? el("span", { class: "tag warn", text: `${unreadable} could not be read` }) : null,
      el("button", { class: "btn small", text: "Copy table", title: "Tab-separated, for a spreadsheet", onclick: () => copyText([["Run", "Setup", ...cols.map(([h]) => h)].join("\t"),
        ...rows.map((r) => [r.name, r.setup, ...cols.map(([, get]) => { const v = get(r); return finite(v) ? +v.toFixed(6) : ""; })].join("\t"))].join("\n"), "Comparison copied — paste into a spreadsheet") }),
      el("button", { class: "btn small ghost", text: "Clear comparison", onclick: clearSelection })),
    card("Growth of $100", "each run from its own first session · hover for values", overlay),
    el("div", { class: "fact-card" }, el("h3", { text: "Metrics (best in each column in bold)" }),
      el("table", { class: "table cmp-table" }, el("thead", {}, el("tr", {}, el("th", { text: "Run" }), el("th", { text: "Setup" }), ...cols.map(([h]) => el("th", { class: "num", text: h })))),
        el("tbody", {}, ...rows.map((r) => el("tr", { class: "clickable", onclick: () => { state.selected = []; afterSelection(); openRun(r.d.path); } },
          el("td", { class: "cmp-name" }, el("i", { class: "swatch", style: { background: r.color } }), el("b", { class: "cmp-no", text: `${r.k + 1}` }), el("span", { class: r.d.label ? "" : "mono", text: r.name, title: r.d.alpha })),
          el("td", { class: "muted nowrap", text: r.setup }),
          ...cols.map(([, get, format], c) => { const v = get(r); return el("td", { class: `num ${finite(v) && v === best[c] ? "best" : ""}`, text: format(v) }); })))))),
    rows.length > 1 ? el("div", { class: "fact-card" }, el("h3", { text: "Correlation of daily returns" }),
      el("table", { class: "table corr-table" }, el("thead", {}, el("tr", {}, el("th", { text: "" }), ...rows.map((r) => el("th", { class: "num" }, el("i", { class: "swatch", style: { background: r.color } }), el("b", { class: "cmp-no", text: `${r.k + 1}` }))))),
        el("tbody", {}, ...rows.map((r, i) => el("tr", {}, el("td", {}, el("i", { class: "swatch", style: { background: r.color } }), el("b", { class: "cmp-no", text: `${r.k + 1}` }), el("span", { class: "muted", text: r.setup })),
          ...corr[i].map((v, j) => el("td", { class: `num ${i !== j && finite(v) && Math.abs(v) >= 0.7 ? "warn-text" : ""}`, text: finite(v) ? fmt.num(v) : "—" })))))),
      el("p", { class: "prop-note", text: "Pearson correlation on the sessions both runs share (at least 20). Above 0.7 the runs are largely the same bet." })) : null));
  drawWhenSized(overlay, () => multiLineChart(overlay, { format: (v) => fmt.num(v, 0), baseline: 100, log: state.scale === "log", label: "Growth of $100",
    series: rows.map((r) => ({ name: `${r.k + 1}. ${r.name.length > 42 ? r.name.slice(0, 41) + "…" : r.name} · ${r.setup}`, short: `#${r.k + 1}`, title: r.d.alpha, dates: r.d.series.dates,
      values: r.d.series.account.map((v) => (v / r.d.series.account[0]) * 100 * (1 + (r.d.series.net_return[0] || 0))), color: r.color })) }));
}

// ------------------------------------------------------------------ holdings, reports, settings
function renderHoldings() {
  const d = state.runDetail, box = $("#holdings");
  const h = d && d.holdings;
  if (!h || !h.top || !h.top.length) { setChildren(box, el("div", { class: "muted pad", text: "No positions recorded for this run." })); return; }
  const max = Math.max(1e-9, ...h.top.map((r) => Math.abs(r.weight)));
  setChildren(box, el("table", { class: "hold-table" },
    el("thead", {}, el("tr", {}, el("th", { text: `Largest positions · ${h.date}` }), el("th", { class: "num", text: "Weight" }), el("th", { text: `${fmt.int(h.count)} names held` }))),
    el("tbody", {}, h.top.map((r) => el("tr", {}, el("td", { text: r.instrument }), el("td", { class: `num ${tone(r.weight)}`, text: fmt.signedPct(r.weight, 2) }),
      el("td", { class: "wbar-cell" }, el("div", { class: "wtrack" }, el("div", { class: "mid" }),
        el("div", { class: `wbar${r.weight < 0 ? " short" : ""}`, style: r.weight >= 0 ? `left:50%;width:${(r.weight / max) * 50}%` : `right:50%;width:${(-r.weight / max) * 50}%` }))))))));
}
function runReusable(d) { return !!d && d.read_only !== true && d.reports_supported !== false; }
function reportsSupported(d) { return runReusable(d); }
function renderReportMenu() {
  const d = state.runDetail, menu = $("#report-menu");
  const reports = (d && d.reports) || [];
  const find = (level) => reports.find((r) => r.level === level && (level !== "summary" || r.holdings));
  const item = (label, profile, level) => {
    const existing = find(level);
    return el("button", { disabled: !reportsSupported(d) || d.status !== "completed", onclick: () => { if (!reportsSupported(d)) return; menu.hidden = true; existing ? openReport(existing.path, label) : buildReport(profile); } },
      existing ? `Open ${label}` : `Generate ${label}`, el("small", { text: existing ? `built ${existing.created.slice(0, 13).replace("T", " ")}` : "renders in the background" }));
  };
  setChildren(menu, item("overview", "overview", "overview"), item("detailed", "detailed", "detailed"), item("holdings", "holdings", "summary"),
    el("div", { class: "sep" }), el("button", { disabled: !reportsSupported(d) || !reports.length, onclick: () => { if (!reportsSupported(d)) return; menu.hidden = true; if (reports[0]) api("open", { path: reports[0].path }).catch((e) => toast("Cannot open", e.message, "bad")); } },
      "Open latest in browser", el("small", { text: "system default browser" })));
}
async function buildReport(profile) {
  if (!state.run || !reportsSupported(state.runDetail)) return;
  try { startPolling(await api("report", { run: state.run, profile })); } catch (e) { toast("Cannot build the report", e.message, "bad"); }
}
function openReport(path, label = "report") {
  if (!reportsSupported(state.runDetail)) return;
  path = String(path).replace(/\\/g, "/");
  const root = state.overview.data_root.replace(/\\/g, "/").replace(/\/$/, "") + "/runs/";
  if (!path.startsWith(root)) { toast("Report outside the data folder", path, "bad"); return; }
  $("#report-frame").src = "/files/" + path.slice(root.length).split("/").map(encodeURIComponent).join("/");
  const tab = $("#report-tab");
  tab.hidden = false;
  tab.firstChild.textContent = `Report · ${label}`;
  setResultTab("report");
}
async function useRunSettings() {
  const d = state.runDetail;
  if (!runReusable(d)) { toast("Settings cannot be reused", (d && d.compatibility_note) || "This preserved run uses an unsupported contract.", "bad"); return; }
  if (!await restoreSavedSec(d)) return false;
  if (d.dataset_path) {
    if (!priceRow(d.dataset_path)) { toast("Saved trading prices unavailable", "Select the original price dataset before reusing this run.", "bad"); return false; }
    await loadDataset(d.dataset_path, true);
    if (state.datasetPath !== d.dataset_path) return false;
  }
  const cfg = d.settings || {};
  setAlpha(d.alpha || "");
  setMode(d.mode);
  if (d.mode === "signed") {
    if (finite(cfg.gross_exposure)) { $("#gross").value = cfg.gross_exposure; $("#gross-range").value = cfg.gross_exposure; }
    if (finite(cfg.annual_borrow_rate)) $("#borrow").value = +(cfg.annual_borrow_rate * 100).toFixed(4);
    setTransform(cfg.signal_transform || "raw");
  } else {
    if (finite(cfg.topk)) $("#topk").value = cfg.topk;
    if (finite(cfg.n_drop)) $("#ndrop").value = cfg.n_drop;
  }
  if (finite(cfg.open_cost)) $("#cost").value = +(cfg.open_cost * 1e4).toFixed(3);
  $("#start").value = d.period.start || ""; $("#end").value = d.period.end || "";
  if (d.warmup_sessions) $("#warmup").value = d.warmup_sessions;
  const universe = d.universe === "all" ? "" : d.universe;
  if ([...$("#universe").options].some((o) => o.value === universe)) { $("#universe").value = universe; renderUniversePanel(); }
  renderPeriod();
  persist();
  toast("Settings loaded from the run", "Edit and press ⌘R to run again.");
  return true;
}
function runMoreMenu(event) {
  const d = state.runDetail;
  if (!d) return;
  const r = state.runs.find((x) => x.path === d.path) || { ...d, factor_starred: d.factor_starred };
  const box = event.currentTarget.getBoundingClientRect();
  contextMenu(box.right - 220, box.bottom + 4, [
    { label: "Use all settings", icon: "settings", disabled: !runReusable(d), action: () => useRunSettings() },  // the toolbar button hides in a narrow window
    { label: "Rename…", icon: "edit", action: () => renameRun(r) },
    { label: r.note ? "Edit note…" : "Add note…", icon: "edit", action: () => editNote(r) },
    { label: "Copy alpha", icon: "copy", action: () => copyText(d.alpha, "Alpha copied") },
    { label: "Copy metrics (tab-separated)", icon: "copy", action: () => copyMetrics() },
    { label: r.factor_starred ? "Unstar this factor" : "Star this factor", icon: "star", checked: r.factor_starred, action: () => annotateFactor({ factor: d.factor, alpha: d.alpha }, { starred: !r.factor_starred }) },
    { label: revealLabel(), icon: "folder", action: () => api("reveal", { path: d.path }).catch((e) => toast("Cannot reveal", e.message, "bad")) },
    "sep",
    { label: "Move to Trash…", icon: "trash", danger: true, action: () => trashRuns([r]) },
  ]);
}

function copyMetrics() {
  const d = state.runDetail;
  if (!d || !d.series || !d.series.dates) return;
  const w = windowSlice(d), st = statsOf(w.returns, cashDaily(d));
  const rows = [["Run", d.label || d.alpha], ["Alpha", d.alpha], ["Mode", d.mode], ["Universe", d.universe],
    ["Window", `${w.dates[0]} to ${w.dates[w.dates.length - 1]}`], ["Sessions", st.observations], ["Sharpe", st.sharpe], ["CAGR", st.cagr],
    ["Max drawdown", st.maxDD], ["Volatility", st.vol], ["Total return", st.total], ["Sortino", st.sortino], ["Calmar", st.calmar], ["Hit rate", st.hitRate]];
  copyText(rows.map(([k, v]) => `${k}\t${finite(v) ? +v.toFixed(6) : v}`).join("\n"), "Metrics copied — paste into a spreadsheet");
}
function showShortcuts() {
  const keys = [["⌘R", "Run the backtest, or simulate in Alpha Lab"], ["⌘.", "Stop the running task"], ["⌘K", "Command palette: commands, runs, datasets"],
    ["⌘1 / ⌘2 / ⌘3", "Qlib Validation, Alpha Lab, Data"], ["⌘D", "Choose trading prices for the current view"], ["⌘I", "Show or hide the settings panel"], ["⌘J", "Show or hide the console panel"],
    ["↑ ↓", "Move through runs"], ["S", "Star or unstar the selected runs"], ["F2", "Rename the run"], ["⌫", "Move the selected runs to the Trash"],
    ["⌘-click, Shift-click, ⌘A", "Select several runs to compare"], ["Esc", "Clear the selection, close menus"],
    ["Drag on the equity chart", "Analyse part of a run; double-click to reset"], ["Right-click", "Run, factor and alpha menus"]];
  dialog({ title: "Keyboard and mouse", confirm: "Close", cancel: null,
    detail: el("table", { class: "table keys" }, el("tbody", {}, keys.map(([k, v]) => el("tr", {}, el("td", {}, el("kbd", { text: k })), el("td", { text: v }))))) });
}

// ================================================================= jobs
async function runBacktest() {
  if (state.job && state.job.state === "running") { toast("A task is already running", state.job.label); return; }
  if (!state.datasetPath || !priceEligible(state.dataset)) { toast("Choose backtest prices", "Select eligible ordinary OHLCV prices in the Qlib Validation toolbar.", "bad"); return; }
  if (!await checkAlpha()) { toast("Native Qlib expression required", "The saved text was kept. Fix the expression shown in Qlib Validation; it is not translated automatically.", "bad"); return; }
  persist();
  try { startPolling(await api("backtest", formValues())); } catch (e) { toast("Cannot start the backtest", e.message, "bad"); }
}
async function startImport() {
  persist();
  try {
    startPolling(await api("import", { table: $("#table").value, start: $("#imp-start").value, end: $("#imp-end").value, size: $("#imp-size").value,
      adv: $("#imp-adv").value, history: $("#imp-history").value, workers: $("#imp-workers").value, make_default: $("#imp-default").checked,
      min_price: $("#imp-minprice").value, groups: $("#imp-groups").value }));
    toast("Import started", "Progress is shown in the status bar and console.");
  } catch (e) { toast("Cannot start the import", e.message, "bad"); }
}
async function startRefresh() {
  if (state.job && state.job.state === "running") { toast("A task is already running", state.job.label); return; }
  try {
    startPolling(await api("refresh-data", { min_price: $("#refresh-minprice").value, make_default: $("#imp-default").checked }));
    toast("Market data refresh started", "About an hour; progress is shown in the status bar and console.");
  } catch (e) { toast("Cannot start the refresh", e.message, "bad"); }
}
async function showSources() {
  try {
    const s = await get("sources");
    $("#sources-note").textContent = s.table ? `Latest table: ${home(s.table)}` : "No Yahoo table yet: the first refresh fetches everything.";
    if (s.screener && !$("#imp-groups").value) $("#imp-groups").value = s.screener;
    if (s.table && !$("#table").value) $("#table").value = s.table;
  } catch (e) { /* informational */ }
}
async function cancelJob() {
  if (researchUI.overview && researchUI.overview.worker.running) return stopResearch();
  if (!state.job || state.job.state !== "running") return;
  try { const r = await api("cancel", {}); if (!r.cancelled) toast("This step cannot be interrupted", "It will stop when the step finishes."); }
  catch (e) { toast("Cancel failed", e.message, "bad"); }
}
function startPolling(job) {
  state.job = job; state.jobSince = 0;
  consoleLine(`▶ ${job.label}`, "l-stage");
  if (state.bottomTab !== "console") $("#console-badge").hidden = false;
  if ($("#center").classList.contains("no-bottom") && job.kind !== "report") toggleBottom(true);
  renderJob(job);
  clearTimeout(state.poll);
  const tick = async () => {
    try {
      const update = await get("job", { id: state.job.id, since: state.jobSince });
      state.job = update; state.jobSince = update.log_index;
      for (const line of update.log) if (!line.startsWith("▶")) consoleLine(line, /^Error|failed/i.test(line) ? "l-err" : /^Completed/.test(line) ? "l-ok" : /^(Verifying|Scoring|Simulating|Rendering|Scanning|Classifying|Loading|Ranking|Writing|Converting|Reading every|Hashing|Snapshot|Ready)/.test(line) ? "l-stage" : "");
      renderJob(update);
      if (update.state === "running") state.poll = setTimeout(tick, 350);
      else finished(update);
    } catch (e) { state.poll = setTimeout(tick, 1500); }
  };
  state.poll = setTimeout(tick, 150);
}
function consoleLine(text, cls = "") {
  const box = $("#console"), pane = box.parentElement;
  const atBottom = pane.scrollHeight - pane.scrollTop - pane.clientHeight < 40;
  const time = new Date().toTimeString().slice(0, 8);
  box.append(el("span", { class: "l-time", text: `${time}  ` }), el("span", { class: cls, text: `${text}\n` }));
  while (box.childNodes.length > 3000) box.firstChild.remove();
  if (atBottom) pane.scrollTop = pane.scrollHeight;
}
function renderJob(job) {
  const running = job && job.state === "running";
  const researching = researchUI.overview && researchUI.overview.worker.running;
  $("#run").disabled = running; $("#run-2").disabled = running; $("#import").disabled = running; $("#lab-run").disabled = running;
  updateSecRunButtons();
  $("#refresh-data").disabled = running;
  if (state.view === "evidence") workflowUpdateButtons();
  if (researching) $("#run").disabled = true;
  $("#cancel").disabled = !(running && job.cancellable);
  $(".statusbar").classList.toggle("busy", running);
  const dot = $("#sb-state .sb-dot");
  dot.className = `sb-dot ${running ? "busy" : job && job.state === "failed" ? "bad" : ""}`;
  $("#sb-progress").hidden = !running;
  $("#sb-cancel").hidden = !(running && job.cancellable);
  if (!job) { $("#sb-text").textContent = "Ready"; return; }
  const determinate = running && finite(job.progress);
  const bar = $("#sb-bar");
  bar.parentElement.classList.toggle("indeterminate", running && !determinate);
  bar.style.width = determinate ? `${Math.round(job.progress * 100)}%` : "";
  const stage = determinate ? `${job.progress_label} ${Math.round(job.progress * 100)}%` : job.stage;
  $("#sb-text").textContent = running ? `${job.label} — ${stage}` : job.state === "completed" ? `Completed · ${job.label}` : job.state === "cancelled" ? `Cancelled · ${job.label}` : `Failed · ${job.error || job.label}`;
  $("#sb-time").textContent = fmt.clock(job.elapsed);
}
async function finished(job) {
  renderJob(job);
  if (typeof workflowChronologyJobFinished === "function") workflowChronologyJobFinished(job);
  if (job.state === "cancelled") { toast("Task cancelled", job.label); return; }
  if (job.state === "failed") { toast(`${{ backtest: "Backtest", "backtest-batch": "Backtests", import: "Import", alpha: "Alpha Lab", refresh: "Refresh" }[job.kind] || "Report"} failed`, job.error || "", "bad"); setBottomTab("console"); return; }
  if (["chronology-trial", "chronology-final"].includes(job.kind)) {
    toast("Chronological evaluation finished", job.receipt?.access_label || job.label, "good");
    setView("evidence");
    await refreshWorkflow();
    await refreshWorkflowChronology();
    return;
  }
  if (job.kind === "build-dictionary" || job.kind === "sec-dictionary-build" || job.kind === "sec-dictionary") {
    await searchSecDictionary(true); await Promise.all([checkAlpha(), checkLab()]); return;
  }
  if (job.kind === "alpha") {
    toast("Alpha Lab finished", job.label, "good");
    await refreshLab();
    try { state.lab.batch = await get("alpha-batch", { path: job.result }); } catch (e) { toast("Cannot read the batch", e.message, "bad"); return; }
    if (state.lab.batch.rows.length === 1) selectLab(state.lab.batch.rows[0].id); else renderLabBatch();
    return;
  }
  if (job.kind === "backtest-batch") {
    let batch = null;
    try { batch = await get("backtest-batch", { path: job.result }); } catch (e) { toast("Cannot read the batch", e.message, "bad"); }
    if (state.runFilter === "trash") setRunFilter("all");
    await refreshRuns();
    const runs = (batch && batch.runs) || [];
    const failed = (batch && batch.errors) || [];
    toast(`${runs.length} backtest${runs.length === 1 ? "" : "s"} completed${failed.length ? `, ${failed.length} failed` : ""}`,
      failed.length ? `First failure: ${failed[0].alpha} — ${failed[0].error}` : "Opened side by side in Compare.", failed.length ? "bad" : "good");
    if (runs.length) await openRun(runs[runs.length - 1]);
    state.selected = runs.length > 1 ? runs.slice(0, MAX_COMPARE) : [];
    afterSelection();
    return;
  }
  if (["backtest", "candidate-validation", "portfolio-validation"].includes(job.kind)) {
    toast("Backtest completed", home(job.result), "good");
    if (state.runFilter === "trash") setRunFilter("all");
    await refreshRuns();
    state.selected = [];
    afterSelection();
    await openRun(job.result);
    if (["candidate-validation", "portfolio-validation"].includes(job.kind)) setView("research");
    if (state.resultTab === "compare" || state.resultTab === "report") setResultTab("overview");
    document.querySelector(`.run-item[data-path="${CSS.escape(job.result)}"]`)?.scrollIntoView({ block: "nearest" });
  } else if (job.kind === "import" || job.kind === "refresh") {
    toast("Dataset ready", home(job.result), "good");
    state.detailDataset = job.result;
    await refreshDatasets();
    await loadDataset(null, true);
  } else if (job.kind === "report") {
    if (state.run) { const d = await get("run", { path: state.run }).catch(() => null); if (d) { state.runDetail.reports = d.reports; renderReportMenu(); } }
    openReport(job.result, "report");
  }
}

// ================================================================= Alpha Lab
const LAB_DEFAULTS = { universe: "TOP3000", delay: "1", neutralization: "MARKET", decay: "0", truncation: "0.08", pasteurize: true,
  cost: "0", start: "", end: "", oos: "", workers: "4" };
function groupDescription(source) {
  const taxonomy = source?.taxonomy_metadata || {};
  const display = value => typeof value === "object" ? JSON.stringify(value) : String(value);
  const labels = ["industry", "sector", "country"].filter(group => taxonomy[group] != null)
    .map(group => {
      const value = display(taxonomy[group]);
      const concise = group === "industry" && /SEC SIC/i.test(value) ? "SEC SIC4"
        : group === "sector" && /OSHA SIC division/i.test(value) ? "coarse OSHA SIC division"
        : group === "country" && /countryba.*business.address/i.test(value) ? "SEC business address (not exchange/currency)"
        : value.length > 60 ? value.slice(0, 57) + "…" : value;
      return `${group}: ${concise}`;
    });
  return { caption: labels.length ? labels.join(" · ") : "Dated classification labels; taxonomy not recorded.",
    details: ["Unknown groups are excluded. Dated observations are not backfilled.",
      ...Object.entries(taxonomy).map(([key, value]) => `${key.replaceAll("_", " ")}: ${display(value)}`),
      source?.availability_policy ? "Availability: " + display(source.availability_policy) : "Availability policy not recorded.",
      source?.historical_identity_limit ? "Historical identity: " + display(source.historical_identity_limit) : "Historical identity limitations not recorded.",
      source?.max_age_days != null ? `Groups expire after ${source.max_age_days} days.` : "No group expiry recorded.",
      "Metadata checked only; workers verify events when groups are used."] };
}
function groupIdentityMatches(expected, actual) {
  return !expected || ["manifest_sha256", "events_sha256"].every(key => !expected[key] || expected[key] === actual?.[key]);
}
function savedLabGroup(result) {
  const identity = result.classification?.source_identity;
  return { path: typeof identity?.path === "string" ? identity.path : "", identity: identity || null,
    error: identity && typeof identity.path !== "string" ? "Saved classification provenance has no source path. Restore the original source before using groups." : "" };
}
function restoreLabGroup(result) {
  const saved = savedLabGroup(result);
  ++state.lab.groupEpoch;
  state.lab.groupOverlay = saved.path; state.lab.groupIdentity = saved.identity;
  state.lab.groupRestoreError = saved.error; state.lab.groupExplicit = true;
  state.lab.groupMetadata = null;
}
async function loadLabGroups() {
  const epoch = ++state.lab.groupEpoch, dataset = state.lab.datasetPath, path = state.lab.groupOverlay;
  state.lab.groupMetadata = null; state.lab.groupLoading = !!dataset; renderLabGroups();
  if (!dataset) return;
  try {
    let metadata = await get("alpha-groups", { dataset, group_overlay: path });
    if (epoch !== state.lab.groupEpoch || dataset !== state.lab.datasetPath || path !== state.lab.groupOverlay) return;
    if (!state.lab.groupExplicit && !path && metadata.overlays?.length === 1) {
      const sole = metadata.overlays[0];
      state.lab.groupOverlay = sole.path; state.lab.groupExplicit = true;
      metadata = { ...metadata, selected: sole, groups: ["market", ...sole.groups] };
    }
    if (metadata.selected && !groupIdentityMatches(state.lab.groupIdentity, metadata.selected)) {
      metadata = { ...metadata, groups: ["market"], error: "Saved classification identity differs from this source. Restore the original overlay or explicitly choose a new source." };
    }
    state.lab.groupMetadata = metadata;
  } catch (error) {
    if (epoch !== state.lab.groupEpoch || dataset !== state.lab.datasetPath || path !== state.lab.groupOverlay) return;
    state.lab.groupMetadata = { overlays: [], groups: ["market"], embedded_groups: [], diagnostics: [], error: error.message };
  }
  state.lab.groupLoading = false; renderLabGroups(); persist(); onLabInput();
}
function renderLabGroups() {
  const select = $("#lab-group-source"), neutral = $("#lab-neutralization");
  if (!select || !neutral) return;
  const m = state.lab.groupMetadata, path = state.lab.groupOverlay;
  setChildren(select, el("option", { value: "", text: "Embedded groups / no overlay" }),
    ...(m?.overlays || []).map(source => el("option", { value: source.path, text: source.path.split(/[\\/]/).pop() })));
  if (path && !(m?.overlays || []).some(source => source.path === path)) setChildren(select, ...select.options, el("option", { value: path, text: "Selected source unavailable — " + path.split(/[\\/]/).pop() }));
  select.value = path; $("#lab-group-path").value = path;
  const error = state.lab.groupRestoreError || m?.error || "";
  const available = error || state.lab.groupLoading ? ["market"] : m?.groups || ["market"];
  const chosen = neutral.value;
  for (const option of neutral.options) {
    // Freeze the submitted value before changing the label: HTML options
    // without a value attribute otherwise derive their value from text.
    option.value = option.value;
    const supported = ["MARKET", "NONE"].includes(option.value) || available.includes(option.value.toLowerCase());
    option.disabled = !supported;
    option.title = supported ? "" : error || `${option.value} needs groups from a compatible source.`;
    option.textContent = option.value + (supported ? "" : " — unavailable");
  }
  // Preserve a selected disabled value: changing the source must not change the strategy.
  neutral.value = chosen;
  const waiting = !state.lab.groupExplicit && (m?.overlays?.length || 0) > 1;
  const description = groupDescription(m?.selected);
  $("#lab-group-caption").textContent = state.lab.groupLoading ? "Checking classification source metadata…" :
    [error, path ? description.caption
      : `Embedded groups: ${(m?.embedded_groups || []).filter(g => g !== "market").join(", ") || "none"}.`,
    waiting ? "Multiple compatible overlays found. Choose a source explicitly, or keep embedded groups / no overlay." : ""].filter(Boolean).join(" ");
  $("#lab-group-details").hidden = !path && !m?.diagnostics?.length;
  $("#lab-group-detail-text").textContent = [...(path ? description.details : []),
    ...((m?.diagnostics || []).map(item => item.path.split(/[\\/]/).pop() + ": " + item.message))].join("\n\n");
  $("#lab-group-caption").classList.toggle("bad", !!error);
  const unsupported = chosen && !["MARKET", "NONE"].includes(chosen) && !available.includes(chosen.toLowerCase());
  $("#lab-neutralization-note").textContent = unsupported ? (chosen === "SUBINDUSTRY" ? "Subindustry is unsupported; no subindustry taxonomy is supplied."
    : `${chosen} needs groups. ${error || "Choose a compatible classification source for the selected Trading prices."}`) : "";
}
function chooseLabGroup(path) {
  state.lab.groupOverlay = path.trim(); state.lab.groupExplicit = true;
  state.lab.groupIdentity = null; state.lab.groupRestoreError = "";
  loadLabGroups(); persist();
}
function setLabDelay(value) {
  state.lab.delay = String(value);
  $$("#lab-delay button").forEach((b) => b.classList.toggle("active", b.dataset.delay === state.lab.delay));
}
function restoreLab(saved) {
  saved = preferenceObject(saved);
  const v = { ...LAB_DEFAULTS };
  for (const key of Object.keys(LAB_DEFAULTS)) if (typeof saved[key] === typeof LAB_DEFAULTS[key]) v[key] = saved[key];
  if (!["0", "1"].includes(v.delay)) v.delay = LAB_DEFAULTS.delay;
  if (Object.hasOwn(saved, "group_overlay")) {
    state.lab.groupOverlay = typeof saved.group_overlay === "string" ? saved.group_overlay : "";
    state.lab.groupExplicit = true; state.lab.groupIdentity = saved.group_identity || null;
    state.lab.groupRestoreError = saved.group_restore_error || "";
  }
  const neutral = $("#lab-neutralization");
  if (!["MARKET", "NONE", "SECTOR", "INDUSTRY", "COUNTRY"].includes(v.neutralization) && ![...neutral.options].some(o => o.value === v.neutralization)) setChildren(neutral, ...neutral.options, el("option", { value: v.neutralization, text: v.neutralization + " — unsupported", disabled: true }));
  $("#lab-universe").value = v.universe; neutral.value = v.neutralization; $("#lab-decay").value = v.decay;
  $("#lab-truncation").value = v.truncation; $("#lab-pasteurize").checked = v.pasteurize !== false; $("#lab-cost").value = v.cost;
  $("#lab-start").value = v.start; $("#lab-end").value = v.end; $("#lab-oos").value = v.oos; $("#lab-workers").value = v.workers;
  setLabDelay(v.delay);
  if (["all", "starred"].includes(saved.filter)) state.lab.filter = saved.filter;
  if (typeof saved.text === "string") $("#lab-text").value = saved.text;
  renderLabGroups();
}
function labValues() {
  return { expressions: $("#lab-text").value, universe: $("#lab-universe").value, delay: state.lab.delay,
    neutralization: $("#lab-neutralization").value, decay: $("#lab-decay").value, truncation: $("#lab-truncation").value,
    pasteurization: $("#lab-pasteurize").checked, cost_bps: $("#lab-cost").value, start: $("#lab-start").value, end: $("#lab-end").value,
    oos_start: $("#lab-oos").value, workers: $("#lab-workers").value, force: $("#lab-force").checked, dataset: state.lab.datasetPath || "",
    group_overlay: state.lab.groupOverlay, ...secRequest() };
}
async function runLab() {
  if (state.job && state.job.state === "running") { toast("A task is already running", state.job.label); return; }
  if (!state.lab.datasetPath || !priceEligible(state.lab.dataset)) { toast("Choose Trading prices", "Select eligible ordinary OHLCV prices in the Alpha Lab toolbar.", "bad"); return; }
  if (state.lab.groupLoading) await loadLabGroups();
  const neutralization = $("#lab-neutralization").value;
  if (!["MARKET", "NONE"].includes(neutralization) && (state.lab.groupRestoreError || state.lab.groupMetadata?.error)) {
    toast("Classification source unavailable", state.lab.groupRestoreError || state.lab.groupMetadata.error, "bad"); return;
  }
  if (!await checkLab()) { toast("Alpha Lab syntax check failed", "Fix the expression shown in Alpha Lab. Saved text is not translated automatically.", "bad"); return; }
  persist();
  try { startPolling(await api("alpha-run", labValues())); } catch (e) { toast("Cannot start Alpha Lab", e.message, "bad"); }
}
function runCurrent() { if (state.view === "evidence") runWorkflowValidation(); else if (state.view === "auto") startResearchCampaign(); else if (state.view === "lab") runLab(); else runBacktest(); }
let labTimer = null;
let labCheckEpoch = 0;
function onLabInput() {
  ++labCheckEpoch; state.labValid = false; delete state.sec.resolved.lab;
  updateSecRunButtons(); renderSecBindings(); clearTimeout(labTimer); labTimer = setTimeout(checkLab, 300);
}
async function checkLab() {
  const text = $("#lab-text").value, status = $("#lab-status");
  const dataset = state.lab.datasetPath;
  const epoch = ++labCheckEpoch, path = state.lab.groupOverlay, groupEpoch = state.lab.groupEpoch;
  const generation = state.sec.generation;
  state.labValid = false; updateSecRunButtons();
  if (!text.trim()) { status.textContent = ""; status.className = "syntax"; return false; }
  try {
    const inputs = JSON.stringify(secRequest());
    // Syntax and field availability only; this endpoint must never load a price panel.
    const r = await api("alpha-check", { text, ...(dataset ? { dataset } : {}), group_overlay: path, ...secRequest() });
    if (epoch !== labCheckEpoch || generation !== state.sec.generation || groupEpoch !== state.lab.groupEpoch || path !== state.lab.groupOverlay || text !== $("#lab-text").value || dataset !== state.lab.datasetPath || inputs !== JSON.stringify(secRequest())) return false;
    const bad = r.lines.filter((l) => !l.ok);
    const sourceError = state.lab.groupRestoreError || state.lab.groupMetadata?.error;
    if (sourceError && r.group_consumers?.length) {
      status.className = "syntax bad"; status.textContent = sourceError; return false;
    }
    status.className = `syntax ${bad.length ? "bad" : "ok"}`;
    status.textContent = bad.length ? `Line ${bad[0].line}: ${secEditorError(bad[0].message)}` : `✓ ${r.lines.length} alpha${r.lines.length === 1 ? "" : "s"}`;
    status.title = bad.map((l) => `Line ${l.line}: ${secEditorError(l.message)}`).join("\n");
    state.labValid = bad.length === 0 && r.lines.length > 0;
    acceptSecInputs(r, "lab"); updateSecRunButtons();
    return bad.length === 0 && r.lines.length > 0;
  } catch (e) {
    if (epoch !== labCheckEpoch || generation !== state.sec.generation || groupEpoch !== state.lab.groupEpoch || path !== state.lab.groupOverlay || text !== $("#lab-text").value || dataset !== state.lab.datasetPath) return false;
    delete state.sec.resolved.lab; renderSecBindings(); updateSecRunButtons();
    status.className = "syntax bad"; status.textContent = secEditorError(e.message); return false;
  }
}
function renderLabPeriod() {
  const box = $("#lab-period-nav"), d = state.lab.dataset;
  if (!box || state.view !== "lab") return;
  const calendar = (d && d.calendar) || [];
  if (calendar.length < 10) { setChildren(box, el("div", { class: "chart-empty", text: "Choose a dataset to set the window." })); $("#lab-period-summary").textContent = ""; return; }
  const bounds = windowBounds(calendar, 252);
  const universe = Object.keys(d.universes || {})[0], u = universe && d.universes[universe];
  const fullStart = u && u.full_size_from ? Math.min(bounds.max - 1, Math.max(bounds.min, sessionIndex(calendar, u.full_size_from, "start") + 1)) : bounds.min;
  const start = $("#lab-start").value, end = $("#lab-end").value;
  const i0 = start ? Math.max(bounds.min, sessionIndex(calendar, start, "start")) : fullStart;
  const i1 = end ? Math.min(bounds.max + 1, sessionIndex(calendar, end, "end")) : calendar.length - 1;
  const maxIndex = calendar.length - 1;
  const summary = (a, b, dflt) => setChildren($("#lab-period-summary"), periodSummary(calendar, a, b, dflt));
  summary(i0, i1, !start && !end);
  drawWhenSized(box, () => rangeNavigator(box, {
    dates: calendar, area: coverageSeries(d, universe), minIndex: bounds.min, maxIndex, range: [i0, Math.min(i1, maxIndex)], label: "Simulation window",
    markers: fullStart > bounds.min ? [{ i: fullStart, label: `full ${universe}` }] : [],
    onChange: (a, b, final) => {
      summary(a, b, false);
      if (final) {
        $("#lab-start").value = a === fullStart ? "" : calendar[a];
        $("#lab-end").value = b === maxIndex ? "" : calendar[b];
        summary(a, b, a === fullStart && b === maxIndex);
        persist();
      }
    },
  }));
}
function applyLabPreset(kind) {
  const d = state.lab.dataset, calendar = (d && d.calendar) || [];
  if (calendar.length < 10) return;
  const bounds = windowBounds(calendar, 252);
  if (kind === "universe") { $("#lab-start").value = ""; $("#lab-end").value = ""; }
  else if (kind === "all") { $("#lab-start").value = calendar[bounds.min]; $("#lab-end").value = ""; }
  else {
    const range = presetRange(kind, calendar, { min: bounds.min, max: calendar.length - 1 }, null);
    if (range) { $("#lab-start").value = range[0] || calendar[bounds.min]; $("#lab-end").value = ""; }
  }
  renderLabPeriod(); persist();
}
async function refreshLab() {
  try {
    state.lab.rows = await get("alpha-library", { sort: state.lab.sort });
    if (state.lab.filter === "trash") state.trash = await get("trash");
  } catch (e) { toast("Cannot read the alpha library", e.message, "bad"); return; }
  renderLabList();
}
function setLabFilter(filter) {
  state.lab.filter = filter;
  $$("#lab-filter button").forEach((b) => b.classList.toggle("active", b.dataset.filter === filter));
  if (filter === "trash") get("trash").then((t) => { state.trash = t; renderLabList(); }).catch((e) => toast("Cannot read the trash", e.message, "bad"));
  else renderLabList();
  persist();
}
function labItem(r, trashed = false) {
  const chosen = state.lab.chosen.includes(r.id);
  return el("div", { class: `run-item${r.id === state.lab.selected || chosen ? " selected" : ""}${trashed ? " trashed" : ""}`, "data-id": r.id, title: labEngineNote(r),
    onclick: (e) => {
      if (trashed) return;
      if (e.metaKey || e.ctrlKey) { state.lab.chosen = chosen ? state.lab.chosen.filter((x) => x !== r.id) : state.lab.chosen.concat(r.id); renderLabList(); return; }
      state.lab.chosen = []; selectLab(r.id);
    },
    oncontextmenu: (e) => { e.preventDefault(); onLabContext(e, r, trashed); } },
    el("div", { class: "ri-top" },
      trashed ? null : starButton(r.starred, r.starred ? "Unstar alpha" : "Star alpha", () => starAlphas([r], !r.starred)),
      el("span", { class: `ri-title${r.label ? "" : " mono"}`, text: r.label || r.expression, title: r.expression }),
      trashed ? el("button", { class: "btn small", text: "Restore", onclick: (e) => { e.stopPropagation(); deleteAlphas([r], false); } })
        : el("span", { class: `ri-metric ${tone(r.fitness)}`, title: "Fitness", text: fmt.num(r.fitness) })),
    r.label ? el("div", { class: "ri-alpha", text: r.expression }) : null,
    el("div", { class: "ri-sub" }, el("span", { class: `ri-chip${r.engine === "alphalab-3" ? "" : " warn"}`, text: r.engine || "engine unknown", title: labEngineNote(r) }),
      el("span", { class: "ri-chip", text: (r.settings || {}).universe || "" }), el("span", { class: "ri-chip", text: `d${(r.settings || {}).delay ?? ""}` }),
      r.status === "ok"
        ? [el("span", { class: tone(r.sharpe), text: `Sharpe ${fmt.num(r.sharpe)}` }), el("span", { text: `TO ${fmt.pct(r.turnover, 0)}` }),
          finite(r.self_correlation) ? el("span", { class: r.self_correlation >= 0.7 ? "warn" : "", text: `corr ${fmt.num(r.self_correlation)}` }) : null]
        : el("span", { class: "warn", text: r.error || "error" })));
}
function renderLabList() {
  const filter = $("#lab-search").value.trim().toLowerCase();
  const list = $("#lab-list");
  $("#lab-sort").hidden = state.lab.filter === "trash";
  if (state.lab.filter === "trash") {
    const rows = (state.trash.alphas || []).filter((r) => !filter || (r.expression || "").toLowerCase().includes(filter));
    $("#lab-count").textContent = rows.length ? `${rows.length} hidden` : "";
    setChildren(list, ...(rows.length ? rows.map((r) => labItem(r, true)) : [el("div", { class: "list-empty", text: "No hidden alphas. Deleted alphas appear here and can be restored." })]));
    return;
  }
  let rows = state.lab.rows.filter((r) => !filter || `${r.expression} ${r.label || ""}`.toLowerCase().includes(filter));
  if (state.lab.filter === "starred") rows = rows.filter((r) => r.starred);
  $("#lab-count").textContent = state.lab.rows.length ? `${rows.length}/${state.lab.rows.length}` : "";
  if (!rows.length) { setChildren(list, el("div", { class: "list-empty", text: state.lab.rows.length ? (state.lab.filter === "starred" ? "No starred alphas yet." : "No alphas match the filter.") : "No alphas yet — write some and press ⌘R." })); return; }
  setChildren(list, ...rows.map((r) => labItem(r)));
}
function onLabContext(e, r, trashed) {
  if (trashed) {
    contextMenu(e.clientX, e.clientY, [{ label: "Restore", icon: "restore", action: () => deleteAlphas([r], false) }, { label: "Copy expression", icon: "copy", action: () => copyText(r.expression, "Expression copied") }]);
    return;
  }
  const targets = state.lab.chosen.length > 1 && state.lab.chosen.includes(r.id) ? state.lab.rows.filter((x) => state.lab.chosen.includes(x.id)) : [r];
  const many = targets.length > 1, allStarred = targets.every((t) => t.starred);
  contextMenu(e.clientX, e.clientY, [
    many ? { header: `${targets.length} alphas selected` } : { label: "Open", icon: "open", action: () => selectLab(r.id) },
    { label: allStarred ? "Unstar" : "Star", icon: "star", checked: allStarred, action: () => starAlphas(targets, !allStarred) },
    many ? null : { label: "Rename…", icon: "edit", action: () => renameAlpha(r) },
    many ? null : { label: "Use saved settings", icon: "settings", action: () => useLabSettings(r.id) },
    { label: many ? "Add expressions to the editor" : "Add to the editor", icon: "load", action: async () => { try {
      const results = await Promise.all(targets.map(t => get("alpha-result", { id: t.id })));
      const selection = d => { const s = savedSec(d); return JSON.stringify(s ? {catalog:s.catalog,mode:secMode(s),registry:s.registry || null,bindings:s.bindings} : null); };
      const sec = selection(results[0]);
      if (results.some(d => selection(d) !== sec)) { toast("Different saved SEC inputs", "Load alphas with the same saved SEC catalog and bindings together.", "bad"); return; }
      const text = $("#lab-text").value.trimEnd();
      const groupSelection = d => JSON.stringify(savedLabGroup(d));
      const group = groupSelection(results[0]);
      if (results.some(d => groupSelection(d) !== group)) { toast("Different saved classification sources", "Add alphas with the same saved classification provenance together.", "bad"); return; }
      const currentGroup = JSON.stringify({path: state.lab.groupOverlay, identity: state.lab.groupIdentity, error: state.lab.groupRestoreError});
      if (text && currentGroup !== group) { toast("Editor classification source differs", "Clear the editor or use saved settings to preserve each alpha's original source.", "bad"); return; }
      if (text && selection({sec:secRequest().sec}) !== sec) { toast("Editor SEC inputs differ", "Clear the editor or use saved settings before adding this alpha, so existing expressions keep their inputs.", "bad"); return; }
      if (!await restoreSavedSec(results[0])) return;
      restoreLabGroup(results[0]);
      await loadLabGroups();
      $("#lab-text").value = `${text ? text + "\n" : ""}${targets.map((t) => t.expression).join("\n")}\n`; onLabInput(); persist();
    } catch(error) { toast("Cannot reuse saved alphas", error.message, "bad"); } } },
    { label: many ? "Copy expressions" : "Copy expression", icon: "copy", action: () => copyText(targets.map((t) => t.expression).join("\n"), "Copied") },
    "sep",
    { label: many ? `Delete ${targets.length} alphas…` : "Delete…", icon: "trash", danger: true, action: () => deleteAlphas(targets, true) },
  ]);
}
async function starAlphas(rows, starred) {
  try { for (const r of rows) await api("annotate", { kind: "alphas", key: r.id, starred }); rows.forEach((r) => { r.starred = starred; }); renderLabList(); }
  catch (e) { toast("Cannot update the star", e.message, "bad"); }
}
async function renameAlpha(r) {
  const value = await dialog({ title: "Rename alpha", message: "A short label shown instead of the expression.", input: { value: r.label || "", placeholder: r.expression.slice(0, 60) }, confirm: "Save" });
  if (value === null) return;
  try { await api("annotate", { kind: "alphas", key: r.id, label: value }); r.label = value.trim(); renderLabList(); }
  catch (e) { toast("Cannot rename", e.message, "bad"); }
}
async function deleteAlphas(rows, deleted) {
  if (deleted) {
    const ok = await dialog({ title: rows.length === 1 ? "Delete this alpha from the list?" : `Delete ${rows.length} alphas from the list?`,
      message: "They are hidden from the library list and can be restored from Trash. Saved results are kept.", confirm: "Delete", danger: true });
    if (!ok) return;
  }
  try {
    await api("alpha-delete", { ids: rows.map((r) => r.id), deleted });
    state.lab.chosen = [];
    await refreshLab();
    if (deleted) toast(`Deleted ${rows.length} alpha${rows.length === 1 ? "" : "s"}`, "", "", { label: "Undo", run: () => deleteAlphas(rows, false) });
    else toast(`Restored ${rows.length} alpha${rows.length === 1 ? "" : "s"}`, "", "good");
  } catch (e) { toast("Cannot update the library", e.message, "bad"); }
}
async function selectLab(id) {
  state.lab.selected = id;
  $$("#lab-list .run-item").forEach((i) => i.classList.toggle("selected", i.dataset.id === id));
  let d;
  try { d = await get("alpha-result", { id }); } catch (e) { toast("Cannot read the alpha", e.message, "bad"); return; }
  if (state.lab.selected !== id) return;
  state.lab.detail = { ...d, engine: d.engine || (state.lab.rows.find((r) => r.id === id) || {}).engine }; state.lab.view = "detail";
  renderLabDetail();
}
function labCheckTags(checks) {
  return (checks || []).map((c) => el("span", { class: `tag ${c.result === "PASS" ? "good" : c.result === "FAIL" ? "bad" : ""}`,
    title: `value ${finite(c.value) ? fmt.num(c.value, 3) : "—"} · limit ${c.limit}`, text: `${c.name.replace(/_/g, " ").toLowerCase()} ${c.result === "PASS" ? "✓" : c.result === "FAIL" ? "✗" : "?"}` }));
}
function labEngineNote(row) {
  if (row.engine === "alphalab-1") return "alphalab-1 · Preserved result from before the assignment and capped-neutrality fixes; these numbers have not been recomputed. Simulate again with alphalab-3 for corrected calculations.";
  if (row.engine === "alphalab-2") return "alphalab-2 · Corrected engine assignments; preserved result from before the capped-neutrality fix. Truncation could break neutrality. Simulate again with alphalab-3 for corrected calculations.";
  if (row.engine === "alphalab-3") return "alphalab-3 · Corrected assignments and capped group neutrality";
  return `${row.engine || "Engine version unavailable"} · Do not assume this result includes the alphalab-3 corrections.`;
}
function labEngineBanner(row) {
  return el("div", { class: `console-notice${row.engine === "alphalab-3" ? "" : " compatibility-note"}`, text: labEngineNote(row) });
}
async function useLabSettings(identifier = null) {
  try {
    const d = identifier ? await get("alpha-result", { id: identifier }) : state.lab.detail;
    if (!d || !await restoreSavedSec(d)) return false;
    // Old price-only results mean explicitly no overlay, even if discovery
    // now offers a sole compatible source.
    restoreLabGroup(d);
    const dataset = typeof d.dataset === "object" ? d.dataset.dataset : d.dataset;
    if (dataset) {
      if (!priceRow(dataset)) { toast("Saved trading prices unavailable", "Select the original price dataset before reusing this alpha.", "bad"); return false; }
      await loadDataset(dataset, false, "lab");
      if (state.lab.datasetPath !== dataset) return false;
    }
    const cfg = d.settings || {};
    restoreLab({ ...cfg, text: d.expression, delay: String(cfg.delay ?? 1), decay: String(cfg.decay ?? 0),
      truncation: String(cfg.truncation ?? .08), cost: String(cfg.cost_bps ?? 0), pasteurize: cfg.pasteurization !== false });
    await loadLabGroups();
    renderLabPeriod(); persist(); await checkLab(); return true;
  } catch (error) { toast("Cannot reuse saved alpha", error.message, "bad"); return false; }
}
function labClassificationProvenance(result) {
  const classification = result.classification, source = classification?.source_identity;
  const summary = result.summary || {};
  const display = value => typeof value === "object" ? JSON.stringify(value) : String(value);
  const percentage = value => typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : display(value);
  const coverage = summary.classification_coverage ?? classification?.coverage;
  const exposure = summary.max_abs_group_net_exposure;
  const inactive = summary.inactive_neutralized_days;
  const diagnostics = [
    coverage != null ? "Classification coverage: " + percentage(coverage) : null,
    exposure != null ? "Max absolute group net exposure: " + (typeof exposure === "number" && Number.isFinite(exposure)
      ? exposure === 0 ? "0%" : Math.abs(exposure) < 1e-8 ? "<0.000001%" : `${(Math.abs(exposure) * 100).toFixed(6)}%` : display(exposure)) : null,
    inactive != null ? "Inactive neutralized days: " + (typeof inactive === "number" && Number.isFinite(inactive) ? Math.round(inactive).toLocaleString("en-US") : display(inactive)) : null,
  ].filter(Boolean);
  const neutralization = String(result.settings?.neutralization || "MARKET").toUpperCase();
  const prices = typeof result.dataset === "string" ? result.dataset : result.dataset?.dataset || result.dataset?.path;
  const sourceLabel = source?.path || (["MARKET", "NONE"].includes(neutralization) ? "not required"
    : "embedded groups from Trading prices" + (prices ? ": " + prices : " (saved price source path unavailable)"));
  const description = groupDescription(source);
  return el("div", { class: "console-notice lab-classification" },
    el("div", { text: "Classification source: " + sourceLabel }),
    source ? el("div", { text: description.caption }) : null,
    source ? el("details", {}, el("summary", { text: "Saved source identity and limitations" }),
      el("div", { text: "Saved identity: " + display(source) }), ...description.details.map(text => el("div", { text }))) : null,
    ...diagnostics.map(text => el("div", { text })));
}
function renderLabDetail() {
  const d = state.lab.detail, body = $("#lab-body");
  $("#lab-empty").hidden = true; body.hidden = false;
  const back = state.lab.batch ? el("button", { class: "btn small", text: "← Batch results", onclick: () => renderLabBatch() }) : null;
  if (d.error) {
    setChildren(body, el("div", { class: "run-head" }, el("div", { class: "run-title" }, el("span", { class: "dot failed" }), el("span", { class: "mono", text: d.expression })), back),
      labEngineBanner(d), labClassificationProvenance(d), el("div", { class: "lab-error", text: d.error }));
    return;
  }
  const s = d.summary || {}, cfg = d.settings || {};
  const tags = [cfg.universe, `delay ${cfg.delay}`, `decay ${cfg.decay}`, cfg.neutralization, `truncation ${cfg.truncation}`,
    cfg.cost_bps ? `${cfg.cost_bps} bps` : "no costs", `${s.start} → ${s.end}`, `${fmt.int(s.days)} sessions`];
  setChildren(body, 
    el("div", { class: "run-head" }, el("div", { class: "run-title" }, el("span", { class: "dot completed" }), el("span", { class: "mono", text: d.expression, title: d.expression })),
      el("div", { class: "run-meta" }, ...tags.map((t) => el("span", { class: "tag", text: String(t) })), back)),
    labEngineBanner(d), secProvenance(d), labClassificationProvenance(d), el("button", { class: "btn small", text: "Use saved settings", onclick: () => useLabSettings() }), el("div", { class: "kpis" },
      kpiTile("Sharpe", fmt.num(s.sharpe), "annualized", tone(s.sharpe)), kpiTile("Fitness", fmt.num(s.fitness), "BRAIN formula", tone(s.fitness)),
      kpiTile("Returns", fmt.signedPct(s.returns), "on half the book", tone(s.returns)), kpiTile("Turnover", fmt.pct(s.turnover), "traded per day"),
      kpiTile("Drawdown", fmt.pct(s.drawdown), "of half the book", "neg"), kpiTile("Margin", `${fmt.num(s.margin_bps, 1)} bps`, "PnL per $ traded", tone(s.margin_bps))),
    el("div", { class: "lab-checks" }, ...labCheckTags(d.checks),
      el("span", { class: "tag", text: `long/short ${fmt.int(s.long_count)}/${fmt.int(s.short_count)}` }),
      finite(d.self_correlation) ? el("span", { class: `tag ${d.self_correlation >= 0.7 ? "bad" : ""}`, text: `self-corr ${fmt.num(d.self_correlation)}` }) : null),
    el("div", { class: "charts" },
      el("div", { class: "chart-card" }, el("div", { class: "chart-head" }, el("span", { class: "chart-title", text: "Cumulative PnL" }),
        el("span", { class: "chart-note", text: `${fmt.money(cfg.booksize)} book · ${fmt.money(s.pnl)} total` })), el("div", { class: "chart", id: "lab-pnl" }), el("div", { class: "chart short", id: "lab-dd" }))),
    labYearly(d.yearly || [], d.periods || {}));
  drawWhenSized($("#lab-pnl"), renderLabCharts);
}
function labYearly(yearly, periods) {
  const row = (label, r) => el("tr", {}, el("td", { text: String(label) }), el("td", { class: `num ${tone(r.sharpe)}`, text: fmt.num(r.sharpe) }),
    el("td", { class: `num ${tone(r.returns)}`, text: fmt.signedPct(r.returns) }), el("td", { class: "num", text: fmt.pct(r.turnover) }),
    el("td", { class: `num ${tone(r.fitness)}`, text: fmt.num(r.fitness) }), el("td", { class: "num", text: fmt.pct(r.drawdown) }),
    el("td", { class: "num", text: finite(r.margin_bps) ? fmt.num(r.margin_bps, 1) : "—" }), el("td", { class: "num", text: fmt.int(r.days) }));
  const extra = Object.entries(periods).map(([name, r]) => row(name.replace(/_/g, " "), r));
  return el("div", { class: "fact-card lab-years" }, el("h3", { text: "By year" }),
    el("table", { class: "table" }, el("thead", {}, el("tr", {}, ...["Year", "Sharpe", "Returns", "Turnover", "Fitness", "Drawdown", "Margin bps", "Days"].map((h, i) => el("th", { class: i ? "num" : "", text: h })))),
      el("tbody", {}, ...yearly.map((r) => row(r.year, r)), ...extra)));
}
function renderLabCharts() {
  const d = state.lab.detail;
  if (state.view !== "lab" || !d || !d.series || !$("#lab-pnl")) return;
  const s = d.series, group = crosshairGroup();
  group.add(lineChart($("#lab-pnl"), { dates: s.dates, values: s.cumulative, baseline: 0, format: fmt.money, label: "Cumulative PnL", group,
    tooltip: (i) => [["PnL", fmt.money(s.cumulative[i]), tone(s.cumulative[i])], ["Drawdown", fmt.money(s.drawdown[i]), s.drawdown[i] < 0 ? "neg" : ""]] }));
  group.add(lineChart($("#lab-dd"), { dates: s.dates, values: s.drawdown, kind: "dd", format: fmt.money, axisX: false, label: "Drawdown", group,
    tooltip: (i) => [["Drawdown", fmt.money(s.drawdown[i]), s.drawdown[i] < 0 ? "neg" : ""]] }));
}
function renderLabBatch() {
  const b = state.lab.batch, body = $("#lab-body");
  if (!b) return;
  state.lab.view = "batch";
  $("#lab-empty").hidden = true; body.hidden = false;
  const ok = b.rows.filter((r) => !r.error).sort((x, y) => (y.summary.fitness ?? -1e9) - (x.summary.fitness ?? -1e9));
  const failed = b.rows.filter((r) => r.error);
  const passes = (r) => (r.checks || []).filter((c) => c.result === "PASS").length;
  const cfg = b.settings || {};
  const counts = b.diagnostics?.version === 1 ? b.diagnostics.alphalab : null;
  const reused = counts?.cache_hit_count, computed = counts?.new_computation_count;
  // Computations count distinct uncached alphas, including failed simulations.
  const validCounts = Number.isSafeInteger(reused) && Number.isSafeInteger(computed)
    && reused >= 0 && computed >= 0 && reused <= b.rows.length && computed <= b.rows.length
    && reused + computed <= b.rows.length;
  const resultBadge = `${ok.length} results` + (validCounts ? ` · ${reused} reused · ${computed} computed` : "");
  setChildren(body, 
    el("div", { class: "run-head" }, el("div", { class: "run-title" }, el("span", { class: "dot completed" }), el("span", { text: `Batch of ${b.rows.length} alphas` })),
      el("div", { class: "run-meta" }, ...[cfg.universe, `delay ${cfg.delay}`, `decay ${cfg.decay}`, cfg.neutralization, `truncation ${cfg.truncation}`,
        resultBadge, failed.length ? `${failed.length} failed` : ""].filter(Boolean).map((t) => el("span", { class: "tag", text: String(t) })))),
    el("div", { class: "fact-card" }, el("h3", { text: "Leaderboard (click a row for details)" }),
      el("table", { class: "table lab-table" },
        el("thead", {}, el("tr", {}, ...["#", "Fitness", "Sharpe", "Returns", "Turnover", "Drawdown", "Self-corr", "Checks", "Alpha"].map((h, i) => el("th", { class: i && i < 8 ? "num" : "", text: h })))),
        el("tbody", {}, ...ok.map((r, i) => el("tr", { class: "clickable", onclick: () => selectLab(r.id) },
          el("td", { text: String(i + 1) }), el("td", { class: `num ${tone(r.summary.fitness)}`, text: fmt.num(r.summary.fitness) }),
          el("td", { class: `num ${tone(r.summary.sharpe)}`, text: fmt.num(r.summary.sharpe) }), el("td", { class: `num ${tone(r.summary.returns)}`, text: fmt.signedPct(r.summary.returns) }),
          el("td", { class: "num", text: fmt.pct(r.summary.turnover) }), el("td", { class: "num", text: fmt.pct(r.summary.drawdown) }),
          el("td", { class: `num ${finite(r.self_correlation) && r.self_correlation >= 0.7 ? "neg" : ""}`, text: finite(r.self_correlation) ? fmt.num(r.self_correlation) : "—" }),
          el("td", { class: "num", text: `${passes(r)}/${(r.checks || []).length}` }), el("td", { class: "mono ellipsis", text: r.expression, title: `${r.expression}\n${labEngineNote(r)}` },
            el("span", { class: r.engine === "alphalab-3" ? "tag" : "tag warn", text: r.engine || "engine unknown" }))))))),
    b.top_pairs.length ? el("div", { class: "fact-card" }, el("h3", { text: "Most correlated pairs in this batch" }),
      el("ul", { class: "limits" }, ...b.top_pairs.slice(0, 8).map((p) => {
        const name = (id) => (b.rows.find((r) => r.id === id) || {}).expression || id;
        return el("li", {}, el("b", { class: p.correlation >= 0.7 ? "neg" : "", text: fmt.num(p.correlation) }), ` ${name(p.a)}  ~  ${name(p.b)}`);
      }))) : null,
    failed.length ? el("div", { class: "fact-card" }, el("h3", { text: "Failed" }), el("ul", { class: "limits" }, ...failed.map((r) => el("li", {}, el("span", { class: "mono", text: r.expression }), ` — ${r.error}`)))) : null);
}

// ================================================================= status bar
function updateStatusRight() {
  const lab = state.view === "lab", d = lab ? state.lab.dataset : state.dataset;
  $("#sb-dataset").textContent = d ? `${lab ? "Lab prices" : "Backtest prices"}: ${datasetTitle(d)} · ${lab ? $("#lab-universe").value : $("#universe").value || "all symbols"}` : "no trading prices selected";
}

// ================================================================= command palette
function commands() {
  const list = [
    ["Run", "Run Qlib validation", "⌘R", runBacktest], ["Run", "Stop the running task", "⌘.", cancelJob],
    ["View", "Qlib Validation", "⌘1", () => setView("research")], ["View", "Alpha Lab", "⌘2", () => setView("lab")], ["View", "Data", "⌘3", () => setView("data")], ["View", "Research", "⌘4", () => setView("auto")],
    ["Run", "Simulate the Alpha Lab alphas", "", runLab],
    ["View", "Toggle bottom panel", "⌘J", () => toggleBottom()], ["View", "Toggle settings panel", "⌘I", () => toggleInspector()], ["View", "Console", "", () => { toggleBottom(true); setBottomTab("console"); }],
    ["View", "Holdings", "", () => { toggleBottom(true); setBottomTab("holdings"); }], ["View", "Universe membership", "", () => { toggleBottom(true); setBottomTab("universe"); }],
    ["Runs", "Show starred", "", () => { setView("research"); setRunFilter("starred"); }], ["Runs", "Group by factor", "", () => { setView("research"); setRunFilter("factors"); }],
    ["Runs", "Open the trash", "", () => { setView("research"); setRunFilter("trash"); }],
    ["Strategy", "Signed weights", "", () => { setMode("signed"); persist(); }], ["Strategy", "Long-only Top-k", "", () => { setMode("long-only"); persist(); }],
    ["Window", "Whole dataset", "", () => applyPreset("all")], ["Window", "From the full universe", "", () => applyPreset("universe")],
    ["Window", "Last 10 years", "", () => applyPreset("10y")], ["Window", "Last 5 years", "", () => applyPreset("5y")], ["Window", "Last 3 years", "", () => applyPreset("3y")],
    ["Window", "Last year", "", () => applyPreset("1y")], ["Window", "Year to date", "", () => applyPreset("ytd")],
    ["Help", "Keyboard and mouse shortcuts", "?", showShortcuts], ["Params", "Reset parameters", "", resetParams], ["Data", "Choose dataset…", "⌘D", showDatasetPopover],
    ["Data", "Open the data folder", "", () => api("reveal", { path: "" }).catch((e) => toast("Cannot open", e.message, "bad"))],
    ["Data", "Import a table…", "", () => { setView("data"); $(".iview[data-view=data] details").open = true; $("#table").focus(); }],
    ["Data", "Refresh market data from Yahoo", "", () => { setView("data"); startRefresh(); }],
  ];
  if (state.runDetail && runReusable(state.runDetail)) {
    list.push(["Report", "Overview report", "", () => buildOrOpen("overview", "overview")], ["Report", "Detailed report", "", () => buildOrOpen("detailed", "detailed")],
      ["Report", "Holdings report", "", () => buildOrOpen("holdings", "summary")], ["Run", "Use this run's settings", "", useRunSettings],
      ["Run", state.runDetail.starred ? "Unstar this run" : "Star this run", "S", () => starRuns([state.runs.find((r) => r.path === state.runDetail.path)].filter(Boolean), !state.runDetail.starred)],
      ["Run", "Signal diagnostics (IC, quantiles)", "", () => setResultTab("signal")], ["Run", "Monthly returns", "", () => setResultTab("returns")],
      ["Run", "Reveal run folder", "", () => api("reveal", { path: state.run }).catch((e) => toast("Cannot reveal", e.message, "bad"))]);
  }
  for (const ex of (state.overview && state.overview.examples) || []) list.push(["Example", ex.label, ex.alpha.slice(0, 36), () => setAlpha(ex.alpha)]);
  for (const d of state.datasets.filter(priceEligible)) list.push(["Trading prices", datasetTitle(d), datasetMetadata(d), () => loadDataset(d.path, false, state.view === "lab" ? "lab" : "research")]);
  for (const r of state.runs.slice(0, 60)) list.push(["Run", `${r.starred ? "★ " : ""}${runTitle(r)}`, `${shortTime(r.created)} · Sharpe ${fmt.num(r.sharpe)}`, () => { setView("research"); openRun(r.path); }]);
  return list;
}
function buildOrOpen(profile, level) {
  if (!reportsSupported(state.runDetail)) return;
  const existing = ((state.runDetail && state.runDetail.reports) || []).find((r) => r.level === level && (level !== "summary" || r.holdings));
  existing ? openReport(existing.path, profile) : buildReport(profile);
}
let paletteItems = [], paletteIndex = 0;
function openPalette() {
  if (!state.datasets.length) get("datasets").then((rows) => { state.datasets = rows; }).catch(() => {});
  $("#palette").hidden = false;
  $("#palette-input").value = "";
  filterPalette();
  $("#palette-input").focus();
}
function closePalette() { $("#palette").hidden = true; }
function filterPalette() {
  const query = $("#palette-input").value.trim().toLowerCase();
  const score = (text) => {
    if (!query) return 1;
    const t = text.toLowerCase();
    if (t.includes(query)) return 100 - t.indexOf(query);
    let i = 0;
    for (const c of t) if (c === query[i]) i++;
    return i === query.length ? 10 : 0;
  };
  paletteItems = commands().map((c) => ({ c, s: score(`${c[0]} ${c[1]} ${c[2]}`) })).filter((x) => x.s > 0).sort((a, b) => b.s - a.s).slice(0, 60).map((x) => x.c);
  paletteIndex = 0;
  renderPalette();
}
function renderPalette() {
  const list = $("#palette-list");
  setChildren(list, ...paletteItems.map(([group, label, hint], i) => el("div", { class: `p-item${i === paletteIndex ? " active" : ""}`, role: "option",
    onmousemove: () => { if (paletteIndex !== i) { paletteIndex = i; renderPalette(); } }, onclick: () => runPalette(i) },
    el("span", { class: "p-group", text: group }), el("span", { class: "p-label", text: label }), hint ? el("span", { class: "p-hint", text: hint }) : null)));
  if (!paletteItems.length) list.append(el("div", { class: "list-empty", text: "No matching commands." }));
  list.children[paletteIndex]?.scrollIntoView({ block: "nearest" });
}
function runPalette(i) { const item = paletteItems[i]; closePalette(); if (item) item[3](); }

// ================================================================= boot
function bind() {
  bindSecBrowser();
  $$(".act[data-view]").forEach((b) => b.addEventListener("click", () => { setView(b.dataset.view); persist(); }));
  $("#toggle-bottom").addEventListener("click", () => toggleBottom());
  $("#toggle-inspector").addEventListener("click", () => toggleInspector());
  $("#bottom-hide").addEventListener("click", () => toggleBottom(false));
  $("#open-folder").addEventListener("click", () => api("reveal", { path: "" }).catch((e) => toast("Cannot open", e.message, "bad")));
  $("#sb-root").addEventListener("click", () => api("reveal", { path: "" }).catch((e) => toast("Cannot open", e.message, "bad")));
  $$("#mode button").forEach((b) => b.addEventListener("click", () => { setMode(b.dataset.mode); persist(); }));
  $("#universe").addEventListener("change", () => { state.settings.universe = $("#universe").value; renderUniversePanel(); updateStatusRight(); renderPeriod(); persist(); });
  $("#tb-dataset").addEventListener("click", (e) => { e.stopPropagation(); showDatasetPopover("research"); });
  $("#tb-lab-dataset").addEventListener("click", (e) => { e.stopPropagation(); showDatasetPopover("lab"); });
  $("#alpha").addEventListener("input", () => { onAlphaInput(); persist(); });
  $("#alpha").addEventListener("scroll", syncEditorScroll);
  $("#alpha").addEventListener("keydown", (e) => { if (e.key === "Tab") { e.preventDefault(); document.execCommand("insertText", false, "  "); } });
  $("#gross-range").addEventListener("input", () => { $("#gross").value = $("#gross-range").value; persist(); });
  $("#gross").addEventListener("input", () => { $("#gross-range").value = $("#gross").value; });
  $$(".props input").forEach((i) => i.addEventListener("change", persist));
  for (const id of ["#start", "#end", "#warmup"]) $(id).addEventListener("change", renderPeriod);
  for (const id of ["#lab-start", "#lab-end"]) $(id).addEventListener("change", renderLabPeriod);
  $$("#transform button").forEach((b) => b.addEventListener("click", () => { setTransform(b.dataset.transform); persist(); }));
  $$("#period-presets button").forEach((b) => b.addEventListener("click", () => applyPreset(b.dataset.range)));
  $$("#lab-period-presets button").forEach((b) => b.addEventListener("click", () => applyLabPreset(b.dataset.range)));
  $("#reset-params").addEventListener("click", resetParams);
  $("#run").addEventListener("click", runCurrent); $("#run-2").addEventListener("click", runBacktest);
  $("#lab-run").addEventListener("click", runLab);
  $("#lab-text").addEventListener("input", () => { onLabInput(); persist(); });
  $("#lab-text").addEventListener("keydown", (e) => { if (e.key === "Tab") { e.preventDefault(); document.execCommand("insertText", false, "  "); } });
  $$("#lab-delay button").forEach((b) => b.addEventListener("click", () => { setLabDelay(b.dataset.delay); persist(); }));
  $$("#lab-props input, #lab-props select").forEach((i) => i.addEventListener("change", persist));
  $("#lab-group-source").addEventListener("change", event => chooseLabGroup(event.target.value));
  $("#lab-group-path").addEventListener("change", event => chooseLabGroup(event.target.value));
  $("#lab-group-refresh").addEventListener("click", loadLabGroups);
  $("#lab-neutralization").addEventListener("change", () => { renderLabGroups(); onLabInput(); });
  $("#lab-group-choose").addEventListener("click", async () => {
    try {
      const bridge = window.pywebview?.api;
      if (bridge?.choose_folder) {
        const path = await bridge.choose_folder(state.lab.groupOverlay || "");
        if (path) chooseLabGroup(path);
      } else if (bridge?.choose_table) {
        const manifest = await bridge.choose_table(state.lab.groupOverlay ? state.lab.groupOverlay + "/manifest.json" : "");
        if (manifest && /[\\/]manifest\.json$/i.test(manifest)) chooseLabGroup(manifest.replace(/[\\/]manifest\.json$/i, ""));
        else if (manifest) toast("Choose manifest.json", "Select the classification folder's manifest.json using All files.", "bad");
      } else { $("#lab-group-path").focus(); toast("Enter the classification folder", "The native chooser is available in the desktop app."); }
    } catch (error) { toast("Cannot choose classification source", error.message, "bad"); }
  });
  $("#lab-reset").addEventListener("click", () => { restoreLab({ text: $("#lab-text").value }); renderLabPeriod(); persist(); });
  $("#lab-refresh").addEventListener("click", refreshLab);
  $("#lab-search").addEventListener("input", renderLabList);
  $$("#lab-filter button").forEach((b) => b.addEventListener("click", () => setLabFilter(b.dataset.filter)));
  $$("#lab-sort button").forEach((b) => b.addEventListener("click", () => {
    state.lab.sort = b.dataset.sort; $$("#lab-sort button").forEach((x) => x.classList.toggle("active", x === b)); refreshLab();
  }));
  $("#lab-examples").addEventListener("click", () => {
    const text = (state.overview && state.overview.lab_examples) || "";
    if (!text) { toast("No example set found", "The bundled teaching examples are missing. Restore the framework's qlib_yahoo/examples/price_volume.txt resource."); return; }
    $("#lab-text").value = text; onLabInput(); persist();
  });
  $("#cancel").addEventListener("click", cancelJob); $("#sb-cancel").addEventListener("click", cancelJob);
  $("#import").addEventListener("click", startImport);
  $("#refresh-data").addEventListener("click", startRefresh);
  $("#choose-table").addEventListener("click", async () => {
    if (window.pywebview && window.pywebview.api && window.pywebview.api.choose_table) {
      const path = await window.pywebview.api.choose_table($("#table").value || "");
      if (path) { $("#table").value = path; persist(); }
    } else { $("#table").focus(); toast("Type the table path", "The native file dialog is available in the desktop app."); }
  });
  $("#runs-refresh").addEventListener("click", () => refreshRuns());
  $("#datasets-refresh").addEventListener("click", refreshDatasets);
  $("#run-search").addEventListener("input", renderRunList);
  $$("#run-filter button").forEach((b) => b.addEventListener("click", () => setRunFilter(b.dataset.filter)));
  $$("#run-sort button").forEach((b) => b.addEventListener("click", () => {
    state.runSort = b.dataset.sort; $$("#run-sort button").forEach((x) => x.classList.toggle("active", x === b)); renderRunList(); persist();
  }));
  $("#runs-list").addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); moveRunSelection(1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); moveRunSelection(-1); }
    else if ((e.key === "Delete" || e.key === "Backspace") && state.runFilter !== "trash" && state.runFilter !== "factors") { e.preventDefault(); trashRuns(listTargets()); }
    else if (e.key.toLowerCase() === "s" && !e.metaKey && !e.ctrlKey) { const t = listTargets(); if (t.length) { e.preventDefault(); starRuns(t, !t.every((r) => r.starred)); } }
    else if (e.key === "F2") { const t = listTargets(); if (t.length === 1) { e.preventDefault(); renameRun(t[0]); } }
    else if (e.key.toLowerCase() === "a" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); state.selected = visibleRuns().map((r) => r.path); afterSelection(); }
    else if (e.key === "Escape" && state.selected.length) { e.preventDefault(); clearSelection(); }
  });
  $$("#main-tabs .tab").forEach((t) => t.addEventListener("click", (e) => { if (e.target.id !== "report-close") setResultTab(t.dataset.tab); }));
  $("#report-close").addEventListener("click", (e) => { e.stopPropagation(); $("#report-tab").hidden = true; $("#report-frame").src = "about:blank"; setResultTab("overview"); });
  $$("#bottom-tabs .tab").forEach((t) => t.addEventListener("click", () => setBottomTab(t.dataset.ptab)));
  $("#console-clear").addEventListener("click", () => { $("#console").textContent = ""; });
  $$("#scale button").forEach((b) => b.addEventListener("click", () => { state.scale = b.dataset.scale; $$("#scale button").forEach((x) => x.classList.toggle("active", x === b)); renderResultTab(); persist(); }));
  $("#use-settings").addEventListener("click", useRunSettings);
  $("#star-run").addEventListener("click", () => { const d = state.runDetail; const r = d && state.runs.find((x) => x.path === d.path); if (r) starRuns([r], !r.starred); });
  $("#run-more").addEventListener("click", runMoreMenu);
  $("#report-btn").addEventListener("click", (e) => { e.stopPropagation(); renderReportMenu(); $("#report-menu").hidden = !$("#report-menu").hidden; });
  $("#examples-btn").addEventListener("click", (e) => { e.stopPropagation(); $("#examples-menu").hidden = !$("#examples-menu").hidden; });
  document.addEventListener("click", (e) => {
    if (!e.target.closest(".menu-wrap")) $$(".menu").forEach((m) => { m.hidden = true; });
    if (!e.target.closest("#dataset-popover") && !e.target.closest("#tb-dataset") && !e.target.closest("#tb-lab-dataset")) $("#dataset-popover").hidden = true;
  });
  $("#palette-btn").addEventListener("click", openPalette);
  $("#palette").addEventListener("mousedown", (e) => { if (e.target.id === "palette") closePalette(); });
  $("#palette-input").addEventListener("input", filterPalette);
  $("#palette-input").addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); paletteIndex = Math.min(paletteItems.length - 1, paletteIndex + 1); renderPalette(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); paletteIndex = Math.max(0, paletteIndex - 1); renderPalette(); }
    else if (e.key === "Enter") { e.preventDefault(); runPalette(paletteIndex); }
    else if (e.key === "Escape") { e.preventDefault(); closePalette(); }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "?" && !e.metaKey && !e.ctrlKey && !/^(INPUT|TEXTAREA|SELECT)$/.test((document.activeElement || {}).tagName || "")) { e.preventDefault(); showShortcuts(); return; }
    if (e.key === "Escape") { $$(".menu").forEach((m) => { m.hidden = true; }); $("#dataset-popover").hidden = true; if (!$("#palette").hidden) closePalette(); }
    if (!(e.metaKey || e.ctrlKey)) return;
    const key = e.key.toLowerCase();
    const actions = { r: runCurrent, ".": cancelJob, k: openPalette, j: () => toggleBottom(), i: () => toggleInspector(), d: () => showDatasetPopover(),
      "1": () => setView("research"), "2": () => setView("lab"), "3": () => setView("data"), "4": () => setView("auto") };
    if (actions[key]) { e.preventDefault(); actions[key](); }
  });
  let resizeTimer = null;
  window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(redrawAll, 120); });
  bindResizers();
}

async function boot() {
  $("#sb-version").textContent = `UI ${UI_VERSION}`;
  bind();
  session.token = (await get("session")).token;
  initializeResearch();
  if (typeof initializeWorkflow === "function") initializeWorkflow();
  const [overview, settings] = await Promise.all([get("overview"), get("settings").catch(() => ({}))]);
  state.overview = overview; state.settings = preferenceObject(settings);
  const savedSec = preferenceObject(state.settings.sec);
  state.sec.catalog = typeof savedSec.catalog === "string" ? savedSec.catalog : "";
  state.sec.bindings = savedSec.bindings && typeof savedSec.bindings === "object" && !Array.isArray(savedSec.bindings) ? savedSec.bindings : {};
  state.sec.mode = secMode(savedSec);
  state.sec.registry = savedSec.registry || null;
  state.sec.restoreError = typeof savedSec.restore_error === "string" ? savedSec.restore_error : null;
  renderSecBindings();
  // An early dialog load may have captured the catalog before settings arrived.
  if ($("#sec-dialog").open) loadSecCatalogs().catch(error => secMessage(error.message, true));
  $("#sb-root").textContent = home(overview.data_root);
  $("#sb-root").title = `${overview.data_root} — show the folder`;
  $("#sb-engine .sb-dot").className = `sb-dot ${overview.signed.available ? "" : "off"}`;
  $("#sb-engine").title = overview.signed.detail;
  $("#sb-engine span:last-child").textContent = overview.signed.available ? "Signed engine ready" : "Signed engine not installed";
  $("#sb-version").textContent = `UI ${UI_VERSION} · Py ${overview.python} · ${overview.version}`;
  setChildren($("#examples-menu"), ...overview.examples.map((ex) => el("button", { onclick: () => { setAlpha(ex.alpha); persist(); $("#examples-menu").hidden = true; } },
    ex.label, el("small", { text: ex.alpha }))));
  $("#alpha").value = overview.examples[0] ? overview.examples[0].alpha : "";
  $("#imp-end").value = new Date().toISOString().slice(0, 10);
  restoreSettings(state.settings);
  $$("#scale button").forEach((b) => b.classList.toggle("active", b.dataset.scale === state.scale));
  $$("#run-filter button").forEach((b) => b.classList.toggle("active", b.dataset.filter === state.runFilter));
  $$("#run-sort button").forEach((b) => b.classList.toggle("active", b.dataset.sort === state.runSort));
  $$("#lab-filter button").forEach((b) => b.classList.toggle("active", b.dataset.filter === state.lab.filter));
  state.mode = state.settings.mode === "long-only" ? "long-only" : "signed";
  setTransform(state.settings.transform || state.transform);
  renderEditor();
  state.datasets = await get("datasets");
  await restorePriceSelections();
  await refreshRuns();
  const remembered = state.settings.run && state.runs.find((r) => r.path === state.settings.run);
  state.run = remembered ? remembered.path : null;
  // Restore tab styling without fetching or drawing a hidden validation result.
  setResultTab(state.resultTab);
  setView("lab");
  await Promise.all([checkAlpha(), checkLab()]);
  state.restoring = false;
  if (overview.job && overview.job.state === "running") startPolling(overview.job);
  else consoleLine(`Ready · data folder ${home(overview.data_root)}`, "l-stage");
}

function showBootFailure(error) {
  const box = $("#boot-error");
  box.hidden = false;
  setChildren(box, el("div", { text: `Console could not start: ${error.message || String(error)}. UI ${UI_VERSION}. Reload to retry.` }),
    el("button", { class: "btn small", text: "Reload console", onclick: () => window.location.reload() }));
  $("#sb-text").textContent = "Console startup failed";
  for (const id of ["#run", "#run-2", "#lab-run"]) $(id).disabled = true;
}
boot().catch(showBootFailure);
