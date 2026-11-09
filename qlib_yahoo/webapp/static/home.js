"use strict";
/* Start view: a guided first session. Three steps from an empty data folder to a tested idea,
   then a map of the workbench. Built from the console's own primitives; no new transport. */
const homeUI = { campaigns: null, loading: false };
const HOME_STARTERS = [
  "-rank(ts_delta(close, 5))",
  "rank(log(volume) - ts_mean(log(volume), 20))",
  "-rank(ts_std_dev(returns, 20))",
];
const HOME_MAP = [
  ["lab", "Alpha Lab", "⌘2", "Fast BRAIN-style simulation of many expressions at once; results are kept in your alpha library."],
  ["research", "Validation", "⌘1", "A careful Qlib long/short backtest of one expression: costs, borrow, next-open fills, a full account ledger."],
  ["discover", "Discover", "⌘5", "Autonomous alpha discovery and machine-learning campaigns under a budget, with a trial ledger and a single-use holdout."],
  ["data", "Data", "⌘3", "Datasets, downloads, imports and quality scans. Every dataset is a dated, sealed snapshot."],
  ["evidence", "Evidence", "", "What was tried, when and with which settings; validation of saved candidates and portfolios."],
  ["auto", "Research", "⌘4", "Hypotheses from the literature, optionally proposed by an agent, tested under the same sealed periods."],
];

function homeIcon() {
  const svg = svgEl("svg", { viewBox: "0 0 20 20" });
  svg.append(svgEl("path", { d: "M3 9.5L10 3l7 6.5" }), svgEl("path", { d: "M5 8.5V17h10V8.5" }), svgEl("path", { d: "M8.5 17v-5h3v5" }));
  return svg;
}
function homeStatus() {
  const ready = state.datasets.filter(priceEligible);
  return { ready, fallback: defaultPriceRow() || ready[0] || null, library: state.lab.rows.length, campaigns: homeUI.campaigns };
}
function homeButton(text, action, cls = "") { return el("button", { class: `btn ${cls}`, type: "button", text, onclick: action }); }
function homeStep(number, title, status, done, body, actions, note) {
  return el("section", { class: `home-step${done ? " done" : ""}` },
    el("div", { class: "home-step-head" }, el("span", { class: "home-step-number", text: done ? "✓" : String(number) }), el("h2", { text: title })),
    el("div", { class: `home-status${done ? " done" : ""}`, text: status }),
    ...body, el("div", { class: "home-actions" }, ...actions), note ? el("p", { class: "home-note", text: note }) : null);
}

function initializeHome() {
  const button = el("button", { class: "act", "data-view": "home", title: "Start (⌘0)", "aria-label": "Start",
    onclick: () => { setView("home"); persist(); } }, homeIcon(), el("span", { class: "act-label", text: "Start" }));
  $(".activitybar").prepend(button);
  $("#explorer").append(el("div", { class: "xview", "data-view": "home" },
    el("div", { class: "pane-head" }, "Getting started"),
    el("div", { class: "home-checklist", id: "home-checklist" }),
    el("p", { class: "home-side-note", text: "Everything runs on this computer. Nothing is traded, no broker is connected and no result is investment advice." })));
  $("#center").prepend(el("div", { class: "cview home-workspace", "data-view": "home", id: "home-view" }));
  $("#inspector").append(el("div", { class: "iview", "data-view": "home" },
    el("div", { class: "pane-head" }, "This installation"),
    el("div", { class: "props home-facts", id: "home-facts" })));
}

async function homeRender() {
  if (state.view !== "home") return;
  const s = homeStatus();
  renderHomeChecklist(s);
  renderHomeFacts();
  const view = $("#home-view");
  const noData = !s.ready.length;
  const defaultName = s.fallback ? datasetTitle(s.fallback) : "none";
  setChildren(view,
    el("header", { class: "home-hero" },
      el("div", { class: "home-eyebrow", text: "Tubby Quant · local quant research workbench" }),
      el("h1", { text: "Start here" }),
      el("p", { text: "Three steps take you from an empty data folder to a tested idea. Each step opens the tool it belongs to, so the rest of the workbench is never a surprise." })),
    el("div", { class: "home-steps" },
      homeStep(1, "Get prices",
        noData ? "No prepared prices yet. Choose one of the three ways below." : `${s.ready.length} dataset${s.ready.length === 1 ? "" : "s"} ready · default: ${defaultName}`, !noData,
        [el("p", { text: "Alpha Lab, Validation and Discover all read the same sealed daily snapshots. The quickest start is artificial demonstration data; real research needs real prices." })],
        [homeButton("Create demonstration data", homeCreateDemo, noData ? "primary" : ""),
         homeButton("Download US daily prices (Yahoo)", homeDownload), homeButton("Import a table…", homeImport)],
        "Demonstration data is labelled artificial everywhere it appears: 200 fictional stocks over six years with three planted signals, so the examples have something to find. The Yahoo download builds a TOP3000 universe from every US common stock and takes about an hour."),
      homeStep(2, "Simulate an alpha",
        s.library ? `${fmt.int(s.library)} alpha${s.library === 1 ? "" : "s"} in your library` : "Nothing simulated yet.", s.library > 0,
        [el("p", { text: "Write expressions BRAIN-style: bare field names, one alpha per line, and a leading minus is fine." }),
         el("code", { class: "home-code", text: HOME_STARTERS.join("\n") })],
        [homeButton("Simulate these three now", () => homeOpenLab(true), !noData && !s.library ? "primary" : ""),
         homeButton("Open Alpha Lab", () => homeOpenLab(false))],
        "Each run is saved to the library with its settings and the dataset's identity. Promising alphas go on to Validation, which trades them through a full account with costs."),
      homeStep(3, "Let the machine search",
        s.campaigns === null ? "Loading campaigns…" : s.campaigns.length ? `${s.campaigns.length} campaign${s.campaigns.length === 1 ? "" : "s"} so far` : "No campaign yet.", !!(s.campaigns && s.campaigns.length),
        [el("p", { text: "Discover proposes ideas, screens them on development data only, validates a budgeted few, reviews them with a deflated Sharpe ratio and promotes survivors. Finalize then scores promoted ideas once on a reserved holdout. Machine learning searches training recipes the same way." })],
        [homeButton("Run a short demo campaign", () => homeDemoCampaign("alpha"), !noData && s.library && !(s.campaigns && s.campaigns.length) ? "primary" : ""),
         homeButton("Discover alphas", () => homeOpenDiscover("alpha")), homeButton("Search model recipes", () => homeOpenDiscover("ml"))],
        "Every trial lands in a hash-chained ledger, and the holdout can be read once per candidate. A campaign that promotes nothing is a result too.")),
    el("h2", { class: "home-map-title", text: "Where things are" }),
    el("div", { class: "home-map" }, ...HOME_MAP.map(([view, title, key, text]) => el("button", { class: "home-map-card", type: "button",
      onclick: () => { setView(view); persist(); } },
      el("div", { class: "home-map-head" }, el("b", { text: title }), key ? el("kbd", { text: key }) : null),
      el("p", { text })))),
    el("p", { class: "home-foot", text: "Research software only: it places no trades, connects to no broker and makes no recommendation. Keyboard help: press ? anywhere." }));
  if (homeUI.campaigns === null && !homeUI.loading) {
    homeUI.loading = true;
    try { homeUI.campaigns = await get("lab-campaigns"); } catch (e) { homeUI.campaigns = []; }
    finally { homeUI.loading = false; }
    if (state.view === "home") homeRender();
  }
}
function renderHomeChecklist(s) {
  const rows = [["Prices ready", s.ready.length > 0, s.ready.length ? datasetTitle(s.fallback) : "Create demonstration data or download real prices"],
    ["An alpha simulated", s.library > 0, s.library ? `${fmt.int(s.library)} in the library` : "Alpha Lab · ⌘2"],
    ["A campaign run", !!(s.campaigns && s.campaigns.length), s.campaigns && s.campaigns.length ? `${s.campaigns.length} campaign${s.campaigns.length === 1 ? "" : "s"}` : "Discover · ⌘5"]];
  setChildren($("#home-checklist"), ...rows.map(([title, done, sub]) => el("div", { class: `home-check${done ? " done" : ""}` },
    el("span", { class: "home-check-dot" }), el("div", {}, el("b", { text: title }), el("small", { text: sub })))));
}
function renderHomeFacts() {
  const o = state.overview || {};
  const fact = (label, value, action) => el("div", { class: "home-fact" }, el("span", { class: "home-fact-label", text: label }),
    action ? el("button", { class: "btn small ghost home-fact-value", text: value, onclick: action }) : el("span", { class: "home-fact-value", text: value }));
  setChildren($("#home-facts"),
    el("div", { class: "group-title", text: "Status" }),
    fact("Signed engine", o.signed && o.signed.available ? "Ready" : "Not installed"),
    fact("Data folder", home(o.data_root || ""), () => api("reveal", { path: "" }).catch((e) => toast("Cannot open", e.message, "bad"))),
    fact("Version", `${o.version || "local"} · Python ${o.python || ""}`),
    el("div", { class: "group-title", text: "Learn more" }),
    el("p", { class: "prop-note", text: "docs/CONSOLE_QUICKSTART.md walks through this screen; docs/LAB_DOMAINS.md explains Discover's rules; the console guide in docs describes every view." }),
    el("div", { class: "home-actions" }, homeButton("Keyboard shortcuts", showShortcuts, "small"), homeButton("Data folder", () => setView("data"), "small")));
}

async function homeCreateDemo() {
  if (state.job && state.job.state === "running") { toast("A task is already running", state.job.label); return; }
  try {
    startPolling(await api("demo-data", {}));
    toast("Creating demonstration data", "A few seconds. The dataset is artificial and labelled as such.");
  } catch (e) { toast("Cannot create demonstration data", e.message, "bad"); }
}
async function homeDownload() {
  const ok = await dialog({ title: "Download US daily prices from Yahoo?", confirm: "Start download",
    message: "Fetches every US common stock's daily history with split and dividend events, builds a table and a TOP3000 snapshot, then scans its quality. About an hour; an interrupted fetch resumes and nothing earlier is overwritten." });
  if (ok) { setView("data"); startRefresh(); }
}
function homeImport() { setView("data"); const details = $(".iview[data-view=data] details"); if (details) details.open = true; $("#table").focus(); }
function homeOpenLab(run) {
  // "Simulate these three now" runs exactly the three starters; "Open Alpha Lab" keeps whatever is in the editor.
  const editor = $("#lab-text"), starters = HOME_STARTERS.join("\n"), current = editor.value.trim();
  if (run || !current) {
    if (run && current && current !== starters) toast("Alpha Lab", "The editor now holds the three starter alphas; your previous expressions were replaced.", "");
    editor.value = starters; onLabInput();
  }
  setView("lab"); persist();
  if (run) runLab(); else $("#lab-text").focus();
}
function homeOpenDiscover(domain) { if (typeof discoverOpenNew === "function") discoverOpenNew(domain); else setView("discover"); }
function homeDemoCampaign(domain) { if (typeof discoverRunDemo === "function") discoverRunDemo(domain); }
function homePrimary() { const s = homeStatus(); if (!s.ready.length) homeCreateDemo(); else homeOpenLab(true); }
