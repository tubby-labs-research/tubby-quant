"use strict";
/* Tubby Qlib Console — shared utilities: DOM helpers, formatting, the loopback API, toasts,
   context menus and dialogs. Loaded before charts.js and app.js. */

const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => Array.from(root.querySelectorAll(s));
const SVG = "http://www.w3.org/2000/svg";

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "html") node.innerHTML = value;
    else if (key === "style" && typeof value === "object") Object.assign(node.style, value);
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) if (child !== null && child !== undefined && child !== false) node.append(child);
  return node;
}
/* Replace a node's children, skipping null/false entries (the DOM would print them as text). */
function setChildren(node, ...children) {
  node.replaceChildren(...children.flat(Infinity).filter((c) => c !== null && c !== undefined && c !== false));
  return node;
}
function svgEl(tag, attrs = {}, text) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}
function icon(name, cls = "") {
  const node = svgEl("svg", { viewBox: "0 0 16 16", class: `i ${cls}`, "aria-hidden": "true" });
  node.innerHTML = ICONS[name] || "";
  return node;
}
const ICONS = {
  star: '<path d="M8 1.8l1.9 3.9 4.3.6-3.1 3 .7 4.3L8 11.6l-3.8 2 .7-4.3-3.1-3 4.3-.6z"/>',
  trash: '<path d="M2.8 4.2h10.4M6.2 4.2V2.8h3.6v1.4M4.2 4.2l.6 9h6.4l.6-9"/>',
  restore: '<path d="M3 8a5 5 0 1 0 1.5-3.6M3 2.8v2.6h2.6"/>',
  copy: '<rect x="5.2" y="5.2" width="8" height="8" rx="1.5"/><path d="M10.8 5.2V3.6a1 1 0 0 0-1-1H3.6a1 1 0 0 0-1 1v6.2a1 1 0 0 0 1 1h1.6"/>',
  edit: '<path d="M10.6 2.8l2.6 2.6-7.6 7.6H3v-2.6z"/>',
  compare: '<path d="M2.5 12.5l3.5-4 3 2.5 4.5-6"/><path d="M2.5 9l3.5-2.5 3 1.5 4.5-4" opacity=".45"/>',
  play: '<path d="M4.5 3l8 5-8 5z"/>',
  folder: '<path d="M2 4.8a1.5 1.5 0 0 1 1.5-1.5h2.8l1.4 1.6h4.8a1.5 1.5 0 0 1 1.5 1.5v5.3a1.5 1.5 0 0 1-1.5 1.5h-9A1.5 1.5 0 0 1 2 11.7z"/>',
  load: '<path d="M8 2.5v8M4.8 7.5L8 10.7l3.2-3.2M3 13.5h10"/>',
  open: '<path d="M6 3H3.5v9.5H13V10M9 3h4v4M13 3L7.5 8.5"/>',
  filter: '<path d="M2.5 3.5h11L9.2 8.6v3.9l-2.4 1.1V8.6z"/>',
  window: '<rect x="2" y="3" width="12" height="10" rx="1.5"/><path d="M5.5 3v10M10.5 3v10" opacity=".5"/>',
  settings: '<circle cx="8" cy="8" r="2.2"/><path d="M8 1.8v1.6M8 12.6v1.6M1.8 8h1.6M12.6 8h1.6M3.6 3.6l1.1 1.1M11.3 11.3l1.1 1.1M3.6 12.4l1.1-1.1M11.3 4.7l1.1-1.1"/>',
  check: '<path d="M3 8.5l3 3 7-7"/>',
  chevron: '<path d="M6 4l4 4-4 4"/>',
  close: '<path d="M4 4l8 8M12 4l-8 8"/>',
};

// ----------------------------------------------------------------------------- platform wording
// localStorage "tqc.platform" = "windows" previews the Windows wording on a Mac (development aid).
const PLATFORM_NAME = (() => { try { const forced = localStorage.getItem("tqc.platform"); if (forced) return forced === "windows" ? "Win32" : forced; } catch (e) { /* storage unavailable */ }
  return navigator.platform || navigator.userAgent || ""; })();
const IS_MAC = /Mac|iPhone|iPad/.test(PLATFORM_NAME);
const IS_WINDOWS = /Win/.test(PLATFORM_NAME);
const KEY_SIGNS = /[\u2318\u232B\u2325]/;  // ⌘ ⌫ ⌥ as written in the page
/* Shortcut labels are written the macOS way (⌘R, ⌫); elsewhere they read Ctrl+R, Del. */
function platformText(text, mac = IS_MAC) {
  if (mac || !text || !KEY_SIGNS.test(text)) return text;
  return text.replace(/\u2318-/g, "Ctrl-").replace(/\u2318\//g, "Ctrl/").replace(/\u2318\s?/g, "Ctrl+")
    .replace(/\u232B/g, "Del").replace(/\u2325/g, "Alt+");
}
const LABEL_ATTRIBUTES = ["title", "placeholder", "aria-label"];
function localizeKeys(root, mac = IS_MAC) {
  if (mac || !root) return;
  if (root.nodeType === Node.TEXT_NODE) { if (KEY_SIGNS.test(root.nodeValue)) root.nodeValue = platformText(root.nodeValue, mac); return; }
  if (root.nodeType !== Node.ELEMENT_NODE) return;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  for (let node = walker.nextNode(); node; node = walker.nextNode()) if (KEY_SIGNS.test(node.nodeValue)) node.nodeValue = platformText(node.nodeValue, mac);
  for (const element of [root, ...root.querySelectorAll("[title],[placeholder],[aria-label]")]) {
    for (const name of LABEL_ATTRIBUTES) {
      const value = element.getAttribute(name);
      if (value && KEY_SIGNS.test(value)) element.setAttribute(name, platformText(value, mac));
    }
  }
}
if (!IS_MAC) {  // keep every label, including those drawn later, in this platform's words
  const observer = new MutationObserver((records) => {
    for (const record of records) {
      if (record.type === "characterData" || record.type === "attributes") localizeKeys(record.target);
      record.addedNodes.forEach((node) => localizeKeys(node));
    }
  });
  const start = () => {
    localizeKeys(document.body);
    observer.observe(document.body, { childList: true, subtree: true, characterData: true, attributes: true, attributeFilter: LABEL_ATTRIBUTES });
  };
  if (document.body) start(); else document.addEventListener("DOMContentLoaded", start);
}

const finite = (v) => typeof v === "number" && isFinite(v);
const fmt = {
  pct: (v, d = 1) => finite(v) ? `${(v * 100).toFixed(d)}%` : "—",
  signedPct: (v, d = 1) => finite(v) ? `${v > 0 ? "+" : ""}${(v * 100).toFixed(d)}%` : "—",
  bps: (v, d = 1) => finite(v) ? `${(v * 1e4).toFixed(d)} bp` : "—",
  num: (v, d = 2) => finite(v) ? Number(v).toFixed(d) : "—",
  signed: (v, d = 2) => finite(v) ? `${v > 0 ? "+" : ""}${Number(v).toFixed(d)}` : "—",
  int: (v) => finite(v) ? Math.round(v).toLocaleString("en-US") : "—",
  money: (v) => {
    if (!finite(v)) return "—";
    const a = Math.abs(v), s = v < 0 ? "-" : "";
    if (a >= 1e9) return `${s}$${(a / 1e9).toFixed(2)}B`;
    if (a >= 1e6) return `${s}$${(a / 1e6).toFixed(2)}M`;
    if (a >= 1e3) return `${s}$${(a / 1e3).toFixed(a >= 1e5 ? 0 : 1)}K`;
    return `${s}$${a.toFixed(0)}`;
  },
  clock: (seconds) => { const s = Math.max(0, Math.floor(seconds || 0)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; },
  bytes: (b) => !finite(b) ? "—" : b > 1e9 ? `${(b / 1e9).toFixed(2)} GB` : `${(b / 1e6).toFixed(1)} MB`,
  years: (sessions) => finite(sessions) ? `${(sessions / 252).toFixed(sessions < 252 ? 2 : 1)} yrs` : "—",
};
const tone = (v) => !finite(v) ? "" : v > 0 ? "pos" : v < 0 ? "neg" : "";
const home = (p) => (p || "").replace(/^\/Users\/[^/]+/, "~").replace(/^[A-Za-z]:\\Users\\[^\\]+/, "~");
const shortTime = (iso) => (iso || "").slice(5, 16).replace("T", " ");
const escapeHtml = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const session = { token: "" };
async function api(name, data, method) {
  const verb = method || (data === undefined ? "GET" : "POST");
  let url = `/api/${name}`;
  const options = { method: verb, headers: {} };
  if (verb === "GET" && data) url += "?" + new URLSearchParams(data).toString();
  if (verb === "POST") {
    options.headers["Content-Type"] = "application/json";
    options.headers["X-Token"] = session.token;
    options.body = JSON.stringify(data || {});
  }
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}
const get = (name, params) => api(name, params || {}, "GET");

function toast(title, message = "", kind = "", action = null) {
  const node = el("div", { class: `toast ${kind}`, role: kind === "bad" ? "alert" : "status" },
    el("div", { class: "toast-text" }, el("b", { text: title }), message ? el("div", { text: message }) : null),
    action ? el("button", { class: "toast-action", text: action.label, onclick: () => { node.remove(); action.run(); } }) : null);
  $("#toasts").append(node);
  setTimeout(() => node.remove(), action ? 9000 : kind === "bad" ? 9000 : 4500);
}

async function copyText(text, what = "Copied") {
  try { await navigator.clipboard.writeText(text); toast(what, text.length > 80 ? text.slice(0, 80) + "…" : text); }
  catch (e) {
    const area = el("textarea", { style: { position: "fixed", opacity: "0" } }); area.value = text; document.body.append(area);
    area.select(); document.execCommand("copy"); area.remove(); toast(what, text.slice(0, 80));
  }
}

// ----------------------------------------------------------------------------- context menu
let openMenu = null;
function closeContextMenu() { if (openMenu) { openMenu.remove(); openMenu = null; } }
/* items: [{label, icon, hint, danger, disabled, checked, action}] or "sep" or {header: text} */
function contextMenu(x, y, items) {
  closeContextMenu();
  const menu = el("div", { class: "ctx", role: "menu" });
  const buttons = [];
  for (const item of items) {
    if (!item) continue;
    if (item === "sep") { menu.append(el("div", { class: "ctx-sep" })); continue; }
    if (item.header) { menu.append(el("div", { class: "ctx-head", text: item.header })); continue; }
    const button = el("button", { class: `ctx-item${item.danger ? " danger" : ""}`, role: "menuitem", disabled: item.disabled,
      onclick: (e) => { e.stopPropagation(); closeContextMenu(); item.action && item.action(); } },
      el("span", { class: "ctx-icon" }, item.icon ? icon(item.icon, item.checked ? "filled" : "") : null),
      el("span", { class: "ctx-label", text: item.label }), item.hint ? el("span", { class: "ctx-hint", text: item.hint }) : null);
    buttons.push(button);
    menu.append(button);
  }
  document.body.append(menu);
  const box = menu.getBoundingClientRect();
  menu.style.left = `${Math.max(6, Math.min(x, window.innerWidth - box.width - 6))}px`;
  menu.style.top = `${Math.max(6, Math.min(y, window.innerHeight - box.height - 6))}px`;
  openMenu = menu;
  let index = -1;
  menu.tabIndex = -1; menu.focus();
  menu.addEventListener("keydown", (e) => {
    const enabled = buttons.filter((b) => !b.disabled);
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      index = (index + (e.key === "ArrowDown" ? 1 : -1) + enabled.length) % enabled.length;
      enabled[index].focus();
    } else if (e.key === "Escape") { e.preventDefault(); closeContextMenu(); }
  });
  return menu;
}
document.addEventListener("mousedown", (e) => { if (openMenu && !openMenu.contains(e.target)) closeContextMenu(); }, true);
window.addEventListener("blur", closeContextMenu);
window.addEventListener("resize", closeContextMenu);

// ----------------------------------------------------------------------------- dialogs
function dialog({ title, message = "", input = null, confirm = "OK", cancel = "Cancel", danger = false, detail = null }) {
  return new Promise((resolve) => {
    const field = input ? el("input", { type: "text", class: "dlg-input", value: input.value || "", placeholder: input.placeholder || "",
      maxlength: input.maxlength || 120, "aria-label": title }) : null;
    const done = (value) => { overlay.remove(); document.removeEventListener("keydown", keys, true); resolve(value); };
    const ok = el("button", { class: `btn ${danger ? "danger" : "primary"}`, text: confirm, onclick: () => done(field ? field.value : true) });
    const overlay = el("div", { class: "overlay dlg-overlay", onmousedown: (e) => { if (e.target === overlay) done(null); } },
      el("div", { class: "dlg", role: "dialog", "aria-modal": "true", "aria-label": title },
        el("div", { class: "dlg-title", text: title }), message ? el("div", { class: "dlg-message", text: message }) : null,
        detail ? el("div", { class: "dlg-detail" }, detail) : null, field,
        el("div", { class: "dlg-actions" }, cancel === null ? null : el("button", { class: "btn", text: cancel, onclick: () => done(null) }), ok)));
    const keys = (e) => {
      if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); done(null); }
      else if (e.key === "Enter" && (field ? document.activeElement === field : true)) { e.preventDefault(); e.stopPropagation(); done(field ? field.value : true); }
    };
    document.addEventListener("keydown", keys, true);
    document.body.append(overlay);
    (field || ok).focus();
    if (field) field.select();
  });
}
