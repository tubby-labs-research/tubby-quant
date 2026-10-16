"use strict";
/* Tubby Qlib Console — SVG charts drawn from theme tokens.
   Line charts (linked crosshair, drag-to-zoom), multi-series overlays with legend and end labels,
   bar charts, a calendar heatmap table and the range navigator used for backtest windows. */

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const SERIES = (i) => css(`--cat-${(i % 6) + 1}`) || "#3987e5";

function niceStep(range, count) {
  const raw = range / Math.max(1, count), power = Math.pow(10, Math.floor(Math.log10(raw || 1))), unit = raw / power;
  return (unit <= 1 ? 1 : unit <= 2 ? 2 : unit <= 2.5 ? 2.5 : unit <= 5 ? 5 : 10) * power;
}
function niceTicks(lo, hi, count = 4) {
  if (!(hi > lo)) { const pad = Math.abs(lo || 1) * 0.1; lo -= pad; hi += pad; }
  const step = niceStep(hi - lo, count), ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) ticks.push(+v.toFixed(12));
  return ticks.length ? ticks : [lo, hi];
}
function logTicks(lo, hi) {
  const ticks = [];
  for (let p = Math.floor(Math.log10(lo)); p <= Math.ceil(Math.log10(hi)); p++) {
    for (const m of [1, 2, 5]) { const v = m * Math.pow(10, p); if (v >= lo && v <= hi) ticks.push(v); }
  }
  return ticks.length >= 2 ? ticks : [lo, hi];
}
function timeTicks(dates, width) {
  // year starts, or month starts for windows shorter than about two years
  const n = dates.length;
  if (n < 2) return [];
  const years = [];
  for (let i = 1; i < n; i++) if (dates[i].slice(0, 4) !== dates[i - 1].slice(0, 4)) years.push(i);
  const room = Math.max(2, Math.floor(width / 64));
  if (years.length >= 2) {
    const every = Math.max(1, Math.ceil(years.length / room));
    return years.filter((_, k) => k % every === 0).map((i) => ({ i, label: dates[i].slice(0, 4) }));
  }
  const months = [];
  for (let i = 1; i < n; i++) if (dates[i].slice(0, 7) !== dates[i - 1].slice(0, 7)) months.push(i);
  const every = Math.max(1, Math.ceil(months.length / room));
  return months.filter((_, k) => k % every === 0).map((i) => ({ i, label: dates[i].slice(0, 7) }));
}

// Draw now when the container has a size; otherwise retry on the next frames (hidden or not yet laid out).
function drawWhenSized(container, draw, attempts = 30) {
  if (container && container.isConnected && container.clientWidth > 0) { draw(); return; }
  if (attempts > 0) requestAnimationFrame(() => drawWhenSized(container, draw, attempts - 1));
}

/* A group of charts sharing one crosshair (by date). */
function crosshairGroup() {
  const members = [];
  return {
    add(c) { if (c) members.push(c); return c; },
    show(date) { members.forEach((c) => c.showDate(date)); },
    hide() { members.forEach((c) => c.hide()); },
  };
}

/* Single-series line/area chart.
   cfg: dates, values, format, label, kind ("dd" for drawdown), baseline, log, axisX, tooltip(i) -> [[k, v, cls]],
        group (crosshairGroup), onBrush(i0, i1) (drag to select), onReset() (double-click). */
function lineChart(container, cfg) {
  releaseChart(container);
  container.innerHTML = "";
  container.classList.toggle("dd", cfg.kind === "dd");
  const dates = cfg.dates || [], values = cfg.values || [];
  if (values.filter(finite).length < 2) { container.append(el("div", { class: "chart-empty", text: cfg.empty || "Not enough data" })); return null; }
  const width = Math.max(200, container.clientWidth), height = Math.max(70, container.clientHeight);
  const pad = { l: 58, r: cfg.endLabel === false ? 12 : 64, t: 8, b: cfg.axisX === false ? 6 : 20 };
  const w = width - pad.l - pad.r, h = height - pad.t - pad.b;
  const good = values.filter(finite);
  const log = cfg.log && good.every((v) => v > 0);
  const tf = log ? Math.log10 : (v) => v;
  let lo = Math.min(...good), hi = Math.max(...good);
  if (cfg.baseline !== undefined) { lo = Math.min(lo, cfg.baseline); hi = Math.max(hi, cfg.baseline); }
  if (cfg.kind === "dd") { hi = 0; lo = Math.min(lo, -0.01); }
  let ticks;
  if (log) { const span = hi / lo; lo /= Math.pow(span, 0.04); hi *= Math.pow(span, 0.04); ticks = logTicks(lo, hi); }
  else { ticks = niceTicks(lo, hi, cfg.kind === "dd" ? 3 : 4); lo = Math.min(lo, ticks[0]); hi = Math.max(hi, ticks[ticks.length - 1]); }
  const last = values.length - 1;
  const x = (i) => pad.l + (i / Math.max(1, last)) * w;
  const y = (v) => pad.t + (1 - (tf(v) - tf(lo)) / (tf(hi) - tf(lo) || 1)) * h;
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": cfg.label || "chart" });
  for (const t of ticks) {
    const yy = Math.round(y(t)) + 0.5;
    svg.append(svgEl("line", { x1: pad.l, x2: pad.l + w, y1: yy, y2: yy, class: "gridline" }));
    svg.append(svgEl("text", { x: pad.l - 8, y: yy + 3.5, "text-anchor": "end", class: "axis" }, cfg.format(t)));
  }
  const base = cfg.kind === "dd" ? 0 : cfg.baseline;
  if (base !== undefined) { const yy = Math.round(y(base)) + 0.5; svg.append(svgEl("line", { x1: pad.l, x2: pad.l + w, y1: yy, y2: yy, class: "zero" })); }
  if (cfg.axisX !== false) for (const tick of timeTicks(dates, w)) svg.append(svgEl("text", { x: x(tick.i), y: height - 5, "text-anchor": "middle", class: "axis" }, tick.label));
  const segments = [];
  let current = [];
  values.forEach((v, i) => { if (finite(v)) current.push(`${x(i).toFixed(1)},${y(v).toFixed(1)}`); else if (current.length) { segments.push(current); current = []; } });
  if (current.length) segments.push(current);
  const floor = cfg.kind === "dd" ? y(0) : pad.t + h;
  for (const seg of segments) {
    const first = seg[0].split(",")[0], end = seg[seg.length - 1].split(",")[0];
    if (cfg.wash !== false) svg.append(svgEl("polygon", { points: `${first},${floor} ${seg.join(" ")} ${end},${floor}`, class: "wash" }));
    svg.append(svgEl("polyline", { points: seg.join(" "), class: "line" }));
  }
  let lastGood = last; while (lastGood > 0 && !finite(values[lastGood])) lastGood--;
  svg.append(svgEl("circle", { cx: x(lastGood), cy: y(values[lastGood]), r: 3.5, class: "end-dot" }));
  if (cfg.endLabel !== false) svg.append(svgEl("text", { x: x(lastGood) + 8, y: y(values[lastGood]) + 4, class: "end-label" }, cfg.format(values[lastGood])));
  const cross = svgEl("line", { y1: pad.t, y2: pad.t + h, class: "cross", visibility: "hidden" });
  const dot = svgEl("circle", { r: 4, class: "hover-dot", visibility: "hidden" });
  const brush = svgEl("rect", { y: pad.t, height: h, class: "brush", visibility: "hidden" });
  svg.append(brush, cross, dot);
  const hit = svgEl("rect", { x: pad.l, y: 0, width: w, height, fill: "transparent", class: cfg.onBrush ? "brushable" : "" });
  svg.append(hit);
  container.append(svg);
  const tip = el("div", { class: "tooltip", hidden: true });
  container.append(tip);
  const indexOf = new Map(dates.map((d, i) => [d, i]));
  const api = {
    show(i) {
      if (i < 0 || i > last || !finite(values[i])) return api.hide();
      const xx = Math.round(x(i)) + 0.5;
      cross.setAttribute("x1", xx); cross.setAttribute("x2", xx); cross.setAttribute("visibility", "visible");
      dot.setAttribute("cx", x(i)); dot.setAttribute("cy", y(values[i])); dot.setAttribute("visibility", "visible");
      if (cfg.tooltip) {
        tip.hidden = false;
        setChildren(tip, el("div", { class: "t-date", text: dates[i] }),
          ...cfg.tooltip(i).map(([k, v, cls]) => el("div", { class: "t-row" }, el("span", { text: k }), el("b", { class: cls || "", text: v }))));
        const box = svg.getBoundingClientRect(), px = (x(i) / width) * box.width;
        const left = px + 14 + tip.offsetWidth > box.width ? px - 14 - tip.offsetWidth : px + 14;
        tip.style.left = `${Math.max(0, left)}px`; tip.style.top = "6px";
      }
    },
    showDate(date) { const i = indexOf.get(date); if (i === undefined) api.hide(); else api.show(i); },
    hide() { cross.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); tip.hidden = true; },
  };
  const indexAt = (event) => {
    const box = svg.getBoundingClientRect();
    const px = ((event.clientX - box.left) / box.width) * width;
    return Math.max(0, Math.min(last, Math.round(((px - pad.l) / w) * last)));
  };
  let drag = null;
  svg.addEventListener("mousemove", (event) => {
    const i = indexAt(event);
    if (drag !== null) {
      const a = Math.min(drag, i), b = Math.max(drag, i);
      brush.setAttribute("x", x(a)); brush.setAttribute("width", Math.max(1, x(b) - x(a))); brush.setAttribute("visibility", "visible");
    }
    if (cfg.group) cfg.group.show(dates[i]); else api.show(i);
  });
  svg.addEventListener("mouseleave", () => { if (cfg.group) cfg.group.hide(); else api.hide(); });
  if (cfg.onBrush) {
    svg.addEventListener("mousedown", (event) => { if (event.button === 0) { drag = indexAt(event); event.preventDefault(); } });
    const up = (event) => {
      if (drag === null) return;
      const i = indexAt(event), a = Math.min(drag, i), b = Math.max(drag, i);
      drag = null; brush.setAttribute("visibility", "hidden");
      if (b - a >= 5) cfg.onBrush(a, b);
    };
    window.addEventListener("mouseup", up, { once: false });
    container._cleanup = () => window.removeEventListener("mouseup", up);
    svg.addEventListener("dblclick", () => cfg.onReset && cfg.onReset());
  }
  return api;
}

/* Several series on one date axis (union of dates). series: [{name, dates, values, color, dash}].
   Legend always shown; direct end labels for up to four series. */
function multiLineChart(container, cfg) {
  container.innerHTML = "";
  const series = (cfg.series || []).filter((s) => s.values && s.values.filter(finite).length >= 2);
  if (!series.length) { container.append(el("div", { class: "chart-empty", text: cfg.empty || "Not enough data" })); return null; }
  const all = Array.from(new Set(series.flatMap((s) => s.dates))).sort();
  const pos = new Map(all.map((d, i) => [d, i]));
  const width = Math.max(200, container.clientWidth), height = Math.max(90, container.clientHeight);
  const labels = series.length <= 4 && cfg.endLabels !== false;
  const pad = { l: 58, r: labels ? 118 : 14, t: 8, b: 20 };
  const w = width - pad.l - pad.r, h = height - pad.t - pad.b;
  const good = series.flatMap((s) => s.values.filter(finite));
  const log = cfg.log && good.every((v) => v > 0);
  const tf = log ? Math.log10 : (v) => v;
  let lo = Math.min(...good), hi = Math.max(...good);
  if (cfg.baseline !== undefined) { lo = Math.min(lo, cfg.baseline); hi = Math.max(hi, cfg.baseline); }
  let ticks;
  if (log) { const span = hi / lo; lo /= Math.pow(span, 0.04); hi *= Math.pow(span, 0.04); ticks = logTicks(lo, hi); }
  else { ticks = niceTicks(lo, hi, 4); lo = Math.min(lo, ticks[0]); hi = Math.max(hi, ticks[ticks.length - 1]); }
  const x = (i) => pad.l + (i / Math.max(1, all.length - 1)) * w;
  const y = (v) => pad.t + (1 - (tf(v) - tf(lo)) / (tf(hi) - tf(lo) || 1)) * h;
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": cfg.label || "chart" });
  for (const t of ticks) {
    const yy = Math.round(y(t)) + 0.5;
    svg.append(svgEl("line", { x1: pad.l, x2: pad.l + w, y1: yy, y2: yy, class: "gridline" }));
    svg.append(svgEl("text", { x: pad.l - 8, y: yy + 3.5, "text-anchor": "end", class: "axis" }, cfg.format(t)));
  }
  if (cfg.baseline !== undefined) { const yy = Math.round(y(cfg.baseline)) + 0.5; svg.append(svgEl("line", { x1: pad.l, x2: pad.l + w, y1: yy, y2: yy, class: "zero" })); }
  for (const tick of timeTicks(all, w)) svg.append(svgEl("text", { x: x(tick.i), y: height - 5, "text-anchor": "middle", class: "axis" }, tick.label));
  const ends = [];
  series.forEach((s, k) => {
    const color = s.color || SERIES(k);
    const pts = s.dates.map((d, i) => finite(s.values[i]) ? `${x(pos.get(d)).toFixed(1)},${y(s.values[i]).toFixed(1)}` : null).filter(Boolean);
    svg.append(svgEl("polyline", { points: pts.join(" "), class: "mline", stroke: color, "stroke-dasharray": s.dash || null }));
    let j = s.values.length - 1; while (j > 0 && !finite(s.values[j])) j--;
    ends.push({ x: x(pos.get(s.dates[j])), y: y(s.values[j]), color, text: `${s.short || s.name} ${cfg.format(s.values[j])}` });
  });
  if (labels) {  // nudge end labels apart (12px), keeping order
    ends.sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 13) ends[i].y = ends[i - 1].y + 13;
    for (const e of ends) {
      svg.append(svgEl("circle", { cx: e.x, cy: e.y, r: 3, fill: e.color }));
      svg.append(svgEl("text", { x: pad.l + w + 8, y: e.y + 4, class: "end-label" }, e.text.length > 17 ? e.text.slice(0, 16) + "…" : e.text));
    }
  }
  const cross = svgEl("line", { y1: pad.t, y2: pad.t + h, class: "cross", visibility: "hidden" });
  svg.append(cross, svgEl("rect", { x: pad.l, y: 0, width: w, height, fill: "transparent" }));
  container.append(svg);
  const legend = el("div", { class: "legend" }, series.map((s, k) => el("span", { class: "legend-item" },
    el("i", { style: { background: s.color || SERIES(k) }, class: s.dash ? "dashed" : "" }), el("span", { text: s.name, title: s.title || s.name }))));
  container.append(legend);
  const tip = el("div", { class: "tooltip", hidden: true });
  container.append(tip);
  const lookup = series.map((s) => new Map(s.dates.map((d, i) => [d, s.values[i]])));
  svg.addEventListener("mousemove", (event) => {
    const box = svg.getBoundingClientRect(), px = ((event.clientX - box.left) / box.width) * width;
    const i = Math.max(0, Math.min(all.length - 1, Math.round(((px - pad.l) / w) * (all.length - 1))));
    const xx = Math.round(x(i)) + 0.5;
    cross.setAttribute("x1", xx); cross.setAttribute("x2", xx); cross.setAttribute("visibility", "visible");
    tip.hidden = false;
    setChildren(tip, el("div", { class: "t-date", text: all[i] }), ...series.map((s, k) => {
      const v = lookup[k].get(all[i]);
      return el("div", { class: "t-row" }, el("span", {}, el("i", { class: "t-swatch", style: { background: s.color || SERIES(k) } }), s.short || s.name),
        el("b", { text: finite(v) ? cfg.format(v) : "—" }));
    }));
    const tx = (x(i) / width) * box.width;
    tip.style.left = `${Math.max(0, tx + 14 + tip.offsetWidth > box.width ? tx - 14 - tip.offsetWidth : tx + 14)}px`; tip.style.top = "6px";
  });
  svg.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); tip.hidden = true; });
  return {};
}

/* Vertical bars. cfg: labels, values, format, color(v, i) -> css color, tooltip(i), dense (no gaps, for daily/monthly bars),
   line: optional overlay values (e.g. a rolling mean) drawn on the same scale. */
function barChart(container, cfg) {
  container.innerHTML = "";
  const labels = cfg.labels || [], values = cfg.values || [];
  if (!values.filter(finite).length) { container.append(el("div", { class: "chart-empty", text: cfg.empty || "No data" })); return; }
  const width = Math.max(200, container.clientWidth), height = Math.max(70, container.clientHeight);
  const pad = { l: 58, r: 12, t: cfg.valueLabels ? 16 : 8, b: 20 };
  const w = width - pad.l - pad.r, h = height - pad.t - pad.b;
  const pool = values.filter(finite).concat((cfg.line || []).filter(finite));
  const top = Math.max(0, ...pool), bottom = Math.min(0, ...pool), room = cfg.valueLabels ? (top - bottom) * 0.12 : 0;
  const ticks = niceTicks(bottom - (bottom < 0 ? room : 0), top + (top > 0 ? room : 0), 4);
  const lo = ticks[0], hi = ticks[ticks.length - 1];
  const y = (v) => pad.t + (1 - (v - lo) / (hi - lo || 1)) * h;
  const slot = w / values.length, bw = cfg.dense ? Math.max(0.6, slot - (slot > 3 ? 0.6 : 0)) : Math.max(2, Math.min(42, slot * 0.68));
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": cfg.label || "bar chart" });
  for (const t of ticks) {
    const yy = Math.round(y(t)) + 0.5;
    svg.append(svgEl("line", { x1: pad.l, x2: pad.l + w, y1: yy, y2: yy, class: "gridline" }));
    svg.append(svgEl("text", { x: pad.l - 8, y: yy + 3.5, "text-anchor": "end", class: "axis" }, cfg.format(t)));
  }
  const zero = Math.round(y(0)) + 0.5;
  values.forEach((v, i) => {
    if (!finite(v)) return;
    const top = Math.min(y(v), y(0)), bh = Math.max(0.8, Math.abs(y(v) - y(0)));
    const r = svgEl("rect", { x: pad.l + i * slot + (slot - bw) / 2, y: top, width: bw, height: bh, rx: cfg.dense ? 0 : 2,
      fill: cfg.color ? cfg.color(v, i) : SERIES(0), class: "bar" });
    if (cfg.tooltip) r.append(svgEl("title", {}, cfg.tooltip(i)));
    svg.append(r);
    if (cfg.valueLabels) svg.append(svgEl("text", { x: pad.l + i * slot + slot / 2, y: v >= 0 ? top - 4 : top + bh + 11, "text-anchor": "middle", class: "bar-label" }, cfg.format(v)));
  });
  svg.append(svgEl("line", { x1: pad.l, x2: pad.l + w, y1: zero, y2: zero, class: "zero" }));
  if (cfg.line) {
    const pts = cfg.line.map((v, i) => finite(v) ? `${(pad.l + i * slot + slot / 2).toFixed(1)},${y(v).toFixed(1)}` : null).filter(Boolean);
    svg.append(svgEl("polyline", { points: pts.join(" "), class: "overlay-line" }));
  }
  const every = cfg.dense ? null : Math.max(1, Math.ceil(labels.length / Math.max(2, Math.floor(w / 48))));
  if (cfg.dense) {
    for (const tick of timeTicks(labels, w)) svg.append(svgEl("text", { x: pad.l + tick.i * slot + slot / 2, y: height - 5, "text-anchor": "middle", class: "axis" }, tick.label));
  } else {
    labels.forEach((label, i) => { if (i % every === 0) svg.append(svgEl("text", { x: pad.l + i * slot + slot / 2, y: height - 5, "text-anchor": "middle", class: "axis" }, String(label))); });
  }
  container.append(svg);
}

/* Diverging fill for signed values: CSS tokens --div-neg-1..5, --div-mid, --div-pos-1..5. */
function divergingColor(v, scale) {
  if (!finite(v)) return "transparent";
  const step = Math.min(5, Math.ceil((Math.abs(v) / (scale || 1)) * 5));
  if (step === 0 || Math.abs(v) < 1e-12) return css("--div-mid");
  return css(`--div-${v > 0 ? "pos" : "neg"}-${step}`);
}

/* Monthly returns table: rows of years, 12 month columns plus the year total. */
function monthlyHeatmap(container, table) {
  container.innerHTML = "";
  if (!table.years.length) { container.append(el("div", { class: "chart-empty", text: "Not enough data" })); return; }
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const cells = table.years.flatMap((y) => table.cells[y].filter(finite).map(Math.abs));
  cells.sort((a, b) => a - b);
  const scale = cells.length ? cells[Math.floor(cells.length * 0.9)] || cells[cells.length - 1] : 0.05;
  const totals = table.years.map((y) => Math.abs(table.totals[y] || 0)).sort((a, b) => a - b);
  const totalScale = totals[Math.floor(totals.length * 0.9)] || totals[totals.length - 1] || 0.2;
  const cell = (v, s) => el("td", { class: `hm-cell ${Math.min(5, Math.ceil((Math.abs(v) / s) * 5)) >= 4 ? "strong" : ""}`,
    style: { background: divergingColor(v, s) }, text: finite(v) ? (v * 100).toFixed(1) : "", title: finite(v) ? fmt.signedPct(v, 2) : "" });
  container.append(el("table", { class: "heatmap" },
    el("thead", {}, el("tr", {}, el("th", { text: "" }), ...months.map((m) => el("th", { text: m })), el("th", { class: "hm-total-h", text: "Year" }))),
    el("tbody", {}, ...table.years.map((year) => el("tr", {}, el("th", { text: year }),
      ...table.cells[year].map((v) => finite(v) ? cell(v, scale) : el("td", { class: "hm-cell empty" })),
      finite(table.totals[year]) ? cell(table.totals[year], totalScale) : el("td", { class: "hm-cell empty" }))))));
}

/* Range navigator: a track over `dates` with a draggable [i0, i1] selection.
   cfg: dates, values (optional sparkline), area (optional second, filled series), minIndex (earliest allowed start),
   markers [{i, label}], range [i0, i1], onChange(i0, i1, final), height. */
function rangeNavigator(container, cfg) {
  releaseChart(container);
  container.innerHTML = "";
  const dates = cfg.dates || [];
  const n = dates.length;
  if (n < 3) { container.append(el("div", { class: "chart-empty", text: "No sessions" })); return null; }
  const width = Math.max(240, container.clientWidth), height = cfg.height || Math.max(40, container.clientHeight);
  const pad = { l: 8, r: 8, t: 4, b: 16 };
  const w = width - pad.l - pad.r, h = height - pad.t - pad.b;
  const x = (i) => pad.l + (i / (n - 1)) * w;
  const minIndex = Math.max(0, cfg.minIndex || 0), maxIndex = Math.min(n - 1, cfg.maxIndex ?? n - 1);
  let [i0, i1] = cfg.range || [minIndex, maxIndex];
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, class: "nav-svg", role: "slider", "aria-label": cfg.label || "Window" });
  svg.append(svgEl("rect", { x: pad.l, y: pad.t, width: w, height: h, rx: 5, class: "nav-track" }));
  const drawSeries = (vals, cls) => {
    const good = vals.filter(finite);
    if (good.length < 2) return;
    const lo = Math.min(...good), hi = Math.max(...good);
    const yy = (v) => pad.t + 3 + (1 - (v - lo) / (hi - lo || 1)) * (h - 6);
    const step = Math.max(1, Math.floor(vals.length / (w * 1.5)));
    const pts = [];
    for (let i = 0; i < vals.length; i += step) if (finite(vals[i])) pts.push(`${x(i).toFixed(1)},${yy(vals[i]).toFixed(1)}`);
    if (cls === "nav-area") svg.append(svgEl("polygon", { points: `${x(0)},${pad.t + h} ${pts.join(" ")} ${x(vals.length - 1)},${pad.t + h}`, class: cls }));
    else svg.append(svgEl("polyline", { points: pts.join(" "), class: cls }));
  };
  if (cfg.area) drawSeries(cfg.area, "nav-area");
  if (cfg.values) drawSeries(cfg.values, "nav-line");
  if (minIndex > 0) {
    svg.append(svgEl("rect", { x: pad.l, y: pad.t, width: x(minIndex) - pad.l, height: h, class: "nav-blocked" }));
  }
  for (const m of cfg.markers || []) {
    const mx = Math.round(x(m.i)) + 0.5;
    svg.append(svgEl("line", { x1: mx, x2: mx, y1: pad.t, y2: pad.t + h, class: "nav-marker" }));
    if (m.label) svg.append(svgEl("text", { x: mx + 4, y: pad.t + 11, class: "nav-marker-label" }, m.label));
  }
  const shadeL = svgEl("rect", { y: pad.t, height: h, class: "nav-shade" });
  const shadeR = svgEl("rect", { y: pad.t, height: h, class: "nav-shade" });
  const sel = svgEl("rect", { y: pad.t, height: h, class: "nav-sel" });
  const hL = svgEl("g", { class: "nav-handle", tabindex: "0", role: "slider", "aria-label": "Window start" });
  const hR = svgEl("g", { class: "nav-handle", tabindex: "0", role: "slider", "aria-label": "Window end" });
  for (const g of [hL, hR]) {
    g.append(svgEl("rect", { y: pad.t - 2, width: 9, height: h + 4, rx: 3, x: -4.5, class: "nav-grip" }));
    g.append(svgEl("line", { x1: -1.2, x2: -1.2, y1: pad.t + h / 2 - 5, y2: pad.t + h / 2 + 5, class: "nav-grip-line" }));
    g.append(svgEl("line", { x1: 1.2, x2: 1.2, y1: pad.t + h / 2 - 5, y2: pad.t + h / 2 + 5, class: "nav-grip-line" }));
  }
  svg.append(shadeL, shadeR, sel, hL, hR);
  for (const tick of timeTicks(dates, w)) svg.append(svgEl("text", { x: x(tick.i), y: height - 3, "text-anchor": "middle", class: "axis" }, tick.label));
  container.append(svg);
  const place = () => {
    shadeL.setAttribute("x", pad.l); shadeL.setAttribute("width", Math.max(0, x(i0) - pad.l));
    shadeR.setAttribute("x", x(i1)); shadeR.setAttribute("width", Math.max(0, pad.l + w - x(i1)));
    sel.setAttribute("x", x(i0)); sel.setAttribute("width", Math.max(1, x(i1) - x(i0)));
    hL.setAttribute("transform", `translate(${x(i0)},0)`); hR.setAttribute("transform", `translate(${x(i1)},0)`);
    hL.setAttribute("aria-valuetext", dates[i0]); hR.setAttribute("aria-valuetext", dates[i1]);
  };
  place();
  const clamp = (a, b) => {
    a = Math.max(minIndex, Math.min(a, maxIndex - 1)); b = Math.max(a + 1, Math.min(b, maxIndex));
    return [a, b];
  };
  const indexAt = (event) => {
    const box = svg.getBoundingClientRect();
    return Math.round(((((event.clientX - box.left) / box.width) * width - pad.l) / w) * (n - 1));
  };
  const emit = (final) => cfg.onChange && cfg.onChange(i0, i1, final);
  let mode = null, anchor = 0, start0 = 0, end0 = 0;
  svg.addEventListener("mousedown", (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    const i = indexAt(event);
    if (event.target.closest(".nav-handle") === hL) mode = "l";
    else if (event.target.closest(".nav-handle") === hR) mode = "r";
    else if (i > i0 && i < i1) { mode = "move"; anchor = i; start0 = i0; end0 = i1; }
    else { mode = "new"; anchor = Math.max(minIndex, Math.min(i, maxIndex)); [i0, i1] = clamp(anchor, anchor + 1); place(); }
    svg.classList.add("dragging");
  });
  const move = (event) => {
    if (!mode) return;
    const i = indexAt(event);
    if (mode === "l") [i0, i1] = clamp(Math.min(i, i1 - 1), i1);
    else if (mode === "r") [i0, i1] = clamp(i0, Math.max(i, i0 + 1));
    else if (mode === "move") {
      const len = end0 - start0, shift = Math.max(minIndex - start0, Math.min(maxIndex - end0, i - anchor));
      i0 = start0 + shift; i1 = i0 + len;
    } else [i0, i1] = clamp(Math.min(anchor, i), Math.max(anchor, i));
    place(); emit(false);
  };
  const up = () => { if (!mode) return; mode = null; svg.classList.remove("dragging"); emit(true); };
  window.addEventListener("mousemove", move);
  window.addEventListener("mouseup", up);
  container._cleanup = () => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); };
  svg.addEventListener("dblclick", () => { [i0, i1] = [minIndex, maxIndex]; place(); emit(true); });
  const key = (which) => (event) => {
    const big = event.shiftKey ? 21 : 1;
    const delta = event.key === "ArrowLeft" ? -big : event.key === "ArrowRight" ? big : 0;
    if (!delta) return;
    event.preventDefault();
    if (which === "l") [i0, i1] = clamp(Math.min(i0 + delta, i1 - 1), i1); else [i0, i1] = clamp(i0, Math.max(i1 + delta, i0 + 1));
    place(); emit(true);
  };
  hL.addEventListener("keydown", key("l")); hR.addEventListener("keydown", key("r"));
  return { set(a, b) { [i0, i1] = clamp(a, b); place(); } };
}

/* Release window listeners held by a chart container before it is redrawn or removed. */
function releaseChart(container) { if (container && container._cleanup) { container._cleanup(); container._cleanup = null; } }
