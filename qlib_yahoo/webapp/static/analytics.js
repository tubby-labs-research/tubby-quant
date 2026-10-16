"use strict";
/* Tubby Qlib Console — performance statistics computed in the page from a run's daily net returns.
   The definitions follow qlib_yahoo/metrics.py (sample standard deviation, 252 sessions a year,
   CAGR from summed log returns, drawdown from an initial NAV of 1), so the full window reproduces the
   run's saved summary and any sub-window is measured the same way. */

const YEAR = 252;

function statsOf(returns, cashDaily = 0) {
  const r = returns.filter(finite);
  const n = r.length;
  const out = { observations: n };
  if (!n) return out;
  const mean = r.reduce((a, b) => a + b, 0) / n;
  let ss = 0, down = 0, m3 = 0, m4 = 0;
  for (const v of r) { const d = v - mean; ss += d * d; m3 += d * d * d; m4 += d * d * d * d; if (v < 0) down += v * v; }
  const std = n > 1 ? Math.sqrt(ss / (n - 1)) : null;
  let log = 0, peak = 0, maxDD = 0, longest = 0, current = 0;
  for (const v of r) {
    log += Math.log1p(v);
    if (log >= peak) { peak = log; current = 0; } else { current += 1; longest = Math.max(longest, current); }
    maxDD = Math.min(maxDD, Math.expm1(log - peak));
  }
  const downside = Math.sqrt(down / n);
  const up = r.filter((v) => v > 0), dn = r.filter((v) => v < 0);
  const variance = ss / n;
  Object.assign(out, {
    mean, std, vol: std !== null ? std * Math.sqrt(YEAR) : null,
    sharpe: std ? ((mean - cashDaily) / std) * Math.sqrt(YEAR) : null,
    sortino: downside ? ((mean - cashDaily) / downside) * Math.sqrt(YEAR) : null,
    total: Math.expm1(log), cagr: Math.expm1((log * YEAR) / n), maxDD,
    hitRate: up.length / n, best: Math.max(...r), worst: Math.min(...r),
    avgUp: up.length ? up.reduce((a, b) => a + b, 0) / up.length : null,
    avgDown: dn.length ? dn.reduce((a, b) => a + b, 0) / dn.length : null,
    skew: variance > 0 ? (m3 / n) / Math.pow(variance, 1.5) : null,
    kurtosis: variance > 0 ? (m4 / n) / (variance * variance) - 3 : null,
    longestDD: longest,
  });
  out.calmar = out.maxDD < 0 ? out.cagr / Math.abs(out.maxDD) : null;
  return out;
}

/* NAV path from returns (starts at `start` before the first return) and its drawdown. */
function navOf(returns, start = 1) {
  const nav = [];
  let v = start;
  for (const r of returns) { v *= 1 + (finite(r) ? r : 0); nav.push(v); }
  return nav;
}
function drawdownOf(nav, openingNAV = nav[0]) {
  // Closing observations do not contain the capital at risk before session one.
  let peak = openingNAV;
  return nav.map((v) => { peak = Math.max(peak, v); return peak > 0 ? v / peak - 1 : 0; });
}

/* Compounded returns per calendar month and year. */
function monthlyTable(dates, returns) {
  const cells = {}, totals = {}, logs = {}, yearLogs = {};
  dates.forEach((d, i) => {
    const r = returns[i];
    if (!finite(r)) return;
    const year = d.slice(0, 4), month = +d.slice(5, 7) - 1;
    if (!logs[year]) { logs[year] = new Array(12).fill(null); yearLogs[year] = 0; }
    logs[year][month] = (logs[year][month] || 0) + Math.log1p(r);
    yearLogs[year] += Math.log1p(r);
  });
  const years = Object.keys(logs).sort().reverse();
  for (const y of years) { cells[y] = logs[y].map((v) => (v === null ? null : Math.expm1(v))); totals[y] = Math.expm1(yearLogs[y]); }
  return { years, cells, totals };
}

function annualStats(dates, returns, cashDaily = 0) {
  const byYear = new Map();
  dates.forEach((d, i) => { const y = d.slice(0, 4); if (!byYear.has(y)) byYear.set(y, []); byYear.get(y).push(returns[i]); });
  return Array.from(byYear, ([year, r]) => ({ year, ...statsOf(r, cashDaily) }));
}

/* Rolling annualized Sharpe over `window` sessions (null until the window fills). */
function rollingSharpe(returns, window = 126) {
  const out = new Array(returns.length).fill(null);
  let sum = 0, sq = 0, count = 0;
  const q = [];
  for (let i = 0; i < returns.length; i++) {
    const v = finite(returns[i]) ? returns[i] : 0;
    q.push(v); sum += v; sq += v * v; count++;
    if (q.length > window) { const old = q.shift(); sum -= old; sq -= old * old; count--; }
    if (count === window) {
      const mean = sum / count, varr = (sq - count * mean * mean) / (count - 1);
      out[i] = varr > 0 ? (mean / Math.sqrt(varr)) * Math.sqrt(YEAR) : null;
    }
  }
  return out;
}

/* The deepest drawdown episodes: start (peak), trough, recovery (or open), depth and length in sessions. */
function drawdownEpisodes(dates, nav, top = 5, openingNAV = nav[0], openingDate = "Opening NAV") {
  const episodes = [];
  let peak = openingNAV, peakAt = -1, trough = openingNAV, troughAt = -1, inDD = false;
  const peakDate = () => peakAt < 0 ? openingDate : dates[peakAt];
  for (let i = 0; i < nav.length; i++) {
    if (nav[i] >= peak) {
      if (inDD) episodes.push({ start: peakDate(), trough: dates[troughAt], end: dates[i], depth: trough / peak - 1, length: i - peakAt, recovered: true });
      peak = nav[i]; peakAt = i; trough = nav[i]; troughAt = i; inDD = false;
    } else {
      inDD = true;
      if (nav[i] < trough) { trough = nav[i]; troughAt = i; }
    }
  }
  if (inDD) episodes.push({ start: peakDate(), trough: dates[troughAt], end: null, depth: trough / peak - 1, length: nav.length - 1 - peakAt, recovered: false });
  return episodes.sort((a, b) => a.depth - b.depth).slice(0, top);
}

/* Pearson correlation of two date-aligned return series (dates as keys). */
function correlationOf(a, b) {
  const map = new Map(a.dates.map((d, i) => [d, a.values[i]]));
  const xs = [], ys = [];
  b.dates.forEach((d, i) => { const v = map.get(d); if (finite(v) && finite(b.values[i])) { xs.push(v); ys.push(b.values[i]); } });
  const n = xs.length;
  if (n < 20) return null;
  const mx = xs.reduce((s, v) => s + v, 0) / n, my = ys.reduce((s, v) => s + v, 0) / n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; syy += (ys[i] - my) ** 2; }
  return sxx && syy ? sxy / Math.sqrt(sxx * syy) : null;
}

/* Mean of values per calendar month (for monthly IC bars) and a trailing mean of those months. */
function monthlyMeans(dates, values) {
  const months = [], means = [];
  let key = null, sum = 0, count = 0;
  dates.forEach((d, i) => {
    const m = d.slice(0, 7);
    if (m !== key && key !== null) { months.push(key); means.push(count ? sum / count : null); sum = 0; count = 0; }
    key = m;
    if (finite(values[i])) { sum += values[i]; count++; }
  });
  if (key !== null) { months.push(key); means.push(count ? sum / count : null); }
  return { months, means };
}
function trailingMean(values, window) {
  return values.map((_, i) => {
    if (i + 1 < window) return null;
    const part = values.slice(i + 1 - window, i + 1).filter(finite);
    return part.length ? part.reduce((a, b) => a + b, 0) / part.length : null;
  });
}
