// 왼쪽 조작 패널(슬라이더·프리셋·검색)과 오른쪽 격자 상세 패널

import * as store from "./state.js";
import { gridSummary } from "./model.js";
import { patternCheck, similarYears } from "./rain.js";
import { nearestFlooded, indexOf } from "./search.js";

const SLIDERS = [
  { left: "Less rain", right: "More rain" },
  { left: "South-focused", right: "North-focused" },
  { left: "Monsoon-type", right: "Downpour-type" },
];

// β 부호별로 "Rain this area is sensitive to"를 말로 옮긴 표현 (PC1–PC3)
const SENSITIVITY = [
  { pos: "heavy rain across Seoul", neg: "light rain across Seoul" },
  { pos: "rain concentrated in the north", neg: "rain concentrated in the south" },
  { pos: "short, intense downpours", neg: "long monsoon-type rain" },
];

let D;  // {grids, model, years, clusters, meta}
let livingYears = 2;

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

export function fmtPct(p) {
  const v = p * 100;
  if (v < 0.1) return "under 0.1%";
  if (v > 99.5) return "over 99%";
  return v < 10 ? `${v.toFixed(1)}%` : `${Math.round(v)}%`;
}

// ---------------------------------------------------------------- 조작 패널

export function initControls(data, on) {
  D = data;
  const { meta } = D;

  $("presets").innerHTML = meta.presets.map((p, i) =>
    `<button data-i="${i}">${esc(p.label)}</button>`).join("");
  $("presets").addEventListener("click", (e) => {
    const i = e.target.dataset?.i;
    if (i !== undefined) on.onPc([...meta.presets[i].pc]);
  });

  $("sliders").innerHTML = SLIDERS.map((s, k) => `
    <div class="slider">
      <div class="slider-head"><b>PC${k + 1}</b><output id="pc-val-${k}"></output></div>
      <input type="range" id="pc-${k}" min="${meta.slider_min[k]}" max="${meta.slider_max[k]}" step="0.05">
      <div class="slider-ends"><span>${s.left}</span><span>${s.right}</span></div>
    </div>`).join("");
  SLIDERS.forEach((_, k) => {
    $(`pc-${k}`).addEventListener("input", (e) => {
      const pc = [...store.get().pc];
      pc[k] = +e.target.value;
      on.onPc(pc);
    });
  });

  $("calc-btn").addEventListener("click", on.onCalc);
  $("search-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = $("search-input").value.trim();
    if (q) on.onSearch(q);
  });
  $("search-results").addEventListener("click", (e) => {
    const li = e.target.closest("li");
    if (li) on.onPick(+li.dataset.i);
  });
  $("detail-close").addEventListener("click", on.onClear);
  $("detail-body").addEventListener("click", (e) => {
    const L = e.target.dataset?.living;
    if (L) { livingYears = +L; renderDetail(store.get()); }
  });
}

export function renderControls(st) {
  const { meta, years } = D;
  st.pc.forEach((v, k) => {
    $(`pc-${k}`).value = v;
    $(`pc-val-${k}`).textContent = v.toFixed(2);
  });
  $("presets").querySelectorAll("button").forEach((b) => {
    const p = meta.presets[b.dataset.i].pc;
    b.classList.toggle("on", p.every((v, k) => Math.abs(v - st.pc[k]) < 1e-6));
  });

  const pat = patternCheck(meta, years, st.pc);
  $("pattern-warn").hidden = !pat.unusual;
  $("pattern-warn").textContent = `This rainfall pattern has no precedent in past years (closest year: ${pat.year}). Read the probabilities with caution.`;

  const stale = store.isStale();
  $("calc-btn").classList.toggle("pulse", stale && !!st.result);
  $("calc-status").textContent = !st.result ? "" :
    stale ? "The sliders have changed. Recalculate to update the map." : "Result for the current sliders.";
}

export function setSearchMsg(msg) {
  $("search-msg").textContent = msg || "";
}

export function showResults(results) {
  $("search-results").innerHTML = results.map((r, i) =>
    `<li data-i="${i}"><b>${esc(r.label)}</b>${r.sub ? `<span>${esc(r.sub)}</span>` : ""}</li>`).join("");
}

// ---------------------------------------------------------------- 상세 패널

function scenarioBlock(st, g) {
  const { grids } = D;
  if (!st.result) {
    return `<div class="big muted">Not calculated yet</div><p class="note">Press “Calculate flood probability” to see it.</p>`;
  }
  const s = gridSummary(st.result.eta, grids.cluster[g], grids.log_q[g]);
  let delta = "";
  if (st.prevResult) {
    const d = (s.mean - st.prevResult.mean[g]) * 100;
    if (Math.abs(d) >= 0.05) {
      delta = `<span class="delta ${d > 0 ? "up" : "down"}">${d > 0 ? "▲" : "▼"}${Math.abs(d).toFixed(1)}%p</span>`;
    }
  }
  return `<div class="big">${fmtPct(s.mean)} <small>(${fmtPct(s.q05)}–${fmtPct(s.q95)})</small> ${delta}</div>
    ${store.isStale() ? `<p class="note stale">Based on the previous rainfall. The sliders changed, so recalculate.</p>` : ""}`;
}

function sensitivityText(c) {
  const cl = D.clusters[c];
  const sig = cl.beta_mean
    .map((m, k) => ({ k, m, sig: cl.beta_q05[k] > 0 || cl.beta_q95[k] < 0 }))
    .filter((b) => b.sig)
    .sort((a, b) => Math.abs(b.m) - Math.abs(a.m));
  if (!sig.length) return "The data does not clearly show which rainfall pattern this area is most sensitive to.";
  return sig.map((b) => b.m > 0 ? SENSITIVITY[b.k].pos : SENSITIVITY[b.k].neg).join(", ") + " raise the risk here.";
}

function neighborMax(st, g) {
  if (!st.result) return null;
  const { grids } = D;
  let best = null;
  for (let dx = -1; dx <= 1; dx++) {
    for (let dy = -1; dy <= 1; dy++) {
      if (!dx && !dy) continue;
      const n = indexOf(grids.ix[g] + dx, grids.iy[g] + dy);
      if (n >= 0 && (!best || st.result.mean[n] > best.p)) best = { p: st.result.mean[n], n };
    }
  }
  return best;
}

function floodedDetail(st, place, g) {
  const { grids, meta, clusters } = D;
  const years = grids.years[g];
  const L = livingYears;
  const sim = similarYears(meta, D.years, st.pc);
  const nb = neighborMax(st, g);
  const cl = clusters[grids.cluster[g]];
  const width = (grids.p_annual_q95[g] - grids.p_annual_q05[g]) * 100;
  const assign = grids.assign[g] === "hdbscan" ? "grouped by shared flood-year pattern"
    : `attached to nearest cluster (${Math.round(grids.attach_dist_m[g])}m)`;

  return `
    <section><h3>Flood probability in this scenario</h3>${scenarioBlock(st, g)}</section>
    <section><h3>Annual flood probability</h3>
      <div class="mid">In an average year: ${fmtPct(grids.p_annual[g])}
        <small>(${fmtPct(grids.p_annual_q05[g])}–${fmtPct(grids.p_annual_q95[g])})</small></div></section>
    <section><h3>Flood probability over residence
      <span class="seg">${[2, 4].map((y) => `<button data-living="${y}" class="${y === L ? "on" : ""}">${y} yr</button>`).join("")}</span></h3>
      <div class="mid">Living here ${L} years: ${fmtPct(grids[`p_${L}yr`][g])}
        <small>(${fmtPct(grids[`p_${L}yr_q05`][g])}–${fmtPct(grids[`p_${L}yr_q95`][g])})</small></div></section>
    <section><h3>Recorded flood history</h3><p>Flooding recorded in ${years.join(", ")}</p></section>
    <section><h3>Rain this area is sensitive to</h3><p>${sensitivityText(grids.cluster[g])}</p></section>
    <section><h3>Similar past years</h3>
      <ul class="sim">${sim.map((s) => `<li><span>${s.year}</span>${years.includes(s.year)
        ? `<b class="yes">This cell flooded</b>` : `<span class="no">No recorded flooding</span>`}</li>`).join("")}</ul></section>
    <section><h3>Nearby cells</h3><p>${nb ? `Max within 1 cell (about 100m): ${fmtPct(nb.p)}`
      : st.result ? "No flood-history cells among the 8 surrounding cells." : "Shown after calculation."}</p></section>
    <section><h3>Basis of estimate</h3>
      <p>Cluster ${cl.cluster} (${cl.size.toLocaleString()} cells, main flood years ${esc(cl.top_combos)}) · ${assign} ·
      annual-probability 90% interval width ${width.toFixed(1)}%p</p></section>`;
}

function emptyDetail(st, place) {
  const { grids, meta } = D;
  const near = nearestFlooded(place.ix, place.iy);
  let nearText = "No flood-history cells within 300m.";
  if (near) {
    const g = near.idx;
    const p = st.result && !store.isStale()
      ? `this scenario ${fmtPct(st.result.mean[g])}` : `annual ${fmtPct(grids.p_annual[g])}`;
    nearText = `The nearest flood-history cell is ${Math.round(near.dist)}m away, with probability ${p}.`;
  }
  return `
    <section><div class="big muted">No recorded flooding</div>
      <p>No flood damage reports were filed for this cell in ${meta.train_years[0]}–${meta.train_years[meta.train_years.length - 1]}, so no probability is calculated.</p>
      <p>${nearText}</p></section>`;
}

function footer() {
  const s = D.meta.unpredictable_share;
  return `<footer class="notice">
    <p>Trained on 2010–2025 records (excluding 2015 and 2021) based on disaster-relief and insurance damage reports.</p>
    <p>No record does not mean flooding is impossible. In cross-validation, ${Math.round(s[2022] * 100)}% of flooded cells in 2022 and
      ${Math.round(s[2025] * 100)}% in 2025 had no earlier record.</p>
    <p>Probabilities are not a forecast; they estimate “if this rain came.”</p></footer>`;
}

export function renderDetail(st) {
  const panel = $("detail");
  const place = st.place;
  panel.hidden = !place;
  document.body.classList.toggle("has-detail", !!place);
  if (!place) return;
  const head = `<header><h2>${esc(place.label)}</h2><p class="sub">Cell (${place.ix}, ${place.iy})</p></header>`;
  const body = place.idx >= 0 ? floodedDetail(st, place, place.idx) : emptyDetail(st, place);
  $("detail-body").innerHTML = head + body + footer();
}
