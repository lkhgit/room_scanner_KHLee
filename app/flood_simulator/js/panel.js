// Left control panel (sliders, presets, search) and right grid cell detail panel

import * as store from "./state.js";
import { gridSummary } from "./model.js";
import { patternCheck, similarYears } from "./rain.js";
import { nearestFlooded, indexOf } from "./search.js";

const SLIDERS = [
  { left: "비 적음", right: "비 많음" },
  { left: "남쪽 집중", right: "북쪽 집중" },
  { left: "장마형", right: "집중호우형" },
];

// Wording of "the rain this area is sensitive to" for each sign of β (PC1–PC3)
const SENSITIVITY = [
  { pos: "서울 전체에 비가 많을 때", neg: "서울 전체에 비가 적을 때" },
  { pos: "북쪽에 비가 몰릴 때", neg: "남쪽에 비가 몰릴 때" },
  { pos: "짧고 강한 집중호우일 때", neg: "길게 내리는 장마형 비일 때" },
];

let D;  // {grids, model, years, clusters, meta}
let livingYears = 2;

// Returns the element with the given id.
const $ = (id) => document.getElementById(id);
// Escapes text for safe use inside HTML.
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// Formats a probability as a percentage, with "under 0.1%" and "over 99%" at the ends.
export function fmtPct(p) {
  const v = p * 100;
  if (v < 0.1) return "0.1% 미만";
  if (v > 99.5) return "99% 초과";
  return v < 10 ? `${v.toFixed(1)}%` : `${Math.round(v)}%`;
}

// ---------------------------------------------------------------- control panel

// Builds the presets, sliders and search box, and wires them to the handlers.
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

// Updates the sliders, presets, unusual-pattern warning and calculation status.
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
  $("pattern-warn").textContent = `과거에 없던 강우 패턴입니다 (가장 가까운 해: ${pat.year}년). 확률을 조심해서 읽으세요.`;

  const stale = store.isStale();
  $("calc-btn").classList.toggle("pulse", stale && !!st.result);
  $("calc-status").textContent = !st.result ? "" :
    stale ? "슬라이더가 바뀌었습니다. 다시 계산하면 지도가 갱신됩니다." : "현재 슬라이더로 계산한 결과입니다.";
}

// Shows a message under the search box.
export function setSearchMsg(msg) {
  $("search-msg").textContent = msg || "";
}

// Lists search results to choose from.
export function showResults(results) {
  $("search-results").innerHTML = results.map((r, i) =>
    `<li data-i="${i}"><b>${esc(r.label)}</b>${r.sub ? `<span>${esc(r.sub)}</span>` : ""}</li>`).join("");
}

// ---------------------------------------------------------------- detail panel

// Returns the HTML for a cell's flood probability in the current scenario, with its change since the last run.
function scenarioBlock(st, g) {
  const { grids } = D;
  if (!st.result) {
    return `<div class="big muted">아직 계산 전</div><p class="note">“침수 확률 계산”을 누르면 표시됩니다.</p>`;
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
    ${store.isStale() ? `<p class="note stale">이전 강우 기준입니다. 슬라이더가 바뀌었으니 다시 계산하세요.</p>` : ""}`;
}

// Describes which rain patterns raise this cluster's risk, from the β values whose 90% interval excludes 0.
function sensitivityText(c) {
  const cl = D.clusters[c];
  const sig = cl.beta_mean
    .map((m, k) => ({ k, m, sig: cl.beta_q05[k] > 0 || cl.beta_q95[k] < 0 }))
    .filter((b) => b.sig)
    .sort((a, b) => Math.abs(b.m) - Math.abs(a.m));
  if (!sig.length) return "어떤 강우 양상에 특히 민감한지는 데이터로 분명하지 않습니다.";
  return sig.map((b) => b.m > 0 ? SENSITIVITY[b.k].pos : SENSITIVITY[b.k].neg).join(", ") + " 위험이 커집니다.";
}

// Finds the highest flood probability among the 8 neighbouring cells.
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

// Returns the detail panel HTML for a cell with flood history.
function floodedDetail(st, place, g) {
  const { grids, meta, clusters } = D;
  const years = grids.years[g];
  const L = livingYears;
  const sim = similarYears(meta, D.years, st.pc);
  const nb = neighborMax(st, g);
  const cl = clusters[grids.cluster[g]];
  const width = (grids.p_annual_q95[g] - grids.p_annual_q05[g]) * 100;
  const assign = grids.assign[g] === "hdbscan" ? "같은 침수 연도 패턴으로 묶임"
    : `가까운 클러스터에 붙임 (${Math.round(grids.attach_dist_m[g])}m)`;

  return `
    <section><h3>이 시나리오의 침수 확률</h3>${scenarioBlock(st, g)}</section>
    <section><h3>연간 침수 확률</h3>
      <div class="mid">평균적인 해 기준 ${fmtPct(grids.p_annual[g])}
        <small>(${fmtPct(grids.p_annual_q05[g])}–${fmtPct(grids.p_annual_q95[g])})</small></div></section>
    <section><h3>거주 기간 침수 확률
      <span class="seg">${[2, 4].map((y) => `<button data-living="${y}" class="${y === L ? "on" : ""}">${y}년</button>`).join("")}</span></h3>
      <div class="mid">${L}년 거주 시 ${fmtPct(grids[`p_${L}yr`][g])}
        <small>(${fmtPct(grids[`p_${L}yr_q05`][g])}–${fmtPct(grids[`p_${L}yr_q95`][g])})</small></div></section>
    <section><h3>실제 침수 이력</h3><p>${years.map((y) => `${y}년`).join(", ")} 침수 기록</p></section>
    <section><h3>이 지역이 민감한 비</h3><p>${sensitivityText(grids.cluster[g])}</p></section>
    <section><h3>비슷했던 해</h3>
      <ul class="sim">${sim.map((s) => `<li><span>${s.year}년</span>${years.includes(s.year)
        ? `<b class="yes">이 격자 침수</b>` : `<span class="no">침수 기록 없음</span>`}</li>`).join("")}</ul></section>
    <section><h3>주변 격자</h3><p>${nb ? `반경 1칸(약 100m) 안 최대 ${fmtPct(nb.p)}`
      : st.result ? "주변 8칸에 침수 이력 격자가 없습니다." : "계산 후 표시됩니다."}</p></section>
    <section><h3>추정 근거</h3>
      <p>클러스터 ${cl.cluster} (격자 ${cl.size.toLocaleString()}개, 주요 침수 연도 ${esc(cl.top_combos)}) · ${assign} ·
      연간 확률 90% 구간 폭 ${width.toFixed(1)}%p</p></section>`;
}

// Returns the detail panel HTML for a cell without flood history, pointing to the nearest cell that has some.
function emptyDetail(st, place) {
  const { grids, meta } = D;
  const near = nearestFlooded(place.ix, place.iy);
  let nearText = "반경 300m 안에 침수 이력이 있는 격자가 없습니다.";
  if (near) {
    const g = near.idx;
    const p = st.result && !store.isStale()
      ? `이 시나리오 ${fmtPct(st.result.mean[g])}` : `연간 ${fmtPct(grids.p_annual[g])}`;
    nearText = `가장 가까운 침수 이력 격자는 ${Math.round(near.dist)}m 떨어져 있고, 그 격자의 확률은 ${p}입니다.`;
  }
  return `
    <section><div class="big muted">기록 없음</div>
      <p>${meta.train_years[0]}–${meta.train_years[meta.train_years.length - 1]}년 피해 신고 기록이 없는 격자라 확률을 계산하지 않습니다.</p>
      <p>${nearText}</p></section>`;
}

// Returns the footer that explains what the data can and cannot tell.
function footer() {
  const s = D.meta.unpredictable_share;
  return `<footer class="notice">
    <p>재난지원금·보험 피해 신고 기반 2010–2025년 기록(2015·2021년 제외)으로 학습했습니다.</p>
    <p>기록이 없다고 침수가 불가능한 것은 아닙니다. 교차검증에서 2022년에는 침수 격자의 ${Math.round(s[2022] * 100)}%,
      2025년에는 ${Math.round(s[2025] * 100)}%가 그 이전에 기록이 없던 곳이었습니다.</p>
    <p>확률은 예보가 아니라 “이런 비가 온다면”의 추정치입니다.</p></footer>`;
}

// Shows or hides the detail panel for the selected place.
export function renderDetail(st) {
  const panel = $("detail");
  const place = st.place;
  panel.hidden = !place;
  document.body.classList.toggle("has-detail", !!place);
  if (!place) return;
  const head = `<header><h2>${esc(place.label)}</h2><p class="sub">격자 (${place.ix}, ${place.iy})</p></header>`;
  const body = place.idx >= 0 ? floodedDetail(st, place, place.idx) : emptyDetail(st, place);
  $("detail-body").innerHTML = head + body + footer();
}
