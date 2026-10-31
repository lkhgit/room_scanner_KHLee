// Checks that the app's calculations (js/model.js, js/rain.js) match the Python ones.
// The reference values are tools/reference.json, written by check_reference.py (full precision of scenario_params.json).
//
// Run:
//   py "app\flood_simulator\tools\check_reference.py"     (from the repository root)
//   node "app\flood_simulator\tools\check_model.mjs"
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { compute, gridSummary } from "../js/model.js";
import { reconstruct, idwWeights, rainField } from "../js/rain.js";

const TOOLS = dirname(fileURLToPath(import.meta.url));
// Reads a JSON file.
const read = (p) => JSON.parse(readFileSync(p, "utf-8"));
// Reads a JSON file from the app's data folder.
const data = (n) => read(join(TOOLS, "..", "data", n));

const grids = data("grids.json"), model = data("model.json"), pca = data("pca.json"), meta = data("meta.json");
const ref = read(join(TOOLS, "reference.json"));
let ok = true;

for (const sc of ref.scenarios) {
  const r = compute(model, grids, sc.pc);
  let maxDiff = 0;
  for (let g = 0; g < grids.n; g++) maxDiff = Math.max(maxDiff, Math.abs(r.mean[g] - sc.mean[g]));
  let qDiff = 0;
  for (const [g, q05, q95] of sc.quantiles) {
    const s = gridSummary(r.eta, grids.cluster[g], grids.log_q[g]);
    qDiff = Math.max(qDiff, Math.abs(s.q05 - q05), Math.abs(s.q95 - q95));
  }
  const pass = maxDiff < 5e-4 && qDiff < 5e-4;
  ok &&= pass;
  console.log(`${sc.label.padEnd(8)} 평균 최대 차이 ${maxDiff.toExponential(1)}, 분위수 최대 차이 ${qDiff.toExponential(1)} -> ${pass ? "통과" : "실패"}`);
}

// At every combination of slider ends, rainfall is not negative and 24h ≥ 1h
const K = meta.slider_min.length;
let bad = 0;
for (let m = 0; m < 2 ** K; m++) {
  const pc = meta.slider_min.map((lo, k) => (m >> k) & 1 ? meta.slider_max[k] : lo);
  const { max1h, max24h } = reconstruct(pca, pc);
  max1h.forEach((h, r) => { if (h < 0 || max24h[r] < h || !isFinite(max24h[r])) bad++; });
}
ok &&= bad === 0;
console.log(`슬라이더 양 끝 조합 강우 이상값 ${bad}개 -> ${bad === 0 ? "통과" : "실패"}`);

// Feeding in 2022's PCs should give back 2022's real regional rainfall, approximately (4 PCA axes explain 94% of the variance)
const y2022 = data("years.json").find((y) => y.year === 2022);
const rec = reconstruct(pca, y2022.pc);
const err = Math.max(...rec.max24h.map((v, r) => Math.abs(v - y2022.max24h[r])));
console.log(`2022년 24h 강우 복원 최대 오차 ${err.toFixed(1)}mm (PC1–PC3 근사, 참고값)`);

// IDW map: weights sum to 1, values lie between the regional min and max, and right on a station the value is its region's
const b = { south: 37.42, west: 126.76, north: 37.70, east: 127.19 };
const W = 120, H = 90, st = meta.stations;
const w = idwWeights(st, b, W, H, 5);
let sumErr = 0;
for (let i = 0; i < W * H; i++) {
  let s = 0;
  for (let r = 0; r < 5; r++) s += w[i * 5 + r];
  sumErr = Math.max(sumErr, Math.abs(s - 1));
}
const field = rainField(w, rec.max24h);
const lo = Math.min(...rec.max24h), hi = Math.max(...rec.max24h);
const inRange = field.every((v) => v >= lo - 1e-3 && v <= hi + 1e-3);
const onStation = idwWeights([...st], { south: st[0].lat, north: st[0].lat, west: st[0].lon, east: st[0].lon }, 1, 1, 5);
const exact = Math.abs(rainField(onStation, rec.max24h)[0] - rec.max24h[st[0].region - 1]) < 1e-3;
const idwOk = sumErr < 1e-5 && inRange && exact;
ok &&= idwOk;
console.log(`IDW 가중치 합 오차 ${sumErr.toExponential(1)}, 범위 안 ${inRange}, 관측소 위 값 일치 ${exact} -> ${idwOk ? "통과" : "실패"}`);

process.exit(ok ? 0 : 1);
