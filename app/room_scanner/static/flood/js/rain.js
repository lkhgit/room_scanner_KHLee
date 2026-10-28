// 슬라이더 PC → 권역 강우 복원, 지도용 IDW 보간, 패턴 신뢰도, 비슷한 해. 화면 코드와 분리한 순수 함수.

export const REGION_NAMES = ["North", "Southeast", "South-central", "Northeast", "West"];
export const RAIN_SCALE = { max1h: 100, max24h: 450 };  // 지도 색 고정 척도 (mm)
const IDW_POWER = 2;

// x̂ = mean + std ⊙ (PC · components), 음수는 0, 1h > 24h이면 24h를 1h로 올림
export function reconstruct(pca, pc) {
  const x = pca.mean.map((m, j) =>
    m + pca.std[j] * pc.reduce((s, v, k) => s + v * pca.components[k][j], 0));
  const max1h = [], max24h = [];
  for (let r = 0; r < x.length / 2; r++) {
    const h1 = Math.max(0, x[2 * r]);
    const h24 = Math.max(h1, x[2 * r + 1]);
    max1h.push(h1);
    max24h.push(h24);
  }
  return { max1h, max24h };
}

// 화면 픽셀마다 권역별 IDW 가중치 (관측소 가중치를 소속 권역별로 합산, 합 = 1).
// 관측소마다 소속 권역의 강우를 주므로 값 = Σ_r w_r · 권역 r 강우. 한 번만 계산해 둔다.
export function idwWeights(stations, bounds, width, height, nRegions) {
  const { south, west, north, east } = bounds;
  const lat0 = (south + north) / 2;
  const mx = 111320 * Math.cos((lat0 * Math.PI) / 180), my = 110540;
  const w = new Float32Array(width * height * nRegions);
  for (let y = 0; y < height; y++) {
    const lat = north - ((y + 0.5) * (north - south)) / height;
    for (let x = 0; x < width; x++) {
      const lon = west + ((x + 0.5) * (east - west)) / width;
      const o = (y * width + x) * nRegions;
      let sum = 0;
      for (const s of stations) {
        const d2 = ((lon - s.lon) * mx) ** 2 + ((lat - s.lat) * my) ** 2;
        const v = 1 / Math.max(d2, 1) ** (IDW_POWER / 2);
        w[o + s.region - 1] += v;
        sum += v;
      }
      for (let r = 0; r < nRegions; r++) w[o + r] /= sum;
    }
  }
  return w;
}

export function rainField(weights, values) {
  const nR = values.length, n = weights.length / nR;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let v = 0;
    for (let r = 0; r < nR; r++) v += weights[i * nR + r] * values[r];
    out[i] = v;
  }
  return out;
}

function distance(a, b, scale) {
  let s = 0;
  for (let k = 0; k < scale.length; k++) s += ((a[k] - b[k]) / scale[k]) ** 2;
  return Math.sqrt(s);
}

// 가장 가까운 과거 연도까지의 거리가 과거 연도끼리의 최대 최근접 거리보다 멀면 "과거에 없던 패턴"
export function patternCheck(meta, years, pc) {
  let best = null;
  for (const y of years) {
    const d = distance(pc, y.pc, meta.pattern_scale);
    if (!best || d < best.dist) best = { year: y.year, dist: d };
  }
  return { ...best, unusual: best.dist > meta.pattern_max_nn_dist };
}

// 침수 데이터가 있는 해 중 PC1–PC3이 가장 가까운 k개
export function similarYears(meta, years, pc, k = 3) {
  return years
    .filter((y) => y.has_flood_data)
    .map((y) => ({ year: y.year, dist: distance(pc, y.pc, meta.pattern_scale) }))
    .sort((a, b) => a.dist - b.dist)
    .slice(0, k);
}
