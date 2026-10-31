// Slider PCs -> regional rainfall, IDW interpolation for the map, pattern check, similar years. Pure functions kept apart from the view code.

export const REGION_NAMES = ["North", "Southeast", "South-central", "Northeast", "West"];
export const RAIN_SCALE = { max1h: 100, max24h: 450 };  // fixed colour scale of the map (mm)
const IDW_POWER = 2;

// Rebuilds regional rainfall: x̂ = mean + std ⊙ (PC · components); negatives become 0, and if 1h > 24h, 24h is raised to 1h.
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

// IDW weight of every region for every map pixel (station weights summed by the station's region, total = 1).
// Each station gets its region's rainfall, so value = Σ_r w_r · rainfall of region r. Computed once.
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

// Interpolated rainfall for every map pixel from the regional values and the IDW weights.
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

// Euclidean distance between two PC vectors, each axis divided by its scale.
function distance(a, b, scale) {
  let s = 0;
  for (let k = 0; k < scale.length; k++) s += ((a[k] - b[k]) / scale[k]) ** 2;
  return Math.sqrt(s);
}

// Finds the nearest past year; the pattern is "unusual" if it is farther away than any past year is from its nearest neighbour.
export function patternCheck(meta, years, pc) {
  let best = null;
  for (const y of years) {
    const d = distance(pc, y.pc, meta.pattern_scale);
    if (!best || d < best.dist) best = { year: y.year, dist: d };
  }
  return { ...best, unusual: best.dist > meta.pattern_max_nn_dist };
}

// Returns the k years with flood data whose PC1–PC3 are closest.
export function similarYears(meta, years, pc, k = 3) {
  return years
    .filter((y) => y.has_flood_data)
    .map((y) => ({ year: y.year, dist: distance(pc, y.pc, meta.pattern_scale) }))
    .sort((a, b) => a.dist - b.dist)
    .slice(0, k);
}
