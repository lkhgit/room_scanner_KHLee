// 장소 검색(카카오), WGS84 → EPSG:5179 변환, 격자 찾기

const EPSG5179 = "+proj=tmerc +lat_0=38 +lon_0=127.5 +k=0.9996 +x_0=1000000 +y_0=2000000 " +
  "+ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs";
const toGrid = proj4("EPSG:4326", EPSG5179);
const SEOUL_BOUNDS = { sw: [37.413, 126.734], ne: [37.715, 127.27] };

let cellIndex = null;  // "ix,iy" → 격자 배열 위치
let meta = null;

export function init(grids, appMeta) {
  meta = appMeta;
  cellIndex = new Map();
  for (let g = 0; g < grids.n; g++) cellIndex.set(`${grids.ix[g]},${grids.iy[g]}`, g);
}

export function indexOf(ix, iy) {
  const g = cellIndex.get(`${ix},${iy}`);
  return g === undefined ? -1 : g;
}

// 위경도 → 격자 번호. 침수 이력 격자가 아니면 idx = −1
export function locate(lat, lng) {
  const [X, Y] = toGrid.forward([lng, lat]);
  const ix = Math.round((X - meta.grid_origin[0]) / meta.cell_size);
  const iy = Math.round((Y - meta.grid_origin[1]) / meta.cell_size);
  return { ix, iy, idx: indexOf(ix, iy) };
}

// 반경 radius(m) 안 가장 가까운 침수 이력 격자 (격자 중심 간 거리)
export function nearestFlooded(ix, iy, radius = 300) {
  const r = Math.floor(radius / meta.cell_size);
  let best = null;
  for (let dx = -r; dx <= r; dx++) {
    for (let dy = -r; dy <= r; dy++) {
      const d = Math.hypot(dx, dy) * meta.cell_size;
      if (d === 0 || d > radius) continue;
      const g = indexOf(ix + dx, iy + dy);
      if (g >= 0 && (!best || d < best.dist)) best = { idx: g, dist: d };
    }
  }
  return best;
}

// ---------------------------------------------------------------- 카카오 검색 (서버가 REST 키로 대신 검색: /api/places)

export function inSeoul({ lat, lng }) {
  return lat >= SEOUL_BOUNDS.sw[0] && lat <= SEOUL_BOUNDS.ne[0] &&
         lng >= SEOUL_BOUNDS.sw[1] && lng <= SEOUL_BOUNDS.ne[1];
}

// 키워드 검색 → 결과 없으면 주소 검색. 서울 범위 밖 결과는 뺀다.
export async function search(query) {
  const r = await fetch(`/api/places?q=${encodeURIComponent(query)}`);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error || `Place search failed (${r.status})`);
  return body.filter(inSeoul);
}
