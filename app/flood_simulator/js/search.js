// Place search (Kakao), WGS84 -> EPSG:5179 conversion, grid cell lookup

const EPSG5179 = "+proj=tmerc +lat_0=38 +lon_0=127.5 +k=0.9996 +x_0=1000000 +y_0=2000000 " +
  "+ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs";
const toGrid = proj4("EPSG:4326", EPSG5179);
const SEOUL_BOUNDS = { sw: [37.413, 126.734], ne: [37.715, 127.27] };

let cellIndex = null;  // "ix,iy" -> position in the grid arrays
let meta = null;

// Builds the lookup from grid indices to positions in the grid arrays.
export function init(grids, appMeta) {
  meta = appMeta;
  cellIndex = new Map();
  for (let g = 0; g < grids.n; g++) cellIndex.set(`${grids.ix[g]},${grids.iy[g]}`, g);
}

// Returns the position of grid cell (ix, iy) in the grid arrays, or -1.
export function indexOf(ix, iy) {
  const g = cellIndex.get(`${ix},${iy}`);
  return g === undefined ? -1 : g;
}

// Latitude/longitude -> grid cell. idx = −1 if the cell has no flood history.
export function locate(lat, lng) {
  const [X, Y] = toGrid.forward([lng, lat]);
  const ix = Math.round((X - meta.grid_origin[0]) / meta.cell_size);
  const iy = Math.round((Y - meta.grid_origin[1]) / meta.cell_size);
  return { ix, iy, idx: indexOf(ix, iy) };
}

// Nearest grid cell with flood history within radius metres (distance between cell centres).
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

// ---------------------------------------------------------------- Kakao search

let kakaoReady = null;

// Loads the Kakao Maps SDK once, using the JavaScript key in config.js.
function loadKakao() {
  if (kakaoReady) return kakaoReady;
  const key = window.FLOOD_APP_CONFIG?.KAKAO_JS_KEY;
  if (!key) return Promise.reject(new Error("config.js에 카카오 JavaScript 키가 없습니다. 지도를 눌러 격자를 고를 수 있습니다."));
  kakaoReady = new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = `https://dapi.kakao.com/v2/maps/sdk.js?appkey=${key}&libraries=services&autoload=false`;
    s.onload = () => window.kakao.maps.load(resolve);
    s.onerror = () => {
      kakaoReady = null;
      reject(new Error("카카오 SDK를 불러오지 못했습니다. 키와 등록 도메인을 확인하세요."));
    };
    document.head.appendChild(s);
  });
  return kakaoReady;
}

// Searches Kakao places by keyword within Seoul.
function keywordSearch(query) {
  const { kakao } = window;
  const places = new kakao.maps.services.Places();
  const bounds = new kakao.maps.LatLngBounds(
    new kakao.maps.LatLng(...SEOUL_BOUNDS.sw), new kakao.maps.LatLng(...SEOUL_BOUNDS.ne));
  return new Promise((resolve) => {
    places.keywordSearch(query, (data, status) => {
      if (status !== kakao.maps.services.Status.OK) return resolve([]);
      resolve(data.map((d) => ({
        label: d.place_name,
        sub: d.road_address_name || d.address_name,
        lat: +d.y, lng: +d.x,
      })));
    }, { bounds });
  });
}

// Searches Kakao by address.
function addressSearch(query) {
  const { kakao } = window;
  const geocoder = new kakao.maps.services.Geocoder();
  return new Promise((resolve) => {
    geocoder.addressSearch(query, (data, status) => {
      if (status !== kakao.maps.services.Status.OK) return resolve([]);
      resolve(data.map((d) => ({
        label: d.road_address?.address_name || d.address_name,
        sub: d.road_address ? d.address_name : "",
        lat: +d.y, lng: +d.x,
      })));
    });
  });
}

// Tells whether a point lies inside Seoul's bounding box.
function inSeoul({ lat, lng }) {
  return lat >= SEOUL_BOUNDS.sw[0] && lat <= SEOUL_BOUNDS.ne[0] &&
         lng >= SEOUL_BOUNDS.sw[1] && lng <= SEOUL_BOUNDS.ne[1];
}

// Keyword search, then address search if nothing is found. Results outside Seoul are dropped.
export async function search(query) {
  await loadKakao();
  let results = (await keywordSearch(query)).filter(inSeoul);
  if (!results.length) results = (await addressSearch(query)).filter(inSeoul);
  return results;
}
