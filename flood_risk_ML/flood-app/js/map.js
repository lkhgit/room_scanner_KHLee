// Map: rain layer (IDW-interpolated raster), flood probability layer, district borders, selection marker, legend

import { REGION_NAMES, RAIN_SCALE, idwWeights, rainField } from "./rain.js";

export const FLOOD_BINS = [0.01, 0.05, 0.10, 0.25, 0.50];
export const FLOOD_COLORS = ["#fff5c2", "#fed976", "#feb24c", "#fd8d3c", "#e31a1c", "#800026"];
const FLOOD_LABELS = ["1% 미만", "1–5%", "5–10%", "10–25%", "25–50%", "50% 이상"];
// Sequential blues (ColorBrewer Blues), interpolated continuously with the rainfall
const BLUES = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"]
  .map((h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)));
const RASTER_WIDTH = 480;  // width of the interpolated raster in pixels (about 90 m per pixel)

let map, gridLayer, gridPolys, guLayer, stationLayer, stationMarks, rainOverlay, rainCard, legend, toggle;
let marker, cellOutline;
let raster;  // {canvas, ctx, width, height, weights, mask}
let rainMetric = "max24h";
let lastRain = null;
let stations = [];
let handlers = {};

// Returns the legend colour for a flood probability.
export function floodColor(p) {
  let i = 0;
  while (i < FLOOD_BINS.length && p >= FLOOD_BINS[i]) i++;
  return FLOOD_COLORS[i];
}

// Returns the RGB colour for a rainfall amount on the blue scale.
function rainRgb(mm, metric) {
  const t = Math.min(1, Math.max(0, mm / RAIN_SCALE[metric])) * (BLUES.length - 1);
  const i = Math.min(BLUES.length - 2, Math.floor(t)), f = t - i;
  return BLUES[i].map((c, k) => Math.round(c + (BLUES[i + 1][k] - c) * f));
}
// Returns the CSS colour for a rainfall amount.
const rainCss = (mm, metric) => `rgb(${rainRgb(mm, metric).join(",")})`;

// Returns the latitude/longitude bounding box of a GeoJSON.
function geoBounds(geojson) {
  let south = 90, north = -90, west = 180, east = -180;
  // Walks nested coordinate arrays and widens the bounds.
  const walk = (c) => {
    if (typeof c[0] === "number") {
      west = Math.min(west, c[0]); east = Math.max(east, c[0]);
      south = Math.min(south, c[1]); north = Math.max(north, c[1]);
    } else c.forEach(walk);
  };
  geojson.features.forEach((f) => walk(f.geometry.coordinates));
  return { south, west, north, east };
}

// Mask that is set only for pixels inside the Seoul border.
function buildMask(geojson, b, width, height) {
  const c = document.createElement("canvas");
  c.width = width; c.height = height;
  const ctx = c.getContext("2d");
  // Converts longitude/latitude to a pixel position on the mask canvas.
  const px = ([lon, lat]) => [((lon - b.west) / (b.east - b.west)) * width,
                              ((b.north - lat) / (b.north - b.south)) * height];
  ctx.beginPath();
  for (const f of geojson.features) {
    const polys = f.geometry.type === "Polygon" ? [f.geometry.coordinates] : f.geometry.coordinates;
    for (const rings of polys) {
      for (const ring of rings) {
        ring.forEach((p, i) => (i ? ctx.lineTo(...px(p)) : ctx.moveTo(...px(p))));
        ctx.closePath();
      }
    }
  }
  ctx.fill("evenodd");
  const a = ctx.getImageData(0, 0, width, height).data;
  const mask = new Uint8Array(width * height);
  for (let i = 0; i < mask.length; i++) mask[i] = a[i * 4 + 3];
  return mask;
}

// Creates the rain raster canvas over Seoul with its IDW weights and border mask.
function initRaster(seoul) {
  const b = geoBounds(seoul);
  const lat0 = ((b.south + b.north) / 2) * Math.PI / 180;
  const width = RASTER_WIDTH;
  const height = Math.round(width * ((b.north - b.south) * 110540) / ((b.east - b.west) * 111320 * Math.cos(lat0)));
  const canvas = document.createElement("canvas");
  canvas.width = width; canvas.height = height;
  raster = {
    canvas, ctx: canvas.getContext("2d"), width, height,
    weights: idwWeights(stations, b, width, height, REGION_NAMES.length),
    mask: buildMask(seoul, b, width, height),
  };
  rainOverlay = L.imageOverlay(canvas.toDataURL(), [[b.south, b.west], [b.north, b.east]], {
    opacity: 0.85, interactive: false,
  });
}

// Creates the Leaflet map with its layers, rain card, legend and rain/flood toggle.
export function init(el, grids, seoul, stationList, on) {
  handlers = on;
  stations = stationList;
  map = L.map(el, { preferCanvas: true, zoomControl: true }).setView([37.5665, 126.978], 11);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19, attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);

  const renderer = L.canvas({ padding: 0.3 });
  gridPolys = grids.corners.map((c) =>
    L.polygon(c, { renderer, stroke: false, fillOpacity: 0.8, fillColor: "#ccc", interactive: false }));
  gridLayer = L.layerGroup(gridPolys);

  initRaster(seoul);
  guLayer = L.geoJSON(seoul, {
    style: { color: "#334155", weight: 0.8, opacity: 0.5, fill: false }, interactive: false,
  }).addTo(map);
  stationMarks = stations.map((s) => L.circleMarker([s.lat, s.lon], {
    radius: 3, color: "#fff", weight: 1, fillColor: "#0f172a", fillOpacity: 0.9,
  }).bindTooltip("", { direction: "top", offset: [0, -4] }));
  stationLayer = L.layerGroup(stationMarks);

  rainCard = L.control({ position: "bottomleft" });
  rainCard.onAdd = () => {
    const div = L.DomUtil.create("div", "map-card rain-card");
    L.DomEvent.disableClickPropagation(div);
    div.addEventListener("click", (e) => {
      const m = e.target.dataset?.metric;
      if (m) { rainMetric = m; if (lastRain) showRain(lastRain); renderRainLegend(); }
    });
    return div;
  };

  legend = L.control({ position: "bottomright" });
  legend.onAdd = () => L.DomUtil.create("div", "map-card legend");

  toggle = L.control({ position: "topright" });
  toggle.onAdd = () => {
    const div = L.DomUtil.create("div", "map-card toggle");
    div.innerHTML = `<button data-mode="rain">강우</button><button data-mode="flood">침수 확률</button>`;
    L.DomEvent.disableClickPropagation(div);
    div.addEventListener("click", (e) => {
      const m = e.target.dataset?.mode;
      if (m) handlers.onMode?.(m);
    });
    return div;
  };
  rainCard.addTo(map);
  legend.addTo(map);
  toggle.addTo(map);

  map.on("click", (e) => handlers.onClick?.(e.latlng.lat, e.latlng.lng));
}

// Shows the rain or the flood layer and the matching legend.
export function setMode(mode) {
  toggle.getContainer().querySelectorAll("button").forEach((b) =>
    b.classList.toggle("on", b.dataset.mode === mode));
  const rain = mode === "rain";
  for (const l of [rainOverlay, stationLayer]) rain ? l.addTo(map) : l.remove();
  rain ? gridLayer.remove() : gridLayer.addTo(map);
  rainCard.getContainer().hidden = !rain;
  rain ? renderRainLegend() : renderFloodLegend();
  guLayer.bringToFront();
  marker?.bringToFront();
  cellOutline?.bringToFront();
}

// Colours every grid cell by its mean flood probability.
export function showFlood(mean) {
  for (let g = 0; g < gridPolys.length; g++) gridPolys[g].setStyle({ fillColor: floodColor(mean[g]) });
}

// Paints the rain raster from regional rainfall values, inside the Seoul border only.
function paintRaster(values) {
  const { ctx, width, height, weights, mask } = raster;
  const field = rainField(weights, values);
  const img = ctx.createImageData(width, height);
  for (let i = 0; i < field.length; i++) {
    if (!mask[i]) continue;
    const [r, g, b] = rainRgb(field[i], rainMetric);
    img.data.set([r, g, b, mask[i]], i * 4);
  }
  ctx.putImageData(img, 0, 0);
  rainOverlay.setUrl(raster.canvas.toDataURL());
}

// Shows a rainfall scenario on the map, the station tooltips and the regional rain card.
export function showRain(rain) {
  lastRain = rain;
  const values = rain[rainMetric];
  paintRaster(values);
  const unit = rainMetric === "max1h" ? "1시간" : "24시간";
  stations.forEach((s, i) => stationMarks[i].setTooltipContent(
    `${s.name} · R${s.region} ${REGION_NAMES[s.region - 1]}<br>최대 ${unit} ${values[s.region - 1].toFixed(0)}mm`));

  const rows = REGION_NAMES.map((name, r) => {
    const v = values[r];
    const w = Math.min(100, (v / RAIN_SCALE[rainMetric]) * 100);
    return `<div class="bar-row"><span class="bar-name">R${r + 1} ${name}</span>
      <span class="bar"><i style="width:${w}%;background:${rainCss(v, rainMetric)}"></i></span>
      <span class="bar-val">${v.toFixed(0)}mm</span></div>`;
  }).join("");
  rainCard.getContainer().innerHTML = `
    <div class="card-title">권역별 최대 강우
      <span class="seg"><button data-metric="max1h" class="${rainMetric === "max1h" ? "on" : ""}">1시간</button><button data-metric="max24h" class="${rainMetric === "max24h" ? "on" : ""}">24시간</button></span>
    </div>${rows}
    <p class="card-note">지도 색은 관측소 27곳에 소속 권역 강우를 주고 거리 가중(IDW)으로 보간한 값입니다.</p>`;
}

// Draws the flood probability legend.
function renderFloodLegend() {
  legend.getContainer().innerHTML = `<div class="card-title">이 비가 오면 침수 확률</div>` +
    FLOOD_LABELS.map((l, i) => `<div class="lg-row"><i style="background:${FLOOD_COLORS[i]}"></i>${l}</div>`).join("") +
    `<div class="card-note">색이 없는 곳 = 2010–2025년 침수 기록 없음</div>`;
}

// Draws the rainfall legend for the selected metric.
function renderRainLegend() {
  const max = RAIN_SCALE[rainMetric];
  const stops = BLUES.map((c, i) => `rgb(${c.join(",")}) ${(i / (BLUES.length - 1)) * 100}%`).join(",");
  legend.getContainer().innerHTML = `<div class="card-title">최대 ${rainMetric === "max1h" ? "1시간" : "24시간"} 강우</div>
    <div class="ramp" style="background:linear-gradient(to right,${stops})"></div>
    <div class="ramp-lab"><span>0</span><span>${max / 2}</span><span>${max}mm 이상</span></div>`;
}

// Marks the selected place and outlines its grid cell.
export function setPlace(place, corners) {
  marker?.remove();
  cellOutline?.remove();
  marker = cellOutline = null;
  if (!place) return;
  marker = L.circleMarker([place.lat, place.lng], {
    radius: 7, color: "#fff", weight: 2, fillColor: "#1d4ed8", fillOpacity: 1,
  }).addTo(map);
  if (corners) {
    cellOutline = L.polygon(corners, { color: "#111", weight: 2, fill: false, interactive: false }).addTo(map);
  }
}

// Zooms the map to a point.
export function flyTo(lat, lng) {
  map.setView([lat, lng], Math.max(map.getZoom(), 16));
}

// Tells Leaflet that the map's size changed.
export function invalidate() {
  map?.invalidateSize();
}
