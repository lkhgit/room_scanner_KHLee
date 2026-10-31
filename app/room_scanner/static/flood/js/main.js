// Loads the data and wires the modules together. What happens in each situation (the table in the plan) is decided here.

import * as store from "./state.js";
import * as rain from "./rain.js";
import * as model from "./model.js";
import * as mapView from "./map.js";
import * as search from "./search.js";
import * as panel from "./panel.js";

// Fetches one JSON file from the data folder.
async function loadJson(name) {
  const res = await fetch(`data/${name}`);
  if (!res.ok) throw new Error(`Failed to load ${name} (${res.status})`);
  return res.json();
}

let D;
let searchResults = [];

// Calculates flood probabilities for the current slider values and switches the map to flood mode.
function calculate() {
  const st = store.get();
  const t = performance.now();
  const result = model.compute(D.model, D.grids, st.pc);
  console.log(`Flood probability calculated in ${(performance.now() - t).toFixed(0)}ms`);
  store.set({ prevResult: st.result, result, mode: "flood" });
}

// Selects a place and finds the grid cell it falls in.
function selectPlace(label, lat, lng) {
  const cell = search.locate(lat, lng);
  store.set({ place: { label, lat, lng, ...cell } });
}

// Selects a place on the flood map (calculating first if needed) and zooms to it.
function pickPlace(label, lat, lng) {
  if (!store.get().result) calculate();
  else store.set({ mode: "flood" });
  selectPlace(label, lat, lng);
  mapView.flyTo(lat, lng);
}

const handlers = {
  // Shows the rain map for new slider values.
  onPc(pc) {
    store.set({ pc, mode: "rain" });
  },
  onCalc: calculate,
  // Switches between the rain and flood maps, calculating first if needed.
  onMode(mode) {
    if (mode === "flood" && !store.get().result) calculate();
    else store.set({ mode });
  },
  // Searches for a place and selects it, or lists the results to choose from.
  async onSearch(query) {
    panel.setSearchMsg("Searching…");
    panel.showResults([]);
    try {
      searchResults = await search.search(query);
    } catch (e) {
      panel.setSearchMsg(e.message);
      return;
    }
    if (!searchResults.length) return panel.setSearchMsg("No results found in Seoul.");
    if (searchResults.length === 1) return handlers.onPick(0);
    panel.setSearchMsg(`Choose one of ${searchResults.length} results.`);
    panel.showResults(searchResults);
  },
  // Selects one search result.
  onPick(i) {
    const r = searchResults[i];
    panel.showResults([]);
    panel.setSearchMsg("");
    pickPlace(r.label, r.lat, r.lng);
  },
  // Selects the point clicked on the map.
  onClick(lat, lng) {
    selectPlace("Location picked on map", lat, lng);
  },
  // Clears the selected place.
  onClear() {
    store.set({ place: null });
  },
};

// Loads all data, sets up the map and panels, shows the first rain map, and selects the place given in the URL.
async function main() {
  const [grids, mdl, pca, years, clusters, meta, seoul] = await Promise.all([
    loadJson("grids.json"), loadJson("model.json"), loadJson("pca.json"),
    loadJson("years.json"), loadJson("clusters.json"), loadJson("meta.json"),
    loadJson("seoul.geojson"),
  ]);
  D = { grids, model: mdl, pca, years, clusters, meta };

  search.init(grids, meta);
  mapView.init("map", grids, seoul, meta.stations, handlers);
  panel.initControls(D, handlers);

  store.subscribe(["pc"], (st) => {
    panel.renderControls(st);
    mapView.showRain(rain.reconstruct(pca, st.pc));
  });
  store.subscribe(["result"], (st) => {
    mapView.showFlood(st.result.mean);
    panel.renderControls(st);
  });
  store.subscribe(["mode"], (st) => mapView.setMode(st.mode));
  store.subscribe(["pc", "result", "place"], (st) => panel.renderDetail(st));
  store.subscribe(["place"], (st) => {
    const p = st.place;
    mapView.setPlace(p, p && p.idx >= 0 ? grids.corners[p.idx] : null);
    setTimeout(mapView.invalidate, 0);
  });

  // First visit: rain map of the normal-year preset, detail panel closed
  store.set({ pc: [...meta.presets[0].pc], mode: "rain" });
  document.getElementById("loading").remove();

  // URL parameters ?lat=..&lng=..&label=..
  const q = new URLSearchParams(location.search);
  if (q.has("lat") && q.has("lng")) {
    const lat = parseFloat(q.get("lat")), lng = parseFloat(q.get("lng"));
    if (Number.isFinite(lat) && Number.isFinite(lng)) {
      if (!search.inSeoul({ lat, lng })) panel.setSearchMsg("Not available outside Seoul");
      else pickPlace(q.get("label") || "Selected location", lat, lng);
    }
  }
}

main().catch((e) => {
  document.getElementById("loading").textContent = `Failed to load: ${e.message}`;
  console.error(e);
});
