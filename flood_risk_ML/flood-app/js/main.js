// 데이터 로드와 모듈 연결. 상황별 동작(계획 문서의 표)은 여기서 정한다.

import * as store from "./state.js";
import * as rain from "./rain.js";
import * as model from "./model.js";
import * as mapView from "./map.js";
import * as search from "./search.js";
import * as panel from "./panel.js";

async function loadJson(name) {
  const res = await fetch(`data/${name}`);
  if (!res.ok) throw new Error(`${name}을 불러오지 못했습니다 (${res.status})`);
  return res.json();
}

let D;
let searchResults = [];

function calculate() {
  const st = store.get();
  const t = performance.now();
  const result = model.compute(D.model, D.grids, st.pc);
  console.log(`침수 확률 계산 ${(performance.now() - t).toFixed(0)}ms`);
  store.set({ prevResult: st.result, result, mode: "flood" });
}

function selectPlace(label, lat, lng) {
  const cell = search.locate(lat, lng);
  store.set({ place: { label, lat, lng, ...cell } });
}

const handlers = {
  onPc(pc) {
    store.set({ pc, mode: "rain" });
  },
  onCalc: calculate,
  onMode(mode) {
    if (mode === "flood" && !store.get().result) calculate();
    else store.set({ mode });
  },
  async onSearch(query) {
    panel.setSearchMsg("검색 중…");
    panel.showResults([]);
    try {
      searchResults = await search.search(query);
    } catch (e) {
      panel.setSearchMsg(e.message);
      return;
    }
    if (!searchResults.length) return panel.setSearchMsg("서울 안에서 결과를 찾지 못했습니다.");
    if (searchResults.length === 1) return handlers.onPick(0);
    panel.setSearchMsg(`${searchResults.length}개 결과 중 하나를 고르세요.`);
    panel.showResults(searchResults);
  },
  onPick(i) {
    const r = searchResults[i];
    panel.showResults([]);
    panel.setSearchMsg("");
    if (!store.get().result) calculate();
    else store.set({ mode: "flood" });
    selectPlace(r.label, r.lat, r.lng);
    mapView.flyTo(r.lat, r.lng);
  },
  onClick(lat, lng) {
    selectPlace("지도에서 고른 위치", lat, lng);
  },
  onClear() {
    store.set({ place: null });
  },
};

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

  // 처음 접속: 평년 프리셋의 강우 지도, 상세 패널 닫힘
  store.set({ pc: [...meta.presets[0].pc], mode: "rain" });
  document.getElementById("loading").remove();
}

main().catch((e) => {
  document.getElementById("loading").textContent = `불러오기 실패: ${e.message}`;
  console.error(e);
});
