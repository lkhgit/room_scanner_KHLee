// 앱 상태와 변경 알림. 화면 모듈은 subscribe로 필요한 키만 듣는다.
const state = {
  pc: [0, 0, 0],         // 현재 슬라이더 값 PC1–PC3
  result: null,          // 마지막 계산 {pc, eta, mean}
  prevResult: null,      // 직전 계산 (변화량 비교용)
  place: null,           // 선택 장소 {label, lat, lng, ix, iy, idx}
  mode: "rain",          // 지도 종류 "rain" | "flood"
};

const listeners = [];

export function get() {
  return state;
}

export function set(patch) {
  Object.assign(state, patch);
  const keys = Object.keys(patch);
  for (const { key, fn } of listeners) {
    if (keys.includes(key)) fn(state);
  }
}

export function subscribe(keys, fn) {
  for (const key of keys) listeners.push({ key, fn });
}

// 마지막 계산 이후 슬라이더가 바뀌었는지
export function isStale() {
  const r = state.result;
  if (!r) return true;
  return r.pc.some((v, i) => Math.abs(v - state.pc[i]) > 1e-9);
}
