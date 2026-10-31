// App state and change notifications. View modules use subscribe() to listen only to the keys they need.
const state = {
  pc: [0, 0, 0],         // current slider values PC1–PC3
  result: null,          // last calculation {pc, eta, mean}
  prevResult: null,      // calculation before that (to compare changes)
  place: null,           // selected place {label, lat, lng, ix, iy, idx}
  mode: "rain",          // map type "rain" | "flood"
};

const listeners = [];

// Returns the app state.
export function get() {
  return state;
}

// Updates the state and calls the listeners of the keys that changed.
export function set(patch) {
  Object.assign(state, patch);
  const keys = Object.keys(patch);
  for (const { key, fn } of listeners) {
    if (keys.includes(key)) fn(state);
  }
}

// Calls fn whenever one of the given keys changes.
export function subscribe(keys, fn) {
  for (const key of keys) listeners.push({ key, fn });
}

// Tells whether the sliders changed since the last calculation.
export function isStale() {
  const r = state.result;
  if (!r) return true;
  return r.pc.some((v, i) => Math.abs(v - state.pc[i]) > 1e-9);
}
