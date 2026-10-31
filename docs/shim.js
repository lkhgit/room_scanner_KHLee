// Static snapshot: answers the Room Scanner server's API calls (and the flood map's data requests) from the data
// scripts next to this file, so the app code runs unchanged. Data is loaded with <script> tags rather than fetch()
// because browsers allow scripts, but not fetch(), on a page opened by double-click (file://).
(() => {
  const base = document.currentScript.src;
  // Resolves a path relative to this script.
  const at = (p) => new URL(p, base).href;
  const realFetch = window.fetch.bind(window);
  const store = (window.RS_DATA = window.RS_DATA || {});
  const pending = {};

  // Loads a data script once and returns the value it stores under key.
  function data(key, file) {
    if (key in store) return Promise.resolve(store[key]);
    return (pending[key] = pending[key] || new Promise((ok, fail) => {
      const s = document.createElement("script");
      s.src = at(file);
      s.onload = () => (key in store ? ok(store[key]) : fail(new Error(`${file} did not define ${key}`)));
      s.onerror = () => fail(new Error(`Could not load ${file}`));
      document.head.appendChild(s);
    }));
  }
  // Builds a JSON Response.
  const json = (body, status = 200) =>
    new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  // Turns an error into a 404 JSON Response.
  const fail = (e) => json({ error: e.message }, 404);

  // Builds a Response with the GLB model decoded from base64.
  function glb(b64) {
    const s = atob(b64), bytes = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i);
    return new Response(bytes, { headers: { "Content-Type": "model/gltf-binary" } });
  }

  // Answers a place search from the saved results, or explains that the demo only knows those places.
  async function searchPlaces(q) {
    const places = await data("places", "flood/data/places.js").catch(() => ({}));
    const key = q && Object.keys(places).find((k) => k.includes(q) || q.includes(k));
    if (key) return json(places[key]);
    return json({ error: `Place search needs the Kakao API, so this demo only knows "${Object.keys(places).join('", "')}". Click the map to pick any other spot.` }, 404);
  }

  // Answers a local request from the data scripts, or returns null to let the real fetch handle it.
  function route(path, query) {
    let m = path.match(/\/flood\/data\/([\w-]+)\.(json|geojson)$/);
    if (m) return data(`flood/${m[1]}.${m[2]}`, `flood/data/${m[1]}.js`).then((v) => json(v), fail);
    m = path.match(/\/api\/(places|rooms(?:\/[^/]+(\/mold|\/model\.glb)?)?)$/);
    if (!m) return null;
    if (m[1] === "places") return searchPlaces(query.get("q") || "");
    if (m[1] === "rooms") return data("rooms", "data/rooms.js").then((v) => json(v), fail);
    if (m[2] === "/model.glb") return data("model", "data/room/model.js").then(glb, fail);
    if (m[2] === "/mold") return data("mold", "data/room/mold.js").then((v) => json(v), fail);
    return data("room", "data/room/room.js").then((v) => json(v), fail);
  }

  window.fetch = (input, opts) => {
    const url = new URL(typeof input === "string" || input instanceof URL ? input : input.url, location.href);
    const local = url.protocol === location.protocol && url.host === location.host;
    return (local && route(url.pathname, url.searchParams)) || realFetch(input, opts);
  };
})();
