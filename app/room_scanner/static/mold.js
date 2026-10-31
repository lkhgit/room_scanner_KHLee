// Mold tab: the 3D room model (room.glb) with the mold index M of each icosahedron face projected onto it.
// room.glb has its origin at the scanner, y up and the rig's az 0° along -z, so a surface point's direction
// from the origin picks the face that measured it. Without a room model, the bare icosahedron is shown.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const M_MAX = 6; // VTT mold index scale 0-6
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const MODE_NAMES = { off: "HVAC off", heat: "Heating", cool: "Cooling" };

// Turns an HTML string into a DOM element.
const el = (html) => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstChild; };

// Rig direction (az from -z towards +x, el up) -> unit vector in the model frame.
const dirOf = (az, el) => { const a = az * Math.PI / 180, e = el * Math.PI / 180;
  return new THREE.Vector3(Math.sin(a) * Math.cos(e), Math.sin(e), -Math.cos(a) * Math.cos(e)); };

const VERT = `
varying vec3 vPos; varying vec2 vUv;
void main() { vPos = position; vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
// Returns the fragment shader for nf faces: colours each surface point by the mold index of the face it points at.
const FRAG = (nf) => `
#define NF ${nf}
uniform sampler2D uMap; uniform float uHasMap; uniform float uAlpha; uniform float uMax;
uniform vec3 uDir[NF]; uniform float uM[NF]; uniform int uSel;
varying vec3 vPos; varying vec2 vUv;
// olive / drab green, darker and browner as M grows
vec3 mold(float t) { return mix(vec3(0.62, 0.62, 0.36), vec3(0.29, 0.27, 0.10), clamp(t, 0.0, 1.0)); }
void main() {
  vec3 d = normalize(vPos);
  // soft nearest-face lookup: the closest face dominates, neighbours blend in only at the seams
  float best = -2.0; int bi = 0;
  for (int i = 0; i < NF; i++) { float c = dot(d, uDir[i]); if (c > best) { best = c; bi = i; } }
  float wsum = 0.0, m = 0.0;
  for (int i = 0; i < NF; i++) { float w = exp(30.0 * (dot(d, uDir[i]) - best)); wsum += w; m += w * uM[i]; }
  m /= wsum;
  // room photo washed out to a hazy off-white so it never competes with the mold colour
  vec3 tex = uHasMap > 0.5 ? texture2D(uMap, vUv).rgb : vec3(0.8);
  float g = dot(tex, vec3(0.299, 0.587, 0.114));
  vec3 base = mix(mix(tex, vec3(g), 0.75), vec3(0.95, 0.95, 0.93), 0.6);
  // no colour below M = 0.3 (nothing growing), full colour from M = 3 (visible mold)
  float cover = smoothstep(0.3, 3.0, m);
  vec3 col = mix(base, mold(m / uMax), uAlpha * cover);
  if (bi == uSel) col = mix(col, vec3(0.25, 0.55, 0.95), 0.28);
  gl_FragColor = vec4(col, 1.0);
}`;

// Makes a text label sprite (used for the N/E/S/W compass letters).
function sprite(text, color) {
  const c = document.createElement("canvas"); c.width = c.height = 128;
  const g = c.getContext("2d"); g.font = "bold 84px sans-serif"; g.fillStyle = color; g.textAlign = "center"; g.textBaseline = "middle";
  g.fillText(text, 64, 68);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false }));
  s.scale.setScalar(0.5); return s;
}

// Mounts the mold viewer (3D model, time slider, face details) into container and returns its controls.
export function mountMold(container, roomId, data, room) {
  const faces = data.faces, n = data.time.n;
  const [y0, mo0, d0, h0] = data.time.start.match(/\d+/g).map(Number);
  const baseMs = Date.UTC(y0, mo0 - 1, d0, h0);
  // Formats an hour index of the simulation as a date and time in KST.
  const fmtTime = (i) => { const d = new Date(baseMs + i * 3600e3);
    return `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()}, ${d.getUTCFullYear()} ${String(d.getUTCHours()).padStart(2, "0")}:00 KST`; };

  const state = { t: Math.round(n * 0.62), scen: "central", face: null, series: "M", alpha: 0.9 };
  const root = el(`<div class="mold">
    <div class="side">
      <div class="note" id="prov"></div>
      <h3>Time</h3><input type="range" id="t" min="0" max="${n - 1}" value="${state.t}">
      <div id="tlabel" style="font-weight:600"></div>
      <div class="seg" style="margin-top:6px"><button data-t="winter">Jan 15</button><button data-t="spring">Apr 15</button><button data-t="summer">Jul 15</button><button data-t="autumn">Oct 15</button></div>
      <h3>Scenario</h3>
      <div class="seg" id="scen"><button data-s="central" class="on">Central</button><button data-s="worst">Worst case</button></div>
      <h3>Mold overlay</h3><input type="range" id="alpha" min="0" max="100" value="${state.alpha * 100}">
      <div class="note" id="panonote"></div>
      <h3>At this hour</h3><dl class="kv" id="now"></dl>
      <h3>Measured inputs</h3><dl class="kv" id="inputs"></dl>
      <h3>Notes</h3><div id="warns" class="note"></div>
    </div>
    <div class="stage">
      <canvas class="gl"></canvas>
      <div class="legend"><div>Mold index M (0–6)</div><div class="ramp"></div><div class="tk"><span>0</span><span>3</span><span>6</span></div></div>
      <div class="facebox" id="facebox" hidden></div>
    </div></div>`);
  container.appendChild(root);
  // Returns the first element inside the viewer that matches a CSS selector.
  const $ = (s) => root.querySelector(s);

  // ---- scene
  const canvas = $("canvas.gl");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 50);
  camera.position.set(2.6, 2.0, 3.2);
  const controls = new OrbitControls(camera, canvas); controls.enableDamping = true;
  const group = new THREE.Group(); scene.add(group);

  const faceIds = faces.map((f) => f.face);
  const faceDirs = faces.map((f) => dirOf(f.az, f.el));
  const uM = new Float32Array(faces.length);
  const shared = { uAlpha: { value: state.alpha }, uMax: { value: M_MAX }, uDir: { value: faceDirs },
                   uM: { value: uM }, uSel: { value: -1 } };
  const frag = FRAG(faces.length);
  // Makes the shader material for one mesh, with the room photo as its texture if there is one.
  const makeMat = (map) => {
    if (map) { map.colorSpace = THREE.NoColorSpace; map.needsUpdate = true; }   // sample the photo as stored (sRGB)
    return new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: frag, side: THREE.FrontSide,
      uniforms: { ...shared, uMap: { value: map || null }, uHasMap: { value: map ? 1 : 0 } } });
  };
  let pickables = [];
  const compass = new THREE.Group(); scene.add(compass);

  // Places the N/E/S/W compass letters and the north arrow around the model.
  function placeCompass(r) {
    compass.clear();
    [["N", 0, "#ff6b6b"], ["E", 90, "#8a93a3"], ["S", 180, "#8a93a3"], ["W", 270, "#8a93a3"]].forEach(([t, b, col]) => {
      const a = b * Math.PI / 180, s = sprite(t, col); s.scale.setScalar(r * 0.22);
      s.position.set(Math.sin(a) * r * 1.15, 0, -Math.cos(a) * r * 1.15); compass.add(s);
    });
    compass.add(new THREE.ArrowHelper(new THREE.Vector3(0, 0, -1), new THREE.Vector3(0, 0, -r * 0.7), r * 0.3, 0xff6b6b, r * 0.08, r * 0.05));
  }
  // Points the camera at an object, from inside the room or from outside.
  function frame(obj, inward) {
    const box = new THREE.Box3().setFromObject(obj), c = box.getCenter(new THREE.Vector3()), s = box.getSize(new THREE.Vector3()).length();
    controls.target.copy(c);
    camera.near = s / 200; camera.far = s * 20; camera.updateProjectionMatrix();
    camera.position.set(c.x + s * (inward ? 0.45 : 0.6), c.y + s * (inward ? 0.8 : 0.5), c.z + s * (inward ? 0.45 : 0.75));
    placeCompass(s * 0.55);
  }

  // Shows the bare icosahedron of measured faces when there is no room model, with a note why.
  function showIcosahedron(note) {
    const G = data.geometry;
    const pos = [];
    faceIds.forEach((fid) => { const tri = G && G.triangles[fid]; if (tri) tri.forEach((vi) => pos.push(...G.vertices[vi])); });
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    geom.setAttribute("uv", new THREE.Float32BufferAttribute(new Float32Array(pos.length / 3 * 2), 2));
    const mesh = new THREE.Mesh(geom, makeMat(null)); mesh.material.side = THREE.DoubleSide;
    group.add(mesh, new THREE.LineSegments(new THREE.EdgesGeometry(geom, 1), new THREE.LineBasicMaterial({ color: 0x20242c })));
    pickables = [mesh];
    frame(group, false);
    $("#panonote").textContent = note;
  }

  if (room.steps.model && room.steps.model.state === "done") {
    $("#panonote").textContent = "Loading the 3D room model…";
    new GLTFLoader().load(`/api/rooms/${roomId}/model.glb`, (gltf) => {
      if (!alive) return;
      gltf.scene.traverse((o) => { if (o.isMesh) { o.material = makeMat(o.material.map); pickables.push(o); } });
      group.add(gltf.scene);
      frame(group, true);
      $("#panonote").textContent = "Mold index of each measured direction, projected onto the 3D room model from the scanner position, "
        + "assuming the surface scan started level and facing the same way as the camera's start position (yaw 0°). "
        + "The near walls are hidden so you can see inside.";
    }, undefined, () => showIcosahedron("The 3D room model could not be loaded, so the measured faces are shown instead."));
  } else {
    showIcosahedron(room.has.photos || room.has.panorama
      ? "The 3D room model is not ready yet (see the Sunlight tab), so the measured faces are shown for now."
      : "This scan has no photos, so there is no 3D room model; the measured faces are shown instead.");
  }

  // Rotates the model so that north points the right way for the room's heading.
  function setHeading() { group.rotation.y = -((room.params.heading_deg || 0) * Math.PI / 180); }
  setHeading();

  // Resizes the renderer when the canvas size changes.
  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    if (canvas.width !== w * devicePixelRatio || canvas.height !== h * devicePixelRatio) {
      renderer.setPixelRatio(devicePixelRatio); renderer.setSize(w, h, false);
      camera.aspect = w / h; camera.updateProjectionMatrix();
    }
  }
  let raf = 0, alive = true;
  (function loop() { if (!alive) return; raf = requestAnimationFrame(loop); resize(); controls.update(); renderer.render(scene, camera); })();

  // ---- data -> colours / text
  const faceById = new Map(faces.map((f) => [f.face, f]));
  // Returns the hourly weather and indoor series.
  const S = () => data.series;
  // Returns a face's mold index at the current time for the current scenario.
  const colorAt = (f) => f[state.scen].M[state.t];

  // Updates the colours and the readouts for the current time.
  function refresh() {
    faceIds.forEach((fid, k) => { uM[k] = colorAt(faceById.get(fid)); });
    $("#tlabel").textContent = fmtTime(state.t);
    const s = S(), i = state.t;
    const mode = data.mode_labels[s.mode[i]];
    $("#now").innerHTML = [
      ["Outdoor temperature", `${s.theta_e[i].toFixed(1)} °C`],
      ["Ground temperature", `${s.theta_g[i].toFixed(1)} °C`],
      ["Indoor temperature", `${s.theta_i[i].toFixed(1)} °C`],
      ["Indoor humidity", `${s.rh_i[state.scen][i].toFixed(0)} %`],
      ["HVAC", MODE_NAMES[mode] || mode],
      ["Weather data", s.data_flag[i] ? "Interpolated" : "Observed"],
    ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
    drawFace();
  }

  const dv = data.params.dv_max;
  $("#inputs").innerHTML = [
    ["Moisture load Δv_max", `${(state.scen === "worst" ? dv.worst : dv.central).toFixed(1)} g/m³ (${dv.judgment})`],
    ["Simulated year", String(data.period.year)],
    ["Weather station", `#${data.station.id}, ${data.station.distance_km} km away`],
    ["Ground depth", `${data.params.ground_depth_m} m`],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  $("#prov").innerHTML = data.provisional_message ? `<div class="notice warn">${data.provisional_message}</div>` : "";
  $("#warns").innerHTML = data.warnings.length ? "<ul style='padding-left:16px;margin:0'>" + data.warnings.map((w) => `<li>${w}</li>`).join("") + "</ul>" : "None.";

  // ---- selected face panel
  // Draws the selected face's details and its chart of M, surface RH or surface temperature over the year.
  function drawFace() {
    const box = $("#facebox");
    if (state.face == null) { box.hidden = true; return; }
    const f = faceById.get(state.face), r = f[state.scen];
    const maxM = Math.max(...r.M);
    box.hidden = false;
    box.innerHTML = `<b>Face ${f.face}</b> <span class="note">(${f.zone === "M" ? "mid" : f.zone === "T" ? "top" : f.zone === "B" ? "bottom" : f.zone || ""} · az ${f.az}° · el ${f.el}°)</span>
      <div class="seg" style="margin:6px 0"><button data-k="M">Mold M</button><button data-k="rh_si">Surface RH</button><button data-k="theta_si">Surface temp</button></div>
      <canvas></canvas>
      <div class="note">${f.material.replace(/_/g, " ")} · peak M this year ${maxM.toFixed(2)} · temperature factor f = ${r && f.f ? f.f[state.scen].toFixed(2) : "–"} (${f.f_judgment})${f.excluded ? " · excluded" : ""}</div>`;
    box.querySelectorAll("[data-k]").forEach((b) => { b.classList.toggle("on", b.dataset.k === state.series);
      b.onclick = () => { state.series = b.dataset.k; drawFace(); }; });
    const cv = box.querySelector("canvas"), w = cv.clientWidth, h = 130, dpr = devicePixelRatio;
    cv.width = w * dpr; cv.height = h * dpr;
    const g = cv.getContext("2d"); g.scale(dpr, dpr);
    const ys = r[state.series], lo = state.series === "M" ? 0 : Math.min(...ys), hi = Math.max(...ys, lo + 1e-6);
    const style = getComputedStyle(document.documentElement);
    g.strokeStyle = style.getPropertyValue("--accent"); g.lineWidth = 1;
    g.beginPath();
    const step = Math.max(1, Math.floor(ys.length / w));
    for (let x = 0; x < w; x++) { const v = ys[Math.min(ys.length - 1, x * step)], y = h - 4 - (v - lo) / (hi - lo) * (h - 12);
      x ? g.lineTo(x, y) : g.moveTo(x, y); }
    g.stroke();
    g.fillStyle = style.getPropertyValue("--muted"); g.font = "10px sans-serif";
    g.fillText(hi.toFixed(state.series === "M" ? 2 : 1), 2, 10); g.fillText(lo.toFixed(state.series === "M" ? 2 : 1), 2, h - 6);
    const tx = state.t / (n - 1) * w; g.strokeStyle = "#e8731a"; g.beginPath(); g.moveTo(tx, 0); g.lineTo(tx, h); g.stroke();
  }

  // Highlights the selected face on the model.
  function highlight() { shared.uSel.value = faceIds.indexOf(state.face); }

  // ---- interaction
  const ray = new THREE.Raycaster(), ptr = new THREE.Vector2();
  let down = null;
  canvas.addEventListener("pointerdown", (e) => { down = [e.clientX, e.clientY]; });
  canvas.addEventListener("pointerup", (e) => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
    const r = canvas.getBoundingClientRect();
    ptr.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ptr, camera);
    const hit = ray.intersectObjects(pickables, false)[0];
    if (hit) {    // the face whose direction (seen from the scanner) is closest to the clicked point
      const d = group.worldToLocal(hit.point.clone()).normalize();
      let k = 0; faceDirs.forEach((v, i) => { if (v.dot(d) > faceDirs[k].dot(d)) k = i; });
      state.face = faceIds[k];
    } else state.face = null;
    highlight(); drawFace();
  });
  $("#t").oninput = (e) => { state.t = +e.target.value; refresh(); };
  root.querySelectorAll("[data-t]").forEach((b) => b.onclick = () => {
    const m = { winter: [0, 15], spring: [3, 15], summer: [6, 15], autumn: [9, 15] }[b.dataset.t];
    state.t = Math.min(n - 1, Math.round((Date.UTC(y0, m[0], m[1], 14) - baseMs) / 3600e3)); $("#t").value = state.t; refresh();
  });
  root.querySelectorAll("[data-s]").forEach((b) => b.onclick = () => {
    state.scen = b.dataset.s; root.querySelectorAll("[data-s]").forEach((x) => x.classList.toggle("on", x === b)); refresh();
  });
  $("#alpha").oninput = (e) => { state.alpha = e.target.value / 100; shared.uAlpha.value = state.alpha; };
  if (!data.geometry && !(room.steps.model && room.steps.model.state === "done")) $("#warns").insertAdjacentHTML("afterbegin", "<div class='notice warn'>The face layout could not be reconstructed, so the 3D view is empty.</div>");
  refresh();

  return {
    setHeading,
    destroy() { alive = false; cancelAnimationFrame(raf); renderer.dispose(); root.remove(); },
  };
}
