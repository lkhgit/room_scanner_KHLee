// Scan wizard: room details -> photo module -> surface module -> room. The server owns the scan state.
const $ = (s, r = document) => r.querySelector(s);
// Escapes text for safe use inside HTML.
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
// POSTs JSON to the server and throws an Error with the server's message if the request fails.
const post = async (url, body) => {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
  const b = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(b.error || `Request failed (${r.status})`);
  return b;
};

const STEPS = [["details", "Room details"], ["photo", "Photos"], ["surface", "Surface scan"], ["finish", "Create room"]];
const STEP_OF = { null: "details", photo_connect: "photo", photo_connecting: "photo", leveling: "photo", capturing: "photo",
  surface_connect: "surface", surface_connecting: "surface", surface_running: "surface", ready: "finish", done: "finish" };
const MATERIALS = [["very_sensitive", "Very sensitive (wood, paper, gypsum board)"], ["sensitive", "Sensitive (wallpaper, painted plaster) — default"],
  ["medium_resistant", "Medium resistant (cement, plastic)"], ["resistant", "Resistant (glass, metal, tile)"]];
const KEYS = { w: ["tilt", 1], s: ["tilt", -1], a: ["pan", -1], d: ["pan", 1] };

// Mounts the scan wizard into app and returns a function that removes it.
export function mountScan(app) {
  let st = { stage: null }, shown = undefined, timer = null, stopped = false, ports = [], size = "medium", busy = false;
  let previewOn = false, previewUrl = null, lastShot = null;

  app.innerHTML = `<div class="page scan"><h1>Scan a room</h1>
    <ol class="wiz" id="wiz"></ol><div id="err"></div><div id="body"></div>
    <details class="card log" id="logbox"><summary class="dim">Scan log</summary><pre id="log"></pre></details>
    <div class="row"><a class="btn" href="#/">Back to Home</a><button class="btn" id="cancel" hidden>Cancel this scan</button></div></div>`;

  $("#cancel").onclick = async () => {
    if (!confirm("Cancel this scan? Data already captured stays in the scans folder, but no room is created.")) return;
    act(() => post("/api/scan/cancel"));
  };

  // Shows an error message above the wizard, or clears it.
  function error(msg) { $("#err").innerHTML = msg ? `<div class="notice bad">${esc(msg)}</div>` : ""; }
  // Runs one wizard action with the buttons disabled, then redraws with the new state.
  async function act(fn) {
    if (busy) return; busy = true; error("");
    document.querySelectorAll("#body button").forEach((b) => b.disabled = true);
    try { const r = await fn(); if (r && "stage" in r) st = r; } catch (e) { error(e.message); }
    busy = false;
    document.querySelectorAll("#body button").forEach((b) => b.disabled = false);
    render(false);
  }
  // Loads the list of serial ports from the server.
  async function loadPorts() {
    try { const r = await fetch("/api/scan/ports"); const b = await r.json(); if (!r.ok) throw new Error(b.error); ports = b; }
    catch (e) { ports = []; error(e.message); }
  }
  // Returns the HTML for a serial port drop-down.
  const portSelect = (id, label, hint) => `<label>${label}<select id="${id}">
      <option value="">Choose a port…</option>${ports.map((p) => `<option value="${esc(p.device)}">${esc(p.device)} — ${esc(p.description)}</option>`).join("")}</select>
      <span class="dim">${hint}</span></label>`;
  // Restores a port choice from localStorage and saves it when it changes.
  const remember = (id) => { const el = $("#" + id); if (!el) return; try { el.value = localStorage.getItem("port_" + id) || ""; } catch {}
    el.onchange = () => { try { localStorage.setItem("port_" + id, el.value); } catch {} }; };

  // Draws the step indicator and the scan log.
  function renderWiz() {
    const cur = STEPS.findIndex(([k]) => k === STEP_OF[st.stage]);
    $("#wiz").innerHTML = STEPS.map(([, l], i) => `<li class="${i < cur ? "done" : i === cur ? "on" : ""}">${l}</li>`).join("");
    $("#cancel").hidden = !st.stage || st.stage === "done";
    $("#logbox").hidden = !st.stage;
    $("#log").textContent = (st.log || []).join("\n");
  }

  // Full redraw only when the stage changes; otherwise just refresh the live numbers.
  function render(force) {
    renderWiz();
    if (st.error && !busy) error(st.error);
    if (!force && shown === st.stage) { live(); return; }
    shown = st.stage;
    previewOn = false;
    const body = $("#body"), b = (h) => body.innerHTML = h;
    switch (st.stage) {
      case null:
        b(`<div class="card"><h2>1. Room details</h2>
          <div class="form">
            <label>Room name<input id="f-name" placeholder="e.g. Mangwon-dong basement A"></label>
            <label>Place (address or landmark)<input id="f-site" placeholder="Used to look up outdoor weather"></label>
            <label>Latitude<input id="f-lat" inputmode="decimal" placeholder="optional"></label>
            <label>Longitude<input id="f-lon" inputmode="decimal" placeholder="optional"></label>
            <label>North heading (°)<input id="f-hd" inputmode="decimal" placeholder="optional"><span class="dim">Compass bearing of the panorama centre. Leave empty to use the compass sensor on the photo module, if fitted.</span></label>
            <label>Camera height (m)<input id="f-h" inputmode="decimal" value="1.2"></label>
          </div>
          <button class="btn primary" id="go">Start scan</button></div>`);
        $("#go").onclick = () => act(() => post("/api/scan/start", { name: $("#f-name").value, site: $("#f-site").value,
          lat: $("#f-lat").value, lon: $("#f-lon").value, heading_deg: $("#f-hd").value, camera_height_m: $("#f-h").value }));
        break;
      case "photo_connect": case "photo_connecting":
        b(`<div class="card"><h2>2. Photos — photo module</h2>
          <p>Connect the <b>photo module</b> (rig Arduino with <code>rig_motor</code> and the XIAO camera) by USB, and place it where the room should be photographed.</p>
          <p><b>North:</b> if the rig has a compass sensor (GY-BNO055), aim the camera along the sensor's X axis <b>before</b> you press Connect
            (do this again after a stopped capture). North is measured at that moment, before any motor moves.</p>
          <div class="form">${portSelect("p-rig", "Rig (Arduino)", "Pan/tilt motors")}${portSelect("p-cam", "Camera (XIAO ESP32S3)", "USB camera serial")}</div>
          <div class="row"><button class="btn primary" id="conn" ${st.stage === "photo_connecting" ? "disabled" : ""}>
            ${st.stage === "photo_connecting" ? '<span class="spin"></span>Connecting…' : "Connect"}</button>
            <button class="btn" id="refresh">Refresh ports</button>
            <button class="btn" id="skip">Skip photos</button></div>
          <p class="dim">Skipping photos means sunlight cannot be calculated for this room.</p></div>`);
        remember("p-rig"); remember("p-cam");
        $("#refresh").onclick = async () => { await loadPorts(); render(true); };
        $("#conn").onclick = () => act(() => post("/api/scan/photo/connect", { arduino: $("#p-rig").value, camera: $("#p-cam").value }));
        $("#skip").onclick = () => act(() => post("/api/scan/photo/skip"));
        break;
      case "leveling":
        b(`<div class="card"><h2>2. Photos — level the camera</h2>
          <p>Move the camera until the room's vertical edges are parallel to the yellow lines and the horizon sits on the green line.
            The position you confirm becomes yaw 0°, pitch 0°.</p>
          <div class="preview"><img id="pv" alt="Camera preview"><div class="guide"></div></div>
          <div class="pad">
            <button class="btn" data-k="w">▲ Tilt up <kbd>W</kbd></button>
            <div class="row"><button class="btn" data-k="a">◀ Pan left <kbd>A</kbd></button><button class="btn" data-k="d">Pan right ▶ <kbd>D</kbd></button></div>
            <button class="btn" data-k="s">▼ Tilt down <kbd>S</kbd></button>
          </div>
          <div class="row"><span class="dim">Step</span><div class="seg" id="size">
            ${[["fine", "0.35°"], ["medium", "2°"], ["coarse", "10°"]].map(([k, l], i) => `<button class="${k === size ? "on" : ""}" data-s="${k}">${l} <kbd>${i + 1}</kbd></button>`).join("")}</div>
            <span class="dim" id="pos"></span></div>
          <div class="row"><button class="btn primary" id="cap">Level OK — start capture <kbd>Enter</kbd></button>
            <button class="btn" id="disc">Disconnect</button></div></div>`);
        body.querySelectorAll("[data-k]").forEach((x) => x.onclick = () => nudge(x.dataset.k));
        body.querySelectorAll("[data-s]").forEach((x) => x.onclick = () => { size = x.dataset.s; body.querySelectorAll("[data-s]").forEach((y) => y.classList.toggle("on", y === x)); });
        $("#cap").onclick = () => act(() => post("/api/scan/photo/capture"));
        $("#disc").onclick = () => act(() => post("/api/scan/photo/stop"));
        previewOn = true; previewLoop();
        break;
      case "capturing":
        b(`<div class="card"><h2>2. Photos — capturing</h2>
          <p>Keep out of the camera's view until the capture finishes.</p>
          <div class="bar-prog"><div id="prog"></div></div><p id="cnt" class="dim"></p>
          <div class="preview"><img id="last" alt="Last photo"></div>
          <button class="btn" id="stop">Stop capture</button></div>`);
        $("#stop").onclick = () => act(() => post("/api/scan/photo/stop"));
        break;
      case "surface_connect": case "surface_connecting":
        b(`<div class="card"><h2>3. Surface scan — surface module</h2>
          <p>${st.photo.count ? `${st.photo.count} photos captured. ` : ""}Now connect the <b>surface module</b> (Arduino with <code>surface_scan</code>) by USB.
            Put it where the photo module stood, level, with the sensor facing <b>the same direction the camera faced when the capture started</b>
            (the levelled yaw 0° position), <b>before</b> you press Start: the module takes that pose as az 0°, el 0° when the port opens,
            and the mold colours are placed on the room model on the assumption that it matches the camera's start position.</p>
          <div class="form">${portSelect("p-surf", "Surface module (Arduino)", "IR + humidity sensors")}
            <label>Main wall material<select id="mat">${MATERIALS.map(([k, l]) => `<option value="${k}" ${k === "sensitive" ? "selected" : ""}>${l}</option>`).join("")}</select></label></div>
          <div class="row"><button class="btn primary" id="sgo" ${st.stage === "surface_connecting" ? "disabled" : ""}>
            ${st.stage === "surface_connecting" ? '<span class="spin"></span>Starting…' : "Start surface scan"}</button>
            <button class="btn" id="refresh">Refresh ports</button></div>
          <p class="dim">One scan measures 20 faces and takes about a minute and a half.</p></div>`);
        remember("p-surf");
        $("#refresh").onclick = async () => { await loadPorts(); render(true); };
        $("#sgo").onclick = () => act(() => post("/api/scan/surface/start", { port: $("#p-surf").value, material: $("#mat").value }));
        break;
      case "surface_running":
        b(`<div class="card"><h2>3. Surface scan — measuring</h2>
          <p id="sstate"></p><div class="bar-prog"><div id="prog"></div></div><p id="cnt" class="dim"></p>
          <button class="btn" id="sstop">Stop scan</button></div>`);
        $("#sstop").onclick = () => act(() => post("/api/scan/surface/stop"));
        break;
      case "ready":
        b(`<div class="card"><h2>4. Create room</h2>
          <p>Scan complete: ${st.photo.count ? `${st.photo.count} photos` : "no photos"} and one 20-face surface scan.</p>
          <p class="dim">Everything is saved in <code>scans/${esc(st.id)}</code>, including a <code>bundle.zip</code> you can upload again later.</p>
          <button class="btn primary" id="fin">Create room and see results</button></div>`);
        $("#fin").onclick = () => act(async () => { const r = await post("/api/scan/finish"); location.hash = `#/room/${r.id}`; });
        break;
      case "done":
        b(`<div class="card"><h2>Done</h2><p>The room has been created.</p>
          <div class="row"><a class="btn primary" href="#/room/${esc(st.room_id)}">Open the room</a>
          <button class="btn" id="again">Scan another room</button></div></div>`);
        $("#again").onclick = () => act(() => post("/api/scan/cancel"));
        break;
    }
    live();
  }

  // Updates the live numbers (angles, progress, face count) without redrawing the step.
  function live() {
    const p = st.photo || {}, s = st.surface || {};
    if (st.stage === "leveling" && $("#pos")) {
      const c = p.compass || {};
      $("#pos").textContent = `tilt ${(p.pitch ?? 0).toFixed(1)}°  pan ${(p.yaw ?? 0).toFixed(1)}°` +
        (p.azimuth != null ? `  ·  camera faces ${p.azimuth.toFixed(1)}° (true north)`
          : c.sensor === false ? "  ·  no compass sensor: north will not be recorded" : "  ·  compass not readable: north will not be recorded");
    }
    if (st.stage === "capturing") {
      $("#prog").style.width = `${p.total ? (100 * p.shot / p.total) : 0}%`;
      $("#cnt").textContent = p.total ? `${p.shot} / ${p.total} photos  ·  ${p.plan}  ·  pitch ${p.pitch}°, yaw ${p.yaw}°` : "Preparing…";
      if (p.last && p.last !== lastShot) { lastShot = p.last; $("#last").src = `/api/scan/photo/last?f=${encodeURIComponent(p.last)}`; }
    }
    if (st.stage === "surface_running") {
      $("#sstate").innerHTML = s.state === "scanning" ? '<span class="spin"></span>Measuring faces…'
        : '<span class="spin"></span>Waiting for the module to be ready…';
      $("#prog").style.width = `${100 * (s.faces || 0) / 20}%`;
      $("#cnt").textContent = `${s.faces || 0} / 20 faces${s.heading != null ? `  ·  module heading ${s.heading}°` : ""}`;
    }
  }

  // Moves the rig one step on the axis for a W/A/S/D key.
  async function nudge(k) {
    const [axis, dir] = KEYS[k];
    try { const r = await post("/api/scan/photo/move", { axis, size, dir }); st = r; live(); } catch (e) { error(e.message); }
  }
  // Handles the leveling keys: W/A/S/D to move, 1/2/3 for step size, Enter to start the capture.
  function onKey(e) {
    if (st.stage !== "leveling" || e.target.closest("input,select,textarea")) return;
    const k = e.key.toLowerCase();
    if (KEYS[k]) { e.preventDefault(); nudge(k); }
    else if (["1", "2", "3"].includes(k)) { const b = document.querySelectorAll("[data-s]")[+k - 1]; b && b.click(); }
    else if (e.key === "Enter") { e.preventDefault(); $("#cap") && $("#cap").click(); }
  }
  addEventListener("keydown", onKey);

  // Keeps fetching camera frames for the leveling preview while it is shown.
  async function previewLoop() {
    while (previewOn && !stopped) {
      try {
        const r = await fetch("/api/scan/photo/preview", { cache: "no-store" });
        if (!r.ok) { await new Promise((res) => setTimeout(res, 800)); continue; }
        const url = URL.createObjectURL(await r.blob());
        const img = $("#pv"); if (!img) { URL.revokeObjectURL(url); break; }
        img.src = url; if (previewUrl) URL.revokeObjectURL(previewUrl); previewUrl = url;
      } catch { await new Promise((res) => setTimeout(res, 800)); }
    }
  }

  // Reloads the scan state from the server and redraws, faster while capturing or scanning.
  async function poll() {
    clearTimeout(timer);
    try { const r = await fetch("/api/scan"); if (r.ok) st = await r.json(); } catch { /* keep polling */ }
    if (stopped) return;
    if (!busy) render(false);
    timer = setTimeout(poll, st.stage === "capturing" || st.stage === "surface_running" ? 700 : 1500);
  }

  (async () => { await loadPorts(); poll(); })();
  return () => { stopped = true; previewOn = false; clearTimeout(timer); removeEventListener("keydown", onKey); if (previewUrl) URL.revokeObjectURL(previewUrl); };
}
