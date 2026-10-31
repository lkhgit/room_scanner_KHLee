// Home and Results screens. All on-screen text is English.
import { mountMold } from "/static/mold.js";
import { mountScan } from "/static/scan.js";

const app = document.getElementById("app");
let cleanup = null;
// Returns the first element that matches a CSS selector.
const $ = (s, r = document) => r.querySelector(s);
// Escapes text for safe use inside HTML.
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
// Fetches a JSON API and throws an Error with the server's message if the request fails.
const api = async (url, opts) => {
  const r = await fetch(url, opts);
  const body = await r.json().catch(() => null);
  if (!r.ok) throw Object.assign(new Error(body?.error || `Request failed (${r.status})`), { body });
  if (body === null) throw new Error(`The server's reply to ${url} could not be read.`);
  return body;
};

// Shows the screen for the current URL hash (home, scan wizard or a room's results).
function route() {
  if (cleanup) { cleanup(); cleanup = null; }
  const m = location.hash.match(/^#\/room\/([0-9a-f]{12})/);
  if (m) results(m[1]); else if (location.hash === "#/scan") scan(); else home();
}
addEventListener("hashchange", route);
route();

// ------------------------------------------------------------------ Home
// Shows the home screen: scan and upload options and the list of previous rooms.
async function home() {
  app.innerHTML = `<div class="page">
    <h1>Check a room before you rent it</h1>
    <p class="lead">Scan a room, or upload a scan bundle, and see sunlight, mold risk and flood risk for the same room on one screen.</p>
    <div class="start">
      <div class="card"><h2>Scan a room</h2>
        <p class="dim">Uses the scanning rig connected to this computer.</p>
        <a class="btn primary" href="#/scan" style="display:inline-block;text-decoration:none">Start Room Scan</a></div>
      <div class="card"><h2>Upload a scan</h2>
        <div class="drop" id="drop">Drop a <b>.zip</b> scan bundle here<br><span class="dim">or</span><br>
          <button class="btn" id="pick">Upload Scan (.zip)</button><input type="file" id="file" accept=".zip" hidden></div>
      </div>
    </div>
    <div id="msg"></div>
    <div class="card" id="prev" hidden><h2>Previous rooms</h2><ul class="rooms" id="rooms"></ul></div></div>`;
  const msg = $("#msg"), drop = $("#drop"), file = $("#file");
  // Uploads a scan bundle and opens the new room, or shows what is wrong with the file.
  async function send(f) {
    msg.innerHTML = `<div class="notice"><span class="spin"></span>Checking ${esc(f.name)}…</div>`;
    const fd = new FormData(); fd.append("file", f);
    try {
      const r = await fetch("/api/upload", { method: "POST", body: fd });
      const b = await r.json();
      if (!r.ok) {
        msg.innerHTML = `<div class="notice bad"><b>This file can't be used.</b><ul>${(b.problems || [b.error]).map((p) => `<li>${esc(p)}</li>`).join("")}</ul></div>`;
        return;
      }
      location.hash = `#/room/${b.id}`;
    } catch (e) { msg.innerHTML = `<div class="notice bad">Upload failed: ${esc(e.message)}</div>`; }
  }
  $("#pick").onclick = () => file.click();
  file.onchange = () => file.files[0] && send(file.files[0]);
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); e.dataTransfer.files[0] && send(e.dataTransfer.files[0]); };
  const list = await api("/api/rooms").catch(() => []);
  if (list.length) {
    $("#prev").hidden = false;
    $("#rooms").innerHTML = list.map((r) => `<li><a href="#/room/${r.id}">${esc(r.name)}</a>
      <span class="dim">${new Date(r.created_at).toLocaleString("en-US", { dateStyle: "medium", timeStyle: "short" })}</span></li>`).join("");
  }
}

// Shows the scan wizard.
function scan() {
  cleanup = mountScan(app);
}

// ------------------------------------------------------------------ Results
const STEP_LABEL = { stitch: "Stitch photos into a panorama", model: "Build the 3D room model", sun: "Calculate sunlight", mold: "Run the mold simulation" };
const TAB_STEPS = { sun: ["stitch", "model", "sun"], mold: ["mold"], flood: [] };

// Shows a room's results screen and keeps it updated while steps run.
async function results(id) {
  let room;
  try { room = await api(`/api/rooms/${id}`); } catch { app.innerHTML = `<div class="page"><div class="notice bad">This room was not found.</div><a href="#/">Back to Home</a></div>`; return; }
  const ui = { tab: sessionStorage.getItem("tab") || "mold", mold: null, moldKey: null, sunDate: null, sunKey: null, floodKey: null };
  app.innerHTML = `
    <div class="bar" id="bar"></div>
    <div class="tabs" id="tabs"></div>
    <div id="panes"><div class="pane" id="p-sun"></div><div class="pane" id="p-mold"></div><div class="pane" id="p-flood"></div></div>`;

  // Draws the bar with the room name, location and north heading, and wires the Apply button.
  function renderBar() {
    const p = room.params;
    $("#bar").innerHTML = `<span class="name">${esc(room.name)}</span>
      <label>Latitude<input id="lat" value="${p.lat ?? ""}"></label>
      <label>Longitude<input id="lon" value="${p.lon ?? ""}"></label>
      <label>North heading (°)<input id="hd" value="${p.heading_deg ?? ""}" placeholder="0"></label>
      <button class="btn" id="apply">Apply</button>
      <span class="conf dim" id="barmsg">${p.heading_deg == null ? "No north heading in this scan; 0° is assumed." : ""}</span>`;
    $("#apply").onclick = async () => {
      $("#barmsg").textContent = "Applying…";
      try {
        const body = { heading_deg: $("#hd").value };
        if ($("#lat").value !== "" && $("#lon").value !== "") { body.lat = $("#lat").value; body.lon = $("#lon").value; }
        const np = await api(`/api/rooms/${id}/params`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
        room.params = np; $("#barmsg").textContent = "Applied. Affected results are recalculating.";
        poll();
      } catch (e) { $("#barmsg").textContent = e.message; }
    };
  }
  // Draws the Sunlight / Mold / Flood tabs with a status dot for each.
  function renderTabs() {
    // Returns the states of the steps a tab depends on.
    const st = (t) => TAB_STEPS[t].map((s) => room.steps[s].state);
    // Picks the status dot class for a tab from its steps' states.
    const dot = (t) => { const s = st(t); return s.includes("failed") ? "failed" : s.includes("running") || s.includes("pending") ? "running" : s.length && s.every((x) => x === "done") ? "done" : ""; };
    $("#tabs").innerHTML = [["sun", "Sunlight"], ["mold", "Mold"], ["flood", "Flood risk"]].map(([t, l]) =>
      `<button class="tab ${ui.tab === t ? "on" : ""}" data-t="${t}">${l}<span class="dot ${dot(t)}"></span></button>`).join("");
    document.querySelectorAll(".tab").forEach((b) => b.onclick = () => { ui.tab = b.dataset.t; sessionStorage.setItem("tab", ui.tab); renderTabs(); renderPanes(); });
  }

  // Returns the HTML for a list of processing steps, with Retry buttons for failed ones.
  function progress(steps, extra = "") {
    // Returns the icon for a step state.
    const icon = (s) => ({ done: "✓", running: '<span class="spin"></span>', failed: "✕", skipped: "–", waiting: "…" }[s] || "·");
    const failed = steps.filter((s) => room.steps[s].state === "failed");
    return `<div class="status"><h2>${steps.length ? "Working on this room" : ""}</h2><div class="steps">${steps.map((s) => {
      const x = room.steps[s]; return `<div>${icon(x.state)} <b>${STEP_LABEL[s]}</b><div class="dim" style="margin-left:22px">${esc(x.message || (x.state === "pending" ? "Waiting to start" : ""))}</div></div>`; }).join("")}</div>
      ${failed.map((s) => `<div class="notice bad">${esc(room.steps[s].message)}
        <div style="margin-top:6px"><button class="btn" data-retry="${s}">Retry</button></div>
        ${room.steps[s].detail ? `<details><summary class="dim">Technical details</summary><pre>${esc(room.steps[s].detail)}</pre></details>` : ""}</div>`).join("")}${extra}</div>`;
  }
  // Wires the Retry buttons inside a pane.
  function bindRetry(pane) {
    pane.querySelectorAll("[data-retry]").forEach((b) => b.onclick = async () => { await api(`/api/rooms/${id}/retry/${b.dataset.retry}`, { method: "POST" }); poll(); });
  }

  // Shows the mold viewer once the simulation is done, otherwise its progress.
  async function renderMold() {
    const pane = $("#p-mold"), s = room.steps.mold;
    const key = `${s.state}|${room.params.lat}|${room.params.lon}|${room.steps.model.state}`;
    if (s.state === "done") {
      if (ui.moldKey === key) { ui.mold && ui.mold.setHeading(); return; }
      ui.mold && ui.mold.destroy(); pane.innerHTML = "";
      ui.moldKey = key;
      try {
        const data = await api(`/api/rooms/${id}/mold`);
        ui.mold = mountMold(pane, id, data, room);
      } catch (e) { pane.innerHTML = `<div class="status"><div class="notice bad">${esc(e.message)}</div></div>`; }
    } else {
      ui.mold && ui.mold.destroy(); ui.mold = null; ui.moldKey = null;
      pane.innerHTML = progress(["mold"]); bindRetry(pane);
    }
  }

  // Shows the sunlight viewer with date presets, or the reason it is not available yet.
  function renderSun() {
    const pane = $("#p-sun");
    if (room.steps.model.state === "skipped") { pane.innerHTML = `<div class="status"><div class="notice warn">This scan has no photos, so sunlight cannot be calculated.</div></div>`; return; }
    if (room.steps.model.state !== "done") { pane.innerHTML = progress(["stitch", "model", "sun"]); bindRetry(pane); ui.sunKey = null; return; }
    if (room.params.lat == null) { pane.innerHTML = `<div class="status"><div class="notice warn">Set a location in the bar above to calculate sunlight.</div></div>`; ui.sunKey = null; return; }
    const key = `${room.params.lat}|${room.params.lon}|${room.params.heading_deg}`;
    if (ui.sunKey === key) return;
    ui.sunKey = key;
    const y = new Date().getFullYear();
    const presets = [["Winter solstice", `${y}-12-21`], ["Spring equinox", `${y}-03-20`], ["Summer solstice", `${y}-06-21`], ["Autumn equinox", `${y}-09-22`]];
    ui.sunDate = ui.sunDate || presets[0][1];
    pane.innerHTML = `<div style="display:flex;flex-direction:column;width:100%">
      <div class="bar"><label>Date<input type="date" id="sd" value="${ui.sunDate}"></label>
        ${presets.map(([l, d]) => `<button class="btn" data-d="${d}">${l}</button>`).join("")}
        <span class="conf dim" id="sunmsg"></span></div>
      <div style="flex:1;min-height:0;position:relative"><iframe id="sunframe" title="Sunlight"></iframe></div></div>`;
    // Requests the sunlight result for a date and loads it into the viewer when ready.
    const load = async (d) => {
      ui.sunDate = d; $("#sd").value = d; $("#sunmsg").innerHTML = '<span class="spin"></span>Recalculating…';
      $("#sunframe").style.opacity = ".35";
      try {
        await api(`/api/rooms/${id}/sun`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ date: d }) });
        for (let i = 0; i < 600; i++) {
          const r = await api(`/api/rooms/${id}/sun/ready?date=${d}`);
          if (r.ready) { if (ui.sunDate !== d) return; $("#sunframe").src = `/rooms/${id}/sun-view?date=${d}`; $("#sunframe").style.opacity = ""; $("#sunmsg").textContent = ""; return; }
          if (room.steps.sun.state === "failed") break;
          await new Promise((r) => setTimeout(r, 1000));
        }
        $("#sunmsg").textContent = "The sunlight calculation failed. See the Sunlight step for details.";
      } catch (e) { $("#sunmsg").textContent = e.message; }
    };
    $("#sd").onchange = (e) => e.target.value && load(e.target.value);
    pane.querySelectorAll("[data-d]").forEach((b) => b.onclick = () => load(b.dataset.d));
    load(ui.sunDate);
  }

  // Shows the flood map centred on the room's location.
  function renderFlood() {
    const pane = $("#p-flood"), p = room.params;
    const key = `${p.lat}|${p.lon}`;
    if (ui.floodKey === key) return;
    ui.floodKey = key;
    const q = p.lat != null ? `?lat=${p.lat}&lng=${p.lon}&label=${encodeURIComponent("Scan location")}` : "";
    pane.innerHTML = `<iframe title="Flood risk" src="/flood/${q}"></iframe>`;
  }

  // Shows the selected tab's pane and renders it.
  function renderPanes() {
    ["sun", "mold", "flood"].forEach((t) => $(`#p-${t}`).classList.toggle("on", ui.tab === t));
    if (ui.tab === "mold") renderMold(); else if (ui.tab === "sun") renderSun(); else renderFlood();
  }

  let timer = null, stopped = false;
  // Reloads the room state and redraws, polling faster while steps are running.
  async function poll() {
    clearTimeout(timer);
    try { room = await api(`/api/rooms/${id}`); } catch { /* keep polling */ }
    if (stopped) return;
    renderTabs(); renderPanes();
    const busy = Object.values(room.steps).some((s) => s.state === "running" || s.state === "pending");
    timer = setTimeout(poll, busy ? 2000 : 8000);
  }
  renderBar();
  const notices = room.warnings.length ? room.warnings : [];
  if (notices.length) {
    const b = document.createElement("div"); b.className = "notice warn"; b.style.margin = "0";
    b.innerHTML = notices.map(esc).join("<br>"); $("#bar").after(b);
  }
  renderTabs(); renderPanes();
  poll();
  cleanup = () => { stopped = true; clearTimeout(timer); ui.mold && ui.mold.destroy(); };
}
