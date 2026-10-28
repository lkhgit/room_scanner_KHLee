"""Processing steps. Each step runs one of the existing scripts as a subprocess and keeps its output in the room folder.

    photo branch:  stitch -> model -> sun        (needs photos or a panorama)
    surface branch: mold                          (needs the surface scan)
A failure in one branch never stops the other. Finished steps are not redone.
"""
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from datetime import date
from pathlib import Path

import yaml

from . import i18n, rooms
from .geometry import icosahedron
from .settings import KMA_CACHE, MOLD_DIR, ROOT, SUNLIGHT_DIR, CFG

MODEL_LOCK = threading.Semaphore(1)       # one heavy room-model run at a time (GPU memory)
_running = set()
_running_lock = threading.Lock()


def _claim(key) -> bool:
    with _running_lock:
        if key in _running:
            return False
        _running.add(key)
        return True


def _release(key) -> None:
    with _running_lock:
        _running.discard(key)


def run_script(script, args, cwd, log_path, on_line=None):
    """Runs a Python script; returns (returncode, tail of output). Output is also appended to log_path."""
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    cmd = [sys.executable, "-u", str(script), *map(str, args)]
    tail = []
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        p = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            log.write(line)
            log.flush()
            tail.append(line.rstrip())
            tail = tail[-40:]
            if on_line:
                on_line(line.rstrip())
        p.wait()
    return p.returncode, "\n".join(tail)


def _key(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:10]


# ------------------------------------------------------------------ start / restart
def start(room_id: str) -> None:
    r = rooms.load(room_id)
    if r["has"]["photos"] or r["has"]["panorama"]:
        threading.Thread(target=photo_branch, args=(room_id,), daemon=True).start()
    else:
        rooms.set_step(room_id, "stitch", "skipped", "No photos in this bundle.")
        rooms.set_step(room_id, "model", "skipped", "No photos in this bundle.")
        rooms.set_step(room_id, "sun", "skipped", "No photos in this bundle.")
    threading.Thread(target=mold_step, args=(room_id,), daemon=True).start()


def retry(room_id: str, step: str) -> None:
    if step in ("stitch", "model", "sun"):
        threading.Thread(target=photo_branch, args=(room_id, True), daemon=True).start()
    elif step == "mold":
        threading.Thread(target=mold_step, args=(room_id, True), daemon=True).start()


def params_changed(room_id: str, old: dict, new: dict) -> None:
    """Re-runs only what depends on what changed."""
    moved = (old.get("lat"), old.get("lon")) != (new.get("lat"), new.get("lon"))
    turned = old.get("heading_deg") != new.get("heading_deg")
    r = rooms.load(room_id)
    if moved:
        threading.Thread(target=mold_step, args=(room_id,), daemon=True).start()
    if (moved or turned) and r["steps"]["model"]["state"] == "done":
        threading.Thread(target=sun_step, args=(room_id,), daemon=True).start()


def recover_after_restart() -> None:
    """Steps that were running when the server stopped are marked so they can be retried."""
    for item in rooms.list_rooms():
        for s, st in item["steps"].items():
            if st["state"] == "running":
                rooms.set_step(item["id"], s, "failed", "The server restarted while this step was running.")


# ------------------------------------------------------------------ photo branch
def photo_branch(room_id: str, force: bool = False) -> None:
    if not _claim((room_id, "photo")):
        return
    try:
        stitch_step(room_id, force) and model_step(room_id, force) and sun_step(room_id)
    finally:
        _release((room_id, "photo"))


def _done(room_id, step, force):
    return not force and rooms.load(room_id)["steps"][step]["state"] == "done"


def stitch_step(room_id, force=False) -> bool:
    d = rooms.room_dir(room_id)
    work = d / "work" / "stitch"
    work.mkdir(parents=True, exist_ok=True)
    pano = next((p for p in work.glob("panorama.*") if "1024x512" not in p.name), None)
    if _done(room_id, "stitch", force) and pano:
        return True
    src = next((p for p in (d / "bundle").glob("panorama.*")), None)
    if src:
        shutil.copy(src, work / src.name)
        rooms.set_step(room_id, "stitch", "done", "Using the panorama from the bundle.")
        return True
    # OpenCV on Windows cannot open paths with non-ASCII characters (e.g. "메이커 최종"),
    # so the scripts work on a copy of the photos in an ASCII-only temp folder.
    with tempfile.TemporaryDirectory(prefix="rs_stitch_", dir=_ascii_tmp()) as tmp:
        photos = Path(tmp) / "photos"
        shutil.copytree(d / "bundle" / "photos", photos)
        if (d / "bundle" / "calib.json").exists():
            shutil.copy(d / "bundle" / "calib.json", photos / "calib.json")
        log = work / "stitch.log"
        notes = _calibrate(room_id, photos, log)
        rooms.set_step(room_id, "stitch", "running", "Stitching photos into a panorama...")
        rc, tail = run_script(SUNLIGHT_DIR / "stitch_panorama.py", [photos, "--out", "panorama.jpg"],
                              SUNLIGHT_DIR, log)
        if rc != 0 or not (photos / "panorama.jpg").exists():
            rooms.set_step(room_id, "stitch", "failed", "Stitching the photos failed.", tail)
            return False
        for f in photos.glob("panorama*.jpg"):
            shutil.move(str(f), work / f.name)
        for name in ("calib.json", "meta_refined.json"):          # kept next to the panorama for inspection
            if (photos / name).exists():
                shutil.copy(photos / name, work / name)
    rooms.set_step(room_id, "stitch", "done", f"Panorama ready ({'; '.join(notes)}).")
    return True


def _calibrate(room_id, photos, log):
    """The two steps that used to be run by hand before stitching:
        calibrate.py -> calib.json        camera model (field of view, optical centre, roll, lens distortion)
        refine.py    -> meta_refined.json each photo's real direction (motor sag, gear backlash)
    stitch_panorama.py picks both up from the photo folder. Files shipped in the bundle are kept as they are.
    A failure is not fatal (too few matching features in a plain room, for instance): stitching then falls back
    to the nominal field of view. Returns short notes for the step message."""
    if (photos / "meta_refined.json").exists():
        return ["using the photo corrections from the bundle"]
    notes = []
    if (photos / "calib.json").exists():
        notes.append("camera calibration from the bundle")
    else:
        rooms.set_step(room_id, "stitch", "running", "Calibrating the camera from the photos...")
        rc, _ = run_script(SUNLIGHT_DIR / "calibrate.py", [photos], SUNLIGHT_DIR, log)
        ok = rc == 0 and (photos / "calib.json").exists()
        notes.append("camera calibrated" if ok else "camera calibration skipped, see the log")
    rooms.set_step(room_id, "stitch", "running", "Correcting each photo's direction...")
    rc, _ = run_script(SUNLIGHT_DIR / "refine.py", [photos], SUNLIGHT_DIR, log)
    ok = rc == 0 and (photos / "meta_refined.json").exists()
    notes.append("photo directions corrected" if ok else "photo correction skipped, see the log")
    return notes


def _ascii_tmp():
    """A temp parent folder whose path is plain ASCII (None = the system default when that already is)."""
    if tempfile.gettempdir().isascii():
        return None
    p = Path(os.environ.get("SystemDrive", "C:") + "\\") / "Temp"
    p.mkdir(exist_ok=True)
    return str(p)


def panorama_path(room_id):
    d = rooms.room_dir(room_id) / "work" / "stitch"
    return next((p for p in d.glob("panorama.*") if "1024x512" not in p.name), None)


def model_dir(room_id):
    return rooms.room_dir(room_id) / "work" / "model" / "panorama"


def model_step(room_id, force=False) -> bool:
    mdir = model_dir(room_id)
    if _done(room_id, "model", force) and (mdir / "room.glb").exists():
        return True
    pano = panorama_path(room_id)
    if not pano:
        rooms.set_step(room_id, "model", "failed", "No panorama is available.")
        return False
    rooms.set_step(room_id, "model", "running", "Waiting for the room-model slot...")
    with MODEL_LOCK:
        rooms.set_step(room_id, "model", "running", "Building the 3D room model (this can take several minutes)...")
        h = rooms.load(room_id)["params"]["camera_height_m"]
        rc, tail = run_script(SUNLIGHT_DIR / "panorama to model_oneformer_v2.py",
                              [pano, "--out", pano.parent.parent / "model", "--cam_height", h],
                              SUNLIGHT_DIR, rooms.room_dir(room_id) / "work" / "model.log")
    if rc != 0 or not (mdir / "room.glb").exists():
        rooms.set_step(room_id, "model", "failed", "Building the room model failed.", tail)
        return False
    rooms.set_step(room_id, "model", "done", "Room model ready.")
    return True


# ------------------------------------------------------------------ sunlight (re-run when conditions change)
def season_dates(year):
    return [date(year, 12, 21), date(year, 3, 20), date(year, 6, 21), date(year, 9, 22)]


def sun_dir(room_id, params, day):
    k = _key(round(params["lat"], 4), round(params["lon"], 4), params.get("heading_deg") or 0, day)
    return rooms.room_dir(room_id) / "work" / "sun" / k


def sun_step(room_id, day=None) -> bool:
    r = rooms.load(room_id)
    p = r["params"]
    if p["lat"] is None or p["lon"] is None:
        rooms.set_step(room_id, "sun", "waiting", "Set a location to calculate sunlight.")
        return False
    day = day or date.today().replace(month=12, day=21).isoformat()
    out = sun_dir(room_id, p, day)
    if (out / "sun_viewer.html").exists():
        rooms.set_step(room_id, "sun", "done", "Sunlight ready.")
        return True
    if not _claim((room_id, "sun", out.name)):
        return False
    try:
        rooms.set_step(room_id, "sun", "running", "Calculating sunlight...")
        out.mkdir(parents=True, exist_ok=True)
        sc = CFG["sun"]
        args = [model_dir(room_id), "--heading", p.get("heading_deg") or 0, "--lat", p["lat"], "--lon", p["lon"],
                "--date", day, "--step", sc["step_min"], "--grid", sc["grid_m"], "--out", out]
        rc, tail = run_script(ROOT / "tools" / "sun_runner.py", args, SUNLIGHT_DIR, out / "sun.log")
        if rc != 0 or not (out / "sun_viewer.html").exists():
            rooms.set_step(room_id, "sun", "failed", "The sunlight calculation failed.", tail)
            return False
        rooms.set_step(room_id, "sun", "done", "Sunlight ready.")
        return True
    finally:
        _release((room_id, "sun", out.name))


# ------------------------------------------------------------------ mold
def sim_year():
    return CFG["mold"].get("sim_year") or date.today().year - 1


def mold_dir(room_id, p):
    return rooms.room_dir(room_id) / "work" / "mold" / _key(round(p["lat"], 4), round(p["lon"], 4), sim_year())


def mold_result(room_id):
    p = rooms.load(room_id)["params"]
    if p["lat"] is None or p["lon"] is None:
        return None
    f = mold_dir(room_id, p) / "sim" / f"viewer_{sim_year()}.json"
    return f if f.exists() else None


def mold_step(room_id, force=False) -> bool:
    r = rooms.load(room_id)
    p = r["params"]
    if p["lat"] is None or p["lon"] is None:
        rooms.set_step(room_id, "mold", "waiting", "Set a location to run the mold simulation.")
        return False
    out = mold_dir(room_id, p)
    if not force and mold_result(room_id):
        rooms.set_step(room_id, "mold", "done", "Mold simulation ready.")
        return True
    if not _claim((room_id, "mold")):
        return False
    try:
        rooms.set_step(room_id, "mold", "running", "Running the mold simulation (first run downloads weather data)...")
        if out.exists():
            shutil.rmtree(out)
        sess = out / "session"
        shutil.copytree(rooms.room_dir(room_id) / "bundle" / "surface", sess)
        data = yaml.safe_load((sess / "session.yaml").read_text(encoding="utf-8")) or {}
        data.setdefault("site", {}).update({"lat": p["lat"], "lon": p["lon"]})
        (sess / "session.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        args = ["--session-dir", sess, "--out-dir", out / "sim", "--cache-dir", KMA_CACHE, "--years", sim_year()]
        rc, tail = run_script(MOLD_DIR / "mold_simulator.py", args, MOLD_DIR, out / "mold.log")
        if rc != 0 or not mold_result(room_id):
            msg = "The mold simulation failed."
            if "KMA_SERVICE_KEY" in tail:
                msg = "The weather-service key is not set, so the mold simulation cannot download weather data."
            rooms.set_step(room_id, "mold", "failed", msg, tail)
            return False
        rooms.set_step(room_id, "mold", "done", "Mold simulation ready.")
        return True
    finally:
        _release((room_id, "mold"))


def mold_payload(room_id):
    """The viewer JSON with every on-screen string in English, plus the icosahedron geometry."""
    f = mold_result(room_id)
    if not f:
        return None
    v = json.loads(f.read_text(encoding="utf-8"))
    v["warnings"] = [i18n.warning(w) for w in v.get("warnings", [])]
    v["provisional_message"] = i18n.provisional(v.get("provisional_message"))
    st = v.get("station") or {}
    v["station"] = {"id": st.get("id"), "distance_km": st.get("distance_km")}
    v.pop("site", None)
    v["geometry"] = icosahedron(v["faces"])
    return _finite(v)


def _finite(x):
    """NaN/inf -> None. The simulator writes NaN for events that never happen (e.g. t_M3 when M never reaches 3),
    and NaN is not valid JSON for the browser."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_finite(v) for v in x]
    return x
