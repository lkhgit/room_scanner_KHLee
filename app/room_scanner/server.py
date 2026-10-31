"""Room Scanner local server. Run: python server.py  (then open http://localhost:8000)"""
import gzip
import json
import tempfile
import threading
from datetime import date
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory

from rs import pipeline, places, rooms, scanner
from rs.bundle import BundleError
from rs.settings import CFG, ROOT

app = Flask(__name__, static_folder=str(ROOT / "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 ** 3


def room_or_404(room_id):
    """Loads a room by id, or answers 404 if it does not exist."""
    try:
        return rooms.load(room_id)
    except (KeyError, OSError):
        abort(404, "Room not found.")


@app.errorhandler(404)
def not_found(e):
    """Returns API 404s as JSON and leaves other 404s as they are."""
    if request.path.startswith("/api/"):
        return jsonify(error=getattr(e, "description", "Not found.")), 404
    return e


@app.get("/")
def index():
    """Serves the single-page app."""
    return send_from_directory(app.static_folder, "index.html")


@app.get("/flood/")
def flood_index():
    """Serves the flood map page."""
    return send_from_directory(ROOT / "static" / "flood", "index.html")


@app.get("/flood/<path:p>")
def flood_files(p):
    """Serves the flood map's scripts and data files."""
    return send_from_directory(ROOT / "static" / "flood", p)


@app.get("/api/places")
def place_search():
    """Searches place names for the flood tab through the Kakao API."""
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])
    try:
        return jsonify(places.search(q))
    except places.PlaceError as e:
        return jsonify(error=str(e)), 502


@app.post("/api/upload")
def upload():
    """Takes an uploaded scan bundle, creates a room from it and starts processing."""
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(problems=["No file was uploaded."]), 400
    if not f.filename.lower().endswith(".zip"):
        return jsonify(problems=["Please upload a .zip scan bundle."]), 400
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "upload.zip"
        f.save(path)
        try:
            room, warnings = rooms.create(path, f.filename)
        except BundleError as e:
            return jsonify(problems=e.problems), 400
    pipeline.start(room["id"])
    return jsonify(id=room["id"], warnings=warnings)


# ------------------------------------------------------------------ scan wizard
@app.errorhandler(scanner.ScanError)
def scan_error(e):
    """Returns scan wizard errors as JSON with the current scan state."""
    return jsonify(error=str(e), scan=scanner.state()), 409


def _num(v, lo, hi):
    """Parses an optional number and checks that it lies between lo and hi."""
    if v in (None, ""):
        return None
    x = float(v)
    if not lo <= x <= hi:
        raise ValueError
    return x


@app.get("/api/scan")
def scan_state():
    """Returns the current scan wizard state."""
    return jsonify(scanner.state())


@app.get("/api/scan/ports")
def scan_ports():
    """Lists the serial ports the scan modules can be connected to."""
    try:
        return jsonify(scanner.ports())
    except ImportError:
        return jsonify(error="pyserial is not installed. Run: pip install -r requirements.txt"), 500


@app.post("/api/scan/start")
def scan_start():
    """Starts a new scan with the room details from the form."""
    b = request.get_json(force=True, silent=True) or {}
    try:
        info = {"name": (b.get("name") or "").strip(), "site": (b.get("site") or "").strip(),
                "lat": _num(b.get("lat"), -90, 90), "lon": _num(b.get("lon"), -180, 180),
                "heading_deg": _num(b.get("heading_deg"), -360, 360),
                "camera_height_m": _num(b.get("camera_height_m"), 0.2, 3) or 1.2}
    except (TypeError, ValueError):
        return jsonify(error="Check the numbers: latitude -90 to 90, longitude -180 to 180, "
                             "heading in degrees, camera height 0.2 to 3 m."), 400
    if (info["lat"] is None) != (info["lon"] is None):
        return jsonify(error="Enter both latitude and longitude, or neither."), 400
    if info["heading_deg"] is not None:
        info["heading_deg"] %= 360
    return jsonify(scanner.start(info))


@app.post("/api/scan/cancel")
def scan_cancel():
    """Cancels the scan in progress."""
    scanner.cancel()
    return jsonify(scanner.state())


@app.post("/api/scan/photo/connect")
def scan_photo_connect():
    """Connects the pan-tilt rig and the camera for the photo step."""
    b = request.get_json(force=True, silent=True) or {}
    if not b.get("arduino") or not b.get("camera"):
        return jsonify(error="Choose the rig port and the camera port."), 400
    if b["arduino"] == b["camera"]:
        return jsonify(error="The rig and the camera must be on different ports."), 400
    scanner.photo_connect(b["arduino"], b["camera"])
    return jsonify(scanner.state())


@app.get("/api/scan/photo/preview")
def scan_preview():
    """Returns one live camera frame for the leveling preview."""
    try:
        data = scanner.preview()
    except scanner.ScanError:
        raise
    except Exception as e:
        return jsonify(error=f"Could not read the camera: {e}"), 502
    return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/api/scan/photo/move")
def scan_move():
    """Moves the rig by one preset step on the pan or tilt axis."""
    b = request.get_json(force=True, silent=True) or {}
    if b.get("axis") not in ("pan", "tilt") or b.get("size") not in scanner.STEP_PRESETS:
        return jsonify(error="Unknown move."), 400
    try:
        scanner.move(b["axis"], b["size"], 1 if float(b.get("dir", 1)) > 0 else -1)
    except scanner.ScanError:
        raise
    except Exception as e:
        return jsonify(error=f"The rig did not respond: {e}"), 502
    return jsonify(scanner.state())


@app.post("/api/scan/photo/capture")
def scan_capture():
    """Starts the automatic photo capture."""
    scanner.capture()
    return jsonify(scanner.state())


@app.post("/api/scan/photo/stop")
def scan_photo_stop():
    """Stops the photo capture."""
    scanner.photo_stop()
    return jsonify(scanner.state())


@app.post("/api/scan/photo/skip")
def scan_photo_skip():
    """Skips the photo step."""
    scanner.photo_skip()
    return jsonify(scanner.state())


@app.get("/api/scan/photo/last")
def scan_last_photo():
    """Returns the most recently captured photo."""
    p = scanner.last_photo()
    if not p or not p.exists():
        abort(404, "No photo yet.")
    return send_file(p, mimetype="image/jpeg", max_age=0)


@app.post("/api/scan/surface/start")
def scan_surface_start():
    """Starts the surface scan on the chosen port."""
    b = request.get_json(force=True, silent=True) or {}
    if not b.get("port"):
        return jsonify(error="Choose the surface module port."), 400
    scanner.surface_start(b["port"], b.get("material"))
    return jsonify(scanner.state())


@app.post("/api/scan/surface/stop")
def scan_surface_stop():
    """Stops the surface scan."""
    scanner.surface_stop()
    return jsonify(scanner.state())


@app.post("/api/scan/finish")
def scan_finish():
    """Turns the finished scan into a room."""
    try:
        room_id = scanner.finish()
    except BundleError as e:
        return jsonify(error="The scan could not be turned into a room: " + "; ".join(e.problems)), 400
    return jsonify(id=room_id)


@app.get("/api/rooms")
def list_rooms():
    """Lists all rooms."""
    return jsonify(rooms.list_rooms())


@app.get("/api/rooms/<room_id>")
def get_room(room_id):
    """Returns one room with today's date and the mold simulation year."""
    r = room_or_404(room_id)
    r["today"] = date.today().isoformat()
    r["sim_year"] = pipeline.sim_year()
    return jsonify(r)


@app.post("/api/rooms/<room_id>/params")
def set_params(room_id):
    """Updates a room's location or heading and recalculates what depends on it."""
    old = room_or_404(room_id)["params"]
    body = request.get_json(force=True, silent=True) or {}
    new = dict(old)
    try:
        if "lat" in body or "lon" in body:
            lat, lon = float(body["lat"]), float(body["lon"])
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError
            new["lat"], new["lon"] = lat, lon
        if "heading_deg" in body:
            new["heading_deg"] = None if body["heading_deg"] in (None, "") else float(body["heading_deg"]) % 360
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Enter a latitude between -90 and 90, a longitude between -180 and 180, "
                             "and a heading in degrees."), 400
    if new == old:
        return jsonify(new)
    rooms.update(room_id, lambda r: r["params"].update(new))
    pipeline.params_changed(room_id, old, new)
    return jsonify(new)


@app.post("/api/rooms/<room_id>/retry/<step>")
def retry(room_id, step):
    """Runs one processing step of a room again."""
    room_or_404(room_id)
    if step not in rooms.STEPS:
        abort(404, "Unknown step.")
    pipeline.retry(room_id, step)
    return jsonify(ok=True)


@app.get("/api/rooms/<room_id>/mold")
def mold(room_id):
    """Returns the mold result of a room as gzipped JSON."""
    room_or_404(room_id)
    data = pipeline.mold_payload(room_id)
    if data is None:
        abort(404, "The mold result is not ready.")
    body = gzip.compress(json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), 5)
    return Response(body, mimetype="application/json", headers={"Content-Encoding": "gzip"})


@app.get("/api/rooms/<room_id>/panorama")
def panorama(room_id):
    """Returns the room's panorama image."""
    room_or_404(room_id)
    p = pipeline.panorama_path(room_id)
    if not p:
        abort(404, "No panorama yet.")
    return send_file(p, max_age=60)


@app.get("/api/rooms/<room_id>/model.glb")
def room_model(room_id):
    """Returns the room's 3D model."""
    room_or_404(room_id)
    p = pipeline.model_dir(room_id) / "room.glb"
    if not p.exists():
        abort(404, "No room model yet.")
    return send_file(p, mimetype="model/gltf-binary", max_age=60)


@app.post("/api/rooms/<room_id>/sun")
def sun_request(room_id):
    """Starts the sunlight calculation for one date in the background."""
    r = room_or_404(room_id)
    day = (request.get_json(force=True, silent=True) or {}).get("date", "")
    try:
        date.fromisoformat(day)
    except ValueError:
        return jsonify(error="Enter the date as YYYY-MM-DD."), 400
    if r["steps"]["model"]["state"] != "done":
        return jsonify(error="The room model is not ready yet."), 409
    threading.Thread(target=pipeline.sun_step, args=(room_id, day), daemon=True).start()
    return jsonify(ok=True)


@app.get("/api/rooms/<room_id>/sun/ready")
def sun_ready(room_id):
    """Tells whether the sunlight result for a date is ready."""
    r = room_or_404(room_id)
    day = request.args.get("date", "")
    return jsonify(ready=(pipeline.sun_dir(room_id, r["params"], day) / "sun_viewer.html").exists()
                   if r["params"]["lat"] is not None else False)


@app.get("/rooms/<room_id>/sun-view")
def sun_view(room_id):
    """Serves the sunlight viewer page for a date."""
    r = room_or_404(room_id)
    f = pipeline.sun_dir(room_id, r["params"], request.args.get("date", "")) / "sun_viewer.html"
    if not f.exists():
        abort(404)
    return send_file(f, mimetype="text/html")


@app.get("/api/rooms/<room_id>/log/<step>")
def step_log(room_id, step):
    """Returns the log text of one processing step."""
    r = room_or_404(room_id)
    return jsonify(detail=r["steps"].get(step, {}).get("detail", ""))


if __name__ == "__main__":
    pipeline.recover_after_restart()
    s = CFG["server"]
    app.run(host=s["host"], port=s["port"], threaded=True)
