"""Scan wizard: drives the two existing devices from the server, then turns the result into a room.

    photo module    rig_motor.ino (Arduino) + xiao_cam_serial.ino (XIAO)  -> ../sunlight_scanner/panorama.py classes
    surface module  surface_scan.ino (Arduino)                            -> ../mold_scanner/logger.py classes
The two modules are used one after the other. Raw data is kept in scans/<id>/ next to a bundle.zip
that can be uploaded again later.
"""
import importlib
import json
import sys
import threading
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from . import pipeline, rooms
from .settings import MOLD_DIR, ROOT, SUNLIGHT_DIR

SCANS_DIR = ROOT / "scans"
STEP_PRESETS = {"fine": 4, "medium": 23, "coarse": 114}   # same as panorama.py keys 1/2/3 (~0.35°, 2°, 10°)
READY_TIMEOUT_S = 20

_lock = threading.RLock()       # guards _s
_dev = threading.Lock()         # one serial conversation at a time with the photo module
_s = None


def _script(name, folder):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    return importlib.import_module(name)


def ports():
    from serial.tools import list_ports
    return [{"device": p.device, "description": p.description} for p in sorted(list_ports.comports())]


class ScanError(Exception):
    pass


def _need(stage):
    if _s is None:
        raise ScanError("No scan is in progress.")
    if _s["stage"] not in stage:
        raise ScanError("The scan is not at the right step for this action.")


def _log(msg):
    with _lock:
        if _s is not None:
            _s["log"].append(f"[{datetime.now():%H:%M:%S}] {msg}")
            del _s["log"][:-200]


def state():
    with _lock:
        if _s is None:
            return {"stage": None}
        out = {k: v for k, v in _s.items() if not k.startswith("_")}
        out["log"] = out["log"][-40:]
        return out


# ------------------------------------------------------------------ session
def start(info):
    global _s
    with _lock:
        if _s is not None and _s["stage"] not in ("done",):
            raise ScanError("A scan is already in progress. Cancel it first.")
        sid = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]
        d = SCANS_DIR / sid
        d.mkdir(parents=True)
        _s = {"id": sid, "dir": str(d), "stage": "photo_connect", "info": info, "log": [], "error": None,
              "photo": {"shot": 0, "total": 0, "plan": "", "count": 0, "pitch": 0.0, "yaw": 0.0,
                        "compass": None, "azimuth": None},
              "surface": {"faces": 0, "state": "", "heading": None},
              "_rig": None, "_cam": None, "_abort": None, "_ser": None, "_north": None}
    return state()


def cancel():
    global _s
    with _lock:
        if _s is None:
            return
        if _s["_abort"]:
            _s["_abort"].set()
        if _s["_ser"]:
            try:
                _s["_ser"].write(b"s")
            except Exception:
                pass
    _close_photo()
    with _lock:
        _s = None


def _fail(msg, exc=None):
    msg += f" ({exc})" if exc else ""
    with _lock:
        if _s is not None:
            _s["error"] = msg
    _log("Error: " + msg)


# ------------------------------------------------------------------ photo module
def photo_connect(arduino, camera):
    with _lock:
        _need(("photo_connect",))
        _s.update(error=None, stage="photo_connecting")
    rig = cam = None
    try:
        pano = _script("panorama", SUNLIGHT_DIR)
        with _dev:
            _log(f"Connecting to the rig on {arduino}...")
            rig = pano.Rig(arduino)
            # True north comes from the compass sensor on the rig, measured before any motor moves
            # (the camera must be aimed along the sensor's X axis at this moment), as panorama.py does.
            north, compass = None, {"sensor": bool(rig.imu_ok)}
            if rig.imu_ok:
                north = rig.measure_heading()
                if north:
                    az = (north["heading"] + pano.MAG_DECLINATION_DEG) % 360
                    compass.update(azimuth=round(az, 1), spread=round(north["spread"], 1), mag_calib=north["cal"][3])
                    _log(f"North measured before the motors moved: camera faces {az:.1f}° "
                         f"(±{north['spread']:.1f}°, compass calibration {north['cal'][3]}/3).")
                    if north["spread"] > 2:
                        _log("The reading is unsteady: move metal and electronics away from the rig and connect again.")
                else:
                    _log("The compass sensor could not be read, so north is not recorded.")
            else:
                _log("No compass sensor found on the rig: north is not recorded.")
            _log(f"Connecting to the camera on {camera}...")
            cam = pano.Camera(camera)
            cam.set_size(pano.PREVIEW_SIZE)
    except Exception as e:
        for dev in (rig, cam):
            if dev:
                dev.close()
        with _lock:
            if _s is not None:
                _s["stage"] = "photo_connect"
        raise ScanError(f"Could not connect to the photo module: {e}")
    with _lock:
        _s.update(_rig=rig, _cam=cam, _north=north, stage="leveling")
        _s["photo"].update(compass=compass, azimuth=compass.get("azimuth"), pitch=0.0, yaw=0.0)
    _log("Photo module connected. Level the camera, then start the capture.")


def preview():
    with _lock:
        _need(("leveling",))
        cam = _s["_cam"]
    with _dev:
        return cam.grab_jpeg()


def move(axis, size, direction):
    pano = _script("panorama", SUNLIGHT_DIR)
    with _lock:
        _need(("leveling",))
        rig = _s["_rig"]
    n = STEP_PRESETS[size] * (1 if direction > 0 else -1)
    with _dev:
        (rig.move_pan if axis == "pan" else rig.move_tilt)(n)
    with _lock:
        _s["photo"].update(pitch=round(pano.steps_to_deg(rig.tilt), 2), yaw=round(pano.steps_to_deg(rig.pan), 2))
        start = _s["_north"]
        if start:       # same readout as panorama.py's leveling window
            _s["photo"]["azimuth"] = round((start["heading"] + pano.MAG_DECLINATION_DEG + pano.steps_to_deg(rig.pan)) % 360, 1)


def capture():
    with _lock:
        _need(("leveling",))
        _s["stage"] = "capturing"
        _s["_abort"] = threading.Event()
    threading.Thread(target=_capture, daemon=True).start()


def _capture():
    """panorama.capture() without the OpenCV windows; writes the same JPEG names and meta.json."""
    pano = _script("panorama", SUNLIGHT_DIR)
    with _lock:
        rig, cam, abort, start = _s["_rig"], _s["_cam"], _s["_abort"], _s["_north"]
        out = Path(_s["dir"]) / "photos"
    try:
        with _dev:
            # the pan done while levelling counts: the spot confirmed here becomes yaw 0°
            north = pano.north_info(start, pano.steps_to_deg(rig.pan), rig.imu_calib_loaded) if start else None
            rig.set_zero()
            if north:
                _log(f"North: the camera faces {north['camera_azimuth_at_yaw0_deg']:.1f}° at yaw 0°.")
            rows = pano.plan_rows()
            total = sum(n for _, n in rows)
            plan = ", ".join(f"{p:+d}° x {n}" for p, n in rows)
            with _lock:
                _s["photo"].update(total=total, plan=plan, shot=0)
            _log(f"Capture plan: {plan} ({total} photos)")
            cam.set_size(pano.CAPTURE_SIZE)
            out.mkdir(parents=True, exist_ok=True)
            meta = {"hfov_deg": pano.HFOV_DEG, "vfov_deg": pano.VFOV_DEG,
                    "steps_per_rev": pano.STEPS_PER_REV, "north": north, "images": []}
            shot = 0
            for ri, (pitch, n) in enumerate(rows):
                rig.tilt_to(pitch)
                for ci in range(n):
                    if abort.is_set():
                        raise InterruptedError
                    rig.pan_to(ci * 360.0 / n)
                    time.sleep(pano.SETTLE_S)
                    data = cam.grab_jpeg()
                    fname = f"r{ri:02d}_c{ci:02d}.jpg"
                    (out / fname).write_bytes(data)
                    item = {"file": fname, "row": ri, "col": ci,
                            "yaw_deg": pano.steps_to_deg(rig.pan),
                            "pitch_deg": pano.steps_to_deg(rig.tilt)}
                    if north:       # this photo's compass bearing (true north, clockwise)
                        item["azimuth_deg"] = (north["camera_azimuth_at_yaw0_deg"] + item["yaw_deg"]) % 360
                    meta["images"].append(item)
                    shot += 1
                    with _lock:
                        _s["photo"].update(shot=shot, last=fname, pitch=round(pano.steps_to_deg(rig.tilt), 2),
                                           yaw=round(pano.steps_to_deg(rig.pan), 2))
                rig.pan_to(0)
                (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        with _lock:
            _s["photo"]["count"] = shot
            _s["stage"] = "surface_connect"
        _log(f"Photo capture finished: {shot} photos.")
    except InterruptedError:
        _log("Photo capture stopped.")
        with _lock:
            if _s is not None:
                _s["stage"] = "photo_connect"
    except Exception as e:
        _fail("The photo capture failed", e)
        with _lock:
            if _s is not None:
                _s["stage"] = "photo_connect"
    finally:
        _close_photo()


def _close_photo():
    """Back to level and facing forward (as panorama.py does on exit), then release the ports."""
    with _lock:
        if _s is None:
            return
        rig, cam = _s["_rig"], _s["_cam"]
        _s["_rig"] = _s["_cam"] = None
    with _dev:
        if rig:
            try:
                rig.pan_to(0)
                rig.tilt_to(0)
            except Exception as e:
                _log(f"Could not return the rig to its start position: {e}")
            rig.close()
        if cam:
            cam.close()


def photo_stop():
    with _lock:
        _need(("capturing", "leveling"))
        if _s["stage"] == "leveling":
            _s["stage"] = "photo_connect"
        elif _s["_abort"]:
            _s["_abort"].set()
            return
    _close_photo()


def photo_skip():
    with _lock:
        _need(("photo_connect",))
    _close_photo()
    with _lock:
        _s["stage"] = "surface_connect"
        _s["photo"]["count"] = 0
    _log("Photos skipped. Sunlight will not be available for this room.")


def last_photo():
    with _lock:
        if _s is None or not _s["photo"].get("last"):
            return None
        return Path(_s["dir"]) / "photos" / _s["photo"]["last"]


# ------------------------------------------------------------------ surface module
def surface_start(port, material):
    with _lock:
        _need(("surface_connect",))
        _s.update(error=None, stage="surface_connecting")
        info, sid = _s["info"], _s["id"]
        sdir = Path(_s["dir"]) / "surface"
    try:
        lg = _script("logger", MOLD_DIR)
        import serial
        sdir.mkdir(parents=True, exist_ok=True)
        # same defaults as logger.py's command line
        args = SimpleNamespace(session=sid, site=info.get("site") or "", lat=info.get("lat"), lon=info.get("lon"),
                               station=None, ir_model="MLX90614ESF-BAA", emissivity=1.0,
                               material=material or "sensitive", notes="")
        session_path = sdir / "session.yaml"
        session = lg.load_or_create_session(session_path, args)
        session["materials"]["default"] = args.material
        lg.save_session(session_path, session)
        parser = lg.Parser(lg.ScanWriter(sdir / "scans.csv"), session, session_path)
        parser.log = _log
        ncst = None
        grid = lg.resolve_logger_grid(session, session_path, _log)
        if grid:
            ncst = parser.on_scan = lg.NcstCollector(sdir / "outdoor.csv", grid, _log)
        try:
            ser = serial.Serial(port, 115200, timeout=1)
        except Exception as e:
            raise ScanError(f"Could not open {port}: {e}")
    except Exception as e:
        with _lock:
            if _s is not None:
                _s["stage"] = "surface_connect"
        raise e if isinstance(e, ScanError) else ScanError(f"Could not start the surface scan: {e}")
    stop = threading.Event()
    ctrl = lg.Controller(ser, parser, 1, 60, auto_start=True, stop=stop)
    with _lock:
        _s.update(stage="surface_running", _ser=ser)
        _s["surface"].update(state="waiting", faces=0)
    _log(f"Port {port} open. Waiting for the surface module to be ready...")
    threading.Thread(target=_surface, args=(lg, ser, parser, ctrl, stop, ncst, sdir), daemon=True).start()


def _surface(lg, ser, parser, ctrl, stop, ncst, sdir):
    raw = (sdir / "serial_raw.log").open("a", encoding="utf-8")
    opened = time.monotonic()

    def feed(line):
        raw.write(f"{lg.now_iso()}\t{line.rstrip()}\n")
        raw.flush()
        parser.handle(line)
        if line.startswith("#INFO,ERROR"):      # surface_scan.ino stops in setup() (while (1)) after this
            _fail("The surface module reported a hardware error: " + line.strip()[len("#INFO,ERROR"):].strip()
                  + ". Check the sensor wiring, then press Start again.")
            stop.set()
            return
        with _lock:
            if _s is None:
                stop.set()
                return
            sf = _s["surface"]
            if parser.cur_id is not None:
                sf.update(state="scanning", faces=len(parser.buf))
            sf["heading"] = parser.session["heading"].get("deg")

    # The UNO resets when the port opens and then prints #INFO,ready. If that line is missed, send n anyway.
    ready = threading.Event()
    send_on_ready = parser.on_ready
    parser.on_ready = lambda: (ready.set(), send_on_ready())
    orig_tick = ctrl.tick

    def tick():
        orig_tick()
        if not ready.is_set() and time.monotonic() - opened > READY_TIMEOUT_S:
            ready.set()
            _log("The surface module did not report ready. Sending the scan command anyway.")
            ctrl.send_scan()

    ctrl.tick = tick
    try:
        lg.run_serial(ser, parser, ctrl, stop, feed)
    except Exception as e:
        _fail("Lost the connection to the surface module", e)
    finally:
        try:
            ser.close()
        except Exception:
            pass
        raw.close()
        if ncst:
            ncst.wait()
    with _lock:
        if _s is None:
            return
        _s["_ser"] = None
        if parser.scans_saved:
            _s["surface"]["state"] = "saved"
            _s["stage"] = "ready"
        else:
            _s["surface"]["state"] = "failed"
            _s["stage"] = "surface_connect"
            if not _s["error"]:
                _s["error"] = "The surface scan was not saved. Check the module and try again."
    _log("Surface scan saved." if parser.scans_saved else "Surface scan not saved.")


def surface_stop():
    with _lock:
        _need(("surface_running",))
        ser = _s["_ser"]
    if ser:
        ser.write(b"s")


# ------------------------------------------------------------------ bundle -> room
def finish():
    with _lock:
        _need(("ready",))
        d, info, sid = Path(_s["dir"]), _s["info"], _s["id"]
    manifest = {
        "format_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        # only filled-in values: an empty lat/lon here would hide the coordinates logger.py geocoded into session.yaml
        "site": {k: v for k, v in (("name", info.get("site")), ("lat", info.get("lat")), ("lon", info.get("lon")))
                 if v not in (None, "")},
        "camera_height_m": info.get("camera_height_m") or 1.2,
        "source": "scan wizard",
    }
    if info.get("heading_deg") is not None:      # typed in by the user; otherwise rooms.create uses the compass north in photos/meta.json
        manifest["heading"] = {"deg": info["heading_deg"], "confidence": "manual"}
    zpath = d / "bundle.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        photos = d / "photos"
        if (photos / "meta.json").exists():
            for f in sorted(photos.iterdir()):
                z.write(f, f"photos/{f.name}")
        for name in ("scans.csv", "session.yaml", "outdoor.csv"):
            if (d / "surface" / name).exists():
                z.write(d / "surface" / name, f"surface/{name}")
    room, _ = rooms.create(zpath, info.get("name") or info.get("site") or sid)
    if info.get("name"):
        rooms.update(room["id"], lambda r: r.update(name=info["name"]))
    pipeline.start(room["id"])
    with _lock:
        _s["stage"] = "done"
        _s["room_id"] = room["id"]
    _log(f"Room created from {zpath}.")
    return room["id"]
