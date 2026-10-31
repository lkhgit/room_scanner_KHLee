"""Room storage: one folder per room under rooms/, with room.json holding its state."""
import json
import re
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path

import yaml

from . import bundle
from .settings import ROOMS_DIR

_lock = threading.RLock()
STEPS = ("stitch", "model", "sun", "mold")


def now() -> str:
    """Returns the current local time as an ISO string with the time zone."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def room_dir(room_id: str) -> Path:
    """Returns a room's folder, raising KeyError for an invalid or unknown id."""
    if not re.fullmatch(r"[0-9a-f]{12}", room_id):
        raise KeyError(room_id)
    d = ROOMS_DIR / room_id
    if not d.is_dir():
        raise KeyError(room_id)
    return d


def load(room_id: str) -> dict:
    """Reads a room's room.json."""
    with _lock:
        return json.loads((room_dir(room_id) / "room.json").read_text(encoding="utf-8"))


def save(room: dict) -> None:
    """Writes a room's room.json atomically (through a temp file)."""
    with _lock:
        path = room_dir(room["id"]) / "room.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(room, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)


def update(room_id: str, fn) -> dict:
    """Loads a room, lets fn change it, and saves it, all under the lock."""
    with _lock:
        room = load(room_id)
        fn(room)
        save(room)
        return room


def set_step(room_id: str, step: str, state: str, message: str = "", detail: str = "") -> None:
    """Records the state, message and log tail of one processing step."""
    def fn(r):
        """Stores the step entry in the room."""
        r["steps"][step] = {"state": state, "message": message, "detail": detail[-4000:], "updated": now()}
    update(room_id, fn)


def list_rooms() -> list:
    """Lists all rooms with their summary fields, newest first."""
    out = []
    for d in sorted(ROOMS_DIR.iterdir(), reverse=True):
        try:
            r = json.loads((d / "room.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append({k: r.get(k) for k in ("id", "name", "created_at", "params", "steps")})
    out.sort(key=lambda r: r["created_at"] or "", reverse=True)
    return out


def create(zip_path: Path, display_name: str):
    """Validates and extracts a bundle. Returns (room dict, warnings); raises bundle.BundleError."""
    manifest, layout, warnings = bundle.inspect(zip_path)
    room_id = uuid.uuid4().hex[:12]
    d = ROOMS_DIR / room_id
    try:
        bundle.extract(zip_path, layout, d / "bundle")
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise
    sess = yaml.safe_load((d / "bundle" / "surface" / "session.yaml").read_text(encoding="utf-8")) or {}
    site = {**(sess.get("site") or {}), **{k: v for k, v in (manifest.get("site") or {}).items() if v is not None}}
    heading = dict(manifest.get("heading") or {})
    deg = heading.get("deg")
    if deg is None:         # no heading typed into the manifest: the photo module's compass north, then the surface module's
        deg = bundle.photo_north(d / "bundle" / "photos" / "meta.json")
        if deg is not None:
            heading["confidence"] = "compass sensor on the photo module"
        else:
            deg = (sess.get("heading") or {}).get("deg")
    room = {
        "id": room_id,
        "name": site.get("name") or Path(display_name).stem or "Untitled room",
        "created_at": now(),
        "manifest": manifest,
        "warnings": warnings,
        "has": {"photos": (d / "bundle" / "photos" / "meta.json").exists(),
                "panorama": any((d / "bundle").glob("panorama.*")),
                "surface": True},
        "params": {"lat": site.get("lat"), "lon": site.get("lon"),
                   "heading_deg": float(deg) if deg is not None else None,
                   "heading_confidence": heading.get("confidence"),
                   "camera_height_m": float(manifest.get("camera_height_m") or 1.2)},
        "steps": {s: {"state": "pending", "message": "", "detail": "", "updated": now()} for s in STEPS},
    }
    save_new(d, room)
    return room, warnings


def save_new(d: Path, room: dict) -> None:
    """Writes room.json for a room folder that was just created."""
    (d / "room.json").write_text(json.dumps(room, ensure_ascii=False, indent=2), encoding="utf-8")
