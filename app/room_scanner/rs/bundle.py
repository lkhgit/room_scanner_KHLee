"""Scan bundle (.zip) format: validation and extraction.

Layout (a single wrapping folder and non-UTF-8 file names are tolerated):
    manifest.json            required
    photos/meta.json + *.jpg photo scan (panorama.py output)       } one of these two, or
    panorama.jpg             already-stitched 2:1 panorama           } neither (surface-only, with a warning)
    calib.json               optional camera calibration
    surface/scans.csv        required (logger.py session)
    surface/session.yaml     required
    surface/outdoor.csv      optional
"""
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

import yaml

FORMAT_VERSION = 1
MAX_UNCOMPRESSED = 2 * 1024 ** 3
IMG_EXT = ("jpg", "jpeg", "png")


class BundleError(Exception):
    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def photo_north(meta_path):
    """Compass bearing (true north, clockwise) of the panorama centre from panorama.py's meta.json "north", or None."""
    try:
        n = json.loads(Path(meta_path).read_text(encoding="utf-8-sig")).get("north") or {}
        return float(n["camera_azimuth_at_yaw0_deg"]) % 360
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _decode(info: zipfile.ZipInfo) -> str:
    """Windows zips often store cp949 names without the UTF-8 flag; zipfile then reads them as cp437."""
    name = info.filename
    if info.flag_bits & 0x800:
        return name
    try:
        return name.encode("cp437").decode("cp949")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def _members(zf: zipfile.ZipFile) -> dict:
    out = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        parts = tuple(p for p in PurePosixPath(_decode(info).replace("\\", "/")).parts if p not in ("", "."))
        if not parts or ".." in parts or parts[0].startswith("__MACOSX") or parts[-1] in (".DS_Store", "Thumbs.db"):
            continue
        out[parts] = info
    return out


def _find_root(members):
    for parts in members:
        if parts[-1].lower() == "manifest.json" and len(parts) <= 2:
            return parts[:-1]
    return None


def inspect(zip_path):
    """Returns (manifest, layout, warnings); raises BundleError listing everything wrong."""
    problems, warnings = [], []
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        raise BundleError(["The file is not a valid .zip archive."])
    with zf:
        members = _members(zf)
        if sum(i.file_size for i in members.values()) > MAX_UNCOMPRESSED:
            raise BundleError(["The archive is too large (over 2 GB uncompressed)."])
        root = _find_root(members)
        if root is None:
            raise BundleError(["manifest.json is missing. A scan bundle must contain a manifest.json at its top level."])
        names = {k[len(root):]: v for k, v in members.items() if k[:len(root)] == root}
        lower = {tuple(p.lower() for p in k): k for k in names}

        def find(filename):
            return next((k for lk, k in lower.items() if lk[-1] == filename), None)

        manifest = {}
        try:
            manifest = json.loads(zf.read(names[find("manifest.json")]).decode("utf-8-sig"))
            if not isinstance(manifest, dict):
                raise ValueError
        except (ValueError, KeyError):
            manifest = {}
            problems.append("manifest.json is not valid JSON.")
        if manifest:
            try:
                if manifest.get("format_version") is None:
                    problems.append("manifest.json has no format_version.")
                elif int(manifest["format_version"]) > FORMAT_VERSION:
                    problems.append(f"This bundle uses format version {manifest['format_version']}, "
                                    f"but this app only reads up to version {FORMAT_VERSION}.")
            except (TypeError, ValueError):
                problems.append("manifest.json has an invalid format_version.")

        scans, session = find("scans.csv"), find("session.yaml")
        missing = [n for n, k in (("scans.csv", scans), ("session.yaml", session)) if k is None]
        if missing:
            problems.append("Surface scan files are missing: " + ", ".join(missing) + ".")

        photo_meta = find("meta.json")
        if photo_meta is not None and scans is not None and photo_meta[:-1] == scans[:-1]:
            photo_meta = None
        pano = next((k for lk, k in lower.items()
                     if lk[-1].startswith("panorama") and lk[-1].rsplit(".", 1)[-1] in IMG_EXT
                     and "1024x512" not in lk[-1]), None)
        photo_has_north = False
        if photo_meta is not None:
            try:
                meta = json.loads(zf.read(names[photo_meta]).decode("utf-8-sig"))
                photo_has_north = isinstance(meta.get("north"), dict) and \
                    meta["north"].get("camera_azimuth_at_yaw0_deg") is not None
                files = {k[-1].lower() for k in names if k[:-1] == photo_meta[:-1]}
                absent = [im["file"] for im in meta.get("images", []) if im["file"].lower() not in files]
                if not meta.get("images"):
                    problems.append("The photo meta.json lists no images.")
                elif absent:
                    problems.append(f"{len(absent)} photo(s) listed in meta.json are missing, e.g. {absent[0]}.")
            except (ValueError, KeyError, TypeError):
                problems.append("The photo meta.json is not valid.")
        if photo_meta is None and pano is None:
            warnings.append("No photos or panorama found: the Sunlight tab and the room image on the Mold tab "
                            "will be unavailable.")

        sess = {}
        if scans is not None:
            head = zf.read(names[scans]).decode("utf-8-sig", "replace").splitlines()
            need = {"scan_id", "face", "az_deg", "el_deg", "t_surf_c", "t_air_c", "rh_pct"}
            cols = set(head[0].strip().split(",")) if head else set()
            if not need <= cols:
                problems.append("scans.csv is missing columns: " + ", ".join(sorted(need - cols)) + ".")
            elif len(head) < 2:
                problems.append("scans.csv has no scan rows.")
        if session is not None:
            try:
                sess = yaml.safe_load(zf.read(names[session]).decode("utf-8-sig")) or {}
            except yaml.YAMLError:
                problems.append("session.yaml is not valid YAML.")
        if problems:
            raise BundleError(problems)

        site = {**(sess.get("site") or {}), **{k: v for k, v in (manifest.get("site") or {}).items() if v is not None}}
        if site.get("lat") is None or site.get("lon") is None:
            warnings.append("The bundle has no site coordinates. Set a location in the bar above the results.")
        if (manifest.get("heading") or {}).get("deg") is None and (sess.get("heading") or {}).get("deg") is None \
                and not photo_has_north:
            warnings.append("The bundle has no north heading. Assuming 0° until you set it in the bar above the results.")

        layout = {
            "root": list(root),
            "scans": list(scans), "session": list(session),
            "outdoor": list(find("outdoor.csv") or []) or None,
            "photo_meta": list(photo_meta) if photo_meta else None,
            "calib": list(find("calib.json") or find("camera_calib.json") or []) or None,
            "panorama": list(pano) if pano else None,
        }
        return manifest, layout, warnings


def _copy(zf, info, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(info) as src, open(target, "wb") as out:
        shutil.copyfileobj(src, out)


def extract(zip_path, layout, dest):
    """Extracts into dest with normalized names: manifest.json, photos/, surface/, panorama.*, calib.json."""
    dest.mkdir(parents=True, exist_ok=True)
    root = tuple(layout["root"])
    with zipfile.ZipFile(zip_path) as zf:
        names = {k[len(root):]: v for k, v in _members(zf).items() if k[:len(root)] == root}

        def put(key, target):
            _copy(zf, names[tuple(key)], dest / target)

        _copy(zf, next(v for k, v in names.items() if k[-1].lower() == "manifest.json"), dest / "manifest.json")
        put(layout["scans"], "surface/scans.csv")
        put(layout["session"], "surface/session.yaml")
        if layout["outdoor"]:
            put(layout["outdoor"], "surface/outdoor.csv")
        if layout["calib"]:
            put(layout["calib"], "calib.json")
        if layout["panorama"]:
            put(layout["panorama"], "panorama." + layout["panorama"][-1].rsplit(".", 1)[-1].lower())
        if layout["photo_meta"]:
            folder = tuple(layout["photo_meta"][:-1])
            for k, info in names.items():
                if k[:len(folder)] == folder and len(k) == len(folder) + 1 and \
                        k[-1].lower().rsplit(".", 1)[-1] in IMG_EXT + ("json",):
                    _copy(zf, info, dest / "photos" / k[-1])
