"""Place search for the Flood tab, done on the server with the Kakao REST key in ../mold scanner/kma.py
(so no Kakao JavaScript key is needed). Same order as the old in-browser search: keyword, then address."""
import sys

from .settings import MOLD_DIR

SEOUL_RECT = "126.734,37.413,127.27,37.715"     # lng/lat of the flood data's bounds (search.js SEOUL_BOUNDS)


class PlaceError(Exception):
    pass


def _kma():
    if str(MOLD_DIR) not in sys.path:
        sys.path.insert(0, str(MOLD_DIR))
    import kma
    return kma


def _get(url, params):
    import requests
    kma = _kma()
    key = kma.KAKAO_REST_KEY.strip()
    if not key:
        raise PlaceError("Kakao REST key is EMPTY - enter it in app/mold_scanner/kma.py (KAKAO_REST_KEY). "
                         "You can still click the map to pick a cell.")
    try:
        r = requests.get(url, params=params, timeout=15, headers={"Authorization": f"KakaoAK {key}"})
    except requests.RequestException as e:
        raise PlaceError(f"Could not reach the Kakao place search: {e}")
    if r.status_code in (401, 403):
        raise PlaceError("The Kakao REST key was rejected. Check KAKAO_REST_KEY in app/mold_scanner/kma.py.")
    if not r.ok:
        raise PlaceError(f"The Kakao place search failed ({r.status_code}).")
    return r.json().get("documents") or []


def search(query):
    kma = _kma()
    out = [{"label": d["place_name"], "sub": d.get("road_address_name") or d.get("address_name") or "",
            "lat": float(d["y"]), "lng": float(d["x"])}
           for d in _get(kma.KAKAO_KEYWORD_URL, {"query": query, "rect": SEOUL_RECT, "size": 15})]
    if not out:
        out = [{"label": (d.get("road_address") or {}).get("address_name") or d["address_name"],
                "sub": d["address_name"] if d.get("road_address") else "",
                "lat": float(d["y"]), "lng": float(d["x"])}
               for d in _get(kma.KAKAO_ADDRESS_URL, {"query": query, "size": 15})]
    return out
