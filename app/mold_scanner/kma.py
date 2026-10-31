"""
Shared module for the KMA (Korea Meteorological Administration) and Kakao APIs, used by logger.py and mold_simulator.py

    API keys          written directly in this file
    p_sat             saturation vapour pressure (original spec §4.1)
    latlon_to_grid    latitude/longitude -> KMA 5 km grid (nx, ny)   (revised spec §2.2)
    fetch_ncst        ultra-short-term observation = outdoor air at the time of the scan   (revised spec §2.1)
    outdoor.csv       outdoor air at scan time, kept in the session folder   (revised spec §2.4)
    geocode           place name -> coordinates with the Kakao Local API
    station coords    asos_stations.yaml (hand-written table), asos_stations.csv (station list from the KMA open data portal)
"""

import csv
import math
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import unquote
from zoneinfo import ZoneInfo

import numpy as np
import yaml

# ============================ API KEYS REQUIRED ============================
# !! The API keys below are intentionally EMPTY. Enter your own keys here !!
# Never commit real keys to git.
#   KMA_SERVICE_KEY : data.go.kr (Korea Meteorological Administration) service key
#   KAKAO_REST_KEY  : Kakao Developers REST API key (only needed for place-name search)
KMA_SERVICE_KEY = ""   # <-- ENTER your data.go.kr service key here
KAKAO_REST_KEY = ""    # <-- ENTER your Kakao REST API key here
# ===========================================================================

KST = ZoneInfo("Asia/Seoul")
HERE = Path(__file__).parent
STATION_YAML = HERE / "asos_stations.yaml"
STATION_CSV_NAMES = ("asos_stations.csv", "관측지점정보_종관기상관측.csv")   # also reads the file under the name the portal gives it
STATION_CSV = next((HERE / n for n in STATION_CSV_NAMES if (HERE / n).exists()), HERE / STATION_CSV_NAMES[0])


def service_key() -> str:
    """Returns the KMA service key, raising an error if it has not been entered."""
    key = KMA_SERVICE_KEY.strip()
    if not key:
        raise RuntimeError("!!! KMA_SERVICE_KEY is EMPTY - enter your data.go.kr service key in "
                           "app/mold_scanner/kma.py !!!")
    return unquote(key) if "%" in key else key   # so a URL-encoded key is not encoded twice


def p_sat(T):
    """Saturation vapour pressure [hPa]. Magnus formula (Alduchov-Eskridge, over water), also used below 0 °C."""
    T = np.asarray(T, dtype=float)
    return 6.1094 * np.exp(17.625 * T / (T + 243.04))


def iso_hour(h: datetime) -> str:
    """Formats a full hour as an ISO string in KST."""
    return h.strftime("%Y-%m-%dT%H:00:00+09:00")


def parse_kst(s: str) -> datetime:
    """ISO 8601 string -> naive datetime in KST."""
    dt = datetime.fromisoformat(s)
    return dt.astimezone(KST).replace(tzinfo=None) if dt.tzinfo else dt


# ---------------------------------------------------------------- grid conversion (§2.2)
def latlon_to_grid(lat: float, lon: float) -> tuple[int, int]:
    """KMA Lambert conformal conic projection (the 5 km grid of the local forecast)."""
    RE, GRID, SLAT1, SLAT2, OLON, OLAT, XO, YO = 6371.00877, 5.0, 30.0, 60.0, 126.0, 38.0, 43, 136
    D = math.pi / 180
    re = RE / GRID
    slat1, slat2, olon, olat = SLAT1 * D, SLAT2 * D, OLON * D, OLAT * D
    sn = math.tan(math.pi / 4 + slat2 / 2) / math.tan(math.pi / 4 + slat1 / 2)
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
    sf = math.tan(math.pi / 4 + slat1 / 2) ** sn * math.cos(slat1) / sn
    ro = re * sf / math.tan(math.pi / 4 + olat / 2) ** sn
    ra = re * sf / math.tan(math.pi / 4 + lat * D / 2) ** sn
    theta = lon * D - olon
    if theta > math.pi:
        theta -= 2 * math.pi
    if theta < -math.pi:
        theta += 2 * math.pi
    theta *= sn
    return int(math.floor(ra * math.sin(theta) + XO + 0.5)), int(math.floor(ro - ra * math.cos(theta) + YO + 0.5))


def station_coords(stn: int):
    """Station (latitude, longitude, name). Looks in asos_stations.yaml first, then asos_stations.csv."""
    if STATION_YAML.exists():
        with STATION_YAML.open(encoding="utf-8") as f:
            table = yaml.safe_load(f) or {}
        rec = table.get(int(stn))
        if rec:
            return float(rec["lat"]), float(rec["lon"]), rec.get("name")
    if STATION_CSV.exists():
        df = load_station_list(STATION_CSV)
        hit = df[df["stn"] == int(stn)]
        if len(hit):
            r = hit.iloc[0]
            return float(r["lat"]), float(r["lon"]), r["name"]
    return None


def resolve_grid(site: dict, station: int | None):
    """(nx, ny, source). Uses site.lat/lon, then the station coordinates. None if neither exists."""
    if site.get("lat") is not None and site.get("lon") is not None:
        return (*latlon_to_grid(float(site["lat"]), float(site["lon"])), "site.lat/lon")
    if station is not None:
        c = station_coords(station)
        if c:
            return (*latlon_to_grid(c[0], c[1]), f"관측소 {station} 좌표")
    return None


# ---------------------------------------------------------------- ultra-short-term observations (§2.1)
NCST_URL = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"


def _ncst_request(base: datetime, nx: int, ny: int) -> dict:
    """Calls the ultra-short-term observation API once and returns the JSON reply. (Tests swap in a fake reply.)"""
    import requests
    r = requests.get(NCST_URL, timeout=20, params={
        "serviceKey": service_key(), "pageNo": 1, "numOfRows": 10, "dataType": "JSON",
        "base_date": f"{base:%Y%m%d}", "base_time": f"{base:%H}00", "nx": nx, "ny": ny})
    r.raise_for_status()
    return r.json()


def _ncst_once(base: datetime, nx: int, ny: int):
    """(T1H, REH), or None on failure. A missing key raises RuntimeError as it is."""
    import requests
    try:
        resp = _ncst_request(base, nx, ny)["response"]
        if resp["header"]["resultCode"] != "00":
            return None
        items = resp["body"]["items"]["item"]
        vals = {it["category"]: float(it["obsrValue"]) for it in items}
        return vals["T1H"], vals["REH"]
    except (requests.RequestException, ValueError, KeyError, TypeError):
        return None


def fetch_ncst(hour: datetime, nx: int, ny: int):
    """Ultra-short-term observation for the full hour `hour`, or the hour before if that fails. An outdoor.csv row or None."""
    for base in (hour, hour - timedelta(hours=1)):
        got = _ncst_once(base, nx, ny)
        if got:
            t, rh = got
            return {"hour": base, "source": "ncst", "t_out_c": t, "rh_out_pct": rh,
                    "p_out_hpa": round(float(rh / 100 * p_sat(t)), 3), "nx": nx, "ny": ny,
                    "fetched_at": datetime.now(KST).isoformat(timespec="seconds")}
    return None


# ---------------------------------------------------------------- outdoor.csv (§2.4)
OUTDOOR_COLUMNS = ["hour", "source", "t_out_c", "rh_out_pct", "p_out_hpa", "nx", "ny", "fetched_at"]


def read_outdoor(path: Path) -> list[dict]:
    """Reads outdoor.csv into a list of rows (empty if the file does not exist)."""
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            num = lambda k: float(r[k]) if r.get(k) not in (None, "") else None
            out.append({"hour": parse_kst(r["hour"]), "source": r["source"],
                        "t_out_c": num("t_out_c"), "rh_out_pct": num("rh_out_pct"),
                        "p_out_hpa": num("p_out_hpa"),
                        "nx": int(r["nx"]) if r.get("nx") else None,
                        "ny": int(r["ny"]) if r.get("ny") else None,
                        "fetched_at": r.get("fetched_at", "")})
    return out


def append_outdoor(path: Path, rows: list[dict]) -> int:
    """Adds only rows whose (hour, source) is not there yet. Returns the number of rows added."""
    existing = read_outdoor(path)
    keys = {(r["hour"], r["source"]) for r in existing}
    new = []
    for r in rows:
        k = (r["hour"], r["source"])
        if k not in keys:
            keys.add(k)
            new.append(r)
    if not new:
        return 0
    allrows = sorted(existing + new, key=lambda r: (r["hour"], r["source"]))
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(OUTDOOR_COLUMNS)
        for r in allrows:
            w.writerow([iso_hour(r["hour"]), r["source"], r["t_out_c"], r["rh_out_pct"], r["p_out_hpa"],
                        "" if r.get("nx") is None else r["nx"], "" if r.get("ny") is None else r["ny"],
                        r.get("fetched_at", "")])
    tmp.replace(path)
    return len(new)


def collect_ncst(path: Path, hours: list[datetime], nx: int, ny: int) -> tuple[int, list[datetime]]:
    """Fetches the hours that have no ncst row yet and adds them to outdoor.csv. Returns (rows added, hours not found)."""
    have = {r["hour"] for r in read_outdoor(path) if r["source"] == "ncst"}
    rows, failed = [], []
    for h in sorted(hours):
        if h in have:
            continue
        row = fetch_ncst(h, nx, ny)
        if row:
            rows.append(row)
            have.add(row["hour"])
        if row is None or row["hour"] != h:
            failed.append(h)
    return append_outdoor(path, rows), failed


# ---------------------------------------------------------------- Kakao geocoding, station list
KAKAO_ADDRESS_URL = "https://dapi.kakao.com/v2/local/search/address.json"
KAKAO_KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two latitude/longitude points."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def _kakao_search(url: str, query: str):
    """First result of a Kakao Local search as (latitude, longitude, address), or None."""
    import requests
    key = KAKAO_REST_KEY.strip()
    if not key:
        raise RuntimeError("!!! KAKAO_REST_KEY is EMPTY - enter your Kakao REST API key in app/mold_scanner/kma.py "
                           "(or set site.lat/lon directly in session.yaml) !!!")
    r = requests.get(url, params={"query": query, "size": 1}, timeout=30,
                     headers={"Authorization": f"KakaoAK {key}"})
    if r.status_code in (401, 403):
        raise RuntimeError(f"카카오 API 인증 실패 ({r.status_code}) - REST 키와 카카오맵 사용 설정 확인: "
                           f"{r.text[:200]}")
    r.raise_for_status()
    docs = r.json().get("documents") or []
    if not docs:
        return None
    d = docs[0]
    label = d.get("address_name") or ""
    if d.get("place_name"):
        label = f"{d['place_name']} ({label})"
    return float(d["y"]), float(d["x"]), label


def geocode(name: str):
    """Place name -> (latitude, longitude, address, query used). If nothing is found, drops the last word and tries again."""
    words = name.split()
    for n in range(len(words), 0, -1):
        query = " ".join(words[:n])
        for url in (KAKAO_ADDRESS_URL, KAKAO_KEYWORD_URL):
            hit = _kakao_search(url, query)
            if hit:
                return (*hit, query)
    return None


def load_station_list(path: Path):
    """Stations currently in operation (stn, name, lat, lon), from the KMA open data portal's station list CSV."""
    import pandas as pd
    if not path.exists():
        raise SystemExit(f"관측소 목록이 없음 - 기상자료개방포털(data.kma.go.kr) 지점정보에서 CSV를 받아 "
                         f"{path.parent}에 {' 또는 '.join(STATION_CSV_NAMES)}로 저장하거나, "
                         "config에 station_id를 적을 것")
    for enc in ("utf-8-sig", "cp949"):
        try:
            df = pd.read_csv(path, encoding=enc, dtype=str)
            break
        except UnicodeDecodeError:
            continue
    df.columns = [c.strip() for c in df.columns]
    if "지점번호" not in df.columns and "지점" in df.columns:   # the downloaded file calls this column '지점'
        df = df.rename(columns={"지점": "지점번호"})
    cols = {"지점번호": "stn", "지점명": "name", "위도": "lat", "경도": "lon"}
    missing = set(cols) - set(df.columns)
    if missing:
        raise SystemExit(f"{path.name}에 열 {sorted(missing)}이 없음 (지점정보 CSV인지 확인)")
    df = df.apply(lambda s: s.str.strip()).rename(columns=cols)
    if "종료일" in df:                     # only the current row of each station's location history
        df = df[df["종료일"].isna() | (df["종료일"] == "")]
    for c in ("stn", "lat", "lon"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["stn", "lat", "lon"]).drop_duplicates("stn", keep="last")
    return df.assign(stn=df["stn"].astype(int))[["stn", "name", "lat", "lon"]].reset_index(drop=True)
