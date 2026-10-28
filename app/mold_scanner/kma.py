"""
기상청·카카오 API 공용 모듈 (logger.py와 mold_simulator.py가 함께 쓴다)

    인증키            이 파일에 직접 적는다
    p_sat             포화수증기압 (원 명세 §4.1)
    latlon_to_grid    위경도 -> 기상청 5 km 격자 (nx, ny)   (수정 명세 §2.2)
    fetch_ncst        초단기실황 = 측정 시점 외기           (수정 명세 §2.1)
    outdoor.csv       세션 폴더의 측정 시점 외기 기록       (수정 명세 §2.4)
    geocode           카카오 로컬 API로 장소 이름 -> 좌표
    관측소 좌표       asos_stations.yaml (손으로 적은 표), asos_stations.csv (기상자료개방포털 지점정보)
"""

import csv
import math
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import unquote
from zoneinfo import ZoneInfo

import numpy as np
import yaml

# ---------------------------------------------------------------- 인증키
# 이 파일을 다른 사람에게 줄 때는 키를 지우고 줄 것
KMA_SERVICE_KEY = "sPmc/m2JcuMYosHKGMxKbtu54xQpgoJyAELQcsrfDgo89ZEalq+bKLmE1jMFnarJzY5u8OkLeDY0Of/TSwOWWg=="     # 공공데이터포털 일반 인증키 (Decoding 키, Encoding 키 모두 가능)
KAKAO_REST_KEY = "426fd4cca1d583baf2a45b9a6975c2f9"      # 카카오 REST API 키 (장소 이름으로 좌표를 찾을 때만 필요)

KST = ZoneInfo("Asia/Seoul")
HERE = Path(__file__).parent
STATION_YAML = HERE / "asos_stations.yaml"
STATION_CSV_NAMES = ("asos_stations.csv", "관측지점정보_종관기상관측.csv")   # 포털에서 받은 이름 그대로도 읽음
STATION_CSV = next((HERE / n for n in STATION_CSV_NAMES if (HERE / n).exists()), HERE / STATION_CSV_NAMES[0])


def service_key() -> str:
    key = KMA_SERVICE_KEY.strip()
    if not key:
        raise RuntimeError("KMA_SERVICE_KEY가 비어 있음 - kma.py에 공공데이터포털 인증키를 적을 것")
    return unquote(key) if "%" in key else key   # Encoding 키를 넣어도 이중 인코딩되지 않게


def p_sat(T):
    """포화수증기압 [hPa]. Magnus식(Alduchov-Eskridge, 물 기준), 0 °C 이하도 같은 식."""
    T = np.asarray(T, dtype=float)
    return 6.1094 * np.exp(17.625 * T / (T + 243.04))


def iso_hour(h: datetime) -> str:
    return h.strftime("%Y-%m-%dT%H:00:00+09:00")


def parse_kst(s: str) -> datetime:
    """ISO 8601 문자열 -> KST 기준 naive datetime."""
    dt = datetime.fromisoformat(s)
    return dt.astimezone(KST).replace(tzinfo=None) if dt.tzinfo else dt


# ---------------------------------------------------------------- 격자 변환 (§2.2)
def latlon_to_grid(lat: float, lon: float) -> tuple[int, int]:
    """기상청 람베르트 정각원추 투영 (동네예보 5 km 격자)."""
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
    """관측소 (위도, 경도, 이름). asos_stations.yaml을 먼저, 없으면 asos_stations.csv를 본다."""
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
    """(nx, ny, 근거). site.lat/lon -> 관측소 좌표 순서. 둘 다 없으면 None."""
    if site.get("lat") is not None and site.get("lon") is not None:
        return (*latlon_to_grid(float(site["lat"]), float(site["lon"])), "site.lat/lon")
    if station is not None:
        c = station_coords(station)
        if c:
            return (*latlon_to_grid(c[0], c[1]), f"관측소 {station} 좌표")
    return None


# ---------------------------------------------------------------- 초단기실황 (§2.1)
NCST_URL = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"


def _ncst_request(base: datetime, nx: int, ny: int) -> dict:
    """초단기실황 한 번 호출. 응답 JSON. (테스트에서 가짜 응답으로 바꿔 끼운다)"""
    import requests
    r = requests.get(NCST_URL, timeout=20, params={
        "serviceKey": service_key(), "pageNo": 1, "numOfRows": 10, "dataType": "JSON",
        "base_date": f"{base:%Y%m%d}", "base_time": f"{base:%H}00", "nx": nx, "ny": ny})
    r.raise_for_status()
    return r.json()


def _ncst_once(base: datetime, nx: int, ny: int):
    """(T1H, REH) 또는 실패 시 None. 키가 없으면 RuntimeError를 그대로 올린다."""
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
    """정시 hour의 초단기실황. 실패하면 한 시간 전을 요청한다. outdoor.csv 행 또는 None."""
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
    """(hour, source)가 아직 없는 행만 추가한다. 추가한 행 수."""
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
    """hours 중 ncst 행이 없는 정시를 요청해 outdoor.csv에 추가한다. (추가 수, 못 구한 정시)"""
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


# ---------------------------------------------------------------- 카카오 지오코딩, 관측소 목록
KAKAO_ADDRESS_URL = "https://dapi.kakao.com/v2/local/search/address.json"
KAKAO_KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def _kakao_search(url: str, query: str):
    """카카오 로컬 검색의 첫 결과 (위도, 경도, 주소) 또는 None."""
    import requests
    key = KAKAO_REST_KEY.strip()
    if not key:
        raise RuntimeError("KAKAO_REST_KEY가 비어 있음 - 장소 이름으로 좌표를 찾으려면 kma.py에 카카오 REST 키를 "
                           "적을 것 (또는 session.yaml에 site.lat/lon을 직접 적을 것)")
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
    """장소 이름 -> (위도, 경도, 주소, 실제 검색어). 못 찾으면 뒤 단어를 하나씩 빼며 다시 찾는다."""
    words = name.split()
    for n in range(len(words), 0, -1):
        query = " ".join(words[:n])
        for url in (KAKAO_ADDRESS_URL, KAKAO_KEYWORD_URL):
            hit = _kakao_search(url, query)
            if hit:
                return (*hit, query)
    return None


def load_station_list(path: Path):
    """기상자료개방포털 지점정보 CSV에서 현재 운영 중인 지점 (stn, name, lat, lon)."""
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
    if "지점번호" not in df.columns and "지점" in df.columns:   # 내려받은 파일은 열 이름이 '지점'
        df = df.rename(columns={"지점": "지점번호"})
    cols = {"지점번호": "stn", "지점명": "name", "위도": "lat", "경도": "lon"}
    missing = set(cols) - set(df.columns)
    if missing:
        raise SystemExit(f"{path.name}에 열 {sorted(missing)}이 없음 (지점정보 CSV인지 확인)")
    df = df.apply(lambda s: s.str.strip()).rename(columns=cols)
    if "종료일" in df:                     # 위치 이력 중 현재 행만
        df = df[df["종료일"].isna() | (df["종료일"] == "")]
    for c in ("stn", "lat", "lon"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["stn", "lat", "lon"]).drop_duplicates("stn", keep="last")
    return df.assign(stn=df["stn"].astype(int))[["stn", "name", "lat", "lon"]].reset_index(drop=True)
