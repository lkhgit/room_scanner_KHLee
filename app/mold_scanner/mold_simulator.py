#!/usr/bin/env python3
"""
Surface temperature/humidity and mold (VTT) simulator

From a session folder written by logger.py and KMA weather data, simulates each face's surface temperature and
humidity hour by hour over a year (or the last 365 days) and computes the VTT mold index M (0-6). The formulas and
defaults follow the "surface temperature/humidity and mold (VTT) simulation implementation spec" (original spec) and
the "mold simulator revision spec: same-day measurement, parameter reliability, weather period" (revised spec).

Input
    <session-dir>/scans.csv, session.yaml   logger.py output
    <session-dir>/outdoor.csv                outdoor air at scan time (collected by the logger from the ultra-short-term
                                             observations; missing hours are filled in here)
    config.yaml                              settings (defaults if missing)
    asos_stations.yaml / .csv                station coordinates (for the grid conversion and automatic station choice)
    kma.py                                   KMA and Kakao API keys and API functions

The station is chosen from config station_id, then session.yaml asos_station, then the station nearest the site that
measures ground temperature. If the site has no coordinates, site.name is looked up with the Kakao Local API and
saved to session.yaml.

Each face's f and dv_max from the measurements come with an uncertainty sigma. Even with a single scan, the measured
value is used whenever it can be computed; a large sigma only adds a warning (judgment "uncertain"). Substitute
values are used only when no value can be computed, and the result is then marked "provisional". Either way two
scenarios are simulated, central and worst; the worst scenario uses f - 2 sigma and dv_max + 2 sigma.

Output (<session-dir>/sim/, <period> = a year or last365_<start date>)
    params.json          f, sigma_f, judgment; dv_max, sigma, judgment; outdoor source and ground lag per scan;
                         periods; provisional flag; warnings
    hourly_<period>.csv  rows = scenario x hour x face
    summary_<period>.csv rows = faces, columns per scenario (_central, _worst). The # lines at the top are comments
    compare_periods.csv  when several periods were run
    viewer_<period>.json for the HTML viewer (column arrays, per scenario)
    viewer_index.json    periods and files for the viewer
    plots/               PNG per face with --plot
Weather data cache: <session-dir>/../kma_cache/

Usage
    put KMA_SERVICE_KEY (and KAKAO_REST_KEY if needed) in kma.py
    python mold_simulator.py --session-dir data/banjiha_A [--years 2025 2024 | --period last365] [--plot]
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import kma
from kma import p_sat
from logger import KST, N_FACES, save_session

TZ = "Asia/Seoul"
TZ_SUFFIX = "+09:00"
SCENARIOS = ("central", "worst")


def now_iso() -> str:
    """Returns the current time in KST as an ISO string."""
    return datetime.now(KST).isoformat(timespec="seconds")


def today() -> pd.Timestamp:
    """Today at 00:00 (KST, naive). Tests swap this out."""
    return pd.Timestamp.now(tz=TZ).tz_localize(None).normalize()


# ---------------------------------------------------------------- settings (original spec §7, revised spec §6.1)
DEFAULTS = {
    "station_id": None,                    # None: taken from session.yaml or the site
    "ground_depth_m": 1.0,
    "period": "year",                      # year | last365
    "sim_years": None,                     # None: the last complete year
    "heating_months": [11, 12, 1, 2, 3],
    "cooling_months": [6, 7, 8, 9],
    "hvac_hours": list(range(18, 24)) + list(range(0, 8)),
    "T_heat_set": 20.0,
    "T_cool_set": 26.0,
    "T_low": 0.0,
    "T_high": 20.0,
    "round_to_class": True,
    "g_min": 0.3,
    "dv_max_override": None,
    "f_override": {},
    "outdoor_source": "auto",              # auto (ASOS first) | asos | ncst
    "ground_lag_max_days": 3,
    "sigma_ir": 0.5,
    "sigma_sht": 0.2,
    "sigma_site": 0.3,
    "sigma_rh_in": 1.8,
    "sigma_rh_out": 5.0,
    "sigma_beh": 1.0,
    "sigma_f_max": 0.10,
    "sigma_dv_max": 1.0,
    "f_central": 0.75,
    "f_low": 0.60,
    "dv_central": 4.0,
    "dv_high": 6.0,
    "material_class": "sensitive",
    "material_by_face": {},
    "C_mat": 0.5,
    "sd_max": 0.3,
}


def parse_hours(spec) -> list[int]:
    """Turns [18, 19, ...] or ["18-23", "0-7"] into a list of hours."""
    hours = []
    for h in spec:
        if isinstance(h, str) and "-" in h:
            a, b = (int(x) for x in h.split("-"))
            hours += list(range(a, b + 1))
        else:
            hours.append(int(h))
    return sorted(set(hours))


def load_config(path: Path | None) -> tuple[dict, dict]:
    """Returns (merged settings, the entries the user wrote)."""
    user = {}
    if path is not None:
        with path.open(encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        unknown = set(user) - set(DEFAULTS)
        if unknown:
            print(f"경고: config에 알 수 없는 항목 {sorted(unknown)} - 무시")
    cfg = {**DEFAULTS, **{k: v for k, v in user.items() if k in DEFAULTS}}
    cfg["hvac_hours"] = parse_hours(cfg["hvac_hours"])
    cfg["f_override"] = {int(k): float(v) for k, v in (cfg["f_override"] or {}).items()}
    cfg["material_by_face"] = {int(k): v for k, v in (cfg["material_by_face"] or {}).items()}
    if cfg["outdoor_source"] not in ("auto", "asos", "ncst"):
        raise SystemExit("outdoor_source는 auto, asos, ncst 중 하나여야 함")
    return cfg, user


# ---------------------------------------------------------------- moist air formulas (original spec §4.1)
def rho_v(p, T):
    """Vapour pressure [hPa] -> vapour density [g/m3]."""
    return 216.68 * np.asarray(p, dtype=float) / (np.asarray(T, dtype=float) + 273.15)


def rho_sat(T):
    """Saturation vapour density [g/m3]."""
    return rho_v(p_sat(T), T)


def dv_to_dp(dv, T):
    """Vapour density difference [g/m3] -> vapour pressure difference [hPa]."""
    return np.asarray(dv, dtype=float) * (np.asarray(T, dtype=float) + 273.15) / 216.68


def g_shape(theta_e24, T_low=0.0, T_high=20.0):
    """Assumption A1: shape function of the indoor moisture load, a function of the 24-hour moving mean outdoor temperature."""
    return np.clip((T_high - np.asarray(theta_e24, dtype=float)) / (T_high - T_low), 0.0, 1.0)


# ---------------------------------------------------------------- ASOS download (original spec §2.2)
HOURLY_URL = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
DAILY_URL = "http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
HOURLY_FIELDS = ["tm", "ta", "hm", "pv"]
GROUND_COLS = {0.5: "avgM05Te", 1.0: "avgM10Te", 1.5: "avgM15Te", 3.0: "avgM30Te", 5.0: "avgM50Te"}
DAILY_FIELDS = ["tm"] + list(GROUND_COLS.values())
MAX_GAP_H = 6
PARTIAL_CACHE_H = 3                        # hours a month that has not ended stays cached
HOURLY_MIN, DAILY_MIN = 0.99, 0.95         # required data coverage over the period (revised spec §5)


def _service_key() -> str:
    """Returns the KMA service key, or stops with the error message if it is missing."""
    try:
        return kma.service_key()
    except RuntimeError as e:
        raise SystemExit(str(e))


def _kma_request(url: str, params: dict) -> list[dict]:
    """Calls a KMA ASOS API and collects the items of every page."""
    import requests
    items, page = [], 1
    while True:
        r = requests.get(url, timeout=60, params={
            "serviceKey": _service_key(), "pageNo": page, "numOfRows": 999,
            "dataType": "JSON", "dataCd": "ASOS", **params})
        r.raise_for_status()
        try:
            resp = r.json()["response"]
        except (ValueError, KeyError):
            raise RuntimeError(f"ASOS 응답을 읽을 수 없음 (인증키 확인): {r.text[:300]}")
        code = resp["header"]["resultCode"]
        if code == "03":                   # NO_DATA
            return items
        if code != "00":
            raise RuntimeError(f"ASOS 오류 {code}: {resp['header']['resultMsg']}")
        body = resp["body"]
        got = (body.get("items") or {}).get("item", [])
        if isinstance(got, dict):
            got = [got]
        items += got
        if not got or len(items) >= int(body["totalCount"]):
            return items
        page += 1


def fetch_month(kind: str, station: int, month: pd.Period, cache_dir: Path) -> pd.DataFrame:
    """One month of ASOS data (as strings). Finished months are cached for good, the current month for a few hours."""
    fields = HOURLY_FIELDS if kind == "hourly" else DAILY_FIELDS
    yesterday = today() - pd.Timedelta(days=1)
    start = month.start_time
    month_end = month.end_time.normalize()
    end = min(month_end, yesterday)
    if end < start:
        return pd.DataFrame(columns=fields)
    complete = end == month_end
    base = f"{station}_{kind}_{month.strftime('%Y%m')}"
    path = cache_dir / (f"{base}.csv" if complete else f"{base}_partial.csv")
    if path.exists():
        fresh = complete or (datetime.now().timestamp() - path.stat().st_mtime) < PARTIAL_CACHE_H * 3600
        if fresh:
            return pd.read_csv(path, dtype=str)

    print(f"ASOS {kind} {month.strftime('%Y-%m')} 내려받는 중 (관측소 {station})", flush=True)
    if kind == "hourly":
        params = {"dateCd": "HR", "startDt": f"{start:%Y%m%d}", "startHh": "00",
                  "endDt": f"{end:%Y%m%d}", "endHh": "23", "stnIds": station}
        items = _kma_request(HOURLY_URL, params)
    else:
        params = {"dateCd": "DAY", "startDt": f"{start:%Y%m%d}", "endDt": f"{end:%Y%m%d}",
                  "stnIds": station}
        items = _kma_request(DAILY_URL, params)
    df = pd.DataFrame(items).reindex(columns=fields)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(path)
    return df


def fetch_range(kind: str, station: int, start: pd.Timestamp, end: pd.Timestamp,
                cache_dir: Path) -> pd.DataFrame:
    """Fetches ASOS data month by month over a date range and joins it."""
    months = pd.period_range(start, end, freq="M")
    frames = [fetch_month(kind, station, m, cache_dir) for m in months]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def asos_hourly_raw(station: int, start_h: pd.Timestamp, end_h: pd.Timestamp, cache_dir: Path) -> pd.DataFrame:
    """Hourly ta, hm, pv over [start_h, end_h] (missing values are NaN, not interpolated). pv is computed from ta and hm if missing."""
    grid = pd.date_range(start_h, end_h, freq="h")
    raw = fetch_range("hourly", station, start_h.normalize(), end_h.normalize(), cache_dir)
    if raw.empty:
        return pd.DataFrame(np.nan, index=grid, columns=["ta", "hm", "pv"])
    raw["tm"] = pd.to_datetime(raw["tm"])
    df = raw.set_index("tm")[["ta", "hm", "pv"]].apply(pd.to_numeric, errors="coerce")
    df = df[~df.index.duplicated()].reindex(grid)
    no_pv = df["pv"].isna() & df["ta"].notna() & df["hm"].notna()
    df.loc[no_pv, "pv"] = df.loc[no_pv, "hm"] / 100 * p_sat(df.loc[no_pv, "ta"])
    return df


def ground_daily(station: int, start: pd.Timestamp, end: pd.Timestamp, depth: float,
                 cache_dir: Path) -> pd.Series:
    """Daily ground temperature (missing days are left out)."""
    daily = fetch_range("daily", station, start, end, cache_dir)
    col = GROUND_COLS[depth]
    if daily.empty or col not in daily:
        return pd.Series(dtype=float)
    tg = pd.Series(pd.to_numeric(daily[col], errors="coerce").to_numpy(),
                   index=pd.to_datetime(daily["tm"])).dropna()
    return tg[~tg.index.duplicated()]


def check_coverage(ok: pd.Series, need: float, label: str, what: str) -> None:
    """Stops with the list of empty months if the share of True in ok (one bool per time) is below need."""
    frac = float(ok.mean()) if len(ok) else 0.0
    if frac >= need:
        return
    miss = (~ok).groupby(ok.index.to_period("M")).sum()
    months = ", ".join(f"{p.strftime('%Y-%m')} {int(n)}개" for p, n in miss[miss > 0].items())
    raise SystemExit(f"{label}: {what} 충족률 {frac:.1%} < {need:.0%} - 비어 있는 달: {months}")


def _long_gaps(missing: pd.Series, max_len: int) -> np.ndarray:
    """True for hours that belong to a gap longer than max_len hours."""
    runs = (missing != missing.shift()).cumsum()
    length = missing.groupby(runs).transform("size")
    return (missing & (length > max_len)).to_numpy()


def load_weather(station: int, start: pd.Timestamp, end: pd.Timestamp, depth: float,
                 cache_dir: Path, warnings: list, label: str) -> pd.DataFrame:
    """Hourly outdoor and ground temperature over [start 00:00, end 23:00]. Columns: theta_e, p_e, theta_e24, theta_g, data_flag."""
    pre = start - pd.Timedelta(days=1)    # the day before, for theta_e24
    df = asos_hourly_raw(station, pre, end + pd.Timedelta(hours=23), cache_dir)
    in_range = df.index >= start
    check_coverage((df["ta"].notna() & df["pv"].notna())[in_range], HOURLY_MIN, label, "외기 시간 자료")

    tg = ground_daily(station, start - pd.Timedelta(days=2), end + pd.Timedelta(days=1), depth, cache_dir)
    days = pd.date_range(start, end, freq="D")
    check_coverage(pd.Series(days.isin(tg.index), index=days), DAILY_MIN, label, "지중온도 일자료")
    if (~days.isin(tg.index)).any():
        warnings.append(f"{label}: 지중온도 결측 {(~days.isin(tg.index)).sum()}일 - 앞뒤 날짜로 보간함")

    flag = np.zeros(len(df), dtype=int)
    for col, name in (("ta", "기온"), ("pv", "증기압")):
        long = _long_gaps(df[col].isna(), MAX_GAP_H)
        if (long & in_range).any():
            warnings.append(f"{label}: {name} 결측 {(long & in_range).sum()}시간이 "
                            f"{MAX_GAP_H}시간보다 긴 구간에 있어 보간함 (data_flag=1)")
        flag |= long
        df[col] = df[col].interpolate("time", limit_direction="both")

    w = pd.DataFrame(index=df.index)
    w["theta_e"] = df["ta"]
    w["p_e"] = df["pv"]
    w["theta_e24"] = w["theta_e"].rolling(24, min_periods=1).mean()
    # daily means placed at 12:00 of each day and interpolated linearly to hours (original spec §4.5)
    noon = (tg.index + pd.Timedelta(hours=12)).asi8
    w["theta_g"] = np.interp(df.index.asi8, noon, tg.to_numpy())
    w["data_flag"] = flag
    return w.loc[start:]


# ---------------------------------------------------------------- station choice
MAX_CANDIDATES = 10                        # number of nearby stations to check for ground temperature
FAR_KM = 30.0


def has_ground_temp(station: int, depth: float, month: pd.Period, cache_dir: Path) -> bool:
    """Tells whether the month's daily data has ground temperature. Non-ASOS stations leave an empty cache, so they are not asked again."""
    try:
        d = fetch_month("daily", station, month, cache_dir)
    except RuntimeError:
        return False
    col = GROUND_COLS[depth]
    return col in d and pd.to_numeric(d[col], errors="coerce").notna().any()


def choose_station(session: dict, session_path: Path, depth: float, month: pd.Period,
                   cache_dir: Path, warnings: list) -> dict:
    """The station nearest the site coordinates (geocoded from the name if missing) that measures ground temperature."""
    site = session.get("site") or {}
    lat, lon = site.get("lat"), site.get("lon")
    if lat is None or lon is None:
        name = (site.get("name") or "").strip()
        if not name:
            raise SystemExit("장소 정보가 없음 - session.yaml에 site.name 또는 site.lat/lon, "
                             "아니면 config에 station_id를 적을 것")
        try:
            hit = kma.geocode(name)
        except RuntimeError as e:
            raise SystemExit(str(e))
        if hit is None:
            raise SystemExit(f"'{name}'의 위치를 카카오에서 찾지 못함 - site.lat/lon 또는 station_id를 직접 적을 것")
        lat, lon, address, query = hit
        print(f"장소 '{name}' -> '{query}' 검색 결과: {address} ({lat:.4f}, {lon:.4f})")
        site.update(lat=lat, lon=lon, geocoded_from=query, geocoded_address=address)
        session["site"] = site
        save_session(session_path, session)
    coord_source = (f"카카오 '{site['geocoded_from']}' -> {site['geocoded_address']}"
                    if site.get("geocoded_address") else "session.yaml site.lat/lon")

    stations = kma.load_station_list(kma.STATION_CSV)
    stations["km"] = kma.haversine_km(lat, lon, stations["lat"], stations["lon"])
    for s in stations.nsmallest(MAX_CANDIDATES, "km").itertuples():
        if not has_ground_temp(s.stn, depth, month, cache_dir):
            print(f"관측소 {s.stn} {s.name} ({s.km:.1f} km): {depth} m 지중온도 없음 - 건너뜀")
            continue
        if s.km > FAR_KM:
            warnings.append(f"가장 가까운 관측소 {s.stn} {s.name}가 {s.km:.0f} km 떨어져 있음")
        return {"id": int(s.stn), "name": s.name, "distance_km": round(float(s.km), 1),
                "source": "nearest", "site_lat": lat, "site_lon": lon, "coord_source": coord_source}
    raise SystemExit(f"가까운 관측소 {MAX_CANDIDATES}곳 모두 {depth} m 지중온도 자료가 없음 - "
                     "config에 station_id를 직접 적을 것")


# ---------------------------------------------------------------- measurement data (original spec §2.1)
def load_session(session_dir: Path) -> tuple[dict, pd.DataFrame]:
    """Reads session.yaml and scans.csv, with the scan timestamps in KST."""
    with (session_dir / "session.yaml").open(encoding="utf-8") as f:
        session = yaml.safe_load(f) or {}
    scans = pd.read_csv(session_dir / "scans.csv")
    scans["timestamp"] = (pd.to_datetime(scans["timestamp"], utc=True)
                          .dt.tz_convert(TZ).dt.tz_localize(None))
    return session, scans


def face_table(scans: pd.DataFrame, session: dict, cfg: dict, warnings: list) -> pd.DataFrame:
    """Position, material and exclusion of each face. Rows = faces 1..20."""
    geo = scans.groupby("face")[["zone", "az_deg", "el_deg"]].first()
    missing = sorted(set(range(1, N_FACES + 1)) - set(geo.index))
    if missing:
        warnings.append(f"scans.csv에 면 {missing}의 데이터가 없음")
    faces = geo.reindex(range(1, N_FACES + 1))
    faces.index.name = "face"

    mats = session.get("materials") or {}
    by_face = {int(k): v for k, v in (mats.get("by_face") or {}).items()}
    default = mats.get("default") or cfg["material_class"]
    faces["material"] = [by_face.get(k, cfg["material_by_face"].get(k, default)) for k in faces.index]
    bad = sorted(set(faces["material"]) - set(MATERIALS))
    if bad:
        raise SystemExit(f"알 수 없는 재료 등급 {bad} - {list(MATERIALS)} 중에서 고를 것")

    excluded = {int(k) for k in (session.get("excluded_faces") or [])} | set(missing)
    faces["excluded"] = [k in excluded for k in faces.index]
    return faces


def filter_scans(scans: pd.DataFrame, faces: pd.DataFrame, cfg: dict, warnings: list) -> pd.DataFrame:
    """Drops noisy readings (IR standard deviation above sd_max) and excluded faces."""
    noisy = scans["t_surf_sd_c"] > cfg["sd_max"]
    if noisy.any():
        warnings.append(f"IR 표준편차 > sd_max {cfg['sd_max']} °C 인 면·스캔 {noisy.sum()}개를 f 추출에서 뺌")
    excl = scans["face"].isin(faces.index[faces["excluded"]])
    return scans[~noisy & ~excl]


# ---------------------------------------------------------------- outdoor air at scan time (revised spec §2)
def select_outdoor(rows: list[dict], source: str) -> pd.DataFrame:
    """One outdoor row per full hour. auto prefers ASOS. Columns: t_out_c, p_out_hpa, source."""
    order = {"auto": ["asos", "ncst"], "asos": ["asos"], "ncst": ["ncst"]}[source]
    df = pd.DataFrame(rows, columns=kma.OUTDOOR_COLUMNS)
    df = df[df["source"].isin(order) & df["t_out_c"].notna() & df["p_out_hpa"].notna()].copy()
    df["hour"] = pd.to_datetime(df["hour"])
    df["rank"] = df["source"].map({s: i for i, s in enumerate(order)})
    df = df.sort_values(["hour", "rank"]).drop_duplicates("hour")
    return df.set_index("hour")[["t_out_c", "p_out_hpa", "source"]].astype(
        {"t_out_c": float, "p_out_hpa": float})


def fill_outdoor(path: Path, need: list[pd.Timestamp], session: dict, station: int, cfg: dict,
                 cache_dir: Path, warnings: list) -> list[dict]:
    """Fills the hours in need that outdoor.csv lacks from ASOS (if available), then from the ultra-short-term observations."""
    src = cfg["outdoor_source"]
    if src in ("auto", "asos"):
        raw = asos_hourly_raw(station, min(need), max(need), cache_dir)
        have = {pd.Timestamp(r["hour"]) for r in kma.read_outdoor(path) if r["source"] == "asos"}
        rows = [{"hour": h.to_pydatetime(), "source": "asos", "t_out_c": float(r.ta),
                 "rh_out_pct": None if pd.isna(r.hm) else float(r.hm), "p_out_hpa": round(float(r.pv), 3),
                 "nx": None, "ny": None, "fetched_at": now_iso()}
                for h, r in raw.loc[need].iterrows()
                if h not in have and pd.notna(r.ta) and pd.notna(r.pv)]
        kma.append_outdoor(path, rows)

    if src in ("auto", "ncst"):
        sel = select_outdoor(kma.read_outdoor(path), src)
        missing = [h for h in need if h not in sel.index]
        if missing:
            grid = session.get("grid") or {}
            g = ((grid["nx"], grid["ny"]) if grid.get("nx") is not None
                 else (kma.resolve_grid(session.get("site") or {}, station) or (None, None))[:2])
            if g[0] is None:
                warnings.append("격자 좌표를 정할 수 없어(site.lat/lon, 관측소 좌표 없음) 초단기실황으로 보충하지 못함")
            else:
                try:
                    print(f"초단기실황으로 외기 {len(missing)}시간 보충 시도 (격자 {g[0]}, {g[1]})", flush=True)
                    kma.collect_ncst(path, [h.to_pydatetime() for h in missing], g[0], g[1])
                except RuntimeError as e:
                    warnings.append(f"초단기실황을 가져오지 못함: {e}")
    return kma.read_outdoor(path)


def ground_for_day(tg: pd.Series, day: pd.Timestamp):
    """Ground temperature of that day, or of the nearest earlier day. (value, lag in days) or (NaN, NaN)."""
    prior = tg[tg.index <= day]
    if prior.empty:
        return np.nan, np.nan
    return float(prior.iloc[-1]), int((day - prior.index[-1]).days)


def scan_conditions(scans: pd.DataFrame, outdoor: pd.DataFrame, tg: pd.Series) -> pd.DataFrame:
    """Indoor conditions per scan, outdoor air at the same hour, ground temperature and its lag. Rows = scan_id."""
    per = scans.groupby("scan_id").first()[["timestamp", "t_air_c", "rh_pct"]]
    per["out_hour"] = per["timestamp"].dt.floor("h")
    o = outdoor.reindex(per["out_hour"])
    per["theta_e"] = o["t_out_c"].to_numpy()
    per["p_e"] = o["p_out_hpa"].to_numpy()
    per["out_source"] = o["source"].to_numpy()
    e24, n24 = [], []
    for h in per["out_hour"]:            # needs at least 18 of the last 24 full hours (revised spec §2.5)
        win = outdoor["t_out_c"].reindex(pd.date_range(h - pd.Timedelta(hours=23), h, freq="h"))
        n = int(win.notna().sum())
        n24.append(n)
        e24.append(win.mean() if n >= 18 else np.nan)
    per["theta_e24"], per["n_e24"] = e24, n24
    g = [ground_for_day(tg, d) for d in per["timestamp"].dt.normalize()]
    per["theta_g"] = [v for v, _ in g]
    per["ground_lag"] = [lag for _, lag in g]
    return per


# ---------------------------------------------------------------- parameter extraction and uncertainty (revised spec §4)
DV_CLASSES = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])


def fit_origin(x, y) -> float:
    """Least-squares slope of a line through the origin."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return float(np.sum(x * y) / np.sum(x * x))


def round_dv_class(v: float) -> float:
    """Rounds a dv_max value to the nearest humidity class."""
    return float(DV_CLASSES[np.abs(DV_CLASSES - v).argmin()])


def dv_max_fit(g, dv) -> float:
    """Fits dv_max as the slope of the moisture excess against g, never below 0."""
    return max(fit_origin(g, dv), 0.0)


def f_fit(theta_si, theta_i, theta_g) -> float:
    """Fits the temperature factor f from surface, indoor and ground temperatures."""
    theta_g = np.asarray(theta_g, dtype=float)
    return fit_origin(np.asarray(theta_i) - theta_g, np.asarray(theta_si) - theta_g)


def sigma_f(f: float, delta, sd, lag, cfg: dict) -> float:
    """Uncertainty of f: a random part per scan plus a systematic part shared by all scans (revised spec §4.1)."""
    delta, sd, lag = (np.asarray(v, dtype=float) for v in (delta, sd, lag))
    rs2 = sd ** 2 + ((1 - f) * 0.2 * lag) ** 2
    sys2 = cfg["sigma_ir"] ** 2 + f ** 2 * cfg["sigma_sht"] ** 2 + (1 - f) ** 2 * cfg["sigma_site"] ** 2
    S2 = np.sum(delta ** 2)
    return float(np.sqrt(np.sum(rs2 * delta ** 2) / S2 ** 2 + sys2 * np.sum(delta) ** 2 / S2 ** 2))


def sigma_dv_max(g, theta_e, theta_i, cfg: dict) -> float:
    """Uncertainty of dv_max (revised spec §4.2)."""
    g = np.asarray(g, dtype=float)
    rs2 = (rho_sat(theta_e) * cfg["sigma_rh_out"] / 100) ** 2 + cfg["sigma_beh"] ** 2
    ssys = rho_sat(theta_i) * cfg["sigma_rh_in"] / 100
    G2 = np.sum(g ** 2)
    return float(np.sqrt(np.sum(g ** 2 * rs2) / G2 ** 2 + np.sum(g * ssys) ** 2 / G2 ** 2))


def extract_dv_max(per: pd.DataFrame, cfg: dict, warnings: list) -> dict:
    """dv_max and its judgment, with the value for each scenario in central/worst."""
    ok = per[["theta_e", "p_e", "theta_e24"]].notna().all(axis=1)
    if (~ok).any():
        warnings.append(f"외기 자료(정시 값, 직전 24시간 중 18개 이상)가 없는 스캔 {int((~ok).sum())}개를 "
                        "dv_max 추출에서 뺌")
    use = per[ok]
    out = {"n_scans": int(len(use)), "g_mean": None, "dv_max_raw": None, "sigma": None}
    if cfg["dv_max_override"] is not None:
        v = float(cfg["dv_max_override"])
        return {**out, "judgment": "override", "central": v, "worst": v}

    if len(use):
        g = g_shape(use["theta_e24"], cfg["T_low"], cfg["T_high"])
        out["g_mean"] = float(np.mean(g))
        if out["g_mean"] < cfg["g_min"]:
            warnings.append(f"평균 g_s {out['g_mean']:.2f} < g_min {cfg['g_min']}: "
                            "외기가 따뜻한 시기의 측정이라 dv_max가 불확실함")
        if np.any(g > 0):
            p_i = use["rh_pct"] / 100 * p_sat(use["t_air_c"])
            dv = rho_v(p_i, use["t_air_c"]) - rho_v(use["p_e"], use["theta_e"])
            raw = fit_origin(g, dv)
            sig = sigma_dv_max(g, use["theta_e"], use["t_air_c"], cfg)
            out.update(dv_max_raw=raw, sigma=sig)
            if raw < 0:
                warnings.append(f"측정 dv_max가 음수({raw:.2f} g/m3) - 0으로 둠")
            # the measured value is used even when uncertain; the worst scenario adds 2 sigma, capped at the top humidity class of 10 g/m3
            clipped = max(raw, 0.0)
            central = round_dv_class(clipped) if cfg["round_to_class"] else clipped
            worst = max(central, min(clipped + 2 * sig, float(DV_CLASSES[-1])))
            judgment = "measured" if sig <= cfg["sigma_dv_max"] else "uncertain"
            if judgment == "uncertain":
                warnings.append(f"dv_max 불확도 큼 (sigma {sig:.2f} > {cfg['sigma_dv_max']} g/m3) - "
                                f"측정값 {clipped:.2f} 사용 (중앙 {central}, 보수 {worst:.2f})")
            return {**out, "judgment": judgment, "central": central, "worst": worst}

    why = "g가 모두 0 (외기가 따뜻해 계산 불가)" if out["g_mean"] is not None else "외기 자료가 있는 스캔 없음"
    warnings.append(f"dv_max 대체값 사용 ({why}) - 중앙 {cfg['dv_central']}, 보수 {cfg['dv_high']} g/m3")
    return {**out, "judgment": "substitute", "central": float(cfg["dv_central"]), "worst": float(cfg["dv_high"])}


def extract_f(valid: pd.DataFrame, per: pd.DataFrame, faces: pd.DataFrame, cfg: dict,
              warnings: list) -> pd.DataFrame:
    """f, sigma_f, judgment and per-scenario f for each face.

    Whenever f can be computed, the measured value is used regardless of its uncertainty (measured / uncertain / anomaly).
    Substitutes are used only when it cannot be computed: no scan of that face, or no indoor-ground difference (substitute).
    """
    lag_ok = per["ground_lag"].notna() & (per["ground_lag"] <= cfg["ground_lag_max_days"])
    if (~lag_ok).any():
        warnings.append(f"지중온도가 없거나 {cfg['ground_lag_max_days']}일보다 늦은 스캔 "
                        f"{int((~lag_ok).sum())}개를 f 추출에서 뺌")
    rows = valid.merge(per.loc[lag_ok, ["theta_g", "ground_lag"]], left_on="scan_id", right_index=True)
    out, subs, uncertain, anomalies, clipped = [], [], [], [], []
    for k in faces.index:
        d = rows[rows["face"] == k]
        rec = {"face": k, "n_scans": int(len(d)), "f_raw": None, "sigma_f": None}
        delta = d["t_air_c"] - d["theta_g"]
        if len(d) and float((delta ** 2).sum()) > 0:
            f_raw = f_fit(d["t_surf_c"], d["t_air_c"], d["theta_g"])
            f_c = min(max(f_raw, 0.0), 1.0)
            rec.update(f_raw=f_raw, sigma_f=sigma_f(f_c, delta, d["t_surf_sd_c"], d["ground_lag"], cfg))
        s = rec["sigma_f"]
        if k in cfg["f_override"]:
            v = cfg["f_override"][k]
            rec.update(judgment="override", f_central=v, f_worst=v)
        elif faces.at[k, "excluded"]:
            rec.update(judgment="excluded", f_central=None, f_worst=None)
        elif rec["f_raw"] is None:
            rec.update(judgment="substitute", f_central=cfg["f_central"], f_worst=cfg["f_low"])
            subs.append(k)
        else:
            f_c = min(max(rec["f_raw"], 0.0), 1.0)
            if not (-2 * s <= rec["f_raw"] <= 1 + 2 * s):
                judgment = "anomaly"
                anomalies.append(k)
            elif s <= cfg["sigma_f_max"]:
                judgment = "measured"
            else:
                judgment = "uncertain"
                uncertain.append(k)
            if judgment != "anomaly" and f_c != rec["f_raw"]:
                clipped.append(k)
            rec.update(judgment=judgment, f_central=f_c, f_worst=max(0.0, f_c - 2 * s))
        out.append(rec)
    if subs:
        warnings.append(f"면 {subs}: 스캔이 없거나 실내-지중 온도 차가 0이라 f를 계산할 수 없음 - "
                        f"대체값 {cfg['f_central']} / {cfg['f_low']} 사용")
    if uncertain:
        warnings.append(f"면 {uncertain}: f 불확도 큼 (sigma_f > {cfg['sigma_f_max']}) - 측정값 사용")
    if anomalies:
        warnings.append(f"면 {anomalies}: f가 [0, 1]을 2 sigma 넘게 벗어남 (이상) - [0, 1]로 잘라 사용. "
                        "방사율이 낮은 면(유리·금속)이나 열원이 시야에 든 면인지 확인")
    if clipped:
        warnings.append(f"면 {clipped}: f가 [0, 1]을 조금 벗어나 잘라냄")
    return pd.DataFrame(out).set_index("face")


def provisional_message(fres: pd.DataFrame, dv: dict) -> str | None:
    """The '잠정 결과: ...' (provisional result) sentence if any substitute value is used."""
    faces = [int(k) for k in fres.index[fres["judgment"] == "substitute"]]
    parts = []
    if faces:
        parts.append("면 전체" if len(faces) == len(fres[fres["judgment"] != "excluded"])
                     else "면 " + ", ".join(map(str, faces)))
    if dv["judgment"] == "substitute":
        parts.append("Δv_max")
    return f"잠정 결과: 대체값 사용({', '.join(parts)})" if parts else None


# ---------------------------------------------------------------- periods (revised spec §5)
def resolve_periods(period: str, years, now_day: pd.Timestamp) -> list[dict]:
    """Turns the period setting into a list of periods: whole past years or the last 365 days."""
    if period == "last365":
        start, end = now_day - pd.Timedelta(days=365), now_day - pd.Timedelta(days=1)
        return [{"label": f"last365_{start:%Y%m%d}", "mode": "last365", "year": None,
                 "start": start, "end": end}]
    if period != "year":
        raise SystemExit("period는 year 또는 last365여야 함")
    last_ok = now_day.year - 1
    years = [int(y) for y in (years or [last_ok])]
    bad = [y for y in years if y > last_ok]
    if bad:
        raise SystemExit(f"끝나지 않은 연도 {bad}는 쓸 수 없음 - 사용 가능한 마지막 연도: {last_ok}")
    return [{"label": str(y), "mode": "year", "year": y,
             "start": pd.Timestamp(y, 1, 1), "end": pd.Timestamp(y, 12, 31)} for y in years]


# ---------------------------------------------------------------- hourly conditions (original spec §4.4-4.6)
MODE_OFF, MODE_HEAT, MODE_COOL = 0, 1, 2


def hvac_mode(index: pd.DatetimeIndex, cfg: dict) -> np.ndarray:
    """HVAC mode for every hour: heating or cooling during the set hours and months, otherwise off."""
    on = np.isin(index.hour, cfg["hvac_hours"])
    heat = on & np.isin(index.month, cfg["heating_months"])
    cool = on & np.isin(index.month, cfg["cooling_months"]) & ~heat
    return np.where(heat, MODE_HEAT, np.where(cool, MODE_COOL, MODE_OFF))


def indoor_conditions(w: pd.DataFrame, cfg: dict, dv_max: float):
    """Indoor temperature from assumptions A3/A4 and indoor vapour pressure from A1. (mode, theta_i, p_i)"""
    mode = hvac_mode(w.index, cfg)
    theta_e = w["theta_e"].to_numpy()
    theta_i = np.select([mode == MODE_HEAT, mode == MODE_COOL],
                        [cfg["T_heat_set"], cfg["T_cool_set"]], theta_e)
    g = g_shape(w["theta_e24"].to_numpy(), cfg["T_low"], cfg["T_high"])
    p_i = w["p_e"].to_numpy() + dv_to_dp(dv_max * g, theta_i)
    p_i = np.minimum(p_i, p_sat(theta_i))
    return mode, theta_i, p_i


def surface_temperature(theta_g, theta_i, hvac_on, f):
    """Assumptions A2-A4: theta_g + f (theta_i - theta_g) while heating or cooling runs, otherwise theta_g."""
    theta_g, theta_i = np.asarray(theta_g, dtype=float), np.asarray(theta_i, dtype=float)
    return np.where(hvac_on, theta_g + np.asarray(f) * (theta_i - theta_g), theta_g)


def surface_rh(p_i, theta_si):
    """Assumption A5: vapour pressure at the surface = indoor vapour pressure. [%], can exceed 100 (condensation)."""
    return 100 * np.asarray(p_i, dtype=float) / p_sat(theta_si)


# ---------------------------------------------------------------- VTT model (original spec §5)
MATERIALS = {   # k1(M<1), k1(M>=1), A, B, C, RH_min
    "very_sensitive":   dict(k1a=1.0,   k1b=2.0,   A=1.0, B=7.0, C=2.0, RH_min=80.0),
    "sensitive":        dict(k1a=0.578, k1b=0.386, A=0.3, B=6.0, C=1.0, RH_min=80.0),
    "medium_resistant": dict(k1a=0.072, k1b=0.097, A=0.0, B=5.0, C=1.5, RH_min=85.0),
    "resistant":        dict(k1a=0.033, k1b=0.014, A=0.0, B=3.0, C=1.0, RH_min=85.0),
}


def rh_crit(T, RH_min):
    """Critical relative humidity. The cubic curve is for wood, so the max with RH_min is used (implementation choice)."""
    T = np.asarray(T, dtype=float)
    cubic = -0.00267 * T**3 + 0.160 * T**2 - 3.13 * T + 100.0
    return np.where(T <= 20, np.maximum(cubic, RH_min), RH_min)


def material_arrays(names) -> dict:
    """Collects each VTT material parameter into an array with one value per face."""
    return {key: np.array([MATERIALS[n][key] for n in names]) for key in MATERIALS["sensitive"]}


def vtt_step(M, T, RH, n_off, mat: dict, C_mat: float):
    """Advances one hour. n_off = hours since growth conditions ended. Returns (M, n_off)."""
    rc = rh_crit(T, mat["RH_min"])
    grow = (T > 0) & (RH >= rc)

    Tg = np.where(grow, T, 1.0)            # avoids log warnings in cells that are not growing
    RHg = np.where(grow, RH, 100.0)
    k1 = np.where(M < 1, mat["k1a"], mat["k1b"])
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(grow, (rc - RHg) / (rc - 100.0), 0.0)
    M_max = mat["A"] + mat["B"] * x - mat["C"] * x**2
    k2 = np.maximum(1 - np.exp(2.3 * (M - M_max)), 0.0)
    rate_day = k1 * k2 / (7 * np.exp(-0.68 * np.log(Tg) - 13.9 * np.log(RHg) + 66.02))

    n_off = np.where(grow, 0, n_off + 1)
    decline = C_mat * np.select([n_off <= 6, n_off <= 24], [-0.00133, 0.0], -0.000667)
    M = np.clip(M + np.where(grow, rate_day / 24, decline), 0.0, 6.0)
    return M, n_off


def run_vtt(T, RH, materials, C_mat: float, M0=0.0) -> np.ndarray:
    """T, RH: (hour, face) arrays. Returns M after each hour, in the same shape."""
    T, RH = np.atleast_2d(T), np.atleast_2d(RH)
    mat = material_arrays(materials)
    M = np.full(T.shape[1], float(M0))
    n_off = np.zeros(T.shape[1], dtype=int)
    out = np.empty_like(T, dtype=float)
    for t in range(T.shape[0]):
        M, n_off = vtt_step(M, T[t], RH[t], n_off, mat, C_mat)
        out[t] = M
    return out


# ---------------------------------------------------------------- simulation
def simulate(w: pd.DataFrame, f: np.ndarray, materials: list[str], cfg: dict, dv_max: float) -> dict:
    """Simulates one period. Per-face arrays are (hour, face)."""
    mode, theta_i, p_i = indoor_conditions(w, cfg, dv_max)
    theta_g = w["theta_g"].to_numpy()
    theta_si = surface_temperature(theta_g[:, None], theta_i[:, None], (mode != MODE_OFF)[:, None],
                                   np.asarray(f)[None, :])
    phi = surface_rh(p_i[:, None], theta_si)
    rh_si = np.minimum(phi, 100.0)
    M = run_vtt(theta_si, rh_si, materials, cfg["C_mat"])
    return {"mode": mode, "theta_i": theta_i, "p_i": p_i, "theta_si": theta_si,
            "rh_si": rh_si, "condensation": phi > 100.0, "M": M}


def iso(ts) -> str:
    """Formats a timestamp as ISO with the KST offset."""
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%S") + TZ_SUFFIX


def first_time(index: pd.DatetimeIndex, hit: np.ndarray):
    """Returns the first time a condition holds, or None."""
    i = np.flatnonzero(hit)
    return iso(index[i[0]]) if len(i) else None


SUMMARY_METRICS = ["M_max_year", "M_final", "first_M1_time", "first_M3_time", "condensation_hours", "months_rh80"]


def summarize(index: pd.DatetimeIndex, res: dict, faces: pd.DataFrame) -> pd.DataFrame:
    """Per-face summary of one scenario. Columns = SUMMARY_METRICS."""
    rh_month = pd.DataFrame(res["rh_si"], index=index).groupby(index.month).mean()
    rows = []
    for j, k in enumerate(faces.index):
        M = res["M"][:, j]
        rows.append({
            "face": k,
            "M_max_year": round(float(M.max()), 3),
            "M_final": round(float(M[-1]), 3),
            "first_M1_time": first_time(index, M >= 1),
            "first_M3_time": first_time(index, M >= 3),
            "condensation_hours": int(res["condensation"][:, j].sum()),
            "months_rh80": [int(m) for m in rh_month.index[rh_month[j] >= 80]],
        })
    return pd.DataFrame(rows).set_index("face")


def scenario_summary(index: pd.DatetimeIndex, results: dict, faces: pd.DataFrame) -> pd.DataFrame:
    """Per-face position and f judgment plus the summary of each scenario (columns end in _central, _worst)."""
    out = pd.DataFrame({"zone": faces["zone"], "az": faces["az_deg"], "el": faces["el_deg"],
                        "f_judgment": faces["judgment"],
                        "sigma_f": faces["sigma_f"].map(lambda v: None if pd.isna(v) else round(float(v), 3)),
                        "f_central": faces["f_central"].astype(float).round(3),
                        "f_worst": faces["f_worst"].astype(float).round(3)})
    for scen, res in results.items():
        s = summarize(index, res, faces)
        out = out.join(s.add_suffix(f"_{scen}"))
    return out


# ---------------------------------------------------------------- output (original spec §6, revised spec §6.3)
def write_text(path: Path, text: str) -> None:
    """Writes a text file atomically (through a temp file)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_hourly(path: Path, w: pd.DataFrame, results: dict, faces: pd.DataFrame) -> None:
    """Writes the hourly results of every scenario, hour and face to a CSV file."""
    frames = []
    for scen, res in results.items():
        n, K = res["M"].shape
        rep = lambda a: np.repeat(np.asarray(a), K)
        frames.append(pd.DataFrame({
            "timestamp": rep(w.index.strftime("%Y-%m-%dT%H:%M:%S") + TZ_SUFFIX),
            "scenario": scen,
            "face": np.tile(faces.index.to_numpy(), n),
            "mode": rep(np.array(["off", "heat", "cool"])[res["mode"]]),
            "theta_e": rep(w["theta_e"].round(2)),
            "theta_g": rep(w["theta_g"].round(2)),
            "theta_i": rep(np.round(res["theta_i"], 2)),
            "p_e": rep(w["p_e"].round(3)),
            "p_i": rep(np.round(res["p_i"], 3)),
            "theta_si": np.round(res["theta_si"].ravel(), 2),
            "rh_si": np.round(res["rh_si"].ravel(), 2),
            "condensation": res["condensation"].ravel().astype(int),
            "M": np.round(res["M"].ravel(), 4),
            "data_flag": rep(w["data_flag"]),
        }))
    tmp = path.with_suffix(".csv.tmp")
    pd.concat(frames, ignore_index=True).to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(path)


def write_summary(path: Path, summary: pd.DataFrame, notes: list[str]) -> None:
    """Writes notes at the top as comment lines starting with '# ' (pandas: read_csv(..., comment='#'))."""
    out = summary.copy()
    for c in out.columns:
        if c.startswith("months_rh80"):
            out[c] = out[c].map(lambda ms: ";".join(map(str, ms)))
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        for line in notes:
            f.write(f"# {line}\n")
        out.to_csv(f)
    tmp.replace(path)


def runs(mask: np.ndarray) -> list[list[int]]:
    """Runs of True as a list of [start index, length]."""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [[int(s), int(e - s)] for s, e in zip(starts, ends)]


def _num(v, d=3):
    """Rounds a number, or returns None for missing values."""
    return None if v is None or pd.isna(v) else round(float(v), d)


def viewer_json(period: dict, session: dict, station: dict, w: pd.DataFrame, results: dict,
                all_faces: pd.DataFrame, sim_faces: pd.DataFrame, summary: pd.DataFrame, dv: dict,
                cfg: dict, provisional: str | None, warnings: list) -> dict:
    """Data for the HTML viewer. Time axis = start + step_h; per-face values are length-n arrays per scenario. Excluded faces are null."""
    r = lambda a, d: np.round(np.asarray(a, dtype=float), d).tolist()
    col = {k: j for j, k in enumerate(sim_faces.index)}
    faces = []
    for k, fc in all_faces.iterrows():
        rec = {"face": int(k), "zone": fc["zone"] if isinstance(fc["zone"], str) else None,
               "az": _num(fc["az_deg"], 2), "el": _num(fc["el_deg"], 2),
               "material": fc["material"], "excluded": bool(fc["excluded"]),
               "f_judgment": fc["judgment"], "f_raw": _num(fc["f_raw"]), "sigma_f": _num(fc["sigma_f"]),
               "f": {"central": _num(fc["f_central"]), "worst": _num(fc["f_worst"])}}
        for scen in SCENARIOS:
            if k not in col:
                rec[scen] = None
                continue
            j, res, s = col[k], results[scen], summary.loc[k]
            rec[scen] = {
                "summary": {"M_max": s[f"M_max_year_{scen}"], "M_final": s[f"M_final_{scen}"],
                            "t_M1": s[f"first_M1_time_{scen}"], "t_M3": s[f"first_M3_time_{scen}"],
                            "condensation_h": int(s[f"condensation_hours_{scen}"]),
                            "months_rh80": s[f"months_rh80_{scen}"]},
                "M": r(res["M"][:, j], 3),
                "theta_si": r(res["theta_si"][:, j], 2),
                "rh_si": r(res["rh_si"][:, j], 1),
                "condensation": runs(res["condensation"][:, j]),
            }
        faces.append(rec)
    any_res = results["central"]
    return {
        "schema": "mold-sim/2",
        "session": session.get("session"),
        "period": {"label": period["label"], "mode": period["mode"], "year": period["year"],
                   "start": iso(period["start"]), "end": iso(period["end"] + pd.Timedelta(hours=23))},
        "site": session.get("site"),
        "heading_deg": (session.get("heading") or {}).get("deg"),
        "station": station,
        "time": {"start": iso(w.index[0]), "step_h": 1, "n": len(w)},
        "provisional": provisional is not None,
        "provisional_message": provisional,
        "scenarios": list(SCENARIOS),
        "params": {"dv_max": {"judgment": dv["judgment"], "raw": _num(dv["dv_max_raw"]),
                              "sigma": _num(dv["sigma"]), "central": _num(dv["central"]),
                              "worst": _num(dv["worst"])},
                   "C_mat": cfg["C_mat"], "ground_depth_m": cfg["ground_depth_m"]},
        "mode_labels": ["off", "heat", "cool"],
        "series": {
            "theta_e": r(w["theta_e"], 2), "theta_g": r(w["theta_g"], 2),
            "theta_i": r(any_res["theta_i"], 2),
            "rh_i": {scen: r(100 * res["p_i"] / p_sat(res["theta_i"]), 1) for scen, res in results.items()},
            "mode": any_res["mode"].astype(int).tolist(),
            "data_flag": w["data_flag"].astype(int).tolist(),
        },
        "faces": faces,
        "warnings": warnings,
    }


def save_plots(out_dir: Path, label: str, w: pd.DataFrame, results: dict, faces: pd.DataFrame) -> None:
    """Saves a PNG chart of M and surface RH over the period for each face."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pdir = out_dir / "plots"
    pdir.mkdir(exist_ok=True)
    for j, k in enumerate(faces.index):
        fig, ax = plt.subplots(figsize=(11, 3.5))
        ax2 = ax.twinx()
        ax2.plot(w.index, results["worst"]["rh_si"][:, j], color="#9bb7d4", lw=0.5, label="surface RH (worst)")
        ax.plot(w.index, results["central"]["M"][:, j], color="#c0392b", lw=1.6, label="M central")
        ax.plot(w.index, results["worst"]["M"][:, j], color="#c0392b", lw=1.2, ls="--", label="M worst")
        ax.set_ylim(0, 6)
        ax2.set_ylim(0, 100)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.set_ylabel("Mould index M")
        ax2.set_ylabel("Surface RH (%)")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title(f"Face {k} ({faces.at[k, 'zone']}, az {faces.at[k, 'az_deg']}°, "
                     f"el {faces.at[k, 'el_deg']}°) - {label}")
        fig.tight_layout()
        fig.savefig(pdir / f"face{k:02d}_{label}.png", dpi=110)
        plt.close(fig)


def print_ranking(label: str, summary: pd.DataFrame, provisional: str | None) -> None:
    """Prints the faces ranked by their worst-scenario peak mold index."""
    if provisional:
        print(f"\n*** {provisional} ***")
    print(f"\n[{label}] 면별 최대 곰팡이 지수 M (중앙 / 보수 시나리오, M >= 3: 육안 관찰 수준)")
    for k, s in summary.sort_values("M_max_year_worst", ascending=False).iterrows():
        mark = "  <== 육안 관찰 수준" if s["M_max_year_worst"] >= 3 else ""
        print(f"  면 {k:2d} ({s['zone']}, az {s['az']:5.1f}, el {s['el']:6.2f}, f {s['f_judgment']})  "
              f"M_max {s['M_max_year_central']:.2f} / {s['M_max_year_worst']:.2f}  "
              f"결로 {s['condensation_hours_central']:4d}h / {s['condensation_hours_worst']:4d}h{mark}")


# ---------------------------------------------------------------- main
def main() -> None:
    """Parses the arguments, extracts f and dv_max from the scans, simulates each period and writes the results."""
    ap = argparse.ArgumentParser(description="표면 온습도·곰팡이(VTT) 시뮬레이터")
    ap.add_argument("--session-dir", required=True, help="logger.py 세션 폴더")
    ap.add_argument("--config", help="config.yaml (기본: 세션 폴더 또는 이 스크립트 폴더의 config.yaml)")
    ap.add_argument("--years", type=int, nargs="+", help="시뮬레이션 기상 연도 (끝난 연도만)")
    ap.add_argument("--period", choices=["year", "last365"], help="year(연도별) 또는 last365(어제까지 365일)")
    ap.add_argument("--out-dir", help="출력 폴더 (기본: <session-dir>/sim)")
    ap.add_argument("--cache-dir", help="ASOS 캐시 폴더 (기본: <session-dir>/../kma_cache)")
    ap.add_argument("--plot", action="store_true", help="면별 M·표면 RH 그래프 PNG 저장")
    args = ap.parse_args()

    session_dir = Path(args.session_dir)
    out_dir = Path(args.out_dir) if args.out_dir else session_dir / "sim"
    cache_dir = Path(args.cache_dir) if args.cache_dir else session_dir.parent / "kma_cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(args.config) if args.config else next(
        (p for p in (session_dir / "config.yaml", Path(__file__).with_name("config.yaml")) if p.exists()),
        None)
    cfg, _ = load_config(cfg_path)
    session, scans = load_session(session_dir)
    period_mode = "year" if args.years else (args.period or cfg["period"])
    periods = resolve_periods(period_mode, args.years or cfg["sim_years"], today())
    depth = float(cfg["ground_depth_m"])
    if depth not in GROUND_COLS:
        raise SystemExit(f"ground_depth_m은 {list(GROUND_COLS)} 중 하나여야 함 (현재 {depth})")
    warnings: list[str] = []

    if cfg["station_id"] is not None:
        station_info = {"id": int(cfg["station_id"]), "source": "config station_id"}
    elif session.get("asos_station"):
        station_info = {"id": int(session["asos_station"]), "source": "session.yaml asos_station"}
    else:
        station_info = choose_station(session, session_dir / "session.yaml", depth,
                                      pd.Period(periods[0]["start"], freq="M"), cache_dir, warnings)
    station = cfg["station_id"] = station_info["id"]
    print(f"세션 {session.get('session')} | 설정 {cfg_path or '기본값'} | "
          f"관측소 {station} {station_info.get('name', '')} ({station_info['source']}) | "
          f"지중온도 {depth} m | 기간 {[p['label'] for p in periods]}")

    # 1. outdoor air and ground temperature at scan time -> dv_max and f per face (with uncertainty and judgment)
    faces = face_table(scans, session, cfg, warnings)
    valid = filter_scans(scans, faces, cfg, warnings)
    times = scans.groupby("scan_id")["timestamp"].first()
    floors = times.dt.floor("h")
    need = sorted({f - pd.Timedelta(hours=i) for f in floors for i in range(24)})
    rows = fill_outdoor(session_dir / "outdoor.csv", need, session, station, cfg, cache_dir, warnings)
    outdoor = select_outdoor(rows, cfg["outdoor_source"])
    lag_max = int(cfg["ground_lag_max_days"])
    tg_meas = ground_daily(station, times.min().normalize() - pd.Timedelta(days=lag_max + 2),
                           times.max().normalize(), depth, cache_dir)
    per = scan_conditions(scans, outdoor, tg_meas)
    dv = extract_dv_max(per, cfg, warnings)
    fres = extract_f(valid, per, faces, cfg, warnings)
    faces = faces.join(fres)
    sim_faces = faces[~faces["excluded"] & faces["f_central"].notna()]
    provisional = provisional_message(fres, dv)
    if provisional:
        print(f"*** {provisional} ***")
    print(f"dv_max 중앙 {dv['central']} / 보수 {_num(dv['worst'], 2)} g/m3 ({dv['judgment']}, "
          f"sigma {_num(dv['sigma'], 2)}) | f 판정 {fres['judgment'].value_counts().to_dict()} | "
          f"시뮬레이션 면 {len(sim_faces)}개")

    # 2. simulation for each period (central and worst scenarios)
    index_entries, compare = [], {}
    for p in periods:
        pwarn: list[str] = []
        w = load_weather(station, p["start"], p["end"], depth, cache_dir, pwarn, p["label"])
        results = {scen: simulate(w, sim_faces[f"f_{scen}"].to_numpy(dtype=float),
                                  list(sim_faces["material"]), cfg, dv[scen]) for scen in SCENARIOS}
        summary = scenario_summary(w.index, results, sim_faces)
        notes = [provisional] if provisional else []
        if p["mode"] == "last365":
            notes.append("last365 결과는 시작 계절이 연도 모드와 달라(첫 시각에 M = 0) 연도 결과와 직접 비교하지 않음")

        write_hourly(out_dir / f"hourly_{p['label']}.csv", w, results, sim_faces)
        write_summary(out_dir / f"summary_{p['label']}.csv", summary, notes)
        view = viewer_json(p, session, station_info, w, results, faces, sim_faces, summary, dv, cfg,
                           provisional, warnings + pwarn)
        write_text(out_dir / f"viewer_{p['label']}.json",
                   json.dumps(view, ensure_ascii=False, separators=(",", ":")))
        if args.plot:
            save_plots(out_dir, p["label"], w, results, sim_faces)

        warnings += pwarn
        index_entries.append({"label": p["label"], "mode": p["mode"], "file": f"viewer_{p['label']}.json",
                              "M_max_central": float(summary["M_max_year_central"].max()),
                              "M_max_worst": float(summary["M_max_year_worst"].max()),
                              "provisional": provisional is not None})
        compare[p["label"]] = summary[[f"M_max_year_{s}" for s in SCENARIOS]]
        print_ranking(p["label"], summary, provisional)

    if len(periods) > 1:
        cmp = pd.concat(compare, axis=1)
        cmp.columns = [f"{c}_{lab}" for lab, c in cmp.columns]
        cmp.to_csv(out_dir / "compare_periods.csv", encoding="utf-8")

    # 3. params.json, viewer_index.json
    params = {
        "generated_at": now_iso(),
        "session": session.get("session"),
        "site": session.get("site"),
        "heading": session.get("heading"),
        "grid": session.get("grid"),
        "station": station_info,
        "periods": [{"label": p["label"], "mode": p["mode"], "start": iso(p["start"]),
                     "end": iso(p["end"] + pd.Timedelta(hours=23))} for p in periods],
        "provisional": provisional is not None,
        "provisional_message": provisional,
        "dv_max": dv,
        "faces": [{"face": int(k), "zone": fc["zone"] if isinstance(fc["zone"], str) else None,
                   "material": fc["material"], "excluded": bool(fc["excluded"]),
                   "judgment": fc["judgment"], "n_scans": int(fc["n_scans"]),
                   "f_raw": _num(fc["f_raw"], 4), "sigma_f": _num(fc["sigma_f"], 4),
                   "f_central": _num(fc["f_central"], 4), "f_worst": _num(fc["f_worst"], 4)}
                  for k, fc in faces.iterrows()],
        "scans": [{"scan_id": int(sid), "timestamp": iso(s["timestamp"]), "out_hour": iso(s["out_hour"]),
                   "out_source": s["out_source"] if isinstance(s["out_source"], str) else None,
                   "theta_e": _num(s["theta_e"], 2), "theta_e24": _num(s["theta_e24"], 2),
                   "n_e24": int(s["n_e24"]), "theta_g": _num(s["theta_g"], 2),
                   "ground_lag": None if pd.isna(s["ground_lag"]) else int(s["ground_lag"])}
                  for sid, s in per.iterrows()],
        "excluded_faces": [int(k) for k in faces.index[faces["excluded"]]],
        "config": cfg,
        "config_file": str(cfg_path) if cfg_path else None,
        "warnings": warnings,
    }
    write_text(out_dir / "params.json", json.dumps(params, ensure_ascii=False, indent=2,
                                                   default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    write_text(out_dir / "viewer_index.json", json.dumps(
        {"schema": "mold-sim/2", "session": session.get("session"), "periods": index_entries},
        ensure_ascii=False, indent=2))

    if warnings:
        print("\n경고")
        for msg in warnings:
            print(f"  - {msg}")
    print(f"\n결과 폴더: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
