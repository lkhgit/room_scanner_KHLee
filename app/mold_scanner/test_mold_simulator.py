"""Tests for mold_simulator.py.

Tests 1-12 from the original spec §8, tests 13-27 from the revised spec §7. Every API call is replaced by a fake
reply or a cache file.
Run: pytest test_mold_simulator.py
"""

import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import yaml

import kma
import mold_simulator as ms

rel = lambda x: pytest.approx(x, rel=0.01)
CFG = ms.DEFAULTS


def ncst_response(code="00", t=15.2, rh=60.0):
    """Builds a fake ultra-short-term observation reply (code "00" with T1H/REH, or an error code)."""
    if code != "00":
        return {"response": {"header": {"resultCode": code, "resultMsg": "NO_DATA"}}}
    return {"response": {"header": {"resultCode": "00", "resultMsg": "OK"},
                         "body": {"items": {"item": [{"category": "T1H", "obsrValue": str(t)},
                                                     {"category": "REH", "obsrValue": str(rh)}]}}}}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Tests never use the network: ASOS comes from the cache only, and ultra-short-term observations answer 'no data' by default."""
    def no_asos(*a, **k):
        """Fails the test if an ASOS request is made."""
        raise AssertionError("테스트 중 ASOS API 호출 - 캐시 파일이 빠졌음")
    monkeypatch.setattr(ms, "_kma_request", no_asos)
    monkeypatch.setattr(kma, "_ncst_request", lambda base, nx, ny: ncst_response("03"))
    monkeypatch.setattr(kma, "_kakao_search", lambda url, q: None)


# ---------------------------------------------------------------- 1-7 moist air, parameters, surface temperature
def test_01_p_sat():
    """Saturation vapour pressure at 20 °C and 0 °C."""
    assert ms.p_sat(20) == pytest.approx(23.33, abs=0.01)
    assert ms.p_sat(0) == pytest.approx(6.109, abs=0.001)


def test_02_rho_v():
    """Vapour density from vapour pressure."""
    assert ms.rho_v(23.33, 20) == rel(17.25)


def test_03_g_shape():
    """Moisture load shape g at cold, mild and warm outdoor temperatures."""
    assert ms.g_shape([-5, 10, 25]).tolist() == [1.0, 0.5, 0.0]


def test_04_dv_max():
    """dv_max fit and rounding to the humidity class."""
    v = ms.dv_max_fit([0.25, 0.5], [1.5, 3.0])
    assert v == rel(6.0)
    assert ms.round_dv_class(v) == 6


def test_05_f_fit():
    """Temperature factor f from one surface, indoor and ground reading."""
    assert ms.f_fit([17], [20], [10]) == rel(0.70)


def test_06_surface_heating():
    """Surface temperature while heating."""
    assert ms.surface_temperature(5, 20, True, 0.70) == rel(15.5)


def test_07_surface_off():
    """Surface temperature with the HVAC off."""
    assert ms.surface_temperature(5, 20, False, 0.70) == rel(5.0)


# ---------------------------------------------------------------- 8-11 VTT
def hours_to_reach(M, level):
    """Returns the first hour (1-based) at which M reaches level."""
    return int(np.argmax(M >= level)) + 1


def test_08_vtt_growth_very_sensitive():
    """VTT growth time to M = 1 and M = 3 for very sensitive material at 22 °C, 97 % RH."""
    n = 24 * 40
    M = ms.run_vtt(np.full((n, 1), 22.0), np.full((n, 1), 97.0), ["very_sensitive"], 0.5)[:, 0]
    assert hours_to_reach(M, 1) == pytest.approx(234, rel=0.03)
    assert hours_to_reach(M, 3) == pytest.approx(467, rel=0.03)


def test_09_vtt_growth_sensitive():
    """VTT growth time and maximum M for sensitive material."""
    n = 24 * 200
    M = ms.run_vtt(np.full((n, 1), 22.0), np.full((n, 1), 97.0), ["sensitive"], 0.5)[:, 0]
    assert hours_to_reach(M, 1) / 24 == rel(16.8)
    assert M.max() == pytest.approx(4.68, abs=0.01)


def test_10_vtt_decline():
    """Mold index decline in dry conditions."""
    M = ms.run_vtt(np.full((48, 1), 20.0), np.full((48, 1), 50.0), ["very_sensitive"], 1.0, M0=2.0)
    assert M[-1, 0] == pytest.approx(1.976, abs=0.001)


def test_11_no_growth_below_zero():
    """No growth below 0 °C, only the decline."""
    M = ms.run_vtt(np.full((10, 1), -1.0), np.full((10, 1), 99.0), ["very_sensitive"], 1.0, M0=1.0)
    dM = np.diff(np.concatenate([[1.0], M[:, 0]]))
    assert np.all(dM <= 0) and dM[:6] == pytest.approx([-0.00133] * 6)   # decline formula (0 for hours 7-24)


# ---------------------------------------------------------------- automatic station choice
def test_haversine_one_degree_latitude():
    """Haversine distance of one degree of latitude."""
    assert kma.haversine_km(37, 127, 38, 127) == pytest.approx(111.19, abs=0.1)


def test_choose_station_by_name(tmp_path, monkeypatch):
    """Site with only a name -> geocoded with fewer words -> the nearest station without ground temperature is skipped."""
    stations = tmp_path / "asos_stations.csv"
    stations.write_text(
        "지점번호 ,시작일 ,종료일 ,지점명 ,지점주소 ,위도 ,경도 ,관측장소 해발고도(m) \n"
        "108,1907-10-01,2016-12-31,서울,옛 위치,37.0,127.5,85\n"
        "108,2017-01-01,,서울,서울 종로구,37.5714,126.9658,85\n"
        "112,1904-08-01,,인천,인천 중구,37.4777,126.6249,71\n"
        "999,2010-01-01,,가짜,망원동,37.5560,126.9060,10\n", encoding="cp949")
    monkeypatch.setattr(kma, "STATION_CSV", stations)

    cache = tmp_path / "kma_cache"
    cache.mkdir()
    cols = list(ms.GROUND_COLS.values())
    pd.DataFrame({"tm": ["2024-01-01"], **{c: [""] for c in cols}}).to_csv(cache / "999_daily_202401.csv", index=False)
    pd.DataFrame({"tm": ["2024-01-01"], **{c: [5.0] for c in cols}}).to_csv(cache / "108_daily_202401.csv", index=False)
    pd.DataFrame({"tm": ["2024-01-01"], **{c: [5.0] for c in cols}}).to_csv(cache / "112_daily_202401.csv", index=False)

    queries = []

    def fake_search(url, query):
        """Records each query and finds only "망원동" by address search."""
        queries.append(query)
        if query == "망원동" and url == kma.KAKAO_ADDRESS_URL:
            return 37.5560, 126.9050, "서울 마포구 망원동"
        return None
    monkeypatch.setattr(kma, "_kakao_search", fake_search)

    session = {"session": "t", "site": {"name": "망원동 반지하", "lat": None, "lon": None}}
    spath = tmp_path / "session.yaml"
    info = ms.choose_station(session, spath, 1.0, pd.Period("2024-01", freq="M"), cache, [])

    assert queries[:2] == ["망원동 반지하", "망원동 반지하"] and "망원동" in queries
    assert info["id"] == 108 and info["distance_km"] == pytest.approx(5.6, abs=0.5)
    saved = yaml.safe_load(spath.read_text(encoding="utf-8"))
    assert saved["site"]["lat"] == pytest.approx(37.556) and saved["site"]["geocoded_from"] == "망원동"


# ---------------------------------------------------------------- 13-16 outdoor air at scan time
def test_13_grid():
    """KMA grid cells for Seoul and Busan."""
    assert kma.latlon_to_grid(37.5665, 126.978) == (60, 127)
    assert kma.latlon_to_grid(37.5714, 126.9658) == (60, 127)
    assert kma.latlon_to_grid(35.1796, 129.0756) == (98, 76)


def test_14_base_time_fallback(tmp_path, monkeypatch):
    """Falls back to the previous hour when the current hour has no observation."""
    asked = []

    def fake(base, nx, ny):
        """Answers 'no data' for 20:00 and a valid reading for other hours."""
        asked.append(base.hour)
        return ncst_response("03") if base.hour == 20 else ncst_response("00", 15.2, 60)
    monkeypatch.setattr(kma, "_ncst_request", fake)
    path = tmp_path / "outdoor.csv"
    added, failed = kma.collect_ncst(path, [datetime(2026, 10, 21, 20)], 60, 127)
    rows = kma.read_outdoor(path)
    assert asked == [20, 19] and added == 1
    assert rows[0]["hour"] == datetime(2026, 10, 21, 19) and rows[0]["source"] == "ncst"
    assert rows[0]["p_out_hpa"] == pytest.approx(0.6 * float(kma.p_sat(15.2)), abs=0.001)


def test_15_outdoor_missing():
    """Without outdoor data dv_max falls back to the substitute and the scan is left out."""
    assert kma.fetch_ncst(datetime(2026, 10, 21, 20), 60, 127) is None     # both requests answer 'no data'
    per = pd.DataFrame({"timestamp": [pd.Timestamp("2026-10-21 20:56")], "t_air_c": [23.3], "rh_pct": [33.2],
                        "theta_e": [np.nan], "p_e": [np.nan], "theta_e24": [np.nan]})
    warnings = []
    dv = ms.extract_dv_max(per, CFG, warnings)
    assert dv["judgment"] == "substitute" and dv["n_scans"] == 0
    assert any("dv_max 추출에서 뺌" in w for w in warnings)


def test_16_source_priority():
    """auto prefers ASOS over the ultra-short-term observation; ncst picks the observation."""
    h = datetime(2026, 10, 20, 12)
    rows = [{"hour": h, "source": "ncst", "t_out_c": 18.0, "rh_out_pct": 50, "p_out_hpa": 10.3},
            {"hour": h, "source": "asos", "t_out_c": 17.0, "rh_out_pct": 55, "p_out_hpa": 10.6}]
    assert ms.select_outdoor(rows, "auto").loc[pd.Timestamp(h), "t_out_c"] == 17.0
    assert ms.select_outdoor(rows, "ncst").loc[pd.Timestamp(h), "t_out_c"] == 18.0


# ---------------------------------------------------------------- 17-18 ground temperature lag
def test_17_ground_lag():
    """Ground temperature from an earlier day, and its lag in the f uncertainty."""
    tg = pd.Series([20.1, 20.0], index=pd.to_datetime(["2026-10-18", "2026-10-19"]))
    v, lag = ms.ground_for_day(tg, pd.Timestamp("2026-10-21"))
    assert v == 20.0 and lag == 2
    zero = {**CFG, "sigma_ir": 0, "sigma_sht": 0, "sigma_site": 0}
    assert ms.sigma_f(0.0, [1.0], [0.0], [lag], zero) == pytest.approx(0.4)    # 0.2 °C x lag


def test_18_ground_lag_too_old():
    """Scans whose ground temperature is too old are left out of the f estimate."""
    scans = pd.DataFrame({"scan_id": [1], "face": [1], "t_surf_c": [17.0], "t_surf_sd_c": [0.04], "t_air_c": [20.0]})
    per = pd.DataFrame({"theta_g": [10.0], "ground_lag": [4]}, index=pd.Index([1], name="scan_id"))
    faces = pd.DataFrame({"excluded": [False]}, index=pd.Index([1], name="face"))
    warnings = []
    fres = ms.extract_f(scans, per, faces, CFG, warnings)
    assert fres.at[1, "n_scans"] == 0 and fres.at[1, "judgment"] == "substitute"
    assert any("f 추출에서 뺌" in w for w in warnings)


# ---------------------------------------------------------------- 19-22 uncertainty and substitutes
@pytest.mark.parametrize("delta, expect, judged", [(1, 0.531, "uncertain"), (2, 0.266, "uncertain"),
                                                    (6, 0.089, "measured")])
def test_19_sigma_f(delta, expect, judged):
    """sigma_f and its judgment for several indoor-ground differences."""
    s = ms.sigma_f(0.75, [delta], [0.04], [1], CFG)
    assert s == pytest.approx(expect, abs=0.005)
    assert ("measured" if s <= CFG["sigma_f_max"] else "uncertain") == judged


def one_scan_f(theta_si, theta_i=20.0, theta_g=18.0):
    """Runs extract_f on a single scan of one face."""
    scans = pd.DataFrame({"scan_id": [1], "face": [1], "t_surf_c": [theta_si], "t_surf_sd_c": [0.04],
                          "t_air_c": [theta_i]})
    per = pd.DataFrame({"theta_g": [theta_g], "ground_lag": [1]}, index=pd.Index([1], name="scan_id"))
    faces = pd.DataFrame({"excluded": [False]}, index=pd.Index([1], name="face"))
    return ms.extract_f(scans, per, faces, CFG, [])


def test_single_scan_uses_measured_f():
    """One scan, 2 °C between indoor and ground: the measured f is used even though it is uncertain."""
    fres = one_scan_f(19.66)                                   # f = 1.66 / 2 = 0.83
    assert fres.at[1, "judgment"] == "uncertain"
    assert fres.at[1, "f_central"] == pytest.approx(0.83)
    s = fres.at[1, "sigma_f"]
    assert fres.at[1, "f_worst"] == pytest.approx(max(0.0, 0.83 - 2 * s))
    assert ms.provisional_message(fres, {"judgment": "uncertain"}) is None


def test_anomaly_f_is_clipped_not_substituted():
    """An f outside [0, 1] by more than 2 sigma is clipped, not replaced."""
    fres = one_scan_f(22.0)                                    # f = 4 / 2 = 2.0 > 1 + 2 sigma
    assert fres.at[1, "judgment"] == "anomaly" and fres.at[1, "f_central"] == 1.0


def test_single_scan_uses_measured_dv():
    """A single scan's dv_max is used even though it is uncertain."""
    per = pd.DataFrame({"t_air_c": [23.3], "rh_pct": [60.0], "theta_e": [10.0], "p_e": [8.0],
                        "theta_e24": [14.0]})
    warnings = []
    dv = ms.extract_dv_max(per, CFG, warnings)
    raw = dv["dv_max_raw"]
    assert dv["judgment"] == "uncertain" and dv["central"] == ms.round_dv_class(max(raw, 0))
    assert dv["worst"] == pytest.approx(max(dv["central"], min(max(raw, 0) + 2 * dv["sigma"], 10)))
    assert any("측정값" in w for w in warnings)


def test_20_sigma_dv_single_autumn_scan():
    """dv_max uncertainty of a single autumn scan."""
    s = ms.sigma_dv_max([0.25], [15.0], [23.3], CFG)
    assert s == pytest.approx(4.98, abs=0.05) and s > CFG["sigma_dv_max"]


def test_21_sigma_dv_winter_day():
    """dv_max uncertainty of a whole winter day of scans."""
    s = ms.sigma_dv_max(np.ones(24), np.zeros(24), np.full(24, 20.0), CFG)
    assert s == pytest.approx(0.37, abs=0.01) and s <= CFG["sigma_dv_max"]


def test_22_substitute_scenarios():
    """Substitute f and dv_max for both scenarios, and the provisional message."""
    faces = pd.DataFrame({"excluded": [False, False]}, index=pd.Index([1, 2], name="face"))
    empty = pd.DataFrame(columns=["scan_id", "face", "t_surf_c", "t_surf_sd_c", "t_air_c"])
    per = pd.DataFrame({"theta_g": [], "ground_lag": [], "theta_e": [], "p_e": [], "theta_e24": [],
                        "t_air_c": [], "rh_pct": []})
    fres = ms.extract_f(empty, per, faces, CFG, [])
    dv = ms.extract_dv_max(per, CFG, [])
    assert fres["f_central"].tolist() == [0.75, 0.75] and fres["f_worst"].tolist() == [0.60, 0.60]
    assert (dv["central"], dv["worst"]) == (4.0, 6.0)
    assert ms.provisional_message(fres, dv) == "잠정 결과: 대체값 사용(면 전체, Δv_max)"


# ---------------------------------------------------------------- 23-26 weather periods
def test_23_default_period():
    """The default period is the last complete year."""
    (p,) = ms.resolve_periods("year", None, pd.Timestamp("2026-10-21"))
    assert (p["label"], p["start"], p["end"]) == ("2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"))


def test_24_unfinished_year():
    """A year that has not ended yet is refused."""
    with pytest.raises(SystemExit, match="사용 가능한 마지막 연도: 2025"):
        ms.resolve_periods("year", [2026], pd.Timestamp("2026-10-21"))


def test_25_last365():
    """last365 covers exactly 365 days, also across a leap day."""
    (p,) = ms.resolve_periods("last365", None, pd.Timestamp("2026-10-21"))
    assert p["start"] == pd.Timestamp("2025-10-21") and p["end"] == pd.Timestamp("2026-10-20")
    assert len(pd.date_range(p["start"], p["end"], freq="D")) == 365
    (leap,) = ms.resolve_periods("last365", None, pd.Timestamp("2028-03-15"))   # includes 2028-02-29
    assert len(pd.date_range(leap["start"], leap["end"], freq="D")) == 365


def test_26_coverage():
    """Too little data coverage stops the run and names the month."""
    idx = pd.date_range("2025-01-01", periods=8760, freq="h")
    ok = pd.Series(True, index=idx)
    ok.iloc[100:276] = False                                                     # 2 % missing
    with pytest.raises(SystemExit, match="2025-01"):
        ms.check_coverage(ok, ms.HOURLY_MIN, "2025", "외기 시간 자료")


# ---------------------------------------------------------------- full run (synthetic ASOS cache)
STATION = 108


def ground(days):
    """Synthetic daily ground temperature for the given days."""
    doy = days.dayofyear.to_numpy()
    return 14 + 7 * np.sin(2 * np.pi * (doy - 120) / 365)


def write_cache(cache, start, end, suffix="", last_day=None, ground_value=None):
    """Synthetic hourly and daily data for the months in [start, end]. pv is left empty so it is computed from ta and hm."""
    for m in pd.period_range(start, end, freq="M"):
        stop = min(m.end_time.normalize(), last_day) if last_day is not None else m.end_time.normalize()
        hrs = pd.date_range(m.start_time, stop + pd.Timedelta(hours=23), freq="h")
        doy = hrs.dayofyear.to_numpy()
        ta = 12 - 14 * np.cos(2 * np.pi * (doy - 15) / 365) + 4 * np.sin(2 * np.pi * (hrs.hour - 9) / 24)
        hm = 65 + 10 * np.cos(2 * np.pi * hrs.hour / 24)
        h = pd.DataFrame({"tm": hrs.strftime("%Y-%m-%d %H:%M"), "ta": ta.round(1),
                          "hm": hm.round(0), "pv": ""})
        h.loc[100:103, "ta"] = np.nan                 # gap of 6 hours or less -> interpolated
        name = m.strftime("%Y%m") + suffix
        h.to_csv(cache / f"{STATION}_hourly_{name}.csv", index=False)
        days = pd.date_range(m.start_time, stop, freq="D")
        d = pd.DataFrame({"tm": days.strftime("%Y-%m-%d")})
        for c in ms.GROUND_COLS.values():
            d[c] = ground_value if ground_value is not None else ground(days).round(1)
        d.to_csv(cache / f"{STATION}_daily_{name}.csv", index=False)


def write_session(sdir, rows, site=None):
    """Writes scans.csv and session.yaml for a test session."""
    pd.DataFrame(rows, columns=["scan_id", "timestamp", "face", "zone", "az_deg", "el_deg", "t_surf_c",
                                "t_surf_sd_c", "t_air_c", "rh_pct", "t_sensor_c"]
                 ).to_csv(sdir / "scans.csv", index=False)
    session = {"session": sdir.name, "site": site or {"name": "test", "lat": 37.5, "lon": 127.0},
               "asos_station": STATION, "heading": {"deg": 90.0},
               "materials": {"default": "sensitive", "by_face": {}}, "excluded_faces": []}
    (sdir / "session.yaml").write_text(yaml.safe_dump(session, allow_unicode=True), encoding="utf-8")


def run_main(monkeypatch, sdir, *extra):
    """Runs mold_simulator.main() on a session folder and returns its output folder."""
    monkeypatch.setattr(sys, "argv", ["mold_simulator.py", "--session-dir", str(sdir), *extra])
    ms.main()
    return sdir / "sim"


def test_12_full_run(tmp_path, monkeypatch):
    """Two days of measurements in January (f 0.57-0.95, judged 'measured' since it is winter) -> full run over 2024 (a leap year)."""
    sdir = tmp_path / "data" / "winter"
    sdir.mkdir(parents=True)
    cache = tmp_path / "data" / "kma_cache"
    cache.mkdir()
    write_cache(cache, pd.Timestamp(2023, 12, 1), pd.Timestamp(2025, 1, 31))
    times = pd.date_range("2025-01-10 00:30", "2025-01-11 23:30", freq="h")
    rows = []
    for s, (t, g) in enumerate(zip(times, ground(times.normalize())), start=1):
        for k in range(1, 21):
            f = 0.55 + 0.02 * k
            rows.append([s, t.strftime("%Y-%m-%dT%H:%M:%S+09:00"), k, "M", 36.0 * ((k - 1) // 2),
                         10.81, round(g + f * (20 - g), 2), 0.05, 20.0, 55.0, 21.0])
    write_session(sdir, rows)
    out = run_main(monkeypatch, sdir, "--years", "2024")

    hourly = pd.read_csv(out / "hourly_2024.csv")
    assert hourly.groupby(["scenario", "face"]).size().eq(8784).all() and hourly["face"].nunique() == 20
    assert set(hourly["scenario"]) == {"central", "worst"}
    assert not hourly.isna().any().any()
    assert hourly["M"].between(0, 6).all()

    params = json.loads((out / "params.json").read_text(encoding="utf-8"))
    f = {p["face"]: p for p in params["faces"]}
    assert f[1]["judgment"] == "measured" and f[1]["f_central"] == pytest.approx(0.57, abs=0.02)
    assert f[20]["f_central"] == pytest.approx(0.95, abs=0.02) and f[20]["f_worst"] < f[20]["f_central"]
    assert params["dv_max"]["judgment"] == "measured" and params["dv_max"]["central"] in (0, 2, 4, 6, 8, 10)
    assert params["provisional"] is False and params["scans"][0]["out_source"] == "asos"

    view = json.loads((out / "viewer_2024.json").read_text(encoding="utf-8"))
    n = view["time"]["n"]
    assert n == 8784 and len(view["faces"]) == 20 and view["provisional"] is False
    assert all(len(view["series"][k]) == n for k in ("theta_e", "theta_g", "theta_i", "mode", "data_flag"))
    assert all(len(view["series"]["rh_i"][s]) == n for s in ms.SCENARIOS)
    assert all(len(fc[s]["M"]) == n and len(fc[s]["rh_si"]) == n for fc in view["faces"] for s in ms.SCENARIOS)
    summary = pd.read_csv(out / "summary_2024.csv", comment="#")
    assert {"M_max_year_central", "M_max_year_worst", "f_judgment"} <= set(summary.columns)
    assert (out / "viewer_index.json").exists()


def test_27_reproduce_first_measurement(tmp_path, monkeypatch, capsys):
    """One scan at 2026-10-21 20:56 -> runs without stopping. f is measured (high uncertainty);
    dv_max cannot be computed without 24 hours of outdoor data -> substitute, and the result is marked provisional (Δv_max)."""
    monkeypatch.setattr(ms, "today", lambda: pd.Timestamp("2026-10-21"))
    sdir = tmp_path / "data" / "20261021_2054"
    sdir.mkdir(parents=True)
    cache = tmp_path / "data" / "kma_cache"
    cache.mkdir()
    write_cache(cache, pd.Timestamp(2024, 12, 1), pd.Timestamp(2026, 1, 31))
    write_cache(cache, pd.Timestamp(2026, 10, 1), pd.Timestamp(2026, 10, 1), suffix="_partial",
                last_day=pd.Timestamp("2026-10-20"), ground_value=21.5)   # up to yesterday, 2 °C below indoor

    ts = "2026-10-21T20:56:16+09:00"
    rows = [[1, ts, k, "M", 36.0 * ((k - 1) // 2), 10.81, round(23.0 + 0.05 * k, 2), 0.04, 23.25, 33.2, 24.3]
            for k in range(1, 21)]
    write_session(sdir, rows, site={"name": "망원동 반지하", "lat": 37.556, "lon": 126.906})
    kma.append_outdoor(sdir / "outdoor.csv", [{"hour": datetime(2026, 10, 21, 20), "source": "ncst",
                                               "t_out_c": 15.0, "rh_out_pct": 60.0, "p_out_hpa": 10.23,
                                               "nx": 59, "ny": 127, "fetched_at": ts}])
    out = run_main(monkeypatch, sdir)

    params = json.loads((out / "params.json").read_text(encoding="utf-8"))
    assert params["periods"][0]["label"] == "2025"
    assert params["provisional"] is True
    faces = params["faces"]
    assert {p["judgment"] for p in faces} <= {"uncertain", "anomaly"}
    assert len({p["f_central"] for p in faces}) > 1                       # each face has its own measured f
    assert params["dv_max"]["judgment"] == "substitute"
    scan = params["scans"][0]
    assert scan["out_source"] == "ncst" and scan["ground_lag"] == 1 and scan["theta_e24"] is None
    assert (out / "summary_2025.csv").read_text(encoding="utf-8").startswith("# 잠정 결과: 대체값 사용(Δv_max)")
    assert "잠정 결과" in capsys.readouterr().out
