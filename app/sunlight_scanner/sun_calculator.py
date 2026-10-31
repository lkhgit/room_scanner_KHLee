"""
sun_calculator.py — calculates the direct sunlight entering a 3D room model with window openings (room.glb)
                    and saves the result as an HTML page for the browser (sun_viewer.html)
    (takes the output folder of "panorama to model_oneformer.py" as its input)

    (virtual env) .venv\\Scripts\\python.exe sun_calculator.py out\\panorama_test_2 --heading 180
    (virtual env) .venv\\Scripts\\python.exe sun_calculator.py out\\panorama_test_2 --heading 180 --date 2026-10-18 --time 14:30

The algorithm follows sunform_engine.py of SunForm (github.com/JakeWhiteArchitecture/sunform, MIT License):
    sun position (Spencer 1971) -> split surfaces down to a maximum edge length -> from each vertex offset 1 cm
    along its normal, cast a ray towards the sun (Möller–Trumbore) -> if nothing blocks it, it is sunlit at that
    time -> accumulate over time + Laplacian smoothing
    Changes
    - For indoor analysis the normal is not "always up" as in SunForm but follows the surface winding (into the room).
      Light from behind a surface (normal · sun ≤ 0) casts no ray and counts as shade.
    - Times are local (--tz, default +9 = Korea) instead of SunForm's UTC.
    - Ray intersection is vectorised with numpy (all vertices at one time step share the sun direction).
    - Finds which window each sunlit ray came through, giving the inflow per window and how deep it reaches the floor.

Steps
    1. Sun position: altitude/azimuth every --step minutes on the --date day(s), or else on the solstices and
       equinoxes of --year; only times with the sun up are kept
       + clear-sky direct normal irradiance (DNI, W/m²) (Meinel model, air mass by Kasten–Young)
    2. Read room.glb: surfaces are told apart by material name (wall_XX / floor / ceiling). Every surface casts shadows.
    3. Analysis grid: floor + walls (and ceiling with --ceiling) split until no edge is longer than --grid (m)
    4. Ray test at every time step -> sunlit vertices (bits), irradiance = DNI × cos(angle of incidence)
    5. Intersect sunlit rays with the window rectangles (windows in layout.json) -> inflow W per window, depth on the floor
    6. Save: sun_viewer.html (one file holding the model and the results), sun_hours.json (summary figures)

Assumptions
    Clear-sky direct sunlight only (no diffuse sky light or reflections). No outside obstacles such as buildings or trees.
    Windows are open holes without glass (no glass transmittance applied).

Coordinates
    room.glb is y-up: +X = right of the panorama's centre column, +Y = up, −Z = the direction the centre column (u=0) faces.
    --heading = the real bearing the panorama's centre column faces (north 0°, east 90°, clockwise).
    The sun direction in model coordinates is the sun azimuth minus heading (same convention as SunForm's X=east, Y=up, −Z=north).
    Without --heading the "heading_deg" in layout.json is used, and without that 0° (centre column = north) is assumed.
"""

import argparse
import base64
import json
import math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix

SEASONS = [("winter", "동지", 12, 21), ("equinox", "춘·추분", 3, 21), ("summer", "하지", 6, 21)]
DIR8 = ["북", "북동", "동", "남동", "남", "남서", "서", "북서"]
SOLAR_CONST = 1361.0        # solar constant outside the atmosphere (W/m²)
RAY_OFFSET = 0.01           # distance the ray start is lifted off the surface (m), same as SunForm
MIN_T = 1e-4                # intersections closer than this are ignored (avoids self-intersection), same as SunForm


def to_yup(p):
    """z-up (x, y, z) → y-up (x, z, -y). Same as to_yup in panorama to model_oneformer.py."""
    p = np.asarray(p, float)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1)


# ================================================================== 1. sun
def get_sun_positions(lat, lon, day, step_min, tz):
    """
    Port of SunForm's get_sun_positions (Spencer 1971 Fourier series).
    hour is local time: UTC = hour - tz. Returns arrays (hour, azimuth°, altitude°) for the times the sun is up.
    Azimuth is north 0°, clockwise (east 90°, south 180°).
    """
    doy = day.timetuple().tm_yday
    B = 2 * math.pi * (doy - 1) / 365
    decl = (0.006918 - 0.399912 * math.cos(B) + 0.070257 * math.sin(B)
            - 0.006758 * math.cos(2 * B) + 0.000907 * math.sin(2 * B)
            - 0.002697 * math.cos(3 * B) + 0.00148 * math.sin(3 * B))           # declination (rad)
    eot = 229.18 * (0.000075 + 0.001868 * math.cos(B) - 0.032077 * math.sin(B)
                    - 0.014615 * math.cos(2 * B) - 0.040849 * math.sin(2 * B))  # equation of time (minutes)

    hour = np.arange(0, 24, step_min / 60)
    solar_time = hour - tz + (eot + 4 * lon) / 60
    ha = np.radians((solar_time - 12) * 15)                                     # hour angle
    la = math.radians(lat)
    sin_alt = math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * np.cos(ha)
    alt = np.arcsin(np.clip(sin_alt, -1, 1))
    cos_az = (math.sin(decl) - np.sin(alt) * math.sin(la)) / np.maximum(np.cos(alt) * math.cos(la), 1e-12)
    az = np.degrees(np.arccos(np.clip(cos_az, -1, 1)))
    az = np.where(ha > 0, 360 - az, az)                                         # west in the afternoon
    up = alt > 0
    return hour[up], az[up], np.degrees(alt[up])


def sun_direction(az, alt, heading):
    """Azimuth/altitude → unit vector towards the sun in model coordinates (y-up), rotated by heading to fit the model."""
    a, h = np.radians(np.asarray(az) - heading), np.radians(alt)
    return np.stack([np.sin(a) * np.cos(h), np.sin(h), -np.cos(a) * np.cos(h)], -1)


def clear_sky_dni(alt):
    """Clear-sky direct normal irradiance (W/m²). Meinel: DNI = 1361·0.7^(AM^0.678), air mass AM from Kasten–Young."""
    z = 90 - np.asarray(alt, float)
    am = 1 / (np.cos(np.radians(z)) + 0.50572 * (96.07995 - z) ** -1.6364)
    return SOLAR_CONST * 0.7 ** (am ** 0.678)


def bearing_of(v, heading):
    """Horizontal direction v in model coordinates → real bearing (north 0°, clockwise)."""
    return (heading + math.degrees(math.atan2(v[0], -v[2]))) % 360


def dir8(b):
    """Returns the 8-point compass name for a bearing."""
    return DIR8[int((b + 22.5) // 45) % 8]


# ================================================================== 2-3. mesh
def load_surfaces(glb_path):
    """room.glb → {material name: triangle array (N,3,3)}. Node transforms are applied and winding is kept."""
    scene = trimesh.load(glb_path, force="scene", process=False)
    out = {}
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        g = scene.geometry[gname]
        mat = getattr(g.visual, "material", None)
        name = getattr(mat, "name", None) or gname
        out.setdefault(name, []).append(trimesh.transform_points(g.vertices, T)[g.faces])
    return {k: np.concatenate(v) for k, v in out.items()}


def subdivide_to_max_edge(tris, max_edge):
    """
    Port of SunForm's subdivide_to_max_edge: a triangle whose longest edge exceeds max_edge is split in two at that edge's midpoint.
    The vertices are rotated so the longest edge is (r0, r1) before splitting, which keeps the winding (normal direction).
    """
    done, cur = [], np.asarray(tris, float)
    while len(cur):
        L = np.stack([np.linalg.norm(cur[:, 1] - cur[:, 0], axis=1),
                      np.linalg.norm(cur[:, 2] - cur[:, 1], axis=1),
                      np.linalg.norm(cur[:, 0] - cur[:, 2], axis=1)], 1)
        big = L.max(1) > max_edge
        done.append(cur[~big])
        cur, k = cur[big], L[big].argmax(1)
        idx = (np.arange(3)[None] + k[:, None]) % 3
        r = np.take_along_axis(cur, idx[:, :, None], 1)
        m = (r[:, 0] + r[:, 1]) / 2
        cur = np.concatenate([np.stack([r[:, 0], m, r[:, 2]], 1), np.stack([m, r[:, 1], r[:, 2]], 1)])
    return np.concatenate(done)


def build_unique_vertices(tris, quant=1e-4):
    """Port of SunForm's build_unique_vertices: rounds to a 0.1 mm grid and merges identical points."""
    flat = tris.reshape(-1, 3)
    key = np.round(flat / quant).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return flat[first], inv.reshape(-1).reshape(-1, 3)


def build_analysis_mesh(surfaces, names, grid):
    """
    Splits the surfaces to analyse and joins them into one vertex array. Each surface keeps its own vertices
    (so normals do not mix at corners).
    Normal = surface winding (into the room). Vertex area = 1/3 of each attached triangle (SunForm compute_vertex_voronoi_areas).
    Ray start = vertex + normal×RAY_OFFSET + (along the surface towards its centre)×RAY_OFFSET.
    Why also push along the surface: a vertex on the floor-wall edge lifted only along its normal starts on the wall
    plane, so its intersection with the wall is at distance 0 (below MIN_T) and is not counted as blocked → light
    would wrongly reach the foot of walls without windows.
    """
    pos, idx, nrm, area, sid, org = [], [], [], [], [], []
    off = 0
    for i, name in enumerate(names):
        fine = subdivide_to_max_edge(surfaces[name], grid)
        v, f = build_unique_vertices(fine)
        cr = np.cross(fine[:, 1] - fine[:, 0], fine[:, 2] - fine[:, 0])
        n = cr.sum(0)
        n /= np.linalg.norm(n)
        a = np.zeros(len(v))
        for k in range(3):
            np.add.at(a, f[:, k], np.linalg.norm(cr, axis=1) / 6)
        inward = (v * a[:, None]).sum(0) / a.sum() - v              # vertex → surface centre
        inward -= np.outer(inward @ n, n)                            # in-plane part only
        inward /= np.maximum(np.linalg.norm(inward, axis=1, keepdims=True), 1e-9)
        pos.append(v); idx.append(f + off); nrm.append(np.repeat(n[None], len(v), 0))
        area.append(a); sid.append(np.full(len(v), i))
        org.append(v + (n[None] + inward) * RAY_OFFSET)
        off += len(v)
    return (np.concatenate(pos), np.concatenate(idx), np.concatenate(nrm),
            np.concatenate(area), np.concatenate(sid), np.concatenate(org))


def smooth_vertex_field(values, idx, iterations=2, lam=0.5):
    """Port of SunForm's smooth_vertex_field: blends with the neighbour mean to soften jagged shadow edges. Flat values stay as they are."""
    n = len(values)
    i, j = idx.ravel(), idx[:, [1, 2, 0]].ravel()
    A = coo_matrix((np.ones(2 * len(i)), (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
    A.data[:] = 1
    deg = np.asarray(A.sum(1)).ravel()
    v = values.astype(float).copy()
    for _ in range(iterations):
        avg = np.where(deg > 0, A @ v / np.maximum(deg, 1), v)
        v = (1 - lam) * v + lam * avg
    return v


# ================================================================== 4. rays
def first_hit(O, d, tris, min_t=MIN_T, chunk=4096):
    """
    Vectorised Möller–Trumbore (SunForm ray_triangle_intersect), using the fact that all rays share direction d.
    Returns for each ray the nearest intersection distance t (inf if none) and the triangle hit (-1 if none).
    """
    v0 = tris[:, 0]
    e1, e2 = tris[:, 1] - v0, tris[:, 2] - v0
    p = np.cross(d, e2)
    det = (e1 * p).sum(1)
    ok = np.abs(det) > 1e-10                              # skip triangles parallel to the ray
    ids, v0, e1, e2, p, inv = np.nonzero(ok)[0], v0[ok], e1[ok], e2[ok], p[ok], 1 / det[ok]
    t_best = np.full(len(O), np.inf)
    id_best = np.full(len(O), -1)
    if not len(ids):
        return t_best, id_best
    for s in range(0, len(O), chunk):
        tv = O[s:s + chunk, None, :] - v0[None]
        u = (tv * p).sum(2) * inv
        q = np.cross(tv, e1)
        v = (q @ d) * inv
        t = (q * e2).sum(2) * inv
        t = np.where((u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > min_t), t, np.inf)
        j = t.argmin(1)
        tb = t[np.arange(len(j)), j]
        t_best[s:s + chunk] = tb
        id_best[s:s + chunk] = np.where(np.isfinite(tb), ids[j], -1)
    return t_best, id_best


def analyze_day(day, a, mesh, occluders, windows):
    """Computes the sunlit vertices and statistics for every time step of one day."""
    pos, idx, nrm, area, sid, org, floor_id = mesh
    hours, az, alt = get_sun_positions(a.lat, a.lon, day, a.step, a.tz)
    dirs = sun_direction(az, alt, a.heading)
    dni = clear_sky_dni(alt)
    S, V, W = len(hours), len(pos), len(windows)
    dt = a.step / 60
    is_floor = sid == floor_id

    win_tris = np.array([[w["corners"][0], w["corners"][1], w["corners"][2]] for w in windows]
                        + [[w["corners"][0], w["corners"][2], w["corners"][3]] for w in windows]).reshape(-1, 3, 3)
    win_of_tri = np.r_[np.arange(W), np.arange(W)]

    lit = np.zeros((S, V), bool)
    E_sum = np.zeros(V)                                   # daily irradiation Wh/m²
    st = {"total_W": [], "flux_W": [], "floor_m2": [], "wall_m2": [],
          "win_W": [[] for _ in windows], "win_flux_W": [[] for _ in windows], "win_depth": [[] for _ in windows]}
    for s in range(S):
        d = dirs[s]
        cosi = nrm @ d
        cand = np.nonzero(cosi > 0)[0]                    # only light from in front of the surface
        O = org[cand]
        t, _ = first_hit(O, d, occluders)
        free = ~np.isfinite(t)
        on, O_on = cand[free], O[free]
        lit[s, on] = True
        e = dni[s] * cosi[on]                             # W/m²
        E_sum[on] += e * dt
        p_on = e * area[on]                               # W received by each vertex

        st["total_W"].append(float(p_on.sum()))
        st["floor_m2"].append(float(area[on][is_floor[on]].sum()))
        st["wall_m2"].append(float(area[on][~is_floor[on]].sum()))

        wid = np.full(len(on), -1)
        if W and len(on):                                 # the window each sunlit ray came through
            _, tri = first_hit(O_on, d, win_tris, min_t=0)
            wid = np.where(tri >= 0, win_of_tri[np.maximum(tri, 0)], -1)
        flux = 0.0
        for k, w in enumerate(windows):
            f = float(dni[s] * max(0.0, float(np.dot(w["n_out"], d))) * w["area_m2"])
            flux += f
            sel = wid == k
            st["win_W"][k].append(float(p_on[sel].sum()))
            st["win_flux_W"][k].append(f)
            fl = sel & is_floor[on]
            depth = (pos[on][fl] - w["corners"][0]) @ (-w["n_out"]) if fl.any() else np.zeros(0)
            st["win_depth"][k].append(float(depth.max()) if len(depth) else 0.0)
        st["flux_W"].append(flux)

    hours_v = smooth_vertex_field(lit.sum(0) * dt, idx)
    return dict(hours=hours, az=az, alt=alt, dirs=dirs, dni=dni, lit=lit,
                hours_v=hours_v, E_sum=E_sum, stats=st, dt=dt)


# ================================================================== 6. save
def b64(arr, dtype):
    """Encodes an array as base64 with the given dtype."""
    return base64.b64encode(np.ascontiguousarray(arr, dtype=dtype).tobytes()).decode()


def hhmm(h):
    """Formats hours as HH:MM."""
    m = int(round(h * 60))
    return f"{m // 60:02d}:{m % 60:02d}"


def main():
    """Parses the arguments, loads the room, calculates sunlight for each day and writes sun_hours.json and sun_viewer.html."""
    ap = argparse.ArgumentParser(description="방 3D 모델(room.glb) 실내 직사광 계산 → sun_viewer.html")
    ap.add_argument("room", help="panorama to model_oneformer.py 출력 폴더 (또는 그 안의 room.glb)")
    ap.add_argument("--heading", type=float, default=None,
                    help="파노라마 가운데 열이 바라보는 방위각(°, 북 0 · 동 90 · 남 180 · 서 270)")
    ap.add_argument("--lat", type=float, default=37.5665, help="위도 (기본: 서울)")
    ap.add_argument("--lon", type=float, default=126.9780, help="경도 (기본: 서울)")
    ap.add_argument("--tz", type=float, default=9, help="표준시 UTC 오프셋 (시간, 한국 = 9)")
    ap.add_argument("--year", type=int, default=date.today().year, help="--date가 없을 때 절기 계산에 쓸 연도")
    ap.add_argument("--date", action="append", default=[],
                    help="계산할 날짜 YYYY-MM-DD (여러 번 쓸 수 있음). 주면 절기(동지/춘추분/하지)는 계산하지 않음")
    ap.add_argument("--time", default="12:00", help="HTML을 열었을 때 처음 보여줄 시각 HH:MM")
    ap.add_argument("--step", type=float, default=5, help="시간 간격 (분)")
    ap.add_argument("--grid", type=float, default=0.1, help="분석 격자 최대 변 길이 (m). 작을수록 정밀·느림")
    ap.add_argument("--ceiling", action="store_true", help="천장도 분석 면에 포함")
    ap.add_argument("--out", default=None, help="출력 폴더 (기본: room.glb가 있는 폴더)")
    a = ap.parse_args()

    room = Path(a.room)
    glb_path = room / "room.glb" if room.is_dir() else room
    room_dir = glb_path.parent
    out_dir = Path(a.out) if a.out else room_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    layout_path = room_dir / "layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8")) if layout_path.exists() else {}

    if a.heading is None:
        if "heading_deg" in layout:
            a.heading = float(layout["heading_deg"])
            print(f"방위: layout.json heading_deg = {a.heading:.1f}°")
        else:
            a.heading = 0.0
            print("경고: --heading 이 없고 layout.json 에도 heading_deg 가 없어 0°(파노라마 가운데 = 북쪽)로 가정합니다")

    # 2. mesh
    surfaces = load_surfaces(glb_path)
    occluders = np.concatenate(list(surfaces.values()))
    wall_names = sorted(n for n in surfaces if n.startswith("wall_"))
    names = wall_names + ["floor"] + (["ceiling"] if a.ceiling and "ceiling" in surfaces else [])
    if "floor" not in surfaces:
        raise SystemExit(f"{glb_path} 에 floor 재질이 없습니다")
    pos, idx, nrm, area, sid, org = build_analysis_mesh(surfaces, names, a.grid)
    floor_id = names.index("floor")
    if nrm[sid == floor_id][0, 1] < 0:                    # if the model's winding is reversed, flip everything
        print("경고: 바닥 법선이 아래를 향해 모든 면의 법선을 뒤집습니다")
        org -= 2 * nrm * RAY_OFFSET
        nrm, idx = -nrm, idx[:, ::-1].copy()
    mesh = (pos, idx, nrm, area, sid, org, floor_id)
    print(f"[1/4] {glb_path}: 면 {len(names)}개, 분석 꼭짓점 {len(pos):,}개 (격자 {a.grid * 100:.0f} cm), "
          f"차폐 삼각형 {len(occluders)}개")

    surf_info = []
    for i, n in enumerate(names):
        sel = sid == i
        out_n = -nrm[sel][0]
        if n.startswith("wall_"):
            b = bearing_of(out_n, a.heading)
            label = f"벽 {int(n[5:])} ({dir8(b)}쪽)"
        else:
            b, label = None, {"floor": "바닥", "ceiling": "천장"}[n]
        surf_info.append({"name": n, "label": label, "bearing": b, "area_m2": float(area[sel].sum())})

    walls_by_idx = {w["index"]: w for w in layout.get("walls", [])}
    windows = []
    for k, w in enumerate(layout.get("windows", [])):
        wall = walls_by_idx.get(w["wall"])
        if wall is None:
            continue
        n_out = -to_yup(wall["normal"])
        b = bearing_of(n_out, a.heading)
        windows.append({"id": k, "wall": w["wall"], "label": f"창 {k + 1}", "facing": dir8(b), "bearing": b,
                        "corners": to_yup(w["corners_3d"]), "n_out": n_out,
                        "area_m2": float(w["width_m"] * w["height_m"]),
                        "width_m": w["width_m"], "height_m": w["height_m"], "sill_m": w["sill_height_m"]})
    print(f"      창문 {len(windows)}개: " + ", ".join(
        f"{w['label']} {w['facing']}향 {w['width_m']:.2f}x{w['height_m']:.2f} m" for w in windows))
    if not windows:
        print("경고: layout.json 에 창문이 없습니다 (직사광이 들어오지 않음)")

    # 1, 4, 5. calculation for each day
    days = [(f"date{i}", "지정일", datetime.strptime(d, "%Y-%m-%d").date()) for i, d in enumerate(a.date)]
    if not days:                                          # the three seasonal days only when --date is not given
        days = [(k, lab, date(a.year, m, dd)) for k, lab, m, dd in SEASONS]
    bytes_per_step = (len(pos) + 7) // 8
    day_data, summary = [], []
    print(f"[2/4] 직사광 계산: 위도 {a.lat}, 경도 {a.lon}, UTC{a.tz:+g}, 방위 {a.heading:.1f}°, {a.step:g}분 간격")
    for key, label, d in days:
        r = analyze_day(d, a, mesh, occluders, windows)
        st, dt = r["stats"], r["dt"]
        total_kwh, flux_kwh = sum(st["total_W"]) * dt / 1000, sum(st["flux_W"]) * dt / 1000
        surf_sum = []
        for i, s in enumerate(surf_info):
            sel = sid == i
            hv = r["lit"][:, sel].sum(0) * dt
            surf_sum.append({**s, "mean_h": float((hv * area[sel]).sum() / area[sel].sum()),
                             "max_h": float(hv.max()), "kwh": float((r["E_sum"][sel] * area[sel]).sum() / 1000)})
        win_sum = [{"id": w["id"], "sun_h": float(sum(f > 0 for f in st["win_flux_W"][k]) * dt),
                    "kwh_in": float(sum(st["win_W"][k]) * dt / 1000),
                    "kwh_flux": float(sum(st["win_flux_W"][k]) * dt / 1000),
                    "max_depth_m": float(max(st["win_depth"][k], default=0))} for k, w in enumerate(windows)]
        S = len(r["hours"])
        print(f"      {label} ({d}): 해 {hhmm(r['hours'][0]) if S else '-'}~{hhmm(r['hours'][-1]) if S else '-'}, "
              f"실내 직사광 {total_kwh:.2f} kWh (창문 통과량 {flux_kwh:.2f} kWh"
              + (f", 격자 오차 {100 * (total_kwh / flux_kwh - 1):+.1f}%)" if flux_kwh > 0 else ")"))
        for w, ws in zip(windows, win_sum):
            print(f"        {w['label']}({w['facing']}향): 직사 {ws['sun_h']:.1f} h, 유입 {ws['kwh_in']:.2f} kWh, "
                  f"바닥 최대 깊이 {ws['max_depth_m']:.2f} m")
        summary.append({"key": key, "label": label, "date": str(d), "total_kwh": total_kwh, "flux_kwh": flux_kwh,
                        "surfaces": surf_sum, "windows": win_sum})
        day_data.append({
            "key": key, "label": label, "date": str(d),
            "steps": [{"hour": round(float(h), 4), "az": round(float(z), 2), "alt": round(float(t), 2),
                       "dir": [round(float(c), 5) for c in v], "dni": round(float(n), 1)}
                      for h, z, t, v, n in zip(r["hours"], r["az"], r["alt"], r["dirs"], r["dni"])],
            "lit": base64.b64encode(np.packbits(r["lit"], axis=1, bitorder="little").tobytes()).decode()
                   if S else "",
            "hours": b64(r["hours_v"], "<f4"),
            "stats": {k: ([[round(x, 2) for x in row] for row in v] if k.startswith("win_")
                          else [round(x, 2) for x in v]) for k, v in st.items()},
            "summary": summary[-1],
        })

    # 6. save
    print("[3/4] sun_hours.json 저장")
    meta = {"source": str(glb_path), "heading_deg": a.heading, "lat": a.lat, "lon": a.lon, "tz": a.tz,
            "step_min": a.step, "grid_m": a.grid,
            "assumptions": "맑은 날 직사광만, 바깥 장애물 없음, 창문 = 뚫린 구멍(유리 투과율 미적용)"}
    win_json = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in w.items()} for w in windows]
    (out_dir / "sun_hours.json").write_text(json.dumps(
        {**meta, "windows": win_json, "days": summary}, indent=2, ensure_ascii=False), encoding="utf-8")

    hh, mm = (int(x) for x in a.time.split(":"))
    data = {**meta, "init_hour": hh + mm / 60, "bytes_per_step": bytes_per_step,
            "info": f"{layout.get('source', glb_path.parent.name)} · 위도 {a.lat:.3f}, 경도 {a.lon:.3f} · "
                    f"파노라마 가운데 방위 {a.heading:.0f}° ({dir8(a.heading)})",
            "mesh": {"pos": b64(pos, "<f4"), "nrm": b64(nrm, "<f4"), "idx": b64(idx, "<u4"),
                     "area": b64(area, "<f4"), "surf": b64(sid, "u1")},
            "surfaces": surf_info, "windows": win_json, "days": day_data}
    html = (VIEWER_HTML
            .replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
            .replace("__GLB_BASE64__", base64.b64encode(glb_path.read_bytes()).decode()))
    (out_dir / "sun_viewer.html").write_text(html, encoding="utf-8")
    print(f"[4/4] 저장 완료 → {(out_dir / 'sun_viewer.html').resolve()} (브라우저로 열기, "
          f"{len(html) / 1e6:.1f} MB)")


VIEWER_HTML = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Room Sunlight</title>
<style>
  :root { --bg:#1e1f24; --panel:#25272e; --line:#383b45; --text:#e9e9ec; --dim:#9c9ea8; --sun:#ffc94d; --blue:#4a6cf7; }
  html, body { margin:0; height:100%; background:var(--bg); color:var(--text);
               font:13px/1.45 system-ui, "Malgun Gothic", sans-serif; overflow:hidden; }
  #view { position:fixed; top:0; right:0; bottom:0; left:360px; }
  #panel { position:fixed; left:0; top:0; bottom:0; width:360px; box-sizing:border-box; padding:16px 16px 24px;
           overflow-y:auto; background:var(--panel); border-right:1px solid var(--line); }
  h1 { font-size:16px; margin:0 0 2px; }
  .sub { color:var(--dim); font-size:12px; }
  .label { color:var(--dim); font-size:11px; margin:14px 0 4px; }
  .seg { display:flex; gap:4px; flex-wrap:wrap; }
  button { padding:5px 10px; border:1px solid var(--line); background:#2f323b; color:var(--text);
           border-radius:6px; cursor:pointer; font:inherit; }
  button.on { background:var(--blue); border-color:var(--blue); }
  .clock { display:flex; align-items:baseline; gap:10px; margin-top:10px; }
  .clock b { font-size:30px; font-variant-numeric:tabular-nums; letter-spacing:.5px; }
  .clock span { color:var(--dim); font-size:12px; }
  .clock button { margin-left:auto; white-space:nowrap; }
  input[type=range] { width:100%; accent-color:var(--sun); margin:6px 0 4px; }
  #chart { width:100%; height:120px; display:block; background:#1d1f25; border-radius:6px; cursor:pointer; }
  .chartkey { display:flex; gap:12px; color:var(--dim); font-size:11px; margin-top:4px; }
  .chartkey i { display:inline-block; width:10px; height:3px; vertical-align:middle; margin-right:4px; }
  dl { display:grid; grid-template-columns:auto 1fr; gap:4px 12px; margin:12px 0 0; }
  dt { color:var(--dim); } dd { margin:0; text-align:right; font-variant-numeric:tabular-nums; }
  table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; margin-top:4px; }
  th, td { padding:4px 3px; border-bottom:1px solid var(--line); text-align:right; }
  th:first-child, td:first-child { text-align:left; }
  th { color:var(--dim); font-weight:500; font-size:11px; }
  .bar { height:10px; border-radius:3px; margin-top:4px; }
  .ticks { display:flex; justify-content:space-between; color:var(--dim); font-size:11px; }
  .note { color:var(--dim); font-size:11px; margin-top:16px; }
  #views { position:fixed; top:12px; right:12px; display:flex; gap:6px; }
  [hidden] { display:none !important; }
  @media (max-width:760px) {
    #panel { top:auto; right:0; width:auto; height:50vh; border-right:0; border-top:1px solid var(--line); }
    #view { left:0; bottom:50vh; }
  }
</style></head><body>
<aside id="panel">
  <h1>실내 일조 분석</h1>
  <div class="sub" id="info"></div>

  <div class="label">날짜</div>
  <div class="seg" id="days"></div>
  <div class="label">보기</div>
  <div class="seg">
    <button id="m0" class="on">지정 시각</button>
    <button id="m1">하루 일조시간</button>
  </div>

  <div class="clock"><b id="clock">--:--</b><span id="sunpos"></span><button id="play">▶ 재생</button></div>
  <input type="range" id="slider" min="0" max="0" value="0">
  <canvas id="chart"></canvas>
  <div class="chartkey"><span><i style="background:#ffc94d"></i>실내에 닿은 직사광 (W)</span>
    <span><i style="background:#6d8cff"></i>바닥에 닿은 면적 (m²)</span></div>

  <div id="timeSec">
    <dl id="nowStats"></dl>
    <div class="label">창문별</div>
    <table><thead><tr><th>창문</th><th>방향</th><th>유입 W</th><th>바닥 깊이</th></tr></thead>
      <tbody id="winNow"></tbody></table>
  </div>
  <div id="daySec" hidden>
    <dl id="dayStats"></dl>
    <div class="label">면별 일조시간</div>
    <table><thead><tr><th>면</th><th>평균 h</th><th>최대 h</th><th>kWh</th></tr></thead>
      <tbody id="surfDay"></tbody></table>
    <div class="label">창문별 하루</div>
    <table><thead><tr><th>창문</th><th>방향</th><th>직사 h</th><th>유입 kWh</th><th>최대 깊이</th></tr></thead>
      <tbody id="winDay"></tbody></table>
  </div>

  <div class="label" id="legendTitle">직사광 세기 (W/m²)</div>
  <div class="bar" id="legendBar"></div>
  <div class="ticks" id="legendTicks"></div>

  <div class="note">맑은 날 직사광만 계산했습니다 (하늘 산란광·반사광 제외). 창밖 건물·나무 같은 바깥 장애물은 없고,
    창문은 유리 없이 뚫린 구멍으로 가정했습니다. 노란 빛줄기는 창문 네 모서리에서 태양 반대 방향으로 쏜 광선입니다.
    ← → 키로 시각을 옮길 수 있습니다.</div>
</aside>
<div id="view"></div>
<div id="views">
  <button id="v_outside" class="on">바깥 위에서</button>
  <button id="v_top">위에서</button>
  <button id="v_inside">카메라 위치</button>
</div>

<script type="importmap">
{"imports": {"three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
             "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}
</script>
<script type="application/json" id="data">__DATA__</script>
<script type="text/plain" id="glb">__GLB_BASE64__</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const D = JSON.parse(document.getElementById('data').textContent);
function bytesOf(s) { const bin = atob(s); const u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; }
const dec = (s, T) => new T(bytesOf(s).buffer);
const M = { pos: dec(D.mesh.pos, Float32Array), nrm: dec(D.mesh.nrm, Float32Array),
            idx: dec(D.mesh.idx, Uint32Array), area: dec(D.mesh.area, Float32Array), surf: dec(D.mesh.surf, Uint8Array) };
const V = M.area.length;
for (const d of D.days) { d.bits = d.lit ? bytesOf(d.lit) : new Uint8Array(0); d.hoursArr = dec(d.hours, Float32Array); }
const DIR8 = ['북','북동','동','남동','남','남서','서','북서'];
const dir8 = b => DIR8[Math.floor(((b % 360) + 360 + 22.5) / 45) % 8];
const hhmm = h => { const m = Math.round(h * 60); return String(Math.floor(m / 60)).padStart(2, '0') + ':' + String(m % 60).padStart(2, '0'); };
const fmt = (x, n = 0) => x.toLocaleString('ko-KR', { minimumFractionDigits: n, maximumFractionDigits: n });
document.getElementById('info').textContent = D.info;

// ---------------------------------------------------------------- 장면
const host = document.getElementById('view');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(devicePixelRatio);
host.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1e1f24);
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 500);
const controls = new OrbitControls(camera, renderer.domElement);

// 분석 메쉬: 꼭짓점마다 lit(0/1)과 val(W/m² 또는 일조시간)을 넣고 셰이더로 색칠
const geo = new THREE.BufferGeometry();
geo.setAttribute('position', new THREE.BufferAttribute(M.pos, 3));
const litAttr = new THREE.BufferAttribute(new Float32Array(V), 1).setUsage(THREE.DynamicDrawUsage);
const valAttr = new THREE.BufferAttribute(new Float32Array(V), 1).setUsage(THREE.DynamicDrawUsage);
geo.setAttribute('lit', litAttr);
geo.setAttribute('val', valAttr);
geo.setIndex(new THREE.BufferAttribute(M.idx, 1));
const HOUR_STOPS = [[0,'#1b0d33'],[1,'#57176b'],[2,'#a6245a'],[3,'#e34a33'],[4,'#f99430'],[6,'#feeeb0']];
const IRR_STOPS = [[0,'#ff6a0d'],[500,'#ffcc40'],[1000,'#fffbe5']];
const glslColor = h => { const c = new THREE.Color(h); return `vec3(${c.r.toFixed(3)},${c.g.toFixed(3)},${c.b.toFixed(3)})`; };
const glslRamp = (name, stops) => {
  let s = `vec3 ${name}(float x){\n`;
  for (let i = 0; i < stops.length - 1; i++) {
    const [a, ca] = stops[i], [b, cb] = stops[i + 1];
    s += `  if (x < ${b.toFixed(1)}) return mix(${glslColor(ca)}, ${glslColor(cb)}, clamp((x - ${a.toFixed(1)}) / ${(b - a).toFixed(1)}, 0., 1.));\n`;
  }
  return s + `  return ${glslColor(stops[stops.length - 1][1])};\n}\n`;
};
const heatMat = new THREE.ShaderMaterial({
  uniforms: { uMode: { value: 0 }, uOpacity: { value: 0.88 } },
  vertexShader: `attribute float lit; attribute float val; varying float vLit; varying float vVal;
    void main(){ vLit = lit; vVal = val; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: `uniform int uMode; uniform float uOpacity; varying float vLit; varying float vVal;
    ${glslRamp('hoursRamp', HOUR_STOPS)}${glslRamp('irrRamp', IRR_STOPS)}
    void main(){
      if (uMode == 0) {
        float a = smoothstep(0.3, 0.7, vLit);
        if (a < 0.01) discard;
        gl_FragColor = vec4(irrRamp(vVal / max(vLit, 1e-3)), 0.9 * a);
      } else {
        gl_FragColor = vec4(hoursRamp(vVal), uOpacity);
      }
    }`,
  transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -4,
});
const heat = new THREE.Mesh(geo, heatMat);
heat.renderOrder = 1;
scene.add(heat);
geo.computeBoundingBox();
const box = geo.boundingBox.clone();
const center = box.getCenter(new THREE.Vector3());
const size = box.getSize(new THREE.Vector3()).length();
const floorY = box.min.y;

// 사진 텍스처 방 (room.glb)
let room = null; const roomMats = [];
const glbBytes = bytesOf(document.getElementById('glb').textContent.trim());
new GLTFLoader().parse(glbBytes.buffer, '', g => {
  room = g.scene; scene.add(room);
  room.traverse(o => { if (o.isMesh) roomMats.push(o.material); });
  update();
}, err => { document.getElementById('info').textContent = '모델 로드 실패: ' + err; });

// 방위 표시: 방 바깥 바닥 높이에 북/동/남/서 글자 + 북쪽 화살표
function textSprite(txt, color) {
  const c = document.createElement('canvas'); c.width = c.height = 128;
  const g = c.getContext('2d'); g.fillStyle = color; g.font = 'bold 72px sans-serif';
  g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(txt, 64, 68);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false }));
  s.scale.setScalar(size * 0.07); return s;
}
const modelDir = b => { const r = (b - D.heading_deg) * Math.PI / 180; return new THREE.Vector3(Math.sin(r), 0, -Math.cos(r)); };
const R0 = Math.max(box.max.x - box.min.x, box.max.z - box.min.z) * 0.5 + size * 0.12;
[['북', 0, '#ff6b6b'], ['동', 90, '#cfd2da'], ['남', 180, '#cfd2da'], ['서', 270, '#cfd2da']].forEach(([t, b, col]) => {
  const s = textSprite(t, col); s.position.copy(center).addScaledVector(modelDir(b), R0); s.position.y = floorY; scene.add(s);
});
scene.add(new THREE.ArrowHelper(modelDir(0), new THREE.Vector3(center.x, floorY, center.z).addScaledVector(modelDir(0), R0 * 0.55),
  R0 * 0.3, 0xff6b6b, size * 0.03, size * 0.02));

// 태양: 하루 경로 + 현재 위치
const SUN_R = size * 1.1;
const sunBall = new THREE.Mesh(new THREE.SphereGeometry(size * 0.035, 24, 16), new THREE.MeshBasicMaterial({ color: 0xffd34d }));
scene.add(sunBall);
const sunLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineDashedMaterial({ color: 0xffd34d, dashSize: size * 0.02, gapSize: size * 0.015, transparent: true, opacity: 0.6 }));
scene.add(sunLine);
const sunToRoom = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffd34d, transparent: true, opacity: 0.25 }));
scene.add(sunToRoom);

// 빛줄기: 창문 모서리 → 태양 반대 방향 광선이 실내 면에 닿는 점
const beams = new THREE.Group(); scene.add(beams);
const beamFill = new THREE.MeshBasicMaterial({ color: 0xffd66b, transparent: true, opacity: 0.09, side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending });
const beamEdge = new THREE.LineBasicMaterial({ color: 0xffd66b, transparent: true, opacity: 0.7 });
const raycaster = new THREE.Raycaster();

// ---------------------------------------------------------------- 상태
let dayIdx = 0, stepIdx = 0, mode = 0, timer = null;
const slider = document.getElementById('slider');
const daysEl = document.getElementById('days');
D.days.forEach((d, i) => {
  const b = document.createElement('button'); b.textContent = `${d.label} ${d.date.slice(5)}`;
  b.onclick = () => setDay(i); daysEl.appendChild(b);
});
function nearestStep(day, h) {
  let best = 0; day.steps.forEach((s, i) => { if (Math.abs(s.hour - h) < Math.abs(day.steps[best].hour - h)) best = i; });
  return best;
}
function setDay(i) {
  const prevHour = D.days[dayIdx].steps[stepIdx]?.hour ?? D.init_hour;
  dayIdx = i; const d = D.days[i];
  [...daysEl.children].forEach((b, k) => b.classList.toggle('on', k === i));
  slider.max = Math.max(0, d.steps.length - 1);
  stepIdx = d.steps.length ? nearestStep(d, prevHour) : 0; slider.value = stepIdx;
  const pts = d.steps.map(s => new THREE.Vector3(...s.dir).multiplyScalar(SUN_R).add(center));
  sunLine.geometry.dispose(); sunLine.geometry = new THREE.BufferGeometry().setFromPoints(pts); sunLine.computeLineDistances();
  update();
}
function setMode(m) {
  mode = m;
  document.getElementById('m0').classList.toggle('on', m === 0);
  document.getElementById('m1').classList.toggle('on', m === 1);
  document.getElementById('timeSec').hidden = m !== 0;
  document.getElementById('daySec').hidden = m !== 1;
  const stops = m === 0 ? IRR_STOPS : HOUR_STOPS, max = stops[stops.length - 1][0];
  document.getElementById('legendTitle').textContent = m === 0 ? '직사광 세기 (W/m²) — 색이 없는 곳은 그늘' : '하루 직사광 시간 (h)';
  document.getElementById('legendBar').style.background =
    `linear-gradient(90deg, ${stops.map(([v, c]) => `${c} ${(v / max * 100).toFixed(0)}%`).join(', ')})`;
  document.getElementById('legendTicks').innerHTML = stops.map(([v]) => `<span>${v}${m === 1 && v === max ? '+' : ''}</span>`).join('');
  update();
}
document.getElementById('m0').onclick = () => setMode(0);
document.getElementById('m1').onclick = () => setMode(1);
slider.oninput = () => { stepIdx = +slider.value; update(); };
function step(k) { const n = D.days[dayIdx].steps.length; if (!n) return; stepIdx = (stepIdx + k + n) % n; slider.value = stepIdx; update(); }
addEventListener('keydown', e => { if (e.key === 'ArrowRight') step(1); if (e.key === 'ArrowLeft') step(-1); });
const playBtn = document.getElementById('play');
playBtn.onclick = () => {
  if (timer) { clearInterval(timer); timer = null; playBtn.textContent = '▶ 재생'; return; }
  if (mode !== 0) setMode(0);
  timer = setInterval(() => step(1), 120); playBtn.textContent = '❚❚ 정지';
};

// ---------------------------------------------------------------- 갱신
function update() {
  const d = D.days[dayIdx], s = d.steps[stepIdx];
  const lit = litAttr.array, val = valAttr.array;
  if (mode === 0 && s) {
    const B = D.bytes_per_step, base = stepIdx * B, bits = d.bits, [dx, dy, dz] = s.dir, n = M.nrm;
    for (let i = 0; i < V; i++) {
      const on = (bits[base + (i >> 3)] >> (i & 7)) & 1;
      lit[i] = on;
      val[i] = on ? s.dni * Math.max(0, n[3 * i] * dx + n[3 * i + 1] * dy + n[3 * i + 2] * dz) : 0;
    }
  } else { lit.fill(1); val.set(d.hoursArr); }
  litAttr.needsUpdate = valAttr.needsUpdate = true;
  heatMat.uniforms.uMode.value = mode;
  for (const m of roomMats) m.color.setScalar(mode === 0 ? 0.42 : 1);

  sunBall.visible = sunToRoom.visible = !!s;
  if (s) {
    sunBall.position.copy(new THREE.Vector3(...s.dir).multiplyScalar(SUN_R).add(center));
    sunToRoom.geometry.dispose(); sunToRoom.geometry = new THREE.BufferGeometry().setFromPoints([sunBall.position, center]);
  }
  updateBeams(s);
  updatePanel(d, s);
  drawChart();
}

function updateBeams(s) {
  for (const c of beams.children) c.geometry.dispose();
  beams.clear();
  if (mode !== 0 || !s || !room) return;
  const sun = new THREE.Vector3(...s.dir), into = sun.clone().negate();
  const fill = [], edge = [];
  for (const w of D.windows) {
    const nOut = new THREE.Vector3(...w.n_out);
    if (nOut.dot(sun) <= 0) continue;                         // 창문이 해를 등지고 있음
    const C = w.corners.map(c => new THREE.Vector3(...c));
    const mid = C.reduce((a, b) => a.add(b), new THREE.Vector3()).multiplyScalar(0.25);
    const P = C.map(c => {
      const o = c.clone().lerp(mid, 0.004).addScaledVector(nOut, -0.004);
      raycaster.set(o, into);
      const hit = raycaster.intersectObject(room, true)[0];
      return hit ? hit.point : o.clone().addScaledVector(into, size);
    });
    for (let k = 0; k < 4; k++) {
      const a = C[k], b = C[(k + 1) % 4], pa = P[k], pb = P[(k + 1) % 4];
      fill.push(a, b, pb, a, pb, pa);
      edge.push(a, pa, pa, pb);
    }
  }
  if (fill.length) {
    beams.add(new THREE.Mesh(new THREE.BufferGeometry().setFromPoints(fill), beamFill));
    beams.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(edge), beamEdge));
  }
}

function updatePanel(d, s) {
  const st = d.stats, sm = d.summary;
  document.getElementById('clock').textContent = s ? hhmm(s.hour) : '--:--';
  document.getElementById('sunpos').textContent = s ? `고도 ${s.alt.toFixed(1)}° · 방위 ${s.az.toFixed(0)}° ${dir8(s.az)}` : '해가 뜨지 않음';
  const floor = D.surfaces.find(x => x.name === 'floor');
  if (mode === 0) {
    const i = stepIdx, ok = !!s;
    const rows = ok ? [
      ['맑은 날 직달일사', `${fmt(s.dni)} W/m²`],
      ['창문을 통과하는 직사광', `${fmt(st.flux_W[i])} W`],
      ['실내 면에 닿은 직사광', `${fmt(st.total_W[i])} W`],
      ['빛이 닿은 바닥', `${fmt(st.floor_m2[i], 2)} m² (${fmt(100 * st.floor_m2[i] / floor.area_m2, 0)}%)`],
      ['빛이 닿은 벽', `${fmt(st.wall_m2[i], 2)} m²`],
    ] : [];
    document.getElementById('nowStats').innerHTML = rows.map(([a, b]) => `<dt>${a}</dt><dd>${b}</dd>`).join('');
    document.getElementById('winNow').innerHTML = D.windows.length ? D.windows.map((w, k) => {
      const W = ok ? st.win_W[k][i] : 0, dep = ok ? st.win_depth[k][i] : 0;
      return `<tr><td>${w.label}</td><td>${w.facing}향</td><td>${fmt(W)}</td><td>${dep > 0 ? fmt(dep, 2) + ' m' : '–'}</td></tr>`;
    }).join('') : '<tr><td colspan="4">창문 없음</td></tr>';
  } else {
    const t0 = d.steps[0], t1 = d.steps[d.steps.length - 1];
    document.getElementById('dayStats').innerHTML = [
      ['해 뜬 시간', t0 ? `${hhmm(t0.hour)} ~ ${hhmm(t1.hour)}` : '–'],
      ['실내로 들어온 직사광', `${fmt(sm.total_kwh, 2)} kWh`],
      ['창문 통과량 (검산)', `${fmt(sm.flux_kwh, 2)} kWh`],
    ].map(([a, b]) => `<dt>${a}</dt><dd>${b}</dd>`).join('');
    document.getElementById('surfDay').innerHTML = sm.surfaces.map(x =>
      `<tr><td>${x.label}</td><td>${fmt(x.mean_h, 2)}</td><td>${fmt(x.max_h, 1)}</td><td>${fmt(x.kwh, 2)}</td></tr>`).join('');
    document.getElementById('winDay').innerHTML = D.windows.length ? sm.windows.map((x, k) =>
      `<tr><td>${D.windows[k].label}</td><td>${D.windows[k].facing}향</td><td>${fmt(x.sun_h, 1)}</td><td>${fmt(x.kwh_in, 2)}</td><td>${x.max_depth_m > 0 ? fmt(x.max_depth_m, 2) + ' m' : '–'}</td></tr>`).join('')
      : '<tr><td colspan="5">창문 없음</td></tr>';
  }
}

// 하루 그래프: 실내 직사광 W(노랑, 채움) + 바닥 일사 면적 m²(파랑) + 현재 시각 선
const chart = document.getElementById('chart');
function chartX(d, i, w) { const n = d.steps.length; return n > 1 ? 8 + (w - 16) * i / (n - 1) : w / 2; }
function drawChart() {
  const d = D.days[dayIdx], st = d.stats, n = d.steps.length;
  const w = chart.clientWidth, h = chart.clientHeight, r = devicePixelRatio;
  chart.width = w * r; chart.height = h * r;
  const g = chart.getContext('2d'); g.scale(r, r); g.clearRect(0, 0, w, h);
  if (!n) return;
  const top = 10, bot = h - 18;
  const maxW = Math.max(1, ...st.flux_W), maxA = Math.max(0.01, ...st.floor_m2);
  g.fillStyle = '#9c9ea8'; g.font = '10px sans-serif'; g.textAlign = 'center';
  for (let hr = Math.ceil(d.steps[0].hour / 3) * 3; hr <= d.steps[n - 1].hour; hr += 3) {
    const i = d.steps.findIndex(s => s.hour >= hr); if (i < 0) continue;
    const x = chartX(d, i, w); g.fillText(String(hr), x, h - 5);
    g.fillStyle = '#2c2f37'; g.fillRect(x, top, 1, bot - top); g.fillStyle = '#9c9ea8';
  }
  const path = (arr, max) => { g.beginPath(); arr.forEach((v, i) => { const x = chartX(d, i, w), y = bot - (bot - top) * v / max; i ? g.lineTo(x, y) : g.moveTo(x, y); }); };
  path(st.total_W, maxW); g.lineTo(chartX(d, n - 1, w), bot); g.lineTo(chartX(d, 0, w), bot); g.closePath();
  g.fillStyle = 'rgba(255,201,77,.25)'; g.fill();
  path(st.total_W, maxW); g.strokeStyle = '#ffc94d'; g.lineWidth = 1.5; g.stroke();
  path(st.floor_m2, maxA); g.strokeStyle = '#6d8cff'; g.lineWidth = 1.2; g.stroke();
  const x = chartX(d, stepIdx, w);
  g.fillStyle = '#ffffff'; g.fillRect(x - 0.5, top - 4, 1, bot - top + 4);
  g.textAlign = 'left'; g.fillStyle = '#9c9ea8'; g.fillText(`최대 ${fmt(Math.max(...st.total_W))} W`, 10, 10);
}
chart.onclick = e => {
  const d = D.days[dayIdx], n = d.steps.length; if (!n) return;
  const rect = chart.getBoundingClientRect();
  stepIdx = Math.max(0, Math.min(n - 1, Math.round((e.clientX - rect.left - 8) / (rect.width - 16) * (n - 1))));
  slider.value = stepIdx; if (mode !== 0) setMode(0); else update();
};

// ---------------------------------------------------------------- 시점
function setView(id) {
  controls.enableZoom = controls.enablePan = id !== 'inside';
  if (id === 'outside') { controls.target.copy(center); camera.position.set(center.x + size * 0.55, center.y + size * 0.85, center.z + size * 0.55); camera.fov = 50; }
  if (id === 'top') { controls.target.copy(center); camera.position.set(center.x, center.y + size * 1.7, center.z + 0.001); camera.fov = 50; }
  if (id === 'inside') { controls.target.set(0, 0, 0); camera.position.set(0, 0, 0.001); camera.fov = 75; }
  camera.updateProjectionMatrix();
  for (const b of document.querySelectorAll('#views button')) b.classList.toggle('on', b.id === 'v_' + id);
}
for (const id of ['outside', 'top', 'inside']) document.getElementById('v_' + id).onclick = () => setView(id);
function resize() {
  const w = host.clientWidth, h = host.clientHeight;
  renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); drawChart();
}
addEventListener('resize', resize);
// 주소 뒤 ?t=09:30&day=1&mode=1&view=top 으로 처음 상태를 정할 수 있음
const q = new URLSearchParams(location.search);
if (q.get('t')) { const [h, m] = q.get('t').split(':').map(Number); D.init_hour = h + (m || 0) / 60; }
dayIdx = Math.min(D.days.length - 1, +(q.get('day') || 0));
resize(); setView(['outside', 'top', 'inside'].includes(q.get('view')) ? q.get('view') : 'outside');
stepIdx = D.days[dayIdx].steps.length ? nearestStep(D.days[dayIdx], D.init_hour) : 0;
setDay(dayIdx); setMode(q.get('mode') === '1' ? 1 : 0);
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
</script></body></html>
"""


if __name__ == "__main__":
    main()
