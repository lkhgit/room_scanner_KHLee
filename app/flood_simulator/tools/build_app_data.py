"""
침수 확률 지도 앱(flood-app)이 읽을 JSON을 분석 결과 파일에서 만든다.

입력 (침수 위험/)
  격자 예측 결과/grid_flood_risk.csv     격자별 q_g, 연간·2년·4년 확률
  격자 예측 결과/scenario_params.json    사후 표본 α, β와 PC 표준화 값
  격자 클러스터링 결과/grid_clusters.csv  격자별 침수 연도 조합, 클러스터 배정 방식
  격자 클러스터링 결과/cluster_summary.csv
  강우 권역화 결과/region_year_features_k5.csv, region_assignment.csv
  예측 모델 결과/cv_results.csv
  data/station_coords.csv, data/seoul_gu.geojson   (fetch_geo.py로 내려받음)

출력 (flood-app/data/)
  grids.json     격자 13,054개 (열 단위 배열), 네 꼭짓점 위경도 포함
  model.json     pc_mean, pc_std, α (S, C), β (S, C, K)
  pca.json       슬라이더 → 권역 강우 복원용 PCA 파라미터 (PC1–PC3만)
  years.json     연도별 PC1–PC3, 권역별 최대 1h·24h 강우, 침수 데이터 유무
  clusters.json  클러스터별 크기, 대표 연도 조합, β 요약
  meta.json      슬라이더 범위, 프리셋, 패턴 신뢰도 기준, 안내 문구 수치, 관측소 좌표·권역
  seoul.geojson  서울 25개 구 경계 (강우 지도 마스크와 경계선)

슬라이더는 PC1–PC3만 쓴다. PC4는 침수 모델 입력이 아니고, 강우 복원에서도 0으로 둔다.

검산
  1. model.json으로 연간 확률을 다시 계산해 grid_flood_risk.csv의 p_annual과 비교
  2. years.json의 PC를 '최종 강우량 데이터.csv'와 비교
  3. 격자 중심 → WGS84 → EPSG:5179 왕복 후 격자 찾기 식이 자기 ix, iy를 돌려주는지

실행:
    & "C:\\Users\\KHLee\\AppData\\Local\\Python\\bin\\python.exe" "flood-app\\tools\\build_app_data.py"
"""
import json
import os

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.special import expit

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(APP_DIR)
OUT_DIR = os.path.join(APP_DIR, "data")

RISK_PATH = os.path.join(BASE_DIR, "격자 예측 결과", "grid_flood_risk.csv")
PARAMS_PATH = os.path.join(BASE_DIR, "격자 예측 결과", "scenario_params.json")
CLUSTERS_PATH = os.path.join(BASE_DIR, "격자 클러스터링 결과", "grid_clusters.csv")
CLUSTER_SUMMARY_PATH = os.path.join(BASE_DIR, "격자 클러스터링 결과", "cluster_summary.csv")
REGION_FEATURES_PATH = os.path.join(BASE_DIR, "강우 권역화 결과", "region_year_features_k5.csv")
REGION_ASSIGN_PATH = os.path.join(BASE_DIR, "강우 권역화 결과", "region_assignment.csv")
FINAL_PC_PATH = os.path.join(BASE_DIR, "최종 강우량 데이터.csv")
CV_PATH = os.path.join(BASE_DIR, "예측 모델 결과", "cv_results.csv")
STATION_PATH = os.path.join(BASE_DIR, "data", "station_coords.csv")
SEOUL_GU_PATH = os.path.join(BASE_DIR, "data", "seoul_gu.geojson")

# '강우량 차원 축소 방법.txt'의 값 그대로
FEATURES = ["R1_max1h", "R1_max24h", "R2_max1h", "R2_max24h", "R3_max1h",
            "R3_max24h", "R4_max1h", "R4_max24h", "R5_max1h", "R5_max24h"]
PCA_MEAN = [42.5220588235, 163.6176470588, 42.4068627451, 162.4191176471, 44.4740896359,
            157.3425770308, 40.7426470588, 160.2990196078, 42.5166316527, 150.1123949580]
PCA_STD = [10.7578831641, 63.6128350006, 12.5744639029, 79.1179164536, 16.6153532987,
           82.6494932153, 9.3466208297, 76.8875059376, 12.3986368504, 68.3287537912]
PCA_COMPONENTS = [
    [0.2101970085, 0.2993260555, 0.3143041771, 0.3615778543, 0.3142783965,
     0.3520592758, 0.2874883394, 0.3369664639, 0.2933773034, 0.3629223611],
    [0.6634584533, 0.4618301124, -0.4445844862, -0.0522403640, -0.2906166184,
     -0.0229601004, -0.1418593797, 0.1260657606, -0.1468471299, 0.0598789002],
    [0.4148873545, -0.2245327943, 0.0115507633, -0.2180038014, -0.0297190861,
     -0.2968199658, 0.4788149315, -0.3004096332, 0.5600618425, -0.0873507858],
    [-0.3388481870, 0.1677106813, 0.0249032413, -0.1202067571, -0.5117573595,
     -0.2475389278, 0.5738544876, 0.3762822943, -0.1515348615, 0.1579681287],
]

N_PCS = 3  # 슬라이더로 쓰는 주성분 수 (PC4 제외)
NO_DATA_YEARS = [2015, 2021]
TRAIN_YEARS = [y for y in range(2010, 2026) if y not in NO_DATA_YEARS]
PRESET_YEARS = [2010, 2011, 2022, 2025]
GRID_ORIGIN = (935550.0, 1937350.0)  # ix = 0, iy = 0 격자 중심 (EPSG:5179)
CELL = 100.0
MODEL_DECIMALS = 4

to_wgs = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)
to_5179 = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)


def round_list(a, d):
    return np.round(np.asarray(a, dtype=float), d).tolist()


def write_json(name, obj):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    print(f"  {name:<14} {os.path.getsize(path) / 1024:8.0f} KB")


# ---------------------------------------------------------------- 파일별 생성

def build_grids(risk, clusters):
    """격자 표를 열 단위 배열로. 꼭짓점은 [서남, 동남, 동북, 서북] 순서의 [lat, lon]."""
    g = risk.join(clusters[["combo", "assign", "attach_dist_m"]], on="cell_id", how="left")
    if g["combo"].isna().any():
        raise SystemExit("grid_clusters.csv에 없는 격자가 grid_flood_risk.csv에 있습니다.")

    corners = []
    for dx, dy in [(-50, -50), (50, -50), (50, 50), (-50, 50)]:
        lon, lat = to_wgs.transform(g["x"].values + dx, g["y"].values + dy)
        corners.append(np.stack([lat, lon], axis=1))
    corners = np.round(np.stack(corners, axis=1), 6)  # (격자, 4, 2)

    prob_cols = [c for c in risk.columns if c.startswith("p_")]
    out = {
        "n": len(g),
        "cell_id": g["cell_id"].tolist(),
        "ix": g["ix"].tolist(),
        "iy": g["iy"].tolist(),
        "corners": corners.tolist(),
        "cluster": g["cluster"].tolist(),
        "log_q": round_list(g["log_q"], 6),
        "years": [[int(y) for y in str(c).split("+")] for c in g["combo"]],
        "assign": g["assign"].tolist(),
        "attach_dist_m": round_list(g["attach_dist_m"], 1),
    }
    for c in prob_cols:
        out[c] = round_list(g[c], 4)
    return out


def build_model(params):
    return {
        "description": params["description"],
        "pcs": params["pcs"],
        "pc_mean": params["pc_mean"],
        "pc_std": params["pc_std"],
        "alpha": round_list(params["alpha"], MODEL_DECIMALS),
        "beta": round_list(params["beta"], MODEL_DECIMALS),
    }


def project_pcs(features):
    z = (features[FEATURES].values - np.array(PCA_MEAN)) / np.array(PCA_STD)
    return z @ np.array(PCA_COMPONENTS).T


def build_years(region_features):
    wide = region_features.pivot(index="year", columns="region",
                                 values=["max_1h_mean", "max_24h_mean"])
    feats = pd.DataFrame(index=wide.index)
    for r in sorted(region_features["region"].unique()):
        feats[f"R{r}_max1h"] = wide[("max_1h_mean", r)]
        feats[f"R{r}_max24h"] = wide[("max_24h_mean", r)]
    pcs = project_pcs(feats)
    rows = []
    for i, year in enumerate(feats.index):
        rows.append({
            "year": int(year),
            "pc": round_list(pcs[i, :N_PCS], 6),
            "max1h": round_list(feats.loc[year, FEATURES[0::2]], 2),
            "max24h": round_list(feats.loc[year, FEATURES[1::2]], 2),
            "has_flood_data": int(year) in TRAIN_YEARS,
        })
    cols = [f"PC{k + 1}" for k in range(N_PCS)]
    return rows, pd.DataFrame(pcs[:, :N_PCS], index=feats.index, columns=cols)


def build_clusters(summary, beta, pcs):
    """β 사후 표본의 평균과 5–95% 분위수. 구간이 0을 포함하지 않으면 그 축에 '민감'하다고 본다."""
    rows = []
    for _, r in summary.iterrows():
        c = int(r["cluster"])
        b = beta[:, c, :]
        rows.append({
            "cluster": c,
            "size": int(r["size"]),
            "top_combos": r["top_combos"],
            "beta_mean": round_list(b.mean(axis=0), 4),
            "beta_q05": round_list(np.quantile(b, 0.05, axis=0), 4),
            "beta_q95": round_list(np.quantile(b, 0.95, axis=0), 4),
            "pcs": pcs,
        })
    return rows


def build_meta(pc_table, pc_mean, cv, assignment, stations):
    """슬라이더 범위·프리셋·패턴 신뢰도 기준·권역과 관측소 좌표.
    패턴 거리는 PC1–PC3을 연도 간 표준편차로 나눈 공간에서 잰다."""
    pc_scale = pc_table.std(ddof=0).values
    zs = pc_table.values / pc_scale
    d = np.linalg.norm(zs[:, None, :] - zs[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    st = stations.merge(assignment[["stn", "k5"]], on="stn")

    share = cv[(cv["metric"] == "unpredictable_share") & (cv["model"] == "all")]
    share = {int(y): round(float(v), 4) for y, v in zip(share["year"], share["value"])}

    return {
        "slider_min": round_list(pc_table.min(), 1),
        "slider_max": round_list(pc_table.max(), 1),
        "presets": [{"label": "평년", "pc": round_list(pc_mean[:N_PCS], 4)}]
                   + [{"label": f"{y}년 비", "year": y,
                       "pc": round_list(pc_table.loc[y], 4)} for y in PRESET_YEARS],
        "pattern_scale": round_list(pc_scale, 6),
        "pattern_max_nn_dist": round(float(d.min(axis=1).max()), 4),
        "train_years": TRAIN_YEARS,
        "no_data_years": NO_DATA_YEARS,
        "unpredictable_share": share,
        "grid_origin": list(GRID_ORIGIN),
        "cell_size": CELL,
        "regions": [{"id": int(r), "stations": g["name"].str.rstrip("*").tolist()}
                    for r, g in assignment.groupby("k5")],
        "stations": [{"name": r["name"].rstrip("*"), "stn": int(r["stn"]), "region": int(r["k5"]),
                      "lat": float(r["lat"]), "lon": float(r["lon"])} for _, r in st.iterrows()],
    }


# ---------------------------------------------------------------- 검산

def check_annual(model, grids, years, risk):
    alpha, beta = np.array(model["alpha"]), np.array(model["beta"])
    labels, log_q = np.array(grids["cluster"]), np.array(grids["log_q"])
    pc = np.array([y["pc"][:len(model["pcs"])] for y in years if y["year"] in TRAIN_YEARS])
    Z = (pc - np.array(model["pc_mean"])) / np.array(model["pc_std"])
    eta = alpha[:, :, None] + np.einsum("sck,yk->scy", beta, Z)
    p = np.zeros((len(alpha), len(labels)))
    for y in range(Z.shape[0]):
        p += expit(eta[:, labels, y] + log_q[None, :])
    p_annual = (p / Z.shape[0]).mean(axis=0)
    return float(np.abs(p_annual - risk["p_annual"].values).max())


def check_grid_lookup(risk):
    lon, lat = to_wgs.transform(risk["x"].values, risk["y"].values)
    X, Y = to_5179.transform(lon, lat)
    ix = np.round((X - GRID_ORIGIN[0]) / CELL).astype(int)
    iy = np.round((Y - GRID_ORIGIN[1]) / CELL).astype(int)
    return int(((ix != risk["ix"].values) | (iy != risk["iy"].values)).sum())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1/3] 입력 로드")
    risk = pd.read_csv(RISK_PATH, encoding="utf-8-sig")
    clusters = pd.read_csv(CLUSTERS_PATH, encoding="utf-8-sig").set_index("cell_id")
    summary = pd.read_csv(CLUSTER_SUMMARY_PATH, encoding="utf-8-sig")
    region_features = pd.read_csv(REGION_FEATURES_PATH, encoding="utf-8-sig")
    cv = pd.read_csv(CV_PATH, encoding="utf-8-sig")
    with open(PARAMS_PATH, encoding="utf-8") as f:
        params = json.load(f)
    if params["cell_id"] != risk["cell_id"].tolist():
        raise SystemExit("scenario_params.json과 grid_flood_risk.csv의 격자 순서가 다릅니다.")
    print(f"  격자 {len(risk):,}개, 사후 표본 {len(params['alpha'])}개, 클러스터 {len(params['alpha'][0])}개")

    print("\n[2/3] 앱용 파일 저장")
    grids = build_grids(risk, clusters)
    model = build_model(params)
    years, pc_table = build_years(region_features)
    write_json("grids.json", grids)
    write_json("model.json", model)
    write_json("pca.json", {"features": FEATURES, "mean": PCA_MEAN, "std": PCA_STD,
                            "components": PCA_COMPONENTS[:N_PCS]})
    write_json("years.json", years)
    write_json("clusters.json", build_clusters(summary, np.array(model["beta"]), model["pcs"]))
    assignment = pd.read_csv(REGION_ASSIGN_PATH, encoding="utf-8-sig")
    stations = pd.read_csv(STATION_PATH, encoding="utf-8-sig")
    if len(stations) != len(assignment):
        raise SystemExit("station_coords.csv의 관측소 수가 region_assignment.csv와 다릅니다.")
    write_json("meta.json", build_meta(pc_table, params["pc_mean"], cv, assignment, stations))
    with open(SEOUL_GU_PATH, encoding="utf-8") as f:
        write_json("seoul.geojson", json.load(f))

    print("\n[3/3] 검산")
    diff = check_annual(model, grids, years, risk)
    print(f"  연간 확률 재계산 vs p_annual: 최대 차이 {diff:.1e} -> {'통과' if diff < 5e-4 else '실패'}")
    final = pd.read_csv(FINAL_PC_PATH, encoding="utf-8-sig").set_index("year")
    pc_diff = float((pc_table - final.loc[pc_table.index, pc_table.columns]).abs().max().max())
    print(f"  연도별 PC vs 최종 강우량 데이터.csv: 최대 차이 {pc_diff:.1e} -> {'통과' if pc_diff < 1e-6 else '실패'}")
    miss = check_grid_lookup(risk)
    print(f"  격자 찾기 왕복: 어긋난 격자 {miss}개 -> {'통과' if miss == 0 else '실패'}")
    if diff >= 5e-4 or pc_diff >= 1e-6 or miss:
        raise SystemExit("검산 실패")
    print(f"\n저장 완료: {OUT_DIR}")


if __name__ == "__main__":
    main()
