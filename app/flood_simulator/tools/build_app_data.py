"""
Builds the JSON files the flood probability map app (flood_simulator) reads, from the analysis result files.

Input (flood_risk_ML/)
  Grid-level prediction results/grid_flood_risk.csv     q_g and annual / 2-year / 4-year probability per grid cell
  Grid-level prediction results/scenario_params.json    posterior samples of α, β and the PC standardisation
  Grid clustering results/grid_clusters.csv  flood-year combination and cluster assignment method per cell
  Grid clustering results/cluster_summary.csv
  Rainfall zonation results/region_year_features_k5.csv, region_assignment.csv
  Prediction model results/cv_results.csv
  data/station_coords.csv, data/seoul_gu.geojson   (downloaded by fetch_geo.py)

Output (app/flood_simulator/data/)
  grids.json     13,054 grid cells (column arrays), including the latitude/longitude of the four corners
  model.json     pc_mean, pc_std, α (S, C), β (S, C, K)
  pca.json       PCA parameters to rebuild regional rainfall from the sliders (PC1–PC3 only)
  years.json     PC1–PC3 per year, max 1h / 24h rainfall per region, whether flood data exists
  clusters.json  size, typical year combinations and β summary per cluster
  meta.json      slider ranges, presets, pattern check threshold, numbers used in the notes, station coordinates and regions
  seoul.geojson  borders of Seoul's 25 districts (rain map mask and outlines)

The sliders use PC1–PC3 only. PC4 is not an input of the flood model and is set to 0 when rebuilding rainfall.

Checks
  1. Recompute the annual probability from model.json and compare with p_annual in grid_flood_risk.csv
  2. Compare the PCs in years.json with 'rainfall data final ver.csv'
  3. Check that the grid lookup returns each cell's own ix, iy after a cell centre -> WGS84 -> EPSG:5179 round trip

Run:
    py "app\\flood_simulator\\tools\\build_app_data.py"     (from the repository root)
"""
import json
import os

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.special import expit

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(APP_DIR)), "flood_risk_ML")
OUT_DIR = os.path.join(APP_DIR, "data")

RISK_PATH = os.path.join(BASE_DIR, "Grid-level prediction results", "grid_flood_risk.csv")
PARAMS_PATH = os.path.join(BASE_DIR, "Grid-level prediction results", "scenario_params.json")
CLUSTERS_PATH = os.path.join(BASE_DIR, "Grid clustering results", "grid_clusters.csv")
CLUSTER_SUMMARY_PATH = os.path.join(BASE_DIR, "Grid clustering results", "cluster_summary.csv")
REGION_FEATURES_PATH = os.path.join(BASE_DIR, "Rainfall zonation results", "region_year_features_k5.csv")
REGION_ASSIGN_PATH = os.path.join(BASE_DIR, "Rainfall zonation results", "region_assignment.csv")
FINAL_PC_PATH = os.path.join(BASE_DIR, "rainfall data final ver.csv")
CV_PATH = os.path.join(BASE_DIR, "Prediction model results", "cv_results.csv")
STATION_PATH = os.path.join(BASE_DIR, "data", "station_coords.csv")
SEOUL_GU_PATH = os.path.join(BASE_DIR, "data", "seoul_gu.geojson")

# values exactly as in '강우량 차원 축소 방법.txt' (the PCA settings file)
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

N_PCS = 3  # number of principal components used as sliders (PC4 left out)
NO_DATA_YEARS = [2015, 2021]
TRAIN_YEARS = [y for y in range(2010, 2026) if y not in NO_DATA_YEARS]
PRESET_YEARS = [2010, 2011, 2022, 2025]
GRID_ORIGIN = (935550.0, 1937350.0)  # centre of grid cell ix = 0, iy = 0 (EPSG:5179)
CELL = 100.0
MODEL_DECIMALS = 4

to_wgs = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)
to_5179 = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)


def round_list(a, d):
    """Rounds an array to d decimals and returns it as a list."""
    return np.round(np.asarray(a, dtype=float), d).tolist()


def write_json(name, obj):
    """Writes one compact JSON file to the output folder and prints its size."""
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    print(f"  {name:<14} {os.path.getsize(path) / 1024:8.0f} KB")


# ---------------------------------------------------------------- one builder per output file

def build_grids(risk, clusters):
    """Turns the grid table into column arrays. Corners are [lat, lon] in the order [SW, SE, NE, NW]."""
    g = risk.join(clusters[["combo", "assign", "attach_dist_m"]], on="cell_id", how="left")
    if g["combo"].isna().any():
        raise SystemExit("grid_clusters.csv에 없는 격자가 grid_flood_risk.csv에 있습니다.")

    corners = []
    for dx, dy in [(-50, -50), (50, -50), (50, 50), (-50, 50)]:
        lon, lat = to_wgs.transform(g["x"].values + dx, g["y"].values + dy)
        corners.append(np.stack([lat, lon], axis=1))
    corners = np.round(np.stack(corners, axis=1), 6)  # (cell, 4, 2)

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
    """Keeps the model parameters the app needs, with α and β rounded."""
    return {
        "description": params["description"],
        "pcs": params["pcs"],
        "pc_mean": params["pc_mean"],
        "pc_std": params["pc_std"],
        "alpha": round_list(params["alpha"], MODEL_DECIMALS),
        "beta": round_list(params["beta"], MODEL_DECIMALS),
    }


def project_pcs(features):
    """Projects yearly regional rainfall features onto the principal components."""
    z = (features[FEATURES].values - np.array(PCA_MEAN)) / np.array(PCA_STD)
    return z @ np.array(PCA_COMPONENTS).T


def build_years(region_features):
    """Builds the per-year PCs and regional max rainfall, and returns the PC table for the checks."""
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
    """Mean and 5–95% quantiles of the β posterior samples. A cluster counts as 'sensitive' to an axis if that interval excludes 0."""
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
    """Slider ranges, presets, the pattern check threshold, regions and station coordinates.
    Pattern distances are measured on PC1–PC3 divided by their standard deviation across years."""
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


# ---------------------------------------------------------------- checks

def check_annual(model, grids, years, risk):
    """Recomputes the annual flood probability from model.json; returns the largest difference from p_annual."""
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
    """Counts grid cells that the WGS84 round trip maps to a different ix, iy."""
    lon, lat = to_wgs.transform(risk["x"].values, risk["y"].values)
    X, Y = to_5179.transform(lon, lat)
    ix = np.round((X - GRID_ORIGIN[0]) / CELL).astype(int)
    iy = np.round((Y - GRID_ORIGIN[1]) / CELL).astype(int)
    return int(((ix != risk["ix"].values) | (iy != risk["iy"].values)).sum())


def main():
    """Loads the inputs, writes every app data file, and runs the three checks."""
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
    print(f"  연도별 PC vs rainfall data final ver.csv: 최대 차이 {pc_diff:.1e} -> {'통과' if pc_diff < 1e-6 else '실패'}")
    miss = check_grid_lookup(risk)
    print(f"  격자 찾기 왕복: 어긋난 격자 {miss}개 -> {'통과' if miss == 0 else '실패'}")
    if diff >= 5e-4 or pc_diff >= 1e-6 or miss:
        raise SystemExit("검산 실패")
    print(f"\n저장 완료: {OUT_DIR}")


if __name__ == "__main__":
    main()
