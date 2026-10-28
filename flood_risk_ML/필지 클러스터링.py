"""
침수 이력이 있는 100m 격자를 '잠긴 해들의 강우 PC 평균 서명'으로 클러스터링하는 스크립트.

1. 격자별 침수 이력: 격자 x 연도(2010~2025) 침수 여부 표 (2015, 2021은 자료 없음 -> NaN)
2. 평균 서명: 격자마다 잠긴 해들의 PC1~PC4 평균
3. 표준화: 서명 4개 열을 격자 전체에 대해 열별 표준화
4. HDBSCAN: 4차원 서명으로 클러스터링, 노이즈(-1) 표시
5. 지리적 처리: 노이즈 격자를 가장 가까운 클러스터 격자(중심점 거리)의 라벨로 부착
6. 점검: 클러스터 크기, 침수 연도 수, 지도

한 번도 침수되지 않은 격자는 처음부터 제외한다.

실행:
    & "C:\\Users\\KHLee\\AppData\\Local\\Python\\bin\\python.exe" "필지 클러스터링.py"
"""
import json
import os

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FLOOD_PATH = os.path.join(BASE_DIR, "합친 침수흔적도", "서울시_침수흔적도_2010_2025.shp")
PC_PATH = os.path.join(BASE_DIR, "최종 강우량 데이터.csv")
RAIN_FEATURE_PATH = os.path.join(BASE_DIR, "강우량 클러스터링된 데이터.csv")
OUT_DIR = os.path.join(BASE_DIR, "격자 클러스터링 결과")
SOURCE_CRS = "EPSG:5179"

CELL_SIZE_M = 100
YEARS = list(range(2010, 2026))
NO_DATA_YEARS = [2015, 2021]  # 침수흔적도 자료 없음
PC_COLS = ["PC1", "PC2", "PC3", "PC4"]

MIN_CLUSTER_SIZE = 100
MIN_SAMPLES = 20

# '강우량 차원 축소 방법.txt'의 변환 파라미터 (PC 값 검증용)
RAIN_FEATURES = ["R1_max1h", "R1_max24h", "R2_max1h", "R2_max24h", "R3_max1h",
                 "R3_max24h", "R4_max1h", "R4_max24h", "R5_max1h", "R5_max24h"]
RAIN_MEAN = np.array([42.5220588235, 163.6176470588, 42.4068627451, 162.4191176471, 44.4740896359,
                      157.3425770308, 40.7426470588, 160.2990196078, 42.5166316527, 150.1123949580])
RAIN_STD = np.array([10.7578831641, 63.6128350006, 12.5744639029, 79.1179164536, 16.6153532987,
                     82.6494932153, 9.3466208297, 76.8875059376, 12.3986368504, 68.3287537912])
RAIN_COMPONENTS = np.array([
    [0.2101970085, 0.2993260555, 0.3143041771, 0.3615778543, 0.3142783965, 0.3520592758, 0.2874883394, 0.3369664639, 0.2933773034, 0.3629223611],
    [0.6634584533, 0.4618301124, -0.4445844862, -0.0522403640, -0.2906166184, -0.0229601004, -0.1418593797, 0.1260657606, -0.1468471299, 0.0598789002],
    [0.4148873545, -0.2245327943, 0.0115507633, -0.2180038014, -0.0297190861, -0.2968199658, 0.4788149315, -0.3004096332, 0.5600618425, -0.0873507858],
    [-0.3388481870, 0.1677106813, 0.0249032413, -0.1202067571, -0.5117573595, -0.2475389278, 0.5738544876, 0.3762822943, -0.1515348615, 0.1579681287],
])

PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c",
    "#fabebe", "#008080", "#e6beff", "#9a6324", "#800000", "#aaffc3", "#808000", "#000075",
]


def load_year_pcs():
    """연도별 PC1~PC4를 읽고, 변환 파라미터로 다시 계산한 값과 일치하는지 확인."""
    pcs = pd.read_csv(PC_PATH, encoding="utf-8-sig").set_index("year")[PC_COLS]
    raw = pd.read_csv(RAIN_FEATURE_PATH, encoding="utf-8-sig").set_index("year")
    manual = ((raw[RAIN_FEATURES].values - RAIN_MEAN) / RAIN_STD) @ RAIN_COMPONENTS.T
    assert np.allclose(manual, pcs.loc[raw.index].values, atol=1e-6), "PC 값이 변환 파라미터와 다릅니다"
    print(f"  PC 검증 완료: 변환 파라미터로 계산한 값과 '최종 강우량 데이터.csv' 일치 ({len(pcs)}개 연도)")
    return pcs.loc[YEARS]


def build_flood_history():
    """침수 폴리곤과 교차하는 격자만 골라 격자 x 연도 침수 여부 표를 만든다."""
    floods = gpd.read_file(FLOOD_PATH)[["YEAR", "geometry"]]
    floods["YEAR"] = floods["YEAR"].astype(int)

    # 격자를 셀 크기에 맞춰 정렬 (build_grid.py와 같은 방식)
    minx, miny, maxx, maxy = floods.total_bounds
    minx = np.floor(minx / CELL_SIZE_M) * CELL_SIZE_M
    miny = np.floor(miny / CELL_SIZE_M) * CELL_SIZE_M
    nx = int(np.ceil((maxx - minx) / CELL_SIZE_M))
    ny = int(np.ceil((maxy - miny) / CELL_SIZE_M))
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny))
    ix, iy = ix.ravel(), iy.ravel()
    x0, y0 = minx + ix * CELL_SIZE_M, miny + iy * CELL_SIZE_M
    grid = gpd.GeoDataFrame(
        {"ix": ix, "iy": iy},
        geometry=shapely.box(x0, y0, x0 + CELL_SIZE_M, y0 + CELL_SIZE_M),
        crs=SOURCE_CRS,
    )
    print(f"  격자 생성: {len(grid):,}개 셀 (cell={CELL_SIZE_M}m)")

    joined = gpd.sjoin(grid, floods, predicate="intersects", how="inner")
    flooded = joined.reset_index()[["index", "YEAR"]].drop_duplicates()

    history = pd.crosstab(flooded["index"], flooded["YEAR"]).clip(upper=1)
    history = history.reindex(columns=YEARS, fill_value=0).astype(float)
    history[NO_DATA_YEARS] = np.nan

    cells = grid.loc[history.index].copy()
    cells.index.name = "cell_id"
    history.index.name = "cell_id"
    print(f"  침수 이력 있는 격자: {len(cells):,}개 (나머지 {len(grid) - len(cells):,}개는 제외)")
    return cells, history


def mean_signature(history, year_pcs):
    """잠긴 해들의 PC1~PC4 평균. 자료 없는 해(NaN)는 잠기지 않은 해처럼 빠진다."""
    flags = history.fillna(0).values
    n_years = flags.sum(axis=1)
    sig = flags @ year_pcs.values / n_years[:, None]
    return pd.DataFrame(sig, index=history.index, columns=PC_COLS), n_years.astype(int)


def attach_noise(xy, labels):
    """노이즈 격자를 가장 가까운 클러스터 격자의 라벨로 부착. 붙은 거리(m)도 함께 반환."""
    labels = labels.copy()
    assign = np.where(labels >= 0, "hdbscan", "nearest")
    dist = np.zeros(len(labels))

    noise = np.where(labels < 0)[0]
    if len(noise):
        known = np.where(labels >= 0)[0]
        d, idx = BallTree(xy[known]).query(xy[noise], k=1)
        labels[noise] = labels[known[idx[:, 0]]]
        dist[noise] = d[:, 0]
    return labels, assign, dist


def cluster_cells(xy, history, year_pcs):
    """서명 -> 표준화 -> HDBSCAN -> 노이즈 부착. history에 있는 연도 열만 쓴다 (교차검증 재클러스터링용).

    xy: history와 같은 순서의 격자 중심점 좌표. 모든 격자는 history 연도 중 한 번 이상 잠겨 있어야 한다.
    """
    sig, n_years = mean_signature(history, year_pcs.loc[history.columns])
    X = StandardScaler().fit_transform(sig.values)
    hdb_labels = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES, copy=True).fit_predict(X)
    labels, assign, dist = attach_noise(xy, hdb_labels)
    return sig, n_years, hdb_labels, labels, assign, dist


def year_combo(row):
    return "+".join(str(y) for y in YEARS if row[y] == 1)


def summarize(df, label_col):
    rows = []
    for lab, g in df.groupby(label_col):
        combos = g["combo"].value_counts().head(3)
        rows.append({
            "cluster": lab,
            "size": len(g),
            "n_years_mean": round(g["n_years"].mean(), 2),
            "n_years_min": g["n_years"].min(),
            "n_years_max": g["n_years"].max(),
            "top_combos": ", ".join(f"{c}({n})" for c, n in combos.items()),
            **{c: round(g[c].mean(), 3) for c in PC_COLS},
        })
    return pd.DataFrame(rows)


def save_map(cells_4326, label_col, assign_col, path):
    center = cells_4326.geometry.union_all().centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=11, tiles="cartodbpositron")
    data = cells_4326[["geometry", "combo", "n_years", "hdbscan_label", "attach_dist_m", label_col, assign_col]].copy()
    data = data.rename(columns={label_col: "cluster", assign_col: "assign"})

    def style(feature):
        p = feature["properties"]
        return {
            "fillColor": PALETTE[p["cluster"] % len(PALETTE)],
            "color": "#000000" if p["assign"] != "hdbscan" else PALETTE[p["cluster"] % len(PALETTE)],
            "weight": 1.2 if p["assign"] != "hdbscan" else 0.3,
            "fillOpacity": 0.7,
        }

    folium.GeoJson(
        json.loads(data.to_json()),
        style_function=style,
        tooltip=folium.GeoJsonTooltip(
            fields=["cluster", "assign", "hdbscan_label", "attach_dist_m", "combo", "n_years"],
            aliases=["클러스터", "배정 방식", "HDBSCAN 라벨", "부착 거리(m)", "침수 연도", "침수 연도 수"],
        ),
    ).add_to(m)
    m.save(path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1/6] 연도별 PC 로드")
    year_pcs = load_year_pcs()

    print("\n[2/6] 격자별 침수 이력")
    cells, history = build_flood_history()
    print("  연도별 침수 격자 수:")
    print(history.sum().astype(int).to_string())
    history.to_csv(os.path.join(OUT_DIR, "grid_flood_history.csv"), encoding="utf-8-sig")

    df = cells.copy()
    df["x"] = df.geometry.centroid.x
    df["y"] = df.geometry.centroid.y

    print("\n[3/6] 평균 서명 + 표준화")
    print(f"[4/6] HDBSCAN (min_cluster_size={MIN_CLUSTER_SIZE}, min_samples={MIN_SAMPLES})")
    sig, n_years, hdb_labels, labels, assign, dist = cluster_cells(df[["x", "y"]].values, history, year_pcs)
    print(f"  서로 다른 서명(침수 연도 조합) 수: {len(np.unique(sig.values.round(10), axis=0))}")
    n_noise = int((hdb_labels < 0).sum())
    print(f"  클러스터 {hdb_labels.max() + 1}개, 노이즈 {n_noise:,}개 ({n_noise / len(hdb_labels):.1%})")

    df["n_years"] = n_years
    df["combo"] = history.apply(year_combo, axis=1)
    df[PC_COLS] = sig
    df["hdbscan_label"] = hdb_labels

    print("\n[5/6] 노이즈 지리적 처리 (최근접 클러스터 격자에 부착)")
    df["cluster"] = labels
    df["assign"] = assign
    df["attach_dist_m"] = dist.round(1)
    noise_dist = pd.Series(dist[hdb_labels < 0])
    print(f"  노이즈 {n_noise:,}개 부착, 거리(m) 분포:")
    print(noise_dist.describe(percentiles=[0.5, 0.9, 0.99]).round(1).to_string())

    print("\n[6/6] 점검 및 저장")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    hdb_summary = summarize(df, "hdbscan_label")
    print("  HDBSCAN 결과 (노이즈 = -1):")
    print(hdb_summary.to_string(index=False))
    hdb_summary.to_csv(os.path.join(OUT_DIR, "cluster_summary_hdbscan.csv"), index=False, encoding="utf-8-sig")

    summary = summarize(df, "cluster")
    print("\n  노이즈 부착 후:")
    print(summary[["cluster", "size", "n_years_mean", "n_years_min", "n_years_max"]].to_string(index=False))
    summary.to_csv(os.path.join(OUT_DIR, "cluster_summary.csv"), index=False, encoding="utf-8-sig")

    cells_4326 = df.to_crs("EPSG:4326")
    save_map(cells_4326, "cluster", "assign", os.path.join(OUT_DIR, "cluster_map.html"))

    out_cols = ["ix", "iy", "x", "y", "n_years", "combo"] + PC_COLS + \
        ["hdbscan_label", "cluster", "assign", "attach_dist_m"]
    df[out_cols].to_csv(os.path.join(OUT_DIR, "grid_clusters.csv"), encoding="utf-8-sig")
    cells_4326[out_cols + ["geometry"]].to_file(os.path.join(OUT_DIR, "grid_clusters.geojson"), driver="GeoJSON")
    print(f"\n저장 완료: {OUT_DIR}")


if __name__ == "__main__":
    main()
