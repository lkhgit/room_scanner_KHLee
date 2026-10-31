"""
Rainfall zonation of Seoul's AWS stations + charts

Follows the document 'Seoul rainfall zonation code plan'.
  1. Load and clean          load_hourly
  2. Station zonation        station_correlation, cluster_stations
  3. Station × year summary  station_year_summary
  4. Region × year features  region_year_features
  5. Validation              validate
  6. Charts                  plot_regions

Required packages: pandas, numpy, scipy, matplotlib
Optional packages: pyarrow (parquet output), geopandas + shapely (region map), plotly (Sankey diagram)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

BASE = Path(__file__).resolve().parent

# ──────────────────────────────────────────────
# settings (no numbers written directly in the code)
# ──────────────────────────────────────────────
CFG = {
    "CSV_PATH": BASE / "merged data" / "강우량_2009_2026.csv",
    "OUT_DIR": BASE / "Rainfall zonation results",
    "EXCLUDE_STATIONS": ["북한산", "북악산"],
    "YEARS": list(range(2010, 2027)),
    "TRAIN_YEARS": list(range(2010, 2026)),
    "SEASON_MONTHS": [6, 7, 8, 9],
    "WET_HOUR_MM": 1.0,
    "WET_HOUR_MM_STRICT": 10.0,          # for the threshold sensitivity check
    "MIN_OVERLAP_H": 500,
    "LINKAGE": "average",
    "K_LIST": [3, 4, 5, 6],
    "ELBOW_K_RANGE": (1, 12),            # range of k shown in the elbow chart
    "MIN_SEASON_COVERAGE": 0.9,
    "MAX_MISSING_H_24": 2,
    "WINDOW_H": 24,
    "PERIODS": [(2010, 2017), (2018, 2026)],  # for the period stability check
    "KNOWN_STORM": {"year": 2022, "stations": ["강남", "서초"]},
    # for the map (only the map is skipped if missing)
    # station coordinates CSV: columns = name, lat, lon  (name matching the station names in the rainfall CSV)
    "STATION_COORDS": BASE / "data" / "station_coords.csv",
    # Seoul district borders, shp/geojson
    "SEOUL_GU_PATH": BASE / "data" / "seoul_gu.geojson",
    "CRS": "EPSG:5179",
}

REGION_COLORS = ["#2a78b5", "#e8833a", "#3a9e5a", "#c94a4a", "#8a63b8", "#a0785a"]

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# ──────────────────────────────────────────────
# 1. load and clean
# ──────────────────────────────────────────────
def load_hourly(csv_path, cfg):
    """Hour × station table. Missing hours are NaN."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = ["stn", "name", "t", "rain"]
    df["t"] = pd.to_datetime(df["t"], format="%Y-%m-%d %H:%M")
    df["name"] = df["name"].astype(str).str.strip()

    hourly = df.pivot_table(index="t", columns="name", values="rain", aggfunc="first")

    # put it on a gap-free hourly axis (missing hours = NaN, not filled with 0)
    full_index = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    hourly = hourly.reindex(full_index)
    hourly.index.name = "t"

    drop = [c for c in hourly.columns if any(ex in c for ex in cfg["EXCLUDE_STATIONS"])]
    hourly = hourly.drop(columns=drop)
    hourly = hourly[hourly.index.year.isin(cfg["YEARS"])]

    stn_map = df[["stn", "name"]].drop_duplicates().set_index("name")["stn"]
    print(f"[load] {hourly.shape[0]:,}시간 × {hourly.shape[1]}개 관측소 (제외: {drop})")
    return hourly, stn_map


def save_table(df, path):
    """Saves a table as parquet, or as gzipped CSV if pyarrow is missing."""
    try:
        df.to_parquet(path)
    except ImportError:
        path = path.with_suffix(".csv.gz")
        df.to_csv(path, encoding="utf-8-sig")
    print(f"[save] {path.name}")


# ──────────────────────────────────────────────
# 2. station zonation
# ──────────────────────────────────────────────
def station_correlation(hourly, cfg, wet_mm=None):
    """Pearson correlation between station pairs, using only the summer hours with rain."""
    wet_mm = cfg["WET_HOUR_MM"] if wet_mm is None else wet_mm
    season = hourly[hourly.index.month.isin(cfg["SEASON_MONTHS"])]
    wet = season[(season >= wet_mm).any(axis=1)]
    corr = wet.corr(method="pearson", min_periods=cfg["MIN_OVERLAP_H"])
    return corr


def _relabel_by_leaf_order(labels, leaves):
    """Renumbers the regions 1, 2, 3... from the left of the dendrogram."""
    order = []
    for i in leaves:
        if labels[i] not in order:
            order.append(labels[i])
    remap = {old: new for new, old in enumerate(order, start=1)}
    return np.array([remap[l] for l in labels])


def distance_matrix(corr):
    """Distance = 1 − correlation (square matrix)."""
    dist = 1.0 - corr.values
    dist = np.where(np.isnan(dist), 1.0, dist)   # pairs without a correlation count as uncorrelated
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0.0)
    return np.clip(dist, 0.0, None)


def cluster_stations(corr, cfg):
    """Hierarchical clustering on distance = 1 − correlation. Returns the linkage matrix and the region assignment for each k."""
    stations = list(corr.columns)
    dist = distance_matrix(corr)

    Z = linkage(squareform(dist, checks=False), method=cfg["LINKAGE"])
    leaves = dendrogram(Z, no_plot=True)["leaves"]

    assignment = pd.DataFrame(index=pd.Index(stations, name="name"))
    for k in cfg["K_LIST"]:
        labels = fcluster(Z, t=k, criterion="maxclust")
        assignment[f"k{k}"] = _relabel_by_leaf_order(labels, leaves)
    return Z, assignment


# ──────────────────────────────────────────────
# 3. station × year summary
# ──────────────────────────────────────────────
def station_year_summary(hourly, cfg):
    """Annual maximum 1-hour / 24-hour rainfall and summer data coverage for every station and year."""
    win = cfg["WINDOW_H"]
    roll24 = hourly.rolling(win, min_periods=win - cfg["MAX_MISSING_H_24"]).sum()

    years = hourly.index.year
    season_mask = hourly.index.month.isin(cfg["SEASON_MONTHS"])

    max_1h = hourly.groupby(years).max()
    max_24h = roll24.groupby(roll24.index.year).max()
    coverage = hourly[season_mask].notna().groupby(years[season_mask]).mean()

    summary = pd.concat(
        {
            "max_1h": max_1h.stack(future_stack=True),
            "max_24h": max_24h.stack(future_stack=True),
            "season_coverage": coverage.stack(future_stack=True),
        },
        axis=1,
    )
    summary.index.names = ["year", "name"]
    summary = summary.reset_index()

    summary["season_coverage"] = summary["season_coverage"].fillna(0.0)
    summary["usable"] = summary["season_coverage"] >= cfg["MIN_SEASON_COVERAGE"]
    summary.loc[~summary["usable"], ["max_1h", "max_24h"]] = np.nan
    summary["train"] = summary["year"].isin(cfg["TRAIN_YEARS"])
    return summary[["name", "year", "max_1h", "max_24h", "season_coverage", "usable", "train"]]


# ──────────────────────────────────────────────
# 4. region × year features
# ──────────────────────────────────────────────
def region_year_features(summary, assignment, k):
    """Station features come first; the regional mean and maximum are taken last."""
    reg = assignment[f"k{k}"].rename("region")
    df = summary.merge(reg, left_on="name", right_index=True)

    g = df.groupby(["region", "year"])
    long = pd.DataFrame(
        {
            "max_1h_mean": g["max_1h"].mean(),
            "max_1h_max": g["max_1h"].max(),
            "max_24h_mean": g["max_24h"].mean(),
            "max_24h_max": g["max_24h"].max(),
            "n_stations": g["usable"].sum().astype(int),
            "n_members": g["name"].count(),
        }
    ).reset_index()
    long["train"] = long["year"].isin(CFG["TRAIN_YEARS"])

    # input for the flood model: one row per year, 2 columns per region
    wide = long.pivot(index="year", columns="region", values=["max_1h_mean", "max_24h_mean"])
    wide.columns = [f"R{r}_{'max1h' if c == 'max_1h_mean' else 'max24h'}" for c, r in wide.columns]
    wide = wide[sorted(wide.columns, key=lambda c: (int(c.split("_")[0][1:]), c))]
    return long, wide


# ──────────────────────────────────────────────
# 5. validation
# ──────────────────────────────────────────────
def adjusted_rand_index(a, b):
    """Adjusted Rand index between two labelings (1 = identical clusterings)."""
    ct = pd.crosstab(np.asarray(a), np.asarray(b)).values.astype(float)
    c2 = lambda x: x * (x - 1) / 2
    sum_ij = c2(ct).sum()
    sum_a = c2(ct.sum(axis=1)).sum()
    sum_b = c2(ct.sum(axis=0)).sum()
    expected = sum_a * sum_b / c2(ct.sum())
    max_idx = (sum_a + sum_b) / 2
    return 1.0 if max_idx == expected else (sum_ij - expected) / (max_idx - expected)


def _upper(values):
    """Values above the diagonal of a square matrix."""
    return values[np.triu_indices_from(values, k=1)]


def validate(hourly, corr, assignment, summary, region_tables, cfg, report_dir):
    """Runs the validation checks (cohesion, stability, sensitivity, border stations, known storm, missing data) and writes validation.txt."""
    lines = []
    global_median = np.nanmedian(_upper(corr.values))
    lines.append(f"전체 관측소 쌍 상관 중앙값: {global_median:.3f}\n")

    # cohesion within regions
    rows = []
    for k in cfg["K_LIST"]:
        for r, members in assignment.groupby(f"k{k}").groups.items():
            sub = corr.loc[members, members].values
            within = np.nanmean(_upper(sub)) if len(members) > 1 else np.nan
            rows.append({"k": k, "region": r, "n": len(members), "within_corr": within,
                         "pass": bool(len(members) == 1 or within > global_median),
                         "stations": ", ".join(members)})
    cohesion = pd.DataFrame(rows)
    cohesion.to_csv(report_dir / "cohesion.csv", index=False, encoding="utf-8-sig")
    lines.append("[권역 내 응집도] 모든 권역 내부 상관 > 전체 중앙값")
    for k, g in cohesion.groupby("k"):
        lines.append(f"  k={k}: {'통과' if g['pass'].all() else '실패'}  "
                     f"(내부 상관 {g['within_corr'].min():.3f} ~ {g['within_corr'].max():.3f})")

    # period stability
    lines.append("\n[기간 안정성] ARI (앞 기간 vs 뒤 기간)")
    period_assign = []
    for y0, y1 in cfg["PERIODS"]:
        sub = hourly[(hourly.index.year >= y0) & (hourly.index.year <= y1)]
        _, a = cluster_stations(station_correlation(sub, cfg), cfg)
        period_assign.append(a)
    for k in cfg["K_LIST"]:
        a, b = period_assign[0][f"k{k}"], period_assign[1][f"k{k}"].reindex(period_assign[0].index)
        lines.append(f"  k={k}: ARI = {adjusted_rand_index(a, b):.3f}")

    # threshold sensitivity
    lines.append(f"\n[기준 민감도] ARI ({cfg['WET_HOUR_MM']}mm 기준 vs {cfg['WET_HOUR_MM_STRICT']}mm 기준)")
    _, strict = cluster_stations(station_correlation(hourly, cfg, cfg["WET_HOUR_MM_STRICT"]), cfg)
    for k in cfg["K_LIST"]:
        lines.append(f"  k={k}: ARI = {adjusted_rand_index(assignment[f'k{k}'], strict[f'k{k}']):.3f}")

    # border stations: stations whose mean correlation with another region is higher than with their own
    # (hierarchical clusters only split as k grows, so this is recorded instead of stations that 'move')
    lines.append("\n[경계 관측소] 다른 권역과의 평균 상관이 자기 권역보다 높은 관측소")
    for k in cfg["K_LIST"]:
        lab = assignment[f"k{k}"]
        flagged = []
        for s in corr.index:
            means = {r: corr.loc[s, lab.index[(lab == r) & (lab.index != s)]].mean()
                     for r in lab.unique()}
            own = means[lab[s]]
            best_r = max(means, key=lambda r: -np.inf if np.isnan(means[r]) else means[r])
            if best_r != lab[s] and not np.isnan(own):
                flagged.append(f"{s}(R{lab[s]}→R{best_r})")
        lines.append(f"  k={k}: {', '.join(flagged) if flagged else '없음'}")

    # known storm
    ks = cfg["KNOWN_STORM"]
    lines.append(f"\n[알려진 폭우] {ks['year']}년, {'·'.join(ks['stations'])} 권역의 연 최대 1시간 강우 순위")
    for k in cfg["K_LIST"]:
        lab = assignment[f"k{k}"]
        targets = {lab[s] for s in lab.index if any(t in s for t in ks["stations"])}
        long = region_tables[k][0]
        yr = long[long["year"] == ks["year"]].sort_values("max_1h_mean", ascending=False)
        top = yr.iloc[0]["region"] if len(yr) else None
        lines.append(f"  k={k}: 대상 권역 {sorted(f'R{t}' for t in targets)}, 1위 R{top} "
                     f"→ {'통과' if top in targets else '실패'}")

    # missing data handling
    lines.append("\n[결측 처리]")
    leaked = summary[~summary["usable"] & summary[["max_1h", "max_24h"]].notna().any(axis=1)]
    lines.append(f"  관측 비율 미달인데 값이 들어간 관측소-연도: {len(leaked)}건 (0이어야 함)")
    unusable = summary[~summary["usable"]].groupby("year")["name"].apply(", ".join)
    for y, names in unusable.items():
        lines.append(f"  {y}: NaN 처리 - {names}")
    for k in cfg["K_LIST"]:
        long = region_tables[k][0]
        short = long[long["n_stations"] < long["n_members"]]
        lines.append(f"  k={k}: 사용 관측소 수가 줄어든 권역-연도 {len(short)}건, "
                     f"유효 관측소 0개 {int((long['n_stations'] == 0).sum())}건")

    text = "\n".join(lines)
    (report_dir / "validation.txt").write_text(text, encoding="utf-8")
    print(text)


# ──────────────────────────────────────────────
# 6. charts
# ──────────────────────────────────────────────
def _cut_height(Z, k):
    """Height at which to cut for k regions (halfway between two merge heights)."""
    n = Z.shape[0] + 1
    lo = Z[n - k - 1, 2]
    hi = Z[n - k, 2]
    return (lo + hi) / 2


def _find_elbow(ks, values):
    """The elbow = the point farthest from the straight line joining both ends."""
    x = (ks - ks[0]) / (ks[-1] - ks[0])
    y = (values - values[-1]) / (values[0] - values[-1])
    gap = (1 - x) - y   # how far below the line y = 1 − x
    return int(ks[np.argmax(gap)])


def plot_elbow(Z, corr, cfg, report_dir):
    """Finds the elbow from the within-region sum of squared distances (W) and the merge heights for each k, and charts both."""
    dist = distance_matrix(corr)
    n = dist.shape[0]
    k_lo, k_hi = cfg["ELBOW_K_RANGE"]
    ks = np.arange(k_lo, min(k_hi, n) + 1)

    # W(k) = Σ_regions (1 / 2n_c) Σ_{i,j∈c} d_ij²  (same definition as the k-means SSE)
    wss = []
    for k in ks:
        labels = fcluster(Z, t=k, criterion="maxclust")
        w = 0.0
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            w += (dist[np.ix_(idx, idx)] ** 2).sum() / (2 * len(idx))
        wss.append(w)
    wss = np.array(wss)

    # merge height when going from k to k−1 regions (larger = a forced merge)
    heights = np.array([Z[n - k, 2] if k >= 2 else np.nan for k in ks])

    elbow_k = _find_elbow(ks, wss)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    ax.plot(ks, wss, "o-", color="#2a78b5")
    ax.plot(elbow_k, wss[ks == elbow_k], "o", ms=13, mfc="none", mec="#c94a4a", mew=2)
    ax.annotate(f"팔꿈치 k={elbow_k}", (elbow_k, wss[ks == elbow_k][0]), xytext=(12, 12),
                textcoords="offset points", color="#c94a4a")
    ax.set_xlabel("권역 수 k")
    ax.set_ylabel("권역 내 거리 제곱합 W(k)")
    ax.set_title("팔꿈치 그림: 권역 내 흩어짐")

    ax = axes[1]
    ax.bar(ks[1:], heights[1:], color="#8a8a8a")
    ax.set_xlabel("권역 수 k")
    ax.set_ylabel("k개 → k-1개 병합 높이 (1 - 상관)")
    ax.set_title("병합 높이: 크게 뛰는 곳 직전이 자연스러운 k")

    for ax in axes:
        ax.set_xticks(ks)
        ax.spines[["top", "right"]].set_visible(False)
        for k in cfg["K_LIST"]:
            ax.axvspan(k - 0.4, k + 0.4, color="#e8833a", alpha=0.08, lw=0)
    fig.tight_layout()
    fig.savefig(report_dir / "0_elbow.png", dpi=150)
    plt.close(fig)

    table = pd.DataFrame({"k": ks, "wss": wss, "merge_height": heights})
    table.to_csv(report_dir / "elbow.csv", index=False, encoding="utf-8-sig")
    print(f"\n[elbow] 팔꿈치 k = {elbow_k}")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return elbow_k


def plot_dendrogram(Z, corr, cfg, report_dir):
    """Draws the station dendrogram with the cut height for each k."""
    fig, ax = plt.subplots(figsize=(13, 6))
    dendrogram(Z, labels=list(corr.columns), ax=ax, leaf_rotation=90,
               color_threshold=0, above_threshold_color="#555555")
    for k in cfg["K_LIST"]:
        h = _cut_height(Z, k)
        ax.axhline(h, ls="--", lw=1, color="#c94a4a", alpha=0.7)
        ax.text(ax.get_xlim()[1], h, f" k={k}", va="center", fontsize=9, color="#c94a4a")
    ax.set_ylabel("거리 (1 - 상관)")
    ax.set_title("관측소 계층적 군집 덴드로그램 (6–9월, 비 온 시간)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(report_dir / "2_dendrogram.png", dpi=150)
    plt.close(fig)


def plot_sorted_heatmap(Z, corr, assignment, cfg, report_dir):
    """Draws the station correlation matrix in dendrogram order, with the regions outlined, for each k."""
    leaves = dendrogram(Z, no_plot=True)["leaves"]
    order = [corr.columns[i] for i in leaves]
    mat = corr.loc[order, order].values
    n = len(order)

    fig, axes = plt.subplots(2, 2, figsize=(16, 15))
    for ax, k in zip(axes.flat, cfg["K_LIST"]):
        im = ax.imshow(mat, cmap="Blues", vmin=np.nanmin(mat), vmax=1)
        ax.set_xticks(range(n), order, rotation=90, fontsize=7)
        ax.set_yticks(range(n), order, fontsize=7)
        labs = assignment.loc[order, f"k{k}"].values
        start = 0
        for i in range(1, n + 1):
            if i == n or labs[i] != labs[start]:
                ax.add_patch(Rectangle((start - 0.5, start - 0.5), i - start, i - start,
                                       fill=False, ec=REGION_COLORS[labs[start] - 1], lw=2.5))
                start = i
        ax.set_title(f"k = {k}")
    fig.colorbar(im, ax=axes, shrink=0.6, label="피어슨 상관")
    fig.suptitle("덴드로그램 순서로 정렬한 관측소 상관 (사각형 = 권역)", fontsize=14)
    fig.savefig(report_dir / "3_corr_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_region_map(assignment, cfg, report_dir):
    """Draws a map of the regions as Voronoi cells around the stations, clipped to Seoul."""
    coords_path, gu_path = cfg["STATION_COORDS"], cfg["SEOUL_GU_PATH"]
    if not coords_path.exists():
        print(f"[map] 관측소 좌표 파일이 없어 지도를 건너뜀: {coords_path}")
        return
    try:
        import geopandas as gpd
        from shapely.geometry import MultiPoint
        from shapely.ops import voronoi_diagram
    except ImportError:
        print("[map] geopandas/shapely가 없어 지도를 건너뜀")
        return

    xy = pd.read_csv(coords_path, encoding="utf-8-sig")
    xy["name"] = xy["name"].astype(str).str.strip()
    xy = xy[xy["name"].isin(assignment.index)]
    pts = gpd.GeoDataFrame(xy, geometry=gpd.points_from_xy(xy["lon"], xy["lat"]),
                           crs="EPSG:4326").to_crs(cfg["CRS"])
    missing = set(assignment.index) - set(pts["name"])
    if missing:
        print(f"[map] 좌표가 없는 관측소: {sorted(missing)}")

    gu = None
    if gu_path.exists():
        gu = gpd.read_file(gu_path).to_crs(cfg["CRS"])
        boundary = gu.union_all() if hasattr(gu, "union_all") else gu.unary_union
    else:
        print(f"[map] 구 경계 파일이 없어 관측소 볼록 껍질로 자름: {gu_path}")
        boundary = MultiPoint(list(pts.geometry)).convex_hull.buffer(2000)

    vor = voronoi_diagram(MultiPoint(list(pts.geometry)), envelope=boundary.envelope.buffer(10000))
    cells = gpd.GeoDataFrame(geometry=list(vor.geoms), crs=cfg["CRS"])
    cells = gpd.sjoin(cells, pts[["name", "geometry"]], predicate="contains").drop(columns="index_right")
    cells["geometry"] = cells.geometry.intersection(boundary)

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    for ax, k in zip(axes.flat, cfg["K_LIST"]):
        reg = cells["name"].map(assignment[f"k{k}"])
        cells.plot(ax=ax, color=[REGION_COLORS[r - 1] for r in reg], alpha=0.55,
                   edgecolor="white", linewidth=0.8)
        if gu is not None:
            gu.boundary.plot(ax=ax, color="#333333", linewidth=0.5)
        pts.plot(ax=ax, color="black", markersize=12)
        for _, p in pts.iterrows():
            ax.annotate(p["name"], (p.geometry.x, p.geometry.y), xytext=(3, 3),
                        textcoords="offset points", fontsize=7)
        handles = [Rectangle((0, 0), 1, 1, color=REGION_COLORS[r - 1], alpha=0.55)
                   for r in range(1, k + 1)]
        ax.legend(handles, [f"R{r}" for r in range(1, k + 1)], loc="lower left", fontsize=8)
        ax.set_title(f"k = {k}")
        ax.set_axis_off()
    fig.suptitle("강우 권역 지도 (관측소 보로노이 영역)", fontsize=14)
    fig.tight_layout()
    fig.savefig(report_dir / "1_region_map.png", dpi=150)
    plt.close(fig)


def plot_sankey(assignment, cfg, report_dir):
    """Draws a Sankey diagram of how the regions split as k grows."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[sankey] plotly가 없어 생키 그림을 건너뜀")
        return

    ks = cfg["K_LIST"]
    nodes, node_idx, colors = [], {}, []
    for k in ks:
        for r in sorted(assignment[f"k{k}"].unique()):
            node_idx[(k, r)] = len(nodes)
            nodes.append(f"k{k}·R{r}")
            colors.append(REGION_COLORS[r - 1])

    src, tgt, val, hover = [], [], [], []
    for k0, k1 in zip(ks[:-1], ks[1:]):
        grp = assignment.groupby([f"k{k0}", f"k{k1}"])
        for (r0, r1), members in grp.groups.items():
            src.append(node_idx[(k0, r0)])
            tgt.append(node_idx[(k1, r1)])
            val.append(len(members))
            hover.append(", ".join(members))

    fig = go.Figure(go.Sankey(
        node=dict(label=nodes, color=colors, pad=15, thickness=18),
        link=dict(source=src, target=tgt, value=val, customdata=hover,
                  hovertemplate="%{customdata}<extra></extra>"),
    ))
    fig.update_layout(title="k를 늘릴 때 권역이 쪼개지는 모습", font=dict(size=12))
    fig.write_html(report_dir / "4_k_sankey.html")


def plot_region_year(region_tables, cfg, report_dir):
    """Draws heat maps of the regional annual maximum rainfall per year for each k."""
    for k in cfg["K_LIST"]:
        long = region_tables[k][0]
        fig, axes = plt.subplots(2, 1, figsize=(14, 2.2 + 1.1 * k * 2), sharex=True)
        for ax, col, title, cmap in [
            (axes[0], "max_1h_mean", "연 최대 1시간 강우 (mm)", "Blues"),
            (axes[1], "max_24h_mean", "연 최대 24시간 강우 (mm)", "Purples"),
        ]:
            mat = long.pivot(index="region", columns="year", values=col)
            cnt = long.pivot(index="region", columns="year", values="n_stations")
            mem = long.pivot(index="region", columns="year", values="n_members")
            im = ax.imshow(mat.values, cmap=cmap, aspect="auto")
            vmid = np.nanmean(mat.values)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    v = mat.values[i, j]
                    txt = "NaN" if np.isnan(v) else f"{v:.0f}"
                    c = "white" if (not np.isnan(v) and v > vmid * 1.3) else "black"
                    ax.text(j, i - 0.12, txt, ha="center", va="center", fontsize=9, color=c)
                    ax.text(j, i + 0.25, f"{int(cnt.values[i, j])}/{int(mem.values[i, j])}",
                            ha="center", va="center", fontsize=6.5, color=c)
            ax.set_yticks(range(mat.shape[0]), [f"R{r}" for r in mat.index])
            ax.set_xticks(range(mat.shape[1]), mat.columns)
            ax.set_title(title)
            fig.colorbar(im, ax=ax, pad=0.01)
        fig.suptitle(f"권역 × 연도 피처 (k={k}, 작은 숫자 = 사용 관측소/소속 관측소)", fontsize=12)
        fig.tight_layout()
        fig.savefig(report_dir / f"5_region_year_k{k}.png", dpi=150)
        plt.close(fig)


def plot_regions(Z, corr, assignment, region_tables, cfg, report_dir):
    """Draws all charts."""
    plot_elbow(Z, corr, cfg, report_dir)
    plot_region_map(assignment, cfg, report_dir)
    plot_dendrogram(Z, corr, cfg, report_dir)
    plot_sorted_heatmap(Z, corr, assignment, cfg, report_dir)
    plot_sankey(assignment, cfg, report_dir)
    plot_region_year(region_tables, cfg, report_dir)
    print(f"[plot] 그림 저장 완료: {report_dir}")


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────
def main(cfg=CFG):
    """Loads the rainfall data, zones the stations, builds the region × year features, validates them and draws the charts."""
    out_dir = cfg["OUT_DIR"]
    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    hourly, stn_map = load_hourly(cfg["CSV_PATH"], cfg)
    save_table(hourly, out_dir / "rain_hourly_clean.parquet")

    corr = station_correlation(hourly, cfg)
    corr.to_csv(out_dir / "station_corr.csv", encoding="utf-8-sig")

    Z, assignment = cluster_stations(corr, cfg)
    assignment.insert(0, "stn", assignment.index.map(stn_map))
    assignment.to_csv(out_dir / "region_assignment.csv", encoding="utf-8-sig")
    assignment = assignment.drop(columns="stn")

    summary = station_year_summary(hourly, cfg)
    summary.to_csv(out_dir / "station_year_summary.csv", index=False, encoding="utf-8-sig")

    region_tables = {}
    for k in cfg["K_LIST"]:
        long, wide = region_year_features(summary, assignment, k)
        long.to_csv(out_dir / f"region_year_features_k{k}.csv", index=False, encoding="utf-8-sig")
        wide.to_csv(out_dir / f"region_year_wide_k{k}.csv", encoding="utf-8-sig")
        region_tables[k] = (long, wide)

    for k in cfg["K_LIST"]:
        print(f"\n[k={k}]")
        for r, members in assignment.groupby(f"k{k}").groups.items():
            print(f"  R{r} ({len(members)}): {', '.join(members)}")

    validate(hourly, corr, assignment, summary, region_tables, cfg, report_dir)
    plot_regions(Z, corr, assignment, region_tables, cfg, report_dir)


if __name__ == "__main__":
    main()
