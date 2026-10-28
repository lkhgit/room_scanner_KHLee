"""
서울시 침수흔적도(연도별 shapefile)를 읽어 100m 격자 단위 침수 빈도/침수심을 계산하고
data/flood_grid.geojson 캐시 파일로 저장하는 전처리 스크립트.

실행:
    "C:\\Users\\KHLee\\AppData\\Local\\Python\\bin\\python.exe" build_grid.py
"""
import glob
import json
import os
import re

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE_DIR, "data", "flood_grid.geojson")
SOURCE_CRS = "EPSG:5179"  # 중부원점 TM, 원본 shapefile 좌표계
CELL_SIZE_M = 100  # 격자 한 변 길이(미터). 필요하면 조정.


def find_year_shapefiles():
    """연도 폴더(YYYY년 침수흔적도)를 순회하며 (year, shp_path) 목록을 반환."""
    pairs = []
    for entry in sorted(os.listdir(BASE_DIR)):
        m = re.match(r"^(\d{4})년\s*침수흔적도$", entry)
        if not m:
            continue
        folder = os.path.join(BASE_DIR, entry)
        if not os.path.isdir(folder):
            continue
        shp_list = glob.glob(os.path.join(folder, "*.shp"))
        if not shp_list:
            print(f"[경고] {entry}: shp 파일 없음, 건너뜀")
            continue
        pairs.append((int(m.group(1)), shp_list[0]))
    return pairs


def load_year(year, shp_path):
    """한 연도 shapefile을 읽어 geometry / F_SHIM / year 만 남긴 GeoDataFrame 반환."""
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(SOURCE_CRS)
    elif gdf.crs.to_string() != SOURCE_CRS:
        gdf = gdf.to_crs(SOURCE_CRS)

    depth_col = None
    for cand in ("F_SHIM", "f_shim"):
        if cand in gdf.columns:
            depth_col = cand
            break

    out = gpd.GeoDataFrame(
        {
            "year": year,
            "depth": pd.to_numeric(gdf[depth_col], errors="coerce") if depth_col else np.nan,
        },
        geometry=gdf.geometry.values,
        crs=SOURCE_CRS,
    )

    # 유효하지 않은/빈 geometry 정리
    out = out[~out.geometry.is_empty & out.geometry.notna()].copy()
    invalid = ~out.geometry.is_valid
    if invalid.any():
        out.loc[invalid, "geometry"] = out.loc[invalid, "geometry"].apply(
            lambda g: g.buffer(0)
        )
    out = out[~out.geometry.is_empty & out.geometry.notna()].copy()
    out = out[out.geometry.is_valid].copy()

    print(f"  {year}: {len(gdf)}건 로드 -> 정리 후 {len(out)}건")
    return out


def build_grid(bounds, cell_size):
    minx, miny, maxx, maxy = bounds
    # 격자를 셀 크기에 맞춰 정렬
    minx = np.floor(minx / cell_size) * cell_size
    miny = np.floor(miny / cell_size) * cell_size
    xs = np.arange(minx, maxx + cell_size, cell_size)
    ys = np.arange(miny, maxy + cell_size, cell_size)

    cells = []
    cell_ids = []
    cid = 0
    for x in xs[:-1]:
        for y in ys[:-1]:
            cells.append(box(x, y, x + cell_size, y + cell_size))
            cell_ids.append(cid)
            cid += 1

    grid = gpd.GeoDataFrame({"cell_id": cell_ids}, geometry=cells, crs=SOURCE_CRS)
    print(f"격자 생성: {len(grid):,}개 셀 (cell={cell_size}m)")
    return grid


def main():
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

    year_shp_pairs = find_year_shapefiles()
    print(f"연도별 shapefile {len(year_shp_pairs)}개 발견: "
          f"{[y for y, _ in year_shp_pairs]}")

    print("\n[1/5] 연도별 shapefile 로드 및 정리")
    year_gdfs = []
    for year, shp_path in year_shp_pairs:
        year_gdfs.append(load_year(year, shp_path))
    floods = pd.concat(year_gdfs, ignore_index=True)
    floods = gpd.GeoDataFrame(floods, geometry="geometry", crs=SOURCE_CRS)
    print(f"전체 침수 폴리곤: {len(floods):,}건")

    print("\n[2/5] 격자 생성")
    grid = build_grid(floods.total_bounds, CELL_SIZE_M)

    print("\n[3/5] 격자-침수폴리곤 공간 조인")
    joined = gpd.sjoin(grid, floods, predicate="intersects", how="inner")
    print(f"교차 매칭: {len(joined):,}건")

    print("\n[4/5] 연도 내 중복(이중 출처) 축약 후 셀별 집계")
    # 같은 (cell_id, year) 안에서는 최댓값 침수심으로 축약 (2022/2025 이중 출처 대응)
    per_cell_year = (
        joined.groupby(["cell_id", "year"], as_index=False)["depth"].max()
    )

    def agg_cell(g):
        years = sorted(g["year"].tolist())
        depths = g.set_index("year")["depth"]
        avg_depth = float(depths.mean(skipna=True)) if depths.notna().any() else None
        max_depth = float(depths.max(skipna=True)) if depths.notna().any() else None
        freq = len(years)
        weighted_score = freq * avg_depth if avg_depth is not None else float(freq)
        years_detail = [
            {"year": int(y), "depth": None if pd.isna(d) else round(float(d), 2)}
            for y, d in depths.items()
        ]
        return pd.Series(
            {
                "freq": freq,
                "avg_depth": None if avg_depth is None else round(avg_depth, 3),
                "max_depth": None if max_depth is None else round(max_depth, 3),
                "weighted_score": round(weighted_score, 3),
                "years_detail": json.dumps(years_detail, ensure_ascii=False),
            }
        )

    stats = per_cell_year.groupby("cell_id").apply(agg_cell, include_groups=False)
    stats = stats.reset_index()

    result = grid.merge(stats, on="cell_id", how="inner")
    print(f"침수 이력 있는 셀: {len(result):,}개 "
          f"(빈도 최댓값={result['freq'].max()}, "
          f"weighted_score 최댓값={result['weighted_score'].max():.2f})")

    print("\n[5/5] EPSG:4326 변환 및 저장")
    result = result.to_crs("EPSG:4326")
    result.to_file(OUT_PATH, driver="GeoJSON")
    print(f"저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    main()
