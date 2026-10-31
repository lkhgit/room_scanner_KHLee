"""
Preprocessing script that reads the Seoul flood trace maps (one shapefile per year), computes the flood frequency
and depth on a 100 m grid and saves them as the cache file data/flood_grid.geojson.

Run:
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
SRC_DIR = os.path.join(BASE_DIR, "flood map")
OUT_PATH = os.path.join(BASE_DIR, "data", "flood_grid.geojson")
SOURCE_CRS = "EPSG:5179"  # Korea central belt TM, the coordinate system of the source shapefiles
CELL_SIZE_M = 100  # grid cell side length (metres). Adjust if needed.


def find_year_shapefiles():
    """Walks the year folders (YYYY년 침수흔적도) and returns a list of (year, shp_path)."""
    pairs = []
    for entry in sorted(os.listdir(SRC_DIR)):
        m = re.match(r"^(\d{4})년\s*침수흔적도$", entry)
        if not m:
            continue
        folder = os.path.join(SRC_DIR, entry)
        if not os.path.isdir(folder):
            continue
        shp_list = glob.glob(os.path.join(folder, "*.shp"))
        if not shp_list:
            print(f"[경고] {entry}: shp 파일 없음, 건너뜀")
            continue
        pairs.append((int(m.group(1)), shp_list[0]))
    return pairs


def load_year(year, shp_path):
    """Reads one year's shapefile and returns a GeoDataFrame with only geometry / F_SHIM / year."""
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

    # clean up invalid/empty geometries
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
    """Builds a square grid of cell_size cells covering the bounds, aligned to multiples of the cell size."""
    minx, miny, maxx, maxy = bounds
    # align the grid to the cell size
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
    """Loads every year's flood polygons, joins them to the grid, aggregates per cell and saves data/flood_grid.geojson."""
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
    # within one (cell_id, year), keep the deepest flood depth (handles the two sources in 2022/2025)
    per_cell_year = (
        joined.groupby(["cell_id", "year"], as_index=False)["depth"].max()
    )

    def agg_cell(g):
        """Summarises one cell: flood years, mean and max depth, frequency and the frequency × depth score."""
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
