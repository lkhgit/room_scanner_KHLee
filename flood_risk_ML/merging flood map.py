"""
Brings the Seoul flood trace maps (one shapefile per year, 2010-2025) to a common schema and merges them into a
single shapefile (merged flood map/서울시_침수흔적도_2010_2025.shp).

- YEAR comes from the folder year (the source F_YR has some typos and is kept for reference only).
- GU_NAM is filled in again from the first 5 digits of ADM_CD (the district code) so the spelling is consistent.
- For 2022 every polygon of the official Seoul layer is kept, and polygons of the disaster relief layer that overlap
  a Seoul polygon are removed.

Run:
    & "C:\\Users\\KHLee\\AppData\\Local\\Python\\bin\\python.exe" "merging flood map.py"
"""
import glob
import os
import re

import geopandas as gpd
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "flood map")
OUT_DIR = os.path.join(BASE_DIR, "merged flood map")
OUT_PATH = os.path.join(OUT_DIR, "서울시_침수흔적도_2010_2025.shp")
SOURCE_CRS = "EPSG:5179"

STR_COLS = [
    "F_YR", "INV_YR", "ADM_CD", "GU_NAM", "F_ZONE_NM", "F_DISA_NM", "F_RSN_DTL",
    "TYPE", "F_SAT_YMD", "F_END_YMD", "F_SAT_TM", "F_END_TM", "SRC",
]
NUM_COLS = ["F_SHIM", "F_AREA", "F_AVR_HGT"]
OUT_COLS = ["YEAR"] + NUM_COLS + STR_COLS

# Seoul district code (first 5 digits of ADM_CD) -> district name
GU_CODES = {
    "11110": "종로구", "11140": "중구", "11170": "용산구", "11200": "성동구",
    "11215": "광진구", "11230": "동대문구", "11260": "중랑구", "11290": "성북구",
    "11305": "강북구", "11320": "도봉구", "11350": "노원구", "11380": "은평구",
    "11410": "서대문구", "11440": "마포구", "11470": "양천구", "11500": "강서구",
    "11530": "구로구", "11545": "금천구", "11560": "영등포구", "11590": "동작구",
    "11620": "관악구", "11650": "서초구", "11680": "강남구", "11710": "송파구",
    "11740": "강동구",
}

SEOUL_LAYER_2022 = "서울시_2022(최종)"


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


def read_shp(shp_path):
    """Reads a shapefile; older files without a .cpg (2010-2020) are read as cp949."""
    has_cpg = os.path.exists(os.path.splitext(shp_path)[0] + ".cpg")
    gdf = gpd.read_file(shp_path) if has_cpg else gpd.read_file(shp_path, encoding="cp949")
    if gdf.crs is None:
        gdf = gdf.set_crs(SOURCE_CRS)
    elif gdf.crs.to_string() != SOURCE_CRS:
        gdf = gdf.to_crs(SOURCE_CRS)
    return gdf


def normalize_adm_cd(value):
    """Turns the many forms such as '1111017400' or 1.1110174e9 into one 10-digit string."""
    if pd.isna(value) or str(value).strip() == "":
        return None
    try:
        return str(int(float(value))).zfill(10)
    except ValueError:
        return str(value).strip()


def to_str(series):
    """Strips values and turns empty or missing ones into None."""
    return series.map(lambda v: None if pd.isna(v) or str(v).strip() == "" else str(v).strip())


def parse_2025_datetime(gdf):
    """'20250822 - 1031' -> (F_SAT_YMD='20250822', F_SAT_TM='10')."""
    parts = gdf["피해일시"].fillna("").str.split("-", n=1, expand=True)
    ymd = parts[0].str.strip()
    hhmm = parts[1].str.strip() if 1 in parts else pd.Series("", index=gdf.index)
    sat_ymd = ymd.where(ymd.str.fullmatch(r"\d{8}"), None)
    sat_tm = hhmm.map(
        lambda t: str(int(t.zfill(4)[:2])) if isinstance(t, str) and re.fullmatch(r"\d{3,4}", t) else None
    )
    return sat_ymd, sat_tm


def to_common_schema(year, gdf):
    """Brings one year's source data to the common schema (OUT_COLS)."""
    df = pd.DataFrame(index=gdf.index)
    df["YEAR"] = year
    for col in NUM_COLS:
        df[col] = pd.to_numeric(gdf[col], errors="coerce") if col in gdf else None
    for col in STR_COLS:
        df[col] = to_str(gdf[col]) if col in gdf else None

    if year == 2022:
        df["SRC"] = gdf["layer"].map(lambda l: "서울시" if l == SEOUL_LAYER_2022 else "재난지원금")
    elif "TYPE2" in gdf:  # 2025
        df["SRC"] = to_str(gdf["TYPE2"])
    else:
        df["SRC"] = "서울시"

    if "피해일시" in gdf:  # 2025
        df["F_SAT_YMD"], df["F_SAT_TM"] = parse_2025_datetime(gdf)

    for col in ("F_SAT_YMD", "F_END_YMD"):
        # '2022-08-08' -> '20220808', '20220808  2100' (date + time) -> '20220808'
        df[col] = df[col].map(lambda v: v.split()[0].replace("-", "") if isinstance(v, str) else v)

    df["ADM_CD"] = gdf["ADM_CD"].map(normalize_adm_cd)
    df["GU_NAM"] = df["ADM_CD"].map(lambda c: GU_CODES.get(c[:5]) if c else None).fillna(df["GU_NAM"])

    out = gpd.GeoDataFrame(df[OUT_COLS], geometry=gdf.geometry.values, crs=SOURCE_CRS)

    # clean up invalid/empty geometries
    out = out[~out.geometry.is_empty & out.geometry.notna()].copy()
    invalid = ~out.geometry.is_valid
    if invalid.any():
        out.loc[invalid, "geometry"] = out.loc[invalid, "geometry"].buffer(0)
    out = out[~out.geometry.is_empty & out.geometry.notna() & out.geometry.is_valid].copy()

    print(f"  {year}: {len(gdf):,}건 로드 -> 정리 후 {len(out):,}건 (invalid 수정 {int(invalid.sum())}건)")
    return out


def drop_2022_overlaps(gdf):
    """Removes the 2022 disaster relief polygons whose area overlaps a Seoul polygon."""
    seoul = gdf[gdf["SRC"] == "서울시"]
    disaster = gdf[gdf["SRC"] == "재난지원금"]

    pairs = gpd.sjoin(disaster[["geometry"]], seoul[["geometry"]], predicate="intersects")
    left = disaster.geometry.loc[pairs.index].values
    right = seoul.geometry.loc[pairs["index_right"]].values
    overlap_area = gpd.GeoSeries(left, crs=SOURCE_CRS).intersection(
        gpd.GeoSeries(right, crs=SOURCE_CRS)
    ).area.values
    overlapping = pairs.index[overlap_area > 0].unique()

    kept = disaster.drop(index=overlapping)
    print(f"  2022 중복 제거: 재난지원금 {len(disaster):,}건 중 {len(overlapping):,}건 제거 "
          f"-> {len(kept):,}건 유지 (서울시 {len(seoul):,}건은 모두 유지)")
    return pd.concat([seoul, kept])


def main():
    """Loads every year's shapefile, brings it to the common schema, merges them and saves the result."""
    year_shp_pairs = find_year_shapefiles()
    print(f"연도별 shapefile {len(year_shp_pairs)}개 발견: {[y for y, _ in year_shp_pairs]}")

    print("\n[1/3] 연도별 로드 및 공통 스키마 변환")
    year_gdfs = []
    for year, shp_path in year_shp_pairs:
        gdf = to_common_schema(year, read_shp(shp_path))
        if year == 2022:
            gdf = drop_2022_overlaps(gdf)
        year_gdfs.append(gdf)

    print("\n[2/3] 병합")
    merged = gpd.GeoDataFrame(
        pd.concat(year_gdfs, ignore_index=True), geometry="geometry", crs=SOURCE_CRS
    )
    merged = merged.sort_values("YEAR", kind="stable").reset_index(drop=True)
    print(f"전체: {len(merged):,}건")
    print(merged.groupby(["YEAR", "SRC"]).size().to_string())

    print("\n[3/3] 저장")
    os.makedirs(OUT_DIR, exist_ok=True)
    merged.to_file(OUT_PATH, encoding="UTF-8")
    print(f"저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    main()
