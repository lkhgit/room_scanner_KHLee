"""
Downloads the station coordinates and Seoul district borders the rain map needs, into flood_risk_ML/data/.
Same paths and formats as STATION_COORDS and SEOUL_GU_PATH in Rainfall clustering.py.

Sources
  Station coordinates: KMA National Climate Data Center, AWS station information (records valid on 2026-10-22)
              https://data.kma.go.kr/tmeta/stn/selectStnList.do?pgmNo=123
              cleaned copy: https://github.com/gud1396-coder/weather-station-explorer (data/aws.json)
  Seoul district borders: Statistics Korea 2013 administrative areas (southkorea/seoul-maps, kostat/2013 simple)
              https://github.com/southkorea/seoul-maps

Output (flood_risk_ML/data/)
  station_coords.csv   name, stn, lat, lon  (name = station name in region_assignment.csv)
  seoul_gu.geojson     polygons of Seoul's 25 districts (WGS84)

Run:
    py "app\\flood_simulator\\tools\\fetch_geo.py"     (from the repository root)
"""
import csv
import json
import os
import urllib.request

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(APP_DIR)), "flood_risk_ML")
OUT_DIR = os.path.join(BASE_DIR, "data")
ASSIGN_PATH = os.path.join(BASE_DIR, "Rainfall zonation results", "region_assignment.csv")

AWS_URL = "https://raw.githubusercontent.com/gud1396-coder/weather-station-explorer/HEAD/data/aws.json"
GU_URL = ("https://raw.githubusercontent.com/southkorea/seoul-maps/master/"
          "kostat/2013/json/seoul_municipalities_geo_simple.json")


def fetch_json(url):
    """Downloads and parses a JSON file."""
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    """Writes station_coords.csv for the stations in region_assignment.csv and saves the Seoul district GeoJSON."""
    os.makedirs(OUT_DIR, exist_ok=True)
    aws = {s["id"]: s for s in fetch_json(AWS_URL)}
    with open(ASSIGN_PATH, encoding="utf-8-sig") as f:
        stations = list(csv.DictReader(f))

    missing = [s["name"] for s in stations if s["stn"] not in aws]
    if missing:
        raise SystemExit(f"좌표가 없는 관측소: {missing}")
    path = os.path.join(OUT_DIR, "station_coords.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "stn", "lat", "lon"])
        for s in stations:
            a = aws[s["stn"]]
            w.writerow([s["name"], s["stn"], a["latitude"], a["longitude"]])
    print(f"  station_coords.csv  관측소 {len(stations)}개")

    gu = fetch_json(GU_URL)
    with open(os.path.join(OUT_DIR, "seoul_gu.geojson"), "w", encoding="utf-8") as f:
        json.dump(gu, f, ensure_ascii=False)
    print(f"  seoul_gu.geojson    구 {len(gu['features'])}개")


if __name__ == "__main__":
    main()
