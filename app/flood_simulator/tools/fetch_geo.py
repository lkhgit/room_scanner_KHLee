"""
강우 지도에 필요한 관측소 좌표와 서울 구 경계를 내려받아 flood_risk_ML/data/에 저장한다.
Rainfall clustering.py의 STATION_COORDS, SEOUL_GU_PATH와 같은 경로·형식이다.

출처
  관측소 좌표: 기상청 국가기후데이터센터 AWS 관측지점정보 (2026-10-22 기준 유효 이력)
              https://data.kma.go.kr/tmeta/stn/selectStnList.do?pgmNo=123
              정리본: https://github.com/gud1396-coder/weather-station-explorer (data/aws.json)
  서울 구 경계: 통계청 2013 행정구역 (southkorea/seoul-maps, kostat/2013 simple)
              https://github.com/southkorea/seoul-maps

출력 (flood_risk_ML/data/)
  station_coords.csv   name, stn, lat, lon  (name = region_assignment.csv의 지점명)
  seoul_gu.geojson     서울 25개 구 폴리곤 (WGS84)

실행:
    py "app\\flood_simulator\\tools\\fetch_geo.py"     (저장소 최상위 폴더에서)
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
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
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
