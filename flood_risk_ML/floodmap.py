"""
Seoul flood frequency heat map web app.

Reads data/flood_grid.geojson (made by build_grid.py) and renders it as a folium map.
Clicking a grid cell shows its flood frequency and depth history in a popup.

Run:
    & "C:\\Users\\KHLee\\AppData\\Local\\Python\\bin\\python.exe" floodmap.py
    http://127.0.0.1:5000
"""
import json
import os
import time

import branca.colormap as cm
import folium
import requests
from flask import Flask, jsonify, render_template, request
from jinja2 import Template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRID_PATH = os.path.join(BASE_DIR, "data", "flood_grid.geojson")
SEOUL_CENTER = [37.5665, 126.9780]

# rough bounding box of Seoul: biases search results towards Seoul (not a hard filter)
SEOUL_VIEWBOX = "126.734,37.715,127.269,37.413"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_USER_AGENT = "SeoulFloodHeatmap/1.0 (https://github.com/lkhgit/room_scanner_KHLee)"
NOMINATIM_MIN_INTERVAL_SEC = 1.1  # follows the Nominatim usage policy (one request per second)

app = Flask(__name__)
_last_geocode_call = 0.0


def load_grid():
    """Reads the grid GeoJSON and adds the display texts for the tooltip and popup."""
    with open(GRID_PATH, "r", encoding="utf-8") as f:
        grid = json.load(f)

    for feature in grid["features"]:
        props = feature["properties"]
        raw_years_detail = props.get("years_detail") or []
        years_detail = (
            json.loads(raw_years_detail) if isinstance(raw_years_detail, str) else raw_years_detail
        )
        years_detail.sort(key=lambda d: d["year"])
        props["years_text"] = ", ".join(
            f"{d['year']}년: {d['depth']}m" if d["depth"] is not None else f"{d['year']}년"
            for d in years_detail
        )
        props["freq_text"] = f"{props['freq']}회"
        props["avg_depth_text"] = (
            f"{props['avg_depth']}m" if props.get("avg_depth") is not None else "정보 없음"
        )
        props["max_depth_text"] = (
            f"{props['max_depth']}m" if props.get("max_depth") is not None else "정보 없음"
        )
    return grid


class AddressSearchControl(folium.MacroElement):
    """Search control that flies the map to an address or place name.

    It must be rendered inside the map iframe (the folium map object itself) to reach the Leaflet map variable
    directly, so it is a MacroElement rather than part of the outer Flask template.
    """

    _template = Template(
        """
        {% macro html(this, kwargs) %}
        <div style="position: fixed; top: 70px; left: 10px; z-index: 1000;
                    background: #fff; padding: 8px; border-radius: 6px;
                    box-shadow: 0 1px 5px rgba(0,0,0,0.4); font-family: sans-serif;">
            <form id="{{ this.get_name() }}-form" style="display:flex; gap:4px;">
                <input id="{{ this.get_name() }}-input" type="text"
                       placeholder="주소나 장소명 검색 (예: 강남역)"
                       style="width: 220px; padding: 4px 6px; font-size: 13px;"/>
                <button type="submit" style="padding: 4px 10px; font-size: 13px;">검색</button>
            </form>
            <div id="{{ this.get_name() }}-msg" style="font-size: 12px; color: #b91c1c; margin-top: 4px;"></div>
        </div>
        {% endmacro %}

        {% macro script(this, kwargs) %}
        (function() {
            var map = {{ this._parent.get_name() }};
            var marker = null;
            var form = document.getElementById("{{ this.get_name() }}-form");
            var input = document.getElementById("{{ this.get_name() }}-input");
            var msg = document.getElementById("{{ this.get_name() }}-msg");

            form.addEventListener("submit", function(e) {
                e.preventDefault();
                var q = input.value.trim();
                if (!q) { return; }
                msg.style.color = "#374151";
                msg.textContent = "검색 중...";
                fetch("/api/geocode?q=" + encodeURIComponent(q))
                    .then(function(r) { return r.json(); })
                    .then(function(data) {
                        if (data.error) {
                            msg.style.color = "#b91c1c";
                            msg.textContent = data.error;
                            return;
                        }
                        if (!data.results || data.results.length === 0) {
                            msg.style.color = "#b91c1c";
                            msg.textContent = "검색 결과가 없습니다.";
                            return;
                        }
                        var first = data.results[0];
                        map.flyTo([first.lat, first.lon], 16);
                        if (marker) {
                            marker.setLatLng([first.lat, first.lon]);
                        } else {
                            marker = L.marker([first.lat, first.lon]).addTo(map);
                        }
                        marker.bindPopup(first.display_name).openPopup();
                        msg.textContent = "";
                    })
                    .catch(function() {
                        msg.style.color = "#b91c1c";
                        msg.textContent = "검색 중 오류가 발생했습니다.";
                    });
            });
        })();
        {% endmacro %}
        """
    )

    def __init__(self):
        """Registers the control under a fixed name."""
        super().__init__()
        self._name = "AddressSearchControl"


def build_map(grid):
    """Builds the folium map with the coloured grid, tooltip, popup, colour scale and search box."""
    scores = [f["properties"]["weighted_score"] for f in grid["features"]]
    colormap = cm.LinearColormap(
        colors=["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"],
        vmin=min(scores),
        vmax=max(scores),
        caption="침수 위험 점수 (빈도 × 평균 침수심)",
    )

    fmap = folium.Map(
        location=SEOUL_CENTER,
        zoom_start=11,
        tiles="CartoDB positron",
        prefer_canvas=True,
    )

    def style_function(feature):
        """Colours a cell by its risk score."""
        score = feature["properties"]["weighted_score"]
        return {
            "fillColor": colormap(score),
            "color": colormap(score),
            "weight": 0.3,
            "fillOpacity": 0.75,
        }

    def highlight_function(_feature):
        """Style of a cell under the mouse."""
        return {"weight": 2, "color": "#000000", "fillOpacity": 0.9}

    folium.GeoJson(
        grid,
        name="침수 빈도 격자",
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["freq_text"],
            aliases=["침수 빈도"],
            sticky=True,
        ),
        popup=folium.GeoJsonPopup(
            fields=["freq_text", "avg_depth_text", "max_depth_text", "years_text"],
            aliases=["침수 빈도", "평균 침수심", "최대 침수심", "연도별 이력"],
            localize=True,
            max_width=320,
        ),
    ).add_to(fmap)

    colormap.add_to(fmap)
    folium.LayerControl().add_to(fmap)
    AddressSearchControl().add_to(fmap)
    return fmap


@app.route("/api/geocode")
def geocode():
    """Looks up a place with Nominatim (at most one request per second) and returns up to 5 results."""
    global _last_geocode_call

    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "검색어를 입력하세요.", "results": []}), 400

    elapsed = time.monotonic() - _last_geocode_call
    if elapsed < NOMINATIM_MIN_INTERVAL_SEC:
        time.sleep(NOMINATIM_MIN_INTERVAL_SEC - elapsed)

    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={
                "q": query,
                "format": "jsonv2",
                "limit": 5,
                "countrycodes": "kr",
                "viewbox": SEOUL_VIEWBOX,
                "accept-language": "ko",
            },
            headers={"User-Agent": NOMINATIM_USER_AGENT},
            timeout=5,
        )
        _last_geocode_call = time.monotonic()
        resp.raise_for_status()
        raw_results = resp.json()
    except requests.RequestException:
        return jsonify({"error": "지오코딩 서비스 호출에 실패했습니다.", "results": []}), 502

    results = [
        {
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "display_name": r.get("display_name", query),
        }
        for r in raw_results
    ]
    return jsonify({"results": results})


@app.route("/")
def index():
    """Renders the page with the flood heat map."""
    grid = load_grid()
    fmap = build_map(grid)
    map_html = fmap._repr_html_()
    return render_template(
        "index.html",
        map_html=map_html,
        cell_count=len(grid["features"]),
        max_freq=max(f["properties"]["freq"] for f in grid["features"]),
    )


if __name__ == "__main__":
    app.run(debug=True)
