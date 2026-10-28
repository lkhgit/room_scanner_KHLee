"""Runs sunlight scanner/sun_calculator.py unchanged, but with every on-screen string in English.

Usage: sun_runner.py <same arguments as sun_calculator.py>
The calculation code is not touched: the viewer template is translated in memory before main() runs, and the
generated files are checked afterwards for leftover Korean text.
"""
import importlib.util
import re
import sys
from pathlib import Path

SUN = Path(__file__).resolve().parents[2] / "sunlight scanner" / "sun_calculator.py"

# Longest first. Applied to the viewer template and to the generated viewer / json files.
PAIRS = [
    ("실내에 닿은 직사광 (W)", "Direct sun reaching the room (W)"),
    ("바닥에 닿은 면적 (m²)", "Floor area in sun (m²)"),
    ("직사광 세기 (W/m²) — 색이 없는 곳은 그늘", "Direct sun intensity (W/m²) — uncoloured = shade"),
    ("하루 직사광 시간 (h)", "Direct sun hours per day (h)"),
    ("직사광 세기 (W/m²)", "Direct sun intensity (W/m²)"),
    ("맑은 날 직사광만 계산했습니다 (하늘 산란광·반사광 제외). 창밖 건물·나무 같은 바깥 장애물은 없고,",
     "Clear-sky direct sunlight only (no sky-diffuse or reflected light). No outside obstacles such as buildings or trees,"),
    ("창문은 유리 없이 뚫린 구멍으로 가정했습니다. 노란 빛줄기는 창문 네 모서리에서 태양 반대 방향으로 쏜 광선입니다.",
     "and windows are treated as open holes without glass. Yellow beams are rays cast from the four window corners away from the sun."),
    ("← → 키로 시각을 옮길 수 있습니다.", "Use the ← → keys to move the time."),
    ("모델 로드 실패: ", "Failed to load model: "),
    ("맑은 날 직달일사", "Clear-sky direct normal irradiance"),
    ("창문을 통과하는 직사광", "Direct sun through windows"),
    ("실내 면에 닿은 직사광", "Direct sun on room surfaces"),
    ("빛이 닿은 바닥", "Sunlit floor"),
    ("빛이 닿은 벽", "Sunlit walls"),
    ("해 뜬 시간", "Sunrise to sunset"),
    ("실내로 들어온 직사광", "Direct sun entering the room"),
    ("창문 통과량 (검산)", "Through windows (check)"),
    ("해가 뜨지 않음", "Sun below horizon"),
    ("고도 ", "Altitude "),
    ("방위 ", "Azimuth "),
    ("창문 없음", "No windows"),
    ("하루 일조시간", "Sun hours"),
    ("지정 시각", "Selected time"),
    ("면별 일조시간", "Sun hours by surface"),
    ("창문별 하루", "Windows, whole day"),
    ("창문별", "By window"),
    ("실내 일조 분석", "Indoor Sunlight"),
    ("바깥 위에서", "Outside, above"),
    ("카메라 위치", "Camera position"),
    ("위에서", "Top view"),
    ("▶ 재생", "▶ Play"),
    ("❚❚ 정지", "❚❚ Pause"),
    ("<th>창문</th>", "<th>Window</th>"),
    ("<th>방향</th>", "<th>Facing</th>"),
    ("<th>유입 W</th>", "<th>In W</th>"),
    ("<th>바닥 깊이</th>", "<th>Floor depth</th>"),
    ("<th>면</th>", "<th>Surface</th>"),
    ("<th>평균 h</th>", "<th>Mean h</th>"),
    ("<th>최대 h</th>", "<th>Max h</th>"),
    ("<th>직사 h</th>", "<th>Sun h</th>"),
    ("<th>유입 kWh</th>", "<th>In kWh</th>"),
    ("<th>최대 깊이</th>", "<th>Max depth</th>"),
    ("최대 ${", "Max ${"),
    ("${w.facing}향", "${w.facing}-facing"),
    ("${D.windows[k].facing}향", "${D.windows[k].facing}-facing"),
    ("'ko-KR'", "'en-US'"),
    ('lang="ko"', 'lang="en"'),
    ('<div class="label">날짜</div>', '<div class="label">Date</div>'),
    ('<div class="label">보기</div>', '<div class="label">View</div>'),
    ("['북','북동','동','남동','남','남서','서','북서']", "['N','NE','E','SE','S','SW','W','NW']"),
    ("[['북', 0,", "[['N', 0,"), ("['동', 90,", "['E', 90,"), ("['남', 180,", "['S', 180,"), ("['서', 270,", "['W', 270,"),
]
# Generated labels (Python f-strings in sun_calculator.main) and data strings.
REGEX = [
    (re.compile(r"벽 (\d+) \((\w+)쪽\)"), r"Wall \1 (\2 side)"),
    (re.compile(r"창 (\d+)"), r"Window \1"),
    (re.compile(r"지정일"), "Chosen date"),
    (re.compile(r"동지"), "Winter solstice"), (re.compile(r"춘분"), "Spring equinox"),
    (re.compile(r"하지"), "Summer solstice"), (re.compile(r"추분"), "Autumn equinox"),
    (re.compile(r"바닥"), "Floor"), (re.compile(r"천장"), "Ceiling"),
    (re.compile(r"위도 ([-\d.]+), 경도 ([-\d.]+) · 파노라마 가운데 방위 (\d+)° \((\w+)\)"),
     r"Latitude \1, longitude \2 · panorama centre bearing \3° (\4)"),
    (re.compile(r"맑은 날 직사광만, 바깥 장애물 없음, 창문 = 뚫린 구멍\(유리 투과율 미적용\)"),
     "Clear-sky direct sun only; no outside obstacles; windows are open holes (no glass transmittance)"),
]
HANGUL = re.compile(r"[가-힣]")

# Brighter sun-hours colours. Rooms with small windows get mostly 0-1 h, which the original ramp shows as
# near-black purple; here shade is a mid slate and even half an hour of sun stands out.
HOUR_STOPS_OLD = "const HOUR_STOPS = [[0,'#1b0d33'],[1,'#57176b'],[2,'#a6245a'],[3,'#e34a33'],[4,'#f99430'],[6,'#feeeb0']];"
HOUR_STOPS_NEW = ("const HOUR_STOPS = [[0,'#5a6175'],[0.5,'#a04fb0'],[1,'#e0508a'],[2,'#f77b45'],[3,'#fdb24a'],"
                  "[4,'#fee08a'],[6,'#fffbe8']];")


# The viewer's shader writes glslColor() values straight to the screen, but THREE.Color converts the hex stops to
# linear first, so every ramp colour renders much darker than its legend (shade looked near-black). For the sun-hours
# ramp only, convert back so the model matches the legend.
GAMMA_FIXES = [
    ("const glslColor = h => { const c = new THREE.Color(h);",
     "const glslColor = (h, raw) => { const c = new THREE.Color(h); if (raw) c.convertLinearToSRGB();"),
    ("const glslRamp = (name, stops) => {", "const glslRamp = (name, stops, raw) => {"),
    ("return mix(${glslColor(ca)}, ${glslColor(cb)},", "return mix(${glslColor(ca, raw)}, ${glslColor(cb, raw)},"),
    ("return ${glslColor(stops[stops.length - 1][1])};", "return ${glslColor(stops[stops.length - 1][1], raw)};"),
    ("${glslRamp('hoursRamp', HOUR_STOPS)}", "${glslRamp('hoursRamp', HOUR_STOPS, true)}"),
]


def restyle(text: str) -> str:
    text = text.replace(HOUR_STOPS_OLD, HOUR_STOPS_NEW)
    for old, new in GAMMA_FIXES:
        text = text.replace(old, new)
    return text


def translate(text: str) -> str:
    for ko, en in PAIRS:
        text = text.replace(ko, en)
    for rx, en in REGEX:
        text = rx.sub(en, text)
    return text


def visible_korean(text: str):
    """Korean outside // comments."""
    return [ln.strip() for ln in text.splitlines() if HANGUL.search(re.sub(r"//.*$", "", ln))]


def main():
    spec = importlib.util.spec_from_file_location("sun_calculator", SUN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    mod.DIR8 = names
    if HOUR_STOPS_OLD not in mod.VIEWER_HTML or any(old not in mod.VIEWER_HTML for old, _ in GAMMA_FIXES):
        print("WARNING: sun_calculator.py's viewer template changed; the brighter sun-hours colours were not fully applied")
    mod.VIEWER_HTML = restyle(translate(mod.VIEWER_HTML))
    sys.argv = [str(SUN)] + sys.argv[1:]
    mod.main()
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else None
    if out:
        for name in ("sun_viewer.html", "sun_hours.json"):
            f = out / name
            if f.exists():
                text = translate(f.read_text(encoding="utf-8"))
                f.write_text(text, encoding="utf-8")
                left = visible_korean(text) if name.endswith(".html") else [t for t in [text] if HANGUL.search(t)]
                if left:
                    print(f"WARNING: untranslated Korean text remains in {name}: {left[:3]}")


if __name__ == "__main__":
    main()
