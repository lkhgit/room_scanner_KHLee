"""Turns Korean text produced by the existing Python scripts into the English shown on screen.

Unknown messages fall back to a generic English line; the original is kept in the technical log.
"""
import re

FALLBACK = "Simulation note (original text is in the technical log)."

_RULES = [
    (r"^(.+): 지중온도 결측 (\d+)일", lambda m: f"Ground temperature is missing for {m[2]} day(s) in {m[1]}; interpolated from neighbouring days."),
    (r"^(.+): (기온|증기압) 결측 (\d+)시간이 (\d+)시간보다 긴 구간에 있어 보간함",
     lambda m: f"{m[1]}: {'Air temperature' if m[2] == '기온' else 'Vapour pressure'} is missing for {m[3]} hour(s) "
               f"inside gaps longer than {m[4]} h; interpolated and flagged."),
    (r"^가장 가까운 관측소 (\d+) .+가 ([\d.]+) km 떨어져 있음", lambda m: f"The nearest weather station (#{m[1]}) is {m[2]} km away."),
    (r"^scans\.csv에 면 (.+)의 데이터가 없음", lambda m: f"No scan data for face(s) {m[1]}."),
    (r"^IR 표준편차 > sd_max ([\d.]+) °C 인 면·스캔 (\d+)개",
     lambda m: f"{m[2]} face/scan reading(s) with an IR standard deviation above {m[1]} °C were left out of the f estimate."),
    (r"^격자 좌표를 정할 수 없어", lambda m: "The forecast grid could not be determined (no coordinates), so outdoor data was not supplemented."),
    (r"^초단기실황을 가져오지 못함", lambda m: "The latest outdoor observations could not be fetched."),
    (r"^외기 자료.*없는 스캔 (\d+)개", lambda m: f"{m[1]} scan(s) without outdoor data were left out of the humidity-load (Δv_max) estimate."),
    (r"^평균 g_s ([\d.]+) < g_min ([\d.]+)",
     lambda m: f"Mean g_s {m[1]} is below g_min {m[2]}: the scan was taken while outdoor air was warm, so Δv_max is uncertain."),
    (r"^측정 dv_max가 음수\(([-\d.]+) g/m3\)", lambda m: f"Measured Δv_max was negative ({m[1]} g/m³); set to 0."),
    (r"^dv_max 불확도 큼 \(sigma ([\d.]+) > ([\d.]+) g/m3\) - 측정값 ([\d.]+) 사용 \(중앙 ([\d.]+), 보수 ([\d.]+)\)",
     lambda m: f"Δv_max is uncertain (sigma {m[1]} > {m[2]} g/m³); using the measured {m[3]} g/m³ (central {m[4]}, worst case {m[5]})."),
    (r"^dv_max 대체값 사용 \((.+)\) - 중앙 ([\d.]+), 보수 ([\d.]+) g/m3",
     lambda m: "Using substitute Δv_max ("
               + ("outdoor air too warm to compute" if m[1].startswith("g가") else "no scans with outdoor data")
               + f") - central {m[2]}, worst case {m[3]} g/m³."),
    (r"^지중온도가 없거나 (\d+)일보다 늦은 스캔 (\d+)개", lambda m: f"{m[2]} scan(s) with no ground temperature within {m[1]} day(s) were left out of the f estimate."),
    (r"^면 (\[.*\]): 스캔이 없거나.*대체값 ([\d.]+) / ([\d.]+) 사용",
     lambda m: f"Face(s) {m[1]}: f could not be computed (no scan, or no indoor-ground temperature difference); using substitutes {m[2]} / {m[3]}."),
    (r"^면 (\[.*\]): f 불확도 큼 \(sigma_f > ([\d.]+)\)", lambda m: f"Face(s) {m[1]}: f is uncertain (sigma_f > {m[2]}); using the measured value."),
    (r"^면 (\[.*\]): f가 \[0, 1\]을 2 sigma",
     lambda m: f"Face(s) {m[1]}: f is outside [0, 1] by more than 2 sigma (anomalous); clipped. "
               "Check for low-emissivity surfaces (glass, metal) or heat sources in view."),
    (r"^면 (\[.*\]): f가 \[0, 1\]을 조금 벗어나", lambda m: f"Face(s) {m[1]}: f was slightly outside [0, 1]; clipped."),
]

_PROV = re.compile(r"^잠정 결과: 대체값 사용\((.+)\)$")


def warning(text: str) -> str:
    """Translates one Korean simulator warning into English, or returns a generic line if it is unknown."""
    if not re.search(r"[가-힣]", text):
        return text
    for pat, fn in _RULES:
        m = re.match(pat, text)
        if m:
            return fn(m)
    return FALLBACK


def provisional(text):
    """Translates the simulator's "provisional result" message into English."""
    if not text:
        return None
    m = _PROV.match(text)
    if not m:
        return "Provisional result: substitute values were used."
    parts = []
    raw = m[1]
    if raw.startswith("면 전체"):
        parts.append("all faces")
        rest = raw[len("면 전체"):].lstrip(", ")
    elif raw.startswith("면 "):
        head, _, rest = raw[2:].partition("Δv_max")
        faces = head.strip().rstrip(",").strip()
        parts.append(f"faces {faces}" if faces else "")
        rest = "Δv_max" if "Δv_max" in raw else ""
    else:
        rest = raw
    if "Δv_max" in rest:
        parts.append("Δv_max")
    parts = [p for p in parts if p]
    return "Provisional result: substitute values were used (" + ", ".join(parts) + ")."


def bearing_name(b: float) -> str:
    """Returns the 8-point compass name (N, NE, ...) for a bearing in degrees."""
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int(((b % 360) + 22.5) // 45) % 8]
