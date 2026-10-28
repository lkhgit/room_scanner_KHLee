#!/usr/bin/env python3
"""
표면 온습도 측정 기기 PC 로거

surface_scan.ino(v2)의 시리얼 출력을 받아 세션 폴더에 저장한다.
    <out-dir>/<session>/session.yaml   세션 정보 (방위각, 장소, 센서, 재료 등급)
    <out-dir>/<session>/scans.csv      스캔 데이터 (한 행 = 한 스캔의 한 면)
    <out-dir>/<session>/serial_raw.log 수신한 원본 줄 전체 (문제 추적용)
    <out-dir>/<session>/outdoor.csv    스캔 직후 가져온 초단기실황 외기 (스캔 시각과 직전 23시간)

초단기실황은 기상청 단기예보 조회서비스로 가져오며, 인증키는 kma.py에 적는다.
격자는 site.lat/lon -> 관측소 좌표 -> 장소 이름(카카오 지오코딩) 순서로 정해 session.yaml의
grid에 적는다. 키가 없거나 요청이 실패해도 측정은 계속한다. 끄려면 --no-ncst.

사용법
    pip install pyserial pyyaml
    python logger.py --port COM3 --session banjiha_A --site "망원동 반지하" --lat 37.556 --lon 126.906
    (macOS/Linux 포트 예: /dev/tty.usbmodem14101, /dev/ttyACM0)

    로거가 포트를 열면 아두이노 우노가 리셋되고, 기기가 #INFO,ready를 보내면
    로거가 n(스캔 1회)을 보낸다. 기본값은 스캔 1회를 저장한 뒤 로거가 종료되는 것이다.
    여러 번 측정하려면 --scans 24 --interval 60 처럼 횟수와 간격(분)을 준다.
    --no-start를 주면 자동으로 보내지 않고 사용자가 n을 입력해 시작한다.
    포트를 열기 전에 기기를 원위치(센서 정면·수평)에 두어야 한다.
    로거 없이 전원만 넣으면 기기는 움직이지 않고 대기한다.

실행 중 키보드 입력 (입력 후 Enter)
    n 지금 스캔   s 스캔 즉시 정지(원위치 복귀 후 로거 종료)   q 로거 종료

같은 --session 이름으로 다시 실행하면 기존 scans.csv 뒤에 이어 쓰고,
scan_id는 기존 최대값 다음부터 이어지도록 다시 매긴다.

테스트: --replay <파일> 로 시리얼 대신 저장된 출력 텍스트를 읽는다.
"""

import argparse
import csv
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

KST = ZoneInfo("Asia/Seoul")
COLUMNS = ["scan_id", "timestamp", "face", "zone", "az_deg", "el_deg",
           "t_surf_c", "t_surf_sd_c", "t_air_c", "rh_pct", "t_sensor_c"]
N_FACES = 20


def now_iso() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


# ---------------------------------------------------------------- session.yaml
def load_or_create_session(path: Path, args) -> dict:
    if path.exists():
        with path.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {
        "session": args.session,
        "created_at": now_iso(),
        "site": {"name": args.site, "lat": args.lat, "lon": args.lon},
        "asos_station": args.station,
        "device": {
            "firmware": "surface_scan v2",
            "ir_sensor": args.ir_model,
            "emissivity": args.emissivity,
        },
        "heading": {"deg": None, "mag_calib": None, "sys_calib": None, "measured_at": None},
        "materials": {"default": args.material, "by_face": {}},
        "excluded_faces": [],
        "notes": args.notes or "",
    }


def save_session(path: Path, session: dict) -> None:
    tmp = path.with_suffix(".yaml.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        yaml.safe_dump(session, f, allow_unicode=True, sort_keys=False)
    tmp.replace(path)


# ---------------------------------------------------------------- scans.csv
class ScanWriter:
    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        if path.exists() and path.stat().st_size > 0:
            with path.open(encoding="utf-8") as f:
                ids = [int(r["scan_id"]) for r in csv.DictReader(f) if r.get("scan_id")]
            self.offset = max(ids, default=0)
        else:
            with path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(COLUMNS)

    def write_scan(self, rows: list[list[str]], timestamp: str) -> int:
        """한 스캔의 20행을 한 번에 기록한다. 기록된 scan_id를 돌려준다."""
        scan_id = self.offset + int(rows[0][0])
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for r in rows:
                w.writerow([scan_id, timestamp] + r[2:])
        return scan_id


# ---------------------------------------------------------------- 파서
class Parser:
    """아두이노의 한 줄 단위 프로토콜을 해석한다."""

    def __init__(self, writer: ScanWriter, session: dict, session_path: Path, on_ready=None):
        self.on_ready = on_ready
        # 스캔 결과 알림: on_scan_done(outcome) — outcome은 "saved", "discarded", "aborted"
        self.on_scan_done = None
        self.on_scan_begin = None
        self.on_scan = None                # 스캔을 저장한 직후 on_scan(timestamp) - 초단기실황 수집
        self.writer = writer
        self.session = session
        self.session_path = session_path
        self.buf: list[list[str]] = []
        self.cur_id: str | None = None
        self.scans_saved = 0

    def log(self, msg: str) -> None:
        print(f"[{datetime.now(KST):%H:%M:%S}] {msg}", flush=True)

    def handle(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        if line.startswith("#"):
            self.handle_meta(line)
        else:
            self.handle_data(line)

    def handle_meta(self, line: str) -> None:
        parts = line.split(",")
        tag = parts[0]
        if tag == "#HDR":
            if parts[1:] != COLUMNS:
                self.log(f"경고: 펌웨어 열 정의가 로거와 다름 -> {parts[1:]}")
        elif tag == "#SESSION":
            kv = dict(zip(parts[1::2], parts[2::2]))
            h = self.session["heading"]
            try:
                h["deg"] = float(kv.get("heading_deg", "nan"))
            except ValueError:
                h["deg"] = None
            if h["deg"] is not None and h["deg"] != h["deg"]:  # NaN
                h["deg"] = None
            h["mag_calib"] = int(kv.get("mag_calib", 0))
            h["sys_calib"] = int(kv.get("sys_calib", 0))
            h["measured_at"] = now_iso()
            save_session(self.session_path, self.session)
            self.log(f"방위각 {h['deg']}° (지자기 보정 {h['mag_calib']}/3) 저장")
            if h["mag_calib"] < 3:
                self.log("경고: 지자기 보정이 3이 아니라 방위각을 믿기 어려움")
        elif tag == "#SCAN_BEGIN":
            self.buf, self.cur_id = [], parts[1]
            self.log(f"스캔 {parts[1]} 시작")
            if self.on_scan_begin:
                self.on_scan_begin()
        elif tag == "#SCAN_END":
            self.finish_scan(parts[1])
        elif tag == "#ABORT":
            self.log(f"스캔 {parts[1]} 중단됨 - 저장하지 않음")
            self.buf, self.cur_id = [], None
            if self.on_scan_done:
                self.on_scan_done("aborted")
        elif tag == "#INFO":
            self.log("기기: " + ",".join(parts[1:]))
            if parts[1:2] == ["ready"] and self.on_ready:
                self.on_ready()
        else:
            self.log(f"알 수 없는 메타 줄: {line}")

    def handle_data(self, line: str) -> None:
        row = next(csv.reader([line]))
        if len(row) != len(COLUMNS):
            self.log(f"경고: 열 개수 {len(row)} (기대 {len(COLUMNS)}) - 무시: {line}")
            return
        if self.cur_id is None or row[0] != self.cur_id:
            self.log(f"경고: SCAN_BEGIN 없이 들어온 데이터 - 무시: {line}")
            return
        try:
            [float(row[i]) for i in (4, 5, 6, 7, 8, 9, 10)]
            int(row[2])
        except ValueError:
            self.log(f"경고: 숫자가 아닌 값 - 무시: {line}")
            return
        self.buf.append(row)

    def finish_scan(self, scan_id: str) -> None:
        outcome = "discarded"
        if scan_id != self.cur_id:
            self.log(f"경고: SCAN_END {scan_id}가 진행 중 스캔 {self.cur_id}와 다름 - 버림")
        elif len(self.buf) != N_FACES:
            self.log(f"경고: 스캔 {scan_id}의 면 데이터가 {len(self.buf)}개 - 버림")
        else:
            outcome = "saved"
            ts = now_iso()
            sid = self.writer.write_scan(self.buf, ts)
            self.scans_saved += 1
            if self.on_scan:
                self.on_scan(ts)
            temps = [float(r[6]) for r in self.buf]
            self.log(f"스캔 {sid} 저장 (표면 {min(temps):.1f}~{max(temps):.1f} °C, "
                     f"실내 {self.buf[0][8]} °C / {self.buf[0][9]} %)")
            sds = [float(r[7]) for r in self.buf]
            if max(sds) > 0.3:
                self.log(f"주의: IR 표준편차 최대 {max(sds):.2f} °C - 흔들림이나 노이즈 확인")
        self.buf, self.cur_id = [], None
        if self.on_scan_done:
            self.on_scan_done(outcome)


# ---------------------------------------------------------------- 스캔 횟수·간격 관리
class Controller:
    """정해진 횟수만큼 n을 보내고, 다 저장하면 종료 신호를 낸다.

    scans=0이면 q를 입력할 때까지 계속한다. 버려진 스캔은 횟수에 넣지 않고
    바로 다시 시도하며, 연속 MAX_RETRY번 버려지면 종료한다.
    """
    MAX_RETRY = 3

    def __init__(self, ser, parser: Parser, scans: int, interval_min: float,
                 auto_start: bool, stop: threading.Event, clock=time.monotonic):
        self.ser, self.parser, self.scans = ser, parser, scans
        self.interval_s = interval_min * 60
        self.stop, self.clock = stop, clock
        self.next_at: float | None = None
        self.begin_at: float | None = None
        self.retries = 0
        parser.on_scan_begin = self.on_begin
        parser.on_scan_done = self.on_done
        if auto_start:
            parser.on_ready = self.send_scan

    def send_scan(self) -> None:
        self.ser.write(b"n")
        self.next_at = None
        done = self.parser.scans_saved
        total = "무제한" if self.scans == 0 else str(self.scans)
        self.parser.log(f"스캔 명령(n) 전송 ({done + 1}/{total})")

    def on_begin(self) -> None:
        self.begin_at = self.clock()

    def on_done(self, outcome: str) -> None:
        if outcome == "aborted":
            self.parser.log("정지 명령으로 스캔이 중단되어 로거를 종료함")
            self.stop.set()
            return
        if outcome == "discarded":
            self.retries += 1
            if self.retries >= self.MAX_RETRY:
                self.parser.log(f"스캔이 연속 {self.retries}번 버려져 종료함 - 배선과 serial_raw.log 확인")
                self.stop.set()
            else:
                self.parser.log("버려진 스캔을 다시 시도함")
                self.send_scan()
            return
        self.retries = 0
        if self.scans and self.parser.scans_saved >= self.scans:
            self.parser.log(f"스캔 {self.parser.scans_saved}회 저장 완료 - 로거를 종료함")
            self.stop.set()
            return
        start = self.begin_at if self.begin_at is not None else self.clock()
        self.next_at = start + self.interval_s
        wait = max(0, self.next_at - self.clock())
        self.parser.log(f"다음 스캔까지 {wait / 60:.0f}분 대기 (q+Enter로 종료)")

    def tick(self) -> None:
        if self.next_at is not None and self.clock() >= self.next_at:
            self.send_scan()


def run_serial(ser, parser: Parser, ctrl: Controller, stop: threading.Event, feed) -> None:
    while not stop.is_set():
        data = ser.readline()
        if data:
            feed(data.decode("utf-8", errors="replace"))
        ctrl.tick()


# ---------------------------------------------------------------- 초단기실황 (측정 시점 외기)
def resolve_logger_grid(session: dict, session_path: Path, log):
    """초단기실황 격자. site.lat/lon -> 관측소 좌표 -> 장소 이름 지오코딩. 정하면 session.yaml에 적는다."""
    import kma
    site = session.get("site") or {}
    got = kma.resolve_grid(site, session.get("asos_station"))
    if got is None and site.get("name"):
        try:
            hit = kma.geocode(site["name"])
        except Exception as e:
            log(f"경고: 장소 '{site['name']}'의 좌표를 찾지 못함 ({e})")
            hit = None
        if hit:
            lat, lon, address, query = hit
            site.update(lat=lat, lon=lon, geocoded_from=query, geocoded_address=address)
            session["site"] = site
            log(f"장소 '{site['name']}' -> {address} ({lat:.4f}, {lon:.4f})")
            got = kma.resolve_grid(site, None)
    if got is None:
        log("경고: 격자를 정할 수 없어 초단기실황을 모으지 않음 (--lat/--lon 또는 --station을 줄 것)")
        return None
    nx, ny, how = got
    session["grid"] = {"nx": nx, "ny": ny, "from": how}
    save_session(session_path, session)
    log(f"초단기실황 격자 ({nx}, {ny}) - {how}")
    return session["grid"]


class NcstCollector:
    """스캔 시각과 직전 23시간의 초단기실황 중 outdoor.csv에 없는 것을 백그라운드로 가져온다."""

    def __init__(self, path: Path, grid: dict, log):
        self.path, self.grid, self.log = path, grid, log
        self.lock = threading.Lock()
        self.threads: list[threading.Thread] = []

    def __call__(self, ts: str) -> None:
        t = threading.Thread(target=self.work, args=(ts,), daemon=True)
        t.start()
        self.threads.append(t)

    def work(self, ts: str) -> None:
        import kma
        end = kma.parse_kst(ts).replace(minute=0, second=0, microsecond=0)
        hours = [end - timedelta(hours=i) for i in range(24)]
        with self.lock:
            try:
                added, failed = kma.collect_ncst(self.path, hours, self.grid["nx"], self.grid["ny"])
                msg = f"초단기실황 {added}시간 추가"
                if failed:
                    msg += f", {len(failed)}시간은 못 가져옴 (시뮬레이터가 나중에 보충)"
                self.log(msg)
            except Exception as e:
                self.log(f"경고: 초단기실황 수집 실패 - {e}")

    def wait(self, timeout_s: float = 120) -> None:
        pending = [t for t in self.threads if t.is_alive()]
        if pending:
            self.log("초단기실황 수집이 끝나기를 기다리는 중")
            for t in pending:
                t.join(timeout_s)


# ---------------------------------------------------------------- 실행
def stdin_forwarder(ser, stop: threading.Event) -> None:
    for line in sys.stdin:
        cmd = line.strip().lower()
        if cmd == "q":
            stop.set()
            return
        if cmd in ("n", "s"):
            ser.write(cmd.encode())
        elif cmd:
            print("명령: n 지금 스캔, s 스캔 정지, q 종료", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="표면 온습도 측정 기기 로거")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--port", help="시리얼 포트 (예: COM3, /dev/ttyACM0)")
    src.add_argument("--replay", help="시리얼 대신 읽을 출력 텍스트 파일 (테스트용)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "data"),
                    help="세션 폴더를 만들 상위 폴더 (기본: logger.py가 있는 폴더의 data)")
    ap.add_argument("--session", default=datetime.now(KST).strftime("%Y%m%d_%H%M"))
    ap.add_argument("--site", default="")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--station", type=int,
                    help="ASOS 관측소 번호 (서울 108). 비우면 시뮬레이터가 장소로 가장 가까운 곳을 고름")
    ap.add_argument("--ir-model", default="MLX90614ESF-BAA")
    ap.add_argument("--emissivity", type=float, default=1.0)
    ap.add_argument("--material", default="sensitive",
                    choices=["very_sensitive", "sensitive", "medium_resistant", "resistant"])
    ap.add_argument("--notes", default="")
    ap.add_argument("--scans", type=int, default=1,
                    help="저장할 스캔 횟수. 다 저장하면 로거 종료 (0이면 q를 누를 때까지 계속)")
    ap.add_argument("--interval", type=float, default=60,
                    help="여러 번 스캔할 때 스캔 시작 간격(분). 기본 60")
    ap.add_argument("--no-start", action="store_true",
                    help="기기가 준비돼도 스캔을 자동 시작하지 않음 (n을 직접 입력)")
    ap.add_argument("--no-ncst", action="store_true", help="스캔 직후 초단기실황(외기) 수집을 끔")
    args = ap.parse_args()

    sdir = Path(args.out_dir) / args.session
    sdir.mkdir(parents=True, exist_ok=True)
    session_path = sdir / "session.yaml"
    session = load_or_create_session(session_path, args)
    save_session(session_path, session)
    writer = ScanWriter(sdir / "scans.csv")
    parser = Parser(writer, session, session_path)
    raw = (sdir / "serial_raw.log").open("a", encoding="utf-8")
    parser.log(f"세션 폴더: {sdir.resolve()}")
    ncst = None
    if not args.no_ncst and not args.replay:
        grid = resolve_logger_grid(session, session_path, parser.log)
        if grid:
            ncst = parser.on_scan = NcstCollector(sdir / "outdoor.csv", grid, parser.log)
    if writer.offset:
        parser.log(f"기존 데이터에 이어 씀 (scan_id {writer.offset + 1}부터)")

    def feed(line: str) -> None:
        raw.write(f"{now_iso()}\t{line.rstrip()}\n")
        raw.flush()
        parser.handle(line)

    try:
        if args.replay:
            with open(args.replay, encoding="utf-8") as f:
                for line in f:
                    feed(line)
        else:
            import serial
            ser = serial.Serial(args.port, args.baud, timeout=1)
            stop = threading.Event()
            ctrl = Controller(ser, parser, args.scans, args.interval,
                              auto_start=not args.no_start, stop=stop)
            plan = "1회" if args.scans == 1 else (
                "q를 누를 때까지" if args.scans == 0 else f"{args.scans}회, {args.interval:g}분 간격")
            if args.no_start:
                parser.log(f"포트 열림 - 기기가 대기함. n+Enter로 시작 ({plan}), q로 종료")
            else:
                parser.log(f"포트 열림 - 아두이노가 준비되면 스캔 시작 ({plan}). q+Enter로 종료")
            threading.Thread(target=stdin_forwarder, args=(ser, stop), daemon=True).start()
            run_serial(ser, parser, ctrl, stop, feed)
            ser.close()
    except KeyboardInterrupt:
        pass
    finally:
        raw.close()
        if ncst:
            ncst.wait()
        if parser.cur_id is not None:
            parser.log(f"종료 시 진행 중이던 스캔 {parser.cur_id}는 저장되지 않음")
        parser.log(f"종료. 이번 실행에서 저장한 스캔 {parser.scans_saved}개")


if __name__ == "__main__":
    main()
