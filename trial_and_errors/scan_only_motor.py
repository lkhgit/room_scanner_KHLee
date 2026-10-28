"""
scan.py — 팬·틸트 리그로 구형 파노라마용 사진을 촬영하는 PC 측 지휘 스크립트

    pip install pyserial opencv-python numpy

사용 예
    python scan.py --uno COM3 --esp COM4                  # 전체 스캔
    python scan.py --uno COM3 --esp COM4 --jog            # 시작 전 수평 맞추기
    python scan.py --uno COM3 --calib P 5                 # 팬 5바퀴 → 표시 확인으로 스텝 보정
    python scan.py --uno COM3 --esp COM4 --test           # 정면 1장만 촬영
    python scan.py --uno COM3 --sim --jog                 # 카메라 없이 모터만으로 전체 스캔 시뮬레이션

동작 개요 (행마다)
    틸트 A(+방향 접근) → 팬 0, 45, ..., 315° 정지 촬영 → 팬 360°에서 확인용 1장
    → A P 0  (= 370° 역회전 + 10° 정회전으로 원위치, 케이블도 풀림)
    → 모터 코일 해제 상태로 휴식 → 온도 확인 → 다음 행
틸트 행을 아래에서 위 순서로 찍으면 틸트는 항상 + 방향으로만 움직여 백래시 보정 이동이 필요 없음.
"""
import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import serial

try:
    import cv2
    import numpy as np
    HAVE_CV = True
except ImportError:
    HAVE_CV = False

# ------------------------------------------------------------------ 설정
# (틸트 각도, 해당 행의 팬 촬영 수) — 화각 측정 후 조정. 아래→위 순서 유지.
ROWS = [
    (-60, 6),
    (-30, 8),
    (0, 8),
    (30, 8),
    (60, 6),
    (90, 1),
]
STEPS_PER_REV = {"P": 4075.77, "T": 4075.77}   # --calib 결과로 교체
BACKLASH_DEG = {"P": 10.0, "T": 8.0}
MIN_STEP_US = 1800

SETTLE_S = 0.4            # 이동 후 진동이 가라앉을 때까지 대기
ROW_REST_S = 15.0         # 행 사이 휴식 (모터 코일 꺼진 상태)

ESP_PAUSE_C = 65.0        # 행이 끝났을 때 이 온도 이상이면 카메라 정지 후 냉각
ESP_RESUME_C = 50.0       # 이 온도 이하로 내려가면 재개
ESP_ABORT_C = 75.0        # 촬영 중 이 온도를 넘으면 해당 행 중단 → 냉각 후 재촬영

USE_NTC = False           # A0에 NTC를 달았다면 True
MOTOR_PAUSE_C = 55.0
MOTOR_RESUME_C = 45.0

DRIFT_WARN_PX = 6.0       # 0°와 360° 사진의 어긋남 경고 기준 (800px 폭 기준)
USE_EXPOSURE_LOCK = True


# ------------------------------------------------------------------ 장치 래퍼
class Rig:
    def __init__(self, port):
        self.s = serial.Serial(port, 115200, timeout=60)
        self._wait_ready()

    def _wait_ready(self):
        # UNO는 포트를 열면 리셋됨
        t0 = time.time()
        while time.time() - t0 < 5:
            line = self.s.readline().decode(errors="ignore").strip()
            if line == "READY":
                return
        self.cmd("Q")  # 이미 켜져 있던 경우

    def cmd(self, c):
        self.s.write((c + "\n").encode())
        line = self.s.readline().decode(errors="ignore").strip()
        if not line.startswith(("OK", "NTC")):
            raise RuntimeError(f"UNO: {c!r} → {line!r}")
        return line

    def state(self, line=None):
        line = line or self.cmd("Q")
        out = {}
        for tok in line.split()[1:]:
            k, v = tok.split("=")
            out[k] = float(v)
        return out

    def configure(self):
        for ax in "PT":
            self.cmd(f"S {ax} {STEPS_PER_REV[ax]}")
            self.cmd(f"B {ax} {BACKLASH_DEG[ax]}")
        self.cmd(f"D {MIN_STEP_US}")

    def approach(self, ax, deg):
        return self.state(self.cmd(f"A {ax} {deg:.3f}"))

    def move(self, ax, deg):
        return self.state(self.cmd(f"M {ax} {deg:.3f}"))

    def release(self):
        try:
            self.cmd("R")
        except Exception:
            pass

    def motor_temp(self):
        raw = int(self.cmd("N").split()[1])
        raw = min(max(raw, 1), 1022)
        r = 10000.0 * raw / (1023 - raw)            # 5V—10k—A0—NTC—GND
        inv_t = 1 / 298.15 + math.log(r / 10000.0) / 3950.0
        return 1 / inv_t - 273.15


class Cam:
    label = ""

    def __init__(self, port):
        s = serial.Serial()
        s.port, s.baudrate, s.timeout = port, 921600, 10
        s.dtr = False      # 포트 열 때 리셋 방지
        s.rts = False
        s.open()
        self.s = s
        time.sleep(0.5)
        self.s.reset_input_buffer()
        info = self.line("I")
        print("ESP32:", info)
        if "cam=0" in info:
            # 카메라가 꺼져 있으면 재초기화 시도 (부팅 시 init 실패 대비)
            for _ in range(3):
                r = self.line("W")
                print("  카메라 재시작:", r)
                if r.startswith("OK"):
                    break
                time.sleep(1)
            else:
                raise RuntimeError(
                    "카메라 초기화 실패 — Sense 확장보드 커넥터를 다시 눌러 끼우고 "
                    "XIAO의 R 버튼을 누른 뒤 다시 실행하세요 (0x105 = 센서 미검출)")

    def line(self, c):
        self.s.write((c + "\n").encode())
        return self.s.readline().decode(errors="ignore").strip()

    def temp(self):
        return float(self.line("T").split()[1])

    def capture(self, retries=3):
        for attempt in range(retries):
            self.s.reset_input_buffer()
            self.s.write(b"C\n")
            head = self.s.readline().decode(errors="ignore").strip()
            if not head.startswith("IMG"):
                print(f"  ! 헤더 오류: {head!r}")
                continue
            _, n, t = head.split()
            n = int(n)
            data = self.s.read(n)
            tail = self.s.readline() + self.s.readline()
            if len(data) == n and data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9":
                return data, float(t)
            print(f"  ! JPEG 손상 (시도 {attempt + 1})")
        raise RuntimeError("촬영 실패")

    def sleep(self):
        self.line("S")

    def wake(self):
        r = self.line("W")
        if not r.startswith("OK"):
            raise RuntimeError(r)


class FakeCam:
    """카메라 없이 모터만 시험할 때 쓰는 가짜 카메라 (--sim).
    실제 촬영·전송 시간을 흉내 내고, 각도가 적힌 회색 이미지를 만들어 저장한다.
    → 촬영 순서·이동 시간·모터 발열·탈조를 카메라 없이 확인할 수 있다."""
    fake = True

    def __init__(self, capture_s=1.2):
        self.capture_s = capture_s   # UXGA JPEG 촬영+USB 전송에 걸리는 대략적 시간
        self.label = ""
        print("가짜 카메라 사용 (--sim): 모터 동작만 시험합니다")

    def line(self, c):
        return "OK sim"

    def temp(self):
        return 0.0

    def capture(self, retries=3):
        time.sleep(self.capture_s)
        if HAVE_CV:
            img = np.full((600, 800, 3), 90, np.uint8)
            cv2.putText(img, self.label or "sim", (30, 320),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
            ok, buf = cv2.imencode(".jpg", img)
            return buf.tobytes(), 0.0
        return b"\xff\xd8\xff\xd9", 0.0   # 최소 JPEG 껍데기

    def sleep(self):
        pass

    def wake(self):
        pass


# ------------------------------------------------------------------ 보조 기능
def drift_px(jpg_a, jpg_b):
    if not HAVE_CV:
        return None
    imgs = []
    for j in (jpg_a, jpg_b):
        im = cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_GRAYSCALE)
        scale = 800 / im.shape[1]
        im = cv2.resize(im, None, fx=scale, fy=scale)
        imgs.append(np.float32(im))
    (dx, dy), _ = cv2.phaseCorrelate(imgs[0], imgs[1])
    return math.hypot(dx, dy)


def cool_camera(cam):
    t = cam.temp()
    if t < ESP_PAUSE_C:
        return False
    print(f"  카메라 {t:.1f}°C → 정지 후 냉각")
    cam.sleep()
    while t > ESP_RESUME_C:
        time.sleep(5)
        t = cam.temp()
        print(f"    {t:.1f}°C", end="\r")
    print()
    cam.wake()
    return True   # 재시작했으니 노출 재고정 필요


def cool_motor(rig):
    if not USE_NTC:
        return
    t = rig.motor_temp()
    if t < MOTOR_PAUSE_C:
        return
    print(f"  모터 {t:.1f}°C → 냉각 대기")
    while t > MOTOR_RESUME_C:
        time.sleep(5)
        t = rig.motor_temp()


def lock_exposure(rig, cam):
    """수평 정면(팬 0, 틸트 0)에서 자동노출을 안정시킨 뒤 고정"""
    if not USE_EXPOSURE_LOCK or getattr(cam, "fake", False):
        return
    rig.approach("T", 0)
    rig.approach("P", 0)
    cam.line("U")
    for _ in range(3):
        cam.capture()
        time.sleep(0.6)
    print(" ", cam.line("L"))


def jog_mode(rig):
    print("조그 모드: 예) p+200, t-50 (하프스텝), 빈 줄 = 완료 후 현재 위치를 0으로")
    while True:
        s = input("> ").strip().lower()
        if not s:
            break
        try:
            ax, n = s[0].upper(), int(s[1:])
            print(rig.cmd(f"J {ax} {n}"))
        except Exception as e:
            print("형식 오류:", e)
    print(rig.cmd("Z"))


# ------------------------------------------------------------------ 스캔
def shoot_row(rig, cam, tilt, n_pan, outdir, log):
    st = rig.approach("T", tilt)
    first = None
    for i in range(n_pan):
        pan = i * 360.0 / n_pan
        st = rig.approach("P", pan) if i == 0 else rig.move("P", pan)
        time.sleep(SETTLE_S)
        name = f"t{tilt:+04d}_p{pan:06.2f}.jpg"
        cam.label = f"tilt {st['T']:+.1f}  pan {st['P']:.1f}"
        jpg, t = cam.capture()
        (outdir / name).write_bytes(jpg)
        log.append({"file": name, "tilt_deg": st["T"], "pan_deg": st["P"],
                    "tilt_steps": st["TS"], "pan_steps": st["PS"],
                    "esp_temp_c": t, "time": time.time()})
        temp_s = "" if getattr(cam, "fake", False) else f"  ESP {t:.1f}°C"
        print(f"  {name}  (실제 P={st['P']:.2f}° T={st['T']:.2f}°){temp_s}")
        first = first or jpg
        if t > ESP_ABORT_C:
            rig.approach("P", 0)
            return False

    if n_pan > 1 and getattr(cam, "fake", False):
        rig.move("P", 360)          # 실제 스캔과 같은 동작 (360°까지 돈 뒤 되감기)
        time.sleep(SETTLE_S)
    elif n_pan > 1:
        # 360°에서 확인용 1장 → 0° 사진과 비교해 탈조 검사
        rig.move("P", 360)
        time.sleep(SETTLE_S)
        check, _ = cam.capture()
        d = drift_px(first, check)
        if d is not None:
            flag = "  ← 탈조 의심!" if d > DRIFT_WARN_PX else ""
            print(f"  0°/360° 어긋남: {d:.1f}px{flag}")
            log[-1]["drift_px_row"] = d
    rig.approach("P", 0)          # 370° 역회전 + 10° 정회전
    return True


def scan(rig, cam):
    tag = "_sim" if getattr(cam, "fake", False) else ""
    outdir = Path("scans") / (datetime.now().strftime("%Y%m%d_%H%M%S") + tag)
    outdir.mkdir(parents=True)
    log = []
    meta = {"rows": ROWS, "steps_per_rev": STEPS_PER_REV,
            "backlash_deg": BACKLASH_DEG, "shots": log}

    lock_exposure(rig, cam)
    t_start = time.time()
    try:
        r = 0
        while r < len(ROWS):
            tilt, n = ROWS[r]
            print(f"[행 {r + 1}/{len(ROWS)}] 틸트 {tilt:+d}°, {n}장")
            n_before = len(log)
            ok = shoot_row(rig, cam, tilt, n, outdir, log)
            rig.release()

            if cool_camera(cam):
                lock_exposure(rig, cam)
            cool_motor(rig)

            if not ok:
                print("  과열로 이 행을 다시 촬영합니다")
                del log[n_before:]
                continue
            r += 1
            if r < len(ROWS):
                time.sleep(ROW_REST_S)

        rig.approach("T", 0)
        rig.approach("P", 0)
    finally:
        rig.release()
        (outdir / "shots.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(f"저장: {outdir}  ({len(log)}장, {(time.time() - t_start) / 60:.1f}분)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uno", required=True)
    ap.add_argument("--esp")
    ap.add_argument("--jog", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--calib", nargs=2, metavar=("AXIS", "REVS"))
    ap.add_argument("--sim", action="store_true",
                    help="카메라 없이 모터만으로 전체 스캔 동작을 시뮬레이션")
    a = ap.parse_args()

    rig = Rig(a.uno)
    rig.configure()

    try:
        if a.calib:
            ax, revs = a.calib[0].upper(), int(a.calib[1])
            steps = round(STEPS_PER_REV[ax] * revs)
            input(f"{ax}축에 표시를 하고 Enter → {steps} 스텝 회전")
            print(rig.cmd(f"J {ax} {steps}"))
            print("표시가 정확히 제자리면 OK. 어긋났으면 어긋난 각도 δ(°, 더 돌았으면 +)를 재서")
            print(f"  새 값 = {STEPS_PER_REV[ax]} × 360·{revs} / (360·{revs} + δ)")
            return

        if a.jog:
            jog_mode(rig)

        if a.sim:
            cam = FakeCam()
        elif not a.esp:
            return
        else:
            cam = Cam(a.esp)

        if a.test:
            jpg, t = cam.capture()
            Path("test.jpg").write_bytes(jpg)
            print(f"test.jpg 저장, ESP {t:.1f}°C")
            return

        scan(rig, cam)
    except KeyboardInterrupt:
        print("\n중단됨")
    finally:
        rig.release()


if __name__ == "__main__":
    sys.exit(main())
