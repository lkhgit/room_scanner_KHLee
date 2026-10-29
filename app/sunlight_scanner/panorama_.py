#!/usr/bin/env python3
"""
팬·틸트 리그 파노라마 촬영 (PC 측)

  1단계  수평 맞추기: 미리보기를 보며 키보드로 모터를 직접 움직여 카메라를 수평으로 맞춤
  2단계  자동 촬영: 틸트 행(row)마다 360° 돌며 촬영. 위쪽은 40°까지만, 바닥 쪽은 찍지 않음
  결과   captures/<날짜_시간>/ 에 JPEG + 각 사진의 yaw/pitch가 담긴 meta.json
         방위 센서(GY-BNO055)가 있으면 정북 방향도 meta.json의 "north"에 저장
         - 프로그램을 켜기 전에 카메라가 센서 X축 방향을 바라보도록 세팅해 둘 것
         - 북쪽은 모터를 한 번도 움직이기 전(연결 직후)에 잰다
         - 수평 맞추기에서 팬을 돌린 양은 스텝 수로 계산해 반영한다

설치:  pip install pyserial opencv-python numpy
실행:  python panorama.py --arduino COM5 --camera COM6     (macOS/Linux는 /dev/tty... )
"""
import argparse
import datetime
import json
import math
import os
import time

import cv2
import numpy as np
import serial

# ===================== 하드웨어 설정 =====================
STEPS_PER_REV = 4096      # 28BYJ-48 하프스텝 기준 출력축 1회전 (개체에 따라 실측 ~4076)
PAN_DIR = 1               # 'D'를 눌렀을 때 반대로 돌면 -1
TILT_DIR = -1             # 'W'를 눌렀을 때 카메라가 아래를 보면 부호 반대로
STEP_DELAY_US = 1500      # 하프스텝 간격 (탈조되면 키우기)

# ===================== 촬영 설정 =====================
HFOV_DEG = 80.0           # 카메라 가로 화각 (이어붙이기 결과로 맞춘 값)
VFOV_DEG = 65.0           # 카메라 세로 화각
OVERLAP = 0.30            # 이웃한 사진끼리 겹치는 비율
ROW_PITCHES = [-30, 0, 40]   # 촬영할 틸트 각도 (0 = 수평). 덮는 범위 약 -62° ~ +72°
MIN_PITCH_DEG = -35       # 이보다 아래로는 절대 촬영하지 않음 (바닥 쪽 제외)
MAX_PITCH_DEG = 75        # 이보다 위로는 올리지 않음 (줄 걸림 방지)
PAN_BACKLASH_DEG = 10     # 백래시 보정: 항상 같은 방향(정방향)으로 목표에 도착하도록 넘겼다가 돌아옴
TILT_BACKLASH_DEG = 5
SETTLE_S = 0.8            # 이동 후 흔들림이 멎을 때까지 대기
PREVIEW_SIZE = "VGA"
CAPTURE_SIZE = "UXGA"

# ===================== 방위(북쪽) 설정 =====================
MAG_DECLINATION_DEG = -8.6   # 자편각 (동쪽 +). 서울은 자북이 진북보다 약 8~9° 서쪽 -> 음수
HEADING_SAMPLES = 30         # 북쪽을 잴 때 평균 낼 측정 횟수


def deg_to_steps(deg):
    return int(round(deg * STEPS_PER_REV / 360.0))


def steps_to_deg(steps):
    return steps * 360.0 / STEPS_PER_REV


def heading_of_x_axis(m, g):
    """센서 좌표의 자기장 m, 중력 g(위쪽)로 센서 X축이 가리키는 방위각(자북 기준, 시계 방향 0~360°).
    기울어져 있어도 맞게 나오도록 수평면에 투영해서 계산 (센서 내부의 축 규약과 무관)."""
    up = g / np.linalg.norm(g)
    east = np.cross(m, up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    return math.degrees(math.atan2(east[0], north[0])) % 360


# ---------------------------------------------------------
class Rig:
    """아두이노 팬·틸트 리그. 위치는 PC가 하프스텝 단위로 추적한다."""

    def __init__(self, port):
        self.ser = serial.Serial(port, 115200, timeout=0.5)
        time.sleep(2.0)                       # UNO는 포트를 열면 리셋됨
        self._wait_for("READY", 8, ping=True)
        self.pan = 0
        self.tilt = 0
        self.cmd(f"S {STEP_DELAY_US}")
        self.cmd("H 0")
        self.imu_ok, self.imu_calib_loaded = self.imu_status()
        if not self.imu_ok:
            print("방위 센서: 없음 (북쪽은 저장되지 않음)")
        else:
            print("방위 센서: 연결됨, 저장된 지자기 보정값 " +
                  ("불러옴" if self.imu_calib_loaded else "없음 ※ bno055_compass_save.ino로 먼저 보정·저장하세요"))

    def imu_status(self):
        """(센서 있음, EEPROM 보정값 불러옴)"""
        self.ser.reset_input_buffer()
        self.ser.write(b"I\n")
        deadline = time.time() + 1.0
        while time.time() < deadline:
            line = self.ser.readline().decode(errors="ignore").strip()
            if line.startswith("ERR"):
                return False, False
            if line.startswith("I "):
                parts = line.split()
                return True, len(parts) > 2 and parts[2] == "1"
        return False, False

    # ---------------- 방위 센서 ----------------
    def read_imu(self):
        """센서 값 1회. 실패하면 None.
        반환: dict(heading=센서 X축의 자북 기준 방위각(시계 방향, °), euler=센서 자체 heading, cal=(sys,gyro,acc,mag))"""
        self.ser.reset_input_buffer()
        self.ser.write(b"N\n")
        deadline = time.time() + 1.0
        while time.time() < deadline:
            line = self.ser.readline().decode(errors="ignore").strip()
            if line.startswith("ERR"):
                return None
            if line.startswith("N "):
                v = [int(x) for x in line.split()[1:]]
                if len(v) != 8:
                    return None
                m = np.array(v[0:3], float) / 16.0       # µT (보정값 적용됨)
                g = np.array(v[3:6], float) / 100.0      # 가속도 m/s² (가만히 있을 때 위쪽을 가리킴)
                cal = v[7]
                if np.linalg.norm(g) < 5 or np.linalg.norm(m) < 5:
                    return None
                return dict(heading=heading_of_x_axis(m, g), euler=v[6] / 16.0,
                            cal=((cal >> 6) & 3, (cal >> 4) & 3, (cal >> 2) & 3, cal & 3),
                            mag=m.tolist(), grav=g.tolist())
        return None

    def measure_heading(self, n=HEADING_SAMPLES):
        """모터 전류를 끊고 (코일 전류가 지자기를 왜곡하지 않게) n번 재서 평균"""
        if not self.imu_ok:
            return None
        self.release()
        time.sleep(0.3)
        rs = []
        for _ in range(n):
            r = self.read_imu()
            if r:
                rs.append(r)
            time.sleep(0.03)
        if len(rs) < n // 2:
            return None
        h = np.radians([r["heading"] for r in rs])
        c, s = np.cos(h).mean(), np.sin(h).mean()
        mean = math.degrees(math.atan2(s, c)) % 360
        spread = math.degrees(math.sqrt(max(0.0, -2 * math.log(max(math.hypot(c, s), 1e-9)))))
        return dict(heading=mean, spread=spread, euler=rs[-1]["euler"], cal=rs[-1]["cal"], n=len(rs))

    def _wait_for(self, token, timeout, ping=False):
        deadline = time.time() + timeout
        if ping:
            self.ser.write(b"?\n")
        while time.time() < deadline:
            line = self.ser.readline().decode(errors="ignore").strip()
            if line == token:
                return
            if ping and not line:
                self.ser.write(b"?\n")
        raise RuntimeError(f"아두이노 응답 없음 (기다린 응답: {token})")

    def cmd(self, line, timeout=60):
        self.ser.reset_input_buffer()
        self.ser.write((line + "\n").encode())
        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = self.ser.readline().decode(errors="ignore").strip()
            if resp == "OK":
                return
            if resp.startswith("ERR"):
                raise RuntimeError(f"아두이노 오류: {line} -> {resp}")
        raise RuntimeError(f"아두이노 타임아웃: {line}")

    # 상대 이동 (하프스텝)
    def move_pan(self, n):
        if n:
            self.cmd(f"P {PAN_DIR * n}")
            self.pan += n

    def move_tilt(self, n):
        if n:
            self.cmd(f"T {TILT_DIR * n}")
            self.tilt += n

    # 절대 이동 (도) - 목표에 항상 정방향으로 도착하도록 백래시 보정
    def pan_to(self, deg):
        diff = deg_to_steps(deg) - self.pan
        if diff < 0:
            bl = deg_to_steps(PAN_BACKLASH_DEG)
            self.move_pan(diff - bl)
            self.move_pan(bl)
        else:
            self.move_pan(diff)

    def tilt_to(self, deg):
        if deg < MIN_PITCH_DEG:
            raise ValueError(f"틸트 {deg}°는 최저 한계 {MIN_PITCH_DEG}°보다 낮음")
        if deg > MAX_PITCH_DEG:
            raise ValueError(f"틸트 {deg}°는 최고 한계 {MAX_PITCH_DEG}°보다 높음")
        diff = deg_to_steps(deg) - self.tilt
        if diff < 0:
            bl = deg_to_steps(TILT_BACKLASH_DEG)
            self.move_tilt(diff - bl)
            self.move_tilt(bl)
        else:
            self.move_tilt(diff)

    def set_zero(self):
        self.pan = 0
        self.tilt = 0

    def release(self):
        try:
            self.cmd("R", timeout=3)
        except Exception:
            pass

    def close(self):
        self.release()
        self.ser.close()


# ---------------------------------------------------------
class Camera:
    """XIAO ESP32S3 Sense - USB 시리얼로 JPEG 수신."""

    def __init__(self, port):
        self.ser = serial.Serial(port, 921600, timeout=2)
        time.sleep(1.5)
        self.ser.reset_input_buffer()
        for _ in range(10):
            self.ser.write(b"?\n")
            line = self.ser.readline().decode(errors="ignore").strip()
            if line == "READY":
                return
            if line.startswith("ERR"):
                raise RuntimeError(f"카메라 오류: {line} (리본 커넥터·PSRAM 설정 확인)")
        raise RuntimeError("카메라 응답 없음 (포트, USB CDC On Boot 설정 확인)")

    def _read_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.ser.read(n - len(buf))
            if not chunk:
                raise RuntimeError("이미지 수신 중 타임아웃")
            buf.extend(chunk)
        return bytes(buf)

    def set_size(self, name):
        self.ser.reset_input_buffer()
        self.ser.write(f"F {name}\n".encode())
        deadline = time.time() + 5
        while time.time() < deadline:
            line = self.ser.readline().decode(errors="ignore").strip()
            if line == "OK":
                return
            if line.startswith("ERR"):
                raise RuntimeError(f"해상도 변경 실패: {name}")
        raise RuntimeError("해상도 변경 타임아웃")

    def grab_jpeg(self):
        self.ser.reset_input_buffer()
        self.ser.write(b"C\n")
        deadline = time.time() + 10
        while time.time() < deadline:
            line = self.ser.readline()
            if line.startswith(b"IMG "):
                n = int(line[4:].strip())
                return self._read_exact(n)
            if line.startswith(b"ERR"):
                raise RuntimeError(line.decode(errors="ignore").strip())
        raise RuntimeError("촬영 응답 타임아웃")

    def grab_bgr(self):
        data = self.grab_jpeg()
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    def close(self):
        self.ser.close()


# ---------------------------------------------------------
def put_text(img, lines):
    for i, t in enumerate(lines):
        y = 22 + 22 * i
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def leveling(rig, cam, start_azimuth=None):
    """키보드로 모터를 움직여 카메라를 수평으로 맞춘다. Enter = 확정, Q/ESC = 종료.
    반환: None(취소) 또는 수평 맞추기 동안 팬을 돌린 각도(°, 위에서 봐서 시계 방향 +)"""
    cam.set_size(PREVIEW_SIZE)
    step_presets = {ord("1"): 4, ord("2"): 23, ord("3"): 114}   # 약 0.35°, 2°, 10°
    step = 23
    win = "Leveling"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        img = cam.grab_bgr()
        if img is None:
            continue
        h, w = img.shape[:2]
        # 가운데 수평선 + 세로 기준선 (방의 세로 모서리가 이 선들과 평행하면 수평)
        cv2.line(img, (0, h // 2), (w, h // 2), (0, 255, 0), 1)
        for x in (w // 4, w // 2, 3 * w // 4):
            cv2.line(img, (x, 0), (x, h), (0, 220, 255), 1)
        lines = [
            "W/S: tilt up/down   A/D: pan left/right",
            f"1/2/3: step size  (now {steps_to_deg(step):.2f} deg)",
            f"tilt offset {steps_to_deg(rig.tilt):+.1f} deg  pan {steps_to_deg(rig.pan):+.1f} deg",
            "ENTER: level OK -> start capture   Q/ESC: quit",
        ]
        if start_azimuth is not None:
            az = (start_azimuth + steps_to_deg(rig.pan)) % 360
            lines.append(f"camera azimuth {az:5.1f} deg (true north, measured before motors moved)")
        put_text(img, lines)
        cv2.imshow(win, img)
        k = cv2.waitKey(1) & 0xFF

        if k == ord("w"):
            rig.move_tilt(step)
        elif k == ord("s"):
            rig.move_tilt(-step)
        elif k == ord("a"):
            rig.move_pan(-step)
        elif k == ord("d"):
            rig.move_pan(step)
        elif k in step_presets:
            step = step_presets[k]
        elif k in (13, 10):            # Enter
            cv2.destroyWindow(win)
            pan_offset = steps_to_deg(rig.pan)
            rig.set_zero()             # 현재 방향 = yaw 0°, pitch 0°
            return pan_offset
        elif k in (ord("q"), 27):
            cv2.destroyWindow(win)
            return None


def plan_rows():
    """행마다 360°를 몇 장으로 나눌지 계산. 수평에서 먼 행일수록 장 수가 줄어든다."""
    eff_h = HFOV_DEG * (1 - OVERLAP)
    rows = []
    for p in sorted(ROW_PITCHES):
        if p < MIN_PITCH_DEG or p > MAX_PITCH_DEG:
            continue
        near = max(0.0, abs(p) - VFOV_DEG / 2)   # 그 행에서 수평에 가장 가까운 가장자리
        n = max(1, math.ceil(360 * math.cos(math.radians(near)) / eff_h))
        rows.append((p, n))
    return rows


def north_info(start, pan_offset_deg, calib_loaded):
    """start: 모터를 움직이기 전 잰 값 (그때 카메라 = 센서 X축 방향)
    pan_offset_deg: 그 뒤 수평 맞추기에서 팬을 돌린 각도 (Enter를 누른 곳이 yaw 0°)
    yaw는 위에서 볼 때 시계 방향으로 증가 (D 키 방향; 이어붙이기가 정상이면 이 방향이 맞음)."""
    if start is None:
        return None
    start_true = (start["heading"] + MAG_DECLINATION_DEG) % 360   # 시작 시 카메라의 정북 기준 방위각
    true0 = (start_true + pan_offset_deg) % 360                   # yaw 0°일 때 카메라의 정북 기준 방위각
    north_yaw = (-true0) % 360                                    # 카메라가 진북을 보는 yaw
    lon = (north_yaw + 180) % 360 - 180                           # 파노라마 경도(-180~180)
    info = {
        "assumption": "프로그램 시작 시(모터 이동 전) 카메라가 GY-BNO055 X축 방향을 바라봄",
        "camera_azimuth_at_yaw0_deg": true0,          # 정북 기준, 시계 방향
        "start_camera_azimuth_magnetic_deg": start["heading"],   # 모터 이동 전, 자북 기준
        "leveling_pan_offset_deg": pan_offset_deg,
        "declination_deg": MAG_DECLINATION_DEG,
        "north_yaw_deg": north_yaw,                   # 이 yaw에서 카메라가 진북을 봄
        "north_pano_u": (lon + 180) / 360,            # stitch_panorama.py 결과에서 진북의 가로 위치 (0=왼쪽 끝, 1=오른쪽 끝)
        "spread_deg": start["spread"],
        "calib_loaded_from_eeprom": calib_loaded,
        "calib_status": dict(zip(["sys", "gyro", "accel", "mag"], start["cal"])),
        "sensor_euler_heading_deg": start["euler"],
    }
    return info


def save_meta(meta, out_dir):
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def capture(rig, cam, out_dir, north=None):
    rows = plan_rows()
    total = sum(n for _, n in rows)
    print("촬영 계획:", ", ".join(f"{p:+d}° x {n}장" for p, n in rows), f"(총 {total}장)")
    if north:
        print(f"정북: yaw 0° 카메라 방위 {north['camera_azimuth_at_yaw0_deg']:.1f}°, "
              f"카메라가 정북을 보는 yaw {north['north_yaw_deg']:.1f}°")

    cam.set_size(CAPTURE_SIZE)
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "hfov_deg": HFOV_DEG, "vfov_deg": VFOV_DEG,
        "steps_per_rev": STEPS_PER_REV, "north": north, "images": [],
    }
    win = "Capture"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    shot = 0

    # 아래 행부터 위로 -> 틸트는 항상 위쪽으로 이동하므로 백래시 방향이 일정
    for ri, (pitch, n) in enumerate(rows):
        rig.tilt_to(pitch)
        for ci in range(n):
            rig.pan_to(ci * 360.0 / n)       # 한 행 안에서는 항상 정방향으로만 회전
            time.sleep(SETTLE_S)
            data = cam.grab_jpeg()

            fname = f"r{ri:02d}_c{ci:02d}.jpg"
            with open(os.path.join(out_dir, fname), "wb") as f:
                f.write(data)
            item = {
                "file": fname, "row": ri, "col": ci,
                "yaw_deg": steps_to_deg(rig.pan),      # 실제 스텝 수로 환산한 각도
                "pitch_deg": steps_to_deg(rig.tilt),
            }
            if north:                                  # 이 사진의 카메라 방위각 (진북 기준, 시계 방향)
                item["azimuth_deg"] = (north["camera_azimuth_at_yaw0_deg"] + item["yaw_deg"]) % 360
            meta["images"].append(item)
            shot += 1
            print(f"[{shot}/{total}] pitch {pitch:+d}°  yaw {steps_to_deg(rig.pan):6.1f}°  -> {fname}")

            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                cv2.imshow(win, cv2.resize(img, (640, int(640 * img.shape[0] / img.shape[1]))))
            if (cv2.waitKey(1) & 0xFF) == 27:
                raise KeyboardInterrupt

        # 행이 끝나면 역방향으로 되감아 선 꼬임을 풀고, 정방향으로 0°에 도착
        rig.pan_to(0)

        save_meta(meta, out_dir)

    cv2.destroyWindow(win)
    print(f"완료: {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arduino", required=True, help="아두이노 포트 (예: COM5)")
    ap.add_argument("--camera", required=True, help="XIAO 포트 (예: COM6)")
    ap.add_argument("--out", default="captures")
    args = ap.parse_args()

    rig = cam = None
    try:
        rig = Rig(args.arduino)

        # 북쪽은 모터를 한 번도 움직이기 전에 잰다 (이때 카메라 = 센서 X축 방향)
        start = rig.measure_heading() if rig.imu_ok else None
        start_az = None
        if start:
            start_az = (start["heading"] + MAG_DECLINATION_DEG) % 360
            print(f"정북 측정 (모터 이동 전): 카메라 방위 {start_az:.1f}°  "
                  f"(자북 기준 {start['heading']:.1f}°, 흔들림 ±{start['spread']:.1f}°, 지자기 보정 상태 {start['cal'][3]}/3)")
            if start["spread"] > 2:
                print("  ※ 측정값이 흔들립니다. 근처 금속·전자기기를 치우고 다시 실행하는 것을 권장합니다.")
        elif rig.imu_ok:
            print("※ 방위 센서를 읽지 못해 북쪽을 저장하지 않습니다.")

        cam = Camera(args.camera)
        pan_offset = leveling(rig, cam, start_az)
        if pan_offset is None:
            print("취소됨")
            return
        north = north_info(start, pan_offset, rig.imu_calib_loaded) if start else None
        out_dir = os.path.join(args.out, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        capture(rig, cam, out_dir, north)
    except KeyboardInterrupt:
        print("\n중단됨")
    finally:
        if rig:
            try:                       # 수평·정면 위치로 복귀
                rig.pan_to(0)
                rig.tilt_to(0)
            except Exception as e:
                print("원위치 복귀 실패:", e)
            rig.close()
        if cam:
            cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
