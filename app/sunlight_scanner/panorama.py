#!/usr/bin/env python3
"""
Pan-tilt rig panorama capture (PC side)

  Step 1  Leveling: move the motors with the keyboard while watching the preview until the camera is level
  Step 2  Automatic capture: one 360° turn per tilt row. Upwards only to 40°, the floor is not photographed
  Result  JPEGs in captures/<date_time>/ plus meta.json with each photo's yaw/pitch
          With a compass sensor (GY-BNO055), true north is also saved under "north" in meta.json
          - before starting the program, aim the camera along the sensor's X axis
          - north is measured before the motors move at all (right after connecting)
          - the pan turned during leveling is counted from its steps and taken into account

Install: pip install pyserial opencv-python numpy
Run:     python panorama.py --arduino COM5 --camera COM6     (on macOS/Linux, /dev/tty... )
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

# ===================== hardware settings =====================
STEPS_PER_REV = 4096      # one output shaft turn of a 28BYJ-48 in half steps (measures ~4076 on some units)
PAN_DIR = 1               # -1 if pressing 'D' turns the wrong way
TILT_DIR = -1             # flip the sign if pressing 'W' makes the camera look down
STEP_DELAY_US = 1500      # time between half steps (increase if the motor skips steps)

# ===================== capture settings =====================
HFOV_DEG = 80.0           # horizontal field of view of the camera (tuned from stitching results)
VFOV_DEG = 65.0           # vertical field of view of the camera
OVERLAP = 0.30            # overlap between neighbouring photos
ROW_PITCHES = [-30, 0, 40]   # tilt angles to capture (0 = level). Covers about -62° to +72°
MIN_PITCH_DEG = -35       # never capture below this (leaves out the floor)
MAX_PITCH_DEG = 75        # never tilt above this (keeps the cable from catching)
PAN_BACKLASH_DEG = 10     # backlash: overshoot and come back so the target is always reached in the same (forward) direction
TILT_BACKLASH_DEG = 5
SETTLE_S = 0.8            # wait after a move until the shaking stops
PREVIEW_SIZE = "VGA"
CAPTURE_SIZE = "UXGA"

# ===================== heading (north) settings =====================
MAG_DECLINATION_DEG = -8.6   # magnetic declination (east +). In Seoul magnetic north is about 8-9° west of true north -> negative
HEADING_SAMPLES = 30         # number of readings averaged when measuring north


def deg_to_steps(deg):
    """Converts degrees to motor half steps."""
    return int(round(deg * STEPS_PER_REV / 360.0))


def steps_to_deg(steps):
    """Converts motor half steps to degrees."""
    return steps * 360.0 / STEPS_PER_REV


def heading_of_x_axis(m, g):
    """Bearing of the sensor's X axis (magnetic north, clockwise 0-360°) from the magnetic field m and gravity g (pointing up)
    in sensor coordinates. Projected onto the horizontal plane so it is right even when tilted (independent of the sensor's own axis conventions)."""
    up = g / np.linalg.norm(g)
    east = np.cross(m, up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    return math.degrees(math.atan2(east[0], north[0])) % 360


# ---------------------------------------------------------
class Rig:
    """Arduino pan-tilt rig. The PC keeps track of the position in half steps."""

    def __init__(self, port):
        """Opens the rig's serial port, waits for READY, sets the step speed and checks for the compass sensor."""
        self.ser = serial.Serial(port, 115200, timeout=0.5)
        time.sleep(2.0)                       # the UNO resets when the port is opened
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
                  ("불러옴" if self.imu_calib_loaded else "없음 ※ bno055_compass.ino로 먼저 보정·저장하세요"))

    def imu_status(self):
        """Returns (sensor present, calibration loaded from EEPROM)."""
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

    # ---------------- compass sensor ----------------
    def read_imu(self):
        """One sensor reading, or None on failure.
        Returns dict(heading=bearing of the sensor's X axis from magnetic north (clockwise, °), euler=the sensor's own heading, cal=(sys,gyro,acc,mag))"""
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
                m = np.array(v[0:3], float) / 16.0       # µT (calibration applied)
                g = np.array(v[3:6], float) / 100.0      # acceleration m/s² (points up when at rest)
                cal = v[7]
                if np.linalg.norm(g) < 5 or np.linalg.norm(m) < 5:
                    return None
                return dict(heading=heading_of_x_axis(m, g), euler=v[6] / 16.0,
                            cal=((cal >> 6) & 3, (cal >> 4) & 3, (cal >> 2) & 3, cal & 3),
                            mag=m.tolist(), grav=g.tolist())
        return None

    def measure_heading(self, n=HEADING_SAMPLES):
        """Switches the motor coils off (so their current does not disturb the magnetic field) and averages n readings."""
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
        """Waits for a given reply line, optionally pinging with ? until the Arduino answers."""
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
        """Sends one command and waits for OK, raising an error on ERR or timeout."""
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

    # relative moves (half steps)
    def move_pan(self, n):
        """Moves the pan motor by n half steps and updates the tracked position."""
        if n:
            self.cmd(f"P {PAN_DIR * n}")
            self.pan += n

    def move_tilt(self, n):
        """Moves the tilt motor by n half steps and updates the tracked position."""
        if n:
            self.cmd(f"T {TILT_DIR * n}")
            self.tilt += n

    # absolute moves (degrees) - backlash compensated so the target is always reached in the forward direction
    def pan_to(self, deg):
        """Turns the pan to an absolute angle, always arriving in the forward direction."""
        diff = deg_to_steps(deg) - self.pan
        if diff < 0:
            bl = deg_to_steps(PAN_BACKLASH_DEG)
            self.move_pan(diff - bl)
            self.move_pan(bl)
        else:
            self.move_pan(diff)

    def tilt_to(self, deg):
        """Tilts to an absolute angle within the allowed range, always arriving in the forward direction."""
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
        """Makes the current position yaw 0°, pitch 0°."""
        self.pan = 0
        self.tilt = 0

    def release(self):
        """Switches the motor coils off, ignoring errors."""
        try:
            self.cmd("R", timeout=3)
        except Exception:
            pass

    def close(self):
        """Switches the coils off and closes the port."""
        self.release()
        self.ser.close()


# ---------------------------------------------------------
class Camera:
    """XIAO ESP32S3 Sense - receives JPEGs over USB serial."""

    def __init__(self, port):
        """Opens the camera's serial port and waits for READY."""
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
        """Reads exactly n bytes from the camera or fails on timeout."""
        buf = bytearray()
        while len(buf) < n:
            chunk = self.ser.read(n - len(buf))
            if not chunk:
                raise RuntimeError("이미지 수신 중 타임아웃")
            buf.extend(chunk)
        return bytes(buf)

    def set_size(self, name):
        """Sets the camera resolution (e.g. VGA, UXGA)."""
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
        """Captures one photo and returns the JPEG bytes."""
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
        """Captures one photo as an OpenCV BGR image."""
        data = self.grab_jpeg()
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    def close(self):
        """Closes the camera port."""
        self.ser.close()


# ---------------------------------------------------------
def put_text(img, lines):
    """Draws lines of white text with a black outline at the top left of an image."""
    for i, t in enumerate(lines):
        y = 22 + 22 * i
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def leveling(rig, cam, start_azimuth=None):
    """Levels the camera by moving the motors from the keyboard. Enter = confirm, Q/ESC = quit.
    Returns None (cancelled) or the pan angle turned while leveling (°, clockwise seen from above is +)"""
    cam.set_size(PREVIEW_SIZE)
    step_presets = {ord("1"): 4, ord("2"): 23, ord("3"): 114}   # about 0.35°, 2°, 10°
    step = 23
    win = "Leveling"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        img = cam.grab_bgr()
        if img is None:
            continue
        h, w = img.shape[:2]
        # horizontal centre line + vertical guides (level when the room's vertical edges run parallel to them)
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
            rig.set_zero()             # current direction = yaw 0°, pitch 0°
            return pan_offset
        elif k in (ord("q"), 27):
            cv2.destroyWindow(win)
            return None


def plan_rows():
    """Works out how many photos each row needs for 360°. Rows farther from level need fewer."""
    eff_h = HFOV_DEG * (1 - OVERLAP)
    rows = []
    for p in sorted(ROW_PITCHES):
        if p < MIN_PITCH_DEG or p > MAX_PITCH_DEG:
            continue
        near = max(0.0, abs(p) - VFOV_DEG / 2)   # the row's edge closest to level
        n = max(1, math.ceil(360 * math.cos(math.radians(near)) / eff_h))
        rows.append((p, n))
    return rows


def north_info(start, pan_offset_deg, calib_loaded):
    """Works out true north for meta.json.
    start: reading taken before the motors moved (the camera then points along the sensor's X axis)
    pan_offset_deg: pan turned during leveling afterwards (where Enter was pressed is yaw 0°)
    yaw grows clockwise seen from above (the D key direction; correct if the stitching comes out right)."""
    if start is None:
        return None
    start_true = (start["heading"] + MAG_DECLINATION_DEG) % 360   # camera bearing from true north at the start
    true0 = (start_true + pan_offset_deg) % 360                   # camera bearing from true north at yaw 0°
    north_yaw = (-true0) % 360                                    # yaw at which the camera faces true north
    lon = (north_yaw + 180) % 360 - 180                           # panorama longitude (-180 to 180)
    info = {
        "assumption": "프로그램 시작 시(모터 이동 전) 카메라가 GY-BNO055 X축 방향을 바라봄",
        "camera_azimuth_at_yaw0_deg": true0,          # from true north, clockwise
        "start_camera_azimuth_magnetic_deg": start["heading"],   # before the motors moved, from magnetic north
        "leveling_pan_offset_deg": pan_offset_deg,
        "declination_deg": MAG_DECLINATION_DEG,
        "north_yaw_deg": north_yaw,                   # the camera faces true north at this yaw
        "north_pano_u": (lon + 180) / 360,            # horizontal position of true north in the stitch_panorama.py result (0 = left edge, 1 = right edge)
        "spread_deg": start["spread"],
        "calib_loaded_from_eeprom": calib_loaded,
        "calib_status": dict(zip(["sys", "gyro", "accel", "mag"], start["cal"])),
        "sensor_euler_heading_deg": start["euler"],
    }
    return info


def save_meta(meta, out_dir):
    """Writes meta.json to the output folder."""
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def capture(rig, cam, out_dir, north=None):
    """Captures every planned row and column of photos and writes meta.json after each row."""
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

    # rows from the bottom up -> tilt always moves upwards, so the backlash direction stays the same
    for ri, (pitch, n) in enumerate(rows):
        rig.tilt_to(pitch)
        for ci in range(n):
            rig.pan_to(ci * 360.0 / n)       # within a row, always turn forwards only
            time.sleep(SETTLE_S)
            data = cam.grab_jpeg()

            fname = f"r{ri:02d}_c{ci:02d}.jpg"
            with open(os.path.join(out_dir, fname), "wb") as f:
                f.write(data)
            item = {
                "file": fname, "row": ri, "col": ci,
                "yaw_deg": steps_to_deg(rig.pan),      # angle from the actual step count
                "pitch_deg": steps_to_deg(rig.tilt),
            }
            if north:                                  # this photo's camera bearing (true north, clockwise)
                item["azimuth_deg"] = (north["camera_azimuth_at_yaw0_deg"] + item["yaw_deg"]) % 360
            meta["images"].append(item)
            shot += 1
            print(f"[{shot}/{total}] pitch {pitch:+d}°  yaw {steps_to_deg(rig.pan):6.1f}°  -> {fname}")

            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                cv2.imshow(win, cv2.resize(img, (640, int(640 * img.shape[0] / img.shape[1]))))
            if (cv2.waitKey(1) & 0xFF) == 27:
                raise KeyboardInterrupt

        # at the end of a row, wind back to untwist the cable and arrive at 0° going forwards
        rig.pan_to(0)

        save_meta(meta, out_dir)

    cv2.destroyWindow(win)
    print(f"완료: {out_dir}")


def main():
    """Connects the rig, measures north, runs leveling and the capture, then returns the rig to its start position."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--arduino", required=True, help="아두이노 포트 (예: COM5)")
    ap.add_argument("--camera", required=True, help="XIAO 포트 (예: COM6)")
    ap.add_argument("--out", default="captures")
    args = ap.parse_args()

    rig = cam = None
    try:
        rig = Rig(args.arduino)

        # north is measured before the motors have moved at all (the camera then points along the sensor's X axis)
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
            try:                       # back to the level, forward-facing position
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
