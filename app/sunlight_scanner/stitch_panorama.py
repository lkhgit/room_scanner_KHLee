#!/usr/bin/env python3
"""
촬영 사진 -> 정거방형(equirectangular, 2:1) 파노라마

panorama.py가 저장한 meta.json의 모터 각도(yaw/pitch)로 각 사진을 구면에 투영해 이어붙인다.
calibrate.py로 구한 카메라 보정값(화각, 광학 중심, 장착 기울어짐 roll, 렌즈 왜곡)이 있으면 자동으로 쓴다.
  찾는 순서: --calib 로 지정한 파일 -> 촬영 폴더의 calib.json -> 상위 폴더의 camera_calib.json
보정값이 없으면 화각만으로 이어붙인다 (meta.json 또는 --hfov/--vfov).
refine.py가 만든 meta_refined.json(사진별 방향 보정)이 있으면 그것을 가장 먼저 쓴다.

실행:  python stitch_panorama.py captures/20261020_195800
결과:  같은 폴더에 panorama.jpg (기본 4096x2048), panorama_1024x512.jpg (HorizonNet 입력 크기)

옵션:
  --calib FILE     사용할 보정 파일 지정
  --no-calib       보정 파일을 무시하고 화각만 사용
  --raw            meta_refined.json(사진별 방향 보정)을 무시
  --width 4096     출력 가로 크기 (세로는 절반)
  --hfov / --vfov  화각 직접 지정 (지정하면 보정 파일은 쓰지 않음)
  --yaw-sign -1    좌우가 뒤집혀 이어지면 -1
  --rotate 0       사진이 누워 있거나 뒤집혀 찍혔으면 90 / 180 / 270 (보정 파일과 함께는 사용 불가)
  --feather 2      겹치는 부분 섞기 강도 (클수록 사진 가운데를 더 신뢰)
  --out NAME       저장 파일 이름 (기본 panorama.jpg)
"""
import argparse
import json
import math
import os

import cv2
import numpy as np


def load_meta(folder):
    with open(os.path.join(folder, "meta.json"), encoding="utf-8") as f:
        return json.load(f)


def load_image(path, rotate=0):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"사진을 읽을 수 없음: {path}")
    if rotate == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == 180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif rotate == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def vfov_from_hfov(hfov, w, h):
    """정사각형 픽셀이라고 보고 사진 비율로 세로 화각 계산 (4:3이면 85 -> 약 69)"""
    return math.degrees(2 * math.atan(math.tan(math.radians(hfov) / 2) * h / w))


def camera_from_fov(hfov, vfov, w, h):
    """보정값 없이 화각만으로 만든 카메라 모델"""
    f = (w / 2) / math.tan(math.radians(hfov) / 2)
    fy = (h / 2) / math.tan(math.radians(vfov) / 2)
    return dict(f=f, aspect=fy / f, cx=(w - 1) / 2, cy=(h - 1) / 2,
                roll=0.0, myaw=0.0, k1=0.0, k2=0.0)


def camera_from_calib(c, w, h):
    """calib.json -> 카메라 모델 (사진 크기가 다르면 비례 조정)"""
    cw, ch = c.get("image_size", [w, h])
    sx = w / cw
    return dict(f=c["f_px"] * sx, aspect=c["aspect"], cx=c["cx"] * sx, cy=c["cy"] * (h / ch),
                roll=math.radians(c["roll_deg"]), myaw=math.radians(c["mount_yaw_deg"]),
                k1=c["k1"], k2=c["k2"])


def find_calib(folder, explicit=None):
    if explicit:
        return explicit
    for p in (os.path.join(folder, "calib.json"),
              os.path.join(os.path.dirname(os.path.abspath(folder.rstrip("/\\"))), "camera_calib.json")):
        if os.path.exists(p):
            return p
    return None


def resolve_fov(meta, folder, hfov, vfov, rotate):
    """회전 전 원본 사진 기준 화각을 정하고, 회전했으면 가로·세로를 바꾼다."""
    raw = cv2.imread(os.path.join(folder, meta["images"][0]["file"]), cv2.IMREAD_COLOR)
    h0, w0 = raw.shape[:2]
    if hfov is None:
        hfov = meta["hfov_deg"]
        if vfov is None:
            vfov = meta["vfov_deg"]
    if vfov is None:
        vfov = vfov_from_hfov(hfov, w0, h0)
    if rotate in (90, 270):
        hfov, vfov = vfov, hfov
    return hfov, vfov


def _undistort_radius_inverse(ru, k1, k2, iters=8):
    """보정된 반경 ru -> 사진 위 반경 rd  (ru = rd * (1 + k1 rd^2 + k2 rd^4) 를 뉴턴법으로 풂)"""
    if k1 == 0 and k2 == 0:
        return ru
    rd = ru.copy()
    for _ in range(iters):
        r2 = rd * rd
        g = rd * (1 + k1 * r2 + k2 * r2 * r2) - ru
        dg = 1 + 3 * k1 * r2 + 5 * k2 * r2 * r2
        rd = rd - g / np.where(np.abs(dg) < 1e-6, 1e-6, dg)
    return rd


class EquirectGrid:
    """출력 파노라마의 각 픽셀 방향을 미리 계산해 둔다."""

    def __init__(self, W):
        self.W = W
        self.H = W // 2
        lon = ((np.arange(W, dtype=np.float32) + 0.5) / W * 2 - 1) * np.pi
        self.lat_rows = (0.5 - (np.arange(self.H, dtype=np.float32) + 0.5) / self.H) * np.pi
        self.sin_lon = np.sin(lon)[None, :]
        self.cos_lon = np.cos(lon)[None, :]

    def warp(self, img, yaw_deg, pitch_deg, hfov=None, vfov=None, feather=2.0, cam=None, extra_roll_deg=0.0):
        """사진 한 장을 파노라마 좌표로 투영. (시작 행, 끝 행, 투영된 사진, 가중치) 또는 None"""
        h, w = img.shape[:2]
        if cam is None:
            cam = camera_from_fov(hfov, vfov, w, h)
        f, fy = cam["f"], cam["f"] * cam["aspect"]
        cx, cy = cam["cx"], cam["cy"]
        roll = cam["roll"] + math.radians(extra_roll_deg)   # 장착 roll + 사진별 roll 보정

        psi = math.radians(yaw_deg)
        th = math.radians(pitch_deg)

        # 이 사진이 닿을 수 있는 행만 계산 (속도·메모리 절약)
        half_diag = math.atan(math.hypot((w / 2) / f, (h / 2) / fy)) * (1 + max(0.0, cam["k1"]))
        margin = 0.05 + abs(roll) + abs(cam["myaw"])
        lat_hi, lat_lo = th + half_diag + margin, th - half_diag - margin
        rows = np.where((self.lat_rows <= lat_hi) & (self.lat_rows >= lat_lo))[0]
        if rows.size == 0:
            return None
        r0, r1 = rows[0], rows[-1] + 1
        lat = self.lat_rows[r0:r1][:, None]
        cl, sl = np.cos(lat), np.sin(lat)
        dx = cl * self.sin_lon
        dy = np.broadcast_to(sl, dx.shape)
        dz = cl * self.cos_lon

        # 리그 좌표축: 앞(f), 오른쪽(r), 위(u)
        cp, sp, ct, st = math.cos(psi), math.sin(psi), math.cos(th), math.sin(th)
        Z = dx * (ct * sp) + dy * st + dz * (ct * cp)
        X = dx * cp - dz * sp
        Y = dx * (-st * sp) + dy * ct + dz * (-st * cp)

        # 카메라 장착 오차 되돌리기: v_cam = Ry(myaw)^T Rz(roll)^T (X, Y, Z)
        if roll or cam["myaw"]:
            cr, sr = math.cos(roll), math.sin(roll)
            X, Y = cr * X + sr * Y, -sr * X + cr * Y
            cyw, syw = math.cos(cam["myaw"]), math.sin(cam["myaw"])
            X, Z = cyw * X - syw * Z, syw * X + cyw * Z

        valid = Z > 1e-3
        Zs = np.where(valid, Z, 1.0)
        xu, yu = X / Zs, Y / Zs

        # 렌즈 왜곡 적용 (보정된 좌표 -> 실제 사진 좌표)
        if cam["k1"] or cam["k2"]:
            k1, k2 = cam["k1"], cam["k2"]
            # 사진 모서리까지의 반경 안에서만 역변환이 유일하므로 그 범위로 제한
            rd_max = math.hypot(max(cx, w - 1 - cx) / f, max(cy, h - 1 - cy) / fy)
            ru_max = rd_max * (1 + k1 * rd_max ** 2 + k2 * rd_max ** 4)
            ru = np.sqrt(xu * xu + yu * yu)
            rd = _undistort_radius_inverse(ru, k1, k2)
            back = rd * (1 + k1 * rd * rd + k2 * rd ** 4)
            valid &= (ru <= ru_max * 1.001) & (np.abs(back - ru) < 1e-4) & (rd >= 0)
            scale = np.where(ru > 1e-9, rd / np.maximum(ru, 1e-9), 1.0)
            xu, yu = xu * scale, yu * scale

        mx = (cx + f * xu).astype(np.float32)
        my = (cy - fy * yu).astype(np.float32)
        valid &= (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
        if not valid.any():
            return None

        # 가장자리로 갈수록 작아지는 가중치 -> 이음매가 부드럽게 섞임
        wx = np.clip(1 - np.abs(mx - (w - 1) / 2) / (w / 2), 0, 1)
        wy = np.clip(1 - np.abs(my - (h - 1) / 2) / (h / 2), 0, 1)
        wgt = (wx * wy) ** feather
        wgt[~valid] = 0
        mx[~valid] = -1
        my[~valid] = -1

        warped = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return r0, r1, warped, wgt


def stitch(folder, width, hfov, vfov, yaw_sign, rotate, feather, verbose=True,
           calib_path=None, use_calib=True, use_refined=True):
    meta = load_meta(folder)

    cam = None
    refined = os.path.join(folder, "meta_refined.json")
    if use_refined and use_calib and hfov is None and vfov is None and os.path.exists(refined):
        meta = json.load(open(refined, encoding="utf-8"))
        h0, w0 = cv2.imread(os.path.join(folder, meta["images"][0]["file"])).shape[:2]
        cam = camera_from_calib(meta["camera"], w0, h0)
        if verbose:
            print(f"사진별 방향 보정 사용: {refined}")
    if cam is None and use_calib and hfov is None and vfov is None:
        path = find_calib(folder, calib_path)
        if path:
            if rotate:
                raise SystemExit("보정 파일과 --rotate는 함께 쓸 수 없습니다. --no-calib를 붙이세요.")
            c = json.load(open(path, encoding="utf-8"))
            h0, w0 = cv2.imread(os.path.join(folder, meta["images"][0]["file"])).shape[:2]
            cam = camera_from_calib(c, w0, h0)
            if verbose:
                print(f"보정 파일 사용: {path}")
                print(f"  화각 {c['hfov_deg']:.1f}°/{c['vfov_deg']:.1f}°  roll {c['roll_deg']:+.2f}°  "
                      f"k1 {c['k1']:+.3f}")
    if cam is None:
        hfov, vfov = resolve_fov(meta, folder, hfov, vfov, rotate)
        if verbose:
            print(f"화각만 사용: 가로 {hfov:.2f}°  세로 {vfov:.2f}°")

    grid = EquirectGrid(width)
    acc = np.zeros((grid.H, grid.W, 3), np.float32)
    wsum = np.zeros((grid.H, grid.W), np.float32)

    images = meta["images"]
    for k, item in enumerate(images):
        img = load_image(os.path.join(folder, item["file"]), rotate)
        res = grid.warp(img, item["yaw_deg"] * yaw_sign, item["pitch_deg"], hfov, vfov, feather, cam,
                        extra_roll_deg=item.get("extra_roll_deg", 0.0))
        if res is None:
            continue
        r0, r1, warped, wgt = res
        acc[r0:r1] += warped.astype(np.float32) * wgt[..., None]
        wsum[r0:r1] += wgt
        if verbose:
            print(f"[{k + 1}/{len(images)}] {item['file']}  yaw {item['yaw_deg']:6.1f}°  pitch {item['pitch_deg']:+5.1f}°")

    covered = wsum > 0
    pano = np.zeros((grid.H, grid.W, 3), np.uint8)
    pano[covered] = np.clip(acc[covered] / wsum[covered][:, None], 0, 255).astype(np.uint8)
    return pano, covered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="captures/<날짜_시간> 폴더")
    ap.add_argument("--calib", default=None)
    ap.add_argument("--no-calib", action="store_true")
    ap.add_argument("--raw", action="store_true", help="meta_refined.json(사진별 방향 보정)을 쓰지 않음")
    ap.add_argument("--width", type=int, default=4096)
    ap.add_argument("--hfov", type=float, default=None)
    ap.add_argument("--vfov", type=float, default=None)
    ap.add_argument("--yaw-sign", type=int, default=1, choices=[1, -1])
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--feather", type=float, default=2.0)
    ap.add_argument("--out", default="panorama.jpg")
    args = ap.parse_args()

    pano, covered = stitch(args.folder, args.width, args.hfov, args.vfov, args.yaw_sign, args.rotate,
                           args.feather, calib_path=args.calib, use_calib=not args.no_calib,
                           use_refined=not args.raw)

    stem = os.path.splitext(args.out)[0]
    out_full = os.path.join(args.folder, args.out)
    out_small = os.path.join(args.folder, f"{stem}_1024x512.jpg")
    cv2.imwrite(out_full, pano, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(out_small, cv2.resize(pano, (1024, 512), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 95])

    rows_any = np.where(covered.any(axis=1))[0]
    if rows_any.size:
        top = 90 - rows_any[0] / pano.shape[0] * 180
        low = 90 - (rows_any[-1] + 1) / pano.shape[0] * 180
        print(f"덮인 범위: 위도 {low:+.0f}° ~ {top:+.0f}°")
    print(f"저장: {out_full}\n      {out_small}")


if __name__ == "__main__":
    main()
