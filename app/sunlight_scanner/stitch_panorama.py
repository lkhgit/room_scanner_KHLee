#!/usr/bin/env python3
"""
Captured photos -> equirectangular (2:1) panorama

Projects each photo onto a sphere using the motor angles (yaw/pitch) in panorama.py's meta.json and blends them.
Camera calibration from calibrate.py (field of view, optical centre, mounting roll, lens distortion) is used
automatically when found.
  Search order: the file given with --calib -> calib.json in the capture folder -> camera_calib.json in the parent folder
Without calibration, only the field of view is used (from meta.json or --hfov/--vfov).
meta_refined.json from refine.py (per-photo direction corrections) is used first when it exists.

Run:     python stitch_panorama.py captures/20261020_195800
Result:  panorama.jpg (4096x2048 by default) and panorama_1024x512.jpg (HorizonNet input size) in the same folder

Options:
  --calib FILE     calibration file to use
  --no-calib       ignore calibration files and use only the field of view
  --raw            ignore meta_refined.json (per-photo direction corrections)
  --width 4096     output width (height is half)
  --hfov / --vfov  field of view given directly (calibration files are then not used)
  --yaw-sign -1    -1 if the photos join mirrored left-right
  --rotate 0       90 / 180 / 270 if the photos were taken sideways or upside down (not with a calibration file)
  --feather 2      blending strength in the overlaps (higher trusts the photo centres more)
  --out NAME       output file name (default panorama.jpg)
"""
import argparse
import json
import math
import os

import cv2
import numpy as np


def load_meta(folder):
    """Reads meta.json from a capture folder."""
    with open(os.path.join(folder, "meta.json"), encoding="utf-8") as f:
        return json.load(f)


def load_image(path, rotate=0):
    """Reads a photo and rotates it by 0, 90, 180 or 270 degrees."""
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
    """Vertical field of view from the photo's aspect ratio, assuming square pixels (4:3: 85 -> about 69)."""
    return math.degrees(2 * math.atan(math.tan(math.radians(hfov) / 2) * h / w))


def camera_from_fov(hfov, vfov, w, h):
    """Camera model from the field of view alone, without calibration."""
    f = (w / 2) / math.tan(math.radians(hfov) / 2)
    fy = (h / 2) / math.tan(math.radians(vfov) / 2)
    return dict(f=f, aspect=fy / f, cx=(w - 1) / 2, cy=(h - 1) / 2,
                roll=0.0, myaw=0.0, k1=0.0, k2=0.0)


def camera_from_calib(c, w, h):
    """calib.json -> camera model (scaled if the photo size differs)."""
    cw, ch = c.get("image_size", [w, h])
    sx = w / cw
    return dict(f=c["f_px"] * sx, aspect=c["aspect"], cx=c["cx"] * sx, cy=c["cy"] * (h / ch),
                roll=math.radians(c["roll_deg"]), myaw=math.radians(c["mount_yaw_deg"]),
                k1=c["k1"], k2=c["k2"])


def find_calib(folder, explicit=None):
    """Finds the calibration file: the given one, the folder's calib.json, or camera_calib.json in the parent folder."""
    if explicit:
        return explicit
    for p in (os.path.join(folder, "calib.json"),
              os.path.join(os.path.dirname(os.path.abspath(folder.rstrip("/\\"))), "camera_calib.json")):
        if os.path.exists(p):
            return p
    return None


def resolve_fov(meta, folder, hfov, vfov, rotate):
    """Field of view of the original (unrotated) photo, with width and height swapped if rotated."""
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
    """Corrected radius ru -> radius rd on the photo (solves ru = rd * (1 + k1 rd^2 + k2 rd^4) with Newton's method)."""
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
    """Precomputes the direction of every pixel of the output panorama."""

    def __init__(self, W):
        """Precomputes the longitude and latitude of the panorama pixels for an output width W."""
        self.W = W
        self.H = W // 2
        lon = ((np.arange(W, dtype=np.float32) + 0.5) / W * 2 - 1) * np.pi
        self.lat_rows = (0.5 - (np.arange(self.H, dtype=np.float32) + 0.5) / self.H) * np.pi
        self.sin_lon = np.sin(lon)[None, :]
        self.cos_lon = np.cos(lon)[None, :]

    def warp(self, img, yaw_deg, pitch_deg, hfov=None, vfov=None, feather=2.0, cam=None, extra_roll_deg=0.0):
        """Projects one photo into panorama coordinates. Returns (first row, end row, projected photo, weights) or None."""
        h, w = img.shape[:2]
        if cam is None:
            cam = camera_from_fov(hfov, vfov, w, h)
        f, fy = cam["f"], cam["f"] * cam["aspect"]
        cx, cy = cam["cx"], cam["cy"]
        roll = cam["roll"] + math.radians(extra_roll_deg)   # mounting roll + per-photo roll correction

        psi = math.radians(yaw_deg)
        th = math.radians(pitch_deg)

        # only the rows this photo can reach are computed (saves time and memory)
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

        # rig axes: forward (f), right (r), up (u)
        cp, sp, ct, st = math.cos(psi), math.sin(psi), math.cos(th), math.sin(th)
        Z = dx * (ct * sp) + dy * st + dz * (ct * cp)
        X = dx * cp - dz * sp
        Y = dx * (-st * sp) + dy * ct + dz * (-st * cp)

        # undo the camera mounting error: v_cam = Ry(myaw)^T Rz(roll)^T (X, Y, Z)
        if roll or cam["myaw"]:
            cr, sr = math.cos(roll), math.sin(roll)
            X, Y = cr * X + sr * Y, -sr * X + cr * Y
            cyw, syw = math.cos(cam["myaw"]), math.sin(cam["myaw"])
            X, Z = cyw * X - syw * Z, syw * X + cyw * Z

        valid = Z > 1e-3
        Zs = np.where(valid, Z, 1.0)
        xu, yu = X / Zs, Y / Zs

        # apply lens distortion (corrected coordinates -> actual photo coordinates)
        if cam["k1"] or cam["k2"]:
            k1, k2 = cam["k1"], cam["k2"]
            # the inverse is unique only within the radius of the photo corners, so limit it to that
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

        # weights fall off towards the edges -> seams blend smoothly
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
    """Stitches all photos of a capture folder into one panorama; returns the image and the covered-pixel mask."""
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
    """Parses the arguments, stitches the panorama and saves it at full size and at 1024x512."""
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
