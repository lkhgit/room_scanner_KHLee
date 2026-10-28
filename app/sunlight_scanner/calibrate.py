#!/usr/bin/env python3
"""
카메라 보정값 자동 추정

촬영 폴더의 사진끼리 겹치는 부분에서 특징점(SIFT)을 찾아 짝지은 뒤,
모터 각도(meta.json)는 그대로 믿고 카메라 쪽 값들을 맞춘다.
  - 화각(초점거리), 세로 비율
  - 광학 중심 (cx, cy)
  - 카메라 장착 기울어짐 (roll), 장착 좌우 틀어짐 (mount yaw)
  - 렌즈 왜곡 (k1, k2)
결과는 calib.json으로 저장되고, stitch_panorama.py가 자동으로 읽어 쓴다.
카메라를 다시 장착하지 않는 한 한 번 구한 값을 다른 촬영에도 쓸 수 있다.

설치:  python -m pip install scipy
실행:  python calibrate.py captures/20261020_210550
       -> captures/20261020_210550/calib.json 과 같은 폴더의 상위에 camera_calib.json 저장
"""
import argparse
import itertools
import json
import math
import os

import cv2
import numpy as np
from scipy.optimize import least_squares

PARAMS = ["f", "aspect", "cx", "cy", "roll", "myaw", "k1", "k2"]


# ---------------- 회전 ----------------
def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rig_rotation(yaw_deg, pitch_deg):
    """모터 각도 -> 카메라 좌표(오른쪽, 위, 앞)를 세계 좌표로 바꾸는 회전 (stitch와 동일)"""
    psi, th = math.radians(yaw_deg), math.radians(pitch_deg)
    cp, sp, ct, st = math.cos(psi), math.sin(psi), math.cos(th), math.sin(th)
    r = [cp, 0, -sp]
    u = [-st * sp, ct, -st * cp]
    f = [ct * sp, st, ct * cp]
    return np.array([r, u, f]).T


def mount_rotation(p):
    """카메라가 마운트에 붙은 방향 오차 (roll: 광축 둘레 기울어짐, myaw: 좌우 틀어짐)"""
    return Rz(p["roll"]) @ Ry(p["myaw"])


def pixels_to_cam_rays(p, P):
    """픽셀 -> 카메라 좌표 광선 (렌즈 왜곡 보정 포함)"""
    x = (P[:, 0] - p["cx"]) / p["f"]
    y = -(P[:, 1] - p["cy"]) / (p["f"] * p["aspect"])
    r2 = x * x + y * y
    s = 1 + p["k1"] * r2 + p["k2"] * r2 * r2
    v = np.stack([x * s, y * s, np.ones_like(x)], 1)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ---------------- 특징점 매칭 ----------------
def match_pairs(folder, items, max_angle=75):
    sift = cv2.SIFT_create(nfeatures=4000)
    feats = []
    for it in items:
        g = cv2.imread(os.path.join(folder, it["file"]), cv2.IMREAD_GRAYSCALE)
        kp, des = sift.detectAndCompute(g, None)
        feats.append((np.float32([k.pt for k in kp]), des))

    def axis(it):
        return rig_rotation(it["yaw_deg"], it["pitch_deg"])[:, 2]

    bf = cv2.BFMatcher()
    out = []
    for i, j in itertools.combinations(range(len(items)), 2):
        ang = math.degrees(math.acos(np.clip(axis(items[i]) @ axis(items[j]), -1, 1)))
        if ang > max_angle:
            continue
        (pi, di), (pj, dj) = feats[i], feats[j]
        if di is None or dj is None or len(di) < 2 or len(dj) < 2:
            continue
        good = [a for a, b in (m for m in bf.knnMatch(di, dj, k=2) if len(m) == 2)
                if a.distance < 0.75 * b.distance]
        if len(good) < 15:
            continue
        A = pi[[g.queryIdx for g in good]]
        B = pj[[g.trainIdx for g in good]]
        Hm, mask = cv2.findHomography(A, B, cv2.RANSAC, 10.0)
        if Hm is None or mask.sum() < 12:
            continue
        mask = mask.ravel().astype(bool)
        out.append((i, j, A[mask].astype(np.float64), B[mask].astype(np.float64)))
    return out


# ---------------- 최적화 ----------------
class Problem:
    def __init__(self, items, pairs, W):
        self.R = [rig_rotation(it["yaw_deg"], it["pitch_deg"]) for it in items]
        self.I = np.concatenate([np.full(len(A), i) for i, j, A, B in pairs])
        self.J = np.concatenate([np.full(len(A), j) for i, j, A, B in pairs])
        self.PA = np.concatenate([A for i, j, A, B in pairs])
        self.PB = np.concatenate([B for i, j, A, B in pairs])
        self.pix_per_rad = (W / 2) / math.tan(math.radians(40))   # 오차를 픽셀로 환산하는 고정 계수

    def world_rays(self, p, P, idx):
        v = pixels_to_cam_rays(p, P) @ mount_rotation(p).T
        out = np.empty_like(v)
        for k in np.unique(idx):
            sel = idx == k
            out[sel] = v[sel] @ self.R[k].T
        return out

    def residual(self, x, free, fixed):
        p = dict(fixed)
        p.update(zip(free, x))
        c = np.cross(self.world_rays(p, self.PA, self.I), self.world_rays(p, self.PB, self.J))
        return np.arcsin(np.clip(np.linalg.norm(c, axis=1), 0, 1)) * self.pix_per_rad

    def fit(self, free, start):
        sol = least_squares(self.residual, [start[k] for k in free], args=(free, start),
                            loss="soft_l1", f_scale=3.0, x_scale="jac")
        p = dict(start)
        p.update(zip(free, sol.x))
        return p, self.residual(sol.x, free, start)

    def keep(self, mask):
        self.I, self.J, self.PA, self.PB = self.I[mask], self.J[mask], self.PA[mask], self.PB[mask]


def params_to_json(p, W, H, err):
    return {
        "hfov_deg": 2 * math.degrees(math.atan((W / 2) / p["f"])),
        "vfov_deg": 2 * math.degrees(math.atan((H / 2) / (p["f"] * p["aspect"]))),
        "image_size": [W, H],
        "f_px": p["f"], "aspect": p["aspect"], "cx": p["cx"], "cy": p["cy"],
        "roll_deg": math.degrees(p["roll"]), "mount_yaw_deg": math.degrees(p["myaw"]),
        "k1": p["k1"], "k2": p["k2"],
        "median_error_px": err,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.folder, "meta.json"), encoding="utf-8"))
    items = meta["images"]
    H, W = cv2.imread(os.path.join(args.folder, items[0]["file"])).shape[:2]

    print("특징점 매칭 중...")
    pairs = match_pairs(args.folder, items)
    prob = Problem(items, pairs, W)
    print(f"  사진 쌍 {len(pairs)}개, 대응점 {len(prob.I)}개")
    if len(prob.I) < 100:
        raise SystemExit("대응점이 너무 적습니다. 무늬가 많은 곳에서 찍은 폴더로 다시 시도하세요.")

    f0 = (W / 2) / math.tan(math.radians(meta.get("hfov_deg", 80)) / 2)
    start = dict(f=f0, aspect=1.0, cx=(W - 1) / 2, cy=(H - 1) / 2, roll=0.0, myaw=0.0, k1=0.0, k2=0.0)
    before = np.median(prob.residual([], [], {**start, "aspect": (H / 2) / math.tan(
        math.radians(meta.get("vfov_deg", 65)) / 2) / f0}))

    # 1차: 왜곡 없이 맞춘 뒤 이상치(잘못된 매칭) 제거
    p, res = prob.fit(["f", "aspect", "cx", "cy", "roll", "myaw"], start)
    thr = max(15.0, 4 * float(np.median(res)))
    prob.keep(res < thr)
    print(f"  잘못된 매칭 제거 후 대응점 {len(prob.I)}개 (기준 {thr:.0f}px)")

    # 2차: 전체
    p, res = prob.fit(PARAMS, p)
    err = float(np.median(res))
    out = params_to_json(p, W, H, err)

    print(f"\n보정 전 (meta.json 화각만 사용) 오차 중앙값: {before:.1f}px")
    print(f"보정 후 오차 중앙값: {err:.1f}px\n")
    print(f"  가로 화각   {out['hfov_deg']:.2f}°")
    print(f"  세로 화각   {out['vfov_deg']:.2f}°")
    print(f"  광학 중심   ({out['cx']:.1f}, {out['cy']:.1f})  (사진 중앙 {(W - 1) / 2:.1f}, {(H - 1) / 2:.1f})")
    print(f"  roll        {out['roll_deg']:+.2f}°   (카메라가 광축 둘레로 기울어진 각도)")
    print(f"  장착 yaw    {out['mount_yaw_deg']:+.2f}°")
    print(f"  렌즈 왜곡   k1 {out['k1']:+.4f}  k2 {out['k2']:+.4f}")

    path = os.path.join(args.folder, "calib.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    shared = os.path.join(os.path.dirname(os.path.abspath(args.folder.rstrip("/\\"))), "camera_calib.json")
    json.dump(out, open(shared, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n저장: {path}\n      {shared}  (calib.json이 없는 다른 촬영 폴더는 이 값을 사용)")


if __name__ == "__main__":
    main()
