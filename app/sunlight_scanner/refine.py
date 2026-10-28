#!/usr/bin/env python3
"""
사진별 방향 자동 보정

모터가 처지거나 기어 유격으로 어긋나서, 사진이 명령한 각도와 다른 곳을 보고 찍힌 경우를 사진 내용으로 바로잡는다.
  1단계  겹치는 사진끼리 특징점(SIFT)을 맞춰, 카메라 값과 사진마다의 yaw/pitch/roll 오차를 함께 추정
  2단계  특징점이 거의 없는 사진(흰 벽·천장)은 이웃 사진과 윤곽이 가장 잘 겹치는 각도를 직접 탐색
  3단계  그래도 정할 수 없으면 같은 행 사진들의 보정값 중앙값을 사용
결과는 meta_refined.json으로 저장되고 (원본 meta.json은 그대로), stitch_panorama.py가 자동으로 읽어 쓴다.

필요: calibrate.py로 만든 calib.json 또는 camera_calib.json (없으면 meta.json의 화각에서 시작)
실행: python refine.py captures/20261021_010000
"""
import argparse
import itertools
import json
import math
import os

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from calibrate import Rz, rig_rotation, mount_rotation, pixels_to_cam_rays
from stitch_panorama import EquirectGrid, camera_from_calib, find_calib

MIN_POINTS = 15          # 이보다 대응점이 적은 사진은 특징점만으로 정하지 않음
WARN_DEG = 15            # 보정량이 이보다 크면 경고
GATE_PX = 320            # 모터 각도로 예측한 위치에서 이만큼 이상 떨어진 매칭은 버림 (반복 무늬 오매칭 제거)


# ---------------------------------------------------------------- 카메라 모델 공용
def cam_rays_to_pixels(p, v):
    """카메라 좌표 광선 -> 픽셀 (렌즈 왜곡 포함)"""
    x, y = v[:, 0] / v[:, 2], v[:, 1] / v[:, 2]
    ru = np.hypot(x, y)
    rd = ru.copy()
    for _ in range(8):
        r2 = rd * rd
        rd -= (rd * (1 + p["k1"] * r2 + p["k2"] * r2 * r2) - ru) / (1 + 3 * p["k1"] * r2 + 5 * p["k2"] * r2 * r2)
    s = np.where(ru > 1e-9, rd / np.maximum(ru, 1e-9), 1.0)
    return np.stack([p["cx"] + p["f"] * x * s, p["cy"] - p["f"] * p["aspect"] * y * s], 1)


def load_camera(folder, meta, W, H):
    path = find_calib(folder)
    if path:
        c = json.load(open(path, encoding="utf-8"))
        cam = camera_from_calib(c, W, H)
        print(f"카메라 보정값: {path}")
    else:
        f = (W / 2) / math.tan(math.radians(meta["hfov_deg"]) / 2)
        fy = (H / 2) / math.tan(math.radians(meta["vfov_deg"]) / 2)
        cam = dict(f=f, aspect=fy / f, cx=(W - 1) / 2, cy=(H - 1) / 2, roll=0.0, myaw=0.0, k1=0.0, k2=0.0)
        print("카메라 보정 파일이 없어 meta.json 화각에서 시작합니다 (calibrate.py를 먼저 실행하는 것을 권장)")
    return cam


# ---------------------------------------------------------------- 1단계: 특징점
def match_pairs(folder, items, cam):
    sift = cv2.SIFT_create(nfeatures=5000)
    feats = []
    for it in items:
        g = cv2.imread(os.path.join(folder, it["file"]), cv2.IMREAD_GRAYSCALE)
        kp, des = sift.detectAndCompute(g, None)
        feats.append((np.float64([k.pt for k in kp]), des))

    R = [rig_rotation(it["yaw_deg"], it["pitch_deg"]) for it in items]
    M = mount_rotation(cam)
    bf = cv2.BFMatcher()
    pairs = []
    for i, j in itertools.combinations(range(len(items)), 2):
        if R[i][:, 2] @ R[j][:, 2] < math.cos(math.radians(85)):
            continue
        (pi, di), (pj, dj) = feats[i], feats[j]
        if di is None or dj is None or len(di) < 2 or len(dj) < 2:
            continue
        good = [a for a, b in (m for m in bf.knnMatch(di, dj, k=2) if len(m) == 2) if a.distance < 0.8 * b.distance]
        if len(good) < 8:
            continue
        A = pi[[g.queryIdx for g in good]]
        B = pj[[g.trainIdx for g in good]]
        # 모터 각도 기준 예측 위치에서 너무 먼 매칭 제거
        vb = (pixels_to_cam_rays(cam, A) @ M.T @ R[i].T) @ R[j] @ M
        front = vb[:, 2] > 0.1
        pred = cam_rays_to_pixels(cam, np.where(front[:, None], vb, [[0, 0, 1]]))
        sel = front & (np.linalg.norm(pred - B, axis=1) < GATE_PX)
        if sel.sum() < 8:
            continue
        A, B = A[sel], B[sel]
        Hm, mask = cv2.findHomography(A, B, cv2.RANSAC, 6.0)
        if Hm is None or mask.sum() < 8:
            continue
        m = mask.ravel().astype(bool)
        pairs.append((i, j, A[m], B[m]))
    return pairs


INTR = ["f", "aspect", "cx", "cy", "roll", "myaw", "k1", "k2"]


def bundle_adjust(items, pairs, cam, ref, W):
    N = len(items)
    I = np.concatenate([np.full(len(A), i) for i, j, A, B in pairs])
    J = np.concatenate([np.full(len(A), j) for i, j, A, B in pairs])
    PA = np.concatenate([A for i, j, A, B in pairs])
    PB = np.concatenate([B for i, j, A, B in pairs])
    pix = (W / 2) / math.tan(math.radians(40))

    def rotations(p, corr):
        out = []
        for k, it in enumerate(items):
            dy, dp, dr = corr[k]
            R = rig_rotation(it["yaw_deg"] + math.degrees(dy), it["pitch_deg"] + math.degrees(dp))
            out.append(R @ Rz(dr) @ mount_rotation(p))
        return np.array(out)

    def unpack(x):
        p = dict(zip(INTR, x[:8]))
        corr = x[8:].reshape(N, 3).copy()
        corr[ref] = 0
        return p, corr

    def resid(x):
        p, corr = unpack(x)
        Rs = rotations(p, corr)
        wa = np.einsum("kij,kj->ki", Rs[I], pixels_to_cam_rays(p, PA))
        wb = np.einsum("kij,kj->ki", Rs[J], pixels_to_cam_rays(p, PB))
        r = np.arcsin(np.clip(np.linalg.norm(np.cross(wa, wb), axis=1), 0, 1)) * pix
        reg = np.degrees(corr.ravel()) * 0.3      # 근거 없는 큰 보정 억제
        return np.concatenate([r, reg])

    def sparsity():
        # 각 대응점 오차는 카메라 값 8개와 두 사진의 보정값에만 의존 -> 미분 계산을 크게 줄임
        n = len(PA)
        S = lil_matrix((n + N * 3, 8 + N * 3), dtype=int)
        S[:n, :8] = 1
        rows = np.arange(n)
        for a in range(3):
            S[rows, 8 + I * 3 + a] = 1
            S[rows, 8 + J * 3 + a] = 1
        for q in range(N * 3):
            S[n + q, 8 + q] = 1
        return S

    x0 = np.concatenate([[cam[k] for k in INTR], np.zeros(N * 3)])
    sol = least_squares(resid, x0, loss="soft_l1", f_scale=3.0, x_scale="jac",
                        jac_sparsity=sparsity(), max_nfev=200)
    p, corr = unpack(sol.x)
    r = resid(sol.x)[:len(PA)]
    # 오차가 큰 매칭을 빼고 한 번 더
    keep = r < max(10.0, 4 * np.median(r))
    I, J, PA, PB = I[keep], J[keep], PA[keep], PB[keep]
    sol = least_squares(resid, sol.x, loss="soft_l1", f_scale=2.0, x_scale="jac",
                        jac_sparsity=sparsity(), max_nfev=200)
    p, corr = unpack(sol.x)
    r = resid(sol.x)[:len(PA)]
    counts = np.bincount(np.concatenate([I, J]), minlength=N)
    return p, np.degrees(corr), counts, float(np.median(r))


# ---------------------------------------------------------------- 2단계: 윤곽 일치 탐색
def highpass(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA)
    h = g - cv2.GaussianBlur(g, (0, 0), 3)
    return h / (h.std() + 1e-6)


def half_cam(p):
    c = dict(p)
    c["f"] /= 2
    c["cx"] = (c["cx"] + 0.5) / 2 - 0.5
    c["cy"] = (c["cy"] + 0.5) / 2 - 0.5
    return c


def photometric_search(target_hp, it, corr0, others, cam_half, grids):
    """others: [(hp, yaw, pitch, extra_roll)] 로 만든 기준 윤곽과 가장 잘 겹치는 (dyaw, dpitch) 탐색.
    grids: (저해상도, 고해상도) - 저해상도로 넓게 찾고 고해상도로 좁혀 찾는다."""
    refs = []
    for grid in grids:
        acc = np.zeros((grid.H, grid.W), np.float32)
        cnt = np.zeros_like(acc)
        for hp, y, pt, er in others:
            res = grid.warp(hp, y, pt, feather=0, cam=cam_half, extra_roll_deg=er)
            if res is None:
                continue
            r0, r1, w, m = res
            acc[r0:r1] += w * (m > 0)
            cnt[r0:r1] += (m > 0)
        refs.append((grid, np.where(cnt > 0, acc / np.maximum(cnt, 1), 0), cnt))

    def err(level, dy, dp):
        grid, ref, cnt = refs[level]
        res = grid.warp(target_hp, it["yaw_deg"] + dy, it["pitch_deg"] + dp, feather=0, cam=cam_half,
                        extra_roll_deg=corr0[2])
        if res is None:
            return np.inf
        r0, r1, w, m = res
        o = (m > 0) & (cnt[r0:r1] > 0)
        if o.sum() < 1500 * (grid.W / 1024) ** 2:
            return np.inf
        return float(np.mean((w[o] - ref[r0:r1][o]) ** 2))

    # 저해상도: pitch 1° / yaw 2° 간격, 처짐은 주로 아래쪽이라 아래로 넓게
    best = (err(0, corr0[0], corr0[1]), corr0[0], corr0[1])
    coarse = []
    for dp in np.arange(corr0[1] - 12, corr0[1] + 6.01, 1.0):
        for dy in np.arange(corr0[0] - 6, corr0[0] + 6.01, 2.0):
            e = err(0, dy, dp)
            if np.isfinite(e):
                coarse.append(e)
            if e < best[0]:
                best = (e, dy, dp)
    contrast = best[0] / np.median(coarse) if coarse else np.inf   # 1보다 충분히 작아야 뚜렷한 최솟값
    # 고해상도: 주변을 촘촘히
    fine = (err(1, best[1], best[2]), best[1], best[2])
    for dp in np.arange(best[2] - 1.0, best[2] + 1.01, 0.25):
        for dy in np.arange(best[1] - 2.0, best[1] + 2.01, 0.5):
            e = err(1, dy, dp)
            if e < fine[0]:
                fine = (e, dy, dp)
    return fine, contrast


# ---------------------------------------------------------------- 메인
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    args = ap.parse_args()
    folder = args.folder

    meta = json.load(open(os.path.join(folder, "meta.json"), encoding="utf-8"))
    items = meta["images"]
    N = len(items)
    imgs = [cv2.imread(os.path.join(folder, it["file"])) for it in items]
    H, W = imgs[0].shape[:2]
    cam = load_camera(folder, meta, W, H)

    print("특징점 매칭 중...")
    pairs = match_pairs(folder, items, cam)
    print(f"  사진 쌍 {len(pairs)}개, 대응점 {sum(len(p[2]) for p in pairs)}개")

    # 기준 사진: 수평(0°)에 가장 가까운 행에서 대응점이 가장 많은 사진
    cnt0 = np.zeros(N, int)
    for i, j, A, B in pairs:
        cnt0[i] += len(A)
        cnt0[j] += len(A)
    min_abs_pitch = min(abs(it["pitch_deg"]) for it in items)
    level = [k for k, it in enumerate(items) if abs(it["pitch_deg"]) - min_abs_pitch < 1]
    ref = max(level, key=lambda k: cnt0[k])

    print("사진별 방향 추정 중 (번들 조정)...")
    p, corr, counts, med = bundle_adjust(items, pairs, cam, ref, W)
    print(f"  특징점 오차 중앙값 {med:.2f}px  (기준 사진 {items[ref]['file']})")

    method = ["특징점" if counts[k] >= MIN_POINTS or k == ref else "" for k in range(N)]

    # 2·3단계: 특징점이 부족한 사진
    weak = [k for k in range(N) if not method[k]]
    if weak:
        print(f"특징점이 부족한 사진 {len(weak)}장 -> 윤곽 일치 탐색")
        grids = (EquirectGrid(512), EquirectGrid(1024))
        ch = half_cam(p)
        hps = [highpass(im) for im in imgs]
        for k in weak:
            row = items[k]["row"]
            same = [corr[q] for q in range(N) if items[q]["row"] == row and method[q]]
            start = np.median(same, axis=0) if same else np.zeros(3)
            others = [(hps[q], items[q]["yaw_deg"] + corr[q][0], items[q]["pitch_deg"] + corr[q][1], corr[q][2])
                      for q in range(N) if method[q]]
            (e, dy, dp), contrast = photometric_search(hps[k], items[k], start, others, ch, grids)
            print(f"  {items[k]['file']}: 최솟값 뚜렷함 {contrast:.2f} (0.8 미만이면 채택)")
            if np.isfinite(e) and contrast < 0.8:
                corr[k] = [dy, dp, start[2]]
                method[k] = "윤곽 탐색"
            else:
                corr[k] = start
                method[k] = "행 중앙값" if same else "보정 없음"

    # 결과 출력
    print("\n사진         yaw보정  pitch보정  roll보정  대응점  방법")
    warn = []
    for k, it in enumerate(items):
        d = corr[k]
        flag = ""
        if np.abs(d[:2]).max() > WARN_DEG:
            flag = "  <- 보정량이 큼, 사진 확인 필요"
            warn.append(it["file"])
        print(f"{it['file']}  {d[0]:+6.2f}°  {d[1]:+7.2f}°  {d[2]:+6.2f}°  {counts[k]:5d}  {method[k]}{flag}")

    rows = sorted({it["row"] for it in items})
    print("\n행별 pitch 보정 중앙값 (명령 각도 대비 실제로 처진 정도):")
    for r in rows:
        v = [corr[k][1] for k in range(N) if items[k]["row"] == r]
        print(f"  {np.mean([items[k]['pitch_deg'] for k in range(N) if items[k]['row'] == r]):+5.1f}° 행: "
              f"{np.median(v):+5.2f}°  (범위 {min(v):+.1f}° ~ {max(v):+.1f}°)")

    # 저장
    out = dict(meta)
    out["camera"] = {
        "hfov_deg": 2 * math.degrees(math.atan((W / 2) / p["f"])),
        "vfov_deg": 2 * math.degrees(math.atan((H / 2) / (p["f"] * p["aspect"]))),
        "image_size": [W, H], "f_px": p["f"], "aspect": p["aspect"], "cx": p["cx"], "cy": p["cy"],
        "roll_deg": math.degrees(p["roll"]), "mount_yaw_deg": math.degrees(p["myaw"]),
        "k1": p["k1"], "k2": p["k2"], "median_error_px": med,
    }
    new_items = []
    for k, it in enumerate(items):
        n = dict(it)
        n["motor_yaw_deg"], n["motor_pitch_deg"] = it["yaw_deg"], it["pitch_deg"]
        n["yaw_deg"] = it["yaw_deg"] + float(corr[k][0])
        n["pitch_deg"] = it["pitch_deg"] + float(corr[k][1])
        n["extra_roll_deg"] = float(corr[k][2])
        n["refine_method"] = method[k]
        new_items.append(n)
    out["images"] = new_items
    path = os.path.join(folder, "meta_refined.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n저장: {path}")
    if warn:
        print("※ 보정량이 큰 사진:", ", ".join(warn))


if __name__ == "__main__":
    main()
