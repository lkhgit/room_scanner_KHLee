"""
panorama to model.py — 360° 2:1 파노라마 한 장으로 방의 3D 모델을 만드는 스크립트

    (가상환경) .venv\\Scripts\\python.exe "panorama to model.py" panorama_test.jpg --cam_height 1.2

처리 순서
    1. HorizonNet으로 벽-천장 / 벽-바닥 경계와 벽 모서리를 감지 (HorizonNet/ 폴더의 코드 재사용)
    2. 카메라 높이(실측값)를 기준으로 모서리를 미터 단위 3D 좌표로 복원 → 방을 "접는" 단계
    3. 벽마다 정면에서 본 직사각형 텍스처를 파노라마에서 펴냄 (벽 평면에 대한 정사영 이미지)
       → 다음 단계에서 이 이미지를 그대로 창문 감지 모델에 넣을 수 있음
    4. 바닥/천장은 위에서 내려다본 텍스처를 만듦
    5. 색 투영: 각 면을 촘촘한 격자(--color_cell, 기본 4cm)로 나누고, 꼭짓점마다 파노라마 색을 투영해 저장
       → 텍스처(MTL)를 읽지 못하는 뷰어(예: Visual Studio)에서도 색이 보임
    6. 3D 모델 저장: viewer.html(브라우저 뷰어), room_colored.ply, room.obj, room.glb

출력 폴더 (--out/<이미지 이름>/)
    viewer.html     ★ 가장 확실한 보기 방법. 브라우저로 열면 텍스처 그대로 보임
                    (모델이 파일 안에 들어 있음, three.js 라이브러리만 인터넷에서 불러옴)
    room_colored.ply  꼭짓점 색(vertex color)만으로 색을 표현한 메쉬. 텍스처 없이도 색이 보여서
                    MeshLab / Blender / CloudCompare 등 거의 모든 3D 프로그램에서 색이 나옴
    room.obj/.mtl   텍스처(PNG) + 꼭짓점 색을 모두 가진 메쉬. 텍스처를 읽는 프로그램(Blender 등)은 선명한
                    텍스처로, 읽지 못하는 프로그램은 꼭짓점 색으로 표시함 (PNG와 같은 폴더에 있어야 함)
    room.glb        텍스처가 포함된 단일 파일 (Blender, 웹 glTF 뷰어 등)
    wall_XX.png     벽 텍스처 (창문 감지 입력)
    floor.png, ceiling.png
    layout.json     벽마다 3D 모서리, 크기(m), 텍스처 크기 → wall_px_to_3d()로 픽셀을 3D로 변환
    overlay.png     복원한 3D 방을 다시 파노라마에 투영해 그린 선 (검증용: 실제 경계와 맞아야 함)
    raw.png         HorizonNet 원시 출력 (위: 모서리 확률, 아래: 경계선)

좌표계 (이 스크립트 내부와 layout.json)
    원점 = 카메라 렌즈 중심, 단위 m, 오른손 좌표계, z축이 위쪽.
    +y = 파노라마 가운데 열(u=0)이 바라보는 방향, +x = 그 오른쪽.
    바닥은 z = -cam_height, 천장은 z = z_ceil (> 0).
    ※ HorizonNet 원본 코드(post_proc.np_coor2xy)는 y 부호가 반대라서 방이 좌우 반전됨.
      여기서는 반전 없는 좌표계를 쓰려고 변환 함수를 직접 정의함 (pano_px_to_ray / point_to_pano_px).
    GLB/OBJ로 저장할 때만 일반 3D 프로그램 관례에 맞춰 y-up으로 바꿈: (x, y, z) → (x, z, -y)
"""
import argparse
import base64
import io
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates

# HorizonNet 코드는 이 파일 옆의 HorizonNet/ 폴더(클론)에서 import
HORIZONNET_DIR = Path(__file__).resolve().parent / "HorizonNet"
sys.path.insert(0, str(HORIZONNET_DIR))

DEFAULT_PTH = HORIZONNET_DIR / "ckpt" / "resnet50_rnn__mp3d.pth"   # Matterport3D(실사)로 학습된 가중치
MODEL_W, MODEL_H = 1024, 512                                        # HorizonNet 입력 크기 고정
MAX_TEX_SIDE = 4096                                                 # 텍스처 한 변 최대 픽셀


# ================================================================== 1. 경계 감지 (HorizonNet)
def ensure_weights(pth):
    """가중치 파일이 없으면 Hugging Face에서 내려받음 (최초 1회, 약 330MB)"""
    pth = Path(pth)
    if pth.exists():
        return pth
    from huggingface_hub import hf_hub_download
    print(f"가중치 다운로드: {pth.name}")
    return Path(hf_hub_download(repo_id="sunset1995/HorizonNet", filename=pth.name,
                                local_dir=str(pth.parent)))


def detect_layout(pano_small, pth, cuboid):
    """
    pano_small: 1024x512 RGB uint8 배열
    반환
        cor_px : (2N, 2) 모서리 픽셀 좌표 (1024x512 기준, 열 x / 행 y).
                 짝수 행 = 천장 모서리, 홀수 행 = 같은 열의 바닥 모서리. 열이 증가하는 순서(시계방향).
        z0, z1 : HorizonNet 내부 값. 카메라→천장 거리를 50으로 둘 때 바닥은 z1(<0). 검증용으로만 사용.
        raw_vis: HorizonNet 원시 출력 시각화 이미지
    """
    import torch
    from model import HorizonNet
    from inference import inference
    from misc.utils import load_trained_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_trained_model(HorizonNet, str(ensure_weights(pth))).to(device).eval()

    # [1, 3, 512, 1024], 값 범위 0~1 (정규화는 모델 안에서 함)
    x = torch.from_numpy(pano_small.transpose(2, 0, 1).copy()).float()[None] / 255.0
    with torch.no_grad():
        # inference(): 모델 출력(경계 2줄 + 모서리 확률) → 피크 검출 → 맨해튼 벽 투표 → 모서리
        cor_id, z0, z1, raw_vis = inference(net=net, x=x, device=device,
                                            visualize=True, force_cuboid=cuboid)
    cor_px = cor_id * np.array([MODEL_W, MODEL_H], np.float32)   # 0~1 정규화 → 픽셀
    return cor_px, float(z0), float(z1), raw_vis


# ================================================================== 2. 파노라마 ↔ 3D 변환
# equirectangular 규칙: 열 → 수평각 u ∈ (-π, π), 행 → 앙각 v ∈ (-π/2, π/2) (위쪽이 +)
#   u = ((col + 0.5) / W - 0.5) · 2π         (HorizonNet post_proc.np_coorx2u 와 같은 식)
#   v = -((row + 0.5) / H - 0.5) · π         (HorizonNet post_proc.np_coory2v 와 같은 식)
# 광선 방향 = (cos v · sin u,  cos v · cos u,  sin v)  → u=0이면 +y, u=+90°이면 +x(오른쪽)

def pano_px_to_uv(col, row, W, H):
    u = ((np.asarray(col) + 0.5) / W - 0.5) * 2 * np.pi
    v = -((np.asarray(row) + 0.5) / H - 0.5) * np.pi
    return u, v


def point_to_pano_px(xyz, W, H):
    """3D 점(…,3) → 그 점이 찍힌 파노라마 픽셀 (col, row). 위 식의 역변환."""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    u = np.arctan2(x, y)                    # 수평각
    v = np.arctan2(z, np.hypot(x, y))       # 앙각
    col = (u / (2 * np.pi) + 0.5) * W - 0.5
    row = (-v / np.pi + 0.5) * H - 0.5
    return col, row


def build_room(cor_px, cam_h):
    """
    모서리 픽셀 → 3D 방.
    바닥 모서리: 바닥 평면 z = -cam_h 와 광선의 교점.
        광선을 수평거리 c 만큼 가면 높이는 c·tan v 이므로  c = -cam_h / tan(v_floor)
        → (x, y) = (c·sin u, c·cos u)
    천장 높이: 같은 열의 천장 모서리는 같은 수평거리 c 에 있으므로 z = c·tan(v_ceil). 모서리별 값의 평균.
    """
    ceil_px, floor_px = cor_px[0::2], cor_px[1::2]
    u, v_floor = pano_px_to_uv(floor_px[:, 0], floor_px[:, 1], MODEL_W, MODEL_H)
    _, v_ceil = pano_px_to_uv(ceil_px[:, 0], ceil_px[:, 1], MODEL_W, MODEL_H)

    z_floor = -cam_h
    c = z_floor / np.tan(v_floor)                       # 카메라~모서리 수평거리 (>0)
    floor_xy = np.stack([c * np.sin(u), c * np.cos(u)], 1)
    z_ceil_each = c * np.tan(v_ceil)
    z_ceil = float(z_ceil_each.mean())

    # 열 증가 순서 = 위에서 볼 때 시계방향(CW). 혹시 반대면 뒤집어서 항상 CW로 맞춤.
    # (신발끈 공식: 면적이 음수면 CW)
    if polygon_signed_area(floor_xy) > 0:
        floor_xy = floor_xy[::-1].copy()
    return floor_xy, z_floor, z_ceil, z_ceil_each


def polygon_signed_area(xy):
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


# ================================================================== 3. 텍스처 펴기
def sample_pano(pano, col, row):
    """파노라마에서 (col, row) 위치 색을 쌍선형 보간으로 가져옴. 좌우 끝은 이어져 있으므로 grid-wrap."""
    out = np.stack([
        map_coordinates(pano[..., ch].astype(np.float32), [row, col], order=1, mode="grid-wrap")
        for ch in range(3)
    ], -1)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_walls(floor_xy, z_floor, z_ceil, px_per_m):
    """
    벽 i = 바닥 다각형의 변 Pᵢ → Pᵢ₊₁.
    다각형이 CW이므로, 방 안에서 벽을 바라볼 때 Pᵢ가 왼쪽 끝, Pᵢ₊₁이 오른쪽 끝이 됨
    (그래서 텍스처가 좌우 반전 없이 사람이 보는 대로 나옴).
    """
    walls = []
    n = len(floor_xy)
    height = z_ceil - z_floor
    for i in range(n):
        L, R = floor_xy[i], floor_xy[(i + 1) % n]
        width = float(np.linalg.norm(R - L))
        ppm = min(px_per_m, MAX_TEX_SIDE / max(width, height))      # 너무 크면 해상도 낮춤
        tex_w, tex_h = max(1, round(width * ppm)), max(1, round(height * ppm))
        d = (R - L) / width                                          # 벽 방향 단위벡터 (왼→오)
        walls.append({
            "index": i,
            "name": f"wall_{i:02d}",
            # 3D 모서리 4개 (z-up 좌표계): 왼쪽 아래, 오른쪽 아래, 오른쪽 위, 왼쪽 위
            "left_bottom": [float(L[0]), float(L[1]), z_floor],
            "right_bottom": [float(R[0]), float(R[1]), z_floor],
            "right_top": [float(R[0]), float(R[1]), z_ceil],
            "left_top": [float(L[0]), float(L[1]), z_ceil],
            "width_m": width,
            "height_m": height,
            "texture": f"wall_{i:02d}.png",
            "texture_size": [tex_w, tex_h],
            "px_per_m": [tex_w / width, tex_h / height],
            # 방 안쪽을 향하는 법선. CW 다각형에서는 d를 시계방향으로 90° 돌린 것: (d_y, -d_x)
            "normal": [float(d[1]), float(-d[0]), 0.0],
        })
    return walls


def wall_px_to_3d(wall, col, row):
    """
    벽 텍스처의 픽셀 좌표 → 3D 점 (z-up, m).  창문 감지 bbox를 3D로 옮길 때 사용.
    col, row 는 픽셀 "경계" 기준 연속 좌표: (0,0) = 텍스처 왼쪽 위 모서리, (tex_w, tex_h) = 오른쪽 아래 모서리.
    예) bbox (x1,y1,x2,y2) → 네 꼭짓점 wall_px_to_3d(w,x1,y1), (x2,y1), (x2,y2), (x1,y2)
    """
    tex_w, tex_h = wall["texture_size"]
    L = np.array(wall["left_bottom"][:2])
    R = np.array(wall["right_bottom"][:2])
    z_top, z_bot = wall["left_top"][2], wall["left_bottom"][2]
    s = np.asarray(col, float) / tex_w                 # 0(왼쪽 끝) ~ 1(오른쪽 끝)
    t = np.asarray(row, float) / tex_h                 # 0(천장) ~ 1(바닥)
    xy = L + np.multiply.outer(s, R - L)
    z = z_top + t * (z_bot - z_top)
    return np.concatenate([xy, np.asarray(z)[..., None]], -1)


def render_wall_texture(pano, wall):
    """텍셀 중심마다 3D 점을 만들고(wall_px_to_3d), 그 점이 찍힌 파노라마 픽셀 색을 가져옴."""
    tex_w, tex_h = wall["texture_size"]
    cols, rows = np.meshgrid(np.arange(tex_w) + 0.5, np.arange(tex_h) + 0.5)   # 텍셀 중심
    xyz = wall_px_to_3d(wall, cols, rows)
    pc, pr = point_to_pano_px(xyz, pano.shape[1], pano.shape[0])
    return sample_pano(pano, pc, pr)


def render_plane_texture(pano, floor_xy, z, px_per_m):
    """
    바닥/천장: 다각형을 감싸는 사각형을 위에서 내려다본 정사영 이미지.
    이미지 열 → x 증가, 행 → y 감소 (행 0 = y 최대) 즉 +y가 위쪽인 평면도.
    ※ 천장 텍스처도 '위에서 본' 방향으로 저장되므로, 이미지 뷰어로 보면 아래에서 올려다본 것과 좌우 반대.
    """
    (xmin, ymin), (xmax, ymax) = floor_xy.min(0), floor_xy.max(0)
    ppm = min(px_per_m, MAX_TEX_SIDE / max(xmax - xmin, ymax - ymin))
    tex_w, tex_h = max(1, round((xmax - xmin) * ppm)), max(1, round((ymax - ymin) * ppm))
    cols, rows = np.meshgrid(np.arange(tex_w) + 0.5, np.arange(tex_h) + 0.5)
    x = xmin + cols / tex_w * (xmax - xmin)
    y = ymax - rows / tex_h * (ymax - ymin)
    xyz = np.stack([x, y, np.full_like(x, z)], -1)
    pc, pr = point_to_pano_px(xyz, pano.shape[1], pano.shape[0])
    tex = sample_pano(pano, pc, pr)
    bounds = (float(xmin), float(ymin), float(xmax), float(ymax))
    return tex, bounds


# ================================================================== 4. 메쉬 만들기
def triangulate(poly):
    """
    귀 자르기(ear clipping) 삼각분할. poly: (N,2), 위에서 볼 때 반시계(CCW).
    L자처럼 오목한 방도 올바르게 채움. 반환 삼각형도 CCW.
    """
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def inside(p, a, b, c):
        return cross(a, b, p) >= -1e-12 and cross(b, c, p) >= -1e-12 and cross(c, a, p) >= -1e-12

    idx, tris = list(range(len(poly))), []
    while len(idx) > 3:
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if cross(a, b, c) <= 1e-12:              # 오목한 꼭짓점이나 일직선 → 귀 아님
                continue
            if any(inside(poly[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue                              # 다른 꼭짓점이 안에 있으면 귀 아님
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            # 귀를 못 찾음(일직선 꼭짓점만 남은 경우 등) → 가장 납작한 꼭짓점 제거 후 계속
            k = min(range(len(idx)), key=lambda k: abs(cross(poly[idx[k - 1]], poly[idx[k]],
                                                               poly[idx[(k + 1) % len(idx)]])))
            idx.pop(k)
    tris.append(tuple(idx))
    return tris


def build_surfaces(floor_xy, z_floor, z_ceil, walls, wall_texs, floor_tex, ceil_tex, bounds):
    """
    면(surface) 목록: 각각 위치/UV/법선/삼각형/텍스처.
    UV는 OBJ 관례(v=0이 이미지 아래)로 만듦. GLB로 쓸 때 v를 뒤집음.
    삼각형 꼭짓점 순서는 '방 안에서 볼 때 반시계' → 법선이 방 안쪽을 향함.
    그래서 밖에서 보면 앞쪽 벽이 투명해져 방 내부가 보이는 '인형의 집' 시점이 됨.
    """
    surfaces = []
    for w, tex in zip(walls, wall_texs):
        pos = np.array([w["left_bottom"], w["right_bottom"], w["right_top"], w["left_top"]])
        uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
        surfaces.append({"name": w["name"], "pos": pos, "uv": uv,
                         "normal": np.array(w["normal"]), "tris": [(0, 1, 2), (0, 2, 3)],
                         "image": tex, "file": w["texture"]})

    xmin, ymin, xmax, ymax = bounds
    ccw = floor_xy[::-1]                                  # CW → CCW (위에서 볼 때)
    tris = triangulate(ccw)
    uv = np.stack([(ccw[:, 0] - xmin) / (xmax - xmin), (ccw[:, 1] - ymin) / (ymax - ymin)], 1)
    # 바닥: 위(방 안)에서 볼 때 CCW → 그대로. 천장: 아래에서 보므로 순서를 뒤집음.
    surfaces.append({"name": "floor", "pos": np.c_[ccw, np.full(len(ccw), z_floor)], "uv": uv,
                     "normal": np.array([0, 0, 1.0]), "tris": tris,
                     "image": floor_tex, "file": "floor.png"})
    surfaces.append({"name": "ceiling", "pos": np.c_[ccw, np.full(len(ccw), z_ceil)], "uv": uv,
                     "normal": np.array([0, 0, -1.0]), "tris": [t[::-1] for t in tris],
                     "image": ceil_tex, "file": "ceiling.png"})
    return surfaces


def subdivide_and_color(s, pano, cell):
    """
    색 투영용 촘촘한 메쉬 만들기.
    면 s의 삼각형마다 한 변이 약 cell(m)이 되도록 작은 삼각형으로 균일 분할하고,
    새 꼭짓점마다 그 3D 위치가 찍힌 파노라마 픽셀 색(point_to_pano_px → sample_pano)을 꼭짓점 색으로 넣음.

    삼각형 (A, B, C)를 n등분: 꼭짓점 P(i, j) = A + (i/n)(B−A) + (j/n)(C−A),  i + j ≤ n
      작은 삼각형 (P(i,j), P(i+1,j), P(i,j+1))  과  (P(i+1,j), P(i+1,j+1), P(i,j+1))
      → 둘 다 원래 삼각형과 꼭짓점 순서(앞면 방향)가 같음. UV도 같은 비율로 보간.
    반환: 원래 면 정보 + "pos", "uv", "tris"(numpy), "color"(N,3 uint8)가 촘촘한 버전으로 바뀐 dict
    """
    all_pos, all_uv, all_tri, base = [], [], [], 0
    for a, b, c in s["tris"]:
        A, B, C = s["pos"][a], s["pos"][b], s["pos"][c]
        uA, uB, uC = s["uv"][a], s["uv"][b], s["uv"][c]
        longest = max(np.linalg.norm(B - A), np.linalg.norm(C - B), np.linalg.norm(A - C))
        n = max(1, math.ceil(longest / cell))

        ij = np.array([(i, j) for i in range(n + 1) for j in range(n + 1 - i)], float)
        wi, wj = ij[:, :1] / n, ij[:, 1:] / n             # 무게 (B쪽, C쪽)
        all_pos.append(A + wi * (B - A) + wj * (C - A))
        all_uv.append(uA + wi * (uB - uA) + wj * (uC - uA))

        def vid(i, j):                                    # (i, j) 꼭짓점 번호. i행에는 n+1−i개가 있음
            return base + i * (n + 1) - i * (i - 1) // 2 + j

        for i in range(n):
            for j in range(n - i):
                all_tri.append((vid(i, j), vid(i + 1, j), vid(i, j + 1)))
                if j < n - i - 1:
                    all_tri.append((vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)))
        base += len(ij)

    pos = np.concatenate(all_pos)
    pc, pr = point_to_pano_px(pos, pano.shape[1], pano.shape[0])     # ← 색 투영
    return {**s, "pos": pos, "uv": np.concatenate(all_uv), "tris": np.array(all_tri),
            "color": sample_pano(pano, pc, pr)}


def to_yup(p):
    """z-up (x, y, z) → y-up (x, z, -y). 회전이라(행렬식 +1) 면 방향과 좌우가 보존됨."""
    p = np.asarray(p, float)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1)


# ================================================================== 5. 저장
def png_bytes(img):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def write_obj(out_dir, surfaces):
    """
    room.obj + room.mtl. 텍스처 PNG는 같은 폴더에 따로 저장되어 있어야 함.
    면에 "color"가 있으면 꼭짓점 줄을 'v x y z r g b'(r,g,b는 0~1)로 써서 꼭짓점 색도 함께 저장.
    (OBJ 표준 확장 문법으로 Blender, MeshLab 등이 읽음. 지원하지 않는 프로그램은 뒤의 r g b를 무시)
    """
    lines, mtl = ["# panorama to model.py", "mtllib room.mtl"], []
    v_off = 1                                             # OBJ 인덱스는 1부터
    for si, s in enumerate(surfaces):
        n = to_yup(s["normal"])
        lines.append(f"g {s['name']}")
        lines.append(f"usemtl {s['name']}")
        rgb = s["color"] / 255.0 if "color" in s else None
        for k, p in enumerate(to_yup(s["pos"])):
            col = f" {rgb[k, 0]:.3f} {rgb[k, 1]:.3f} {rgb[k, 2]:.3f}" if rgb is not None else ""
            lines.append(f"v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}{col}")
        for t in s["uv"]:
            lines.append(f"vt {t[0]:.5f} {t[1]:.5f}")
        lines.append(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}")   # 면(surface)마다 법선 1개 → si+1번
        for tri in s["tris"]:
            # f 꼭짓점/UV/법선 — 꼭짓점과 UV는 같은 번호를 씀
            lines.append("f " + " ".join(f"{v_off + i}/{v_off + i}/{si + 1}" for i in tri))
        v_off += len(s["pos"])
        mtl += [f"newmtl {s['name']}", "Ka 1 1 1", "Kd 1 1 1", "Ks 0 0 0", "illum 1",
                f"map_Kd {s['file']}", ""]
    (out_dir / "room.obj").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "room.mtl").write_text("\n".join(mtl), encoding="utf-8")


def write_ply(path, surfaces):
    """
    꼭짓점 색 메쉬를 binary PLY로 저장 (텍스처 없음 → 어떤 뷰어에서도 색이 보임).
    헤더(텍스트) 뒤에 꼭짓점마다 [x y z: float32 ×3][r g b: uint8 ×3],
    면마다 [꼭짓점 수 3: uint8][번호: int32 ×3] 가 이어짐.
    """
    pos = np.concatenate([to_yup(s["pos"]) for s in surfaces]).astype("<f4")
    col = np.concatenate([s["color"] for s in surfaces]).astype(np.uint8)
    offsets = np.cumsum([0] + [len(s["pos"]) for s in surfaces[:-1]])
    tris = np.concatenate([s["tris"] + o for s, o in zip(surfaces, offsets)]).astype("<i4")

    vert = np.zeros(len(pos), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                     ("r", "u1"), ("g", "u1"), ("b", "u1")])
    vert["x"], vert["y"], vert["z"] = pos.T
    vert["r"], vert["g"], vert["b"] = col.T
    face = np.zeros(len(tris), dtype=[("n", "u1"), ("i", "<i4", (3,))])
    face["n"], face["i"] = 3, tris

    header = "\n".join([
        "ply", "format binary_little_endian 1.0", "comment panorama to model.py (y-up, meters)",
        f"element vertex {len(vert)}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        f"element face {len(face)}", "property list uchar int vertex_indices", "end_header", ""])
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vert.tobytes())
        f.write(face.tobytes())


def build_glb(surfaces):
    """
    glTF 2.0 바이너리(.glb) 한 파일에 메쉬 + PNG 텍스처를 모두 넣음.
    구조: [12바이트 헤더][JSON 청크][BIN 청크]. 면마다 primitive 1개 + 재질 1개.
    재질은 KHR_materials_unlit(조명 무시) → 사진 색이 그대로 보임. 지원 안 하는 뷰어는 일반 재질로 표시.
    """
    binbuf = bytearray()
    gltf = {"asset": {"version": "2.0", "generator": "panorama to model.py"},
            "extensionsUsed": ["KHR_materials_unlit"],
            "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": "room"}],
            "meshes": [{"primitives": []}], "materials": [], "textures": [], "images": [],
            "samplers": [{"magFilter": 9729, "minFilter": 9729, "wrapS": 33071, "wrapT": 33071}],
            "accessors": [], "bufferViews": [], "buffers": []}

    def add_view(data, target=None):
        while len(binbuf) % 4:                            # 4바이트 정렬
            binbuf.append(0)
        view = {"buffer": 0, "byteOffset": len(binbuf), "byteLength": len(data)}
        if target:
            view["target"] = target
        binbuf.extend(data)
        gltf["bufferViews"].append(view)
        return len(gltf["bufferViews"]) - 1

    def add_accessor(arr, comp, typ, target, minmax=False):
        acc = {"bufferView": add_view(arr.tobytes(), target), "componentType": comp,
               "count": len(arr), "type": typ}
        if minmax:                                        # POSITION은 min/max 필수
            acc["min"], acc["max"] = arr.min(0).tolist(), arr.max(0).tolist()
        gltf["accessors"].append(acc)
        return len(gltf["accessors"]) - 1

    for i, s in enumerate(surfaces):
        pos = to_yup(s["pos"]).astype(np.float32)
        nrm = np.repeat(to_yup(s["normal"])[None], len(pos), 0).astype(np.float32)
        uv = s["uv"].astype(np.float32).copy()
        uv[:, 1] = 1.0 - uv[:, 1]                         # glTF UV 원점은 이미지 왼쪽 위
        idx = np.array(s["tris"], np.uint32).reshape(-1)
        img_view = add_view(png_bytes(s["image"]))
        gltf["images"].append({"bufferView": img_view, "mimeType": "image/png", "name": s["name"]})
        gltf["textures"].append({"source": i, "sampler": 0})
        gltf["materials"].append({"name": s["name"],
                                  "pbrMetallicRoughness": {"baseColorTexture": {"index": i},
                                                           "metallicFactor": 0.0,
                                                           "roughnessFactor": 1.0},
                                  "extensions": {"KHR_materials_unlit": {}}})
        gltf["meshes"][0]["primitives"].append({
            "attributes": {"POSITION": add_accessor(pos, 5126, "VEC3", 34962, minmax=True),
                           "NORMAL": add_accessor(nrm, 5126, "VEC3", 34962),
                           "TEXCOORD_0": add_accessor(uv, 5126, "VEC2", 34962)},
            "indices": add_accessor(idx, 5125, "SCALAR", 34963),
            "material": i, "mode": 4})

    while len(binbuf) % 4:
        binbuf.append(0)
    gltf["buffers"].append({"byteLength": len(binbuf)})
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)                           # JSON 청크는 공백으로 정렬
    total = 12 + 8 + len(js) + 8 + len(binbuf)
    return (struct.pack("<III", 0x46546C67, 2, total)
            + struct.pack("<II", len(js), 0x4E4F534A) + js
            + struct.pack("<II", len(binbuf), 0x004E4942) + bytes(binbuf))


VIEWER_HTML = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Room Viewer</title>
<style>
  html, body { margin: 0; height: 100%; background: #1e1f24; color: #eee; font: 14px sans-serif; overflow: hidden; }
  #ui { position: fixed; top: 12px; left: 12px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  button { padding: 6px 12px; border: 1px solid #555; background: #2c2e35; color: #eee; border-radius: 6px; cursor: pointer; }
  button.on { background: #4a6cf7; border-color: #4a6cf7; }
  #info { opacity: .75; }
</style></head><body>
<div id="ui">
  <button id="outside" class="on">바깥에서 보기</button>
  <button id="inside">카메라 위치에서 둘러보기</button>
  <span id="info">__INFO__</span>
</div>
<script type="importmap">
{"imports": {"three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
             "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}
</script>
<script type="text/plain" id="glb">__GLB_BASE64__</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1e1f24);
scene.add(new THREE.AmbientLight(0xffffff, 1));
const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.01, 200);
const controls = new OrbitControls(camera, renderer.domElement);
let box = null;

// 파일 안에 base64로 들어 있는 room.glb 를 읽음 (인터넷 없이도 모델 데이터는 있음)
const b64 = document.getElementById('glb').textContent.trim();
const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
new GLTFLoader().parse(bytes.buffer, '', gltf => {
  scene.add(gltf.scene);
  box = new THREE.Box3().setFromObject(gltf.scene);
  outside();
}, err => { document.getElementById('info').textContent = '모델 로드 실패: ' + err; });

function outside() {   // 방 밖 위쪽에서 내려다봄. 면이 안쪽만 그려지므로 앞쪽 벽은 투명해짐
  const c = box.getCenter(new THREE.Vector3()), s = box.getSize(new THREE.Vector3()).length();
  controls.enableZoom = controls.enablePan = true;
  controls.target.copy(c);
  camera.position.set(c.x + s * 0.6, c.y + s * 0.9, c.z + s * 0.6);
  camera.fov = 50; camera.updateProjectionMatrix();
  setActive('outside');
}
function inside() {    // 원점 = 촬영한 카메라 위치. 드래그하면 그 자리에서 고개를 돌림
  controls.enableZoom = controls.enablePan = false;
  controls.target.set(0, 0, 0);
  camera.position.set(0, 0, 0.001);
  camera.fov = 75; camera.updateProjectionMatrix();
  setActive('inside');
}
function setActive(id) {
  for (const b of document.querySelectorAll('button')) b.classList.toggle('on', b.id === id);
}
document.getElementById('outside').onclick = outside;
document.getElementById('inside').onclick = inside;
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
</script></body></html>
"""


def draw_overlay(pano, walls, path):
    """
    검증용: 복원한 3D 방의 모서리선(천장선/바닥선 = 빨강, 벽 모서리 세로선 = 노랑)을
    다시 파노라마에 투영해 그림. 사진 속 실제 경계와 겹치면 3D 복원이 맞는 것.
    """
    H, W = pano.shape[:2]
    img = Image.fromarray(pano.copy())
    draw = ImageDraw.Draw(img)
    lw = max(1, W // 600)

    def polyline(xyz, color):
        c, r = point_to_pano_px(xyz, W, H)
        for k in range(len(c) - 1):
            if abs(c[k + 1] - c[k]) < W / 2:              # 좌우 끝을 넘어가는 구간은 건너뜀
                draw.line([(c[k], r[k]), (c[k + 1], r[k + 1])], fill=color, width=lw)

    t = np.linspace(0, 1, 200)[:, None]
    for w in walls:
        lb, rb, rt, lt = (np.array(w[k]) for k in ("left_bottom", "right_bottom", "right_top", "left_top"))
        polyline(lb + t * (rb - lb), (255, 40, 40))       # 바닥선
        polyline(lt + t * (rt - lt), (255, 40, 40))       # 천장선
        polyline(lb + t * (lt - lb), (255, 220, 0))       # 벽 모서리 세로선
    img.save(path)


# ================================================================== 실행
def main():
    ap = argparse.ArgumentParser(description="360° 파노라마 → 텍스처 입힌 방 3D 모델")
    ap.add_argument("image", help="2:1 equirectangular 파노라마 (수평이 맞춰져 있어야 함)")
    ap.add_argument("--out", default="out", help="출력 상위 폴더")
    ap.add_argument("--cam_height", type=float, default=1.2,
                    help="바닥에서 카메라 렌즈까지 높이(m). 모델 전체 크기가 이 값에 비례함")
    ap.add_argument("--px_per_m", type=float, default=200, help="텍스처 해상도 (200 = 1px당 5mm)")
    ap.add_argument("--color_cell", type=float, default=0.04,
                    help="꼭짓점 색 격자 간격(m). 작을수록 PLY/OBJ 색이 선명하지만 파일이 커짐")
    ap.add_argument("--cuboid", action="store_true", help="벽 4개짜리 직육면체 방으로 강제")
    ap.add_argument("--pth", default=str(DEFAULT_PTH), help="HorizonNet 가중치 경로")
    a = ap.parse_args()

    src = Path(a.image)
    pano = np.array(Image.open(src).convert("RGB"))          # 텍스처는 원본 해상도에서 샘플링
    H, W = pano.shape[:2]
    if abs(W / H - 2) > 0.05:
        print(f"경고: 가로세로 비율이 {W / H:.2f}:1 입니다 (2:1이어야 정확함)")
    out_dir = Path(a.out) / src.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 경계 감지 — 모델 입력만 1024x512로 줄임
    small = np.array(Image.fromarray(pano).resize((MODEL_W, MODEL_H), Image.BICUBIC))
    print("[1/6] HorizonNet 경계 감지 …")
    cor_px, z0, z1, raw_vis = detect_layout(small, a.pth, a.cuboid)
    Image.fromarray(raw_vis).save(out_dir / "raw.png")

    # 2. 3D 복원
    floor_xy, z_floor, z_ceil, z_ceil_each = build_room(cor_px, a.cam_height)
    # 검증: HorizonNet 내부 값으로 계산한 천장 높이와 비교 (카메라→천장 : 카메라→바닥 = 50 : |z1|)
    z_ceil_hn = a.cam_height * 50.0 / abs(z1)
    print(f"[2/6] 모서리 {len(floor_xy)}개, 방 높이 {z_ceil - z_floor:.2f} m "
          f"(카메라 위 천장 {z_ceil:.2f} m, HorizonNet 내부값으로는 {z_ceil_hn:.2f} m, "
          f"모서리별 편차 ±{z_ceil_each.std():.3f} m), 바닥 면적 {-polygon_signed_area(floor_xy):.1f} m²")

    # 3. 벽 텍스처
    walls = make_walls(floor_xy, z_floor, z_ceil, a.px_per_m)
    wall_texs = []
    for w in walls:
        tex = render_wall_texture(pano, w)
        Image.fromarray(tex).save(out_dir / w["texture"])
        wall_texs.append(tex)
        # 검증: 텍스처 네 귀퉁이를 3D로 변환하면 벽 모서리와 정확히 같아야 함
        tw, th = w["texture_size"]
        assert np.allclose(wall_px_to_3d(w, 0, th), w["left_bottom"])
        assert np.allclose(wall_px_to_3d(w, tw, 0), w["right_top"])
        print(f"      {w['name']}: {w['width_m']:.2f} x {w['height_m']:.2f} m → {tw}x{th}px")
    print("[3/6] 벽 텍스처 저장")

    # 4. 바닥/천장 텍스처
    floor_tex, bounds = render_plane_texture(pano, floor_xy, z_floor, a.px_per_m)
    ceil_tex, _ = render_plane_texture(pano, floor_xy, z_ceil, a.px_per_m)
    Image.fromarray(floor_tex).save(out_dir / "floor.png")
    Image.fromarray(ceil_tex).save(out_dir / "ceiling.png")
    print("[4/6] 바닥/천장 텍스처 저장")

    # 5. 색 투영: 면을 촘촘하게 나누고 꼭짓점마다 파노라마 색을 넣음
    surfaces = build_surfaces(floor_xy, z_floor, z_ceil, walls, wall_texs, floor_tex, ceil_tex, bounds)
    dense = [subdivide_and_color(s, pano, a.color_cell) for s in surfaces]
    n_v, n_f = sum(len(s["pos"]) for s in dense), sum(len(s["tris"]) for s in dense)
    print(f"[5/6] 색 투영: 꼭짓점 {n_v:,}개, 삼각형 {n_f:,}개 (격자 {a.color_cell * 100:.0f} cm)")

    # 6. 모델 저장
    write_ply(out_dir / "room_colored.ply", dense)        # 꼭짓점 색만
    write_obj(out_dir, dense)                             # 텍스처 + 꼭짓점 색
    glb = build_glb(surfaces)                             # 텍스처 (면 수가 적은 원래 메쉬)
    (out_dir / "room.glb").write_bytes(glb)
    info = (f"{src.name} · 벽 {len(walls)}개 · 높이 {z_ceil - z_floor:.2f} m · "
            f"카메라 높이 {a.cam_height} m 기준")
    html = VIEWER_HTML.replace("__INFO__", info).replace(
        "__GLB_BASE64__", base64.b64encode(glb).decode())
    (out_dir / "viewer.html").write_text(html, encoding="utf-8")

    layout = {
        "source": src.name, "source_size": [W, H],
        "coordinate_system": "origin=camera, meters, right-handed, z-up, +y = panorama center column",
        "cam_height": a.cam_height, "z_floor": z_floor, "z_ceil": z_ceil,
        "floor_polygon_cw": floor_xy.tolist(),
        "corners_px_1024x512": cor_px.tolist(),
        "walls": walls,
        "floor": {"texture": "floor.png", "bounds_xyxy": bounds},
        "ceiling": {"texture": "ceiling.png", "bounds_xyxy": bounds},
    }
    (out_dir / "layout.json").write_text(json.dumps(layout, indent=2, ensure_ascii=False), encoding="utf-8")
    draw_overlay(pano, walls, out_dir / "overlay.png")
    print(f"[6/6] 저장 완료: {out_dir.resolve()}")
    print(f"      3D 모델 보기 → {(out_dir / 'viewer.html').resolve()} (브라우저로 열기)")
    print(f"      색 있는 메쉬  → {(out_dir / 'room_colored.ply').resolve()}")


if __name__ == "__main__":
    main()
