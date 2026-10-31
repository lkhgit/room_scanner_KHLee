"""
panorama to model_oneformer_v2.py — builds a 3D model of a room from a single 360° 2:1 panorama,
                                    detects the windows on each wall with OneFormer and paints them on the model
    (the version of "panorama to model_oneformer.py" with fixed post-processing in step 1, corner detection)

    (virtual env) .venv\\Scripts\\python.exe "panorama to model_oneformer_v2.py" panorama_test.jpg --cam_height 1.2

Changes in v2 (step 1)
    Fixes two problems of using HorizonNet's inference() as it is
    a. False corners in the middle of a wall (probability peaks from door frames or vertical pillar lines)
       → lift the ceiling boundary on both sides of a peak into the floor plan and compare the directions. If the
         turn is smaller than --min_corner_angle (default 30°), both sides are the same wall and the peak is dropped.
         A Manhattan room has an even number of corners, so with an odd count the least-turning peak is dropped too.
    b. Corners found correctly but the room folds up wrongly
       → HorizonNet's post-processing (post_proc.gen_ww) assumes the walls are parallel to the panorama's x/y axes
         (u = 0°, 90°, …). The input should have been aligned with preprocess.py first; without that step, corners
         were pushed to the intersections of axis-aligned walls.
         → estimate the wall direction from the ceiling boundary, rotate the panorama horizontally (np.roll),
           infer again and shift the resulting columns back (turn off with --no_align).
    The log prints the raw peaks, dropped peaks, turn angles, alignment rotation and the number of walls the
    post-processing had to force in (forced infer), and raw_peaks.png marks them
    (top band: green = kept peak, red = dropped peak / bottom: yellow = final corner columns)

Steps
    1. Detect the wall-ceiling / wall-floor boundaries and the wall corners with HorizonNet (reuses the code in the
       HorizonNet/ folder; the post-processing is done here)
    2. Rebuild the corners as 3D coordinates in metres from the measured camera height → the step that "folds" the room
    3. Unwrap a rectangular, face-on texture of every wall from the panorama (orthographic image of the wall plane)
    4. ★ Window detection: feed each wall texture from step 3 to OneFormer (shi-labs/oneformer_ade20k_swin_large) for
       semantic segmentation; pixels whose "windowpane" probability exceeds --window_thresh (default 0.3) form the
       window mask. Blobs smaller than --min_window_area (m²) are dropped.
       → paint the mask over the wall texture in a translucent colour (the painted texture goes on the 3D model)
       → convert the bbox of every blob (connected component) to 3D with wall_px_to_3d() and save it in layout.json
       → cut that bbox (a rectangle with edges parallel/perpendicular to the floor that encloses the window
         segment) out of the wall mesh as a hole
    5. Floor and ceiling get a texture seen from above
    6. Colour projection: split every surface into a fine grid (--color_cell, default 4 cm) and store the panorama
       colour projected onto each vertex
       (wall vertices on the window mask get the window colour blended in)
       → colours show even in viewers that cannot read textures (MTL), e.g. Visual Studio
    7. Save the 3D model: viewer.html (browser viewer), room_colored.ply, room.obj, room.glb

Output folder (--out/<image name>/)
    viewer.html     ★ the most reliable way to look at it. Opened in a browser it shows the textures and window
                    holes as they are (the model is inside the file; only the three.js library is loaded online)
    room_colored.ply  mesh coloured by vertex colours only. Colours show without textures, so they appear in
                    almost any 3D program, such as MeshLab / Blender / CloudCompare
    room.obj/.mtl   mesh with both textures (PNG) and vertex colours. Programs that read textures (Blender etc.)
                    show the sharp textures, the others the vertex colours (the PNGs must be in the same folder)
    room.glb        a single file with the textures included (Blender, web glTF viewers, etc.)
    wall_XX.png         original wall texture (input of the window detection)
    wall_XX_window.png  window mask (white = window)
    wall_XX_painted.png wall texture with the windows painted (goes on the 3D model, but the painted part is cut
                        out as a hole and so not visible)
    floor.png, ceiling.png
    layout.json     3D corners, size (m) and texture size of every wall + "windows": 3D rectangles of the detected windows
    overlay.png     lines of the rebuilt 3D room (red/yellow) and the detected window rectangles (light blue)
                    projected back onto the panorama (for checking: they should match the real edges and windows)
    raw.png         raw HorizonNet output (top: corner probability, bottom: boundaries)

Coordinates (inside this script and in layout.json)
    origin = centre of the camera lens, units m, right-handed, z axis up.
    +y = the direction the panorama's centre column (u=0) faces, +x = to its right.
    The floor is at z = -cam_height, the ceiling at z = z_ceil (> 0).
    ※ HorizonNet's original code (post_proc.np_coor2xy) has the y sign flipped, which mirrors the room left-right.
      To use an unmirrored frame here, the conversion functions are defined directly (pano_px_to_ray / point_to_pano_px).
    Only when saving GLB/OBJ is it changed to y-up, as usual in 3D programs: (x, y, z) → (x, z, -y)
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
from scipy.ndimage import binary_erosion, find_objects, label, map_coordinates

# HorizonNet code is imported from the HorizonNet/ folder (a git submodule) next to this file
HORIZONNET_DIR = Path(__file__).resolve().parent / "HorizonNet"
sys.path.insert(0, str(HORIZONNET_DIR))

DEFAULT_PTH = HORIZONNET_DIR / "ckpt" / "resnet50_rnn__mp3d.pth"   # weights trained on Matterport3D (real photos)
MODEL_W, MODEL_H = 1024, 512                                        # fixed HorizonNet input size
MAX_TEX_SIDE = 4096                                                 # largest texture side in pixels

SEG_MODEL = "shi-labs/oneformer_ade20k_swin_large"                  # window detection model (ADE20K, 150 classes)
WINDOW_COLOR = (0, 180, 255)                                        # colour to paint windows with (light blue, RGB)


# ================================================================== 1. boundary detection (HorizonNet)
def ensure_weights(pth):
    """Downloads the weights from Hugging Face if the file is missing (first run only, about 330 MB)."""
    pth = Path(pth)
    if pth.exists():
        return pth
    from huggingface_hub import hf_hub_download
    print(f"가중치 다운로드: {pth.name}")
    return Path(hf_hub_download(repo_id="sunset1995/HorizonNet", filename=pth.name,
                                local_dir=str(pth.parent)))


def run_horizonnet(net, device, img):
    """
    img: 1024x512 RGB uint8 → network output (same conversion as the first part of HorizonNet's inference())
    Returns
        y_bon  : (2, 1024) row of the ceiling boundary / floor boundary in every column
        y_cor  : (1024,) wall corner probability 0-1 in every column
        raw_vis: visualisation of the raw output (top 30 rows = corner probability band, 3 rows gap, bottom 512 rows = boundaries)
    """
    import torch
    from dataset import visualize_a_data

    # [1, 3, 512, 1024], values 0-1 (normalisation happens inside the model)
    x = torch.from_numpy(img.transpose(2, 0, 1).copy()).float()[None] / 255.0
    with torch.no_grad():
        y_bon, y_cor = net(x.to(device))
    y_bon, y_cor = y_bon.cpu(), torch.sigmoid(y_cor).cpu()
    raw_vis = visualize_a_data(x[0], y_bon[0], y_cor[0])
    y_bon = (y_bon[0].numpy() / np.pi + 0.5) * MODEL_H - 0.5          # elevation (radians) → row
    y_bon[0] = np.clip(y_bon[0], 1, MODEL_H / 2 - 1)
    y_bon[1] = np.clip(y_bon[1], MODEL_H / 2 + 1, MODEL_H - 2)
    return y_bon, y_cor[0, 0].numpy(), raw_vis


def ceiling_xy(ceil_row):
    """Ceiling boundary (a row per column) → floor plan coordinates seen from above (1024, 2), in HorizonNet's frame (z0=50)."""
    from misc import post_proc
    return post_proc.np_coor2xy(np.stack([np.arange(MODEL_W), ceil_row], 1), z=50)


def estimate_yaw(ceil_row, step=4):
    """
    Estimates the wall direction θ of the room (degrees, -45 to 45). Rotating the panorama by -θ makes the walls
    parallel to the x/y axes.
    Lifts the ceiling boundary into the floor plan, cuts it into short pieces every `step` columns, multiplies the
    piece directions by 4 (folding the 90° period into one) and votes weighted by piece length → circular mean of the
    directions near the most common one (actually ±10°).
    (increasing column u by δ rotates every floor plan point by +δ, so the columns to shift = -θ/360·W)
    """
    xy = ceiling_xy(ceil_row)
    d = np.roll(xy, -step, 0) - xy                     # the last piece joins column 0 (the panorama wraps around)
    w = np.hypot(d[:, 0], d[:, 1])
    a4 = 4 * np.arctan2(d[:, 1], d[:, 0])
    hist = np.bincount(np.round(np.degrees(a4)).astype(int) % 360, w, 360)
    hist = np.convolve(np.r_[hist[-5:], hist, hist[:5]], np.ones(11), "valid")   # ±5° circular smoothing
    mode = np.radians(hist.argmax())
    near = np.abs(np.angle(np.exp(1j * (a4 - mode)))) < np.radians(40)          # leave out pieces at corners or occlusions
    return float(np.degrees(np.angle(np.sum(w[near] * np.exp(1j * a4[near])))) / 4)


def segment_direction(xy, a, b, margin=4):
    """Main direction (unit vector) of the ceiling boundary points from column a to b (wrapping). The margin columns near corners are blurry and left out."""
    n = (b - a) % MODEL_W
    idx = (a + np.arange(n + 1)) % MODEL_W
    if n > 2 * margin + 4:
        idx = idx[margin:-margin]
    p = xy[idx] - xy[idx].mean(0)
    return np.linalg.svd(p, full_matrices=False)[2][0]


def corner_angles(xy, peaks):
    """Angle between the wall directions left and right of every peak (0-90°): near 90° at a real corner, near 0° on the same wall."""
    n = len(peaks)
    dirs = [segment_direction(xy, peaks[i - 1], peaks[i]) for i in range(n)]   # dirs[i] = the stretch from peak i-1 to i
    cos = [abs(float(dirs[i] @ dirs[(i + 1) % n])) for i in range(n)]
    return np.degrees(np.arccos(np.minimum(cos, 1.0)))


def filter_peaks(xy, peaks, min_angle, min_n=4):
    """
    Drops peaks one at a time, least-turning first, re-evaluating after each (dropping one merges the stretches
    on both sides, which changes the angles).
        turn < min_angle → "직선" (straight: a false corner on the same wall)
        still odd count  → "홀수" (odd: a Manhattan room has an even number of corners)
    Stops at min_n corners. Returns the remaining peaks and [(column, angle, reason), …]
    """
    peaks, removed = list(peaks), []
    while len(peaks) > min_n:
        ang = corner_angles(xy, peaks)
        i = int(ang.argmin())
        if ang[i] < min_angle:
            reason = "직선"
        elif len(peaks) % 2:
            reason = "홀수"
        else:
            break
        removed.append((int(peaks[i]), float(ang[i]), reason))
        del peaks[i]
    return np.array(peaks, int), removed


def detect_layout(pano_small, pth, cuboid, min_v=0.05, r=0.05, min_angle=30.0, align=True):
    """
    pano_small: 1024x512 RGB uint8 array
    Order: first inference → estimate wall direction → rotate the panorama and infer again → find peaks → drop
           false peaks → Manhattan wall voting (post_proc.gen_ww) → shift the corner columns back to the original panorama
    Returns
        cor_px : (2N, 2) corner pixel coordinates (in 1024x512, column x / row y).
                 even rows = ceiling corners, odd rows = floor corners in the same column. Ordered by increasing column (clockwise).
        z0, z1 : HorizonNet internal values. With the camera-to-ceiling distance set to 50, the floor is at z1 (<0). Only used for checking.
        raw_vis: visualisation of HorizonNet's raw output (original panorama, first inference)
        info   : diagnostics (alignment rotation, peaks, dropped peaks, number of walls gen_ww had to force)
    """
    import torch
    from shapely.geometry import Polygon
    if not (HORIZONNET_DIR / "model.py").exists():   # submodule not checked out (cloned without --recurse-submodules)
        sys.exit(f"HorizonNet code not found in {HORIZONNET_DIR}\n"
                 "Run this in the repository root:  git submodule update --init\n"
                 "(downloaded as a ZIP? extract https://github.com/lkhgit/HorizonNet into that folder instead)")
    from model import HorizonNet
    from inference import find_N_peaks
    from misc import post_proc
    from misc.utils import load_trained_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_trained_model(HorizonNet, str(ensure_weights(pth))).to(device).eval()

    y_bon, y_cor, raw_vis = run_horizonnet(net, device, pano_small)
    yaw, dx = 0.0, 0
    if align:
        yaw = estimate_yaw(y_bon[0])
        dx = int(round(-yaw / 360 * MODEL_W))
        if dx:                                           # infer again on the rotated image; from here on column coordinates refer to the rotated image
            y_bon, y_cor, _ = run_horizonnet(net, device, np.roll(pano_small, dx, axis=1))

    z0 = 50
    _, z1 = post_proc.np_refine_by_fix_z(*y_bon, z0)
    r_px = int(round(MODEL_W * r / 2))                   # minimum distance between peaks (NMS window)
    if cuboid:
        peaks_raw = find_N_peaks(y_cor, r=r_px, min_v=0, N=4)[0]
        peaks, removed = peaks_raw, []
    else:
        peaks_raw = find_N_peaks(y_cor, r=r_px, min_v=min_v)[0]
        peaks, removed = filter_peaks(ceiling_xy(y_bon[0]), peaks_raw, min_angle)

    # Manhattan wall voting (as in inference()). If the polygon self-intersects, fall back to a cuboid
    tol = abs(0.16 * z1 / 1.6)
    cor, xy_cor = post_proc.gen_ww(peaks, y_bon[0], z0, tol=tol, force_cuboid=cuboid)
    fallback = False
    if not cuboid:
        xy2d = np.zeros((len(xy_cor), 2), np.float32)
        for i in range(len(xy_cor)):
            xy2d[i, xy_cor[i]["type"]] = xy_cor[i]["val"]
            xy2d[i, xy_cor[i - 1]["type"]] = xy_cor[i - 1]["val"]
        if not Polygon(xy2d).is_valid:
            fallback = True
            peaks = find_N_peaks(y_cor, r=r_px, min_v=0, N=4)[0]
            cor, xy_cor = post_proc.gen_ww(peaks, y_bon[0], z0, tol=tol, force_cuboid=True)
    cor = np.hstack([cor, post_proc.infer_coory(cor[:, 1], z1 - z0, z0)[:, None]])

    # rotated columns → original panorama columns, back in increasing column order
    unroll = lambda c: (np.asarray(c, np.float64) - dx) % MODEL_W
    cor[:, 0] = unroll(cor[:, 0])
    cor = np.roll(cor, -int(cor[:, 0].argmin()), axis=0)
    cor_px = np.zeros((len(cor) * 2, 2), np.float32)
    cor_px[0::2] = cor[:, [0, 1]]
    cor_px[1::2] = cor[:, [0, 2]]

    actions = [c.get("action", "ori") for c in xy_cor]
    info = {
        "yaw_deg": yaw, "roll_px": dx,
        "peaks_raw": [[float(unroll(p)), float(y_cor[p])] for p in peaks_raw],     # [original column, probability]
        "peaks_kept": [float(unroll(p)) for p in peaks],
        "peaks_removed": [[float(unroll(p)), a, why] for p, a, why in removed],    # [original column, turn angle, reason]
        "forced_infer": actions.count("forced infer"),
        "forced_change": actions.count("forced change"),
        "fallback_cuboid": fallback,
    }
    return cor_px, float(z0), float(z1), raw_vis, info


def draw_peaks(raw_vis, info, cor_px):
    """Marks the peaks on raw_vis: green (kept) / red (dropped) in the top band, yellow (final corner columns) below."""
    img = Image.fromarray(raw_vis)
    d = ImageDraw.Draw(img)
    kept = set(round(c) for c in info["peaks_kept"])
    for c, _ in info["peaks_raw"]:
        d.line([(c, 0), (c, 29)], fill=(0, 255, 0) if round(c) in kept else (255, 0, 0), width=3)
    for c in cor_px[0::2, 0]:
        d.line([(c, 33), (c, img.height - 1)], fill=(255, 220, 0), width=1)
    return np.array(img)


# ================================================================== 2. panorama ↔ 3D conversion
# equirectangular convention: column → horizontal angle u ∈ (-π, π), row → elevation v ∈ (-π/2, π/2) (up is +)
#   u = ((col + 0.5) / W - 0.5) · 2π         (same formula as HorizonNet post_proc.np_coorx2u)
#   v = -((row + 0.5) / H - 0.5) · π         (same formula as HorizonNet post_proc.np_coory2v)
# ray direction = (cos v · sin u,  cos v · cos u,  sin v)  → +y for u=0, +x (right) for u=+90°

def pano_px_to_uv(col, row, W, H):
    """Panorama pixel (col, row) → horizontal angle u and elevation v in radians."""
    u = ((np.asarray(col) + 0.5) / W - 0.5) * 2 * np.pi
    v = -((np.asarray(row) + 0.5) / H - 0.5) * np.pi
    return u, v


def point_to_pano_px(xyz, W, H):
    """3D points (…,3) → the panorama pixel (col, row) that shows them. Inverse of the formulas above."""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    u = np.arctan2(x, y)                    # horizontal angle
    v = np.arctan2(z, np.hypot(x, y))       # elevation
    col = (u / (2 * np.pi) + 0.5) * W - 0.5
    row = (-v / np.pi + 0.5) * H - 0.5
    return col, row


def build_room(cor_px, cam_h):
    """
    Corner pixels → 3D room.
    Floor corners: intersection of the ray with the floor plane z = -cam_h.
        After a horizontal distance c the ray is at height c·tan v, so c = -cam_h / tan(v_floor)
        → (x, y) = (c·sin u, c·cos u)
    Ceiling height: the ceiling corner in the same column is at the same horizontal distance c, so z = c·tan(v_ceil).
    The mean over the corners is used.
    """
    ceil_px, floor_px = cor_px[0::2], cor_px[1::2]
    u, v_floor = pano_px_to_uv(floor_px[:, 0], floor_px[:, 1], MODEL_W, MODEL_H)
    _, v_ceil = pano_px_to_uv(ceil_px[:, 0], ceil_px[:, 1], MODEL_W, MODEL_H)

    z_floor = -cam_h
    c = z_floor / np.tan(v_floor)                       # horizontal distance camera to corner (>0)
    floor_xy = np.stack([c * np.sin(u), c * np.cos(u)], 1)
    z_ceil_each = c * np.tan(v_ceil)
    z_ceil = float(z_ceil_each.mean())

    # increasing columns = clockwise (CW) seen from above. If it comes out the other way, reverse it so it is always CW.
    # (shoelace formula: negative area means CW)
    if polygon_signed_area(floor_xy) > 0:
        floor_xy = floor_xy[::-1].copy()
    return floor_xy, z_floor, z_ceil, z_ceil_each


def polygon_signed_area(xy):
    """Signed area of a polygon by the shoelace formula (negative = clockwise)."""
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


# ================================================================== 3. texture unwrapping
def sample_pano(pano, col, row):
    """Bilinearly samples the panorama colour at (col, row). The left and right edges join, hence grid-wrap."""
    out = np.stack([
        map_coordinates(pano[..., ch].astype(np.float32), [row, col], order=1, mode="grid-wrap")
        for ch in range(3)
    ], -1)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_walls(floor_xy, z_floor, z_ceil, px_per_m):
    """
    Wall i = edge Pᵢ → Pᵢ₊₁ of the floor polygon.
    Since the polygon is CW, Pᵢ is the left end and Pᵢ₊₁ the right end when facing the wall from inside the room
    (so the texture comes out as a person sees it, not mirrored).
    """
    walls = []
    n = len(floor_xy)
    height = z_ceil - z_floor
    for i in range(n):
        L, R = floor_xy[i], floor_xy[(i + 1) % n]
        width = float(np.linalg.norm(R - L))
        ppm = min(px_per_m, MAX_TEX_SIDE / max(width, height))      # lower the resolution if it gets too big
        tex_w, tex_h = max(1, round(width * ppm)), max(1, round(height * ppm))
        d = (R - L) / width                                          # unit vector along the wall (left → right)
        walls.append({
            "index": i,
            "name": f"wall_{i:02d}",
            # the 4 corners in 3D (z-up): bottom left, bottom right, top right, top left
            "left_bottom": [float(L[0]), float(L[1]), z_floor],
            "right_bottom": [float(R[0]), float(R[1]), z_floor],
            "right_top": [float(R[0]), float(R[1]), z_ceil],
            "left_top": [float(L[0]), float(L[1]), z_ceil],
            "width_m": width,
            "height_m": height,
            "texture": f"wall_{i:02d}.png",
            "texture_size": [tex_w, tex_h],
            "px_per_m": [tex_w / width, tex_h / height],
            # normal pointing into the room. For a CW polygon it is d turned 90° clockwise: (d_y, -d_x)
            "normal": [float(d[1]), float(-d[0]), 0.0],
        })
    return walls


def wall_px_to_3d(wall, col, row):
    """
    Pixel coordinates on a wall texture → 3D point (z-up, m). Used to move window bboxes into 3D.
    col, row are continuous coordinates on pixel edges: (0,0) = top left corner of the texture, (tex_w, tex_h) = bottom right corner.
    e.g. bbox (x1,y1,x2,y2) → four corners wall_px_to_3d(w,x1,y1), (x2,y1), (x2,y2), (x1,y2)
    """
    tex_w, tex_h = wall["texture_size"]
    L = np.array(wall["left_bottom"][:2])
    R = np.array(wall["right_bottom"][:2])
    z_top, z_bot = wall["left_top"][2], wall["left_bottom"][2]
    s = np.asarray(col, float) / tex_w                 # 0 (left end) to 1 (right end)
    t = np.asarray(row, float) / tex_h                 # 0 (ceiling) to 1 (floor)
    xy = L + np.multiply.outer(s, R - L)
    z = z_top + t * (z_bot - z_top)
    return np.concatenate([xy, np.asarray(z)[..., None]], -1)


def render_wall_texture(pano, wall):
    """Makes a 3D point for every texel centre (wall_px_to_3d) and takes the colour of the panorama pixel that shows it."""
    tex_w, tex_h = wall["texture_size"]
    cols, rows = np.meshgrid(np.arange(tex_w) + 0.5, np.arange(tex_h) + 0.5)   # texel centres
    xyz = wall_px_to_3d(wall, cols, rows)
    pc, pr = point_to_pano_px(xyz, pano.shape[1], pano.shape[0])
    return sample_pano(pano, pc, pr)


def render_plane_texture(pano, floor_xy, z, px_per_m):
    """
    Floor/ceiling: orthographic image, seen from above, of the rectangle around the polygon.
    Image columns → increasing x, rows → decreasing y (row 0 = largest y), i.e. a floor plan with +y at the top.
    ※ The ceiling texture is also stored as seen from above, so in an image viewer it is mirrored compared with looking up at it.
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


# ================================================================== 3-2. window detection (OneFormer, semantic)
def load_oneformer(name):
    """
    Loads the OneFormer model and processor (the first run downloads about 900 MB from Hugging Face).
    Returns processor, model, device, window_ids
        window_ids = the ADE20K windowpane class ids (just 8 in this model).
                     ADE20K's original name is "windowpane, window", but this model's config (id2label) stores it
                     shortened to 'window ' (with a trailing space). So the names are split at commas and the
                     classes with a piece exactly equal to "windowpane" or "window" are taken.
                     (Not hard-coding the id keeps it working with other ADE20K models.)
    """
    import torch
    from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = OneFormerProcessor.from_pretrained(name)
    model = OneFormerForUniversalSegmentation.from_pretrained(name).to(device).eval()
    window_ids = [int(i) for i, n in model.config.id2label.items()
                  if {p.strip() for p in n.lower().split(",")} & {"windowpane", "window"}]
    if not window_ids:
        raise RuntimeError(f"{name} 의 클래스 목록에 'windowpane'이 없습니다")
    print(f"      OneFormer 창문 클래스: "
          + ", ".join(f"{i}={model.config.id2label[i]!r}" for i in window_ids))
    return processor, model, device, window_ids


def segment_windows(img, processor, model, device, window_ids, thresh):
    """
    One wall texture → window mask (H, W) bool.
    task_inputs=["semantic"]: one OneFormer model does semantic, instance and panoptic segmentation, and this text
    input picks the task. Only semantic (one class per pixel) is used here.
    The class score map is computed directly, the same way as post_process_semantic_segmentation
    (class probability per query × mask probability per query, summed over all queries).
    But instead of argmax (the single top class), the scores are normalised to sum to 1 per pixel and a pixel counts
    as window when its window probability exceeds thresh → lowering thresh also catches doubtful window pixels.
    """
    import torch
    import torch.nn.functional as F

    inputs = processor(images=Image.fromarray(img), task_inputs=["semantic"], return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    cls = outputs.class_queries_logits.softmax(-1)[..., :-1]         # (1, Q, 150), the last "no object" class left out
    masks = outputs.masks_queries_logits.sigmoid()                    # (1, Q, h, w)
    score = torch.einsum("bqc,bqhw->bchw", cls, masks)                # (1, 150, h, w) score per class
    score = F.interpolate(score, size=img.shape[:2], mode="bilinear", align_corners=False)[0]
    prob = score / score.sum(0, keepdim=True).clamp(min=1e-6)         # class probabilities sum to 1 per pixel
    win_prob = prob[window_ids].sum(0)                                # window probability (H, W)
    return (win_prob > thresh).cpu().numpy()


def extract_windows(wall, mask, min_area_m2):
    """
    Window mask → list of windows + cleaned-up mask.
    Each connected blob of pixels in the mask is one window; blobs smaller than min_area_m2 (noise) are dropped.
    Each blob's bbox (in texture pixels) becomes a 3D rectangle via wall_px_to_3d().
    The wall texture's horizontal axis is parallel to the floor and its vertical axis perpendicular, so the bbox is
    exactly "the rectangle with edges parallel/perpendicular to the floor that encloses the blob" = the hole to cut.
    Overlapping bboxes would make overlapping holes and a broken mesh, so overlapping bboxes are merged into one
    bbox around both.
    ※ Being a bbox, it can come out larger than the real window when a window is detected at a slant or two join in an L.
    """
    ppm_x, ppm_y = wall["px_per_m"]
    lab, _ = label(mask)                                   # 0 = background, 1..n = blob number
    clean = np.zeros_like(mask)
    rects = []                                             # [c0, r0, c1, r1, number of mask pixels]
    for k, sl in enumerate(find_objects(lab), start=1):
        part = lab[sl] == k
        if part.sum() / (ppm_x * ppm_y) < min_area_m2:      # pixel count → m²
            continue
        clean[sl] |= part
        rects.append([sl[1].start, sl[0].start, sl[1].stop, sl[0].stop, int(part.sum())])

    merged = True
    while merged:                                          # merge until no overlapping pair is left
        merged = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                a, b = rects[i], rects[j]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    rects[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]), a[4] + b[4]]
                    rects.pop(j)
                    merged = True
                    break
            if merged:
                break

    windows = []
    for c0, r0, c1, r1, n_px in sorted(rects):             # windows from left to right
        area_m2 = n_px / (ppm_x * ppm_y)
        corners = [wall_px_to_3d(wall, c, r).tolist()       # top left → top right → bottom right → bottom left
                   for c, r in ((c0, r0), (c1, r0), (c1, r1), (c0, r1))]
        windows.append({
            "wall": wall["index"],
            "bbox_px": [int(c0), int(r0), int(c1), int(r1)],  # in wall texture pixels (x1, y1, x2, y2)
            "corners_3d": corners,
            "width_m": (c1 - c0) / ppm_x,
            "height_m": (r1 - r0) / ppm_y,
            "sill_height_m": (wall["height_m"] - r1 / ppm_y),   # height from the floor to the bottom of the window
            "area_m2": float(area_m2),                          # actual mask area (at most the bbox area)
        })
    return windows, clean


def paint_windows(tex, mask, color, alpha):
    """
    Paints the window pixels of a wall texture.
    Inside: blended with the original colour (half and half at alpha=0.5) → the view outside still shows through
    Border (the outer 3 px of the mask): solid colour → the window outline stands out
    """
    out = tex.astype(np.float32)
    c = np.array(color, np.float32)
    out[mask] = out[mask] * (1 - alpha) + c * alpha
    edge = mask & ~binary_erosion(mask, iterations=3)
    out[edge] = c
    return out.astype(np.uint8)


def paint_vertex_colors(ds, mask, color, alpha):
    """
    Blends the window colour into the vertex colours of the dense mesh (from subdivide_and_color) as well.
    Vertex UV (0-1, v=0 at the bottom) → reads the mask at wall texture pixel (col = u·W, row = (1−v)·H).
    """
    th, tw = mask.shape
    col = ds["uv"][:, 0] * tw - 0.5
    row = (1 - ds["uv"][:, 1]) * th - 0.5
    on = map_coordinates(mask.astype(np.uint8), [row, col], order=0, mode="nearest").astype(bool)
    rgb = ds["color"].astype(np.float32)
    rgb[on] = rgb[on] * (1 - alpha) + np.array(color, np.float32) * alpha
    return {**ds, "color": rgb.astype(np.uint8)}


# ================================================================== 4. building the mesh
def triangulate(poly):
    """
    Ear clipping triangulation. poly: (N,2), counter-clockwise (CCW) seen from above.
    Also fills concave rooms such as L shapes correctly. The returned triangles are CCW too.
    """
    def cross(o, a, b):
        """2D cross product of the vectors o→a and o→b."""
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def inside(p, a, b, c):
        """Tells whether point p lies inside triangle (a, b, c)."""
        return cross(a, b, p) >= -1e-12 and cross(b, c, p) >= -1e-12 and cross(c, a, p) >= -1e-12

    idx, tris = list(range(len(poly))), []
    while len(idx) > 3:
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if cross(a, b, c) <= 1e-12:              # concave or collinear vertex → not an ear
                continue
            if any(inside(poly[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue                              # not an ear if another vertex lies inside
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            # no ear found (e.g. only collinear vertices left) → remove the flattest vertex and go on
            k = min(range(len(idx)), key=lambda k: abs(cross(poly[idx[k - 1]], poly[idx[k]],
                                                               poly[idx[(k + 1) % len(idx)]])))
            idx.pop(k)
    tris.append(tuple(idx))
    return tris


def wall_mesh_with_holes(wall, holes):
    """
    Wall mesh with rectangular holes (windows). holes: [(c0, r0, c1, r1), ...] in wall texture pixels, not overlapping.
    Cutting the wall into a grid at every x (column) and y (row) of the wall and hole corners puts every hole edge on a
    grid line, so each cell is either "entirely inside a hole" or "entirely wall".
    → leave out the cells whose centre lies in a hole and fill the others with 2 triangles each.
    Without holes there is one cell = the same rectangular wall as before.
    Returns pos (N,3), uv (N,2, v=0 at the bottom), tris [(i0,i1,i2), ...] (counter-clockwise seen from inside the room)
    """
    tw, th = wall["texture_size"]
    xs = sorted({0, tw, *(h[0] for h in holes), *(h[2] for h in holes)})
    ys = sorted({0, th, *(h[1] for h in holes), *(h[3] for h in holes)})
    gx, gy = np.meshgrid(xs, ys)                          # grid points (rows in ys order, columns in xs order)
    pos = wall_px_to_3d(wall, gx, gy).reshape(-1, 3)
    uv = np.stack([gx / tw, 1 - gy / th], -1).reshape(-1, 2).astype(float)

    def vid(i, j):                                        # index of grid point xs[i], ys[j]
        """Index of grid point (xs[i], ys[j]) in the vertex array."""
        return j * len(xs) + i

    tris = []
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            cx, cy = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            if any(c0 < cx < c1 and r0 < cy < r1 for c0, r0, c1, r1 in holes):
                continue                                  # a hole cell
            # bottom left → bottom right → top right → top left (row numbers grow downwards)
            lb, rb, rt, lt = vid(i, j + 1), vid(i + 1, j + 1), vid(i + 1, j), vid(i, j)
            tris += [(lb, rb, rt), (lb, rt, lt)]
    return pos, uv, tris


def build_surfaces(floor_xy, z_floor, z_ceil, walls, wall_texs, floor_tex, ceil_tex, bounds, wall_holes):
    """
    List of surfaces, each with positions/UVs/normal/triangles/texture.
    UVs follow the OBJ convention (v=0 at the bottom of the image); v is flipped when writing GLB.
    Triangle vertices are counter-clockwise seen from inside the room → the normals point into the room.
    Seen from outside, the near walls therefore turn see-through, giving a "doll's house" view into the room.
    wall_holes[i]: window holes to cut in wall i (see wall_mesh_with_holes)
    """
    surfaces = []
    for w, tex, holes in zip(walls, wall_texs, wall_holes):
        pos, uv, tris = wall_mesh_with_holes(w, holes)
        surfaces.append({"name": w["name"], "pos": pos, "uv": uv,
                         "normal": np.array(w["normal"]), "tris": tris,
                         "image": tex, "file": w["texture"]})

    xmin, ymin, xmax, ymax = bounds
    ccw = floor_xy[::-1]                                  # CW → CCW (seen from above)
    tris = triangulate(ccw)
    uv = np.stack([(ccw[:, 0] - xmin) / (xmax - xmin), (ccw[:, 1] - ymin) / (ymax - ymin)], 1)
    # floor: CCW seen from above (inside the room) → as is. Ceiling: seen from below, so the order is reversed.
    surfaces.append({"name": "floor", "pos": np.c_[ccw, np.full(len(ccw), z_floor)], "uv": uv,
                     "normal": np.array([0, 0, 1.0]), "tris": tris,
                     "image": floor_tex, "file": "floor.png"})
    surfaces.append({"name": "ceiling", "pos": np.c_[ccw, np.full(len(ccw), z_ceil)], "uv": uv,
                     "normal": np.array([0, 0, -1.0]), "tris": [t[::-1] for t in tris],
                     "image": ceil_tex, "file": "ceiling.png"})
    return surfaces


def subdivide_and_color(s, pano, cell):
    """
    Builds the dense mesh for colour projection.
    Splits every triangle of surface s evenly into small triangles with edges of about cell (m), and gives every new
    vertex the colour of the panorama pixel that shows its 3D position (point_to_pano_px → sample_pano).

    Triangle (A, B, C) split n ways: vertices P(i, j) = A + (i/n)(B−A) + (j/n)(C−A),  i + j ≤ n
      small triangles (P(i,j), P(i+1,j), P(i,j+1)) and (P(i+1,j), P(i+1,j+1), P(i,j+1))
      → both keep the original vertex order (front face direction). UVs are interpolated in the same ratio.
    Returns the surface dict with "pos", "uv", "tris" (numpy) and "color" (N,3 uint8) replaced by the dense versions
    """
    all_pos, all_uv, all_tri, base = [], [], [], 0
    for a, b, c in s["tris"]:
        A, B, C = s["pos"][a], s["pos"][b], s["pos"][c]
        uA, uB, uC = s["uv"][a], s["uv"][b], s["uv"][c]
        longest = max(np.linalg.norm(B - A), np.linalg.norm(C - B), np.linalg.norm(A - C))
        n = max(1, math.ceil(longest / cell))

        ij = np.array([(i, j) for i in range(n + 1) for j in range(n + 1 - i)], float)
        wi, wj = ij[:, :1] / n, ij[:, 1:] / n             # weights (towards B, towards C)
        all_pos.append(A + wi * (B - A) + wj * (C - A))
        all_uv.append(uA + wi * (uB - uA) + wj * (uC - uA))

        def vid(i, j):                                    # index of vertex (i, j); row i holds n+1−i vertices
            """Index of the subdivision vertex (i, j) in the vertex array."""
            return base + i * (n + 1) - i * (i - 1) // 2 + j

        for i in range(n):
            for j in range(n - i):
                all_tri.append((vid(i, j), vid(i + 1, j), vid(i, j + 1)))
                if j < n - i - 1:
                    all_tri.append((vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)))
        base += len(ij)

    pos = np.concatenate(all_pos)
    pc, pr = point_to_pano_px(pos, pano.shape[1], pano.shape[0])     # ← colour projection
    return {**s, "pos": pos, "uv": np.concatenate(all_uv), "tris": np.array(all_tri),
            "color": sample_pano(pano, pc, pr)}


def to_yup(p):
    """z-up (x, y, z) → y-up (x, z, -y). It is a rotation (determinant +1), so face orientation and handedness are kept."""
    p = np.asarray(p, float)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1)


# ================================================================== 5. saving
def png_bytes(img):
    """Encodes an image as PNG bytes."""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def write_obj(out_dir, surfaces):
    """
    room.obj + room.mtl. The texture PNGs must be saved separately in the same folder.
    If a surface has "color", vertex lines are written as 'v x y z r g b' (r,g,b in 0-1) to store vertex colours too.
    (A common OBJ extension read by Blender, MeshLab etc.; programs without support ignore the trailing r g b)
    """
    lines, mtl = ["# panorama to model.py", "mtllib room.mtl"], []
    v_off = 1                                             # OBJ indices start at 1
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
        lines.append(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}")   # one normal per surface → number si+1
        for tri in s["tris"]:
            # f vertex/UV/normal — vertex and UV share the same index
            lines.append("f " + " ".join(f"{v_off + i}/{v_off + i}/{si + 1}" for i in tri))
        v_off += len(s["pos"])
        mtl += [f"newmtl {s['name']}", "Ka 1 1 1", "Kd 1 1 1", "Ks 0 0 0", "illum 1",
                f"map_Kd {s['file']}", ""]
    (out_dir / "room.obj").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "room.mtl").write_text("\n".join(mtl), encoding="utf-8")


def write_ply(path, surfaces):
    """
    Saves the vertex-coloured mesh as binary PLY (no textures → colours show in any viewer).
    After the text header come [x y z: float32 ×3][r g b: uint8 ×3] for every vertex
    and [vertex count 3: uint8][indices: int32 ×3] for every face.
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
    Puts the mesh and all PNG textures into one glTF 2.0 binary (.glb) file.
    Layout: [12-byte header][JSON chunk][BIN chunk]. One primitive and one material per surface.
    Materials use KHR_materials_unlit (no lighting) → the photo colours show as they are. Viewers without support show a normal material.
    """
    binbuf = bytearray()
    gltf = {"asset": {"version": "2.0", "generator": "panorama to model.py"},
            "extensionsUsed": ["KHR_materials_unlit"],
            "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": "room"}],
            "meshes": [{"primitives": []}], "materials": [], "textures": [], "images": [],
            "samplers": [{"magFilter": 9729, "minFilter": 9729, "wrapS": 33071, "wrapT": 33071}],
            "accessors": [], "bufferViews": [], "buffers": []}

    def add_view(data, target=None):
        """Appends data to the binary buffer (4-byte aligned) and adds a buffer view for it."""
        while len(binbuf) % 4:                            # 4-byte alignment
            binbuf.append(0)
        view = {"buffer": 0, "byteOffset": len(binbuf), "byteLength": len(data)}
        if target:
            view["target"] = target
        binbuf.extend(data)
        gltf["bufferViews"].append(view)
        return len(gltf["bufferViews"]) - 1

    def add_accessor(arr, comp, typ, target, minmax=False):
        """Adds an accessor for an array, with min/max when required."""
        acc = {"bufferView": add_view(arr.tobytes(), target), "componentType": comp,
               "count": len(arr), "type": typ}
        if minmax:                                        # POSITION requires min/max
            acc["min"], acc["max"] = arr.min(0).tolist(), arr.max(0).tolist()
        gltf["accessors"].append(acc)
        return len(gltf["accessors"]) - 1

    for i, s in enumerate(surfaces):
        pos = to_yup(s["pos"]).astype(np.float32)
        nrm = np.repeat(to_yup(s["normal"])[None], len(pos), 0).astype(np.float32)
        uv = s["uv"].astype(np.float32).copy()
        uv[:, 1] = 1.0 - uv[:, 1]                         # glTF's UV origin is the top left of the image
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
    js += b" " * (-len(js) % 4)                           # the JSON chunk is padded with spaces
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


def draw_overlay(pano, walls, windows, path):
    """
    For checking: draws the edges of the rebuilt 3D room (ceiling/floor lines = red, vertical wall corners = yellow)
    and the detected 3D window rectangles (light blue) projected back onto the panorama.
    If they lie on the real edges and windows in the photo, the 3D reconstruction and window placement are right.
    """
    H, W = pano.shape[:2]
    img = Image.fromarray(pano.copy())
    draw = ImageDraw.Draw(img)
    lw = max(1, W // 600)

    def polyline(xyz, color):
        """Draws a 3D polyline projected onto the panorama."""
        c, r = point_to_pano_px(xyz, W, H)
        for k in range(len(c) - 1):
            if abs(c[k + 1] - c[k]) < W / 2:              # skip segments that cross the left/right edge
                draw.line([(c[k], r[k]), (c[k + 1], r[k + 1])], fill=color, width=lw)

    t = np.linspace(0, 1, 200)[:, None]
    for w in walls:
        lb, rb, rt, lt = (np.array(w[k]) for k in ("left_bottom", "right_bottom", "right_top", "left_top"))
        polyline(lb + t * (rb - lb), (255, 40, 40))       # floor line
        polyline(lt + t * (rt - lt), (255, 40, 40))       # ceiling line
        polyline(lb + t * (lt - lb), (255, 220, 0))       # vertical wall corner
    for win in windows:
        cs = np.array(win["corners_3d"])
        for k in range(4):                                # the rectangle's four edges
            polyline(cs[k] + t * (cs[(k + 1) % 4] - cs[k]), WINDOW_COLOR)
    img.save(path)


# ================================================================== main
def main():
    """Parses the arguments and runs the seven steps from panorama to 3D model files."""
    ap = argparse.ArgumentParser(description="360° 파노라마 → 텍스처 입힌 방 3D 모델")
    ap.add_argument("image", help="2:1 equirectangular 파노라마 (수평이 맞춰져 있어야 함)")
    ap.add_argument("--out", default="out", help="출력 상위 폴더")
    ap.add_argument("--cam_height", type=float, default=1.2,
                    help="바닥에서 카메라 렌즈까지 높이(m). 모델 전체 크기가 이 값에 비례함")
    ap.add_argument("--px_per_m", type=float, default=200, help="텍스처 해상도 (200 = 1px당 5mm)")
    ap.add_argument("--color_cell", type=float, default=0.04,
                    help="꼭짓점 색 격자 간격(m). 작을수록 PLY/OBJ 색이 선명하지만 파일이 커짐")
    ap.add_argument("--cuboid", action="store_true", help="벽 4개짜리 직육면체 방으로 강제")
    ap.add_argument("--cor_min_v", type=float, default=0.05,
                    help="모서리 확률이 이보다 큰 피크만 모서리 후보로 봄")
    ap.add_argument("--cor_r", type=float, default=0.05,
                    help="피크 사이 최소 간격 (이미지 너비 비율, 0.05 = 약 9°)")
    ap.add_argument("--min_corner_angle", type=float, default=30,
                    help="피크 양옆 벽이 이 각(°)보다 덜 꺾이면 가짜 모서리로 보고 버림 (0 = 끔, 홀수 보정은 유지)")
    ap.add_argument("--no_align", action="store_true", help="벽 방향 정렬(파노라마 수평 회전)을 하지 않음")
    ap.add_argument("--pth", default=str(DEFAULT_PTH), help="HorizonNet 가중치 경로")
    ap.add_argument("--seg_model", default=SEG_MODEL, help="OneFormer 모델 이름 (ADE20K 계열)")
    ap.add_argument("--window_alpha", type=float, default=0.5,
                    help="창문 칠하기 불투명도 (0 = 안 칠함, 1 = 완전히 덮음)")
    ap.add_argument("--window_thresh", type=float, default=0.3,
                    help="픽셀의 창문 확률이 이보다 크면 창문으로 봄 (낮을수록 많이 잡고 오탐도 늘어남)")
    ap.add_argument("--min_window_area", type=float, default=0.05,
                    help="이보다 작은(m²) 창문 덩어리는 잡음으로 보고 버림")
    a = ap.parse_args()

    src = Path(a.image)
    pano = np.array(Image.open(src).convert("RGB"))          # textures are sampled at the original resolution
    H, W = pano.shape[:2]
    if abs(W / H - 2) > 0.05:
        print(f"경고: 가로세로 비율이 {W / H:.2f}:1 입니다 (2:1이어야 정확함)")
    out_dir = Path(a.out) / src.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. boundary detection — only the model input is shrunk to 1024x512
    small = np.array(Image.fromarray(pano).resize((MODEL_W, MODEL_H), Image.BICUBIC))
    print("[1/7] HorizonNet 경계 감지 …")
    cor_px, z0, z1, raw_vis, det = detect_layout(small, a.pth, a.cuboid, a.cor_min_v, a.cor_r,
                                                 a.min_corner_angle, not a.no_align)
    Image.fromarray(raw_vis).save(out_dir / "raw.png")
    Image.fromarray(draw_peaks(raw_vis, det, cor_px)).save(out_dir / "raw_peaks.png")
    print(f"      벽 방향 정렬: {det['yaw_deg']:+.1f}° → 파노라마를 {det['roll_px']:+d}px 돌려 다시 추론")
    print("      원 피크 (열/확률): " + ", ".join(f"{c:.0f}/{p:.2f}" for c, p in det["peaks_raw"]))
    for c, ang, why in det["peaks_removed"]:
        print(f"      버린 피크: 열 {c:.0f} (꺾인 각 {ang:.1f}°, {why})")
    print(f"      남은 피크: {', '.join(f'{c:.0f}' for c in det['peaks_kept'])} → 최종 모서리 열: "
          + ", ".join(f"{c:.0f}" for c in cor_px[0::2, 0]))
    if det["forced_infer"] or det["forced_change"] or det["fallback_cuboid"]:
        print(f"      주의: 맨해튼 후처리가 벽을 억지로 고침 (끼워 넣음 {det['forced_infer']}, "
              f"방향 바꿈 {det['forced_change']}, 직육면체 대체 {det['fallback_cuboid']})")

    # 2. 3D reconstruction
    floor_xy, z_floor, z_ceil, z_ceil_each = build_room(cor_px, a.cam_height)
    # check: compare with the ceiling height from HorizonNet's internal values (camera→ceiling : camera→floor = 50 : |z1|)
    z_ceil_hn = a.cam_height * 50.0 / abs(z1)
    print(f"[2/7] 모서리 {len(floor_xy)}개, 방 높이 {z_ceil - z_floor:.2f} m "
          f"(카메라 위 천장 {z_ceil:.2f} m, HorizonNet 내부값으로는 {z_ceil_hn:.2f} m, "
          f"모서리별 편차 ±{z_ceil_each.std():.3f} m), 바닥 면적 {-polygon_signed_area(floor_xy):.1f} m²")

    # 3. wall textures
    walls = make_walls(floor_xy, z_floor, z_ceil, a.px_per_m)
    wall_texs = []
    for w in walls:
        tex = render_wall_texture(pano, w)
        Image.fromarray(tex).save(out_dir / w["texture"])
        wall_texs.append(tex)
        # check: the texture's four corners converted to 3D must equal the wall corners exactly
        tw, th = w["texture_size"]
        assert np.allclose(wall_px_to_3d(w, 0, th), w["left_bottom"])
        assert np.allclose(wall_px_to_3d(w, tw, 0), w["right_top"])
        print(f"      {w['name']}: {w['width_m']:.2f} x {w['height_m']:.2f} m → {tw}x{th}px")
    print("[3/7] 벽 텍스처 저장")

    # 4. window detection: OneFormer semantic on each wall texture → window mask → painting + 3D rectangles
    print(f"[4/7] OneFormer 창문 감지 ({a.seg_model}) …")
    processor, seg_net, seg_dev, window_ids = load_oneformer(a.seg_model)
    window_masks, painted_texs, windows = [], [], []
    for w, tex in zip(walls, wall_texs):
        raw_mask = segment_windows(tex, processor, seg_net, seg_dev, window_ids, a.window_thresh)
        found, mask = extract_windows(w, raw_mask, a.min_window_area)
        painted = paint_windows(tex, mask, WINDOW_COLOR, a.window_alpha)
        Image.fromarray((mask * 255).astype(np.uint8)).save(out_dir / f"{w['name']}_window.png")
        Image.fromarray(painted).save(out_dir / f"{w['name']}_painted.png")
        window_masks.append(mask)
        painted_texs.append(painted)
        windows += found
        print(f"      {w['name']}: 창문 {len(found)}개"
              + "".join(f"  [{f['width_m']:.2f} x {f['height_m']:.2f} m, 바닥에서 {f['sill_height_m']:.2f} m]"
                        for f in found))
    del seg_net                                            # free the memory (the model is large)

    # 5. floor/ceiling textures
    floor_tex, bounds = render_plane_texture(pano, floor_xy, z_floor, a.px_per_m)
    ceil_tex, _ = render_plane_texture(pano, floor_xy, z_ceil, a.px_per_m)
    Image.fromarray(floor_tex).save(out_dir / "floor.png")
    Image.fromarray(ceil_tex).save(out_dir / "ceiling.png")
    print("[5/7] 바닥/천장 텍스처 저장")

    # 6. colour projection: split the surfaces finely and give every vertex its panorama colour
    #    walls switch to the textures with the windows painted → the window colour shows in GLB/OBJ/viewer
    #    the window bboxes are cut out of the wall mesh as holes. All painted pixels lie inside the bboxes, so
    #    in the model the paint disappears with the holes, and wall_XX_painted.png stays for checking the detection
    wall_holes = [[win["bbox_px"] for win in windows if win["wall"] == w["index"]] for w in walls]
    surfaces = build_surfaces(floor_xy, z_floor, z_ceil, walls, painted_texs, floor_tex, ceil_tex, bounds,
                              wall_holes)
    for s, w in zip(surfaces, walls):                     # the first len(walls) surfaces are the walls (build_surfaces order)
        s["file"] = f"{w['name']}_painted.png"            # so the OBJ's MTL points to the painted textures
    dense = [subdivide_and_color(s, pano, a.color_cell) for s in surfaces]
    for i, mask in enumerate(window_masks):               # blend the window colour into the wall vertex colours too (for PLY/OBJ)
        dense[i] = paint_vertex_colors(dense[i], mask, WINDOW_COLOR, a.window_alpha)
    n_v, n_f = sum(len(s["pos"]) for s in dense), sum(len(s["tris"]) for s in dense)
    print(f"[6/7] 색 투영: 꼭짓점 {n_v:,}개, 삼각형 {n_f:,}개 (격자 {a.color_cell * 100:.0f} cm)")

    # 7. save the model
    write_ply(out_dir / "room_colored.ply", dense)        # vertex colours only
    write_obj(out_dir, dense)                             # textures + vertex colours
    glb = build_glb(surfaces)                             # textures (the original mesh with few faces)
    (out_dir / "room.glb").write_bytes(glb)
    info = (f"{src.name} · 벽 {len(walls)}개 · 창문 구멍 {len(windows)}개 · 높이 {z_ceil - z_floor:.2f} m · "
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
        "corner_detection": det,
        "walls": walls,
        "windows": windows,
        "window_model": a.seg_model,
        "floor": {"texture": "floor.png", "bounds_xyxy": bounds},
        "ceiling": {"texture": "ceiling.png", "bounds_xyxy": bounds},
    }
    (out_dir / "layout.json").write_text(json.dumps(layout, indent=2, ensure_ascii=False), encoding="utf-8")
    draw_overlay(pano, walls, windows, out_dir / "overlay.png")
    print(f"[7/7] 저장 완료: {out_dir.resolve()}")
    print(f"      3D 모델 보기 → {(out_dir / 'viewer.html').resolve()} (브라우저로 열기)")
    print(f"      색 있는 메쉬  → {(out_dir / 'room_colored.ply').resolve()}")


if __name__ == "__main__":
    main()
