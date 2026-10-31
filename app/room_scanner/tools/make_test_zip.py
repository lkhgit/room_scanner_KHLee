"""Builds a scan bundle (.zip) from an existing logger.py session, for developing without the rig.

    python tools/make_test_zip.py "../mold_scanner/data/20261021_2225" test_bundle.zip [--heading 30] [--fake-panorama]

--fake-panorama adds a synthetic 2:1 test image (compass-coloured bands) so the room-image projection can be checked;
it is not a real room, so the 3D room model built from it is meaningless.
"""
import argparse
import json
import zipfile
from datetime import datetime
from pathlib import Path

import yaml


def fake_panorama(path):
    """Writes a synthetic 2:1 test panorama with coloured gradients and a grid."""
    import numpy as np
    import cv2
    h, w = 1024, 2048
    img = np.zeros((h, w, 3), np.uint8)
    x = np.linspace(0, 1, w)[None, :, None]
    y = np.linspace(0, 1, h)[:, None, None]
    ones = np.ones((h, w, 1))
    img[...] = np.concatenate([x * 255 * ones, ((1 - x) * 160 + y * 80), y * 255 * ones], axis=2).astype(np.uint8)
    for col in range(0, w, w // 24):
        img[:, col:col + 2] = 255
    for row in range(0, h, h // 12):
        img[row:row + 2, :] = 255
    cv2.putText(img, "CENTRE = rig az 0", (w // 2 - 260, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    cv2.imwrite(str(path), img)


def main():
    """Packs a logger.py session (and optionally a fake panorama) into a scan bundle."""
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("out")
    ap.add_argument("--heading", type=float, default=None)
    ap.add_argument("--fake-panorama", action="store_true")
    a = ap.parse_args()
    s = Path(a.session_dir)
    sess = yaml.safe_load((s / "session.yaml").read_text(encoding="utf-8")) or {}
    site = sess.get("site") or {}
    manifest = {
        "format_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "site": {"name": site.get("name"), "lat": site.get("lat"), "lon": site.get("lon")},
        "heading": {"deg": a.heading, "confidence": "manual" if a.heading is not None else None},
        "camera_height_m": 1.2,
    }
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        for name in ("scans.csv", "session.yaml", "outdoor.csv"):
            if (s / name).exists():
                z.write(s / name, f"surface/{name}")
        if a.fake_panorama:
            tmp = Path(a.out).with_suffix(".pano.jpg")
            fake_panorama(tmp)
            z.write(tmp, "panorama.jpg")
            tmp.unlink()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
