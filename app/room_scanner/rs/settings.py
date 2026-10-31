"""Loads config.yaml and resolves paths."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}


def _path(p: str) -> Path:
    """Resolves a config path relative to the app folder unless it is absolute."""
    q = Path(p)
    return (q if q.is_absolute() else ROOT / q).resolve()


SUNLIGHT_DIR = _path(CFG["paths"]["sunlight_dir"])
MOLD_DIR = _path(CFG["paths"]["mold_dir"])
FLOOD_DIR = _path(CFG["paths"]["flood_dir"])
KMA_CACHE = _path(CFG["paths"]["kma_cache"])
ROOMS_DIR = ROOT / "rooms"
ROOMS_DIR.mkdir(exist_ok=True)
