"""Shared helpers: config loading, path resolution, COCO class-id mapping."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Ultralytics / YOLO 80-class index  ->  official COCO json category_id (1..90, with gaps).
COCO80_TO_COCO91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]
assert len(COCO80_TO_COCO91) == 80

# Classes shared between COCO and KITTI (COCO json ids): KITTI Pedestrian, Car, Truck.
SHARED_CATEGORIES = [
    {"id": 1, "name": "person"},
    {"id": 3, "name": "car"},
    {"id": 8, "name": "truck"},
]


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "configs" / "experiment.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve(p: str | os.PathLike) -> Path:
    """Resolve a config path relative to the repo root (absolute paths pass through)."""
    p = Path(os.path.expanduser(str(p)))
    return p if p.is_absolute() else (REPO_ROOT / p)


def kitti_gt_path(cfg: dict) -> Path:
    k = cfg["kitti"]
    n = k.get("max_images") or "all"
    return resolve(cfg["paths"]["kitti_gt"].format(difficulty=k.get("difficulty", "moderate"), n=n))


def get_model_cfg(cfg: dict, name: str) -> dict:
    for m in cfg["models"]:
        if m["name"] == name:
            return m
    names = [m["name"] for m in cfg["models"]]
    raise SystemExit(f"Model '{name}' not in config. Available: {names}")


def environment_info() -> dict:
    """Everything the reproducibility table needs, captured at run time."""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["cudnn"] = torch.backends.cudnn.version()
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_total_mem_MB"] = round(torch.cuda.get_device_properties(0).total_memory / 2**20)
    except Exception as e:  # pragma: no cover
        info["torch_error"] = repr(e)
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except Exception:
        pass
    try:
        import pycocotools  # noqa: F401
        from importlib.metadata import version

        info["pycocotools"] = version("pycocotools")
    except Exception:
        pass
    try:
        info["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        pass
    return info


def dump_json(obj, path: str | os.PathLike, indent: int | None = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)
