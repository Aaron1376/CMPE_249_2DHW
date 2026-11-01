"""Convert KITTI 2D object labels (training/label_2/*.txt) into a COCO-format ground-truth file.

KITTI label line (15 fields, space separated):
    type truncated occluded alpha x1 y1 x2 y2 h w l X Y Z rotation_y
    truncated: 0..1 (fraction outside the image), occluded: 0 visible, 1 partly, 2 largely, 3 unknown

Class mapping to COCO (official COCO json ids):
    Pedestrian -> person (1)    Car -> car (3)    Truck -> truck (8)
Classes with no clean COCO counterpart become iscrowd=1 "don't care" regions, so a COCO
detector is neither rewarded nor penalised for firing on them. This follows KITTI's own
evaluation, which does not count Van detections against Car or Person_sitting against Pedestrian:
    Van -> ignore for car and truck       Person_sitting -> ignore for person
    Cyclist -> ignore for person          DontCare -> ignore for every scored class
    Tram, Misc -> dropped (COCO would call them train/bus/other, which are not scored)

Difficulty (official KITTI object benchmark definitions):
    easy     : box height >= 40 px, fully visible (occluded 0),   truncation <= 15 %
    moderate : box height >= 25 px, partly occluded (<= 1),       truncation <= 30 %
    hard     : box height >= 25 px, difficult to see (<= 2),      truncation <= 50 %
Each scored box stores its level (0 easy, 1 moderate, 2 hard, 3 beyond hard). For an evaluation
at difficulty D, boxes with level <= D are scored and harder ones become ignore regions,
exactly like KITTI's easy / moderate / hard results. The file is written for the difficulty in
the config (default: moderate, KITTI's main ranking); src/run_eval.py also reports all three.

Usage:
    python src/prepare_kitti.py                      # uses configs/experiment.yaml
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from common import SHARED_CATEGORIES, dump_json, kitti_gt_path, load_config, resolve

SCORED = {"Pedestrian": 1, "Car": 3, "Truck": 8}
IGNORE_AS = {"Van": [3, 8], "Person_sitting": [1], "Cyclist": [1]}
DONTCARE = "DontCare"
DROPPED = ["Tram", "Misc"]
LEVELS = {"easy": 0, "moderate": 1, "hard": 2}
# (min height px, max occlusion, max truncation) per level, from the KITTI object benchmark page
LEVEL_RULES = [(40, 0, 0.15), (25, 1, 0.30), (25, 2, 0.50)]


def parse_label_line(line: str) -> dict | None:
    f = line.split()
    if len(f) < 8:
        return None
    return {"type": f[0], "truncated": float(f[1]), "occluded": int(float(f[2])),
            "bbox": [float(f[4]), float(f[5]), float(f[6]), float(f[7])]}


def difficulty_level(height: float, occluded: int, truncated: float) -> int:
    """0 easy, 1 moderate, 2 hard, 3 harder than hard (or unknown occlusion)."""
    for lvl, (min_h, max_occ, max_trunc) in enumerate(LEVEL_RULES):
        if height >= min_h and 0 <= occluded <= max_occ and truncated <= max_trunc:
            return lvl
    return 3


def convert_labels(objects: list[dict], img_id: int, start_ann_id: int, difficulty: int,
                   ignore_regions: bool, scored_ids: list[int], stats: Counter) -> list[dict]:
    """Turn the parsed objects of one image into COCO annotations."""
    anns, aid = [], start_ann_id

    def add(cat_id, xyxy, crowd, obj, level):
        nonlocal aid
        x1, y1, x2, y2 = xyxy
        w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
        if w <= 0 or h <= 0:
            return
        aid += 1
        anns.append({"id": aid, "image_id": img_id, "category_id": cat_id, "bbox": [x1, y1, w, h],
                     "area": w * h, "iscrowd": int(crowd), "kitti_type": obj["type"], "level": level,
                     "always_ignore": int(level is None), "occluded": obj["occluded"],
                     "truncated": obj["truncated"]})

    for o in objects:
        t, b = o["type"], o["bbox"]
        if t in SCORED:
            level = difficulty_level(b[3] - b[1], o["occluded"], o["truncated"])
            crowd = level > difficulty
            if crowd and not ignore_regions:
                stats[f"dropped_{t}_too_hard"] += 1
                continue
            add(SCORED[t], b, crowd, o, level)
            stats[("ignore_box_" if crowd else "scored_") + t] += 1
        elif ignore_regions and t in IGNORE_AS:
            for cid in IGNORE_AS[t]:
                if cid in scored_ids:
                    add(cid, b, True, o, None)
            stats[f"ignore_class_{t}"] += 1
        elif ignore_regions and t == DONTCARE:
            for cid in scored_ids:
                add(cid, b, True, o, None)
            stats["ignore_DontCare"] += 1
        else:
            stats[f"dropped_{t}"] += 1
    return anns


def convert(kitti_root: Path, kcfg: dict, scored_ids: list[int]):
    img_dir, lab_dir = kitti_root / "training" / "image_2", kitti_root / "training" / "label_2"
    if not img_dir.is_dir() or not lab_dir.is_dir():
        raise SystemExit(f"Expected {img_dir} and {lab_dir}. Run scripts/download_data.py --only kitti first.")
    difficulty = LEVELS[kcfg.get("difficulty", "moderate")]
    ignore = bool(kcfg.get("ignore_regions", True))

    label_files = sorted(lab_dir.glob("*.txt"))
    if kcfg.get("max_images"):
        label_files = label_files[: int(kcfg["max_images"])]
    images, annotations, stats = [], [], Counter()
    for lf in tqdm(label_files, desc="KITTI labels"):
        img_path = img_dir / f"{lf.stem}.png"
        if not img_path.exists():
            stats["missing_image"] += 1
            continue
        with Image.open(img_path) as im:  # reads the header only
            W, H = im.size
        img_id = int(lf.stem)
        images.append({"id": img_id, "file_name": f"training/image_2/{img_path.name}", "width": W, "height": H})
        objects = [o for o in (parse_label_line(x) for x in lf.read_text().splitlines()) if o]
        annotations += convert_labels(objects, img_id, len(annotations), difficulty, ignore, scored_ids, stats)
    # make annotation ids unique and 1-based
    for k, a in enumerate(annotations, start=1):
        a["id"] = k
    return images, annotations, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    kcfg = cfg["kitti"]
    scored_ids = cfg["shared_coco_cat_ids"]
    out_path = resolve(args.out) if args.out else kitti_gt_path(cfg)
    images, annotations, stats = convert(resolve(cfg["paths"]["kitti_root"]), kcfg, scored_ids)

    cats = [c for c in SHARED_CATEGORIES if c["id"] in scored_ids]
    gt = {
        "info": {
            "description": "KITTI 2D object, training split (7,481 labeled images), COCO category ids",
            "difficulty": kcfg.get("difficulty", "moderate"),
            "ignore_regions": kcfg.get("ignore_regions", True),
            "max_images": kcfg.get("max_images"),
            "scored": SCORED, "ignore_as": IGNORE_AS, "dontcare": "ignore for all scored classes",
            "dropped": DROPPED, "level_rules": {"easy": LEVEL_RULES[0], "moderate": LEVEL_RULES[1],
                                                 "hard": LEVEL_RULES[2]},
        },
        "images": images,
        "annotations": annotations,
        "categories": cats,
    }
    dump_json(gt, out_path, indent=None)
    dump_json(dict(sorted(stats.items())), out_path.with_suffix(".stats.json"))

    per_cls = Counter(a["category_id"] for a in annotations if not a["iscrowd"])
    names = {c["id"]: c["name"] for c in cats}
    n_ign = sum(a["iscrowd"] for a in annotations)
    print(f"\nWrote {out_path}")
    print(f"  images: {len(images)}   scored boxes: {len(annotations) - n_ign}   ignore regions: {n_ign}")
    for cid, n in sorted(per_cls.items()):
        print(f"  {names[cid]:>8}: {n}")
    print("  stats:", dict(sorted(stats.items())))


if __name__ == "__main__":
    main()
