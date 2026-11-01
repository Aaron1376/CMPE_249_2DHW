"""Build a tiny synthetic dataset (no downloads) to smoke-test the whole pipeline.

Images: the two sample images bundled with Ultralytics (bus.jpg, zidane.jpg), each saved three
times (with a horizontal flip variant). Pseudo-GT: RT-DETR-L detections with score > 0.5, written as
  * a COCO-format GT file (80 categories)                     -> tests/smoke_data/coco_gt.json
  * a KITTI-format folder (PNG + label_2/*.txt, with a Van and a DontCare line per image)
                                                              -> tests/smoke_data/kitti/training/
so `prepare_kitti.py` runs on real KITTI-format files. Numbers produced are meaningless; this only
checks that every script runs end to end.
"""
import json
import os
import shutil
from pathlib import Path

import cv2
import ultralytics
from ultralytics import RTDETR

sys_root = Path(__file__).resolve().parents[1]
out = sys_root / "tests" / "smoke_data"
img_dir = out / "images"
img_dir.mkdir(parents=True, exist_ok=True)

assets = Path(ultralytics.__file__).parent / "assets"
names = []
for src in ("bus.jpg", "zidane.jpg"):
    im = cv2.imread(str(assets / src))
    for k in range(3):
        v = cv2.flip(im, 1) if k % 2 else im
        fn = f"{Path(src).stem}_{k}.jpg"
        cv2.imwrite(str(img_dir / fn), v)
        names.append((fn, v.shape[1], v.shape[0]))

from common import COCO80_TO_COCO91  # noqa: E402

wpath = sys_root / "weights" / "rtdetr-l.pt"
model = RTDETR(str(wpath) if wpath.exists() else "rtdetr-l.pt")
images, anns = [], []
aid = 0
for i, (fn, w, h) in enumerate(names, start=1):
    images.append({"id": i, "file_name": fn, "width": w, "height": h,
                   "camera": "CAM_FRONT" if i % 2 else "CAM_BACK", "night": i == 6, "rain": i == 5,
                   "scene_description": "smoke test"})
    r = model.predict(str(img_dir / fn), conf=0.5, verbose=False, device="cpu")[0]
    for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist()):
        aid += 1
        anns.append({"id": aid, "image_id": i, "category_id": COCO80_TO_COCO91[c],
                     "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})

cats80 = [{"id": cid, "name": n} for cid, n in zip(COCO80_TO_COCO91, model.names.values())]
json.dump({"images": images, "annotations": anns, "categories": cats80}, open(out / "coco_gt.json", "w"))

# KITTI-format copy: <id>.png + <id>.txt with Car / Pedestrian / Truck lines (others -> Misc)
kdir = out / "kitti" / "training"
(kdir / "image_2").mkdir(parents=True, exist_ok=True)
(kdir / "label_2").mkdir(parents=True, exist_ok=True)
to_kitti = {1: "Pedestrian", 3: "Car", 8: "Truck"}
for im in images:
    stem = f"{im['id']:06d}"
    cv2.imwrite(str(kdir / "image_2" / f"{stem}.png"), cv2.imread(str(img_dir / im["file_name"])))
    lines = []
    for a in anns:
        if a["image_id"] != im["id"]:
            continue
        x, y, w, h = a["bbox"]
        t = to_kitti.get(a["category_id"], "Misc")
        lines.append(f"{t} 0.00 0 0.00 {x:.2f} {y:.2f} {x + w:.2f} {y + h:.2f} 1.5 1.6 3.9 0.0 1.5 20.0 0.00")
    lines.append("Van 0.00 0 0.00 5.00 5.00 60.00 50.00 2.0 1.8 4.5 0.0 1.7 20.0 0.00")
    lines.append("DontCare -1 -1 -10 70.00 5.00 120.00 40.00 -1 -1 -1 -1000 -1000 -1000 -10")
    (kdir / "label_2" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
print(f"{len(images)} images, {len(anns)} pseudo-GT boxes -> {out} (COCO json + KITTI folder)")
