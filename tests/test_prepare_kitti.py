"""Unit tests for the KITTI -> COCO conversion, the difficulty levels and the ignore-region handling.
No dataset download needed: a tiny fake KITTI folder is written to a temp dir."""
import contextlib
import io
from collections import Counter

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from prepare_kitti import convert, convert_labels, difficulty_level, parse_label_line
from run_eval import kitti_gt_at

SCORED_IDS = [1, 3, 8]

# Real KITTI label format: type trunc occ alpha x1 y1 x2 y2 h w l X Y Z ry
LABELS = """Car 0.00 0 -1.58 587.01 173.33 614.12 200.12 1.65 1.67 3.64 -0.65 1.71 46.70 -1.59
Car 0.00 0 1.85 387.63 181.54 423.81 203.12 1.67 1.87 3.69 -16.53 2.39 58.49 1.57
Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01
Pedestrian 0.80 3 -2.00 0.00 150.00 20.00 260.00 1.80 0.50 0.80 -5.00 1.50 6.00 -2.00
Van 0.00 1 -1.00 800.00 160.00 900.00 230.00 2.00 1.80 4.50 3.00 1.70 20.00 -1.00
Cyclist 0.00 0 1.00 300.00 170.00 340.00 240.00 1.70 0.60 1.80 -8.00 1.60 20.00 1.00
Truck 0.00 0 -1.50 1000.00 100.00 1200.00 250.00 3.00 2.50 9.00 10.00 1.50 30.00 -1.50
Tram 0.00 0 -1.50 100.00 100.00 300.00 250.00 3.50 2.50 15.00 -10.00 1.50 30.00 -1.50
DontCare -1 -1 -10 500.00 170.00 560.00 190.00 -1 -1 -1 -1000 -1000 -1000 -10
"""


def test_parse_line():
    o = parse_label_line(LABELS.splitlines()[0])
    assert o["type"] == "Car" and o["occluded"] == 0 and o["truncated"] == 0.0
    assert o["bbox"] == [587.01, 173.33, 614.12, 200.12]
    assert parse_label_line("") is None


def test_difficulty_levels_match_kitti_definitions():
    assert difficulty_level(40, 0, 0.15) == 0          # easy boundary
    assert difficulty_level(39.9, 0, 0.0) == 1         # too short for easy -> moderate
    assert difficulty_level(50, 1, 0.0) == 1           # partly occluded -> moderate
    assert difficulty_level(50, 0, 0.20) == 1          # truncation 20 % -> moderate
    assert difficulty_level(50, 2, 0.0) == 2           # largely occluded -> hard
    assert difficulty_level(50, 0, 0.45) == 2          # truncation 45 % -> hard
    assert difficulty_level(24, 0, 0.0) == 3           # shorter than 25 px -> beyond hard
    assert difficulty_level(50, 3, 0.0) == 3           # unknown occlusion -> beyond hard
    assert difficulty_level(50, 0, 0.60) == 3          # truncation > 50 % -> beyond hard


def test_convert_labels_mapping_and_ignore_regions():
    objs = [parse_label_line(x) for x in LABELS.splitlines()]
    st = Counter()
    anns = convert_labels(objs, img_id=7, start_ann_id=0, difficulty=1, ignore_regions=True,
                          scored_ids=SCORED_IDS, stats=st)
    scored = [a for a in anns if not a["iscrowd"]]
    crowd = [a for a in anns if a["iscrowd"]]
    # scored at moderate: car(h=26.8, easy fails on height -> moderate), car(h=21.6 -> beyond hard -> ignore),
    # pedestrian (easy), truck (easy). The truncated/unknown-occlusion pedestrian is an ignore region.
    assert sorted((a["kitti_type"], a["category_id"]) for a in scored) == [
        ("Car", 3), ("Pedestrian", 1), ("Truck", 8)]
    types_crowd = Counter(a["kitti_type"] for a in crowd)
    assert types_crowd["Van"] == 2                       # car + truck
    assert types_crowd["Cyclist"] == 1                   # person
    assert types_crowd["DontCare"] == 3                  # every scored class
    assert types_crowd["Car"] == 1 and types_crowd["Pedestrian"] == 1
    assert "Tram" not in {a["kitti_type"] for a in anns}
    van = [a for a in crowd if a["kitti_type"] == "Van"]
    assert {a["category_id"] for a in van} == {3, 8} and all(a["always_ignore"] for a in van)
    x, y, w, h = scored[0]["bbox"]
    assert w > 0 and h > 0 and all(a["image_id"] == 7 for a in anns)


def test_convert_labels_without_ignore_regions():
    objs = [parse_label_line(x) for x in LABELS.splitlines()]
    anns = convert_labels(objs, 1, 0, 1, False, SCORED_IDS, Counter())
    assert all(a["iscrowd"] == 0 for a in anns) and len(anns) == 3


def test_convert_folder(tmp_path):
    (tmp_path / "training" / "image_2").mkdir(parents=True)
    (tmp_path / "training" / "label_2").mkdir(parents=True)
    for stem in ("000000", "000001"):
        Image.fromarray(np.zeros((375, 1242, 3), np.uint8)).save(tmp_path / "training" / "image_2" / f"{stem}.png")
        (tmp_path / "training" / "label_2" / f"{stem}.txt").write_text(LABELS)
    images, anns, stats = convert(tmp_path, {"difficulty": "moderate", "ignore_regions": True}, SCORED_IDS)
    assert [im["id"] for im in images] == [0, 1]
    assert images[0]["width"] == 1242 and images[0]["height"] == 375
    assert images[0]["file_name"] == "training/image_2/000000.png"
    assert len({a["id"] for a in anns}) == len(anns) and min(a["id"] for a in anns) == 1
    images, _, _ = convert(tmp_path, {"difficulty": "moderate", "max_images": 1}, SCORED_IDS)
    assert len(images) == 1


def _gt_dict():
    objs = [parse_label_line(x) for x in LABELS.splitlines()]
    anns = convert_labels(objs, 1, 0, 1, True, SCORED_IDS, Counter())
    return {"images": [{"id": 1, "width": 1242, "height": 375, "file_name": "x.png"}],
            "categories": [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}, {"id": 8, "name": "truck"}],
            "annotations": anns}


def test_difficulty_variants():
    g = _gt_dict()
    n = {lvl: len(kitti_gt_at(g, lvl).getAnnIds(iscrowd=False)) for lvl in (0, 1, 2)}
    # easy: pedestrian + truck (car of height 26.8 is moderate); moderate adds that car; hard adds nothing here
    assert n == {0: 2, 1: 3, 2: 3}
    assert g["annotations"][0]["iscrowd"] in (0, 1)  # original dict untouched by the variant builder


def test_detection_on_dontcare_is_ignored():
    g = _gt_dict()
    gt = kitti_gt_at(g, 1)
    perfect = [{"image_id": 1, "category_id": a["category_id"], "bbox": a["bbox"], "score": 0.9}
               for a in gt.dataset["annotations"] if not a["iscrowd"]]
    on_dontcare = [{"image_id": 1, "category_id": 3, "bbox": [505.0, 172.0, 50.0, 16.0], "score": 0.99}]
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(gt, gt.loadRes(perfect + on_dontcare), "bbox")
        E.evaluate(); E.accumulate(); E.summarize()
    assert abs(E.stats[0] - 1.0) < 1e-6   # the top-scoring detection on DontCare does not hurt AP
