"""Thin wrapper around pycocotools COCOeval returning a flat, JSON-friendly dict."""
from __future__ import annotations

import contextlib
import io

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

STAT_NAMES = [
    "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large",
]


def _per_class(E: COCOeval, coco_gt: COCO) -> dict:
    """Per-class AP@[.5:.95], AP50, AP75 and AR100 from the accumulated arrays (area=all, maxDet=100)."""
    prec = E.eval["precision"]  # [T, R, K, A, M]
    rec = E.eval["recall"]      # [T, K, A, M]
    iou_thrs = E.params.iouThrs
    out = {}
    for k, cat_id in enumerate(E.params.catIds):
        cat_id = int(cat_id)  # pycocotools.loadCats returns None for numpy ints
        name = coco_gt.cats[cat_id]["name"]

        def ap_at(t_idx=None):
            p = prec[:, :, k, 0, -1] if t_idx is None else prec[t_idx, :, k, 0, -1]
            p = p[p > -1]
            return float(np.mean(p)) if p.size else float("nan")

        r = rec[:, k, 0, -1]
        r = r[r > -1]
        out[name] = {
            "category_id": int(cat_id),
            "AP": ap_at(),
            "AP50": ap_at(int(np.argmin(np.abs(iou_thrs - 0.5)))),
            "AP75": ap_at(int(np.argmin(np.abs(iou_thrs - 0.75)))),
            "AR100": float(np.mean(r)) if r.size else float("nan"),
            "num_gt": int(len(coco_gt.getAnnIds(catIds=[cat_id], imgIds=E.params.imgIds, iscrowd=False))),
        }
    return out


def evaluate(coco_gt: COCO, detections: list[dict], cat_ids: list[int] | None = None,
             img_ids: list[int] | None = None, verbose: bool = True) -> dict:
    """Run standard COCO bbox evaluation. Returns {'overall': {...}, 'per_class': {...}, 'summary': str}."""
    cat_ids = cat_ids or sorted(coco_gt.getCatIds())
    img_ids = img_ids or sorted(coco_gt.getImgIds())

    img_set = set(img_ids)
    dets = [d for d in detections if d["category_id"] in set(cat_ids) and d["image_id"] in img_set]
    if not dets:
        return {"overall": {k: 0.0 for k in STAT_NAMES}, "per_class": {}, "summary": "no detections",
                "num_images": len(img_ids), "num_dets": 0}

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(dets)
    E = COCOeval(coco_gt, coco_dt, iouType="bbox")
    E.params.catIds = cat_ids
    E.params.imgIds = img_ids

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        E.evaluate()
        E.accumulate()
        E.summarize()
    summary = buf.getvalue()
    if verbose:
        print(summary)

    overall = {name: float(v) for name, v in zip(STAT_NAMES, E.stats)}
    return {
        "overall": overall,
        "per_class": _per_class(E, coco_gt),
        "summary": summary,
        "num_images": len(img_ids),
        "num_dets": len(dets),
    }
