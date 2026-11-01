"""Failure analysis: error-type breakdown + qualitative side-by-side examples.

For each model, detections above an operating threshold (default score >= 0.3) are
greedily matched to GT per image and class at IoU >= 0.5. Every unmatched detection
is labelled as one of:
    localization : same class GT exists with 0.1 <= IoU < 0.5   (box in the right place but loose/offset)
    duplicate    : same class GT with IoU >= 0.5 already matched (double detection)
    confusion    : different-class GT with IoU >= 0.5            (e.g. truck called car)
    background   : nothing nearby                                (hallucination or unlabeled object)
Detections on iscrowd=1 ignore regions (KITTI DontCare / Van / Cyclist / too-hard boxes) are counted as
'ignored'. On KITTI, detections shorter than 25 px are dropped first, as in the KITTI moderate protocol.
Unmatched GT are misses, broken down by COCO size (small < 32^2 <= medium < 96^2 <= large).

Outputs (report/figures/failures/, results/failure_<dataset>.csv):
    error_breakdown_<dataset>.png   stacked counts per model
    examples_<dataset>_<k>.jpg      GT vs each model on the images with the most misses

Usage:
    python src/failure_analysis.py --dataset kitti
    python src/failure_analysis.py --dataset coco --models YOLO11l RT-DETR-L --num-examples 6
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import kitti_gt_path, load_config, resolve  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
ERR_COLORS = {"localization": "#2a78d6", "duplicate": "#6250d6", "confusion": "#eda100", "background": "#e34948",
              "miss_small": "#86b6ef", "miss_medium": "#3987e5", "miss_large": "#104281"}


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: N x 4, b: M x 4 in xywh."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(ax1[:, None], bx1[None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(ay1[:, None], by1[None]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return inter / np.maximum(union, 1e-9)


def ioa_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over the area of a (detections) vs ignore regions b. Same rule pycocotools uses for crowd."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    iw = np.clip(np.minimum((a[:, 0] + a[:, 2])[:, None], (b[:, 0] + b[:, 2])[None])
                 - np.maximum(a[:, 0][:, None], b[:, 0][None]), 0, None)
    ih = np.clip(np.minimum((a[:, 1] + a[:, 3])[:, None], (b[:, 1] + b[:, 3])[None])
                 - np.maximum(a[:, 1][:, None], b[:, 1][None]), 0, None)
    return iw * ih / np.maximum((a[:, 2] * a[:, 3])[:, None], 1e-9)


def size_bucket(area: float) -> str:
    return "small" if area < 32 ** 2 else ("medium" if area < 96 ** 2 else "large")


def analyse(gt_by_img, dets_by_img, cat_ids, thr, min_det_h=0.0):
    counts = Counter()
    misses_per_img = Counter()
    for img_id, all_gts in gt_by_img.items():
        gts = [g for g in all_gts if g["category_id"] in cat_ids and not g.get("iscrowd", 0)]
        crowd = [g for g in all_gts if g["category_id"] in cat_ids and g.get("iscrowd", 0)]
        c_box = np.array([g["bbox"] for g in crowd], dtype=float).reshape(-1, 4)
        c_cls = np.array([g["category_id"] for g in crowd])
        dets = sorted([d for d in dets_by_img.get(img_id, []) if d["score"] >= thr and d["category_id"] in cat_ids
                       and d["bbox"][3] >= min_det_h],
                      key=lambda d: -d["score"])
        g_box = np.array([g["bbox"] for g in gts], dtype=float).reshape(-1, 4)
        g_cls = np.array([g["category_id"] for g in gts])
        d_box = np.array([d["bbox"] for d in dets], dtype=float).reshape(-1, 4)
        ious = iou_matrix(d_box, g_box)
        ioas = ioa_matrix(d_box, c_box)
        matched = np.zeros(len(gts), bool)
        for i, d in enumerate(dets):
            same = g_cls == d["category_id"]
            cand = np.where(same & ~matched)[0]
            if len(cand) and ious[i, cand].max() >= 0.5:
                matched[cand[np.argmax(ious[i, cand])]] = True
                counts["tp"] += 1
                continue
            c_same = c_cls == d["category_id"]
            if c_same.any() and ioas[i, c_same].max() >= 0.5:
                counts["ignored"] += 1   # on a "don't care" region: neither TP nor FP
                continue
            if same.any() and ious[i, same].max() >= 0.5:
                counts["duplicate"] += 1
            elif same.any() and ious[i, same].max() >= 0.1:
                counts["localization"] += 1
            elif (~same).any() and ious[i, ~same].max() >= 0.5:
                counts["confusion"] += 1
            else:
                counts["background"] += 1
        for g, m in zip(gts, matched):
            if not m:
                counts[f"miss_{size_bucket(g['bbox'][2] * g['bbox'][3])}"] += 1
                misses_per_img[img_id] += 1
        counts["gt"] += len(gts)
    return counts, misses_per_img


def draw(img, boxes, color_bgr, labels=None, dashed=False):
    for k, b in enumerate(boxes):
        x, y, w, h = [int(round(v)) for v in b]
        if dashed:
            for (p, q) in (((x, y), (x + w, y)), ((x + w, y), (x + w, y + h)), ((x + w, y + h), (x, y + h)),
                           ((x, y + h), (x, y))):
                n = max(1, int(np.hypot(q[0] - p[0], q[1] - p[1]) // 10))
                for s in range(0, n, 2):
                    a = (int(p[0] + (q[0] - p[0]) * s / n), int(p[1] + (q[1] - p[1]) * s / n))
                    c = (int(p[0] + (q[0] - p[0]) * (s + 1) / n), int(p[1] + (q[1] - p[1]) * (s + 1) / n))
                    cv2.line(img, a, c, color_bgr, 2)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), color_bgr, 2)
        if labels:
            cv2.putText(img, labels[k], (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1,
                        cv2.LINE_AA)
    return img


def hex_to_bgr(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", choices=["coco", "kitti"], required=True)
    ap.add_argument("--models", nargs="*", default=None, help="default: all models in config with results")
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--num-examples", type=int, default=6)
    ap.add_argument("--report-dir", default="report")
    args = ap.parse_args()

    cfg = load_config(args.config)
    results = resolve(cfg["paths"]["results"])
    if args.dataset == "coco":
        gt_path, img_root = resolve(cfg["paths"]["coco_ann"]), resolve(cfg["paths"]["coco_images"])
    else:
        gt_path, img_root = kitti_gt_path(cfg), resolve(cfg["paths"]["kitti_root"])
    cat_ids = set(cfg["shared_coco_cat_ids"])  # analyse the shared classes on both datasets
    out_fig = resolve(args.report_dir) / "figures" / "failures"
    out_fig.mkdir(parents=True, exist_ok=True)

    gt = json.load(open(gt_path))
    imgs = {im["id"]: im for im in gt["images"]}
    cat_name = {c["id"]: c["name"] for c in gt["categories"]}
    models = args.models or [m["name"] for m in cfg["models"]]
    models = [m for m in models if (results / args.dataset / m / "detections.json").exists()]
    if not models:
        raise SystemExit("No detections found; run src/run_eval.py first.")

    # restrict GT to images actually evaluated (supports --limit runs)
    first = json.load(open(results / args.dataset / models[0] / "metrics.json"))
    n_eval = first["num_images"]
    eval_ids = sorted(imgs)[:n_eval]
    gt_by_img = {i: [] for i in eval_ids}
    for a in gt["annotations"]:
        if a["image_id"] in gt_by_img:
            gt_by_img[a["image_id"]].append(a)

    rows, all_dets, total_misses = [], {}, Counter()
    for m in models:
        dets = json.load(open(results / args.dataset / m / "detections.json"))
        by_img = defaultdict(list)
        for d in dets:
            by_img[d["image_id"]].append(d)
        all_dets[m] = by_img
        min_h = 25.0 if (args.dataset == "kitti" and cfg["kitti"].get("min_det_height", True)) else 0.0
        c, misses = analyse(gt_by_img, by_img, cat_ids, args.score_thr, min_h)
        total_misses.update(misses)
        rows.append({"model": m, **{k: c.get(k, 0) for k in
                     ["gt", "tp", "ignored", "localization", "duplicate", "confusion", "background",
                      "miss_small", "miss_medium", "miss_large"]}})
    df = pd.DataFrame(rows)
    df["recall@thr"] = df["tp"] / df["gt"].clip(lower=1)
    df["precision@thr"] = df["tp"] / (df[["tp", "localization", "duplicate", "confusion", "background"]]
                                      .sum(axis=1).clip(lower=1))
    df.to_csv(results / f"failure_{args.dataset}.csv", index=False)
    print(df.to_string(index=False))

    # stacked error breakdown
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.2), sharey=True)
    for ax, keys, title in ((axes[0], ["localization", "duplicate", "confusion", "background"],
                             f"False positives (score >= {args.score_thr})"),
                            (axes[1], ["miss_small", "miss_medium", "miss_large"], "Missed GT by object size")):
        left = np.zeros(len(df))
        for k in keys:
            ax.barh(df["model"], df[k], left=left, color=ERR_COLORS[k], label=k.replace("miss_", ""),
                    height=0.55, edgecolor="#fcfcfb", linewidth=2)
            left += df[k].values
        for yi, t in enumerate(left):
            ax.text(t, yi, f"  {int(t)}", va="center", fontsize=8)
        ax.set_title(title, loc="left", fontsize=10)
        ax.invert_yaxis()
        ax.legend(fontsize=7.5, ncol=len(keys), loc="lower center", bbox_to_anchor=(0.5, -0.38), frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.15)
        ax.set_xlim(left=0, right=max(1.0, ax.get_xlim()[1]))
    fig.tight_layout()
    fig.savefig(out_fig / f"error_breakdown_{args.dataset}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_fig / f"error_breakdown_{args.dataset}.pdf", bbox_inches="tight")
    plt.close(fig)

    # qualitative: images with the most misses summed over models
    worst = [i for i, _ in total_misses.most_common(args.num_examples)]
    if len(worst) < args.num_examples:  # fall back to the most crowded images
        crowded = sorted(gt_by_img, key=lambda i: -len(gt_by_img[i]))
        worst += [i for i in crowded if i not in worst][: args.num_examples - len(worst)]
    for k, img_id in enumerate(worst):
        im = imgs[img_id]
        base = cv2.imread(str(img_root / im["file_name"]))
        if base is None:
            print(f"skip (cannot read) {im['file_name']}")
            continue
        panels = []
        g = [a for a in gt_by_img[img_id] if a["category_id"] in cat_ids and not a.get("iscrowd", 0)]
        p = draw(base.copy(), [a["bbox"] for a in g], (255, 255, 255),
                 [cat_name.get(a["category_id"], "?") for a in g], dashed=True)
        cv2.putText(p, "Ground truth", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(p)
        for mi, m in enumerate(models):
            d = [x for x in all_dets[m].get(img_id, []) if x["score"] >= args.score_thr and x["category_id"] in cat_ids]
            col = hex_to_bgr(PALETTE[mi % len(PALETTE)])
            q = draw(base.copy(), [a["bbox"] for a in g], (255, 255, 255), dashed=True)
            q = draw(q, [x["bbox"] for x in d], col, [f"{cat_name.get(x['category_id'], '?')} {x['score']:.2f}" for x in d])
            cv2.putText(q, m, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
            panels.append(q)
        h = 360
        panels = [cv2.resize(x, (int(x.shape[1] * h / x.shape[0]), h)) for x in panels]
        cols = 1 if panels[0].shape[1] > 2.5 * h else 2   # stack wide KITTI frames vertically
        while len(panels) % cols:
            panels.append(np.full_like(panels[0], 252))
        grid = np.vstack([np.hstack(panels[r:r + cols]) for r in range(0, len(panels), cols)])
        tag = im.get("scene_description", im["file_name"])
        cv2.imwrite(str(out_fig / f"examples_{args.dataset}_{k}.jpg"), grid)
        print(f"example {k}: image {img_id} ({tag}) misses={total_misses[img_id]}")
    print(f"\nFigures -> {out_fig}\nCSV -> {results}/failure_{args.dataset}.csv")


if __name__ == "__main__":
    main()
