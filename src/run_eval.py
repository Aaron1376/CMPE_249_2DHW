"""Run one (model, dataset) experiment: inference + timing + GPU memory + COCO metrics.

Every model and both datasets go through this one script, so the preprocessing,
thresholds and metric code are identical across runs.

Outputs (results/<dataset>/<model>/):
    detections.json   COCO-format detections (official category ids)
    latency.csv       per-image preprocess / inference / postprocess ms
    metrics.json      overall + per-class metrics, speed summary, memory, config, environment
    command.txt       exact command line used

Examples:
    python src/run_eval.py --dataset coco     --model YOLO11l
    python src/run_eval.py --dataset kitti    --model RT-DETR-L
    python src/run_eval.py --dataset coco --model YOLO11l --limit 50 --device cpu   # smoke test
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import io
import sys
import time
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

from coco_metrics import evaluate
from common import (COCO80_TO_COCO91, dump_json, environment_info, get_model_cfg, kitti_gt_path, load_config,
                    resolve)

# KITTI difficulty levels and the minimum box height (px) used at each, from the KITTI object benchmark.
KITTI_LEVELS = {"easy": (0, 40), "moderate": (1, 25), "hard": (2, 25)}


def kitti_gt_at(gt_dict: dict, level: int) -> COCO:
    """COCO object where boxes harder than `level` (and always-ignore classes) are iscrowd ignore regions."""
    d = copy.deepcopy(gt_dict)
    for a in d["annotations"]:
        lvl = a.get("level")
        a["iscrowd"] = int(bool(a.get("always_ignore")) or lvl is None or lvl > level)
    with contextlib.redirect_stdout(io.StringIO()):
        c = COCO()
        c.dataset = d
        c.createIndex()
    return c


def load_model(weights: str, eval_idx: int | None = None):
    from ultralytics import RTDETR, YOLO

    is_detr = "rtdetr" in Path(weights).name.lower()
    # Prefer the copy in <repo>/weights/ (downloaded by scripts/download_weights.py).
    local = resolve("weights") / Path(weights).name
    if local.exists():
        weights = str(local)
    model = RTDETR(weights) if is_detr else YOLO(weights)
    if is_detr and eval_idx is not None:
        # Use fewer decoder layers at inference (documented Ultralytics speed knob, no retraining).
        head = model.model.model[-1]
        head.decoder.eval_idx = eval_idx
    return model, is_detr


def model_complexity(model) -> dict:
    out = {}
    try:
        n_params = sum(p.numel() for p in model.model.parameters())
        out["params_M"] = round(n_params / 1e6, 2)
    except Exception:
        pass
    try:
        from ultralytics.utils.torch_utils import get_flops

        out["GFLOPs_640"] = round(float(get_flops(model.model, imgsz=640)), 1)
    except Exception:
        pass
    return out


def summarize_latency(rows: list[dict]) -> dict:
    out = {}
    for key in ("preprocess", "inference", "postprocess", "total"):
        v = np.array([r[key] for r in rows], dtype=float)
        out[key] = {"mean": float(v.mean()), "median": float(np.median(v)),
                    "p95": float(np.percentile(v, 95)), "std": float(v.std())}
    out["fps_mean"] = 1000.0 / out["total"]["mean"]
    out["n_images_timed"] = len(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", choices=["coco", "kitti"], required=True)
    ap.add_argument("--model", required=True, help="model name from configs/experiment.yaml")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N images (debug)")
    ap.add_argument("--device", default=None, help="override device, e.g. 0 or cpu")
    ap.add_argument("--gt", default=None, help="override GT json path")
    ap.add_argument("--images", default=None, help="override image root")
    ap.add_argument("--out", default=None, help="override output folder")
    args = ap.parse_args()

    cfg = load_config(args.config)
    mcfg = get_model_cfg(cfg, args.model)
    inf = dict(cfg["inference"])
    if mcfg.get("imgsz"):  # optional per-model input size (ablation)
        inf["imgsz"] = mcfg["imgsz"]
    if args.device is not None:
        inf["device"] = args.device
    shared = cfg["shared_coco_cat_ids"]

    if args.dataset == "coco":
        gt_path = resolve(args.gt or cfg["paths"]["coco_ann"])
        img_root = resolve(args.images or cfg["paths"]["coco_images"])
    else:
        gt_path = resolve(args.gt) if args.gt else kitti_gt_path(cfg)
        img_root = resolve(args.images or cfg["paths"]["kitti_root"])

    out_dir = resolve(args.out) if args.out else resolve(cfg["paths"]["results"]) / args.dataset / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "command.txt").write_text("python " + " ".join(sys.argv) + "\n")

    coco_gt = COCO(str(gt_path))
    img_ids = sorted(coco_gt.getImgIds())
    if args.limit:
        img_ids = img_ids[: args.limit]
    imgs = coco_gt.loadImgs(img_ids)

    model, is_detr = load_model(mcfg["weights"], mcfg.get("rtdetr_eval_idx"))
    pred_kw = dict(imgsz=inf["imgsz"], conf=inf["conf"], iou=inf["iou"], max_det=inf["max_det"],
                   device=inf["device"], verbose=False)
    if inf.get("half"):  # only pass when enabled ('half' is deprecated in newer Ultralytics but still honored)
        pred_kw["half"] = True

    import torch

    use_cuda = torch.cuda.is_available() and str(inf["device"]) != "cpu"

    # ---- warm-up (CUDA kernels, cudnn autotune); not timed ----
    warm = [str(img_root / im["file_name"]) for im in imgs[: max(1, inf["warmup"])]]
    for i in range(inf["warmup"]):
        model.predict(warm[i % len(warm)], **pred_kw)
    # actual network input (H, W): YOLO letterboxes keeping the aspect ratio, Ultralytics RT-DETR
    # stretches to a square. Recorded because it matters a lot on wide KITTI images.
    try:
        import cv2

        input_hw = list(model.predictor.preprocess([cv2.imread(warm[0])]).shape[2:])
    except Exception:
        input_hw = None
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # ---- inference, batch=1, one image at a time ----
    detections, lat_rows = [], []
    keep_only_shared = args.dataset == "kitti"
    t_wall = time.perf_counter()
    for im in tqdm(imgs, desc=f"{args.model} on {args.dataset}"):
        res = model.predict(str(img_root / im["file_name"]), **pred_kw)[0]
        sp = res.speed  # ms per image, measured by Ultralytics (CUDA-synchronized)
        lat_rows.append({"image_id": im["id"], "preprocess": sp["preprocess"], "inference": sp["inference"],
                         "postprocess": sp["postprocess"],
                         "total": sp["preprocess"] + sp["inference"] + sp["postprocess"]})
        b = res.boxes
        if b is None or len(b) == 0:
            continue
        xyxy = b.xyxy.cpu().numpy()
        cls = b.cls.cpu().numpy().astype(int)
        scores = b.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, s in zip(xyxy, cls, scores):
            cat_id = COCO80_TO_COCO91[c]
            if keep_only_shared and cat_id not in shared:
                continue
            detections.append({"image_id": im["id"], "category_id": cat_id,
                               "bbox": [round(float(x1), 2), round(float(y1), 2),
                                        round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                               "score": round(float(s), 5)})
    wall_s = time.perf_counter() - t_wall
    peak_mem_mb = torch.cuda.max_memory_allocated() / 2**20 if use_cuda else None
    reserved_mb = torch.cuda.max_memory_reserved() / 2**20 if use_cuda else None

    dump_json(detections, out_dir / "detections.json", indent=None)
    with open(out_dir / "latency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lat_rows[0].keys()))
        w.writeheader()
        w.writerows(lat_rows)

    # ---- metrics ----
    metrics = {}
    if args.dataset == "coco":
        print("== COCO, all 80 classes ==")
        metrics["coco_all"] = evaluate(coco_gt, detections, img_ids=img_ids)
        print(f"== COCO, {len(shared)} classes shared with KITTI ==")
        metrics["coco_shared"] = evaluate(coco_gt, detections, cat_ids=shared, img_ids=img_ids)
    else:
        kcfg = cfg["kitti"]
        filt = kcfg.get("min_det_height", True)

        def dets_for(min_h):
            return [d for d in detections if d["bbox"][3] >= min_h] if filt else detections

        main_level = kcfg.get("difficulty", "moderate")
        metrics["slices"] = {}
        for name, (lvl, min_h) in KITTI_LEVELS.items():
            gt_l = kitti_gt_at(coco_gt.dataset, lvl)
            verbose = name == main_level
            if verbose:
                print(f"== KITTI ({main_level}), {len(shared)} shared classes ==")
            r = evaluate(gt_l, dets_for(min_h), cat_ids=shared, img_ids=img_ids, verbose=verbose)
            if verbose:
                metrics["kitti"] = r
            n_gt = len(gt_l.getAnnIds(imgIds=img_ids, catIds=shared, iscrowd=False))
            metrics["slices"][name] = {"overall": r["overall"], "num_images": len(img_ids), "num_gt": n_gt,
                                       "per_class": {k: v["AP"] for k, v in r["per_class"].items()}}

    record = {
        "model": args.model,
        "family": mcfg.get("family"),
        "weights": mcfg["weights"],
        "rtdetr_eval_idx": mcfg.get("rtdetr_eval_idx"),
        "nms": not is_detr,
        "dataset": args.dataset,
        "gt": str(gt_path),
        "num_images": len(img_ids),
        "inference_settings": inf,
        "network_input_hw": input_hw,
        "latency_ms": summarize_latency(lat_rows),
        "wall_time_s": wall_s,
        "gpu_peak_allocated_MB": peak_mem_mb,
        "gpu_peak_reserved_MB": reserved_mb,
        "complexity": model_complexity(model),
        "metrics": metrics,
        "environment": environment_info(),
        "command": "python " + " ".join(sys.argv),
    }
    dump_json(record, out_dir / "metrics.json")
    lt = record["latency_ms"]
    print(f"\nSaved {out_dir}/metrics.json")
    print(f"latency total mean {lt['total']['mean']:.2f} ms  ({lt['fps_mean']:.1f} FPS), "
          f"peak GPU mem {peak_mem_mb if peak_mem_mb is None else round(peak_mem_mb)} MB")


if __name__ == "__main__":
    main()
