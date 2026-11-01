# 2DHW: YOLO11 vs RT-DETR, COCO to KITTI

Cross-domain evaluation of three COCO-pretrained real-time 2D object detectors, with no retraining:
**YOLO11-L**, **YOLO11-X** (convolutional, one-stage, NMS) and **RT-DETR-L** (transformer, NMS-free).
Each is evaluated on COCO val2017 and on the KITTI 2D object benchmark, then compared on
(1) architecture under domain shift and (2) the speed-accuracy tradeoff.
Everything runs on one local NVIDIA GPU (tested on an RTX 4050 Laptop GPU, 6 GB).

## Results

Measured on an RTX 4050 Laptop GPU (50 W power limit), PyTorch FP32, batch size 1.
"COCO (3 cls)" is COCO scored on the three classes shared with KITTI (person, car, truck),
the fair baseline for the domain-shift drop. KITTI is scored at moderate difficulty with COCO-style AP@[.50:.95].

| Model | COCO AP (80 cls) | COCO AP (3 cls) | KITTI AP | AP drop | COCO latency (ms) | KITTI latency (ms) |
|---|---|---|---|---|---|---|
| YOLO11-L | 52.3 | 53.2 | 38.0 | 15.2 | 23.0 | 21.5 |
| YOLO11-X | 53.6 | 54.1 | 38.1 | 16.0 | 34.1 | 21.8 |
| RT-DETR-L | 52.8 | 52.6 | 39.3 | 13.3 | 32.7 | 33.1 |

- RT-DETR-L loses the least AP under the domain shift, mostly through small objects (KITTI AP_S 31.6 vs 21.8 for YOLO11-L).
  Part of this is a preprocessing effect: Ultralytics stretches RT-DETR inputs to 640x640 but letterboxes YOLO11
  (a KITTI frame becomes 224x640), so objects are about 3x taller for RT-DETR.
- At batch size 1, RT-DETR-L's NMS-free design does not pay off in end-to-end latency (about 1.5x YOLO11-L on KITTI).
- COCO AP is 0.2 to 1.1 points below the published values (53.4 / 54.7 / 53.0 for YOLO11-L / YOLO11-X / RT-DETR-L).
- KITTI AP here is not comparable to the KITTI leaderboard (different metric, training split only).

Full numbers: [results/summary.csv](results/summary.csv), [results/domain_shift.csv](results/domain_shift.csv),
[results/kitti_difficulty.csv](results/kitti_difficulty.csv), [results/per_class.csv](results/per_class.csv).

## Setup

Requirements: an NVIDIA GPU with a current driver, and 64-bit Python 3.10 to 3.12 (the results here were produced
with Python 3.14.0, PyTorch 2.14.1+cu130, Ultralytics 8.4.196 and pycocotools 2.0.11). About 25 GB of free disk
is needed during the KITTI download (about 8 GB remains afterwards).

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1      Linux: source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch with CUDA, using the command from https://pytorch.org/get-started/locally/
(on Windows a plain `pip install torch` gives a CPU-only build; on Linux the PyPI wheels include CUDA). Then:

```bash
pip install -r requirements.txt
python scripts/check_gpu.py            # must end with "GPU OK"
python scripts/download_weights.py     # yolo11l.pt, yolo11x.pt, rtdetr-l.pt into weights/
python scripts/download_data.py        # COCO val2017 + KITTI 2D object into data/
```

If the direct KITTI download is refused, register for a free account at https://www.cvlibs.net/datasets/kitti/,
download "left color images of object data set" and "training labels of object data set", and run:

```bash
python scripts/download_data.py --only kitti --kitti-image-zip <data_object_image_2.zip> --kitti-label-zip <data_object_label_2.zip>
```

## Run

```bash
python run_all_local.py                # check -> test -> weights -> prepare -> eval -> analyze -> package
python run_all_local.py --smoke        # 6 synthetic images through the whole pipeline (quick sanity check)
python run_all_local.py --limit 50 --steps prepare eval     # timing test on 50 images per dataset
```

The full run takes roughly 30 to 60 minutes after the downloads. It is safe to re-run: finished runs are skipped.
To redo one run:

```bash
python run_all_local.py --steps eval --models YOLO11x --datasets coco --force
python run_all_local.py --steps analyze package
```

For fair latency numbers, plug in the charger, set the laptop to maximum performance, close other GPU-heavy
applications and run all models in one sitting.

## Repository layout

```
run_all_local.py              one command: check -> test -> weights -> prepare -> eval -> analyze -> package
configs/experiment.yaml       every setting (models, paths, thresholds, KITTI difficulty)
scripts/check_gpu.py          confirms PyTorch sees the GPU, prints fixes if not
scripts/download_weights.py   checkpoints into weights/
scripts/download_data.py      COCO val2017 + KITTI training images/labels into data/
src/prepare_kitti.py          KITTI labels -> COCO-format GT (3 shared classes, difficulty levels, ignore regions)
src/run_eval.py               one (model, dataset) run: inference, latency, GPU memory, COCO metrics
src/coco_metrics.py           pycocotools wrapper (overall + per-class AP/AR)
src/aggregate.py              all runs -> LaTeX/Markdown tables + figures
src/failure_analysis.py       error-type breakdown + qualitative GT-vs-prediction panels
src/common.py                 config, paths, COCO 80->91 id map, environment capture
tests/                        unit tests (KITTI parsing, difficulty levels, ignore regions) + synthetic smoke data
results/                      summary CSVs and per-run metrics.json (exact command, settings, environment)
logs/                         command logs, nvidia-smi.txt, pip-freeze.txt
```

A single run can be launched directly (Windows PowerShell: `$env:PYTHONPATH="src"`, Linux: `export PYTHONPATH=src`):

```bash
python src/run_eval.py --dataset coco --model YOLO11l
python src/run_eval.py --dataset kitti --model RT-DETR-L
```

Each run writes `results/<dataset>/<model>/{metrics.json, detections.json, latency.csv, command.txt}`.

## Experiment design

| | |
|---|---|
| Models | YOLO11l, YOLO11x (CNN one-stage, NMS); RT-DETR-L (transformer, NMS-free). Optional in the config: RT-DETR-X, RT-DETR-L with 4 decoder layers |
| Datasets | COCO val2017 (5,000 imgs, 80 cls); KITTI 2D object training split (7,481 imgs) |
| Taxonomy | Pedestrian -> person, Car -> car, Truck -> truck; Van, Cyclist, Person_sitting, DontCare -> ignore regions (see `src/prepare_kitti.py`) |
| Settings | imgsz 640, batch 1, conf 0.001, NMS IoU 0.7 (YOLO only), max_det 300, FP32, 20 warm-up images |
| Metrics | AP, AP50, AP75, AP S/M/L, AR, per-class AP; latency (pre/infer/post), FPS, peak GPU memory, params, GFLOPs |

COCO is scored twice: all 80 classes (to compare with published numbers) and the 3 shared classes.
KITTI is scored at easy, moderate and hard.

## Notes

- KITTI classes without a clean COCO match (Van, Cyclist, Person_sitting, DontCare) and boxes harder than the
  evaluated level become `iscrowd=1` ignore regions, mirroring KITTI's own evaluation. Detections shorter than the
  level's minimum height (25 px moderate/hard, 40 px easy) are dropped, as on the KITTI server. Tram and Misc are dropped.
- Latency is PyTorch eager mode on a laptop GPU (not TensorRT on a T4), so absolute numbers are slower than published ones.
  Only the comparison between models, measured in one sitting under the same power settings, is meaningful.
- `data/` and `weights/` are not in the repository; the scripts above download them.
- Ultralytics code and weights are AGPL-3.0.
