"""Download the COCO-pretrained Ultralytics checkpoints into weights/ (about 225 MB for the 3 main models).

Usage:  python scripts/download_weights.py            # models listed in configs/experiment.yaml
        python scripts/download_weights.py --all      # also RT-DETR-X (optional 4th model)
"""
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from common import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also download rtdetr-x.pt")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    names = sorted({m["weights"] for m in load_config(args.config)["models"]})
    if args.all and "rtdetr-x.pt" not in names:
        names.append("rtdetr-x.pt")

    wdir = REPO / "weights"
    wdir.mkdir(exist_ok=True)
    os.chdir(wdir)  # Ultralytics downloads into the current folder
    from ultralytics import RTDETR, YOLO

    for n in names:
        if (wdir / n).exists():
            print(f"have  {n}")
            continue
        (RTDETR if "rtdetr" in n else YOLO)(n)
        print(f"got   {n}  ({(wdir / n).stat().st_size / 2**20:.0f} MB)")
    print(f"Weights in {wdir}")


if __name__ == "__main__":
    main()
