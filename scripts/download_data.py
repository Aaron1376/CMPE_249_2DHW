"""Download and extract the datasets into data/ (works on Windows, macOS and Linux).

    COCO val2017       ~0.8 GB images + ~0.25 GB annotation zip (only instances_val2017.json is kept)
    KITTI 2D object    ~12 GB image zip + 5 MB labels (only the 7,481 labeled training images are extracted, ~6 GB)

Usage:
    python scripts/download_data.py                      # both datasets
    python scripts/download_data.py --only coco
    python scripts/download_data.py --only kitti
    python scripts/download_data.py --only kitti --kitti-image-zip D:/dl/data_object_image_2.zip \
                                                 --kitti-label-zip D:/dl/data_object_label_2.zip

KITTI: the script uses the same public mirror as torchvision's KITTI loader. If it is refused,
register (free) at https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d, download
"left color images of object data set" and "training labels of object data set" in your browser,
and pass the two zips with --kitti-image-zip / --kitti-label-zip.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
COCO_IMG_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
KITTI_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"   # mirror used by torchvision.datasets.Kitti
KITTI_IMAGES, KITTI_LABELS = "data_object_image_2.zip", "data_object_label_2.zip"
KITTI_N = 7481


def download(url: str, dst: Path) -> Path:
    if dst.exists():
        print(f"already downloaded: {dst.name}")
        return dst
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0)) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=dst.name) as bar:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dst)
    return dst


def get_coco(root: Path, keep_archives: bool) -> None:
    coco = root / "coco"
    coco.mkdir(parents=True, exist_ok=True)
    img_dir, ann = coco / "val2017", coco / "annotations" / "instances_val2017.json"

    if not (img_dir.exists() and len(list(img_dir.glob("*.jpg"))) == 5000):
        z = download(COCO_IMG_URL, coco / "val2017.zip")
        print("extracting val2017.zip")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(coco)
        if not keep_archives:
            z.unlink()
    if not ann.exists():
        z = download(COCO_ANN_URL, coco / "annotations_trainval2017.zip")
        with zipfile.ZipFile(z) as zf:
            zf.extract("annotations/instances_val2017.json", coco)
        if not keep_archives:
            z.unlink()
    n = len(list(img_dir.glob("*.jpg")))
    print(f"COCO: {n} images (expect 5000), annotations: {ann.exists()}")
    if n != 5000 or not ann.exists():
        raise SystemExit("COCO download incomplete; re-run this script.")


def _extract_prefix(zpath: Path, prefix: str, dest: Path) -> int:
    """Extract only members under `prefix` (e.g. 'training/image_2/'). Returns the number of files."""
    n = 0
    with zipfile.ZipFile(zpath) as zf:
        members = [m for m in zf.namelist() if m.startswith(prefix) and not m.endswith("/")]
        for m in tqdm(members, desc=f"extract {prefix}", unit="file"):
            zf.extract(m, dest)
            n += 1
    return n


def get_kitti(root: Path, image_zip: str | None, label_zip: str | None, keep_archives: bool) -> None:
    kitti = root / "kitti"
    img_dir, lab_dir = kitti / "training" / "image_2", kitti / "training" / "label_2"
    kitti.mkdir(parents=True, exist_ok=True)

    def have(d, ext):
        return d.is_dir() and len(list(d.glob(f"*{ext}"))) == KITTI_N

    for target, ext, name, user_zip in ((lab_dir, ".txt", KITTI_LABELS, label_zip),
                                        (img_dir, ".png", KITTI_IMAGES, image_zip)):
        if have(target, ext):
            continue
        if user_zip:
            z = Path(user_zip).expanduser()
            if not z.exists():
                raise SystemExit(f"not found: {z}")
        else:
            try:
                z = download(KITTI_URL + name, kitti / name)
            except Exception as e:
                print(f"\nDirect KITTI download of {name} failed ({e}).")
                print("1. Register (free) at https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d")
                print("2. Download 'left color images of object data set' (data_object_image_2.zip, 12 GB)")
                print("   and 'training labels of object data set' (data_object_label_2.zip, 5 MB).")
                print("3. Run: python scripts/download_data.py --only kitti --kitti-image-zip <path> --kitti-label-zip <path>")
                raise SystemExit(1)
        prefix = "training/" + target.name + "/"
        if _extract_prefix(z, prefix, kitti) == 0:
            raise SystemExit(f"{z.name} has no '{prefix}' folder. Is it the right KITTI zip?")
        if not keep_archives and z.parent == kitti:
            z.unlink()
    n_img, n_lab = len(list(img_dir.glob("*.png"))), len(list(lab_dir.glob("*.txt")))
    print(f"KITTI: {n_img} training images, {n_lab} label files (expect {KITTI_N} each)")
    if n_img != KITTI_N or n_lab != KITTI_N:
        raise SystemExit("KITTI looks incomplete; re-run this script.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["coco", "kitti"], default=None)
    ap.add_argument("--root", default=str(REPO / "data"))
    ap.add_argument("--kitti-image-zip", default=None, help="data_object_image_2.zip you downloaded yourself")
    ap.add_argument("--kitti-label-zip", default=None, help="data_object_label_2.zip you downloaded yourself")
    ap.add_argument("--keep-archives", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    free_gb = shutil.disk_usage(root.parent if not root.exists() else root).free / 2**30
    print(f"Data folder: {root}   free disk: {free_gb:.0f} GB (need ~20 GB while the KITTI zip is extracted)")
    if args.only in (None, "coco"):
        get_coco(root, args.keep_archives)
    if args.only in (None, "kitti"):
        get_kitti(root, args.kitti_image_zip, args.kitti_label_zip, args.keep_archives)
    print("Data ready. Next: python run_all_local.py")


if __name__ == "__main__":
    main()
