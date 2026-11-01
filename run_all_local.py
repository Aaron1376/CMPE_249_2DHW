"""Run the whole Track A experiment on your own machine, one step after another.

Steps (in order):
    check     GPU check + record nvidia-smi / pip freeze in logs/
    test      unit tests (KITTI converter, difficulty levels, ignore regions)
    weights   download checkpoints into weights/ (skipped if present)
    prepare   KITTI labels -> COCO-format ground truth with difficulty levels (skipped if present)
    eval      every model x {coco, kitti}; finished runs are skipped, so you can stop and resume
    analyze   tables + figures (report/tables, report/figures) and failure analysis
    package   submission_artifacts.zip (code, configs, metrics, logs, tables, figures; no images/weights)

Usage (from the repo root, with your virtual environment active):
    python run_all_local.py                      # everything
    python run_all_local.py --smoke              # 6-image synthetic test of the full pipeline
    python run_all_local.py --steps eval analyze # only some steps
    python run_all_local.py --models YOLO11l --datasets coco --force   # redo one run
    python run_all_local.py --limit 50           # quick timing test (writes to results_limit50/)

Everything printed is also saved to logs/run_<timestamp>.log (your command log for the submission).
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
from common import kitti_gt_path, load_config, resolve  # noqa: E402

ALL_STEPS = ["check", "test", "weights", "prepare", "eval", "analyze", "package"]
LOG = None


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.write((msg + "\n").encode())
    LOG.flush()


def run(cmd: list[str]) -> None:
    """Run a command, stream its output to the screen and the log, stop on failure."""
    log(f"\n$ {' '.join(cmd)}")
    env = dict(os.environ, PYTHONPATH=str(REPO / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
               PYTHONUNBUFFERED="1")
    p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while chunk := p.stdout.read1(4096):
        sys.stdout.buffer.write(chunk)
        sys.stdout.flush()
        LOG.write(chunk)
    LOG.flush()
    if p.wait() != 0:
        log(f"\nFAILED (exit {p.returncode}): {' '.join(cmd)}\nSee the log above. Fix the problem and re-run; "
            "finished runs are skipped.")
        sys.exit(p.returncode)


def capture(cmd: list[str], out: Path) -> None:
    try:
        out.write_text(subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout)
    except Exception as e:
        out.write_text(f"could not run {cmd[0]}: {e}\n")


def main():
    global LOG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", choices=ALL_STEPS, default=None)
    ap.add_argument("--models", nargs="+", default=None, help="subset of model names from the config")
    ap.add_argument("--datasets", nargs="+", choices=["coco", "kitti"], default=["coco", "kitti"])
    ap.add_argument("--device", default=None, help="override config device: 0 (GPU) or cpu")
    ap.add_argument("--force", action="store_true", help="redo runs/outputs that already exist")
    ap.add_argument("--limit", type=int, default=None, help="only the first N images per dataset (timing test)")
    ap.add_argument("--smoke", action="store_true", help="synthetic 6-image test of the whole pipeline")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    py = sys.executable
    cfg_path = args.config or ("tests/smoke_config.yaml" if args.smoke else "configs/experiment.yaml")
    cfg = load_config(REPO / cfg_path)
    models = args.models or [m["name"] for m in cfg["models"]]
    steps = args.steps or (["test", "prepare", "eval", "analyze"] if args.smoke else ALL_STEPS)
    report_dir = "tests/smoke_report" if args.smoke else "report"
    results = resolve(cfg["paths"]["results"])
    if args.limit:
        results = results.parent / f"{results.name}_limit{args.limit}"

    (REPO / "logs").mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    LOG = open(REPO / "logs" / f"run_{stamp}.log", "wb")
    log(f"== Track A local run {stamp}  config={cfg_path}  steps={steps}  models={models}  datasets={args.datasets}")

    if "check" in steps:
        run([py, "scripts/check_gpu.py"])
        capture(["nvidia-smi"], REPO / "logs" / "nvidia-smi.txt")
        capture([py, "-m", "pip", "freeze"], REPO / "logs" / "pip-freeze.txt")
        log("saved logs/nvidia-smi.txt and logs/pip-freeze.txt (for the reproducibility table)")

    if "test" in steps:
        run([py, "-m", "pytest", "-q", "tests/test_prepare_kitti.py"])

    if "weights" in steps:
        run([py, "scripts/download_weights.py", "--config", cfg_path])

    if args.smoke and ({"prepare", "eval"} & set(steps)) and (
            args.force or not (REPO / "tests/smoke_data/coco_gt.json").exists()):
        run([py, "tests/make_smoke_data.py"])

    if "prepare" in steps and "kitti" in args.datasets:
        gt = kitti_gt_path(cfg)
        if gt.exists() and not args.force:
            log(f"\nprepare: {gt.name} exists, skipping (use --force to rebuild)")
        else:
            run([py, "src/prepare_kitti.py", "--config", cfg_path])

    if "eval" in steps:
        for ds in args.datasets:
            for m in models:
                out = results / ds / m
                if (out / "metrics.json").exists() and not args.force:
                    log(f"\neval: {ds}/{m} done already, skipping")
                    continue
                cmd = [py, "src/run_eval.py", "--config", cfg_path, "--dataset", ds, "--model", m]
                if args.device is not None:
                    cmd += ["--device", str(args.device)]
                if args.limit:
                    cmd += ["--limit", str(args.limit), "--out", str(out)]
                run(cmd)

    if "analyze" in steps:
        if args.limit:
            log("\nanalyze: skipped for --limit runs (timing test only). Per-run numbers are in "
                f"{results}/<dataset>/<model>/metrics.json")
        else:
            run([py, "src/aggregate.py", "--config", cfg_path, "--report-dir", report_dir])
            for ds in args.datasets:
                n = "2" if args.smoke else ("6" if ds == "kitti" else "3")
                run([py, "src/failure_analysis.py", "--config", cfg_path, "--dataset", ds,
                     "--report-dir", report_dir, "--num-examples", n])

    if "package" in steps and not args.smoke:
        zpath = REPO / "submission_artifacts.zip"
        patterns = ["README.md", "LAPTOP_RUNBOOK.md", "requirements.txt", "run_all_local.py",
                    "configs/*", "src/*.py", "scripts/*.py", "tests/*.py", "tests/*.yaml",
                    "results/*.csv", "results/*/*/metrics.json", "results/*/*/command.txt",
                    "results/*/*/latency.csv", "report/*.tex", "report/*.bib",
                    "report/tables/*", "report/figures/*", "report/figures/failures/*", "logs/*"]
        n = 0
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for pat in patterns:
                for f in sorted(REPO.glob(pat)):
                    if f.is_file():
                        z.write(f, f.relative_to(REPO))
                        n += 1
        log(f"\npackage: wrote {zpath.name} ({n} files, {zpath.stat().st_size / 2**20:.1f} MB)")

    log(f"\n== done. Full log: logs/run_{stamp}.log")


if __name__ == "__main__":
    main()
