"""Collect every results/<dataset>/<model>/metrics.json into report tables and figures.

Outputs:
    results/summary.csv                       one row per (model, dataset scope)
    report/tables/*.tex                       booktabs tables, \\input{} from report.tex
    report/tables/*.md                        same tables in markdown (for README / slides)
    report/figures/*.png, *.pdf               speed-accuracy, memory-accuracy, latency breakdown,
                                              per-class AP, domain-shift drop
Usage:
    python src/aggregate.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from common import load_config, resolve  # noqa: E402

# Validated categorical palette (fixed order, one hue per model; never cycled).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SCOPE_LABEL = {"coco_all": "COCO (80 cls)", "coco_shared": "COCO (3 shared cls)", "kitti": "KITTI moderate (3 cls)"}
SCOPE_MARKER = {"coco_all": "o", "coco_shared": "s", "kitti": "^"}
SHARED_CLASSES = ["person", "car", "truck"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "legend.frameon": False,
})


def load_records(results_dir: Path) -> list[dict]:
    recs = []
    for p in sorted(results_dir.glob("*/*/metrics.json")):
        with open(p) as f:
            recs.append(json.load(f))
    return recs


def build_summary(recs: list[dict], model_order: list[str]) -> pd.DataFrame:
    rows = []
    for r in recs:
        lat = r["latency_ms"]
        for scope, m in r["metrics"].items():
            if scope == "slices":
                continue
            o = m["overall"]
            rows.append({
                "model": r["model"], "family": r["family"], "nms": r["nms"], "dataset": r["dataset"],
                "scope": scope, "images": m["num_images"],
                # COCOeval reports -1 when a size bucket has no GT; show it as missing, not -100.
                **{k: (100 * o[k] if o[k] >= 0 else float("nan"))
                   for k in ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR100")},
                "pre_ms": lat["preprocess"]["mean"], "inf_ms": lat["inference"]["mean"],
                "post_ms": lat["postprocess"]["mean"], "total_ms": lat["total"]["mean"],
                "p95_ms": lat["total"]["p95"], "fps": lat["fps_mean"],
                "gpu_mem_MB": r.get("gpu_peak_allocated_MB"),
                "params_M": r.get("complexity", {}).get("params_M"),
                "GFLOPs": r.get("complexity", {}).get("GFLOPs_640"),
                "gpu": r.get("environment", {}).get("gpu", "cpu"),
                "input_hw": "x".join(map(str, r["network_input_hw"])) if r.get("network_input_hw") else "",
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["model"] = pd.Categorical(df["model"], categories=[m for m in model_order if m in set(df["model"])],
                                 ordered=True)
    return df.sort_values(["scope", "model"]).reset_index(drop=True)


# ---------------------------------------------------------------- tables
def fmt(v, nd=1):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return f"{v:.{nd}f}"


def write_table(df: pd.DataFrame, cols: list[tuple[str, str, int]], name: str, caption: str,
                label: str, out_dir: Path):
    """cols: (column, header, decimals). Writes .tex (booktabs) and .md."""
    out_dir.mkdir(parents=True, exist_ok=True)
    heads = [h for _, h, _ in cols]
    body = []
    for _, row in df.iterrows():
        body.append([str(row[c]) if nd is None else fmt(row[c], nd) for c, _, nd in cols])

    # Wide tables span both columns of a two-column report; everything is scaled to fit.
    env, width = ("table*", "\\textwidth") if len(cols) > 6 else ("table", "\\columnwidth")
    tex = [f"\\begin{{{env}}}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           f"\\resizebox{{{width}}}{{!}}{{%",
           "\\begin{tabular}{" + "l" * 1 + "r" * (len(cols) - 1) + "}", "\\toprule",
           " & ".join(heads) + " \\\\", "\\midrule"]
    tex += [" & ".join(x.replace("_", "\\_").replace("%", "\\%") for x in b) + " \\\\" for b in body]
    tex += ["\\bottomrule", "\\end{tabular}}", f"\\end{{{env}}}"]
    (out_dir / f"{name}.tex").write_text("\n".join(tex) + "\n")

    md = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    md += ["| " + " | ".join(b) + " |" for b in body]
    (out_dir / f"{name}.md").write_text(f"**{caption}**\n\n" + "\n".join(md) + "\n")


def per_class_frame(recs, model_order):
    rows = []
    for r in recs:
        for scope in ("coco_shared", "kitti"):
            if scope in r["metrics"]:
                for cls, v in r["metrics"][scope]["per_class"].items():
                    rows.append({"model": r["model"], "scope": scope, "class": cls,
                                 "AP": 100 * v["AP"] if v["AP"] >= 0 else float("nan"),
                                 "AP50": 100 * v["AP50"] if v["AP50"] >= 0 else float("nan"),
                                 "num_gt": v["num_gt"]})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["model"] = pd.Categorical(df["model"], categories=[m for m in model_order if m in set(df["model"])],
                                     ordered=True)
    return df


# ---------------------------------------------------------------- figures
def color_map(models):
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}


def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_tradeoff(df, models, colors, xcol, xlabel, name, out_dir):
    scopes = [s for s in ("coco_all", "kitti") if s in set(df["scope"])]
    if not scopes or df[xcol].isna().all():
        return
    fig, axes = plt.subplots(1, len(scopes), figsize=(5.2 * len(scopes), 3.8), squeeze=False)
    for ax, scope in zip(axes[0], scopes):
        d = df[df["scope"] == scope]
        for fam_nms, ls in ((True, "-"), (False, "--")):  # connect sizes within a family
            dd = d[d["nms"] == fam_nms].sort_values(xcol)
            if len(dd) > 1:
                ax.plot(dd[xcol], dd["AP"], ls, color=INK2, lw=1, alpha=0.6, zorder=1)
        # stagger labels above/below in x order so neighbouring points never overlap
        for k, (_, row) in enumerate(d.sort_values(xcol).iterrows()):
            ax.scatter(row[xcol], row["AP"], s=70, color=colors[row["model"]], marker=SCOPE_MARKER[scope],
                       edgecolor=SURFACE, linewidth=2, zorder=3)
            above = k % 2 == 0
            ax.annotate(f"{row['model']}\n{row['AP']:.1f} AP", (row[xcol], row["AP"]),
                        textcoords="offset points", xytext=(0, 9 if above else -12), fontsize=8, color=INK,
                        ha="center", va="bottom" if above else "top")
        ax.set_title(SCOPE_LABEL[scope], color=INK, loc="left")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("mAP@[.50:.95] (%)")
        ax.margins(x=0.25, y=0.2)
    fig.suptitle("Solid line: YOLO11 sizes (NMS).  Dashed: RT-DETR (NMS-free).  Up-left is better.",
                 fontsize=8.5, color=INK2, y=0.995)
    fig.tight_layout()
    save(fig, out_dir, name)


def fig_latency_breakdown(df, models, out_dir):
    d = df[df["scope"].isin(["coco_all", "kitti"])]
    if d.empty:
        return
    stages = [("pre_ms", "preprocess", "#9ec5f4"), ("inf_ms", "inference", "#2a78d6"),
              ("post_ms", "postprocess (incl. NMS for YOLO)", "#104281")]
    scopes = [s for s in ("coco_all", "kitti") if s in set(d["scope"])]
    fig, axes = plt.subplots(1, len(scopes), figsize=(5.2 * len(scopes), 3.4), squeeze=False, sharey=True)
    for ax, scope in zip(axes[0], scopes):
        dd = d[d["scope"] == scope].sort_values("model")
        y = range(len(dd))
        left = [0.0] * len(dd)
        for col, lab, c in stages:
            vals = dd[col].tolist()
            ax.barh(list(y), vals, left=left, color=c, label=lab, height=0.55, edgecolor=SURFACE, linewidth=2)
            left = [a + b for a, b in zip(left, vals)]
        for yi, tot in zip(y, left):
            ax.text(tot, yi, f"  {tot:.1f} ms", va="center", fontsize=8, color=INK)
        ax.set_yticks(list(y))
        ax.set_yticklabels(dd["model"].astype(str))
        ax.invert_yaxis()
        ax.set_xlabel("mean latency per image, batch=1 (ms)")
        ax.set_title(SCOPE_LABEL[scope], loc="left", color=INK)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.2)
    axes[0][0].legend(loc="lower center", bbox_to_anchor=(0.5 * len(scopes), -0.42), ncol=3, fontsize=8)
    fig.tight_layout()
    save(fig, out_dir, "latency_breakdown")


def fig_per_class(pc, models, colors, out_dir):
    if pc.empty:
        return
    scopes = [s for s in ("coco_shared", "kitti") if s in set(pc["scope"])]
    classes = SHARED_CLASSES
    fig, axes = plt.subplots(1, len(scopes), figsize=(5.6 * len(scopes), 3.6), squeeze=False, sharey=True)
    present = [m for m in models if m in set(pc["model"])]
    w = 0.8 / max(1, len(present))
    for ax, scope in zip(axes[0], scopes):
        d = pc[pc["scope"] == scope]
        for i, m in enumerate(present):
            vals = [d[(d["model"] == m) & (d["class"] == c)]["AP"].mean() for c in classes]
            xs = [j + (i - (len(present) - 1) / 2) * w for j in range(len(classes))]
            ax.bar(xs, vals, width=w * 0.92, color=colors[m], label=m, edgecolor=SURFACE, linewidth=1)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_ylabel("AP@[.50:.95] (%)")
        ax.set_title(SCOPE_LABEL[scope], loc="left", color=INK)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="lower center", ncol=len(present), bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, out_dir, "per_class_ap")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--report-dir", default="report")
    args = ap.parse_args()

    cfg = load_config(args.config)
    results_dir = resolve(cfg["paths"]["results"])
    report_dir = resolve(args.report_dir)
    tdir, fdir = report_dir / "tables", report_dir / "figures"
    models = [m["name"] for m in cfg["models"]]

    recs = load_records(results_dir)
    if not recs:
        raise SystemExit(f"No metrics.json found under {results_dir}. Run src/run_eval.py first.")
    df = build_summary(recs, models)
    df.to_csv(results_dir / "summary.csv", index=False)
    present = [m for m in models if m in set(df["model"].astype(str))]
    colors = color_map(present)

    acc_cols = [("model", "Model", None), ("AP", "AP", 1), ("AP50", "AP$_{50}$", 1), ("AP75", "AP$_{75}$", 1),
                ("AP_small", "AP$_S$", 1), ("AP_medium", "AP$_M$", 1), ("AP_large", "AP$_L$", 1),
                ("AR100", "AR$_{100}$", 1)]
    for scope, cap in (("coco_all", "COCO val2017, all 80 classes."),
                       ("coco_shared", "COCO val2017 restricted to the 3 classes shared with KITTI (person, car, truck)."),
                       ("kitti", "KITTI 2D object (7,481 training images, moderate difficulty), 3 shared classes, "
                                 "no retraining.")):
        d = df[df["scope"] == scope]
        if not d.empty:
            write_table(d, acc_cols, f"acc_{scope}", cap, f"tab:acc_{scope}", tdir)

    # speed / memory tables. KITTI images are wide (1242x375): YOLO letterboxes them to ~224x640,
    # Ultralytics RT-DETR stretches them to 640x640, so KITTI speed differs from COCO speed.
    sp_cols = [("model", "Model", None), ("input_hw", "Input HxW", None), ("params_M", "Params (M)", 1),
               ("GFLOPs", "GFLOPs@640$^2$", 1),
               ("pre_ms", "Pre (ms)", 2), ("inf_ms", "Infer (ms)", 2), ("post_ms", "Post (ms)", 2),
               ("total_ms", "Total (ms)", 2), ("p95_ms", "p95 (ms)", 2), ("fps", "FPS", 1),
               ("gpu_mem_MB", "Peak mem (MB)", 0)]
    for scope in ("coco_all", "kitti"):
        d = df[df["scope"] == scope]
        if not d.empty:
            gpu = d["gpu"].iloc[0]
            write_table(d, sp_cols, f"speed_{scope}",
                        f"Speed and memory on {SCOPE_LABEL[scope]}, batch=1, imgsz=640, {gpu}.",
                        f"tab:speed_{scope}", tdir)

    # domain shift: COCO(shared) -> KITTI
    a = df[df["scope"] == "coco_shared"].set_index("model")
    b = df[df["scope"] == "kitti"].set_index("model")
    common_m = [m for m in present if m in a.index and m in b.index]
    if common_m:
        rows = []
        for m in common_m:
            r = {"model": m}
            for k in ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"):
                r[f"{k}_coco"], r[f"{k}_kitti"] = a.loc[m, k], b.loc[m, k]
                r[f"{k}_drop"] = a.loc[m, k] - b.loc[m, k]
            rows.append(r)
        shift = pd.DataFrame(rows)
        shift.to_csv(results_dir / "domain_shift.csv", index=False)
        write_table(shift, [("model", "Model", None), ("AP_coco", "AP COCO", 1), ("AP_kitti", "AP KITTI", 1),
                            ("AP_drop", "$\\Delta$AP", 1), ("AP50_drop", "$\\Delta$AP$_{50}$", 1),
                            ("AP75_drop", "$\\Delta$AP$_{75}$", 1), ("AP_small_drop", "$\\Delta$AP$_S$", 1),
                            ("AP_medium_drop", "$\\Delta$AP$_M$", 1), ("AP_large_drop", "$\\Delta$AP$_L$", 1)],
                    "domain_shift", "Drop from COCO (3 shared classes) to KITTI (moderate). Positive = worse on KITTI.",
                    "tab:shift", tdir)

    # per class
    pc = per_class_frame(recs, models)
    if not pc.empty:
        pc.to_csv(results_dir / "per_class.csv", index=False)
        wide = pc.pivot_table(index=["class"], columns=["scope", "model"], values="AP", observed=False)
        rows = []
        for cls in SHARED_CLASSES:
            if cls not in wide.index:
                continue
            r = {"class": cls}
            for m in present:
                for scope, tag in (("coco_shared", "C"), ("kitti", "K")):
                    r[f"{m}_{tag}"] = wide.loc[cls].get((scope, m), float("nan"))
            gt_n = pc[(pc["class"] == cls) & (pc["scope"] == "kitti")]["num_gt"]
            r["kitti_gt"] = int(gt_n.iloc[0]) if len(gt_n) else float("nan")
            rows.append(r)
        pct = pd.DataFrame(rows)
        cols = [("class", "Class", None)]
        for m in present:
            cols += [(f"{m}_C", f"{m} C", 1), (f"{m}_K", f"{m} K", 1)]
        cols += [("kitti_gt", "KITTI GT", 0)]
        write_table(pct, cols, "per_class", "Per-class AP@[.50:.95]. C = COCO, K = KITTI (moderate).",
                    "tab:per_class", tdir)

    # KITTI difficulty levels (easy / moderate / hard), used in the failure analysis
    srows = []
    for r in recs:
        for name, sl in r["metrics"].get("slices", {}).items():
            o = sl["overall"]
            row = {"model": r["model"], "level": name, "gt": sl.get("num_gt"),
                   **{k: (100 * o[k] if o[k] >= 0 else float("nan")) for k in ("AP", "AP50", "AP75")}}
            for cls, ap_c in sl.get("per_class", {}).items():
                row[f"AP_{cls}"] = 100 * ap_c if ap_c is not None and ap_c >= 0 else float("nan")
            srows.append(row)
    if srows:
        sl = pd.DataFrame(srows)
        order = {"easy": 0, "moderate": 1, "hard": 2}
        sl["_o"] = sl["level"].map(order)
        sl["model"] = pd.Categorical(sl["model"], categories=[m for m in models if m in set(sl["model"])],
                                     ordered=True)
        sl = sl.sort_values(["model", "_o"]).drop(columns="_o")
        sl.to_csv(results_dir / "kitti_difficulty.csv", index=False)
        cols = [("model", "Model", None), ("level", "Level", None), ("gt", "GT", 0), ("AP", "AP", 1),
                ("AP50", "AP$_{50}$", 1), ("AP75", "AP$_{75}$", 1)]
        cols += [(f"AP_{c}", f"AP {c}", 1) for c in SHARED_CLASSES if f"AP_{c}" in sl.columns]
        write_table(sl, cols, "slices", "KITTI results by official difficulty level (failure analysis). Harder "
                    "levels add more occluded, truncated and smaller (25-39 px) objects.", "tab:slices", tdir)

    # figures
    fig_tradeoff(df, present, colors, "total_ms", "mean end-to-end latency per image (ms)", "speed_accuracy", fdir)
    fig_tradeoff(df, present, colors, "gpu_mem_MB", "peak GPU memory allocated (MB)", "memory_accuracy", fdir)
    fig_tradeoff(df, present, colors, "params_M", "parameters (M)", "params_accuracy", fdir)
    fig_latency_breakdown(df, present, fdir)
    fig_per_class(pc, present, colors, fdir)

    print(df[["model", "scope", "AP", "AP50", "AP75", "total_ms", "fps", "gpu_mem_MB"]].to_string(index=False))
    print(f"\nTables -> {tdir}\nFigures -> {fdir}\nCSV -> {results_dir}/summary.csv")


if __name__ == "__main__":
    main()
