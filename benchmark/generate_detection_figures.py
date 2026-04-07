#!/usr/bin/env python
"""
generate_detection_figures.py — Visual performance report for the people-detection benchmark.

Figures generated
-----------------
  1. detection_metrics.png    – mAP@50, mAP@75, mAP@50:95 grouped bar chart
  2. detection_speed.png      – Latency (ms) and FPS bar chart
  3. detection_samples.png    – Side-by-side detection visualisations on sample images
  4. detection_summary.png    – Combined 2×2 dashboard (metrics + speed + 2 sample images)

Usage
-----
    cd benchmark/
    python generate_detection_figures.py
    python generate_detection_figures.py --results results/people_detection_XYZ.json
    python generate_detection_figures.py --out-dir my_figures/
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image as PILImage

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "results" / "figures"

# ── Style ─────────────────────────────────────────────────────────────────────
BG         = "#0d0d0d"
PANEL_BG   = "#111111"
TEXT       = "#cccccc"
GRID_COLOR = "#222222"
ACCENT     = "#58a6ff"

MODEL_COLORS = {
    "yolo11n":       "#58a6ff",
    "yolo11s":       "#7ee787",
    "mobilenet_ssd": "#f78166",
}
MODEL_LABELS = {
    "yolo11n":       "YOLOv11n (nano)",
    "yolo11s":       "YOLOv11s (small)",
    "mobilenet_ssd": "MobileNet SSD",
}

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    GRID_COLOR,
    "axes.labelcolor":   TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "text.color":        TEXT,
    "grid.color":        GRID_COLOR,
    "legend.facecolor":  PANEL_BG,
    "legend.edgecolor":  GRID_COLOR,
    "font.family":       "monospace",
})


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def load_results(path: str | None) -> dict:
    if path:
        return json.loads(Path(path).read_text())
    files = sorted(glob.glob(str(RESULTS_DIR / "people_detection_*.json")))
    if not files:
        raise FileNotFoundError(
            "No results found. Run run_benchmark_people_detection.py first."
        )
    return json.loads(Path(files[-1]).read_text())


def _style_ax(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=TEXT, fontsize=11, pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT, fontsize=9)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — Accuracy metrics
# ──────────────────────────────────────────────────────────────────────────────

def fig_metrics(data: dict, out: Path):
    models   = list(data["models"].keys())
    labels   = [MODEL_LABELS.get(m, m) for m in models]
    map50    = [data["models"][m]["metrics"]["map50"]  for m in models]
    map75    = [data["models"][m]["metrics"]["map75"]  for m in models]
    map_avg  = [data["models"][m]["metrics"]["map_avg"] for m in models]

    x    = np.arange(len(models))
    w    = 0.26
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    _style_ax(ax, title="Person Detection Accuracy — COCO128",
              ylabel="mAP (person class)")

    bars50  = ax.bar(x - w,    map50,   w, label="mAP@50",     color="#58a6ff", alpha=0.9)
    bars75  = ax.bar(x,        map75,   w, label="mAP@75",     color="#7ee787", alpha=0.9)
    barsavg = ax.bar(x + w,    map_avg, w, label="mAP@50:95",  color="#f78166", alpha=0.9)

    for bars in (bars50, bars75, barsavg):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8, color=TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, min(1.0, max(map50 + [0.1]) * 1.35))
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — Speed comparison
# ──────────────────────────────────────────────────────────────────────────────

def fig_speed(data: dict, out: Path):
    models  = list(data["models"].keys())
    labels  = [MODEL_LABELS.get(m, m) for m in models]
    latency = [data["models"][m]["metrics"]["mean_latency_ms"] for m in models]
    fps     = [data["models"][m]["metrics"]["fps"]             for m in models]
    colors  = [MODEL_COLORS.get(m, ACCENT) for m in models]

    fig, (ax_lat, ax_fps) = plt.subplots(1, 2, figsize=(10, 4.5), facecolor=BG)

    # Latency
    _style_ax(ax_lat, title="Inference Latency", ylabel="Latency (ms)")
    bars = ax_lat.bar(labels, latency, color=colors, alpha=0.9, width=0.55)
    for bar, val in zip(bars, latency):
        ax_lat.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f} ms", ha="center", va="bottom", fontsize=9, color=TEXT)
    ax_lat.set_ylim(0, max(latency) * 1.35)

    # FPS
    _style_ax(ax_fps, title="Throughput (FPS)", ylabel="Frames per Second")
    bars = ax_fps.bar(labels, fps, color=colors, alpha=0.9, width=0.55)
    for bar, val in zip(bars, fps):
        ax_fps.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9, color=TEXT)
    ax_fps.set_ylim(0, max(fps) * 1.35)

    device = data.get("device", "GPU")
    fig.suptitle(f"Speed Benchmark ({device.upper()})", color=TEXT, fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 — Sample detection visualisations
# ──────────────────────────────────────────────────────────────────────────────

def _draw_boxes(ax, img: PILImage.Image, detections: list[dict], color: str, title: str):
    """Draw bounding boxes on a matplotlib axes."""
    import matplotlib.patches as patches
    ax.imshow(np.array(img))
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        rect = patches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            boxstyle="square,pad=0",
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 4, f"{det['confidence']:.2f}",
                color=color, fontsize=7, fontweight="bold",
                bbox=dict(facecolor=BG, alpha=0.6, pad=1, edgecolor="none"))
    ax.set_title(title, color=TEXT, fontsize=8, pad=4)
    ax.axis("off")


def run_inference_for_viz(
    models_cfg: dict,
    image_paths: list[str],
    conf: float,
    device: str,
) -> dict[str, list[dict]]:
    """
    Run fresh inference on a small set of images for visualisation.
    Returns { model_key: [{"bbox":[x1,y1,x2,y2], "confidence":float}, ...] per image }.
    Actually returns { model_key: list of per-image detection lists }.
    """
    from models.yolov11 import YOLOv11Model
    from models.mobilenet_ssd import MobileNetSSDModel

    all_preds: dict[str, list[list[dict]]] = {}

    for mkey, mcfg in models_cfg.items():
        model = mcfg["model"]
        model.load()
        preds_per_image = []
        for img_path in image_paths:
            result = model.detect(img_path, conf_threshold=conf)
            preds = [{"bbox": d.bbox, "confidence": d.confidence}
                     for d in result.detections] if not result.error else []
            preds_per_image.append(preds)
        model.unload()
        all_preds[mkey] = preds_per_image

    return all_preds


def fig_samples(
    data: dict,
    out: Path,
    n_samples: int = 4,
    conf: float = 0.25,
):
    """
    Pick n_samples images that had person detections (from benchmark results)
    and re-run inference for visualisation.
    """
    # Pick diverse sample images from benchmark results
    first_model = next(iter(data["models"]))
    per_image = data["models"][first_model].get("per_image", [])

    # Filter to images with ground-truth persons and successful predictions
    candidates = [
        item for item in per_image
        if item.get("n_gt_persons", 0) > 0 and Path(item["image"]).exists()
    ][:n_samples * 3]

    sample_paths = [c["image"] for c in candidates[:n_samples]]
    if not sample_paths:
        print("  Warning: no suitable sample images found for visualisation")
        return

    device = data.get("device", "cpu")
    from models.yolov11 import YOLOv11Model
    from models.mobilenet_ssd import MobileNetSSDModel

    model_instances = {
        "yolo11n":       YOLOv11Model(variant="nano",  device=device),
        "yolo11s":       YOLOv11Model(variant="small", device=device),
        "mobilenet_ssd": MobileNetSSDModel(device=device),
    }
    # Only viz models that were benchmarked
    model_instances = {k: v for k, v in model_instances.items() if k in data["models"]}

    n_models = len(model_instances)
    n_imgs   = len(sample_paths)

    fig, axes = plt.subplots(
        n_imgs, n_models + 1,
        figsize=((n_models + 1) * 4, n_imgs * 3.5),
        facecolor=BG,
    )
    if n_imgs == 1:
        axes = [axes]  # make iterable over rows

    # Load and draw originals in first column
    for row, img_path in enumerate(sample_paths):
        img = PILImage.open(img_path).convert("RGB")
        ax  = axes[row][0]
        ax.imshow(np.array(img))
        ax.set_title("Ground Truth", color=TEXT, fontsize=8, pad=4)
        ax.axis("off")

    # Run each model column
    for col, (mkey, model) in enumerate(model_instances.items(), start=1):
        print(f"  Visualising {mkey} …")
        model.load()
        color = MODEL_COLORS.get(mkey, ACCENT)
        label = MODEL_LABELS.get(mkey, mkey)
        for row, img_path in enumerate(sample_paths):
            result = model.detect(img_path, conf_threshold=conf)
            img    = PILImage.open(img_path).convert("RGB")
            preds  = [{"bbox": d.bbox, "confidence": d.confidence}
                      for d in result.detections] if not result.error else []
            _draw_boxes(axes[row][col], img, preds, color,
                        f"{label} ({len(preds)} people)")
        model.unload()

    fig.suptitle("Person Detection — Sample Results", color=TEXT, fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4 — Latency distribution (box plots)
# ──────────────────────────────────────────────────────────────────────────────

def fig_latency_dist(data: dict, out: Path):
    models  = list(data["models"].keys())
    labels  = [MODEL_LABELS.get(m, m) for m in models]
    colors  = [MODEL_COLORS.get(m, ACCENT) for m in models]
    lat_data = [data["models"][m].get("latencies_ms", []) for m in models]

    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=BG)
    _style_ax(ax, title="Latency Distribution per Model",
              xlabel="Model", ylabel="Latency (ms)")

    bp = ax.boxplot(
        lat_data,
        labels=labels,
        patch_artist=True,
        medianprops=dict(color=TEXT, linewidth=2),
        whiskerprops=dict(color=GRID_COLOR),
        capprops=dict(color=GRID_COLOR),
        flierprops=dict(markerfacecolor=GRID_COLOR, markersize=3),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5 — Summary dashboard (2 × 2)
# ──────────────────────────────────────────────────────────────────────────────

def fig_dashboard(data: dict, out: Path):
    models  = list(data["models"].keys())
    labels  = [MODEL_LABELS.get(m, m) for m in models]
    colors  = [MODEL_COLORS.get(m, ACCENT) for m in models]

    map50   = [data["models"][m]["metrics"]["map50"]           for m in models]
    map75   = [data["models"][m]["metrics"]["map75"]           for m in models]
    latency = [data["models"][m]["metrics"]["mean_latency_ms"] for m in models]
    fps     = [data["models"][m]["metrics"]["fps"]             for m in models]
    lat_d   = [data["models"][m].get("latencies_ms", [])       for m in models]

    fig = plt.figure(figsize=(14, 10), facecolor=BG)
    gs  = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32)

    # ── top-left: mAP comparison ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _style_ax(ax1, title="Person Detection Accuracy (COCO128)", ylabel="mAP")
    x = np.arange(len(models))
    w = 0.3
    b50 = ax1.bar(x - w/2, map50, w, label="mAP@50",  color="#58a6ff", alpha=0.9)
    b75 = ax1.bar(x + w/2, map75, w, label="mAP@75",  color="#7ee787", alpha=0.9)
    for bars in (b50, b75):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                     f"{h:.3f}", ha="center", va="bottom", fontsize=7.5, color=TEXT)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylim(0, min(1.0, max(map50 + [0.1]) * 1.45))
    ax1.legend(fontsize=8)

    # ── top-right: FPS ────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _style_ax(ax2, title="Throughput (FPS)", ylabel="Frames / Second")
    bars = ax2.bar(labels, fps, color=colors, alpha=0.9, width=0.5)
    for bar, val in zip(bars, fps):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=9, color=TEXT)
    ax2.set_ylim(0, max(fps) * 1.35)

    # ── bottom-left: latency box plot ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    _style_ax(ax3, title="Latency Distribution", ylabel="ms / image")
    bp = ax3.boxplot(
        lat_d,
        labels=labels,
        patch_artist=True,
        medianprops=dict(color=TEXT, linewidth=2),
        whiskerprops=dict(color=GRID_COLOR),
        capprops=dict(color=GRID_COLOR),
        flierprops=dict(markerfacecolor=GRID_COLOR, markersize=3),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # ── bottom-right: summary table ───────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(PANEL_BG)
    ax4.axis("off")

    col_labels = ["Model", "mAP@50", "mAP@75", "Latency\n(ms)", "FPS"]
    rows = [
        [
            MODEL_LABELS.get(m, m),
            f"{data['models'][m]['metrics']['map50']:.3f}",
            f"{data['models'][m]['metrics']['map75']:.3f}",
            f"{data['models'][m]['metrics']['mean_latency_ms']:.1f}",
            f"{data['models'][m]['metrics']['fps']:.1f}",
        ]
        for m in models
    ]

    table = ax4.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.6)
    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor(PANEL_BG if r > 0 else "#1a1a2e")
        cell.set_edgecolor(GRID_COLOR)
        cell.set_text_props(color=TEXT)
        if r == 0:
            cell.set_text_props(color=ACCENT, fontweight="bold")

    ax4.set_title("Summary Table", color=TEXT, fontsize=10, pad=8)

    device = data.get("device", "GPU")
    fig.suptitle(
        f"People Detection Benchmark — CV Approach B  ({device.upper()}, COCO128)",
        color=TEXT, fontsize=13, y=1.01,
    )
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate detection benchmark figures")
    parser.add_argument("--results", default=None, help="Path to results JSON")
    parser.add_argument("--out-dir", default=str(FIGURES_DIR),
                        help="Output directory for figures")
    parser.add_argument("--no-samples", action="store_true",
                        help="Skip sample image visualisation (requires re-running models)")
    parser.add_argument("--n-samples", type=int, default=4,
                        help="Number of sample images to visualise (default: 4)")
    args = parser.parse_args()

    print("Loading results …")
    data = load_results(args.results)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = data.get("timestamp", "latest")

    print("\nGenerating figures …")
    fig_metrics(data, out_dir / f"detection_metrics_{ts}.png")
    fig_speed(  data, out_dir / f"detection_speed_{ts}.png")
    fig_latency_dist(data, out_dir / f"detection_latency_dist_{ts}.png")
    fig_dashboard(data, out_dir / f"detection_summary_{ts}.png")

    if not args.no_samples:
        print("Running inference for sample visualisations …")
        try:
            conf = data.get("conf", 0.25)
            fig_samples(data, out_dir / f"detection_samples_{ts}.png",
                        n_samples=args.n_samples, conf=conf)
        except Exception as e:
            print(f"  Warning: sample visualisation failed ({e})")

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
