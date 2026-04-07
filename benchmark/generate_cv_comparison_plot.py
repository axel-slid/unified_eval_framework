#!/usr/bin/env python
"""
generate_cv_comparison_plot.py
3×3 grid: rows = 3 images, cols = YOLOv11n / YOLOv11s / MobileNet SSD
Light mode, bounding boxes drawn over each image.
"""
from __future__ import annotations

import json
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

RESULTS_DIR = Path(__file__).parent / "results"

DETECTOR_LABELS = {
    "yolo11n":       "YOLOv11n",
    "yolo11s":       "YOLOv11s",
    "mobilenet_ssd": "MobileNet SSD",
}
DETECTOR_COLORS = {
    "yolo11n":       "#0969da",
    "yolo11s":       "#1a7f37",
    "mobilenet_ssd": "#9a3412",
}
DETECTORS = ["yolo11n", "yolo11s", "mobilenet_ssd"]

# Pick 3 visually interesting images
CHOSEN_STEMS = [
    "rally-board-65-rightsight-2-group-view",
    "download (4)",
    "download",
]


def load_data():
    files = sorted(glob.glob(str(RESULTS_DIR / "pipeline_people_*.json")))
    for f in reversed(files):
        d = json.load(open(f))
        if "20260406" in d["timestamp"] and len(d.get("cv_results", {})) == 3:
            return d
    raise FileNotFoundError("No result file with all 3 detectors found")


def draw_boxes(ax, detections, color):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        w, h = x2 - x1, y2 - y1
        rect = mpatches.FancyBboxPatch(
            (x1, y1), w, h,
            boxstyle="round,pad=1.5",
            linewidth=2.2,
            edgecolor=color,
            facecolor=color,
            alpha=0.12,
        )
        ax.add_patch(rect)
        rect2 = mpatches.FancyBboxPatch(
            (x1, y1), w, h,
            boxstyle="round,pad=1.5",
            linewidth=2.2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect2)
        # confidence dot
        ax.text(
            x1 + 3, y1 + 3, f"{det['confidence']:.0%}",
            fontsize=7, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, edgecolor="none", alpha=0.85),
            verticalalignment="top", clip_on=True,
        )


def main():
    data = load_data()

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(wspace=0.04, hspace=0.12)

    for row, stem in enumerate(CHOSEN_STEMS):
        # find image path
        img_path = next(
            (p for p in data["images"] if stem in p), None
        )
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        for col, det_key in enumerate(DETECTORS):
            ax = axes[row][col]
            ax.imshow(img)
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
            ax.axis("off")
            ax.set_facecolor("white")

            det_data = data["cv_results"][det_key].get(stem, {})
            detections = det_data.get("detections", [])
            n = det_data.get("n_persons", 0)
            lat = det_data.get("latency_ms", 0)
            color = DETECTOR_COLORS[det_key]

            draw_boxes(ax, detections, color)

            # count badge
            ax.text(
                W - 4, H - 4,
                f"{n} person{'s' if n != 1 else ''}  ·  {lat:.0f}ms",
                fontsize=8, color="white", fontweight="bold",
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#24292f", edgecolor="none", alpha=0.75),
                clip_on=True,
            )

            # column header (top row only)
            if row == 0:
                ax.set_title(
                    DETECTOR_LABELS[det_key],
                    fontsize=12, fontweight="bold",
                    color=color, pad=8,
                )

        # row label on leftmost panel
        short = Path(img_path).stem[:30]
        axes[row][0].set_ylabel(short, fontsize=8, color="#57606a", labelpad=6)

    out = RESULTS_DIR / "cv_comparison_plot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
