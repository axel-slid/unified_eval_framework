#!/usr/bin/env python
"""
generate_detection_plot.py — Light-mode figure: YOLOv11s boxes + Qwen3-VL-4B
participant/talking verdicts overlaid on the Logitech rally-board image.
"""
from __future__ import annotations

import json
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from PIL import Image
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"
PEOPLE_DIR  = Path(__file__).parent.parent.parent / "people_images"
IMAGE_STEM  = "rally-board-65-rightsight-2-group-view"
IMAGE_PATH  = PEOPLE_DIR / f"{IMAGE_STEM}.png"


def load_data():
    files = sorted(glob.glob(str(RESULTS_DIR / "pipeline_people_*.json")))
    for f in reversed(files):
        d = json.load(open(f))
        if "20260406" in d["timestamp"] and "qwen3vl_4b" in d["vlm_results"]:
            return d
    raise FileNotFoundError("No matching result file found")


def main():
    data   = load_data()
    stem   = IMAGE_STEM
    dets   = data["cv_results"]["yolo11s"][stem]["detections"]
    vlm    = data["vlm_results"]["qwen3vl_4b"][stem]

    img = Image.open(IMAGE_PATH).convert("RGB")
    W, H = img.size

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.imshow(img)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    # colour scheme
    COLORS = {
        ("participant", "talking"):      ("#1a7f37", "#2da44e"),  # green — active speaker
        ("participant", "silent"):       ("#0969da", "#218bff"),  # blue  — participant
        ("non-participant", "talking"):  ("#9a6700", "#d1a000"),  # amber — odd case
        ("non-participant", "silent"):   ("#cf222e", "#fa4549"),  # red   — not part.
    }

    for det, vlm_p in zip(dets, vlm):
        x1, y1, x2, y2 = det["bbox"]
        conf  = det["confidence"]
        part  = vlm_p.get("participant")
        talk  = vlm_p.get("talking")
        pidx  = vlm_p["person_idx"]

        role_key = (
            "participant" if part else "non-participant",
            "talking"     if talk else "silent",
        )
        edge_col, fill_col = COLORS[role_key]

        # Box
        rect = mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            boxstyle="round,pad=2",
            linewidth=2.5,
            edgecolor=edge_col,
            facecolor=fill_col + "22",
        )
        ax.add_patch(rect)

        # Label pill
        part_str = "Participant" if part else "Non-participant"
        talk_str = " · Talking" if talk else " · Silent"
        label    = f"#{pidx}  {part_str}{talk_str}  ({conf:.0%})"

        # pill background
        pill_y = max(y1 - 6, 14)
        ax.text(
            x1 + 4, pill_y, label,
            fontsize=7.5, fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=edge_col,
                edgecolor="none",
                alpha=0.92,
            ),
            verticalalignment="bottom",
            clip_on=True,
        )

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#2da44e", edgecolor="#1a7f37", label="Participant · Talking"),
        mpatches.Patch(facecolor="#218bff", edgecolor="#0969da", label="Participant · Silent"),
        mpatches.Patch(facecolor="#d1a000", edgecolor="#9a6700", label="Non-participant · Talking"),
        mpatches.Patch(facecolor="#fa4549", edgecolor="#cf222e", label="Non-participant · Silent"),
    ]
    legend = ax.legend(
        handles=legend_elements,
        loc="lower left",
        fontsize=9,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#d0d7de",
        title="Qwen3-VL-4B verdict",
        title_fontsize=9,
    )
    legend.get_title().set_fontweight("bold")

    ax.set_title(
        f"YOLOv11s detection  ·  Qwen3-VL-4B classification  ·  {len(dets)} persons detected",
        fontsize=11, fontweight="bold", color="#24292f", pad=10,
    )

    out = RESULTS_DIR / "pipeline_logitech_plot.png"
    plt.tight_layout(pad=0.5)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
