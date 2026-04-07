#!/usr/bin/env python
"""
face_detection/plot_logitech.py — Logitech rally-board image with YOLOv8-Face
dilated bboxes colour-coded by Qwen3-VL-4B participant/talking verdict.

Colour scheme (matches generate_detection_plot.py):
  Green  — Participant · Talking
  Blue   — Participant · Silent
  Amber  — Non-participant · Talking
  Red    — Non-participant · Silent

Usage
-----
    cd benchmark/face_detection/
    python plot_logitech.py
    python plot_logitech.py --results results/pipeline_20260406_221234.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image as PILImage

HERE        = Path(__file__).parent
RESULTS_DIR = HERE / "results"
IMAGE_STEM  = "rally-board-65-rightsight-2-group-view"
IMAGE_PATH  = HERE.parent.parent / "people_images" / f"{IMAGE_STEM}.png"

COLORS = {
    ("participant",     "talking"): ("#1a7f37", "#2da44e"),  # green
    ("participant",     "silent"):  ("#0969da", "#218bff"),  # blue
    ("non-participant", "talking"): ("#9a6700", "#d1a000"),  # amber
    ("non-participant", "silent"):  ("#cf222e", "#fa4549"),  # red
}


def latest_json(d: Path) -> Path:
    jsons = sorted(d.glob("pipeline_*.json"))
    if not jsons:
        sys.exit(f"No pipeline JSON found in {d}")
    return jsons[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=None)
    args = parser.parse_args()

    json_path = Path(args.results) if args.results else latest_json(RESULTS_DIR)
    data      = json.loads(json_path.read_text())
    vlm_key   = next(iter(data["vlm_results"]))

    dets    = data["cv_results"].get(IMAGE_STEM, {}).get("detections", [])
    persons = data["vlm_results"][vlm_key].get(IMAGE_STEM, [])

    img  = PILImage.open(IMAGE_PATH).convert("RGB")
    W, H = img.size

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.imshow(np.array(img))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    for det in dets:
        f_idx  = det["face_idx"]
        conf   = det["confidence"]
        x1, y1, x2, y2 = det["dilated_bbox"]

        person = next((p for p in persons if p.get("face_idx") == f_idx), None)
        part   = person.get("participant") if person else None
        talk   = person.get("talking")     if person else None

        role_key  = (
            "participant"     if part else "non-participant",
            "talking"         if talk else "silent",
        )
        edge_col, fill_col = COLORS[role_key]

        # Box
        ax.add_patch(mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            boxstyle="round,pad=2",
            linewidth=2.5,
            edgecolor=edge_col,
            facecolor=fill_col + "22",
        ))

        # Label pill
        part_str = "Participant" if part else "Non-participant"
        talk_str = " · Talking"  if talk else " · Silent"
        label    = f"#{f_idx}  {part_str}{talk_str}  ({conf:.0%})"

        pill_y = max(y1 - 6, 14)
        ax.text(
            x1 + 4, pill_y, label,
            fontsize=7.5, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=edge_col,
                      edgecolor="none", alpha=0.92),
            verticalalignment="bottom", clip_on=True,
        )

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#2da44e", edgecolor="#1a7f37", label="Participant · Talking"),
        mpatches.Patch(facecolor="#218bff", edgecolor="#0969da", label="Participant · Silent"),
        mpatches.Patch(facecolor="#d1a000", edgecolor="#9a6700", label="Non-participant · Talking"),
        mpatches.Patch(facecolor="#fa4549", edgecolor="#cf222e", label="Non-participant · Silent"),
    ]
    legend = ax.legend(
        handles=legend_elements, loc="lower left",
        fontsize=9, framealpha=0.92, facecolor="white", edgecolor="#d0d7de",
        title="Qwen3-VL-4B verdict", title_fontsize=9,
    )
    legend.get_title().set_fontweight("bold")

    vlbl = vlm_key.replace("qwen3vl_", "Qwen3-VL-").replace("_int8", "-int8")
    ax.set_title(
        f"YOLOv8-Face detection  ·  {vlbl} classification  ·  "
        f"{len(dets)} faces detected  ·  dilated {data.get('dilate', 2.0)}×",
        fontsize=11, fontweight="bold", color="#24292f", pad=10,
    )

    out = RESULTS_DIR / "pipeline_logitech_face_plot.png"
    plt.tight_layout(pad=0.5)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
