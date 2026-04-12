#!/usr/bin/env python
"""
plot_head_vs_face.py — Side-by-side comparison of YOLOv8-Face vs YOLOv8-Head
+ Qwen3-VL-4B participant classification on the Logitech rally-board image.

Usage
-----
    cd benchmark/face_detection/
    python plot_head_vs_face.py
    python plot_head_vs_face.py --face results/pipeline_20260406_221234.json \
                                 --head results/pipeline_head_20260410_173924.json
"""
from __future__ import annotations

import argparse
import json
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

PART_COLOR    = ("#1a7f37", "#2da44e")   # green  (edge, fill)
NONPART_COLOR = ("#cf222e", "#fa4549")   # red    (edge, fill)


def latest(pattern: str) -> Path:
    files = sorted(RESULTS_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file matching {pattern} in {RESULTS_DIR}")
    return files[-1]


def draw_panel(ax, img, dets: list, persons: list, title: str):
    W, H = img.size
    ax.imshow(np.array(img))
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.axis("off")

    for det in dets:
        f_idx           = det["face_idx"]
        conf            = det["confidence"]
        x1, y1, x2, y2 = det["dilated_bbox"]

        person   = next((p for p in persons if p.get("face_idx") == f_idx), None)
        part     = person.get("participant") if person else None
        edge_col, fill_col = PART_COLOR if part else NONPART_COLOR

        ax.add_patch(mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            boxstyle="round,pad=2",
            linewidth=2.2,
            edgecolor=edge_col,
            facecolor=fill_col + "22",
        ))
        part_str = "P" if part else "NP"
        ax.text(x1 + 4, max(y1 - 5, 12),
                f"#{f_idx} {part_str} {conf:.0%}",
                fontsize=7, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=edge_col,
                          edgecolor="none", alpha=0.9),
                verticalalignment="bottom", clip_on=True)

    n_det  = len(dets)
    n_part = sum(1 for p in persons if p.get("participant"))
    ax.set_title(
        f"{title}\n"
        f"{n_det} detected  ·  {n_part} participant  ·  {n_det - n_part} non-participant",
        fontsize=10, fontweight="bold", color="#24292f", pad=8, linespacing=1.5,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--face", default=None, help="Face pipeline JSON")
    parser.add_argument("--head", default=None, help="Head pipeline JSON")
    args = parser.parse_args()

    face_json = Path(args.face) if args.face else latest("pipeline_2*.json")
    head_json = Path(args.head) if args.head else latest("pipeline_head_*.json")

    face_data = json.loads(face_json.read_text())
    head_data = json.loads(head_json.read_text())

    img = PILImage.open(IMAGE_PATH).convert("RGB")

    # ── extract face panel data ────────────────────────────────────────────────
    face_vlm_key = next(iter(face_data["vlm_results"]))
    face_dets    = face_data["cv_results"].get(IMAGE_STEM, {}).get("detections", [])
    face_persons = face_data["vlm_results"][face_vlm_key].get(IMAGE_STEM, [])

    face_dilate  = face_data.get("dilate", 2.0)
    face_model   = face_data.get("detector", "YOLOv8-Face")
    face_conf    = face_data.get("conf", "—")

    # ── extract head panel data ────────────────────────────────────────────────
    head_vlm_key = next(iter(head_data["vlm_results"]))
    head_dets    = head_data["cv_results"].get(IMAGE_STEM, {}).get("detections", [])
    head_persons = head_data["vlm_results"][head_vlm_key].get(IMAGE_STEM, [])

    head_dilate  = head_data.get("dilate", 1.5)
    head_model   = head_data.get("detector", "YOLOv8-Head (nano)")
    head_conf    = head_data.get("conf", "—")

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(26, 8))
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(wspace=0.04)

    draw_panel(axes[0], img, face_dets, face_persons,
               f"YOLOv8-Face  ·  {face_vlm_key}\nconf ≥ {face_conf}  ·  dilated {face_dilate}×")

    draw_panel(axes[1], img, head_dets, head_persons,
               f"{head_model}  ·  {head_vlm_key}\nconf ≥ {head_conf}  ·  dilated {head_dilate}×")

    # Shared legend
    legend_elements = [
        mpatches.Patch(facecolor="#2da44e", edgecolor="#1a7f37", label="Participant"),
        mpatches.Patch(facecolor="#fa4549", edgecolor="#cf222e", label="Non-participant"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=10, framealpha=0.92, facecolor="white",
               edgecolor="#d0d7de", bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Face vs Head Detector Comparison  ·  Qwen3-VL-4B Participant Classification",
        fontsize=13, fontweight="bold", color="#24292f", y=1.01,
    )

    out = RESULTS_DIR / "head_vs_face_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
