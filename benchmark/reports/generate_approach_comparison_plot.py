#!/usr/bin/env python
"""
generate_approach_comparison_plot.py
Three-panel light-mode figure for the Logitech rally-board image:
  Panel 1 — Original image
  Panel 2 — Approach A: Qwen3-VL-4B asked to output JSON bounding boxes directly
  Panel 3 — Approach B: YOLOv11s detection + Qwen3-VL-4B classification (crop + full image)
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

RESULTS_DIR = Path(__file__).parent.parent / "results"
PEOPLE_DIR  = Path(__file__).parent.parent.parent / "people_images"
STEM        = "rally-board-65-rightsight-2-group-view"
IMAGE_PATH  = PEOPLE_DIR / f"{STEM}.png"

PART_COLOR    = "#1a7f37"   # green  — participant
NONPART_COLOR = "#cf222e"   # red    — non-participant
TALK_COLOR    = "#0969da"   # blue   — talking (approach B only)

BOX_ALPHA = 0.15


def draw_box(ax, x1, y1, x2, y2, edge_col, label=None, fontsize=7.5):
    w, h = x2 - x1, y2 - y1
    rect = mpatches.FancyBboxPatch(
        (x1, y1), w, h,
        boxstyle="round,pad=1.5",
        linewidth=2,
        edgecolor=edge_col,
        facecolor=edge_col,
        alpha=BOX_ALPHA,
    )
    ax.add_patch(rect)
    # edge only (solid)
    rect2 = mpatches.FancyBboxPatch(
        (x1, y1), w, h,
        boxstyle="round,pad=1.5",
        linewidth=2,
        edgecolor=edge_col,
        facecolor="none",
    )
    ax.add_patch(rect2)
    if label:
        ax.text(
            x1 + 3, max(y1 - 4, 12), label,
            fontsize=fontsize, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=edge_col, edgecolor="none", alpha=0.9),
            verticalalignment="bottom", clip_on=True,
        )


def load_approach_a():
    files = sorted(glob.glob(str(RESULTS_DIR / "approach_a_*.json")))
    d = json.load(open(files[-1]))
    return d["results"]["qwen3vl_4b"][STEM]


def load_approach_b():
    files = sorted(glob.glob(str(RESULTS_DIR / "pipeline_people_*.json")))
    cv, vlm = None, None
    for f in reversed(files):
        d = json.load(open(f))
        if "20260406" not in d["timestamp"]:
            continue
        if cv is None and "yolo11s" in d.get("cv_results", {}):
            cv = d["cv_results"]["yolo11s"][STEM]["detections"]
        if vlm is None and "qwen3vl_4b" in d.get("vlm_results", {}):
            vlm = d["vlm_results"]["qwen3vl_4b"][STEM]
        if cv is not None and vlm is not None:
            break
    return cv, vlm


def main():
    img = Image.open(IMAGE_PATH).convert("RGB")
    W, H = img.size
    arr = img

    a_data    = load_approach_a()
    cv, b_vlm = load_approach_b()

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor("white")

    titles = [
        "Original",
        "Approach A — VLM-only\nQwen3-VL-4B outputs JSON bboxes directly",
        "Approach B — YOLO + VLM\nYOLOv11s detection · Qwen3-VL-4B classification\n(full room + crop)",
    ]

    for ax, title in zip(axes, titles):
        ax.imshow(arr)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")
        ax.set_facecolor("white")
        ax.set_title(title, fontsize=10, fontweight="bold", color="#24292f",
                     pad=8, loc="center", linespacing=1.5)

    # ── Panel 2: Approach A ───────────────────────────────────────────────────
    ax_a = axes[1]
    people_a = a_data.get("people", [])
    for p in people_a:
        x1, y1, x2, y2 = p["bbox"]
        role  = p.get("role", "")
        color = PART_COLOR if role == "participant" else NONPART_COLOR
        label = f"{'Part.' if role == 'participant' else 'Non-part.'} {p['id']}"
        draw_box(ax_a, x1, y1, x2, y2, color, label=label)

    # Approach A legend
    leg_a = [
        mpatches.Patch(facecolor=PART_COLOR,    edgecolor=PART_COLOR,    label="Participant"),
        mpatches.Patch(facecolor=NONPART_COLOR,  edgecolor=NONPART_COLOR, label="Non-participant"),
    ]
    ax_a.legend(handles=leg_a, loc="lower left", fontsize=8.5, framealpha=0.92,
                facecolor="white", edgecolor="#d0d7de")

    n_part_a    = sum(1 for p in people_a if p.get("role") == "participant")
    n_nonpart_a = sum(1 for p in people_a if p.get("role") == "non-participant")
    ax_a.text(4, H - 6,
              f"{len(people_a)} detected  ·  {n_part_a} participant  ·  {n_nonpart_a} non-participant",
              fontsize=8, color="#57606a",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d0d7de", alpha=0.9),
              verticalalignment="bottom")

    # ── Panel 3: Approach B ───────────────────────────────────────────────────
    ax_b = axes[2]
    for det, vlm_p in zip(cv, b_vlm):
        x1, y1, x2, y2 = det["bbox"]
        part  = vlm_p.get("participant")
        talk  = vlm_p.get("talking")
        pidx  = vlm_p["person_idx"]

        if part:
            color = TALK_COLOR if talk else PART_COLOR
            label = f"#{ pidx } {'Talking' if talk else 'Participant'}"
        else:
            color = NONPART_COLOR
            label = f"#{pidx} Non-part."

        draw_box(ax_b, x1, y1, x2, y2, color, label=label)

    leg_b = [
        mpatches.Patch(facecolor=TALK_COLOR,    edgecolor=TALK_COLOR,    label="Participant · Talking"),
        mpatches.Patch(facecolor=PART_COLOR,    edgecolor=PART_COLOR,    label="Participant · Silent"),
        mpatches.Patch(facecolor=NONPART_COLOR,  edgecolor=NONPART_COLOR, label="Non-participant"),
    ]
    ax_b.legend(handles=leg_b, loc="lower left", fontsize=8.5, framealpha=0.92,
                facecolor="white", edgecolor="#d0d7de")

    n_part_b    = sum(1 for p in b_vlm if p.get("participant"))
    n_nonpart_b = sum(1 for p in b_vlm if not p.get("participant"))
    n_talking_b = sum(1 for p in b_vlm if p.get("talking"))
    ax_b.text(4, H - 6,
              f"{len(cv)} detected  ·  {n_part_b} participant  ·  {n_nonpart_b} non-participant  ·  {n_talking_b} talking",
              fontsize=8, color="#57606a",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d0d7de", alpha=0.9),
              verticalalignment="bottom")

    # ── Top annotation bar ────────────────────────────────────────────────────
    fig.text(0.5, 0.01,
             "Approach A: single VLM call, model hallucinates bboxes directly  |  "
             "Approach B: YOLO grounds detections, VLM reasons over full room + crop",
             ha="center", fontsize=9, color="#57606a",
             style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = RESULTS_DIR / "approach_comparison_plot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
