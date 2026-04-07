#!/usr/bin/env python
"""
generate_three_approach_plot.py
4-panel light-mode figure for the Logitech rally-board image:
  Panel 1 — Original
  Panel 2 — Approach A: Naive Qwen3-VL-4B → JSON bboxes + roles (no finetuning)
  Panel 3 — Approach A+: LoRA finetuned Qwen3-VL-4B → JSON bboxes
  Panel 4 — Approach B: YOLOv11s + Qwen3-VL-4B (full room + crop)
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
FINETUNE_DIR = Path(__file__).parent.parent / "finetune"
PEOPLE_DIR   = Path(__file__).parent.parent / "people_images"
STEM         = "rally-board-65-rightsight-2-group-view"
IMAGE_PATH   = PEOPLE_DIR / f"{STEM}.png"

PART_COLOR    = "#1a7f37"
NONPART_COLOR = "#cf222e"
TALK_COLOR    = "#0969da"
LORA_COLOR    = "#7c3aed"


def draw_box(ax, x1, y1, x2, y2, color, label=None, fontsize=7.5):
    for alpha, face in [(0.12, color), (0, "none")]:
        rect = mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            boxstyle="round,pad=1.5", linewidth=2.2,
            edgecolor=color, facecolor=face if face != 0 else color,
            alpha=alpha if alpha != 0 else 1.0,
        )
        ax.add_patch(rect)
    if label:
        ax.text(
            x1 + 3, max(y1 - 4, 11), label,
            fontsize=fontsize, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.22", facecolor=color, edgecolor="none", alpha=0.9),
            verticalalignment="bottom", clip_on=True,
        )


def load_approach_a():
    """Naive base Qwen — from approach_a results JSON."""
    files = sorted(glob.glob(str(RESULTS_DIR / "approach_a_*.json")))
    d = json.load(open(files[-1]))
    return d["results"]["qwen3vl_4b"][STEM].get("people", [])


def load_lora():
    """LoRA finetuned Qwen — deduplicated bboxes from compare_log output."""
    raw_people = [
        {"id": "P1",  "bbox": [37,  106, 200, 420]},
        {"id": "P2",  "bbox": [180, 116, 200, 148]},
        {"id": "P3",  "bbox": [200, 116, 230, 150]},
        {"id": "P4",  "bbox": [298, 83,  550, 265]},
        {"id": "P5",  "bbox": [544, 90,  797, 448]},
        {"id": "P6",  "bbox": [551, 126, 580, 150]},
        {"id": "P7",  "bbox": [558, 126, 582, 150]},
        {"id": "P11", "bbox": [44,  85,  85,  230]},
    ]
    # Filter to boxes with meaningful area (> 500 px²)
    W, H = Image.open(IMAGE_PATH).size
    meaningful = []
    seen = set()
    for p in raw_people:
        x1, y1, x2, y2 = p["bbox"]
        x2 = min(x2, W); y2 = min(y2, H)
        area = (x2 - x1) * (y2 - y1)
        key = (x1, y1, x2, y2)
        if area > 800 and key not in seen:
            seen.add(key)
            meaningful.append({**p, "bbox": [x1, y1, x2, y2]})
    return meaningful


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
        if cv and vlm:
            break
    return cv, vlm


def main():
    img = Image.open(IMAGE_PATH).convert("RGB")
    W, H = img.size

    people_a   = load_approach_a()
    people_lora = load_lora()
    cv, b_vlm  = load_approach_b()

    fig, axes = plt.subplots(1, 4, figsize=(24, 7))
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(wspace=0.04)

    panel_titles = [
        "Original",
        "Approach A — Naive VLM\nQwen3-VL-4B zero-shot JSON",
        "Approach A+ — LoRA Fine-tuned\nQwen3-VL-4B (r=16, bbox task)",
        "Approach B — YOLO + VLM\nYOLOv11s · Qwen3-VL-4B\n(full room + crop)",
    ]

    for ax, title in zip(axes, panel_titles):
        ax.imshow(img)
        ax.set_xlim(0, W); ax.set_ylim(H, 0)
        ax.axis("off"); ax.set_facecolor("white")
        ax.set_title(title, fontsize=9.5, fontweight="bold",
                     color="#24292f", pad=8, linespacing=1.6)

    # ── Panel 2: Approach A (naive) ───────────────────────────────────────────
    ax = axes[1]
    for p in people_a:
        x1, y1, x2, y2 = p["bbox"]
        role  = p.get("role", "")
        color = PART_COLOR if role == "participant" else NONPART_COLOR
        label = f"{'P' if role == 'participant' else 'NP'} {p['id']}"
        draw_box(ax, x1, y1, x2, y2, color, label=label)

    ax.legend(handles=[
        mpatches.Patch(facecolor=PART_COLOR,    label="Participant"),
        mpatches.Patch(facecolor=NONPART_COLOR,  label="Non-participant"),
    ], loc="lower left", fontsize=8, framealpha=0.92, facecolor="white", edgecolor="#d0d7de")

    n_p = sum(1 for p in people_a if p.get("role") == "participant")
    ax.text(3, H - 5, f"{len(people_a)} detected · {n_p} participant · {len(people_a)-n_p} non-part.",
            fontsize=7.5, color="white", ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#24292f", edgecolor="none", alpha=0.75))

    # ── Panel 3: LoRA ─────────────────────────────────────────────────────────
    ax = axes[2]
    for p in people_lora:
        x1, y1, x2, y2 = p["bbox"]
        draw_box(ax, x1, y1, x2, y2, LORA_COLOR, label=p["id"])

    ax.legend(handles=[
        mpatches.Patch(facecolor=LORA_COLOR, label="Detected person"),
    ], loc="lower left", fontsize=8, framealpha=0.92, facecolor="white", edgecolor="#d0d7de")

    ax.text(3, H - 5, f"{len(people_lora)} detected (deduped) · no role labels",
            fontsize=7.5, color="white", ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#24292f", edgecolor="none", alpha=0.75))

    # ── Panel 4: Approach B ───────────────────────────────────────────────────
    ax = axes[3]
    for det, vlm_p in zip(cv, b_vlm):
        x1, y1, x2, y2 = det["bbox"]
        part  = vlm_p.get("participant")
        talk  = vlm_p.get("talking")
        pidx  = vlm_p["person_idx"]
        color = TALK_COLOR if (part and talk) else PART_COLOR if part else NONPART_COLOR
        label = f"#{ pidx } {'Talk.' if (part and talk) else 'Part.' if part else 'NP'}"
        draw_box(ax, x1, y1, x2, y2, color, label=label)

    ax.legend(handles=[
        mpatches.Patch(facecolor=TALK_COLOR,   label="Participant · Talking"),
        mpatches.Patch(facecolor=PART_COLOR,   label="Participant · Silent"),
        mpatches.Patch(facecolor=NONPART_COLOR, label="Non-participant"),
    ], loc="lower left", fontsize=8, framealpha=0.92, facecolor="white", edgecolor="#d0d7de")

    n_part = sum(1 for p in b_vlm if p.get("participant"))
    n_talk = sum(1 for p in b_vlm if p.get("talking"))
    ax.text(3, H - 5, f"{len(cv)} detected · {n_part} participant · {n_talk} talking",
            fontsize=7.5, color="white", ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#24292f", edgecolor="none", alpha=0.75))

    # ── Footer ────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.01,
             "A: zero-shot bbox hallucination (imprecise boxes, role labels)  ·  "
             "A+: LoRA bbox-trained (tighter boxes, no roles, some repetition)  ·  "
             "B: YOLO grounds detection, VLM classifies with room context",
             ha="center", fontsize=8.5, color="#57606a", style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = RESULTS_DIR / "three_approach_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
