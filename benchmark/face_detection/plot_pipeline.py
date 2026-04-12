#!/usr/bin/env python
"""
face_detection/plot_pipeline.py — Figures for the YOLOv8-Face + Qwen3 pipeline.

Figures
-------
  1. pipeline_detections.png  – face bboxes (original + dilated) on every image
  2. pipeline_crops_vlm.png   – dilated face crops with Qwen3 P/T labels
  3. pipeline_summary.png     – per-image counts + talking heatmap
  4. pipeline_matrix.png      – images × faces matrix with full crop + verdict

Usage
-----
    cd benchmark/face_detection/
    python plot_pipeline.py                          # latest results JSON
    python plot_pipeline.py --results results/pipeline_20260406_123456.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image as PILImage

HERE        = Path(__file__).parent
RESULTS_DIR = HERE / "results"

# ── style ─────────────────────────────────────────────────────────────────────
BG       = "#ffffff"
PANEL_BG = "#f5f7fa"
TEXT     = "#1a1a2e"
GRID     = "#dce1e8"
SUBTITLE = "#555f6e"

FACE_COLOR   = "#5B8DB8"   # steel blue — matches model palette
DILATE_COLOR = "#E8735A"   # coral — dilated box
QWEN_COLOR   = "#3BAA78"   # emerald

YES_COLOR = "#16a34a"
NO_COLOR  = "#dc2626"
UNK_COLOR = "#9ca3af"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "text.color":        TEXT,
    "grid.color":        GRID,
    "legend.facecolor":  BG,
    "legend.edgecolor":  GRID,
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ── helpers ───────────────────────────────────────────────────────────────────

def latest_json(d: Path) -> Path:
    jsons = sorted(d.glob("pipeline_*.json"))
    if not jsons:
        sys.exit(f"No pipeline JSON found in {d}")
    return jsons[-1]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def _yn_str(v):
    return "YES" if v is True else ("NO" if v is False else "?")

def _yn_color(v):
    return YES_COLOR if v is True else (NO_COLOR if v is False else UNK_COLOR)

def _short(name: str, n: int = 14) -> str:
    return name if len(name) <= n else name[:n-1] + "…"

def _style_ax(ax, title="", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=7)
    if ylabel:
        ax.set_ylabel(ylabel, color=SUBTITLE, fontsize=8.5)


# ── Figure 1 — Face detections: original + dilated boxes ─────────────────────

def fig_detections(data: dict, out: Path):
    image_paths = data["images"]
    n_imgs      = len(image_paths)
    dilate      = data.get("dilate", 2.0)

    fig, axes = plt.subplots(1, n_imgs, figsize=(n_imgs * 4.0, 4.5), facecolor=BG)
    if n_imgs == 1:
        axes = [axes]

    for col, img_path in enumerate(image_paths):
        ax   = axes[col]
        stem = Path(img_path).stem
        img  = PILImage.open(img_path).convert("RGB")
        dets = data["cv_results"].get(stem, {}).get("detections", [])

        ax.imshow(np.array(img))
        ax.axis("off")

        for det in dets:
            # Original face bbox (blue)
            x1, y1, x2, y2 = det["original_bbox"]
            ax.add_patch(mpatches.FancyBboxPatch(
                (x1, y1), x2-x1, y2-y1, boxstyle="square,pad=0",
                linewidth=1.5, edgecolor=FACE_COLOR, facecolor="none", linestyle="--",
            ))
            # Dilated bbox (coral, solid)
            d1, d2, d3, d4 = det["dilated_bbox"]
            ax.add_patch(mpatches.FancyBboxPatch(
                (d1, d2), d3-d1, d4-d2, boxstyle="square,pad=0",
                linewidth=2, edgecolor=DILATE_COLOR, facecolor="none",
            ))
            ax.text(d1 + 2, d2 + 12, f"{det['confidence']:.2f}",
                    color="white", fontsize=6,
                    bbox=dict(facecolor=DILATE_COLOR, alpha=0.8, pad=1, edgecolor="none"))

        n = len(dets)
        ax.set_title(f"{_short(Path(img_path).name)}\n{n} face(s)",
                     color=TEXT, fontsize=8.5, pad=4)

    # Legend
    handles = [
        mpatches.Patch(edgecolor=FACE_COLOR,   facecolor="none", linestyle="--",
                       linewidth=1.5, label="YOLOv8-Face bbox"),
        mpatches.Patch(edgecolor=DILATE_COLOR, facecolor="none",
                       linewidth=2,   label=f"Dilated {dilate}× (crop region)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Stage 1 — YOLOv8-Face Detection  (dashed = original · solid = dilated crop)",
                 color=TEXT, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout(pad=0.4)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  {out}")


# ── Figure 2 — Crop + VLM labels ─────────────────────────────────────────────

def fig_crops_vlm(data: dict, out: Path):
    vlm_keys = list(data["vlm_results"].keys())

    all_crops = []
    for img_path in data["images"]:
        stem = Path(img_path).stem
        for det in data["cv_results"].get(stem, {}).get("detections", []):
            all_crops.append({"stem": stem, **det})

    if not all_crops:
        print("  No crops — skipping fig_crops_vlm")
        return

    n_crops = len(all_crops)
    n_cols  = min(n_crops, 10)
    n_rows  = -(-n_crops // n_cols)
    row_h   = 2.8 + len(vlm_keys) * 0.6

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.0, n_rows * row_h),
                             facecolor=BG, squeeze=False)
    axes_flat = axes.flatten()

    for i, crop_info in enumerate(all_crops):
        ax    = axes_flat[i]
        ax.axis("off")
        ax.set_facecolor(BG)

        cp = crop_info.get("crop_path")
        if cp and Path(cp).exists():
            ax.imshow(np.array(PILImage.open(cp).convert("RGB")))

        stem  = crop_info["stem"]
        f_idx = crop_info["face_idx"]
        conf  = crop_info["confidence"]
        ax.set_title(f"{_short(stem, 10)}  f{f_idx}\n{conf:.2f}",
                     color=TEXT, fontsize=6.5, pad=3, linespacing=1.3)

        for vi, vk in enumerate(vlm_keys):
            persons = data["vlm_results"].get(vk, {}).get(stem, [])
            person  = next((p for p in persons if p["face_idx"] == f_idx), None)
            vlbl    = vk.replace("qwen3vl_", "Qwen3-VL-").replace("_int8", "-int8")

            if person:
                pv   = person.get("participant")
                line = f"P:{_yn_str(pv)}"
                bgc  = _yn_color(pv)
            else:
                line = "—"
                bgc  = UNK_COLOR

            y_pos = -0.08 - vi * 0.22
            ax.text(0.5, y_pos, f"{vlbl[:14]}\n{line}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=5.8, color="white",
                    bbox=dict(facecolor=bgc, alpha=0.88, pad=1.5,
                              edgecolor="none", boxstyle="round,pad=0.25"))

    for j in range(n_crops, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Stage 2 — Qwen3-VL-4B on Dilated Face Crops   (P=Participant)",
                 color=TEXT, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout(pad=0.4)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  {out}")


# ── Figure 3 — Summary dashboard ─────────────────────────────────────────────

def fig_summary(data: dict, out: Path):
    vlm_keys    = list(data["vlm_results"].keys())
    image_paths = data["images"]
    short_names = [_short(Path(p).name, 16) for p in image_paths]
    n_imgs  = len(image_paths)
    n_vlms  = len(vlm_keys)

    face_counts = []
    for img_path in image_paths:
        stem = Path(img_path).stem
        face_counts.append(data["cv_results"].get(stem, {}).get("n_faces", 0))

    vlm_counts = {}
    for vk in vlm_keys:
        part_list, nonpart_list = [], []
        for img_path in image_paths:
            stem    = Path(img_path).stem
            persons = data["vlm_results"][vk].get(stem, [])
            n_part  = sum(1 for p in persons if p.get("participant") is True)
            part_list.append(n_part)
            nonpart_list.append(len(persons) - n_part)
        vlm_counts[vk] = {"participant": part_list, "non_participant": nonpart_list}

    n_cols = 1 + n_vlms + 1
    fig, axes = plt.subplots(1, n_cols,
                             figsize=(n_cols * max(5, n_imgs * 1.4), 5.5),
                             facecolor=BG)
    if n_cols == 1:
        axes = [axes]

    x = np.arange(n_imgs)

    # ── Face count bar ────────────────────────────────────────────────────────
    ax0 = axes[0]
    _style_ax(ax0, title="Faces Detected (YOLOv8-Face)", ylabel="Count")
    bars = ax0.bar(x, face_counts, 0.5, color=FACE_COLOR, alpha=0.85)
    for bar, v in zip(bars, face_counts):
        if v > 0:
            ax0.text(bar.get_x() + bar.get_width()/2, v + 0.05, str(v),
                     ha="center", va="bottom", fontsize=8, color=TEXT)
    ax0.set_xticks(x)
    ax0.set_xticklabels(short_names, fontsize=7, rotation=30, ha="right")
    ax0.set_ylim(0, max(face_counts) + 2.5)

    # ── Per-VLM participant / non-participant bars ────────────────────────────
    for vi, vk in enumerate(vlm_keys):
        ax  = axes[1 + vi]
        lbl = vk.replace("qwen3vl_", "Qwen3-VL-").replace("_int8", "-int8")
        _style_ax(ax, title=lbl, ylabel="Count")
        ax.bar(x - 0.12, vlm_counts[vk]["participant"],    0.22, label="Participant",     color=YES_COLOR, alpha=0.85)
        ax.bar(x + 0.12, vlm_counts[vk]["non_participant"], 0.22, label="Non-participant", color=NO_COLOR,  alpha=0.85)
        for xi, (np_, nnp) in enumerate(zip(vlm_counts[vk]["participant"], vlm_counts[vk]["non_participant"])):
            for xoff, val in [(-0.12, np_), (0.12, nnp)]:
                if val > 0:
                    ax.text(xi + xoff, val + 0.05, str(val),
                            ha="center", va="bottom", fontsize=6.5, color=TEXT)
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=7, rotation=30, ha="right")
        ax.legend(fontsize=7)
        ax.set_ylim(0, max(face_counts) + 2.5)

    # ── Participant fraction heatmap ──────────────────────────────────────────
    ax_h = axes[-1]
    ax_h.set_facecolor(PANEL_BG)
    matrix = []
    for vk in vlm_keys:
        row_vals = []
        for img_path in image_paths:
            stem    = Path(img_path).stem
            persons = data["vlm_results"][vk].get(stem, [])
            n_total = len(persons)
            n_part  = sum(1 for p in persons if p.get("participant") is True)
            row_vals.append(n_part / n_total if n_total > 0 else 0.0)
        matrix.append(row_vals)

    mat = np.array(matrix)
    im  = ax_h.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax_h.set_xticks(range(n_imgs))
    ax_h.set_xticklabels(short_names, fontsize=7, rotation=30, ha="right", color=TEXT)
    ax_h.set_yticks(range(n_vlms))
    ax_h.set_yticklabels(
        [vk.replace("qwen3vl_", "Qwen3-VL-").replace("_int8", "-int8") for vk in vlm_keys],
        fontsize=8)
    for r in range(n_vlms):
        for c in range(n_imgs):
            ax_h.text(c, r, f"{mat[r,c]:.0%}", ha="center", va="center",
                      fontsize=9, fontweight="bold",
                      color="white" if mat[r,c] > 0.55 else TEXT)
    ax_h.set_title("Participant Fraction", color=TEXT, fontsize=10, fontweight="bold", pad=7)
    plt.colorbar(im, ax=ax_h, fraction=0.06, pad=0.04)

    fig.suptitle("Pipeline Summary — YOLOv8-Face + Qwen3-VL-4B  (Participant / Non-participant)",
                 color=TEXT, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(pad=0.8, w_pad=1.2)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  {out}")


# ── Figure 4 — Image × Face matrix (cols=images, rows=faces) ─────────────────

def fig_matrix(data: dict, out: Path, vlm_key: str = "qwen3vl_4b"):
    """
    Columns = images, rows = face index.
    Each cell shows the dilated face crop + Participant/Talking verdict below.
    """
    image_paths = data["images"]
    n_imgs      = len(image_paths)

    # Find max faces across all images
    max_faces = max(
        data["cv_results"].get(Path(p).stem, {}).get("n_faces", 0)
        for p in image_paths
    )
    if max_faces == 0:
        print("  No faces detected — skipping matrix")
        return

    n_rows = max_faces
    n_cols = n_imgs

    cell_h = 3.2
    cell_w = 2.4
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(cell_w * n_cols, cell_h * n_rows),
                             squeeze=False, facecolor=BG)
    fig.patch.set_facecolor(BG)

    for col, img_path in enumerate(image_paths):
        stem    = Path(img_path).stem
        dets    = data["cv_results"].get(stem, {}).get("detections", [])
        persons = data["vlm_results"].get(vlm_key, {}).get(stem, [])

        for row in range(n_rows):
            ax = axes[row][col]
            ax.axis("off")
            ax.set_facecolor("#f0f0f0")

            if row < len(dets):
                det   = dets[row]
                f_idx = det["face_idx"]
                cp    = det.get("crop_path")

                if cp and Path(cp).exists():
                    ax.imshow(np.array(PILImage.open(cp).convert("RGB")))

                person = next((p for p in persons if p.get("face_idx") == f_idx), None)
                if person:
                    pv = person.get("participant")
                    p_color = _yn_color(pv)
                    ax.text(0.5, -0.05,
                            f"Participant: {_yn_str(pv)}",
                            transform=ax.transAxes, ha="center", va="top",
                            fontsize=7.5, color="white",
                            bbox=dict(facecolor=p_color, alpha=0.92, pad=2,
                                      edgecolor="none", boxstyle="round,pad=0.3"))
                    # Confidence
                    ax.text(0.02, 0.02, f"{det['confidence']:.2f}",
                            transform=ax.transAxes, ha="left", va="bottom",
                            fontsize=6.5, color="white",
                            bbox=dict(facecolor="#333", alpha=0.6, pad=1.5,
                                      edgecolor="none"))

            # Column header on top row
            if row == 0:
                short = _short(Path(img_path).name, 18)
                ax.set_title(short, fontsize=8.5, color=TEXT,
                             fontweight="bold", pad=5)

        # Row label (face index) on leftmost column
        for row in range(n_rows):
            if col == 0:
                axes[row][0].set_ylabel(f"Face {row}", fontsize=9, color=SUBTITLE,
                                        rotation=0, labelpad=42, va="center")

    vlbl = vlm_key.replace("qwen3vl_", "Qwen3-VL-").replace("_int8", "-int8")
    fig.suptitle(
        f"Face Pipeline Matrix — YOLOv8-Face + {vlbl}\n"
        f"Columns = images · Rows = face index · Dilated {data.get('dilate', 2.0)}× crops",
        color=TEXT, fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout(h_pad=1.8, w_pad=0.4)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  {out}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=None)
    parser.add_argument("--out-dir", default=str(RESULTS_DIR / "pipeline_figures"))
    args = parser.parse_args()

    json_path = Path(args.results) if args.results else latest_json(RESULTS_DIR)
    print(f"Loading {json_path}")
    data    = load(json_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = data.get("timestamp", "latest")
    vlm_key = next(iter(data["vlm_results"]), "qwen3vl_4b")

    print("Generating figures …")
    fig_detections(data, out_dir / f"pipeline_detections_{ts}.png")
    fig_crops_vlm( data, out_dir / f"pipeline_crops_vlm_{ts}.png")
    fig_summary(   data, out_dir / f"pipeline_summary_{ts}.png")
    fig_matrix(    data, out_dir / f"pipeline_matrix_{ts}.png", vlm_key=vlm_key)
    print(f"\nAll figures → {out_dir}/")


if __name__ == "__main__":
    main()
