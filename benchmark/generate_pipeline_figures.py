#!/usr/bin/env python
"""
generate_pipeline_figures.py — Visualise the two-stage people analysis pipeline.

Figures
-------
  1. pipeline_detections_<ts>.png    – 3 rows (one per detector) × N image cols
  2. pipeline_crops_vlm_<ts>.png     – Person crops with SmolVLM + Qwen3 labels
  3. pipeline_summary_<ts>.png       – Per-image counts + talking heatmap
  4. pipeline_single_image_<ts>.png  – One image: original | 3 detectors | Qwen3 crops

Usage
-----
    cd benchmark/
    python generate_pipeline_figures.py
    python generate_pipeline_figures.py --results results/pipeline_people_XYZ.json
    python generate_pipeline_figures.py --single-image "download (1).jpeg"
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image as PILImage

ROOT        = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# ── Light-mode style ──────────────────────────────────────────────────────────
BG         = "#ffffff"
PANEL_BG   = "#f5f7fa"
TEXT       = "#1a1a2e"
GRID       = "#dce1e8"
SUBTITLE   = "#555f6e"

MODEL_COLORS = {
    "yolo11n":       "#3b82f6",   # blue
    "yolo11s":       "#22c55e",   # green
    "mobilenet_ssd": "#ef4444",   # red
}
MODEL_LABELS = {
    "yolo11n":       "YOLOv11n (nano)",
    "yolo11s":       "YOLOv11s (small)",
    "mobilenet_ssd": "MobileNet SSD",
}
VLM_COLORS = {
    "smolvlm":    "#8b5cf6",
    "qwen3vl_4b": "#f97316",
}
VLM_LABELS = {
    "smolvlm":    "SmolVLM2-2.2B",
    "qwen3vl_4b": "Qwen3-VL-4B",
}
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


# ──────────────────────────────────────────────────────────────────────────────
def load_results(path: str | None) -> dict:
    if path:
        return json.loads(Path(path).read_text())
    files = sorted(glob.glob(str(RESULTS_DIR / "pipeline_people_*.json")))
    if not files:
        raise FileNotFoundError("No pipeline results. Run run_pipeline_people_analysis.py first.")
    return json.loads(Path(files[-1]).read_text())


def _style_ax(ax, title="", ylabel="", xlabel=""):
    ax.set_facecolor(PANEL_BG)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=7)
    if ylabel:
        ax.set_ylabel(ylabel, color=SUBTITLE, fontsize=8.5)
    if xlabel:
        ax.set_xlabel(xlabel, color=SUBTITLE, fontsize=8.5)


def _yn_str(val):
    if val is True:  return "YES"
    if val is False: return "NO"
    return "?"

def _yn_color(val):
    if val is True:  return YES_COLOR
    if val is False: return NO_COLOR
    return UNK_COLOR


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — CV Detections  (3 detector rows × N image cols — horizontal)
# ──────────────────────────────────────────────────────────────────────────────

def fig_detections(data: dict, out: Path):
    detectors   = list(data["cv_results"].keys())
    image_paths = data["images"]
    n_imgs = len(image_paths)
    n_det  = len(detectors)

    # Horizontal layout: detectors as rows, images as columns
    fig, axes = plt.subplots(
        n_det, n_imgs,
        figsize=(n_imgs * 3.8, n_det * 3.2),
        facecolor=BG,
    )
    axes = np.array(axes)
    if axes.ndim == 1:
        axes = axes.reshape(n_det, n_imgs)

    for row, det_key in enumerate(detectors):
        color = MODEL_COLORS.get(det_key, "#333")
        label = MODEL_LABELS.get(det_key, det_key)
        for col, img_path in enumerate(image_paths):
            ax   = axes[row][col]
            stem = Path(img_path).stem
            img  = PILImage.open(img_path).convert("RGB")
            dets = data["cv_results"][det_key].get(stem, {}).get("detections", [])

            ax.imshow(np.array(img))
            ax.set_facecolor("#000")
            ax.axis("off")

            for det in dets:
                x1, y1, x2, y2 = det["bbox"]
                rect = mpatches.FancyBboxPatch(
                    (x1, y1), x2 - x1, y2 - y1,
                    boxstyle="square,pad=0",
                    linewidth=2, edgecolor=color, facecolor="none",
                )
                ax.add_patch(rect)
                ax.text(x1 + 2, y1 + 10, f"{det['confidence']:.2f}",
                        color="white", fontsize=6.5,
                        bbox=dict(facecolor=color, alpha=0.75, pad=1, edgecolor="none"))

            if col == 0:
                ax.set_ylabel(label, color=color, fontsize=9, fontweight="bold",
                              labelpad=6)
                ax.yaxis.set_label_position("left")

            n = len(dets)
            title_color = color
            ax.set_title(f"{n} {'person' if n == 1 else 'people'}",
                         color=title_color, fontsize=8.5, pad=3)

    # Column headers = short image names
    for col, img_path in enumerate(image_paths):
        name = Path(img_path).name
        axes[0][col].annotate(
            name, xy=(0.5, 1.18), xycoords="axes fraction",
            ha="center", va="bottom", fontsize=8, color=SUBTITLE,
        )

    fig.suptitle("Stage 1 — CV Person Detection (all models × all images)",
                 color=TEXT, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(pad=0.5, h_pad=0.4, w_pad=0.3)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — Crop VLM analysis (horizontal, more columns)
# ──────────────────────────────────────────────────────────────────────────────

def fig_crops_vlm(data: dict, out: Path, max_cols: int = 9):
    crop_manifest = data["crop_manifest"]
    vlm_results   = data["vlm_results"]
    vlm_keys      = [k for k in vlm_results if k in VLM_LABELS]

    all_crops = []
    for img_path in data["images"]:
        stem = Path(img_path).stem
        for c in crop_manifest.get(stem, []):
            all_crops.append({"stem": stem, **c})

    if not all_crops:
        print("  No crops, skipping")
        return

    n_crops = len(all_crops)
    n_cols  = min(n_crops, max_cols)
    n_rows  = -(-n_crops // n_cols)

    row_h = 2.6 + len(vlm_keys) * 0.55
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.0, n_rows * row_h),
                             facecolor=BG)
    axes_flat = np.array(axes).flatten() if n_rows * n_cols > 1 else np.array([axes])

    for i, crop_info in enumerate(all_crops):
        ax = axes_flat[i]
        ax.set_facecolor(BG)
        ax.axis("off")

        cp = crop_info.get("crop_path")
        if cp and Path(cp).exists():
            try:
                ax.imshow(np.array(PILImage.open(cp).convert("RGB")))
            except Exception:
                pass

        stem  = crop_info["stem"]
        p_idx = crop_info["person_idx"]
        conf  = crop_info["confidence"]

        short = (stem[:9] + "…") if len(stem) > 9 else stem
        ax.set_title(f"{short}  p{p_idx}\nconf {conf:.2f}",
                     color=TEXT, fontsize=6.5, pad=3, linespacing=1.3)

        # One label chip per VLM
        for vi, vk in enumerate(vlm_keys):
            persons = vlm_results.get(vk, {}).get(stem, [])
            person  = next((p for p in persons if p["person_idx"] == p_idx), None)
            vlbl    = VLM_LABELS.get(vk, vk)

            if person:
                pv = person.get("participant")
                tv = person.get("talking")
                line = f"P:{_yn_str(pv)}  T:{_yn_str(tv)}"
                bg_c = _yn_color(tv)
            else:
                line  = "—"
                bg_c  = UNK_COLOR

            y_pos = -0.07 - vi * 0.20
            ax.text(0.5, y_pos,
                    f"{vlbl[:12]}\n{line}",
                    transform=ax.transAxes,
                    ha="center", va="top", fontsize=5.8,
                    color="white",
                    bbox=dict(facecolor=bg_c, alpha=0.88, pad=1.5,
                              edgecolor="none", boxstyle="round,pad=0.25"))

    for j in range(n_crops, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Stage 2 — VLM Crop Analysis   (P = Participant · T = Talking)",
                 color=TEXT, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout(pad=0.4)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 — Summary dashboard (horizontal)
# ──────────────────────────────────────────────────────────────────────────────

def fig_summary(data: dict, out: Path):
    detectors   = list(data["cv_results"].keys())
    vlm_keys    = [k for k in data["vlm_results"] if k in VLM_LABELS]
    image_paths = data["images"]
    short_names = [Path(p).name[:16] for p in image_paths]
    n_imgs  = len(image_paths)
    n_vlms  = len(vlm_keys)

    # Collect CV counts
    det_counts = {d: [] for d in detectors}
    for img_path in image_paths:
        stem = Path(img_path).stem
        for d in detectors:
            det_counts[d].append(data["cv_results"][d].get(stem, {}).get("n_persons", 0))

    # Collect VLM counts
    vlm_counts = {}
    for vk in vlm_keys:
        part_list, talk_list = [], []
        for img_path in image_paths:
            stem    = Path(img_path).stem
            persons = data["vlm_results"][vk].get(stem, [])
            part_list.append(sum(1 for p in persons if p.get("participant") is True))
            talk_list.append(sum(1 for p in persons if p.get("talking") is True))
        vlm_counts[vk] = {"participant": part_list, "talking": talk_list}

    crop_det = data["detector_for_crops"]

    x = np.arange(n_imgs)
    # Layout: 2 rows × (1 + n_vlms) cols  — wide
    n_cols = 1 + n_vlms + 1   # detection | smolvlm | qwen | heatmap
    fig, axes = plt.subplots(1, n_cols,
                             figsize=(n_cols * max(5, n_imgs * 1.4), 5.5),
                             facecolor=BG)

    # ── Col 0: CV detection counts ────────────────────────────────────────────
    ax0 = axes[0]
    _style_ax(ax0, title="Persons Detected", ylabel="Count")
    w = 0.22
    offsets = np.linspace(-(len(detectors)-1)*w/2, (len(detectors)-1)*w/2, len(detectors))
    for offset, dk in zip(offsets, detectors):
        bars = ax0.bar(x + offset, det_counts[dk], w * 0.92,
                       label=MODEL_LABELS.get(dk, dk),
                       color=MODEL_COLORS.get(dk, "#aaa"), alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax0.text(bar.get_x() + bar.get_width()/2, h + 0.05, str(int(h)),
                         ha="center", va="bottom", fontsize=6.5, color=TEXT)
    ax0.set_xticks(x)
    ax0.set_xticklabels(short_names, fontsize=7, rotation=30, ha="right")
    ax0.legend(fontsize=7, loc="upper right")
    ax0.set_ylim(0, max(max(v) for v in det_counts.values()) + 2.5)

    # ── Cols 1…n_vlms: VLM participant + talking ──────────────────────────────
    for vi, vk in enumerate(vlm_keys):
        ax = axes[1 + vi]
        vlbl = VLM_LABELS.get(vk, vk)
        _style_ax(ax, title=f"{vlbl}", ylabel="Count")
        n_det = [data["cv_results"][crop_det].get(Path(p).stem, {}).get("n_persons", 0)
                 for p in image_paths]
        ax.bar(x - 0.25, n_det,                   0.22,
               label="Detected", color="#94a3b8", alpha=0.6)
        ax.bar(x,        vlm_counts[vk]["participant"], 0.22,
               label="Participant", color=YES_COLOR, alpha=0.85)
        ax.bar(x + 0.25, vlm_counts[vk]["talking"],    0.22,
               label="Talking", color="#f97316", alpha=0.85)
        for xi, (nd, np_, nt) in enumerate(zip(
            n_det, vlm_counts[vk]["participant"], vlm_counts[vk]["talking"]
        )):
            for xoff, val in [(-0.25, nd), (0, np_), (0.25, nt)]:
                if val > 0:
                    ax.text(xi + xoff, val + 0.05, str(val),
                            ha="center", va="bottom", fontsize=6.5, color=TEXT)
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=7, rotation=30, ha="right")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylim(0, max(n_det + [1]) + 2.5)

    # ── Last col: talking fraction heatmap ────────────────────────────────────
    ax_h = axes[-1]
    ax_h.set_facecolor(PANEL_BG)

    matrix = []
    for vk in vlm_keys:
        row_vals = []
        for img_path in image_paths:
            stem    = Path(img_path).stem
            persons = data["vlm_results"][vk].get(stem, [])
            n_total = len(persons)
            n_talk  = sum(1 for p in persons if p.get("talking") is True)
            row_vals.append(n_talk / n_total if n_total > 0 else 0.0)
        matrix.append(row_vals)

    mat = np.array(matrix)
    im  = ax_h.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1,
                      aspect="auto", interpolation="nearest")
    ax_h.set_xticks(range(n_imgs))
    ax_h.set_xticklabels(short_names, fontsize=7, rotation=30, ha="right", color=TEXT)
    ax_h.set_yticks(range(n_vlms))
    ax_h.set_yticklabels([VLM_LABELS.get(vk, vk) for vk in vlm_keys], fontsize=8)
    for r in range(n_vlms):
        for c in range(n_imgs):
            ax_h.text(c, r, f"{mat[r,c]:.0%}", ha="center", va="center",
                      fontsize=8.5, fontweight="bold",
                      color="white" if mat[r,c] > 0.55 else TEXT)
    ax_h.set_title("Talking Fraction", color=TEXT, fontsize=10, fontweight="bold", pad=7)
    plt.colorbar(im, ax=ax_h, fraction=0.06, pad=0.04,
                 label="Fraction talking")

    fig.suptitle("Pipeline Summary — Meeting Room People Analysis",
                 color=TEXT, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(pad=0.8, w_pad=1.2)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4 — Single image deep-dive  (NEW)
#   Left: original  |  3 detection columns  |  Qwen3 crop grid
# ──────────────────────────────────────────────────────────────────────────────

def fig_single_image(data: dict, out: Path, image_name: str | None = None):
    """
    Horizontal layout for one image:
      Col 0        : original image (large)
      Cols 1–3     : detector overlays (YOLOv11n, YOLOv11s, MobileNet SSD)
      Cols 4+      : Qwen3-VL-4B crop thumbnails, one per detected person
    """
    # Pick image
    image_paths = data["images"]
    if image_name:
        chosen = next((p for p in image_paths if Path(p).name == image_name), None)
        if chosen is None:
            print(f"  Warning: '{image_name}' not found, using first image")
            chosen = image_paths[0]
    else:
        chosen = image_paths[0]

    stem      = Path(chosen).stem
    orig_img  = PILImage.open(chosen).convert("RGB")
    detectors = list(data["cv_results"].keys())
    qwen_key  = "qwen3vl_4b"

    # Gather Qwen crops for this image
    crops         = data["crop_manifest"].get(stem, [])
    qwen_persons  = data["vlm_results"].get(qwen_key, {}).get(stem, [])
    n_crops       = len(crops)

    # Layout: 1 (original) + 3 (detectors) + n_crops  →  very wide
    n_cols  = 1 + len(detectors) + max(n_crops, 1)
    col_w   = 3.0
    fig_w   = n_cols * col_w
    fig_h   = 5.5

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)

    # Use gridspec: first col wider
    from matplotlib.gridspec import GridSpec
    widths = [2.5] + [1.5] * len(detectors) + [1.2] * max(n_crops, 1)
    gs = GridSpec(1, n_cols, figure=fig, width_ratios=widths,
                  wspace=0.04, left=0.01, right=0.99, top=0.88, bottom=0.04)

    # ── Col 0: original ───────────────────────────────────────────────────────
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(np.array(orig_img))
    ax_orig.axis("off")
    ax_orig.set_title("Original", color=TEXT, fontsize=10, fontweight="bold", pad=5)

    # ── Cols 1–3: detectors ───────────────────────────────────────────────────
    for ci, det_key in enumerate(detectors, start=1):
        ax = fig.add_subplot(gs[0, ci])
        color = MODEL_COLORS.get(det_key, "#333")
        label = MODEL_LABELS.get(det_key, det_key)
        dets  = data["cv_results"][det_key].get(stem, {}).get("detections", [])

        ax.imshow(np.array(orig_img))
        ax.axis("off")

        for det in dets:
            x1, y1, x2, y2 = det["bbox"]
            rect = mpatches.FancyBboxPatch(
                (x1, y1), x2 - x1, y2 - y1,
                boxstyle="square,pad=0",
                linewidth=2.5, edgecolor=color, facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(x1 + 2, y1 + 12, f"{det['confidence']:.2f}",
                    color="white", fontsize=6,
                    bbox=dict(facecolor=color, alpha=0.8, pad=1.2, edgecolor="none"))

        n = len(dets)
        ax.set_title(f"{label}\n{n} {'person' if n == 1 else 'people'}",
                     color=color, fontsize=8.5, fontweight="bold", pad=4)

    # ── Cols 4+: Qwen3-VL-4B crops ───────────────────────────────────────────
    separator_col = 1 + len(detectors)

    # Vertical divider drawn on the figure
    fig.add_artist(
        plt.Line2D(
            [(separator_col - 0.2) / n_cols, (separator_col - 0.2) / n_cols],
            [0.02, 0.95],
            transform=fig.transFigure,
            color=GRID, linewidth=1.5, linestyle="--",
        )
    )

    # Column header
    if n_crops > 0:
        header_ax = fig.add_subplot(gs[0, separator_col:])
        header_ax.axis("off")
        header_ax.set_title(
            f"Qwen3-VL-4B  — {n_crops} person crops",
            color=VLM_COLORS.get(qwen_key, "#f97316"),
            fontsize=9.5, fontweight="bold", pad=4,
        )

    for ci, crop_info in enumerate(crops):
        col_idx = separator_col + ci
        if col_idx >= n_cols:
            break
        ax = fig.add_subplot(gs[0, col_idx])
        ax.axis("off")
        ax.set_facecolor(BG)

        cp = crop_info.get("crop_path")
        if cp and Path(cp).exists():
            try:
                crop_img = PILImage.open(cp).convert("RGB")
                ax.imshow(np.array(crop_img))
            except Exception:
                pass

        p_idx  = crop_info["person_idx"]
        person = next((p for p in qwen_persons if p["person_idx"] == p_idx), None)

        if person:
            pv = person.get("participant")
            tv = person.get("talking")
            # Two annotation boxes below
            p_color = YES_COLOR if pv else (NO_COLOR if pv is False else UNK_COLOR)
            t_color = YES_COLOR if tv else (NO_COLOR if tv is False else UNK_COLOR)

            ax.text(0.5, -0.04,
                    f"Participant: {_yn_str(pv)}",
                    transform=ax.transAxes,
                    ha="center", va="top", fontsize=7, color="white",
                    bbox=dict(facecolor=p_color, alpha=0.9, pad=2,
                              edgecolor="none", boxstyle="round,pad=0.3"))
            ax.text(0.5, -0.18,
                    f"Talking: {_yn_str(tv)}",
                    transform=ax.transAxes,
                    ha="center", va="top", fontsize=7, color="white",
                    bbox=dict(facecolor=t_color, alpha=0.9, pad=2,
                              edgecolor="none", boxstyle="round,pad=0.3"))

            # Reason snippets (truncated)
            p_reason = (person.get("participant_response") or "")[:55]
            t_reason = (person.get("talking_response")     or "")[:55]
            ax.set_title(f"p{p_idx}", color=SUBTITLE, fontsize=7.5, pad=3)
        else:
            ax.set_title(f"p{p_idx}", color=SUBTITLE, fontsize=7.5, pad=3)

    fig.suptitle(
        f"Deep Dive — {Path(chosen).name}  "
        f"|  Original · CV Detections · Qwen3-VL-4B Interpretation",
        color=TEXT, fontsize=12, fontweight="bold",
    )
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",      default=None,  help="Path to pipeline JSON")
    parser.add_argument("--out-dir",      default=str(FIGURES_DIR))
    parser.add_argument("--single-image", default=None,
                        help="Filename (e.g. 'download (1).jpeg') for the deep-dive figure")
    args = parser.parse_args()

    print("Loading results …")
    data    = load_results(args.results)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = data.get("timestamp", "latest")

    print("\nGenerating figures …")
    fig_detections(data,  out_dir / f"pipeline_detections_{ts}.png")
    fig_crops_vlm( data,  out_dir / f"pipeline_crops_vlm_{ts}.png")
    fig_summary(   data,  out_dir / f"pipeline_summary_{ts}.png")

    # Deep-dive: generate one per image
    for img_path in data["images"]:
        img_name = Path(img_path).name
        safe     = img_name.replace(" ", "_").replace("(", "").replace(")", "")
        fig_single_image(data, out_dir / f"pipeline_single_{safe}_{ts}.png",
                         image_name=img_name)

    print(f"\nAll figures → {out_dir}/")


if __name__ == "__main__":
    main()
