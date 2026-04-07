#!/usr/bin/env python
"""
face_detection/plot.py — Generate plots from a face_detection results JSON.

Produces:
  1. latency_comparison.png   — bar chart: mean latency ± std per model
  2. faces_per_image.png      — grouped bar: faces detected per image per model
  3. montage_<image>.png      — side-by-side annotated images for each photo
  4. dashboard.png            — combined summary dashboard

Usage
-----
    cd benchmark/face_detection/
    python plot.py                           # auto-picks latest results JSON
    python plot.py --results results/results_20260406_214432.json
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
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image as PILImage

HERE = Path(__file__).parent

PALETTE = {
    "mtcnn":      "#5B8DB8",   # steel blue
    "retinaface": "#E8735A",   # coral red
    "yolov8face": "#3BAA78",   # emerald green
}


# ── helpers ───────────────────────────────────────────────────────────────────

def latest_json(results_dir: Path) -> Path:
    jsons = sorted(results_dir.glob("results_*.json"))
    if not jsons:
        sys.exit(f"No results JSON found in {results_dir}")
    return jsons[-1]


def load_results(path: Path) -> dict:
    return json.loads(path.read_text())


def short_name(filename: str, max_len: int = 14) -> str:
    stem = Path(filename).stem
    return stem if len(stem) <= max_len else stem[:max_len - 1] + "…"


# ── plot 1: latency comparison ────────────────────────────────────────────────

def plot_latency(data: dict, out_dir: Path) -> Path:
    models  = list(data["models"].keys())
    labels  = [data["models"][k]["label"] for k in models]
    means   = [data["models"][k]["summary"]["mean_latency_ms"] for k in models]
    stds    = [data["models"][k]["summary"]["std_latency_ms"]  for k in models]
    fps_val = [data["models"][k]["summary"]["fps"]             for k in models]
    colors  = [PALETTE.get(k, "#888") for k in models]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Face Detection — Latency & Throughput", fontsize=14, fontweight="bold", y=1.01)

    # Latency bar
    ax = axes[0]
    bars = ax.bar(labels, means, yerr=stds, capsize=6, color=colors,
                  edgecolor="white", linewidth=0.8, width=0.55)
    for bar, val, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 1,
                f"{val:.1f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean inference latency (ms)", fontsize=11)
    ax.set_title("Mean Latency ± Std", fontsize=12)
    ax.set_ylim(0, max(means) * 1.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # FPS bar
    ax = axes[1]
    bars2 = ax.bar(labels, fps_val, color=colors, edgecolor="white", linewidth=0.8, width=0.55)
    for bar, val in zip(bars2, fps_val):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Frames per second", fontsize=11)
    ax.set_title("Throughput (FPS)", fontsize=12)
    ax.set_ylim(0, max(fps_val) * 1.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    out = out_dir / "latency_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ── plot 2: faces per image ───────────────────────────────────────────────────

def plot_faces_per_image(data: dict, out_dir: Path) -> Path:
    models    = list(data["models"].keys())
    labels    = [data["models"][k]["label"] for k in models]
    img_names = [short_name(p) for p in data["images"]]
    n_imgs    = len(img_names)
    n_models  = len(models)

    faces_matrix = np.zeros((n_models, n_imgs), dtype=int)
    for mi, key in enumerate(models):
        for ii, img_full in enumerate(data["images"]):
            img_name = Path(img_full).name
            row = next((r for r in data["models"][key]["per_image"]
                        if r["image"] == img_name), None)
            if row:
                faces_matrix[mi, ii] = row["n_faces"]

    x = np.arange(n_imgs)
    width = 0.25
    offsets = np.linspace(-(n_models - 1) * width / 2,
                          (n_models - 1) * width / 2, n_models)

    fig, ax = plt.subplots(figsize=(13, 5))
    for mi, (key, label) in enumerate(zip(models, labels)):
        bars = ax.bar(x + offsets[mi], faces_matrix[mi], width,
                      label=label, color=PALETTE.get(key, "#888"),
                      edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars, faces_matrix[mi]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        str(val), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(img_names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Faces detected", fontsize=11)
    ax.set_title("Faces Detected per Image", fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, faces_matrix.max() + 2)

    fig.tight_layout()
    out = out_dir / "faces_per_image.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ── plot 3: annotated image montages ─────────────────────────────────────────

def build_montages(data: dict, vis_dir: Path, out_dir: Path) -> list[Path]:
    """For each source image, create a side-by-side panel of all 3 annotated versions."""
    models    = list(data["models"].keys())
    labels    = [data["models"][k]["label"] for k in models]
    img_names = [Path(p).name for p in data["images"]]
    out_paths = []

    for img_name in img_names:
        stem = Path(img_name).stem
        panels = []
        for key in models:
            candidates = list(vis_dir.glob(f"{stem}__{key}.*"))
            if not candidates:
                continue
            ann = cv2.imread(str(candidates[0]))
            panels.append(ann)

        if not panels:
            continue

        # Resize all panels to same height
        target_h = 400
        resized = []
        for p in panels:
            h, w = p.shape[:2]
            scale = target_h / h
            resized.append(cv2.resize(p, (int(w * scale), target_h)))

        row = np.concatenate(resized, axis=1)

        # Add model name headers (light grey bar)
        header_h = 32
        header = np.full((header_h, row.shape[1], 3), 240, dtype=np.uint8)
        col_w = row.shape[1] // len(resized)
        for i, label in enumerate(labels[:len(resized)]):
            cv2.putText(header, label, (i * col_w + 8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 50, 50), 1)
        composite = np.concatenate([header, row], axis=0)

        out_path = out_dir / f"montage_{stem}.jpg"
        cv2.imwrite(str(out_path), composite)
        out_paths.append(out_path)

    return out_paths


# ── plot 4: image × model matrix ─────────────────────────────────────────────

def plot_image_model_matrix(
    data: dict,
    vis_dir: Path,
    out_dir: Path,
    n_images: int = 3,
) -> Path:
    """
    Grid: rows = images (first n_images), columns = models.
    Each cell is the annotated image for that (image, model) pair.
    Column headers = model names, row labels = image filenames.
    """
    models    = list(data["models"].keys())
    labels    = [data["models"][k]["label"] for k in models]
    # Always make the last row the Logitech rally-board image; fill the rest from the top
    logitech = next((p for p in data["images"] if "rally" in Path(p).name.lower()), None)
    others   = [p for p in data["images"] if p != logitech]
    img_paths = (others[: n_images - 1] + [logitech]) if logitech else data["images"][:n_images]
    n_cols    = len(models)
    n_rows    = len(img_paths)

    # Colour each column header to match the model's palette colour
    col_colors = [PALETTE.get(k, "#888888") for k in models]

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 4 * n_rows),
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    for col, (key, label, hdr_color) in enumerate(zip(models, labels, col_colors)):
        for row, img_full in enumerate(img_paths):
            img_name = Path(img_full).name
            stem     = Path(img_name).stem
            ax       = axes[row][col]

            candidates = list(vis_dir.glob(f"{stem}__{key}.*"))
            if candidates:
                bgr = cv2.imread(str(candidates[0]))
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                ax.imshow(rgb)
            else:
                ax.set_facecolor("#eeeeee")
                ax.text(0.5, 0.5, "not found", ha="center", va="center",
                        transform=ax.transAxes, color="#999999")

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Row label on leftmost column
            if col == 0:
                short = stem if len(stem) <= 16 else stem[:15] + "…"
                ax.set_ylabel(short, fontsize=10, color="#333333",
                              rotation=0, labelpad=60, va="center", ha="right")

            # Column header on top row — coloured rectangle + white text
            if row == 0:
                ax.set_title(label, fontsize=12, fontweight="bold",
                             color="white", pad=0,
                             bbox=dict(boxstyle="square,pad=0.35",
                                       facecolor=hdr_color, edgecolor="none"))

    fig.suptitle("Face Detection — Model × Image Matrix",
                 fontsize=14, fontweight="bold", color="#111111", y=1.01)
    fig.tight_layout(h_pad=0.4, w_pad=0.4)

    out = out_dir / "model_image_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ── plot 5: dashboard ─────────────────────────────────────────────────────────

def plot_dashboard(data: dict, out_dir: Path) -> Path:
    models  = list(data["models"].keys())
    labels  = [data["models"][k]["label"] for k in models]
    means   = [data["models"][k]["summary"]["mean_latency_ms"] for k in models]
    stds    = [data["models"][k]["summary"]["std_latency_ms"]  for k in models]
    fps_val = [data["models"][k]["summary"]["fps"]             for k in models]
    totals  = [data["models"][k]["summary"]["total_faces"]     for k in models]
    colors  = [PALETTE.get(k, "#888") for k in models]

    img_names = [short_name(p) for p in data["images"]]
    n_imgs    = len(img_names)
    n_models  = len(models)
    faces_matrix = np.zeros((n_models, n_imgs), dtype=int)
    for mi, key in enumerate(models):
        for ii, img_full in enumerate(data["images"]):
            img_name = Path(img_full).name
            row = next((r for r in data["models"][key]["per_image"]
                        if r["image"] == img_name), None)
            if row:
                faces_matrix[mi, ii] = row["n_faces"]

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

    title_kw = dict(fontsize=11, fontweight="bold", color="#222222", pad=8)
    label_kw = dict(fontsize=9,  color="#555555")

    def style_ax(ax, title):
        ax.set_facecolor("#f7f7f7")
        ax.set_title(title, **title_kw)
        ax.tick_params(colors="#555555", labelsize=8)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("#cccccc")
        ax.spines["bottom"].set_color("#cccccc")
        ax.grid(axis="y", color="#dddddd", linestyle="--", alpha=0.8)

    # ── Panel 1: Latency bar ──
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(labels, means, yerr=stds, capsize=5, color=colors,
                   edgecolor="white", linewidth=0.8, width=0.5)
    for bar, val, std in zip(bars, means, stds):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + max(means) * 0.02,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=8.5,
                 color="#222222", fontweight="bold")
    ax1.set_ylabel("ms", **label_kw)
    ax1.set_ylim(0, max(means) * 1.45)
    style_ax(ax1, "Mean Latency ± Std (ms)")

    # ── Panel 2: FPS bar ──
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(labels, fps_val, color=colors, edgecolor="white", linewidth=0.8, width=0.5)
    for bar, val in zip(bars2, fps_val):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(fps_val) * 0.02,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=8.5,
                 color="#222222", fontweight="bold")
    ax2.set_ylabel("FPS", **label_kw)
    ax2.set_ylim(0, max(fps_val) * 1.3)
    style_ax(ax2, "Throughput (FPS)")

    # ── Panel 3: Total faces donut ──
    ax3 = fig.add_subplot(gs[0, 2])
    wedges, texts, autotexts = ax3.pie(
        totals, labels=labels, colors=colors, autopct="%d",
        startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"color": "#222222", "fontsize": 9},
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color("#222222")
    ax3.set_title("Total Faces Detected", **title_kw)
    ax3.set_facecolor("#f7f7f7")

    # ── Panel 4 (bottom, full width): faces per image ──
    ax4 = fig.add_subplot(gs[1, :])
    x = np.arange(n_imgs)
    width = 0.25
    offsets = np.linspace(-(n_models - 1) * width / 2, (n_models - 1) * width / 2, n_models)
    for mi, (key, label) in enumerate(zip(models, labels)):
        bars4 = ax4.bar(x + offsets[mi], faces_matrix[mi], width,
                        label=label, color=PALETTE.get(key, "#888"),
                        edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars4, faces_matrix[mi]):
            if val > 0:
                ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                         str(val), ha="center", va="bottom", fontsize=7.5, color="#333333")
    ax4.set_xticks(x)
    ax4.set_xticklabels(img_names, rotation=20, ha="right", fontsize=8.5, color="#444444")
    ax4.set_ylabel("Faces detected", **label_kw)
    ax4.set_ylim(0, faces_matrix.max() + 2)
    ax4.legend(framealpha=0.9, facecolor="white", edgecolor="#cccccc", fontsize=9)
    style_ax(ax4, "Faces Detected per Image  (all models)")

    fig.suptitle("Face Detection Benchmark  ·  people_images",
                 fontsize=15, fontweight="bold", color="#111111", y=1.01)

    out = out_dir / "dashboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=None,
                        help="Path to results JSON (default: latest in results/)")
    parser.add_argument("--out-dir", default=str(HERE / "results"),
                        help="Where to save plots")
    args = parser.parse_args()

    results_dir = Path(args.out_dir)
    json_path   = Path(args.results) if args.results else latest_json(results_dir)
    print(f"Loading results from {json_path}")
    data = load_results(json_path)

    # Find the matching vis directory (vis_<timestamp>)
    ts      = json_path.stem.replace("results_", "")
    vis_dir = results_dir / f"vis_{ts}"

    print("Generating plots …")

    p1 = plot_latency(data, results_dir)
    print(f"  {p1}")

    p2 = plot_faces_per_image(data, results_dir)
    print(f"  {p2}")

    montages = []
    if vis_dir.exists():
        montages = build_montages(data, vis_dir, results_dir)
        for m in montages:
            print(f"  {m}")

        p_matrix = plot_image_model_matrix(data, vis_dir, results_dir, n_images=3)
        print(f"  {p_matrix}")
    else:
        print(f"  (vis dir {vis_dir} not found — skipping montages and matrix)")

    p4 = plot_dashboard(data, results_dir)
    print(f"  {p4}")

    print("\nDone.")
    return {"latency": p1, "per_image": p2, "montages": montages, "dashboard": p4}


if __name__ == "__main__":
    main()
