#!/usr/bin/env python
"""
generate_plot.py — PNG summary plot for the environment monitoring benchmark.

Layout:
  Top row    : accuracy heatmap (model × category) + overall bar chart
  Bottom rows: example image panels — one clean + one messy per category,
               each annotated with per-model prediction chips and rationale

Usage:
    python generate_plot.py
    python generate_plot.py --results results/env_monitoring_results_XYZ.json
"""

from __future__ import annotations

import argparse
import glob
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image as PILImage

RESULTS_DIR = Path(__file__).parent.parent / "results"
CATEGORIES = ["blinds", "chairs", "table", "whiteboard"]

MODEL_COLORS = ["#58a6ff", "#f78166", "#7ee787", "#d2a8ff", "#ffa657", "#79c0ff", "#ff7b72", "#56d364"]

CLASS_COLOR = {
    "present_clean": "#3d9",
    "present_messy": "#e54",
    "not_present":   "#999",
    "uncertain":     "#fa0",
}
BG = "#0d0d0d"
PANEL_BG = "#111111"
TEXT_COLOR = "#cccccc"
GRID_COLOR = "#222222"


def load_latest(path: str | None) -> dict:
    if path:
        return json.loads(Path(path).read_text())
    files = sorted(glob.glob(str(RESULTS_DIR / "env_monitoring_results_*.json")))
    if not files:
        raise FileNotFoundError("No results found")
    return json.loads(Path(files[-1]).read_text())


def build_lookup(data: dict) -> dict[str, dict[str, dict]]:
    lookup: dict[str, dict[str, dict]] = {}
    for k, v in data["models"].items():
        for r in v["results"]:
            lookup.setdefault(r["image_filename"], {})[k] = r
    return lookup


def pick_examples(data: dict, lookup: dict) -> dict[str, dict[str, str]]:
    """
    For each category, pick one clean and one messy image filename.
    Prefer images where at least one model got it right and one wrong.
    """
    examples: dict[str, dict[str, str]] = {}
    by_cat: dict[str, list[str]] = {}
    for fn, models in lookup.items():
        any_r = next(iter(models.values()))
        ct = any_r.get("change_type", "")
        label = any_r.get("label", "")
        by_cat.setdefault(ct, {}).setdefault(label, [])
        by_cat[ct][label].append(fn)

    for ct in CATEGORIES:
        examples[ct] = {}
        for label in ("clean", "messy"):
            candidates = by_cat.get(ct, {}).get(label, [])
            if not candidates:
                continue
            # Prefer one with mixed correct/wrong across models
            best = candidates[0]
            for fn in candidates:
                preds = [lookup[fn][k].get("predicted_class") for k in data["models"] if k in lookup[fn]]
                gts = [lookup[fn][k].get("gt_class") for k in data["models"] if k in lookup[fn]]
                n_correct = sum(p == g for p, g in zip(preds, gts) if p)
                if 0 < n_correct < len(preds):
                    best = fn
                    break
            examples[ct][label] = best
    return examples


def draw_heatmap(ax, data: dict):
    model_keys = list(data["models"].keys())
    model_names = [v["model_name"].replace(" (", "\n(") for v in data["models"].values()]
    matrix = []
    for k in model_keys:
        m = data["models"][k]["metrics"]
        row = [m["per_change_type"].get(ct, {}).get("accuracy", 0.0) for ct in CATEGORIES]
        matrix.append(row)

    mat = np.array(matrix)
    cmap = LinearSegmentedColormap.from_list("rg", ["#3a0000", "#1a1a1a", "#003a00"])
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(CATEGORIES)))
    ax.set_xticklabels(CATEGORIES, color=TEXT_COLOR, fontsize=8)
    ax.set_yticks(range(len(model_keys)))
    ax.set_yticklabels(model_names, color=TEXT_COLOR, fontsize=7)
    ax.tick_params(colors=TEXT_COLOR, length=0)
    ax.set_facecolor(BG)

    for i in range(len(model_keys)):
        for j in range(len(CATEGORIES)):
            val = mat[i, j]
            color = "#3d9" if val >= 0.7 else "#fa0" if val >= 0.4 else "#e54"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    color=color, fontsize=9, fontweight="bold")

    ax.set_title("Accuracy by Model & Category", color=TEXT_COLOR, fontsize=9, pad=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)


def draw_overall_bars(ax, data: dict, model_colors: dict):
    model_keys = list(data["models"].keys())
    model_names = [data["models"][k]["model_name"].split(" (")[0] for k in model_keys]
    accs = [data["models"][k]["metrics"]["accuracy"] for k in model_keys]
    s1_accs = [data["models"][k]["metrics"]["stage1_accuracy"] for k in model_keys]

    x = np.arange(len(model_keys))
    w = 0.35
    bars1 = ax.bar(x - w/2, accs, w, color=[model_colors[k] for k in model_keys], alpha=0.9, label="Overall Acc")
    bars2 = ax.bar(x + w/2, s1_accs, w, color=[model_colors[k] for k in model_keys], alpha=0.4, label="Stage1 Det")

    for bar, val in zip(bars1, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.0%}", ha="center", va="bottom", color=TEXT_COLOR, fontsize=7)
    for bar, val in zip(bars2, s1_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.0%}", ha="center", va="bottom", color=TEXT_COLOR, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=25, ha="right", color=TEXT_COLOR, fontsize=7)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], color=TEXT_COLOR, fontsize=7)
    ax.tick_params(colors=TEXT_COLOR, length=0)
    ax.set_facecolor(BG)
    ax.set_title("Overall Accuracy (solid) vs Stage 1 Detection (faded)", color=TEXT_COLOR, fontsize=9, pad=8)
    ax.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=TEXT_COLOR, framealpha=0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)


def draw_image_ax(ax, fn: str, lookup: dict, title: str):
    """Draw just the image in ax, preserving aspect ratio."""
    row_data = lookup.get(fn, {})
    any_r = next(iter(row_data.values()), {})
    image_path = any_r.get("image_path", "")
    gt = any_r.get("gt_class", "")
    label = any_r.get("label", "")

    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.set_xticks([])
    ax.set_yticks([])

    try:
        img = PILImage.open(image_path).convert("RGB")
        ax.imshow(np.array(img), aspect="equal")
    except Exception:
        ax.text(0.5, 0.5, "no image", ha="center", va="center",
                color="#444", transform=ax.transAxes)

    gt_color = "#3d9" if "clean" in gt else "#e54"
    ax.set_title(f"{title}  |  GT: {label}", color=gt_color, fontsize=7, pad=4, fontweight="bold")


def draw_chips_ax(ax, fn: str, data: dict, lookup: dict, model_colors: dict):
    """Draw per-model prediction chips and rationale in ax."""
    row_data = lookup.get(fn, {})
    any_r = next(iter(row_data.values()), {})
    gt = any_r.get("gt_class", "")

    ax.set_facecolor(PANEL_BG)
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)

    model_keys = list(data["models"].keys())
    n = len(model_keys)
    slot_h = 1.0 / n

    for i, k in enumerate(model_keys):
        r = row_data.get(k)
        if not r:
            continue
        pred = r.get("predicted_class") or "—"
        correct = pred == gt
        desc = r.get("description", "") or ""
        s1 = {True: "s1:yes", False: "s1:no", None: "s1:?"}[r.get("stage1_detected")]
        s2_raw = r.get("stage2_ready")
        s2 = {True: "s2:ready", False: "s2:not_ready", None: ""}[s2_raw] if s2_raw is not None else ""

        accent = model_colors[k]
        chip_bg = "#0c1a0c" if correct else "#1a0c0c"
        tick = "✓" if correct else "✗"
        tick_color = "#3d9" if correct else "#e54"
        pred_color = CLASS_COLOR.get(pred, "#888")
        model_short = data["models"][k]["model_name"].split(" (")[0][:20]

        y_top = 1.0 - i * slot_h
        y_bot = y_top - slot_h
        pad = 0.015

        ax.add_patch(mpatches.FancyBboxPatch(
            (pad, y_bot + pad), 1 - 2*pad, slot_h - 2*pad,
            boxstyle="round,pad=0.01", transform=ax.transAxes,
            facecolor=chip_bg, edgecolor=accent, linewidth=1.0, clip_on=True
        ))

        y_mid = (y_top + y_bot) / 2
        # Model name
        ax.text(0.04, y_mid + slot_h * 0.22, model_short,
                transform=ax.transAxes, fontsize=6, color=accent,
                va="center", fontweight="bold")
        # Prediction class
        ax.text(0.04, y_mid,
                pred, transform=ax.transAxes, fontsize=6,
                color=pred_color, va="center", fontweight="bold")
        # Stage labels
        ax.text(0.04, y_mid - slot_h * 0.22,
                f"{s1}  {s2}", transform=ax.transAxes, fontsize=5,
                color="#666", va="center")
        # Tick
        ax.text(0.97, y_mid, tick, transform=ax.transAxes,
                fontsize=10, color=tick_color, va="center", ha="right", fontweight="bold")
        # Rationale
        if desc:
            wrapped = textwrap.fill(desc[:160], width=38)
            ax.text(0.38, y_mid, wrapped, transform=ax.transAxes,
                    fontsize=4.5, color="#777", va="center",
                    linespacing=1.3, clip_on=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=None)
    args = parser.parse_args()

    data = load_latest(args.results)
    model_keys = list(data["models"].keys())
    model_colors = {k: MODEL_COLORS[i % len(MODEL_COLORS)] for i, k in enumerate(model_keys)}
    lookup = build_lookup(data)
    examples = pick_examples(data, lookup)
    ts = data.get("timestamp", "latest")

    n_cats = len(CATEGORIES)
    # Layout: 2 top panels + n_cats * 2 image panels
    n_example_cols = n_cats * 2  # clean + messy per category

    fig = plt.figure(figsize=(22, 14), facecolor=BG)
    fig.suptitle(
        f"Environment Monitoring — Two-Stage Pipeline  ·  {len(model_keys)} models  ·  {ts}",
        color=TEXT_COLOR, fontsize=12, fontweight="bold", y=0.98
    )

    outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.35, height_ratios=[1, 2.2])

    # ── Top: heatmap + bar chart ──────────────────────────────────────────────
    top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.35, width_ratios=[1.2, 1])
    ax_heat = fig.add_subplot(top[0])
    ax_heat.set_facecolor(BG)
    draw_heatmap(ax_heat, data)

    ax_bar = fig.add_subplot(top[1])
    ax_bar.set_facecolor(BG)
    draw_overall_bars(ax_bar, data, model_colors)

    # ── Bottom: image row + chips row per example column ─────────────────────
    # Each column = one example. Split each into image (top) + chips (bottom).
    bot = gridspec.GridSpecFromSubplotSpec(
        2, n_example_cols, subplot_spec=outer[1],
        wspace=0.06, hspace=0.08, height_ratios=[1.1, 1.6]
    )

    col = 0
    for ct in CATEGORIES:
        for label in ("clean", "messy"):
            fn = examples.get(ct, {}).get(label)
            ax_img = fig.add_subplot(bot[0, col])
            ax_chips = fig.add_subplot(bot[1, col])
            if fn:
                draw_image_ax(ax_img, fn, lookup, f"{ct.upper()} · {label}")
                draw_chips_ax(ax_chips, fn, data, lookup, model_colors)
            else:
                ax_img.axis("off")
                ax_chips.axis("off")
            col += 1

    # ── Model legend ──────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=model_colors[k], label=data["models"][k]["model_name"].split(" (")[0])
        for k in model_keys
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=len(model_keys),
               facecolor=PANEL_BG, labelcolor=TEXT_COLOR, fontsize=8,
               framealpha=0.9, edgecolor=GRID_COLOR, bbox_to_anchor=(0.5, 0.01))

    out = RESULTS_DIR / f"env_monitoring_plot_{ts}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Plot → {out}")


if __name__ == "__main__":
    main()
