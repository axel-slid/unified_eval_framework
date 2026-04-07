#!/usr/bin/env python
"""
generate_prompting_report.py — Figures + HTML report for the prompting-techniques benchmark.

Reads results JSON from run_benchmark_prompting_techniques.py and produces:
  1.  Grouped bar — Item Accuracy by model × technique
  2.  Grouped bar — Room Accuracy by model × technique
  3.  Grouped bar — Room F1 by model × technique
  4.  Heatmap — per-item accuracy per (model, technique)
  5.  Technique delta bars — gain vs DIRECT baseline (item acc + room F1)
  6.  Latency comparison — mean latency per model × technique
  7.  Per-model small multiples — all metrics, one subplot per model
  8.  Radar / spider — per-technique profile across all metrics
  9.  Confusion breakdown (TP/FP/FN/TN stacked) per model × technique
  10. Parse-error rate bar chart

All figures are embedded as base64 PNGs in a single dark-themed HTML report.

Usage:
    cd benchmark
    python generate_prompting_report.py
    python generate_prompting_report.py --results results/prompting_techniques_results_XYZ.json
    python generate_prompting_report.py --save-pngs          # also save individual PNGs
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ── Style constants ────────────────────────────────────────────────────────────

BG        = "#0d0d0d"
PANEL_BG  = "#111111"
TEXT      = "#cccccc"
GRID      = "#1e1e1e"
ACCENT    = "#58a6ff"

TECHNIQUE_COLORS = {
    "direct":            "#58a6ff",
    "cot":               "#7ee787",
    "few_shot":          "#f78166",
    "few_shot_per_item": "#d2a8ff",
}
TECHNIQUE_LABELS = {
    "direct":            "Direct",
    "cot":               "Chain-of-Thought",
    "few_shot":          "Few-Shot Batch",
    "few_shot_per_item": "Few-Shot Per-Item",
}

MODEL_PALETTE = [
    "#58a6ff", "#f78166", "#7ee787", "#d2a8ff",
    "#ffa657", "#79c0ff", "#ff7b72", "#56d364",
]

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ── Matplotlib theme ───────────────────────────────────────────────────────────

def _apply_theme(fig, axes_flat):
    fig.patch.set_facecolor(BG)
    for ax in axes_flat:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(color=GRID, linewidth=0.5)


def _fig(nrows=1, ncols=1, w=10, h=5):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))
    axes_arr = np.array(axes)
    axes_flat = axes_arr.flatten().tolist()
    _apply_theme(fig, axes_flat)
    return fig, axes_flat


def _to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=BG, dpi=130)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _bar_label(ax, bars, fmt=".1%", pad=3, color=TEXT, fontsize=7):
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + pad * 0.003,
                format(h, fmt),
                ha="center", va="bottom",
                color=color, fontsize=fontsize,
            )


# ── Data helpers ───────────────────────────────────────────────────────────────

def _model_short(name: str) -> str:
    """Shorten model display names."""
    name = name.replace("SmolVLM2 (", "SmolVLM\n(").replace("InternVL3 (", "InternVL3\n(")
    name = name.replace("Qwen3-VL (", "Qwen3-VL\n(")
    return name.rstrip(")")


def _get_metric(data: dict, tech: str, metric: str, default=0.0):
    return data.get(tech, {}).get("metrics", {}).get(metric, default)


def _get_per_item(data: dict, tech: str, iid: str) -> float:
    return data.get(tech, {}).get("metrics", {}).get("per_item", {}).get(iid, {}).get("accuracy", 0.0)


# ── Figure 1 — Item Accuracy grouped bar ──────────────────────────────────────

def fig_item_accuracy(all_results, techniques, model_keys, model_names):
    fig, (ax,) = _fig(1, 1, w=max(10, len(model_keys) * 2.5), h=5)
    x = np.arange(len(model_keys))
    n = len(techniques)
    width = 0.7 / n
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    for i, tech in enumerate(techniques):
        vals = [_get_metric(all_results[mk], tech, "item_accuracy") for mk in model_keys]
        bars = ax.bar(x + offsets[i], vals, width, label=TECHNIQUE_LABELS[tech],
                      color=TECHNIQUE_COLORS.get(tech, MODEL_PALETTE[i]), alpha=0.88)
        _bar_label(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels([_model_short(n) for n in model_names], fontsize=7)
    ax.set_ylabel("Item Accuracy", color=TEXT)
    ax.set_ylim(0, 1.12)
    ax.set_title("Item Accuracy by Model & Technique", color=TEXT, fontsize=11)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    return _to_b64(fig)


# ── Figure 2 — Room Accuracy grouped bar ──────────────────────────────────────

def fig_room_accuracy(all_results, techniques, model_keys, model_names):
    fig, (ax,) = _fig(1, 1, w=max(10, len(model_keys) * 2.5), h=5)
    x = np.arange(len(model_keys))
    n = len(techniques)
    width = 0.7 / n
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    for i, tech in enumerate(techniques):
        vals = [_get_metric(all_results[mk], tech, "room_accuracy") for mk in model_keys]
        bars = ax.bar(x + offsets[i], vals, width, label=TECHNIQUE_LABELS[tech],
                      color=TECHNIQUE_COLORS.get(tech, MODEL_PALETTE[i]), alpha=0.88)
        _bar_label(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels([_model_short(n) for n in model_names], fontsize=7)
    ax.set_ylabel("Room Accuracy", color=TEXT)
    ax.set_ylim(0, 1.12)
    ax.set_title("Room-Level Accuracy by Model & Technique", color=TEXT, fontsize=11)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    return _to_b64(fig)


# ── Figure 3 — Room F1 grouped bar ────────────────────────────────────────────

def fig_room_f1(all_results, techniques, model_keys, model_names):
    fig, (ax,) = _fig(1, 1, w=max(10, len(model_keys) * 2.5), h=5)
    x = np.arange(len(model_keys))
    n = len(techniques)
    width = 0.7 / n
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    for i, tech in enumerate(techniques):
        vals = [_get_metric(all_results[mk], tech, "room_f1") for mk in model_keys]
        bars = ax.bar(x + offsets[i], vals, width, label=TECHNIQUE_LABELS[tech],
                      color=TECHNIQUE_COLORS.get(tech, MODEL_PALETTE[i]), alpha=0.88)
        _bar_label(ax, bars, fmt=".3f")

    ax.set_xticks(x)
    ax.set_xticklabels([_model_short(n) for n in model_names], fontsize=7)
    ax.set_ylabel("Room-Ready F1", color=TEXT)
    ax.set_ylim(0, 1.12)
    ax.set_title("Room-Ready F1 Score by Model & Technique", color=TEXT, fontsize=11)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    fig.tight_layout()
    return _to_b64(fig)


# ── Figure 4 — Per-item accuracy heatmap ──────────────────────────────────────

def fig_per_item_heatmap(all_results, techniques, model_keys, model_names, checklist):
    item_ids = [str(c["id"]) for c in checklist]
    item_labels = [c["item"][:35] for c in checklist]
    n_tech = len(techniques)
    cols = n_tech

    fig, axes = plt.subplots(1, cols, figsize=(cols * max(5, len(model_keys) * 1.2), max(4, len(item_ids) * 0.6 + 1.5)))
    if cols == 1:
        axes = [axes]
    fig.patch.set_facecolor(BG)

    cmap = LinearSegmentedColormap.from_list("rg", ["#1f0d0d", "#e54", "#fa0", "#8bc34a", "#3d9"])

    for col, tech in enumerate(techniques):
        ax = axes[col]
        ax.set_facecolor(PANEL_BG)

        # rows=items, cols=models
        matrix = np.zeros((len(item_ids), len(model_keys)))
        for j, mk in enumerate(model_keys):
            for i, iid in enumerate(item_ids):
                matrix[i, j] = _get_per_item(all_results[mk], tech, iid)

        im = ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")

        ax.set_xticks(range(len(model_keys)))
        ax.set_xticklabels([_model_short(n) for n in model_names], rotation=30, ha="right", fontsize=6, color=TEXT)
        ax.set_yticks(range(len(item_ids)))
        ax.set_yticklabels(item_labels if col == 0 else [], fontsize=7, color=TEXT)
        ax.set_title(TECHNIQUE_LABELS[tech], color=TEXT, fontsize=9)

        for i in range(len(item_ids)):
            for j in range(len(model_keys)):
                val = matrix[i, j]
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=6, color="white" if val < 0.6 else "black")

        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)

    fig.suptitle("Per-Item Accuracy Heatmap (items × models, per technique)", color=TEXT, fontsize=10, y=1.01)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02).ax.tick_params(colors=TEXT, labelsize=7)
    fig.tight_layout()
    return _to_b64(fig)


# ── Figure 5 — Technique delta vs DIRECT baseline ─────────────────────────────

def fig_technique_delta(all_results, techniques, model_keys, model_names):
    if "direct" not in techniques:
        return None

    non_direct = [t for t in techniques if t != "direct"]
    if not non_direct:
        return None

    fig, axes_arr = plt.subplots(1, 2, figsize=(max(10, len(model_keys) * 2), 5))
    axes_arr = list(axes_arr)
    _apply_theme(fig, axes_arr)

    metrics = [("item_accuracy", "Item Accuracy Delta"), ("room_f1", "Room F1 Delta")]

    for ax, (metric_key, title) in zip(axes_arr, metrics):
        x = np.arange(len(model_keys))
        n = len(non_direct)
        width = 0.7 / n
        offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

        for i, tech in enumerate(non_direct):
            deltas = []
            for mk in model_keys:
                base = _get_metric(all_results[mk], "direct", metric_key)
                val = _get_metric(all_results[mk], tech, metric_key)
                deltas.append(val - base)

            colors = [TECHNIQUE_COLORS.get(tech, MODEL_PALETTE[i])] * len(deltas)
            bar_colors = ["#3d9" if d >= 0 else "#e54" for d in deltas]
            bars = ax.bar(x + offsets[i], deltas, width,
                          label=TECHNIQUE_LABELS[tech], color=bar_colors, alpha=0.85)
            for bar, d in zip(bars, deltas):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        d + (0.005 if d >= 0 else -0.015),
                        f"{d:+.1%}", ha="center", va="bottom" if d >= 0 else "top",
                        color=TEXT, fontsize=7)

        ax.axhline(0, color=TEXT, linewidth=0.8, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([_model_short(n) for n in model_names], fontsize=7)
        ax.set_title(title + " vs Direct Baseline", color=TEXT, fontsize=10)
        ax.set_ylabel("Δ (technique − direct)", color=TEXT)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0) if "accuracy" in metric_key else matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:+.2f}"))

        handles = [mpatches.Patch(color=TECHNIQUE_COLORS.get(t, MODEL_PALETTE[j]), label=TECHNIQUE_LABELS[t])
                   for j, t in enumerate(non_direct)]
        ax.legend(handles=handles, facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

    fig.suptitle("Technique Gain vs Direct Baseline", color=TEXT, fontsize=11)
    fig.tight_layout()
    return _to_b64(fig)


# ── Figure 6 — Latency comparison ─────────────────────────────────────────────

def fig_latency(all_results, techniques, model_keys, model_names):
    fig, (ax,) = _fig(1, 1, w=max(10, len(model_keys) * 2.5), h=5)
    x = np.arange(len(model_keys))
    n = len(techniques)
    width = 0.7 / n
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    for i, tech in enumerate(techniques):
        vals = [_get_metric(all_results[mk], tech, "mean_latency_ms") for mk in model_keys]
        bars = ax.bar(x + offsets[i], vals, width, label=TECHNIQUE_LABELS[tech],
                      color=TECHNIQUE_COLORS.get(tech, MODEL_PALETTE[i]), alpha=0.88)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 5,
                        f"{v:.0f}", ha="center", va="bottom", color=TEXT, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([_model_short(n) for n in model_names], fontsize=7)
    ax.set_ylabel("Mean Latency (ms)", color=TEXT)
    ax.set_title("Mean Inference Latency per Image by Model & Technique", color=TEXT, fontsize=11)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    fig.tight_layout()
    return _to_b64(fig)


# ── Figure 7 — Per-model small multiples ──────────────────────────────────────

def fig_per_model_small_multiples(all_results, techniques, model_keys, model_names, checklist):
    item_ids = [str(c["id"]) for c in checklist]
    item_short = [c["item"][:20] + "…" if len(c["item"]) > 20 else c["item"] for c in checklist]

    ncols = min(3, len(model_keys))
    nrows = (len(model_keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 5, nrows * 4 + 0.5),
                              squeeze=False)
    fig.patch.set_facecolor(BG)
    all_axes = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    for idx, mk in enumerate(model_keys):
        ax = all_axes[idx]
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(color=GRID, linewidth=0.4, axis="y")

        x = np.arange(len(item_ids))
        n = len(techniques)
        width = 0.8 / n
        offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

        for i, tech in enumerate(techniques):
            vals = [_get_per_item(all_results[mk], tech, iid) for iid in item_ids]
            ax.bar(x + offsets[i], vals, width,
                   color=TECHNIQUE_COLORS.get(tech, MODEL_PALETTE[i]),
                   alpha=0.85, label=TECHNIQUE_LABELS[tech])

        ax.set_xticks(x)
        ax.set_xticklabels(item_short, rotation=35, ha="right", fontsize=6, color=TEXT)
        ax.set_ylim(0, 1.15)
        ax.tick_params(colors=TEXT, labelsize=7)
        ax.set_title(model_names[idx], color=TEXT, fontsize=8)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

        if idx == 0:
            ax.legend(facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=6)

    # Hide unused axes
    for idx in range(len(model_keys), len(all_axes)):
        all_axes[idx].set_visible(False)

    fig.suptitle("Per-Item Accuracy by Technique — Each Model", color=TEXT, fontsize=11, y=1.01)
    fig.tight_layout()
    return _to_b64(fig)


# ── Figure 8 — Radar chart per technique ──────────────────────────────────────

def fig_radar(all_results, techniques, model_keys, model_names):
    """One radar per technique showing model profiles across 5 metrics."""
    radar_metrics = ["item_accuracy", "room_accuracy", "room_f1", "room_precision", "room_recall"]
    radar_labels = ["Item Acc", "Room Acc", "F1", "Precision", "Recall"]
    N = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ncols = len(techniques)
    fig = plt.figure(figsize=(ncols * 4.5, 4.5))
    fig.patch.set_facecolor(BG)

    for col, tech in enumerate(techniques):
        ax = fig.add_subplot(1, ncols, col + 1, polar=True)
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT, labelsize=7)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_labels, color=TEXT, fontsize=8)
        ax.set_ylim(0, 1)
        ax.yaxis.set_tick_params(labelsize=6)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"], color="#555", fontsize=5)
        ax.grid(color=GRID, linewidth=0.5)
        ax.spines["polar"].set_edgecolor(GRID)
        ax.set_title(TECHNIQUE_LABELS[tech], color=TEXT, fontsize=9, pad=12)

        for i, mk in enumerate(model_keys):
            vals = [_get_metric(all_results[mk], tech, m) for m in radar_metrics]
            vals += vals[:1]
            color = MODEL_PALETTE[i % len(MODEL_PALETTE)]
            ax.plot(angles, vals, color=color, linewidth=1.5, label=_model_short(model_names[i]).replace("\n", " "))
            ax.fill(angles, vals, color=color, alpha=0.08)

        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
                  facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=6)

    fig.suptitle("Model Profiles by Technique (Radar)", color=TEXT, fontsize=11, y=1.02)
    fig.tight_layout()
    return _to_b64(fig)


# ── Figure 9 — Confusion stacked bar (TP/FP/FN/TN) ───────────────────────────

def fig_confusion(all_results, techniques, model_keys, model_names):
    n_models = len(model_keys)
    n_tech = len(techniques)
    total_cols = n_models * n_tech

    fig, (ax,) = _fig(1, 1, w=max(12, total_cols * 1.2), h=5)
    ax.set_facecolor(PANEL_BG)

    xtick_pos = []
    xtick_labels = []
    colors_map = {"TP": "#3d9", "FP": "#fa0", "FN": "#e54", "TN": "#555"}
    bottoms_dict: dict[str, float] = {}

    bar_data: list[tuple[float, float, float, float, float, str, str]] = []
    x = 0
    for mk, model_name in zip(model_keys, model_names):
        for tech in techniques:
            m = all_results[mk].get(tech, {}).get("metrics", {})
            n = m.get("n_valid", 1) or 1
            tp = m.get("tp", 0) / n
            fp = m.get("fp", 0) / n
            fn = m.get("fn", 0) / n
            tn = m.get("tn", 0) / n
            bar_data.append((x, tp, fp, fn, tn, TECHNIQUE_LABELS[tech], model_name))
            xtick_pos.append(x)
            xtick_labels.append(f"{TECHNIQUE_LABELS[tech][:3]}")
            x += 1
        x += 0.5  # gap between models

    plotted_labels = set()
    for (xpos, tp, fp, fn, tn, t_label, m_label) in bar_data:
        bottom = 0.0
        for key, val in [("TN", tn), ("FP", fp), ("FN", fn), ("TP", tp)]:
            label = key if key not in plotted_labels else "_"
            ax.bar(xpos, val, 0.6, bottom=bottom, color=colors_map[key],
                   label=label, alpha=0.9)
            if val > 0.05:
                ax.text(xpos, bottom + val / 2, f"{key}\n{val:.0%}",
                        ha="center", va="center", fontsize=5.5, color="white")
            bottom += val
            plotted_labels.add(key)

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels, fontsize=6, color=TEXT)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of Images", color=TEXT)
    ax.set_title("Room-Ready Confusion Breakdown (TP/FP/FN/TN) per Model × Technique", color=TEXT, fontsize=10)

    # Model name annotations
    x_cursor = 0
    for mk, model_name in zip(model_keys, model_names):
        center = x_cursor + (n_tech - 1) / 2
        ax.text(center, 1.07, _model_short(model_name).replace("\n", " ")[:18],
                ha="center", va="bottom", fontsize=6, color=ACCENT)
        x_cursor += n_tech + 0.5

    handles = [mpatches.Patch(color=c, label=k) for k, c in colors_map.items()]
    ax.legend(handles=handles, facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8,
              loc="upper right")
    fig.tight_layout()
    return _to_b64(fig)


# ── Figure 10 — Parse error rate ──────────────────────────────────────────────

def fig_parse_errors(all_results, techniques, model_keys, model_names):
    fig, (ax,) = _fig(1, 1, w=max(10, len(model_keys) * 2.5), h=4)
    x = np.arange(len(model_keys))
    n = len(techniques)
    width = 0.7 / n
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    for i, tech in enumerate(techniques):
        rates = []
        for mk in model_keys:
            m = all_results[mk].get(tech, {}).get("metrics", {})
            total = m.get("n_images", 1) or 1
            errs = m.get("parse_errors", 0)
            rates.append(errs / total)
        bars = ax.bar(x + offsets[i], rates, width, label=TECHNIQUE_LABELS[tech],
                      color=TECHNIQUE_COLORS.get(tech, MODEL_PALETTE[i]), alpha=0.88)
        _bar_label(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels([_model_short(n) for n in model_names], fontsize=7)
    ax.set_ylabel("Parse Error Rate", color=TEXT)
    ax.set_ylim(0, min(1.2, max(0.1, 1.1)))
    ax.set_title("Parse / Inference Error Rate by Model & Technique", color=TEXT, fontsize=11)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    return _to_b64(fig)


# ── HTML builder ───────────────────────────────────────────────────────────────

_HTML_STYLE = """
body { font-family: 'Courier New', monospace; background: #0d0d0d; color: #ccc; padding: 2rem; }
h1 { color: #fff; font-size: 1.4rem; letter-spacing: 3px; text-transform: uppercase; }
h2 { color: #888; font-size: .9rem; letter-spacing: 2px; text-transform: uppercase;
     border-bottom: 1px solid #222; padding-bottom: .4rem; margin-top: 2.5rem; }
h3 { color: #58a6ff; font-size: .85rem; letter-spacing: 1px; margin-top: 1.5rem; }
table { border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; font-size: .80rem; }
th { background: #161616; color: #aaa; padding: 7px 12px; text-align: left;
     border-bottom: 1px solid #2a2a2a; }
td { padding: 8px 12px; border-bottom: 1px solid #1a1a1a; vertical-align: top; }
tr:hover td { background: #111; }
.tag  { display:inline-block; background:#1a1a1a; color:#666;
        padding:2px 10px; border-radius:3px; font-size:.75rem; }
.note { font-size:11px; color:#555; margin-top:6px; font-style:italic; }
.fig  { margin: 1.5rem 0; text-align: center; }
.fig img { max-width: 100%; border: 1px solid #222; border-radius: 4px; }
.fig-caption { font-size:11px; color:#555; margin-top:6px; }
.good { color: #3d9; font-weight: 700; }
.warn { color: #fa0; font-weight: 700; }
.bad  { color: #e54; font-weight: 700; }
.tech-direct   { color: #58a6ff; }
.tech-cot      { color: #7ee787; }
.tech-few_shot { color: #f78166; }
"""

TECHNIQUE_CSS = {"direct": "tech-direct", "cot": "tech-cot", "few_shot": "tech-few_shot"}


def _acc_cls(v: float) -> str:
    if v >= 0.9: return "good"
    if v >= 0.6: return "warn"
    return "bad"


def _build_summary_table(all_results, techniques, model_keys, model_names) -> str:
    cols = ["Model", "Technique", "Item Acc", "Room Acc", "Room F1", "Precision", "Recall",
            "Parse Err", "Latency (ms)"]
    th = "".join(f"<th>{c}</th>" for c in cols)
    rows = ""
    for mk, mname in zip(model_keys, model_names):
        for tech in techniques:
            m = all_results[mk].get(tech, {}).get("metrics", {})
            if not m:
                continue
            ia, ra, f1 = m["item_accuracy"], m["room_accuracy"], m["room_f1"]
            rows += (
                f"<tr>"
                f"<td>{mname}</td>"
                f"<td><span class='{TECHNIQUE_CSS.get(tech, '')}'>{TECHNIQUE_LABELS.get(tech, tech)}</span></td>"
                f"<td><span class='{_acc_cls(ia)}'>{ia:.1%}</span></td>"
                f"<td><span class='{_acc_cls(ra)}'>{ra:.1%}</span></td>"
                f"<td><span class='{_acc_cls(f1)}'>{f1:.3f}</span></td>"
                f"<td>{m['room_precision']:.3f}</td>"
                f"<td>{m['room_recall']:.3f}</td>"
                f"<td>{m['parse_errors']}/{m['n_images']}</td>"
                f"<td>{m['mean_latency_ms']:.0f}</td>"
                f"</tr>"
            )
    return f"<table><tr>{th}</tr>{rows}</table>"


def _build_per_item_table(all_results, techniques, model_keys, model_names, checklist) -> str:
    item_ids = [str(c["id"]) for c in checklist]
    header_cells = "<th>Item</th><th>Description</th>"
    for mk, mn in zip(model_keys, model_names):
        for tech in techniques:
            header_cells += f"<th>{mn[:14]}<br><span class='{TECHNIQUE_CSS.get(tech, '')}'>{tech}</span></th>"
    rows = ""
    for item in checklist:
        iid = str(item["id"])
        rows += f"<tr><td>{iid}</td><td style='color:#aaa'>{item['item']}</td>"
        for mk in model_keys:
            for tech in techniques:
                acc = _get_per_item(all_results[mk], tech, iid)
                rows += f"<td><span class='{_acc_cls(acc)}'>{acc:.1%}</span></td>"
        rows += "</tr>"
    return f"<table><tr>{header_cells}</tr>{rows}</table>"


def _fig_block(b64: str | None, caption: str) -> str:
    if b64 is None:
        return ""
    return (
        f'<div class="fig">'
        f'<img src="data:image/png;base64,{b64}" alt="{caption}">'
        f'<div class="fig-caption">{caption}</div>'
        f'</div>'
    )


def build_html(data: dict, figures: dict, timestamp: str) -> str:
    checklist = data["checklist"]
    test_set = data.get("test_set", "—")
    techniques = data.get("techniques", list(TECHNIQUE_LABELS.keys()))
    all_results = data["models"]
    model_keys = list(all_results.keys())
    model_names = [
        next((v.get("model_name") for v in t.values() if "model_name" in v), mk)
        for mk, t in all_results.items()
    ]

    summary_table = _build_summary_table(all_results, techniques, model_keys, model_names)
    item_table = _build_per_item_table(all_results, techniques, model_keys, model_names, checklist)

    tech_desc_rows = "".join(
        f"<tr><td class='{TECHNIQUE_CSS.get(t, '')}'>{TECHNIQUE_LABELS.get(t, t)}</td><td style='color:#aaa'>{_technique_description(t)}</td></tr>"
        for t in techniques
    )

    figs_html = "".join([
        _fig_block(figures.get("item_accuracy"),         "Fig 1 — Item Accuracy by Model & Technique"),
        _fig_block(figures.get("room_accuracy"),         "Fig 2 — Room-Level Accuracy by Model & Technique"),
        _fig_block(figures.get("room_f1"),               "Fig 3 — Room-Ready F1 Score"),
        _fig_block(figures.get("per_item_heatmap"),      "Fig 4 — Per-Item Accuracy Heatmap"),
        _fig_block(figures.get("technique_delta"),       "Fig 5 — Technique Gain vs Direct Baseline"),
        _fig_block(figures.get("latency"),               "Fig 6 — Mean Inference Latency"),
        _fig_block(figures.get("per_model_multiples"),   "Fig 7 — Per-Model Per-Item Breakdown"),
        _fig_block(figures.get("radar"),                 "Fig 8 — Model Profiles by Technique (Radar)"),
        _fig_block(figures.get("confusion"),             "Fig 9 — Room-Ready Confusion Breakdown"),
        _fig_block(figures.get("parse_errors"),          "Fig 10 — Parse / Inference Error Rate"),
    ])

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Prompting Techniques Benchmark — {timestamp}</title>
<style>{_HTML_STYLE}</style>
</head>
<body>
<h1>Prompting Techniques Benchmark</h1>
<p class="tag">{timestamp}</p>
<p class="note">Test set: {test_set} · Binary checklist evaluation · No LLM judge</p>

<h2>Techniques</h2>
<table>
  <tr><th>Technique</th><th>Description</th></tr>
  {tech_desc_rows}
</table>

<h2>Summary Metrics</h2>
{summary_table}

<h2>Per-Item Accuracy</h2>
{item_table}

<h2>Figures</h2>
{figs_html}

</body>
</html>"""


def _technique_description(t: str) -> str:
    return {
        "direct":            "N calls per image (one per item). Each call: 'Is [condition] true? Answer Yes or No.'",
        "cot":               "1 call per image. Model reasons step-by-step through each item, then outputs a Check/Output list.",
        "few_shot":          "1 call per image. 2 reference images (READY + NOT READY) prepended, then full checklist evaluated at once.",
        "few_shot_per_item": "N calls per image (one per item). Each call shows reference images + asks targeted per-item classification.",
    }.get(t, t)


# ── Main ───────────────────────────────────────────────────────────────────────

def merge_model_jsons(paths: list[str]) -> dict:
    """Merge per-model result JSONs (each has one model) into a single combined dict."""
    merged: dict = {}
    for path in paths:
        with open(path) as f:
            d = json.load(f)
        if not merged:
            merged = {k: v for k, v in d.items() if k != "models"}
            merged["models"] = {}
        for mk, mv in d.get("models", {}).items():
            merged["models"][mk] = mv
        # union techniques
        for t in d.get("techniques", []):
            if t not in merged.get("techniques", []):
                merged.setdefault("techniques", []).append(t)
    return merged


def load_latest_results(path: str | None, merge_dir: str | None) -> dict:
    if merge_dir:
        jsons = sorted(glob.glob(str(Path(merge_dir) / "*.json")))
        if not jsons:
            raise FileNotFoundError(f"No JSON files found in {merge_dir}")
        print(f"Merging {len(jsons)} per-model result files from {merge_dir}")
        return merge_model_jsons(jsons)
    if path:
        with open(path) as f:
            return json.load(f)
    # Auto-pick latest single-file result
    candidates = sorted(
        glob.glob(str(RESULTS_DIR / "prompting_techniques_results_*.json")),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No prompting_techniques_results_*.json found in {RESULTS_DIR}. "
            "Run run_benchmark_prompting_techniques.py first."
        )
    print(f"Auto-selected results: {candidates[0]}")
    with open(candidates[0]) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate prompting techniques report")
    parser.add_argument("--results", default=None, help="Path to single results JSON")
    parser.add_argument("--merge-dir", default=None, help="Directory of per-model JSON files to merge")
    parser.add_argument("--save-pngs", action="store_true", help="Also save individual PNG files")
    args = parser.parse_args()

    data = load_latest_results(args.results, getattr(args, "merge_dir", None))
    timestamp = data["timestamp"]
    checklist = data["checklist"]
    techniques = data.get("techniques", ["direct", "cot", "few_shot"])
    all_results = data["models"]
    model_keys = list(all_results.keys())
    model_names = [
        next((v.get("model_name") for v in t.values() if "model_name" in v), mk)
        for mk, t in all_results.items()
    ]

    print(f"Generating report for: {timestamp}")
    print(f"  Models:     {model_keys}")
    print(f"  Techniques: {techniques}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    figures: dict[str, str | None] = {}

    print("  Rendering figures...")

    figures["item_accuracy"]       = fig_item_accuracy(all_results, techniques, model_keys, model_names)
    figures["room_accuracy"]       = fig_room_accuracy(all_results, techniques, model_keys, model_names)
    figures["room_f1"]             = fig_room_f1(all_results, techniques, model_keys, model_names)
    figures["per_item_heatmap"]    = fig_per_item_heatmap(all_results, techniques, model_keys, model_names, checklist)
    figures["technique_delta"]     = fig_technique_delta(all_results, techniques, model_keys, model_names)
    figures["latency"]             = fig_latency(all_results, techniques, model_keys, model_names)
    figures["per_model_multiples"] = fig_per_model_small_multiples(all_results, techniques, model_keys, model_names, checklist)
    figures["radar"]               = fig_radar(all_results, techniques, model_keys, model_names)
    figures["confusion"]           = fig_confusion(all_results, techniques, model_keys, model_names)
    figures["parse_errors"]        = fig_parse_errors(all_results, techniques, model_keys, model_names)

    html = build_html(data, figures, timestamp)
    html_path = RESULTS_DIR / f"prompting_techniques_report_{timestamp}.html"
    html_path.write_text(html)
    print(f"\nHTML report → {html_path}")

    if args.save_pngs:
        png_dir = RESULTS_DIR / f"prompting_techniques_figs_{timestamp}"
        png_dir.mkdir(exist_ok=True)
        for name, b64 in figures.items():
            if b64:
                (png_dir / f"{name}.png").write_bytes(base64.b64decode(b64))
        print(f"PNGs saved → {png_dir}/")


if __name__ == "__main__":
    main()
