#!/usr/bin/env python3
"""
generate_binary_figures.py — Save individual figures from the binary benchmark results:
  - heatmap.png
  - bar_chart.png
  - prompts_table.png

Usage:
    cd benchmark
    python generate_binary_figures.py
    python generate_binary_figures.py --out results/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from run_benchmark_env_monitoring_binary import QUESTIONS
from generate_binary_report import latest_per_model

RESULTS_DIR = Path(__file__).parent.parent / "results"

CAT_COLORS = {
    "blinds":     "#3b82f6",
    "chairs":     "#8b5cf6",
    "table":      "#f97316",
    "whiteboard": "#10b981",
}

def acc_color(acc: float) -> str:
    if acc >= 0.9: return "#16a34a"
    if acc >= 0.7: return "#65a30d"
    if acc >= 0.5: return "#d97706"
    return "#dc2626"

def heat_color(acc: float) -> str:
    if acc >= 0.9: return "#bbf7d0"
    if acc >= 0.8: return "#d9f99d"
    if acc >= 0.7: return "#fef08a"
    if acc >= 0.6: return "#fed7aa"
    return "#fecaca"

def heat_text(acc: float) -> str:
    if acc >= 0.7: return "#14532d"
    if acc >= 0.6: return "#713f12"
    return "#7f1d1d"


def save_heatmap(all_results: dict, out: Path) -> None:
    categories = sorted({
        ct for d in all_results.values()
        for ct in d["metrics"]["per_change_type"]
    })
    model_keys = sorted(all_results, key=lambda k: -all_results[k]["metrics"]["accuracy"])
    model_names = [all_results[k]["model_name"] for k in model_keys]
    cols = categories + ["overall"]

    fig, ax = plt.subplots(figsize=(len(cols) * 1.4 + 1, len(model_keys) * 0.7 + 1))
    ax.set_xlim(0, len(cols))
    ax.set_ylim(0, len(model_keys))
    ax.axis("off")
    fig.patch.set_facecolor("#f9fafb")

    cell_w, cell_h = 1.0, 1.0
    for row_i, key in enumerate(model_keys):
        m = all_results[key]["metrics"]
        y = len(model_keys) - row_i - 1
        for col_i, ct in enumerate(cols):
            acc = m["accuracy"] if ct == "overall" else m["per_change_type"].get(ct, {}).get("accuracy", 0)
            correct = "" if ct == "overall" else f"{m['per_change_type'].get(ct,{}).get('correct',0)}/{m['per_change_type'].get(ct,{}).get('total',0)}"
            bg = heat_color(acc)
            fg = heat_text(acc)
            rect = mpatches.FancyBboxPatch(
                (col_i + 0.04, y + 0.06), cell_w - 0.08, cell_h - 0.12,
                boxstyle="round,pad=0.02", facecolor=bg, edgecolor="white", linewidth=1.5
            )
            ax.add_patch(rect)
            ax.text(col_i + 0.5, y + 0.58, f"{acc:.0%}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=fg)
            if correct:
                ax.text(col_i + 0.5, y + 0.25, correct, ha="center", va="center",
                        fontsize=7.5, color=fg, alpha=0.6)

    # column headers
    for col_i, ct in enumerate(cols):
        color = CAT_COLORS.get(ct, "#374151")
        ax.text(col_i + 0.5, len(model_keys) + 0.15, ct,
                ha="center", va="bottom", fontsize=10, fontweight="600", color=color)

    # row labels
    for row_i, name in enumerate(model_names):
        y = len(model_keys) - row_i - 1
        ax.text(-0.1, y + 0.5, name, ha="right", va="center", fontsize=9.5, color="#111827")

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  heatmap     → {out}")


def save_bar_chart(all_results: dict, out: Path) -> None:
    categories = sorted({
        ct for d in all_results.values()
        for ct in d["metrics"]["per_change_type"]
    })
    model_keys = sorted(all_results, key=lambda k: -all_results[k]["metrics"]["accuracy"])
    model_names = [all_results[k]["model_name"] for k in model_keys]

    x = np.arange(len(model_keys))
    all_cats = ["overall"] + categories
    width = 0.8 / len(all_cats)

    fig, ax = plt.subplots(figsize=(max(10, len(model_keys) * 1.6), 5))
    fig.patch.set_facecolor("#f9fafb")
    ax.set_facecolor("#ffffff")

    for i, ct in enumerate(all_cats):
        vals = [
            all_results[k]["metrics"]["accuracy"] * 100 if ct == "overall"
            else all_results[k]["metrics"]["per_change_type"].get(ct, {}).get("accuracy", 0) * 100
            for k in model_keys
        ]
        color = CAT_COLORS.get(ct, "#374151")
        alpha = 0.85 if ct != "overall" else 0.3
        edge = color
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                      color=color, alpha=alpha, edgecolor=edge, linewidth=1.2, label=ct)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=9, color="#374151")
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
    ax.tick_params(axis="y", colors="#9ca3af", labelsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#e5e7eb")
    ax.yaxis.grid(True, color="#f3f4f6", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, framealpha=0, labelcolor="#374151")

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  bar_chart   → {out}")


def save_prompts_table(out: Path) -> None:
    import textwrap

    categories = sorted(QUESTIONS.keys())
    WRAP = 80  # chars per line for the prompt column

    # Pre-wrap each prompt and count lines
    rows = []
    for ct in categories:
        q = QUESTIONS[ct]
        lines = [l.strip() for l in q["question"].strip().split("\n") if l.strip()]
        wrapped = "\n".join(
            textwrap.fill(l, width=WRAP) for l in lines
        )
        n_lines = wrapped.count("\n") + 1
        rows.append((ct, wrapped, q["yes_means"], n_lines))

    LINE_H = 0.22   # inches per text line
    ROW_PAD = 0.25  # inches padding per row
    HEADER_H = 0.5
    total_h = HEADER_H + sum(r[3] * LINE_H + ROW_PAD for r in rows)

    fig = plt.figure(figsize=(13, total_h))
    fig.patch.set_facecolor("#f9fafb")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    col_x = {"cat": 0.02, "prompt": 0.14, "yes": 0.93}

    # Header (drawn at top in data coords)
    y_cur = total_h - 0.15
    for label, cx in [("CATEGORY", col_x["cat"]), ("PROMPT", col_x["prompt"]), ("YES →", col_x["yes"])]:
        ax.text(cx, y_cur, label, fontsize=8, fontweight="700", color="#6b7280",
                va="top", fontfamily="monospace")
    y_cur -= 0.2
    ax.plot([0.01, 0.99], [y_cur, y_cur], color="#e5e7eb", linewidth=1)
    y_cur -= 0.1

    for i, (ct, prompt, yes_means, n_lines) in enumerate(rows):
        row_h = n_lines * LINE_H + ROW_PAD
        y_mid = y_cur - row_h / 2

        color = CAT_COLORS.get(ct, "#374151")
        ax.text(col_x["cat"], y_mid, ct, fontsize=11, fontweight="700",
                color=color, va="center")
        ax.text(col_x["prompt"], y_cur - 0.05, prompt, fontsize=8,
                color="#374151", va="top", fontfamily="monospace",
                linespacing=1.5)
        label_color = "#16a34a" if yes_means == "clean" else "#dc2626"
        ax.text(col_x["yes"], y_mid, yes_means, fontsize=9, fontweight="600",
                color=label_color, va="center")

        y_cur -= row_h
        if i < len(rows) - 1:
            ax.plot([0.01, 0.99], [y_cur, y_cur], color="#f3f4f6", linewidth=0.8)

    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  prompts     → {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else RESULTS_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = latest_per_model(RESULTS_DIR)
    if not all_results:
        print("No results found.")
        return

    print(f"Saving figures to {out_dir}/")
    save_heatmap(all_results, out_dir / "heatmap.png")
    save_bar_chart(all_results, out_dir / "bar_chart.png")
    save_prompts_table(out_dir / "prompts_table.png")


if __name__ == "__main__":
    main()
