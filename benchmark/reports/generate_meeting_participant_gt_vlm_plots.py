#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

BG = "#ffffff"
PANEL_BG = "#fbfbf9"
TEXT = "#202124"
GRID = "#d9d4cb"
COLORS = {
    "accuracy": "#2a9d8f",
    "f1": "#264653",
    "precision": "#457b9d",
    "recall": "#e9c46a",
    "participant": "#2a9d8f",
    "non_participant": "#a8dadc",
    "tp": "#2a9d8f",
    "tn": "#577590",
    "fp": "#e76f51",
    "fn": "#f4a261",
}

LABELS = {
    "qwen3vl_4b": "Qwen3-VL 4B",
    "qwen3vl_8b": "Qwen3-VL 8B",
    "qwen3vl_4b_int8": "Qwen3-VL 4B int8",
    "qwen3vl_8b_int8": "Qwen3-VL 8B int8",
    "gemma_e2b_8bit_hf": "Gemma E2B 8-bit",
    "gemma_e4b_4bit": "Gemma E4B 4-bit",
}

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL_BG,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "text.color": TEXT,
    "grid.color": GRID,
    "legend.facecolor": "white",
    "legend.edgecolor": GRID,
    "font.family": "monospace",
})


def latest_results_json() -> Path:
    files = sorted(glob.glob(str(RESULTS_DIR / "meeting_participant_gt_vlm_eval_*.json")))
    if not files:
        raise FileNotFoundError("No meeting_participant_gt_vlm_eval_*.json found in benchmark/results.")
    return Path(files[-1])


def load_results(path: Path) -> dict:
    return json.loads(path.read_text())


def style_ax(ax, title: str, ylabel: str = "") -> None:
    ax.set_title(title, fontsize=11, pad=8, color=TEXT)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=TEXT)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)


def ordered_models(data: dict) -> list[str]:
    keys = list(data["models"].keys())
    preferred = [
        "qwen3vl_4b",
        "qwen3vl_8b",
        "qwen3vl_4b_int8",
        "qwen3vl_8b_int8",
        "gemma_e2b_8bit_hf",
        "gemma_e4b_4bit",
    ]
    ordered = [k for k in preferred if k in keys]
    ordered.extend([k for k in keys if k not in ordered])
    return ordered


def label_for(model_key: str) -> str:
    return LABELS.get(model_key, model_key)


def fig_summary(data: dict, out: Path) -> None:
    models = ordered_models(data)
    labels = [label_for(m) for m in models]
    x = np.arange(len(models))
    w = 0.2

    acc = [data["models"][m]["metrics"]["accuracy"] for m in models]
    f1 = [data["models"][m]["metrics"]["f1_participant"] for m in models]
    prec = [data["models"][m]["metrics"]["precision_participant"] for m in models]
    rec = [data["models"][m]["metrics"]["recall_participant"] for m in models]

    fig, ax = plt.subplots(figsize=(12, 5.2), facecolor=BG)
    style_ax(ax, "GT-Box Participant Classification Metrics", "Score")
    bars1 = ax.bar(x - 1.5 * w, acc, w, color=COLORS["accuracy"], label="Accuracy")
    bars2 = ax.bar(x - 0.5 * w, f1, w, color=COLORS["f1"], label="F1")
    bars3 = ax.bar(x + 0.5 * w, prec, w, color=COLORS["precision"], label="Precision")
    bars4 = ax.bar(x + 1.5 * w, rec, w, color=COLORS["recall"], label="Recall")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, ncol=4, loc="upper left")
    for bars in (bars1, bars2, bars3, bars4):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def fig_confusion(data: dict, out: Path) -> None:
    models = ordered_models(data)
    labels = [label_for(m) for m in models]
    x = np.arange(len(models))
    w = 0.18

    tp = [data["models"][m]["metrics"]["tp"] for m in models]
    tn = [data["models"][m]["metrics"]["tn"] for m in models]
    fp = [data["models"][m]["metrics"]["fp"] for m in models]
    fn = [data["models"][m]["metrics"]["fn"] for m in models]

    fig, ax = plt.subplots(figsize=(12, 5.2), facecolor=BG)
    style_ax(ax, "Confusion Counts on GT Person Crops", "Count")
    bars1 = ax.bar(x - 1.5 * w, tp, w, color=COLORS["tp"], label="TP")
    bars2 = ax.bar(x - 0.5 * w, tn, w, color=COLORS["tn"], label="TN")
    bars3 = ax.bar(x + 0.5 * w, fp, w, color=COLORS["fp"], label="FP")
    bars4 = ax.bar(x + 1.5 * w, fn, w, color=COLORS["fn"], label="FN")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.legend(fontsize=8, ncol=4, loc="upper left")
    ymax = max(tp + tn + fp + fn + [1])
    ax.set_ylim(0, ymax * 1.2)
    for bars in (bars1, bars2, bars3, bars4):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.25, f"{int(h)}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def fig_role_accuracy(data: dict, out: Path) -> None:
    models = ordered_models(data)
    labels = [label_for(m) for m in models]
    x = np.arange(len(models))
    w = 0.34

    part = [data["models"][m]["metrics"]["participant_breakdown"]["participant"]["accuracy"] for m in models]
    nonpart = [data["models"][m]["metrics"]["participant_breakdown"]["non-participant"]["accuracy"] for m in models]

    fig, ax = plt.subplots(figsize=(12, 5.2), facecolor=BG)
    style_ax(ax, "Role-Wise Accuracy", "Accuracy")
    bars1 = ax.bar(x - w / 2, part, w, color=COLORS["participant"], label="Participant")
    bars2 = ax.bar(x + w / 2, nonpart, w, color=COLORS["non_participant"], label="Non-participant")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, loc="upper left")
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def fig_latency(data: dict, out: Path) -> None:
    models = ordered_models(data)
    labels = [label_for(m) for m in models]
    lat = [data["models"][m]["metrics"]["mean_latency_ms"] for m in models]

    fig, ax = plt.subplots(figsize=(12, 4.8), facecolor=BG)
    style_ax(ax, "Mean Per-Crop Latency", "Latency (ms)")
    bars = ax.bar(labels, lat, color="#8ecae6")
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ymax = max(lat + [1.0])
    ax.set_ylim(0, ymax * 1.2)
    for bar, val in zip(bars, lat):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.02, f"{val:.0f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PNG plots from meeting_participant_gt_vlm_eval results.")
    parser.add_argument("--results", default=None, help="Path to meeting_participant_gt_vlm_eval_*.json")
    parser.add_argument("--out-dir", default=str(FIGURES_DIR))
    args = parser.parse_args()

    results_path = Path(args.results) if args.results else latest_results_json()
    data = load_results(results_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = results_path.stem
    summary = out_dir / f"{stem}_summary.png"
    confusion = out_dir / f"{stem}_confusion.png"
    role_acc = out_dir / f"{stem}_role_accuracy.png"
    latency = out_dir / f"{stem}_latency.png"

    fig_summary(data, summary)
    fig_confusion(data, confusion)
    fig_role_accuracy(data, role_acc)
    fig_latency(data, latency)

    print(f"Saved: {summary}")
    print(f"Saved: {confusion}")
    print(f"Saved: {role_acc}")
    print(f"Saved: {latency}")


if __name__ == "__main__":
    main()
