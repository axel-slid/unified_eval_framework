#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ROOT.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ROOT))

from benchmark.runs.run_meeting_participant_bbox_eval import (
    YOLOv8HeadModel,
    box_iou,
    load_dataset,
)
from models.yolov11 import YOLOv11Model
from models.yolov8_face import YOLOv8FaceModel


BG = "#ffffff"
PANEL_BG = "#fbfbf9"
TEXT = "#202124"
GRID = "#d9d4cb"
MODEL_COLORS = {
    "face": "#e76f51",
    "head": "#f4a261",
    "person": "#2a9d8f",
}
MODEL_LABELS = {
    "face": "Face",
    "head": "Head",
    "person": "Person",
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
    files = sorted(glob.glob(str(RESULTS_DIR / "meeting_participant_bbox_eval_*.json")))
    if not files:
        raise FileNotFoundError("No meeting participant bbox eval results found.")
    return Path(files[-1])


def style_ax(ax, title: str, ylabel: str | None = None) -> None:
    ax.set_title(title, fontsize=11, pad=8, color=TEXT)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=TEXT)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)


def load_results(path: Path) -> dict:
    return json.loads(path.read_text())


def compute_best_iou_distributions(conf_threshold: float) -> dict[str, dict[str, list[float]]]:
    items = load_dataset()
    model_builders = {
        "face": lambda: YOLOv8FaceModel(device="cpu"),
        "head": lambda: YOLOv8HeadModel(device="cpu"),
        "person": lambda: YOLOv11Model(variant="small", device="cpu"),
    }
    out: dict[str, dict[str, list[float]]] = {}

    for model_key, builder in model_builders.items():
        model = builder()
        model.load()
        all_best: list[float] = []
        part_best: list[float] = []
        nonpart_best: list[float] = []

        for item in items:
            result = model.detect(str(item["image_path"]), conf_threshold=conf_threshold)
            preds = [det.bbox for det in result.detections] if not result.error else []
            for gt in item["gts"]:
                best_iou = max((box_iou(pred, gt["bbox"]) for pred in preds), default=0.0)
                all_best.append(best_iou)
                if gt["role"] == "participant":
                    part_best.append(best_iou)
                else:
                    nonpart_best.append(best_iou)

        model.unload()
        out[model_key] = {
            "all": all_best,
            "participant": part_best,
            "non-participant": nonpart_best,
        }

    return out


def figure_summary(results: dict, out_path: Path) -> None:
    models = ["face", "head", "person"]
    x = np.arange(len(models))
    w = 0.24

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), facecolor=BG)

    precision = [results["models"][m]["bbox_precision"] for m in models]
    recall = [results["models"][m]["bbox_recall"] for m in models]
    f1 = [results["models"][m]["bbox_f1"] for m in models]
    ax = axes[0]
    style_ax(ax, "Bbox Match Metrics", "Score")
    bars1 = ax.bar(x - w, precision, w, label="Precision", color="#457b9d")
    bars2 = ax.bar(x, recall, w, label="Recall", color="#2a9d8f")
    bars3 = ax.bar(x + w, f1, w, label="F1", color="#264653")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, loc="upper left")
    for bars in (bars1, bars2, bars3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    style_ax(ax, "Matched IoU + Latency", "Value")
    mean_iou = [results["models"][m]["mean_matched_iou"] for m in models]
    latency = [results["models"][m]["mean_latency_ms"] / 150.0 for m in models]
    bars1 = ax.bar(x - w / 2, mean_iou, w, label="Mean matched IoU", color="#e9c46a")
    bars2 = ax.bar(x + w / 2, latency, w, label="Latency / 150", color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax.set_ylim(0, max(1.0, max(latency + mean_iou) * 1.25))
    ax.legend(fontsize=8, loc="upper left")
    for bar, val in zip(bars1, mean_iou):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, [results["models"][m]["mean_latency_ms"] for m in models]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.0f}ms", ha="center", va="bottom", fontsize=8)

    ax = axes[2]
    style_ax(ax, "Participant vs Non-Participant Recall", "Recall")
    part = [results["models"][m]["participant_breakdown"]["participant"]["recall"] for m in models]
    nonpart = [results["models"][m]["participant_breakdown"]["non-participant"]["recall"] for m in models]
    bars1 = ax.bar(x - w / 2, part, w, label="Participant", color="#2a9d8f")
    bars2 = ax.bar(x + w / 2, nonpart, w, label="Non-participant", color="#a8dadc")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, loc="upper left")
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Meeting Participant Dataset: Detector Comparison", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def figure_overlap(best_ious: dict[str, dict[str, list[float]]], out_path: Path) -> None:
    models = ["face", "head", "person"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), facecolor=BG)

    ax = axes[0]
    style_ax(ax, "Best IoU to Any GT Person Box", "Best IoU")
    data = [best_ious[m]["all"] for m in models]
    bp = ax.boxplot(
        data,
        patch_artist=True,
        tick_labels=[MODEL_LABELS[m] for m in models],
        medianprops={"color": "#1f2937", "linewidth": 1.6},
        whiskerprops={"color": GRID},
        capprops={"color": GRID},
    )
    for patch, model_key in zip(bp["boxes"], models):
        patch.set_facecolor(MODEL_COLORS[model_key])
        patch.set_alpha(0.6)
        patch.set_edgecolor(MODEL_COLORS[model_key])
    ax.set_ylim(0, 1.0)
    for idx, model_key in enumerate(models, start=1):
        vals = best_ious[model_key]["all"]
        ax.text(idx, min(0.95, max(vals) + 0.05), f"mean {statistics.mean(vals):.2f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    style_ax(ax, "CDF of Best IoU", "Fraction of GT boxes")
    for model_key in models:
        vals = np.sort(np.array(best_ious[model_key]["all"]))
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, y, label=MODEL_LABELS[model_key], linewidth=2.2, color=MODEL_COLORS[model_key])
    ax.axvline(0.5, color="#6b7280", linestyle="--", linewidth=1.2)
    ax.text(0.505, 0.08, "IoU=0.5", fontsize=8, color="#6b7280")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Overlap Against Person-Sized Ground Truth", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots for meeting participant bbox evaluation.")
    parser.add_argument("--results", default=None, help="Path to meeting_participant_bbox_eval_*.json")
    parser.add_argument("--out-dir", default=str(FIGURES_DIR))
    args = parser.parse_args()

    results_path = Path(args.results) if args.results else latest_results_json()
    results = load_results(results_path)
    conf_threshold = results["models"]["person"]["confidence_threshold"]
    best_ious = compute_best_iou_distributions(conf_threshold=conf_threshold)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "meeting_participant_detector_summary.png"
    overlap_path = out_dir / "meeting_participant_detector_overlap.png"

    figure_summary(results, summary_path)
    figure_overlap(best_ious, overlap_path)

    print(f"Saved: {summary_path}")
    print(f"Saved: {overlap_path}")


if __name__ == "__main__":
    main()
