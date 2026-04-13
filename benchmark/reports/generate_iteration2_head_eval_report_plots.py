#!/usr/bin/env python
"""
Generate image-heavy plots for iteration2 head/participant evaluation results.

Usage:
  cd benchmark
  python reports/generate_iteration2_head_eval_report_plots.py \
    --results /abs/path/to/iteration2_head_participant_eval_<timestamp>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = ROOT.parent / "data" / "iteration2" / "eval_results" / "head_qwen3vl_4b_iteration2"

PART_COLOR = "#16a34a"
NONPART_COLOR = "#dc2626"
PRED_COLOR = "#0ea5e9"
BG = "#faf8f4"
PANEL = "#fffdf8"


def load_results(path: str | None) -> tuple[dict, Path]:
    if path:
        p = Path(path)
    else:
        matches = sorted(DEFAULT_RESULTS_ROOT.rglob("iteration2_head_participant_eval_*.json"))
        if not matches:
            raise FileNotFoundError("No iteration2_head_participant_eval_*.json found")
        p = matches[-1]
    return json.loads(p.read_text()), p


def ensure_dir(results_path: Path) -> Path:
    out_dir = results_path.parent / "report_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def gt_color(role: str) -> str:
    return PART_COLOR if role == "participant" else NONPART_COLOR


def draw_box(ax, bbox, color, label=None, linestyle="-", linewidth=2.0, alpha=1.0):
    x1, y1, x2, y2 = bbox
    ax.add_patch(
        mpatches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
        )
    )
    if label:
        ax.text(
            x1,
            max(10, y1 - 4),
            label,
            fontsize=7,
            color="white",
            bbox=dict(facecolor=color, edgecolor="none", alpha=0.92, pad=1.5),
        )


def save_summary_overview(data: dict, out_dir: Path) -> str:
    det = data["head_detector"]["metrics"]
    best = data["vlm"]["dilation_results"][data["vlm"]["best_dilation_key"]]["metrics"]
    labels = [
        "Det Precision", "Det Recall", "Det F1", "Det IoU",
        "Cls Acc", "End-to-End", "Part F1",
    ]
    vals = [
        det["precision"], det["recall"], det["f1"], det["mean_matched_iou"],
        best["accuracy_on_valid"], best["end_to_end_accuracy_over_gt"], best["f1_participant"],
    ]
    colors = ["#2563eb", "#059669", "#7c3aed", "#f97316", "#0ea5e9", "#16a34a", "#dc2626"]

    fig, ax = plt.subplots(figsize=(10.5, 5.2), facecolor=BG)
    ax.set_facecolor(PANEL)
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_title("Iteration2 Head Detection + Role Classification Overview")
    ax.tick_params(axis="x", rotation=18)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    p = out_dir / "iteration2_overview_summary.png"
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return str(p)


def save_dilation_sweep(data: dict, out_dir: Path) -> str:
    dil = data["vlm"]["dilation_results"]
    keys = list(dil.keys())
    xs = [dil[k]["dilation"] for k in keys]
    valid_acc = [dil[k]["metrics"]["accuracy_on_valid"] for k in keys]
    e2e = [dil[k]["metrics"]["end_to_end_accuracy_over_gt"] for k in keys]
    f1 = [dil[k]["metrics"]["f1_participant"] for k in keys]
    parse = [dil[k]["metrics"]["parse_errors"] for k in keys]

    fig, ax1 = plt.subplots(figsize=(10.5, 5.2), facecolor=BG)
    ax1.set_facecolor(PANEL)
    ax1.plot(xs, valid_acc, marker="o", linewidth=2.5, color="#2563eb", label="Accuracy on valid")
    ax1.plot(xs, e2e, marker="o", linewidth=2.5, color="#16a34a", label="End-to-end over GT")
    ax1.plot(xs, f1, marker="o", linewidth=2.5, color="#dc2626", label="Participant F1")
    ax1.set_ylim(0, 1.0)
    ax1.set_xlabel("Dilation scale")
    ax1.set_ylabel("Metric value")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax2 = ax1.twinx()
    ax2.bar(xs, parse, width=0.08, alpha=0.15, color="#111827", label="Parse errors")
    ax2.set_ylabel("Parse errors")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
    ax1.set_title("Role Metrics Across Crop Dilations")
    fig.tight_layout()
    p = out_dir / "iteration2_dilation_sweep.png"
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return str(p)


def save_confusion_heatmap(data: dict, out_dir: Path) -> str:
    best = data["vlm"]["dilation_results"][data["vlm"]["best_dilation_key"]]["metrics"]
    mat = np.array([[best["tp"], best["fn"]], [best["fp"], best["tn"]]])
    fig, ax = plt.subplots(figsize=(5.5, 4.8), facecolor=BG)
    ax.set_facecolor(PANEL)
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1], ["Pred participant", "Pred non-participant"])
    ax.set_yticks([0, 1], ["GT participant", "GT non-participant"])
    ax.set_title("Best-Dilation Role Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    fig.tight_layout()
    p = out_dir / "iteration2_best_dilation_confusion.png"
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return str(p)


def save_confusion_breakdown_panel(data: dict, out_dir: Path) -> str:
    best = data["vlm"]["dilation_results"][data["vlm"]["best_dilation_key"]]["metrics"]
    counts = {
        "TP": best["tp"],
        "TN": best["tn"],
        "FP": best["fp"],
        "FN": best["fn"],
    }
    mat = np.array([[counts["TP"], counts["FN"]], [counts["FP"], counts["TN"]]])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), facecolor=BG)
    ax1.set_facecolor(PANEL)
    im = ax1.imshow(mat, cmap="Blues")
    ax1.set_xticks([0, 1], ["Pred participant", "Pred non-participant"])
    ax1.set_yticks([0, 1], ["GT participant", "GT non-participant"])
    ax1.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    ax2.set_facecolor(PANEL)
    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    colors = ["#16a34a", "#2563eb", "#f59e0b", "#dc2626"]
    bars = ax2.bar(labels, values, color=colors)
    ax2.set_title("Confusion Breakdown")
    ax2.set_ylabel("Count")
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, value in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, value + max(0.2, max(values) * 0.02), str(value), ha="center", fontsize=10)

    fig.suptitle("Best-Dilation False Positives / False Negatives / Confusion Summary", fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = out_dir / "iteration2_confusion_breakdown_panel.png"
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return str(p)


def save_per_image_counts(data: dict, out_dir: Path) -> str:
    records = data["head_detector"]["records"]
    x = np.arange(len(records))
    gt = [r["n_gt"] for r in records]
    pred = [r["n_pred"] for r in records]
    matched = [len(r["matches"]) for r in records]
    labels = [Path(r["file_name"]).stem for r in records]

    fig, ax = plt.subplots(figsize=(13.5, 5.5), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.bar(x - 0.22, gt, 0.22, label="GT", color="#94a3b8")
    ax.bar(x, pred, 0.22, label="Pred", color="#0ea5e9")
    ax.bar(x + 0.22, matched, 0.22, label="Matched", color="#16a34a")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_title("Per-Image Detection Counts")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    p = out_dir / "iteration2_per_image_detection_counts.png"
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return str(p)


def save_iou_vs_correct(data: dict, out_dir: Path) -> str:
    best = data["vlm"]["dilation_results"][data["vlm"]["best_dilation_key"]]["records"]
    xs = [r["iou"] for r in best]
    ys = [r["dilation"] for r in best]
    colors = [PART_COLOR if r["correct"] else NONPART_COLOR for r in best]
    sizes = [40 + 180 * max(0.0, r["confidence"]) for r in best]

    fig, ax = plt.subplots(figsize=(8.8, 5.5), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.scatter(xs, ys, s=sizes, c=colors, alpha=0.7, edgecolors="white", linewidths=0.6)
    ax.set_xlabel("Matched IoU")
    ax.set_ylabel("Crop dilation")
    ax.set_title("Matched IoU vs Classification Outcome (Best Dilation Records)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ok_patch = mpatches.Patch(color=PART_COLOR, label="Correct classification")
    bad_patch = mpatches.Patch(color=NONPART_COLOR, label="Incorrect classification")
    ax.legend(handles=[ok_patch, bad_patch], fontsize=8, loc="lower right")
    fig.tight_layout()
    p = out_dir / "iteration2_iou_vs_correctness.png"
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return str(p)


def save_detection_overlay_sheet(data: dict, out_dir: Path) -> str:
    records = data["head_detector"]["records"]
    n = len(records)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 4.2), facecolor=BG)
    axes = np.array(axes).reshape(-1)

    for ax, rec in zip(axes, records):
        img = Image.open(rec["image_path"]).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
        matched_pred_idxs = {m["pred_idx"] for m in rec["matches"]}
        matched_gt_idxs = {m["gt_idx"] for m in rec["matches"]}

        for idx, match in enumerate(rec["matches"]):
            draw_box(ax, match["gt_bbox"], gt_color(match["gt_role"]), label=f"GT {match['gt_role'][0].upper()}{idx+1}")
            draw_box(ax, match["pred_bbox"], PRED_COLOR, label=f"Pred {idx+1}", linestyle="--", linewidth=1.8)

        unmatched_gt = []
        # Rebuild GT view from matches only if present in file via match indices
        # We use counts in title for quick scan.
        for pred_idx, pred in enumerate(rec["predictions"]):
            if pred_idx not in matched_pred_idxs:
                draw_box(ax, pred["bbox"], "#f59e0b", label="Unmatched pred", linestyle=":", linewidth=1.6, alpha=0.9)

        ax.set_title(
            f"{rec['file_name']}\nGT={rec['n_gt']} Pred={rec['n_pred']} Matched={len(rec['matches'])}",
            fontsize=9,
        )

    for ax in axes[len(records):]:
        ax.axis("off")
    fig.suptitle("Full-Image Detection Overlays (Matched GT + Predicted Head Boxes)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    p = out_dir / "iteration2_detection_overlay_sheet.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return str(p)


def save_best_dilation_overlay_sheet(data: dict, out_dir: Path) -> str:
    best_key = data["vlm"]["best_dilation_key"]
    records = data["vlm"]["dilation_results"][best_key]["records"]
    by_image = {}
    for rec in records:
        by_image.setdefault(rec["file_name"], []).append(rec)

    items = sorted(by_image.items())
    cols = 2
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 4.4), facecolor=BG)
    axes = np.array(axes).reshape(-1)

    for ax, (file_name, recs) in zip(axes, items):
        img = Image.open(recs[0]["image_path"]).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
        correct = sum(1 for r in recs if r["correct"])
        for rec in recs:
            color = PART_COLOR if rec["predicted_participant"] else NONPART_COLOR if rec["predicted_participant"] is False else "#a855f7"
            role_short = "P" if rec["gt_role"] == "participant" else "NP"
            pred_short = "P" if rec["predicted_participant"] is True else "NP" if rec["predicted_participant"] is False else "?"
            draw_box(
                ax,
                rec["dilated_bbox"],
                color,
                label=f"{rec['person_id']} gt:{role_short} pred:{pred_short}",
                linewidth=2.0,
            )
        ax.set_title(f"{file_name}\nCorrect {correct}/{len(recs)}", fontsize=9)

    for ax in axes[len(items):]:
        ax.axis("off")
    fig.suptitle("Best-Dilation Classification Overlays", fontsize=14, fontweight="bold")
    fig.tight_layout()
    p = out_dir / "iteration2_best_dilation_overlay_sheet.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return str(p)


def save_crop_gallery(data: dict, out_dir: Path) -> str | None:
    best_key = data["vlm"]["best_dilation_key"]
    records = data["vlm"]["dilation_results"][best_key]["records"]
    incorrect = [r for r in records if not r["correct"] and Path(r["crop_path"]).exists()][:6]
    correct = [r for r in records if r["correct"] and Path(r["crop_path"]).exists()][:6]
    chosen = (incorrect + correct)[:12]
    if not chosen:
        return None

    cols = 4
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.6), facecolor=BG)
    axes = np.array(axes).reshape(-1)
    for ax, rec in zip(axes, chosen):
        img = Image.open(rec["crop_path"]).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
        color = PART_COLOR if rec["correct"] else NONPART_COLOR
        pred_name = "participant" if rec["predicted_participant"] is True else "non-participant" if rec["predicted_participant"] is False else "unparsed"
        title = (
            f"{rec['file_name']} {rec['person_id']}\n"
            f"gt={rec['gt_role']} pred={pred_name}\n"
            f"IoU={rec['iou']:.2f} conf={rec['confidence']:.2f}"
        )
        ax.set_title(title, fontsize=8, color=color)
    for ax in axes[len(chosen):]:
        ax.axis("off")
    fig.suptitle("Best-Dilation Crop Gallery (Incorrect first, then correct)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    p = out_dir / "iteration2_best_dilation_crop_gallery.png"
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return str(p)


def save_family_breakdown(data: dict, out_dir: Path) -> str:
    best_key = data["vlm"]["best_dilation_key"]
    records = data["vlm"]["dilation_results"][best_key]["records"]
    groups = {"all_participants": [], "non_participants": []}
    for rec in records:
        key = "all_participants" if rec["file_name"].startswith("all_participants") else "non_participants"
        groups[key].append(rec)

    labels = []
    accs = []
    part_accs = []
    nonpart_accs = []
    for name, recs in groups.items():
        labels.append(name)
        accs.append(sum(1 for r in recs if r["correct"]) / len(recs) if recs else 0.0)
        part = [r for r in recs if r["gt_role"] == "participant"]
        nonpart = [r for r in recs if r["gt_role"] == "non-participant"]
        part_accs.append(sum(1 for r in part if r["correct"]) / len(part) if part else 0.0)
        nonpart_accs.append(sum(1 for r in nonpart if r["correct"]) / len(nonpart) if nonpart else 0.0)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.bar(x - 0.22, accs, 0.22, label="Overall", color="#0ea5e9")
    ax.bar(x, part_accs, 0.22, label="Participant", color=PART_COLOR)
    ax.bar(x + 0.22, nonpart_accs, 0.22, label="Non-participant", color=NONPART_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_title("Best-Dilation Accuracy by Image Family")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "iteration2_family_breakdown.png"
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return str(p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate rich plots for iteration2 head participant eval.")
    parser.add_argument("--results", default=None, help="Path to iteration2_head_participant_eval_<timestamp>.json")
    args = parser.parse_args()

    data, results_path = load_results(args.results)
    out_dir = ensure_dir(results_path)

    saved = [
        save_summary_overview(data, out_dir),
        save_dilation_sweep(data, out_dir),
        save_confusion_heatmap(data, out_dir),
        save_confusion_breakdown_panel(data, out_dir),
        save_per_image_counts(data, out_dir),
        save_iou_vs_correct(data, out_dir),
        save_detection_overlay_sheet(data, out_dir),
        save_best_dilation_overlay_sheet(data, out_dir),
        save_family_breakdown(data, out_dir),
    ]
    crop_gallery = save_crop_gallery(data, out_dir)
    if crop_gallery:
        saved.append(crop_gallery)

    print(f"Results JSON: {results_path}")
    print(f"Saved {len(saved)} plots to {out_dir}")
    for path in saved:
        print(f"Plot -> {path}")


if __name__ == "__main__":
    main()
