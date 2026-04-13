#!/usr/bin/env python
"""
run_qwen3vl_4b_int8_env_monitoring_prompt_sweep.py

Evaluate environment-monitoring binary metrics using only Qwen3-VL-4B int8, while
trying 5 different prompting techniques for the chairs category.

Categories:
  - table
  - blinds
  - whiteboard
  - chairs

For non-chair categories, the script uses one fixed prompt.
For chairs, it runs five prompt variants and reports:
  - chair-only metrics per variant
  - full-dataset metrics when that chair variant is combined with the fixed prompts

Usage:
    cd benchmark
    python runs/run_qwen3vl_4b_int8_env_monitoring_prompt_sweep.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Set GPU visibility before importing model modules that may import torch.
GPU_ID = os.environ.get("GPU_ID")
if GPU_ID and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import GenerationConfig, ModelConfig, load_config
from models import MODEL_REGISTRY


ROOT = Path(__file__).resolve().parent.parent


def resolve_dataset_csv() -> Path:
    candidates = [
        ROOT / "environment_monitoring_dataset" / "unified_annotations.csv",
        ROOT.parent / "environment_monitoring_dataset" / "unified_annotations.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find environment_monitoring_dataset/unified_annotations.csv "
        f"in any expected location: {candidates}"
    )


DATASET_CSV = resolve_dataset_csv()
FIGURES_SUBDIR = "figures"


FIXED_PROMPTS: dict[str, dict] = {
    "table": {
        "question": (
            "Look carefully at the table surface in this image.\n"
            "Ignore permanently installed equipment only.\n"
            "Is there anything left on the table surface such as laptops, cups, bottles, papers, bags, chargers, or personal items?\n"
            "End your response with a final line containing only: yes or no"
        ),
        "yes_means": "messy",
    },
    "blinds": {
        "question": (
            "Look carefully at the windows and any coverings in this image.\n"
            "Are the blinds or shades open so natural light is coming in?\n"
            "End your response with a final line containing only: yes or no"
        ),
        "yes_means": "clean",
    },
    "whiteboard": {
        "question": (
            "Look carefully at the whiteboard or writable wall surface in this image.\n"
            "Are there any visible marks, writing, diagrams, residue, or drawings on it?\n"
            "End your response with a final line containing only: yes or no"
        ),
        "yes_means": "messy",
    },
}


CHAIR_PROMPTS: list[dict] = [
    {
        "key": "chair_direct_binary",
        "label": "Direct Binary",
        "question": (
            "Are all visible chairs fully tucked into the table in a neat meeting-ready arrangement?\n"
            "End your response with a final line containing only: yes or no"
        ),
        "yes_means": "clean",
    },
    {
        "key": "chair_step_by_step",
        "label": "Step by Step",
        "question": (
            "Look at every chair around the table.\n"
            "Think step by step about whether each chair is pushed in or clearly left out of place.\n"
            "If any chair is noticeably pulled out, turned away, isolated, or abandoned away from the table, answer no.\n"
            "End your response with a final line containing only: yes or no"
        ),
        "yes_means": "clean",
    },
    {
        "key": "chair_meeting_ready_rule",
        "label": "Meeting-Ready Rule",
        "question": (
            "Decide whether this room is meeting-ready with respect to chairs only.\n"
            "A yes answer means the chairs are generally arranged around the table and pushed in.\n"
            "A no answer means at least one chair is clearly displaced, pulled out, sideways, or away from the table.\n"
            "End your response with a final line containing only: yes or no"
        ),
        "yes_means": "clean",
    },
    {
        "key": "chair_count_and_verify",
        "label": "Count and Verify",
        "question": (
            "First inspect the chairs one by one around the table.\n"
            "Verify whether every visible chair is positioned as if the room were reset after a meeting.\n"
            "If even one visible chair is obviously not pushed in or is left away from the table, answer no.\n"
            "End your response with a final line containing only: yes or no"
        ),
        "yes_means": "clean",
    },
    {
        "key": "chair_strict_negative",
        "label": "Strict Negative Test",
        "question": (
            "Answer yes only if there is no evidence of any chair being left out.\n"
            "If you can see a chair that is pulled back, rotated away, separated from the table, or not reset, answer no.\n"
            "End your response with a final line containing only: yes or no"
        ),
        "yes_means": "clean",
    },
    {
        "key": "chair_single_failure_rule",
        "label": "Single Failure Rule",
        "question": (
            "Check whether every visible chair is tucked in.\n"
            "If even a single chair is not tucked in, answer no.\n"
            "Answer yes only if all visible chairs are tucked in.\n"
            "End your response with a final line containing only: yes or no"
        ),
        "yes_means": "clean",
    },
    {
        "key": "chair_zero_tolerance",
        "label": "Zero Tolerance",
        "question": (
            "Use a zero-tolerance rule for chair placement.\n"
            "If one or more chairs are even clearly slightly left out from the table, answer no.\n"
            "Only answer yes if none of the visible chairs are left out.\n"
            "End your response with a final line containing only: yes or no"
        ),
        "yes_means": "clean",
    },
    {
        "key": "chair_exception_scan",
        "label": "Exception Scan",
        "question": (
            "Scan the room for any exception to a fully reset chair layout.\n"
            "A single chair that is pulled out, angled away, or detached from the table means the answer is no.\n"
            "If there are no such exceptions, answer yes.\n"
            "End your response with a final line containing only: yes or no"
        ),
        "yes_means": "clean",
    },
    {
        "key": "chair_all_or_nothing",
        "label": "All or Nothing",
        "question": (
            "This is an all-or-nothing check.\n"
            "Answer yes only if every visible chair is tucked in around the table.\n"
            "If even one visible chair is not tucked in, answer no.\n"
            "End your response with a final line containing only: yes or no"
        ),
        "yes_means": "clean",
    },
    {
        "key": "chair_reset_state",
        "label": "Reset State",
        "question": (
            "Decide whether the chairs are in a fully reset state after room cleanup.\n"
            "Fully reset means all visible chairs are pushed in.\n"
            "If any single visible chair is left out, the room is not reset and the answer is no.\n"
            "End your response with a final line containing only: yes or no"
        ),
        "yes_means": "clean",
    },
    {
        "key": "chair_strict_negative_messy_refs_2",
        "label": "Strict Negative + 2 Messy Refs",
        "question": (
            "The reference images above are NOT READY examples where at least one visible chair is left out.\n"
            "Now apply the same strict rule to the test image.\n"
            "Answer yes only if every visible chair is tucked in.\n"
            "If even one visible chair is not tucked in, answer no.\n"
            "End your response with a final line containing only: yes or no"
        ),
        "yes_means": "clean",
        "few_shot": {"mode": "messy_only", "n_messy": 2, "n_clean": 0},
    },
    {
        "key": "chair_strict_negative_messy_refs_4",
        "label": "Strict Negative + 4 Messy Refs",
        "question": (
            "The reference images above are NOT READY examples where one or more visible chairs are left out.\n"
            "Use those failures as visual references.\n"
            "For the test image, answer yes only if all visible chairs are tucked in.\n"
            "If even one visible chair is not tucked in, answer no.\n"
            "End your response with a final line containing only: yes or no"
        ),
        "yes_means": "clean",
        "few_shot": {"mode": "messy_only", "n_messy": 4, "n_clean": 0},
    },
    {
        "key": "chair_strict_negative_mixed_refs_2x2",
        "label": "Strict Negative + 2 Clean 2 Messy",
        "question": (
            "The reference images above show READY and NOT READY chair layouts.\n"
            "READY means all visible chairs are tucked in.\n"
            "NOT READY means even one visible chair is left out.\n"
            "Apply the same rule to the test image.\n"
            "End your response with a final line containing only: yes or no"
        ),
        "yes_means": "clean",
        "few_shot": {"mode": "mixed", "n_messy": 2, "n_clean": 2},
    },
    {
        "key": "chair_strict_negative_mixed_refs_1x3",
        "label": "Strict Negative + 1 Clean 3 Messy",
        "question": (
            "Use the reference images above to compare chair layouts.\n"
            "A single visible chair left out is enough to make the answer no.\n"
            "Only answer yes if every visible chair in the test image is tucked in.\n"
            "End your response with a final line containing only: yes or no"
        ),
        "yes_means": "clean",
        "few_shot": {"mode": "mixed", "n_messy": 3, "n_clean": 1},
    },
]


def parse_yes_no(response: str) -> bool | None:
    text = (response or "").strip().lower()
    lines = [line.strip().rstrip(".,!") for line in text.splitlines() if line.strip()]
    if lines:
        last = lines[-1]
        if last == "yes":
            return True
        if last == "no":
            return False
    if re.search(r"\byes\b", text):
        return True
    if re.search(r"\bno\b", text):
        return False
    return None


def predict_label(answer: bool | None, yes_means: str) -> str | None:
    if answer is None:
        return None
    if answer:
        return yes_means
    return "clean" if yes_means == "messy" else "messy"


def compute_metrics(results: list[dict]) -> dict:
    total = correct = parse_errors = 0
    by_type: dict[str, dict] = {}
    latencies: list[int] = []

    for row in results:
        pred = row.get("predicted_label")
        gt = row["label"]
        ct = row["change_type"]

        total += 1
        if pred == gt:
            correct += 1
        if pred is None:
            parse_errors += 1

        bucket = by_type.setdefault(ct, {"total": 0, "correct": 0, "parse_errors": 0})
        bucket["total"] += 1
        if pred == gt:
            bucket["correct"] += 1
        if pred is None:
            bucket["parse_errors"] += 1

        lat = row.get("latency_ms", 0)
        if lat:
            latencies.append(lat)

    return {
        "accuracy": round(correct / total, 4) if total else 0.0,
        "n_images": total,
        "parse_errors": parse_errors,
        "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else 0,
        "per_change_type": {
            ct: {
                "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0.0,
                "correct": v["correct"],
                "total": v["total"],
                "parse_errors": v["parse_errors"],
            }
            for ct, v in by_type.items()
        },
    }


def _style_ax(ax, title: str, ylabel: str = "") -> None:
    ax.set_title(title, fontsize=11, pad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_axisbelow(True)


def save_plots(out_dir: Path, timestamp: str, fixed_results: dict, chair_variants: dict, best_key: str) -> list[str]:
    figures_dir = out_dir / FIGURES_SUBDIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    keys = list(chair_variants.keys())
    labels = [chair_variants[k]["label"] for k in keys]
    chair_acc = [chair_variants[k]["chair_metrics"]["accuracy"] for k in keys]
    overall_acc = [chair_variants[k]["combined_metrics"]["accuracy"] for k in keys]
    parse_errs = [chair_variants[k]["chair_metrics"]["parse_errors"] for k in keys]

    x = np.arange(len(keys))
    w = 0.38

    # Plot 1: chair prompt sweep accuracy
    fig, ax = plt.subplots(figsize=(11, 5.2), facecolor="white")
    ax.set_facecolor("#fbfbf9")
    _style_ax(ax, "Chair Prompt Techniques", "Accuracy")
    b1 = ax.bar(x - w / 2, chair_acc, w, color="#2a9d8f", label="Chair-only accuracy")
    b2 = ax.bar(x + w / 2, overall_acc, w, color="#457b9d", label="Overall accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, loc="upper left")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    p1 = figures_dir / f"qwen3vl_4b_int8_chair_prompt_accuracy_{timestamp}.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(str(p1))

    # Plot 2: chair parse errors
    fig, ax = plt.subplots(figsize=(10.5, 4.8), facecolor="white")
    ax.set_facecolor("#fbfbf9")
    _style_ax(ax, "Chair Prompt Parse Errors", "Count")
    bars = ax.bar(labels, parse_errs, color="#e76f51")
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ymax = max(parse_errs + [1])
    ax.set_ylim(0, ymax * 1.25)
    for bar, val in zip(bars, parse_errs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, str(val), ha="center", va="bottom", fontsize=8)
    p2 = figures_dir / f"qwen3vl_4b_int8_chair_prompt_parse_errors_{timestamp}.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(str(p2))

    # Plot 3: category accuracy using best chair prompt
    best_overall = chair_variants[best_key]["combined_metrics"]["per_change_type"]
    categories = ["table", "blinds", "whiteboard", "chairs"]
    vals = [best_overall.get(ct, {}).get("accuracy", 0.0) for ct in categories]
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor="white")
    ax.set_facecolor("#fbfbf9")
    _style_ax(ax, f"Best Prompt Category Accuracy ({chair_variants[best_key]['label']})", "Accuracy")
    bars = ax.bar(categories, vals, color=["#f97316", "#3b82f6", "#10b981", "#8b5cf6"])
    ax.set_ylim(0, 1.0)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    p3 = figures_dir / f"qwen3vl_4b_int8_best_prompt_category_accuracy_{timestamp}.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(str(p3))

    # Plot 4: examples from best chair prompt
    records = chair_variants[best_key]["chair_records"]
    positives = [r for r in records if r["predicted_label"] == r["label"] and not r["error"]]
    negatives = [r for r in records if r["predicted_label"] != r["label"] and not r["error"]]
    example_records = (negatives[:3] + positives[:3])[:6]
    if example_records:
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), facecolor="white")
        axes = axes.flatten()
        for ax, rec in zip(axes, example_records):
            img = Image.open(rec["image_path"]).convert("RGB")
            ax.imshow(img)
            ax.axis("off")
            ok = rec["predicted_label"] == rec["label"]
            color = "#16a34a" if ok else "#dc2626"
            title = (
                f"{rec['sample_id']}\n"
                f"gt={rec['label']} pred={rec['predicted_label'] or '?'}"
            )
            ax.set_title(title, fontsize=9, color=color, pad=5)
            snippet = (rec["raw_response"] or "").strip().replace("\n", " ")
            if len(snippet) > 90:
                snippet = snippet[:87] + "..."
            ax.text(
                0.02, 0.02, snippet or "(no response)",
                transform=ax.transAxes,
                fontsize=7,
                color="white",
                bbox=dict(facecolor="black", alpha=0.65, edgecolor="none", boxstyle="round,pad=0.25"),
                va="bottom",
            )
        for ax in axes[len(example_records):]:
            ax.axis("off")
        fig.suptitle(f"Best Chair Prompt Examples: {chair_variants[best_key]['label']}", fontsize=13, fontweight="bold")
        p4 = figures_dir / f"qwen3vl_4b_int8_best_chair_prompt_examples_{timestamp}.png"
        fig.tight_layout()
        fig.savefig(p4, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved.append(str(p4))

    return saved


def load_samples() -> list[dict]:
    with open(DATASET_CSV) as f:
        return list(csv.DictReader(f))


def build_chair_ref_images(all_chair_samples: list[dict], test_sample: dict, few_shot_cfg: dict) -> list[tuple[str, str]]:
    test_path = test_sample["image_path"]
    clean_candidates = [
        s for s in all_chair_samples
        if s["image_path"] != test_path and s["label"] == "clean" and Path(s["image_path"]).exists()
    ]
    messy_candidates = [
        s for s in all_chair_samples
        if s["image_path"] != test_path and s["label"] == "messy" and Path(s["image_path"]).exists()
    ]

    refs: list[tuple[str, str]] = []
    n_clean = few_shot_cfg.get("n_clean", 0)
    n_messy = few_shot_cfg.get("n_messy", 0)
    mode = few_shot_cfg.get("mode", "mixed")

    if mode == "messy_only":
        refs.extend((s["image_path"], "not_ready") for s in messy_candidates[:n_messy])
        return refs

    refs.extend((s["image_path"], "ready") for s in clean_candidates[:n_clean])
    refs.extend((s["image_path"], "not_ready") for s in messy_candidates[:n_messy])
    return refs


def load_qwen4b_int8_model(config_path: str) -> tuple[object, ModelConfig]:
    cfg = load_config(config_path)
    mcfg = next((m for m in cfg.enabled_models if m.key == "qwen3vl_4b_int8"), None)
    if mcfg is None:
        raise SystemExit("Model key 'qwen3vl_4b_int8' not found/enabled in benchmark_config.yaml")
    cls = MODEL_REGISTRY.get(mcfg.cls_name)
    if cls is None:
        raise SystemExit(f"Model class {mcfg.cls_name!r} is not available")

    run_cfg = ModelConfig(
        key=mcfg.key,
        enabled=mcfg.enabled,
        cls_name=mcfg.cls_name,
        model_path=mcfg.model_path,
        dtype=mcfg.dtype,
        generation=GenerationConfig(max_new_tokens=96, do_sample=False),
    )
    return cls(run_cfg), run_cfg


def run_once(model, sample: dict, prompt_cfg: dict, retries: int, all_chair_samples: list[dict] | None = None) -> dict:
    image_path = sample["image_path"]
    sid = sample["image_filename"].replace(".png", "").replace(".jpg", "")[:35]

    if not Path(image_path).exists():
        return {
            **sample,
            "sample_id": sid,
            "predicted_label": None,
            "answer": None,
            "raw_response": "",
            "error": "image not found",
            "latency_ms": 0,
        }

    total_latency = 0
    raw_response = ""
    answer = None
    error = None
    for attempt in range(retries + 1):
        if prompt_cfg.get("few_shot"):
            refs = build_chair_ref_images(all_chair_samples or [], sample, prompt_cfg["few_shot"])
            result = model.run_few_shot(refs, image_path, prompt_cfg["question"])
        else:
            result = model.run(image_path, prompt_cfg["question"])
        total_latency += round(result.latency_ms)
        raw_response = result.response
        error = result.error
        if error:
            break
        answer = parse_yes_no(result.response)
        if answer is not None:
            break

    predicted = predict_label(answer, prompt_cfg["yes_means"])
    return {
        **sample,
        "sample_id": sid,
        "predicted_label": predicted,
        "answer": answer,
        "raw_response": raw_response,
        "error": error,
        "latency_ms": total_latency,
        "n_refs": len(build_chair_ref_images(all_chair_samples or [], sample, prompt_cfg["few_shot"])) if prompt_cfg.get("few_shot") else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-VL-4B int8 env-monitoring eval with 5 chair prompt techniques.")
    parser.add_argument("--config", default=str(ROOT / "benchmark_config.yaml"))
    parser.add_argument("--retries", type=int, default=2, help="Retries on parse failure")
    parser.add_argument("--out-dir", default=str(ROOT / "results"))
    args = parser.parse_args()

    samples = load_samples()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Dataset: {DATASET_CSV}")
    print(f"Samples: {len(samples)}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    model, model_cfg = load_qwen4b_int8_model(args.config)
    print(f"\nLoading model: {model_cfg.key} from {model_cfg.model_path}")
    model.load()

    fixed_results: dict[str, dict] = {}
    for category, prompt_cfg in FIXED_PROMPTS.items():
        category_rows = [s for s in samples if s["change_type"] == category]
        print(f"\nRunning fixed prompt for {category} ({len(category_rows)} images)")
        records = []
        for sample in category_rows:
            rec = run_once(model, sample, prompt_cfg, args.retries)
            records.append(rec)
            print(
                f"  [{rec['sample_id']}] {category}: "
                f"pred={rec['predicted_label']} gt={rec['label']} "
                f"{'ERR' if rec['error'] else round(rec['latency_ms'])}"
            )
        fixed_results[category] = {
            "prompt": prompt_cfg,
            "records": records,
            "metrics": compute_metrics(records),
        }

    chair_rows = [s for s in samples if s["change_type"] == "chairs"]
    chair_variants: dict[str, dict] = {}
    for chair_prompt in CHAIR_PROMPTS:
        print(f"\nRunning chair variant: {chair_prompt['label']} ({len(chair_rows)} images)")
        records = []
        for sample in chair_rows:
            rec = run_once(model, sample, chair_prompt, args.retries, all_chair_samples=chair_rows)
            records.append(rec)
            print(
                f"  [{rec['sample_id']}] chairs/{chair_prompt['key']}: "
                f"pred={rec['predicted_label']} gt={rec['label']} refs={rec['n_refs']} "
                f"{'ERR' if rec['error'] else round(rec['latency_ms'])}"
            )

        combined_records = []
        for category in ("table", "blinds", "whiteboard"):
            combined_records.extend(fixed_results[category]["records"])
        combined_records.extend(records)

        chair_variants[chair_prompt["key"]] = {
            "label": chair_prompt["label"],
            "prompt": chair_prompt,
            "chair_metrics": compute_metrics(records),
            "combined_metrics": compute_metrics(combined_records),
            "chair_records": records,
        }

    model.unload()

    best_key = max(
        chair_variants,
        key=lambda k: (
            chair_variants[k]["chair_metrics"]["accuracy"],
            chair_variants[k]["combined_metrics"]["accuracy"],
        ),
    )

    out = {
        "timestamp": timestamp,
        "dataset": str(DATASET_CSV),
        "model_key": model_cfg.key,
        "model_path": model_cfg.model_path,
        "fixed_category_results": fixed_results,
        "chair_prompt_variants": chair_variants,
        "best_chair_prompt_key": best_key,
        "best_chair_prompt_label": chair_variants[best_key]["label"],
    }

    out_path = out_dir / f"qwen3vl_4b_int8_env_monitoring_prompt_sweep_{timestamp}.json"
    out_path.write_text(json.dumps(out, indent=2))
    saved_plots = save_plots(out_dir, timestamp, fixed_results, chair_variants, best_key)

    print("\nChair prompt sweep summary:")
    for key, result in chair_variants.items():
        cm = result["chair_metrics"]
        om = result["combined_metrics"]
        print(
            f"  {key}: chair_acc={cm['accuracy']:.1%} "
            f"overall_acc={om['accuracy']:.1%} "
            f"parse_errors={cm['parse_errors']}"
        )
    print(f"\nBest chair prompt: {best_key} ({chair_variants[best_key]['label']})")
    print(f"Results -> {out_path}")
    for plot_path in saved_plots:
        print(f"Plot -> {plot_path}")


if __name__ == "__main__":
    main()
