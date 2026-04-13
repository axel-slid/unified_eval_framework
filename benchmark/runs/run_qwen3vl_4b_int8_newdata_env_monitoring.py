#!/usr/bin/env python
"""
Evaluate the new data/ dataset with Qwen3-VL-4B int8 only.

Dataset expected:
  data/data4.12.26/labels.csv
  data/data4.12.26/meeting_room/{chairs,whiteboard,tables,blinds}/*.png

This script:
  - runs fixed prompts for whiteboard / blinds / table
  - tries multiple chair strategies, including crop-based split-image checks
  - saves JSON results plus PNG plots/examples

Usage:
  cd benchmark
  GPU_ID=2 python runs/run_qwen3vl_4b_int8_newdata_env_monitoring.py
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
PROJECT_ROOT = ROOT.parent
DATA_ROOT = PROJECT_ROOT / "data" / "data4.12.26"
LABELS_CSV = DATA_ROOT / "labels.csv"
DEFAULT_OUT_DIR = DATA_ROOT / "eval_results" / "qwen3vl_4b_int8_newdata_env_monitoring"

FIXED_PROMPTS: dict[str, dict] = {
    "whiteboard": {
        "question": (
            "Look at the whiteboard or writable wall surface in this image.\n"
            "Are there any visible marks, writing, residue, diagrams, or drawings on it?\n"
            "End your response with a final line containing only: yes or no"
        ),
        "yes_means": "messy",
    },
    "blinds": {
        "question": (
            "Look at the windows and coverings in this image.\n"
            "Are the blinds or shades open so outside light is clearly coming in?\n"
            "End your response with a final line containing only: yes or no"
        ),
        "yes_means": "clean",
    },
    "tables": {
        "question": (
            "Look only at the table surface in this image.\n"
            "Are there any loose items left on the table such as papers, bottles, cups, devices, chargers, or personal objects?\n"
            "End your response with a final line containing only: yes or no"
        ),
        "yes_means": "messy",
    },
}

CHAIR_STRATEGIES = [
    {
        "key": "strict_negative_full",
        "label": "Strict Negative Full",
        "mode": "single",
        "question": (
            "Answer yes only if every visible chair is tucked in.\n"
            "If even one visible chair is pulled out, angled away, or left out, answer no.\n"
            "End your response with a final line containing only: yes or no"
        ),
        "yes_means": "clean",
    },
    {
        "key": "single_failure_rule",
        "label": "Single Failure Rule",
        "mode": "single",
        "question": (
            "Check whether every visible chair is tucked in.\n"
            "If a single visible chair is not tucked in, answer no.\n"
            "Only answer yes if all visible chairs are tucked in.\n"
            "End your response with a final line containing only: yes or no"
        ),
        "yes_means": "clean",
    },
    {
        "key": "all_or_nothing",
        "label": "All or Nothing",
        "mode": "single",
        "question": (
            "This is an all-or-nothing check for chairs.\n"
            "Yes means all visible chairs are tucked in.\n"
            "No means at least one visible chair is left out.\n"
            "End your response with a final line containing only: yes or no"
        ),
        "yes_means": "clean",
    },
    {
        "key": "strict_negative_lr",
        "label": "Strict Negative LR Halves",
        "mode": "split_lr",
        "question": (
            "Inspect this crop for chairs.\n"
            "Answer yes only if every visible chair in this crop is tucked in.\n"
            "If even one visible chair in this crop is left out, answer no.\n"
            "End your response with a final line containing only: yes or no"
        ),
        "yes_means": "clean",
    },
    {
        "key": "strict_negative_quads",
        "label": "Strict Negative Quadrants",
        "mode": "split_quads",
        "question": (
            "Inspect this crop for chair placement.\n"
            "If you can see any chair not tucked in, answer no.\n"
            "Answer yes only if all visible chairs in this crop are tucked in.\n"
            "End your response with a final line containing only: yes or no"
        ),
        "yes_means": "clean",
    },
    {
        "key": "full_then_lr_verify",
        "label": "Full + LR Verify",
        "mode": "full_plus_lr",
        "question": (
            "Use a strict rule: one visible chair left out means no.\n"
            "For this image region, answer yes only if every visible chair here is tucked in.\n"
            "End your response with a final line containing only: yes or no"
        ),
        "yes_means": "clean",
    },
    {
        "key": "full_then_quads_verify",
        "label": "Full + Quadrants Verify",
        "mode": "full_plus_quads",
        "question": (
            "Use a strict rule: if one visible chair is left out, the answer is no.\n"
            "For this image region, answer yes only if every visible chair here is tucked in.\n"
            "End your response with a final line containing only: yes or no"
        ),
        "yes_means": "clean",
    },
]


def parse_yes_no(response: str) -> bool | None:
    text = (response or "").strip().lower()
    lines = [line.strip().rstrip(".,!") for line in text.splitlines() if line.strip()]
    if lines:
        if lines[-1] == "yes":
            return True
        if lines[-1] == "no":
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
    for row in results:
        pred = row.get("predicted_label")
        gt = row["label"]
        ct = row["subtask"]
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
    return {
        "accuracy": round(correct / total, 4) if total else 0.0,
        "n_images": total,
        "parse_errors": parse_errors,
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


def load_qwen_model(config_path: str):
    cfg = load_config(config_path)
    mcfg = next((m for m in cfg.enabled_models if m.key == "qwen3vl_4b_int8"), None)
    if mcfg is None:
        raise SystemExit("qwen3vl_4b_int8 not enabled in benchmark_config.yaml")
    cls = MODEL_REGISTRY.get(mcfg.cls_name)
    if cls is None:
        raise SystemExit(f"Missing model class: {mcfg.cls_name}")
    run_cfg = ModelConfig(
        key=mcfg.key,
        enabled=mcfg.enabled,
        cls_name=mcfg.cls_name,
        model_path=mcfg.model_path,
        dtype=mcfg.dtype,
        generation=GenerationConfig(max_new_tokens=96, do_sample=False),
    )
    return cls(run_cfg), run_cfg


def load_samples() -> list[dict]:
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Missing labels CSV: {LABELS_CSV}")
    with open(LABELS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    discovered_paths = {p.name: p for p in DATA_ROOT.rglob("*") if p.is_file()}
    normalized = []
    skipped = []
    for row in rows:
        subtask = row["subtask"]
        image_name = Path(row["output_path"]).name
        folder = {"table": "tables"}.get(subtask, subtask)
        room_type = row.get("room_type", "meeting_room")
        actual_path = DATA_ROOT / room_type / folder / image_name
        if not actual_path.exists():
            discovered = discovered_paths.get(image_name)
            if discovered is None:
                skipped.append((row["output_path"], room_type, subtask, image_name))
                continue
            actual_path = discovered
        label = row["label"]
        if subtask == "chairs":
            mapped = "clean" if "neat" in label or label.endswith("_clean") else "messy"
        elif subtask == "whiteboard":
            mapped = "clean" if label.endswith("_clean") else "messy"
        elif subtask == "blinds":
            mapped = "clean" if label.endswith("_up") or label.endswith("_clean") else "messy"
        elif subtask == "table":
            mapped = "clean" if "clean" in label else "messy"
            subtask = "tables"
        else:
            mapped = "clean" if "clean" in label else "messy"

        normalized.append(
            {
                **row,
                "subtask": subtask,
                "image_path": str(actual_path),
                "image_filename": image_name,
                "label": mapped,
            }
        )
    if skipped:
        print(f"Warning: skipping {len(skipped)} rows with missing images")
        for output_path, room_type, subtask, image_name in skipped[:10]:
            print(
                f"  missing image: room_type={room_type} subtask={subtask} "
                f"file={image_name} output_path={output_path}"
            )
    return normalized


def crop_regions(image_path: Path, mode: str, tmp_dir: Path) -> list[Path]:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    regions = []

    def save_crop(box, suffix):
        crop = img.crop(box)
        out = tmp_dir / f"{image_path.stem}__{suffix}.png"
        crop.save(out)
        regions.append(out)

    if mode in {"split_lr", "full_plus_lr"}:
        save_crop((0, 0, w // 2, h), "left")
        save_crop((w // 2, 0, w, h), "right")
    elif mode in {"split_quads", "full_plus_quads"}:
        save_crop((0, 0, w // 2, h // 2), "q1")
        save_crop((w // 2, 0, w, h // 2), "q2")
        save_crop((0, h // 2, w // 2, h), "q3")
        save_crop((w // 2, h // 2, w, h), "q4")
    return regions


def run_prompt(model, image_path: str, prompt: str) -> tuple[bool | None, str, int, str | None]:
    r = model.run(image_path, prompt)
    ans = None if r.error else parse_yes_no(r.response)
    return ans, r.response, round(r.latency_ms), r.error


def evaluate_chair_strategy(model, sample: dict, strategy: dict, tmp_dir: Path) -> dict:
    image_path = Path(sample["image_path"])
    mode = strategy["mode"]

    answers = []
    raw_parts = []
    total_latency = 0
    error = None

    if mode == "single":
        ans, raw, lat, err = run_prompt(model, str(image_path), strategy["question"])
        answers.append(ans)
        raw_parts.append(("full", raw))
        total_latency += lat
        error = err
    else:
        if mode.startswith("full_plus"):
            ans, raw, lat, err = run_prompt(model, str(image_path), strategy["question"])
            answers.append(ans)
            raw_parts.append(("full", raw))
            total_latency += lat
            error = err or error
        for crop_path in crop_regions(image_path, mode, tmp_dir):
            ans, raw, lat, err = run_prompt(model, str(crop_path), strategy["question"])
            answers.append(ans)
            raw_parts.append((crop_path.stem, raw))
            total_latency += lat
            error = err or error

    # strict aggregation: any visible "no" means messy; all yes means clean; else unparsed
    if any(a is False for a in answers):
        final_answer = False
    elif answers and all(a is True for a in answers):
        final_answer = True
    else:
        final_answer = None

    pred = predict_label(final_answer, strategy["yes_means"])
    return {
        **sample,
        "strategy_key": strategy["key"],
        "predicted_label": pred,
        "answer": final_answer,
        "raw_response": "\n\n".join(f"[{name}]\n{text}" for name, text in raw_parts),
        "error": error,
        "latency_ms": total_latency,
        "n_views": len(raw_parts),
    }


def save_plots(out_dir: Path, timestamp: str, fixed_results: dict, chair_variants: dict, best_key: str) -> list[str]:
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    keys = list(chair_variants.keys())
    labels = [chair_variants[k]["label"] for k in keys]
    chair_acc = [chair_variants[k]["chair_metrics"]["accuracy"] for k in keys]
    overall_acc = [chair_variants[k]["combined_metrics"]["accuracy"] for k in keys]

    x = np.arange(len(keys))
    w = 0.38

    fig, ax = plt.subplots(figsize=(12, 5.5), facecolor="white")
    ax.set_facecolor("#fbfbf9")
    ax.bar(x - w / 2, chair_acc, w, label="Chair-only accuracy", color="#2a9d8f")
    ax.bar(x + w / 2, overall_acc, w, label="Overall accuracy", color="#457b9d")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_title("Chair Strategy Sweep on New Dataset")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    p1 = figures_dir / f"qwen4b_int8_newdata_chair_strategy_accuracy_{timestamp}.png"
    fig.savefig(p1, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(str(p1))

    best = chair_variants[best_key]["combined_metrics"]["per_change_type"]
    cats = ["tables", "blinds", "whiteboard", "chairs"]
    vals = [best.get(c, {}).get("accuracy", 0.0) for c in cats]
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor="white")
    ax.set_facecolor("#fbfbf9")
    ax.bar(cats, vals, color=["#f97316", "#3b82f6", "#10b981", "#8b5cf6"])
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Best Strategy Category Accuracy ({chair_variants[best_key]['label']})")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    fig.tight_layout()
    p2 = figures_dir / f"qwen4b_int8_newdata_best_strategy_category_accuracy_{timestamp}.png"
    fig.savefig(p2, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(str(p2))

    examples = chair_variants[best_key]["chair_records"]
    bad = [r for r in examples if r["predicted_label"] != r["label"] and not r["error"]][:3]
    good = [r for r in examples if r["predicted_label"] == r["label"] and not r["error"]][:3]
    chosen = (bad + good)[:6]
    if chosen:
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), facecolor="white")
        axes = axes.flatten()
        for ax, rec in zip(axes, chosen):
            img = Image.open(rec["image_path"]).convert("RGB")
            ax.imshow(img)
            ax.axis("off")
            ok = rec["predicted_label"] == rec["label"]
            color = "#16a34a" if ok else "#dc2626"
            ax.set_title(
                f"{rec['image_filename']}\ngt={rec['label']} pred={rec['predicted_label']} views={rec['n_views']}",
                fontsize=8,
                color=color,
            )
            snippet = (rec["raw_response"] or "").replace("\n", " ")
            if len(snippet) > 100:
                snippet = snippet[:97] + "..."
            ax.text(
                0.02, 0.02, snippet,
                transform=ax.transAxes,
                fontsize=6.5,
                color="white",
                bbox=dict(facecolor="black", alpha=0.65, edgecolor="none", boxstyle="round,pad=0.2"),
                va="bottom",
            )
        for ax in axes[len(chosen):]:
            ax.axis("off")
        fig.tight_layout()
        p3 = figures_dir / f"qwen4b_int8_newdata_best_strategy_examples_{timestamp}.png"
        fig.savefig(p3, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved.append(str(p3))
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-VL-4B int8 eval on new data/ folder with chair strategy sweep.")
    parser.add_argument("--config", default=str(ROOT / "benchmark_config.yaml"))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    samples = load_samples()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_chair_crop_cache"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Dataset: {LABELS_CSV}")
    print(f"Samples: {len(samples)}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    model, model_cfg = load_qwen_model(args.config)
    print(f"Loading model: {model_cfg.key}")
    model.load()

    fixed_results = {}
    for subtask_key, prompt_cfg in FIXED_PROMPTS.items():
        rows = [s for s in samples if s["subtask"] == subtask_key]
        records = []
        print(f"\nRunning fixed prompt for {subtask_key}: {len(rows)} images")
        for sample in rows:
            ans, raw, lat, err = run_prompt(model, sample["image_path"], prompt_cfg["question"])
            pred = predict_label(ans, prompt_cfg["yes_means"])
            rec = {
                **sample,
                "predicted_label": pred,
                "answer": ans,
                "raw_response": raw,
                "error": err,
                "latency_ms": lat,
            }
            records.append(rec)
            print(f"  {sample['image_filename']}: pred={pred} gt={sample['label']} {lat if not err else 'ERR'}")
        fixed_results[subtask_key] = {"prompt": prompt_cfg, "records": records, "metrics": compute_metrics(records)}

    chair_rows = [s for s in samples if s["subtask"] == "chairs"]
    chair_variants = {}
    for strategy in CHAIR_STRATEGIES:
        print(f"\nRunning chair strategy: {strategy['label']} ({strategy['mode']})")
        records = []
        for sample in chair_rows:
            rec = evaluate_chair_strategy(model, sample, strategy, tmp_dir)
            records.append(rec)
            print(f"  {sample['image_filename']}: pred={rec['predicted_label']} gt={sample['label']} views={rec['n_views']}")
        combined = []
        for key in ("tables", "blinds", "whiteboard"):
            combined.extend(fixed_results[key]["records"])
        combined.extend(records)
        chair_variants[strategy["key"]] = {
            "label": strategy["label"],
            "strategy": strategy,
            "chair_metrics": compute_metrics(records),
            "combined_metrics": compute_metrics(combined),
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
        "timestamp": ts,
        "dataset": str(LABELS_CSV),
        "model_key": model_cfg.key,
        "model_path": model_cfg.model_path,
        "fixed_category_results": fixed_results,
        "chair_strategy_results": chair_variants,
        "best_chair_strategy_key": best_key,
        "best_chair_strategy_label": chair_variants[best_key]["label"],
    }
    out_path = out_dir / f"qwen3vl_4b_int8_newdata_env_monitoring_{ts}.json"
    out_path.write_text(json.dumps(out, indent=2))

    saved_plots = save_plots(out_dir, ts, fixed_results, chair_variants, best_key)
    print("\nChair strategy summary:")
    for key, result in chair_variants.items():
        print(
            f"  {key}: chair_acc={result['chair_metrics']['accuracy']:.1%} "
            f"overall_acc={result['combined_metrics']['accuracy']:.1%}"
        )
    print(f"\nBest chair strategy: {best_key} ({chair_variants[best_key]['label']})")
    print(f"Results -> {out_path}")
    for plot_path in saved_plots:
        print(f"Plot -> {plot_path}")


if __name__ == "__main__":
    main()
