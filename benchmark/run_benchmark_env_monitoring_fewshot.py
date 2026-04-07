#!/usr/bin/env python
"""
run_benchmark_env_monitoring_fewshot.py — Few-shot grounded environment monitoring benchmark.

Each inference shows the model 4 reference images (2 clean, 2 messy) for the
category before asking it to classify the test image.  This single-stage approach
replaces the two-stage pipeline: the examples implicitly anchor what the object
looks like and what "ready" vs "not_ready" means.

Output classes:
  present_clean  — model predicted ready
  present_messy  — model predicted not_ready
  uncertain      — unparseable response

Ground-truth mapping:
  clean  → present_clean
  messy  → present_messy

Usage:
    cd benchmark
    CUDA_VISIBLE_DEVICES=1 conda run -p /mnt/shared/dils/envs/Qwen3VL-env \\
        python run_benchmark_env_monitoring_fewshot.py --all

    # Specific models only:
    python run_benchmark_env_monitoring_fewshot.py --models qwen3vl_4b internvl
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
from datetime import datetime
from pathlib import Path

from config import BenchmarkConfig, load_config
from models import MODEL_REGISTRY

DATASET_CSV = (
    Path(__file__).parent.parent
    / "environment_monitoring_dataset"
    / "unified_annotations.csv"
)

CLASSES = ["present_clean", "present_messy", "uncertain"]

# ── Category-specific prompts (single-stage, shown after examples) ─────────────

PROMPTS = {
    "whiteboard": (
        "You have seen 4 reference images above showing READY (clean whiteboard) "
        "and NOT READY (whiteboard with markings).\n\n"
        "Now look at this new image.\n"
        "Does the whiteboard have ANY markings, text, drawings, or writing on it?\n"
        "Reply on the first line with: ready (completely blank) or not_ready (any marks present).\n"
        "Then on a new line describe exactly what you observe on the whiteboard."
    ),
    "blinds": (
        "You have seen 4 reference images above showing READY (coverings uniformly closed) "
        "and NOT READY (gaps, outdoor view, or uneven slats).\n\n"
        "Now look at this new image.\n"
        "Can you see any daylight, outdoor view, or gaps through the window coverings?\n"
        "Reply on the first line with: not_ready (any gaps or outdoor view visible) "
        "or ready (uniformly closed with no gaps).\n"
        "Then on a new line describe what you see."
    ),
    "chairs": (
        "You have seen 4 reference images above showing READY (chairs neatly around table) "
        "and NOT READY (chairs significantly displaced or out of place).\n\n"
        "Now look at this new image.\n"
        "Are any chairs significantly displaced — pulled far from the table, "
        "pushed against a wall, or clearly abandoned?\n"
        "Minor variation is fine.\n"
        "Reply on the first line with: not_ready (significant displacement) "
        "or ready (chairs generally arranged around the table).\n"
        "Then on a new line describe the chair arrangement."
    ),
    "table": (
        "You have seen 4 reference images above showing READY (clear table surface) "
        "and NOT READY (personal items left on table).\n\n"
        "Now look at this new image.\n"
        "Does the table have ANY personal items — laptops, notebooks, cups, bottles, "
        "phones, bags, papers, food?\n"
        "Only permanently installed equipment is acceptable.\n"
        "Reply on the first line with: not_ready (any items present) or ready (surface clear).\n"
        "Then on a new line list what you see on the table."
    ),
}


# ── Reference shot selection ──────────────────────────────────────────────────

def build_shot_pool(samples: list[dict]) -> dict[str, dict[str, list[str]]]:
    """
    For each category, collect sorted lists of image paths by label.
    Returns: {category: {"clean": [...paths...], "messy": [...paths...]}}
    """
    pool: dict[str, dict[str, list[str]]] = {}
    for s in samples:
        ct = s["change_type"]
        label = s["label"]
        pool.setdefault(ct, {"clean": [], "messy": []})
        pool[ct][label].append(s["image_path"])
    # Sort for determinism
    for ct in pool:
        pool[ct]["clean"].sort()
        pool[ct]["messy"].sort()
    return pool


def get_shots(
    pool: dict[str, dict[str, list[str]]],
    ct: str,
    test_path: str,
    n_each: int = 2,
) -> list[tuple[str, str]]:
    """
    Pick n_each clean + n_each messy reference images for category ct,
    excluding test_path.  Returns list of (image_path, 'ready'/'not_ready').
    """
    clean_cands = [p for p in pool[ct]["clean"] if p != test_path]
    messy_cands = [p for p in pool[ct]["messy"] if p != test_path]

    shots: list[tuple[str, str]] = []
    shots += [(p, "ready") for p in clean_cands[:n_each]]
    shots += [(p, "not_ready") for p in messy_cands[:n_each]]
    return shots


# ── Response parsing ──────────────────────────────────────────────────────────

def parse_ready(response: str) -> bool | None:
    text = response.strip().lower()
    first_line = text.split("\n")[0].strip()
    if "not_ready" in first_line or "not ready" in first_line:
        return False
    if "ready" in first_line:
        return True
    if "not_ready" in text or "not ready" in text:
        return False
    if "ready" in text:
        return True
    return None


def extract_description(response: str) -> str:
    lines = response.strip().split("\n")
    desc_lines = [l.strip() for l in lines[1:] if l.strip()]
    return " ".join(desc_lines)


def gt_class(label: str) -> str:
    return "present_clean" if label == "clean" else "present_messy"


def classify(ready: bool | None) -> str:
    if ready is True:
        return "present_clean"
    if ready is False:
        return "present_messy"
    return "uncertain"


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    total = correct = 0
    by_type: dict[str, dict] = {}
    class_confusion: dict[str, dict[str, int]] = {c: {c2: 0 for c2 in CLASSES} for c in CLASSES}

    for r in results:
        pred_cls = r.get("predicted_class")
        if pred_cls is None:
            continue
        gt = r["gt_class"]
        ct = r["change_type"]
        total += 1
        if pred_cls == gt:
            correct += 1
        class_confusion[gt][pred_cls] += 1
        by_type.setdefault(ct, {"total": 0, "correct": 0})
        by_type[ct]["total"] += 1
        if pred_cls == gt:
            by_type[ct]["correct"] += 1

    acc = correct / total if total else 0.0
    per_type = {
        ct: {
            "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0.0,
            "correct": v["correct"],
            "total": v["total"],
        }
        for ct, v in by_type.items()
    }
    return {
        "accuracy": round(acc, 4),
        "n_images": total,
        "class_confusion": class_confusion,
        "per_change_type": per_type,
    }


# ── Core runner ───────────────────────────────────────────────────────────────

def run_fewshot_benchmark(
    cfg: BenchmarkConfig,
    model_filter: list[str] | None = None,
    n_shots: int = 2,
) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(DATASET_CSV) as f:
        samples = list(csv.DictReader(f))

    for s in samples:
        s["gt_class"] = gt_class(s["label"])

    shot_pool = build_shot_pool(samples)

    print(f"\nDataset: {DATASET_CSV}")
    print(f"Samples: {len(samples)}  |  shots per category: {n_shots} clean + {n_shots} messy")
    by_type: dict[str, int] = {}
    for s in samples:
        by_type[s["change_type"]] = by_type.get(s["change_type"], 0) + 1
    for ct, n in sorted(by_type.items()):
        print(f"  {ct}: {n} images")

    models_to_run = [
        m for m in cfg.enabled_models
        if model_filter is None or m.key in model_filter
    ]
    if not models_to_run:
        print("No models selected.")
        return

    all_results: dict[str, dict] = {}

    for mcfg in models_to_run:
        cls = MODEL_REGISTRY.get(mcfg.cls_name)
        if cls is None:
            print(f"[SKIP] Unknown class '{mcfg.cls_name}'")
            continue

        model = cls(mcfg)
        print(f"\n{'='*60}\nModel: {model.name}\n{'='*60}")
        model.load()

        # Check few-shot support
        try:
            model.run_few_shot([], "", "")
        except NotImplementedError:
            print(f"  [SKIP] {model.name} does not support run_few_shot")
            model.unload()
            continue
        except Exception:
            pass  # Expected to fail with empty inputs — support exists

        model_results: list[dict] = []

        for sample in samples:
            image_path = sample["image_path"]
            sid = sample["image_filename"].replace(".png", "")[:35]
            ct = sample["change_type"]
            gt = sample["gt_class"]

            if not Path(image_path).exists():
                print(f"  [{sid}] WARNING: image not found")
                model_results.append({
                    **sample,
                    "predicted_class": None,
                    "raw_response": "",
                    "description": "",
                    "error": "image not found",
                    "latency_ms": 0,
                    "model_name": model.name,
                    "model_path": mcfg.model_path,
                    "n_shots": n_shots,
                })
                continue

            shots = get_shots(shot_pool, ct, image_path, n_each=n_shots)
            if len(shots) < 2:
                print(f"  [{sid}] WARNING: not enough shot images for {ct}")

            result = model.run_few_shot(shots, image_path, PROMPTS[ct])

            if result.error:
                print(f"  [{sid}] ERROR: {result.error}")
                model_results.append({
                    **sample,
                    "predicted_class": None,
                    "raw_response": "",
                    "description": "",
                    "error": result.error,
                    "latency_ms": 0,
                    "model_name": model.name,
                    "model_path": mcfg.model_path,
                    "n_shots": n_shots,
                })
                continue

            ready = parse_ready(result.response)
            pred_cls = classify(ready)
            description = extract_description(result.response)
            correct = pred_cls == gt
            mark = "✓" if correct else "✗"

            print(
                f"  [{sid}] {mark} [{ct}] "
                f"→ {pred_cls} (gt={gt}) {result.latency_ms:.0f}ms"
            )
            if description:
                print(f"    → {description[:130]}")

            model_results.append({
                **sample,
                "predicted_class": pred_cls,
                "raw_response": result.response,
                "description": description,
                "error": None,
                "latency_ms": round(result.latency_ms),
                "model_name": model.name,
                "model_path": mcfg.model_path,
                "n_shots": n_shots,
            })

        model.unload()

        metrics = compute_metrics(model_results)
        all_results[mcfg.key] = {
            "model_name": model.name,
            "model_path": mcfg.model_path,
            "metrics": metrics,
            "results": model_results,
        }

        print(f"\n  accuracy={metrics['accuracy']:.1%}")
        for ct, m in sorted(metrics["per_change_type"].items()):
            print(f"  [{ct}] {m['accuracy']:.1%} ({m['correct']}/{m['total']})")

        # Incremental checkpoint
        checkpoint = {
            "timestamp": timestamp,
            "mode": "few_shot",
            "n_shots": n_shots,
            "dataset": str(DATASET_CSV),
            "models": all_results,
        }
        ckpt_path = cfg.output_dir / f"env_monitoring_fewshot_{timestamp}.json"
        ckpt_path.write_text(json.dumps(checkpoint, indent=2))

    out = {
        "timestamp": timestamp,
        "mode": "few_shot",
        "n_shots": n_shots,
        "dataset": str(DATASET_CSV),
        "models": all_results,
    }
    json_path = cfg.output_dir / f"env_monitoring_fewshot_{timestamp}.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults → {json_path}")
    print_summary(all_results)


# ── Console summary ───────────────────────────────────────────────────────────

def print_summary(all_results: dict) -> None:
    categories = sorted({
        ct for data in all_results.values()
        for ct in data["metrics"]["per_change_type"]
    })
    print(f"\n{'='*70}\nENVIRONMENT MONITORING — FEW-SHOT SUMMARY\n{'='*70}")
    col_w = max(len(ct) for ct in categories)
    header = f"{'Model':<38} {'Overall':>8}  " + "  ".join(f"{ct:>{col_w}}" for ct in categories)
    print(header)
    print("-" * len(header))
    for key, data in all_results.items():
        m = data["metrics"]
        row = f"{data['model_name']:<38} {m['accuracy']:>7.1%}  "
        for ct in categories:
            ct_m = m["per_change_type"].get(ct, {})
            row += f"  {ct_m.get('accuracy', 0):>{col_w}.1%}"
        print(row)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Few-shot grounded environment monitoring benchmark")
    parser.add_argument("--config", default="benchmark_config.yaml")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--shots", type=int, default=2,
                        help="Number of clean + messy reference images per category (default: 2 each)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_filter = args.models if args.models else None

    print(f"Config:  {args.config}")
    print(f"Dataset: {DATASET_CSV}")
    print(f"Mode:    few-shot ({args.shots} clean + {args.shots} messy per category)")
    print(f"Models:  {model_filter or [m.key for m in cfg.enabled_models]}")

    run_fewshot_benchmark(cfg, model_filter=model_filter, n_shots=args.shots)
