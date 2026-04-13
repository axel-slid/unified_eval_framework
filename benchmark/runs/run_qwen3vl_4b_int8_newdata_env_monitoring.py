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
import base64
import csv
import html
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

STRICT_CHAIR_SUFFIX = "End your response with a final line containing only: yes or no"


def chair_strategy(
    key: str,
    label: str,
    mode: str,
    question: str,
    few_shot: dict | None = None,
) -> dict:
    out = {
        "key": key,
        "label": label,
        "mode": mode,
        "question": question,
        "yes_means": "clean",
    }
    if few_shot:
        out["few_shot"] = few_shot
    return out


def build_chair_strategies() -> list[dict]:
    return [
        chair_strategy(
            "strict_negative_full",
            "Strict Negative Full",
            "single",
            (
                "Answer yes only if every visible chair is tucked in.\n"
                "If even one visible chair is pulled out, angled away, rotated out from the table, "
                f"or left out, answer no.\n{STRICT_CHAIR_SUFFIX}"
            ),
        ),
        chair_strategy(
            "single_failure_rule",
            "Single Failure Rule",
            "single",
            (
                "Check whether every visible chair is tucked in.\n"
                "If a single visible chair is not tucked in, answer no.\n"
                f"Only answer yes if all visible chairs are tucked in.\n{STRICT_CHAIR_SUFFIX}"
            ),
        ),
        chair_strategy(
            "visible_only_exception_scan",
            "Visible Exception Scan",
            "single",
            (
                "Scan all visible chairs one by one.\n"
                "Ignore chairs you cannot see.\n"
                "If you notice even one visible chair sticking out from the table or aisle, answer no.\n"
                f"Answer yes only when every visible chair looks pushed in neatly.\n{STRICT_CHAIR_SUFFIX}"
            ),
        ),
        chair_strategy(
            "count_then_fail",
            "Count Then Fail",
            "single",
            (
                "Mentally count the visible chairs that are not tucked in.\n"
                "If the count is 0, answer yes.\n"
                f"If the count is 1 or more, answer no.\n{STRICT_CHAIR_SUFFIX}"
            ),
        ),
        chair_strategy(
            "strict_negative_lr",
            "Strict Negative LR Halves",
            "split_lr",
            (
                "Inspect this crop for chairs.\n"
                "Answer yes only if every visible chair in this crop is tucked in.\n"
                f"If even one visible chair in this crop is left out, answer no.\n{STRICT_CHAIR_SUFFIX}"
            ),
        ),
        chair_strategy(
            "strict_negative_quads",
            "Strict Negative Quadrants",
            "split_quads",
            (
                "Inspect this crop for chair placement.\n"
                "If you can see any chair not tucked in, answer no.\n"
                f"Answer yes only if all visible chairs in this crop are tucked in.\n{STRICT_CHAIR_SUFFIX}"
            ),
        ),
        chair_strategy(
            "vertical_thirds",
            "Vertical Thirds",
            "split_vertical_thirds",
            (
                "Inspect this vertical slice for chairs.\n"
                "If any visible chair in this slice is pulled out or crooked, answer no.\n"
                f"Answer yes only if all visible chairs in this slice are tucked in.\n{STRICT_CHAIR_SUFFIX}"
            ),
        ),
        chair_strategy(
            "horizontal_bands",
            "Horizontal Bands",
            "split_horizontal_bands",
            (
                "Inspect this horizontal band for chairs.\n"
                "If one visible chair is not pushed in, answer no.\n"
                f"Answer yes only if every visible chair in this band is tucked in.\n{STRICT_CHAIR_SUFFIX}"
            ),
        ),
        chair_strategy(
            "overlap_quads",
            "Overlap Quadrants",
            "split_overlap_quads",
            (
                "Inspect this overlapping crop for chairs near the table.\n"
                "If any visible chair appears left out, not aligned, or not tucked in, answer no.\n"
                f"Answer yes only if every visible chair in this crop is tucked in.\n{STRICT_CHAIR_SUFFIX}"
            ),
        ),
        chair_strategy(
            "full_then_lr_verify",
            "Full + LR Verify",
            "full_plus_lr",
            (
                "Use a strict rule: one visible chair left out means no.\n"
                f"For this image region, answer yes only if every visible chair here is tucked in.\n{STRICT_CHAIR_SUFFIX}"
            ),
        ),
        chair_strategy(
            "full_then_quads_verify",
            "Full + Quadrants Verify",
            "full_plus_quads",
            (
                "Use a strict rule: if one visible chair is left out, the answer is no.\n"
                f"For this image region, answer yes only if every visible chair here is tucked in.\n{STRICT_CHAIR_SUFFIX}"
            ),
        ),
        chair_strategy(
            "full_then_vertical_thirds",
            "Full + Vertical Thirds",
            "full_plus_vertical_thirds",
            (
                "Apply a zero-tolerance rule for chair placement in this region.\n"
                f"Yes means every visible chair is tucked in; no means at least one is left out.\n{STRICT_CHAIR_SUFFIX}"
            ),
        ),
        chair_strategy(
            "full_then_overlap_quads",
            "Full + Overlap Quads",
            "full_plus_overlap_quads",
            (
                "Use zero tolerance for chairs in this region.\n"
                f"If one visible chair is not tucked in, answer no; otherwise answer yes.\n{STRICT_CHAIR_SUFFIX}"
            ),
        ),
        chair_strategy(
            "messy_refs_2",
            "Messy Refs x2",
            "single",
            (
                "Use the reference images labeled not_ready as examples of messy chairs.\n"
                "For the test image, answer yes only if every visible chair is tucked in.\n"
                f"If the test image resembles the messy references in any visible chair, answer no.\n{STRICT_CHAIR_SUFFIX}"
            ),
            few_shot={"mode": "messy_only", "n_messy": 2},
        ),
        chair_strategy(
            "mixed_refs_1x1",
            "Mixed Refs 1+1",
            "single",
            (
                "Use the ready references as clean examples and the not_ready references as messy examples.\n"
                "Answer yes only if the test image matches the clean pattern where all visible chairs are tucked in.\n"
                f"If even one visible chair looks like the messy references, answer no.\n{STRICT_CHAIR_SUFFIX}"
            ),
            few_shot={"mode": "mixed", "n_clean": 1, "n_messy": 1},
        ),
        chair_strategy(
            "mixed_refs_2x2",
            "Mixed Refs 2+2",
            "single",
            (
                "Compare the test image against the clean and messy references.\n"
                "Answer yes only if all visible chairs resemble the clean references.\n"
                f"If any visible chair resembles the messy references by being left out, answer no.\n{STRICT_CHAIR_SUFFIX}"
            ),
            few_shot={"mode": "mixed", "n_clean": 2, "n_messy": 2},
        ),
        chair_strategy(
            "lr_with_messy_refs",
            "LR Halves + Messy Refs",
            "split_lr",
            (
                "Use the not_ready references as examples of chairs left out.\n"
                "For this crop, answer yes only if every visible chair is tucked in.\n"
                f"If any visible chair in this crop resembles a messy reference, answer no.\n{STRICT_CHAIR_SUFFIX}"
            ),
            few_shot={"mode": "messy_only", "n_messy": 2},
        ),
        chair_strategy(
            "quads_with_mixed_refs",
            "Quads + Mixed Refs",
            "split_quads",
            (
                "Use the ready and not_ready references to judge this crop.\n"
                "Answer yes only if every visible chair in this crop matches the ready pattern.\n"
                f"If any visible chair matches the not_ready pattern, answer no.\n{STRICT_CHAIR_SUFFIX}"
            ),
            few_shot={"mode": "mixed", "n_clean": 1, "n_messy": 1},
        ),
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
    elif mode in {"split_vertical_thirds", "full_plus_vertical_thirds"}:
        x1 = w // 3
        x2 = (2 * w) // 3
        save_crop((0, 0, x1, h), "v1")
        save_crop((x1, 0, x2, h), "v2")
        save_crop((x2, 0, w, h), "v3")
    elif mode in {"split_horizontal_bands", "full_plus_horizontal_bands"}:
        y1 = h // 3
        y2 = (2 * h) // 3
        save_crop((0, 0, w, y1), "h1")
        save_crop((0, y1, w, y2), "h2")
        save_crop((0, y2, w, h), "h3")
    elif mode in {"split_overlap_quads", "full_plus_overlap_quads"}:
        ox = int(w * 0.15)
        oy = int(h * 0.15)
        mx = w // 2
        my = h // 2
        save_crop((0, 0, min(w, mx + ox), min(h, my + oy)), "oq1")
        save_crop((max(0, mx - ox), 0, w, min(h, my + oy)), "oq2")
        save_crop((0, max(0, my - oy), min(w, mx + ox), h), "oq3")
        save_crop((max(0, mx - ox), max(0, my - oy), w, h), "oq4")
    return regions


def run_prompt(
    model,
    image_path: str,
    prompt: str,
    retries: int,
    refs: list[tuple[str, str]] | None = None,
) -> tuple[bool | None, str, int, str | None]:
    total_latency = 0
    last_response = ""
    last_error = None
    answer = None
    for _ in range(retries + 1):
        r = model.run_few_shot(refs, image_path, prompt) if refs else model.run(image_path, prompt)
        total_latency += round(r.latency_ms)
        last_response = r.response
        last_error = r.error
        if r.error:
            break
        answer = parse_yes_no(r.response)
        if answer is not None:
            break
    return answer, last_response, total_latency, last_error


def evaluate_chair_strategy(
    model,
    sample: dict,
    strategy: dict,
    tmp_dir: Path,
    retries: int,
    all_chair_samples: list[dict],
) -> dict:
    image_path = Path(sample["image_path"])
    mode = strategy["mode"]
    refs = build_chair_ref_images(all_chair_samples, sample, strategy["few_shot"]) if strategy.get("few_shot") else None

    answers = []
    raw_parts = []
    total_latency = 0
    error = None

    if mode == "single":
        ans, raw, lat, err = run_prompt(model, str(image_path), strategy["question"], retries, refs)
        answers.append(ans)
        raw_parts.append(("full", raw))
        total_latency += lat
        error = err
    else:
        if mode.startswith("full_plus"):
            ans, raw, lat, err = run_prompt(model, str(image_path), strategy["question"], retries, refs)
            answers.append(ans)
            raw_parts.append(("full", raw))
            total_latency += lat
            error = err or error
        for crop_path in crop_regions(image_path, mode, tmp_dir):
            ans, raw, lat, err = run_prompt(model, str(crop_path), strategy["question"], retries, refs)
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
        "n_refs": len(refs or []),
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


def img_b64(path: str | Path) -> str | None:
    try:
        p = Path(path)
        ext = p.suffix.lower().strip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/png")
        return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"
    except Exception:
        return None


def build_html_report(
    timestamp: str,
    out_path: Path,
    fixed_results: dict,
    chair_variants: dict,
    report_chair_key: str,
    plot_paths: list[str],
    model_key: str,
) -> str:
    report_variant = chair_variants[report_chair_key]
    sections = []

    def metric_block(title: str, metrics: dict) -> str:
        rows = []
        for cat, vals in metrics.get("per_change_type", {}).items():
            rows.append(
                f"<tr><td>{html.escape(cat)}</td><td>{vals['correct']}/{vals['total']}</td>"
                f"<td>{vals['accuracy']:.1%}</td><td>{vals['parse_errors']}</td></tr>"
            )
        return (
            f"<div class='metric-card'><h3>{html.escape(title)}</h3>"
            f"<p><strong>Accuracy:</strong> {metrics.get('accuracy', 0.0):.1%} "
            f"<span class='muted'>({metrics.get('n_images', 0)} images, parse errors {metrics.get('parse_errors', 0)})</span></p>"
            f"<table><tr><th>Category</th><th>Correct</th><th>Accuracy</th><th>Parse Errors</th></tr>{''.join(rows)}</table>"
            f"</div>"
        )

    plot_html = "".join(
        (
            f"<figure class='plot-card'><img src='{img_b64(p) or ''}' alt='{html.escape(Path(p).name)}'>"
            f"<figcaption>{html.escape(Path(p).name)}</figcaption></figure>"
        )
        for p in plot_paths
        if Path(p).exists()
    )

    records_by_category = {
        "whiteboard": fixed_results["whiteboard"]["records"],
        "blinds": fixed_results["blinds"]["records"],
        "tables": fixed_results["tables"]["records"],
        "chairs": report_variant["chair_records"],
    }

    for category, records in records_by_category.items():
        cards = []
        for rec in sorted(records, key=lambda r: (r["label"], r["image_filename"])):
            img_src = img_b64(rec["image_path"]) or ""
            pred = rec.get("predicted_label")
            gt = rec["label"]
            ok = pred == gt
            status_class = "ok" if ok else "bad"
            answer = rec.get("answer")
            answer_str = "yes" if answer is True else "no" if answer is False else "unparsed"
            cards.append(
                "<article class='card'>"
                f"<img src='{img_src}' alt='{html.escape(rec['image_filename'])}'>"
                f"<div class='meta'><div class='filename'>{html.escape(rec['image_filename'])}</div>"
                f"<div class='badges'><span class='badge gt'>gt={html.escape(gt)}</span>"
                f"<span class='badge {status_class}'>pred={html.escape(str(pred))}</span>"
                f"<span class='badge neutral'>answer={html.escape(answer_str)}</span>"
                f"<span class='badge neutral'>latency={rec.get('latency_ms', 0)}ms</span>"
                f"<span class='badge neutral'>views={rec.get('n_views', 1)}</span>"
                f"<span class='badge neutral'>refs={rec.get('n_refs', 0)}</span></div>"
                f"<details><summary>Model response</summary><pre>{html.escape(rec.get('raw_response') or '')}</pre></details>"
                "</div></article>"
            )
        sections.append(
            f"<section><h2>{html.escape(category.title())}</h2><div class='grid'>{''.join(cards)}</div></section>"
        )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Qwen3-VL-4B Int8 New Data Report {html.escape(timestamp)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background:#f6f3ee; color:#1f2937; margin:0; padding:24px; }}
.wrap {{ max-width: 1560px; margin: 0 auto; }}
h1 {{ margin:0 0 8px; font-size:32px; }}
h2 {{ margin:28px 0 14px; font-size:20px; }}
h3 {{ margin:0 0 8px; font-size:16px; }}
p, li {{ line-height:1.45; }}
.muted {{ color:#6b7280; }}
.hero {{ background:white; border:1px solid #e5ded3; border-radius:18px; padding:20px 22px; box-shadow:0 12px 30px rgba(0,0,0,0.06); }}
.metrics {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:18px; }}
.metric-card, .plot-card, .card {{ background:white; border:1px solid #e5ded3; border-radius:16px; box-shadow:0 10px 24px rgba(0,0,0,0.05); }}
.metric-card {{ padding:16px; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid #eee7de; }}
.plots {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(380px, 1fr)); gap:18px; margin-top:14px; }}
.plot-card {{ padding:12px; }}
.plot-card img {{ width:100%; border-radius:12px; display:block; }}
.plot-card figcaption {{ margin-top:8px; font-size:13px; color:#6b7280; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(320px, 1fr)); gap:16px; }}
.card {{ overflow:hidden; }}
.card img {{ width:100%; height:240px; object-fit:cover; display:block; background:#ece7df; }}
.meta {{ padding:14px; }}
.filename {{ font-size:13px; color:#6b7280; margin-bottom:10px; word-break:break-word; }}
.badges {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:10px; }}
.badge {{ font-size:12px; padding:5px 8px; border-radius:999px; background:#f3efe8; }}
.badge.gt {{ background:#e7f0ff; color:#1d4ed8; }}
.badge.ok {{ background:#e7f8ee; color:#15803d; }}
.badge.bad {{ background:#fdecec; color:#b91c1c; }}
.badge.neutral {{ background:#f3efe8; color:#6b7280; }}
details summary {{ cursor:pointer; font-size:13px; color:#374151; }}
pre {{ white-space:pre-wrap; background:#faf8f4; border:1px solid #eee7de; border-radius:10px; padding:10px; font-size:12px; line-height:1.4; overflow:auto; }}
@media (max-width: 900px) {{ .metrics {{ grid-template-columns:1fr; }} body {{ padding:16px; }} }}
</style>
</head>
<body>
<div class="wrap">
  <div class="hero">
    <h1>Qwen3-VL-4B Int8 New-Data Report</h1>
    <p class="muted">{html.escape(timestamp)} · model <strong>{html.escape(model_key)}</strong> · chair strategy <strong>{html.escape(report_variant['label'])}</strong> (`{html.escape(report_chair_key)}`)</p>
    <p>This report includes all generated plots plus every prediction card. Chairs are shown using the <strong>messy_refs_2</strong> prompting setup when present.</p>
  </div>

  <div class="metrics">
    {metric_block("Fixed Categories", compute_metrics([r for v in fixed_results.values() for r in v["records"]]))}
    {metric_block(f"Chair Strategy: {report_variant['label']}", report_variant["chair_metrics"])}
  </div>

  <section>
    <h2>Plots</h2>
    <div class="plots">{plot_html}</div>
  </section>

  {''.join(sections)}
</div>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-VL-4B int8 eval on new data/ folder with chair strategy sweep.")
    parser.add_argument("--config", default=str(ROOT / "benchmark_config.yaml"))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--retries", type=int, default=2, help="Retries on parse failure")
    parser.add_argument("--target-chair-accuracy", type=float, default=0.70)
    parser.add_argument("--max-chair-strategies", type=int, default=0, help="0 means evaluate all strategies")
    parser.add_argument("--chair-strategy", default="", help="Optional single chair strategy key to run, e.g. messy_refs_2")
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
    print(f"Target chair accuracy: {args.target_chair_accuracy:.2f}")

    model, model_cfg = load_qwen_model(args.config)
    print(f"Loading model: {model_cfg.key}")
    model.load()

    fixed_results = {}
    for subtask_key, prompt_cfg in FIXED_PROMPTS.items():
        rows = [s for s in samples if s["subtask"] == subtask_key]
        records = []
        print(f"\nRunning fixed prompt for {subtask_key}: {len(rows)} images")
        for sample in rows:
            ans, raw, lat, err = run_prompt(model, sample["image_path"], prompt_cfg["question"], args.retries)
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
    chair_strategies = build_chair_strategies()
    if args.chair_strategy:
        chair_strategies = [s for s in chair_strategies if s["key"] == args.chair_strategy]
        if not chair_strategies:
            raise SystemExit(f"Unknown chair strategy: {args.chair_strategy}")
    if args.max_chair_strategies > 0:
        chair_strategies = chair_strategies[:args.max_chair_strategies]

    for idx, strategy in enumerate(chair_strategies, start=1):
        print(f"\nRunning chair strategy: {strategy['label']} ({strategy['mode']})")
        records = []
        for sample in chair_rows:
            rec = evaluate_chair_strategy(model, sample, strategy, tmp_dir, args.retries, chair_rows)
            records.append(rec)
            print(
                f"  {sample['image_filename']}: pred={rec['predicted_label']} "
                f"gt={sample['label']} views={rec['n_views']} refs={rec['n_refs']}"
            )
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
        chair_acc = chair_variants[strategy["key"]]["chair_metrics"]["accuracy"]
        overall_acc = chair_variants[strategy["key"]]["combined_metrics"]["accuracy"]
        print(
            f"  -> strategy {idx}/{len(chair_strategies)} "
            f"chair_acc={chair_acc:.1%} overall_acc={overall_acc:.1%}"
        )
        if chair_acc >= args.target_chair_accuracy:
            print(
                f"Stopping early: {strategy['key']} reached chair accuracy "
                f"{chair_acc:.1%} >= {args.target_chair_accuracy:.1%}"
            )
            break

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
        "target_chair_accuracy": args.target_chair_accuracy,
        "retries": args.retries,
        "fixed_category_results": fixed_results,
        "chair_strategy_results": chair_variants,
        "best_chair_strategy_key": best_key,
        "best_chair_strategy_label": chair_variants[best_key]["label"],
        "target_hit": chair_variants[best_key]["chair_metrics"]["accuracy"] >= args.target_chair_accuracy,
    }
    out_path = out_dir / f"qwen3vl_4b_int8_newdata_env_monitoring_{ts}.json"
    out_path.write_text(json.dumps(out, indent=2))

    saved_plots = save_plots(out_dir, ts, fixed_results, chair_variants, best_key)
    report_key = "messy_refs_2" if "messy_refs_2" in chair_variants else best_key
    html_path = out_dir / f"qwen3vl_4b_int8_newdata_env_monitoring_{ts}.html"
    html_path.write_text(
        build_html_report(
            timestamp=ts,
            out_path=html_path,
            fixed_results=fixed_results,
            chair_variants=chair_variants,
            report_chair_key=report_key,
            plot_paths=saved_plots,
            model_key=model_cfg.key,
        )
    )
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
    print(f"HTML -> {html_path}")


if __name__ == "__main__":
    main()
