#!/usr/bin/env python
"""
run_meeting_participant_gt_vlm_eval.py — Evaluate VLM participant classification
using the ground-truth person boxes from meeting_participation_dataset/allBoxes.json.

For each annotated person:
1. Crop the GT bbox from the full image (with optional padding).
2. Ask the VLM whether this person is a meeting participant or non-participant.
3. Compare against the annotated role and report accuracy/F1/confusion stats.

This isolates classification quality from detection quality because the boxes come
directly from the dataset annotations rather than from a detector.

Example:
    cd benchmark
    python runs/run_meeting_participant_gt_vlm_eval.py \
      --models qwen3vl_4b qwen3vl_8b qwen3vl_4b_int8 qwen3vl_8b_int8 gemma_e2b_8bit_hf
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from config import BenchmarkConfig, GenerationConfig, ModelConfig, load_config
from models import MODEL_REGISTRY


DATASET_JSON = PROJECT_ROOT / "meeting_participation_dataset" / "allBoxes.json"
DATASET_DIR = PROJECT_ROOT / "meeting_participation_dataset"
RESULTS_DIR = ROOT / "results"
CROPS_DIR = RESULTS_DIR / "meeting_participant_gt_crops"

PROMPT_PARTICIPANT = (
    "You are given two images: Image 1 is the full meeting room, Image 2 is a crop of one annotated person "
    "from that room. Determine whether this person is a genuine meeting participant.\n\n"
    "Definitions:\n"
    '- "participant": seated or standing at the table, engaged in the meeting.\n'
    '- "non-participant": in the background, passing by, or not engaged in the meeting.\n\n'
    'Answer with exactly one word first: "PARTICIPANT" or "NON-PARTICIPANT". '
    "Then give one short reason."
)


DEFAULT_MODELS = [
    "qwen3vl_4b",
    "qwen3vl_8b",
    "qwen3vl_4b_int8",
    "qwen3vl_8b_int8",
    "gemma_e2b_8bit_hf",
]


def crop_with_padding(img: Image.Image, bbox: list[float], pad: float = 0.10) -> Image.Image:
    width, height = img.size
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad, bh * pad
    cx1 = max(0, x1 - px)
    cy1 = max(0, y1 - py)
    cx2 = min(width, x2 + px)
    cy2 = min(height, y2 + py)
    return img.crop((cx1, cy1, cx2, cy2))


def parse_participant_label(text: str) -> bool | None:
    t = text.strip().upper()
    if t.startswith("NON-PARTICIPANT") or t.startswith("NON PARTICIPANT"):
        return False
    if t.startswith("PARTICIPANT"):
        return True
    head = t[:40]
    if "NON-PARTICIPANT" in head or "NON PARTICIPANT" in head:
        return False
    if "PARTICIPANT" in head:
        return True
    return None


def load_samples() -> list[dict]:
    payload = json.loads(DATASET_JSON.read_text())
    samples: list[dict] = []
    for ann in payload["annotations"]:
        image_path = DATASET_DIR / ann["file_name"]
        with Image.open(image_path) as img:
            width, height = img.size
        samples.append(
            {
                "image_name": ann["file_name"],
                "image_path": image_path,
                "width": width,
                "height": height,
                "people": ann["people"],
            }
        )
    return samples


def build_model_lookup(cfg: BenchmarkConfig) -> dict[str, ModelConfig]:
    return {model.key: model for model in cfg.enabled_models}


def save_gt_crops(samples: list[dict], crop_pad: float) -> list[dict]:
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for sample in samples:
        image = Image.open(sample["image_path"]).convert("RGB")
        stem = Path(sample["image_name"]).stem

        for idx, person in enumerate(sample["people"]):
            crop = crop_with_padding(image, person["bbox"], pad=crop_pad)
            crop_path = CROPS_DIR / f"{stem}__gt_person{idx:02d}.png"
            crop.save(crop_path)
            manifest.append(
                {
                    "image_name": sample["image_name"],
                    "image_path": str(sample["image_path"]),
                    "crop_path": str(crop_path),
                    "person_idx": idx,
                    "person_id": person["id"],
                    "bbox": person["bbox"],
                    "gt_role": person["role"],
                }
            )

    return manifest


def compute_metrics(records: list[dict]) -> dict:
    total = len(records)
    valid = [r for r in records if r["predicted_participant"] is not None]
    correct = sum(1 for r in valid if r["correct"])

    tp = sum(1 for r in valid if r["predicted_participant"] is True and r["gt_participant"] is True)
    tn = sum(1 for r in valid if r["predicted_participant"] is False and r["gt_participant"] is False)
    fp = sum(1 for r in valid if r["predicted_participant"] is True and r["gt_participant"] is False)
    fn = sum(1 for r in valid if r["predicted_participant"] is False and r["gt_participant"] is True)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = correct / len(valid) if valid else 0.0

    role_breakdown = {}
    for role_name, role_value in [("participant", True), ("non-participant", False)]:
        subset = [r for r in valid if r["gt_participant"] is role_value]
        role_breakdown[role_name] = {
            "n": len(subset),
            "accuracy": (sum(1 for r in subset if r["correct"]) / len(subset)) if subset else 0.0,
        }

    latencies = [r["latency_ms"] for r in valid if r["latency_ms"] is not None]

    return {
        "n_total": total,
        "n_valid": len(valid),
        "n_unparsed": total - len(valid),
        "accuracy": accuracy,
        "precision_participant": precision,
        "recall_participant": recall,
        "f1_participant": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "mean_latency_ms": float(statistics.mean(latencies)) if latencies else 0.0,
        "participant_breakdown": role_breakdown,
    }


def evaluate_model(model_cfg: ModelConfig, crop_manifest: list[dict]) -> dict:
    cls = MODEL_REGISTRY.get(model_cfg.cls_name)
    if cls is None:
        raise RuntimeError(f"Model class {model_cfg.cls_name!r} not available for {model_cfg.key}")

    model = cls(model_cfg)
    print(f"\n=== {model_cfg.key} ===")
    model.load()

    records: list[dict] = []
    for item in crop_manifest:
        gt_participant = item["gt_role"] == "participant"
        result = model.run_two_image(item["image_path"], item["crop_path"], PROMPT_PARTICIPANT)
        pred_participant = None if result.error else parse_participant_label(result.response)
        correct = (pred_participant == gt_participant) if pred_participant is not None else False

        records.append(
            {
                **item,
                "gt_participant": gt_participant,
                "predicted_participant": pred_participant,
                "correct": correct,
                "response": result.response,
                "latency_ms": result.latency_ms,
                "error": result.error,
            }
        )

        pred_str = (
            "participant" if pred_participant is True else
            "non-participant" if pred_participant is False else
            "unparsed"
        )
        print(
            f"  {item['image_name']} {item['person_id']}: "
            f"gt={item['gt_role']} pred={pred_str} "
            f"{'OK' if correct else 'MISS'}"
        )

    model.unload()
    return {
        "model_key": model_cfg.key,
        "model_path": model_cfg.model_path,
        "dtype": model_cfg.dtype,
        "metrics": compute_metrics(records),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VLM participant classification on GT boxes.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model keys from benchmark_config.yaml",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "benchmark_config.yaml"),
        help="Path to benchmark_config.yaml",
    )
    parser.add_argument("--crop-pad", type=float, default=0.10)
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_lookup = build_model_lookup(cfg)
    missing = [m for m in args.models if m not in model_lookup]
    if missing:
        raise SystemExit(
            "These model keys are not enabled or not present in benchmark_config.yaml: "
            + ", ".join(missing)
        )

    samples = load_samples()
    crop_manifest = save_gt_crops(samples, crop_pad=args.crop_pad)
    print(f"Loaded {len(samples)} images and {len(crop_manifest)} GT person crops from {DATASET_JSON}")

    all_results = {
        "dataset": str(DATASET_JSON),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "crop_pad": args.crop_pad,
        "models": {},
    }

    for model_key in args.models:
        model_cfg = model_lookup[model_key]
        # Keep responses short and deterministic for classification.
        model_cfg = ModelConfig(
            key=model_cfg.key,
            enabled=model_cfg.enabled,
            cls_name=model_cfg.cls_name,
            model_path=model_cfg.model_path,
            dtype=model_cfg.dtype,
            generation=GenerationConfig(max_new_tokens=32, do_sample=False),
        )
        all_results["models"][model_key] = evaluate_model(model_cfg, crop_manifest)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"meeting_participant_gt_vlm_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(all_results, indent=2))

    print(f"\nSaved results to {out_path}")
    print("\nSummary:")
    for model_key in args.models:
        metrics = all_results["models"][model_key]["metrics"]
        print(
            f"  {model_key}: acc={metrics['accuracy']:.3f} "
            f"f1={metrics['f1_participant']:.3f} "
            f"valid={metrics['n_valid']}/{metrics['n_total']}"
        )


if __name__ == "__main__":
    main()
