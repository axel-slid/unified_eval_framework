#!/usr/bin/env python
"""
finetune/eval_finetuned.py — Compare base Qwen3-VL-4B vs LoRA fine-tuned model
on person bbox detection.

Metrics
-------
For each image we compute:
  • parse_success      — did the model output valid JSON in our schema?
  • n_people_pred      — how many people did the model detect?
  • n_people_gt        — ground-truth person count (from val JSONL)
  • count_error        — |n_people_pred - n_people_gt|
  • mean_iou           — mean IoU between matched GT/pred boxes (Hungarian matching)
  • bbox_format_valid  — all predicted bboxes are [x1<x2, y1<y2, in-bounds]

Usage
-----
  cd finetune/
  # compare base vs adapter
  python eval_finetuned.py --adapter checkpoints/qwen3vl_bbox_lora/final

  # base only
  python eval_finetuned.py --base-only

  # limit images for a quick check
  python eval_finetuned.py --adapter checkpoints/qwen3vl_bbox_lora/final --max-images 50
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info

ROOT      = Path(__file__).parent
DATA_DIR  = ROOT / "data"
MODEL_PATH = "/mnt/shared/dils/models/Qwen3-VL-4B-Instruct"

# Prompt used for the fine-tuned model (bbox-only, matches training format)
DETECTION_PROMPT_BBOX = (
    "Detect ALL people in this {width}x{height} image.\n"
    "Output ONLY valid JSON, no extra text:\n"
    '{{"people": [{{"id": "P1", "bbox": [x1, y1, x2, y2]}}, ...]}}\n'
    "Bounding boxes must be pixel coordinates [x1, y1, x2, y2] within the image."
)

# Prompt used for the base model (full task, no fine-tuning)
DETECTION_PROMPT_FULL = (
    "You are analyzing a meeting room image ({width}x{height} pixels).\n\n"
    "Your tasks:\n"
    "1. Detect ALL people visible in the image.\n"
    "2. For each person, label their role:\n"
    '   - "participant": seated or standing at the table, engaged with the meeting.\n'
    '   - "non-participant": passing by, in background, not engaged.\n'
    "3. Identify who is the TARGET SPEAKER. Mark at most one is_target_speaker=true.\n\n"
    "Output ONLY valid JSON, no extra text:\n"
    '{{"people": [{{"id": "P1", "bbox": [x1, y1, x2, y2], '
    '"role": "participant", "is_target_speaker": false, '
    '"reason": "short reason"}}, ...]}}\n\n'
    "Bounding boxes must be pixel coordinates [x1, y1, x2, y2] within the "
    "image ({width}x{height}). Do not output any text outside the JSON object."
)


# ── IoU helpers ────────────────────────────────────────────────────────────────

def iou(boxA: list, boxB: list) -> float:
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    aA = (ax2 - ax1) * (ay2 - ay1)
    bA = (bx2 - bx1) * (by2 - by1)
    return inter / (aA + bA - inter)


def hungarian_mean_iou(gt_boxes: list, pred_boxes: list) -> float:
    if not gt_boxes or not pred_boxes:
        return 0.0
    n, m = len(gt_boxes), len(pred_boxes)
    cost = np.zeros((n, m))
    for i, g in enumerate(gt_boxes):
        for j, p in enumerate(pred_boxes):
            cost[i, j] = 1.0 - iou(g, p)
    row_ind, col_ind = linear_sum_assignment(cost)
    ious = [1.0 - cost[r, c] for r, c in zip(row_ind, col_ind)]
    return float(np.mean(ious)) if ious else 0.0


# ── JSON parsing (same as run_approach_a_vlm_only.py) ─────────────────────────

def extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    fenced = re.sub(r"\s*```$", "", fenced, flags=re.MULTILINE).strip()
    try:
        return json.loads(fenced)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ── Model wrapper ──────────────────────────────────────────────────────────────

class QwenInferenceModel:
    def __init__(self, model_path: str, adapter_path: str | None = None):
        os.environ.setdefault("HF_HOME", "/mnt/shared/dils/hf_cache")
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        ).eval()
        if adapter_path:
            print(f"[eval] Loading LoRA adapter from {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.eval()
        self.name = "base" if adapter_path is None else "lora"

    def run(self, image_path: str, width: int, height: int) -> tuple[str, float]:
        prompt_tmpl = DETECTION_PROMPT_BBOX if self.name == "lora" else DETECTION_PROMPT_FULL
        prompt = prompt_tmpl.format(width=width, height=height)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text",  "text": prompt},
        ]}]
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        device = next(self.model.parameters()).device
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        latency = (time.perf_counter() - t0) * 1000
        trimmed = [ids[0][inputs.input_ids.shape[1]:]]
        response = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return response, latency


# ── Evaluation loop ────────────────────────────────────────────────────────────

def evaluate(model: QwenInferenceModel, val_jsonl: Path, max_images: int | None) -> dict:
    examples = []
    with open(val_jsonl) as f:
        for i, line in enumerate(f):
            if max_images and i >= max_images:
                break
            examples.append(json.loads(line))

    metrics = {
        "parse_success": [],
        "count_error":   [],
        "mean_iou":      [],
        "bbox_valid":    [],
        "latency_ms":    [],
        "n_people_gt":   [],
        "n_people_pred": [],
    }

    for ex in examples:
        msgs = ex["messages"]
        # Ground-truth: parse assistant reply
        gt_reply = msgs[1]["content"]
        gt_parsed = extract_json(gt_reply)
        gt_people = gt_parsed.get("people", []) if gt_parsed else []
        gt_boxes  = [p["bbox"] for p in gt_people if p.get("bbox")]

        # Image info from user content
        img_path = msgs[0]["content"][0]["image"]
        try:
            with Image.open(img_path) as im:
                W, H = im.size
        except Exception:
            continue

        response, latency = model.run(img_path, W, H)
        parsed = extract_json(response)

        parse_ok = parsed is not None and "people" in parsed
        metrics["parse_success"].append(float(parse_ok))
        metrics["latency_ms"].append(latency)
        metrics["n_people_gt"].append(len(gt_boxes))

        if not parse_ok:
            metrics["count_error"].append(len(gt_boxes))
            metrics["mean_iou"].append(0.0)
            metrics["bbox_valid"].append(0.0)
            metrics["n_people_pred"].append(0)
            continue

        pred_people = parsed.get("people", [])
        pred_boxes  = []
        all_valid   = True
        for p in pred_people:
            bb = p.get("bbox")
            if isinstance(bb, list) and len(bb) == 4:
                x1, y1, x2, y2 = bb
                if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= W and y2 <= H:
                    pred_boxes.append([x1, y1, x2, y2])
                else:
                    all_valid = False
            else:
                all_valid = False

        metrics["n_people_pred"].append(len(pred_boxes))
        metrics["count_error"].append(abs(len(pred_boxes) - len(gt_boxes)))
        metrics["mean_iou"].append(hungarian_mean_iou(gt_boxes, pred_boxes))
        metrics["bbox_valid"].append(float(all_valid and len(pred_boxes) > 0))

    summary = {k: float(np.mean(v)) for k, v in metrics.items() if v}
    summary["n_evaluated"] = len(examples)
    return summary


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None,
                        help="Path to LoRA adapter directory (fine-tuned model)")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--val-file", default=str(DATA_DIR / "coco_val.jsonl"))
    args = parser.parse_args()

    val_path = Path(args.val_file)
    if not val_path.exists():
        print(f"Val file not found: {val_path}")
        sys.exit(1)

    results = {}

    print("\n=== Evaluating BASE model ===")
    base_model = QwenInferenceModel(MODEL_PATH, adapter_path=None)
    results["base"] = evaluate(base_model, val_path, args.max_images)
    del base_model
    torch.cuda.empty_cache()

    if not args.base_only and args.adapter:
        print("\n=== Evaluating FINE-TUNED model ===")
        ft_model = QwenInferenceModel(MODEL_PATH, adapter_path=args.adapter)
        results["finetuned"] = evaluate(ft_model, val_path, args.max_images)
        del ft_model
        torch.cuda.empty_cache()

    # Print comparison table
    print("\n" + "=" * 65)
    print(f"{'Metric':<25} {'Base':>12} {'Fine-tuned':>12}")
    print("-" * 65)
    keys = ["parse_success", "count_error", "mean_iou", "bbox_valid",
            "latency_ms", "n_people_gt", "n_people_pred", "n_evaluated"]
    for k in keys:
        base_val = results["base"].get(k, float("nan"))
        ft_val   = results.get("finetuned", {}).get(k, float("nan"))
        print(f"{k:<25} {base_val:>12.3f} {ft_val:>12.3f}")
    print("=" * 65)

    out = ROOT / "eval_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
