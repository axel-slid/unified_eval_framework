#!/usr/bin/env python
"""
run_pipeline_people_analysis.py — Two-stage people analysis pipeline.

Stage 1 (CV): All three detectors (YOLOv11n, YOLOv11s, MobileNet SSD) run on
              every image in people_images/ and return person bounding boxes.

Stage 2 (VLM): Each detected person is analysed by passing BOTH the full room
               image and the person crop to the VLM, enabling it to use spatial
               context (position at table, posture relative to room) alongside
               the close-up crop for:
                 Q1. Is this person a meeting participant?
                 Q2. Is this person currently talking / speaking?

               All VLMs are run: SmolVLM2-2.2B, InternVL3-4B (fp + int8),
               Qwen3-VL-4B (fp + int8), Qwen3-VL-8B (int8).

Results saved to benchmark/results/pipeline_people_<timestamp>.json
Crops saved to benchmark/results/crops/

Usage
-----
    cd benchmark/
    python run_pipeline_people_analysis.py
    python run_pipeline_people_analysis.py --vlm smolvlm qwen3vl_4b
    python run_pipeline_people_analysis.py --detector-for-crops yolo11s
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

PEOPLE_DIR  = ROOT.parent / "people_images"
RESULTS_DIR = ROOT / "results"
CROPS_DIR   = RESULTS_DIR / "crops"

# ── Prompts ───────────────────────────────────────────────────────────────────
PROMPT_PARTICIPANT = (
    "You are given two images: Image 1 is the full meeting room, Image 2 is a "
    "crop of one detected person from that room. "
    "Using both images, determine: is this person a genuine meeting participant "
    "(seated or standing at the table, engaged in the meeting)? "
    "Answer YES or NO, then give one short reason."
)

PROMPT_TALKING = (
    "You are given two images: Image 1 is the full meeting room, Image 2 is a "
    "crop of one detected person from that room. "
    "Using both images, is this person currently talking or speaking? "
    "Look at their mouth, facial expression, and body posture for cues. "
    "Answer YES or NO, then give one short reason."
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def parse_yes_no(text: str) -> bool | None:
    t = text.strip().upper()
    if t.startswith("YES"):
        return True
    if t.startswith("NO"):
        return False
    if "YES" in t[:20]:
        return True
    if "NO" in t[:20]:
        return False
    return None


def crop_with_padding(img: Image.Image, bbox: list[float], pad: float = 0.10) -> Image.Image:
    """Crop a bounding box from img with proportional padding."""
    W, H = img.size
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad, bh * pad
    cx1 = max(0, x1 - px)
    cy1 = max(0, y1 - py)
    cx2 = min(W, x2 + px)
    cy2 = min(H, y2 + py)
    return img.crop((cx1, cy1, cx2, cy2))


def save_crop(crop: Image.Image, stem: str, idx: int, detector: str) -> Path:
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    out = CROPS_DIR / f"{stem}__{detector}_person{idx:02d}.png"
    crop.save(out)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 — CV Detection
# ──────────────────────────────────────────────────────────────────────────────

DETECTOR_CONFIGS = {
    "yolo11n":       {"variant": "nano"},
    "yolo11s":       {"variant": "small"},
    "mobilenet_ssd": {},
}


def run_cv_stage(image_paths: list[Path], detectors: list[str], conf: float, device: str) -> dict:
    """Run all detectors on all images. Returns nested dict [detector][image_stem]."""
    from models.yolov11 import YOLOv11Model
    from models.mobilenet_ssd import MobileNetSSDModel

    results: dict[str, dict] = {}

    for det_key in detectors:
        print(f"\n[CV] {det_key}")
        cfg = DETECTOR_CONFIGS[det_key]

        if det_key.startswith("yolo"):
            model = YOLOv11Model(**cfg, device=device)
        else:
            model = MobileNetSSDModel(device=device)

        model.load()
        det_results = {}

        for img_path in image_paths:
            result = model.detect(str(img_path), conf_threshold=conf)
            det_results[img_path.stem] = {
                "image_path": str(img_path),
                "detections": [
                    {"bbox": d.bbox, "confidence": d.confidence}
                    for d in result.detections
                ] if not result.error else [],
                "n_persons":  len(result.detections) if not result.error else 0,
                "latency_ms": result.latency_ms,
                "error":      result.error,
            }
            status = f"  {img_path.name}: {det_results[img_path.stem]['n_persons']} people"
            if result.error:
                status += f" [ERROR: {result.error}]"
            print(status)

        model.unload()
        results[det_key] = det_results

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — Crop + VLM
# ──────────────────────────────────────────────────────────────────────────────

VLM_CONFIGS = {
    "smolvlm": {
        "class": "SmolVLMModel",
        "model_path": "/mnt/shared/dils/models/SmolVLM2-2.2B-Instruct",
        "dtype": "bfloat16",
    },
    "internvl": {
        "class": "InternVLModel",
        "model_path": "OpenGVLab/InternVL3_5-4B-HF",
        "dtype": "bfloat16",
    },
    "internvl_int8": {
        "class": "InternVLModel",
        "model_path": "/mnt/shared/dils/projects/logitech/unified_eval_framework/models/InternVL3_5-4B-HF-int8",
        "dtype": "bfloat16",
    },
    "qwen3vl_4b": {
        "class": "Qwen3VLModel",
        "model_path": "/mnt/shared/dils/models/Qwen3-VL-4B-Instruct",
        "dtype": "bfloat16",
    },
    "qwen3vl_4b_int8": {
        "class": "Qwen3VLModel",
        "model_path": "/mnt/shared/dils/projects/logitech/unified_eval_framework/models/Qwen3-VL-4B-Instruct-int8",
        "dtype": "bfloat16",
    },
    "qwen3vl_8b": {
        "class": "Qwen3VLModel",
        "model_path": "/mnt/shared/dils/projects/logitech/unified_eval_framework/models/Qwen3-VL-8B-Instruct",
        "dtype": "bfloat16",
    },
    "qwen3vl_8b_int8": {
        "class": "Qwen3VLModel",
        "model_path": "/mnt/shared/dils/projects/logitech/unified_eval_framework/models/Qwen3-VL-8B-Instruct-int8",
        "dtype": "bfloat16",
    },
}


def _make_model_cfg(key: str, vcfg: dict):
    from config import ModelConfig, GenerationConfig
    return ModelConfig(
        key=key,
        enabled=True,
        cls_name=vcfg["class"],
        model_path=vcfg["model_path"],
        dtype=vcfg["dtype"],
        generation=GenerationConfig(max_new_tokens=64, do_sample=False),
    )


def run_vlm_stage(
    image_paths: list[Path],
    cv_results: dict,
    detector_for_crops: str,
    vlm_keys: list[str],
) -> dict:
    """
    Crop person bboxes from detector_for_crops detections, run each VLM.
    Returns { vlm_key: { image_stem: [ per_person_dict ] } }.
    """
    from models import MODEL_REGISTRY

    # Build crop manifest (same crops reused across all VLMs)
    crop_manifest: dict[str, list[dict]] = {}  # image_stem → [{crop_path, bbox, ...}]
    det_data = cv_results[detector_for_crops]

    for img_path in image_paths:
        stem = img_path.stem
        img = Image.open(img_path).convert("RGB")
        persons = det_data.get(stem, {}).get("detections", [])
        crops = []
        for i, det in enumerate(persons):
            crop   = crop_with_padding(img, det["bbox"])
            c_path = save_crop(crop, stem, i, detector_for_crops)
            crops.append({
                "crop_path":  str(c_path),
                "bbox":       det["bbox"],
                "confidence": det["confidence"],
                "person_idx": i,
            })
        crop_manifest[stem] = crops

    vlm_results: dict[str, dict] = {}

    for vlm_key in vlm_keys:
        if vlm_key not in VLM_CONFIGS:
            print(f"  [VLM] Unknown key {vlm_key}, skipping")
            continue

        vcfg = VLM_CONFIGS[vlm_key]
        cls  = MODEL_REGISTRY.get(vcfg["class"])
        if cls is None:
            print(f"  [VLM] {vcfg['class']} not available, skipping")
            continue

        mcfg  = _make_model_cfg(vlm_key, vcfg)
        model = cls(mcfg)

        print(f"\n[VLM] {vlm_key}")
        model.load()

        vlm_image_results: dict[str, list] = {}

        for img_path in image_paths:
            stem      = img_path.stem
            crops     = crop_manifest[stem]
            full_path = str(img_path)
            person_results = []

            for crop_info in crops:
                crop_path = crop_info["crop_path"]
                p_res: dict = {**crop_info}

                # Q1: participant? — full room + crop
                r1 = model.run_two_image(full_path, crop_path, PROMPT_PARTICIPANT)
                p_res["participant_response"] = r1.response
                p_res["participant"]          = parse_yes_no(r1.response)
                p_res["participant_latency"]  = r1.latency_ms
                p_res["participant_error"]    = r1.error

                # Q2: talking? — full room + crop
                r2 = model.run_two_image(full_path, crop_path, PROMPT_TALKING)
                p_res["talking_response"] = r2.response
                p_res["talking"]          = parse_yes_no(r2.response)
                p_res["talking_latency"]  = r2.latency_ms
                p_res["talking_error"]    = r2.error

                person_results.append(p_res)
                print(
                    f"  {img_path.name} person {crop_info['person_idx']:02d} → "
                    f"participant={'Y' if p_res['participant'] else 'N' if p_res['participant'] is False else '?'}  "
                    f"talking={'Y' if p_res['talking'] else 'N' if p_res['talking'] is False else '?'}"
                )

            vlm_image_results[stem] = person_results

        model.unload()
        vlm_results[vlm_key] = vlm_image_results

    return vlm_results, crop_manifest


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detectors", nargs="+",
                        default=["yolo11n", "yolo11s", "mobilenet_ssd"],
                        choices=list(DETECTOR_CONFIGS.keys()))
    parser.add_argument("--detector-for-crops", default="yolo11s",
                        help="Which detector's boxes to crop for VLM analysis")
    parser.add_argument("--vlm", nargs="+",
                        default=list(VLM_CONFIGS.keys()),
                        choices=list(VLM_CONFIGS.keys()))
    parser.add_argument("--conf",     type=float, default=0.30)
    parser.add_argument("--device",   default=None)
    parser.add_argument("--cv-json",  default=None,
                        help="Reuse CV results from a previous run (skip Stage 1)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    image_paths = sorted(PEOPLE_DIR.glob("*.jpeg")) + sorted(PEOPLE_DIR.glob("*.jpg")) + sorted(PEOPLE_DIR.glob("*.png"))
    print(f"Images: {len(image_paths)}")

    # ── Stage 1: CV detection (or load cached) ────────────────────────────────
    if args.cv_json:
        print(f"\n====== Stage 1: Loading CV results from {args.cv_json} ======")
        with open(args.cv_json) as f:
            prev = json.load(f)
        cv_results = prev["cv_results"]
    else:
        print("\n====== Stage 1: CV Detection ======")
        cv_results = run_cv_stage(image_paths, args.detectors, args.conf, device)

    # ── Stage 2: VLM crop analysis ────────────────────────────────────────────
    print("\n====== Stage 2: VLM Crop Analysis ======")
    vlm_results, crop_manifest = run_vlm_stage(
        image_paths, cv_results, args.detector_for_crops, args.vlm
    )

    # ── Save results ──────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"pipeline_people_{ts}.json"

    payload = {
        "timestamp":          ts,
        "device":             device,
        "conf":               args.conf,
        "detector_for_crops": args.detector_for_crops,
        "images":             [str(p) for p in image_paths],
        "cv_results":         cv_results,
        "crop_manifest":      crop_manifest,
        "vlm_results":        vlm_results,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
