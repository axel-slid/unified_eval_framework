#!/usr/bin/env python
"""
face_detection/run_pipeline.py — Two-stage face analysis pipeline.

Stage 1 (CV): YOLOv8-Face detects face bounding boxes in every image.
              Each bbox is dilated 2× from its centre before cropping,
              so the crop includes neck/shoulder context around the face.

Stage 2 (VLM): Each dilated crop + full room image is passed to Qwen3-VL-4B:
                 Q1. Is this person a meeting participant?
                 Q2. Is this person currently talking / speaking?

Results saved to face_detection/results/pipeline_<timestamp>.json
Crops    saved to face_detection/results/pipeline_crops/

Usage
-----
    cd benchmark/face_detection/
    python run_pipeline.py
    python run_pipeline.py --vlm qwen3vl_4b --conf 0.3
    python run_pipeline.py --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image as PILImage

HERE        = Path(__file__).parent
BENCH_ROOT  = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BENCH_ROOT))   # for benchmark/models (VLMs)

PEOPLE_DIR  = HERE.parent.parent / "people_images"
RESULTS_DIR = HERE / "results"
CROPS_DIR   = RESULTS_DIR / "pipeline_crops"
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Prompts (same as people-analysis pipeline) ─────────────────────────────────
PROMPT_PARTICIPANT = (
    "You are given two images: Image 1 is the full meeting room, Image 2 is a "
    "crop of one detected face from that room. "
    "Using both images, determine: is this person a genuine meeting participant "
    "(seated or standing at the table, engaged in the meeting)? "
    "Answer YES or NO, then give one short reason."
)

PROMPT_TALKING = (
    "You are given two images: Image 1 is the full meeting room, Image 2 is a "
    "crop of one detected face from that room. "
    "Using both images, is this person currently talking or speaking? "
    "Look at their mouth, facial expression, and body posture for cues. "
    "Answer YES or NO, then give one short reason."
)

VLM_CONFIGS = {
    "qwen3vl_4b": {
        "class":      "Qwen3VLModel",
        "model_path": "/mnt/shared/dils/models/Qwen3-VL-4B-Instruct",
        "dtype":      "bfloat16",
    },
    "qwen3vl_4b_int8": {
        "class":      "Qwen3VLModel",
        "model_path": "/mnt/shared/dils/projects/logitech/unified_eval_framework/models/Qwen3-VL-4B-Instruct-int8",
        "dtype":      "bfloat16",
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_yes_no(text: str) -> bool | None:
    t = text.strip().upper()
    if t.startswith("YES"):   return True
    if t.startswith("NO"):    return False
    if "YES" in t[:20]:       return True
    if "NO"  in t[:20]:       return False
    return None


def dilate_bbox(bbox: list[float], scale: float, W: int, H: int) -> list[float]:
    """Expand bbox by `scale` from its centre, clamped to image bounds."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw = (x2 - x1) / 2 * scale
    hh = (y2 - y1) / 2 * scale
    return [
        max(0,   cx - hw),
        max(0,   cy - hh),
        min(W,   cx + hw),
        min(H,   cy + hh),
    ]


def save_crop(img: PILImage.Image, bbox: list[float], stem: str, idx: int) -> Path:
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop = img.crop((x1, y1, x2, y2))
    out  = CROPS_DIR / f"{stem}__face{idx:02d}.png"
    crop.save(out)
    return out


# ── Stage 1 ───────────────────────────────────────────────────────────────────

def run_cv_stage(
    image_paths: list[Path],
    conf: float,
    device: str,
    dilate: float = 2.0,
) -> dict:
    from models.yolov8_face import YOLOv8FaceModel

    print(f"\n[Stage 1] YOLOv8-Face  (conf={conf}, dilation={dilate}×)")
    model = YOLOv8FaceModel(device=device)
    model.load()

    cv_results: dict[str, dict] = {}

    for img_path in image_paths:
        img        = PILImage.open(img_path).convert("RGB")
        W, H       = img.size
        result     = model.detect(str(img_path), conf_threshold=conf)
        stem       = img_path.stem

        dilated_dets = []
        crops        = []
        for i, det in enumerate(result.detections):
            dbox     = dilate_bbox(det.bbox, dilate, W, H)
            crop_path = save_crop(img, dbox, stem, i)
            dilated_dets.append({
                "original_bbox": det.bbox,
                "dilated_bbox":  dbox,
                "confidence":    det.confidence,
                "face_idx":      i,
                "crop_path":     str(crop_path),
            })

        cv_results[stem] = {
            "image_path": str(img_path),
            "n_faces":    len(dilated_dets),
            "detections": dilated_dets,
            "latency_ms": result.latency_ms,
            "error":      result.error,
        }
        print(f"  {img_path.name}: {len(dilated_dets)} face(s)  {result.latency_ms:.1f} ms")

    model.unload()
    return cv_results


# ── Stage 2 ───────────────────────────────────────────────────────────────────

def run_vlm_stage(
    image_paths: list[Path],
    cv_results: dict,
    vlm_keys: list[str],
) -> dict:
    from models import MODEL_REGISTRY
    from config import ModelConfig, GenerationConfig

    vlm_results: dict[str, dict] = {}

    for vlm_key in vlm_keys:
        if vlm_key not in VLM_CONFIGS:
            print(f"  [VLM] Unknown key {vlm_key!r}, skipping")
            continue
        vcfg = VLM_CONFIGS[vlm_key]
        cls  = MODEL_REGISTRY.get(vcfg["class"])
        if cls is None:
            print(f"  [VLM] {vcfg['class']} not in registry, skipping")
            continue

        mcfg  = ModelConfig(
            key=vlm_key, enabled=True,
            cls_name=vcfg["class"], model_path=vcfg["model_path"],
            dtype=vcfg["dtype"],
            generation=GenerationConfig(max_new_tokens=64, do_sample=False),
        )
        model = cls(mcfg)
        print(f"\n[Stage 2] {vlm_key}")
        model.load()

        img_results: dict[str, list] = {}

        for img_path in image_paths:
            stem      = img_path.stem
            full_path = str(img_path)
            dets      = cv_results.get(stem, {}).get("detections", [])
            persons   = []

            for det in dets:
                crop_path = det["crop_path"]
                rec: dict = {**det}

                r1 = model.run_two_image(full_path, crop_path, PROMPT_PARTICIPANT)
                rec["participant_response"] = r1.response
                rec["participant"]          = parse_yes_no(r1.response)
                rec["participant_latency"]  = r1.latency_ms
                rec["participant_error"]    = r1.error

                r2 = model.run_two_image(full_path, crop_path, PROMPT_TALKING)
                rec["talking_response"] = r2.response
                rec["talking"]          = parse_yes_no(r2.response)
                rec["talking_latency"]  = r2.latency_ms
                rec["talking_error"]    = r2.error

                persons.append(rec)
                print(
                    f"  {img_path.name}  face {det['face_idx']:02d} → "
                    f"participant={'Y' if rec['participant'] else 'N' if rec['participant'] is False else '?'}  "
                    f"talking={'Y' if rec['talking'] else 'N' if rec['talking'] is False else '?'}"
                )

            img_results[stem] = persons

        model.unload()
        vlm_results[vlm_key] = img_results

    return vlm_results


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images",  default=str(PEOPLE_DIR))
    parser.add_argument("--vlm",     nargs="+", default=["qwen3vl_4b"],
                        choices=list(VLM_CONFIGS.keys()))
    parser.add_argument("--conf",    type=float, default=0.25)
    parser.add_argument("--dilate",  type=float, default=2.0,
                        help="Dilation factor for face bbox before cropping (default: 2.0)")
    parser.add_argument("--device",  default=None)
    parser.add_argument("--cv-json", default=None,
                        help="Reuse CV results JSON from a previous run (skip Stage 1)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    images_dir = Path(args.images)
    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    print(f"Device: {device}  |  Images: {len(image_paths)}")

    # Stage 1
    if args.cv_json:
        print(f"\n[Stage 1] Loading cached CV results from {args.cv_json}")
        prev       = json.loads(Path(args.cv_json).read_text())
        cv_results = prev["cv_results"]
    else:
        cv_results = run_cv_stage(image_paths, args.conf, device, args.dilate)

    # Stage 2
    vlm_results = run_vlm_stage(image_paths, cv_results, args.vlm)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"pipeline_{ts}.json"
    out.write_text(json.dumps({
        "timestamp":  ts,
        "device":     device,
        "conf":       args.conf,
        "dilate":     args.dilate,
        "images":     [str(p) for p in image_paths],
        "cv_results": cv_results,
        "vlm_results": vlm_results,
    }, indent=2))
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
