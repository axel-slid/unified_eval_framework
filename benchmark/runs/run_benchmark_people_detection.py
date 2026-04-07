#!/usr/bin/env python
"""
run_benchmark_people_detection.py — Benchmark YOLOv11n, YOLOv11s, and MobileNet SSD
for person detection, targeting the meeting-room use case.

Pipeline
--------
1. Download COCO128 (128 images, mix of COCO val scenes) via ultralytics.
   Labels are in YOLO format: class x_cx y_cy w h  (normalised 0-1).
2. For each model run:
   a. Timed inference on every image → latency statistics.
   b. Compare predicted boxes vs ground-truth using torchmetrics MeanAveragePrecision
      at IoU=0.50 (mAP@50) and IoU=0.75 (mAP@75), person-class only.
3. Save full results to benchmark/results/people_detection_<timestamp>.json.

Usage
-----
    cd benchmark/
    python run_benchmark_people_detection.py
    python run_benchmark_people_detection.py --models yolo11n yolo11s   # subset
    python run_benchmark_people_detection.py --conf 0.3 --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# ── project imports ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.yolov11 import YOLOv11Model
from models.mobilenet_ssd import MobileNetSSDModel
from models.base import DetectionResult

# ── COCO128 torchvision-label class ids for "person" ──────────────────────────
# In YOLO-label txt files class 0 = person (COCO 80-class ordering)
YOLO_PERSON_CLS   = 0
# torchvision SSD uses 1-indexed COCO labels; 1 = person
TV_PERSON_LABEL   = 1


# ──────────────────────────────────────────────────────────────────────────────
# COCO128 helpers
# ──────────────────────────────────────────────────────────────────────────────

def download_coco128() -> Path:
    """Download COCO128 via ultralytics and return the dataset root."""
    from ultralytics.data.utils import download
    from ultralytics.utils import DATASETS_DIR

    data_dir = Path(DATASETS_DIR) / "coco128"
    if not data_dir.exists():
        print("Downloading COCO128 dataset …")
        download(
            "https://ultralytics.com/assets/coco128.zip",
            dir=DATASETS_DIR,
            unzip=True,
        )
    return data_dir


def load_coco128_items(data_dir: Path) -> list[dict]:
    """Return list of {image_path, gt_boxes_xyxy, gt_labels} for images that
    have at least one annotation."""
    img_dir = data_dir / "images" / "train2017"
    lbl_dir = data_dir / "labels" / "train2017"

    items = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        lines = lbl_path.read_text().strip().splitlines()
        if not lines:
            continue

        # Read image dimensions to de-normalise boxes
        from PIL import Image as PILImage
        with PILImage.open(img_path) as img:
            W, H = img.size

        gt_boxes: list[list[float]] = []
        gt_labels: list[int] = []
        for ln in lines:
            parts = ln.split()
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = (cx - bw / 2) * W
            y1 = (cy - bh / 2) * H
            x2 = (cx + bw / 2) * W
            y2 = (cy + bh / 2) * H
            gt_boxes.append([x1, y1, x2, y2])
            gt_labels.append(cls)

        items.append({
            "image_path": str(img_path),
            "gt_boxes":   gt_boxes,
            "gt_labels":  gt_labels,
        })

    return items


# ──────────────────────────────────────────────────────────────────────────────
# mAP computation (person class only, IoU 0.50 and 0.75)
# ──────────────────────────────────────────────────────────────────────────────

def build_metric():
    """Return a fresh torchmetrics MeanAveragePrecision instance."""
    from torchmetrics.detection import MeanAveragePrecision
    return MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=[0.50, 0.75],
        rec_thresholds=None,          # use default 101-point interpolation
        extended_summary=False,
        class_metrics=True,
    )


def run_model_eval(
    model,
    items: list[dict],
    conf: float,
    device: str,
    person_cls_in_preds: int,       # class id the model uses for "person" (0=YOLO, 1=torchvision)
    use_all_classes: bool = False,
) -> dict:
    """
    Run inference on all items, collect latencies and mAP.

    Ground truth is always from COCO128 YOLO labels (person = class 0).
    person_cls_in_preds is the class id the model outputs for person detections.

    Returns a dict with keys:
        latencies_ms, map50, map75, map_avg, per_image_results
    """
    metric = build_metric()
    latencies: list[float] = []
    per_image: list[dict] = []
    errors = 0

    for item in items:
        img_path  = item["image_path"]
        gt_boxes  = item["gt_boxes"]
        gt_labels = item["gt_labels"]

        if use_all_classes:
            result: DetectionResult = model.detect_all_classes(img_path, conf_threshold=conf)
        else:
            result: DetectionResult = model.detect(img_path, conf_threshold=conf)

        if result.error:
            errors += 1
            continue

        latencies.append(result.latency_ms)

        # ── ground truth ──────────────────────────────────────────────────────
        # COCO128 labels are always in YOLO format: person = class 0.
        # We normalise to label=0 for torchmetrics (class-agnostic person eval).
        person_gt_boxes  = [b for b, l in zip(gt_boxes, gt_labels) if l == YOLO_PERSON_CLS]
        person_gt_labels = [0] * len(person_gt_boxes)

        # ── predictions ───────────────────────────────────────────────────────
        # Filter to person class; each model may use a different class id for person.
        person_preds = [
            d for d in result.detections
            if d.class_id == person_cls_in_preds or d.class_name == "person"
        ]
        pred_boxes  = [d.bbox       for d in person_preds]
        pred_scores = [d.confidence for d in person_preds]
        pred_labels = [0] * len(pred_boxes)

        # ── update metric ──
        preds = [{
            "boxes":  torch.tensor(pred_boxes,  dtype=torch.float32) if pred_boxes  else torch.zeros(0, 4),
            "scores": torch.tensor(pred_scores, dtype=torch.float32) if pred_scores else torch.zeros(0),
            "labels": torch.tensor(pred_labels, dtype=torch.int64)   if pred_labels else torch.zeros(0, dtype=torch.int64),
        }]
        targets = [{
            "boxes":  torch.tensor(person_gt_boxes,  dtype=torch.float32) if person_gt_boxes  else torch.zeros(0, 4),
            "labels": torch.tensor(person_gt_labels, dtype=torch.int64)   if person_gt_labels else torch.zeros(0, dtype=torch.int64),
        }]
        metric.update(preds, targets)

        per_image.append({
            "image":        img_path,
            "n_gt_persons": len(person_gt_boxes),
            "n_pred":       len(pred_boxes),
            "latency_ms":   result.latency_ms,
        })

    computed = metric.compute()

    return {
        "latencies_ms":     latencies,
        "mean_latency_ms":  float(np.mean(latencies)) if latencies else 0.0,
        "std_latency_ms":   float(np.std(latencies))  if latencies else 0.0,
        "fps":              1000.0 / float(np.mean(latencies)) if latencies else 0.0,
        "map50":            float(computed.get("map_50",  computed.get("map", torch.tensor(0.0)))),
        "map75":            float(computed.get("map_75",  torch.tensor(0.0))),
        "map_avg":          float(computed.get("map",     torch.tensor(0.0))),
        "n_images":         len(latencies),
        "n_errors":         errors,
        "per_image":        per_image,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "yolo11n":      {"cls": YOLOv11Model,    "kwargs": {"variant": "nano"},  "person_cls": YOLO_PERSON_CLS, "all_classes": True},
    "yolo11s":      {"cls": YOLOv11Model,    "kwargs": {"variant": "small"}, "person_cls": YOLO_PERSON_CLS, "all_classes": True},
    "mobilenet_ssd":{"cls": MobileNetSSDModel,"kwargs": {},                  "person_cls": TV_PERSON_LABEL,  "all_classes": True},
}


def main():
    parser = argparse.ArgumentParser(description="People detection benchmark")
    parser.add_argument("--models",  nargs="+", default=list(MODEL_CONFIGS.keys()),
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Which models to benchmark")
    parser.add_argument("--conf",    type=float, default=0.25,
                        help="Confidence threshold for detections (default: 0.25)")
    parser.add_argument("--device",  default=None,
                        help="Device: 'cuda' or 'cpu' (default: auto)")
    parser.add_argument("--out-dir", default=str(ROOT / "results"),
                        help="Directory to save results JSON")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Download / locate COCO128 ──────────────────────────────────────────
    print("\n=== Loading COCO128 dataset ===")
    data_dir = download_coco128()
    items    = load_coco128_items(data_dir)
    print(f"Found {len(items)} annotated images")

    # ── 2. Run each model ─────────────────────────────────────────────────────
    all_results: dict[str, dict] = {}

    for model_key in args.models:
        cfg = MODEL_CONFIGS[model_key]
        print(f"\n=== {model_key} ===")

        kwargs = dict(cfg["kwargs"])
        if "device" in cfg["cls"].__init__.__code__.co_varnames:
            kwargs["device"] = device

        model = cfg["cls"](**kwargs)
        model.load()

        eval_result = run_model_eval(
            model,
            items,
            conf=args.conf,
            device=device,
            person_cls_in_preds=cfg["person_cls"],
            use_all_classes=cfg["all_classes"],
        )

        model.unload()

        print(f"  mAP@50:        {eval_result['map50']:.4f}")
        print(f"  mAP@75:        {eval_result['map75']:.4f}")
        print(f"  Mean latency:  {eval_result['mean_latency_ms']:.1f} ms")
        print(f"  FPS:           {eval_result['fps']:.1f}")
        print(f"  Images:        {eval_result['n_images']}")

        all_results[model_key] = {
            "model_name":      model_key,
            "conf_threshold":  args.conf,
            "device":          device,
            "metrics": {
                "map50":           eval_result["map50"],
                "map75":           eval_result["map75"],
                "map_avg":         eval_result["map_avg"],
                "mean_latency_ms": eval_result["mean_latency_ms"],
                "std_latency_ms":  eval_result["std_latency_ms"],
                "fps":             eval_result["fps"],
                "n_images":        eval_result["n_images"],
                "n_errors":        eval_result["n_errors"],
            },
            "per_image":       eval_result["per_image"],
            "latencies_ms":    eval_result["latencies_ms"],
        }

    # ── 3. Save results ───────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out  = out_dir / f"people_detection_{ts}.json"
    payload = {
        "timestamp": ts,
        "dataset":   str(data_dir),
        "conf":      args.conf,
        "device":    device,
        "models":    all_results,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
