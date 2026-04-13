#!/usr/bin/env python
"""
Evaluate the iteration2 participant-face dataset with:
1. YOLOv8-Head detection against annotated face/head boxes.
2. Qwen3-VL participant vs non-participant classification on matched head crops.
3. A dilation sweep over the detected boxes used for VLM crops.

Dataset expected:
  data/iteration2/manual_face_annotations.json
  data/iteration2/*.png

Outputs:
  data/iteration2/eval_results/head_qwen3vl_4b_iteration2/<timestamp>/
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

GPU_ID = os.environ.get("GPU_ID")
if GPU_ID and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ROOT.parent
DATA_ROOT = PROJECT_ROOT / "data" / "iteration2"
ANNOTATIONS_PATH = DATA_ROOT / "manual_face_annotations.json"
DEFAULT_OUT_ROOT = DATA_ROOT / "eval_results" / "head_qwen3vl_4b_iteration2"

sys.path.insert(0, str(ROOT))

from config import GenerationConfig, ModelConfig, load_config
from models import MODEL_REGISTRY
from models.base import BaseDetectionModel, Detection, DetectionResult


PROMPT_PARTICIPANT = (
    "You are given two images. Image 1 is the full meeting scene. "
    "Image 2 is a crop around one detected head/face from that scene.\n\n"
    "Decide whether the cropped person is a genuine meeting participant.\n"
    '- "participant": seated or standing with the meeting group, engaged in the meeting.\n'
    '- "non-participant": in the background, walking by, not engaged, or not part of the meeting.\n\n'
    "Answer with exactly one word on the final line: participant or non-participant"
)

WEIGHT_URLS = {
    "nano": "https://github.com/Abcfsa/YOLOv8_head_detector/raw/main/nano.pt",
    "medium": "https://github.com/Abcfsa/YOLOv8_head_detector/raw/main/medium.pt",
}


@dataclass
class GTPerson:
    person_id: str
    bbox: list[float]
    role: str


@dataclass
class Sample:
    file_name: str
    image_path: Path
    people: list[GTPerson]


class YOLOv8HeadModel(BaseDetectionModel):
    name = "YOLOv8-Head"

    def __init__(self, variant: str = "nano", device: str | None = None):
        self.variant = variant
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = ROOT / "face_detection" / "weights" / f"yolov8_{variant}_head.pt"
        self.model = None

    def _ensure_weights(self) -> None:
        if self.weights.exists():
            return
        self.weights.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(WEIGHT_URLS[self.variant], self.weights)

    def load(self) -> None:
        from ultralytics import YOLO

        self._ensure_weights()
        self.model = YOLO(str(self.weights))
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self.model(dummy, device=self._device, verbose=False)

    def detect(self, image_path: str, conf_threshold: float = 0.25) -> DetectionResult:
        try:
            t0 = time.perf_counter()
            results = self.model(image_path, conf=conf_threshold, device=self._device, verbose=False)
            latency_ms = (time.perf_counter() - t0) * 1000
            detections: list[Detection] = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append(
                        Detection(
                            bbox=[x1, y1, x2, y2],
                            confidence=float(box.conf[0]),
                            class_id=int(box.cls[0]) if box.cls is not None else 0,
                            class_name="head",
                        )
                    )
            return DetectionResult(detections=detections, latency_ms=latency_ms)
        except Exception as exc:
            return DetectionResult(detections=[], latency_ms=0.0, error=str(exc))

    def unload(self) -> None:
        del self.model
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_samples(max_images: int = 0) -> list[Sample]:
    data = json.loads(ANNOTATIONS_PATH.read_text())
    samples: list[Sample] = []
    for item in data["annotations"]:
        image_path = DATA_ROOT / item["file_name"]
        if not image_path.exists():
            continue
        people = [
            GTPerson(person_id=p["id"], bbox=[float(v) for v in p["bbox"]], role=p["role"])
            for p in item.get("people", [])
        ]
        samples.append(Sample(file_name=item["file_name"], image_path=image_path, people=people))
    if max_images > 0:
        samples = samples[:max_images]
    return samples


def load_qwen_model(config_path: str, model_key: str = "qwen3vl_4b"):
    cfg = load_config(config_path)
    mcfg = next((m for m in cfg.enabled_models if m.key == model_key), None)
    if mcfg is None:
        raise SystemExit(f"{model_key} not enabled in benchmark_config.yaml")
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


def compute_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def greedy_match(
    gt_boxes: list[list[float]],
    pred_boxes: list[list[float]],
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    candidates = []
    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            iou = compute_iou(gt, pred)
            if iou >= iou_threshold:
                candidates.append((iou, gi, pi))
    candidates.sort(reverse=True)

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append((gi, pi, iou))
    return matches


def dilate_bbox(bbox: list[float], scale: float, width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw = (x2 - x1) * scale / 2.0
    hh = (y2 - y1) * scale / 2.0
    return [
        int(max(0, math.floor(cx - hw))),
        int(max(0, math.floor(cy - hh))),
        int(min(width, math.ceil(cx + hw))),
        int(min(height, math.ceil(cy + hh))),
    ]


def parse_participant_label(text: str) -> bool | None:
    t = (text or "").strip().lower()
    lines = [line.strip().rstrip(".,!") for line in t.splitlines() if line.strip()]
    if lines:
        tail = lines[-1]
        if tail == "participant":
            return True
        if tail == "non-participant":
            return False
    if "non-participant" in t[:80]:
        return False
    if "participant" in t[:80]:
        return True
    return None


def compute_detection_metrics(records: list[dict]) -> dict:
    total_gt = sum(r["n_gt"] for r in records)
    total_pred = sum(r["n_pred"] for r in records)
    matched = sum(len(r["matches"]) for r in records)
    matched_ious = [m["iou"] for r in records for m in r["matches"]]
    precision = matched / total_pred if total_pred else 0.0
    recall = matched / total_gt if total_gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "n_images": len(records),
        "n_gt": total_gt,
        "n_pred": total_pred,
        "n_matched": matched,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_matched_iou": round(sum(matched_ious) / len(matched_ious), 4) if matched_ious else 0.0,
        "mean_latency_ms": round(sum(r["latency_ms"] for r in records) / len(records), 2) if records else 0.0,
    }


def compute_role_metrics(records: list[dict], total_gt: int) -> dict:
    valid = [r for r in records if r["predicted_participant"] is not None]
    tp = sum(1 for r in valid if r["gt_participant"] and r["predicted_participant"])
    tn = sum(1 for r in valid if (not r["gt_participant"]) and (not r["predicted_participant"]))
    fp = sum(1 for r in valid if (not r["gt_participant"]) and r["predicted_participant"])
    fn = sum(1 for r in valid if r["gt_participant"] and (not r["predicted_participant"]))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(valid) if valid else 0.0
    end_to_end_accuracy = (tp + tn) / total_gt if total_gt else 0.0

    per_role = {}
    for name, value in [("participant", True), ("non-participant", False)]:
        subset = [r for r in valid if r["gt_participant"] is value]
        correct = sum(1 for r in subset if r["predicted_participant"] is value)
        per_role[name] = {
            "correct": correct,
            "total_valid": len(subset),
            "accuracy": round(correct / len(subset), 4) if subset else 0.0,
        }

    matched_total = len(records)
    parse_errors = sum(1 for r in records if r["predicted_participant"] is None)
    return {
        "n_gt_total": total_gt,
        "n_matched_detections": matched_total,
        "n_valid_predictions": len(valid),
        "parse_errors": parse_errors,
        "accuracy_on_valid": round(accuracy, 4),
        "end_to_end_accuracy_over_gt": round(end_to_end_accuracy, 4),
        "precision_participant": round(precision, 4),
        "recall_participant": round(recall, 4),
        "f1_participant": round(f1, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "per_role": per_role,
        "mean_latency_ms": round(sum(r["latency_ms"] for r in records) / len(records), 2) if records else 0.0,
    }


def save_crop(image: Image.Image, bbox: list[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(tuple(bbox)).save(path)


def run_detection(
    samples: list[Sample],
    detector: YOLOv8HeadModel,
    conf_threshold: float,
    iou_threshold: float,
) -> tuple[list[dict], list[dict]]:
    detection_records: list[dict] = []
    matched_items: list[dict] = []

    for sample in samples:
        result = detector.detect(str(sample.image_path), conf_threshold=conf_threshold)
        gt_boxes = [p.bbox for p in sample.people]
        pred_boxes = [d.bbox for d in result.detections]
        matches = greedy_match(gt_boxes, pred_boxes, iou_threshold=iou_threshold)

        match_recs = []
        for gt_idx, pred_idx, iou in matches:
            gt_person = sample.people[gt_idx]
            det = result.detections[pred_idx]
            match_recs.append(
                {
                    "gt_idx": gt_idx,
                    "pred_idx": pred_idx,
                    "iou": round(iou, 4),
                    "gt_role": gt_person.role,
                    "gt_bbox": gt_person.bbox,
                    "pred_bbox": det.bbox,
                    "confidence": round(det.confidence, 4),
                }
            )
            matched_items.append(
                {
                    "file_name": sample.file_name,
                    "image_path": str(sample.image_path),
                    "person_id": gt_person.person_id,
                    "gt_bbox": gt_person.bbox,
                    "pred_bbox": det.bbox,
                    "gt_role": gt_person.role,
                    "iou": round(iou, 4),
                    "confidence": round(det.confidence, 4),
                }
            )

        detection_records.append(
            {
                "file_name": sample.file_name,
                "image_path": str(sample.image_path),
                "n_gt": len(sample.people),
                "n_pred": len(result.detections),
                "latency_ms": round(result.latency_ms, 2),
                "error": result.error,
                "matches": match_recs,
                "predictions": [
                    {
                        "bbox": [round(v, 2) for v in d.bbox],
                        "confidence": round(d.confidence, 4),
                    }
                    for d in result.detections
                ],
            }
        )
        print(
            f"  {sample.file_name}: gt={len(sample.people)} pred={len(result.detections)} "
            f"matched={len(matches)} latency={result.latency_ms:.1f}ms"
        )

    return detection_records, matched_items


def evaluate_vlm_dilations(
    samples_by_name: dict[str, Sample],
    matched_items: list[dict],
    dilations: list[float],
    out_dir: Path,
    model,
) -> dict:
    crops_root = out_dir / "crops"
    results: dict[str, dict] = {}
    total_gt = sum(len(s.people) for s in samples_by_name.values())

    for dilation in dilations:
        key = f"{dilation:.2f}"
        print(f"\nEvaluating VLM with dilation={dilation:.2f}")
        records = []
        dilation_dir = crops_root / f"dilate_{key.replace('.', 'p')}"
        for item in matched_items:
            sample = samples_by_name[item["file_name"]]
            image = Image.open(sample.image_path).convert("RGB")
            w, h = image.size
            crop_bbox = dilate_bbox(item["pred_bbox"], dilation, w, h)
            crop_path = dilation_dir / f"{Path(item['file_name']).stem}__{item['person_id']}.png"
            save_crop(image, crop_bbox, crop_path)

            result = model.run_two_image(str(sample.image_path), str(crop_path), PROMPT_PARTICIPANT)
            pred = None if result.error else parse_participant_label(result.response)
            gt_participant = item["gt_role"] == "participant"
            records.append(
                {
                    **item,
                    "crop_path": str(crop_path),
                    "dilation": dilation,
                    "dilated_bbox": crop_bbox,
                    "gt_participant": gt_participant,
                    "predicted_participant": pred,
                    "correct": pred == gt_participant if pred is not None else False,
                    "response": result.response,
                    "latency_ms": round(result.latency_ms, 2),
                    "error": result.error,
                }
            )
            pred_name = (
                "participant" if pred is True else
                "non-participant" if pred is False else
                "unparsed"
            )
            print(
                f"  {item['file_name']} {item['person_id']}: gt={item['gt_role']} "
                f"pred={pred_name} iou={item['iou']:.2f}"
            )
        results[key] = {
            "dilation": dilation,
            "metrics": compute_role_metrics(records, total_gt=total_gt),
            "records": records,
        }
    return results


def save_plots(out_dir: Path, detection_metrics: dict, detection_records: list[dict], dilation_results: dict) -> list[str]:
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor="white")
    ax.set_facecolor("#fbfbf9")
    metric_names = ["precision", "recall", "f1", "mean_matched_iou"]
    metric_vals = [detection_metrics[m] for m in metric_names]
    ax.bar(metric_names, metric_vals, color=["#2563eb", "#059669", "#7c3aed", "#f97316"])
    ax.set_ylim(0, 1.0)
    ax.set_title("YOLOv8-Head vs Iteration2 Face Boxes")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    p = figures_dir / "head_detection_metrics.png"
    fig.tight_layout()
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(str(p))

    dil_keys = list(dilation_results.keys())
    dil_labels = [f"{dilation_results[k]['dilation']:.2f}" for k in dil_keys]
    cls_acc = [dilation_results[k]["metrics"]["accuracy_on_valid"] for k in dil_keys]
    e2e_acc = [dilation_results[k]["metrics"]["end_to_end_accuracy_over_gt"] for k in dil_keys]
    f1_vals = [dilation_results[k]["metrics"]["f1_participant"] for k in dil_keys]

    x = np.arange(len(dil_keys))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor="white")
    ax.set_facecolor("#fbfbf9")
    ax.bar(x - width, cls_acc, width, label="Role accuracy (valid)", color="#2563eb")
    ax.bar(x, e2e_acc, width, label="End-to-end over GT", color="#059669")
    ax.bar(x + width, f1_vals, width, label="Participant F1", color="#7c3aed")
    ax.set_xticks(x)
    ax.set_xticklabels(dil_labels)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Dilation scale")
    ax.set_title("VLM Role Metrics vs Crop Dilation")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    p = figures_dir / "vlm_dilation_metrics.png"
    fig.tight_layout()
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(str(p))

    part_recall = [dilation_results[k]["metrics"]["per_role"]["participant"]["accuracy"] for k in dil_keys]
    nonpart_recall = [dilation_results[k]["metrics"]["per_role"]["non-participant"]["accuracy"] for k in dil_keys]
    parse_errors = [dilation_results[k]["metrics"]["parse_errors"] for k in dil_keys]
    fig, ax1 = plt.subplots(figsize=(10, 5.2), facecolor="white")
    ax1.set_facecolor("#fbfbf9")
    ax1.plot(dil_labels, part_recall, marker="o", label="Participant accuracy", color="#2563eb", linewidth=2)
    ax1.plot(dil_labels, nonpart_recall, marker="o", label="Non-participant accuracy", color="#dc2626", linewidth=2)
    ax1.set_ylim(0, 1.0)
    ax1.set_xlabel("Dilation scale")
    ax1.set_ylabel("Accuracy")
    ax1.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    ax2 = ax1.twinx()
    ax2.bar(dil_labels, parse_errors, alpha=0.18, color="#52525b", label="Parse errors")
    ax2.set_ylabel("Parse errors")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, fontsize=8, loc="lower left")
    ax1.set_title("Per-Role Accuracy and Parse Errors vs Dilation")
    p = figures_dir / "vlm_dilation_role_breakdown.png"
    fig.tight_layout()
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(str(p))

    abs_errors = [abs(r["n_pred"] - r["n_gt"]) for r in detection_records]
    matched_counts = [len(r["matches"]) for r in detection_records]
    image_labels = [Path(r["file_name"]).stem for r in detection_records]
    fig, ax = plt.subplots(figsize=(12, 5.2), facecolor="white")
    ax.set_facecolor("#fbfbf9")
    ax.bar(np.arange(len(image_labels)) - 0.2, [r["n_gt"] for r in detection_records], 0.4, label="GT", color="#94a3b8")
    ax.bar(np.arange(len(image_labels)) + 0.2, [r["n_pred"] for r in detection_records], 0.4, label="Pred", color="#0ea5e9")
    ax.plot(np.arange(len(image_labels)), matched_counts, color="#16a34a", marker="o", linewidth=2, label="Matched")
    ax.set_xticks(np.arange(len(image_labels)))
    ax.set_xticklabels(image_labels, rotation=45, ha="right", fontsize=8)
    ax.set_title("Per-Image Head Detection Counts")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    p = figures_dir / "head_detection_counts_per_image.png"
    fig.tight_layout()
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(str(p))

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate iteration2 face/head participant dataset.")
    parser.add_argument("--config", default=str(ROOT / "benchmark_config.yaml"))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--head-model", choices=["nano", "medium"], default="nano")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--dilations", default="1.0,1.3,1.6,2.0,2.4")
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    samples = load_samples(max_images=args.max_images)
    if not samples:
        raise SystemExit(f"No samples found in {DATA_ROOT}")

    dilations = [float(x.strip()) for x in args.dilations.split(",") if x.strip()]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {ANNOTATIONS_PATH}")
    print(f"Images: {len(samples)}")
    print(f"GT people: {sum(len(s.people) for s in samples)}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    detector = YOLOv8HeadModel(variant=args.head_model)
    print(f"\nLoading detector: {detector.name} ({args.head_model})")
    detector.load()
    print("\nRunning head detection")
    detection_records, matched_items = run_detection(
        samples=samples,
        detector=detector,
        conf_threshold=args.conf,
        iou_threshold=args.iou_threshold,
    )
    detector.unload()

    detection_metrics = compute_detection_metrics(detection_records)
    print(
        f"\nDetection metrics: precision={detection_metrics['precision']:.3f} "
        f"recall={detection_metrics['recall']:.3f} f1={detection_metrics['f1']:.3f} "
        f"mean_iou={detection_metrics['mean_matched_iou']:.3f}"
    )

    samples_by_name = {s.file_name: s for s in samples}
    model, model_cfg = load_qwen_model(args.config, model_key="qwen3vl_4b")
    print(f"\nLoading VLM: {model_cfg.key}")
    model.load()
    dilation_results = evaluate_vlm_dilations(
        samples_by_name=samples_by_name,
        matched_items=matched_items,
        dilations=dilations,
        out_dir=run_dir,
        model=model,
    )
    model.unload()

    best_dilation_key = max(
        dilation_results,
        key=lambda k: (
            dilation_results[k]["metrics"]["end_to_end_accuracy_over_gt"],
            dilation_results[k]["metrics"]["accuracy_on_valid"],
        ),
    )
    plots = save_plots(run_dir, detection_metrics, detection_records, dilation_results)

    out = {
        "timestamp": ts,
        "dataset": str(ANNOTATIONS_PATH),
        "head_detector": {
            "model": detector.name,
            "variant": args.head_model,
            "conf_threshold": args.conf,
            "iou_threshold": args.iou_threshold,
            "metrics": detection_metrics,
            "records": detection_records,
        },
        "vlm": {
            "model_key": model_cfg.key,
            "model_path": model_cfg.model_path,
            "prompt": PROMPT_PARTICIPANT,
            "dilation_results": dilation_results,
            "best_dilation": dilation_results[best_dilation_key]["dilation"],
            "best_dilation_key": best_dilation_key,
        },
        "plots": plots,
    }
    out_path = run_dir / f"iteration2_head_participant_eval_{ts}.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("\nBest dilation:")
    best = dilation_results[best_dilation_key]
    print(
        f"  {best['dilation']:.2f}: valid_acc={best['metrics']['accuracy_on_valid']:.3f} "
        f"e2e_acc={best['metrics']['end_to_end_accuracy_over_gt']:.3f} "
        f"f1={best['metrics']['f1_participant']:.3f}"
    )
    print(f"\nResults -> {out_path}")
    for plot in plots:
        print(f"Plot -> {plot}")


if __name__ == "__main__":
    main()
