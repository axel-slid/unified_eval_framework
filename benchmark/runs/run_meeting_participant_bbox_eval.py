#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from models.base import Detection, DetectionResult
from models.yolov11 import YOLOv11Model
from models.yolov8_face import YOLOv8FaceModel


DATASET_JSON = PROJECT_ROOT / "meeting_participation_dataset" / "allBoxes.json"
DATASET_DIR = PROJECT_ROOT / "meeting_participation_dataset"
RESULTS_DIR = ROOT / "results"
HEAD_WEIGHTS_DIR = ROOT / "face_detection" / "weights"
HEAD_WEIGHT_URL = "https://github.com/Abcfsa/YOLOv8_head_detector/raw/main/nano.pt"


@dataclass
class MatchRecord:
    pred_idx: int
    gt_idx: int
    iou: float
    confidence: float
    gt_role: str


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def greedy_match(
    preds: list[dict],
    gts: list[dict],
    iou_threshold: float,
) -> tuple[list[MatchRecord], list[int], list[int]]:
    candidates: list[tuple[float, float, int, int]] = []
    for pred_idx, pred in enumerate(preds):
        for gt_idx, gt in enumerate(gts):
            iou = box_iou(pred["bbox"], gt["bbox"])
            if iou >= iou_threshold:
                candidates.append((iou, pred["confidence"], pred_idx, gt_idx))

    candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    used_preds: set[int] = set()
    used_gts: set[int] = set()
    matches: list[MatchRecord] = []

    for iou, conf, pred_idx, gt_idx in candidates:
        if pred_idx in used_preds or gt_idx in used_gts:
            continue
        used_preds.add(pred_idx)
        used_gts.add(gt_idx)
        matches.append(
            MatchRecord(
                pred_idx=pred_idx,
                gt_idx=gt_idx,
                iou=iou,
                confidence=conf,
                gt_role=gts[gt_idx]["role"],
            )
        )

    unmatched_pred_indices = [idx for idx in range(len(preds)) if idx not in used_preds]
    unmatched_gt_indices = [idx for idx in range(len(gts)) if idx not in used_gts]
    return matches, unmatched_pred_indices, unmatched_gt_indices


def summarize_role_breakdown(gts: list[dict], matches: list[MatchRecord]) -> dict:
    gt_by_role = {"participant": 0, "non-participant": 0}
    for gt in gts:
        gt_by_role[gt["role"]] = gt_by_role.get(gt["role"], 0) + 1

    matched_by_role = {"participant": [], "non-participant": []}
    for match in matches:
        matched_by_role.setdefault(match.gt_role, []).append(match.iou)

    summary = {}
    for role, total in gt_by_role.items():
        ious = matched_by_role.get(role, [])
        matched = len(ious)
        summary[role] = {
            "gt_count": total,
            "matched_gt_count": matched,
            "recall": matched / total if total else 0.0,
            "mean_iou": float(statistics.mean(ious)) if ious else 0.0,
            "median_iou": float(statistics.median(ious)) if ious else 0.0,
        }
    return summary


def load_dataset() -> list[dict]:
    payload = json.loads(DATASET_JSON.read_text())
    items = []
    for ann in payload["annotations"]:
        image_path = DATASET_DIR / ann["file_name"]
        with Image.open(image_path) as img:
            width, height = img.size
        items.append(
            {
                "image_path": image_path,
                "image_name": ann["file_name"],
                "width": width,
                "height": height,
                "gts": ann["people"],
            }
        )
    return items


class YOLOv8HeadModel:
    name = "YOLOv8-Head"

    def __init__(self, device: str | None = None):
        self._device = device
        self.model = None
        self.weights_path = HEAD_WEIGHTS_DIR / "yolov8_nano_head.pt"

    def _ensure_weights(self) -> None:
        if self.weights_path.exists():
            return
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HEAD_WEIGHT_URL, self.weights_path)

    def load(self) -> None:
        from ultralytics import YOLO

        self._ensure_weights()
        self.model = YOLO(str(self.weights_path))

    def detect(self, image_path: str, conf_threshold: float = 0.25) -> DetectionResult:
        try:
            t0 = time.perf_counter()
            results = self.model(
                image_path,
                conf=conf_threshold,
                device=self._device,
                verbose=False,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append(
                        Detection(
                            bbox=[x1, y1, x2, y2],
                            confidence=float(box.conf[0]),
                            class_id=0,
                            class_name="head",
                        )
                    )
            return DetectionResult(
                detections=detections,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            return DetectionResult(detections=[], latency_ms=0.0, error=str(exc))

    def unload(self) -> None:
        self.model = None


def build_model(model_key: str, device: str | None):
    if model_key == "face":
        return YOLOv8FaceModel(device=device)
    if model_key == "head":
        return YOLOv8HeadModel(device=device)
    if model_key == "person":
        return YOLOv11Model(variant="small", device=device)
    raise ValueError(f"Unknown model_key={model_key!r}")


def run_eval(model_key: str, items: list[dict], conf: float, iou_threshold: float, device: str | None) -> dict:
    model = build_model(model_key, device)
    model.load()

    all_ious: list[float] = []
    latencies: list[float] = []
    image_results: list[dict] = []
    total_gt = 0
    total_preds = 0
    total_matches = 0
    total_errors = 0
    role_gt_totals = {"participant": 0, "non-participant": 0}
    role_match_totals = {"participant": 0, "non-participant": 0}
    role_iou_lists = {"participant": [], "non-participant": []}

    for item in items:
        result = model.detect(str(item["image_path"]), conf_threshold=conf)
        gts = item["gts"]
        preds = [
            {
                "bbox": det.bbox,
                "confidence": det.confidence,
            }
            for det in result.detections
        ] if not result.error else []

        for gt in gts:
            role_gt_totals[gt["role"]] = role_gt_totals.get(gt["role"], 0) + 1

        total_gt += len(gts)
        total_preds += len(preds)

        if result.error:
            total_errors += 1
            image_results.append(
                {
                    "image_name": item["image_name"],
                    "width": item["width"],
                    "height": item["height"],
                    "n_gt": len(gts),
                    "n_pred": 0,
                    "latency_ms": result.latency_ms,
                    "error": result.error,
                    "matches": [],
                    "role_breakdown": summarize_role_breakdown(gts, []),
                }
            )
            continue

        latencies.append(result.latency_ms)
        matches, unmatched_pred_indices, unmatched_gt_indices = greedy_match(preds, gts, iou_threshold)
        total_matches += len(matches)
        for match in matches:
            all_ious.append(match.iou)
            role_match_totals[match.gt_role] = role_match_totals.get(match.gt_role, 0) + 1
            role_iou_lists.setdefault(match.gt_role, []).append(match.iou)

        image_results.append(
            {
                "image_name": item["image_name"],
                "width": item["width"],
                "height": item["height"],
                "n_gt": len(gts),
                "n_pred": len(preds),
                "latency_ms": result.latency_ms,
                "error": None,
                "matches": [asdict(match) for match in matches],
                "unmatched_pred_indices": unmatched_pred_indices,
                "unmatched_gt_indices": unmatched_gt_indices,
                "role_breakdown": summarize_role_breakdown(gts, matches),
            }
        )
        print(
            f"[{model_key}] {item['image_name']}: "
            f"{len(matches)}/{len(gts)} matched, {len(preds)} preds, {result.latency_ms:.1f} ms"
        )

    model.unload()

    precision = total_matches / total_preds if total_preds else 0.0
    recall = total_matches / total_gt if total_gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    role_summary = {}
    for role, gt_total in role_gt_totals.items():
        matched = role_match_totals.get(role, 0)
        role_ious = role_iou_lists.get(role, [])
        role_summary[role] = {
            "gt_count": gt_total,
            "matched_gt_count": matched,
            "recall": matched / gt_total if gt_total else 0.0,
            "mean_iou": float(statistics.mean(role_ious)) if role_ious else 0.0,
            "median_iou": float(statistics.median(role_ious)) if role_ious else 0.0,
        }

    return {
        "model_key": model_key,
        "model_name": getattr(model, "name", model_key),
        "confidence_threshold": conf,
        "iou_threshold": iou_threshold,
        "n_images": len(items),
        "n_errors": total_errors,
        "total_gt_boxes": total_gt,
        "total_pred_boxes": total_preds,
        "total_matched_boxes": total_matches,
        "bbox_precision": precision,
        "bbox_recall": recall,
        "bbox_f1": f1,
        "mean_matched_iou": float(statistics.mean(all_ious)) if all_ious else 0.0,
        "median_matched_iou": float(statistics.median(all_ious)) if all_ious else 0.0,
        "mean_latency_ms": float(statistics.mean(latencies)) if latencies else 0.0,
        "participant_breakdown": role_summary,
        "per_image": image_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate face/head/person bbox models on the meeting participant dataset.")
    parser.add_argument("--models", nargs="+", default=["face", "head", "person"], choices=["face", "head", "person"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    items = load_dataset()
    print(f"Loaded {len(items)} annotated images from {DATASET_JSON}")

    results = {
        "dataset": str(DATASET_JSON),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "models": {},
    }

    for model_key in args.models:
        print(f"\n=== Evaluating {model_key} ===")
        results["models"][model_key] = run_eval(
            model_key=model_key,
            items=items,
            conf=args.conf,
            iou_threshold=args.iou,
            device=args.device,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"meeting_participant_bbox_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
