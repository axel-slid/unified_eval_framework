"""
YOLOv11 person detection model wrapper.

Supports both yolo11n (nano) and yolo11s (small) via the `variant` parameter.
Weights are downloaded automatically from ultralytics on first use.

Variant comparison (COCO val, all classes):
  nano  (yolo11n): mAP@50:95 = 39.5, params = 2.6M
  small (yolo11s): mAP@50:95 = 47.0, params = 9.4M
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from models.base import BaseDetectionModel, Detection, DetectionResult

# COCO class index for "person"
_PERSON_CLS = 0

# Model weights filenames (auto-downloaded by ultralytics)
_WEIGHT_MAP = {
    "nano":  "yolo11n.pt",
    "small": "yolo11s.pt",
}


class YOLOv11Model(BaseDetectionModel):
    """
    YOLOv11 detection model (nano or small variant).

    Args:
        variant: "nano" or "small". Defaults to "nano".
        device:  "cuda", "cpu", or None for auto-detect.
    """

    def __init__(self, variant: str = "nano", device: str | None = None):
        if variant not in _WEIGHT_MAP:
            raise ValueError(f"variant must be 'nano' or 'small', got {variant!r}")
        self.variant = variant
        self.weights = _WEIGHT_MAP[variant]
        self.name = f"YOLOv11{'n' if variant == 'nano' else 's'}"
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    # ------------------------------------------------------------------
    def load(self) -> None:
        from ultralytics import YOLO  # lazy import to keep startup fast

        print(f"[{self.name}] Loading {self.weights} on {self._device} ...")
        self.model = YOLO(self.weights)
        # Warm-up: run one tiny inference so the first timed call isn't skewed
        import numpy as np
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
        print(f"[{self.name}] Ready")

    # ------------------------------------------------------------------
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

            detections: list[Detection] = []
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls != _PERSON_CLS:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append(Detection(
                        bbox=[x1, y1, x2, y2],
                        confidence=float(box.conf[0]),
                        class_id=cls,
                        class_name="person",
                    ))

            return DetectionResult(detections=detections, latency_ms=latency_ms)

        except Exception as exc:
            return DetectionResult(detections=[], latency_ms=0.0, error=str(exc))

    # ------------------------------------------------------------------
    def detect_all_classes(
        self, image_path: str, conf_threshold: float = 0.25
    ) -> DetectionResult:
        """Return detections for ALL classes (needed for COCO mAP evaluation)."""
        try:
            t0 = time.perf_counter()
            results = self.model(
                image_path,
                conf=conf_threshold,
                device=self._device,
                verbose=False,
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            detections: list[Detection] = []
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append(Detection(
                        bbox=[x1, y1, x2, y2],
                        confidence=float(box.conf[0]),
                        class_id=cls,
                        class_name=self.model.names[cls],
                    ))

            return DetectionResult(detections=detections, latency_ms=latency_ms)

        except Exception as exc:
            return DetectionResult(detections=[], latency_ms=0.0, error=str(exc))

    # ------------------------------------------------------------------
    def unload(self) -> None:
        del self.model
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
