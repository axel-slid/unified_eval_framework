"""
MTCNN face detection model wrapper.

Uses facenet-pytorch's MTCNN (Multi-task Cascaded CNN) implementation.
Weights are downloaded automatically on first use (~1 MB).

Reference: Zhang et al., "Joint Face Detection and Alignment using
Multi-task Cascaded Convolutional Networks", IEEE SPL 2016.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from models.base import BaseDetectionModel, Detection, DetectionResult


class MTCNNFaceModel(BaseDetectionModel):
    """
    MTCNN face detector via facenet-pytorch.

    Args:
        device: "cuda", "cpu", or None for auto-detect.
        min_face_size: minimum face size in pixels (default 20).
    """

    name = "MTCNN"

    def __init__(self, device: str | None = None, min_face_size: int = 20):
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.min_face_size = min_face_size
        self.model = None

    def load(self) -> None:
        from facenet_pytorch import MTCNN

        print(f"[{self.name}] Loading MTCNN on {self._device} ...")
        self.model = MTCNN(
            keep_all=True,
            device=self._device,
            min_face_size=self.min_face_size,
            post_process=False,
        )
        # Warm-up
        from PIL import Image
        import numpy as np
        dummy = Image.fromarray(np.zeros((320, 320, 3), dtype=np.uint8))
        self.model.detect(dummy)
        print(f"[{self.name}] Ready")

    def detect(self, image_path: str, conf_threshold: float = 0.25) -> DetectionResult:
        from PIL import Image

        try:
            img = Image.open(image_path).convert("RGB")

            t0 = time.perf_counter()
            boxes, probs = self.model.detect(img)
            latency_ms = (time.perf_counter() - t0) * 1000

            detections: list[Detection] = []
            if boxes is not None and probs is not None:
                for box, prob in zip(boxes, probs):
                    if prob is None or float(prob) < conf_threshold:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    detections.append(Detection(
                        bbox=[x1, y1, x2, y2],
                        confidence=float(prob),
                        class_id=0,
                        class_name="face",
                    ))

            return DetectionResult(detections=detections, latency_ms=latency_ms)

        except Exception as exc:
            return DetectionResult(detections=[], latency_ms=0.0, error=str(exc))

    def unload(self) -> None:
        del self.model
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
