"""
MTCNN face detection model wrapper.

Uses facenet-pytorch's MTCNN (Multi-task Cascaded CNN).
Weights (~1 MB) download automatically on first use.

Reference: Zhang et al., "Joint Face Detection and Alignment using
Multi-task Cascaded Convolutional Networks", IEEE SPL 2016.
"""
from __future__ import annotations
import time
import torch
from .base import BaseFaceDetector, Detection, DetectionResult


class MTCNNFaceModel(BaseFaceDetector):
    """
    Args:
        device:        "cuda", "cpu", or None for auto.
        min_face_size: minimum face height in pixels (default 20).
    """
    name = "MTCNN"

    def __init__(self, device: str | None = None, min_face_size: int = 20):
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.min_face_size = min_face_size
        self.model = None

    def load(self) -> None:
        from facenet_pytorch import MTCNN
        print(f"[{self.name}] Loading on {self._device} …")
        self.model = MTCNN(
            keep_all=True,
            device=self._device,
            min_face_size=self.min_face_size,
            post_process=False,
        )
        # warm-up
        from PIL import Image
        import numpy as np
        self.model.detect(Image.fromarray(np.zeros((320, 320, 3), dtype=np.uint8)))
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
                    detections.append(Detection(bbox=box.tolist(), confidence=float(prob)))
            return DetectionResult(detections=detections, latency_ms=latency_ms)
        except Exception as exc:
            return DetectionResult(detections=[], latency_ms=0.0, error=str(exc))

    def unload(self) -> None:
        del self.model
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
