"""
YOLOv8-Face detection model wrapper.

YOLOv8n backbone fine-tuned on WiderFace. Weights (~6 MB) are stored in
face_detection/weights/yolov8n-face.pt and auto-downloaded if missing.

Model source: arnabdhar/YOLOv8-Face-Detection (HuggingFace)
"""
from __future__ import annotations
import time
from pathlib import Path
import torch
from .base import BaseFaceDetector, Detection, DetectionResult

_WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
_DEFAULT_WEIGHTS = _WEIGHTS_DIR / "yolov8n-face.pt"
_HF_URL = "https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt"


class YOLOv8FaceModel(BaseFaceDetector):
    """
    Args:
        weights: path to .pt file. Auto-downloads if not found.
        device:  "cuda", "cpu", or None for auto.
    """
    name = "YOLOv8-Face"

    def __init__(self, weights: str | Path | None = None, device: str | None = None):
        self.weights = Path(weights) if weights else _DEFAULT_WEIGHTS
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def _ensure_weights(self) -> None:
        if self.weights.exists():
            return
        import urllib.request
        print(f"[{self.name}] Downloading weights → {self.weights} …")
        self.weights.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_HF_URL, self.weights)
        print(f"[{self.name}] Download complete")

    def load(self) -> None:
        from ultralytics import YOLO
        self._ensure_weights()
        print(f"[{self.name}] Loading {self.weights.name} on {self._device} …")
        self.model = YOLO(str(self.weights))
        import numpy as np
        self.model(np.zeros((320, 320, 3), dtype=np.uint8), verbose=False)
        print(f"[{self.name}] Ready")

    def detect(self, image_path: str, conf_threshold: float = 0.25) -> DetectionResult:
        try:
            t0 = time.perf_counter()
            results = self.model(image_path, conf=conf_threshold, device=self._device, verbose=False)
            latency_ms = (time.perf_counter() - t0) * 1000
            detections: list[Detection] = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append(Detection(bbox=[x1, y1, x2, y2], confidence=float(box.conf[0])))
            return DetectionResult(detections=detections, latency_ms=latency_ms)
        except Exception as exc:
            return DetectionResult(detections=[], latency_ms=0.0, error=str(exc))

    def unload(self) -> None:
        del self.model
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
