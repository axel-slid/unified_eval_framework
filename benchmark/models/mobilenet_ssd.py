"""
MobileNet SSD person detection model wrapper.

Uses torchvision's SSDLite320 with MobileNetV3-Large backbone, pre-trained on
COCO. This model is designed for edge deployment — fast on CPU with a small
memory footprint.

Published COCO val benchmarks (torchvision model zoo):
  ssdlite320_mobilenet_v3_large: box mAP@50:95 = 21.3
"""
from __future__ import annotations

import time

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from models.base import BaseDetectionModel, Detection, DetectionResult

# COCO class index for "person" (1-indexed in COCO, 0-indexed label returned by
# torchvision detection models maps label 1 → "person")
_PERSON_LABEL = 1

# COCO 80-class label list (torchvision uses 1-based labels for COCO; 0 = background)
_COCO_LABELS = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "N/A",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
    "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A",
    "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


class MobileNetSSDModel(BaseDetectionModel):
    """
    SSDLite320 with MobileNetV3-Large backbone (torchvision, COCO-pretrained).

    Args:
        device: "cuda", "cpu", or None for auto-detect.
    """

    def __init__(self, device: str | None = None):
        self.name = "MobileNet SSD"
        self._device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self._device_str)
        self.model = None

    # ------------------------------------------------------------------
    def load(self) -> None:
        from torchvision.models.detection import (
            ssdlite320_mobilenet_v3_large,
            SSDLite320_MobileNet_V3_Large_Weights,
        )

        print(f"[{self.name}] Loading SSDLite320 MobileNetV3-Large on {self._device_str} ...")
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.model = ssdlite320_mobilenet_v3_large(weights=weights)
        self.model.to(self.device)
        self.model.eval()

        # Warm-up
        dummy = torch.zeros(1, 3, 320, 320, device=self.device)
        with torch.no_grad():
            self.model(dummy)
        print(f"[{self.name}] Ready")

    # ------------------------------------------------------------------
    def _preprocess(self, image_path: str) -> tuple[torch.Tensor, tuple[int, int]]:
        """Load image → float tensor in [0, 1], return (tensor, (H, W))."""
        img = Image.open(image_path).convert("RGB")
        orig_size = (img.height, img.width)
        tensor = TF.to_tensor(img).unsqueeze(0).to(self.device)
        return tensor, orig_size

    # ------------------------------------------------------------------
    def detect(self, image_path: str, conf_threshold: float = 0.25) -> DetectionResult:
        try:
            tensor, _ = self._preprocess(image_path)

            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = self.model(tensor)
            latency_ms = (time.perf_counter() - t0) * 1000

            detections: list[Detection] = []
            out = outputs[0]
            for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
                if float(score) < conf_threshold:
                    continue
                if int(label) != _PERSON_LABEL:
                    continue
                x1, y1, x2, y2 = box.tolist()
                detections.append(Detection(
                    bbox=[x1, y1, x2, y2],
                    confidence=float(score),
                    class_id=int(label),
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
            tensor, _ = self._preprocess(image_path)

            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = self.model(tensor)
            latency_ms = (time.perf_counter() - t0) * 1000

            detections: list[Detection] = []
            out = outputs[0]
            for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
                if float(score) < conf_threshold:
                    continue
                lbl_idx = int(label)
                name = _COCO_LABELS[lbl_idx] if lbl_idx < len(_COCO_LABELS) else str(lbl_idx)
                x1, y1, x2, y2 = box.tolist()
                detections.append(Detection(
                    bbox=[x1, y1, x2, y2],
                    confidence=float(score),
                    class_id=lbl_idx,
                    class_name=name,
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
