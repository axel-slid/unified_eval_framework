"""
RetinaFace face detection model wrapper.

Uses insightface's RetinaFace detector (buffalo_sc = fast ResNet-based backbone).
Model weights (~30 MB) are downloaded automatically to ~/.insightface/models/
on first use via ONNX Runtime.

Reference: Deng et al., "RetinaFace: Single-Shot Multi-Level Face Localisation
in the Wild", CVPR 2020.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from models.base import BaseDetectionModel, Detection, DetectionResult

# Available insightface model packs:
#   buffalo_sc  – compact, faster (~30 MB)
#   buffalo_l   – large, most accurate (~300 MB)
_DEFAULT_PACK = "buffalo_sc"


class RetinaFaceModel(BaseDetectionModel):
    """
    RetinaFace detector backed by insightface / ONNX Runtime.

    Args:
        model_pack: insightface model pack name ("buffalo_sc" or "buffalo_l").
        device:     "cuda", "cpu", or None for auto-detect.
        det_size:   detection input resolution (H, W). Larger → more accurate,
                    slower. Default (640, 640).
    """

    name = "RetinaFace"

    def __init__(
        self,
        model_pack: str = _DEFAULT_PACK,
        device: str | None = None,
        det_size: tuple[int, int] = (640, 640),
    ):
        self.model_pack = model_pack
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.det_size = det_size
        self.app = None

    def load(self) -> None:
        from insightface.app import FaceAnalysis

        ctx_id = 0 if self._device == "cuda" else -1
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ctx_id == 0
            else ["CPUExecutionProvider"]
        )

        print(f"[{self.name}] Loading {self.model_pack} on {self._device} ...")
        self.app = FaceAnalysis(
            name=self.model_pack,
            providers=providers,
            allowed_modules=["detection"],   # skip recognition/landmark for speed
        )
        self.app.prepare(ctx_id=ctx_id, det_size=self.det_size)

        # Warm-up
        import numpy as np
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self.app.get(dummy)
        print(f"[{self.name}] Ready")

    def detect(self, image_path: str, conf_threshold: float = 0.25) -> DetectionResult:
        import cv2

        try:
            img = cv2.imread(image_path)
            if img is None:
                # PIL fallback for non-standard formats
                from PIL import Image
                import numpy as np
                pil = Image.open(image_path).convert("RGB")
                img = np.array(pil)[:, :, ::-1]  # RGB → BGR for insightface

            t0 = time.perf_counter()
            faces = self.app.get(img)
            latency_ms = (time.perf_counter() - t0) * 1000

            detections: list[Detection] = []
            for face in faces:
                score = float(face.det_score)
                if score < conf_threshold:
                    continue
                x1, y1, x2, y2 = face.bbox.tolist()
                detections.append(Detection(
                    bbox=[x1, y1, x2, y2],
                    confidence=score,
                    class_id=0,
                    class_name="face",
                ))

            return DetectionResult(detections=detections, latency_ms=latency_ms)

        except Exception as exc:
            return DetectionResult(detections=[], latency_ms=0.0, error=str(exc))

    def unload(self) -> None:
        del self.app
        self.app = None
