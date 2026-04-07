"""
RetinaFace face detection model wrapper.

Uses insightface's RetinaFace detector (buffalo_sc backbone, ~30 MB).
Weights download automatically to ~/.insightface/models/ on first use.

Reference: Deng et al., "RetinaFace: Single-Shot Multi-Level Face
Localisation in the Wild", CVPR 2020.
"""
from __future__ import annotations
import time
import torch
from .base import BaseFaceDetector, Detection, DetectionResult


class RetinaFaceModel(BaseFaceDetector):
    """
    Args:
        model_pack: insightface pack — "buffalo_sc" (fast) or "buffalo_l" (accurate).
        device:     "cuda", "cpu", or None for auto.
        det_size:   detection input resolution (H, W). Default (640, 640).
    """
    name = "RetinaFace"

    def __init__(
        self,
        model_pack: str = "buffalo_sc",
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
        print(f"[{self.name}] Loading {self.model_pack} on {self._device} …")
        self.app = FaceAnalysis(
            name=self.model_pack,
            providers=providers,
            allowed_modules=["detection"],
        )
        self.app.prepare(ctx_id=ctx_id, det_size=self.det_size)
        import numpy as np
        self.app.get(np.zeros((320, 320, 3), dtype=np.uint8))
        print(f"[{self.name}] Ready")

    def detect(self, image_path: str, conf_threshold: float = 0.25) -> DetectionResult:
        import cv2
        try:
            img = cv2.imread(image_path)
            if img is None:
                from PIL import Image
                import numpy as np
                pil = Image.open(image_path).convert("RGB")
                img = np.array(pil)[:, :, ::-1]
            t0 = time.perf_counter()
            faces = self.app.get(img)
            latency_ms = (time.perf_counter() - t0) * 1000
            detections = [
                Detection(bbox=f.bbox.tolist(), confidence=float(f.det_score))
                for f in faces
                if float(f.det_score) >= conf_threshold
            ]
            return DetectionResult(detections=detections, latency_ms=latency_ms)
        except Exception as exc:
            return DetectionResult(detections=[], latency_ms=0.0, error=str(exc))

    def unload(self) -> None:
        del self.app
        self.app = None
