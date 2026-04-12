from .base import BaseVLMModel, InferenceResult, BaseDetectionModel, Detection, DetectionResult

# VLM models — wrapped in try/except so detection-only scripts don't fail
# if heavy VLM dependencies (transformers, etc.) aren't importable.
try:
    from .smolvlm import SmolVLMModel
except Exception:
    SmolVLMModel = None  # type: ignore[assignment,misc]

try:
    from .internvl import InternVLModel
except Exception:
    InternVLModel = None  # type: ignore[assignment,misc]

try:
    from .qwen3vl import Qwen3VLModel
except Exception:
    Qwen3VLModel = None  # type: ignore[assignment,misc]

try:
    from .gemma import GemmaModel
except Exception:
    GemmaModel = None  # type: ignore[assignment,misc]

# CV detection models
from .yolov11 import YOLOv11Model
from .mobilenet_ssd import MobileNetSSDModel

# Registry: class name (string from YAML `class:` field) → Python class
MODEL_REGISTRY: dict[str, type[BaseVLMModel]] = {
    k: v for k, v in {
        "SmolVLMModel":  SmolVLMModel,
        "InternVLModel": InternVLModel,
        "Qwen3VLModel":  Qwen3VLModel,
        "GemmaModel":    GemmaModel,
    }.items() if v is not None
}

# Detection model registry (separate from VLM registry)
DETECTION_REGISTRY: dict[str, type[BaseDetectionModel]] = {
    "YOLOv11Model":      YOLOv11Model,
    "MobileNetSSDModel": MobileNetSSDModel,
}

__all__ = [
    "BaseVLMModel",
    "InferenceResult",
    "BaseDetectionModel",
    "Detection",
    "DetectionResult",
    "SmolVLMModel",
    "InternVLModel",
    "Qwen3VLModel",
    "GemmaModel",
    "YOLOv11Model",
    "MobileNetSSDModel",
    "MODEL_REGISTRY",
    "DETECTION_REGISTRY",
]
