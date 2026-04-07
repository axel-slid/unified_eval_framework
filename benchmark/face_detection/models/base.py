"""Shared dataclasses for face detection models."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Detection:
    bbox: list[float]       # [x1, y1, x2, y2] pixels
    confidence: float
    class_name: str = "face"


@dataclass
class DetectionResult:
    detections: list[Detection]
    latency_ms: float
    error: str | None = None


class BaseFaceDetector(ABC):
    name: str = "UnnamedDetector"

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def detect(self, image_path: str, conf_threshold: float = 0.25) -> DetectionResult: ...

    def unload(self) -> None:
        pass
