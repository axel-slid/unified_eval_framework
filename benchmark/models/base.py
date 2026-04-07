from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class InferenceResult:
    response: str
    latency_ms: float
    error: str | None = None


@dataclass
class Detection:
    """Single object detection result."""
    bbox: list[float]       # [x1, y1, x2, y2] in pixel coordinates
    confidence: float
    class_id: int
    class_name: str


@dataclass
class DetectionResult:
    """Output from a detection model for one image."""
    detections: list[Detection]
    latency_ms: float
    error: str | None = None


class BaseVLMModel(ABC):
    """
    Contract every model runner must satisfy.
    Subclasses receive a ModelConfig at construction so they can read
    model_path, dtype, generation params, etc. without hardcoding anything.
    """

    name: str = "UnnamedModel"

    @abstractmethod
    def load(self) -> None:
        """Load weights into memory. Called once before the run loop."""
        ...

    @abstractmethod
    def run(self, image_path: str, question: str) -> InferenceResult:
        """
        Run inference on one (image, question) pair.
        Never raise — catch internally and return InferenceResult with error set.
        """
        ...

    def run_few_shot(
        self,
        ref_images: list[tuple[str, str]],
        test_image_path: str,
        question: str,
    ) -> "InferenceResult":
        """
        Multi-image few-shot inference.
        ref_images: list of (image_path, label) where label is 'ready' or 'not_ready'.
        Override in subclasses that support multiple images.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement run_few_shot")

    def run_two_image(
        self,
        full_image_path: str,
        crop_path: str,
        question: str,
    ) -> "InferenceResult":
        """
        Two-image inference: full room image + person crop shown together.
        Override in subclasses that support multiple images.
        Falls back to crop-only if not implemented.
        """
        return self.run(crop_path, question)

    def unload(self) -> None:
        """Optional: release GPU memory after benchmarking."""
        pass


class BaseDetectionModel(ABC):
    """
    Contract for CV object detection models (YOLO, SSD, etc.).
    Returns structured bounding box results rather than text.
    """

    name: str = "UnnamedDetector"

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory."""
        ...

    @abstractmethod
    def detect(self, image_path: str, conf_threshold: float = 0.25) -> DetectionResult:
        """
        Detect objects in a single image.
        Returns DetectionResult with bounding boxes.
        Never raise — catch internally and return DetectionResult with error set.
        """
        ...

    def unload(self) -> None:
        """Optional: release GPU/CPU memory after benchmarking."""
        pass
