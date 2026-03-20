from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class InferenceResult:
    response: str
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

    def unload(self) -> None:
        """Optional: release GPU memory after benchmarking."""
        pass
