from .base import BaseVLMModel, InferenceResult
from .smolvlm import SmolVLMModel
from .internvl import InternVLModel
from .qwen3vl import Qwen3VLModel

# Registry: class name (string from YAML `class:` field) → Python class
# Add new models here when you create a new runner file
MODEL_REGISTRY: dict[str, type[BaseVLMModel]] = {
    "SmolVLMModel": SmolVLMModel,
    "InternVLModel": InternVLModel,
    "Qwen3VLModel": Qwen3VLModel,
}

__all__ = [
    "BaseVLMModel",
    "InferenceResult",
    "SmolVLMModel",
    "InternVLModel",
    "Qwen3VLModel",
    "MODEL_REGISTRY",
]
