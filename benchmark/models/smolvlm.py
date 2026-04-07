from __future__ import annotations

import os
import time

import torch
from PIL import Image
from transformers import AutoProcessor, SmolVLMForConditionalGeneration

from models.base import BaseVLMModel, InferenceResult
from config import ModelConfig

TILE_SIZE = 378  # must be divisible by scale_factor(3) * patch_size(14) = 42


class SmolVLMModel(BaseVLMModel):

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.name = f"SmolVLM2 ({cfg.model_path.split('/')[-1]})"
        self.model = None
        self.processor = None

    def load(self) -> None:
        print(f"[{self.name}] Loading from {self.cfg.model_path} ...")

        os.environ.setdefault("HF_HOME", "/mnt/shared/dils/hf_cache")
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/mnt/shared/dils/hf_cache")

        local_only = os.path.isdir(self.cfg.model_path)

        self.processor = AutoProcessor.from_pretrained(
            self.cfg.model_path,
            local_files_only=local_only,
        )
        self.processor.image_processor.size = {"longest_edge": TILE_SIZE}
        self.processor.image_processor.max_image_size = {"longest_edge": TILE_SIZE}

        self.model = SmolVLMForConditionalGeneration.from_pretrained(
            self.cfg.model_path,
            torch_dtype=_resolve_dtype(self.cfg.dtype),
            device_map="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
            local_files_only=local_only,
        )
        self.model = self.model.eval()
        print(f"[{self.name}] Ready")

    def run(self, image_path: str, question: str) -> InferenceResult:
        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            text_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            device     = next(self.model.parameters()).device
            model_dtype = next(self.model.parameters()).dtype
            inputs = self.processor(
                text=text_prompt,
                images=image,
                return_tensors="pt",
            ).to(device)
            # Cast floating-point inputs to the model dtype (bfloat16 / float16)
            inputs = {
                k: v.to(model_dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                for k, v in inputs.items()
            }

            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self.cfg.generation.to_dict(),
                )
            latency_ms = (time.perf_counter() - t0) * 1000

            generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
            response = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            return InferenceResult(response=response, latency_ms=latency_ms)

        except Exception as e:
            return InferenceResult(response="", latency_ms=0.0, error=str(e))

    def run_few_shot(
        self,
        ref_images: list[tuple[str, str]],
        test_image_path: str,
        question: str,
    ) -> InferenceResult:
        try:
            pil_images = []
            content = []
            for img_path, label in ref_images:
                tag = "READY" if label == "ready" else "NOT READY"
                content.append({"type": "text", "text": f"[{tag}]"})
                content.append({"type": "image"})
                img = Image.open(img_path).convert("RGB")
                img = img.resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)
                pil_images.append(img)

            content.append({"type": "text", "text": question})
            content.append({"type": "image"})
            test_img = Image.open(test_image_path).convert("RGB")
            test_img = test_img.resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)
            pil_images.append(test_img)

            messages = [{"role": "user", "content": content}]
            text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

            device      = next(self.model.parameters()).device
            model_dtype = next(self.model.parameters()).dtype
            inputs = self.processor(
                text=text_prompt,
                images=pil_images,
                return_tensors="pt",
            ).to(device)
            inputs = {
                k: v.to(model_dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                for k, v in inputs.items()
            }

            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **self.cfg.generation.to_dict())
            latency_ms = (time.perf_counter() - t0) * 1000

            generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
            response = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            return InferenceResult(response=response, latency_ms=latency_ms)
        except Exception as e:
            return InferenceResult(response="", latency_ms=0.0, error=str(e))

    def run_two_image(self, full_image_path: str, crop_path: str, question: str) -> InferenceResult:
        try:
            full_img = Image.open(full_image_path).convert("RGB")
            full_img = full_img.resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)
            crop_img = Image.open(crop_path).convert("RGB")
            crop_img = crop_img.resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)

            messages = [{"role": "user", "content": [
                {"type": "text", "text": "Image 1: full meeting room."},
                {"type": "image"},
                {"type": "text", "text": "Image 2: detected person crop."},
                {"type": "image"},
                {"type": "text", "text": question},
            ]}]
            text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

            device = next(self.model.parameters()).device
            model_dtype = next(self.model.parameters()).dtype
            inputs = self.processor(
                text=text_prompt,
                images=[full_img, crop_img],
                return_tensors="pt",
            ).to(device)
            inputs = {
                k: v.to(model_dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                for k, v in inputs.items()
            }

            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **self.cfg.generation.to_dict())
            latency_ms = (time.perf_counter() - t0) * 1000

            generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
            response = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            return InferenceResult(response=response, latency_ms=latency_ms)
        except Exception as e:
            return InferenceResult(response="", latency_ms=0.0, error=str(e))

    def unload(self) -> None:
        del self.model
        self.model = None
        _empty_cache()


def _resolve_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}.get(name, torch.bfloat16)


def _empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
