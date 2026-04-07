from __future__ import annotations

import os
import time

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from models.base import BaseVLMModel, InferenceResult
from config import ModelConfig


class Qwen3VLModel(BaseVLMModel):

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.name = f"Qwen3-VL ({cfg.model_path.split('/')[-1]})"
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
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.cfg.model_path,
            torch_dtype=_resolve_dtype(self.cfg.dtype),
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            local_files_only=local_only,
        )
        self.model = self.model.eval()
        print(f"[{self.name}] Ready")

    def run(self, image_path: str, question: str) -> InferenceResult:
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text",  "text": question},
                    ],
                }
            ]

            text_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            image_inputs, video_inputs = process_vision_info(messages)

            device = next(self.model.parameters()).device
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)

            t0 = time.perf_counter()
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    **self.cfg.generation.to_dict(),
                )
            latency_ms = (time.perf_counter() - t0) * 1000

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

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
            content = []
            for img_path, label in ref_images:
                tag = "READY" if label == "ready" else "NOT READY"
                content.append({"type": "text", "text": f"[{tag}]"})
                content.append({"type": "image", "image": img_path})
            content.append({"type": "text", "text": question})
            content.append({"type": "image", "image": test_image_path})

            messages = [{"role": "user", "content": content}]
            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            device = next(self.model.parameters()).device
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)

            t0 = time.perf_counter()
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **self.cfg.generation.to_dict())
            latency_ms = (time.perf_counter() - t0) * 1000

            generated_ids_trimmed = [
                out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return InferenceResult(response=response, latency_ms=latency_ms)
        except Exception as e:
            return InferenceResult(response="", latency_ms=0.0, error=str(e))

    def run_two_image(self, full_image_path: str, crop_path: str, question: str) -> InferenceResult:
        try:
            messages = [{"role": "user", "content": [
                {"type": "text", "text": "Image 1: full meeting room."},
                {"type": "image", "image": full_image_path},
                {"type": "text", "text": "Image 2: detected person crop."},
                {"type": "image", "image": crop_path},
                {"type": "text", "text": question},
            ]}]
            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            device = next(self.model.parameters()).device
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)

            t0 = time.perf_counter()
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **self.cfg.generation.to_dict())
            latency_ms = (time.perf_counter() - t0) * 1000

            generated_ids_trimmed = [
                out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return InferenceResult(response=response, latency_ms=latency_ms)
        except Exception as e:
            return InferenceResult(response="", latency_ms=0.0, error=str(e))

    def unload(self) -> None:
        del self.model
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _resolve_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(name, torch.bfloat16)
