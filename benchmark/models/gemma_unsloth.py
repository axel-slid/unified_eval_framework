"""
gemma_unsloth.py — Gemma 4 E2B (and Gemma 3) with optional quantization.

Backend selection (automatic):
  • CUDA  → Unsloth FastVisionModel  (optimal; supports 4-bit BnB)
  • MPS / CPU → HuggingFace transformers + torchao weight-only quant

Quantization levels:
  4-bit  — torchao int4_weight_only  (~4 GB, fastest on device)
  8-bit  — torchao int8_weight_only  (~6 GB)
  bf16   — full bfloat16, no quant   (~10 GB)

2-bit / 1-bit via GPU require GGUF + llama-cpp and are not implemented here.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

import torch
from PIL import Image as PILImage

from models.base import BaseVLMModel, InferenceResult
from config import ModelConfig


# ── Registry ──────────────────────────────────────────────────────────────────

QUANT_CONFIGS: dict[str, dict] = {
    # ── Gemma 4 E2B ──────────────────────────────────────────────────────────
    "gemma4_e2b_4bit": {
        "hf_id":      "google/gemma-4-E2B-it",
        "unsloth_id": "unsloth/gemma-4-E2B-it",
        "quant":      "4bit",
        "label":      "Gemma 4 E2B · 4-bit",
        "vram_hint":  "~4 GB",
    },
    "gemma4_e2b_8bit": {
        "hf_id":      "google/gemma-4-E2B-it",
        "unsloth_id": "unsloth/gemma-4-E2B-it",
        "quant":      "8bit",
        "label":      "Gemma 4 E2B · 8-bit",
        "vram_hint":  "~6 GB",
    },
    "gemma4_e2b_bf16": {
        "hf_id":      "google/gemma-4-E2B-it",
        "unsloth_id": "unsloth/gemma-4-E2B-it",
        "quant":      "bf16",
        "label":      "Gemma 4 E2B · bf16",
        "vram_hint":  "~10 GB",
    },
    # ── Gemma 3 4B ───────────────────────────────────────────────────────────
    "gemma3_4b_4bit": {
        "hf_id":      "google/gemma-3-4b-it",
        "unsloth_id": "unsloth/gemma-3-4b-it",
        "quant":      "4bit",
        "label":      "Gemma 3 4B · 4-bit",
        "vram_hint":  "~4 GB",
    },
    "gemma3_4b_bf16": {
        "hf_id":      "google/gemma-3-4b-it",
        "unsloth_id": "unsloth/gemma-3-4b-it",
        "quant":      "bf16",
        "label":      "Gemma 3 4B · bf16",
        "vram_hint":  "~8 GB",
    },
}


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _has_unsloth() -> bool:
    if not torch.cuda.is_available():
        return False  # unsloth requires CUDA; skip import to avoid warning
    try:
        import unsloth  # noqa: F401
        return True
    except Exception:
        return False


class GemmaUnslothModel(BaseVLMModel):
    """
    Gemma 4 E2B / Gemma 3 multimodal inference.

    Selects backend at load time:
      CUDA + unsloth  → FastVisionModel (fastest)
      MPS / CPU       → AutoModelForImageTextToText + torchao quant
    """

    def __init__(self, cfg: ModelConfig):
        self.cfg   = cfg
        self.qcfg  = QUANT_CONFIGS.get(cfg.key) or QUANT_CONFIGS["gemma4_e2b_4bit"]
        self.name  = self.qcfg["label"]
        self.model     = None
        self.processor = None
        self._device   = _best_device()
        self._backend  = "unsloth" if (self._device == "cuda" and _has_unsloth()) else "hf"

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        if self._backend == "unsloth":
            self._load_unsloth()
        else:
            self._load_hf()

    def _load_unsloth(self) -> None:
        from unsloth import FastVisionModel

        model_id = self.cfg.model_path or self.qcfg["unsloth_id"]
        print(f"[{self.name}] Unsloth path — {model_id}  ({self.qcfg['vram_hint']})")
        quant = self.qcfg["quant"]
        kwargs: dict = {}
        if quant == "4bit":
            kwargs["load_in_4bit"] = True
        elif quant == "8bit":
            kwargs["load_in_8bit"] = True
        elif quant == "bf16":
            kwargs["dtype"] = torch.bfloat16

        self.model, self.processor = FastVisionModel.from_pretrained(model_id, **kwargs)
        FastVisionModel.for_inference(self.model)
        print(f"[{self.name}] Ready (unsloth)")

    def _load_hf(self) -> None:
        from transformers import AutoProcessor, AutoModelForImageTextToText

        model_id = self.cfg.model_path or self.qcfg["hf_id"]
        print(f"[{self.name}] HF path — {model_id}  ({self.qcfg['vram_hint']}, device={self._device})")

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        ).to(self._device)
        self.model.eval()

        # Apply torchao weight-only quantization where supported
        quant = self.qcfg["quant"]
        if quant in ("4bit", "8bit"):
            self._apply_torchao(quant)

        print(f"[{self.name}] Ready (HF transformers)")

    def _apply_torchao(self, quant: str) -> None:
        try:
            from torchao.quantization import quantize_, int4_weight_only, int8_weight_only
            scheme = int4_weight_only() if quant == "4bit" else int8_weight_only()
            quantize_(self.model, scheme)
            print(f"[{self.name}] torchao {quant} quantization applied")
        except Exception as e:
            print(f"[{self.name}] torchao quant skipped ({e}); running bf16")

    def unload(self) -> None:
        del self.model
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Inference ─────────────────────────────────────────────────────────────

    def run(self, image_path: str, question: str) -> InferenceResult:
        return self._infer([image_path], question)

    def run_two_image(
        self, full_image_path: str, crop_path: str, question: str
    ) -> InferenceResult:
        return self._infer([full_image_path, crop_path], question)

    def run_streaming(
        self,
        image_path: str,
        question: str,
        token_callback: Callable[[str, float, int], None],
        done_callback: Callable[[InferenceResult], None],
    ) -> None:
        """Non-blocking streaming inference. Spawns a daemon thread."""
        threading.Thread(
            target=self._stream_worker,
            args=(image_path, question, token_callback, done_callback),
            daemon=True,
        ).start()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _build_inputs_unsloth(self, image_paths: list[str], question: str):
        images  = [PILImage.open(p).convert("RGB") for p in image_paths]
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": question})
        messages   = [{"role": "user", "content": content}]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        img_arg    = images[0] if len(images) == 1 else images
        inputs     = self.processor(
            img_arg, input_text, add_special_tokens=False, return_tensors="pt"
        )
        device = next(self.model.parameters()).device
        return {k: v.to(device) for k, v in inputs.items()}

    def _build_inputs_hf(self, image_paths: list[str], question: str):
        images  = [PILImage.open(p).convert("RGB") for p in image_paths]
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": question})
        messages   = [{"role": "user", "content": content}]
        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(
            text=input_text,
            images=images,
            return_tensors="pt",
        )
        return {k: v.to(self._device) for k, v in inputs.items()}

    def _build_inputs(self, image_paths: list[str], question: str):
        if self._backend == "unsloth":
            return self._build_inputs_unsloth(image_paths, question)
        return self._build_inputs_hf(image_paths, question)

    def _infer(self, image_paths: list[str], question: str) -> InferenceResult:
        try:
            inputs = self._build_inputs(image_paths, question)
            gen_kw = {
                **inputs,
                "max_new_tokens": self.cfg.generation.max_new_tokens,
                "do_sample":      self.cfg.generation.do_sample,
                "use_cache":      True,
            }
            t0 = time.perf_counter()
            with torch.no_grad():
                out_ids = self.model.generate(**gen_kw)
            latency_ms = (time.perf_counter() - t0) * 1000
            in_len   = inputs["input_ids"].shape[1]
            new_ids  = out_ids[0][in_len:]
            response = self.processor.decode(new_ids, skip_special_tokens=True)
            return InferenceResult(response=response, latency_ms=latency_ms)
        except Exception as e:
            return InferenceResult(response="", latency_ms=0.0, error=str(e))

    def _stream_worker(
        self,
        image_path: str,
        question: str,
        token_callback: Callable[[str, float, int], None],
        done_callback: Callable[[InferenceResult], None],
    ) -> None:
        try:
            from transformers import TextIteratorStreamer

            inputs   = self._build_inputs([image_path], question)
            streamer = TextIteratorStreamer(
                self.processor if hasattr(self.processor, "decode") else self.processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            gen_kw = {
                **inputs,
                "max_new_tokens": self.cfg.generation.max_new_tokens,
                "do_sample":      self.cfg.generation.do_sample,
                "use_cache":      True,
                "streamer":       streamer,
            }

            gen_thread = threading.Thread(
                target=lambda: self.model.generate(**gen_kw),
                daemon=True,
            )
            t_start = time.perf_counter()
            gen_thread.start()

            chunks: list[str] = []
            n_tokens = 0
            for chunk in streamer:
                if not chunk:
                    continue
                n_tokens += 1
                chunks.append(chunk)
                elapsed = time.perf_counter() - t_start
                tps     = n_tokens / elapsed if elapsed > 0 else 0.0
                token_callback(chunk, tps, n_tokens)

            gen_thread.join()
            latency_ms = (time.perf_counter() - t_start) * 1000
            done_callback(InferenceResult(response="".join(chunks), latency_ms=latency_ms))

        except Exception as e:
            done_callback(InferenceResult(response="", latency_ms=0.0, error=str(e)))
