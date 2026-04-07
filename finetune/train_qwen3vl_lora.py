#!/usr/bin/env python
"""
finetune/train_qwen3vl_lora.py — LoRA fine-tune Qwen3-VL-4B for person bbox detection.

Architecture
------------
We freeze the vision encoder entirely and apply LoRA to the language model
attention projections (q_proj, k_proj, v_proj, o_proj) and the MLP gates
(gate_proj, up_proj, down_proj).  This is the minimum set needed to teach
the model a new output format while preserving its visual understanding.

Training objective
------------------
Causal LM loss on the FULL sequence (image + user prompt + assistant reply).
The assistant reply is a compact JSON string, so the loss is dominated by
bbox coordinate tokens — exactly what we want.

Usage
-----
  cd finetune/
  # single-GPU
  python train_qwen3vl_lora.py

  # multi-GPU with accelerate
  accelerate launch --num_processes 2 train_qwen3vl_lora.py

  # quick smoke-test (tiny dataset slice)
  python train_qwen3vl_lora.py --max-train-samples 64 --max-steps 20 --batch-size 1

Outputs
-------
  checkpoints/qwen3vl_bbox_lora/   — adapter weights + training state
  checkpoints/qwen3vl_bbox_lora/final/  — merged adapter (load like a normal model)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from qwen_vl_utils import process_vision_info

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
DATA_DIR   = ROOT / "data"
CKPT_DIR   = ROOT / "checkpoints" / "qwen3vl_bbox_lora"
MODEL_PATH = "/mnt/shared/dils/models/Qwen3-VL-4B-Instruct"

# ── LoRA config ────────────────────────────────────────────────────────────────
LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05
# Target the language model's attention + MLP (NOT the vision encoder)
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ── Dataset ────────────────────────────────────────────────────────────────────

class BboxDetectionDataset(Dataset):
    """
    Reads a JSONL file produced by prepare_dataset.py.
    Each line is a dict with a "messages" key in the Qwen3-VL chat format:
      messages[0]: user  → content = [{type:image, image:path}, {type:text, text:...}]
      messages[1]: assistant → content = JSON string
    """

    def __init__(self, jsonl_path: Path, processor, max_samples: int | None = None):
        self.processor = processor
        self.examples: list[dict] = []
        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        messages = ex["messages"]

        # Apply chat template to get the full text prompt
        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,  # assistant turn already included
        )

        # Load image
        image_inputs, video_inputs = process_vision_info(messages)

        # Tokenise
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=False,
            return_tensors="pt",
        )
        # Sequence tensors: [1, seq_len] → squeeze to [seq_len]
        # pixel_values: [N_patches, feat_dim] — already flat, keep as-is
        # image_grid_thw: [1, 3] — keep batch dim so collator can torch.cat to [B, 3]
        SEQ_KEYS = ("input_ids", "attention_mask", "mm_token_type_ids")
        out = {}
        for k, v in inputs.items():
            if k in SEQ_KEYS:
                out[k] = v.squeeze(0)
            else:
                out[k] = v  # pixel_values, image_grid_thw — unchanged
        return out


# ── Label masking ──────────────────────────────────────────────────────────────

def make_labels_mask_user_tokens(
    input_ids: torch.Tensor,
    processor,
) -> torch.Tensor:
    """
    Build labels tensor: -100 for everything EXCEPT the assistant's reply tokens.

    We find the assistant turn start by locating the assistant header token
    sequence in the tokenised input.  Everything before (user prompt + image
    tokens) is masked.  This focuses the loss on bbox coordinates only.
    """
    labels = input_ids.clone()

    # The assistant header produced by apply_chat_template.
    # Qwen3 uses: "<|im_start|>assistant\n"
    assistant_header = "<|im_start|>assistant\n"
    header_ids = processor.tokenizer.encode(
        assistant_header, add_special_tokens=False
    )
    n = len(header_ids)
    seq = input_ids.tolist()

    # Find last occurrence (there may be few-shot turns)
    pos = -1
    for i in range(len(seq) - n, -1, -1):
        if seq[i : i + n] == header_ids:
            pos = i + n  # first token of assistant content
            break

    if pos == -1:
        # Fallback: mask nothing (train on full sequence)
        return labels

    labels[:pos] = -100
    return labels


# ── Custom collator ────────────────────────────────────────────────────────────

class QwenBboxCollator:
    """
    Pads a batch of variable-length tokenised examples and builds label tensors
    that mask the user / image tokens.
    """

    def __init__(self, processor):
        self.processor = processor
        self.pad_token_id = processor.tokenizer.pad_token_id or 0

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].shape[0] for item in batch)

        input_ids_list, attention_mask_list, labels_list, mm_type_ids_list = [], [], [], []
        pixel_values_list, image_grid_thw_list = [], []

        for item in batch:
            seq_len = item["input_ids"].shape[0]
            pad_len = max_len - seq_len

            ids = torch.cat([
                item["input_ids"],
                torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
            ])
            mask = torch.cat([
                item["attention_mask"],
                torch.zeros(pad_len, dtype=torch.long),
            ])
            lbl = make_labels_mask_user_tokens(item["input_ids"], self.processor)
            lbl = torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)])

            input_ids_list.append(ids)
            attention_mask_list.append(mask)
            labels_list.append(lbl)

            # mm_token_type_ids: pad with 0 (text token type)
            if "mm_token_type_ids" in item:
                mm = torch.cat([
                    item["mm_token_type_ids"],
                    torch.zeros(pad_len, dtype=item["mm_token_type_ids"].dtype),
                ])
                mm_type_ids_list.append(mm)

            if "pixel_values" in item:
                pixel_values_list.append(item["pixel_values"])
            if "image_grid_thw" in item:
                image_grid_thw_list.append(item["image_grid_thw"])

        out = {
            "input_ids":      torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels":         torch.stack(labels_list),
        }
        if mm_type_ids_list:
            out["mm_token_type_ids"] = torch.stack(mm_type_ids_list)
        if pixel_values_list:
            # pixel_values: each item is [N_patches, feat_dim] — cat along patch dim
            out["pixel_values"] = torch.cat(pixel_values_list, dim=0)
        if image_grid_thw_list:
            # each item is [1, 3] — cat → [B, 3]
            out["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)

        return out


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model_and_processor(model_path: str, dtype: torch.dtype):
    os.environ.setdefault("HF_HOME", "/mnt/shared/dils/hf_cache")

    print(f"[train] Loading processor from {model_path} ...")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

    print(f"[train] Loading model ({dtype}) ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",          # splits across available GPUs
        local_files_only=True,
    )
    model.config.use_cache = False  # required for gradient checkpointing

    return model, processor


def apply_lora(model) -> "PeftModel":
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_TARGET_MODULES,
        # Only apply to the language model, not the vision encoder
        # (Qwen3-VL names the visual part "visual")
        modules_to_save=[],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


# ── Training ───────────────────────────────────────────────────────────────────

def train(args):
    dtype = torch.bfloat16

    model, processor = load_model_and_processor(MODEL_PATH, dtype)
    model = apply_lora(model)

    train_ds = BboxDetectionDataset(
        DATA_DIR / "coco_train.jsonl", processor, args.max_train_samples
    )
    val_ds = BboxDetectionDataset(
        DATA_DIR / "coco_val.jsonl", processor, args.max_val_samples
    )
    print(f"[train] train={len(train_ds)}  val={len(val_ds)}")

    collator = QwenBboxCollator(processor)

    training_args = TrainingArguments(
        output_dir=str(CKPT_DIR),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,   # critical for multi-modal inputs
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print("[train] Starting training ...")
    trainer.train()

    # Save the final adapter
    final_dir = CKPT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"[train] Adapter saved → {final_dir}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen3-VL-4B for bbox detection")
    parser.add_argument("--epochs",           type=int,   default=3)
    parser.add_argument("--max-steps",        type=int,   default=0,
                        help="Override epochs (0 = use --epochs)")
    parser.add_argument("--batch-size",       type=int,   default=2)
    parser.add_argument("--grad-accum",       type=int,   default=8,
                        help="Gradient accumulation steps (effective batch = batch-size × grad-accum)")
    parser.add_argument("--lr",               type=float, default=2e-4)
    parser.add_argument("--eval-steps",       type=int,   default=200)
    parser.add_argument("--max-train-samples",type=int,   default=None)
    parser.add_argument("--max-val-samples",  type=int,   default=None)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
