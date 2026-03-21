#!/usr/bin/env python


# %%


# %%
"""
quantize/quantize.py — quantize VLMs to int8 and save to disk.

Saves the quantized model to /mnt/shared/dils/models/<model>-int8/
so it can be loaded like any other local model (no re-quantizing on every run).

Usage:
    cd quantize
    conda activate /mnt/shared/dils/envs/Qwen3VL-env
    pip install bitsandbytes optimum

    # quantize all
    python quantize.py --all

    # quantize specific models
    python quantize.py --models internvl qwen3vl_4b qwen3vl_8b

    # list available models
    python quantize.py --list
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
QUANTIZE_DIR = Path(__file__).parent
PROJECT_ROOT = QUANTIZE_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"

os.environ.setdefault("HF_HOME", "/mnt/shared/dils/hf_cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/mnt/shared/dils/hf_cache")

# ── Model registry ─────────────────────────────────────────────────────────────
MODELS = {
    "internvl": {
        "label": "InternVL3-4B",
        "source_path": "OpenGVLab/InternVL3_5-4B-HF",
        "output_dir": str(MODELS_DIR / "InternVL3_5-4B-HF-int8"),
        "loader": "internvl",
    },
    "qwen3vl_4b": {
        "label": "Qwen3-VL-4B",
        "source_path": str(MODELS_DIR / "Qwen3-VL-4B-Instruct"),
        "output_dir": str(MODELS_DIR / "Qwen3-VL-4B-Instruct-int8"),
        "loader": "qwen3vl",
    },
    "qwen3vl_8b": {
        "label": "Qwen3-VL-8B",
        "source_path": str(MODELS_DIR / "Qwen3-VL-8B-Instruct"),
        "output_dir": str(MODELS_DIR / "Qwen3-VL-8B-Instruct-int8"),
        "loader": "qwen3vl",
    },
}


# ── Quantize functions ─────────────────────────────────────────────────────────

def quantize_internvl(source_path: str, output_dir: str) -> None:
    from transformers import AutoProcessor, InternVLForConditionalGeneration, BitsAndBytesConfig

    local_only = os.path.isdir(source_path)
    print(f"  Loading processor...")
    processor = AutoProcessor.from_pretrained(source_path, local_files_only=local_only)

    print(f"  Loading model in int8...")
    model = InternVLForConditionalGeneration.from_pretrained(
        source_path,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        local_files_only=local_only,
    )

    print(f"  Saving to {output_dir} ...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"  Done.")


def quantize_qwen3vl(source_path: str, output_dir: str) -> None:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig

    local_only = os.path.isdir(source_path)
    print(f"  Loading processor...")
    processor = AutoProcessor.from_pretrained(source_path, local_files_only=local_only)

    print(f"  Loading model in int8...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        source_path,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        local_files_only=local_only,
    )

    print(f"  Saving to {output_dir} ...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"  Done.")


LOADERS = {
    "internvl": quantize_internvl,
    "qwen3vl": quantize_qwen3vl,
}


# ── Main ──────────────────────────────────────────────────────────────────────

def run_quantize(keys: list[str]) -> None:
    for key in keys:
        if key not in MODELS:
            print(f"[SKIP] Unknown model '{key}'. Run --list to see options.")
            continue

        m = MODELS[key]
        output = m["output_dir"]

        if Path(output).exists() and any(Path(output).iterdir()):
            print(f"\n[SKIP] {m['label']} — already quantized at {output}")
            print(f"       Delete the folder to re-quantize.")
            continue

        print(f"\n{'='*60}")
        print(f"Quantizing: {m['label']}  →  int8")
        print(f"  Source : {m['source_path']}")
        print(f"  Output : {output}")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            LOADERS[m["loader"]](m["source_path"], output)
            elapsed = time.time() - t0
            size_gb = sum(
                f.stat().st_size for f in Path(output).rglob("*") if f.is_file()
            ) / 1e9
            print(f"\n  Quantized in {elapsed:.0f}s — saved {size_gb:.1f}GB to {output}")
        except Exception as e:
            import traceback
            print(f"\n  ERROR quantizing {key}: {e}")
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Quantize VLMs to int8 and save to disk")
    parser.add_argument("--models", nargs="+", help="Model keys to quantize")
    parser.add_argument("--all", action="store_true", help="Quantize all models")
    parser.add_argument("--list", action="store_true", help="List available models")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable models:")
        for key, m in MODELS.items():
            exists = Path(m["output_dir"]).exists()
            status = "✓ already quantized" if exists else "not yet quantized"
            print(f"  {key:<16} {m['label']:<20} → {m['output_dir']}")
            print(f"  {'':<16} {status}")
        return

    if args.all:
        keys = list(MODELS.keys())
    elif args.models:
        keys = args.models
    else:
        parser.print_help()
        return

    print(f"\nModels to quantize: {keys}")
    print(f"Output dir: {MODELS_DIR}/\n")
    run_quantize(keys)

    print("\n" + "="*60)
    print("DONE. Add these to benchmark_config.yaml:")
    print("="*60)
    for key in keys:
        if key not in MODELS:
            continue
        m = MODELS[key]
        print(f"""
  {key}_int8:
    enabled: true
    class: {_class_for(key)}
    model_path: {m["output_dir"]}
    dtype: bfloat16    # weights already int8 on disk, load normally
    generation:
      max_new_tokens: 256""")


def _class_for(key: str) -> str:
    return {
        "internvl": "InternVLModel",
        "qwen3vl_4b": "Qwen3VLModel",
        "qwen3vl_8b": "Qwen3VLModel",
    }.get(key, "UnknownModel")


if __name__ == "__main__":
    main()
