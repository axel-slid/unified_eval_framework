#!/usr/bin/env bash
# scripts/infer.sh — Run a single VLM inference locally (CPU, Mac-friendly)
#
# Usage:
#   bash scripts/infer.sh -m smolvlm -i inferences/images/001.jpg
#   bash scripts/infer.sh -m smolvlm -i /path/to/image.jpg -p "What objects are on the table?"
#   bash scripts/infer.sh -m qwen3vl  -i inferences/images/001.jpg
#   bash scripts/infer.sh -m internvl -i inferences/images/001.jpg
#
# Flags:
#   -m  model   smolvlm | qwen3vl | internvl  (required)
#   -i  image   path to image file            (required)
#   -p  prompt  question/prompt string        (default: "Describe this image in detail.")
#   -t  tokens  max new tokens                (default: 128)

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL=""
IMAGE=""
PROMPT="Describe this image in detail."
MAX_TOKENS=128

# ── Parse flags ───────────────────────────────────────────────────────────────
while getopts "m:i:p:t:" opt; do
  case $opt in
    m) MODEL="$OPTARG" ;;
    i) IMAGE="$OPTARG" ;;
    p) PROMPT="$OPTARG" ;;
    t) MAX_TOKENS="$OPTARG" ;;
    *) echo "Usage: $0 -m <model> -i <image> [-p <prompt>] [-t <max_tokens>]"; exit 1 ;;
  esac
done

if [[ -z "$MODEL" || -z "$IMAGE" ]]; then
  echo "ERROR: -m <model> and -i <image> are required."
  echo "  Models: smolvlm | qwen3vl | internvl"
  exit 1
fi

if [[ ! -f "$IMAGE" ]]; then
  echo "ERROR: Image not found: $IMAGE"
  exit 1
fi

# ── Select conda env and HuggingFace model id ─────────────────────────────────
case "$MODEL" in
  smolvlm)
    CONDA_ENV="SmolVLM-env"
    FULL_NAME="HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    ARCH="smolvlm"
    ;;
  qwen3vl)
    CONDA_ENV="Qwen3VL-env"
    FULL_NAME="Qwen/Qwen3-VL-4B-Instruct"
    ARCH="qwen3vl"
    ;;
  internvl)
    CONDA_ENV="InternV3-env"
    FULL_NAME="OpenGVLab/InternVL3_5-4B-HF"
    ARCH="internvl"
    ;;
  *)
    echo "ERROR: Unknown model '$MODEL'. Choose: smolvlm | qwen3vl | internvl"
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOCAL_PATH="${PROJECT_ROOT}/models/$(basename "$FULL_NAME")"
PYTHON="/opt/anaconda3/envs/${CONDA_ENV}/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python not found at $PYTHON"
  echo "  Make sure the conda env '${CONDA_ENV}' exists."
  exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Model : $FULL_NAME"
echo "  Image : $IMAGE"
echo "  Prompt: $PROMPT"
echo "  Tokens: $MAX_TOKENS"
if [[ -d "$LOCAL_PATH" ]]; then
  echo "  Source: local ($LOCAL_PATH)"
else
  echo "  Source: HuggingFace Hub (will cache to ~/.cache/huggingface)"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Run inference ─────────────────────────────────────────────────────────────
"$PYTHON" - <<PYEOF
import sys, time
from pathlib import Path
from PIL import Image
import torch

FULL_NAME   = "$FULL_NAME"
ARCH        = "$ARCH"
IMAGE_PATH  = "$IMAGE"
PROMPT      = "$PROMPT"
MAX_TOKENS  = $MAX_TOKENS
LOCAL_PATH  = "$LOCAL_PATH"

path       = LOCAL_PATH if Path(LOCAL_PATH).is_dir() else FULL_NAME
local_only = Path(LOCAL_PATH).is_dir()

t0 = time.time()

if ARCH == "smolvlm":
    from transformers import AutoProcessor, SmolVLMForConditionalGeneration
    proc = AutoProcessor.from_pretrained(path, local_files_only=local_only)
    proc.image_processor.size           = {"longest_edge": 378}
    proc.image_processor.max_image_size = {"longest_edge": 378}
    mdl = SmolVLMForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.float32, device_map="cpu", local_files_only=local_only
    ).eval()
    image = Image.open(IMAGE_PATH).convert("RGB").resize((378, 378), Image.LANCZOS)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text_in = proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs  = proc(text=text_in, images=image, return_tensors="pt").to("cpu")
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)
    response = proc.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

elif ARCH == "qwen3vl":
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    proc = AutoProcessor.from_pretrained(path, local_files_only=local_only)
    mdl  = Qwen3VLForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.float32, device_map="cpu", local_files_only=local_only
    ).eval()
    messages = [{"role": "user", "content": [
        {"type": "image", "image": IMAGE_PATH},
        {"type": "text",  "text": PROMPT},
    ]}]
    text_in = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_inp, vid_inp = process_vision_info(messages)
    inputs = proc(text=[text_in], images=img_inp, videos=vid_inp,
                  padding=True, return_tensors="pt").to("cpu")
    with torch.no_grad():
        gen_ids = mdl.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)
    trimmed  = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
    response = proc.batch_decode(trimmed, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)[0]

elif ARCH == "internvl":
    from transformers import AutoProcessor, InternVLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(path, local_files_only=local_only)
    mdl  = InternVLForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.float32, device_map="cpu", local_files_only=local_only
    ).eval()
    image = Image.open(IMAGE_PATH).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text_in = proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs  = proc(text=text_in, images=image, return_tensors="pt").to("cpu")
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)
    response = proc.decode(out[0], skip_special_tokens=True)

elapsed = time.time() - t0
print(f"\n{'━'*40}")
print(f"  Response ({elapsed:.1f}s):")
print(f"{'━'*40}")
print(response)
print(f"{'━'*40}\n")
PYEOF
