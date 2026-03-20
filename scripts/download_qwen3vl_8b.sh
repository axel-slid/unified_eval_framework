#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="Qwen3VL-env"
PYTHON_VERSION="3.11"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Project root: ${PROJECT_ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Install Miniconda or Anaconda first."
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Using existing conda env: ${ENV_NAME}"
else
  echo "Creating conda env: ${ENV_NAME}"
  conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel

pip install torch torchvision torchaudio

# Qwen3-VL requires transformers built from source (4.57.0 not yet released)
pip install git+https://github.com/huggingface/transformers

pip install \
  "huggingface_hub[cli]" \
  accelerate \
  safetensors \
  sentencepiece \
  pillow \
  numpy \
  qwen-vl-utils \
  einops

# ── Download 8B model ─────────────────────────────────────────────────────────
MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
MODEL_DIR="${PROJECT_ROOT}/models/Qwen3-VL-8B-Instruct"
mkdir -p "${MODEL_DIR}"

if [[ -n "${HF_TOKEN:-}" ]]; then
  hf auth login --token "${HF_TOKEN}" --add-to-git-credential
else
  echo "HF_TOKEN not set (skipping login)"
fi

echo "Downloading model to: ${MODEL_DIR}"
hf download "${MODEL_ID}" --local-dir "${MODEL_DIR}"

echo ""
echo "Done."
echo "Activate with:  conda activate ${ENV_NAME}"
echo "Model path:     ${MODEL_DIR}"
