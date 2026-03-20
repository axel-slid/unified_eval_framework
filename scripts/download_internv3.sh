#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="InternV3-env"
PYTHON_VERSION="3.11"
MODEL_ID="OpenGVLab/InternVL3_5-4B"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="${PROJECT_ROOT}/models/InternVL3_5-4B"

echo "Project root: ${PROJECT_ROOT}"
echo "Model dir: ${MODEL_DIR}"

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

pip install \
  "transformers>=4.52.1" \
  "huggingface_hub[cli]" \
  accelerate \
  safetensors \
  sentencepiece \
  timm \
  pillow \
  numpy \
  eva-decord \
  einops

mkdir -p "${MODEL_DIR}"

if [[ -n "${HF_TOKEN:-}" ]]; then
  hf auth login --token "${HF_TOKEN}" --add-to-git-credential
else
  echo "HF_TOKEN not set (skipping login)"
fi

echo "Downloading model to: ${MODEL_DIR}"
hf download "${MODEL_ID}" \
  --local-dir "${MODEL_DIR}"

echo
echo "Done."
echo "Activate with:"
echo "  conda activate ${ENV_NAME}"
echo
echo "Model path:"
echo "  ${MODEL_DIR}"