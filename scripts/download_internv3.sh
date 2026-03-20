#!/usr/bin/env bash
set -euo pipefail

########################################
# Config
########################################
ENV_NAME="InternV3-env"
PYTHON_VERSION="3.11"
MODEL_ID="OpenGVLab/InternVL3_5-4B"

# Resolve project root (one level up from /scripts)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

MODEL_DIR="${PROJECT_ROOT}/models/InternVL3_5-4B"

########################################
# Setup conda
########################################
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Install Miniconda/Anaconda first."
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

########################################
# Create environment (if needed)
########################################
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Using existing env: ${ENV_NAME}"
else
  echo "Creating env: ${ENV_NAME}"
  conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"

########################################
# Install dependencies
########################################
python -m pip install --upgrade pip

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
  decord \
  einops

########################################
# Prepare model directory
########################################
mkdir -p "${MODEL_DIR}"

########################################
# Hugging Face login (optional)
########################################
if [[ -n "${HF_TOKEN:-}" ]]; then
  huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential
else
  echo "HF_TOKEN not set (skipping login)"
fi

########################################
# Download model
########################################
echo "Downloading model to: ${MODEL_DIR}"

huggingface-cli download "${MODEL_ID}" \
  --local-dir "${MODEL_DIR}" \
  --local-dir-use-symlinks False

########################################
# Done
########################################
echo
echo "Done."
echo "Model saved at: ${MODEL_DIR}"
echo
echo "Activate env with:"
echo "  conda activate ${ENV_NAME}"