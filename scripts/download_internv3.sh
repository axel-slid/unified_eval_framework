#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="InternV3-env"
PYTHON_VERSION="3.11"
MODEL_ID="OpenGVLab/InternVL3_5-4B"
MODEL_DIR="${HOME}/models/InternVL3_5-4B"

echo "==> Checking conda"
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Install Miniconda or Anaconda first."
  exit 1
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "==> Creating conda environment: ${ENV_NAME}"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Environment ${ENV_NAME} already exists. Reusing it."
else
  conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
fi

echo "==> Activating environment"
conda activate "${ENV_NAME}"

echo "==> Upgrading pip"
python -m pip install --upgrade pip setuptools wheel

echo "==> Installing PyTorch"
# CUDA 12.1 wheels; change this if your system needs a different CUDA build or CPU-only.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "==> Installing model dependencies"
pip install \
  "transformers>=4.52.1" \
  "huggingface_hub[cli]" \
  safetensors \
  sentencepiece \
  accelerate \
  timm \
  pillow \
  numpy \
  torchvision \
  decord \
  einops

echo "==> Optional: installing flash-attn build helpers"
pip install ninja packaging

mkdir -p "${MODEL_DIR}"

echo "==> Logging into Hugging Face if token is available"
if [[ -n "${HF_TOKEN:-}" ]]; then
  huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential
else
  echo "HF_TOKEN is not set. Skipping login."
  echo "If you hit auth/rate-limit issues, run:"
  echo "  export HF_TOKEN=your_token_here"
  echo "  huggingface-cli login --token \"\$HF_TOKEN\""
fi

echo "==> Downloading model: ${MODEL_ID}"
huggingface-cli download "${MODEL_ID}" \
  --local-dir "${MODEL_DIR}" \
  --local-dir-use-symlinks False

echo
echo "==> Done"
echo "Environment: ${ENV_NAME}"
echo "Model path:  ${MODEL_DIR}"
echo
echo "To use it later:"
echo "  conda activate ${ENV_NAME}"
echo "  python - <<'PY'"
echo "from transformers import AutoTokenizer, AutoModel"
echo "import torch"
echo "path = '${MODEL_DIR}'"
echo "tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)"
echo "model = AutoModel.from_pretrained("
echo "    path,"
echo "    torch_dtype=torch.bfloat16,"
echo "    low_cpu_mem_usage=True,"
echo "    trust_remote_code=True"
echo ")"
echo "print('Model loaded successfully')"
echo "PY"