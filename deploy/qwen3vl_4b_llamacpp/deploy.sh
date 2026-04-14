#!/usr/bin/env bash
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PORT="${PORT:-8080}"
THREADS="${THREADS:-4}"
CTX_SIZE="${CTX_SIZE:-2048}"
MAX_IMAGE_SIDE="${MAX_IMAGE_SIDE:-720}"
BACKEND="${BACKEND:-auto}"
PROMPT="${PROMPT:-Describe the image clearly and concisely. If there are people, mention what they are doing.}"
MODE="${1:-serve}"
IMAGE_PATH="${2:-}"

RUNNER="${DEPLOY_DIR}/run_qwen3vl_4b_llamacpp.py"

install_llama_cpp() {
  echo "llama-server not found. Attempting to install llama.cpp..."
  if command -v brew >/dev/null 2>&1; then
    brew install llama.cpp
  else
    echo "Homebrew not found. Install llama.cpp manually: https://github.com/ggml-org/llama.cpp" >&2
    exit 1
  fi
}

choose_backend() {
  if [[ "${BACKEND}" != "auto" ]]; then
    echo "${BACKEND}"
    return
  fi
  if command -v llama-server >/dev/null 2>&1; then
    echo "local"
  else
    install_llama_cpp
    if command -v llama-server >/dev/null 2>&1; then
      echo "local"
    elif command -v docker >/dev/null 2>&1; then
      echo "docker"
    else
      echo "none"
    fi
  fi
}

ensure_python_deps() {
  "${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys
mods = ["requests", "PIL", "huggingface_hub"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print("MISSING:" + ",".join(missing))
    sys.exit(3)
PY
}

install_python_deps() {
  echo "Installing Python dependencies..."
  "${PYTHON_BIN}" -m pip install --upgrade requests pillow huggingface_hub
}

BACKEND_CHOSEN="$(choose_backend)"
if [[ "${BACKEND_CHOSEN}" == "none" ]]; then
  echo "Neither llama-server nor docker is available." >&2
  echo "Install llama.cpp (preferred) or Docker, then rerun." >&2
  exit 1
fi

if ! ensure_python_deps >/tmp/qwen_deploy_deps_check.txt 2>&1; then
  if grep -q '^MISSING:' /tmp/qwen_deploy_deps_check.txt; then
    install_python_deps
  else
    cat /tmp/qwen_deploy_deps_check.txt >&2
    exit 1
  fi
fi
rm -f /tmp/qwen_deploy_deps_check.txt

echo "Using backend: ${BACKEND_CHOSEN}"
echo "Preparing GGUF assets if needed..."
"${PYTHON_BIN}" "${RUNNER}" prepare

COMMON_ARGS=(
  --backend "${BACKEND_CHOSEN}"
  --port "${PORT}"
  --threads "${THREADS}"
  --ctx-size "${CTX_SIZE}"
)

if [[ "${MODE}" == "serve" ]]; then
  echo "Starting Qwen3-VL-4B llama.cpp server on port ${PORT}..."
  exec "${PYTHON_BIN}" "${RUNNER}" serve "${COMMON_ARGS[@]}"
elif [[ "${MODE}" == "infer" ]]; then
  if [[ -z "${IMAGE_PATH}" ]]; then
    echo "Usage: $0 infer /absolute/path/to/image.jpg" >&2
    exit 1
  fi
  echo "Running one-shot inference for ${IMAGE_PATH}..."
  exec "${PYTHON_BIN}" "${RUNNER}" infer \
    "${COMMON_ARGS[@]}" \
    --start-server \
    --image "${IMAGE_PATH}" \
    --prompt "${PROMPT}" \
    --max-image-side "${MAX_IMAGE_SIDE}"
else
  echo "Unknown mode: ${MODE}" >&2
  echo "Usage:" >&2
  echo "  $0 serve" >&2
  echo "  $0 infer /absolute/path/to/image.jpg" >&2
  exit 1
fi
