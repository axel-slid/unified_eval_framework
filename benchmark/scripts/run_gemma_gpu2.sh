#!/usr/bin/env bash
# run_gemma_gpu2.sh
# Runs Gemma E2B (4-bit), E2B (8-bit HF), and E4B (4-bit) on GPU 2
# then regenerates the binary figures.
#
# Usage: bash scripts/run_gemma_gpu2.sh
# From: benchmark/ directory

set -euo pipefail

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BENCHMARK_DIR}"

ENV_PATH="${CONDA_ENV_PATH:-/mnt/shared/${USER}/envs/Qwen3VL-env}"
PYTHON="${ENV_PATH}/bin/python"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HOME=/mnt/shared/${USER}/hf_cache
export HUGGINGFACE_HUB_CACHE=/mnt/shared/${USER}/hf_cache
export CUDA_VISIBLE_DEVICES=2

echo "======================================================"
echo " Gemma Benchmark — GPU 2"
echo "======================================================"
echo ""

echo "[1/3] Gemma E2B 4-bit"
"${PYTHON}" runs/run_benchmark_env_monitoring_binary.py --models gemma_e2b_4bit
echo ""

echo "[2/3] Gemma E2B 8-bit HF"
"${PYTHON}" runs/run_benchmark_env_monitoring_binary.py --models gemma_e2b_8bit_hf
echo ""

echo "[3/3] Gemma E4B 4-bit"
"${PYTHON}" runs/run_benchmark_env_monitoring_binary.py --models gemma_e4b_4bit
echo ""

echo "======================================================"
echo " Regenerating figures..."
echo "======================================================"
"${PYTHON}" reports/generate_binary_figures.py

echo ""
echo "Done."
