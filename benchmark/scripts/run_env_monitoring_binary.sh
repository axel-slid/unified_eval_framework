#!/usr/bin/env bash
# run_env_monitoring_binary.sh
# Runs the binary yes/no env monitoring benchmark across all models,
# one process per model to release GPU memory between runs.
#
# Usage:
#   cd benchmark
#   bash run_env_monitoring_binary.sh
#   bash run_env_monitoring_binary.sh --models smolvlm qwen3vl_4b

set -uo pipefail

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BENCHMARK_DIR}"

PYTHON="${PYTHON:-python3}"
EXTRA_ARGS="${@}"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HOME=/mnt/shared/${USER}/hf_cache
export HUGGINGFACE_HUB_CACHE=/mnt/shared/${USER}/hf_cache

RUN_TS=$(date +%Y%m%d_%H%M%S)
MERGE_DIR="results/env_monitoring_binary_run_${RUN_TS}"
mkdir -p "${MERGE_DIR}"

MODELS=(smolvlm internvl internvl_int8 qwen3vl_4b qwen3vl_4b_int8 qwen3vl_8b qwen3vl_8b_int8)
TOTAL=${#MODELS[@]}

# If --models is passed, use that subset instead
if echo "${EXTRA_ARGS}" | grep -q '\-\-models'; then
    MODELS=()
    reading=0
    for arg in ${EXTRA_ARGS}; do
        if [[ "${arg}" == "--models" ]]; then reading=1; continue; fi
        if [[ "${arg}" == --* ]] && [[ "${arg}" != "--models" ]]; then reading=0; fi
        if [[ $reading -eq 1 ]]; then MODELS+=("${arg}"); fi
    done
    TOTAL=${#MODELS[@]}
fi

RESULT_FILES=()

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    IDX=$((i + 1))
    echo ""
    echo "[${IDX}/${TOTAL}] Running model: ${MODEL}"
    echo "======================================================="

    OUT_JSON="${MERGE_DIR}/${MODEL}.json"

    if CUDA_VISIBLE_DEVICES=0 "${PYTHON}" runs/run_benchmark_env_monitoring_binary.py \
        --models "${MODEL}" \
        2>&1; then
        LATEST=$(ls -t results/env_monitoring_binary_*.json 2>/dev/null | head -1 || true)
        if [[ -n "${LATEST}" ]]; then
            cp "${LATEST}" "${OUT_JSON}"
            RESULT_FILES+=("${OUT_JSON}")
            echo "  Saved → ${OUT_JSON}"
        fi
    else
        echo "  [SKIP] ${MODEL} failed (OOM or error) — continuing"
    fi
done

echo ""
echo "======================================================="
echo " All models done."
echo "======================================================="
echo ""
echo "Results in: ${MERGE_DIR}"
echo "Latest HTML report: $(ls -t results/env_monitoring_binary_report_*.html 2>/dev/null | head -1 || echo 'none')"
