#!/usr/bin/env bash
# run_all_models_prompting.sh
# Runs each model in its own process (releases GPU memory between models),
# then generates the merged report.
#
# Usage:
#   cd benchmark
#   bash run_all_models_prompting.sh
#   bash run_all_models_prompting.sh --techniques direct cot   # subset of techniques
#   bash run_all_models_prompting.sh --models smolvlm qwen3vl_4b

set -uo pipefail

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BENCHMARK_DIR}"

PYTHON="/mnt/shared/dils/envs/Qwen3VL-env/bin/python"
TEST_SET="test_sets/meeting_room_sample.json"
EXTRA_ARGS="${@}"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HOME=/mnt/shared/dils/hf_cache
export HUGGINGFACE_HUB_CACHE=/mnt/shared/dils/hf_cache

RUN_TS=$(date +%Y%m%d_%H%M%S)
MERGE_DIR="results/prompting_run_${RUN_TS}"
mkdir -p "${MERGE_DIR}"

MODELS=(smolvlm internvl internvl_int8 qwen3vl_4b qwen3vl_4b_int8 qwen3vl_8b qwen3vl_8b_int8)
TOTAL=${#MODELS[@]}

# If --models is passed in EXTRA_ARGS, use that subset instead
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

    if CUDA_VISIBLE_DEVICES=0 "${PYTHON}" run_benchmark_prompting_techniques.py \
        --test-set "${TEST_SET}" \
        --models "${MODEL}" \
        ${EXTRA_ARGS/--models*/} \
        2>&1; then
        # Find the just-created result file (latest prompting_techniques_results_*.json)
        LATEST=$(ls -t results/prompting_techniques_results_*.json 2>/dev/null | head -1 || true)
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
echo " All models done. Generating merged report..."
echo "======================================================="

"${PYTHON}" generate_prompting_report.py \
    --merge-dir "${MERGE_DIR}" \
    --save-pngs

echo ""
echo "Done. Open results/prompting_techniques_report_*.html"
