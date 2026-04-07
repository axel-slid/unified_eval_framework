#!/usr/bin/env bash
# run_pipeline_all_vlms.sh
# Stage 1 (CV) runs once, then each VLM runs in its own subprocess to free GPU memory.

set -uo pipefail

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BENCHMARK_DIR}"

PYTHON="${PYTHON:-python3}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HOME=/mnt/shared/${USER}/hf_cache
export HUGGINGFACE_HUB_CACHE=/mnt/shared/${USER}/hf_cache
export CUDA_VISIBLE_DEVICES=0

VLMS=(smolvlm internvl internvl_int8 qwen3vl_4b qwen3vl_4b_int8 qwen3vl_8b qwen3vl_8b_int8)
TOTAL=${#VLMS[@]}

# ── Stage 1: CV detection (run once with smolvlm as a no-op VLM placeholder) ──
echo "====== Stage 1: CV Detection ======"
"${PYTHON}" runs/run_pipeline_people_analysis.py \
    --detectors yolo11n yolo11s mobilenet_ssd \
    --vlm smolvlm \
    2>&1

CV_JSON=$(ls -t results/pipeline_people_*.json 2>/dev/null | head -1)
if [[ -z "${CV_JSON}" ]]; then
    echo "ERROR: No CV results found. Aborting."
    exit 1
fi
echo ""
echo "CV results: ${CV_JSON}"

# ── Stage 2: one VLM per subprocess ───────────────────────────────────────────
echo ""
echo "====== Stage 2: VLM Analysis (${TOTAL} models) ======"

for i in "${!VLMS[@]}"; do
    VLM="${VLMS[$i]}"
    IDX=$((i + 1))
    echo ""
    echo "[${IDX}/${TOTAL}] VLM: ${VLM}"
    echo "-------------------------------------------------------"

    ran=0
    for GPU in 0 2 1; do
        echo "  Trying GPU ${GPU}..."
        if CUDA_VISIBLE_DEVICES=${GPU} "${PYTHON}" runs/run_pipeline_people_analysis.py \
            --vlm "${VLM}" \
            --detector-for-crops yolo11s \
            --cv-json "${CV_JSON}" \
            2>&1; then
            echo "  [${VLM}] done (GPU ${GPU})"
            ran=1
            break
        else
            echo "  [${VLM}] failed on GPU ${GPU}, trying next..."
        fi
    done
    if [[ $ran -eq 0 ]]; then
        echo "  [${VLM}] FAILED on all GPUs — skipping"
    fi
done

echo ""
echo "====== All done ======"
echo "Results in: results/pipeline_people_*.json"
