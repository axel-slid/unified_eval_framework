#!/usr/bin/env bash
# run_all_models.sh
# Runs all 4 models sequentially (one per GPU call) then merges into one report.
# Usage: bash run_all_models.sh
# From: benchmark/ directory

set -euo pipefail

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BENCHMARK_DIR}"

ENV_PATH="/mnt/shared/dils/envs/Qwen3VL-env"
PYTHON="${ENV_PATH}/bin/python"
TEST_SET="test_sets/captioning_100.json"

# ── Check OPENAI_API_KEY ──────────────────────────────────────────────────────
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY not set. Run: export OPENAI_API_KEY=sk-..."
  exit 1
fi

# ── Check test set exists ─────────────────────────────────────────────────────
if [[ ! -f "${TEST_SET}" ]]; then
  echo "Test set not found at ${TEST_SET}"
  echo "Downloading 100 images first..."
  "${PYTHON}" test_sets/download_test_images.py --count 100
fi

echo ""
echo "======================================================"
echo " Running VLM Benchmark — all 4 models"
echo "======================================================"
echo ""

export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HOME=/mnt/shared/dils/hf_cache
export HUGGINGFACE_HUB_CACHE=/mnt/shared/dils/hf_cache

# ── SmolVLM2 ─────────────────────────────────────────────────────────────────
echo "[1/4] SmolVLM2-2.2B"
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" run_benchmark.py \
    --test-set "${TEST_SET}" \
    --models smolvlm
echo ""

# ── InternVL3 ────────────────────────────────────────────────────────────────
echo "[2/4] InternVL3-4B"
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" run_benchmark.py \
    --test-set "${TEST_SET}" \
    --models internvl
echo ""

# ── Qwen3-VL 4B ──────────────────────────────────────────────────────────────
echo "[3/4] Qwen3-VL-4B"
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" run_benchmark.py \
    --test-set "${TEST_SET}" \
    --models qwen3vl_4b
echo ""

# ── Qwen3-VL 8B (needs more VRAM — uses GPU 2) ───────────────────────────────
echo "[4/4] Qwen3-VL-8B"
CUDA_VISIBLE_DEVICES=2 "${PYTHON}" run_benchmark.py \
    --test-set "${TEST_SET}" \
    --models qwen3vl_8b
echo ""

# ── Merge all results into one combined report ────────────────────────────────
echo "======================================================"
echo " Merging results into combined report..."
echo "======================================================"

"${PYTHON}" - << 'PYEOF'
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")
from run_benchmark import save_html_report

results_dir = Path("results")

# Load all individual result JSONs and merge into one dict
all_results = {}
for f in sorted(results_dir.glob("results_*.json")):
    if "merged" in f.name:
        continue
    try:
        data = json.loads(f.read_text())
        # Only take the latest run for each model key
        for key, val in data.items():
            all_results[key] = val
    except Exception as e:
        print(f"Skipping {f.name}: {e}")

if not all_results:
    print("No result files found in results/")
    sys.exit(1)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save merged JSON
merged_path = results_dir / f"results_merged_{timestamp}.json"
merged_path.write_text(json.dumps(all_results, indent=2))
print(f"Merged JSON  → {merged_path}")

# Save combined HTML report
html_path = results_dir / f"report_all_models_{timestamp}.html"
save_html_report(all_results, html_path, timestamp)
print(f"HTML report  → {html_path}")

# Print summary table
print(f"\n{'='*65}")
print("SUMMARY")
print(f"{'='*65}")
print(f"{'Model':<32} {'Avg Score':>10} {'Avg Latency':>14} {'N':>5}")
print("-" * 65)
for key, results in all_results.items():
    scores = [r["score"] for r in results if r["score"] is not None]
    latencies = [r["latency_ms"] for r in results if r["latency_ms"] > 0]
    name = results[0]["model_name"] if results else key
    avg_score = f"{statistics.mean(scores):.2f}" if scores else "—"
    avg_lat = f"{statistics.mean(latencies):.0f}ms" if latencies else "—"
    print(f"{name:<32} {avg_score:>10} {avg_lat:>14} {len(scores):>5}")
PYEOF

echo ""
echo "Done. Open results/report_all_models_*.html in a browser."
