nohup bash -c '
export PYTHON=${PYTHON:-python3}
export OPENAI_API_KEY=
export HF_HOME=/mnt/shared/${USER}/hf_cache
export HUGGINGFACE_HUB_CACHE=/mnt/shared/${USER}/hf_cache
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=2 $PYTHON runs/run_benchmark_vqa.py \
    --test-set test_sets/captioning_100.json \
    --all
' >> logs/vqa_run.log 2>&1 &

echo "Started PID $! — tail logs/vqa_run.log to follow"