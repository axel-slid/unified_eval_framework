nohup bash -c '
export PYTHON=/mnt/shared/dils/envs/Qwen3VL-env/bin/python
export OPENAI_API_KEY=
export HF_HOME=/mnt/shared/dils/hf_cache
export HUGGINGFACE_HUB_CACHE=/mnt/shared/dils/hf_cache
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd /mnt/shared/dils/projects/logitech/unified_eval_framework/benchmark

CUDA_VISIBLE_DEVICES=2 $PYTHON run_benchmark_vqa.py \
    --test-set test_sets/captioning_100.json \
    --all
' >> logs/vqa_run.log 2>&1 &

echo "Started PID $! — tail logs/vqa_run.log to follow"