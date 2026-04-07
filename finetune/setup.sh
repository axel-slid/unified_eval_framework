#!/usr/bin/env bash
# finetune/setup.sh — install training dependencies
# scipy and transformers/accelerate/bitsandbytes are already present in the env
set -e
pip install \
  peft==0.14.0 \
  trl==0.17.0 \
  datasets==3.5.0 \
  pycocotools==2.0.8 \
  einops \
  --quiet
echo "Done."
echo ""
echo "Workflow:"
echo "  1. python prepare_dataset.py              # ~30 min — downloads COCO + builds JSONL"
echo "  2. python prepare_dataset.py --max-images 200  # quick smoke-test first"
echo "  3. python train_qwen3vl_lora.py           # full training run"
echo "  4. python train_qwen3vl_lora.py --max-train-samples 64 --max-steps 5 --batch-size 1  # smoke-test"
echo "  5. python eval_finetuned.py --adapter checkpoints/qwen3vl_bbox_lora/final"
