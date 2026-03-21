# Quantize

Quantizes VLMs to int8 and saves them as standalone models in `/models/`.
Once saved, quantized models load like any normal local model — no re-quantizing on every run.

## Why save to disk?

Loading a bfloat16 model and quantizing on-the-fly takes the same RAM as the full model during loading.
Saving the pre-quantized weights means subsequent loads use only the int8 footprint from the start.

## Memory savings

| Model | bfloat16 | int8 | Saving |
|-------|----------|------|--------|
| InternVL3-4B | ~8GB | ~4GB | 50% |
| Qwen3-VL-4B | ~8GB | ~4GB | 50% |
| Qwen3-VL-8B | ~16GB | ~8GB | 50% |

## Setup

```bash
conda activate /mnt/shared/dils/envs/Qwen3VL-env
pip install bitsandbytes optimum
```

## Usage

```bash
cd quantize

# list available models and their status
python quantize.py --list

# quantize specific models
python quantize.py --models internvl qwen3vl_4b qwen3vl_8b

# quantize all
python quantize.py --all
```

Quantized models are saved to:
```
models/
├── InternVL3_5-4B-HF-int8/
├── Qwen3-VL-4B-Instruct-int8/
└── Qwen3-VL-8B-Instruct-int8/
```

## Adding to benchmark

After quantizing, add to `benchmark/benchmark_config.yaml`:

```yaml
  internvl_int8:
    enabled: true
    class: InternVLModel
    model_path: /mnt/shared/dils/projects/logitech/unified_eval_framework/models/InternVL3_5-4B-HF-int8
    dtype: bfloat16    # weights are already int8 on disk — load normally
    generation:
      max_new_tokens: 256

  qwen3vl_4b_int8:
    enabled: true
    class: Qwen3VLModel
    model_path: /mnt/shared/dils/projects/logitech/unified_eval_framework/models/Qwen3-VL-4B-Instruct-int8
    dtype: bfloat16
    generation:
      max_new_tokens: 256

  qwen3vl_8b_int8:
    enabled: true
    class: Qwen3VLModel
    model_path: /mnt/shared/dils/projects/logitech/unified_eval_framework/models/Qwen3-VL-8B-Instruct-int8
    dtype: bfloat16
    generation:
      max_new_tokens: 256
```

Note: use `dtype: bfloat16` in the config — the quantization is baked into the saved weights,
so the loader treats it like a normal local model.

## Run the benchmark

```bash
cd benchmark
conda activate /mnt/shared/dils/envs/Qwen3VL-env
export OPENAI_API_KEY=sk-...

CUDA_VISIBLE_DEVICES=0 python run_benchmark.py \
    --test-set test_sets/captioning_100.json \
    --models internvl_int8 qwen3vl_4b_int8

CUDA_VISIBLE_DEVICES=1 python run_benchmark.py \
    --test-set test_sets/captioning_100.json \
    --models qwen3vl_8b_int8
```
