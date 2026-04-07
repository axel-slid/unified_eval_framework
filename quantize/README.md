# Quantize

Quantizes VLMs to int8 using bitsandbytes and saves them as standalone models.

Once saved, quantized models load like any normal local model — no re-quantizing on every run. This is the key advantage over on-the-fly quantization: loading a bfloat16 model and quantizing it at runtime temporarily requires the same RAM as the full model. Saving pre-quantized weights means subsequent loads only use the int8 footprint from the start.

---

## Memory Savings

| Model | bfloat16 | int8 | Saving |
|-------|----------|------|--------|
| InternVL3-4B | ~8 GB | ~4 GB | 50% |
| Qwen3-VL-4B | ~8 GB | ~4 GB | 50% |
| Qwen3-VL-8B | ~16 GB | ~8 GB | 50% |

**Note on inference speed:** int8 models are often *slower* on GPU in this setup because bitsandbytes dequantizes weights during inference rather than executing native int8 kernels. For real inference speedups, use GGUF int4 models via llama.cpp (see `demo/`).

---

## Setup

```bash
conda activate /mnt/shared/<yourname>/envs/Qwen3VL-env
pip install bitsandbytes optimum
```

---

## Usage

```bash
cd quantize

# list available models and their current status
python quantize.py --list

# quantize specific models
python quantize.py --models internvl qwen3vl_4b qwen3vl_8b

# quantize all supported models
python quantize.py --all
```

Quantized models are saved alongside the originals:

```
models/
├── InternVL3_5-4B-HF-int8/
├── Qwen3-VL-4B-Instruct-int8/
└── Qwen3-VL-8B-Instruct-int8/
```

---

## Adding to the Benchmark

After quantizing, enable the int8 entries in `benchmark/benchmark_config.yaml`:

```yaml
internvl_int8:
  enabled: true
  class: InternVLModel
  model_path: /mnt/shared/<yourname>/models/InternVL3_5-4B-HF-int8
  dtype: bfloat16    # weights are already int8 on disk — load normally

qwen3vl_4b_int8:
  enabled: true
  class: Qwen3VLModel
  model_path: /mnt/shared/<yourname>/models/Qwen3-VL-4B-Instruct-int8
  dtype: bfloat16

qwen3vl_8b_int8:
  enabled: true
  class: Qwen3VLModel
  model_path: /mnt/shared/<yourname>/models/Qwen3-VL-8B-Instruct-int8
  dtype: bfloat16
```

Use `dtype: bfloat16` — the quantization is baked into the saved weights, so the loader treats them like a normal local model.

---

## Running the Benchmark

```bash
cd benchmark
conda activate /mnt/shared/<yourname>/envs/Qwen3VL-env
export OPENAI_API_KEY=sk-...

CUDA_VISIBLE_DEVICES=0 python run_benchmark_vqa.py \
    --test-set test_sets/captioning_100.json \
    --models internvl_int8 qwen3vl_4b_int8

CUDA_VISIBLE_DEVICES=1 python run_benchmark_vqa.py \
    --test-set test_sets/captioning_100.json \
    --models qwen3vl_8b_int8
```
