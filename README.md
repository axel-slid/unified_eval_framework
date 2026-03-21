# Unified VLM Eval Framework

A lightweight, extensible benchmarking framework for vision-language models (VLMs). Models run locally on GPU and are evaluated using two complementary methods:

1. **Captioning benchmark** (`run_benchmark.py`) — models describe images, scored 0–100 by GPT-as-judge
2. **VQA benchmark** (`run_benchmark_vqa.py`) — GPT generates 5 targeted questions per image, all models answer them, GPT judges and acts as its own baseline

Both benchmarks produce embedded-image HTML reports for easy sharing and comparison.

---

## Repo Structure

```
unified_eval_framework/
├── scripts/
│   ├── download_smolvlm.sh           ← env setup + model download for SmolVLM2
│   ├── download_internv3.sh          ← env setup + model download for InternVL3.5
│   ├── download_qwen3vl_4b.sh        ← env setup + model download for Qwen3-VL 4B
│   └── download_qwen3vl_8b.sh        ← env setup + model download for Qwen3-VL 8B
├── inferences/
│   ├── SmolVLM2-2.2B-Base.py        ← standalone test inference for SmolVLM2
│   ├── InternV3_5-4B.py              ← standalone test inference for InternVL3.5
│   ├── Qwen3VL-4B.py                 ← standalone test inference for Qwen3-VL 4B
│   ├── Qwen3VL-8B.py                 ← standalone test inference for Qwen3-VL 8B
│   └── images/                       ← sample images for test inferences
├── benchmark/
│   ├── benchmark_config.yaml         ← all model/judge/run settings (edit paths here)
│   ├── config.py                     ← loads + validates YAML into dataclasses
│   ├── run_benchmark.py              ← captioning benchmark entry point
│   ├── run_benchmark_vqa.py          ← VQA benchmark entry point
│   ├── run_all_models.sh             ← runs all models + merges into one report
│   ├── judge.py                      ← LLM-as-judge scorer (OpenAI API, 0–100)
│   ├── models/
│   │   ├── __init__.py               ← MODEL_REGISTRY (class name → class)
│   │   ├── base.py                   ← BaseVLMModel interface
│   │   ├── smolvlm.py                ← SmolVLM2 runner (supports bfloat16 + int8)
│   │   ├── internvl.py               ← InternVL3 runner (supports bfloat16 + int8)
│   │   └── qwen3vl.py                ← Qwen3-VL runner (supports bfloat16 + int8)
│   ├── test_sets/
│   │   ├── sample.json               ← 3-image smoke test
│   │   ├── captioning_100.json       ← 100-image diverse test set
│   │   ├── generate_test_set.py      ← build a test set from a local image folder
│   │   └── download_test_images.py   ← download images from Wikimedia Commons
│   └── results/                      ← auto-created; JSON + HTML report per run
├── quantize/
│   ├── quantize.py                   ← quantize models to int8 and save to disk
│   └── README.md
├── demo/
│   ├── server.py                     ← CPU-only FastAPI backend (webcam inference)
│   ├── frontend/index.html           ← webcam UI with auto-capture and response log
│   └── README.md
├── docs/
│   └── report_preview.png
└── models/                           ← downloaded weights (gitignored)
```

---

## Prerequisites

- [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html)
- An OpenAI API key (used by the LLM judge)
- A GPU with at least 8GB VRAM (16GB+ for Qwen3-VL 8B)
- Sufficient disk space — model weights are 5–18GB each

> **Shared server tip:** If your home directory has a disk quota, redirect caches before doing anything:
> ```bash
> export HF_HOME=/mnt/shared/<yourname>/hf_cache
> export HUGGINGFACE_HUB_CACHE=/mnt/shared/<yourname>/hf_cache
> echo 'export HF_HOME=/mnt/shared/<yourname>/hf_cache' >> ~/.bashrc
> echo 'export HUGGINGFACE_HUB_CACHE=/mnt/shared/<yourname>/hf_cache' >> ~/.bashrc
> mkdir -p /mnt/shared/<yourname>/envs /mnt/shared/<yourname>/conda_pkgs
> conda config --add envs_dirs /mnt/shared/<yourname>/envs
> conda config --add pkgs_dirs /mnt/shared/<yourname>/conda_pkgs
> ```

---

## Quick Start — Reproduce Our Results

Everything from clone to benchmark, copy-paste top to bottom. Replace `<yourname>` with your username throughout.

```bash
# 1. Clone
git clone https://github.com/axel-slid/unified_eval_framework.git
cd unified_eval_framework

# 2. Redirect caches (shared server only)
export HF_HOME=/mnt/shared/<yourname>/hf_cache
export HUGGINGFACE_HUB_CACHE=/mnt/shared/<yourname>/hf_cache
echo 'export HF_HOME=/mnt/shared/<yourname>/hf_cache' >> ~/.bashrc
echo 'export HUGGINGFACE_HUB_CACHE=/mnt/shared/<yourname>/hf_cache' >> ~/.bashrc

# 3. Download all models (~41GB total)
bash scripts/download_smolvlm.sh         # SmolVLM2-2.2B   (~5GB)  → SmolVLM-env
bash scripts/download_internv3.sh        # InternVL3.5-4B  (~9GB)  → InternV3-env
bash scripts/download_qwen3vl_4b.sh      # Qwen3-VL-4B     (~9GB)  → Qwen3VL-env
bash scripts/download_qwen3vl_8b.sh      # Qwen3-VL-8B     (~18GB) → Qwen3VL-env

# 4. Update model paths in benchmark_config.yaml
#    Change model_path values to point to your local downloads
#    e.g. /mnt/shared/<yourname>/models/SmolVLM2-2.2B-Instruct

# 5. Download 100 test images
cd benchmark
conda activate /mnt/shared/<yourname>/envs/Qwen3VL-env
python test_sets/download_test_images.py --count 100

# 6. Run captioning benchmark (all 4 bfloat16 models)
export OPENAI_API_KEY=sk-...
nohup bash -c '
export PYTHON=/mnt/shared/<yourname>/envs/Qwen3VL-env/bin/python
export OPENAI_API_KEY=sk-...
export HF_HOME=/mnt/shared/<yourname>/hf_cache
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /path/to/unified_eval_framework/benchmark
CUDA_VISIBLE_DEVICES=0 $PYTHON run_benchmark.py --test-set test_sets/captioning_100.json --models smolvlm
CUDA_VISIBLE_DEVICES=0 $PYTHON run_benchmark.py --test-set test_sets/captioning_100.json --models internvl
CUDA_VISIBLE_DEVICES=0 $PYTHON run_benchmark.py --test-set test_sets/captioning_100.json --models qwen3vl_4b
CUDA_VISIBLE_DEVICES=1 $PYTHON run_benchmark.py --test-set test_sets/captioning_100.json --models qwen3vl_8b
' >> logs/benchmark_run.log 2>&1 &

# 7. (Optional) Run VQA benchmark — GPT generates questions + acts as baseline
nohup bash -c '
export PYTHON=/mnt/shared/<yourname>/envs/Qwen3VL-env/bin/python
export OPENAI_API_KEY=sk-...
export HF_HOME=/mnt/shared/<yourname>/hf_cache
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /path/to/unified_eval_framework/benchmark
CUDA_VISIBLE_DEVICES=0 $PYTHON run_benchmark_vqa.py \
    --test-set test_sets/captioning_100.json --all
' >> logs/vqa_run.log 2>&1 &

# Watch progress
tail -f logs/benchmark_run.log
```

Results land in `benchmark/results/` — open the `.html` file in any browser.

---

## Benchmark Results

### Captioning Benchmark (0–100, gpt-5.4-mini judge)

Each model generates a free-form image description. GPT scores 0–100 based on accuracy, completeness, and absence of hallucination. All models run on the same 100 diverse Wikimedia Commons images.

| Model | Params | dtype | GPU | Avg Score | Avg Latency | N |
|-------|--------|-------|-----|-----------|-------------|---|
| SmolVLM2-2.2B-Instruct | 2.2B | bfloat16 | RTX 4060 Ti (16GB) | 73.6 / 100 | 2879ms | 100 |
| InternVL3-4B-HF | 4B | bfloat16 | RTX 4060 Ti (16GB) | 82.2 / 100 | 4760ms | 100 |
| Qwen3-VL-4B-Instruct | 4B | bfloat16 | RTX 4060 Ti (16GB) | 78.6 / 100 | 6195ms | 100 |
| Qwen3-VL-8B-Instruct | 8B | bfloat16 | RTX PRO 6000 (97GB) | 77.0 / 100 | 5254ms | 100 |
| InternVL3-4B-HF | 4B | int8 | RTX PRO 6000 (97GB) | 79.7 / 100 | 16058ms | 100 |
| Qwen3-VL-4B-Instruct | 4B | int8 | RTX PRO 6000 (97GB) | 78.5 / 100 | 21506ms | 100 |
| Qwen3-VL-8B-Instruct | 8B | int8 | RTX PRO 6000 (97GB) | 79.6 / 100 | 21167ms | 100 |

### VQA Benchmark (0–100, GPT-generated questions + GPT baseline)

GPT generates 5 targeted questions per image and provides reference answers. All models answer the same questions and are scored against GPT's reference. GPT itself is scored as a theoretical ceiling.

| Model | Params | dtype | Avg Score | vs GPT Baseline | Avg Latency | N |
|-------|--------|-------|-----------|-----------------|-------------|---|
| GPT Baseline (gpt-5.4-mini) | — | — | 90.9 / 100 | baseline | 1064ms | 100 |
| SmolVLM2-2.2B-Instruct | 2.2B | bfloat16 | 72.0 / 100 | −18.9 | 315ms | 100 |
| InternVL3-4B-HF | 4B | bfloat16 | 65.3 / 100 | −25.6 | 1400ms | 100 |
| Qwen3-VL-4B-Instruct | 4B | bfloat16 | 88.6 / 100 | −2.3 | 2547ms | 100 |
| Qwen3-VL-8B-Instruct | 8B | bfloat16 | 88.3 / 100 | −2.6 | 3267ms | 100 |
| InternVL3-4B-HF | 4B | int8 | 62.9 / 100 | −28.0 | 5415ms | 100 |
| Qwen3-VL-4B-Instruct | 4B | int8 | 87.5 / 100 | −3.5 | 10843ms | 100 |
| Qwen3-VL-8B-Instruct | 8B | int8 | 87.4 / 100 | −3.5 | 13460ms | 100 |

![VQA Benchmark Report](docs/report_preview.png)

### Key Findings

**Qwen3-VL dominates on VQA.** Both Qwen3-VL-4B and 8B score within 2–3 points of the GPT ceiling on VQA tasks, while InternVL3 lags significantly (−25 points). This suggests Qwen3-VL is substantially better at instruction-following and precise question answering — which matters more than captioning ability for real deployment tasks.

**The two benchmarks disagree.** InternVL3 scores highest on captioning (82.2) but lowest on VQA (65.3). SmolVLM2 is the inverse — mediocre at captioning but competitive on VQA for its size. This shows why captioning alone is a poor proxy for task-specific performance.

**int8 quantization trades speed for quality inconsistently.** For Qwen3-VL, int8 loses only 1 point on VQA. For InternVL3, int8 loses 2.4 points. However int8 is 3–4x *slower* on GPU in our setup — this is because bitsandbytes int8 dequantizes during inference rather than using native int8 kernels. For real deployment speedups, use GGUF int4 via llama.cpp instead.

**SmolVLM2 is the efficiency winner.** At 315ms per image (VQA) it is 4–10x faster than other models with only a modest quality penalty. For high-throughput or edge deployment it is the clear choice.

---

## Features

### 1. Captioning Benchmark (`run_benchmark.py`)

Runs one or more models on a test set of images with a fixed prompt, scores responses 0–100 using GPT-as-judge, and produces a combined HTML report with embedded images and per-response score bars.

```bash
cd benchmark
export OPENAI_API_KEY=sk-...

# single model
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py \
    --test-set test_sets/captioning_100.json --models smolvlm

# multiple models in one report
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py \
    --test-set test_sets/captioning_100.json \
    --models smolvlm internvl qwen3vl_4b
```

### 2. VQA Benchmark (`run_benchmark_vqa.py`)

A more rigorous evaluation pipeline:

1. GPT looks at each image and generates 5 targeted questions with reference answers
2. Every model answers all 5 questions per image
3. GPT judges each answer against its reference (0–100)
4. GPT itself answers and is scored as a baseline (~90–95)
5. The `vs GPT` column shows the quality gap to a frontier model

```bash
# run all enabled models + GPT baseline
CUDA_VISIBLE_DEVICES=0 python run_benchmark_vqa.py \
    --test-set test_sets/captioning_100.json --all

# skip GPT baseline to save API cost
CUDA_VISIBLE_DEVICES=0 python run_benchmark_vqa.py \
    --models smolvlm internvl --no-gpt-baseline
```

### 3. int8 Quantization (`quantize/quantize.py`)

Quantizes models to int8 using bitsandbytes and saves them to disk. Once saved, quantized models load like any normal local model — no re-quantizing on every run.

```bash
cd quantize
pip install bitsandbytes

# list models and their status
python quantize.py --list

# quantize all supported models
python quantize.py --all

# quantize specific models
python quantize.py --models internvl qwen3vl_4b qwen3vl_8b
```

int8 memory savings:

| Model | bfloat16 | int8 |
|-------|----------|------|
| InternVL3-4B | ~8GB | ~4GB |
| Qwen3-VL-4B | ~8GB | ~4GB |
| Qwen3-VL-8B | ~16GB | ~8GB |

### 4. CPU Webcam Demo (`demo/`)

A live webcam demo optimised for CPU-only inference (no GPU required). Uses SmolVLM2 in float32 locked to 4 CPU threads to simulate edge hardware like ARM Cortex @ 2.80GHz.

```bash
cd demo
pip install fastapi uvicorn python-multipart
python server.py
# open http://localhost:8080
```

Features: live webcam feed, manual capture, auto-capture mode (every 2/4/8/15s), response log with thumbnails and latency, 4-core CPU load visualisation.

---

## Full Workflow

### Step 1 — Download models

```bash
bash scripts/download_smolvlm.sh
bash scripts/download_internv3.sh
bash scripts/download_qwen3vl_4b.sh
bash scripts/download_qwen3vl_8b.sh
```

| Model | Script | Conda Env | Disk |
|-------|--------|-----------|------|
| SmolVLM2-2.2B-Instruct | `download_smolvlm.sh` | `SmolVLM-env` | ~5GB |
| InternVL3.5-4B-HF | `download_internv3.sh` | `InternV3-env` | ~9GB |
| Qwen3-VL-4B-Instruct | `download_qwen3vl_4b.sh` | `Qwen3VL-env` | ~9GB |
| Qwen3-VL-8B-Instruct | `download_qwen3vl_8b.sh` | `Qwen3VL-env` | ~18GB |

> Qwen3-VL env is created at `/mnt/shared/<yourname>/envs/Qwen3VL-env`. Activate with the full path.

### Step 2 — Update `benchmark_config.yaml`

Change `model_path` for each model to your local download location:

```yaml
models:
  smolvlm:
    enabled: true
    class: SmolVLMModel
    model_path: /mnt/shared/<yourname>/models/SmolVLM2-2.2B-Instruct
    dtype: bfloat16

  internvl:
    enabled: true
    class: InternVLModel
    model_path: OpenGVLab/InternVL3_5-4B-HF   # or local path
    dtype: bfloat16

  qwen3vl_4b:
    enabled: true
    class: Qwen3VLModel
    model_path: /mnt/shared/<yourname>/models/Qwen3-VL-4B-Instruct
    dtype: bfloat16

  qwen3vl_8b:
    enabled: true
    class: Qwen3VLModel
    model_path: /mnt/shared/<yourname>/models/Qwen3-VL-8B-Instruct
    dtype: bfloat16
```

### Step 3 — Download test images

```bash
cd benchmark
conda activate /mnt/shared/<yourname>/envs/Qwen3VL-env
python test_sets/download_test_images.py --count 100
```

### Step 4 — Run benchmark

**Captioning (nohup, all models):**

```bash
mkdir -p logs
nohup bash -c '
export PYTHON=/mnt/shared/<yourname>/envs/Qwen3VL-env/bin/python
export OPENAI_API_KEY=sk-...
export HF_HOME=/mnt/shared/<yourname>/hf_cache
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /path/to/benchmark

CUDA_VISIBLE_DEVICES=0 $PYTHON run_benchmark.py --test-set test_sets/captioning_100.json --models smolvlm
CUDA_VISIBLE_DEVICES=0 $PYTHON run_benchmark.py --test-set test_sets/captioning_100.json --models internvl
CUDA_VISIBLE_DEVICES=0 $PYTHON run_benchmark.py --test-set test_sets/captioning_100.json --models qwen3vl_4b
CUDA_VISIBLE_DEVICES=1 $PYTHON run_benchmark.py --test-set test_sets/captioning_100.json --models qwen3vl_8b
' >> logs/benchmark_run.log 2>&1 &
tail -f logs/benchmark_run.log
```

**VQA (nohup, all models + GPT baseline):**

```bash
nohup bash -c '
export PYTHON=/mnt/shared/<yourname>/envs/Qwen3VL-env/bin/python
export OPENAI_API_KEY=sk-...
export HF_HOME=/mnt/shared/<yourname>/hf_cache
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /path/to/benchmark
CUDA_VISIBLE_DEVICES=0 $PYTHON run_benchmark_vqa.py \
    --test-set test_sets/captioning_100.json --all
' >> logs/vqa_run.log 2>&1 &
tail -f logs/vqa_run.log
```

### Step 5 — View results

Results in `benchmark/results/`:
- `results_<timestamp>.json` — raw scores, responses, latencies
- `report_<timestamp>.html` — side-by-side HTML with embedded images and score bars
- `vqa_results_<timestamp>.json` — VQA raw results
- `vqa_report_<timestamp>.html` — VQA HTML with per-question breakdown

---

## Adding Your Own Model

3 steps, no changes to `run_benchmark.py` or `run_benchmark_vqa.py`.

### 1. Create `benchmark/models/yourmodel.py`

```python
from __future__ import annotations
import time, torch
from PIL import Image
from transformers import AutoProcessor, YourModelClass
from models.base import BaseVLMModel, InferenceResult
from config import ModelConfig

class YourModel(BaseVLMModel):
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.name = f"YourModel ({cfg.model_path.split('/')[-1]})"
        self.model = self.processor = None

    def load(self) -> None:
        self.processor = AutoProcessor.from_pretrained(self.cfg.model_path)
        self.model = YourModelClass.from_pretrained(
            self.cfg.model_path, torch_dtype=torch.bfloat16, device_map="cuda",
        ).eval()

    def run(self, image_path: str, question: str) -> InferenceResult:
        try:
            image = Image.open(image_path).convert("RGB")
            t0 = time.perf_counter()
            # outputs = self.model.generate(...)
            return InferenceResult(response="...", latency_ms=(time.perf_counter()-t0)*1000)
        except Exception as e:
            return InferenceResult(response="", latency_ms=0.0, error=str(e))

    def unload(self) -> None:
        del self.model; self.model = None
        torch.cuda.empty_cache()
```

### 2. Register in `benchmark/models/__init__.py`

```python
from .yourmodel import YourModel
MODEL_REGISTRY = {
    "SmolVLMModel": SmolVLMModel,
    "InternVLModel": InternVLModel,
    "Qwen3VLModel": Qwen3VLModel,
    "YourModel": YourModel,   # ← add this
}
```

### 3. Add to `benchmark_config.yaml`

```yaml
yourmodel:
  enabled: true
  class: YourModel
  model_path: /path/to/model
  dtype: bfloat16
  generation:
    max_new_tokens: 256
```

---

## Scoring Scale

| Range | Meaning |
|-------|---------|
| 90–100 | Fully correct and complete |
| 70–89 | Mostly correct, minor issues |
| 50–69 | Partially correct, missing key details |
| 20–49 | Mostly wrong, some correct elements |
| 0–19 | Wrong, refused, or heavy hallucination |

**Estimated API cost:** ~$0.10–0.20 per 100 images for VQA (question generation + 8 model judgements per image). ~$0.03 per 100 images for captioning.

---

## Published Benchmark Scores

| Model | MMMU | MathVista | DocVQA | ChartQA | TextVQA | OCRBench | AI2D | ScienceQA |
|-------|------|-----------|--------|---------|---------|----------|------|-----------|
| SmolVLM2-2.2B | 42.0 | 51.5 | 80.0 | 68.7 | 73.0 | 72.9 | 70.0 | 89.6 |
| InternVL3.5-4B | 56.6 | 67.2 | 91.6 | 86.0 | 78.4 | — | 82.6 | — |
| Qwen3-VL-4B | ~58 | ~72 | ~94 | ~85 | — | — | — | — |
| Qwen3-VL-8B | ~65 | 85.8 | ~97 | ~89 | — | — | — | — |

> Sources: SmolVLM2 [arXiv:2504.05299](https://arxiv.org/abs/2504.05299) · InternVL3.5 [arXiv:2508.18265](https://arxiv.org/abs/2508.18265) · Qwen3-VL [arXiv:2511.21631](https://arxiv.org/abs/2511.21631)

---

## Troubleshooting

**CUDA out of memory**
```bash
nvidia-smi        # find the PID
kill <pid>        # free it, then re-run
```

**`CondaError: Run conda init before conda activate`** — use explicit python path in nohup scripts:
```bash
export PYTHON=/mnt/shared/<yourname>/envs/Qwen3VL-env/bin/python
$PYTHON run_benchmark.py ...
```

**`ImportError: cannot import name X from transformers`**
```bash
pip install git+https://github.com/huggingface/transformers  # Qwen3-VL needs source
pip install --upgrade "transformers>=4.52.1"                 # others
```

**`ImportError: Package num2words is required`**
```bash
/mnt/shared/<yourname>/envs/Qwen3VL-env/bin/pip install num2words
```

**Judge scores all 0** — `OPENAI_API_KEY` not set. Benchmark still saves responses — export the key and re-run.

**Model ignores the image** — use the `-Instruct` variant, not `-Base`.

**Qwen3-VL env not found** — activate by full path:
```bash
conda activate /mnt/shared/<yourname>/envs/Qwen3VL-env
```

**int8 models are slower than bfloat16** — expected with bitsandbytes on GPU. It dequantizes during inference rather than using native int8 kernels. For true speedup on CPU/edge, use GGUF int4 via llama.cpp.
