# Unified VLM Eval Framework

A lightweight, extensible benchmarking framework for vision-language models (VLMs). Models run locally on GPU and are scored 1–5 by an LLM-as-judge (GPT) on image-question pairs with hand-written rubrics.

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
│   ├── benchmark_config.yaml         ← all model/judge/run settings live here
│   ├── config.py                     ← loads + validates YAML into dataclasses
│   ├── run_benchmark.py              ← entry point
│   ├── run_all_models.sh             ← runs all models + merges into one report
│   ├── judge.py                      ← LLM-as-judge scorer (OpenAI API)
│   ├── models/
│   │   ├── __init__.py               ← MODEL_REGISTRY (class name → class)
│   │   ├── base.py                   ← BaseVLMModel interface
│   │   ├── smolvlm.py
│   │   ├── internvl.py
│   │   └── qwen3vl.py
│   ├── test_sets/
│   │   ├── sample.json               ← small hand-written test set (3 items)
│   │   ├── captioning_100.json       ← auto-generated 100-image captioning test set
│   │   ├── generate_test_set.py      ← build a test set from a local image folder
│   │   └── download_test_images.py   ← download images from Wikimedia Commons
│   └── results/                      ← auto-created; JSON + HTML report per run
└── models/                           ← downloaded model weights (gitignored)
```

---

## Prerequisites

- [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html)
- An OpenAI API key (used by the LLM judge)
- A GPU with at least 8GB VRAM (16GB+ for Qwen3-VL 8B)
- Sufficient disk space — model weights are 5–18GB each

> **Shared server tip:** If your home directory has a disk quota, redirect everything to a larger partition before doing anything else:
> ```bash
> # Redirect HuggingFace cache
> export HF_HOME=/mnt/shared/<yourname>/hf_cache
> export HUGGINGFACE_HUB_CACHE=/mnt/shared/<yourname>/hf_cache
> echo 'export HF_HOME=/mnt/shared/<yourname>/hf_cache' >> ~/.bashrc
> echo 'export HUGGINGFACE_HUB_CACHE=/mnt/shared/<yourname>/hf_cache' >> ~/.bashrc
>
> # Redirect conda envs + packages
> mkdir -p /mnt/shared/<yourname>/envs /mnt/shared/<yourname>/conda_pkgs
> conda config --add envs_dirs /mnt/shared/<yourname>/envs
> conda config --add pkgs_dirs /mnt/shared/<yourname>/conda_pkgs
> ```
> The Qwen3-VL download scripts handle this automatically using explicit env paths.

---

## Quick Start

Everything from clone to benchmark results, copy-paste top to bottom:

```bash
# ── 1. Clone ──────────────────────────────────────────────────────────────────
git clone https://github.com/axel-slid/unified_eval_framework.git
cd unified_eval_framework

# ── 2. Shared server only: redirect caches to avoid home dir quota ─────────────
export HF_HOME=/mnt/shared/<yourname>/hf_cache
export HUGGINGFACE_HUB_CACHE=/mnt/shared/<yourname>/hf_cache
echo 'export HF_HOME=/mnt/shared/<yourname>/hf_cache' >> ~/.bashrc
echo 'export HUGGINGFACE_HUB_CACHE=/mnt/shared/<yourname>/hf_cache' >> ~/.bashrc

# ── 3. Download all models ─────────────────────────────────────────────────────
bash scripts/download_smolvlm.sh         # SmolVLM2-2.2B   (~5GB)  → SmolVLM-env
bash scripts/download_internv3.sh        # InternVL3.5-4B  (~9GB)  → InternV3-env
bash scripts/download_qwen3vl_4b.sh      # Qwen3-VL-4B     (~9GB)  → Qwen3VL-env
bash scripts/download_qwen3vl_8b.sh      # Qwen3-VL-8B     (~18GB) → Qwen3VL-env

# ── 4. Verify each model loads (should print an image description) ─────────────
conda activate SmolVLM-env
python inferences/SmolVLM2-2.2B-Base.py

conda activate InternV3-env
python inferences/InternV3_5-4B.py

conda activate /mnt/shared/<yourname>/envs/Qwen3VL-env
python inferences/Qwen3VL-4B.py
python inferences/Qwen3VL-8B.py

# ── 5. Download 100 test images ────────────────────────────────────────────────
cd benchmark
conda activate /mnt/shared/<yourname>/envs/Qwen3VL-env
python test_sets/download_test_images.py --count 100

# ── 6. Run all models and generate combined report ─────────────────────────────
export OPENAI_API_KEY=sk-...
bash run_all_models.sh
```

Results are saved to `benchmark/results/` — open `report_all_models_<timestamp>.html` in any browser.

---

## Sample Report

100-image captioning benchmark, scored by `gpt-5.4-mini` as judge:

| Model | Params | Avg Score | Avg Latency | N |
|-------|--------|-----------|-------------|---|
| SmolVLM2-2.2B-Instruct | 2.2B | 3.70 / 5 | 3448ms | 100 |
| InternVL3-4B-HF | 4B | 4.00 / 5 | 7789ms | 100 |
| Qwen3-VL-4B-Instruct | 4B | — | — | — |
| Qwen3-VL-8B-Instruct | 8B | — | — | — |

Full results: [results.html](results.html)

![Benchmark report screenshot](docs/report_preview.png)

---

## Full Workflow

### Step 1 — Clone the repo

```bash
git clone https://github.com/axel-slid/unified_eval_framework.git
cd unified_eval_framework
```

---

### Step 2 — Download models

Each model has a download script that creates its conda environment and downloads weights.

| Model | Script | Env | Size |
|-------|--------|-----|------|
| SmolVLM2-2.2B-Instruct | `download_smolvlm.sh` | `SmolVLM-env` | ~5GB |
| InternVL3.5-4B-HF | `download_internv3.sh` | `InternV3-env` | ~9GB |
| Qwen3-VL-4B-Instruct | `download_qwen3vl_4b.sh` | `/mnt/shared/<you>/envs/Qwen3VL-env` | ~9GB |
| Qwen3-VL-8B-Instruct | `download_qwen3vl_8b.sh` | `/mnt/shared/<you>/envs/Qwen3VL-env` | ~18GB |

```bash
bash scripts/download_smolvlm.sh
bash scripts/download_internv3.sh
bash scripts/download_qwen3vl_4b.sh   # creates shared env + downloads 4B
bash scripts/download_qwen3vl_8b.sh   # reuses same env, downloads 8B weights
```

> **Note on Qwen3-VL:** Scripts create the conda env at `/mnt/shared/<yourname>/envs/Qwen3VL-env` to avoid home directory quota issues. Both 4B and 8B share this one environment. Activate with the full path: `conda activate /mnt/shared/<yourname>/envs/Qwen3VL-env`

> **Gated models:** Set `HF_TOKEN` before running:
> ```bash
> export HF_TOKEN=hf_...
> bash scripts/download_smolvlm.sh
> ```

---

### Step 3 — Verify inference works

```bash
# SmolVLM2
conda activate SmolVLM-env
python inferences/SmolVLM2-2.2B-Base.py

# InternVL3.5
conda activate InternV3-env
python inferences/InternV3_5-4B.py

# Qwen3-VL 4B
conda activate /mnt/shared/<yourname>/envs/Qwen3VL-env
python inferences/Qwen3VL-4B.py

# Qwen3-VL 8B (needs 16GB+ VRAM)
python inferences/Qwen3VL-8B.py
```

If you see a text description printed to stdout, the model is working.

---

### Step 4 — Configure the benchmark

Open `benchmark/benchmark_config.yaml` and update model paths to your local downloads:

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

Set `enabled: false` to skip a model without removing its config.

---

### Step 5 — Prepare a test set

#### Option A — Built-in sample (3 images, quick smoke test)

`benchmark/test_sets/sample.json` — use this first to verify the pipeline end to end.

#### Option B — Download 100 diverse images from the web

```bash
cd benchmark
conda activate /mnt/shared/<yourname>/envs/Qwen3VL-env
python test_sets/download_test_images.py --count 100
# → generates test_sets/captioning_100.json automatically

# options:
python test_sets/download_test_images.py --count 100 --query "product packaging"
python test_sets/download_test_images.py --count 100 --delay 2.5   # slower if hitting 429s
```

#### Option C — Build from your own images

```bash
python test_sets/generate_test_set.py \
    --images /path/to/your/images \
    --output test_sets/your_test_set.json
```

#### Option D — Write manually

```json
[
  {
    "id": "001",
    "image": "test_sets/images/001.jpg",
    "question": "What text is visible in this image?",
    "reference_answer": "EXIT",
    "rubric": "Award full marks if all visible text is correctly identified. Penalize hallucinated text."
  }
]
```

`reference_answer` can be `""` for captioning — the judge scores from the image directly.

---

### Step 6 — Run the benchmark

#### Run all 4 models with one command (recommended)

```bash
cd benchmark
conda activate /mnt/shared/<yourname>/envs/Qwen3VL-env
export OPENAI_API_KEY=sk-...
bash run_all_models.sh
```

This runs each model sequentially, unloading between runs to free GPU memory, then merges all results into a single `report_all_models_<timestamp>.html`.

#### Run individual models

```bash
cd benchmark
export OPENAI_API_KEY=sk-...

# check free GPUs first
nvidia-smi

CUDA_VISIBLE_DEVICES=0 python run_benchmark.py \
    --test-set test_sets/captioning_100.json \
    --models smolvlm

CUDA_VISIBLE_DEVICES=0 python run_benchmark.py \
    --test-set test_sets/captioning_100.json \
    --models internvl qwen3vl_4b

# 8B needs 16GB+ VRAM — use a bigger GPU
CUDA_VISIBLE_DEVICES=1 python run_benchmark.py \
    --test-set test_sets/captioning_100.json \
    --models qwen3vl_8b
```

---

### Step 7 — View results

Results land in `benchmark/results/`:

- `results_<timestamp>.json` — raw scores, responses, latencies, judge reasons
- `report_<timestamp>.html` — side-by-side comparison, open in any browser
- `report_all_models_<timestamp>.html` — combined 4-model report from `run_all_models.sh`

Terminal summary printed after each run:
```
=================================================================
SUMMARY
=================================================================
Model                            Avg Score    Avg Latency      N
-----------------------------------------------------------------
SmolVLM2 (SmolVLM2-2.2B)             3.70         3448ms    100
InternVL3 (InternVL3_5-4B)           4.00         7789ms    100
Qwen3-VL (Qwen3-VL-4B-Instruct)        —              —      —
Qwen3-VL (Qwen3-VL-8B-Instruct)        —              —      —
```

---

## Adding Your Own Model

3 steps, no changes to `run_benchmark.py`.

### 1. Create `benchmark/models/yourmodel.py`

Copy `smolvlm.py` as a template and implement `load()` and `run()`:

```python
from __future__ import annotations
import time
import torch
from PIL import Image
from transformers import AutoProcessor, YourModelClass
from models.base import BaseVLMModel, InferenceResult
from config import ModelConfig

class YourModel(BaseVLMModel):

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.name = f"YourModel ({cfg.model_path.split('/')[-1]})"
        self.model = None
        self.processor = None

    def load(self) -> None:
        self.processor = AutoProcessor.from_pretrained(self.cfg.model_path)
        self.model = YourModelClass.from_pretrained(
            self.cfg.model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        ).eval()

    def run(self, image_path: str, question: str) -> InferenceResult:
        try:
            image = Image.open(image_path).convert("RGB")
            # ... your inference code here ...
            t0 = time.perf_counter()
            # outputs = self.model.generate(...)
            latency_ms = (time.perf_counter() - t0) * 1000
            response = "..."
            return InferenceResult(response=response, latency_ms=latency_ms)
        except Exception as e:
            return InferenceResult(response="", latency_ms=0.0, error=str(e))

    def unload(self) -> None:
        del self.model
        self.model = None
        torch.cuda.empty_cache()
```

### 2. Register in `benchmark/models/__init__.py`

```python
from .yourmodel import YourModel

MODEL_REGISTRY = {
    "SmolVLMModel": SmolVLMModel,
    "InternVLModel": InternVLModel,
    "Qwen3VLModel": Qwen3VLModel,
    "YourModel": YourModel,      # ← add this
}
```

### 3. Add to `benchmark/benchmark_config.yaml`

```yaml
models:
  yourmodel:
    enabled: true
    class: YourModel
    model_path: /path/to/your/model
    dtype: bfloat16
    generation:
      max_new_tokens: 256
```

Run it:
```bash
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --models yourmodel
```

---

## Scoring

Responses are scored 1–5 by the judge model. The judge sees the image directly so it can verify accuracy without needing a reference answer.

| Score | Meaning |
|-------|---------|
| 5 | Fully correct and complete |
| 4 | Mostly correct, minor issues |
| 3 | Partially correct, missing key details |
| 2 | Mostly wrong, minor correct elements |
| 1 | Completely wrong or refused to answer |

**Estimated API cost:** ~$0.03 per 100 images × 2 models with `gpt-5.4-mini`.

---

## Model Results

| Model | Params | Avg Score (/ 5) | Avg Latency | Test Set |
|-------|--------|-----------------|-------------|----------|
| SmolVLM2-2.2B-Instruct | 2.2B | 3.70 | 3448ms | captioning 100 |
| InternVL3.5-4B-HF | 4B | 4.00 | 7789ms | captioning 100 |
| Qwen3-VL-4B-Instruct | 4B | — | — | — |
| Qwen3-VL-8B-Instruct | 8B | — | — | — |
| *(your model here)* | — | — | — | — |

---

## Configuration Reference

```yaml
output_dir: results

judge:
  model: gpt-5.4-mini          # OpenAI model used for scoring
  max_tokens: 256
  timeout_seconds: 30

generation_defaults:
  max_new_tokens: 256
  do_sample: false

models:
  smolvlm:
    enabled: true
    class: SmolVLMModel        # must match MODEL_REGISTRY key in models/__init__.py
    model_path: /path/to/model # local path or HuggingFace repo ID
    dtype: bfloat16            # float32 | float16 | bfloat16
    generation:
      max_new_tokens: 256      # overrides generation_defaults for this model
```

---

## Troubleshooting

**CUDA out of memory**
```bash
nvidia-smi              # find the PID hogging memory
kill <pid>              # free it, then re-run
```
If running multiple models in sequence still OOMs, run them one at a time using `--models <key>`.

**Disk quota exceeded**
Redirect HuggingFace cache and conda envs to a larger partition (see Prerequisites). The Qwen3-VL scripts do this automatically.

**`ImportError: cannot import name X from transformers`**
```bash
pip install --upgrade "transformers>=4.52.1"
# Qwen3-VL needs transformers from source:
pip install git+https://github.com/huggingface/transformers
```

**`ImportError: Package num2words is required`**
```bash
/mnt/shared/<yourname>/envs/Qwen3VL-env/bin/pip install num2words
```

**Judge scores all 0**
`OPENAI_API_KEY` is not set. The benchmark still runs and saves responses — set the key and re-run to get scores.

**Model produces garbage / ignores the image**
Use the `-Instruct` variant, not `-Base`. Base models are not fine-tuned for instruction following.

**Qwen3-VL conda env not found**
Activate by full path, not name:
```bash
conda activate /mnt/shared/<yourname>/envs/Qwen3VL-env
```
