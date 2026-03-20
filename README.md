# Unified VLM Eval Framework

A lightweight, extensible benchmarking framework for vision-language models (VLMs). Models run locally on GPU and are scored 1–5 by an LLM-as-judge (currently GPT) on image-question pairs with hand-written rubrics.

---

## Repo Structure

```
unified_eval_framework/
├── scripts/
│   ├── download_smolvlm.sh           ← env setup + model download for SmolVLM2
│   └── download_internv3.sh          ← env setup + model download for InternVL3.5
├── inferences/
│   ├── SmolVLM2-2.2B-Base.py        ← standalone test inference for SmolVLM2
│   ├── InternV3_5-4B.py              ← standalone test inference for InternVL3.5
│   └── images/                       ← sample images for test inferences
├── benchmark/
│   ├── benchmark_config.yaml         ← all model/judge/run settings live here
│   ├── config.py                     ← loads + validates YAML into dataclasses
│   ├── run_benchmark.py              ← entry point
│   ├── judge.py                      ← LLM-as-judge scorer (OpenAI API)
│   ├── models/
│   │   ├── __init__.py               ← MODEL_REGISTRY (class name → class)
│   │   ├── base.py                   ← BaseVLMModel interface
│   │   ├── smolvlm.py
│   │   └── internvl.py
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
- A GPU with at least 8GB VRAM (CPU works but is very slow)
- Sufficient disk space on a large partition — model weights are 4–10GB each

> **Shared server tip:** If your home directory has a disk quota, redirect the HuggingFace cache before doing anything else:
> ```bash
> export HF_HOME=/path/to/large/disk/hf_cache
> export HUGGINGFACE_HUB_CACHE=/path/to/large/disk/hf_cache
> echo 'export HF_HOME=/path/to/large/disk/hf_cache' >> ~/.bashrc
> echo 'export HUGGINGFACE_HUB_CACHE=/path/to/large/disk/hf_cache' >> ~/.bashrc
> ```

---

## Full Workflow

### Step 1 — Clone the repo

```bash
git clone <repo-url>
cd unified_eval_framework
```

---

### Step 2 — Download your model

Each model gets its own conda environment. Run the appropriate script from the repo root.

#### SmolVLM2-2.2B

```bash
bash scripts/download_smolvlm.sh
```

This will:
- Create a conda env named `SmolVLM-env` (Python 3.11)
- Install `torch`, `transformers >= 4.52.1`, and all dependencies
- Download `HuggingFaceTB/SmolVLM2-2.2B-Instruct` to `models/SmolVLM2-2.2B-Instruct/`

#### InternVL3.5-4B

```bash
bash scripts/download_internv3.sh
```

This will:
- Create a conda env named `InternV3-env` (Python 3.11)
- Install `torch`, `transformers >= 4.52.1`, `timm`, `einops`, and all dependencies
- Download `OpenGVLab/InternVL3_5-4B` to `models/InternVL3_5-4B/`

> **Gated or private models:** Set `HF_TOKEN` before running and it authenticates automatically:
> ```bash
> export HF_TOKEN=hf_...
> bash scripts/download_smolvlm.sh
> ```

---

### Step 3 — Verify inference works

Before benchmarking, confirm each model loads and produces output. These scripts run a single image through the model and print the response.

```bash
# SmolVLM2
conda activate SmolVLM-env
python inferences/SmolVLM2-2.2B-Base.py

# InternVL3.5
conda activate InternV3-env
python inferences/InternV3_5-4B.py
```

If you see a text description printed to stdout, the model is working. If you want to use a locally downloaded model instead of pulling from HuggingFace, update `MODEL_PATH` at the top of the script:

```python
MODEL_PATH = "../models/SmolVLM2-2.2B-Instruct"   # local path
```

---

### Step 4 — Configure the benchmark

Open `benchmark/benchmark_config.yaml` and check:

1. **Model paths** — update `model_path` to your local path if you downloaded the weights:
   ```yaml
   models:
     smolvlm:
       model_path: /path/to/models/SmolVLM2-2.2B-Instruct  # local
       # model_path: HuggingFaceTB/SmolVLM2-2.2B-Instruct  # or pull from Hub
   ```

2. **Enable/disable models** — set `enabled: false` to skip a model without removing its config:
   ```yaml
   models:
     smolvlm:
       enabled: true
     internvl:
       enabled: false   # skip this one
   ```

3. **Judge model** — currently set to `gpt-5.4-mini`. Change if needed:
   ```yaml
   judge:
     model: gpt-5.4-mini
   ```

---

### Step 5 — Prepare a test set

#### Option A — Use the built-in sample set (3 images, quick smoke test)

The repo includes `benchmark/test_sets/sample.json` with 3 hand-written items. Good for verifying the pipeline works before a full run.

#### Option B — Download 100 diverse images from the web

```bash
cd benchmark
conda activate SmolVLM-env
python test_sets/download_test_images.py --count 100
```

This downloads images from Wikimedia Commons across 10 diverse categories (people, animals, food, cities, sports, etc.) and generates `test_sets/captioning_100.json` automatically.

Options:
```bash
# Custom output location
python test_sets/download_test_images.py --count 100 --output test_sets/images/captioning

# Single topic instead of diverse categories
python test_sets/download_test_images.py --count 100 --query "product packaging"

# Slower download rate if hitting rate limits
python test_sets/download_test_images.py --count 100 --delay 2.5
```

#### Option C — Build a test set from your own images

```bash
python test_sets/generate_test_set.py \
    --images /path/to/your/images \
    --output test_sets/your_test_set.json
```

#### Option D — Write a custom test set manually

Create a JSON file. Each item needs:

```json
[
  {
    "id": "001",
    "image": "test_sets/images/001.jpg",
    "question": "What text is visible in this image?",
    "reference_answer": "EXIT",
    "rubric": "Award full marks if all visible text is correctly identified. Partial credit for partially correct answers. Penalize hallucinated text."
  }
]
```

`reference_answer` can be left empty `""` for open-ended tasks like captioning — the judge will evaluate based on the image directly.

---

### Step 6 — Run the benchmark

```bash
cd benchmark
conda activate SmolVLM-env   # use whichever env has your model's deps
export OPENAI_API_KEY=sk-...

# Run all enabled models against the sample set
python run_benchmark.py

# Run against the 100-image captioning set
python run_benchmark.py --test-set test_sets/captioning_100.json

# Run only specific models
python run_benchmark.py --models smolvlm
python run_benchmark.py --models smolvlm internvl

# Run on a specific GPU (check free GPUs with nvidia-smi)
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --models smolvlm internvl
```

> **Multi-GPU tip:** Run models one at a time if GPU memory is tight:
> ```bash
> CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --models smolvlm
> CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --models internvl
> ```

---

### Step 7 — View results

Results are saved to `benchmark/results/`:

- `results_<timestamp>.json` — raw scores, responses, latencies, and judge reasons per question per model
- `report_<timestamp>.html` — formatted side-by-side comparison table, open in any browser

A summary is also printed to the terminal:

```
============================================================
SUMMARY
============================================================
Model                          Avg Score    Avg Latency      N
--------------------------------------------------------------
SmolVLM2 (SmolVLM2-2.2B)           3.67          812ms      3
InternVL3 (InternVL3_5-4B)         4.33         1204ms      3
```

---

## Adding Your Own Model

Adding a new model requires 3 files/edits — nothing else needs to change.

### 1. Create `benchmark/models/yourmodel.py`

Copy `smolvlm.py` as a template. The only contract is implementing `load()` and `run()`:

```python
from __future__ import annotations
import time
import torch
from PIL import Image
from transformers import AutoProcessor, YourModelClass   # swap in the right class
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
            response = "..."   # decoded output
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
    "YourModel": YourModel,      # ← add this line
}
```

### 3. Add to `benchmark/benchmark_config.yaml`

```yaml
models:
  yourmodel:
    enabled: true
    class: YourModel
    model_path: /path/to/your/model   # or HuggingFace repo ID
    dtype: bfloat16
    generation:
      max_new_tokens: 256
```

Then run:
```bash
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --models yourmodel
```

---

## Scoring

Responses are scored 1–5 by the judge model (default: `gpt-5.4-mini`) using a per-question rubric. The judge also sees the image directly, so it can verify whether the model's description is accurate.

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
| SmolVLM2-2.2B-Instruct | 2.2B | — | — | — |
| InternVL3.5-4B-HF | 4B | — | — | — |
| *(your model here)* | — | — | — | — |

> Run the benchmark and paste your results here.

---

## Configuration Reference

All settings live in `benchmark/benchmark_config.yaml`:

```yaml
output_dir: results            # where JSON + HTML reports are saved

judge:
  model: gpt-5.4-mini          # OpenAI model used for scoring
  max_tokens: 256
  timeout_seconds: 30

generation_defaults:           # applied to all models unless overridden
  max_new_tokens: 256
  do_sample: false

models:
  smolvlm:
    enabled: true              # set false to skip without deleting config
    class: SmolVLMModel        # must match a key in models/__init__.py MODEL_REGISTRY
    model_path: /path/to/model # local path or HuggingFace repo ID
    dtype: bfloat16            # float32 | float16 | bfloat16
    generation:
      max_new_tokens: 256      # overrides generation_defaults for this model
```

---

## Troubleshooting

**CUDA out of memory**
Check if another process is using the GPU:
```bash
nvidia-smi
kill <pid>   # free the memory, then re-run
```

**Disk quota exceeded**
Redirect HuggingFace cache to a larger partition (see Prerequisites above).

**`ImportError: cannot import name X from transformers`**
Your transformers version is too old:
```bash
pip install --upgrade "transformers>=4.52.1"
```
Then restart your Jupyter kernel if running in a notebook.

**Judge scores all 0**
`OPENAI_API_KEY` is not set in the current shell. The benchmark still runs and saves responses — add the key and re-run to get scores.

**Model produces garbage output / ignores the image**
Make sure you're using the `-Instruct` variant of the model, not `-Base`. Base models are not fine-tuned for instruction following.
