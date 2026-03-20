# Unified VLM Eval Framework

A lightweight, extensible benchmarking framework for vision-language models (VLMs). Models are scored by Claude-as-judge on a 1–5 scale across image-question pairs with hand-written rubrics.

---

## Repo Structure

```
unified_eval_framework/
├── scripts/
│   ├── download_smolvlm.sh       ← env setup + model download for SmolVLM2
│   └── download_internv3.sh      ← env setup + model download for InternVL3.5
├── inferences/
│   ├── SmolVLM2-2.2B-Base.py    ← standalone test inference for SmolVLM2
│   ├── InternV3_5-4B.py          ← standalone test inference for InternVL3.5
│   └── images/                   ← sample images for test inferences
├── benchmark/
│   ├── benchmark_config.yaml     ← all model/judge/run settings live here
│   ├── config.py                 ← loads + validates YAML into dataclasses
│   ├── run_benchmark.py          ← entry point
│   ├── judge.py                  ← Claude-as-judge scorer
│   ├── models/
│   │   ├── __init__.py           ← MODEL_REGISTRY (class name → class)
│   │   ├── base.py               ← BaseVLMModel interface
│   │   ├── smolvlm.py
│   │   └── internvl.py
│   ├── test_sets/
│   │   ├── sample.json           ← (image, question, rubric) tuples
│   │   └── images/               ← images referenced by the test set
│   └── results/                  ← auto-created; JSON + HTML report per run
```

---

## Prerequisites

- [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html)
- An Anthropic API key (used by Claude-as-judge)
- A GPU with at least 8GB VRAM recommended (CPU inference works but is slow)

---

## Step 1 — Download Models

Each model gets its own conda environment and is downloaded via the HuggingFace CLI. Run the appropriate script from the repo root.

### SmolVLM2-2.2B

```bash
bash scripts/download_smolvlm.sh
```

This will:
- Create a conda env named `SmolVLM-env` (Python 3.11)
- Install `torch`, `transformers >= 4.52.1`, and all other dependencies
- Download `HuggingFaceTB/SmolVLM2-2.2B-Base` to `models/SmolVLM2-2.2B-Base/`

### InternVL3.5-4B

```bash
bash scripts/download_internv3.sh
```

This will:
- Create a conda env named `InternV3-env` (Python 3.11)
- Install `torch`, `transformers >= 4.52.1`, `timm`, `einops`, and all other dependencies
- Download `OpenGVLab/InternVL3_5-4B` to `models/InternVL3_5-4B/`

> **Private or gated models:** Set `HF_TOKEN` before running the script and it will authenticate automatically.
> ```bash
> export HF_TOKEN=hf_...
> bash scripts/download_smolvlm.sh
> ```

---

## Step 2 — Run Test Inferences

Before benchmarking, verify each model loads and runs correctly with the standalone inference scripts. These scripts run a single image through the model and print the response to stdout.

### SmolVLM2

```bash
conda activate SmolVLM-env
python inferences/SmolVLM2-2.2B-Base.py
```

### InternVL3.5

```bash
conda activate InternV3-env
python inferences/InternV3_5-4B.py
```

Each script auto-detects your hardware (MPS → CUDA → CPU) and runs a sample image through the model. If you see a text description printed, the model is working correctly.

To use a local downloaded model instead of fetching from HuggingFace, update the `MODEL_PATH` variable at the top of the script:

```python
MODEL_PATH = "../models/SmolVLM2-2.2B-Base"   # local path
```

---

## Step 3 — Run the Benchmark

The benchmark runner loads all enabled models from `benchmark_config.yaml`, runs each one against the test set, scores responses with Claude-as-judge, and outputs a JSON result file and an HTML report.

### Setup

```bash
conda activate SmolVLM-env   # or whichever env has your deps installed
export ANTHROPIC_API_KEY=sk-ant-...
```

### Run all enabled models

```bash
cd benchmark
python run_benchmark.py
```

### Options

```bash
# Custom config or test set
python run_benchmark.py --config benchmark_config.yaml --test-set test_sets/sample.json

# Run only specific models (keys must match benchmark_config.yaml)
python run_benchmark.py --models smolvlm
python run_benchmark.py --models smolvlm internvl
```

### Output

Results are saved to `benchmark/results/`:
- `results_<timestamp>.json` — raw scores, responses, latencies, and judge reasons per question per model
- `report_<timestamp>.html` — formatted side-by-side comparison table (open in any browser)

A summary is also printed to the terminal:

```
============================================================
SUMMARY
============================================================
Model                          Avg Score    Avg Latency      N
--------------------------------------------------------------
SmolVLM2-2.2B-Base                  3.67          812ms      3
InternVL3_5-4B                      4.33         1204ms      3
```

---

## Scoring

Responses are scored 1–5 by `claude-sonnet-4-20250514` using a per-question rubric defined in the test set JSON:

| Score | Meaning |
|-------|---------|
| 5 | Fully correct and complete |
| 4 | Mostly correct, minor issues |
| 3 | Partially correct, missing key details |
| 2 | Mostly wrong, minor correct elements |
| 1 | Completely wrong or refused to answer |

---

## Model Results

| Model | Params | Avg Score (/ 5) | Avg Latency | Notes |
|-------|--------|-----------------|-------------|-------|
| SmolVLM2-2.2B-Base | 2.2B | — | — | Lightweight; chat-template input format |
| InternVL3.5-4B | 4B | — | — | Strong spatial reasoning |
| ... | ... | ... | ... | More models coming |

> Run `python run_benchmark.py` and paste your results here.

---

## Adding a New Model

**1. Create `benchmark/models/yourmodel.py`**

```python
from models.base import BaseVLMModel, InferenceResult
from config import ModelConfig

class YourModel(BaseVLMModel):
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.name = "YourModel"

    def load(self): ...
    def run(self, image_path, question) -> InferenceResult: ...
    def unload(self): ...  # optional
```

**2. Register in `benchmark/models/__init__.py`**

```python
from .yourmodel import YourModel

MODEL_REGISTRY = {
    ...,
    "YourModel": YourModel,
}
```

**3. Add to `benchmark/benchmark_config.yaml`**

```yaml
models:
  yourmodel:
    enabled: true
    class: YourModel
    model_path: org/repo-or-local-path
    dtype: float16
    generation:
      max_new_tokens: 256
```

That's it — no changes to `run_benchmark.py`.

---

## Adding Test Cases

Edit `benchmark/test_sets/sample.json` (or create a new file and pass it with `--test-set`). Each entry needs:

```json
{
  "id": "004",
  "image": "test_sets/images/004.jpg",
  "question": "What text is visible in this image?",
  "rubric": "Award full marks if all visible text is correctly identified. Partial credit for partially correct answers. Penalize hallucinated text."
}
```

---

## Configuration Reference

All settings live in `benchmark/benchmark_config.yaml`:

```yaml
output_dir: results

judge:
  model: claude-sonnet-4-20250514
  max_tokens: 256
  timeout_seconds: 30

generation_defaults:
  max_new_tokens: 256
  do_sample: false

models:
  smolvlm:
    enabled: true
    class: SmolVLMModel
    model_path: HuggingFaceTB/SmolVLM2-2.2B-Instruct
    dtype: float16
    generation:
      max_new_tokens: 256
```

Per-model `generation` settings override `generation_defaults`. Set `enabled: false` to skip a model without removing its config.
