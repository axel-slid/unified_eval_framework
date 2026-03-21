# Unified VLM Eval Framework

A lightweight, extensible benchmarking framework for vision-language models (VLMs). Models run locally on GPU and are evaluated using two methods — a VQA benchmark where GPT generates questions and acts as its own baseline, and a captioning benchmark for quick comparisons.

---

## Results

### VQA Benchmark — GPT-Generated Questions + GPT Baseline

GPT generates 5 targeted questions per image with reference answers. All models answer the same questions and are scored 0–100 against GPT's reference. GPT itself is scored as the theoretical ceiling.

| Model | Params | dtype | Avg Score | vs GPT | Avg Latency | N |
|-------|--------|-------|-----------|--------|-------------|---|
| GPT Baseline (gpt-5.4-mini) | — | — | **90.9 / 100** | baseline | 1064ms | 100 |
| Qwen3-VL-4B-Instruct | 4B | bfloat16 | **88.6 / 100** | −2.3 | 2547ms | 100 |
| Qwen3-VL-8B-Instruct | 8B | bfloat16 | **88.3 / 100** | −2.6 | 3267ms | 100 |
| Qwen3-VL-4B-Instruct | 4B | int8 | **87.5 / 100** | −3.5 | 10843ms | 100 |
| Qwen3-VL-8B-Instruct | 8B | int8 | **87.4 / 100** | −3.5 | 13460ms | 100 |
| SmolVLM2-2.2B-Instruct | 2.2B | bfloat16 | **72.0 / 100** | −18.9 | 315ms | 100 |
| InternVL3-4B-HF | 4B | int8 | **62.9 / 100** | −28.0 | 5415ms | 100 |
| InternVL3-4B-HF | 4B | bfloat16 | **65.3 / 100** | −25.6 | 1400ms | 100 |

**Key findings:**
- Qwen3-VL (4B and 8B) scores within 2–3 points of the GPT ceiling — best instruction-following at this size class
- InternVL3 underperforms significantly on VQA despite being the top captioning model — captioning is a poor proxy for task performance
- int8 Qwen3-VL loses only 1 point vs bfloat16 — quantization barely hurts quality
- SmolVLM2 is 4–10x faster than all other models at 315ms with competitive quality for its size
- int8 models are slower on GPU in our setup — bitsandbytes dequantizes during inference rather than using native int8 kernels; use GGUF int4 via llama.cpp for real speedups

---

## Quick Start

Replace `<yourname>` with your username throughout.

```bash
# 1. Clone
git clone https://github.com/axel-slid/unified_eval_framework.git
cd unified_eval_framework

# 2. Shared server — redirect caches away from home dir quota
export HF_HOME=/mnt/shared/<yourname>/hf_cache
export HUGGINGFACE_HUB_CACHE=/mnt/shared/<yourname>/hf_cache
echo 'export HF_HOME=/mnt/shared/<yourname>/hf_cache' >> ~/.bashrc
echo 'export HUGGINGFACE_HUB_CACHE=/mnt/shared/<yourname>/hf_cache' >> ~/.bashrc
mkdir -p /mnt/shared/<yourname>/envs /mnt/shared/<yourname>/conda_pkgs
conda config --add envs_dirs /mnt/shared/<yourname>/envs
conda config --add pkgs_dirs /mnt/shared/<yourname>/conda_pkgs

# 3. Download all models (~41GB total)
bash scripts/download_smolvlm.sh         # SmolVLM2-2.2B   (~5GB)
bash scripts/download_internv3.sh        # InternVL3.5-4B  (~9GB)
bash scripts/download_qwen3vl_4b.sh      # Qwen3-VL-4B     (~9GB)
bash scripts/download_qwen3vl_8b.sh      # Qwen3-VL-8B     (~18GB)

# 4. Update model_path in benchmark/benchmark_config.yaml to your local paths

# 5. Download 100 test images
cd benchmark
conda activate /mnt/shared/<yourname>/envs/Qwen3VL-env
python test_sets/download_test_images.py --count 100

# 6. Run VQA benchmark (all models + GPT baseline)
mkdir -p logs
nohup bash -c '
export PYTHON=/mnt/shared/<yourname>/envs/Qwen3VL-env/bin/python
export OPENAI_API_KEY=sk-...
export HF_HOME=/mnt/shared/<yourname>/hf_cache
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /path/to/unified_eval_framework/benchmark
CUDA_VISIBLE_DEVICES=0 $PYTHON run_benchmark_vqa.py \
    --test-set test_sets/captioning_100.json --all
' >> logs/vqa_run.log 2>&1 &

tail -f logs/vqa_run.log
```

Results saved to `benchmark/results/vqa_report_<timestamp>.html` — open in any browser.

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
│   │   ├── smolvlm.py                ← SmolVLM2 runner (bfloat16 + int8)
│   │   ├── internvl.py               ← InternVL3 runner (bfloat16 + int8)
│   │   └── qwen3vl.py                ← Qwen3-VL runner (bfloat16 + int8)
│   ├── test_sets/
│   │   ├── sample.json               ← 3-image smoke test
│   │   ├── captioning_100.json       ← 100-image diverse test set
│   │   ├── generate_test_set.py      ← build a test set from a local image folder
│   │   └── download_test_images.py   ← download images from Wikimedia Commons
│   └── results/                      ← auto-created; JSON + HTML reports
├── quantize/
│   ├── quantize.py                   ← quantize models to int8 and save to disk
│   └── README.md
├── demo/
│   ├── server.py                     ← CPU-only FastAPI backend for webcam inference
│   ├── frontend/index.html           ← webcam UI with auto-capture + response log
│   └── README.md
├── docs/
│   └── report_preview.png
└── models/                           ← downloaded weights (gitignored)
```

---

## Full Workflow

### Step 1 — Download models

```bash
bash scripts/download_smolvlm.sh
bash scripts/download_internv3.sh
bash scripts/download_qwen3vl_4b.sh
bash scripts/download_qwen3vl_8b.sh
```

| Model | Conda Env | Disk |
|-------|-----------|------|
| SmolVLM2-2.2B-Instruct | `SmolVLM-env` | ~5GB |
| InternVL3.5-4B-HF | `InternV3-env` | ~9GB |
| Qwen3-VL-4B-Instruct | `/mnt/shared/<you>/envs/Qwen3VL-env` | ~9GB |
| Qwen3-VL-8B-Instruct | `/mnt/shared/<you>/envs/Qwen3VL-env` | ~18GB |

> Qwen3-VL env is created on the shared disk to avoid home quota issues. Activate with full path: `conda activate /mnt/shared/<yourname>/envs/Qwen3VL-env`

---

### Step 2 — Verify inference

```bash
conda activate SmolVLM-env
python inferences/SmolVLM2-2.2B-Base.py

conda activate InternV3-env
python inferences/InternV3_5-4B.py

conda activate /mnt/shared/<yourname>/envs/Qwen3VL-env
python inferences/Qwen3VL-4B.py
python inferences/Qwen3VL-8B.py    # needs 16GB+ VRAM
```

---

### Step 3 — Configure `benchmark_config.yaml`

Update `model_path` for each model to your local download location:

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
    model_path: OpenGVLab/InternVL3_5-4B-HF    # or local path
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

  # int8 variants (after running quantize/quantize.py)
  internvl_int8:
    enabled: true
    class: InternVLModel
    model_path: /mnt/shared/<yourname>/models/InternVL3_5-4B-HF-int8
    dtype: bfloat16    # weights already int8 on disk

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

Set `enabled: false` to skip a model without removing its config.

---

### Step 4 — Prepare a test set

**Option A — Download 100 diverse images (recommended)**

```bash
cd benchmark
python test_sets/download_test_images.py --count 100
# generates test_sets/captioning_100.json automatically
```

**Option B — Built-in 3-image smoke test**

`benchmark/test_sets/sample.json` — use first to verify the pipeline works end to end.

**Option C — Your own images**

```bash
python test_sets/generate_test_set.py \
    --images /path/to/your/images \
    --output test_sets/your_test_set.json
```

**Option D — Write manually**

```json
[
  {
    "id": "001",
    "image": "test_sets/images/001.jpg",
    "question": "What text is visible in this image?",
    "reference_answer": "EXIT",
    "rubric": "Award full marks if all visible text is identified. Penalize hallucinated text."
  }
]
```

---

### Step 5 — Run VQA benchmark (recommended)

```bash
cd benchmark
mkdir -p logs

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

Skip GPT baseline to save API cost:
```bash
$PYTHON run_benchmark_vqa.py --models smolvlm internvl --no-gpt-baseline
```

---

### Step 6 — Run captioning benchmark

```bash
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

# merge all results into one combined HTML report
$PYTHON - << EOF
import json, sys, statistics
from pathlib import Path
from datetime import datetime
sys.path.insert(0, ".")
from run_benchmark import save_html_report

results_dir = Path("results")
all_results = {}
for f in sorted(results_dir.glob("results_*.json")):
    if "merged" in f.name: continue
    data = json.loads(f.read_text())
    for key, val in data.items():
        all_results[key] = val

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
html_path = results_dir / f"report_all_models_{timestamp}.html"
save_html_report(all_results, html_path, timestamp)
print(f"Combined report → {html_path}")
EOF
' >> logs/benchmark_run.log 2>&1 &

tail -f logs/benchmark_run.log
```

---

### Step 7 — (Optional) Quantize to int8

```bash
cd quantize
/mnt/shared/<yourname>/envs/Qwen3VL-env/bin/pip install bitsandbytes

# quantize all models
/mnt/shared/<yourname>/envs/Qwen3VL-env/bin/python quantize.py --all

# or specific models
/mnt/shared/<yourname>/envs/Qwen3VL-env/bin/python quantize.py \
    --models internvl qwen3vl_4b qwen3vl_8b
```

Saves quantized models to `models/InternVL3_5-4B-HF-int8/`, etc. Then update `benchmark_config.yaml` int8 entries to `enabled: true` and re-run the benchmark.

int8 memory savings:

| Model | bfloat16 | int8 |
|-------|----------|------|
| InternVL3-4B | ~8GB | ~4GB |
| Qwen3-VL-4B | ~8GB | ~4GB |
| Qwen3-VL-8B | ~16GB | ~8GB |

---

### Step 8 — (Optional) CPU webcam demo

```bash
cd demo
pip install fastapi uvicorn python-multipart
python server.py
# open http://localhost:8080
```

Webcam demo optimised for CPU-only inference (no GPU). Uses SmolVLM2 in float32 locked to 4 threads to simulate edge hardware. Features: live webcam feed, manual capture, auto-capture every 2/4/8/15s, response log with thumbnails and latency per frame.

---

## Adding Your Own Model

3 steps, no changes to the benchmark runners.

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
            self.cfg.model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
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
    "YourModel": YourModel,
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

## Full Results

### VQA Benchmark (0–100, GPT-generated questions, GPT baseline)

| Model | Params | dtype | GPU | Avg Score | vs GPT | Latency | N |
|-------|--------|-------|-----|-----------|--------|---------|---|
| GPT Baseline (gpt-5.4-mini) | — | — | — | 90.9 / 100 | baseline | 1064ms | 100 |
| Qwen3-VL-4B-Instruct | 4B | bfloat16 | RTX PRO 6000 (97GB) | 88.6 / 100 | −2.3 | 2547ms | 100 |
| Qwen3-VL-8B-Instruct | 8B | bfloat16 | RTX PRO 6000 (97GB) | 88.3 / 100 | −2.6 | 3267ms | 100 |
| Qwen3-VL-4B-Instruct | 4B | int8 | RTX PRO 6000 (97GB) | 87.5 / 100 | −3.5 | 10843ms | 100 |
| Qwen3-VL-8B-Instruct | 8B | int8 | RTX PRO 6000 (97GB) | 87.4 / 100 | −3.5 | 13460ms | 100 |
| SmolVLM2-2.2B-Instruct | 2.2B | bfloat16 | RTX 4060 Ti (16GB) | 72.0 / 100 | −18.9 | 315ms | 100 |
| InternVL3-4B-HF | 4B | bfloat16 | RTX 4060 Ti (16GB) | 65.3 / 100 | −25.6 | 1400ms | 100 |
| InternVL3-4B-HF | 4B | int8 | RTX PRO 6000 (97GB) | 62.9 / 100 | −28.0 | 5415ms | 100 |

### Captioning Benchmark (0–100, GPT judge, free-form description)

| Model | Params | dtype | GPU | Avg Score | Latency | N |
|-------|--------|-------|-----|-----------|---------|---|
| InternVL3-4B-HF | 4B | bfloat16 | RTX 4060 Ti (16GB) | 82.2 / 100 | 4760ms | 100 |
| InternVL3-4B-HF | 4B | int8 | RTX PRO 6000 (97GB) | 79.7 / 100 | 16058ms | 100 |
| Qwen3-VL-4B-Instruct | 4B | bfloat16 | RTX 4060 Ti (16GB) | 78.6 / 100 | 6195ms | 100 |
| Qwen3-VL-8B-Instruct | 8B | int8 | RTX PRO 6000 (97GB) | 79.6 / 100 | 21167ms | 100 |
| Qwen3-VL-4B-Instruct | 4B | int8 | RTX PRO 6000 (97GB) | 78.5 / 100 | 21506ms | 100 |
| Qwen3-VL-8B-Instruct | 8B | bfloat16 | RTX PRO 6000 (97GB) | 77.0 / 100 | 5254ms | 100 |
| SmolVLM2-2.2B-Instruct | 2.2B | bfloat16 | RTX 4060 Ti (16GB) | 73.6 / 100 | 2879ms | 100 |

### Published Benchmark Scores

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
nvidia-smi && kill <pid>
```

**`CondaError: Run conda init before conda activate`** — use the explicit Python path in nohup scripts instead of `conda activate`:
```bash
export PYTHON=/mnt/shared/<yourname>/envs/Qwen3VL-env/bin/python
$PYTHON run_benchmark.py ...
```

**`ImportError: cannot import name X from transformers`**
```bash
# Qwen3-VL needs transformers from source:
pip install git+https://github.com/huggingface/transformers
# Others:
pip install --upgrade "transformers>=4.52.1"
```

**`ImportError: Package num2words is required`**
```bash
/mnt/shared/<yourname>/envs/Qwen3VL-env/bin/pip install num2words
```

**Judge scores all 0** — `OPENAI_API_KEY` not set. The benchmark still saves responses — export the key and re-run just the judging.

**Model ignores the image** — use the `-Instruct` variant not `-Base`.

**int8 models slower than bfloat16 on GPU** — expected with bitsandbytes. It dequantizes weights to float16 during inference rather than using native int8 kernels. Memory savings are real but speed savings are not on GPU. For actual inference speedup on CPU/edge use GGUF int4 via llama.cpp.

**Qwen3-VL env not found** — activate by full path:
```bash
conda activate /mnt/shared/<yourname>/envs/Qwen3VL-env
```
