# Benchmark

All evaluation entry points, model runners, and report generators live here.

---

## Directory Structure

```
benchmark/
├── benchmark_config.yaml     all settings (model paths, judge, generation)
├── config.py                 loads + validates YAML into dataclasses
├── judge.py                  LLM-as-judge scorer (OpenAI API, 0–100 scale)
│
├── runs/                     entry point scripts — run from benchmark/ directory
│   ├── run_benchmark.py                          free-form captioning
│   ├── run_benchmark_vqa.py                      VQA with GPT-generated questions
│   ├── run_benchmark_meeting_room.py             binary checklist vs ground truth (no API)
│   ├── run_benchmark_env_monitoring.py           two-stage presence → readiness
│   ├── run_benchmark_env_monitoring_binary.py    binary clean/messy classification
│   ├── run_benchmark_env_monitoring_fewshot.py   few-shot reference-image variant
│   ├── run_benchmark_people_detection.py         mAP benchmark for CV person detectors
│   ├── run_pipeline_people_analysis.py           CV detection → VLM analysis pipeline
│   ├── run_approach_a_vlm_only.py                VLM-only baseline (no CV)
│   ├── run_face_detection.py                     MTCNN / RetinaFace / YOLOv8-Face
│   └── run_benchmark_prompting_techniques.py     direct / CoT / few-shot comparison
│
├── reports/                  post-run visualisation and report generators
│   ├── generate_plot.py
│   ├── generate_dashboard.py
│   ├── generate_binary_report.py / generate_binary_figures.py
│   ├── generate_pipeline_report.py / generate_pipeline_figures.py
│   ├── generate_prompting_report.py / generate_prompting_examples.py
│   ├── generate_detection_plot.py / generate_detection_figures.py
│   ├── generate_approach_comparison.py / generate_approach_comparison_plot.py
│   ├── generate_three_approach_plot.py
│   ├── generate_cv_comparison_plot.py
│   └── generate_examples_report.py
│
├── scripts/                  shell runners (sequential multi-model execution)
│   ├── run_all_models.sh             run all VLMs on captioning benchmark
│   ├── run_all_models_nohup.sh       nohup wrapper for long runs
│   ├── run_all_models_prompting.sh   prompting technique sweep across all models
│   ├── run_pipeline_all_vlms.sh      CV stage once + all VLMs sequentially
│   └── run_env_monitoring_binary.sh  env monitoring binary sweep
│
├── models/                   VLM and CV model runner classes
│   ├── __init__.py           MODEL_REGISTRY (class name → class)
│   ├── base.py               BaseVLMModel interface + dataclasses
│   ├── smolvlm.py            SmolVLM2-2.2B
│   ├── internvl.py           InternVL3-4B
│   ├── qwen3vl.py            Qwen3-VL (4B + 8B, bfloat16 + int8)
│   ├── yolov11.py            YOLOv11n/s person detector
│   ├── mobilenet_ssd.py      MobileNet SSD person detector
│   ├── yolov8_face.py        YOLOv8-Face face detector
│   ├── mtcnn_face.py         MTCNN face detector
│   └── retinaface.py         RetinaFace face detector
│
├── face_detection/           face detection pipeline and plots
│   ├── run.py                single-model benchmark
│   ├── run_pipeline.py       multi-stage pipeline
│   ├── plot.py / plot_pipeline.py / plot_logitech.py
│   └── results/              output images and JSON (some plots committed)
│
├── test_sets/                test data and download utilities
│   ├── sample.json               3-image smoke test
│   ├── captioning_100.json       100-image diverse test set
│   ├── meeting_room_sample.json  meeting room readiness samples
│   ├── download_test_images.py   download from Wikimedia Commons
│   └── download_test_images.py   download images from Wikimedia Commons
│
├── visualization/            shared plotting utilities
└── results/                  auto-created; JSON + HTML reports (gitignored)
```

---

## Running Benchmarks

All commands assume you are inside the `benchmark/` directory.

### Core benchmarks

```bash
export OPENAI_API_KEY=sk-...

# VQA — GPT-generated questions, all models + GPT ceiling
CUDA_VISIBLE_DEVICES=0 python runs/run_benchmark_vqa.py \
    --test-set test_sets/captioning_100.json --all

# Captioning
CUDA_VISIBLE_DEVICES=0 python runs/run_benchmark.py \
    --test-set test_sets/captioning_100.json --all

# Meeting room checklist (no API key needed)
CUDA_VISIBLE_DEVICES=0 python runs/run_benchmark_meeting_room.py \
    --test-set test_sets/meeting_room_sample.json --all
```

### Environment monitoring

```bash
python runs/run_benchmark_env_monitoring.py --all
python runs/run_benchmark_env_monitoring_binary.py --all
python runs/run_benchmark_env_monitoring_fewshot.py --all
```

### People & face detection

```bash
# CV model mAP benchmark (COCO128)
python runs/run_benchmark_people_detection.py

# Two-stage CV → VLM pipeline
python runs/run_pipeline_people_analysis.py
python runs/run_pipeline_people_analysis.py --vlm smolvlm qwen3vl_4b
python runs/run_pipeline_people_analysis.py --detector-for-crops yolo11s

# VLM-only baseline
python runs/run_approach_a_vlm_only.py

# Face detection comparison
python runs/run_face_detection.py
```

### Prompting techniques

```bash
python runs/run_benchmark_prompting_techniques.py --all
```

### Shell runners (multi-model sequential)

```bash
# Set your env path before running:
export PYTHON=/mnt/shared/<yourname>/envs/Qwen3VL-env/bin/python

bash scripts/run_all_models.sh
bash scripts/run_all_models_prompting.sh
bash scripts/run_pipeline_all_vlms.sh
bash scripts/run_env_monitoring_binary.sh
```

### Report generators

Run after a benchmark to produce figures and HTML reports:

```bash
python reports/generate_dashboard.py --results results/env_monitoring_results_XYZ.json
python reports/generate_prompting_report.py --results results/prompting_techniques_results_XYZ.json
python reports/generate_pipeline_report.py
python reports/generate_approach_comparison.py \
    --approach-a results/approach_a_XYZ.json \
    --approach-b results/pipeline_people_XYZ.json
```

---

## Model Interface

All model runners inherit from `BaseVLMModel` (`models/base.py`):

```python
class BaseVLMModel(ABC):
    name: str

    def load(self) -> None: ...
    def run(self, image_path: str, question: str) -> InferenceResult: ...
    def unload(self) -> None: ...
```

`InferenceResult` fields: `response: str`, `latency_ms: float`, `error: str | None`

CV detection models return `DetectionResult` with a list of `Detection` objects (bbox, confidence, class_id, class_name).

---

## Adding a New Model — 3 Steps

**1. Create `models/yourmodel.py`**

```python
from models.base import BaseVLMModel, InferenceResult
from config import ModelConfig

class YourModel(BaseVLMModel):
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.name = "YourModel"

    def load(self) -> None: ...
    def run(self, image_path: str, question: str) -> InferenceResult: ...
    def unload(self) -> None: ...
```

**2. Register in `models/__init__.py`**

```python
from .yourmodel import YourModel
MODEL_REGISTRY = { ..., "YourModel": YourModel }
```

**3. Add to `benchmark_config.yaml`**

```yaml
models:
  yourmodel:
    enabled: true
    class: YourModel
    model_path: org/repo-or-local-path
    dtype: bfloat16
    generation:
      max_new_tokens: 256
```

No changes to any runner script needed.

---

## Judge

`judge.py` calls the OpenAI API with a structured scoring prompt. Returns a score 0–100 and a brief reason.

Score rubric: 0–20 wrong · 21–40 mostly wrong · 41–60 partial · 61–80 mostly correct · 81–100 fully correct

---

## Output Format

Each run produces:
- `results/<benchmark>_results_<timestamp>.json` — raw per-sample results
- `results/<benchmark>_report_<timestamp>.html` — interactive HTML report with per-model summary and per-image breakdown
