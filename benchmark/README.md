# Benchmark

All evaluation entry points, model runners, and report generators live here.

---

## Directory Structure

```
benchmark/
├── benchmark_config.yaml         all settings (model paths, judge, generation)
├── config.py                     loads + validates YAML into dataclasses
├── judge.py                      LLM-as-judge scorer (OpenAI API, 0–100 scale)
│
├── Core benchmarks
│   ├── run_benchmark_vqa.py              VQA with GPT-generated questions
│   ├── run_benchmark.py                  free-form captioning
│   └── run_benchmark_meeting_room.py     binary checklist vs ground truth (no API)
│
├── Environment monitoring
│   ├── run_benchmark_env_monitoring.py          two-stage presence → readiness
│   ├── run_benchmark_env_monitoring_binary.py   binary clean/messy classification
│   └── run_benchmark_env_monitoring_fewshot.py  few-shot reference-image variant
│
├── People & face detection
│   ├── run_benchmark_people_detection.py  mAP benchmark for CV person detectors
│   ├── run_pipeline_people_analysis.py    CV detection → VLM analysis pipeline
│   ├── run_approach_a_vlm_only.py         VLM-only baseline (no CV)
│   └── run_face_detection.py              MTCNN / RetinaFace / YOLOv8-Face
│
├── Prompting research
│   └── run_benchmark_prompting_techniques.py  direct / CoT / few-shot comparison
│
├── Shell runners
│   ├── run_all_models.sh
│   ├── run_all_models_prompting.sh
│   ├── run_pipeline_all_vlms.sh
│   └── run_env_monitoring_binary.sh
│
├── Report generators
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
├── models/
│   ├── __init__.py           MODEL_REGISTRY mapping class name → class
│   ├── base.py               BaseVLMModel interface + InferenceResult / Detection dataclasses
│   ├── smolvlm.py            SmolVLM2-2.2B runner (tile-based, fast at 315 ms)
│   ├── internvl.py           InternVL3-4B runner (bfloat16 + int8)
│   ├── qwen3vl.py            Qwen3-VL runner (4B + 8B, bfloat16 + int8)
│   ├── yolov11.py            YOLOv11n / YOLOv11s person detector
│   ├── mobilenet_ssd.py      MobileNet SSD person detector
│   ├── yolov8_face.py        YOLOv8-Face face detector
│   ├── mtcnn_face.py         MTCNN face detector
│   └── retinaface.py         RetinaFace face detector
│
├── face_detection/
│   ├── run.py                single-model face detection benchmark
│   ├── run_pipeline.py       multi-stage pipeline (detect → track → analyse)
│   ├── plot.py               result visualizations
│   ├── plot_pipeline.py      pipeline-stage visualizations
│   └── plot_logitech.py      Logitech-device-specific plot styling
│
├── test_sets/
│   ├── sample.json               3-image smoke test
│   ├── captioning_100.json       100-image diverse test set
│   ├── meeting_room_sample.json  meeting room readiness samples
│   ├── download_test_images.py   download images from Wikimedia Commons
│   └── generate_test_set.py      build a test set from a local folder
│
└── results/                  auto-created; JSON + HTML reports (gitignored)
```

---

## Running Benchmarks

All commands assume you are inside the `benchmark/` directory.

### VQA Benchmark

GPT generates 5 targeted questions per image, VLMs answer them, GPT judges each answer 0–100.

```bash
export OPENAI_API_KEY=sk-...

# all enabled models + GPT ceiling baseline
CUDA_VISIBLE_DEVICES=0 python run_benchmark_vqa.py \
    --test-set test_sets/captioning_100.json --all

# specific models only
CUDA_VISIBLE_DEVICES=0 python run_benchmark_vqa.py \
    --test-set test_sets/captioning_100.json --models smolvlm qwen3vl_4b
```

### Captioning Benchmark

VLMs describe each image freely; GPT judges descriptions 0–100.

```bash
python run_benchmark.py --test-set test_sets/captioning_100.json --all
python run_benchmark.py --models smolvlm internvl
```

### Meeting Room Checklist

Binary checklist evaluation against human ground truth. No API key required.

```bash
python run_benchmark_meeting_room.py --test-set test_sets/meeting_room_sample.json --all
```

### Environment Monitoring

Two-stage presence + readiness benchmark.

```bash
python run_benchmark_env_monitoring.py --all
python run_benchmark_env_monitoring.py --models qwen3vl_4b internvl

# binary clean/messy variant (skips Stage 1 presence check)
python run_benchmark_env_monitoring_binary.py --all

# few-shot variant (reference images injected into prompt)
python run_benchmark_env_monitoring_fewshot.py --all
```

### People Detection (CV models)

mAP@50 and mAP@75 benchmark for person detectors on COCO128.

```bash
python run_benchmark_people_detection.py
python run_benchmark_people_detection.py --models yolo11n yolo11s
```

### People Analysis Pipeline (CV + VLM)

Stage 1: CV detectors find persons. Stage 2: VLMs analyse each crop + full room context.

```bash
python run_pipeline_people_analysis.py
python run_pipeline_people_analysis.py --vlm smolvlm qwen3vl_4b
python run_pipeline_people_analysis.py --detector-for-crops yolo11s
```

### VLM-Only Baseline

Single VLM call per image: detect persons + classify roles simultaneously.

```bash
python run_approach_a_vlm_only.py
```

### Face Detection

Compare MTCNN, RetinaFace, and YOLOv8-Face.

```bash
python run_face_detection.py
```

### Prompting Techniques

Compare four strategies on the meeting-room checklist task.

```bash
python run_benchmark_prompting_techniques.py --all
python run_benchmark_prompting_techniques.py --models qwen3vl_4b --techniques direct cot
```

---

## Model Interface

All VLM model runners inherit from `BaseVLMModel` (`models/base.py`):

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

    def load(self) -> None:
        # load weights, processor, tokenizer
        ...

    def run(self, image_path: str, question: str) -> InferenceResult:
        # run inference, return InferenceResult
        ...

    def unload(self) -> None:
        # free GPU memory (optional but recommended between models)
        ...
```

**2. Register in `models/__init__.py`**

```python
from .yourmodel import YourModel

MODEL_REGISTRY = {
    ...
    "YourModel": YourModel,
}
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

No changes to any benchmark runner are needed.

---

## Judge

`judge.py` calls the OpenAI API (model configured in `benchmark_config.yaml`) with a structured scoring prompt. The judge receives the image, the question, the reference answer (if any), and the model's response. It returns a score 0–100 and a brief reason.

Score rubric:
- 0–20: completely wrong
- 21–40: mostly wrong
- 41–60: partially correct
- 61–80: mostly correct
- 81–100: fully correct

---

## Output Format

Each benchmark run produces:
- `results/<benchmark>_results_<timestamp>.json` — raw per-sample results
- `results/<benchmark>_report_<timestamp>.html` — interactive HTML report

HTML reports include a per-model summary table (avg score, avg latency, N samples) and a per-image breakdown with scores and responses from every model.
