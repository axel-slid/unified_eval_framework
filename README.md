# Unified VLM Eval Framework

A modular benchmarking and experimentation framework for vision-language models (VLMs), built around meeting-room and people-analysis use cases. Models run locally on GPU. Evaluation pipelines cover VQA, free-form captioning, structured environment monitoring, people detection, face detection, prompting technique comparison, LoRA fine-tuning, int8 quantization, GGUF deployment, and a desktop GUI.

48,000+ lines of Python, shell, and YAML across the full stack.

---

## Table of Contents

1. [Repo Structure](#repo-structure)
2. [Results](#results)
3. [Setup](#setup)
4. [Benchmarks](#benchmarks)
   - [VQA](#vqa-benchmark)
   - [Captioning](#captioning-benchmark)
   - [Meeting Room Checklist](#meeting-room-checklist)
   - [Environment Monitoring](#environment-monitoring)
   - [People Detection Pipeline](#people-detection-pipeline)
   - [Face Detection](#face-detection)
   - [Prompting Techniques](#prompting-techniques)
5. [Quantization](#quantization)
6. [Fine-Tuning](#fine-tuning)
7. [Deploy (llama.cpp)](#deploy-llamacpp)
8. [Demo GUI](#demo-gui)
9. [Configuration](#configuration)
10. [Adding a New Model](#adding-a-new-model)

---

## Repo Structure

```
unified_eval_framework/
│
├── benchmark/                        evaluation entry points and logic
│   ├── benchmark_config.yaml         single config file for all model paths + settings
│   ├── config.py                     loads YAML into dataclasses
│   ├── judge.py                      LLM-as-judge scorer (OpenAI API, 0–100)
│   │
│   ├── runs/                         entry point scripts
│   │   ├── run_benchmark_vqa.py
│   │   ├── run_benchmark.py
│   │   ├── run_benchmark_meeting_room.py
│   │   ├── run_benchmark_env_monitoring.py
│   │   ├── run_benchmark_env_monitoring_binary.py
│   │   ├── run_benchmark_env_monitoring_fewshot.py
│   │   ├── run_benchmark_people_detection.py
│   │   ├── run_pipeline_people_analysis.py
│   │   ├── run_approach_a_vlm_only.py
│   │   ├── run_face_detection.py
│   │   └── run_benchmark_prompting_techniques.py
│   │
│   ├── reports/                      post-run figure and HTML report generators
│   ├── scripts/                      shell runners for multi-model sequential runs
│   ├── models/                       VLM and CV model runner classes
│   ├── face_detection/               face detection pipeline and plots
│   └── test_sets/                    test data and download utilities
│
├── quantize/
│   ├── quantize.py                   int8 quantize models and save to disk
│   └── README.md
│
├── finetune/
│   ├── train_qwen3vl_lora.py         LoRA fine-tuning for Qwen3-VL-4B
│   ├── prepare_dataset.py            download COCO + filter for people/office
│   ├── eval_finetuned.py             evaluate a checkpoint on val split
│   ├── compare_single.py             base vs fine-tuned side-by-side
│   ├── visualize_predictions.py      batch bbox visualization
│   ├── visualize_5.py                5-checkpoint progression visualization
│   └── README.md
│
├── deploy/
│   └── qwen3vl_4b_llamacpp/          llama.cpp GGUF server for Qwen3-VL-4B
│       ├── serve.sh                  start the server
│       ├── infer.sh                  run inference against a running server
│       ├── deploy.sh                 unified entry point (serve | infer)
│       ├── run_qwen3vl_4b_llamacpp.py
│       ├── examples/                 example images
│       └── logs/                     inference logs (auto-created)
│
├── demo/
│   ├── app.py                        desktop GUI (customtkinter + Docker)
│   ├── prepare_ggufs.py              download GGUF models from HuggingFace
│   ├── models.json                   GGUF model registry for the GUI
│   └── requirements.txt
│
├── scripts/                          model download + env setup scripts
│   ├── download_smolvlm.sh
│   ├── download_internv3.sh
│   ├── download_qwen3vl_4b.sh
│   └── download_qwen3vl_8b.sh
│
├── inferences/                       standalone single-image smoke tests
│   ├── SmolVLM2-2.2B-Base.py
│   ├── InternV3_5-4B.py
│   ├── Qwen3VL-4B.py
│   └── Qwen3VL-8B.py
│
├── docs/
│   ├── report_preview.png
│   └── benchmark_flowchart.md
│
└── models/                           downloaded weights (gitignored, ~41 GB)
```

---

## Results

### VQA Benchmark — GPT-Generated Questions

GPT generates 5 targeted questions per image with reference answers. All models answer the same questions. GPT-as-judge scores each answer 0–100 against the reference. GPT itself is scored as the theoretical ceiling.

| Model | Params | dtype | Avg Score | vs GPT | Avg Latency | N |
|-------|--------|-------|-----------|--------|-------------|---|
| GPT Baseline (gpt-4o-mini) | — | — | 90.9 / 100 | baseline | 1064 ms | 100 |
| Qwen3-VL-4B-Instruct | 4B | bfloat16 | 88.6 / 100 | −2.3 | 2547 ms | 100 |
| Qwen3-VL-8B-Instruct | 8B | bfloat16 | 88.3 / 100 | −2.6 | 3267 ms | 100 |
| Qwen3-VL-4B-Instruct | 4B | int8 | 87.5 / 100 | −3.5 | 10843 ms | 100 |
| Qwen3-VL-8B-Instruct | 8B | int8 | 87.4 / 100 | −3.5 | 13460 ms | 100 |
| SmolVLM2-2.2B-Instruct | 2.2B | bfloat16 | 72.0 / 100 | −18.9 | 315 ms | 100 |
| InternVL3-4B | 4B | bfloat16 | 65.3 / 100 | −25.6 | 1400 ms | 100 |
| InternVL3-4B | 4B | int8 | 62.9 / 100 | −28.0 | 5415 ms | 100 |

**Key findings:**
- Qwen3-VL (4B and 8B) scores within 2–3 points of the GPT ceiling — best instruction-following at this size class
- int8 Qwen3-VL loses only ~1 point vs bfloat16 — quantization barely hurts quality
- SmolVLM2 is 4–10× faster than all other models at 315 ms, competitive for its size
- int8 models are slower on GPU — bitsandbytes dequantizes at runtime. For real speedups use GGUF int4 via llama.cpp
- InternVL3 underperforms on VQA despite competitive captioning — captioning score is a poor proxy for task performance

![Report preview](docs/report_preview.png)

---

## Setup

Replace `<yourname>` with your username throughout.

### 1. Clone the repo

```bash
git clone https://github.com/axel-slid/unified_eval_framework.git
cd unified_eval_framework
```

### 2. Configure shared disk (skip if running locally)

```bash
export HF_HOME=/mnt/shared/<yourname>/hf_cache
export HUGGINGFACE_HUB_CACHE=/mnt/shared/<yourname>/hf_cache
echo 'export HF_HOME=/mnt/shared/<yourname>/hf_cache' >> ~/.bashrc

mkdir -p /mnt/shared/<yourname>/envs /mnt/shared/<yourname>/conda_pkgs
conda config --add envs_dirs /mnt/shared/<yourname>/envs
conda config --add pkgs_dirs /mnt/shared/<yourname>/conda_pkgs
```

### 3. Download models

Each script creates a dedicated conda environment and downloads the model weights.

```bash
bash scripts/download_smolvlm.sh       # SmolVLM2-2.2B   (~5 GB)   → SmolVLM-env
bash scripts/download_internv3.sh      # InternVL3-4B    (~9 GB)   → InternV3-env
bash scripts/download_qwen3vl_4b.sh    # Qwen3-VL-4B     (~9 GB)   → Qwen3VL-env
bash scripts/download_qwen3vl_8b.sh    # Qwen3-VL-8B     (~18 GB)  → Qwen3VL-env
```

Activate the Qwen env (used for most benchmarks):

```bash
conda activate /mnt/shared/<yourname>/envs/Qwen3VL-env
```

### 4. Verify with a smoke test

```bash
conda activate SmolVLM-env
python inferences/SmolVLM2-2.2B-Base.py

conda activate InternV3-env
python inferences/InternV3_5-4B.py

conda activate /mnt/shared/<yourname>/envs/Qwen3VL-env
python inferences/Qwen3VL-4B.py
python inferences/Qwen3VL-8B.py    # needs 16 GB+ VRAM
```

### 5. Update model paths

Edit `benchmark/benchmark_config.yaml` and set the `model_path` for each model to your local download path. See [Configuration](#configuration).

### 6. Download test images

```bash
cd benchmark
python test_sets/download_test_images.py --count 100
# writes benchmark/test_sets/captioning_100.json
```

---

## Benchmarks

All benchmark commands run from the `benchmark/` directory. Set your API key before any run that uses the judge:

```bash
export OPENAI_API_KEY=sk-...
conda activate /mnt/shared/<yourname>/envs/Qwen3VL-env
cd benchmark
```

---

### VQA Benchmark

**File:** `runs/run_benchmark_vqa.py`

GPT-4o-mini generates 5 targeted, image-specific questions per image together with reference answers. Every VLM answers the same questions. GPT-as-judge scores each answer 0–100 against the reference. GPT is also run as a scored ceiling baseline.

```bash
# run all enabled models
CUDA_VISIBLE_DEVICES=0 python runs/run_benchmark_vqa.py \
    --test-set test_sets/captioning_100.json --all

# run specific models
CUDA_VISIBLE_DEVICES=0 python runs/run_benchmark_vqa.py \
    --test-set test_sets/captioning_100.json \
    --models qwen3vl_4b smolvlm

# smoke test (3 images)
CUDA_VISIBLE_DEVICES=0 python runs/run_benchmark_vqa.py \
    --test-set test_sets/sample.json --all

# run in background and tail log
mkdir -p logs
nohup bash -c '
  export OPENAI_API_KEY=sk-...
  export CUDA_VISIBLE_DEVICES=0
  cd /path/to/unified_eval_framework/benchmark
  python runs/run_benchmark_vqa.py --test-set test_sets/captioning_100.json --all
' >> logs/vqa.log 2>&1 &
tail -f logs/vqa.log
```

Outputs: `results/vqa_results_<timestamp>.json` + `results/vqa_report_<timestamp>.html`

---

### Captioning Benchmark

**File:** `runs/run_benchmark.py`

VLMs describe each image freely. GPT judges each description 0–100 using a rubric covering accuracy, completeness, and relevance.

```bash
CUDA_VISIBLE_DEVICES=0 python runs/run_benchmark.py \
    --test-set test_sets/captioning_100.json --all

CUDA_VISIBLE_DEVICES=0 python runs/run_benchmark.py \
    --test-set test_sets/captioning_100.json \
    --models qwen3vl_4b internvl smolvlm
```

Outputs: `results/captioning_results_<timestamp>.json` + HTML report

Generate figures after the run:

```bash
python reports/generate_plot.py --results results/captioning_results_<timestamp>.json
python reports/generate_dashboard.py --results results/captioning_results_<timestamp>.json
```

---

### Meeting Room Checklist

**File:** `runs/run_benchmark_meeting_room.py`

No API key required. VLMs answer a fixed binary checklist per image (e.g., "Is the whiteboard clean?", "Are chairs arranged?"). Predictions are compared to human ground-truth labels.

Metrics: item accuracy, room accuracy, per-item F1.

```bash
CUDA_VISIBLE_DEVICES=0 python runs/run_benchmark_meeting_room.py \
    --test-set test_sets/meeting_room_sample.json --all

CUDA_VISIBLE_DEVICES=0 python runs/run_benchmark_meeting_room.py \
    --test-set test_sets/meeting_room_sample.json \
    --models qwen3vl_4b qwen3vl_8b
```

---

### Environment Monitoring

Three variants covering different classification strategies.

#### Two-stage: presence → readiness

**File:** `runs/run_benchmark_env_monitoring.py`

Stage 1 — Presence: "Is there a [whiteboard / blinds / chairs / table] in this image?"
Stage 2 — Readiness (only if present): "Is the [X] ready for a meeting?"

Predicted classes: `not_present` · `present_clean` · `present_messy` · `uncertain`

```bash
CUDA_VISIBLE_DEVICES=0 python runs/run_benchmark_env_monitoring.py --all

CUDA_VISIBLE_DEVICES=0 python runs/run_benchmark_env_monitoring.py \
    --models qwen3vl_4b internvl
```

#### Binary: clean / messy

**File:** `runs/run_benchmark_env_monitoring_binary.py`

Skips Stage 1 — classifies directly as clean or messy in a single call.

```bash
CUDA_VISIBLE_DEVICES=0 python runs/run_benchmark_env_monitoring_binary.py --all

# shell runner for multi-model sweep
bash scripts/run_env_monitoring_binary.sh
```

Generate report after run:

```bash
python reports/generate_binary_report.py \
    --results results/env_monitoring_binary_results_<timestamp>.json
python reports/generate_binary_figures.py \
    --results results/env_monitoring_binary_results_<timestamp>.json
```

#### Few-shot variant

**File:** `runs/run_benchmark_env_monitoring_fewshot.py`

Injects reference example images (one READY, one NOT READY) into the prompt before the query image.

```bash
CUDA_VISIBLE_DEVICES=0 python runs/run_benchmark_env_monitoring_fewshot.py --all
```

---

### People Detection Pipeline

Three approaches for meeting participant analysis, from pure CV to pure VLM.

#### Two-stage CV → VLM pipeline

**File:** `runs/run_pipeline_people_analysis.py`

Stage 1 — CV detection (YOLOv11n, YOLOv11s, or MobileNet SSD): detect persons and extract bounding-box crops.

Stage 2 — VLM analysis: each detected person is analysed using both the full room image and the person crop. Questions asked per person:
- Is this person a meeting participant?
- Is this person currently speaking?

```bash
# default settings (YOLOv11n + Qwen3-VL-4B)
CUDA_VISIBLE_DEVICES=0 python runs/run_pipeline_people_analysis.py

# choose specific VLMs
CUDA_VISIBLE_DEVICES=0 python runs/run_pipeline_people_analysis.py \
    --vlm smolvlm qwen3vl_4b

# use a larger CV detector
CUDA_VISIBLE_DEVICES=0 python runs/run_pipeline_people_analysis.py \
    --detector-for-crops yolo11s

# run all VLMs sequentially via shell runner
bash scripts/run_pipeline_all_vlms.sh
```

Generate pipeline report:

```bash
python reports/generate_pipeline_report.py
python reports/generate_pipeline_figures.py
python reports/generate_approach_comparison.py \
    --approach-a results/approach_a_<timestamp>.json \
    --approach-b results/pipeline_people_<timestamp>.json
python reports/generate_three_approach_plot.py
```

#### CV-only people detection mAP

**File:** `runs/run_benchmark_people_detection.py`

Benchmarks YOLOv11n, YOLOv11s, and MobileNet SSD on COCO128. Reports mAP@50, mAP@75, and latency.

```bash
python runs/run_benchmark_people_detection.py
```

Generate CV comparison plot:

```bash
python reports/generate_cv_comparison_plot.py
python reports/generate_detection_figures.py
```

#### VLM-only baseline

**File:** `runs/run_approach_a_vlm_only.py`

Single VLM call per image for simultaneous person detection + role classification. No CV pre-processing. Useful as an upper-bound simplicity baseline to compare against the two-stage pipeline.

```bash
CUDA_VISIBLE_DEVICES=0 python runs/run_approach_a_vlm_only.py
```

---

### Face Detection

**File:** `runs/run_face_detection.py`

Benchmarks three face detectors on a custom people-images dataset.

| Model | Backend |
|-------|---------|
| MTCNN | facenet-pytorch |
| RetinaFace | insightface / ONNX Runtime |
| YOLOv8-Face | ultralytics, fine-tuned on WiderFace |

```bash
python runs/run_face_detection.py
```

Pipeline variants and plots live in `benchmark/face_detection/`:

```bash
# single-model benchmark
python benchmark/face_detection/run.py

# multi-stage pipeline
python benchmark/face_detection/run_pipeline.py

# generate plots
python benchmark/face_detection/plot.py
python benchmark/face_detection/plot_pipeline.py
python benchmark/face_detection/plot_logitech.py
```

---

### Prompting Techniques

**File:** `runs/run_benchmark_prompting_techniques.py`

Compares four prompting strategies on the meeting-room checklist task.

| Technique | Description |
|-----------|-------------|
| Direct | One model call per checklist item: "Is [condition] true? Answer Yes or No." |
| Chain-of-Thought | Single call; model reasons step-by-step then outputs the full checklist. |
| Few-Shot Batch | Single call with two reference images prepended (READY + NOT READY). |
| Few-Shot Per-Item | One call per item with reference images showing true/false examples. |

```bash
CUDA_VISIBLE_DEVICES=0 python runs/run_benchmark_prompting_techniques.py --all

CUDA_VISIBLE_DEVICES=0 python runs/run_benchmark_prompting_techniques.py \
    --models qwen3vl_4b smolvlm

# multi-model shell runner
bash scripts/run_all_models_prompting.sh
```

Generate report after run:

```bash
python reports/generate_prompting_report.py \
    --results results/prompting_techniques_results_<timestamp>.json
python reports/generate_prompting_examples.py \
    --results results/prompting_techniques_results_<timestamp>.json
```

---

## Quantization

**File:** `quantize/quantize.py`

Quantizes models to int8 using bitsandbytes and saves them to disk as standalone models. Once saved, they load like any normal local model — no re-quantizing on every run.

### Memory savings

| Model | bfloat16 | int8 | Saving |
|-------|----------|------|--------|
| InternVL3-4B | ~8 GB | ~4 GB | 50% |
| Qwen3-VL-4B | ~8 GB | ~4 GB | 50% |
| Qwen3-VL-8B | ~16 GB | ~8 GB | 50% |

> Note: int8 models are often slower on GPU because bitsandbytes dequantizes weights at inference time. For real speedups use GGUF int4 via llama.cpp (see [Deploy](#deploy-llamacpp)).

### Setup

```bash
conda activate /mnt/shared/<yourname>/envs/Qwen3VL-env
pip install bitsandbytes optimum
```

### Run

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

### Enable in benchmark

After quantizing, enable the int8 entries in `benchmark/benchmark_config.yaml`:

```yaml
qwen3vl_4b_int8:
  enabled: true
  class: Qwen3VLModel
  model_path: /mnt/shared/<yourname>/models/Qwen3-VL-4B-Instruct-int8
  dtype: bfloat16    # weights are already int8 on disk — load normally
```

Then run any benchmark with the int8 model key:

```bash
cd benchmark
CUDA_VISIBLE_DEVICES=0 python runs/run_benchmark_vqa.py \
    --test-set test_sets/captioning_100.json \
    --models qwen3vl_4b_int8 qwen3vl_8b_int8
```

---

## Fine-Tuning

LoRA fine-tuning of Qwen3-VL-4B for person bounding-box detection in meeting-room images. The vision encoder is frozen. LoRA adapters are applied to the language model layers only.

### Setup

```bash
cd finetune
conda activate /mnt/shared/<yourname>/envs/Qwen3VL-env
pip install peft accelerate bitsandbytes trl
# or:
bash setup.sh
```

### 1. Prepare dataset

Downloads COCO 2017 and filters for images with people in office/meeting-room scenes. Generates instruction-tuning data.

```bash
python prepare_dataset.py
# outputs: data/coco_train.jsonl, data/coco_val.jsonl
```

### 2. Train

```bash
# single GPU
python train_qwen3vl_lora.py

# multi-GPU (recommended)
accelerate launch --num_processes 2 train_qwen3vl_lora.py
```

Key hyperparameters (edit in script):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LORA_R` | 16 | LoRA rank |
| `LORA_ALPHA` | 32 | LoRA alpha |
| `LORA_DROPOUT` | 0.05 | Dropout on LoRA layers |
| `LR` | 2e-4 | Learning rate |
| `BATCH_SIZE` | 2 | Per-device batch size |
| `GRAD_ACCUM` | 4 | Gradient accumulation steps |
| `MAX_STEPS` | 1614 | Total training steps |
| `SAVE_STEPS` | 200 | Checkpoint save interval |

Checkpoints save to `checkpoints/qwen3vl_bbox_lora/`. Final adapter at `checkpoints/qwen3vl_bbox_lora/final/`.

### 3. Evaluate

```bash
python eval_finetuned.py \
    --checkpoint checkpoints/qwen3vl_bbox_lora/final \
    --data data/coco_val.jsonl
```

### 4. Visualize

```bash
# base vs fine-tuned side-by-side on one image
python compare_single.py \
    --image data/images/000042.jpg \
    --checkpoint checkpoints/qwen3vl_bbox_lora/final

# batch prediction grid (16 images)
python visualize_predictions.py \
    --checkpoint checkpoints/qwen3vl_bbox_lora/final \
    --data data/coco_val.jsonl \
    --n 16

# 5-checkpoint progression comparison
python visualize_5.py \
    --checkpoints checkpoints/qwen3vl_bbox_lora/checkpoint-200 \
                  checkpoints/qwen3vl_bbox_lora/checkpoint-600 \
                  checkpoints/qwen3vl_bbox_lora/checkpoint-1000 \
                  checkpoints/qwen3vl_bbox_lora/checkpoint-1400 \
                  checkpoints/qwen3vl_bbox_lora/final

# sample grid PNG for reporting
python generate_sample_grid.py \
    --checkpoint checkpoints/qwen3vl_bbox_lora/final \
    --output viz_output/finetune_sample_grid.png
```

### Use the fine-tuned model

```python
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/mnt/shared/<yourname>/models/Qwen3-VL-4B-Instruct",
    torch_dtype="bfloat16",
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "finetune/checkpoints/qwen3vl_bbox_lora/final")
processor = AutoProcessor.from_pretrained("finetune/checkpoints/qwen3vl_bbox_lora/final")
```

### Example output

Base vs fine-tuned comparison:

![Comparison](finetune/viz_output/comparison.png)

Sample prediction grid:

![Sample grid](finetune/viz_output/finetune_sample_grid.png)

---

## Deploy (llama.cpp)

Runs Qwen3-VL-4B as a quantized GGUF model via llama.cpp — no GPU required. Serves an OpenAI-compatible HTTP endpoint on port 8080.

Uses `Qwen3-VL-4B-Instruct-Q8_0.gguf` + `mmproj-Qwen3-VL-4B-Instruct-F16.gguf` downloaded automatically from HuggingFace.

### 1. Start the server

```bash
bash deploy/qwen3vl_4b_llamacpp/serve.sh
```

This will:
- Install `llama.cpp` via Homebrew if not already installed
- Install Python dependencies if missing (`requests`, `pillow`, `huggingface_hub`)
- Download the GGUF + mmproj files if missing (~5 GB)
- Start the server on `http://127.0.0.1:8080`

Leave this terminal running.

### 2. Run inference on an image

In a second terminal:

```bash
bash deploy/qwen3vl_4b_llamacpp/infer.sh deploy/qwen3vl_4b_llamacpp/examples/example1.png
```

```bash
bash deploy/qwen3vl_4b_llamacpp/infer.sh deploy/qwen3vl_4b_llamacpp/examples/example2.png
```

To get just the model's response with no extra output:

```bash
bash deploy/qwen3vl_4b_llamacpp/deploy.sh infer /path/to/image.jpg 2>/dev/null | tail -n 1
```

A log file is saved to `deploy/qwen3vl_4b_llamacpp/logs/` after every inference run.

### Direct Python usage

```bash
# start server
python deploy/qwen3vl_4b_llamacpp/run_qwen3vl_4b_llamacpp.py serve --backend local

# run inference against a running server
python deploy/qwen3vl_4b_llamacpp/run_qwen3vl_4b_llamacpp.py infer \
    --image /path/to/image.jpg \
    --prompt "Describe this image." \
    --port 8080
```

### Health check

```bash
curl http://127.0.0.1:8080/health
```

### Environment variables

```bash
PORT=8080          # server port
THREADS=4          # CPU threads
CTX_SIZE=2048      # context window
BACKEND=auto       # auto | local | docker
PROMPT="Describe the image clearly and concisely."
```

---

## Demo GUI

A desktop application (customtkinter + Docker) that runs GGUF models in llama.cpp Docker containers. Upload an image and compare model outputs side-by-side across all loaded models.

### Setup

```bash
cd demo
pip install -r requirements.txt
```

### Prepare GGUF models

```bash
python prepare_ggufs.py
# downloads models listed in models.json from HuggingFace
```

### Launch

```bash
python app.py
```

The GUI will open. Click **Deploy** next to a model to start its Docker container, then upload an image and click **Run** to compare outputs across all active models.

Docker must be running. Each model runs in its own container on a separate port.

---

## Configuration

`benchmark/benchmark_config.yaml` is the single source of truth for all model paths, generation settings, and judge configuration. Edit this file before running any benchmark.

```yaml
output_dir: results/

judge:
  model: gpt-4o-mini
  max_tokens: 256
  timeout_seconds: 30

generation_defaults:
  max_new_tokens: 256
  do_sample: false

models:
  smolvlm:
    enabled: true
    class: SmolVLMModel
    model_path: /mnt/shared/<yourname>/models/SmolVLM2-2.2B-Instruct
    dtype: bfloat16

  internvl:
    enabled: true
    class: InternVLModel
    model_path: /mnt/shared/<yourname>/models/InternVL3_5-4B-HF
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

  # int8 variants — enable after running quantize/quantize.py
  internvl_int8:
    enabled: false
    class: InternVLModel
    model_path: /mnt/shared/<yourname>/models/InternVL3_5-4B-HF-int8
    dtype: bfloat16

  qwen3vl_4b_int8:
    enabled: false
    class: Qwen3VLModel
    model_path: /mnt/shared/<yourname>/models/Qwen3-VL-4B-Instruct-int8
    dtype: bfloat16

  qwen3vl_8b_int8:
    enabled: false
    class: Qwen3VLModel
    model_path: /mnt/shared/<yourname>/models/Qwen3-VL-8B-Instruct-int8
    dtype: bfloat16
```

Use `enabled: false` to skip a model without removing its config. Per-model `generation` blocks override `generation_defaults`.

---

## Adding a New Model

Three steps, no changes to any runner script needed.

### 1. Create `benchmark/models/yourmodel.py`

```python
from models.base import BaseVLMModel, InferenceResult
from config import ModelConfig

class YourModel(BaseVLMModel):
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.name = "YourModel"

    def load(self) -> None:
        # load model weights into memory
        ...

    def run(self, image_path: str, question: str) -> InferenceResult:
        # run one inference, return InferenceResult(response, latency_ms, error)
        ...

    def unload(self) -> None:
        # optional — free GPU memory between model runs
        ...
```

### 2. Register in `benchmark/models/__init__.py`

```python
from .yourmodel import YourModel

MODEL_REGISTRY = {
    ...
    "YourModel": YourModel,
}
```

### 3. Add to `benchmark/benchmark_config.yaml`

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

The model will now appear in `--all` runs and can be selected by name with `--models yourmodel`.
