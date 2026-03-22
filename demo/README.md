# VLM CPU Demo

Webcam demo for CPU-only inference. Supports SmolVLM2, Qwen3-VL, and InternVL with multi-model selection in the UI.

## Structure

```
demo/
├── server.py        ← FastAPI backend, CPU inference, model switching
├── frontend/
│   └── index.html   ← Webcam UI with model selection + inference log
└── README.md
```

## Setup

```bash
pip install fastapi uvicorn python-multipart pillow torch torchvision transformers
# for Qwen3-VL only:
pip install qwen-vl-utils
```

Download models first:
```bash
bash scripts/download_smolvlm.sh       # SmolVLM2-2.2B (~8 GB, recommended for CPU)
bash scripts/download_qwen3vl_4b.sh    # Qwen3-VL-4B (optional)
bash scripts/download_internv3.sh      # InternVL3.5-4B (optional)
```

## Run

```bash
cd demo

# 4-thread CPU (recommended for 4-core machines)
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python server.py

# auto-load SmolVLM2 on startup
python server.py --model smolvlm

# open browser
# http://localhost:8080
```

## Usage

1. Click **LOAD** — loads the selected model into RAM (SmolVLM2 ~30–60s on CPU)
2. Select which models to compare in the **Models** panel (chips in top-right)
3. Click **START CAMERA** — opens your webcam
4. Hit **CAPTURE** to snap a frame and run inference
5. Set an interval and hit **AUTO** for continuous inference
6. Responses appear in the log with timestamp, latency, and thumbnail

> **Note**: Only one model can be loaded at a time due to the 8 GB RAM constraint.
> The frontend will show all available models as chips, but inference is routed
> to whichever model is currently loaded. Selecting multiple chips lets you
> queue different models if the server is extended to support model switching.

## RAM budget (CPU, float32 / bfloat16)

| Model | dtype | RAM |
|---|---|---|
| SmolVLM2-2.2B | float32 | ~8 GB ✅ fits |
| Qwen3-VL-4B | bfloat16 | ~8 GB ⚠️ tight |
| InternVL3.5-4B | bfloat16 | ~8 GB ⚠️ tight |

## Expected latency (4× ARM @ 2.80GHz)

| max_new_tokens | Latency |
|---|---|
| 32 | ~15–25s |
| 64 (default) | ~25–50s |
| 128 | ~50–90s |
