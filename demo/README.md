# VLM CPU Simulator — Dockerized Multi-Model Demo

<<<<<<< HEAD
A full replacement of the original `demo/` folder. Each VLM model runs in its own Docker container, simulating a 4-core ARM CPU at 2.80 GHz with a RAM budget split across however many containers are active.
=======
Webcam demo for CPU-only inference. Supports SmolVLM2, Qwen3-VL, and InternVL with multi-model selection in the UI.
>>>>>>> 4b9c60c9a620c9fed4d0217f3dd1b443268c3b5e

## Architecture

```
<<<<<<< HEAD
┌──────────────────────────────────────────────────────────┐
│  Browser  →  nginx :3000                                  │
│               ├── /               → frontend:8090         │
│               ├── /api/smolvlm/   → smolvlm:8080          │
│               ├── /api/qwen3vl_4b/→ qwen3vl_4b:8080       │
│               └── /api/internvl_4b/→ internvl_4b:8080     │
└──────────────────────────────────────────────────────────┘

Each model container:
  - 4 CPU cores  (ARM CPU3, 2.80 GHz max / 0.49 GHz min)
  - mem_limit    split across active containers (default ~2.8 GB each)
  - mem_swap     2× mem_limit (swap buffer)
```

## Quick Start

```bash
# Clone or replace your demo/ folder with this one
cd demo_docker/

# Build and run all 3 models + nginx + frontend
docker compose up --build

# Open in browser
open http://localhost:3000
```

## RAM Budget Configuration

By default `docker-compose.yml` gives each model **2800m** (≈2.8 GB). Adjust based on your actual RAM:

| Your total RAM | Models running | Recommended mem_limit |
|---------------|---------------|-----------------------|
| 8 GB          | 1             | 7g                    |
| 8 GB          | 2             | 3800m                 |
| 8 GB          | 3             | 2400m (+ swap)        |
| 16 GB         | 3             | 5g                    |
| 32 GB         | 3             | 10g                   |

Edit `docker-compose.yml`:
```yaml
mem_limit: 2800m      # per-container RAM cap
memswap_limit: 5600m  # total RAM+swap (2× is good)
```

## Running Only Some Models

```bash
# SmolVLM only (most RAM-efficient)
docker compose up --build nginx frontend smolvlm

# Two models
docker compose up --build nginx frontend smolvlm qwen3vl_4b
```

## CPU Simulation

Each container is constrained to **4 CPUs** (`cpus: "4.0"`):

```
ARM CPU3  ·  4 cores
  current: 2.80 GHz  (max load)
  idle:    0.49 GHz  (min load)
  scaling: linear with container CPU % utilization
```

The UI shows **live dynamic frequency** for each model — it scales down when idle and climbs toward 2.80 GHz under inference load.

## UI Features

- **Side-by-side model panels** — each with live CPU freq + RAM gauges
- **Per-model AUTO button** — start continuous capture/infer loop independently
- **▶ ALL button** — start continuous on all loaded models simultaneously
- **Individual LOAD/UNLOAD** — load only the models you have RAM for
- **Camera feed** — shared across all panels (capture fires to all loaded models)
- **Inference log per model** — timestamped with latency

## Directory Structure

```
demo_docker/
├── docker-compose.yml          # Orchestration
├── nginx/
│   └── nginx.conf              # Reverse proxy config
├── demo/
│   ├── Dockerfile.model        # Model worker image
│   ├── Dockerfile.frontend     # Static frontend image
│   ├── server_model.py         # Per-model FastAPI server (env-configured)
│   ├── server_frontend.py      # Serves frontend/
│   ├── requirements.model.txt  # Python deps for model workers
│   └── frontend/
│       └── index.html          # Full multi-model UI
=======
demo/
├── server.py        ← FastAPI backend, CPU inference, model switching
├── frontend/
│   └── index.html   ← Webcam UI with model selection + inference log
>>>>>>> 4b9c60c9a620c9fed4d0217f3dd1b443268c3b5e
└── README.md
```

## HF Model Weights

<<<<<<< HEAD
Models load from HuggingFace Hub on first run (requires internet) or from a local cache at `/mnt/shared/dils/models/` (bind-mount your weights there for offline use):

```yaml
# docker-compose.yml volumes section:
volumes:
  - /your/local/models:/mnt/shared/dils:ro
  - hf_cache:/mnt/hf_cache
```

## Ports
=======
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
>>>>>>> 4b9c60c9a620c9fed4d0217f3dd1b443268c3b5e

| Service      | Internal | External |
|-------------|---------|---------|
| nginx        | 80      | 3000    |
| frontend     | 8090    | —       |
| smolvlm      | 8080    | 8081    |
| qwen3vl_4b   | 8080    | 8082    |
| internvl_4b  | 8080    | 8083    |

<<<<<<< HEAD
Direct model API access: `http://localhost:8081/api/status` etc.
=======
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
>>>>>>> 4b9c60c9a620c9fed4d0217f3dd1b443268c3b5e
