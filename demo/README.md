# VLM CPU Simulator — Dockerized Multi-Model Demo

A full replacement of the original `demo/` folder. Each VLM model runs in its own Docker container, simulating a 4-core ARM CPU at 2.80 GHz with a RAM budget split across however many containers are active.

## Architecture

```
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
└── README.md
```

## HF Model Weights

Models load from HuggingFace Hub on first run (requires internet) or from a local cache at `/mnt/shared/dils/models/` (bind-mount your weights there for offline use):

```yaml
# docker-compose.yml volumes section:
volumes:
  - /your/local/models:/mnt/shared/dils:ro
  - hf_cache:/mnt/hf_cache
```

## Ports

| Service      | Internal | External |
|-------------|---------|---------|
| nginx        | 80      | 3000    |
| frontend     | 8090    | —       |
| smolvlm      | 8080    | 8081    |
| qwen3vl_4b   | 8080    | 8082    |
| internvl_4b  | 8080    | 8083    |

Direct model API access: `http://localhost:8081/api/status` etc.
