# VLM CPU Demo

Webcam demo optimised for CPU-only inference on 4-core ARM devices (or any machine without a GPU).

## Structure

```
demo/
├── server.py        ← FastAPI backend, CPU inference, webcam frame handling
├── frontend/
│   └── index.html   ← Webcam UI with auto-capture and response log
└── README.md
```

## Setup

```bash
pip install fastapi uvicorn python-multipart pillow torch torchvision transformers
```

SmolVLM2 must be downloaded first:
```bash
bash scripts/download_smolvlm.sh
```

## Run

```bash
cd demo

# 4-thread CPU simulation (matches 4-core ARM @ 2.80GHz)
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python server.py

# open in browser
# http://localhost:8080
```

## Usage

1. Click **LOAD** — loads SmolVLM2 into RAM (~8GB, float32, takes 30–60s on CPU)
2. Click **START CAMERA** — opens webcam
3. Hit **📷** to capture a frame and run inference
4. Or set an interval and hit **AUTO** to run continuously every N seconds
5. Responses appear in the log on the right with timestamp, latency, and thumbnail

## Expected latency on 4× ARM @ 2.80GHz

| max_new_tokens | Approx. latency |
|---|---|
| 32 | ~15–25s |
| 64 (default) | ~25–50s |
| 128 | ~50–90s |

Adjust `max_new_tokens` in `server.py` to trade quality for speed.
Set the auto interval to match your expected latency (e.g. 30s).
