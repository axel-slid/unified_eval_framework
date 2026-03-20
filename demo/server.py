"""
demo/server.py — CPU-only VLM demo server with webcam support.

Optimized for 4x ARM Cortex cores @ 2.80GHz (or similar low-power CPU).
Uses SmolVLM2 only (smallest model), float32, low token budget.

Usage:
    cd demo
    pip install fastapi uvicorn python-multipart pillow torch torchvision transformers
    python server.py

    # limit to 4 CPU threads to simulate ARM 4-core:
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python server.py
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

# ── Force CPU, 4 threads ──────────────────────────────────────────────────────
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

DEMO_DIR = Path(__file__).parent
PROJECT_ROOT = DEMO_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "benchmark"))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "hf_cache"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(PROJECT_ROOT / "models" / "hf_cache"))

# ── Models suited for CPU inference ──────────────────────────────────────────
# SmolVLM2 is the only realistic option on CPU — 2.2B in float32 ~= 8GB RAM
# We cap max_new_tokens low to keep latency reasonable
CPU_MODELS = {
    "smolvlm": {
        "label": "SmolVLM2-2.2B",
        "full_name": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "model_path": "/mnt/shared/dils/models/SmolVLM2-2.2B-Instruct",
        "max_new_tokens": 64,    # keep short for CPU speed
        "tile_size": 378,
        "note": "Best for CPU — fast, lightweight",
    },
}

# ── State ─────────────────────────────────────────────────────────────────────
_model = None
_processor = None
_model_key = None
_loading = False

app = FastAPI(title="VLM CPU Demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── CPU info ──────────────────────────────────────────────────────────────────
def cpu_info() -> dict:
    return {
        "threads": torch.get_num_threads(),
        "device": "cpu",
        "model_loaded": _model_key is not None,
        "loading": _loading,
    }


# ── Model loading ─────────────────────────────────────────────────────────────
def _do_load(key: str):
    global _model, _processor, _model_key, _loading
    _loading = True
    try:
        from transformers import AutoProcessor, SmolVLMForConditionalGeneration

        info = CPU_MODELS[key]
        local_only = os.path.isdir(info["model_path"])
        path = info["model_path"] if local_only else info["full_name"]

        print(f"[server] Loading {info['label']} on CPU (float32)...")
        t0 = time.time()

        _processor = AutoProcessor.from_pretrained(path, local_files_only=local_only)
        _processor.image_processor.size = {"longest_edge": info["tile_size"]}
        _processor.image_processor.max_image_size = {"longest_edge": info["tile_size"]}

        _model = SmolVLMForConditionalGeneration.from_pretrained(
            path,
            torch_dtype=torch.float32,   # float32 for CPU — bfloat16 is slower on most CPUs
            device_map="cpu",
            local_files_only=local_only,
        )
        _model = _model.eval()
        _model_key = key
        print(f"[server] Loaded in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"[server] Load failed: {e}")
        traceback.print_exc()
        _model_key = None
    finally:
        _loading = False


def _do_unload():
    global _model, _processor, _model_key
    if _model is not None:
        del _model
        _model = None
    if _processor is not None:
        del _processor
        _processor = None
    _model_key = None
    import gc; gc.collect()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/status")
def status():
    return {
        **cpu_info(),
        "loaded_model": _model_key,
        "available_models": {
            k: {**v, "loaded": k == _model_key}
            for k, v in CPU_MODELS.items()
        },
    }


@app.post("/api/load")
def load_model():
    global _loading
    if _loading:
        raise HTTPException(status_code=409, detail="Already loading")
    if _model_key is not None:
        return {"status": "already_loaded", "model": _model_key}

    import threading
    threading.Thread(target=_do_load, args=("smolvlm",), daemon=True).start()
    return {"status": "loading"}


@app.post("/api/unload")
def unload_model():
    _do_unload()
    return {"status": "unloaded"}


@app.post("/api/infer")
async def infer(
    image: UploadFile = File(...),
    prompt: str = Form("Describe this image in detail."),
):
    if _model is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    if _loading:
        raise HTTPException(status_code=409, detail="Model is still loading")

    img_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Resize to tile size for speed
    tile = CPU_MODELS["smolvlm"]["tile_size"]
    pil_image = pil_image.resize((tile, tile), Image.LANCZOS)

    # Save temp file (needed by processor)
    tmp = DEMO_DIR / "tmp_frame.jpg"
    pil_image.save(tmp, quality=85)

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_prompt = _processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = _processor(text=text_prompt, images=pil_image, return_tensors="pt").to("cpu")

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=CPU_MODELS["smolvlm"]["max_new_tokens"],
                do_sample=False,
            )
        latency_ms = int((time.perf_counter() - t0) * 1000)

        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
        response = _processor.decode(generated_ids[0], skip_special_tokens=True)

        tmp.unlink(missing_ok=True)
        return {
            "response": response,
            "latency_ms": latency_ms,
            "model": CPU_MODELS["smolvlm"]["label"],
            "threads": torch.get_num_threads(),
        }

    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Serve frontend ────────────────────────────────────────────────────────────
frontend_dir = DEMO_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads (default: 4)")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

    print(f"\n VLM CPU Demo")
    print(f" Threads : {args.threads}")
    print(f" URL     : http://{args.host}:{args.port}\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")