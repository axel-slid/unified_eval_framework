"""
demo/server.py — CPU-only VLM demo server with webcam support.

Supports multiple models selectable from the frontend.
Optimized for CPU-only inference with 8 GB RAM budget.

Usage:
    cd demo
    pip install fastapi uvicorn python-multipart pillow torch torchvision transformers
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python server.py

    # auto-load a model on startup:
    python server.py --model smolvlm

Models (only one loaded at a time — 8 GB RAM limit):
  smolvlm      SmolVLM2-2.2B   float32   ~8 GB   RECOMMENDED for CPU
  qwen3vl_4b   Qwen3-VL-4B     bfloat16  ~8 GB   needs swap or quantization
  internvl_4b  InternVL3.5-4B  bfloat16  ~8 GB   needs swap or quantization
"""

from __future__ import annotations

import argparse
import gc
import io
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

# ── CPU thread limits ─────────────────────────────────────────────────────────
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

DEMO_DIR = Path(__file__).parent
PROJECT_ROOT = DEMO_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "benchmark"))

# HF cache location
_hf_home = (
    "/mnt/shared/dils/hf_cache"
    if Path("/mnt/shared/dils").exists()
    else str(PROJECT_ROOT / "models" / "hf_cache")
)
os.environ.setdefault("HF_HOME", _hf_home)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", _hf_home)
os.environ.setdefault("TRANSFORMERS_CACHE", _hf_home)


# ── Model registry ────────────────────────────────────────────────────────────
CPU_MODELS: dict[str, dict] = {
    "smolvlm": {
        "label": "SmolVLM2-2.2B",
        "full_name": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "model_path": "/mnt/shared/dils/models/SmolVLM2-2.2B-Instruct",
        "arch": "smolvlm",
        "dtype": "float32",
        "tile_size": 378,
        "max_new_tokens": 64,
        "note": "Recommended — fits in 8 GB RAM",
    },
    "qwen3vl_4b": {
        "label": "Qwen3-VL-4B",
        "full_name": "Qwen/Qwen3-VL-4B-Instruct",
        "model_path": "/mnt/shared/dils/models/Qwen3-VL-4B-Instruct",
        "arch": "qwen3vl",
        "dtype": "bfloat16",
        "tile_size": 448,
        "max_new_tokens": 64,
        "note": "bfloat16 — may need extra swap",
    },
    "internvl_4b": {
        "label": "InternVL3.5-4B",
        "full_name": "OpenGVLab/InternVL3_5-4B-HF",
        "model_path": "/mnt/shared/dils/models/InternVL3_5-4B-HF",
        "arch": "internvl",
        "dtype": "bfloat16",
        "tile_size": 448,
        "max_new_tokens": 64,
        "note": "bfloat16 — may need extra swap",
    },
}

# ── Runtime state ─────────────────────────────────────────────────────────────
_model = None
_processor = None
_model_key: str | None = None
_loading = False
_load_error: str | None = None

app = FastAPI(title="VLM CPU Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Model management ──────────────────────────────────────────────────────────

def _do_unload():
    global _model, _processor, _model_key
    if _model is not None:
        del _model
        _model = None
    if _processor is not None:
        del _processor
        _processor = None
    _model_key = None
    gc.collect()


def _do_load(key: str):
    global _model, _processor, _model_key, _loading, _load_error
    _loading = True
    _load_error = None
    try:
        info = CPU_MODELS[key]
        local_path = info["model_path"]
        local_only = os.path.isdir(local_path)
        path = local_path if local_only else info["full_name"]
        dtype = torch.bfloat16 if info["dtype"] == "bfloat16" else torch.float32
        arch = info["arch"]

        print(f"[server] Loading {info['label']} ({info['dtype']}) from {'local' if local_only else 'HF hub'}…")
        t0 = time.time()

        if arch == "smolvlm":
            from transformers import AutoProcessor, SmolVLMForConditionalGeneration
            _processor = AutoProcessor.from_pretrained(path, local_files_only=local_only)
            _processor.image_processor.size = {"longest_edge": info["tile_size"]}
            _processor.image_processor.max_image_size = {"longest_edge": info["tile_size"]}
            _model = SmolVLMForConditionalGeneration.from_pretrained(
                path, torch_dtype=dtype, device_map="cpu", local_files_only=local_only
            )

        elif arch == "qwen3vl":
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            _processor = AutoProcessor.from_pretrained(path, local_files_only=local_only)
            _model = Qwen3VLForConditionalGeneration.from_pretrained(
                path, torch_dtype=dtype, device_map="cpu", local_files_only=local_only
            )

        elif arch == "internvl":
            from transformers import AutoProcessor, InternVLForConditionalGeneration
            _processor = AutoProcessor.from_pretrained(path, local_files_only=local_only)
            _model = InternVLForConditionalGeneration.from_pretrained(
                path, torch_dtype=dtype, device_map="cpu", local_files_only=local_only
            )

        else:
            raise ValueError(f"Unknown arch: {arch}")

        _model = _model.eval()
        _model_key = key
        print(f"[server] {info['label']} ready in {time.time()-t0:.1f}s")

    except Exception as e:
        print(f"[server] Load failed: {e}")
        traceback.print_exc()
        _model_key = None
        _load_error = str(e)
    finally:
        _loading = False


# ── Inference ─────────────────────────────────────────────────────────────────

def _run_inference(pil_image: Image.Image, prompt: str, info: dict) -> str:
    arch = info["arch"]

    if arch == "smolvlm":
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text_prompt = _processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = _processor(text=text_prompt, images=pil_image, return_tensors="pt").to("cpu")
        with torch.no_grad():
            outputs = _model.generate(**inputs, max_new_tokens=info["max_new_tokens"], do_sample=False)
        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
        return _processor.decode(generated_ids[0], skip_special_tokens=True)

    elif arch == "qwen3vl":
        tmp = DEMO_DIR / "_tmp_qwen.jpg"
        pil_image.save(tmp, quality=85)
        try:
            from qwen_vl_utils import process_vision_info
            messages = [{"role": "user", "content": [
                {"type": "image", "image": str(tmp)},
                {"type": "text", "text": prompt},
            ]}]
            text_prompt = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = _processor(
                text=[text_prompt], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt"
            ).to("cpu")
            with torch.no_grad():
                generated_ids = _model.generate(**inputs, max_new_tokens=info["max_new_tokens"], do_sample=False)
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
            return _processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        finally:
            tmp.unlink(missing_ok=True)

    elif arch == "internvl":
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text_prompt = _processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = _processor(text=text_prompt, images=pil_image, return_tensors="pt").to("cpu")
        with torch.no_grad():
            outputs = _model.generate(**inputs, max_new_tokens=info["max_new_tokens"], do_sample=False)
        return _processor.decode(outputs[0], skip_special_tokens=True)

    else:
        raise ValueError(f"Unknown arch: {arch}")


# ── API routes ────────────────────────────────────────────────────────────────

@app.get("/api/status")
def status():
    return {
        "threads": torch.get_num_threads(),
        "device": "cpu",
        "model_loaded": _model_key is not None,
        "loaded_model": _model_key,
        "loading": _loading,
        "load_error": _load_error,
        "available_models": {
            k: {**v, "loaded": k == _model_key}
            for k, v in CPU_MODELS.items()
        },
    }


@app.post("/api/load")
def load_model(model_key: str = "smolvlm"):
    global _loading
    if _loading:
        raise HTTPException(status_code=409, detail="Already loading a model")
    if model_key not in CPU_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model key: {model_key!r}. Valid: {list(CPU_MODELS)}")
    if _model_key == model_key:
        return {"status": "already_loaded", "model": model_key}
    if _model_key is not None:
        _do_unload()
    threading.Thread(target=_do_load, args=(model_key,), daemon=True).start()
    return {"status": "loading", "model": model_key}


@app.post("/api/unload")
def unload_model():
    _do_unload()
    return {"status": "unloaded"}


@app.post("/api/infer")
async def infer(
    image: UploadFile = File(...),
    prompt: str = Form("What do you see?"),
    model_key: str = Form("smolvlm"),
):
    if _model is None:
        raise HTTPException(status_code=400, detail="No model loaded — click LOAD first")
    if _loading:
        raise HTTPException(status_code=409, detail="Model is still loading")
    if _model_key != model_key:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model mismatch: '{model_key}' requested but '{_model_key}' is loaded. "
                "Only one model can be loaded at a time (8 GB RAM constraint). "
                "Unload current model then load the desired one."
            ),
        )

    img_bytes = await image.read()
    info = CPU_MODELS[model_key]
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    pil_image = pil_image.resize((info["tile_size"], info["tile_size"]), Image.LANCZOS)

    try:
        t0 = time.perf_counter()
        response = _run_inference(pil_image, prompt, info)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "response": response,
            "latency_ms": latency_ms,
            "model": info["label"],
            "model_key": model_key,
            "threads": torch.get_num_threads(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Static frontend ───────────────────────────────────────────────────────────
frontend_dir = DEMO_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM CPU Demo Server")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--threads", type=int, default=4, help="CPU thread count (default: 4)")
    parser.add_argument("--model", default=None, help=f"Auto-load model on startup. Options: {list(CPU_MODELS)}")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

    print(f"\n  VLM CPU Demo  |  8 GB RAM · CPU-only")
    print(f"  Threads : {args.threads}")
    print(f"  HF Home : {_hf_home}")
    print(f"  URL     : http://{args.host}:{args.port}")
    print(f"  Models  : {', '.join(CPU_MODELS.keys())}\n")

    if args.model:
        if args.model not in CPU_MODELS:
            print(f"  Warning: unknown --model '{args.model}' — skipping auto-load")
        else:
            print(f"  Auto-loading: {args.model} …")
            threading.Thread(target=_do_load, args=(args.model,), daemon=True).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
