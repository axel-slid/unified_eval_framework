"""
demo/server_model.py — Per-model VLM worker container.

Each container represents one model running on a simulated 4-core ARM CPU
at 2.80 GHz. Config comes entirely from environment variables so the same
image handles all three model variants.

Environment Variables (set by docker-compose):
    MODEL_KEY           smolvlm | qwen3vl_4b | internvl_4b
    MODEL_LABEL         display name
    MODEL_FULL_NAME     HuggingFace repo id
    MODEL_ARCH          smolvlm | qwen3vl | internvl
    MODEL_DTYPE         float32 | bfloat16
    MODEL_TILE_SIZE     image resize target (px)
    MODEL_MAX_TOKENS    max generation tokens
    SERVER_PORT         HTTP port (default 8080)
    CPU_CORES           simulated core count (4)
    CPU_FREQ_MHZ        simulated max freq in MHz (2800)
    CPU_MIN_MHZ         simulated min freq in MHz (490)
"""

from __future__ import annotations

import gc
import io
import math
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import psutil
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# ── CPU simulation config ─────────────────────────────────────────────────────
CPU_CORES     = int(os.environ.get("CPU_CORES", "4"))
CPU_FREQ_MHZ  = float(os.environ.get("CPU_FREQ_MHZ", "2800"))
CPU_MIN_MHZ   = float(os.environ.get("CPU_MIN_MHZ", "490"))
CPU_ARCH      = "ARM"

# ── Model config from env ─────────────────────────────────────────────────────
MODEL_KEY       = os.environ.get("MODEL_KEY", "smolvlm")
MODEL_LABEL     = os.environ.get("MODEL_LABEL", "SmolVLM2-2.2B")
MODEL_FULL_NAME = os.environ.get("MODEL_FULL_NAME", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
MODEL_ARCH      = os.environ.get("MODEL_ARCH", "smolvlm")
MODEL_DTYPE_STR = os.environ.get("MODEL_DTYPE", "float32")
MODEL_TILE_SIZE = int(os.environ.get("MODEL_TILE_SIZE", "378"))
MODEL_MAX_TOKENS= int(os.environ.get("MODEL_MAX_TOKENS", "64"))
SERVER_PORT     = int(os.environ.get("SERVER_PORT", "8080"))

# HF model path (local if pre-downloaded, else hub)
_hf_home = os.environ.get("HF_HOME", "/mnt/hf_cache")
LOCAL_MODEL_PATH = f"/mnt/shared/dils/models/{Path(MODEL_FULL_NAME).name}"

# ── Thread limits — match simulated 4-core CPU ────────────────────────────────
torch.set_num_threads(CPU_CORES)
torch.set_num_interop_threads(max(1, CPU_CORES // 2))
os.environ["OMP_NUM_THREADS"]      = str(CPU_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_CORES)
os.environ["MKL_NUM_THREADS"]      = str(CPU_CORES)

# ── Runtime state ─────────────────────────────────────────────────────────────
_model      = None
_processor  = None
_loading    = False
_load_error: str | None = None
_loaded_at: float | None = None
_inference_count = 0
_total_latency_ms = 0.0
_infer_lock = threading.Lock()  # serialize inference (single-model CPU constraint)

# ── CPU frequency simulation ──────────────────────────────────────────────────
_cpu_load_history: list[float] = []

def _simulated_cpu_freq() -> float:
    """
    Simulate dynamic ARM CPU frequency scaling.
    Returns MHz: scales between CPU_MIN_MHZ and CPU_FREQ_MHZ based on load.
    """
    try:
        load = psutil.cpu_percent(interval=None)
    except Exception:
        load = 0.0
    _cpu_load_history.append(load)
    if len(_cpu_load_history) > 5:
        _cpu_load_history.pop(0)
    avg_load = sum(_cpu_load_history) / len(_cpu_load_history)
    # Linear scaling: idle → min_freq, 100% load → max_freq
    freq = CPU_MIN_MHZ + (CPU_FREQ_MHZ - CPU_MIN_MHZ) * (avg_load / 100.0)
    return round(freq, 1)

def _container_ram_mb() -> dict:
    """Read container memory limits from cgroups if available, else psutil."""
    used_mb = None
    limit_mb = None

    # Try cgroup v2 first
    cg2_limit = Path("/sys/fs/cgroup/memory.max")
    cg2_usage = Path("/sys/fs/cgroup/memory.current")
    cg1_limit = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    cg1_usage = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")

    try:
        if cg2_limit.exists():
            raw = cg2_limit.read_text().strip()
            limit_mb = None if raw == "max" else int(raw) // (1024 * 1024)
            used_mb  = int(cg2_usage.read_text().strip()) // (1024 * 1024)
        elif cg1_limit.exists():
            lim = int(cg1_limit.read_text().strip())
            limit_mb = None if lim > (512 * 1024 * 1024 * 1024) else lim // (1024 * 1024)
            used_mb  = int(cg1_usage.read_text().strip()) // (1024 * 1024)
    except Exception:
        pass

    if used_mb is None:
        vm = psutil.virtual_memory()
        used_mb  = (vm.total - vm.available) // (1024 * 1024)
        limit_mb = vm.total // (1024 * 1024)

    return {"used_mb": used_mb, "limit_mb": limit_mb}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title=f"VLM Worker — {MODEL_LABEL}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model management ──────────────────────────────────────────────────────────

def _do_unload():
    global _model, _processor, _loaded_at
    if _model is not None:
        del _model; _model = None
    if _processor is not None:
        del _processor; _processor = None
    _loaded_at = None
    gc.collect()


def _do_load():
    global _model, _processor, _loading, _load_error, _loaded_at

    _loading = True
    _load_error = None
    try:
        local_only = os.path.isdir(LOCAL_MODEL_PATH)
        path       = LOCAL_MODEL_PATH if local_only else MODEL_FULL_NAME
        dtype      = torch.bfloat16 if MODEL_DTYPE_STR == "bfloat16" else torch.float32

        print(f"[{MODEL_KEY}] Loading {MODEL_LABEL} ({MODEL_DTYPE_STR}) …", flush=True)
        t0 = time.time()

        if MODEL_ARCH == "smolvlm":
            from transformers import AutoProcessor, SmolVLMForConditionalGeneration
            proc = AutoProcessor.from_pretrained(path, local_files_only=local_only)
            proc.image_processor.size = {"longest_edge": MODEL_TILE_SIZE}
            proc.image_processor.max_image_size = {"longest_edge": MODEL_TILE_SIZE}
            mdl  = SmolVLMForConditionalGeneration.from_pretrained(
                path, torch_dtype=dtype, device_map="cpu", local_files_only=local_only
            )

        elif MODEL_ARCH == "qwen3vl":
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            proc = AutoProcessor.from_pretrained(path, local_files_only=local_only)
            mdl  = Qwen3VLForConditionalGeneration.from_pretrained(
                path, torch_dtype=dtype, device_map="cpu", local_files_only=local_only
            )

        elif MODEL_ARCH == "internvl":
            from transformers import AutoProcessor, InternVLForConditionalGeneration
            proc = AutoProcessor.from_pretrained(path, local_files_only=local_only)
            mdl  = InternVLForConditionalGeneration.from_pretrained(
                path, torch_dtype=dtype, device_map="cpu", local_files_only=local_only
            )

        else:
            raise ValueError(f"Unknown arch: {MODEL_ARCH!r}")

        _model     = mdl.eval()
        _processor = proc
        _loaded_at = time.time()
        print(f"[{MODEL_KEY}] Ready in {time.time()-t0:.1f}s", flush=True)

    except Exception as e:
        print(f"[{MODEL_KEY}] Load failed: {e}", flush=True)
        traceback.print_exc()
        _load_error = str(e)
    finally:
        _loading = False


# ── Inference ─────────────────────────────────────────────────────────────────

def _run_inference(pil_image: Image.Image, prompt: str) -> str:
    if MODEL_ARCH == "smolvlm":
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text_in  = _processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs   = _processor(text=text_in, images=pil_image, return_tensors="pt").to("cpu")
        with torch.no_grad():
            out = _model.generate(**inputs, max_new_tokens=MODEL_MAX_TOKENS, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return _processor.decode(gen[0], skip_special_tokens=True)

    elif MODEL_ARCH == "qwen3vl":
        tmp = Path("/tmp/_qwen_frame.jpg")
        pil_image.save(tmp, quality=85)
        try:
            from qwen_vl_utils import process_vision_info
            messages = [{"role": "user", "content": [
                {"type": "image", "image": str(tmp)},
                {"type": "text",  "text": prompt},
            ]}]
            text_in = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            img_inp, vid_inp = process_vision_info(messages)
            inputs = _processor(
                text=[text_in], images=img_inp, videos=vid_inp,
                padding=True, return_tensors="pt"
            ).to("cpu")
            with torch.no_grad():
                gen_ids = _model.generate(**inputs, max_new_tokens=MODEL_MAX_TOKENS, do_sample=False)
            trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
            return _processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        finally:
            tmp.unlink(missing_ok=True)

    elif MODEL_ARCH == "internvl":
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text_in  = _processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs   = _processor(text=text_in, images=pil_image, return_tensors="pt").to("cpu")
        with torch.no_grad():
            out = _model.generate(**inputs, max_new_tokens=MODEL_MAX_TOKENS, do_sample=False)
        return _processor.decode(out[0], skip_special_tokens=True)

    raise ValueError(f"Unknown arch: {MODEL_ARCH!r}")


# ── API routes ────────────────────────────────────────────────────────────────

@app.get("/api/status")
def status():
    ram   = _container_ram_mb()
    freq  = _simulated_cpu_freq()
    try:
        cpu_pct = psutil.cpu_percent(interval=None)
    except Exception:
        cpu_pct = 0.0

    uptime_s = int(time.time() - _loaded_at) if _loaded_at else None

    return {
        # Model state
        "model_key":    MODEL_KEY,
        "model_label":  MODEL_LABEL,
        "model_loaded": _model is not None,
        "loading":      _loading,
        "load_error":   _load_error,
        "dtype":        MODEL_DTYPE_STR,
        "arch":         MODEL_ARCH,
        "uptime_s":     uptime_s,

        # Inference stats
        "inference_count":    _inference_count,
        "avg_latency_ms":     round(_total_latency_ms / _inference_count, 1) if _inference_count else None,

        # CPU simulation (ARM 4-core @ 2.80 GHz)
        "cpu": {
            "arch":     CPU_ARCH,
            "cores":    CPU_CORES,
            "freq_mhz": freq,          # dynamic — scales with load
            "min_mhz":  CPU_MIN_MHZ,
            "max_mhz":  CPU_FREQ_MHZ,
            "load_pct": round(cpu_pct, 1),
            "threads":  torch.get_num_threads(),
        },

        # RAM
        "ram": {
            "used_mb":  ram["used_mb"],
            "limit_mb": ram["limit_mb"],
            "pct":      round(ram["used_mb"] / ram["limit_mb"] * 100, 1) if ram["limit_mb"] else None,
        },
    }


@app.post("/api/load")
def load_model():
    global _loading
    if _loading:
        raise HTTPException(status_code=409, detail="Already loading")
    if _model is not None:
        return {"status": "already_loaded", "model": MODEL_KEY}
    _do_unload()
    threading.Thread(target=_do_load, daemon=True).start()
    return {"status": "loading", "model": MODEL_KEY}


@app.post("/api/unload")
def unload_model():
    _do_unload()
    return {"status": "unloaded"}


@app.post("/api/infer")
async def infer(
    image:  UploadFile = File(...),
    prompt: str        = Form("What do you see?"),
):
    global _inference_count, _total_latency_ms

    if _model is None:
        raise HTTPException(status_code=400, detail="Model not loaded — POST /api/load first")
    if _loading:
        raise HTTPException(status_code=409, detail="Model still loading")

    img_bytes = await image.read()
    pil       = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    pil       = pil.resize((MODEL_TILE_SIZE, MODEL_TILE_SIZE), Image.LANCZOS)

    try:
        with _infer_lock:
            t0       = time.perf_counter()
            response = _run_inference(pil, prompt)
            lat_ms   = int((time.perf_counter() - t0) * 1000)

        _inference_count  += 1
        _total_latency_ms += lat_ms

        return {
            "response":    response,
            "latency_ms":  lat_ms,
            "model_key":   MODEL_KEY,
            "model_label": MODEL_LABEL,
            "cpu_freq_mhz": _simulated_cpu_freq(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n  VLM Worker: {MODEL_LABEL}")
    print(f"  CPU:  {CPU_CORES}× {CPU_ARCH} @ {CPU_FREQ_MHZ} MHz (min {CPU_MIN_MHZ} MHz)")
    print(f"  Arch: {MODEL_ARCH} | dtype: {MODEL_DTYPE_STR}")
    print(f"  Port: {SERVER_PORT}\n")
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, log_level="warning")
