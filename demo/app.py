"""
VLM Demo — Unified single-server app.

Replaces docker-compose + nginx + multiple containers with ONE Python process.
All model state is managed in-process; models load/unload on demand.

Usage:
    python app.py              # serves on http://localhost:3000
    python app.py --port 8080  # custom port

Add models:
    • Edit  demo/models.json  and restart, OR
    • Use the "+ Add Model" button in the UI (no restart needed)
"""

from __future__ import annotations

import gc
import io
import json
import os
import threading
import time
import traceback
import webbrowser
from pathlib import Path
from typing import Any

import psutil
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).parent
MODELS_JSON  = HERE / "models.json"
FRONTEND_DIR = HERE / "frontend"
# Shared local model weights (checked before falling back to HuggingFace Hub)
LOCAL_MODELS_DIR = HERE.parent / "models"

DEFAULT_PORT = 3000

# ── Model registry ─────────────────────────────────────────────────────────────
# _configs: key → config dict  (persisted to models.json)
# _states:  key → runtime state (in-memory only)
# _locks:   key → threading.Lock (one per model, serialises inference)

_configs: dict[str, dict] = {}
_states:  dict[str, dict] = {}
_locks:   dict[str, threading.Lock] = {}
_registry_lock = threading.Lock()   # protects adds/removes


def _blank_state() -> dict:
    return {
        "model":           None,
        "processor":       None,
        "loading":         False,
        "load_error":      None,
        "loaded_at":       None,
        "inference_count": 0,
        "total_latency_ms": 0.0,
        "cpu_load_history": [],
    }


def _load_models_json() -> None:
    """Read models.json and register any new keys."""
    if not MODELS_JSON.exists():
        MODELS_JSON.write_text(json.dumps([], indent=2))
        return
    try:
        entries = json.loads(MODELS_JSON.read_text())
    except Exception as exc:
        print(f"[WARNING] Could not parse models.json: {exc}")
        return
    for cfg in entries:
        key = cfg.get("key")
        if not key:
            continue
        if key not in _configs:
            _configs[key] = cfg
            _states[key]  = _blank_state()
            _locks[key]   = threading.Lock()


def _save_models_json() -> None:
    MODELS_JSON.write_text(json.dumps(list(_configs.values()), indent=2))


# Load on startup
_load_models_json()


# ── CPU / RAM telemetry ────────────────────────────────────────────────────────

def _sim_cpu_freq(key: str) -> float:
    """Simulated dynamic CPU frequency based on rolling load average."""
    try:
        load = psutil.cpu_percent(interval=None)
    except Exception:
        load = 0.0
    hist = _states[key]["cpu_load_history"]
    hist.append(load)
    if len(hist) > 5:
        hist.pop(0)
    avg = sum(hist) / len(hist)
    cfg = _configs[key]
    lo  = cfg.get("cpu_min_mhz",  490)
    hi  = cfg.get("cpu_freq_mhz", 2800)
    return round(lo + (hi - lo) * avg / 100.0, 1)


def _ram_info() -> dict:
    """System RAM usage (cgroup-aware for containers, falls back to psutil)."""
    try:
        cg2l = Path("/sys/fs/cgroup/memory.max")
        cg2u = Path("/sys/fs/cgroup/memory.current")
        if cg2l.exists():
            raw = cg2l.read_text().strip()
            lim  = None if raw == "max" else int(raw) // (1024 * 1024)
            used = int(cg2u.read_text().strip()) // (1024 * 1024)
            return {"used_mb": used, "limit_mb": lim}
    except Exception:
        pass
    vm = psutil.virtual_memory()
    return {
        "used_mb":  (vm.total - vm.available) // (1024 * 1024),
        "limit_mb": vm.total // (1024 * 1024),
    }


# ── Model load / unload ────────────────────────────────────────────────────────

def _do_unload(key: str) -> None:
    st = _states[key]
    if st["model"] is not None:
        del st["model"]
        st["model"] = None
    if st["processor"] is not None:
        del st["processor"]
        st["processor"] = None
    st["loaded_at"] = None
    gc.collect()


def _do_load(key: str) -> None:
    st  = _states[key]
    cfg = _configs[key]

    st["loading"]    = True
    st["load_error"] = None

    try:
        arch       = cfg["arch"]
        full_name  = cfg["full_name"]
        dtype_str  = cfg.get("dtype", "float32")
        dtype      = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32
        tile_size  = int(cfg.get("tile_size",  378))
        max_tokens = int(cfg.get("max_tokens", 64))

        # Prefer a pre-downloaded local copy when available
        local_path = LOCAL_MODELS_DIR / Path(full_name).name
        local_only = local_path.is_dir()
        path       = str(local_path) if local_only else full_name

        print(f"[{key}] Loading {cfg['label']} from {'local' if local_only else 'HuggingFace'} "
              f"({dtype_str}) …", flush=True)
        t0 = time.time()

        if arch == "smolvlm":
            from transformers import AutoProcessor, SmolVLMForConditionalGeneration
            proc = AutoProcessor.from_pretrained(path, local_files_only=local_only)
            proc.image_processor.size          = {"longest_edge": tile_size}
            proc.image_processor.max_image_size = {"longest_edge": tile_size}
            mdl  = SmolVLMForConditionalGeneration.from_pretrained(
                path, torch_dtype=dtype, device_map="cpu", local_files_only=local_only
            )

        elif arch == "qwen3vl":
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            proc = AutoProcessor.from_pretrained(path, local_files_only=local_only)
            mdl  = Qwen3VLForConditionalGeneration.from_pretrained(
                path, torch_dtype=dtype, device_map="cpu", local_files_only=local_only
            )

        elif arch == "internvl":
            from transformers import AutoProcessor, InternVLForConditionalGeneration
            proc = AutoProcessor.from_pretrained(path, local_files_only=local_only)
            mdl  = InternVLForConditionalGeneration.from_pretrained(
                path, torch_dtype=dtype, device_map="cpu", local_files_only=local_only
            )

        elif arch == "auto":
            # Generic fallback — works for many HF vision-language models
            from transformers import AutoProcessor, AutoModelForVision2Seq
            proc = AutoProcessor.from_pretrained(path, local_files_only=local_only)
            mdl  = AutoModelForVision2Seq.from_pretrained(
                path, torch_dtype=dtype, device_map="cpu", local_files_only=local_only
            )

        else:
            raise ValueError(f"Unknown arch: {arch!r}. "
                             "Valid values: smolvlm, qwen3vl, internvl, auto")

        st["model"]     = mdl.eval()
        st["processor"] = proc
        st["loaded_at"] = time.time()
        print(f"[{key}] Ready in {time.time() - t0:.1f}s", flush=True)

    except Exception as exc:
        traceback.print_exc()
        st["load_error"] = str(exc)
    finally:
        st["loading"] = False


# ── Inference ──────────────────────────────────────────────────────────────────

def _run_inference(key: str, pil_image: Image.Image, prompt: str) -> str:
    st         = _states[key]
    cfg        = _configs[key]
    arch       = cfg["arch"]
    max_tokens = int(cfg.get("max_tokens", 64))
    mdl        = st["model"]
    proc       = st["processor"]

    if arch == "smolvlm":
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text_in = proc.apply_chat_template(messages, add_generation_prompt=True)
        inputs  = proc(text=text_in, images=pil_image, return_tensors="pt").to("cpu")
        with torch.no_grad():
            out = mdl.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return proc.decode(gen[0], skip_special_tokens=True)

    elif arch == "qwen3vl":
        tmp = Path("/tmp/_vlm_qwen_frame.jpg")
        pil_image.save(tmp, quality=85)
        try:
            from qwen_vl_utils import process_vision_info
            messages = [{"role": "user", "content": [
                {"type": "image", "image": str(tmp)},
                {"type": "text",  "text": prompt},
            ]}]
            text_in = proc.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            img_inp, vid_inp = process_vision_info(messages)
            inputs = proc(
                text=[text_in], images=img_inp, videos=vid_inp,
                padding=True, return_tensors="pt",
            ).to("cpu")
            with torch.no_grad():
                gen_ids = mdl.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
            trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
            return proc.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        finally:
            tmp.unlink(missing_ok=True)

    elif arch in ("internvl", "auto"):
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text_in = proc.apply_chat_template(messages, add_generation_prompt=True)
        inputs  = proc(text=text_in, images=pil_image, return_tensors="pt").to("cpu")
        with torch.no_grad():
            out = mdl.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        return proc.decode(out[0], skip_special_tokens=True)

    raise ValueError(f"Unknown arch: {arch!r}")


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(title="VLM Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── Frontend ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return (FRONTEND_DIR / "index.html").read_text()


# ── Models list & management ───────────────────────────────────────────────────

@app.get("/api/models")
def list_models():
    """Return all registered models with their current runtime state."""
    out = []
    for key, cfg in _configs.items():
        st = _states[key]
        out.append({
            **cfg,
            "loaded":  st["model"] is not None,
            "loading": st["loading"],
        })
    return out


@app.post("/api/models")
def add_model(body: dict):
    """
    Register a new model. Body fields:
        key        – unique slug, e.g. "my_llava"
        label      – display name
        full_name  – HuggingFace repo id, e.g. "llava-hf/llava-1.5-7b-hf"
        arch       – smolvlm | qwen3vl | internvl | auto
        dtype      – float32 | bfloat16
        tile_size  – image resize target in px (default 378)
        max_tokens – max generation tokens (default 64)
        note       – optional description shown in the UI
        color      – optional hex colour for the card title
    """
    key = body.get("key", "").strip().replace(" ", "_")
    if not key:
        raise HTTPException(400, "key is required")
    with _registry_lock:
        if key in _configs:
            raise HTTPException(409, f"Model '{key}' already exists")
        _configs[key] = body
        _states[key]  = _blank_state()
        _locks[key]   = threading.Lock()
        _save_models_json()
    print(f"[registry] Added model: {key}", flush=True)
    return {"status": "added", "key": key}


@app.delete("/api/models/{key}")
def remove_model(key: str):
    """Unload and permanently remove a model from the registry."""
    with _registry_lock:
        if key not in _configs:
            raise HTTPException(404, "Model not found")
        _do_unload(key)
        _configs.pop(key)
        _states.pop(key)
        _locks.pop(key)
        _save_models_json()
    print(f"[registry] Removed model: {key}", flush=True)
    return {"status": "removed", "key": key}


# ── Per-model routes ───────────────────────────────────────────────────────────

@app.get("/api/{key}/status")
def model_status(key: str):
    if key not in _configs:
        raise HTTPException(404, "Unknown model")
    cfg  = _configs[key]
    st   = _states[key]
    ram  = _ram_info()
    freq = _sim_cpu_freq(key)
    try:
        cpu_pct = psutil.cpu_percent(interval=None)
    except Exception:
        cpu_pct = 0.0
    uptime = int(time.time() - st["loaded_at"]) if st["loaded_at"] else None
    return {
        "model_key":    key,
        "model_label":  cfg["label"],
        "model_loaded": st["model"] is not None,
        "loading":      st["loading"],
        "load_error":   st["load_error"],
        "dtype":        cfg.get("dtype", "float32"),
        "arch":         cfg["arch"],
        "uptime_s":     uptime,
        "inference_count": st["inference_count"],
        "avg_latency_ms":  (
            round(st["total_latency_ms"] / st["inference_count"], 1)
            if st["inference_count"] else None
        ),
        "cpu": {
            "arch":     "CPU",
            "cores":    os.cpu_count() or 4,
            "freq_mhz": freq,
            "min_mhz":  cfg.get("cpu_min_mhz",  490),
            "max_mhz":  cfg.get("cpu_freq_mhz", 2800),
            "load_pct": round(cpu_pct, 1),
            "threads":  torch.get_num_threads(),
        },
        "ram": {
            "used_mb":  ram["used_mb"],
            "limit_mb": ram["limit_mb"],
            "pct": (
                round(ram["used_mb"] / ram["limit_mb"] * 100, 1)
                if ram["limit_mb"] else None
            ),
        },
    }


@app.post("/api/{key}/load")
def load_model_route(key: str):
    if key not in _configs:
        raise HTTPException(404, "Unknown model")
    st = _states[key]
    if st["loading"]:
        raise HTTPException(409, "Already loading")
    if st["model"] is not None:
        return {"status": "already_loaded", "model": key}
    _do_unload(key)
    threading.Thread(target=_do_load, args=(key,), daemon=True).start()
    return {"status": "loading", "model": key}


@app.post("/api/{key}/unload")
def unload_model_route(key: str):
    if key not in _configs:
        raise HTTPException(404, "Unknown model")
    _do_unload(key)
    return {"status": "unloaded"}


@app.post("/api/{key}/infer")
async def infer_route(
    key:    str,
    image:  UploadFile = File(...),
    prompt: str        = Form("What do you see?"),
):
    if key not in _configs:
        raise HTTPException(404, "Unknown model")
    st = _states[key]
    if st["model"] is None:
        raise HTTPException(400, "Model not loaded — click LOAD first")
    if st["loading"]:
        raise HTTPException(409, "Model still loading")

    cfg       = _configs[key]
    tile      = int(cfg.get("tile_size", 378))
    img_bytes = await image.read()
    pil       = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    pil       = pil.resize((tile, tile), Image.LANCZOS)

    try:
        with _locks[key]:
            t0       = time.perf_counter()
            response = _run_inference(key, pil, prompt)
            lat_ms   = int((time.perf_counter() - t0) * 1000)

        st["inference_count"]  += 1
        st["total_latency_ms"] += lat_ms

        return {
            "response":     response,
            "latency_ms":   lat_ms,
            "model_key":    key,
            "model_label":  cfg["label"],
            "cpu_freq_mhz": _sim_cpu_freq(key),
        }
    except Exception as exc:
        raise HTTPException(500, str(exc))


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VLM Demo — unified single-server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="HTTP port (default 3000)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't auto-open the browser")
    args = parser.parse_args()

    print(f"\n  ╔══════════════════════════════════════╗")
    print(f"  ║        VLM Demo — CPU Inference      ║")
    print(f"  ╚══════════════════════════════════════╝")
    print(f"\n  URL      →  http://localhost:{args.port}")
    print(f"  Models   →  {MODELS_JSON}")
    print(f"  Loaded   →  {len(_configs)} model(s) in registry")
    print(f"\n  Add models: edit models.json or click '+ Add Model' in the UI")
    print(f"  Press Ctrl+C to stop\n")

    if not args.no_browser:
        def _open():
            time.sleep(1.5)
            webbrowser.open(f"http://localhost:{args.port}")
        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
