"""
VLM Demo — llama.cpp Docker Edition.

Each quantized GGUF model runs in its own llama.cpp Docker container.
Prepare GGUF assets first, then click "Deploy" to start a model.
Upload an image and run inference across all loaded models side-by-side.

Usage:
    cd demo
    pip install -r requirements.txt
    python app.py
"""

from __future__ import annotations

import os

# Set thread counts before any torch/MKL/OpenMP import so the env vars take effect.
_all_cpus = str(os.cpu_count() or 4)
os.environ.setdefault("OMP_NUM_THREADS",   _all_cpus)
os.environ.setdefault("MKL_NUM_THREADS",   _all_cpus)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _all_cpus)

import base64
import gc
import io
import json
import queue as _queue
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional

import customtkinter as ctk
import docker
import psutil
import requests
from PIL import Image, ImageOps


# ── Paths / constants ──────────────────────────────────────────────────────────
HERE         = Path(__file__).parent
PROJECT_ROOT = HERE.parent
MODELS_JSON  = HERE / "models.json"

LLAMA_SERVER_IMAGES = [
    "ghcr.io/ggml-org/llama.cpp:server",
    "ghcr.io/ggerganov/llama.cpp:server",
]
LLAMA_FULL_IMAGES = [
    "ghcr.io/ggml-org/llama.cpp:full",
    "ghcr.io/ggerganov/llama.cpp:full",
]
BASE_PORT           = 8100
CONTAINER_PREFIX    = "vlm_demo_"

# Quantisation preference order when multiple GGUFs exist in a repo
TARGET_QUANT = "Q8_0"
QUANT_PREF = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_K_S", "Q4_0"]

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── Palette ────────────────────────────────────────────────────────────────────
BG_COLOR    = "#1e1e1e"   # window / app background
TOPBAR_FG   = "#181818"   # topbar strip
PANEL_FG    = "#1e1e1e"   # scrollable / panel bg
CARD_FG     = "#252525"   # sidebar / panel background
CARD_BORDER = "#333333"   # separator / border
ACCENT_FG   = "#2a2a2a"   # input field bg
GLASS_SEP   = "#303030"   # hover state
SASH_COLOR  = "#2e2e2e"   # PanedWindow sash

TXT_PRI  = "#d4d4d4"   # primary text
TXT_SEC  = "#888888"   # secondary text
TXT_DIM  = "#555555"   # muted / label text

STATUS_COLOR = {
    "stopped":   "#444444",
    "preparing": "#9d6fe8",
    "deploying": "#d4a017",
    "loading":   "#d4a017",
    "ready":     "#3db870",
    "error":     "#e05050",
}

POLL_MS = 2000

# ── Global log queue (print → UI) ─────────────────────────────────────────────
_log_queue: _queue.SimpleQueue = _queue.SimpleQueue()

def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"{ts}  {msg}"
    print(line, flush=True)
    _log_queue.put(line)


# ══════════════════════════════════════════════════════════════════════════════
# GGUF helpers  (download from HF  or  convert via Docker)
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_model_dir(cfg: dict) -> Optional[Path]:
    raw = cfg.get("model_dir", "")
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p if p.is_dir() else p   # return even if not yet created


def _existing_ggufs(model_dir: Path) -> tuple[list[Path], list[Path]]:
    """Return (main_ggufs, mmproj_ggufs) found in model_dir."""
    if not model_dir.is_dir():
        return [], []
    all_g = sorted(model_dir.glob("*.gguf"))
    main   = [f for f in all_g if "mmproj" not in f.name.lower()]
    mmproj = [f for f in all_g if "mmproj"     in f.name.lower()]
    return main, mmproj


def _best_gguf(files: list[Path]) -> Optional[Path]:
    """Pick the best-ranked GGUF from a list."""
    for q in QUANT_PREF:
        for f in files:
            if q in f.name.upper():
                return f
    return files[0] if files else None


def _resize_image(image_bytes: bytes, max_side: int = 720) -> bytes:
    """Resize image so its longest side is at most max_side px. Returns JPEG bytes."""
    from PIL import Image as _PIL
    img = _PIL.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    if max(w, h) <= max_side:
        return image_bytes
    scale = max_side / max(w, h)
    nw, nh = int(w * scale), int(h * scale)
    buf = io.BytesIO()
    img.resize((nw, nh), _PIL.LANCZOS).save(buf, "JPEG", quality=90)
    return buf.getvalue()


_IMAGE_CACHE: dict[tuple[str, ...], str] = {}


def _resolve_docker_image(docker_client, candidates: list[str], log=None) -> str:
    """Pick the first pullable llama.cpp image, or a cached local copy."""
    key = tuple(candidates)
    cached = _IMAGE_CACHE.get(key)
    if cached:
        return cached

    errors = []
    for image in candidates:
        if log:
            log(f"Pulling {image} …")
        try:
            docker_client.images.pull(image)
            _IMAGE_CACHE[key] = image
            return image
        except Exception as e:
            try:
                docker_client.images.get(image)
                if log:
                    log(f"WARNING: pull failed ({e}), using cached image: {image}")
                _IMAGE_CACHE[key] = image
                return image
            except Exception:
                errors.append(f"{image}: {e}")

    detail = "\n".join(errors)
    raise RuntimeError(
        "Could not find a compatible llama.cpp Docker image.\n"
        f"Tried:\n{detail}"
    )


def download_gguf(cfg: dict, log) -> tuple[Optional[Path], Optional[Path]]:
    """
    Download GGUF files from HuggingFace.
    `log` is a callable that accepts a string.
    Returns (model_path, mmproj_path).
    """
    from huggingface_hub import list_repo_files, hf_hub_download

    repo       = cfg["gguf_hf_repo"]
    model_dir  = _resolve_model_dir(cfg)
    model_dir.mkdir(parents=True, exist_ok=True)

    log(f"Listing files in {repo} …")
    try:
        all_files = list(list_repo_files(repo))
    except Exception as e:
        log(f"ERROR listing repo: {e}")
        return None, None

    gguf_files  = [f for f in all_files if f.endswith(".gguf")]
    main_files  = [f for f in gguf_files if "mmproj" not in f.lower()]
    mmproj_files = [f for f in gguf_files if "mmproj"     in f.lower()]

    if not main_files:
        log("ERROR: No GGUF model files found in repo.")
        return None, None

    # Pick best quantisation
    def _rank(name: str) -> int:
        for i, q in enumerate(QUANT_PREF):
            if q in name.upper():
                return i
        return 99

    chosen_model  = sorted(main_files,  key=_rank)[0]
    chosen_mmproj = sorted(mmproj_files, key=_rank)[0] if mmproj_files else None

    log(f"Downloading model:  {chosen_model}")
    try:
        local_model = hf_hub_download(repo_id=repo, filename=chosen_model,
                                      local_dir=str(model_dir))
        log(f"  → {local_model}")
    except Exception as e:
        log(f"ERROR: {e}")
        return None, None

    local_mmproj = None
    if chosen_mmproj:
        log(f"Downloading mmproj: {chosen_mmproj}")
        try:
            local_mmproj = hf_hub_download(repo_id=repo, filename=chosen_mmproj,
                                           local_dir=str(model_dir))
            log(f"  → {local_mmproj}")
        except Exception as e:
            log(f"WARNING: mmproj download failed: {e}")

    return Path(local_model), Path(local_mmproj) if local_mmproj else None


def convert_gguf(cfg: dict, docker_client, log) -> tuple[Optional[Path], Optional[Path]]:
    """
    Convert local safetensors → GGUF using ghcr.io/ggerganov/llama.cpp:full.
    `log` is a callable that accepts a string.
    Returns (model_path, mmproj_path).
    """
    model_dir = _resolve_model_dir(cfg)
    if not model_dir or not model_dir.is_dir():
        log(f"ERROR: Model directory not found: {model_dir}")
        log("Run the download script in scripts/ first.")
        return None, None

    out_name = f"{model_dir.name}-{TARGET_QUANT}.gguf"
    out_path = model_dir / out_name
    f16_name = f"{model_dir.name}-F16.gguf"
    f16_path = model_dir / f16_name

    if out_path.exists():
        log(f"GGUF already exists: {out_name}")
        _, mmprojs = _existing_ggufs(model_dir)
        return out_path, _best_gguf(mmprojs)

    log(f"Converting {model_dir.name} → GGUF ({TARGET_QUANT}) …")
    log(f"Source: {model_dir}")
    image = _resolve_docker_image(docker_client, LLAMA_FULL_IMAGES, log)

    convert_cmd = [
        "/app/convert_hf_to_gguf.py",
        "/model",
        "--outtype", "f16",
        "--outfile", f"/model/{f16_name}",
    ]
    quantize_cmd = [
        "/app/llama-quantize",
        f"/model/{f16_name}",
        f"/model/{out_name}",
        TARGET_QUANT,
    ]
    log(f"Using image: {image}")
    log(f"Running: python3 {' '.join(convert_cmd)}")
    log(f"Then: {' '.join(quantize_cmd)}\n")

    try:
        if not f16_path.exists():
            container = docker_client.containers.run(
                image,
                entrypoint="python3",
                command=convert_cmd,
                volumes={str(model_dir): {"bind": "/model", "mode": "rw"}},
                remove=True,
                detach=True,
            )
            for chunk in container.logs(stream=True, follow=True):
                line = chunk.decode("utf-8", errors="replace").rstrip()
                if line:
                    log(line)

            exit_code = container.wait()["StatusCode"]
            if exit_code != 0:
                log(f"\nERROR: Conversion exited with code {exit_code}.")
                log("This model architecture may not yet be supported by llama.cpp.")
                return None, None
        else:
            log(f"Reusing existing intermediate GGUF: {f16_name}")

        container = docker_client.containers.run(
            image,
            command=quantize_cmd,
            volumes={str(model_dir): {"bind": "/model", "mode": "rw"}},
            remove=True,
            detach=True,
        )
        for chunk in container.logs(stream=True, follow=True):
            line = chunk.decode("utf-8", errors="replace").rstrip()
            if line:
                log(line)

        exit_code = container.wait()["StatusCode"]
        if exit_code != 0:
            log(f"\nERROR: Quantization exited with code {exit_code}.")
            return None, None

        if f16_path.exists():
            try:
                f16_path.unlink()
            except OSError:
                pass

        log(f"\nDone → {out_path}")
        _, mmprojs = _existing_ggufs(model_dir)
        return out_path, _best_gguf(mmprojs)

    except Exception as e:
        log(f"ERROR: {e}")
        return None, None


# ══════════════════════════════════════════════════════════════════════════════
# DockerModelManager
# ══════════════════════════════════════════════════════════════════════════════

class DockerModelManager:
    def __init__(self):
        self._client: Optional[docker.DockerClient] = None
        self._containers: dict[str, dict] = {}
        self._local_models: dict[str, dict] = {}
        self._port_counter = BASE_PORT
        self._lock = threading.Lock()
        self._docker_error: Optional[str] = None
        try:
            self._connect()
        except Exception as e:
            self._docker_error = str(e)

    def _connect(self):
        try:
            self._client = docker.from_env()
            self._client.ping()
        except Exception as e:
            self._client = None
            raise RuntimeError(
                f"Cannot connect to Docker: {e}\n"
                "Make sure Docker Desktop is running."
            )

    @property
    def client(self):
        return self._client

    @property
    def docker_error(self) -> Optional[str]:
        return self._docker_error

    def _next_port(self) -> int:
        with self._lock:
            p = self._port_counter
            self._port_counter += 1
            return p

    def deploy(self, key: str, cfg: dict, memory_mb: int, cpus: float) -> None:
        if cfg.get("runtime") == "transformers":
            self._deploy_transformers(key, cfg, cpus)
            return

        if self._client is None:
            raise RuntimeError("Docker not connected")

        model_path  = Path(cfg["model_path"]).expanduser().resolve()
        mmproj_path = cfg.get("mmproj_path")
        ctx_size    = cfg.get("ctx_size", 2048)

        if not model_path.exists():
            raise FileNotFoundError(f"GGUF not found: {model_path}")

        self.stop(key)

        port      = self._next_port()
        model_dir = model_path.parent
        volumes   = {str(model_dir): {"bind": "/mnt/model", "mode": "ro"}}

        n_threads = max(1, int(cpus))
        cmd = [
            "--model",         f"/mnt/model/{model_path.name}",
            "--host",          "0.0.0.0",
            "--port",          "8080",
            "--ctx-size",      str(ctx_size),
            "--threads",       str(n_threads),
            "--threads-batch", str(n_threads),
            "--mlock",                          # lock model weights in RAM, no swap
        ]

        if mmproj_path:
            mp = Path(mmproj_path).expanduser().resolve()
            if mp.parent != model_dir:
                volumes[str(mp.parent)] = {"bind": "/mnt/mmproj", "mode": "ro"}
                cmd += ["--mmproj", f"/mnt/mmproj/{mp.name}"]
            else:
                cmd += ["--mmproj", f"/mnt/model/{mp.name}"]

        image = _resolve_docker_image(self._client, LLAMA_SERVER_IMAGES)

        log(f"[deploy] Starting {key} on port {port} (ctx={ctx_size}, cpus={cpus})")
        container = self._client.containers.run(
            image,
            command=cmd,
            name=f"{CONTAINER_PREFIX}{key}",
            ports={"8080/tcp": port},
            volumes=volumes,
            nano_cpus=int(cpus * 1_000_000_000),
            detach=True,
            remove=True,
        )
        log(f"[deploy] Container started for {key} (id={container.short_id})")

        with self._lock:
            self._containers[key] = {"container": container, "port": port}

    def _deploy_transformers(self, key: str, cfg: dict, cpus: float = 4.0) -> None:
        model_dir = _resolve_model_dir(cfg)
        if not model_dir or not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        self.stop(key)

        n_threads = max(1, int(cpus))
        log(f"[deploy] Loading {key} via transformers from {model_dir} (threads={n_threads})")

        import torch

        # Apply thread budget before loading (affects BLAS/MKL used during weight init)
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(max(1, n_threads // 2))
        torch.set_float32_matmul_precision("high")  # use TF32/AMX where available

        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            Qwen3VLForConditionalGeneration,
        )

        family = cfg.get("backend_family", "")
        if not family:
            label = cfg.get("label", "").lower()
            if "qwen3-vl" in label:
                family = "qwen3_vl"
            elif "internvl" in label:
                family = "internvl"
            elif "smolvlm" in label:
                family = "smolvlm"
            else:
                family = "generic_vlm"

        device = "cpu"
        torch_dtype = torch.float32

        log(f"[deploy] Loading processor for {key} (family={family})")
        if family == "qwen3_vl":
            # Limit image resolution to avoid OOM in the vision encoder's O(n²) attention.
            # max_pixels = 1280 * 28 * 28 ≈ 1 MP → ≤ ~5120 patches → safe for MPS.
            processor = AutoProcessor.from_pretrained(
                str(model_dir),
                min_pixels=256 * 28 * 28,
                max_pixels=1280 * 28 * 28,
            )
        else:
            processor = AutoProcessor.from_pretrained(str(model_dir))

        log(f"[deploy] Loading model weights for {key} (dtype={torch_dtype}, device={device})")
        if family == "qwen3_vl":
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                str(model_dir),
                torch_dtype=torch_dtype,
                device_map={"": device},
            )
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                str(model_dir),
                torch_dtype=torch_dtype,
                device_map={"": device},
            )
        device = getattr(model, "device", None)
        model.eval()
        log(f"[deploy] {key} loaded on {device} — threads={n_threads}")

        with self._lock:
            self._local_models[key] = {
                "runtime":   "transformers",
                "family":    family,
                "model":     model,
                "processor": processor,
                "device":    device,
                "n_threads": n_threads,
                "process":   psutil.Process(),
            }

    def stop(self, key: str) -> None:
        with self._lock:
            info = self._containers.pop(key, None)
            local_info = self._local_models.pop(key, None)
        if info:
            try:
                info["container"].stop(timeout=5)
            except Exception:
                pass
        if local_info:
            try:
                del local_info["model"]
                del local_info["processor"]
            except Exception:
                pass
            gc.collect()
        if self._client:
            try:
                self._client.containers.get(f"{CONTAINER_PREFIX}{key}").stop(timeout=5)
            except Exception:
                pass

    def stop_all(self) -> None:
        for key in list(self._containers.keys()):
            self.stop(key)

    def get_status(self, key: str) -> dict:
        with self._lock:
            info = self._containers.get(key)
            local_info = self._local_models.get(key)
        if local_info:
            proc = local_info["process"]
            try:
                cpu_pct = proc.cpu_percent(interval=None)
                mem_mb = proc.memory_info().rss / (1024 * 1024)
            except Exception:
                cpu_pct = 0.0
                mem_mb = 0.0
            return {
                "status": "ready",
                "runtime": "transformers",
                "cpu_pct": cpu_pct,
                "mem_mb": mem_mb,
                "mem_limit_mb": 0.0,
            }
        if not info:
            return {"status": "stopped"}
        try:
            info["container"].reload()
            if info["container"].status != "running":
                with self._lock:
                    self._containers.pop(key, None)
                return {"status": "stopped"}
        except Exception as e:
            return {"status": "error", "detail": str(e)}

        port = info["port"]
        stats = self._container_stats(info["container"])
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=2)
            if r.status_code == 200 and r.json().get("status") == "ok":
                return {"status": "ready", "port": port, **stats}
            return {"status": "loading", "port": port, **stats}
        except requests.exceptions.ConnectionError:
            return {"status": "loading", "port": port, **stats}
        except Exception as e:
            return {"status": "error", "detail": str(e), "port": port, **stats}

    def _container_stats(self, container) -> dict:
        try:
            stats = container.stats(stream=False)
            cpu_total = stats["cpu_stats"]["cpu_usage"].get("total_usage", 0)
            precpu_total = stats["precpu_stats"]["cpu_usage"].get("total_usage", 0)
            system_total = stats["cpu_stats"].get("system_cpu_usage", 0)
            presystem_total = stats["precpu_stats"].get("system_cpu_usage", 0)
            online_cpus = (
                stats["cpu_stats"].get("online_cpus")
                or len(stats["cpu_stats"]["cpu_usage"].get("percpu_usage", []) or [])
                or 1
            )
            cpu_delta = cpu_total - precpu_total
            system_delta = system_total - presystem_total
            cpu_pct = 0.0
            if cpu_delta > 0 and system_delta > 0:
                cpu_pct = (cpu_delta / system_delta) * online_cpus * 100.0

            mem_usage = stats["memory_stats"].get("usage", 0)
            mem_limit = stats["memory_stats"].get("limit", 0)
            return {
                "cpu_pct": cpu_pct,
                "mem_mb": mem_usage / (1024 * 1024),
                "mem_limit_mb": mem_limit / (1024 * 1024) if mem_limit else 0.0,
            }
        except Exception:
            return {"cpu_pct": 0.0, "mem_mb": 0.0, "mem_limit_mb": 0.0}

    def infer(self, key: str, image_bytes: bytes, prompt: str,
              on_token=None) -> dict:
        image_bytes = _resize_image(image_bytes, max_side=720)

        with self._lock:
            info = self._containers.get(key)
            local_info = self._local_models.get(key)
        if local_info:
            return self._infer_transformers(local_info, image_bytes, prompt, on_token)
        if not info:
            raise RuntimeError(f"Model '{key}' is not deployed")

        port = info["port"]
        from PIL import Image as _PILImage

        current_bytes = image_bytes
        _img_for_size = _PILImage.open(io.BytesIO(image_bytes))
        orig_w, orig_h = _img_for_size.size
        log(f"[infer] {key}: image {orig_w}×{orig_h} ({len(image_bytes)//1024} KB) → port {port}")

        t0 = time.perf_counter()
        output_parts: list[str] = []
        for attempt in range(5):
            b64 = base64.b64encode(current_bytes).decode()
            payload = {
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                "max_tokens": 256,
                "stream": True,
            }
            log(f"[infer] {key}: attempt {attempt+1}/5 ({len(current_bytes)//1024} KB)")
            r = requests.post(f"http://localhost:{port}/v1/chat/completions",
                              json=payload, stream=True, timeout=180)
            if r.status_code == 400:
                err_body = {}
                try:
                    err_body = r.json()
                except Exception:
                    pass
                err_type = (err_body.get("error") or {}).get("type", "") or str(err_body)
                if "exceed_context_size" in err_type or "context" in err_type.lower():
                    img = _PILImage.open(io.BytesIO(current_bytes))
                    w, h = img.size
                    nw, nh = int(w * 0.6), int(h * 0.6)
                    log(f"[infer] {key}: context exceeded — resizing {w}×{h} → {nw}×{nh}")
                    if on_token:
                        on_token(f"\n[resizing {w}×{h}→{nw}×{nh}, retrying…]\n")
                    img = img.resize((nw, nh), _PILImage.LANCZOS)
                    buf = io.BytesIO()
                    img.save(buf, "JPEG", quality=85)
                    current_bytes = buf.getvalue()
                    output_parts.clear()
                    continue
            r.raise_for_status()

            for line in r.iter_lines():
                if not line:
                    continue
                if line.startswith(b"data: "):
                    raw = line[6:]
                    if raw == b"[DONE]":
                        break
                    try:
                        chunk_data = json.loads(raw)
                        delta = chunk_data["choices"][0]["delta"].get("content", "")
                        if delta:
                            output_parts.append(delta)
                            if on_token:
                                on_token(delta)
                    except Exception:
                        pass
            break

        lat = int((time.perf_counter() - t0) * 1000)
        response = "".join(output_parts)
        tps = round(len(response.split()) / max(lat / 1000, 1e-6), 1) if response else None
        log(f"[infer] {key}: done in {lat}ms")

        return {"response": response, "latency_ms": lat, "tokens_per_sec": tps}

    def _infer_transformers(self, local_info: dict, image_bytes: bytes, prompt: str,
                            on_token=None) -> dict:
        from PIL import Image
        import torch
        import threading as _th
        from transformers import TextIteratorStreamer

        processor  = local_info["processor"]
        model      = local_info["model"]
        family     = local_info.get("family", "generic_vlm")
        n_threads  = local_info.get("n_threads", 4)
        current_bytes = image_bytes
        tok = getattr(processor, "tokenizer", processor)

        # Re-apply thread budget — another model deploy may have changed the global
        import torch as _torch
        _torch.set_num_threads(n_threads)
        _torch.set_num_interop_threads(max(1, n_threads // 2))

        t0 = time.perf_counter()
        output_parts: list[str] = []

        for attempt in range(5):
            image = Image.open(io.BytesIO(current_bytes)).convert("RGB")
            log(f"[infer] transformers/{family}: attempt {attempt+1}/5, {image.width}×{image.height}, tokenizing…")

            if family == "internvl":
                inputs = processor(
                    images=image,
                    text=f"{processor.image_token}\n{prompt}",
                    return_tensors="pt",
                )
            elif family == "smolvlm":
                messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
                text = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(text=text, images=[image], return_tensors="pt")
            else:
                messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
                inputs = processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt",
                )
            inputs = inputs.to(model.device)

            streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
            _exc: list[Optional[Exception]] = [None]

            def _generate(inputs=inputs, streamer=streamer):
                try:
                    with torch.inference_mode():
                        model.generate(**inputs, max_new_tokens=256,
                                       do_sample=False, streamer=streamer)
                except Exception as e:
                    _exc[0] = e
                    try:
                        streamer.end()
                    except Exception:
                        pass

            log(f"[infer] transformers/{family}: generating…")
            output_parts.clear()
            gen_thread = _th.Thread(target=_generate, daemon=True)
            gen_thread.start()

            for chunk in streamer:
                if chunk:
                    output_parts.append(chunk)
                    if on_token:
                        on_token(chunk)

            gen_thread.join()

            if _exc[0] is not None:
                e = _exc[0]
                if "out of memory" in str(e).lower():
                    w, h = image.width, image.height
                    nw, nh = int(w * 0.6), int(h * 0.6)
                    log(f"[infer] transformers/{family}: OOM — resizing {w}×{h} → {nw}×{nh}")
                    if on_token:
                        on_token(f"\n[OOM — resizing {w}×{h}→{nw}×{nh}, retrying…]\n")
                    gc.collect()
                    buf = io.BytesIO()
                    image.resize((nw, nh), Image.LANCZOS).save(buf, "JPEG", quality=85)
                    current_bytes = buf.getvalue()
                    del inputs
                    gc.collect()
                    continue
                raise e
            break

        lat = int((time.perf_counter() - t0) * 1000)
        output_text = "".join(output_parts)
        toks = len(output_parts)
        tps = round(toks / max(lat / 1000, 1e-6), 1) if toks else None
        log(f"[infer] transformers/{family}: done in {lat}ms ({toks} chunks)")
        return {"response": output_text, "latency_ms": lat, "tokens_per_sec": tps}


# ══════════════════════════════════════════════════════════════════════════════
# ModelCard
# ══════════════════════════════════════════════════════════════════════════════

class ModelCard(ctk.CTkFrame):
    """Flat list-row style — no card box, just structured rows with a bottom rule."""

    def __init__(self, parent, cfg: dict, manager: DockerModelManager, **kwargs):
        super().__init__(parent, corner_radius=0, fg_color="transparent", **kwargs)
        self.cfg     = cfg
        self.manager = manager
        self.key     = cfg["key"]
        self._status = "stopped"
        self._cpu_history: list[float] = []
        self._mem_history: list[float] = []

        # ── Row 1: name + status + buttons ────────────────────────────────────
        row1 = ctk.CTkFrame(self, fg_color="transparent")
        row1.pack(fill="x", pady=(10, 0))
        row1.columnconfigure(1, weight=1)

        # Status square
        self._dot = ctk.CTkLabel(row1, text="▮", text_color=STATUS_COLOR["stopped"],
                                 font=ctk.CTkFont(size=9), width=14)
        self._dot.grid(row=0, column=0, padx=(0, 8), sticky="w")

        color = cfg.get("color", TXT_PRI)
        ctk.CTkLabel(row1, text=cfg["label"],
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=color, anchor="w").grid(row=0, column=1, sticky="w")

        self._status_lbl = ctk.CTkLabel(row1, text="stopped",
                                        font=ctk.CTkFont(family="Menlo", size=10),
                                        text_color=TXT_DIM, anchor="e")
        self._status_lbl.grid(row=0, column=2, padx=(8, 12), sticky="e")

        # Buttons — flat, text-only style
        btn_frame = ctk.CTkFrame(row1, fg_color="transparent")
        btn_frame.grid(row=0, column=3, sticky="e")

        self._deploy_btn = ctk.CTkButton(
            btn_frame, text="deploy", width=58, height=22,
            fg_color="transparent", hover_color=GLASS_SEP,
            border_width=1, border_color=CARD_BORDER,
            text_color=TXT_SEC, font=ctk.CTkFont(size=10),
            corner_radius=2,
            command=self._on_deploy,
        )
        self._deploy_btn.pack(side="left")

        self._stop_btn = ctk.CTkButton(
            btn_frame, text="stop", width=46, height=22,
            fg_color="transparent", hover_color=GLASS_SEP,
            border_width=1, border_color=CARD_BORDER,
            text_color=TXT_SEC, font=ctk.CTkFont(size=10),
            corner_radius=2,
            command=self._on_stop,
        )
        self._stop_btn.pack(side="left", padx=(4, 0))

        # ── Row 2: resource inputs ─────────────────────────────────────────────
        row2 = ctk.CTkFrame(self, fg_color="transparent")
        row2.pack(fill="x", pady=(4, 0), padx=(22, 0))

        self._cpu_var = ctk.StringVar(value=str(cfg.get("default_cpus", 4)))
        self._ram_var = ctk.StringVar(value=str(cfg.get("default_memory_mb", 4000)))

        self._build_field(row2, "cpu", self._cpu_var)
        ctk.CTkLabel(row2, text="", width=16).pack(side="left")
        self._build_field(row2, "ram mb", self._ram_var)

        # ── Row 3: live stats ──────────────────────────────────────────────────
        row3 = ctk.CTkFrame(self, fg_color="transparent")
        row3.pack(fill="x", pady=(3, 0), padx=(22, 0))

        self._stats_hdr = ctk.CTkLabel(
            row3, text="",
            font=ctk.CTkFont(family="Menlo", size=9),
            text_color=TXT_DIM, anchor="w",
        )
        self._stats_hdr.pack(side="left")

        self._stats_canvas = tk.Canvas(
            row3, height=20, bg=BG_COLOR, highlightthickness=0, bd=0,
        )
        self._stats_canvas.pack(side="right", fill="x", expand=True, padx=(10, 0))

        # ── Bottom separator ───────────────────────────────────────────────────
        ctk.CTkFrame(self, height=1, fg_color=CARD_BORDER, corner_radius=0).pack(
            fill="x", pady=(10, 0)
        )

    def _build_field(self, parent, label: str, var: ctk.StringVar):
        ctk.CTkLabel(parent, text=label,
                     font=ctk.CTkFont(size=9), text_color=TXT_DIM,
                     width=36, anchor="w").pack(side="left")
        ctk.CTkEntry(parent, textvariable=var, width=52,
                     fg_color=ACCENT_FG, border_color=CARD_BORDER, border_width=1,
                     font=ctk.CTkFont(family="Menlo", size=10), height=20,
                     corner_radius=2).pack(side="left", padx=(2, 0))

    # ── Deploy / Stop ──────────────────────────────────────────────────────────

    def _on_deploy(self):
        try:
            cpus = float(self._cpu_var.get())
            mem  = int(self._ram_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "CPU and RAM must be numbers.")
            return

        if not self._ensure_gguf():
            return

        self._set_ui_status("deploying", "Deploying…")
        self._set_all_btns("disabled")

        def _run():
            try:
                self.manager.deploy(self.key, self.cfg, mem, cpus)
            except Exception as e:
                import traceback; traceback.print_exc()
                self.after(0, lambda err=str(e): (
                    self._set_ui_status("error", "Error"),
                    messagebox.showerror(f"Deploy failed — {self.cfg['label']}", err),
                    self._set_all_btns("normal"),
                ))

        threading.Thread(target=_run, daemon=True).start()

    def _on_stop(self):
        self._set_ui_status("stopped", "Stopping…")
        threading.Thread(target=lambda: self.manager.stop(self.key), daemon=True).start()

    def _ensure_gguf(self) -> bool:
        """
        Check that a GGUF is available. Auto-detect from model_dir.
        Shows a hint if missing. Returns True if ready.
        """
        # Already set and exists
        raw = self.cfg.get("model_path", "")
        if raw and Path(raw).expanduser().exists():
            self._auto_detect_mmproj()
            return True

        # Scan model_dir for GGUF files
        model_dir = _resolve_model_dir(self.cfg)
        if model_dir:
            main_ggufs, mmprojs = _existing_ggufs(model_dir)
            best = _best_gguf(main_ggufs)
            if best:
                self.cfg["model_path"] = str(best)
                mm = _best_gguf(mmprojs)
                if mm:
                    self.cfg["mmproj_path"] = str(mm)
                self._save_cfg()
                return True

        messagebox.showinfo(
            "GGUF not found",
            f"No GGUF file found for {self.cfg['label']}.\n\n"
            "Run:\n"
            "python3 demo/prepare_ggufs.py\n\n"
            "Then restart the app.",
        )
        return False

    def _auto_detect_mmproj(self):
        raw = self.cfg.get("mmproj_path", "")
        if raw and Path(raw).expanduser().exists():
            return
        model_dir = _resolve_model_dir(self.cfg)
        if not model_dir:
            return
        _, mmprojs = _existing_ggufs(model_dir)
        best = _best_gguf(mmprojs)
        if best:
            self.cfg["mmproj_path"] = str(best)

    # ── Status ─────────────────────────────────────────────────────────────────

    def apply_status(self, info: dict):
        status = info.get("status", "stopped")
        port = info.get("port", "")
        text_map = {
            "stopped":   "—",
            "deploying": "deploying",
            "loading":   "loading",
            "ready":     f":{port}",
            "error":     "error",
        }
        self._set_ui_status(status, text_map.get(status, status))
        self._update_stats(info)
        is_idle = status in ("stopped", "ready", "error")
        self._set_all_btns("normal" if is_idle else "disabled")

    def _set_ui_status(self, status: str, text: str):
        self._status = status
        color = STATUS_COLOR.get(status, TXT_DIM)
        self._dot.configure(text_color=color)
        self._status_lbl.configure(text=text, text_color=color)

    def _set_all_btns(self, state: str):
        for btn in (self._deploy_btn, self._stop_btn):
            btn.configure(state=state)

    def _update_stats(self, info: dict):
        cpu_pct = float(info.get("cpu_pct", 0.0) or 0.0)
        mem_mb  = float(info.get("mem_mb", 0.0) or 0.0)
        mem_lim = float(info.get("mem_limit_mb", 0.0) or 0.0)
        if info.get("status") == "stopped":
            cpu_pct = mem_mb = 0.0

        self._cpu_history.append(cpu_pct)
        self._mem_history.append(mem_mb)
        self._cpu_history = self._cpu_history[-60:]
        self._mem_history = self._mem_history[-60:]

        if cpu_pct > 0 or mem_mb > 0:
            self._stats_hdr.configure(
                text=f"cpu {cpu_pct:.0f}%  ram {mem_mb:.0f}mb",
                text_color=TXT_SEC,
            )
        else:
            self._stats_hdr.configure(text="", text_color=TXT_DIM)
        self._draw_stats(mem_lim)

    def _draw_stats(self, mem_lim: float):
        canvas = self._stats_canvas
        w = int(canvas.winfo_width() or 160)
        h = 20
        canvas.configure(width=w, height=h, bg=CARD_FG)
        canvas.delete("all")
        self._draw_series(canvas, self._cpu_history, w, h, "#3a6090", 100.0)
        scale = mem_lim if mem_lim > 0 else max(self._mem_history or [1.0])
        self._draw_series(canvas, self._mem_history, w, h, "#2a7050", scale)

    def _draw_series(self, canvas, values, w, h, color, max_val):
        if len(values) < 2 or max_val <= 0:
            return
        step = w / max(len(values) - 1, 1)
        pts = []
        for i, v in enumerate(values):
            pts.extend((i * step, h - min(max(v / max_val, 0), 1) * h))
        canvas.create_line(*pts, fill=color, width=1, smooth=True)

    @property
    def is_ready(self) -> bool:
        return self._status == "ready"

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save_cfg(self):
        try:
            all_cfgs = json.loads(MODELS_JSON.read_text())
            for i, c in enumerate(all_cfgs):
                if c.get("key") == self.key:
                    all_cfgs[i] = self.cfg
                    break
            MODELS_JSON.write_text(json.dumps(all_cfgs, indent=2))
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# ImagePanel
# ══════════════════════════════════════════════════════════════════════════════

class ImagePanel(ctk.CTkFrame):
    PLACEHOLDER = "click to upload"
    PREVIEW_SIZE = (340, 240)

    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color=ACCENT_FG, corner_radius=2,
                         border_width=1, border_color=CARD_BORDER,
                         cursor="hand2", **kwargs)
        self._image_bytes: Optional[bytes] = None
        self._tk_img: Optional[ctk.CTkImage] = None

        self._lbl = ctk.CTkLabel(self, text=self.PLACEHOLDER,
                                  font=ctk.CTkFont(family="Menlo", size=10),
                                  text_color=TXT_DIM)
        self._lbl.pack(expand=True)

        self.bind("<Button-1>", lambda _: self._browse())
        self._lbl.bind("<Button-1>", lambda _: self._browse())

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All", "*.*")],
        )
        if not path:
            return
        with open(path, "rb") as f:
            raw = f.read()
        self._set(raw)

    def _set(self, raw: bytes):
        pil = Image.open(io.BytesIO(raw))
        pil = ImageOps.exif_transpose(pil).convert("RGB")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=95)
        self._image_bytes = buf.getvalue()
        pil.thumbnail(self.PREVIEW_SIZE, Image.LANCZOS)
        self._tk_img = ctk.CTkImage(
            light_image=pil,
            dark_image=pil,
            size=pil.size,
        )
        self._lbl.configure(image=self._tk_img, text="")

    @property
    def image_bytes(self) -> Optional[bytes]:
        return self._image_bytes

    def has_image(self) -> bool:
        return self._image_bytes is not None


# ══════════════════════════════════════════════════════════════════════════════
# ResultsPanel
# ══════════════════════════════════════════════════════════════════════════════

class ResultsPanel(ctk.CTkScrollableFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self._blocks: dict[str, dict] = {}

    def prepare(self, keys: list[str], labels: dict[str, str], colors: dict[str, str]):
        for k in list(self._blocks):
            if k not in keys:
                self._blocks.pop(k)["frame"].destroy()

        for key in keys:
            if key in self._blocks:
                self._blocks[key]["_text"] = ""
                self._blocks[key]["body"].configure(text="", text_color=TXT_SEC)
                self._blocks[key]["lat"].configure(text="")
                continue

            frame = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
            frame.pack(fill="x", pady=(0, 0))

            hdr = ctk.CTkFrame(frame, fg_color="transparent")
            hdr.pack(fill="x", pady=(12, 4))

            ctk.CTkLabel(hdr, text=labels.get(key, key),
                         font=ctk.CTkFont(size=11, weight="bold"),
                         text_color=colors.get(key, TXT_PRI)).pack(side="left")

            lat = ctk.CTkLabel(hdr, text="",
                               font=ctk.CTkFont(family="Menlo", size=9),
                               text_color=TXT_DIM)
            lat.pack(side="right")

            body = ctk.CTkLabel(frame, text="",
                                font=ctk.CTkFont(size=12),
                                text_color=TXT_SEC,
                                justify="left", anchor="w", wraplength=460)
            body.pack(fill="x", pady=(0, 0))

            # thin separator line
            ctk.CTkFrame(frame, height=1, fg_color=CARD_BORDER,
                         corner_radius=0).pack(fill="x", pady=(10, 0))

            self._blocks[key] = {"frame": frame, "body": body, "lat": lat, "_text": ""}

    def append_streaming(self, key: str, chunk: str):
        if key not in self._blocks:
            return
        self._blocks[key]["_text"] += chunk
        self._blocks[key]["body"].configure(
            text=self._blocks[key]["_text"], text_color=TXT_PRI)

    def set_result(self, key: str, text: str, latency_ms: int,
                   tps: Optional[float] = None):
        if key not in self._blocks:
            return
        lat_str = f"{latency_ms / 1000:.1f}s"
        if tps:
            lat_str += f"  {tps:.1f}t/s"
        self._blocks[key]["body"].configure(text=text, text_color=TXT_PRI)
        self._blocks[key]["lat"].configure(text=lat_str, text_color=TXT_SEC)

    def set_error(self, key: str, error: str):
        if key not in self._blocks:
            return
        self._blocks[key]["body"].configure(text=f"error: {error}",
                                             text_color="#dc2626")
        self._blocks[key]["lat"].configure(text="")


# ══════════════════════════════════════════════════════════════════════════════
# LogsPanel
# ══════════════════════════════════════════════════════════════════════════════

class LogsPanel(ctk.CTkFrame):
    """Real-time log viewer fed from the global _log_queue."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color=CARD_FG,
                         corner_radius=2,
                         border_width=1, border_color=CARD_BORDER, **kwargs)

        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.pack(fill="x", padx=12, pady=(10, 6))
        ctk.CTkLabel(hdr, text="logs",
                     font=ctk.CTkFont(family="Menlo", size=10),
                     text_color=TXT_DIM).pack(side="left")
        self._clear_btn = ctk.CTkButton(
            hdr, text="clr", width=32, height=18,
            fg_color="transparent", hover_color=GLASS_SEP,
            border_width=1, border_color=CARD_BORDER,
            text_color=TXT_DIM,
            font=ctk.CTkFont(family="Menlo", size=9), corner_radius=2,
            command=self._clear,
        )
        self._clear_btn.pack(side="right")

        self._text = tk.Text(
            self,
            bg=CARD_FG, fg=TXT_DIM,
            insertbackground=TXT_DIM,
            font=("Menlo", 9),
            wrap="word",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            state="disabled",
        )
        self._text.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # colour tags
        self._text.tag_configure("ts",      foreground="#444444")
        self._text.tag_configure("deploy",  foreground="#4a7aaa")
        self._text.tag_configure("infer",   foreground="#3a9060")
        self._text.tag_configure("warn",    foreground="#a08030")
        self._text.tag_configure("err",     foreground="#a04040")
        self._text.tag_configure("default", foreground="#33334a")

    def _classify(self, line: str) -> str:
        if "[deploy]" in line:
            return "deploy"
        if "[infer]" in line:
            if "OOM" in line or "error" in line.lower():
                return "err"
            return "infer"
        if "ERROR" in line or "error" in line:
            return "err"
        if "WARNING" in line or "warn" in line.lower():
            return "warn"
        return "default"

    def append(self, line: str):
        tag = self._classify(line)
        # split timestamp from rest
        parts = line.split("  ", 1)
        self._text.configure(state="normal")
        if len(parts) == 2:
            self._text.insert("end", parts[0] + "  ", "ts")
            self._text.insert("end", parts[1] + "\n", tag)
        else:
            self._text.insert("end", line + "\n", tag)
        self._text.configure(state="disabled")
        self._text.see("end")

    def _clear(self):
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.configure(state="disabled")


# ══════════════════════════════════════════════════════════════════════════════
# App
# ══════════════════════════════════════════════════════════════════════════════

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("VLM Benchmark")
        self.geometry("1400x820")
        self.minsize(1100, 650)
        self.configure(fg_color=BG_COLOR)

        self._manager: Optional[DockerModelManager] = None
        self._cards:   dict[str, ModelCard] = {}

        self._build_ui()
        self._connect_docker()
        self.after(POLL_MS, self._poll)
        self.after(150, self._poll_logs)

    def _build_ui(self):
        # ── Topbar ────────────────────────────────────────────────────────────
        topbar = ctk.CTkFrame(self, fg_color=TOPBAR_FG, height=36, corner_radius=0)
        topbar.pack(fill="x", side="top")
        topbar.pack_propagate(False)

        ctk.CTkLabel(topbar, text="  vlm benchmark",
                     font=ctk.CTkFont(family="Menlo", size=11),
                     text_color=TXT_SEC).pack(side="left", padx=4)

        self._docker_lbl = ctk.CTkLabel(topbar, text="connecting",
                                         font=ctk.CTkFont(family="Menlo", size=9),
                                         text_color=TXT_DIM)
        self._docker_lbl.pack(side="right", padx=16)

        tk.Frame(self, bg=CARD_BORDER, height=1).pack(fill="x")

        # ── Resizable 3-pane layout ───────────────────────────────────────────
        paned = tk.PanedWindow(
            self, orient=tk.HORIZONTAL,
            bg=SASH_COLOR,
            sashwidth=5, sashrelief="flat",
            opaqueresize=True,
        )
        paned.pack(fill="both", expand=True)

        # ── Left — models sidebar ─────────────────────────────────────────────
        left_outer = tk.Frame(paned, bg=CARD_FG)
        paned.add(left_outer, width=290, minsize=180, stretch="never")

        lhdr = tk.Frame(left_outer, bg=CARD_FG, height=34)
        lhdr.pack(fill="x")
        lhdr.pack_propagate(False)
        ctk.CTkLabel(lhdr, text="models", fg_color=CARD_FG,
                     font=ctk.CTkFont(family="Menlo", size=9),
                     text_color=TXT_DIM).pack(side="left", padx=14, pady=8)
        tk.Frame(left_outer, bg=CARD_BORDER, height=1).pack(fill="x")

        self._card_scroll = ctk.CTkScrollableFrame(
            left_outer, fg_color=CARD_FG,
            scrollbar_button_color=CARD_BORDER,
            scrollbar_button_hover_color=GLASS_SEP,
        )
        self._card_scroll.pack(fill="both", expand=True, padx=10)

        # ── Center — image + prompt + results ─────────────────────────────────
        center_outer = tk.Frame(paned, bg=BG_COLOR)
        paned.add(center_outer, minsize=380, stretch="always")

        center = ctk.CTkFrame(center_outer, fg_color="transparent")
        center.pack(fill="both", expand=True, padx=20, pady=14)
        center.columnconfigure(0, weight=1)
        center.rowconfigure(3, weight=1)

        self._img_panel = ImagePanel(center, height=200)
        self._img_panel.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        ctrl = ctk.CTkFrame(center, fg_color="transparent")
        ctrl.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        ctrl.columnconfigure(0, weight=1)

        self._prompt = ctk.CTkEntry(
            ctrl,
            font=ctk.CTkFont(size=12), height=32,
            fg_color=ACCENT_FG,
            border_color=CARD_BORDER, border_width=1,
            corner_radius=2,
            text_color=TXT_PRI,
            placeholder_text="what do you see in this image?",
            placeholder_text_color=TXT_DIM,
        )
        self._prompt.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        self._run_btn = ctk.CTkButton(
            ctrl, text="run", width=68, height=32,
            fg_color="transparent", hover_color=GLASS_SEP,
            border_width=1, border_color=CARD_BORDER,
            text_color=TXT_SEC,
            font=ctk.CTkFont(family="Menlo", size=11),
            corner_radius=2,
            command=self._on_run,
        )
        self._run_btn.grid(row=0, column=1)

        tk.Frame(center_outer, bg=CARD_BORDER, height=1).pack(fill="x",
                                                               before=center)
        ctk.CTkFrame(center, height=1, fg_color=CARD_BORDER,
                     corner_radius=0).grid(row=2, column=0, sticky="ew")

        self._results = ResultsPanel(center)
        self._results.grid(row=3, column=0, sticky="nsew")

        # ── Right — logs ──────────────────────────────────────────────────────
        right_outer = tk.Frame(paned, bg=CARD_FG)
        paned.add(right_outer, width=290, minsize=180, stretch="never")

        self._logs_panel = LogsPanel(right_outer)
        self._logs_panel.pack(fill="both", expand=True)

    def _connect_docker(self):
        def _try():
            self._manager = DockerModelManager()
            if self._manager.client is not None:
                self.after(0, self._on_docker_ok)
            else:
                err = self._manager.docker_error or "Docker unavailable"
                self.after(0, lambda detail=err: self._on_docker_partial(detail))
        threading.Thread(target=_try, daemon=True).start()

    def _on_docker_ok(self):
        self._docker_lbl.configure(text="docker", text_color=STATUS_COLOR["ready"])
        self._load_cards()

    def _on_docker_partial(self, err: str):
        self._docker_lbl.configure(text="docker unavailable", text_color=STATUS_COLOR["error"])
        self._load_cards()

    def _load_cards(self):
        try:
            cfgs = json.loads(MODELS_JSON.read_text())
        except Exception as e:
            messagebox.showerror("models.json error", str(e))
            return
        for cfg in cfgs:
            if not cfg.get("key"):
                continue
            if not cfg.get("enabled", True):
                continue
            model_dir = _resolve_model_dir(cfg)
            model_path = cfg.get("model_path", "")
            has_model_dir = bool(model_dir and model_dir.exists())
            has_model_path = bool(model_path and Path(model_path).expanduser().exists())
            if not has_model_dir and not has_model_path:
                continue
            card = ModelCard(self._card_scroll, cfg, self._manager)
            card.pack(fill="x", pady=(0, 10))
            self._cards[cfg["key"]] = card

    def _on_docker_fail(self, err: str):
        self._docker_lbl.configure(text="● Docker: not found", text_color="#cc4444")
        messagebox.showerror("Docker not available",
                              f"{err}\n\nStart Docker Desktop and restart the app.")

    def _poll(self):
        if self._manager:
            for key, card in self._cards.items():
                try:
                    card.apply_status(self._manager.get_status(key))
                except Exception:
                    pass
        self.after(POLL_MS, self._poll)

    def _poll_logs(self):
        """Drain the global log queue into the LogsPanel — runs on the main thread."""
        try:
            while True:
                line = _log_queue.get_nowait()
                self._logs_panel.append(line)
        except _queue.Empty:
            pass
        self.after(150, self._poll_logs)

    def _on_run(self):
        if not self._img_panel.has_image():
            messagebox.showwarning("No image", "Upload an image first.")
            return

        ready = [k for k, c in self._cards.items() if c.is_ready]
        if not ready:
            messagebox.showwarning("No models ready",
                                   "Deploy at least one model and wait for 'Ready'.")
            return

        prompt = self._prompt.get().strip() or "What do you see in this image?"
        img    = self._img_panel.image_bytes
        labels = {k: self._cards[k].cfg["label"]             for k in ready}
        colors = {k: self._cards[k].cfg.get("color", "#fff") for k in ready}

        self._results.prepare(ready, labels, colors)
        self._run_btn.configure(state="disabled")

        def _worker(key: str):
            def _on_token(chunk: str):
                self.after(0, lambda c=chunk: self._results.append_streaming(key, c))

            try:
                r = self._manager.infer(key, img, prompt, on_token=_on_token)
                self.after(0, lambda r=r: self._results.set_result(
                    key, r["response"], r["latency_ms"], r.get("tokens_per_sec")))
            except Exception as e:
                self.after(0, lambda err=str(e): self._results.set_error(key, err))
            finally:
                self.after(0, lambda: self._run_btn.configure(state="normal"))

        for key in ready:
            threading.Thread(target=_worker, args=(key,), daemon=True).start()

    def on_close(self):
        if self._manager:
            try:
                self._manager.stop_all()
            except Exception:
                pass
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
