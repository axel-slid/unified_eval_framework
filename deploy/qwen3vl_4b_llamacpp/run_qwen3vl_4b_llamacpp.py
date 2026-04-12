#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import io
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from PIL import Image


DEPLOY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOY_DIR.parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "Qwen3-VL-4B-Instruct"
DEFAULT_REPO_ID = "lmstudio-community/Qwen3-VL-4B-Instruct-GGUF"
DEFAULT_PORT = 8080
DEFAULT_CTX = 2048
DEFAULT_THREADS = max(1, os.cpu_count() or 4)
DEFAULT_MAX_IMAGE_SIDE = 720
DOCKER_IMAGES = [
    "ghcr.io/ggml-org/llama.cpp:server",
    "ghcr.io/ggerganov/llama.cpp:server",
]
QUANT_PREF = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_K_S", "Q4_0"]

DEFAULT_PROMPT = (
    "Describe the image clearly and concisely. "
    "If there are people, mention what they are doing."
)


def log(msg: str) -> None:
    print(msg, flush=True)


def _existing_ggufs(model_dir: Path) -> tuple[list[Path], list[Path]]:
    if not model_dir.is_dir():
        return [], []
    all_g = sorted(model_dir.glob("*.gguf"))
    main = [f for f in all_g if "mmproj" not in f.name.lower()]
    mmproj = [f for f in all_g if "mmproj" in f.name.lower()]
    return main, mmproj


def _rank_quant(name: str) -> int:
    upper = name.upper()
    for i, q in enumerate(QUANT_PREF):
        if q in upper:
            return i
    return 999


def _best_gguf(files: list[Path]) -> Optional[Path]:
    if not files:
        return None
    return sorted(files, key=lambda p: (_rank_quant(p.name), p.name))[0]


def find_assets(model_dir: Path) -> tuple[Optional[Path], Optional[Path]]:
    main, mmproj = _existing_ggufs(model_dir)
    return _best_gguf(main), _best_gguf(mmproj)


def prepare_assets(model_dir: Path, repo_id: str) -> tuple[Path, Path]:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub\n"
            "Install with: pip install huggingface_hub"
        ) from exc

    model_dir.mkdir(parents=True, exist_ok=True)
    all_files = list(list_repo_files(repo_id))
    ggufs = [f for f in all_files if f.endswith(".gguf")]
    main_files = [f for f in ggufs if "mmproj" not in f.lower()]
    mmproj_files = [f for f in ggufs if "mmproj" in f.lower()]
    if not main_files:
        raise SystemExit(f"No GGUF model files found in {repo_id}")
    chosen_main = sorted(main_files, key=_rank_quant)[0]
    chosen_mmproj = sorted(mmproj_files, key=_rank_quant)[0] if mmproj_files else None

    log(f"Downloading model from {repo_id}: {chosen_main}")
    main_path = Path(hf_hub_download(repo_id=repo_id, filename=chosen_main, local_dir=str(model_dir)))

    if not chosen_mmproj:
        raise SystemExit(f"No mmproj GGUF found in {repo_id}")
    log(f"Downloading mmproj from {repo_id}: {chosen_mmproj}")
    mmproj_path = Path(hf_hub_download(repo_id=repo_id, filename=chosen_mmproj, local_dir=str(model_dir)))

    return main_path, mmproj_path


def _resize_image(image_bytes: bytes, max_side: int) -> bytes:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    if max(w, h) <= max_side:
        return image_bytes
    scale = max_side / max(w, h)
    nw, nh = int(w * scale), int(h * scale)
    buf = io.BytesIO()
    img.resize((nw, nh), Image.LANCZOS).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def _payload(image_bytes: bytes, prompt: str, max_tokens: int) -> dict:
    b64 = base64.b64encode(image_bytes).decode()
    return {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": max_tokens,
        "stream": False,
    }


def wait_for_server(port: int, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    last_err = None
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return
        except Exception as exc:
            last_err = exc
        time.sleep(1)
    raise SystemExit(f"llama.cpp server on port {port} did not become healthy in {timeout_s}s. Last error: {last_err}")


def infer_once(port: int, image_path: Path, prompt: str, max_tokens: int, max_image_side: int) -> dict:
    image_bytes = _resize_image(image_path.read_bytes(), max_image_side)
    r = requests.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json=_payload(image_bytes, prompt, max_tokens),
        timeout=300,
    )
    r.raise_for_status()
    data = r.json()
    return {"response": data["choices"][0]["message"]["content"], "raw": data}


def _local_llama_server_cmd(
    llama_server_bin: str,
    model_path: Path,
    mmproj_path: Path,
    port: int,
    ctx_size: int,
    threads: int,
    extra_args: list[str],
) -> list[str]:
    return [
        llama_server_bin,
        "--model", str(model_path),
        "--mmproj", str(mmproj_path),
        "--host", "0.0.0.0",
        "--port", str(port),
        "--ctx-size", str(ctx_size),
        "--threads", str(threads),
        "--threads-batch", str(threads),
        "--mlock",
        *extra_args,
    ]


def _docker_server_cmd(
    image: str,
    model_path: Path,
    mmproj_path: Path,
    port: int,
    ctx_size: int,
    threads: int,
    extra_args: list[str],
) -> list[str]:
    model_dir = model_path.parent
    return [
        "docker", "run", "--rm", "--init",
        "-p", f"{port}:8080",
        "-v", f"{model_dir}:/mnt/model:ro",
        image,
        "--model", f"/mnt/model/{model_path.name}",
        "--mmproj", f"/mnt/model/{mmproj_path.name}",
        "--host", "0.0.0.0",
        "--port", "8080",
        "--ctx-size", str(ctx_size),
        "--threads", str(threads),
        "--threads-batch", str(threads),
        "--mlock",
        *extra_args,
    ]


def choose_docker_image() -> str:
    if not shutil.which("docker"):
        raise SystemExit("Docker CLI not found. Install Docker or use --backend local with a llama-server binary.")
    for image in DOCKER_IMAGES:
        probe = subprocess.run(["docker", "image", "inspect", image], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if probe.returncode == 0:
            return image
    for image in DOCKER_IMAGES:
        log(f"Pulling {image} ...")
        pull = subprocess.run(["docker", "pull", image])
        if pull.returncode == 0:
            return image
    raise SystemExit("Could not pull a compatible llama.cpp Docker image.")


def command_prepare(args: argparse.Namespace) -> int:
    model_path, mmproj_path = prepare_assets(args.model_dir, args.repo_id)
    log(f"Model GGUF:  {model_path}")
    log(f"MMProj GGUF: {mmproj_path}")
    return 0


def command_serve(args: argparse.Namespace) -> int:
    model_path, mmproj_path = find_assets(args.model_dir)
    if not model_path or not mmproj_path:
        if args.auto_prepare:
            log("GGUF assets missing, downloading them first ...")
            model_path, mmproj_path = prepare_assets(args.model_dir, args.repo_id)
        else:
            raise SystemExit(f"Missing GGUF assets in {args.model_dir}\nRun with `prepare` first or pass --auto-prepare.")

    extra_args = args.extra_arg or []
    if args.backend == "local":
        llama_server_bin = args.llama_server_bin or shutil.which("llama-server")
        if not llama_server_bin:
            raise SystemExit("Could not find llama-server in PATH. Pass --llama-server-bin or use --backend docker.")
        cmd = _local_llama_server_cmd(llama_server_bin, model_path, mmproj_path, args.port, args.ctx_size, args.threads, extra_args)
    else:
        cmd = _docker_server_cmd(choose_docker_image(), model_path, mmproj_path, args.port, args.ctx_size, args.threads, extra_args)

    log("Starting Qwen3-VL-4B Q8_0 via llama.cpp")
    log("Command:")
    log(" ".join(str(part) for part in cmd))
    proc = subprocess.Popen(cmd)
    log(f"PID: {proc.pid}")
    log(f"Health URL: http://127.0.0.1:{args.port}/health")
    try:
        proc.wait()
        return proc.returncode
    except KeyboardInterrupt:
        log("Stopping server ...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        return 130


def command_infer(args: argparse.Namespace) -> int:
    if args.start_server:
        model_path, mmproj_path = find_assets(args.model_dir)
        if not model_path or not mmproj_path:
            if args.auto_prepare:
                model_path, mmproj_path = prepare_assets(args.model_dir, args.repo_id)
            else:
                raise SystemExit("Missing GGUF assets. Run prepare first or pass --auto-prepare.")
        if args.backend == "local":
            llama_server_bin = args.llama_server_bin or shutil.which("llama-server")
            if not llama_server_bin:
                raise SystemExit("Could not find llama-server in PATH. Pass --llama-server-bin or use --backend docker.")
            cmd = _local_llama_server_cmd(llama_server_bin, model_path, mmproj_path, args.port, args.ctx_size, args.threads, args.extra_arg or [])
        else:
            cmd = _docker_server_cmd(choose_docker_image(), model_path, mmproj_path, args.port, args.ctx_size, args.threads, args.extra_arg or [])
        server_proc = subprocess.Popen(cmd)
        try:
            wait_for_server(args.port, args.health_timeout)
            out = infer_once(args.port, args.image, args.prompt, args.max_tokens, args.max_image_side)
            print(out["response"])
            return 0
        finally:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()
    else:
        wait_for_server(args.port, args.health_timeout)
        out = infer_once(args.port, args.image, args.prompt, args.max_tokens, args.max_image_side)
        print(out["response"])
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL-4B Q8_0 with llama.cpp.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="Download the Qwen3-VL-4B GGUF + mmproj assets.")
    p_prepare.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p_prepare.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p_prepare.set_defaults(func=command_prepare)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    common.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    common.add_argument("--port", type=int, default=DEFAULT_PORT)
    common.add_argument("--ctx-size", type=int, default=DEFAULT_CTX)
    common.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    common.add_argument("--backend", choices=["local", "docker"], default="local")
    common.add_argument("--llama-server-bin", default=None)
    common.add_argument("--auto-prepare", action="store_true")
    common.add_argument("--extra-arg", action="append", default=[])

    p_serve = sub.add_parser("serve", parents=[common], help="Start an OpenAI-compatible llama.cpp server.")
    p_serve.set_defaults(func=command_serve)

    p_infer = sub.add_parser("infer", parents=[common], help="Run one inference request.")
    p_infer.add_argument("--image", type=Path, required=True)
    p_infer.add_argument("--prompt", default=DEFAULT_PROMPT)
    p_infer.add_argument("--max-tokens", type=int, default=256)
    p_infer.add_argument("--max-image-side", type=int, default=DEFAULT_MAX_IMAGE_SIDE)
    p_infer.add_argument("--health-timeout", type=int, default=180)
    p_infer.add_argument("--start-server", action="store_true")
    p_infer.set_defaults(func=command_infer)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
