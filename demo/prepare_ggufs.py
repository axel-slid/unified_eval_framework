#!/usr/bin/env python3
"""
Prepare GGUF assets for the demo.

This script downloads GGUF files from Hugging Face when `gguf_hf_repo` is set,
or converts local safetensors checkpoints into GGUF via llama.cpp Docker.
After each model is prepared, it optionally launches a temporary llama-server
container and waits for `/health` so we know the asset can actually load.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional


HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent
MODELS_JSON = HERE / "models.json"

LLAMA_SERVER_IMAGES = [
    "ghcr.io/ggml-org/llama.cpp:server",
    "ghcr.io/ggerganov/llama.cpp:server",
]
LLAMA_FULL_IMAGES = [
    "ghcr.io/ggml-org/llama.cpp:full",
    "ghcr.io/ggerganov/llama.cpp:full",
]
TARGET_QUANT = "Q8_0"
QUANT_PREF = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_K_S", "Q4_0"]

_IMAGE_CACHE: dict[tuple[str, ...], str] = {}


def log(message: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def _resolve_model_dir(cfg: dict) -> Optional[Path]:
    raw = cfg.get("model_dir", "")
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def _existing_ggufs(model_dir: Path) -> tuple[list[Path], list[Path]]:
    if not model_dir.is_dir():
        return [], []
    all_g = sorted(model_dir.glob("*.gguf"))
    main = [f for f in all_g if "mmproj" not in f.name.lower()]
    mmproj = [f for f in all_g if "mmproj" in f.name.lower()]
    return main, mmproj


def _best_gguf(files: list[Path]) -> Optional[Path]:
    for q in QUANT_PREF:
        for f in files:
            if q in f.name.upper():
                return f
    return files[0] if files else None


def _is_preferred_quantized(path: Optional[Path]) -> bool:
    if path is None:
        return False
    name = path.name.upper()
    return TARGET_QUANT in name


def _resolve_docker_image(client, candidates: list[str]) -> str:
    key = tuple(candidates)
    cached = _IMAGE_CACHE.get(key)
    if cached:
        return cached

    errors = []
    for image in candidates:
        log(f"Pulling {image} …")
        try:
            client.images.pull(image)
            _IMAGE_CACHE[key] = image
            return image
        except Exception as e:
            try:
                client.images.get(image)
                log(f"WARNING: pull failed ({e}), using cached image: {image}")
                _IMAGE_CACHE[key] = image
                return image
            except Exception:
                errors.append(f"{image}: {e}")

    raise RuntimeError(
        "Could not find a compatible llama.cpp Docker image.\n"
        + "\n".join(errors)
    )


def download_gguf(cfg: dict) -> tuple[Optional[Path], Optional[Path]]:
    from huggingface_hub import hf_hub_download, list_repo_files

    repo = cfg["gguf_hf_repo"]
    model_dir = _resolve_model_dir(cfg)
    if model_dir is None:
        raise RuntimeError(f"Missing model_dir for {cfg.get('key')}")
    model_dir.mkdir(parents=True, exist_ok=True)

    log(f"Listing files in {repo} …")
    all_files = list(list_repo_files(repo))
    gguf_files = [f for f in all_files if f.endswith(".gguf")]
    main_files = [f for f in gguf_files if "mmproj" not in f.lower()]
    mmproj_files = [f for f in gguf_files if "mmproj" in f.lower()]
    if not main_files:
        raise RuntimeError(f"No GGUF files found in {repo}")

    def _rank(name: str) -> int:
        for i, q in enumerate(QUANT_PREF):
            if q in name.upper():
                return i
        return 99

    chosen_model = sorted(main_files, key=_rank)[0]
    chosen_mmproj = sorted(mmproj_files, key=_rank)[0] if mmproj_files else None

    log(f"Downloading model: {chosen_model}")
    local_model = Path(
        hf_hub_download(repo_id=repo, filename=chosen_model, local_dir=str(model_dir))
    )
    local_mmproj = None

    if chosen_mmproj:
        log(f"Downloading mmproj: {chosen_mmproj}")
        local_mmproj = Path(
            hf_hub_download(
                repo_id=repo,
                filename=chosen_mmproj,
                local_dir=str(model_dir),
            )
        )

    return local_model, local_mmproj


def convert_gguf(cfg: dict, client) -> tuple[Optional[Path], Optional[Path]]:
    model_dir = _resolve_model_dir(cfg)
    if not model_dir or not model_dir.is_dir():
        raise RuntimeError(f"Model directory not found: {model_dir}")

    out_name = f"{model_dir.name}-{TARGET_QUANT}.gguf"
    out_path = model_dir / out_name
    f16_name = f"{model_dir.name}-F16.gguf"
    f16_path = model_dir / f16_name
    if out_path.exists():
        _, mmprojs = _existing_ggufs(model_dir)
        return out_path, _best_gguf(mmprojs)

    image = _resolve_docker_image(client, LLAMA_FULL_IMAGES)
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

    log(f"Converting {model_dir.name} → GGUF ({TARGET_QUANT})")
    log(f"Running in Docker: python3 {' '.join(convert_cmd)}")
    log(f"Then: {' '.join(quantize_cmd)}")

    if not f16_path.exists():
        container = client.containers.run(
            image,
            entrypoint="python3",
            command=convert_cmd,
            volumes={str(model_dir): {"bind": "/model", "mode": "rw"}},
            remove=True,
            detach=True,
        )
        try:
            for chunk in container.logs(stream=True, follow=True):
                line = chunk.decode("utf-8", errors="replace").rstrip()
                if line:
                    log(line)
            exit_code = container.wait()["StatusCode"]
        finally:
            try:
                container.remove(force=True)
            except Exception:
                pass

        if exit_code != 0 or not f16_path.exists():
            raise RuntimeError(
                f"Conversion failed for {cfg.get('label', cfg.get('key'))} "
                f"(exit code {exit_code})"
            )
    else:
        log(f"Reusing existing intermediate GGUF: {f16_name}")

    container = client.containers.run(
        image,
        command=quantize_cmd,
        volumes={str(model_dir): {"bind": "/model", "mode": "rw"}},
        remove=True,
        detach=True,
    )
    try:
        for chunk in container.logs(stream=True, follow=True):
            line = chunk.decode("utf-8", errors="replace").rstrip()
            if line:
                log(line)
        exit_code = container.wait()["StatusCode"]
    finally:
        try:
            container.remove(force=True)
        except Exception:
            pass

    if exit_code != 0 or not out_path.exists():
        raise RuntimeError(
            f"Quantization failed for {cfg.get('label', cfg.get('key'))} "
            f"(exit code {exit_code})"
        )

    if f16_path.exists():
        try:
            f16_path.unlink()
        except OSError:
            pass

    _, mmprojs = _existing_ggufs(model_dir)
    return out_path, _best_gguf(mmprojs)


def verify_gguf(
    client,
    cfg: dict,
    model_path: Path,
    mmproj_path: Optional[Path],
    timeout_s: int,
) -> None:
    import requests

    image = _resolve_docker_image(client, LLAMA_SERVER_IMAGES)
    model_dir = model_path.parent
    volumes = {str(model_dir): {"bind": "/mnt/model", "mode": "ro"}}
    cmd = [
        "--model", f"/mnt/model/{model_path.name}",
        "--host", "0.0.0.0",
        "--port", "8080",
        "--ctx-size", str(cfg.get("ctx_size", 2048)),
        "--threads", "2",
    ]

    if mmproj_path:
        mp = mmproj_path.expanduser().resolve()
        if mp.parent != model_dir:
            volumes[str(mp.parent)] = {"bind": "/mnt/mmproj", "mode": "ro"}
            cmd += ["--mmproj", f"/mnt/mmproj/{mp.name}"]
        else:
            cmd += ["--mmproj", f"/mnt/model/{mp.name}"]

    log(f"Verifying load for {cfg['label']} …")
    container = client.containers.run(
        image,
        command=cmd,
        ports={"8080/tcp": None},
        volumes=volumes,
        detach=True,
        remove=False,
    )

    try:
        deadline = time.time() + timeout_s
        port = None
        while time.time() < deadline:
            container.reload()
            if container.status != "running":
                logs = container.logs(tail=200).decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"llama-server exited while loading {cfg['label']}.\n{logs}"
                )

            ports = container.attrs["NetworkSettings"]["Ports"].get("8080/tcp") or []
            if ports:
                port = ports[0]["HostPort"]
                try:
                    resp = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
                    if resp.status_code == 200 and resp.json().get("status") == "ok":
                        log(f"Verified {cfg['label']} on localhost:{port}")
                        return
                except requests.RequestException:
                    pass
            time.sleep(2)

        logs = container.logs(tail=200).decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Timed out waiting for {cfg['label']} to become healthy.\n{logs}"
        )
    finally:
        try:
            container.stop(timeout=5)
        except Exception:
            pass
        try:
            container.remove(force=True)
        except Exception:
            pass


def prepare_one(
    client,
    cfg: dict,
    verify: bool,
    timeout_s: int,
) -> dict:
    model_dir = _resolve_model_dir(cfg)
    if not cfg.get("gguf_hf_repo") and (not model_dir or not model_dir.exists()):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    main_ggufs, mmprojs = _existing_ggufs(model_dir) if model_dir else ([], [])
    model_path = _best_gguf(main_ggufs)
    mmproj_path = _best_gguf(mmprojs)

    if cfg.get("gguf_hf_repo") and not _is_preferred_quantized(model_path):
        if model_path:
            log(f"Existing GGUF is not a preferred quantization: {model_path.name}")
        log(f"Preparing from Hugging Face repo: {cfg['gguf_hf_repo']}")
        model_path, mmproj_path = download_gguf(cfg)
    elif model_path:
        log(f"Found existing GGUF for {cfg['label']}: {model_path.name}")
    elif cfg.get("gguf_hf_repo"):
        log(f"Preparing from Hugging Face repo: {cfg['gguf_hf_repo']}")
        model_path, mmproj_path = download_gguf(cfg)
    else:
        log(f"Preparing from local model directory: {model_dir}")
        model_path, mmproj_path = convert_gguf(cfg, client)

    if not model_path or not model_path.exists():
        raise RuntimeError(f"No GGUF ready for {cfg['label']}")

    if verify:
        verify_gguf(client, cfg, model_path, mmproj_path, timeout_s)

    cfg["model_path"] = str(model_path)
    if mmproj_path and mmproj_path.exists():
        cfg["mmproj_path"] = str(mmproj_path)
    elif "mmproj_path" in cfg:
        cfg.pop("mmproj_path", None)
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare GGUF files for the demo")
    parser.add_argument(
        "--models",
        nargs="*",
        help="Specific model keys to prepare. Defaults to all entries in demo/models.json.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Prepare GGUFs without launching llama-server health checks.",
    )
    parser.add_argument(
        "--verify-timeout",
        type=int,
        default=300,
        help="Seconds to wait for each model to report healthy.",
    )
    args = parser.parse_args()

    try:
        import docker
    except ModuleNotFoundError:
        log("ERROR: missing Python dependency 'docker'.")
        log("Install demo dependencies with:")
        log("python3 -m pip install -r demo/requirements.txt")
        return 1

    try:
        cfgs = json.loads(MODELS_JSON.read_text())
    except Exception as e:
        log(f"ERROR reading {MODELS_JSON}: {e}")
        return 1

    wanted = set(args.models or [])
    selected = [cfg for cfg in cfgs if not wanted or cfg.get("key") in wanted]
    if not selected:
        log("No matching models found.")
        return 1

    try:
        client = docker.from_env()
        client.ping()
    except Exception as e:
        log(f"ERROR: cannot connect to Docker: {e}")
        return 1

    updated = []
    failures = []
    skipped = []
    for cfg in cfgs:
        if wanted and cfg.get("key") not in wanted:
            updated.append(cfg)
            continue

        label = cfg.get("label", cfg.get("key", "unknown"))
        log("")
        log(f"=== {label} ===")
        try:
            updated.append(
                prepare_one(
                    client,
                    dict(cfg),
                    verify=not args.skip_verify,
                    timeout_s=args.verify_timeout,
                )
            )
        except FileNotFoundError as e:
            skipped.append((cfg.get("key", label), str(e)))
            updated.append(cfg)
            log(f"Skipping: {e}")
        except Exception as e:
            failures.append((cfg.get("key", label), str(e)))
            updated.append(cfg)
            log(f"ERROR: {e}")

    MODELS_JSON.write_text(json.dumps(updated, indent=2) + "\n")

    if skipped:
        log("")
        log("Skipped models without a local source or configured GGUF repo:")
        for key, err in skipped:
            log(f"- {key}: {err}")

    if failures:
        log("")
        log("Some models failed:")
        for key, err in failures:
            log(f"- {key}: {err}")
        return 1

    log("")
    log("All requested models are prepared.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
