# Qwen3-VL-4B llama.cpp Deployment

This folder packages the repo's `Qwen3-VL-4B` GGUF deployment flow into one place.

It uses:
- `Qwen3-VL-4B-Instruct-Q8_0.gguf`
- `mmproj-Qwen3-VL-4B-Instruct-F16.gguf`
- `llama-server` if installed locally
- Docker as a fallback

Everything needed to run from the repo is inside this folder:
- [deploy.sh](/Users/alexdils/Downloads/berkeley/clubs/ml@b/projects/logitech/unified_eval_framework/deploy/qwen3vl_4b_llamacpp/deploy.sh)
- [run_qwen3vl_4b_llamacpp.py](/Users/alexdils/Downloads/berkeley/clubs/ml@b/projects/logitech/unified_eval_framework/deploy/qwen3vl_4b_llamacpp/run_qwen3vl_4b_llamacpp.py)
- [serve.sh](/Users/alexdils/Downloads/berkeley/clubs/ml@b/projects/logitech/unified_eval_framework/deploy/qwen3vl_4b_llamacpp/serve.sh)
- [infer.sh](/Users/alexdils/Downloads/berkeley/clubs/ml@b/projects/logitech/unified_eval_framework/deploy/qwen3vl_4b_llamacpp/infer.sh)

## One Command

From the repo root:

```bash
bash deploy/qwen3vl_4b_llamacpp/serve.sh
```

That will:
- install lightweight Python dependencies if needed
- download the GGUF + mmproj if missing
- start a llama.cpp OpenAI-compatible server on port `8080`

## One-Shot Test

```bash
bash deploy/qwen3vl_4b_llamacpp/infer.sh meeting_participation_dataset/OriginalFilled.jpg
```

Or directly:

```bash
bash deploy/qwen3vl_4b_llamacpp/deploy.sh serve
```

## Direct Python Usage

Start a persistent server:

```bash
python deploy/qwen3vl_4b_llamacpp/run_qwen3vl_4b_llamacpp.py serve --backend local
```

Run one request against an already-running server:

```bash
python deploy/qwen3vl_4b_llamacpp/run_qwen3vl_4b_llamacpp.py infer \
  --image meeting_participation_dataset/GPTgenerated4.png \
  --prompt "Describe this image." \
  --port 8080
```

## Environment Variables

These wrappers honor:

```bash
PORT=8080
THREADS=4
CTX_SIZE=2048
BACKEND=auto
PROMPT="Describe this image."
```

`BACKEND` may be:
- `auto`
- `local`
- `docker`

## Notes

- This is llama.cpp-style 8-bit quantization (`Q8_0` GGUF), not Hugging Face bitsandbytes int8.
- If `llama-server` exists in `PATH`, the wrapper prefers it.
- Otherwise it falls back to Docker with a llama.cpp server image.
- The server exposes:

```text
http://127.0.0.1:8080/health
http://127.0.0.1:8080/v1/chat/completions
```

## Health Check

```bash
curl http://127.0.0.1:8080/health
```

## Example curl Request

```bash
python - <<'PY'
import base64, json
from pathlib import Path
import requests

img = Path("meeting_participation_dataset/OriginalFilled.jpg").read_bytes()
b64 = base64.b64encode(img).decode()
payload = {
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": "Describe this image."},
        ],
    }],
    "max_tokens": 256,
    "stream": False,
}
r = requests.post("http://127.0.0.1:8080/v1/chat/completions", json=payload, timeout=300)
print(r.json()["choices"][0]["message"]["content"])
PY
```
