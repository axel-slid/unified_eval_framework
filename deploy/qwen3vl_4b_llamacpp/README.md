# Qwen3-VL-4B llama.cpp Deployment

## 1. Start the server

```bash
bash deploy/qwen3vl_4b_llamacpp/serve.sh
```

This will:
- Install `llama.cpp` via Homebrew if not already installed
- Install Python dependencies if missing
- Download the model GGUF files if missing
- Start an OpenAI-compatible server on `http://127.0.0.1:8080`

Leave this terminal running.

## 2. Run inference on an image

In a second terminal, pass any image path:

```bash
bash deploy/qwen3vl_4b_llamacpp/infer.sh deploy/qwen3vl_4b_llamacpp/examples/example1.jpg
```

```bash
bash deploy/qwen3vl_4b_llamacpp/infer.sh deploy/qwen3vl_4b_llamacpp/examples/example2.jpg
```

The response is printed to stdout. A log file is saved to `deploy/qwen3vl_4b_llamacpp/logs/` after each run.

To get just the model's response with no extra output:

```bash
bash deploy/qwen3vl_4b_llamacpp/deploy.sh infer /path/to/image.jpg 2>/dev/null | tail -n 1
```
