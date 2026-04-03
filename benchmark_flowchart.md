# Benchmark Pipeline Flowchart

```mermaid
flowchart TD
    A([Start: run_benchmark.py]) --> B[Parse CLI args\n--config, --test-set, --models]
    B --> C[Load benchmark_config.yaml\nmodels · judge · generation defaults]
    B --> D[Load test set JSON\nid · image · question · rubric]
    C & D --> E[Filter enabled models\noptionally by --models flag]

    E --> F{Models remaining?}
    F -- No --> Z([Exit — nothing to run])
    F -- Yes --> G[Pop next ModelConfig]

    G --> H[Look up class in MODEL_REGISTRY\nSmolVLMModel / InternVLModel / Qwen3VLModel]
    H --> I[model.load\nload weights into GPU memory]
    I --> J{Test items remaining?}

    J -- Yes --> K[Next item\nimage · question · rubric]
    K --> L{Image file exists?}
    L -- No --> M[Log WARNING\nskip item] --> J
    L -- Yes --> N[model.run image question\nInferenceResult: response · latency_ms · error]

    N --> O{Inference error?}
    O -- Yes --> P[Record error\nscore = null] --> J
    O -- No --> Q[judge\nquestion · rubric · response · image\nvia OpenAI GPT API]

    Q --> R{Judge API OK?}
    R -- Error --> S[Record judge error\nscore = null] --> J
    R -- OK --> T[JudgeResult\nscore 0–100 · reason]
    T --> U[Record result\nid · response · latency_ms · score · reason] --> J

    J -- Done --> V[model.unload\nfree GPU memory]
    V --> F

    F -- All done --> W[Save results_timestamp.json\nraw per-item data for all models]
    W --> X[Print summary table\nmodel · avg score · avg latency · N]
    X --> Y[Save report_timestamp.html\nsummary table + per-question detail with score bars]
    Y --> Z2([End])
```

## Key Components

| Component | File | Role |
|---|---|---|
| Entry point | `run_benchmark.py` | Orchestrates the full pipeline |
| Config | `benchmark_config.yaml` | Models, judge settings, generation params |
| Test set | `test_sets/sample.json` | List of `{id, image, question, rubric}` items |
| Model base | `models/base.py` | Abstract `BaseVLMModel` — `load()`, `run()`, `unload()` |
| Models | `models/{smolvlm,internvl,qwen3vl}.py` | Concrete VLM runners |
| Registry | `models/__init__.py` | Maps YAML `class:` string → Python class |
| Judge | `judge.py` | Calls OpenAI GPT to score responses 0–100 |
| Config parser | `config.py` | Parses YAML into typed dataclasses |

## Score Scale (Judge)

| Range | Meaning |
|---|---|
| 0–20 | Completely wrong / hallucination / refusal |
| 21–40 | Mostly wrong, minor correct elements |
| 41–60 | Partially correct, missing key details |
| 61–80 | Mostly correct, minor issues |
| 81–100 | Fully correct and complete |
