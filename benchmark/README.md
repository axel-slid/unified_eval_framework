# VLM Benchmark

## Structure
```
benchmark/
├── benchmark_config.yaml     ← all settings live here
├── config.py                 ← loads + validates the YAML into dataclasses
├── run_benchmark.py          ← entry point
├── judge.py                  ← Claude-as-judge scorer
├── models/
│   ├── __init__.py           ← MODEL_REGISTRY (class name → class)
│   ├── base.py               ← BaseVLMModel interface
│   ├── smolvlm.py
│   └── internvl.py
├── test_sets/
│   └── sample.json           ← (image, question, rubric) tuples
└── results/                  ← auto-created, JSON + HTML report per run
```

## Usage
```bash
conda activate SmolVLM-env
export ANTHROPIC_API_KEY=sk-ant-...

# run all enabled models
python run_benchmark.py

# custom config or test set
python run_benchmark.py --config benchmark_config.yaml --test-set test_sets/sample.json

# only run specific models (must match keys in YAML)
python run_benchmark.py --models smolvlm
```

## Adding a new model — 3 steps

**1. Create `models/yourmodel.py`**
```python
from models.base import BaseVLMModel, InferenceResult
from config import ModelConfig

class YourModel(BaseVLMModel):
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.name = "YourModel"

    def load(self): ...
    def run(self, image_path, question) -> InferenceResult: ...
    def unload(self): ...  # optional
```

**2. Register in `models/__init__.py`**
```python
from .yourmodel import YourModel

MODEL_REGISTRY = {
    ...
    "YourModel": YourModel,
}
```

**3. Add to `benchmark_config.yaml`**
```yaml
models:
  yourmodel:
    enabled: true
    class: YourModel
    model_path: org/repo-or-local-path
    dtype: float16
    generation:
      max_new_tokens: 256
```

That's it. No changes to `run_benchmark.py`.
