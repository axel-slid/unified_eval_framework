"""
config.py — loads and validates benchmark_config.yaml.
All other modules import from here; nothing reads the YAML directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class JudgeConfig:
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 256
    timeout_seconds: int = 30


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    do_sample: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"max_new_tokens": self.max_new_tokens, "do_sample": self.do_sample}


@dataclass
class ModelConfig:
    key: str                    # the dict key from YAML, e.g. "smolvlm"
    enabled: bool
    cls_name: str               # "class" field in YAML → maps to a Python class name
    model_path: str
    dtype: str = "float16"
    generation: GenerationConfig = field(default_factory=GenerationConfig)


@dataclass
class BenchmarkConfig:
    output_dir: Path
    judge: JudgeConfig
    generation_defaults: GenerationConfig
    models: list[ModelConfig]   # only enabled models

    @property
    def enabled_models(self) -> list[ModelConfig]:
        return [m for m in self.models if m.enabled]


# ── Loader ────────────────────────────────────────────────────────────────────

def load_config(path: str | Path = "benchmark_config.yaml") -> BenchmarkConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    gen_defaults = _parse_generation(raw.get("generation_defaults", {}))

    models = []
    for key, mcfg in (raw.get("models") or {}).items():
        if mcfg is None:
            continue
        # merge generation_defaults with per-model overrides
        merged_gen = {**gen_defaults.to_dict(), **mcfg.get("generation", {})}
        models.append(ModelConfig(
            key=key,
            enabled=mcfg.get("enabled", True),
            cls_name=mcfg["class"],
            model_path=mcfg["model_path"],
            dtype=mcfg.get("dtype", "float16"),
            generation=_parse_generation(merged_gen),
        ))

    return BenchmarkConfig(
        output_dir=Path(raw.get("output_dir", "results")),
        judge=_parse_judge(raw.get("judge", {})),
        generation_defaults=gen_defaults,
        models=models,
    )


def _parse_judge(d: dict) -> JudgeConfig:
    return JudgeConfig(
        model=d.get("model", "claude-sonnet-4-20250514"),
        max_tokens=d.get("max_tokens", 256),
        timeout_seconds=d.get("timeout_seconds", 30),
    )


def _parse_generation(d: dict) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=d.get("max_new_tokens", 256),
        do_sample=d.get("do_sample", False),
    )
