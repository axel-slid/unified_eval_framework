# Unified Evaluation Framework

A modular and extensible benchmarking framework for evaluating Vision-Language Models (VLMs) across standardized test sets.

This framework enables reproducible evaluation, easy model integration, configurable benchmarking, and structured result reporting.

---

## Overview

The Unified Evaluation Framework provides:

- A config-driven benchmarking pipeline
- A plug-and-play model interface
- A standardized judging mechanism
- Support for multi-model comparison
- Structured outputs for downstream analysis

---

## Repository Structure

| Directory / File | Description |
|-----------------|------------|
| benchmark/ | Core benchmarking pipeline |
| benchmark/run_benchmark.py | Main entry point |
| benchmark/config.py | Config parsing and validation |
| benchmark/judge.py | Evaluation logic |
| benchmark/benchmark_config.yaml | Default configuration |
| benchmark/test_sets/ | Example datasets |
| inferences/ | Model-specific inference implementations |
| scripts/ | Model download utilities |
| README.md | Project documentation |

---

## Installation

### Requirements

- Python 3.9+
- pip or conda environment

### Setup

```bash
git clone <repo-url>
cd unified_eval_framework
pip install -r requirements.txt
```

---

## Quick Start

```bash
python benchmark/run_benchmark.py
```

---

## Configuration

All benchmark behavior is controlled via a YAML config file.

---

## Test Sets

Test sets are JSON files containing evaluation samples.

---

## Output

Results are saved in the configured output directory.

---

## Model Integration

Adding a new model requires no changes to the core pipeline.

---

## License

Add your license here.
