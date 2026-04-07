#!/usr/bin/env python3
"""
generate_binary_report.py — Merge latest per-model env_monitoring_binary JSONs
into a single combined HTML report without rerunning inference.

Usage:
    cd benchmark
    python generate_binary_report.py
    python generate_binary_report.py --out results/my_report.html
"""

from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime
from pathlib import Path

# Import save_html from the runner
import sys
sys.path.insert(0, str(Path(__file__).parent))
from run_benchmark_env_monitoring_binary import save_html


RESULTS_DIR = Path(__file__).parent / "results"


def latest_per_model(results_dir: Path) -> dict[str, dict]:
    """Return the most recent result entry per model key."""
    files = sorted(results_dir.glob("env_monitoring_binary_*.json"))
    # last-write-wins per model key
    per_model: dict[str, dict] = {}
    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        for key, model_data in data.get("models", {}).items():
            per_model[key] = model_data
    return per_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None, help="Output HTML path")
    args = parser.parse_args()

    all_results = latest_per_model(RESULTS_DIR)
    if not all_results:
        print("No result JSONs found in", RESULTS_DIR)
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"env_monitoring_binary_combined_{timestamp}.html"

    print(f"Models found: {sorted(all_results.keys())}")
    for key, data in all_results.items():
        acc = data["metrics"]["accuracy"]
        per = {ct: f"{m['accuracy']:.0%}" for ct, m in data["metrics"]["per_change_type"].items()}
        print(f"  {key}: overall={acc:.0%}  {per}")

    save_html(all_results, out_path, timestamp)
    print(f"\nReport → {out_path}")


if __name__ == "__main__":
    main()
