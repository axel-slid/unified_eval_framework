#!/usr/bin/env python
"""
run_benchmark.py — entry point.

Usage:
    python run_benchmark.py
    python run_benchmark.py --config benchmark_config.yaml
    python run_benchmark.py --config benchmark_config.yaml --test-set test_sets/sample.json
    python run_benchmark.py --models smolvlm
    python run_benchmark.py --help

Adding a new model:
    1. Create models/yourmodel.py  (inherit BaseVLMModel, implement load + run)
    2. Add the class to MODEL_REGISTRY in models/__init__.py
    3. Add an entry to benchmark_config.yaml under `models:`
    Done — no changes needed in this file.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import statistics
from datetime import datetime
from pathlib import Path

from config import BenchmarkConfig, load_config
from judge import judge
from models import MODEL_REGISTRY


# ── Helpers ───────────────────────────────────────────────────────────────────

def _score_color(score: int | None) -> str:
    if score is None: return "#555"
    if score >= 80: return "#3d9"
    if score >= 60: return "#8bc34a"
    if score >= 40: return "#fa0"
    return "#e54"

def _img_to_b64(image_path: str) -> str | None:
    try:
        ext = Path(image_path).suffix.lower().strip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


# ── Core runner ───────────────────────────────────────────────────────────────

def run_benchmark(
    cfg: BenchmarkConfig,
    test_set: list[dict],
    model_filter: list[str] | None = None,
) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    models_to_run = [
        m for m in cfg.enabled_models
        if model_filter is None or m.key in model_filter
    ]

    if not models_to_run:
        print("No models selected. Check --models flag and enabled: true in config.")
        return

    all_results: dict[str, list[dict]] = {}

    for mcfg in models_to_run:
        cls = MODEL_REGISTRY.get(mcfg.cls_name)
        if cls is None:
            print(f"[SKIP] Unknown class '{mcfg.cls_name}' for model '{mcfg.key}'. "
                  f"Register it in models/__init__.py")
            continue

        model = cls(mcfg)
        print(f"\n{'='*60}\nRunning: {model.name}\n{'='*60}")
        model.load()

        model_results = []
        for item in test_set:
            item_id = item["id"]
            image_path = item["image"]
            question = item["question"]
            rubric = item["rubric"]

            print(f"  [{item_id}] {question[:65]}...")

            if not os.path.exists(image_path):
                print(f"    WARNING: image not found at {image_path}, skipping")
                continue

            result = model.run(image_path, question)

            if result.error:
                print(f"    INFERENCE ERROR: {result.error}")
                judge_result = None
            else:
                print(f"    Response ({result.latency_ms:.0f}ms): {result.response[:80]}...")
                judge_result = judge(
                    question,
                    rubric,
                    result.response,
                    cfg.judge,
                    reference_answer=item.get("reference_answer", ""),
                    image_path=image_path,
                )
                if judge_result.error:
                    print(f"    JUDGE ERROR: {judge_result.error}")
                else:
                    print(f"    Score: {judge_result.score}/100 — {judge_result.reason}")

            model_results.append({
                "id": item_id,
                "image": image_path,
                "question": question,
                "rubric": rubric,
                "response": result.response,
                "latency_ms": round(result.latency_ms, 1),
                "inference_error": result.error,
                "score": judge_result.score if judge_result and not judge_result.error else None,
                "judge_reason": judge_result.reason if judge_result and not judge_result.error else None,
                "judge_error": judge_result.error if judge_result else None,
                "model_name": model.name,
                "model_path": mcfg.model_path,
            })

        model.unload()
        all_results[mcfg.key] = model_results

    # Save raw JSON
    json_path = cfg.output_dir / f"results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results → {json_path}")

    print_summary(all_results)

    html_path = cfg.output_dir / f"report_{timestamp}.html"
    save_html_report(all_results, html_path, timestamp)
    print(f"HTML report  → {html_path}")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(all_results: dict[str, list[dict]]) -> None:
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"{'Model':<28} {'Avg Score':>10} {'Avg Latency':>14} {'N':>5}")
    print("-" * 62)
    for key, results in all_results.items():
        scores = [r["score"] for r in results if r["score"] is not None]
        latencies = [r["latency_ms"] for r in results if r["latency_ms"] > 0]
        name = results[0]["model_name"] if results else key
        avg_score = f"{statistics.mean(scores):.1f}/100" if scores else "—"
        avg_lat = f"{statistics.mean(latencies):.0f}ms" if latencies else "—"
        print(f"{name:<28} {avg_score:>10} {avg_lat:>14} {len(scores):>5}")


# ── HTML report ───────────────────────────────────────────────────────────────

def save_html_report(all_results: dict, path: Path, timestamp: str) -> None:
    model_keys = list(all_results.keys())

    def model_display(key: str) -> str:
        results = all_results.get(key, [])
        return results[0]["model_name"] if results else key

    # ── Summary table ─────────────────────────────────────────────────────────
    summary_rows = ""
    for k in model_keys:
        results = all_results[k]
        scores = [r["score"] for r in results if r["score"] is not None]
        latencies = [r["latency_ms"] for r in results if r["latency_ms"] > 0]
        avg_score = statistics.mean(scores) if scores else None
        avg_lat = f"{statistics.mean(latencies):.0f}ms" if latencies else "—"
        path_str = results[0]["model_path"] if results else ""
        color = _score_color(int(avg_score) if avg_score else None)
        score_display = f"{avg_score:.1f} / 100" if avg_score is not None else "—"
        summary_rows += (
            f"<tr>"
            f"<td>{model_display(k)}</td>"
            f"<td><span style='color:{color};font-weight:600'>{score_display}</span></td>"
            f"<td>{avg_lat}</td>"
            f"<td>{len(scores)}</td>"
            f"<td><code>{path_str}</code></td>"
            f"</tr>"
        )

    # ── Per-question rows ──────────────────────────────────────────────────────
    rows_by_id: dict[str, dict] = {}
    for key, results in all_results.items():
        for r in results:
            if r["id"] not in rows_by_id:
                rows_by_id[r["id"]] = {
                    "id": r["id"],
                    "image": r["image"],
                    "question": r["question"],
                    "models": {},
                }
            rows_by_id[r["id"]]["models"][key] = r

    model_headers = "".join(f"<th>{model_display(k)}</th>" for k in model_keys)

    detail_rows = ""
    for row in rows_by_id.values():
        # Embed image as base64
        img_src = _img_to_b64(row["image"])
        img_html = (
            f'<img src="{img_src}" style="width:120px;height:90px;object-fit:cover;'
            f'border-radius:4px;border:1px solid #2a2a2a;display:block;margin-bottom:6px">'
            if img_src else ""
        )

        detail_rows += (
            f"<tr>"
            f"<td style='min-width:160px'>"
            f"{img_html}"
            f"<b style='font-size:11px'>{row['id']}</b><br>"
            f"<small style='color:#888'>{row['question']}</small>"
            f"</td>"
        )

        for k in model_keys:
            m = row["models"].get(k, {})
            score = m.get("score")
            latency = m.get("latency_ms", "—")
            response = (m.get("response") or "")[:140]
            reason = m.get("judge_reason") or m.get("inference_error") or m.get("judge_error") or ""
            color = _score_color(score)
            score_display = f"{score}/100" if score is not None else "—"

            # Score bar
            bar_width = score if score is not None else 0
            bar_html = (
                f'<div style="background:#1a1a1a;border-radius:2px;height:3px;margin:4px 0 6px">'
                f'<div style="width:{bar_width}%;height:100%;background:{color};border-radius:2px;transition:width .3s"></div>'
                f'</div>'
            )

            detail_rows += (
                f"<td>"
                f"<span style='color:{color};font-weight:700;font-size:13px'>{score_display}</span>"
                f" <small style='color:#555'>({latency}ms)</small>"
                f"{bar_html}"
                f"<small style='color:#aaa;line-height:1.5'>{response}{'…' if len(response)==140 else ''}</small><br>"
                f"<i style='font-size:10px;color:#555'>{reason}</i>"
                f"</td>"
            )
        detail_rows += "</tr>"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>VLM Benchmark — {timestamp}</title>
<style>
  body {{ font-family: 'Courier New', monospace; background: #0d0d0d; color: #ccc; padding: 2rem; }}
  h1 {{ color: #fff; font-size: 1.4rem; letter-spacing: 3px; text-transform: uppercase; }}
  h2 {{ color: #888; font-size: .9rem; letter-spacing: 2px; text-transform: uppercase;
        border-bottom: 1px solid #222; padding-bottom: .4rem; margin-top: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; font-size: .85rem; }}
  th {{ background: #161616; color: #aaa; padding: 7px 12px; text-align: left;
        border-bottom: 1px solid #2a2a2a; }}
  td {{ padding: 10px 12px; border-bottom: 1px solid #1a1a1a; vertical-align: top; }}
  tr:hover td {{ background: #111; }}
  code {{ background: #1a1a1a; padding: 1px 5px; border-radius: 3px; font-size: .8rem; }}
  .tag {{ display:inline-block; background:#1a1a1a; color:#666;
          padding: 2px 10px; border-radius: 3px; font-size: .75rem; }}
</style>
</head>
<body>
<h1>VLM Benchmark Report</h1>
<p class="tag">{timestamp}</p>

<h2>Summary</h2>
<table>
  <tr><th>Model</th><th>Avg Score</th><th>Avg Latency</th><th>N</th><th>Path</th></tr>
  {summary_rows}
</table>

<h2>Per-Question Results</h2>
<table>
  <tr><th>Image / Question</th>{model_headers}</tr>
  {detail_rows}
</table>
</body>
</html>"""

    with open(path, "w") as f:
        f.write(html)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM benchmark runner")
    parser.add_argument("--config", default="benchmark_config.yaml")
    parser.add_argument("--test-set", default="test_sets/sample.json")
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    with open(args.test_set) as f:
        test_set = json.load(f)

    print(f"Config:   {args.config}")
    print(f"Test set: {args.test_set} ({len(test_set)} items)")
    print(f"Models:   {args.models or [m.key for m in cfg.enabled_models]}")

    run_benchmark(cfg, test_set, model_filter=args.models)