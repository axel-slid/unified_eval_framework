#!/usr/bin/env python
"""
run_benchmark_vqa.py — GPT-generated VQA benchmark.

For each image:
  1. GPT generates 5 targeted questions + its own reference answers (gold standard)
  2. Each VLM answers all 5 questions
  3. GPT judges each VLM answer against its own reference answer (0-100)
  4. GPT itself is also scored as a baseline (should score ~95-100)

This gives a much more meaningful benchmark than captioning:
  - Questions are image-specific and targeted
  - Ground truth is consistent (same judge generates and scores)
  - GPT baseline tells you the theoretical ceiling
  - Score gap from GPT = how much quality you lose vs a frontier model

Usage:
    cd benchmark
    conda activate /mnt/shared/dils/envs/Qwen3VL-env
    export OPENAI_API_KEY=sk-...

    python run_benchmark_vqa.py --test-set test_sets/captioning_100.json --models smolvlm internvl
    python run_benchmark_vqa.py --test-set test_sets/captioning_100.json --all
    python run_benchmark_vqa.py --help
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import argparse
import base64
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

import httpx

from config import BenchmarkConfig, load_config
from models import MODEL_REGISTRY

OPENAI_API = "https://api.openai.com/v1/chat/completions"


# ── OpenAI helpers ─────────────────────────────────────────────────────────────

def _encode_image(image_path: str) -> tuple[str, str]:
    ext = Path(image_path).suffix.lower().strip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return b64, mime


def _openai_call(messages: list, model: str, max_tokens: int, timeout: int,
                 response_format: dict | None = None) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    payload = {
        "model": model,
        "max_completion_tokens": max_tokens,
        "temperature": 0,
        "messages": messages,
    }
    if response_format:
        payload["response_format"] = response_format

    r = httpx.post(
        OPENAI_API,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


# ── Step 1: GPT generates 5 questions + reference answers ─────────────────────

QUESTION_GEN_PROMPT = """\
You are a vision QA expert creating a targeted evaluation for vision-language models.

Look at this image carefully and generate exactly 5 questions that test different aspects of visual understanding. Questions should be:
- Specific and answerable from the image alone
- Varied: include questions about objects, counts, colors, spatial relationships, and scene context
- Clear and unambiguous
- Appropriate difficulty — not too trivial, not requiring external knowledge

For each question, also provide the correct reference answer (1-3 sentences).

Return ONLY valid JSON, no preamble, no markdown:
{
  "questions": [
    {"id": 1, "question": "...", "reference_answer": "..."},
    {"id": 2, "question": "...", "reference_answer": "..."},
    {"id": 3, "question": "...", "reference_answer": "..."},
    {"id": 4, "question": "...", "reference_answer": "..."},
    {"id": 5, "question": "...", "reference_answer": "..."}
  ]
}
"""

def generate_questions(image_path: str, cfg) -> list[dict]:
    b64, mime = _encode_image(image_path)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": QUESTION_GEN_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]
    }]
    raw = _openai_call(messages, cfg.judge.model, 1024, cfg.judge.timeout_seconds,
                       response_format={"type": "json_object"})
    parsed = json.loads(raw)
    return parsed["questions"]


# ── Step 2: Judge a VLM answer against reference ──────────────────────────────

JUDGE_PROMPT = """\
You are an impartial judge scoring a vision-language model's answer.

Question: {question}
Reference answer (correct): {reference_answer}
Model's answer: {model_answer}

Score 0-100:
- 90-100: Correct and complete
- 70-89: Mostly correct, minor omissions
- 50-69: Partially correct, missing key details
- 20-49: Mostly wrong but some correct elements
- 0-19: Wrong, refused, or hallucinated

Return ONLY valid JSON:
{{"score": <int 0-100>, "reason": "<one sentence>"}}
"""

def judge_answer(question: str, reference_answer: str, model_answer: str,
                 image_path: str, cfg) -> dict:
    b64, mime = _encode_image(image_path)
    text = JUDGE_PROMPT.format(
        question=question,
        reference_answer=reference_answer,
        model_answer=model_answer,
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]
    }]
    raw = _openai_call(messages, cfg.judge.model, 256, cfg.judge.timeout_seconds,
                       response_format={"type": "json_object"})
    parsed = json.loads(raw)
    return {"score": int(parsed["score"]), "reason": str(parsed["reason"])}


# ── Step 3: GPT answers questions (baseline) ──────────────────────────────────

GPT_ANSWER_PROMPT = "Answer this question about the image concisely and accurately in 1-3 sentences.\n\nQuestion: {question}"

def gpt_answer(question: str, image_path: str, cfg) -> str:
    b64, mime = _encode_image(image_path)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": GPT_ANSWER_PROMPT.format(question=question)},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]
    }]
    return _openai_call(messages, cfg.judge.model, 256, cfg.judge.timeout_seconds)


# ── Core runner ───────────────────────────────────────────────────────────────

def run_vqa_benchmark(
    cfg: BenchmarkConfig,
    test_set: list[dict],
    model_filter: list[str] | None = None,
    run_gpt_baseline: bool = True,
) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    models_to_run = [
        m for m in cfg.enabled_models
        if model_filter is None or m.key in model_filter
    ]

    if not models_to_run and not run_gpt_baseline:
        print("No models selected.")
        return

    # ── Phase 1: generate questions for all images ────────────────────────────
    questions_cache_path = cfg.output_dir / f"vqa_questions_{timestamp}.json"
    print(f"\n{'='*60}")
    print(f"Phase 1: Generating 5 questions per image via {cfg.judge.model}")
    print(f"{'='*60}")

    image_questions: dict[str, list[dict]] = {}
    for i, item in enumerate(test_set):
        image_path = item["image"]
        if not os.path.exists(image_path):
            print(f"  [{item['id']}] WARNING: image not found, skipping")
            continue
        print(f"  [{item['id']}] Generating questions... ({i+1}/{len(test_set)})")
        try:
            qs = generate_questions(image_path, cfg)
            image_questions[item["id"]] = qs
            print(f"         → {len(qs)} questions generated")
        except Exception as e:
            print(f"         ERROR: {e}")
            image_questions[item["id"]] = []

    # Save questions cache
    questions_cache_path.write_text(json.dumps(image_questions, indent=2))
    print(f"\nQuestions saved to {questions_cache_path}")

    # ── Phase 2: GPT baseline answers ─────────────────────────────────────────
    all_results: dict[str, list[dict]] = {}

    if run_gpt_baseline:
        print(f"\n{'='*60}")
        print(f"Phase 2: GPT baseline ({cfg.judge.model})")
        print(f"{'='*60}")

        gpt_results = []
        for item in test_set:
            image_path = item["image"]
            item_id = item["id"]
            qs = image_questions.get(item_id, [])
            if not qs or not os.path.exists(image_path):
                continue

            q_scores = []
            for q in qs:
                try:
                    t0 = time.perf_counter()
                    answer = gpt_answer(q["question"], image_path, cfg)
                    latency_ms = (time.perf_counter() - t0) * 1000
                    judged = judge_answer(q["question"], q["reference_answer"], answer, image_path, cfg)
                    q_scores.append({
                        "question_id": q["id"],
                        "question": q["question"],
                        "reference_answer": q["reference_answer"],
                        "model_answer": answer,
                        "score": judged["score"],
                        "reason": judged["reason"],
                        "latency_ms": round(latency_ms),
                    })
                    print(f"  [{item_id}] Q{q['id']}: {judged['score']}/100")
                except Exception as e:
                    print(f"  [{item_id}] Q{q['id']} ERROR: {e}")

            if q_scores:
                avg = statistics.mean(s["score"] for s in q_scores)
                gpt_results.append({
                    "id": item_id,
                    "image": image_path,
                    "question_scores": q_scores,
                    "avg_score": round(avg, 1),
                    "latency_ms": round(statistics.mean(s["latency_ms"] for s in q_scores)),
                    "model_name": f"GPT Baseline ({cfg.judge.model})",
                    "model_path": cfg.judge.model,
                    "score": round(avg),
                })

        all_results["gpt_baseline"] = gpt_results
        gpt_avg = statistics.mean(r["avg_score"] for r in gpt_results) if gpt_results else 0
        print(f"\nGPT baseline avg: {gpt_avg:.1f}/100")

    # ── Phase 3: run each VLM ─────────────────────────────────────────────────
    for mcfg in models_to_run:
        cls = MODEL_REGISTRY.get(mcfg.cls_name)
        if cls is None:
            print(f"[SKIP] Unknown class '{mcfg.cls_name}'")
            continue

        model = cls(mcfg)
        print(f"\n{'='*60}\nPhase 3 — Model: {model.name}\n{'='*60}")
        model.load()

        model_results = []
        for item in test_set:
            image_path = item["image"]
            item_id = item["id"]
            qs = image_questions.get(item_id, [])
            if not qs or not os.path.exists(image_path):
                continue

            q_scores = []
            for q in qs:
                try:
                    result = model.run(image_path, q["question"])
                    if result.error:
                        print(f"  [{item_id}] Q{q['id']} INFERENCE ERROR: {result.error}")
                        continue

                    judged = judge_answer(
                        q["question"], q["reference_answer"],
                        result.response, image_path, cfg
                    )
                    q_scores.append({
                        "question_id": q["id"],
                        "question": q["question"],
                        "reference_answer": q["reference_answer"],
                        "model_answer": result.response,
                        "score": judged["score"],
                        "reason": judged["reason"],
                        "latency_ms": round(result.latency_ms),
                    })
                    print(f"  [{item_id}] Q{q['id']}: {judged['score']}/100 ({result.latency_ms:.0f}ms)")
                except Exception as e:
                    print(f"  [{item_id}] Q{q['id']} ERROR: {e}")

            if q_scores:
                avg = statistics.mean(s["score"] for s in q_scores)
                model_results.append({
                    "id": item_id,
                    "image": image_path,
                    "question_scores": q_scores,
                    "avg_score": round(avg, 1),
                    "latency_ms": round(statistics.mean(s["latency_ms"] for s in q_scores)),
                    "model_name": model.name,
                    "model_path": mcfg.model_path,
                    "score": round(avg),
                })

        model.unload()
        all_results[mcfg.key] = model_results

        if model_results:
            avg = statistics.mean(r["avg_score"] for r in model_results)
            print(f"\n{model.name} avg: {avg:.1f}/100")

    # ── Save + report ──────────────────────────────────────────────────────────
    json_path = cfg.output_dir / f"vqa_results_{timestamp}.json"
    json_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nRaw results → {json_path}")

    print_vqa_summary(all_results)

    html_path = cfg.output_dir / f"vqa_report_{timestamp}.html"
    save_vqa_html(all_results, image_questions, html_path, timestamp)
    print(f"HTML report  → {html_path}")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_vqa_summary(all_results: dict) -> None:
    print(f"\n{'='*65}\nVQA BENCHMARK SUMMARY\n{'='*65}")
    print(f"{'Model':<42} {'Avg Score':>10} {'vs GPT':>8} {'N':>5}")
    print("-" * 65)

    gpt_avg = None
    if "gpt_baseline" in all_results:
        scores = [r["avg_score"] for r in all_results["gpt_baseline"]]
        gpt_avg = statistics.mean(scores) if scores else None

    for key, results in all_results.items():
        if not results: continue
        scores = [r["avg_score"] for r in results]
        lats = [r["latency_ms"] for r in results]
        name = results[0]["model_name"]
        avg = statistics.mean(scores) if scores else 0
        gap = f"{avg - gpt_avg:+.1f}" if gpt_avg and key != "gpt_baseline" else "—"
        print(f"{name:<42} {avg:>9.1f} {gap:>8} {len(scores):>5}")


# ── HTML report ───────────────────────────────────────────────────────────────

def _score_color(score) -> str:
    if score is None: return "#555"
    s = int(score)
    if s >= 90: return "#3d9"
    if s >= 70: return "#8bc34a"
    if s >= 50: return "#fa0"
    return "#e54"

def _img_b64(image_path: str) -> str | None:
    try:
        ext = Path(image_path).suffix.lower().strip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")
        with open(image_path, "rb") as f:
            return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"
    except: return None


def save_vqa_html(all_results: dict, image_questions: dict, path: Path, timestamp: str) -> None:
    model_keys = list(all_results.keys())

    def model_display(key):
        r = all_results.get(key, [])
        return r[0]["model_name"] if r else key

    # Summary table
    gpt_avg = None
    if "gpt_baseline" in all_results:
        s = [r["avg_score"] for r in all_results["gpt_baseline"]]
        gpt_avg = statistics.mean(s) if s else None

    summary_rows = ""
    for k in model_keys:
        results = all_results[k]
        if not results: continue
        scores = [r["avg_score"] for r in results]
        lats = [r["latency_ms"] for r in results]
        avg = statistics.mean(scores)
        avg_lat = f"{statistics.mean(lats):.0f}ms"
        gap = f"{avg - gpt_avg:+.1f}" if gpt_avg and k != "gpt_baseline" else "baseline"
        color = _score_color(avg)
        summary_rows += (
            f"<tr>"
            f"<td>{model_display(k)}</td>"
            f"<td><span style='color:{color};font-weight:700'>{avg:.1f}/100</span></td>"
            f"<td style='color:{'#3d9' if '+' in str(gap) else '#e54' if '-' in str(gap) else '#888'}'>{gap}</td>"
            f"<td>{avg_lat}</td>"
            f"<td>{len(scores)}</td>"
            f"</tr>"
        )

    # Per-image rows
    all_ids = []
    for results in all_results.values():
        for r in results:
            if r["id"] not in all_ids:
                all_ids.append(r["id"])

    # Build lookup
    lookup: dict[str, dict[str, dict]] = {}
    for key, results in all_results.items():
        for r in results:
            if r["id"] not in lookup:
                lookup[r["id"]] = {}
            lookup[r["id"]][key] = r

    model_headers = "".join(f"<th>{model_display(k)}</th>" for k in model_keys)

    detail_rows = ""
    for img_id in all_ids:
        row_data = lookup.get(img_id, {})
        any_result = next(iter(row_data.values()), {})
        image_path = any_result.get("image", "")
        img_src = _img_b64(image_path)
        img_html = f'<img src="{img_src}" style="width:140px;height:105px;object-fit:cover;border-radius:4px;border:1px solid #2a2a2a;display:block;margin-bottom:6px">' if img_src else ""

        # Get questions for this image
        qs = image_questions.get(img_id, [])
        q_list = "".join(f"<li style='margin:2px 0;color:#777;font-size:10px'>{q['question']}</li>" for q in qs)
        q_html = f"<ol style='padding-left:14px;margin:4px 0'>{q_list}</ol>" if q_list else ""

        detail_rows += (
            f"<tr>"
            f"<td style='min-width:180px'>{img_html}"
            f"<b style='font-size:11px'>#{img_id}</b>{q_html}</td>"
        )

        for k in model_keys:
            r = row_data.get(k, {})
            avg = r.get("avg_score")
            lat = r.get("latency_ms", "—")
            color = _score_color(avg)
            score_display = f"{avg}/100" if avg is not None else "—"

            # Per-question breakdown
            q_breakdown = ""
            for qs_item in r.get("question_scores", []):
                qc = _score_color(qs_item["score"])
                q_breakdown += (
                    f"<div style='margin:3px 0;padding:4px 6px;background:#151515;border-radius:3px;border-left:2px solid {qc}'>"
                    f"<span style='color:{qc};font-weight:600;font-size:11px'>Q{qs_item['question_id']}: {qs_item['score']}/100</span>"
                    f"<br><small style='color:#888;font-size:10px'>{qs_item['reason']}</small>"
                    f"</div>"
                )

            bar_w = int(avg) if avg else 0
            detail_rows += (
                f"<td>"
                f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px'>"
                f"<span style='color:{color};font-weight:700;font-size:14px'>{score_display}</span>"
                f"<small style='color:#555'>{lat}ms</small></div>"
                f"<div style='background:#1a1a1a;border-radius:2px;height:3px;margin-bottom:8px'>"
                f"<div style='width:{bar_w}%;height:100%;background:{color};border-radius:2px'></div></div>"
                f"{q_breakdown}"
                f"</td>"
            )
        detail_rows += "</tr>"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>VQA Benchmark — {timestamp}</title>
<style>
  body {{ font-family: 'Courier New', monospace; background: #0d0d0d; color: #ccc; padding: 2rem; }}
  h1 {{ color: #fff; font-size: 1.4rem; letter-spacing: 3px; text-transform: uppercase; }}
  h2 {{ color: #888; font-size: .9rem; letter-spacing: 2px; text-transform: uppercase;
        border-bottom: 1px solid #222; padding-bottom: .4rem; margin-top: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; font-size: .82rem; }}
  th {{ background: #161616; color: #aaa; padding: 7px 12px; text-align: left;
        border-bottom: 1px solid #2a2a2a; }}
  td {{ padding: 10px 12px; border-bottom: 1px solid #1a1a1a; vertical-align: top; }}
  tr:hover td {{ background: #111; }}
  code {{ background: #1a1a1a; padding: 1px 5px; border-radius: 3px; font-size: .8rem; }}
  .tag {{ display:inline-block; background:#1a1a1a; color:#666;
          padding: 2px 10px; border-radius: 3px; font-size: .75rem; }}
  .note {{ font-size: 11px; color: #555; margin-top: 6px; font-style: italic; }}
</style>
</head>
<body>
<h1>VQA Benchmark Report</h1>
<p class="tag">{timestamp}</p>
<p class="note">5 GPT-generated questions per image · judged by {list(all_results.values())[0][0]["model_path"] if all_results else ""} · GPT answers as baseline</p>

<h2>Summary</h2>
<table>
  <tr><th>Model</th><th>Avg Score</th><th>vs GPT Baseline</th><th>Avg Latency</th><th>N images</th></tr>
  {summary_rows}
</table>

<h2>Per-Image Results</h2>
<table>
  <tr><th>Image / Questions</th>{model_headers}</tr>
  {detail_rows}
</table>
</body>
</html>"""

    path.write_text(html)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQA benchmark — GPT generates questions and judges")
    parser.add_argument("--config", default="benchmark_config.yaml")
    parser.add_argument("--test-set", default="test_sets/captioning_100.json")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model keys to run (from benchmark_config.yaml)")
    parser.add_argument("--all", action="store_true",
                        help="Run all enabled models from config")
    parser.add_argument("--no-gpt-baseline", action="store_true",
                        help="Skip GPT baseline (saves API cost)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    with open(args.test_set) as f:
        test_set = json.load(f)

    if args.all:
        model_filter = None
    elif args.models:
        model_filter = args.models
    else:
        model_filter = None

    print(f"Config:      {args.config}")
    print(f"Test set:    {args.test_set} ({len(test_set)} images)")
    print(f"Judge model: {cfg.judge.model}")
    print(f"Models:      {model_filter or [m.key for m in cfg.enabled_models]}")
    print(f"GPT baseline: {'no' if args.no_gpt_baseline else 'yes'}")

    run_vqa_benchmark(
        cfg, test_set,
        model_filter=model_filter,
        run_gpt_baseline=not args.no_gpt_baseline,
    )
