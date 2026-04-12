#!/usr/bin/env python
"""
run_benchmark_env_monitoring_binary.py — Single binary yes/no question per category.

For each image the model is asked exactly one question:

  table      — "Is there anything left on the table surface?"   yes→messy  no→clean
  blinds     — "Are ALL of the blinds fully up?"                yes→clean  no→messy
  chairs     — "Are all the chairs pushed in around the table?" yes→clean  no→messy
  whiteboard — "Is there any writing or marks on the whiteboard?" yes→messy no→clean

Ground truth is taken directly from the dataset label (clean / messy).

Usage:
    cd benchmark
    conda activate /mnt/shared/dils/envs/Qwen3VL-env

    python run_benchmark_env_monitoring_binary.py --all
    python run_benchmark_env_monitoring_binary.py --models qwen3vl_4b smolvlm
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import argparse
import base64
import csv
import json
import re
from datetime import datetime
from pathlib import Path

from config import BenchmarkConfig, load_config
from models import MODEL_REGISTRY

DATASET_CSV = (
    Path(__file__).parent.parent
    / "environment_monitoring_dataset"
    / "unified_annotations.csv"
)

# ── Binary questions ──────────────────────────────────────────────────────────
# yes_means: which label "yes" corresponds to ("clean" or "messy")

QUESTIONS: dict[str, dict] = {
    "table": {
        "question": (
            "Look at the table surface in this image.\n"
            "Think step by step: describe what you see on the table surface, then decide.\n"
            "Is there anything left on it — laptops, cups, bottles, papers, bags, or any personal items?\n"
            "End your response with a line containing only: yes or no"
        ),
        "yes_means": "messy",
    },
    "blinds": {
        "question": (
            "Look at the windows in this image.\n"
            "Think step by step: describe the state of the blinds or window coverings, then decide.\n"
            "Is natural light coming into the room — are the blinds open or not drawn?\n"
            "End your response with a line containing only: yes or no"
        ),
        "yes_means": "clean",
    },
    "chairs": {
        "question": "yes or no: are the chairs all tucked into the table?",
        "yes_means": "clean",
    },
    "whiteboard": {
        "question": (
            "Look at the whiteboard in this image.\n"
            "Does the whiteboard have any markings on it?\n"
            "End your response with a line containing only: yes or no"
        ),
        "yes_means": "messy",
    },
}


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_yes_no(response: str) -> bool | None:
    """Check the last non-empty line first (model was asked to end with yes/no),
    then fall back to scanning the full text."""
    text = response.strip().lower()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    # Check last line first
    if lines:
        last = lines[-1].rstrip(".,!")
        if last == "yes":
            return True
        if last == "no":
            return False
    # Fallback: scan full text
    if re.search(r"\byes\b", text):
        return True
    if re.search(r"\bno\b", text):
        return False
    return None


def predict_label(answer: bool | None, yes_means: str) -> str | None:
    """Convert parsed yes/no to 'clean' or 'messy'. None = parse failure."""
    if answer is None:
        return None
    if answer:
        return yes_means
    return "clean" if yes_means == "messy" else "messy"


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    total = correct = parse_errors = 0
    by_type: dict[str, dict] = {}
    latencies: list[float] = []
    tps_vals: list[float] = []

    for r in results:
        pred = r.get("predicted_label")
        gt = r["label"]
        ct = r["change_type"]

        if pred is None:
            parse_errors += 1

        total += 1
        if pred == gt:
            correct += 1

        by_type.setdefault(ct, {"total": 0, "correct": 0, "parse_errors": 0})
        by_type[ct]["total"] += 1
        if pred == gt:
            by_type[ct]["correct"] += 1
        if pred is None:
            by_type[ct]["parse_errors"] += 1

        lat = r.get("latency_ms", 0)
        if lat and lat > 0:
            latencies.append(lat)
            n_tok = len(r.get("raw_response", "").split())
            if n_tok > 0:
                tps_vals.append(n_tok / (lat / 1000))

    acc = correct / total if total else 0.0
    per_type = {
        ct: {
            "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0.0,
            "correct": v["correct"],
            "total": v["total"],
            "parse_errors": v["parse_errors"],
        }
        for ct, v in by_type.items()
    }

    avg_latency_ms = round(sum(latencies) / len(latencies)) if latencies else 0
    avg_tps = round(sum(tps_vals) / len(tps_vals), 1) if tps_vals else 0.0

    return {
        "accuracy": round(acc, 4),
        "n_images": total,
        "parse_errors": parse_errors,
        "avg_latency_ms": avg_latency_ms,
        "avg_tps": avg_tps,
        "per_change_type": per_type,
    }


# ── Core runner ───────────────────────────────────────────────────────────────

def run_binary_benchmark(
    cfg: BenchmarkConfig,
    model_filter: list[str] | None = None,
) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(DATASET_CSV) as f:
        samples = list(csv.DictReader(f))

    print(f"\nDataset: {DATASET_CSV}")
    print(f"Samples: {len(samples)}")
    by_type: dict[str, int] = {}
    for s in samples:
        by_type[s["change_type"]] = by_type.get(s["change_type"], 0) + 1
    for ct, n in sorted(by_type.items()):
        print(f"  {ct}: {n} images")

    models_to_run = [
        m for m in cfg.enabled_models
        if model_filter is None or m.key in model_filter
    ]
    if not models_to_run:
        print("No models selected.")
        return

    all_results: dict[str, dict] = {}

    for mcfg in models_to_run:
        cls = MODEL_REGISTRY.get(mcfg.cls_name)
        if cls is None:
            print(f"[SKIP] Unknown class '{mcfg.cls_name}'")
            continue

        model = cls(mcfg)
        print(f"\n{'='*60}\nModel: {model.name}\n{'='*60}")
        model.load()

        model_results: list[dict] = []

        for sample in samples:
            image_path = sample["image_path"]
            sid = sample["image_filename"].replace(".png", "")[:35]
            ct = sample["change_type"]
            gt = sample["label"]

            if not Path(image_path).exists():
                print(f"  [{sid}] WARNING: image not found")
                model_results.append({
                    **sample,
                    "predicted_label": None,
                    "raw_response": "",
                    "answer": None,
                    "error": "image not found",
                    "latency_ms": 0,
                    "model_name": model.name,
                })
                continue

            q = QUESTIONS[ct]

            # Run with up to 2 retries on parse failure
            r = None
            answer = None
            total_latency = 0
            for attempt in range(3):
                r = model.run(image_path, q["question"])
                if r.error:
                    break
                total_latency += r.latency_ms
                answer = parse_yes_no(r.response)
                if answer is not None:
                    break
                if attempt < 2:
                    print(f"  [{sid}] parse fail (attempt {attempt+1}), retrying...")

            if r.error:
                print(f"  [{sid}] ERROR: {r.error}")
                model_results.append({
                    **sample,
                    "predicted_label": None,
                    "raw_response": "",
                    "answer": None,
                    "error": r.error,
                    "latency_ms": 0,
                    "model_name": model.name,
                })
                continue

            predicted = predict_label(answer, q["yes_means"])
            correct = predicted == gt
            mark = "✓" if correct else "✗"
            ans_str = {True: "yes", False: "no", None: "?"}[answer]

            print(
                f"  [{sid}] {mark} [{ct}] "
                f"ans={ans_str} → {predicted} (gt={gt}) "
                f"{round(total_latency)}ms"
            )

            model_results.append({
                **sample,
                "predicted_label": predicted,
                "raw_response": r.response,
                "answer": answer,
                "error": None,
                "latency_ms": round(total_latency),
                "model_name": model.name,
                "model_path": mcfg.model_path,
            })

        model.unload()

        metrics = compute_metrics(model_results)
        all_results[mcfg.key] = {
            "model_name": model.name,
            "model_path": mcfg.model_path,
            "metrics": metrics,
            "results": model_results,
        }

        print(f"\n  accuracy={metrics['accuracy']:.1%}  parse_errors={metrics['parse_errors']}")
        for ct, m in sorted(metrics["per_change_type"].items()):
            print(f"  [{ct}] {m['accuracy']:.1%} ({m['correct']}/{m['total']})")

        # Incremental save
        checkpoint = {"timestamp": timestamp, "dataset": str(DATASET_CSV), "models": all_results}
        ckpt_path = cfg.output_dir / f"env_monitoring_binary_{timestamp}.json"
        ckpt_path.write_text(json.dumps(checkpoint, indent=2))

    out = {"timestamp": timestamp, "dataset": str(DATASET_CSV), "models": all_results}
    json_path = cfg.output_dir / f"env_monitoring_binary_{timestamp}.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults → {json_path}")

    html_path = cfg.output_dir / f"env_monitoring_binary_report_{timestamp}.html"
    save_html(all_results, html_path, timestamp)
    print(f"Report  → {html_path}")


# ── HTML report ───────────────────────────────────────────────────────────────

def _img_b64(image_path: str) -> str | None:
    try:
        ext = Path(image_path).suffix.lower().strip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")
        with open(image_path, "rb") as f:
            return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"
    except Exception:
        return None


def _heat_bg(acc: float) -> str:
    """Green→yellow→red cell background for heatmap (light mode)."""
    if acc >= 0.9: return "#bbf7d0"
    if acc >= 0.8: return "#d9f99d"
    if acc >= 0.7: return "#fef08a"
    if acc >= 0.6: return "#fed7aa"
    return "#fecaca"

def _heat_fg(acc: float) -> str:
    if acc >= 0.7: return "#14532d"
    if acc >= 0.6: return "#713f12"
    return "#7f1d1d"

def _acc_color(acc: float) -> str:
    if acc >= 0.9: return "#16a34a"
    if acc >= 0.7: return "#65a30d"
    if acc >= 0.5: return "#d97706"
    return "#dc2626"

_CAT_COLORS = {
    "blinds":     "#3b82f6",
    "chairs":     "#8b5cf6",
    "table":      "#f97316",
    "whiteboard": "#10b981",
}

_LABEL_COLOR = {
    "clean": "#16a34a",
    "messy": "#dc2626",
}


def save_html(all_results: dict, path: Path, timestamp: str) -> None:
    model_keys = list(all_results.keys())
    categories = sorted({
        ct for data in all_results.values()
        for ct in data["metrics"]["per_change_type"]
    })

    # Sort models by overall accuracy descending
    model_keys = sorted(model_keys, key=lambda k: -all_results[k]["metrics"]["accuracy"])

    # ── Chart.js bar chart data ───────────────────────────────────────────────
    chart_labels_js = json.dumps([all_results[k]["model_name"] for k in model_keys])
    overall_data = [round(all_results[k]["metrics"]["accuracy"] * 100, 1) for k in model_keys]
    cat_datasets = []
    for ct in categories:
        color = _CAT_COLORS.get(ct, "#888")
        data = [
            round(all_results[k]["metrics"]["per_change_type"].get(ct, {}).get("accuracy", 0) * 100, 1)
            for k in model_keys
        ]
        cat_datasets.append({"label": ct, "data": data, "color": color})

    chart_datasets_js = json.dumps([
        {"label": "overall", "data": overall_data,
         "backgroundColor": "rgba(0,0,0,0.08)", "borderColor": "#374151", "borderWidth": 2},
    ] + [
        {"label": d["label"], "data": d["data"],
         "backgroundColor": d["color"] + "33", "borderColor": d["color"], "borderWidth": 2}
        for d in cat_datasets
    ])

    # ── Heatmap matrix: rows=models, cols=categories ──────────────────────────
    heatmap_header = "<tr><th></th>" + "".join(
        "<th style='text-align:center;color:{};letter-spacing:1px'>{}</th>".format(
            _CAT_COLORS.get(ct, "#555"), ct)
        for ct in categories
    ) + "<th style='text-align:center'>overall</th></tr>"

    heatmap_rows = ""
    for key in model_keys:
        m = all_results[key]["metrics"]
        cells = ""
        for ct in categories:
            acc = m["per_change_type"].get(ct, {}).get("accuracy", 0)
            correct = m["per_change_type"].get(ct, {}).get("correct", 0)
            total = m["per_change_type"].get(ct, {}).get("total", 0)
            bg = _heat_bg(acc)
            fg = _heat_fg(acc)
            cells += (
                f"<td style='text-align:center;background:{bg};color:{fg};"
                f"font-weight:700;font-size:1rem;padding:14px 18px'>"
                f"{acc:.0%}"
                f"<div style='font-weight:400;font-size:10px;opacity:.6'>{correct}/{total}</div></td>"
            )
        ov = m["accuracy"]
        cells += (
            f"<td style='text-align:center;background:{_heat_bg(ov)};color:{_heat_fg(ov)};"
            f"font-weight:700;font-size:1rem;padding:14px 18px;border-left:2px solid #e5e7eb'>"
            f"{ov:.0%}</td>"
        )
        heatmap_rows += f"<tr><td style='font-weight:600;white-space:nowrap;padding-right:1.5rem'>{all_results[key]['model_name']}</td>{cells}</tr>"

    # ── Per-image detail rows ─────────────────────────────────────────────────
    all_fnames_by_cat: dict[str, list[str]] = {}
    for data in all_results.values():
        for r in data["results"]:
            ct = r["change_type"]
            fn = r["image_filename"]
            if fn not in all_fnames_by_cat.get(ct, []):
                all_fnames_by_cat.setdefault(ct, []).append(fn)

    lookup: dict[str, dict] = {}
    for key, data in all_results.items():
        for r in data["results"]:
            lookup.setdefault(r["image_filename"], {})[key] = r

    model_th = "".join(
        f"<th style='text-align:center'>{all_results[k]['model_name']}</th>"
        for k in model_keys
    )

    detail_sections = ""
    for ct in categories:
        fnames = all_fnames_by_cat.get(ct, [])
        cat_color = _CAT_COLORS.get(ct, "#555")
        rows_html = ""
        for fn in fnames:
            row_data = lookup.get(fn, {})
            any_r = next(iter(row_data.values()), {})
            img_src = _img_b64(any_r.get("image_path", ""))
            img_html = (
                f'<img src="{img_src}" style="width:130px;height:90px;object-fit:cover;'
                f'border-radius:4px;border:1px solid #e5e7eb;display:block">'
                if img_src else '<div style="width:130px;height:90px;background:#f3f4f6;border-radius:4px"></div>'
            )
            gt = any_r.get("label", "")
            gt_color = _LABEL_COLOR.get(gt, "#555")

            cells = f"<td style='min-width:140px;vertical-align:middle;text-align:center'>{img_html}</td>"
            cells += (
                f"<td style='vertical-align:middle;text-align:center'>"
                f"<span style='color:{gt_color};font-weight:700;font-size:1.05em'>{gt}</span></td>"
            )

            for key in model_keys:
                r = row_data.get(key, {})
                pred = r.get("predicted_label")
                ans = r.get("answer")
                raw = r.get("raw_response", "")
                lat = r.get("latency_ms", "—")
                err = r.get("error")
                reasoning_escaped = raw.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")

                if pred is None and err:
                    cells += "<td style='text-align:center;color:#dc2626;font-size:11px'>⚠ error</td>"
                elif pred is None:
                    cells += (
                        f"<td style='text-align:center;vertical-align:middle;cursor:pointer;"
                        f"background:#fef9c3' onclick=\"showReasoning(`{reasoning_escaped}`)\">"
                        f"<span style='color:#b45309;font-weight:700'>?</span><br>"
                        f"<span style='color:#92400e;font-size:10px'>parse fail · {lat}ms</span></td>"
                    )
                else:
                    correct = pred == gt
                    pred_color = _LABEL_COLOR.get(pred, "#555")
                    bg = "#dcfce7" if correct else "#fee2e2"
                    mark = "✓" if correct else "✗"
                    mark_color = "#16a34a" if correct else "#dc2626"
                    ans_str = {True: "yes", False: "no", None: "?"}[ans]
                    cells += (
                        f"<td style='text-align:center;vertical-align:middle;background:{bg};"
                        f"cursor:pointer' onclick=\"showReasoning(`{reasoning_escaped}`)\">"
                        f"<span style='color:{pred_color};font-weight:700'>{pred}</span> "
                        f"<span style='color:{mark_color}'>{mark}</span><br>"
                        f"<span style='color:#6b7280;font-size:10px'>ans: {ans_str} · {lat}ms</span></td>"
                    )

            rows_html += f"<tr>{cells}</tr>"

        q_second_line = QUESTIONS[ct]['question'].split('\n')[1].strip()
        q_yes_means = QUESTIONS[ct]['yes_means']
        detail_sections += f"""
<h2 style='color:{cat_color};margin-top:2.5rem;letter-spacing:2px;text-transform:uppercase;font-size:.9rem;border-bottom:1px solid #e5e7eb;padding-bottom:.4rem;margin-bottom:.8rem'>
  {ct}
  <span style='color:#9ca3af;font-size:.75rem;font-weight:400;margin-left:1rem;letter-spacing:0'>
    {q_second_line} &nbsp;·&nbsp; yes → {q_yes_means}
  </span>
</h2>
<table>
  <tr><th>Image</th><th>Ground Truth</th>{model_th}</tr>
  {rows_html}
</table>"""

    n_images = sum(len(v) for v in all_fnames_by_cat.values())

    # ── Summary table (7 rows, sorted by overall acc) ────────────────────────
    summary_rows = ""
    for key in model_keys:
        m = all_results[key]["metrics"]
        name = all_results[key]["model_name"]
        ov = m["accuracy"]
        def _cell(ct):
            acc = m["per_change_type"].get(ct, {}).get("accuracy", 0)
            return f"<td style='text-align:center;color:{_acc_color(acc)};font-weight:600'>{acc:.0%}</td>"
        summary_rows += (
            f"<tr>"
            f"<td style='font-weight:600;white-space:nowrap'>{name}</td>"
            + "".join(_cell(ct) for ct in categories)
            + f"<td style='text-align:center;font-weight:700;color:{_acc_color(ov)};border-left:2px solid #e5e7eb'>{ov:.0%}</td>"
            f"</tr>"
        )
    cat_headers = "".join(
        "<th style='text-align:center;color:{}'>{}</th>".format(_CAT_COLORS.get(ct, "#555"), ct)
        for ct in categories
    )
    summary_header = (
        f"<tr><th>Model</th>{cat_headers}"
        f"<th style='text-align:center;border-left:2px solid #e5e7eb'>Overall</th></tr>"
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Env Monitoring Binary — {timestamp}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #f9fafb;
    color: #111827;
    padding: 2rem 2.5rem;
    line-height: 1.5;
  }}
  h1 {{ color: #111827; font-size: 1.3rem; letter-spacing: 1px; margin-bottom: .3rem; }}
  .subtitle {{ color: #9ca3af; font-size: .8rem; margin-bottom: 2rem; }}
  .section-title {{
    font-size: .75rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase;
    color: #6b7280; border-bottom: 1px solid #e5e7eb; padding-bottom: .4rem;
    margin: 2rem 0 .8rem;
  }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1rem; font-size: .82rem; }}
  th {{
    background: #f3f4f6; color: #6b7280; padding: 8px 12px; text-align: left;
    border-bottom: 2px solid #e5e7eb; font-size: .75rem; letter-spacing: 1px; text-transform: uppercase;
  }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #f3f4f6; vertical-align: top; }}
  tr:hover td {{ background: #f9fafb; }}
  .chart-wrap {{
    background: #fff; border: 1px solid #e5e7eb; border-radius: 8px;
    padding: 1.5rem; margin-bottom: 1.5rem; max-width: 860px;
  }}
  .tag {{
    display: inline-block; background: #f3f4f6; color: #6b7280;
    padding: 2px 10px; border-radius: 3px; font-size: .75rem; margin-right: 6px;
  }}
  .heatmap-wrap {{
    background: #fff; border: 1px solid #e5e7eb; border-radius: 8px;
    padding: 1rem 1.5rem; margin-bottom: 1.5rem; display: inline-block;
  }}
  .heatmap-wrap table {{ margin-bottom: 0; }}
  .heatmap-wrap td, .heatmap-wrap th {{ border: 1px solid #e5e7eb; }}
  #modal-overlay {{
    display: none; position: fixed; inset: 0;
    background: rgba(0,0,0,0.4); z-index: 100;
    align-items: center; justify-content: center;
  }}
  #modal-overlay.open {{ display: flex; }}
  #modal-box {{
    background: #fff; border: 1px solid #e5e7eb; border-radius: 10px;
    padding: 1.5rem 2rem; max-width: 680px; width: 90%;
    max-height: 80vh; overflow-y: auto; position: relative;
    box-shadow: 0 20px 60px rgba(0,0,0,0.15);
  }}
  #modal-box h3 {{
    color: #6b7280; font-size: .75rem; letter-spacing: 2px; text-transform: uppercase;
    margin-bottom: 1rem; border-bottom: 1px solid #f3f4f6; padding-bottom: .5rem;
  }}
  #modal-text {{
    color: #374151; font-size: .85rem; white-space: pre-wrap; line-height: 1.7;
    font-family: 'Courier New', monospace;
  }}
  #modal-close {{
    position: absolute; top: .75rem; right: 1rem; background: none;
    border: none; color: #9ca3af; font-size: 1.2rem; cursor: pointer;
  }}
  #modal-close:hover {{ color: #374151; }}
</style>
</head>
<body>
<div id="modal-overlay" onclick="if(event.target===this)closeModal()">
  <div id="modal-box">
    <button id="modal-close" onclick="closeModal()">✕</button>
    <h3>Model Reasoning</h3>
    <pre id="modal-text"></pre>
  </div>
</div>

<h1>Environment Monitoring — Binary Benchmark</h1>
<p class="subtitle">
  <span class="tag">{timestamp}</span>
  <span class="tag">yes/no + reasoning per category</span>
  <span class="tag">{n_images} images</span>
  <span class="tag">{len(model_keys)} models</span>
</p>

<p class="section-title">Results</p>
<table style="max-width:700px;margin-bottom:1.5rem">
  {summary_header}
  {summary_rows}
</table>

<p class="section-title">Accuracy Matrix</p>
<div class="heatmap-wrap">
  <table>
    {heatmap_header}
    {heatmap_rows}
  </table>
</div>

<p class="section-title">Accuracy by Category</p>
<div class="chart-wrap">
  <canvas id="accChart" height="80"></canvas>
</div>

{detail_sections}

<script>
function showReasoning(text) {{
  document.getElementById('modal-text').textContent = text || '(no response)';
  document.getElementById('modal-overlay').classList.add('open');
}}
function closeModal() {{
  document.getElementById('modal-overlay').classList.remove('open');
}}
document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeModal(); }});

new Chart(document.getElementById('accChart'), {{
  type: 'bar',
  data: {{
    labels: {chart_labels_js},
    datasets: {chart_datasets_js}
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ labels: {{ color: '#6b7280', font: {{ size: 11 }} }} }}
    }},
    scales: {{
      x: {{
        ticks: {{ color: '#6b7280', font: {{ size: 10 }} }},
        grid: {{ color: '#f3f4f6' }}
      }},
      y: {{
        min: 0, max: 100,
        ticks: {{
          color: '#6b7280', font: {{ size: 10 }},
          callback: v => v + '%'
        }},
        grid: {{ color: '#f3f4f6' }}
      }}
    }}
  }}
}});
</script>
</body>
</html>"""

    path.write_text(html)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binary yes/no environment monitoring benchmark")
    parser.add_argument("--config", default="benchmark_config.yaml")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_filter = args.models if args.models else None

    print(f"Config:  {args.config}")
    print(f"Dataset: {DATASET_CSV}")
    print(f"Models:  {model_filter or [m.key for m in cfg.enabled_models]}")

    run_binary_benchmark(cfg, model_filter=model_filter)
