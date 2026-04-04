#!/usr/bin/env python
"""
run_benchmark_meeting_room.py — Meeting room readiness checklist benchmark.

Each image is a meeting room photo. The VLM receives the image plus a
predefined checklist and must return a structured JSON response indicating
whether each checklist item is satisfied and whether the room is ready overall.

Evaluation is binary (true/false per item) against human-labelled ground truth —
no LLM judge or API calls required.

Test set format (see test_sets/meeting_room_sample.json):
  {
    "checklist": [{"id": 1, "item": "All chairs tucked in"}, ...],
    "samples": [
      {
        "id": "room_001",
        "image": "test_sets/images/meeting_rooms/room_001.jpg",
        "ground_truth": {
          "items": {"1": true, "2": false, ...},
          "room_ready": false
        }
      }
    ]
  }

Metrics reported:
  - item_accuracy      : fraction of (image × item) pairs correct
  - room_accuracy      : fraction of images where room_ready verdict is correct
  - per_item_accuracy  : per checklist item breakdown
  - room_ready F1      : precision / recall / F1 treating "ready" as positive

Usage:
    cd benchmark
    conda activate /mnt/shared/dils/envs/Qwen3VL-env

    python run_benchmark_meeting_room.py --test-set test_sets/meeting_room_sample.json --all
    python run_benchmark_meeting_room.py --test-set test_sets/meeting_room_sample.json --models smolvlm internvl
    python run_benchmark_meeting_room.py --help
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

from config import BenchmarkConfig, load_config
from models import MODEL_REGISTRY


# ── Prompt construction ────────────────────────────────────────────────────────

def build_prompt(checklist: list[dict]) -> str:
    items_text = "\n".join(
        f"{item['id']}. {item['item']}" for item in checklist
    )
    item_ids = [str(item["id"]) for item in checklist]
    example_items = {iid: True for iid in item_ids}
    example_items[item_ids[-1]] = False  # make example non-trivial

    return f"""\
You are inspecting a meeting room to determine whether it is ready for a meeting.

Carefully examine the image and evaluate each item in the checklist below.

Checklist:
{items_text}

Rules:
- Mark an item true if the condition is clearly and fully met in the image.
- Mark an item false if the condition is not met OR if it cannot be confirmed from the image.
- Set room_ready to true ONLY if every single item is true.

Respond ONLY with a JSON object — no markdown fences, no explanation before or after. Use exactly this structure:
{{
  "items": {{
{chr(10).join(f'    "{iid}": true or false,' for iid in item_ids)}
  }},
  "room_ready": true or false,
  "reasoning": "one or two sentences explaining your assessment"
}}"""


# ── Response parsing ───────────────────────────────────────────────────────────

_JSON_RE = re.compile(r'\{[\s\S]*\}')


def parse_response(response: str, checklist: list[dict]) -> dict | None:
    """
    Extract the JSON payload from the model's raw response.
    Returns None if parsing fails entirely.
    """
    # Try direct parse first
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    # Try to extract the first {...} block (handles markdown fences, preamble, etc.)
    m = _JSON_RE.search(response)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    return None


def coerce_result(parsed: dict, checklist: list[dict]) -> dict:
    """
    Normalise a parsed response dict into a canonical form:
      { "items": {"1": bool, ...}, "room_ready": bool, "reasoning": str }
    Fills missing item keys with False.
    """
    items_raw = parsed.get("items", {})
    items: dict[str, bool] = {}
    for item in checklist:
        key = str(item["id"])
        val = items_raw.get(key, items_raw.get(item["id"], None))
        if isinstance(val, bool):
            items[key] = val
        elif isinstance(val, str):
            items[key] = val.lower() == "true"
        else:
            items[key] = False  # missing → pessimistic

    room_ready_raw = parsed.get("room_ready", False)
    if isinstance(room_ready_raw, str):
        room_ready = room_ready_raw.lower() == "true"
    else:
        room_ready = bool(room_ready_raw)

    return {
        "items": items,
        "room_ready": room_ready,
        "reasoning": str(parsed.get("reasoning", "")),
    }


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(model_results: list[dict], checklist: list[dict]) -> dict:
    """
    Compute aggregate metrics from a list of per-image result dicts.
    Each result dict must have keys: predicted, ground_truth (both coerced).
    """
    item_ids = [str(c["id"]) for c in checklist]
    item_labels = {str(c["id"]): c["item"] for c in checklist}

    total_items = 0
    correct_items = 0
    per_item_correct: dict[str, int] = {iid: 0 for iid in item_ids}
    per_item_total: dict[str, int] = {iid: 0 for iid in item_ids}

    room_correct = 0
    tp = fp = fn = tn = 0  # for room_ready F1 (ready = positive)

    for r in model_results:
        pred = r.get("predicted")
        gt = r["ground_truth"]
        if pred is None:
            continue

        # Item-level
        for iid in item_ids:
            p = pred["items"].get(iid, False)
            g = gt["items"].get(iid, False)
            per_item_total[iid] += 1
            total_items += 1
            if p == g:
                per_item_correct[iid] += 1
                correct_items += 1

        # Room-level
        p_ready = pred["room_ready"]
        g_ready = gt["room_ready"]
        if p_ready == g_ready:
            room_correct += 1
        if p_ready and g_ready:
            tp += 1
        elif p_ready and not g_ready:
            fp += 1
        elif not p_ready and g_ready:
            fn += 1
        else:
            tn += 1

    n = len(model_results)
    item_acc = correct_items / total_items if total_items else 0.0
    room_acc = room_correct / n if n else 0.0

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    per_item_acc = {
        iid: {
            "label": item_labels[iid],
            "accuracy": per_item_correct[iid] / per_item_total[iid] if per_item_total[iid] else 0.0,
            "correct": per_item_correct[iid],
            "total": per_item_total[iid],
        }
        for iid in item_ids
    }

    return {
        "item_accuracy": round(item_acc, 4),
        "room_accuracy": round(room_acc, 4),
        "room_precision": round(precision, 4),
        "room_recall": round(recall, 4),
        "room_f1": round(f1, 4),
        "per_item": per_item_acc,
        "n_images": n,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ── Core runner ───────────────────────────────────────────────────────────────

def run_meeting_room_benchmark(
    cfg: BenchmarkConfig,
    test_set_path: str,
    model_filter: list[str] | None = None,
) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(test_set_path) as f:
        test_set = json.load(f)

    checklist: list[dict] = test_set["checklist"]
    samples: list[dict] = test_set["samples"]
    prompt = build_prompt(checklist)

    print(f"\nChecklist ({len(checklist)} items):")
    for item in checklist:
        print(f"  {item['id']}. {item['item']}")
    print(f"\nSamples: {len(samples)}")

    models_to_run = [
        m for m in cfg.enabled_models
        if model_filter is None or m.key in model_filter
    ]

    if not models_to_run:
        print("No models selected.")
        return

    all_results: dict[str, dict] = {}  # model_key → {metrics, results}

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
            image_path = sample["image"]
            sample_id = sample["id"]
            gt = sample["ground_truth"]

            if not Path(image_path).exists():
                print(f"  [{sample_id}] WARNING: image not found — {image_path}")
                model_results.append({
                    "id": sample_id,
                    "image": image_path,
                    "ground_truth": gt,
                    "predicted": None,
                    "parse_error": "image not found",
                    "raw_response": "",
                    "latency_ms": 0,
                    "model_name": model.name,
                })
                continue

            result = model.run(image_path, prompt)

            if result.error:
                print(f"  [{sample_id}] INFERENCE ERROR: {result.error}")
                model_results.append({
                    "id": sample_id,
                    "image": image_path,
                    "ground_truth": gt,
                    "predicted": None,
                    "parse_error": result.error,
                    "raw_response": result.response,
                    "latency_ms": round(result.latency_ms),
                    "model_name": model.name,
                })
                continue

            parsed = parse_response(result.response, checklist)
            if parsed is None:
                print(f"  [{sample_id}] PARSE ERROR — raw: {result.response[:120]!r}")
                model_results.append({
                    "id": sample_id,
                    "image": image_path,
                    "ground_truth": gt,
                    "predicted": None,
                    "parse_error": "json parse failed",
                    "raw_response": result.response,
                    "latency_ms": round(result.latency_ms),
                    "model_name": model.name,
                })
                continue

            predicted = coerce_result(parsed, checklist)

            # Per-item correctness summary for console
            item_results = []
            for item in checklist:
                iid = str(item["id"])
                p = predicted["items"].get(iid, False)
                g = gt["items"].get(iid, False)
                mark = "✓" if p == g else "✗"
                item_results.append(f"{mark}{iid}({'T' if p else 'F'})")

            room_match = "✓" if predicted["room_ready"] == gt["room_ready"] else "✗"
            print(
                f"  [{sample_id}] {' '.join(item_results)} | "
                f"ready={room_match}({'T' if predicted['room_ready'] else 'F'} vs gt={'T' if gt['room_ready'] else 'F'}) "
                f"| {result.latency_ms:.0f}ms"
            )

            model_results.append({
                "id": sample_id,
                "image": image_path,
                "ground_truth": gt,
                "predicted": predicted,
                "parse_error": None,
                "raw_response": result.response,
                "latency_ms": round(result.latency_ms),
                "model_name": model.name,
                "model_path": mcfg.model_path,
            })

        model.unload()

        metrics = compute_metrics(model_results, checklist)
        all_results[mcfg.key] = {
            "model_name": model.name,
            "model_path": mcfg.model_path,
            "metrics": metrics,
            "results": model_results,
        }

        print(f"\n  item_accuracy={metrics['item_accuracy']:.1%}  "
              f"room_accuracy={metrics['room_accuracy']:.1%}  "
              f"room_F1={metrics['room_f1']:.3f}")

    # ── Save + report ──────────────────────────────────────────────────────────
    out = {
        "timestamp": timestamp,
        "checklist": checklist,
        "test_set": test_set_path,
        "models": all_results,
    }
    json_path = cfg.output_dir / f"meeting_room_results_{timestamp}.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\nRaw results → {json_path}")

    print_summary(all_results, checklist)

    html_path = cfg.output_dir / f"meeting_room_report_{timestamp}.html"
    save_html(all_results, checklist, html_path, timestamp, test_set_path)
    print(f"HTML report  → {html_path}")


# ── Console summary ────────────────────────────────────────────────────────────

def print_summary(all_results: dict, checklist: list[dict]) -> None:
    print(f"\n{'='*70}\nMEETING ROOM BENCHMARK SUMMARY\n{'='*70}")
    print(f"{'Model':<38} {'ItemAcc':>8} {'RoomAcc':>8} {'RoomF1':>8} {'N':>5}")
    print("-" * 70)
    for key, data in all_results.items():
        m = data["metrics"]
        print(
            f"{data['model_name']:<38} "
            f"{m['item_accuracy']:>7.1%} "
            f"{m['room_accuracy']:>8.1%} "
            f"{m['room_f1']:>8.3f} "
            f"{m['n_images']:>5}"
        )

    print()
    # Per-item breakdown across all models
    first = next(iter(all_results.values()), None)
    if first:
        print(f"{'Item':<50} " + "  ".join(f"{k[:10]:>10}" for k in all_results))
        print("-" * 70)
        for item in checklist:
            iid = str(item["id"])
            label = item["item"][:48]
            row = f"{label:<50} "
            for data in all_results.values():
                pi = data["metrics"]["per_item"].get(iid, {})
                acc = pi.get("accuracy", 0)
                row += f"{acc:>9.1%}  "
            print(row)


# ── HTML report ────────────────────────────────────────────────────────────────

def _img_b64(image_path: str) -> str | None:
    try:
        ext = Path(image_path).suffix.lower().strip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")
        with open(image_path, "rb") as f:
            return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"
    except Exception:
        return None


def _acc_color(acc: float) -> str:
    if acc >= 0.9: return "#3d9"
    if acc >= 0.7: return "#8bc34a"
    if acc >= 0.5: return "#fa0"
    return "#e54"


def save_html(
    all_results: dict,
    checklist: list[dict],
    path: Path,
    timestamp: str,
    test_set_path: str,
) -> None:
    model_keys = list(all_results.keys())
    item_ids = [str(c["id"]) for c in checklist]

    # ── Summary table ──────────────────────────────────────────────────────────
    summary_rows = ""
    for key, data in all_results.items():
        m = data["metrics"]
        ia_c = _acc_color(m["item_accuracy"])
        ra_c = _acc_color(m["room_accuracy"])
        f1_c = _acc_color(m["room_f1"])
        summary_rows += (
            f"<tr>"
            f"<td>{data['model_name']}</td>"
            f"<td><span style='color:{ia_c};font-weight:700'>{m['item_accuracy']:.1%}</span></td>"
            f"<td><span style='color:{ra_c};font-weight:700'>{m['room_accuracy']:.1%}</span></td>"
            f"<td><span style='color:{f1_c};font-weight:700'>{m['room_f1']:.3f}</span></td>"
            f"<td style='color:#666;font-size:11px'>"
            f"P={m['room_precision']:.2f} R={m['room_recall']:.2f} "
            f"TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}</td>"
            f"<td>{m['n_images']}</td>"
            f"</tr>"
        )

    # ── Per-item accuracy rows ─────────────────────────────────────────────────
    item_rows = ""
    for item in checklist:
        iid = str(item["id"])
        item_rows += f"<tr><td>{iid}</td><td style='color:#aaa'>{item['item']}</td>"
        for key, data in all_results.items():
            pi = data["metrics"]["per_item"].get(iid, {})
            acc = pi.get("accuracy", 0.0)
            correct = pi.get("correct", 0)
            total = pi.get("total", 0)
            color = _acc_color(acc)
            item_rows += (
                f"<td><span style='color:{color};font-weight:600'>{acc:.1%}</span>"
                f"<span style='color:#444;font-size:10px'> ({correct}/{total})</span></td>"
            )
        item_rows += "</tr>"

    # ── Per-image detail rows ──────────────────────────────────────────────────
    # Collect all sample IDs in order
    all_ids: list[str] = []
    for data in all_results.values():
        for r in data["results"]:
            if r["id"] not in all_ids:
                all_ids.append(r["id"])

    # Build lookup: sample_id → model_key → result
    lookup: dict[str, dict[str, dict]] = {}
    for key, data in all_results.items():
        for r in data["results"]:
            lookup.setdefault(r["id"], {})[key] = r

    model_th = "".join(
        f"<th>{all_results[k]['model_name']}</th>" for k in model_keys
    )

    detail_rows = ""
    for sid in all_ids:
        row_data = lookup.get(sid, {})
        any_r = next(iter(row_data.values()), {})
        image_path = any_r.get("image", "")
        img_src = _img_b64(image_path)
        img_html = (
            f'<img src="{img_src}" style="width:150px;height:110px;object-fit:cover;'
            f'border-radius:4px;border:1px solid #2a2a2a;display:block;margin-bottom:6px">'
            if img_src else '<div style="width:150px;height:110px;background:#1a1a1a;border-radius:4px;display:flex;align-items:center;justify-content:center;color:#333;font-size:10px">no image</div>'
        )

        gt = any_r.get("ground_truth", {})
        gt_items = gt.get("items", {})
        gt_ready = gt.get("room_ready", False)

        # Ground truth column
        gt_items_html = "".join(
            f"<div style='font-size:10px;color:#555;margin:1px 0'>"
            f"<span style='color:{'#3d9' if gt_items.get(iid) else '#e54'}'>"
            f"{'✓' if gt_items.get(iid) else '✗'}</span> {item['item'][:40]}</div>"
            for iid, item in zip(item_ids, checklist)
        )
        gt_ready_color = "#3d9" if gt_ready else "#e54"
        gt_html = (
            f"{gt_items_html}"
            f"<div style='margin-top:6px;font-size:11px;font-weight:700;"
            f"color:{gt_ready_color}'>{'READY' if gt_ready else 'NOT READY'}</div>"
        )

        detail_rows += (
            f"<tr>"
            f"<td style='min-width:170px;vertical-align:top'>{img_html}"
            f"<b style='font-size:11px;color:#888'>#{sid}</b></td>"
            f"<td style='min-width:180px;vertical-align:top'>{gt_html}</td>"
        )

        for key in model_keys:
            r = row_data.get(key, {})
            pred = r.get("predicted")
            lat = r.get("latency_ms", "—")
            parse_err = r.get("parse_error")

            if pred is None:
                detail_rows += (
                    f"<td style='color:#555;font-size:11px'>"
                    f"{'⚠ ' + parse_err if parse_err else '—'}</td>"
                )
                continue

            pred_items = pred.get("items", {})
            pred_ready = pred.get("room_ready", False)
            reasoning = pred.get("reasoning", "")

            items_html = ""
            for iid, item in zip(item_ids, checklist):
                p = pred_items.get(iid, False)
                g = gt_items.get(iid, False)
                match = p == g
                bg = "#0d1f12" if match else "#1f0d0d"
                mark = "✓" if p else "✗"
                mark_color = "#3d9" if p else "#e54"
                items_html += (
                    f"<div style='font-size:10px;margin:1px 0;padding:2px 4px;"
                    f"border-radius:2px;background:{bg}'>"
                    f"<span style='color:{mark_color}'>{mark}</span> "
                    f"<span style='color:#{'aaa' if match else '777'}'>{item['item'][:38]}</span>"
                    f"{'<span style=\"color:#e54;font-size:9px\"> ✗GT</span>' if not match else ''}"
                    f"</div>"
                )

            ready_correct = pred_ready == gt_ready
            ready_color = "#3d9" if pred_ready else "#e54"
            ready_bg = "#0d1f12" if ready_correct else "#1f0d0d"
            items_html += (
                f"<div style='margin-top:6px;padding:3px 6px;border-radius:3px;"
                f"background:{ready_bg};font-size:11px;font-weight:700;color:{ready_color}'>"
                f"{'READY' if pred_ready else 'NOT READY'}"
                f"{'<span style=\"color:#e54\"> ✗</span>' if not ready_correct else ''}"
                f"</div>"
            )

            if reasoning:
                items_html += (
                    f"<div style='margin-top:4px;font-size:9px;color:#555;"
                    f"font-style:italic'>{reasoning[:120]}</div>"
                )

            detail_rows += (
                f"<td style='vertical-align:top'>"
                f"{items_html}"
                f"<div style='color:#444;font-size:10px;margin-top:4px'>{lat}ms</div>"
                f"</td>"
            )

        detail_rows += "</tr>"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Meeting Room Benchmark — {timestamp}</title>
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
  .tag {{ display:inline-block; background:#1a1a1a; color:#666;
          padding: 2px 10px; border-radius: 3px; font-size: .75rem; }}
  .note {{ font-size: 11px; color: #555; margin-top: 6px; font-style: italic; }}
</style>
</head>
<body>
<h1>Meeting Room Readiness Benchmark</h1>
<p class="tag">{timestamp}</p>
<p class="note">Test set: {test_set_path} · Binary checklist evaluation · No LLM judge</p>

<h2>Summary</h2>
<table>
  <tr>
    <th>Model</th>
    <th>Item Accuracy</th>
    <th>Room Accuracy</th>
    <th>Room F1</th>
    <th>Confusion (ready=pos)</th>
    <th>N</th>
  </tr>
  {summary_rows}
</table>

<h2>Per-Item Accuracy</h2>
<table>
  <tr><th>#</th><th>Checklist Item</th>{"".join(f"<th>{all_results[k]['model_name']}</th>" for k in model_keys)}</tr>
  {item_rows}
</table>

<h2>Per-Image Results</h2>
<table>
  <tr><th>Image</th><th>Ground Truth</th>{model_th}</tr>
  {detail_rows}
</table>
</body>
</html>"""

    path.write_text(html)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Meeting room readiness checklist benchmark"
    )
    parser.add_argument("--config", default="benchmark_config.yaml")
    parser.add_argument(
        "--test-set",
        default="test_sets/meeting_room_sample.json",
        help="Path to test set JSON (checklist + samples with ground truth)",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Model keys to run (from benchmark_config.yaml). Omit to run all enabled.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all enabled models from config (same as omitting --models)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.models:
        model_filter = args.models
    else:
        model_filter = None

    print(f"Config:    {args.config}")
    print(f"Test set:  {args.test_set}")
    print(f"Models:    {model_filter or [m.key for m in cfg.enabled_models]}")

    run_meeting_room_benchmark(cfg, args.test_set, model_filter=model_filter)
