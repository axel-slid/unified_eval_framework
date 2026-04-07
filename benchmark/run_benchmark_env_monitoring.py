#!/usr/bin/env python
"""
run_benchmark_env_monitoring.py — Two-stage environment monitoring benchmark.

For each image the model is run twice:
  Stage 1 — Presence detection: "Is there a [X] in this image?" → yes / no
  Stage 2 — Readiness assessment (only if Stage 1 = yes):
             "Is the [X] ready for a meeting?" → ready / not_ready + description

This yields four predicted classes per sample:
  not_present   — Stage 1 said no
  present_clean — Stage 1 yes, Stage 2 ready
  present_messy — Stage 1 yes, Stage 2 not_ready
  uncertain     — Stage 1 yes, Stage 2 unparseable

Ground truth maps directly from the dataset labels:
  clean  → present_clean
  messy  → present_messy

Dataset: environment_monitoring_dataset/unified_annotations.csv
Categories: whiteboard, blinds, chairs, table

Usage:
    cd benchmark
    conda activate /mnt/shared/dils/envs/Qwen3VL-env

    python run_benchmark_env_monitoring.py --all
    python run_benchmark_env_monitoring.py --models qwen3vl_4b internvl
    python run_benchmark_env_monitoring.py --help
"""

from __future__ import annotations

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

CLASSES = ["not_present", "present_clean", "present_messy", "uncertain"]

# ── Stage prompts ─────────────────────────────────────────────────────────────

STAGE1 = {
    "whiteboard": (
        "Look at the walls in this image.\n"
        "Is there a whiteboard or large white writing surface panel visible?\n"
        "Reply with only the single word: yes or no"
    ),
    "blinds": (
        "Look at the windows in this image.\n"
        "Are there any window coverings visible — blinds, roller shades, curtains, or shutters?\n"
        "Reply with only the single word: yes or no"
    ),
    "chairs": (
        "Are there chairs visible in this image?\n"
        "Reply with only the single word: yes or no"
    ),
    "table": (
        "Is there a table visible in this image?\n"
        "Reply with only the single word: yes or no"
    ),
}

STAGE2 = {
    "whiteboard": (
        "Look at the whiteboard in this image.\n"
        "Does it have ANY markings, text, drawings, or writing on it at all?\n"
        "Reply on the first line with: ready (completely clean, no marks whatsoever) or not_ready (any marks present).\n"
        "Then on a new line describe exactly what you observe on the whiteboard surface."
    ),
    "blinds": (
        "Look at the window coverings in this image.\n"
        "Can you see any daylight, outdoor view, or light gaps coming through the coverings?\n"
        "Are any slats, panels, or sections at a different angle from the others?\n"
        "Reply on the first line with: not_ready (any light gaps, outdoor view visible, or uneven sections) "
        "or ready (coverings uniformly block the outdoor view with no gaps).\n"
        "Then on a new line describe specifically what you see — any gaps, light, or unevenness."
    ),
    "chairs": (
        "Look at the chairs in this image.\n"
        "Are any chairs significantly displaced — pulled out more than halfway from the table, "
        "knocked over, pushed against a wall, or clearly abandoned away from the table?\n"
        "Minor positional variation is fine and counts as ready.\n"
        "Reply on the first line with: not_ready (any chair significantly displaced) "
        "or ready (chairs generally arranged around the table).\n"
        "Then on a new line describe the arrangement, noting any chairs that are clearly out of place."
    ),
    "table": (
        "Look at the table surface in this image.\n"
        "Does the table have ANY of the following left on it: "
        "laptops, notebooks, cups, bottles, phones, bags, papers, food, or personal items?\n"
        "Only permanently installed equipment (built-in monitors, cable trays, fixed screens) is acceptable.\n"
        "Reply on the first line with: not_ready (any personal items present) or ready (surface clear).\n"
        "Then on a new line list exactly what you see on the table surface."
    ),
}


# ── Response parsing ──────────────────────────────────────────────────────────

def parse_yes_no(response: str) -> bool | None:
    """Return True=yes, False=no, None=unparseable."""
    text = response.strip().lower()
    # Check first non-empty word or line
    first = text.split()[0] if text.split() else ""
    if first in ("yes", "yes.", "yes,"):
        return True
    if first in ("no", "no.", "no,"):
        return False
    # Fallback: search for yes/no
    if re.search(r"\byes\b", text):
        return True
    if re.search(r"\bno\b", text):
        return False
    return None


def parse_ready(response: str) -> bool | None:
    """Return True=ready, False=not_ready, None=unparseable."""
    text = response.strip().lower()
    first_line = text.split("\n")[0].strip()
    # not_ready before ready to avoid prefix match
    if "not_ready" in first_line or "not ready" in first_line:
        return False
    if "ready" in first_line:
        return True
    # fallback to full text
    if "not_ready" in text or "not ready" in text:
        return False
    if "ready" in text:
        return True
    return None


def extract_description(response: str) -> str:
    """Pull the free-text description from Stage 2 (everything after first line)."""
    lines = response.strip().split("\n")
    desc_lines = [l.strip() for l in lines[1:] if l.strip()]
    return " ".join(desc_lines)


def classify(stage1: bool | None, stage2: bool | None) -> str:
    if stage1 is False or stage1 is None:
        return "not_present"
    if stage2 is True:
        return "present_clean"
    if stage2 is False:
        return "present_messy"
    return "uncertain"


def gt_class(label: str) -> str:
    return "present_clean" if label == "clean" else "present_messy"


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    total = correct = 0
    by_type: dict[str, dict] = {}
    class_confusion: dict[str, dict[str, int]] = {c: {c2: 0 for c2 in CLASSES} for c in CLASSES}
    stage1_correct = stage1_total = 0

    for r in results:
        pred_cls = r.get("predicted_class")
        if pred_cls is None:
            continue
        gt = r["gt_class"]
        ct = r["change_type"]

        total += 1
        if pred_cls == gt:
            correct += 1

        class_confusion[gt][pred_cls] += 1

        # Stage 1: model should always say yes (element always present in gt)
        stage1_total += 1
        if r.get("stage1_detected") is True:
            stage1_correct += 1

        by_type.setdefault(ct, {"total": 0, "correct": 0})
        by_type[ct]["total"] += 1
        if pred_cls == gt:
            by_type[ct]["correct"] += 1

    acc = correct / total if total else 0.0
    stage1_acc = stage1_correct / stage1_total if stage1_total else 0.0

    per_type = {
        ct: {
            "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0.0,
            "correct": v["correct"],
            "total": v["total"],
        }
        for ct, v in by_type.items()
    }

    return {
        "accuracy": round(acc, 4),
        "stage1_accuracy": round(stage1_acc, 4),
        "n_images": total,
        "class_confusion": class_confusion,
        "per_change_type": per_type,
    }


# ── Core runner ───────────────────────────────────────────────────────────────

def run_env_monitoring_benchmark(
    cfg: BenchmarkConfig,
    model_filter: list[str] | None = None,
) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(DATASET_CSV) as f:
        samples = list(csv.DictReader(f))

    for s in samples:
        s["gt_class"] = gt_class(s["label"])

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
            gt = sample["gt_class"]

            if not Path(image_path).exists():
                print(f"  [{sid}] WARNING: image not found")
                model_results.append({**sample, "predicted_class": None,
                                       "stage1_raw": "", "stage2_raw": "",
                                       "stage1_detected": None, "stage2_ready": None,
                                       "description": "", "error": "image not found",
                                       "latency_ms": 0, "model_name": model.name})
                continue

            # ── Stage 1: presence detection ───────────────────────────────────
            r1 = model.run(image_path, STAGE1[ct])
            if r1.error:
                print(f"  [{sid}] STAGE1 ERROR: {r1.error}")
                model_results.append({**sample, "predicted_class": None,
                                       "stage1_raw": "", "stage2_raw": "",
                                       "stage1_detected": None, "stage2_ready": None,
                                       "description": "", "error": r1.error,
                                       "latency_ms": 0, "model_name": model.name})
                continue

            stage1_detected = parse_yes_no(r1.response)
            total_latency = r1.latency_ms

            # ── Stage 2: readiness assessment (only if detected) ──────────────
            stage2_ready = None
            stage2_raw = ""
            description = ""

            if stage1_detected:
                r2 = model.run(image_path, STAGE2[ct])
                if not r2.error:
                    stage2_ready = parse_ready(r2.response)
                    stage2_raw = r2.response
                    description = extract_description(r2.response)
                    total_latency += r2.latency_ms

            pred_cls = classify(stage1_detected, stage2_ready)
            correct = pred_cls == gt
            mark = "✓" if correct else "✗"

            s1_str = {True: "yes", False: "no", None: "?"}[stage1_detected]
            s2_str = {True: "ready", False: "not_ready", None: "-"}[stage2_ready]
            print(
                f"  [{sid}] {mark} [{ct}] "
                f"s1={s1_str} s2={s2_str} → {pred_cls} (gt={gt}) "
                f"{total_latency:.0f}ms"
            )
            if description:
                print(f"    → {description[:130]}")

            model_results.append({
                **sample,
                "predicted_class": pred_cls,
                "stage1_raw": r1.response,
                "stage1_detected": stage1_detected,
                "stage2_raw": stage2_raw,
                "stage2_ready": stage2_ready,
                "description": description,
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

        print(f"\n  accuracy={metrics['accuracy']:.1%}  stage1={metrics['stage1_accuracy']:.1%}")
        for ct, m in sorted(metrics["per_change_type"].items()):
            print(f"  [{ct}] {m['accuracy']:.1%} ({m['correct']}/{m['total']})")

        # Incremental save after each model
        checkpoint = {"timestamp": timestamp, "dataset": str(DATASET_CSV), "models": all_results}
        ckpt_path = cfg.output_dir / f"env_monitoring_results_{timestamp}.json"
        ckpt_path.write_text(json.dumps(checkpoint, indent=2))

    # ── Final save + reports ──────────────────────────────────────────────────
    out = {"timestamp": timestamp, "dataset": str(DATASET_CSV), "models": all_results}
    json_path = cfg.output_dir / f"env_monitoring_results_{timestamp}.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults → {json_path}")

    print_summary(all_results)

    html_path = cfg.output_dir / f"env_monitoring_report_{timestamp}.html"
    save_html(all_results, html_path, timestamp)
    print(f"Report  → {html_path}")


# ── Console summary ───────────────────────────────────────────────────────────

def print_summary(all_results: dict) -> None:
    categories = sorted({
        ct for data in all_results.values()
        for ct in data["metrics"]["per_change_type"]
    })

    print(f"\n{'='*70}\nENVIRONMENT MONITORING — TWO-STAGE PIPELINE SUMMARY\n{'='*70}")
    col_w = max(len(ct) for ct in categories)
    header = f"{'Model':<38} {'Overall':>8} {'S1-Det':>7}  " + "  ".join(f"{ct:>{col_w}}" for ct in categories)
    print(header)
    print("-" * len(header))
    for key, data in all_results.items():
        m = data["metrics"]
        row = (
            f"{data['model_name']:<38} "
            f"{m['accuracy']:>7.1%} "
            f"{m['stage1_accuracy']:>7.1%}  "
        )
        for ct in categories:
            ct_m = m["per_change_type"].get(ct, {})
            row += f"  {ct_m.get('accuracy', 0):>{col_w}.1%}"
        print(row)


# ── HTML report ───────────────────────────────────────────────────────────────

_CLASS_COLOR = {
    "present_clean": "#3d9",
    "present_messy": "#e54",
    "not_present":   "#888",
    "uncertain":     "#fa0",
}


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


def save_html(all_results: dict, path: Path, timestamp: str) -> None:
    model_keys = list(all_results.keys())
    categories = sorted({
        ct for data in all_results.values()
        for ct in data["metrics"]["per_change_type"]
    })

    # ── Summary table ─────────────────────────────────────────────────────────
    summary_rows = ""
    for key, data in all_results.items():
        m = data["metrics"]
        oc = _acc_color(m["accuracy"])
        sc = _acc_color(m["stage1_accuracy"])
        cat_cells = "".join(
            f"<td><span style='color:{_acc_color(m['per_change_type'].get(ct,{}).get('accuracy',0))};font-weight:600'>"
            f"{m['per_change_type'].get(ct,{}).get('accuracy',0):.1%}</span>"
            f"<span style='color:#444;font-size:10px'> "
            f"({m['per_change_type'].get(ct,{}).get('correct',0)}/{m['per_change_type'].get(ct,{}).get('total',0)})"
            f"</span></td>"
            for ct in categories
        )
        summary_rows += (
            f"<tr><td>{data['model_name']}</td>"
            f"<td><span style='color:{oc};font-weight:700'>{m['accuracy']:.1%}</span></td>"
            f"<td><span style='color:{sc};font-weight:700'>{m['stage1_accuracy']:.1%}</span></td>"
            f"{cat_cells}</tr>"
        )

    # ── Per-image rows ────────────────────────────────────────────────────────
    all_ids = []
    for data in all_results.values():
        for r in data["results"]:
            if r["image_filename"] not in all_ids:
                all_ids.append(r["image_filename"])

    lookup: dict[str, dict] = {}
    for key, data in all_results.items():
        for r in data["results"]:
            lookup.setdefault(r["image_filename"], {})[key] = r

    model_th = "".join(f"<th>{all_results[k]['model_name']}</th>" for k in model_keys)

    detail_rows = ""
    for fid in all_ids:
        row_data = lookup.get(fid, {})
        any_r = next(iter(row_data.values()), {})
        img_src = _img_b64(any_r.get("image_path", ""))
        img_html = (
            f'<img src="{img_src}" style="width:140px;height:100px;object-fit:cover;'
            f'border-radius:4px;border:1px solid #2a2a2a;display:block;margin-bottom:4px">'
            if img_src else '<div style="width:140px;height:100px;background:#1a1a1a;border-radius:4px"></div>'
        )
        gt = any_r.get("gt_class", "")
        ct = any_r.get("change_type", "")
        label = any_r.get("label", "")
        gt_color = _CLASS_COLOR.get(gt, "#888")
        gt_html = (
            f"<div style='font-size:11px;font-weight:700;color:{gt_color}'>{gt}</div>"
            f"<div style='font-size:10px;color:#555;margin-top:2px'>{ct} · {label}</div>"
        )

        detail_rows += (
            f"<tr><td style='min-width:155px;vertical-align:top'>{img_html}"
            f"<span style='font-size:9px;color:#444'>{fid[:30]}</span></td>"
            f"<td style='vertical-align:top'>{gt_html}</td>"
        )

        for key in model_keys:
            r = row_data.get(key, {})
            pred_cls = r.get("predicted_class")
            s1 = r.get("stage1_detected")
            s2 = r.get("stage2_ready")
            desc = r.get("description", "")
            lat = r.get("latency_ms", "—")
            err = r.get("error")

            if pred_cls is None:
                detail_rows += f"<td style='color:#555;font-size:11px'>{'⚠ ' + str(err) if err else '—'}</td>"
                continue

            correct = pred_cls == gt
            pred_color = _CLASS_COLOR.get(pred_cls, "#888")
            bg = "#0d1f12" if correct else "#1f0d0d"
            wrong_mark = '' if correct else '<span style="color:#e54;font-size:9px"> ✗</span>'

            s1_str = {True: "yes", False: "no", None: "?"}[s1]
            s2_str = {True: "ready", False: "not_ready", None: "—"}[s2]
            s1_color = "#3d9" if s1 else "#e54"
            s2_color = "#3d9" if s2 else ("#e54" if s2 is False else "#666")

            cell = (
                f"<div style='padding:3px 6px;border-radius:3px;background:{bg};margin-bottom:4px'>"
                f"<span style='color:{pred_color};font-weight:700;font-size:11px'>{pred_cls}</span>"
                f"{wrong_mark}</div>"
                f"<div style='font-size:10px;color:#666'>"
                f"s1: <span style='color:{s1_color}'>{s1_str}</span> · "
                f"s2: <span style='color:{s2_color}'>{s2_str}</span></div>"
            )
            if desc:
                cell += f"<div style='font-size:9px;color:#555;font-style:italic;margin-top:3px'>{desc[:180]}</div>"
            cell += f"<div style='color:#444;font-size:10px;margin-top:3px'>{lat}ms</div>"

            detail_rows += f"<td style='vertical-align:top'>{cell}</td>"

        detail_rows += "</tr>"

    cat_ths = "".join(f"<th>{ct}</th>" for ct in categories)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Env Monitoring — Two-Stage Pipeline — {timestamp}</title>
<style>
  body {{ font-family: 'Courier New', monospace; background: #0d0d0d; color: #ccc; padding: 2rem; }}
  h1 {{ color: #fff; font-size: 1.4rem; letter-spacing: 3px; text-transform: uppercase; }}
  h2 {{ color: #888; font-size: .9rem; letter-spacing: 2px; text-transform: uppercase;
        border-bottom: 1px solid #222; padding-bottom: .4rem; margin-top: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; font-size: .82rem; }}
  th {{ background: #161616; color: #aaa; padding: 7px 12px; text-align: left; border-bottom: 1px solid #2a2a2a; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #1a1a1a; vertical-align: top; }}
  tr:hover td {{ background: #111; }}
  .tag {{ display:inline-block; background:#1a1a1a; color:#666; padding: 2px 10px; border-radius: 3px; font-size: .75rem; }}
  .note {{ font-size: 11px; color: #555; margin-top: 6px; font-style: italic; }}
  .cls {{ display:inline-block; padding: 1px 6px; border-radius: 3px; font-size: 10px; font-weight: 600; }}
</style>
</head>
<body>
<h1>Environment Monitoring — Two-Stage Pipeline</h1>
<p class="tag">{timestamp}</p>
<p class="note">Stage 1: presence detection · Stage 2: readiness assessment · 4 classes: not_present / present_clean / present_messy / uncertain</p>

<h2>Summary</h2>
<table>
  <tr><th>Model</th><th>Overall</th><th>S1 Detection</th>{cat_ths}</tr>
  {summary_rows}
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
    parser = argparse.ArgumentParser(description="Two-stage environment monitoring benchmark")
    parser.add_argument("--config", default="benchmark_config.yaml")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_filter = args.models if args.models else None

    print(f"Config:  {args.config}")
    print(f"Dataset: {DATASET_CSV}")
    print(f"Models:  {model_filter or [m.key for m in cfg.enabled_models]}")

    run_env_monitoring_benchmark(cfg, model_filter=model_filter)
