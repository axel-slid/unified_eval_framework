#!/usr/bin/env python
"""
generate_examples_report.py — Rich per-image examples report.

Reads the latest env_monitoring_results_*.json checkpoint and generates
an HTML showing every image with each model's prediction, correctness,
and full rationale — colour-coded per model and per correctness.

Usage:
    python generate_examples_report.py
    python generate_examples_report.py --results results/env_monitoring_results_XYZ.json
"""

from __future__ import annotations

import argparse
import base64
import json
import glob
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Distinct accent colours per model slot (up to 8)
MODEL_ACCENTS = ["#58a6ff", "#f78166", "#7ee787", "#d2a8ff", "#ffa657", "#79c0ff", "#ff7b72", "#56d364"]

CLASS_COLOR = {
    "present_clean": "#3d9",
    "present_messy": "#e54",
    "not_present":   "#999",
    "uncertain":     "#fa0",
}


def img_b64(path: str) -> str | None:
    try:
        ext = Path(path).suffix.lower().strip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")
        return f"data:{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode()}"
    except Exception:
        return None


def load_latest(path: str | None) -> dict:
    if path:
        return json.loads(Path(path).read_text())
    files = sorted(glob.glob(str(RESULTS_DIR / "env_monitoring_results_*.json")))
    if not files:
        raise FileNotFoundError("No env_monitoring_results_*.json found in results/")
    return json.loads(Path(files[-1]).read_text())


def build_html(data: dict) -> str:
    model_keys = list(data["models"].keys())
    model_names = {k: data["models"][k]["model_name"] for k in model_keys}
    accents = {k: MODEL_ACCENTS[i % len(MODEL_ACCENTS)] for i, k in enumerate(model_keys)}

    # Build lookup: image_filename → model_key → result
    lookup: dict[str, dict[str, dict]] = {}
    all_filenames: list[str] = []
    for k, v in data["models"].items():
        for r in v["results"]:
            fn = r["image_filename"]
            if fn not in lookup:
                lookup[fn] = {}
                all_filenames.append(fn)
            lookup[fn][k] = r

    # Group by change_type
    by_category: dict[str, list[str]] = {}
    for fn in all_filenames:
        any_r = next(iter(lookup[fn].values()))
        ct = any_r.get("change_type", "unknown")
        by_category.setdefault(ct, []).append(fn)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = "".join(
        f"<span style='display:inline-block;margin:0 12px 6px 0;"
        f"padding:3px 10px;border-radius:3px;border-left:4px solid {accents[k]};"
        f"background:#111;font-size:11px;color:#ccc'>{model_names[k]}</span>"
        for k in model_keys
    )

    # ── Summary bar per model ─────────────────────────────────────────────────
    summary_rows = ""
    for k in model_keys:
        m = data["models"][k]["metrics"]
        pt = m["per_change_type"]
        acc_color = lambda a: "#3d9" if a >= 0.8 else "#fa0" if a >= 0.5 else "#e54"
        cats = ["blinds", "chairs", "table", "whiteboard"]
        cat_cells = "".join(
            f"<td><span style='color:{acc_color(pt.get(c,{}).get('accuracy',0))};font-weight:600'>"
            f"{pt.get(c,{}).get('accuracy',0):.0%}</span></td>"
            for c in cats
        )
        summary_rows += (
            f"<tr>"
            f"<td style='border-left:4px solid {accents[k]};padding-left:8px;color:#ccc'>{model_names[k]}</td>"
            f"<td><span style='color:{acc_color(m['accuracy'])};font-weight:700'>{m['accuracy']:.0%}</span></td>"
            f"<td style='color:#888'>{m['stage1_accuracy']:.0%}</td>"
            f"{cat_cells}"
            f"</tr>"
        )

    # ── Per-category image cards ──────────────────────────────────────────────
    category_sections = ""
    for ct in sorted(by_category.keys()):
        fns = by_category[ct]
        # Sort: clean first, then messy
        fns = sorted(fns, key=lambda fn: (lookup[fn][model_keys[0]].get("label", ""), fn))

        cards = ""
        for fn in fns:
            row_data = lookup[fn]
            any_r = next(iter(row_data.values()))
            label = any_r.get("label", "")
            gt = any_r.get("gt_class", "")
            image_path = any_r.get("image_path", "")

            src = img_b64(image_path)
            img_tag = (
                f'<img src="{src}" style="width:100%;height:160px;object-fit:cover;'
                f'border-radius:4px;display:block;margin-bottom:8px">'
                if src else
                '<div style="width:100%;height:160px;background:#1a1a1a;border-radius:4px;margin-bottom:8px"></div>'
            )

            gt_color = CLASS_COLOR.get(gt, "#888")
            card_header = (
                f"<div style='margin-bottom:8px'>"
                f"<span style='font-size:10px;color:#555'>{fn[:38]}</span><br>"
                f"<span style='font-size:11px;font-weight:700;color:{gt_color}'>GT: {gt}</span>"
                f"<span style='font-size:10px;color:#555'> · {label}</span>"
                f"</div>"
            )

            model_blocks = ""
            for k in model_keys:
                r = row_data.get(k)
                if not r:
                    continue
                pred = r.get("predicted_class")
                s1 = r.get("stage1_detected")
                s2 = r.get("stage2_ready")
                desc = r.get("description", "") or r.get("stage2_raw", "")[:200]
                lat = r.get("latency_ms", "—")
                correct = pred == gt

                accent = accents[k]
                bg = "#0c1a0c" if correct else "#1a0c0c"
                pred_color = CLASS_COLOR.get(pred, "#888")
                s1_str = {True: "s1:yes", False: "s1:no", None: "s1:?"}[s1]
                s2_str = {True: "s2:ready", False: "s2:not_ready", None: ""}[s2] if s2 is not None else ""
                tick = "✓" if correct else "✗"
                tick_color = "#3d9" if correct else "#e54"

                model_blocks += (
                    f"<div style='border-left:3px solid {accent};padding:6px 8px;"
                    f"margin-bottom:6px;background:{bg};border-radius:0 4px 4px 0'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:3px'>"
                    f"<span style='font-size:10px;color:{accent};font-weight:700'>{model_names[k]}</span>"
                    f"<span style='font-size:12px;color:{tick_color};font-weight:700'>{tick}</span>"
                    f"</div>"
                    f"<div style='font-size:11px;font-weight:700;color:{pred_color};margin-bottom:2px'>{pred or '—'}</div>"
                    f"<div style='font-size:9px;color:#555;margin-bottom:4px'>{s1_str}{' · ' + s2_str if s2_str else ''} · {lat}ms</div>"
                    "<div style='font-size:9px;color:#777;font-style:italic;line-height:1.4'>"
                    + (desc[:220] if desc else "<span style='color:#333'>no description</span>")
                    + "</div>"
                    f"</div>"
                )

            cards += (
                f"<div style='background:#111;border-radius:6px;padding:12px;"
                f"border:1px solid #1e1e1e;break-inside:avoid'>"
                f"{img_tag}{card_header}{model_blocks}"
                f"</div>"
            )

        category_sections += (
            f"<h2>{ct.upper()}</h2>"
            f"<div style='columns:3;column-gap:16px;margin-bottom:2rem'>{cards}</div>"
        )

    ts = data.get("timestamp", "")
    n_models = len(model_keys)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Examples Report — {ts}</title>
<style>
  body {{ font-family: 'Courier New', monospace; background: #0d0d0d; color: #ccc; padding: 2rem; max-width: 1400px; margin: 0 auto; }}
  h1 {{ color: #fff; font-size: 1.3rem; letter-spacing: 3px; text-transform: uppercase; margin-bottom: .3rem; }}
  h2 {{ color: #888; font-size: .85rem; letter-spacing: 2px; text-transform: uppercase;
        border-bottom: 1px solid #222; padding-bottom: .4rem; margin: 2rem 0 1rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; font-size: .8rem; }}
  th {{ background: #161616; color: #888; padding: 6px 10px; text-align: left; border-bottom: 1px solid #222; }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #181818; }}
  .note {{ font-size: 11px; color: #444; margin: .3rem 0 1.2rem; }}
</style>
</head>
<body>
<h1>Environment Monitoring — Examples Report</h1>
<p class="note">{ts} · {n_models} models · two-stage pipeline</p>

<div style="margin-bottom:1.2rem">{legend_items}</div>

<h2>Summary</h2>
<table>
  <tr><th>Model</th><th>Overall</th><th>S1 Det</th><th>blinds</th><th>chairs</th><th>table</th><th>whiteboard</th></tr>
  {summary_rows}
</table>

{category_sections}
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=None, help="Path to specific results JSON")
    args = parser.parse_args()

    data = load_latest(args.results)
    html = build_html(data)

    ts = data.get("timestamp", "latest")
    out = RESULTS_DIR / f"env_monitoring_examples_{ts}.html"
    out.write_text(html)
    print(f"Report → {out}")


if __name__ == "__main__":
    main()
