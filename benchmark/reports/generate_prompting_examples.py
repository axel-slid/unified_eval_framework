#!/usr/bin/env python
"""
generate_prompting_examples.py — Per-image examples report for prompting-techniques benchmark.

Shows every room image as a card. Each card contains:
  - The room photo
  - Ground truth checklist
  - A (model × technique) matrix of per-item predictions, colour-coded correct/wrong

Usage:
    cd benchmark
    python generate_prompting_examples.py
    python generate_prompting_examples.py --merge-dir results/prompting_run_20260406_171115
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

MODEL_ACCENTS = [
    "#58a6ff", "#f78166", "#7ee787", "#d2a8ff",
    "#ffa657", "#79c0ff", "#ff7b72", "#56d364",
]
TECHNIQUE_LABELS = {
    "direct":            "Direct",
    "cot":               "CoT",
    "few_shot":          "Few-Shot",
    "few_shot_per_item": "FS/Item",
}
TECHNIQUE_COLORS = {
    "direct":            "#58a6ff",
    "cot":               "#7ee787",
    "few_shot":          "#f78166",
    "few_shot_per_item": "#d2a8ff",
}


def img_b64(path: str) -> str | None:
    try:
        ext = Path(path).suffix.lower().strip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")
        return f"data:{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode()}"
    except Exception:
        return None


def load_merge_dir(merge_dir: Path) -> dict:
    """Load all per-model JSONs from a prompting_run_* directory and merge."""
    files = sorted(merge_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files in {merge_dir}")

    merged: dict = {}
    checklist = None
    techniques = None
    timestamp = merge_dir.name

    for f in files:
        d = json.loads(f.read_text())
        if checklist is None:
            checklist = d["checklist"]
            techniques = d["techniques"]
        for model_key, model_data in d["models"].items():
            merged[model_key] = model_data

    return {
        "timestamp": timestamp,
        "checklist": checklist,
        "techniques": techniques,
        "models": merged,
    }


def build_html(data: dict) -> str:
    checklist = data["checklist"]
    techniques = data["techniques"]
    model_keys = list(data["models"].keys())
    accents = {k: MODEL_ACCENTS[i % len(MODEL_ACCENTS)] for i, k in enumerate(model_keys)}

    item_ids = [str(c["id"]) for c in checklist]
    item_short = [c["item"][:28] for c in checklist]  # abbreviated labels

    # ── Collect all sample IDs and image paths ────────────────────────────────
    sample_ids: list[str] = []
    # lookup: sample_id → model_key → technique → result dict
    lookup: dict[str, dict[str, dict[str, dict]]] = {}

    first_tech = techniques[0]
    for model_key, model_data in data["models"].items():
        tech_data = model_data.get(first_tech, {})
        for r in tech_data.get("results", []):
            sid = r["id"]
            if sid not in lookup:
                sample_ids.append(sid)
                lookup[sid] = {}

    for model_key, model_data in data["models"].items():
        for technique in techniques:
            tech_data = model_data.get(technique, {})
            for r in tech_data.get("results", []):
                sid = r["id"]
                lookup.setdefault(sid, {}).setdefault(model_key, {})[technique] = r

    # ── Summary table ─────────────────────────────────────────────────────────
    def acc_color(a: float) -> str:
        if a >= 0.75: return "#3d9"
        if a >= 0.55: return "#fa0"
        return "#e54"

    summary_rows = ""
    for model_key in model_keys:
        model_data = data["models"][model_key]
        model_name = next(
            iter(model_data.values()), {}
        ).get("model_name", model_key)
        for technique in techniques:
            m = model_data.get(technique, {}).get("metrics", {})
            if not m:
                continue
            ia, ra, f1 = m.get("item_accuracy", 0), m.get("room_accuracy", 0), m.get("room_f1", 0)
            pe = m.get("parse_errors", 0)
            lat = m.get("mean_latency_ms", 0)
            tc = TECHNIQUE_COLORS.get(technique, "#888")
            summary_rows += (
                f"<tr>"
                f"<td style='border-left:4px solid {accents[model_key]};padding-left:8px;color:#ccc'>{model_name}</td>"
                f"<td><span style='color:{tc};font-size:10px'>{TECHNIQUE_LABELS.get(technique, technique)}</span></td>"
                f"<td><span style='color:{acc_color(ia)};font-weight:700'>{ia:.0%}</span></td>"
                f"<td><span style='color:{acc_color(ra)};font-weight:700'>{ra:.0%}</span></td>"
                f"<td><span style='color:{acc_color(f1)};font-weight:700'>{f1:.3f}</span></td>"
                f"<td style='color:{'#e54' if pe else '#3d9'}'>{pe}</td>"
                f"<td style='color:#555'>{lat:.0f}ms</td>"
                f"</tr>"
            )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_models = "".join(
        f"<span style='display:inline-block;margin:0 10px 6px 0;padding:3px 10px;"
        f"border-radius:3px;border-left:4px solid {accents[k]};background:#111;"
        f"font-size:11px;color:#ccc'>"
        f"{data['models'][k].get(techniques[0], {}).get('model_name', k)}</span>"
        for k in model_keys
    )
    legend_techniques = "".join(
        f"<span style='display:inline-block;margin:0 10px 6px 0;padding:3px 10px;"
        f"border-radius:3px;background:{TECHNIQUE_COLORS.get(t,'#444')}22;"
        f"border:1px solid {TECHNIQUE_COLORS.get(t,'#444')}55;"
        f"font-size:11px;color:{TECHNIQUE_COLORS.get(t,'#888')}'>"
        f"{TECHNIQUE_LABELS.get(t, t)}</span>"
        for t in techniques
    )

    # ── Per-image cards ───────────────────────────────────────────────────────
    cards_html = ""
    for sid in sample_ids:
        row_data = lookup.get(sid, {})

        # Get image path and GT from any result
        any_r = None
        for mk in model_keys:
            for t in techniques:
                any_r = row_data.get(mk, {}).get(t)
                if any_r:
                    break
            if any_r:
                break

        if not any_r:
            continue

        image_path = any_r.get("image", "")
        gt = any_r.get("ground_truth", {})
        gt_items = gt.get("items", {})
        gt_ready = gt.get("room_ready", False)

        # Image
        src = img_b64(image_path)
        img_html = (
            f'<img src="{src}" style="width:100%;height:180px;object-fit:cover;'
            f'border-radius:4px;display:block;margin-bottom:8px">'
            if src else
            '<div style="width:100%;height:180px;background:#1a1a1a;border-radius:4px;margin-bottom:8px;'
            'display:flex;align-items:center;justify-content:center;color:#333;font-size:10px">no image</div>'
        )

        # Ground truth block
        gt_ready_color = "#3d9" if gt_ready else "#e54"
        gt_items_html = "".join(
            f"<div style='font-size:9px;color:{'#3d9' if gt_items.get(iid) else '#666'};margin:1px 0'>"
            f"{'✓' if gt_items.get(iid) else '✗'} {lbl}</div>"
            for iid, lbl in zip(item_ids, item_short)
        )
        gt_block = (
            f"<div style='margin-bottom:10px;padding:6px 8px;background:#0d0d0d;"
            f"border-radius:4px;border:1px solid #1e1e1e'>"
            f"<div style='font-size:9px;color:#555;margin-bottom:4px;text-transform:uppercase;letter-spacing:1px'>Ground Truth</div>"
            f"{gt_items_html}"
            f"<div style='margin-top:5px;font-size:10px;font-weight:700;color:{gt_ready_color}'>"
            f"{'READY' if gt_ready else 'NOT READY'}</div>"
            f"</div>"
        )

        # Matrix: rows=models, cols=techniques
        # Header row
        matrix_header = (
            "<tr><td style='font-size:9px;color:#444;padding:3px 6px'>Model</td>"
            + "".join(
                f"<td style='font-size:9px;color:{TECHNIQUE_COLORS.get(t,'#888')};padding:3px 6px;"
                f"text-align:center'>{TECHNIQUE_LABELS.get(t,t)}</td>"
                for t in techniques
            )
            + "</tr>"
        )

        matrix_rows = ""
        for model_key in model_keys:
            model_data = data["models"][model_key]
            model_name = model_data.get(techniques[0], {}).get("model_name", model_key)
            short_name = model_name.split("(")[0].strip()[:18]
            accent = accents[model_key]

            row_cells = (
                f"<td style='font-size:9px;color:{accent};padding:3px 6px;"
                f"white-space:nowrap;border-left:2px solid {accent}22'>{short_name}</td>"
            )

            for technique in techniques:
                r = row_data.get(model_key, {}).get(technique)
                if not r or r.get("predicted") is None:
                    err = r.get("parse_error", "—") if r else "—"
                    row_cells += f"<td style='padding:3px 4px;text-align:center'><span style='font-size:9px;color:#333'>{err[:10]}</span></td>"
                    continue

                pred = r["predicted"]
                pred_items = pred.get("items", {})
                pred_ready = pred.get("room_ready", False)

                # Build mini checklist dots
                dots = ""
                n_correct = 0
                for iid in item_ids:
                    p = pred_items.get(iid, False)
                    g = gt_items.get(iid, False)
                    match = p == g
                    if match:
                        n_correct += 1
                    dot_color = "#3d9" if (p and match) else "#1a5c1a" if (not p and match) else "#e54" if p else "#5c1a1a"
                    dots += f"<span style='color:{dot_color};font-size:11px'>{'●' if p else '○'}</span>"

                ready_correct = pred_ready == gt_ready
                ready_color = "#3d9" if pred_ready else "#e54"
                ready_bg = "#0d1f12" if ready_correct else "#1f0d0d"
                lat = r.get("latency_ms", "")

                cell_bg = "#0c150c" if n_correct == len(item_ids) else "#0d0d0d"
                row_cells += (
                    f"<td style='padding:4px 4px;background:{cell_bg};'>"
                    f"<div style='font-size:11px;letter-spacing:1px;margin-bottom:2px'>{dots}</div>"
                    f"<div style='font-size:9px;font-weight:700;color:{ready_color};"
                    f"background:{ready_bg};border-radius:2px;padding:1px 3px;display:inline-block'>"
                    f"{'RDY' if pred_ready else 'NOT'}</div>"
                    f"<div style='font-size:8px;color:#333;margin-top:1px'>{lat}ms</div>"
                    f"</td>"
                )

            matrix_rows += f"<tr>{row_cells}</tr>"

        matrix_html = (
            f"<table style='width:100%;border-collapse:collapse;font-size:9px'>"
            f"{matrix_header}{matrix_rows}"
            f"</table>"
        )

        # Dot legend
        dot_legend = (
            "<div style='font-size:8px;color:#444;margin-top:6px'>"
            "<span style='color:#3d9'>●</span>=pred✓&amp;correct "
            "<span style='color:#1a5c1a'>○</span>=pred✗&amp;correct "
            "<span style='color:#e54'>●</span>=pred✓&amp;wrong "
            "<span style='color:#5c1a1a'>○</span>=pred✗&amp;wrong"
            "</div>"
        )

        is_clean = "clean" in sid
        card_border = "#1a3a1a" if is_clean else "#3a1a1a"
        card_label_color = "#3d9" if is_clean else "#e54"
        card_label = "CLEAN" if is_clean else "MESSY"

        cards_html += (
            f"<div style='background:#111;border-radius:6px;padding:12px;"
            f"border:1px solid {card_border};break-inside:avoid;margin-bottom:16px'>"
            f"{img_html}"
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px'>"
            f"<span style='font-size:10px;color:#555'>{sid}</span>"
            f"<span style='font-size:10px;font-weight:700;color:{card_label_color}'>{card_label}</span>"
            f"</div>"
            f"{gt_block}"
            f"{matrix_html}"
            f"{dot_legend}"
            f"</div>"
        )

    ts = data.get("timestamp", "")

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Prompting Examples — {ts}</title>
<style>
  body {{ font-family: 'Courier New', monospace; background: #0d0d0d; color: #ccc;
          padding: 2rem; max-width: 1600px; margin: 0 auto; }}
  h1 {{ color: #fff; font-size: 1.3rem; letter-spacing: 3px; text-transform: uppercase; margin-bottom: .3rem; }}
  h2 {{ color: #888; font-size: .85rem; letter-spacing: 2px; text-transform: uppercase;
        border-bottom: 1px solid #222; padding-bottom: .4rem; margin: 2rem 0 1rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; font-size: .8rem; }}
  th {{ background: #161616; color: #888; padding: 6px 10px; text-align: left; border-bottom: 1px solid #222; }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #181818; vertical-align: top; }}
  .note {{ font-size: 11px; color: #444; margin: .3rem 0 1.2rem; }}
  .grid {{ columns: 3; column-gap: 16px; }}
  @media (max-width: 1200px) {{ .grid {{ columns: 2; }} }}
  @media (max-width: 700px)  {{ .grid {{ columns: 1; }} }}
</style>
</head>
<body>
<h1>Meeting Room Readiness — Examples Report</h1>
<p class="note">{ts} · {len(model_keys)} models · {len(techniques)} techniques · {len(sample_ids)} images</p>

<div style="margin-bottom:.6rem">{legend_models}</div>
<div style="margin-bottom:1.2rem">{legend_techniques}</div>

<h2>Summary</h2>
<table>
  <tr>
    <th>Model</th><th>Technique</th>
    <th>ItemAcc</th><th>RoomAcc</th><th>RoomF1</th>
    <th>ParseErr</th><th>Latency</th>
  </tr>
  {summary_rows}
</table>

<h2>Per-Image Results</h2>
<p class="note">
  Each card: dots = checklist items 1–5 &nbsp;|&nbsp;
  ● pred=Yes &nbsp; ○ pred=No &nbsp;|&nbsp;
  green = correct &nbsp; red = wrong &nbsp;|&nbsp;
  RDY / NOT = room-ready verdict
</p>
<div class="grid">
{cards_html}
</div>
</body>
</html>"""


def find_latest_merge_dir() -> Path:
    dirs = sorted(RESULTS_DIR.glob("prompting_run_*/"))
    if not dirs:
        raise FileNotFoundError("No prompting_run_* directories found in results/")
    return dirs[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge-dir", default=None, help="Path to prompting_run_* directory")
    args = parser.parse_args()

    merge_dir = Path(args.merge_dir) if args.merge_dir else find_latest_merge_dir()
    print(f"Loading from: {merge_dir}")

    data = load_merge_dir(merge_dir)
    html = build_html(data)

    ts = data["timestamp"]
    out = RESULTS_DIR / f"prompting_examples_{ts}.html"
    out.write_text(html)
    print(f"Report → {out}")


if __name__ == "__main__":
    main()
