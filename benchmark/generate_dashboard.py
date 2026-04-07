#!/usr/bin/env python
"""
generate_dashboard.py — Self-contained HTML dashboard for env monitoring benchmark.

Embeds matplotlib charts (heatmap, bars, confusion matrices) as base64 PNGs
alongside metrics tables in a single dark-themed HTML file.

Usage:
    python generate_dashboard.py
    python generate_dashboard.py --results results/env_monitoring_results_XYZ.json
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
CATEGORIES = ["blinds", "chairs", "table", "whiteboard"]
CLASSES = ["not_present", "present_clean", "present_messy", "uncertain"]

MODEL_COLORS = ["#58a6ff", "#f78166", "#7ee787", "#d2a8ff", "#ffa657", "#79c0ff", "#ff7b72", "#56d364"]

BG = "#0d0d0d"
PANEL_BG = "#111111"
TEXT_COLOR = "#cccccc"
GRID_COLOR = "#222222"

CLASS_COLOR = {
    "present_clean": "#3d9",
    "present_messy": "#e54",
    "not_present": "#999",
    "uncertain": "#fa0",
}


def load_latest(path: str | None) -> dict:
    if path:
        return json.loads(Path(path).read_text())
    files = sorted(glob.glob(str(RESULTS_DIR / "env_monitoring_results_*.json")))
    if not files:
        raise FileNotFoundError("No results found")
    return json.loads(Path(files[-1]).read_text())


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def make_heatmap_bar_chart(data: dict, model_colors: dict) -> str:
    model_keys = list(data["models"].keys())
    model_names = [v["model_name"].replace(" (", "\n(") for v in data["models"].values()]
    model_names_short = [data["models"][k]["model_name"].split(" (")[0] for k in model_keys]

    # ── Heatmap ──────────────────────────────────────────────────────────────
    matrix = []
    for k in model_keys:
        m = data["models"][k]["metrics"]
        row = [m["per_change_type"].get(ct, {}).get("accuracy", 0.0) for ct in CATEGORIES]
        matrix.append(row)
    mat = np.array(matrix)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, len(model_keys) * 0.7 + 1.5)),
                              facecolor=BG, gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.4})

    # Heatmap
    ax = axes[0]
    cmap = LinearSegmentedColormap.from_list("rg", ["#3a0000", "#1a1a1a", "#003a00"])
    ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(CATEGORIES)))
    ax.set_xticklabels(CATEGORIES, color=TEXT_COLOR, fontsize=9)
    ax.set_yticks(range(len(model_keys)))
    ax.set_yticklabels(model_names, color=TEXT_COLOR, fontsize=8)
    ax.tick_params(colors=TEXT_COLOR, length=0)
    ax.set_facecolor(BG)
    for i in range(len(model_keys)):
        for j in range(len(CATEGORIES)):
            val = mat[i, j]
            color = "#3d9" if val >= 0.7 else "#fa0" if val >= 0.4 else "#e54"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")
    ax.set_title("Accuracy by Model & Category", color=TEXT_COLOR, fontsize=10, pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)

    # Bar chart
    ax2 = axes[1]
    accs = [data["models"][k]["metrics"]["accuracy"] for k in model_keys]
    s1_accs = [data["models"][k]["metrics"]["stage1_accuracy"] for k in model_keys]
    x = np.arange(len(model_keys))
    w = 0.35
    ax2.bar(x - w/2, accs, w, color=[model_colors[k] for k in model_keys], alpha=0.9, label="Overall Acc")
    ax2.bar(x + w/2, s1_accs, w, color=[model_colors[k] for k in model_keys], alpha=0.4, label="Stage1 Det")
    for i, (acc, s1) in enumerate(zip(accs, s1_accs)):
        ax2.text(x[i] - w/2, acc + 0.02, f"{acc:.0%}", ha="center", color=TEXT_COLOR, fontsize=7)
        ax2.text(x[i] + w/2, s1 + 0.02, f"{s1:.0%}", ha="center", color=TEXT_COLOR, fontsize=7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names_short, rotation=25, ha="right", color=TEXT_COLOR, fontsize=7)
    ax2.set_ylim(0, 1.2)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], color=TEXT_COLOR, fontsize=8)
    ax2.tick_params(colors=TEXT_COLOR, length=0)
    ax2.set_facecolor(BG)
    ax2.set_title("Overall Acc (solid) vs Stage1 Detection (faded)", color=TEXT_COLOR, fontsize=10, pad=10)
    ax2.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_COLOR, framealpha=0.8)
    ax2.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5)
    ax2.set_axisbelow(True)
    for spine in ax2.spines.values():
        spine.set_edgecolor(GRID_COLOR)

    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


def make_confusion_grid(data: dict, model_colors: dict) -> str:
    model_keys = list(data["models"].keys())
    n = len(model_keys)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * 4.5, rows * 4),
                              facecolor=BG)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx, k in enumerate(model_keys):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        cc = data["models"][k]["metrics"]["class_confusion"]
        mat = np.array([[cc.get(gt, {}).get(pred, 0) for pred in CLASSES] for gt in CLASSES], dtype=float)
        row_sums = mat.sum(axis=1, keepdims=True)
        mat_norm = np.divide(mat, row_sums, where=row_sums > 0)

        cmap = LinearSegmentedColormap.from_list("bl", ["#0a0a0a", "#1a3a1a"])
        ax.imshow(mat_norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        labels_short = ["not_pres", "clean", "messy", "uncert"]
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels_short, color=TEXT_COLOR, fontsize=7, rotation=30, ha="right")
        ax.set_yticks(range(4))
        ax.set_yticklabels(labels_short, color=TEXT_COLOR, fontsize=7)
        ax.tick_params(colors=TEXT_COLOR, length=0)
        ax.set_facecolor(BG)
        for i in range(4):
            for j in range(4):
                raw = int(mat[i, j])
                norm = mat_norm[i, j]
                color = "#3d9" if i == j else "#e54" if norm > 0.3 else TEXT_COLOR
                ax.text(j, i, f"{raw}\n{norm:.0%}", ha="center", va="center",
                        color=color, fontsize=7, fontweight="bold" if i == j else "normal")
        name = data["models"][k]["model_name"].split(" (")[0]
        ax.set_title(name, color=model_colors[k], fontsize=9, pad=6, fontweight="bold")
        ax.set_xlabel("Predicted", color=TEXT_COLOR, fontsize=7)
        ax.set_ylabel("Ground Truth", color=TEXT_COLOR, fontsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(model_colors[k])
            spine.set_linewidth(1.5)

    # Hide unused axes
    for idx in range(len(model_keys), rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].set_visible(False)

    fig.suptitle("Confusion Matrices (row = GT, col = Predicted)", color=TEXT_COLOR,
                 fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


def make_per_category_bars(data: dict, model_colors: dict) -> str:
    model_keys = list(data["models"].keys())
    model_names_short = [data["models"][k]["model_name"].split(" (")[0] for k in model_keys]

    fig, axes = plt.subplots(1, len(CATEGORIES),
                              figsize=(len(CATEGORIES) * 4, 4.5), facecolor=BG)

    for ci, cat in enumerate(CATEGORIES):
        ax = axes[ci]
        accs = [data["models"][k]["metrics"]["per_change_type"].get(cat, {}).get("accuracy", 0)
                for k in model_keys]
        totals = [data["models"][k]["metrics"]["per_change_type"].get(cat, {}).get("total", 0)
                  for k in model_keys]
        colors = [model_colors[k] for k in model_keys]
        x = np.arange(len(model_keys))
        bars = ax.bar(x, accs, color=colors, alpha=0.85)
        for bar, acc, tot in zip(bars, accs, totals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{acc:.0%}\n({int(acc*tot)}/{tot})", ha="center",
                    color=TEXT_COLOR, fontsize=6.5, linespacing=1.3)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names_short, rotation=35, ha="right",
                           color=TEXT_COLOR, fontsize=7)
        ax.set_ylim(0, 1.28)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["0%", "50%", "100%"], color=TEXT_COLOR, fontsize=8)
        ax.tick_params(colors=TEXT_COLOR, length=0)
        ax.set_facecolor(BG)
        ax.set_title(cat.upper(), color=TEXT_COLOR, fontsize=10, fontweight="bold", pad=8)
        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)

    plt.tight_layout()
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


def make_stage_breakdown(data: dict, model_colors: dict) -> str:
    """Stage 1 vs Stage 2 accuracy per model per category."""
    model_keys = list(data["models"].keys())
    model_names_short = [data["models"][k]["model_name"].split(" (")[0] for k in model_keys]

    # Compute per-model, per-category stage1 accuracy
    stage1_by_cat: dict[str, list[float]] = {cat: [] for cat in CATEGORIES}
    for k in model_keys:
        results = data["models"][k]["results"]
        for cat in CATEGORIES:
            cat_results = [r for r in results if r.get("change_type") == cat]
            if not cat_results:
                stage1_by_cat[cat].append(0.0)
                continue
            s1_correct = sum(
                1 for r in cat_results
                if (r.get("stage1_detected") is True and r.get("label") in ("clean", "messy"))
                or (r.get("stage1_detected") is False and r.get("label") == "not_present")
            )
            stage1_by_cat[cat].append(s1_correct / len(cat_results))

    n_cats = len(CATEGORIES)
    fig, axes = plt.subplots(1, n_cats, figsize=(n_cats * 4, 4.5), facecolor=BG)

    for ci, cat in enumerate(CATEGORIES):
        ax = axes[ci]
        s1_accs = stage1_by_cat[cat]
        overall_accs = [data["models"][k]["metrics"]["per_change_type"].get(cat, {}).get("accuracy", 0)
                        for k in model_keys]
        x = np.arange(len(model_keys))
        w = 0.38
        colors = [model_colors[k] for k in model_keys]
        ax.bar(x - w/2, s1_accs, w, color=colors, alpha=0.5, label="Stage1")
        ax.bar(x + w/2, overall_accs, w, color=colors, alpha=0.9, label="Overall")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names_short, rotation=35, ha="right", color=TEXT_COLOR, fontsize=7)
        ax.set_ylim(0, 1.25)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["0%", "50%", "100%"], color=TEXT_COLOR, fontsize=8)
        ax.tick_params(colors=TEXT_COLOR, length=0)
        ax.set_facecolor(BG)
        ax.set_title(cat.upper(), color=TEXT_COLOR, fontsize=10, fontweight="bold", pad=8)
        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        if ci == 0:
            ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_COLOR, framealpha=0.8)

    plt.tight_layout()
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


def acc_color_style(acc: float) -> str:
    if acc >= 0.75:
        return f"color:#3d9;font-weight:700"
    elif acc >= 0.5:
        return f"color:#fa0;font-weight:700"
    else:
        return f"color:#e54;font-weight:700"


def build_metrics_table(data: dict, model_colors: dict) -> str:
    model_keys = list(data["models"].keys())
    header = "<tr><th>Model</th><th>Overall</th><th>Stage1 Det</th>" + \
             "".join(f"<th>{c.upper()}</th>" for c in CATEGORIES) + "</tr>"

    rows_html = ""
    for k in model_keys:
        m = data["models"][k]["metrics"]
        pt = m["per_change_type"]
        acc = m["accuracy"]
        s1 = m["stage1_accuracy"]
        accent = model_colors[k]
        name = data["models"][k]["model_name"]

        def _cat_cell(c):
            cat_acc = pt.get(c, {}).get("accuracy", 0)
            cat_cor = pt.get(c, {}).get("correct", 0)
            cat_tot = pt.get(c, {}).get("total", 0)
            style = acc_color_style(cat_acc)
            return (f"<td><span style='{style}'>{cat_acc:.0%}"
                    f"<span style='color:#555;font-weight:400;font-size:10px'>"
                    f" ({cat_cor}/{cat_tot})</span></span></td>")
        cat_cells = "".join(_cat_cell(c) for c in CATEGORIES)

        rows_html += (
            f"<tr>"
            f"<td style='border-left:4px solid {accent};padding-left:10px'>"
            f"<span style='color:{accent};font-weight:700'>{name}</span></td>"
            f"<td><span style='{acc_color_style(acc)}'>{acc:.1%}</span></td>"
            f"<td style='color:#888'>{s1:.1%}</td>"
            f"{cat_cells}"
            f"</tr>"
        )
    return f"<table>{header}{rows_html}</table>"


def build_confusion_table(data: dict, model_colors: dict) -> str:
    model_keys = list(data["models"].keys())
    html = ""
    for k in model_keys:
        cc = data["models"][k]["metrics"]["class_confusion"]
        accent = model_colors[k]
        name = data["models"][k]["model_name"].split(" (")[0]
        th_cells = "".join(
            "<th style='color:{}'>{}</th>".format(CLASS_COLOR.get(c, "#888"), c)
            for c in CLASSES
        )
        header = (
            f"<h3 style='color:{accent};margin:1.5rem 0 .5rem'>{name}</h3>"
            f"<table style='font-size:11px;margin-bottom:1rem'>"
            f"<tr><th>GT \\ Pred</th>{th_cells}</tr>"
        )
        rows_html = ""
        for gt in CLASSES:
            row_total = sum(cc.get(gt, {}).get(p, 0) for p in CLASSES)
            cells = ""
            for pred in CLASSES:
                val = cc.get(gt, {}).get(pred, 0)
                pct = val / row_total if row_total else 0
                if gt == pred:
                    style = f"background:#0c1a0c;color:#3d9;font-weight:700"
                elif pct > 0.3:
                    style = f"background:#1a0c0c;color:#e54"
                else:
                    style = f"color:#666"
                cells += f"<td style='{style}'>{val} <span style='color:#444'>({pct:.0%})</span></td>"
            gt_color = CLASS_COLOR.get(gt, "#888")
            rows_html += f"<tr><td style='color:{gt_color};font-weight:600'>{gt}</td>{cells}</tr>"
        html += header + rows_html + "</table>"
    return html


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=None)
    args = parser.parse_args()

    data = load_latest(args.results)
    model_keys = list(data["models"].keys())
    model_colors = {k: MODEL_COLORS[i % len(MODEL_COLORS)] for i, k in enumerate(model_keys)}
    ts = data.get("timestamp", "latest")
    n_models = len(model_keys)

    print("Generating heatmap + bar chart...")
    b64_top = make_heatmap_bar_chart(data, model_colors)

    print("Generating per-category bars...")
    b64_cats = make_per_category_bars(data, model_colors)

    print("Generating stage breakdown...")
    b64_stages = make_stage_breakdown(data, model_colors)

    print("Generating confusion matrices...")
    b64_conf = make_confusion_grid(data, model_colors)

    print("Building metrics tables...")
    metrics_table = build_metrics_table(data, model_colors)
    confusion_tables = build_confusion_table(data, model_colors)

    # Model legend chips
    legend = "".join(
        f"<span style='display:inline-block;margin:0 10px 6px 0;padding:4px 12px;"
        f"border-radius:4px;border-left:4px solid {model_colors[k]};"
        f"background:#161616;font-size:11px;color:#ccc'>"
        f"{data['models'][k]['model_name']}</span>"
        for k in model_keys
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Env Monitoring Dashboard — {ts}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Courier New', monospace;
    background: {BG};
    color: #ccc;
    padding: 2rem 2.5rem;
    max-width: 1600px;
    margin: 0 auto;
  }}
  h1 {{
    color: #fff;
    font-size: 1.3rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: .3rem;
  }}
  h2 {{
    color: #888;
    font-size: .8rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-bottom: 1px solid #1e1e1e;
    padding-bottom: .5rem;
    margin: 2.5rem 0 1.2rem;
  }}
  h3 {{ font-size: .85rem; }}
  .note {{ font-size: 11px; color: #444; margin: .4rem 0 1.5rem; }}
  .legend {{ margin-bottom: 1.5rem; }}
  img.chart {{ width: 100%; border-radius: 6px; margin-bottom: .5rem; border: 1px solid #1e1e1e; }}
  table {{
    border-collapse: collapse;
    width: 100%;
    font-size: .78rem;
    margin-bottom: 1.5rem;
  }}
  th {{
    background: #161616;
    color: #888;
    padding: 7px 12px;
    text-align: left;
    border-bottom: 1px solid #222;
    letter-spacing: 1px;
  }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #181818; }}
  tr:hover td {{ background: #141414; }}
  .section {{ margin-bottom: 3rem; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }}
  .confusion-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
    gap: 1.5rem;
    margin-bottom: 1.5rem;
  }}
  .confusion-card {{
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 6px;
    padding: 1rem;
  }}
</style>
</head>
<body>

<h1>Environment Monitoring — Benchmark Dashboard</h1>
<p class="note">{ts} · {n_models} models · two-stage pipeline · 60 images (blinds×10, whiteboard×10, chairs×20, table×20)</p>

<div class="legend">{legend}</div>

<!-- ── SECTION 1: Summary Overview ─────────────────────────────── -->
<div class="section">
  <h2>1 · Overview — Heatmap &amp; Overall Accuracy</h2>
  <img class="chart" src="data:image/png;base64,{b64_top}" alt="heatmap and bar chart">
</div>

<!-- ── SECTION 2: Metrics Table ────────────────────────────────── -->
<div class="section">
  <h2>2 · Metrics Summary Table</h2>
  {metrics_table}
</div>

<!-- ── SECTION 3: Per-Category Accuracy ────────────────────────── -->
<div class="section">
  <h2>3 · Per-Category Accuracy</h2>
  <img class="chart" src="data:image/png;base64,{b64_cats}" alt="per-category accuracy">
</div>

<!-- ── SECTION 4: Stage 1 vs Overall ───────────────────────────── -->
<div class="section">
  <h2>4 · Stage 1 Detection vs Overall Accuracy (per category)</h2>
  <img class="chart" src="data:image/png;base64,{b64_stages}" alt="stage breakdown">
</div>

<!-- ── SECTION 5: Confusion Matrices ───────────────────────────── -->
<div class="section">
  <h2>5 · Confusion Matrices</h2>
  <img class="chart" src="data:image/png;base64,{b64_conf}" alt="confusion matrices">
  <div class="confusion-grid">
    {confusion_tables}
  </div>
</div>

</body>
</html>"""

    out = RESULTS_DIR / f"env_monitoring_dashboard_{ts}.html"
    out.write_text(html)
    print(f"Dashboard → {out}")


if __name__ == "__main__":
    main()
