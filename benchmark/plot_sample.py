import json
import re
import textwrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

# ── Data ──────────────────────────────────────────────────────────────────────
with open("../vqa_results.json") as f:
    raw = json.load(f)

IMAGE_ID = "001"
IMAGE_PATH = "test_sets/images/001.jpg"

ORDER = [
    "gpt_baseline", "smolvlm", "internvl", "qwen3vl_4b",
    "qwen3vl_8b", "internvl_int8", "qwen3vl_4b_int8", "qwen3vl_8b_int8",
]
MODEL_LABELS = [
    "GPT-4.1-mini\n(baseline)", "SmolVLM2\n2.2B", "InternVL3\n4B",
    "Qwen3-VL\n4B", "Qwen3-VL\n8B",
    "InternVL3\n4B-int8", "Qwen3-VL\n4B-int8", "Qwen3-VL\n8B-int8",
]

def get_record(key):
    for r in raw[key]:
        if r["id"] == IMAGE_ID:
            return r

records = {k: get_record(k) for k in ORDER}

# questions from baseline
questions = [q["question"] for q in records["gpt_baseline"]["question_scores"]]
short_qs  = [
    "Prominent structure\nat end of street?",
    "How many people\nwalking on left?",
    "Color of large\nawning on right?",
    "Where is person\nwith bicycle?",
    "What kind of\nscene is shown?",
]

def clean_answer(text, maxlen=90):
    # strip chat template artifacts
    text = re.sub(r"(user|assistant)\s*\n", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)   # remove bold markdown
    text = text.split("\n")[0].strip()               # first line only
    if len(text) > maxlen:
        text = text[:maxlen].rsplit(" ", 1)[0] + "…"
    return text

# ── Palette ───────────────────────────────────────────────────────────────────
BG       = "#ffffff"
PANEL    = "#f7f8fc"
BORDER   = "#d0d4e0"
TEXT_D   = "#1a1d27"
TEXT_DIM = "#6b7080"
GRID_C   = "#e8eaf0"
HDR_BG   = "#1e3a5f"
HDR_FG   = "#ffffff"
ACCENT_B = "#2563eb"

score_cmap = LinearSegmentedColormap.from_list("sc", ["#fca5a5", "#fef9c3", "#bbf7d0"])

def score_color(s, alpha=0.45):
    r, g, b, _ = score_cmap((s - 60) / 40)
    return (r, g, b, alpha)

# ── Layout ────────────────────────────────────────────────────────────────────
N_MODELS = len(ORDER)
N_QS     = len(questions)

fig = plt.figure(figsize=(22, 14), facecolor=BG)
gs  = GridSpec(1, 2, figure=fig,
               left=0.01, right=0.99, top=0.91, bottom=0.02,
               width_ratios=[1, 2.8], wspace=0.03)

ax_img  = fig.add_subplot(gs[0])
ax_tbl  = fig.add_subplot(gs[1])

for ax in (ax_img, ax_tbl):
    ax.set_facecolor(BG)
    ax.axis("off")

# ── Left: image ───────────────────────────────────────────────────────────────
img = Image.open(IMAGE_PATH)
ax_img.imshow(img, aspect="equal")
ax_img.axis("off")
ax_img.set_title(f"Test Image  ·  ID {IMAGE_ID}",
                 color=TEXT_D, fontsize=11, fontweight="bold", pad=8)

# small score badge per model
avg_scores = [records[k]["avg_score"] for k in ORDER]
badge_text = "  ".join(
    f"{lbl.splitlines()[0]}: {sc:.0f}"
    for lbl, sc in zip(MODEL_LABELS, avg_scores)
)

# ── Right: table ──────────────────────────────────────────────────────────────
# Manual table drawing: rows = questions, cols = models
# Extra left col for the question text

COL_W_Q   = 0.13   # question column width (fraction of axes)
COL_W_M   = (1.0 - COL_W_Q) / N_MODELS
ROW_H_HDR = 0.07
ROW_H     = (1.0 - ROW_H_HDR) / N_QS

def draw_cell(ax, x, y, w, h, text, bg, fg=TEXT_D, fontsize=8,
              bold=False, va="center", ha="center", pad=0.008):
    rect = mpatches.FancyBboxPatch(
        (x + 0.002, y + 0.002), w - 0.004, h - 0.004,
        boxstyle="round,pad=0.002",
        facecolor=bg, edgecolor=BORDER, linewidth=0.5,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text,
            transform=ax.transAxes,
            ha=ha, va=va, fontsize=fontsize,
            color=fg, fontweight="bold" if bold else "normal",
            wrap=False,
            multialignment="center")

# Header row: question col + model cols
draw_cell(ax_tbl, 0, 1 - ROW_H_HDR, COL_W_Q, ROW_H_HDR,
          "Question", HDR_BG, fg=HDR_FG, bold=True, fontsize=9)

for ci, lbl in enumerate(MODEL_LABELS):
    cx = COL_W_Q + ci * COL_W_M
    avg = avg_scores[ci]
    cell_bg = HDR_BG
    draw_cell(ax_tbl, cx, 1 - ROW_H_HDR, COL_W_M, ROW_H_HDR,
              f"{lbl}\n({avg:.0f}/100)",
              cell_bg, fg=HDR_FG, bold=True, fontsize=7.5)

# Data rows
for ri, (sq, full_q) in enumerate(zip(short_qs, questions)):
    y = 1 - ROW_H_HDR - (ri + 1) * ROW_H
    row_bg = PANEL if ri % 2 == 0 else BG

    # Question cell
    draw_cell(ax_tbl, 0, y, COL_W_Q, ROW_H,
              sq, row_bg, fg=TEXT_D, fontsize=7.5, bold=True, ha="center")

    for ci, key in enumerate(ORDER):
        cx   = COL_W_Q + ci * COL_W_M
        qs   = records[key]["question_scores"][ri]
        sc   = qs["score"]
        ans  = clean_answer(qs["model_answer"])
        wrapped = "\n".join(textwrap.wrap(ans, width=28))

        cell_bg = score_color(sc)
        fg_col  = TEXT_D

        draw_cell(ax_tbl, cx, y, COL_W_M, ROW_H,
                  f"{wrapped}\n[{sc}/100]",
                  cell_bg, fg=fg_col, fontsize=6.8, ha="center")

# ── Titles ─────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.955,
         "VLM Benchmark  ·  Sample Image Deep-Dive  ·  All Models  ·  5 Questions",
         ha="center", color=TEXT_D, fontsize=15, fontweight="bold")
fig.text(0.5, 0.935,
         "Logitech × ML@Berkeley  ·  Unified Eval Framework  ·  gpt-4.1-mini judge",
         ha="center", color=TEXT_DIM, fontsize=10)

# Legend
green_p = mpatches.Patch(color=score_color(100, 0.7), label="High score")
yell_p  = mpatches.Patch(color=score_color(80,  0.7), label="Mid score")
red_p   = mpatches.Patch(color=score_color(60,  0.7), label="Low score")
fig.legend(handles=[green_p, yell_p, red_p],
           loc="lower right", bbox_to_anchor=(0.99, 0.01),
           facecolor=BG, edgecolor=BORDER, labelcolor=TEXT_D, fontsize=9,
           ncol=3, title="Cell colour", title_fontsize=8)

out = "plot_sample.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
