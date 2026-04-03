"""Simple benchmark pipeline flowchart — light mode, one page."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

fig, ax = plt.subplots(figsize=(9, 11))
ax.set_xlim(0, 9)
ax.set_ylim(0, 11)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

MODELS = [
    "SmolVLM2\n2.2B",
    "InternVL3\n4B",
    "Qwen3-VL\n4B",
    "Qwen3-VL\n8B",
    "InternVL3\n4B int8",
    "Qwen3-VL\n4B int8",
    "Qwen3-VL\n8B int8",
]

# Colors
BLUE   = "#2563eb"
GREEN  = "#16a34a"
ORANGE = "#ea580c"
PURPLE = "#7c3aed"
GRAY   = "#64748b"
LGRAY  = "#f1f5f9"
BORDER = "#cbd5e1"
TEXT   = "#0f172a"
MUTED  = "#64748b"

CX = 4.5  # center x


def rbox(ax, cx, cy, w, h, label, sublabel=None,
         fc=LGRAY, ec=BORDER, tc=TEXT, fontsize=10, bold=True, lw=1.6):
    patch = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3,
    )
    ax.add_patch(patch)
    dy = 0.13 if sublabel else 0
    ax.text(cx, cy + dy, label, ha="center", va="center",
            color=tc, fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            fontfamily="sans-serif", zorder=4)
    if sublabel:
        ax.text(cx, cy - 0.2, sublabel, ha="center", va="center",
                color=MUTED, fontsize=7.5, fontfamily="sans-serif", zorder=4)


def arr(ax, x1, y1, x2, y2, color=GRAY, lw=1.8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=13), zorder=2)


# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(CX, 10.55, "VLM Benchmark Pipeline",
        ha="center", va="center", color=TEXT,
        fontsize=15, fontweight="bold", fontfamily="sans-serif")
ax.text(CX, 10.18, "Image-question pairs  →  parallel VLM inference  →  LLM-as-judge scoring",
        ha="center", va="center", color=MUTED,
        fontsize=8.5, fontfamily="sans-serif")

# ── ① Input: Image + Questions (side by side) ─────────────────────────────────
ax.text(CX, 9.65, "①  Input", ha="center", va="center",
        color=BLUE, fontsize=9, fontweight="bold", fontfamily="sans-serif")

rbox(ax, 2.6, 9.1, 2.8, 0.75, "Test Images",
     sublabel="image · rubric",
     fc="#eff6ff", ec=BLUE, tc=BLUE, fontsize=10)
rbox(ax, 6.4, 9.1, 2.8, 0.75, "Question Test Set",
     sublabel="100 VQA items",
     fc="#f0fdf4", ec=GREEN, tc=GREEN, fontsize=10)

# Arrow merging down to a single point
arr(ax, 2.6, 8.725, 2.6, 8.15, color=BLUE)
arr(ax, 6.4, 8.725, 6.4, 8.15, color=GREEN)
ax.plot([2.6, 6.4], [8.15, 8.15], color=GRAY, lw=1.8, zorder=2)
ax.plot([CX, CX], [8.15, 7.9], color=GRAY, lw=1.8, zorder=2)
arr(ax, CX, 7.9, CX, 7.65, color=GRAY)

# ── ② Inference: Models ───────────────────────────────────────────────────────
ax.text(CX, 7.4, "②  Inference", ha="center", va="center",
        color=PURPLE, fontsize=9, fontweight="bold", fontfamily="sans-serif")

n = len(MODELS)
model_xs = np.linspace(0.65, 8.35, n)
MODEL_Y  = 6.75
MODEL_W  = 1.05
MODEL_H  = 0.85

# Fan-out lines from center top to each model
for mx in model_xs:
    ax.plot([CX, mx], [7.65, MODEL_Y + MODEL_H/2], color="#c4b5fd", lw=1.3, zorder=1)

for mx, label in zip(model_xs, MODELS):
    rbox(ax, mx, MODEL_Y, MODEL_W, MODEL_H, label,
         fc="#faf5ff", ec=PURPLE, tc=PURPLE, fontsize=7.2, lw=1.4)

ax.text(CX, MODEL_Y - MODEL_H/2 - 0.2,
        "4 architectures  ·  3 int8-quantized variants",
        ha="center", va="center", color=MUTED, fontsize=7.5,
        fontfamily="sans-serif", style="italic")

# Fan-in lines from each model bottom to center
for mx in model_xs:
    ax.plot([mx, CX], [MODEL_Y - MODEL_H/2, 5.9], color="#c4b5fd", lw=1.3, zorder=1)
arr(ax, CX, 5.9, CX, 5.65, color=PURPLE)

# ── ③ Grade: GPT Judge ────────────────────────────────────────────────────────
ax.text(CX, 5.4, "③  Grade", ha="center", va="center",
        color=ORANGE, fontsize=9, fontweight="bold", fontfamily="sans-serif")

rbox(ax, CX, 4.85, 3.2, 0.8, "GPT Judge",
     sublabel="score 0–100  ·  one-sentence reason",
     fc="#fff7ed", ec=ORANGE, tc=ORANGE, fontsize=11)

arr(ax, CX, 4.85 - 0.4, CX, 4.05, color=ORANGE)

# ── ④ Output ──────────────────────────────────────────────────────────────────
rbox(ax, CX - 1.7, 3.65, 2.6, 0.65, "results.json",
     sublabel="raw per-item data",
     fc=LGRAY, ec=BORDER, tc=GRAY, fontsize=9)
rbox(ax, CX + 1.7, 3.65, 2.6, 0.65, "report.html",
     sublabel="scores · bars · detail",
     fc=LGRAY, ec=BORDER, tc=GRAY, fontsize=9)

ax.plot([CX, CX - 1.7], [4.05, 3.65 + 0.325], color=GRAY, lw=1.5, zorder=2)
ax.plot([CX, CX + 1.7], [4.05, 3.65 + 0.325], color=GRAY, lw=1.5, zorder=2)

plt.tight_layout(pad=0.3)
plt.savefig("benchmark_flowchart.png", dpi=180, bbox_inches="tight",
            facecolor="white")
print("Saved → benchmark_flowchart.png")
