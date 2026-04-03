import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

# ── Load & aggregate ──────────────────────────────────────────────────────────
with open("../vqa_results.json") as f:
    raw = json.load(f)

MODEL_META = {
    "gpt_baseline":    {"label": "GPT Baseline\n(gpt-5.4-mini)", "params": "~8B",  "quant": "fp16", "group": "cloud"},
    "smolvlm":         {"label": "SmolVLM2\n(2.2B)",             "params": "2.2B", "quant": "fp16", "group": "small"},
    "internvl":        {"label": "InternVL3\n(4B)",               "params": "4B",   "quant": "fp16", "group": "mid"},
    "qwen3vl_4b":      {"label": "Qwen3-VL\n(4B)",               "params": "4B",   "quant": "fp16", "group": "mid"},
    "qwen3vl_8b":      {"label": "Qwen3-VL\n(8B)",               "params": "8B",   "quant": "fp16", "group": "large"},
    "internvl_int8":   {"label": "InternVL3\n(4B-int8)",          "params": "4B",   "quant": "int8", "group": "mid"},
    "qwen3vl_4b_int8": {"label": "Qwen3-VL\n(4B-int8)",          "params": "4B",   "quant": "int8", "group": "mid"},
    "qwen3vl_8b_int8": {"label": "Qwen3-VL\n(8B-int8)",          "params": "8B",   "quant": "int8", "group": "large"},
}

ORDER = [
    "gpt_baseline", "smolvlm", "internvl", "qwen3vl_4b",
    "qwen3vl_8b", "internvl_int8", "qwen3vl_4b_int8", "qwen3vl_8b_int8",
]

models, scores, latencies, p25, p75, p_min, p_max = [], [], [], [], [], [], []
for key in ORDER:
    results = raw[key]
    s = [r["avg_score"] for r in results if "avg_score" in r]
    l = [r["latency_ms"] for r in results if "latency_ms" in r]
    models.append(MODEL_META[key]["label"])
    scores.append(np.mean(s))
    latencies.append(np.mean(l) / 1000)
    p25.append(np.percentile(s, 25))
    p75.append(np.percentile(s, 75))
    p_min.append(np.min(s))
    p_max.append(np.max(s))

scores    = np.array(scores)
latencies = np.array(latencies)
p25       = np.array(p25)
p75       = np.array(p75)
p_min     = np.array(p_min)
p_max     = np.array(p_max)

# ── Light-mode palette ────────────────────────────────────────────────────────
BG        = "#ffffff"
PANEL_BG  = "#f7f8fc"
BORDER    = "#d0d4e0"
TEXT_D    = "#1a1d27"
TEXT_DIM  = "#6b7080"
ACCENT_B  = "#2563eb"
ACCENT_G  = "#16a34a"
ACCENT_O  = "#ea580c"
ACCENT_R  = "#dc2626"
GRID_C    = "#e5e7ef"

score_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
cmap_bar   = LinearSegmentedColormap.from_list("lviz", ["#ef4444", "#f59e0b", "#22c55e"])
bar_colors = [cmap_bar(v) for v in score_norm]

x = np.arange(len(models))
short_labels = [
    "GPT-5.4-mini", "SmolVLM2 2.2B", "InternVL3 4B", "Qwen3-VL 4B",
    "Qwen3-VL 8B", "InternVL3 4B-int8", "Qwen3-VL 4B-int8", "Qwen3-VL 8B-int8",
]

def style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=TEXT_DIM)
    ax.grid(color=GRID_C, linewidth=0.7, zorder=0)

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1 — Accuracy bar chart
# ═══════════════════════════════════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(13, 6), facecolor=BG)
style_ax(ax)

bars = ax.bar(x, scores, width=0.6, color=bar_colors,
              edgecolor=BORDER, linewidth=0.8, zorder=3)

yerr_lo = scores - p25
yerr_hi = p75 - scores
ax.errorbar(x, scores, yerr=[yerr_lo, yerr_hi],
            fmt="none", ecolor=TEXT_DIM, elinewidth=1.4,
            capsize=4, capthick=1.4, zorder=4, alpha=0.7)

for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{score:.1f}",
            ha="center", va="bottom",
            color=TEXT_D, fontsize=9.5, fontweight="bold")

gpt_score = scores[ORDER.index("gpt_baseline")]
ax.axhline(gpt_score, color=ACCENT_B, linewidth=1.2,
           linestyle="--", alpha=0.6, zorder=2)
ax.text(len(models) - 0.42, gpt_score + 0.5,
        f"GPT baseline  {gpt_score:.1f}",
        color=ACCENT_B, fontsize=8.5, ha="right", alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(models, color=TEXT_D, fontsize=9.5)
ax.set_ylabel("Avg VQA Score  (0–100)", color=TEXT_DIM, fontsize=10)
ax.set_ylim(50, 100)
ax.set_xlim(-0.6, len(models) - 0.4)
ax.yaxis.set_tick_params(colors=TEXT_DIM)

fp16_mask = [i for i, k in enumerate(ORDER) if "int8" not in k]
int8_mask = [i for i, k in enumerate(ORDER) if "int8" in k]
for mask, lbl, col in [(fp16_mask, "fp16", ACCENT_G), (int8_mask, "int8 quantized", ACCENT_O)]:
    lo, hi = min(mask), max(mask)
    ax.annotate("", xy=(hi + 0.4, 51.6), xytext=(lo - 0.4, 51.6),
                arrowprops=dict(arrowstyle="-", color=col, lw=1.5))
    ax.text((lo + hi) / 2, 51, lbl, ha="center", va="top", color=col, fontsize=8)

ax.set_title("Visual Question-Answering  ·  Accuracy per Model",
             color=TEXT_D, fontsize=13, fontweight="bold", pad=10)
fig1.text(0.5, 0.01, "Logitech × ML@Berkeley  ·  Unified Eval Framework",
          ha="center", color=TEXT_DIM, fontsize=9)

fig1.tight_layout(rect=[0, 0.03, 1, 1])
fig1.savefig("plot_accuracy.png", dpi=180, bbox_inches="tight", facecolor=BG)
print("Saved → plot_accuracy.png")
plt.close(fig1)

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2 — Latency bar chart
# ═══════════════════════════════════════════════════════════════════════════════
fig2, ax = plt.subplots(figsize=(13, 6), facecolor=BG)
style_ax(ax)

lat_colors = [ACCENT_O if "int8" in k else ACCENT_B for k in ORDER]
ax.bar(x, latencies, width=0.6, color=lat_colors,
       edgecolor=BORDER, linewidth=0.8, zorder=3)

for xi, lat in zip(x, latencies):
    ax.text(xi, lat + 0.1, f"{lat:.1f}s",
            ha="center", va="bottom",
            color=TEXT_D, fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(models, color=TEXT_D, fontsize=9.5)
ax.set_ylabel("Avg Latency per Image (s)", color=TEXT_DIM, fontsize=10)
ax.yaxis.set_tick_params(colors=TEXT_DIM)
ax.set_title("Inference Latency per Model",
             color=TEXT_D, fontsize=13, fontweight="bold", pad=10)

patch_fp16 = mpatches.Patch(color=ACCENT_B, label="fp16")
patch_int8 = mpatches.Patch(color=ACCENT_O, label="int8 quantized")
ax.legend(handles=[patch_fp16, patch_int8],
          facecolor=BG, edgecolor=BORDER,
          labelcolor=TEXT_D, fontsize=9, loc="upper left")

fig2.text(0.5, 0.01, "Logitech × ML@Berkeley  ·  Unified Eval Framework",
          ha="center", color=TEXT_DIM, fontsize=9)
fig2.tight_layout(rect=[0, 0.03, 1, 1])
fig2.savefig("plot_latency.png", dpi=180, bbox_inches="tight", facecolor=BG)
print("Saved → plot_latency.png")
plt.close(fig2)

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3 — Score vs Latency scatter
# ═══════════════════════════════════════════════════════════════════════════════
fig3, ax = plt.subplots(figsize=(9, 7), facecolor=BG)
style_ax(ax)

offsets = {
    "gpt_baseline":    (-1.8,  0.5),
    "smolvlm":         ( 0.15,  0.5),
    "internvl":        ( 0.15, -1.3),
    "qwen3vl_4b":      ( 0.15,  0.5),
    "qwen3vl_8b":      ( 0.15,  0.5),
    "internvl_int8":   (-2.5,  0.5),
    "qwen3vl_4b_int8": ( 0.2,   0.5),
    "qwen3vl_8b_int8": ( 0.2,  -1.3),
}

# Gradient background: red (GPT corner = high latency, high score) →
# green (ideal corner = low latency, high score)
# GPT sits at ~(1.1, 90.9). Ideal corner is top-left.
# We score each point by: high score + low latency = good.
# Gradient drawn as a mesh from red (bottom-right) to green (top-left).
from matplotlib.colors import LinearSegmentedColormap as LSCM

lat_min, lat_max = latencies.min() - 0.5,  latencies.max() + 1.0
sc_min,  sc_max  = scores.min()   - 3.0,   scores.max()   + 2.0

nx, ny = 300, 300
lat_grid = np.linspace(lat_min, lat_max, nx)
sc_grid  = np.linspace(sc_min,  sc_max,  ny)
LAT, SC  = np.meshgrid(lat_grid, sc_grid)

# "goodness" = normalised score − normalised latency; range [−1, 1] → [0, 1]
lat_n = (LAT - lat_min) / (lat_max - lat_min)
sc_n  = (SC  - sc_min)  / (sc_max  - sc_min)
goodness = (sc_n - lat_n + 1) / 2   # 0 = worst (slow+bad), 1 = best (fast+good)

cmap_grad = LSCM.from_list("rg", ["#fca5a5", "#fef9c3", "#bbf7d0"])  # pastel red→yellow→green
ax.imshow(goodness, extent=[lat_min, lat_max, sc_min, sc_max],
          origin="lower", aspect="auto", cmap=cmap_grad,
          alpha=0.35, zorder=1)

for lat, sc, lbl, key in zip(latencies, scores, short_labels, ORDER):
    size = 140 if key == "gpt_baseline" else 100
    ax.scatter(lat, sc, s=size, color="white", edgecolors=TEXT_D,
               linewidths=1.2, zorder=4)
    # small coloured dot inside
    inner_col = ACCENT_B if key == "gpt_baseline" else ACCENT_O if "int8" in key else ACCENT_G
    ax.scatter(lat, sc, s=size * 0.35, color=inner_col, zorder=5)
    dx, dy = offsets.get(key, (0.2, 0.5))
    ax.text(lat + dx, sc + dy, lbl,
            color=TEXT_D, fontsize=8, ha="left", va="bottom", fontweight="bold")

ax.set_xlim(lat_min, lat_max)
ax.set_ylim(sc_min,  sc_max)
ax.set_xlabel("Avg Latency (s)", color=TEXT_DIM, fontsize=10)
ax.set_ylabel("Avg VQA Score", color=TEXT_DIM, fontsize=10)
ax.xaxis.set_tick_params(colors=TEXT_DIM)
ax.yaxis.set_tick_params(colors=TEXT_DIM)
ax.set_title("Score vs. Latency Trade-off",
             color=TEXT_D, fontsize=13, fontweight="bold", pad=10)

ax.annotate("← faster & better", xy=(0.04, 0.96), xycoords="axes fraction",
            color="#15803d", fontsize=9, ha="left", va="top", fontweight="bold")

fp16_dot  = mpatches.Patch(color=ACCENT_G, label="local fp16")
int8_dot  = mpatches.Patch(color=ACCENT_O, label="local int8")
cloud_dot = mpatches.Patch(color=ACCENT_B, label="cloud")
ax.legend(handles=[fp16_dot, int8_dot, cloud_dot],
          facecolor=BG, edgecolor=BORDER,
          labelcolor=TEXT_D, fontsize=9, loc="lower right")

fig3.text(0.5, 0.01, "Logitech × ML@Berkeley  ·  Unified Eval Framework",
          ha="center", color=TEXT_DIM, fontsize=9)
fig3.tight_layout(rect=[0, 0.03, 1, 1])
fig3.savefig("plot_scatter.png", dpi=180, bbox_inches="tight", facecolor=BG)
print("Saved → plot_scatter.png")
plt.close(fig3)

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 4 — Summary table
# ═══════════════════════════════════════════════════════════════════════════════
table_labels = [
    "GPT Baseline (gpt-5.4-mini)", "SmolVLM2 (2.2B)", "InternVL3 (4B)",
    "Qwen3-VL (4B)", "Qwen3-VL (8B)",
    "InternVL3 (4B-int8)", "Qwen3-VL (4B-int8)", "Qwen3-VL (8B-int8)",
]
quants   = ["fp16", "fp16", "fp16", "fp16", "fp16", "int8", "int8", "int8"]
groups   = ["Cloud", "Local", "Local", "Local", "Local", "Local", "Local", "Local"]

# Score delta vs GPT
delta = scores - gpt_score

col_headers = ["Model", "Quant", "Type", "Avg Score", "vs GPT", "P25", "P75", "Avg Latency"]
rows = []
for i, key in enumerate(ORDER):
    d = delta[i]
    d_str = f"{d:+.1f}" if key != "gpt_baseline" else "—"
    rows.append([
        table_labels[i],
        quants[i],
        groups[i],
        f"{scores[i]:.1f}",
        d_str,
        f"{p25[i]:.1f}",
        f"{p75[i]:.1f}",
        f"{latencies[i]:.2f}s",
    ])

fig4, ax = plt.subplots(figsize=(14, 4.8), facecolor=BG)
ax.set_facecolor(BG)
ax.axis("off")

tbl = ax.table(
    cellText=rows,
    colLabels=col_headers,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.9)

# Style header row
for col in range(len(col_headers)):
    cell = tbl[0, col]
    cell.set_facecolor("#1e3a5f")
    cell.set_text_props(color="white", fontweight="bold", fontsize=10)
    cell.set_edgecolor(BORDER)

# Style data rows
for row_idx, (row_data, key) in enumerate(zip(rows, ORDER), start=1):
    is_even = row_idx % 2 == 0
    row_bg  = "#eef2fb" if is_even else BG

    for col_idx in range(len(col_headers)):
        cell = tbl[row_idx, col_idx]
        cell.set_facecolor(row_bg)
        cell.set_edgecolor(BORDER)
        cell.set_text_props(color=TEXT_D)

        # Colour score column by value
        if col_idx == 3:
            v = score_norm[row_idx - 1]
            r, g, b, _ = cmap_bar(v)
            # pastel blend towards white
            cell.set_facecolor((r * 0.35 + 0.65, g * 0.35 + 0.65, b * 0.35 + 0.65))

        # Colour delta column
        if col_idx == 4 and key != "gpt_baseline":
            d = delta[row_idx - 1]
            cell.set_text_props(
                color=ACCENT_G if d >= 0 else ACCENT_R,
                fontweight="bold",
            )

        # Bold model name
        if col_idx == 0:
            cell.set_text_props(color=TEXT_D, fontweight="bold", ha="left")
            cell.PAD = 0.02

        # int8 badge colour
        if col_idx == 1 and row_data[1] == "int8":
            cell.set_text_props(color=ACCENT_O, fontweight="bold")

tbl.auto_set_column_width(list(range(len(col_headers))))

ax.set_title("VLM Benchmark  ·  Summary Table",
             color=TEXT_D, fontsize=13, fontweight="bold", pad=14, loc="left", x=0.01)

fig4.text(0.5, 0.01, "Logitech × ML@Berkeley  ·  Unified Eval Framework  ·  100 captioning images",
          ha="center", color=TEXT_DIM, fontsize=9)

fig4.tight_layout(rect=[0, 0.04, 1, 0.97])
fig4.savefig("plot_table.png", dpi=180, bbox_inches="tight", facecolor=BG)
print("Saved → plot_table.png")
plt.close(fig4)

print("\nAll plots saved.")
