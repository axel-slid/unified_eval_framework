import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap as LSCM

with open("vqa_results.json") as f:
    raw = json.load(f)

ORDER = [
    "gpt_baseline", "smolvlm", "internvl", "qwen3vl_4b",
    "qwen3vl_8b", "internvl_int8", "qwen3vl_4b_int8", "qwen3vl_8b_int8",
]
SHORT_LABELS = [
    "GPT-5.4-mini", "SmolVLM2 2.2B", "InternVL3 4B", "Qwen3-VL 4B",
    "Qwen3-VL 8B", "InternVL3 4B-int8", "Qwen3-VL 4B-int8", "Qwen3-VL 8B-int8",
]

scores, latencies = [], []
for key in ORDER:
    results = raw[key]
    scores.append(np.mean([r["avg_score"] for r in results if "avg_score" in r]))
    latencies.append(np.mean([r["latency_ms"] for r in results if "latency_ms" in r]) / 1000)

scores    = np.array(scores)
latencies = np.array(latencies)

BG       = "#ffffff"
PANEL    = "#f7f8fc"
BORDER   = "#d0d4e0"
TEXT_D   = "#1a1d27"
TEXT_DIM = "#6b7080"
GRID_C   = "#e5e7ef"
ACCENT_B = "#2563eb"
ACCENT_G = "#16a34a"
ACCENT_O = "#ea580c"

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

fig, ax = plt.subplots(figsize=(9, 7), facecolor=BG)
ax.set_facecolor(PANEL)
for spine in ax.spines.values():
    spine.set_edgecolor(BORDER)
    spine.set_linewidth(0.8)
ax.tick_params(colors=TEXT_DIM)
ax.grid(color=GRID_C, linewidth=0.7, zorder=0)

lat_min, lat_max = latencies.min() - 0.5,  latencies.max() + 1.0
sc_min,  sc_max  = scores.min()   - 3.0,   scores.max()   + 2.0

LAT, SC = np.meshgrid(
    np.linspace(lat_min, lat_max, 300),
    np.linspace(sc_min,  sc_max,  300),
)
lat_n    = (LAT - lat_min) / (lat_max - lat_min)
sc_n     = (SC  - sc_min)  / (sc_max  - sc_min)
goodness = (sc_n - lat_n + 1) / 2

cmap_grad = LSCM.from_list("rg", ["#fca5a5", "#fef9c3", "#bbf7d0"])
ax.imshow(goodness, extent=[lat_min, lat_max, sc_min, sc_max],
          origin="lower", aspect="auto", cmap=cmap_grad,
          alpha=0.35, zorder=1)

for lat, sc, lbl, key in zip(latencies, scores, SHORT_LABELS, ORDER):
    size      = 140 if key == "gpt_baseline" else 100
    inner_col = ACCENT_B if key == "gpt_baseline" else ACCENT_O if "int8" in key else ACCENT_G
    ax.scatter(lat, sc, s=size, color="white", edgecolors=TEXT_D,
               linewidths=1.2, zorder=4)
    ax.scatter(lat, sc, s=size * 0.35, color=inner_col, zorder=5)
    dx, dy = offsets.get(key, (0.2, 0.5))
    ax.text(lat + dx, sc + dy, lbl,
            color=TEXT_D, fontsize=8, ha="left", va="bottom", fontweight="bold")

ax.set_xlim(lat_min, lat_max)
ax.set_ylim(sc_min,  sc_max)
ax.set_xlabel("Avg Latency (s)", color=TEXT_DIM, fontsize=10)
ax.set_ylabel("Avg VQA Score",   color=TEXT_DIM, fontsize=10)
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

fig.text(0.5, 0.01, "Logitech × ML@Berkeley  ·  Unified Eval Framework",
         ha="center", color=TEXT_DIM, fontsize=9)
fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig("plot_scatter.png", dpi=180, bbox_inches="tight", facecolor=BG)
print("Saved → plot_scatter.png")
