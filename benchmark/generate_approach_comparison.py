#!/usr/bin/env python
"""
generate_approach_comparison.py — Visual comparison of Approach A vs Approach B.

Approach A: VLM-only (single prompt, full image → structured JSON with bbox + role + speaker)
Approach B: CV + VLM pipeline (YOLO/SSD detection → per-crop VLM Q&A)

Loads the most recent result JSON from each approach and produces:
  1. approach_comparison_latency_<ts>.png   — Latency breakdown (A vs B per image & model)
  2. approach_comparison_counts_<ts>.png    — Person count agreement (A vs B)
  3. approach_comparison_roles_<ts>.png     — Participant / speaker agreement heatmap
  4. approach_comparison_overlay_<ts>.png   — Side-by-side annotated images (one image, A vs B)
  5. approach_comparison_summary_<ts>.png   — Single-page summary dashboard

Usage
-----
    cd benchmark/
    python generate_approach_comparison.py
    python generate_approach_comparison.py \\
        --approach-a results/approach_a_XYZ.json \\
        --approach-b results/pipeline_people_XYZ.json \\
        --overlay-image "photo.jpg"
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from datetime import datetime
from PIL import Image as PILImage

ROOT        = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# ── Style ─────────────────────────────────────────────────────────────────────
BG       = "#ffffff"
PANEL_BG = "#f5f7fa"
TEXT     = "#1a1a2e"
GRID     = "#dce1e8"
SUBTITLE = "#555f6e"

COLOR_A   = "#6366f1"   # indigo  — Approach A
COLOR_B   = "#22c55e"   # green   — Approach B
COLOR_ERR = "#ef4444"

VLM_COLORS = {
    "smolvlm":    "#8b5cf6",
    "qwen3vl_4b": "#f97316",
}
VLM_LABELS = {
    "smolvlm":    "SmolVLM2-2.2B",
    "qwen3vl_4b": "Qwen3-VL-4B",
}
ROLE_YES = "#16a34a"
ROLE_NO  = "#dc2626"
ROLE_UNK = "#9ca3af"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "text.color":        TEXT,
    "grid.color":        GRID,
    "legend.facecolor":  BG,
    "legend.edgecolor":  GRID,
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ──────────────────────────────────────────────────────────────────────────────
# Data loaders
# ──────────────────────────────────────────────────────────────────────────────

def _latest(pattern: str) -> str:
    files = sorted(glob.glob(str(RESULTS_DIR / pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")
    return files[-1]


def load_approach_a(path: str | None) -> dict:
    p = path or _latest("approach_a_*.json")
    return json.loads(Path(p).read_text())


def load_approach_b(path: str | None) -> dict:
    p = path or _latest("pipeline_people_*.json")
    return json.loads(Path(p).read_text())


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _style(ax, title="", ylabel="", xlabel=""):
    ax.set_facecolor(PANEL_BG)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=6)
    if ylabel:
        ax.set_ylabel(ylabel, color=SUBTITLE, fontsize=8.5)
    if xlabel:
        ax.set_xlabel(xlabel, color=SUBTITLE, fontsize=8.5)


def _short(path: str, n: int = 14) -> str:
    name = Path(path).name
    return (name[:n] + "…") if len(name) > n else name


def _b_total_latency(b_data: dict, stem: str, vlm_key: str) -> float:
    """
    Approach B total latency for one image =
      sum of CV detection latencies (all detectors) +
      sum of VLM latencies for all crops from this image.
    """
    lat = 0.0
    for det_res in b_data["cv_results"].values():
        lat += det_res.get(stem, {}).get("latency_ms", 0.0)
    persons = b_data["vlm_results"].get(vlm_key, {}).get(stem, [])
    for p in persons:
        lat += p.get("participant_latency", 0.0)
        lat += p.get("talking_latency",     0.0)
    return lat


def _a_person_count(a_data: dict, stem: str, vlm_key: str) -> int:
    return a_data["results"].get(vlm_key, {}).get(stem, {}).get("n_people", 0)


def _b_person_count(b_data: dict, stem: str, detector: str = "yolo11s") -> int:
    return b_data["cv_results"].get(detector, {}).get(stem, {}).get("n_persons", 0)


def _b_vlm_counts(b_data: dict, stem: str, vlm_key: str) -> tuple[int, int]:
    """Returns (n_participants, n_talking) for Approach B."""
    persons = b_data["vlm_results"].get(vlm_key, {}).get(stem, [])
    n_part = sum(1 for p in persons if p.get("participant") is True)
    n_talk = sum(1 for p in persons if p.get("talking") is True)
    return n_part, n_talk


# ──────────────────────────────────────────────────────────────────────────────
# Shared image list — intersection of images in both result files
# ──────────────────────────────────────────────────────────────────────────────

def shared_stems(a_data: dict, b_data: dict, vlm_key: str) -> list[str]:
    a_stems = set(a_data["results"].get(vlm_key, {}).keys())
    b_stems = set()
    for det_res in b_data["cv_results"].values():
        b_stems |= set(det_res.keys())
    common = sorted(a_stems & b_stems)
    return common


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — Latency breakdown
# ──────────────────────────────────────────────────────────────────────────────

def fig_latency(a_data: dict, b_data: dict, out: Path):
    """
    For each shared VLM (smolvlm / qwen3vl_4b) show per-image latency:
      - Approach A: single inference latency
      - Approach B: CV latency (all detectors) + VLM latency (all crops × 2 questions)
    """
    vlm_keys = [k for k in a_data["results"] if k in b_data.get("vlm_results", {})]
    if not vlm_keys:
        print("  [latency] No shared VLMs, skipping")
        return

    n_vlms = len(vlm_keys)
    fig, axes = plt.subplots(1, n_vlms, figsize=(max(8, 5 * n_vlms), 5.5), facecolor=BG)
    if n_vlms == 1:
        axes = [axes]

    for ax, vlm_key in zip(axes, vlm_keys):
        stems = shared_stems(a_data, b_data, vlm_key)
        x     = np.arange(len(stems))
        lats_a = [
            a_data["results"][vlm_key][s]["latency_ms"]
            for s in stems
        ]
        lats_b = [_b_total_latency(b_data, s, vlm_key) for s in stems]

        w = 0.35
        ax.bar(x - w/2, lats_a, w, label="Approach A (VLM-only)", color=COLOR_A, alpha=0.88)
        ax.bar(x + w/2, lats_b, w, label="Approach B (CV+VLM)",   color=COLOR_B, alpha=0.88)

        for xi, (la, lb) in enumerate(zip(lats_a, lats_b)):
            ax.text(xi - w/2, la + 20, f"{la:.0f}", ha="center", va="bottom",
                    fontsize=6.5, color=COLOR_A)
            ax.text(xi + w/2, lb + 20, f"{lb:.0f}", ha="center", va="bottom",
                    fontsize=6.5, color=COLOR_B)

        _style(ax,
               title=f"Latency — {VLM_LABELS.get(vlm_key, vlm_key)}",
               ylabel="ms / image")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_short(a_data["results"][vlm_key][s]["image_path"]) for s in stems],
            fontsize=7, rotation=30, ha="right",
        )
        ax.legend(fontsize=8)

        # Speedup annotation
        valid = [(la, lb) for la, lb in zip(lats_a, lats_b) if lb > 0]
        if valid:
            ratios = [lb / la for la, lb in valid]
            ax.text(0.97, 0.97,
                    f"B/A ratio: {np.mean(ratios):.1f}×",
                    transform=ax.transAxes,
                    ha="right", va="top", fontsize=8.5, color=TEXT,
                    bbox=dict(facecolor=BG, edgecolor=GRID, pad=4))

    fig.suptitle(
        "Approach A vs B — Inference Latency per Image (ms)",
        color=TEXT, fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout(pad=0.8)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — Person count comparison
# ──────────────────────────────────────────────────────────────────────────────

def fig_person_counts(a_data: dict, b_data: dict, out: Path):
    """
    Per image: number of people detected by
      - Approach A (VLM bbox count)
      - Approach B / YOLOv11s (CV detector)
    Also shows parse success rate for Approach A.
    """
    vlm_keys = [k for k in a_data["results"] if k in b_data.get("vlm_results", {})]
    if not vlm_keys:
        print("  [counts] No shared VLMs, skipping")
        return

    n_vlms = len(vlm_keys)
    fig, axes = plt.subplots(1, n_vlms + 1,
                             figsize=(max(10, 4.5 * (n_vlms + 1)), 5.5),
                             facecolor=BG)

    # ── Per-VLM: person count comparison ─────────────────────────────────────
    for ax, vlm_key in zip(axes[:n_vlms], vlm_keys):
        stems  = shared_stems(a_data, b_data, vlm_key)
        x      = np.arange(len(stems))
        cnt_a  = [_a_person_count(a_data, s, vlm_key) for s in stems]
        cnt_b  = [_b_person_count(b_data, s)           for s in stems]

        w = 0.35
        ax.bar(x - w/2, cnt_a, w, label="Approach A", color=COLOR_A, alpha=0.88)
        ax.bar(x + w/2, cnt_b, w, label="Approach B (YOLOv11s)", color=COLOR_B, alpha=0.88)

        for xi, (ca, cb) in enumerate(zip(cnt_a, cnt_b)):
            for xoff, val, col in [(- w/2, ca, COLOR_A), (w/2, cb, COLOR_B)]:
                if val > 0:
                    ax.text(xi + xoff, val + 0.07, str(val),
                            ha="center", va="bottom", fontsize=7, color=col)

        _style(ax,
               title=f"Person Count — {VLM_LABELS.get(vlm_key, vlm_key)}",
               ylabel="# people detected")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_short(a_data["results"][vlm_key][s]["image_path"]) for s in stems],
            fontsize=7, rotation=30, ha="right",
        )
        ax.legend(fontsize=8)

    # ── Last panel: Approach A parse success rate ────────────────────────────
    ax_p = axes[-1]
    ax_p.set_facecolor(PANEL_BG)
    parse_rates = []
    labels_vk   = []
    for vlm_key in vlm_keys:
        img_res = a_data["results"].get(vlm_key, {})
        n_ok  = sum(1 for r in img_res.values() if r.get("parse_success"))
        n_all = len(img_res)
        rate  = n_ok / n_all if n_all else 0.0
        parse_rates.append(rate)
        labels_vk.append(VLM_LABELS.get(vlm_key, vlm_key))

    colors = [COLOR_A if r >= 0.8 else COLOR_ERR for r in parse_rates]
    bars   = ax_p.bar(range(len(vlm_keys)), parse_rates, color=colors, alpha=0.9)
    for bar, rate in zip(bars, parse_rates):
        ax_p.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                  f"{rate:.0%}", ha="center", va="bottom", fontsize=9, color=TEXT)
    ax_p.set_ylim(0, 1.15)
    ax_p.axhline(1.0, color=GRID, linestyle="--", linewidth=0.8)
    _style(ax_p, title="Approach A — JSON Parse Success Rate", ylabel="Rate")
    ax_p.set_xticks(range(len(vlm_keys)))
    ax_p.set_xticklabels(labels_vk, fontsize=8)

    fig.suptitle("Approach A vs B — Person Detection Count & Parse Reliability",
                 color=TEXT, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(pad=0.8)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 — Role / speaker agreement heatmap
# ──────────────────────────────────────────────────────────────────────────────

def fig_roles(a_data: dict, b_data: dict, out: Path):
    """
    For each (image, VLM) cell show a 4-value stacked bar:
      - Approach A: participants, non-participants, speakers
      - Approach B: participants (from crop Q1), talking (from crop Q2)
    One row per VLM, one group per image.
    """
    vlm_keys = [k for k in a_data["results"] if k in b_data.get("vlm_results", {})]
    if not vlm_keys:
        print("  [roles] No shared VLMs, skipping")
        return

    n_vlms = len(vlm_keys)
    fig, axes = plt.subplots(n_vlms, 1,
                             figsize=(max(10, 2.5 * 6), 4.5 * n_vlms),
                             facecolor=BG)
    if n_vlms == 1:
        axes = [axes]

    for ax, vlm_key in zip(axes, vlm_keys):
        stems = shared_stems(a_data, b_data, vlm_key)
        x     = np.arange(len(stems))
        w     = 0.18

        # Approach A
        a_part    = [a_data["results"][vlm_key][s].get("n_participants",     0) for s in stems]
        a_nonpart = [a_data["results"][vlm_key][s].get("n_non_participants", 0) for s in stems]
        a_speak   = [a_data["results"][vlm_key][s].get("n_target_speakers",  0) for s in stems]

        # Approach B
        b_part_list, b_talk_list = zip(*[_b_vlm_counts(b_data, s, vlm_key) for s in stems]) \
            if stems else ([], [])

        ax.bar(x - 1.5*w, a_part,    w, label="A: participants",     color=COLOR_A,    alpha=0.88)
        ax.bar(x - 0.5*w, a_nonpart, w, label="A: non-participants",  color="#a5b4fc",  alpha=0.88)
        ax.bar(x + 0.5*w, a_speak,   w, label="A: target speaker",   color="#7c3aed",  alpha=0.95)
        ax.bar(x + 1.5*w, b_part_list, w, label="B: participants",   color=COLOR_B,    alpha=0.88)
        ax.bar(x + 2.5*w, b_talk_list, w, label="B: talking",        color="#86efac",  alpha=0.88)

        _style(ax,
               title=f"Role & Speaker Assignments — {VLM_LABELS.get(vlm_key, vlm_key)}",
               ylabel="# people")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_short(a_data["results"][vlm_key][s]["image_path"]) for s in stems],
            fontsize=7.5, rotation=30, ha="right",
        )
        ax.legend(fontsize=7.5, loc="upper right", ncol=2)

    fig.suptitle("Approach A vs B — Role Classification & Speaker Detection",
                 color=TEXT, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(pad=0.8, h_pad=1.5)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4 — Side-by-side annotated overlay for one image
# ──────────────────────────────────────────────────────────────────────────────

def _draw_bbox(ax, bbox, color, label, lw=2.0, fontsize=7):
    if bbox is None or len(bbox) != 4:
        return
    x1, y1, x2, y2 = bbox
    rect = mpatches.FancyBboxPatch(
        (x1, y1), x2 - x1, y2 - y1,
        boxstyle="square,pad=0",
        linewidth=lw, edgecolor=color, facecolor="none",
    )
    ax.add_patch(rect)
    ax.text(x1 + 2, y1 + 12, label, color="white", fontsize=fontsize,
            bbox=dict(facecolor=color, alpha=0.82, pad=1.5, edgecolor="none"))


def fig_overlay(a_data: dict, b_data: dict, image_name: str | None, out: Path):
    """
    For one image show:
      Left  : original image
      Middle: Approach A bbox overlay (role + speaker)
      Right : Approach B overlay (YOLOv11s bboxes, VLM role/talking from best VLM)
    One row per shared VLM.
    """
    vlm_keys = [k for k in a_data["results"] if k in b_data.get("vlm_results", {})]
    if not vlm_keys:
        print("  [overlay] No shared VLMs, skipping")
        return

    # Pick image
    first_vlm = vlm_keys[0]
    all_stems  = list(a_data["results"][first_vlm].keys())
    if image_name:
        stem = next((s for s in all_stems if image_name in s or s in image_name), all_stems[0])
    else:
        stem = all_stems[0]

    img_path = a_data["results"][first_vlm][stem]["image_path"]
    try:
        orig = PILImage.open(img_path).convert("RGB")
    except Exception:
        print(f"  [overlay] Cannot open {img_path}, skipping")
        return

    n_vlms = len(vlm_keys)
    fig, axes = plt.subplots(
        n_vlms, 3,
        figsize=(15, 5.2 * n_vlms),
        facecolor=BG,
    )
    if n_vlms == 1:
        axes = axes.reshape(1, 3)

    for row, vlm_key in enumerate(vlm_keys):
        ax_orig, ax_a, ax_b = axes[row]
        vlbl = VLM_LABELS.get(vlm_key, vlm_key)

        # ── Original ──────────────────────────────────────────────────────────
        ax_orig.imshow(np.array(orig))
        ax_orig.axis("off")
        if row == 0:
            ax_orig.set_title("Original", color=TEXT, fontsize=10, fontweight="bold", pad=5)

        # ── Approach A ────────────────────────────────────────────────────────
        ax_a.imshow(np.array(orig))
        ax_a.axis("off")
        a_res = a_data["results"].get(vlm_key, {}).get(stem, {})
        for p in a_res.get("people", []):
            if not p.get("bbox_valid"):
                continue
            role    = p.get("role", "?")
            speaker = p.get("is_target_speaker", False)
            color   = "#7c3aed" if speaker else (ROLE_YES if role == "participant" else ROLE_NO)
            label   = f"{p['id']} {role[:4]}" + (" SPK" if speaker else "")
            _draw_bbox(ax_a, p["bbox"], color, label, lw=2.5 if speaker else 2.0)
        parse_ok = a_res.get("parse_success", False)
        ax_a.set_title(
            f"Approach A — {vlbl}\n"
            f"({a_res.get('n_people', 0)} people  parse={'OK' if parse_ok else 'FAIL'})",
            color=COLOR_A, fontsize=9, fontweight="bold", pad=4,
        )

        # ── Approach B ────────────────────────────────────────────────────────
        ax_b.imshow(np.array(orig))
        ax_b.axis("off")
        crop_det    = b_data.get("detector_for_crops", "yolo11s")
        b_dets      = b_data["cv_results"].get(crop_det, {}).get(stem, {}).get("detections", [])
        b_persons   = b_data["vlm_results"].get(vlm_key, {}).get(stem, [])
        b_person_map = {p["person_idx"]: p for p in b_persons}

        for i, det in enumerate(b_dets):
            p_vlm   = b_person_map.get(i)
            is_part = p_vlm.get("participant") if p_vlm else None
            is_talk = p_vlm.get("talking")     if p_vlm else None
            color   = ROLE_YES if is_part else (ROLE_NO if is_part is False else ROLE_UNK)
            label   = f"P{i} {'part' if is_part else ('non' if is_part is False else '?')}"
            if is_talk:
                label += " TLK"
                color  = "#f97316"
            _draw_bbox(ax_b, det["bbox"], color, label)

        n_b_part = sum(1 for p in b_persons if p.get("participant") is True)
        n_b_talk = sum(1 for p in b_persons if p.get("talking") is True)
        ax_b.set_title(
            f"Approach B — {vlbl}\n"
            f"({len(b_dets)} detected  {n_b_part} part  {n_b_talk} talking)",
            color=COLOR_B, fontsize=9, fontweight="bold", pad=4,
        )

    # Legend
    legend_patches = [
        mpatches.Patch(color=ROLE_YES,  label="Participant"),
        mpatches.Patch(color=ROLE_NO,   label="Non-participant"),
        mpatches.Patch(color="#7c3aed", label="A: Target speaker"),
        mpatches.Patch(color="#f97316", label="B: Talking"),
        mpatches.Patch(color=ROLE_UNK,  label="Unknown"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=5,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        f"Approach A vs B — Annotated Overlay\n{Path(img_path).name}",
        color=TEXT, fontsize=12, fontweight="bold",
    )
    fig.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5 — Summary dashboard
# ──────────────────────────────────────────────────────────────────────────────

def fig_summary(a_data: dict, b_data: dict, out: Path):
    """
    Single-page summary:
      Row 0: mean latency bar (A vs B per VLM)
      Row 1: parse rate + FPS comparison
      Row 2: mean person count agreement (A vs B/YOLOv11s)
      Row 3: qualitative capability table
    """
    vlm_keys = [k for k in a_data["results"] if k in b_data.get("vlm_results", {})]
    if not vlm_keys:
        print("  [summary] No shared VLMs, skipping")
        return

    fig = plt.figure(figsize=(14, 10), facecolor=BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.4,
                            left=0.07, right=0.97, top=0.90, bottom=0.08)

    labels_vk = [VLM_LABELS.get(k, k) for k in vlm_keys]
    x_vk      = np.arange(len(vlm_keys))
    w         = 0.32

    # ── (0,0) Mean latency ────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    mean_a, mean_b = [], []
    for vlm_key in vlm_keys:
        stems  = shared_stems(a_data, b_data, vlm_key)
        lats_a = [a_data["results"][vlm_key][s]["latency_ms"] for s in stems if s in a_data["results"][vlm_key]]
        lats_b = [_b_total_latency(b_data, s, vlm_key) for s in stems]
        mean_a.append(np.mean(lats_a) if lats_a else 0)
        mean_b.append(np.mean(lats_b) if lats_b else 0)

    bars_a = ax0.bar(x_vk - w/2, mean_a, w, label="Approach A", color=COLOR_A, alpha=0.88)
    bars_b = ax0.bar(x_vk + w/2, mean_b, w, label="Approach B", color=COLOR_B, alpha=0.88)
    for bar, val in list(zip(bars_a, mean_a)) + list(zip(bars_b, mean_b)):
        ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                 f"{val:.0f}", ha="center", va="bottom", fontsize=7.5, color=TEXT)
    _style(ax0, title="Mean Latency (ms/image)", ylabel="ms")
    ax0.set_xticks(x_vk)
    ax0.set_xticklabels(labels_vk, fontsize=8)
    ax0.legend(fontsize=8)

    # ── (0,1) FPS comparison ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    fps_a = [1000.0 / m if m > 0 else 0 for m in mean_a]
    fps_b = [1000.0 / m if m > 0 else 0 for m in mean_b]
    ax1.bar(x_vk - w/2, fps_a, w, label="Approach A", color=COLOR_A, alpha=0.88)
    ax1.bar(x_vk + w/2, fps_b, w, label="Approach B", color=COLOR_B, alpha=0.88)
    _style(ax1, title="Throughput (FPS)", ylabel="frames / sec")
    ax1.set_xticks(x_vk)
    ax1.set_xticklabels(labels_vk, fontsize=8)
    ax1.legend(fontsize=8)

    # ── (0,2) Parse success rate ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    parse_rates = []
    for vlm_key in vlm_keys:
        img_res  = a_data["results"].get(vlm_key, {})
        n_ok     = sum(1 for r in img_res.values() if r.get("parse_success"))
        n_all    = len(img_res)
        parse_rates.append(n_ok / n_all if n_all else 0.0)
    colors = [COLOR_A if r >= 0.8 else COLOR_ERR for r in parse_rates]
    bars = ax2.bar(x_vk, parse_rates, color=colors, alpha=0.9)
    for bar, rate in zip(bars, parse_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{rate:.0%}", ha="center", va="bottom", fontsize=9, color=TEXT)
    ax2.set_ylim(0, 1.15)
    ax2.axhline(1.0, color=GRID, linestyle="--", linewidth=0.8)
    _style(ax2, title="Approach A: JSON Parse Rate", ylabel="Rate")
    ax2.set_xticks(x_vk)
    ax2.set_xticklabels(labels_vk, fontsize=8)

    # ── (1,0–1) Mean person count agreement ───────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    for i, vlm_key in enumerate(vlm_keys):
        stems  = shared_stems(a_data, b_data, vlm_key)
        cnt_a  = [_a_person_count(a_data, s, vlm_key) for s in stems]
        cnt_b  = [_b_person_count(b_data, s)           for s in stems]
        offset = (i - len(vlm_keys)/2 + 0.5) * w * 2
        xi     = np.arange(len(stems))
        ax3.plot(xi + offset, cnt_a, "o-", color=VLM_COLORS.get(vlm_key, COLOR_A),
                 label=f"A – {VLM_LABELS.get(vlm_key, vlm_key)}", alpha=0.9, markersize=6)
        ax3.plot(xi + offset, cnt_b, "s--", color=COLOR_B, alpha=0.7,
                 label=f"B – YOLOv11s", markersize=6)

    # Only show one B legend entry
    handles, lbls = ax3.get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, lbls):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    ax3.legend(h2, l2, fontsize=7.5, loc="upper right")

    first_vlm = vlm_keys[0]
    stems_first = shared_stems(a_data, b_data, first_vlm)
    _style(ax3, title="Person Count per Image — A vs B/YOLOv11s", ylabel="# people")
    ax3.set_xticks(range(len(stems_first)))
    ax3.set_xticklabels(
        [_short(a_data["results"][first_vlm][s]["image_path"]) for s in stems_first],
        fontsize=7.5, rotation=25, ha="right",
    )

    # ── (1,2) Capability table ────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")

    capabilities = [
        ("Single inference per image",    "Yes", "No"),
        ("Bounding boxes",                "VLM (less precise)", "CV (high precision)"),
        ("Role classification",           "Yes",  "Yes (per crop)"),
        ("Target speaker ID",             "Yes",  "Indirect (talking Q)"),
        ("Spatial / scene context",       "Full image", "Crop only"),
        ("Runs w/o GPU (CPU feasible)",   "Possible (small model)", "CV: Yes / VLM: Heavy"),
        ("Output format",                 "Structured JSON", "Separate JSON fields"),
        ("LoRA adapter compatible",       "Yes",  "No (two-stage)"),
    ]

    col_labels  = ["Capability", "Approach A", "Approach B"]
    col_widths  = [0.45, 0.275, 0.275]
    row_h       = 0.10
    y_start     = 0.97

    for ci, (lbl, cw) in enumerate(zip(col_labels, col_widths)):
        xpos = sum(col_widths[:ci])
        ax4.text(xpos + cw/2, y_start, lbl,
                 transform=ax4.transAxes, fontsize=8, fontweight="bold",
                 ha="center", va="top", color=TEXT)

    ax4.plot([0, 1], [y_start - 0.035, y_start - 0.035],
             color=GRID, linewidth=0.8, transform=ax4.transAxes, clip_on=False)

    for ri, (cap, va, vb) in enumerate(capabilities):
        y = y_start - 0.05 - ri * row_h
        bg = "#f0f4ff" if ri % 2 == 0 else BG
        ax4.add_patch(mpatches.FancyBboxPatch(
            (0, y - row_h * 0.85), 1, row_h * 0.9,
            boxstyle="square,pad=0", facecolor=bg, edgecolor="none",
            transform=ax4.transAxes,
        ))
        ax4.text(0.01,                col_widths[0]/2 + y - row_h*0.4, cap,
                 transform=ax4.transAxes, fontsize=6.8, va="center", color=TEXT)
        ax4.text(col_widths[0] + col_widths[1]/2, y - row_h*0.4, va,
                 transform=ax4.transAxes, fontsize=6.5, va="center",
                 ha="center", color=COLOR_A)
        ax4.text(col_widths[0] + col_widths[1] + col_widths[2]/2, y - row_h*0.4, vb,
                 transform=ax4.transAxes, fontsize=6.5, va="center",
                 ha="center", color=COLOR_B)

    ax4.set_title("Capability Comparison", color=TEXT, fontsize=9,
                  fontweight="bold", pad=8)

    fig.suptitle(
        "Approach A (VLM-only)  vs  Approach B (CV + VLM)  —  Summary Dashboard",
        color=TEXT, fontsize=14, fontweight="bold",
    )
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare Approach A vs Approach B")
    parser.add_argument("--approach-a", default=None,
                        help="Path to approach_a_*.json (default: most recent)")
    parser.add_argument("--approach-b", default=None,
                        help="Path to pipeline_people_*.json (default: most recent)")
    parser.add_argument("--overlay-image", default=None,
                        help="Image name (or stem) for the annotated overlay figure")
    parser.add_argument("--out-dir", default=str(FIGURES_DIR))
    args = parser.parse_args()

    print("Loading results …")
    a_data = load_approach_a(args.approach_a)
    b_data = load_approach_b(args.approach_b)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\nGenerating figures …")
    fig_latency(      a_data, b_data, out_dir / f"approach_comparison_latency_{ts}.png")
    fig_person_counts(a_data, b_data, out_dir / f"approach_comparison_counts_{ts}.png")
    fig_roles(        a_data, b_data, out_dir / f"approach_comparison_roles_{ts}.png")
    fig_overlay(      a_data, b_data, args.overlay_image,
                                      out_dir / f"approach_comparison_overlay_{ts}.png")
    fig_summary(      a_data, b_data, out_dir / f"approach_comparison_summary_{ts}.png")

    print(f"\nAll figures → {out_dir}/")


if __name__ == "__main__":
    main()
