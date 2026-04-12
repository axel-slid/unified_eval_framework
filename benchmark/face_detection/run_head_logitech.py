#!/usr/bin/env python
"""
run_head_logitech.py — YOLOv8-Head (Abcfsa/YOLOv8_head_detector) + Qwen3-VL-4B
on the Logitech rally-board image.  Generates the same overlay plot as
plot_logitech.py (participant / non-participant coloured bboxes).

Steps
-----
  1. Download nano.pt from the GitHub repo (if not already cached).
  2. Run YOLOv8-head detection on the Logitech image.
  3. Dilate each bbox and save a crop.
  4. Run Qwen3-VL-4B on each crop (participant query only).
  5. Save a pipeline JSON compatible with plot_logitech.py.
  6. Call plot_logitech.py to render the figure.

Usage
-----
    cd benchmark/face_detection/
    python run_head_logitech.py
    python run_head_logitech.py --conf 0.3 --dilate 1.5 --model medium
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlretrieve

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image as PILImage

HERE        = Path(__file__).parent
BENCH_ROOT  = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BENCH_ROOT))

RESULTS_DIR = HERE / "results"
CROPS_DIR   = RESULTS_DIR / "head_crops"
WEIGHTS_DIR = HERE / "weights"

IMAGE_STEM = "rally-board-65-rightsight-2-group-view"
IMAGE_PATH = HERE.parent.parent / "people_images" / f"{IMAGE_STEM}.png"

WEIGHT_URLS = {
    "nano":   "https://github.com/Abcfsa/YOLOv8_head_detector/raw/main/nano.pt",
    "medium": "https://github.com/Abcfsa/YOLOv8_head_detector/raw/main/medium.pt",
}

PROMPT_PARTICIPANT = (
    "You are given two images: Image 1 is the full meeting room, Image 2 is a "
    "crop of one detected head from that room. "
    "Using both images, determine: is this person a genuine meeting participant "
    "(seated or standing at the table, engaged in the meeting)? "
    "Answer YES or NO, then give one short reason."
)

VLM_KEY  = "qwen3vl_4b"
VLM_PATH = "/mnt/shared/dils/models/Qwen3-VL-4B-Instruct"

PART_COLOR    = ("#1a7f37", "#2da44e")   # green  edge, fill
NONPART_COLOR = ("#cf222e", "#fa4549")   # red    edge, fill


# ── helpers ───────────────────────────────────────────────────────────────────

def download_weights(model: str) -> Path:
    dest = WEIGHTS_DIR / f"yolov8_{model}_head.pt"
    if dest.exists():
        print(f"  Weights cached: {dest}")
        return dest
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    url = WEIGHT_URLS[model]
    print(f"  Downloading {url} …")
    urlretrieve(url, dest)
    print(f"  Saved → {dest}")
    return dest


def dilate_bbox(bbox: list, scale: float, W: int, H: int) -> list:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw = (x2 - x1) / 2 * scale
    hh = (y2 - y1) / 2 * scale
    return [max(0, cx - hw), max(0, cy - hh),
            min(W, cx + hw), min(H, cy + hh)]


def save_crop(img: PILImage.Image, bbox: list, idx: int) -> Path:
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop = img.crop((x1, y1, x2, y2))
    out  = CROPS_DIR / f"{IMAGE_STEM}__head{idx:02d}.png"
    crop.save(out)
    return out


def parse_yes_no(text: str) -> bool | None:
    t = text.strip().upper()
    if t.startswith("YES"): return True
    if t.startswith("NO"):  return False
    if "YES" in t[:20]:     return True
    if "NO"  in t[:20]:     return False
    return None


# ── Stage 1: head detection ───────────────────────────────────────────────────

def run_head_detection(weights_path: Path, conf: float, dilate: float) -> dict:
    from ultralytics import YOLO

    print(f"\n[Stage 1] YOLOv8-Head  weights={weights_path.name}  conf={conf}  dilate={dilate}×")
    model = YOLO(str(weights_path))

    img  = PILImage.open(IMAGE_PATH).convert("RGB")
    W, H = img.size

    t0      = time.perf_counter()
    results = model(str(IMAGE_PATH), conf=conf, verbose=False)
    latency = (time.perf_counter() - t0) * 1000

    detections = []
    for r in results:
        boxes = r.boxes
        for i, box in enumerate(boxes):
            xyxy = box.xyxy[0].tolist()          # pixel coords
            conf_val = float(box.conf[0])
            dbox = dilate_bbox(xyxy, dilate, W, H)
            crop_path = save_crop(img, dbox, i)
            detections.append({
                "original_bbox": xyxy,
                "dilated_bbox":  dbox,
                "confidence":    conf_val,
                "face_idx":      i,               # keep key name for plot_logitech compat
                "crop_path":     str(crop_path),
            })

    print(f"  {IMAGE_PATH.name}: {len(detections)} head(s)  {latency:.1f} ms")
    return {
        IMAGE_STEM: {
            "image_path": str(IMAGE_PATH),
            "n_faces":    len(detections),
            "detections": detections,
            "latency_ms": latency,
            "error":      None,
        }
    }


# ── Stage 2: VLM classification ───────────────────────────────────────────────

def run_vlm_stage(cv_results: dict) -> dict:
    from models import MODEL_REGISTRY
    from config import ModelConfig, GenerationConfig

    cls = MODEL_REGISTRY.get("Qwen3VLModel")
    if cls is None:
        print("  [VLM] Qwen3VLModel not found in registry — skipping")
        return {}

    mcfg = ModelConfig(
        key=VLM_KEY, enabled=True,
        cls_name="Qwen3VLModel", model_path=VLM_PATH,
        dtype="bfloat16",
        generation=GenerationConfig(max_new_tokens=64, do_sample=False),
    )
    model = cls(mcfg)
    print(f"\n[Stage 2] {VLM_KEY}")
    model.load()

    persons = []
    dets    = cv_results.get(IMAGE_STEM, {}).get("detections", [])
    for det in dets:
        rec = {**det}
        r   = model.run_two_image(str(IMAGE_PATH), det["crop_path"], PROMPT_PARTICIPANT)
        rec["participant_response"] = r.response
        rec["participant"]          = parse_yes_no(r.response)
        rec["participant_latency"]  = r.latency_ms
        rec["participant_error"]    = r.error
        # Stub talking so plot_logitech doesn't KeyError if it ever reads it
        rec["talking"] = None
        persons.append(rec)
        print(
            f"  head {det['face_idx']:02d} → "
            f"participant={'Y' if rec['participant'] else 'N' if rec['participant'] is False else '?'}"
            f"  ({r.latency_ms:.0f} ms)"
        )

    model.unload()
    return {VLM_KEY: {IMAGE_STEM: persons}}


# ── Plot (same style as plot_logitech.py) ─────────────────────────────────────

def generate_plot(data: dict, out: Path):
    dets    = data["cv_results"].get(IMAGE_STEM, {}).get("detections", [])
    persons = data["vlm_results"].get(VLM_KEY, {}).get(IMAGE_STEM, [])

    img  = PILImage.open(IMAGE_PATH).convert("RGB")
    W, H = img.size

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.imshow(np.array(img))
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.axis("off")

    for det in dets:
        f_idx            = det["face_idx"]
        conf             = det["confidence"]
        x1, y1, x2, y2  = det["dilated_bbox"]

        person   = next((p for p in persons if p.get("face_idx") == f_idx), None)
        part     = person.get("participant") if person else None
        edge_col, fill_col = PART_COLOR if part else NONPART_COLOR

        ax.add_patch(mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            boxstyle="round,pad=2",
            linewidth=2.5,
            edgecolor=edge_col,
            facecolor=fill_col + "22",
        ))

        part_str = "Participant" if part else "Non-participant"
        label    = f"#{f_idx}  {part_str}  ({conf:.0%})"
        ax.text(x1 + 4, max(y1 - 6, 14), label,
                fontsize=7.5, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=edge_col,
                          edgecolor="none", alpha=0.92),
                verticalalignment="bottom", clip_on=True)

    legend_elements = [
        mpatches.Patch(facecolor="#2da44e", edgecolor="#1a7f37", label="Participant"),
        mpatches.Patch(facecolor="#fa4549", edgecolor="#cf222e", label="Non-participant"),
    ]
    legend = ax.legend(handles=legend_elements, loc="lower left",
                       fontsize=9, framealpha=0.92, facecolor="white",
                       edgecolor="#d0d7de",
                       title="Qwen3-VL-4B verdict", title_fontsize=9)
    legend.get_title().set_fontweight("bold")

    n_det  = len(dets)
    n_part = sum(1 for p in persons if p.get("participant"))
    ax.set_title(
        f"YOLOv8-Head detection  ·  Qwen3-VL-4B classification  ·  "
        f"{n_det} heads detected  ·  {n_part} participant  ·  "
        f"{n_det - n_part} non-participant  ·  "
        f"dilated {data.get('dilate', 1.5)}×",
        fontsize=11, fontweight="bold", color="#24292f", pad=10,
    )

    plt.tight_layout(pad=0.5)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nPlot saved → {out}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="nano", choices=["nano", "medium"])
    parser.add_argument("--conf",   type=float, default=0.3)
    parser.add_argument("--dilate", type=float, default=1.5,
                        help="Dilation factor for head bbox before crop (default: 1.5)")
    parser.add_argument("--skip-vlm", action="store_true",
                        help="Skip VLM stage (detection only)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Stage 1 — head detection
    weights  = download_weights(args.model)
    cv_res   = run_head_detection(weights, args.conf, args.dilate)

    # Stage 2 — VLM
    vlm_res  = {} if args.skip_vlm else run_vlm_stage(cv_res)

    # Save JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = RESULTS_DIR / f"pipeline_head_{ts}.json"
    payload  = {
        "timestamp":   ts,
        "detector":    f"yolov8_head_{args.model}",
        "device":      device,
        "conf":        args.conf,
        "dilate":      args.dilate,
        "images":      [str(IMAGE_PATH)],
        "cv_results":  cv_res,
        "vlm_results": vlm_res,
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"Results JSON → {out_json}")

    # Plot
    out_png = RESULTS_DIR / f"head_logitech_{ts}.png"
    generate_plot(payload, out_png)


if __name__ == "__main__":
    main()
