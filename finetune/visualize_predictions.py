#!/usr/bin/env python
"""
finetune/visualize_predictions.py — Two-pass meeting room analysis.

Pass 1 (fine-tuned Qwen3-VL): bbox detection only — focused, accurate localization.
Pass 2 (base Qwen3-VL):       given the detected bboxes, reason about role + speaker
                               from the full scene context.

Produces a side-by-side comparison:
  Left  — Base model doing everything in one shot
  Right — Fine-tuned detection + base model role/speaker reasoning

Usage
-----
  python visualize_predictions.py \
    --image ../people_images/rally-board-65-rightsight-2-group-view.png \
    --adapter checkpoints/qwen3vl_bbox_lora/checkpoint-954
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

ROOT       = Path(__file__).parent
MODEL_PATH = "/mnt/shared/dils/models/Qwen3-VL-4B-Instruct"

# ── Prompts ────────────────────────────────────────────────────────────────────

BBOX_ONLY_PROMPT = (
    "You are analyzing an image ({width}x{height} pixels).\n\n"
    "Detect ALL people visible in the image and output their bounding boxes.\n\n"
    "Output ONLY valid JSON, no extra text:\n"
    '{{ "people": [{{"id": "P1", "bbox": [x1, y1, x2, y2]}}, ...] }}\n\n'
    "Bounding boxes must be pixel coordinates [x1, y1, x2, y2] within the "
    "image ({width}x{height}). Do not output any text outside the JSON object."
)

# Given detected bboxes, reason about role and speaker from scene context
ROLE_SPEAKER_PROMPT = (
    "You are analyzing a meeting room image ({width}x{height} pixels).\n\n"
    "The following people have already been detected (bounding boxes in pixels):\n"
    "{people_list}\n\n"
    "For each person, determine:\n"
    '  - "role": "participant" if they are seated/standing at the table engaged '
    'in the meeting, or "non-participant" if they are in the background, '
    "passing by, or not engaged.\n"
    '  - "is_target_speaker": true for the ONE person currently talking '
    "(open mouth, leaning forward, others looking at them). false for everyone else. "
    "If nobody is clearly speaking, all false.\n"
    '  - "reason": one short phrase explaining your decision.\n\n'
    "Output ONLY valid JSON:\n"
    '{{ "people": [{{"id": "P1", "bbox": [x1,y1,x2,y2], "role": "participant", '
    '"is_target_speaker": false, "reason": "seated at table"}}] }}\n\n'
    "Keep the same ids and bboxes. Do not output any text outside the JSON object."
)

# Base model single-shot prompt (for left panel comparison)
SINGLE_SHOT_PROMPT = (
    "You are analyzing a meeting room image ({width}x{height} pixels).\n\n"
    "1. Detect ALL people visible in the image.\n"
    "2. For each person label their role:\n"
    '   - "participant": seated or standing at the table, engaged with the meeting.\n'
    '   - "non-participant": passing by, in background, not engaged.\n'
    "3. Identify the TARGET SPEAKER — the ONE person currently talking "
    "(open mouth, others looking at them). At most one is_target_speaker=true.\n\n"
    "Output ONLY valid JSON, no extra text:\n"
    '{{ "people": [{{"id": "P1", "bbox": [x1, y1, x2, y2], "role": "participant", '
    '"is_target_speaker": false, "reason": "short reason"}}, ...] }}\n\n'
    "Bounding boxes must be pixel coordinates [x1, y1, x2, y2] within the "
    "image ({width}x{height}). Do not output any text outside the JSON object."
)

# ── Colours ────────────────────────────────────────────────────────────────────
ROLE_COLORS = {
    "participant":     "#00C853",
    "non-participant": "#FF6D00",
    None:              "#2196F3",
}
SPEAKER_COLOR = "#FFD600"


# ── JSON helpers ───────────────────────────────────────────────────────────────

def extract_json(text: str) -> dict | None:
    text = text.strip()
    for candidate in [text,
                      re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE).strip()]:
        candidate = re.sub(r"\s*```$", "", candidate, flags=re.MULTILINE).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ── Model ──────────────────────────────────────────────────────────────────────

class VLMRunner:
    def __init__(self, model_path: str, adapter_path: str | None = None):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info as pvi
        os.environ.setdefault("HF_HOME", "/mnt/shared/dils/hf_cache")

        self._pvi = pvi
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map="auto", local_files_only=True,
        ).eval()
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path).eval()

    def run(self, image_path: str, prompt: str, max_new_tokens: int = 1024) -> tuple[str, float]:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text",  "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        imgs, vids = self._pvi(messages)
        device = next(self.model.parameters()).device
        inputs = self.processor(
            text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt"
        ).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        latency = (time.perf_counter() - t0) * 1000
        trimmed = [ids[0][inputs.input_ids.shape[1]:]]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0], latency

    def unload(self):
        del self.model
        torch.cuda.empty_cache()


# ── Two-pass pipeline ──────────────────────────────────────────────────────────

def run_two_pass(image_path: str, adapter_path: str) -> tuple[list, float]:
    """
    Pass 1: fine-tuned model → bboxes only.
    Pass 2: base model → role + speaker reasoning given those bboxes.
    """
    with Image.open(image_path) as im:
        W, H = im.size

    # Pass 1 — bbox detection (fine-tuned)
    print("  [pass 1] Loading fine-tuned model for bbox detection ...")
    ft_model = VLMRunner(MODEL_PATH, adapter_path)
    bbox_prompt = BBOX_ONLY_PROMPT.format(width=W, height=H)
    bbox_raw, lat1 = ft_model.run(image_path, bbox_prompt, max_new_tokens=512)
    ft_model.unload()
    print(f"  [pass 1] Raw: {bbox_raw[:200]}")

    parsed = extract_json(bbox_raw)
    people_with_bboxes = parsed.get("people", []) if parsed else []
    print(f"  [pass 1] Detected {len(people_with_bboxes)} people")

    if not people_with_bboxes:
        return [], lat1

    # Clamp bboxes
    clamped = []
    for p in people_with_bboxes:
        bb = p.get("bbox", [])
        if len(bb) == 4:
            x1, y1, x2, y2 = bb
            x1, x2 = max(0, x1), min(W, x2)
            y1, y2 = max(0, y1), min(H, y2)
            if x2 > x1 and y2 > y1:
                clamped.append({"id": p["id"], "bbox": [x1, y1, x2, y2]})
    people_with_bboxes = clamped

    # Pass 2 — role + speaker reasoning (base model)
    people_list_str = "\n".join(
        f'  {p["id"]}: bbox={p["bbox"]}'
        for p in people_with_bboxes
    )
    role_prompt = ROLE_SPEAKER_PROMPT.format(
        width=W, height=H, people_list=people_list_str
    )
    print("  [pass 2] Loading base model for role/speaker reasoning ...")
    base_model = VLMRunner(MODEL_PATH, adapter_path=None)
    role_raw, lat2 = base_model.run(image_path, role_prompt, max_new_tokens=1024)
    base_model.unload()
    print(f"  [pass 2] Raw: {role_raw[:300]}")

    parsed2 = extract_json(role_raw)
    people_final = parsed2.get("people", []) if parsed2 else people_with_bboxes
    return people_final, lat1 + lat2


def run_single_shot_base(image_path: str) -> tuple[list, float]:
    with Image.open(image_path) as im:
        W, H = im.size
    prompt = SINGLE_SHOT_PROMPT.format(width=W, height=H)
    print("  Loading base model ...")
    model = VLMRunner(MODEL_PATH)
    raw, latency = model.run(image_path, prompt, max_new_tokens=2048)
    model.unload()
    print(f"  Raw: {raw[:300]}")
    parsed = extract_json(raw)
    return (parsed.get("people", []) if parsed else []), latency


# ── Drawing ────────────────────────────────────────────────────────────────────

def draw_panel(image_path: str, people: list, title: str, latency_ms: float, subtitle_extra: str = "") -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size

    try:
        font       = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(14, H // 38))
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(12, H // 48))
    except Exception:
        font = font_small = ImageFont.load_default()

    for p in people:
        bbox = p.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        x1, x2 = max(0, x1), min(W, x2)
        y1, y2 = max(0, y1), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        role    = p.get("role")
        speaker = p.get("is_target_speaker", False)
        color   = ROLE_COLORS.get(role, ROLE_COLORS[None])
        lw      = max(2, H // 180)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        if speaker:
            draw.rectangle([x1-lw*2, y1-lw*2, x2+lw*2, y2+lw*2], outline=SPEAKER_COLOR, width=lw+1)

        label = f"{p.get('id','?')}"
        if role:
            label += f" {role[:4]}"   # "part" or "non-"
        if speaker:
            label += " SPEAKING"
        reason = p.get("reason", "")

        tb = draw.textbbox((x1, y1), label, font=font_small)
        lbl_h = tb[3] - tb[1] + 4
        lbl_w = tb[2] - tb[0] + 6
        draw.rectangle([x1, y1 - lbl_h, x1 + lbl_w, y1], fill=color)
        draw.text((x1 + 3, y1 - lbl_h + 1), label, fill="white", font=font_small)

        # Reason tooltip at bottom of box
        if reason:
            rb = draw.textbbox((x1, y2), reason[:40], font=font_small)
            rw = rb[2] - rb[0] + 6
            draw.rectangle([x1, y2, x1 + rw, y2 + (rb[3]-rb[1]) + 4], fill="#00000099")
            draw.text((x1 + 3, y2 + 2), reason[:40], fill="#EEEEEE", font=font_small)

    # Stats
    n_part    = sum(1 for p in people if p.get("role") == "participant")
    n_nonpart = sum(1 for p in people if p.get("role") == "non-participant")
    speakers  = [p.get("id") for p in people if p.get("is_target_speaker")]

    bar_h = max(50, H // 14)
    bar = Image.new("RGB", (W, bar_h), "#1A1A2E")
    bd  = ImageDraw.Draw(bar)
    bd.text((10, 4),        title,    fill="#E0E0E0", font=font)
    stats = f"{len(people)} people  |  {n_part} participants  |  {n_nonpart} non-participants"
    if speakers:
        stats += f"  |  speaker: {', '.join(speakers)}"
    stats += f"  |  {latency_ms/1000:.1f}s"
    if subtitle_extra:
        stats += f"  |  {subtitle_extra}"
    bd.text((10, bar_h//2 + 2), stats, fill="#9E9E9E", font=font_small)

    out = Image.new("RGB", (W, H + bar_h))
    out.paste(bar, (0, 0))
    out.paste(img, (0, bar_h))
    return out


def add_legend(img: Image.Image) -> Image.Image:
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    items = [
        (ROLE_COLORS["participant"],     "participant"),
        (ROLE_COLORS["non-participant"], "non-participant"),
        (SPEAKER_COLOR,                 "target speaker (yellow outline)"),
        (ROLE_COLORS[None],             "unknown role"),
    ]
    pad, sw, sp = 8, 14, 20
    lh = pad * 2 + len(items) * sp
    legend = Image.new("RGB", (260, lh), "#1A1A2E")
    d = ImageDraw.Draw(legend)
    for i, (c, lbl) in enumerate(items):
        y = pad + i * sp
        d.rectangle([pad, y+1, pad+sw, y+sw-1], fill=c)
        d.text((pad+sw+6, y), lbl, fill="#E0E0E0", font=font)
    out = Image.new("RGB", (img.width, img.height + lh))
    out.paste(img, (0, 0))
    out.paste(legend, (0, img.height))
    return out


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",   default="../people_images/rally-board-65-rightsight-2-group-view.png")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--out-dir", default="viz_output")
    args = parser.parse_args()

    img_path = str(Path(args.image).resolve())
    out_dir  = ROOT / args.out_dir
    out_dir.mkdir(exist_ok=True)

    panels = []

    # Left: base model single-shot
    print("\n=== Base model (single-shot) ===")
    people_base, lat_base = run_single_shot_base(img_path)
    print(f"  → {len(people_base)} people")
    panels.append(draw_panel(img_path, people_base, "Base Qwen3-VL-4B  (single-shot)", lat_base))

    # Right: two-pass (fine-tuned bbox + base reasoning)
    if args.adapter:
        print(f"\n=== Two-pass: fine-tuned bbox + base role/speaker ===")
        people_ft, lat_ft = run_two_pass(img_path, args.adapter)
        print(f"  → {len(people_ft)} people")
        ckpt = Path(args.adapter).name
        panels.append(draw_panel(
            img_path, people_ft,
            f"Fine-tuned bbox ({ckpt})  +  base reasoning",
            lat_ft,
            subtitle_extra="2-pass"
        ))

    # Stitch
    if len(panels) == 2:
        W = panels[0].width + panels[1].width
        H = max(panels[0].height, panels[1].height)
        canvas = Image.new("RGB", (W, H), "#0D0D0D")
        canvas.paste(panels[0], (0, 0))
        canvas.paste(panels[1], (panels[0].width, 0))
        final = add_legend(canvas)
    else:
        final = add_legend(panels[0])

    out = out_dir / "comparison.png"
    final.save(out)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
