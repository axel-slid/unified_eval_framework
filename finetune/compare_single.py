#!/usr/bin/env python
"""Quick base vs LoRA comparison on a single image."""
import json, os, time, torch
from pathlib import Path
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info

os.environ.setdefault("HF_HOME", "/mnt/shared/dils/hf_cache")

MODEL_PATH  = "/mnt/shared/dils/models/Qwen3-VL-4B-Instruct"
ADAPTER     = str(Path(__file__).parent / "checkpoints/qwen3vl_bbox_lora/final")
IMAGE_PATH  = "/mnt/shared/dils/projects/logitech/unified_eval_framework/people_images/rally-board-65-rightsight-2-group-view.png"

PROMPT_LORA = (
    "Detect ALL people in this {w}x{h} image.\n"
    "Output ONLY valid JSON, no extra text:\n"
    '{{"people": [{{"id": "P1", "bbox": [x1, y1, x2, y2]}}, ...]}}\n'
    "Bounding boxes must be pixel coordinates [x1, y1, x2, y2] within the image."
)

PROMPT_BASE = (
    "You are analyzing a meeting room image ({w}x{h} pixels).\n\n"
    "Detect ALL people visible in the image. For each person output:\n"
    '  - "role": "participant" or "non-participant"\n'
    "  - is_target_speaker: true for at most one person\n\n"
    "Output ONLY valid JSON:\n"
    '{{"people": [{{"id": "P1", "bbox": [x1, y1, x2, y2], "role": "participant", "is_target_speaker": false}}, ...]}}\n'
    "Bounding boxes must be pixel coordinates within the image. No text outside JSON."
)

def run(model, processor, image_path, prompt):
    with Image.open(image_path) as im:
        W, H = im.size
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text",  "text": prompt.format(w=W, h=H)},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_inputs, vid_inputs = process_vision_info(messages)
    device = next(model.parameters()).device
    inputs = processor(text=[text], images=img_inputs, videos=vid_inputs,
                       padding=True, return_tensors="pt").to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    ms = (time.perf_counter() - t0) * 1000
    trimmed = ids[0][inputs.input_ids.shape[1]:]
    return processor.decode(trimmed, skip_special_tokens=True), ms, W, H

def draw_boxes(image_path, people, out_path, label):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    colors = ["#FF4444", "#44FF44", "#4444FF", "#FF44FF", "#44FFFF", "#FFFF44"]
    for i, p in enumerate(people):
        bb = p.get("bbox", [])
        if len(bb) == 4:
            draw.rectangle(bb, outline=colors[i % len(colors)], width=3)
            draw.text((bb[0]+4, bb[1]+4), p.get("id", f"P{i+1}"), fill=colors[i % len(colors)])
    img.save(out_path)
    print(f"  Saved viz → {out_path}")

def parse_json(text):
    import re, json
    for attempt in [text, re.sub(r'^```(?:json)?\s*', '', text.strip())]:
        try: return json.loads(attempt)
        except: pass
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try: return json.loads(m.group(0))
        except: pass
    return None

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)

# ── BASE ──────────────────────────────────────────────────────────────────────
print("\n=== BASE MODEL ===")
base = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True
).eval()
base_out, base_ms, W, H = run(base, processor, IMAGE_PATH, PROMPT_BASE)
print(f"Latency: {base_ms:.0f} ms")
print(f"Output:\n{base_out}")
base_parsed = parse_json(base_out)
base_people = base_parsed.get("people", []) if base_parsed else []
print(f"People detected: {len(base_people)}")
draw_boxes(IMAGE_PATH, base_people,
           "/mnt/shared/dils/projects/logitech/unified_eval_framework/finetune/viz_output/compare_base.png",
           "base")
del base; torch.cuda.empty_cache()

# ── LORA ──────────────────────────────────────────────────────────────────────
print("\n=== LORA FINE-TUNED MODEL ===")
ft = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True
).eval()
ft = PeftModel.from_pretrained(ft, ADAPTER).eval()
lora_out, lora_ms, _, _ = run(ft, processor, IMAGE_PATH, PROMPT_LORA)
print(f"Latency: {lora_ms:.0f} ms")
print(f"Output:\n{lora_out}")
lora_parsed = parse_json(lora_out)
lora_people = lora_parsed.get("people", []) if lora_parsed else []
print(f"People detected: {len(lora_people)}")
draw_boxes(IMAGE_PATH, lora_people,
           "/mnt/shared/dils/projects/logitech/unified_eval_framework/finetune/viz_output/compare_lora.png",
           "lora")
del ft; torch.cuda.empty_cache()

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print(f"{'':25} {'BASE':>10} {'LORA':>10}")
print("-" * 50)
print(f"{'people detected':<25} {len(base_people):>10} {len(lora_people):>10}")
print(f"{'latency (ms)':<25} {base_ms:>10.0f} {lora_ms:>10.0f}")
print("=" * 50)
print("\nViz saved to finetune/viz_output/compare_base.png and compare_lora.png")
