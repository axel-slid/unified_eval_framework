#!/usr/bin/env python
"""Run base vs LoRA on 5 val images, draw bboxes, save side-by-side comparisons."""
import json, os, re, time, torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info

os.environ.setdefault("HF_HOME", "/mnt/shared/dils/hf_cache")
ROOT       = Path(__file__).parent
MODEL_PATH = "/mnt/shared/dils/models/Qwen3-VL-4B-Instruct"
ADAPTER    = str(ROOT / "checkpoints/qwen3vl_bbox_lora/final")
OUT_DIR    = ROOT / "viz_output" / "5way"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_LORA = (
    "Detect ALL people in this {w}x{h} image.\n"
    "Output ONLY valid JSON, no extra text:\n"
    '{{"people": [{{"id": "P1", "bbox": [x1, y1, x2, y2]}}, ...]}}\n'
    "Bounding boxes must be pixel coordinates [x1, y1, x2, y2] within the image."
)
PROMPT_BASE = (
    "You are analyzing a meeting room image ({w}x{h} pixels).\n"
    "Detect ALL people visible. Output ONLY valid JSON:\n"
    '{{"people": [{{"id": "P1", "bbox": [x1, y1, x2, y2], "role": "participant", "is_target_speaker": false}}, ...]}}\n'
    "Bounding boxes must be pixel coordinates within the image. No text outside JSON."
)

COLORS = ["#FF4444", "#44BB66", "#4488FF", "#FF8800", "#CC44CC",
          "#00CCCC", "#FFCC00", "#FF44AA", "#88CC00", "#0088FF"]

def parse_json(text):
    for s in [text, re.sub(r'^```(?:json)?\s*', '', text.strip())]:
        try: return json.loads(s)
        except: pass
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try: return json.loads(m.group(0))
        except: pass
    return None

def run_model(model, processor, image_path, prompt_tmpl):
    with Image.open(image_path) as im:
        W, H = im.size
    prompt = prompt_tmpl.format(w=W, h=H)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": str(image_path)},
        {"type": "text",  "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(messages)
    device = next(model.parameters()).device
    inputs = processor(text=[text], images=img_in, videos=vid_in,
                       padding=True, return_tensors="pt").to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                             repetition_penalty=1.15)
    ms = (time.perf_counter() - t0) * 1000
    out = processor.decode(ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return out, ms, W, H

def draw(image_path, people, gt_boxes, title, W, H):
    img = Image.open(image_path).convert("RGB").resize((480, int(480 * H / W)))
    sx, sy = 480 / W, (480 * H / W) / H
    draw = ImageDraw.Draw(img)

    # GT boxes in white dashed style (draw twice offset)
    for bb in gt_boxes:
        x1,y1,x2,y2 = int(bb[0]*sx), int(bb[1]*sy), int(bb[2]*sx), int(bb[3]*sy)
        draw.rectangle([x1,y1,x2,y2], outline="#FFFFFF", width=3)
        draw.rectangle([x1,y1,x2,y2], outline="#000000", width=1)

    # Predicted boxes in color
    for i, p in enumerate(people):
        bb = p.get("bbox", [])
        if len(bb) != 4: continue
        x1,y1,x2,y2 = int(bb[0]*sx), int(bb[1]*sy), int(bb[2]*sx), int(bb[3]*sy)
        c = COLORS[i % len(COLORS)]
        draw.rectangle([x1,y1,x2,y2], outline=c, width=3)
        draw.rectangle([x1+2,y1+2,x1+28,y1+14], fill=c)
        draw.text((x1+4, y1+3), p.get("id","?"), fill="#fff")

    # Title bar
    bar = Image.new("RGB", (480, 28), "#1e293b")
    bd = ImageDraw.Draw(bar)
    bd.text((8, 6), title, fill="#f1f5f9")
    combined = Image.new("RGB", (480, img.height + 28))
    combined.paste(bar, (0, 0))
    combined.paste(img, (0, 28))
    return combined

def make_legend(gt_n, pred_n, iou_val, ms):
    leg = Image.new("RGB", (480, 60), "#0f172a")
    d = ImageDraw.Draw(leg)
    d.text((8, 8),  f"GT people: {gt_n}   Predicted: {pred_n}   IoU: {iou_val:.3f}", fill="#94a3b8")
    d.text((8, 28), f"Latency: {ms:.0f} ms", fill="#64748b")
    d.rectangle([0,0,479,59], outline="#334155", width=1)
    return leg

# load val examples
examples = []
with open(ROOT / "data/coco_val.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 5: break
        examples.append(json.loads(line))

# --- BASE MODEL ---
print("Loading base model...")
processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
base = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True).eval()

base_results = []
for i, ex in enumerate(examples):
    img_path = ex["messages"][0]["content"][0]["image"]
    gt = parse_json(ex["messages"][1]["content"])
    gt_boxes = [p["bbox"] for p in gt.get("people", [])] if gt else []
    out, ms, W, H = run_model(base, processor, img_path, PROMPT_BASE)
    parsed = parse_json(out)
    people = parsed.get("people", []) if parsed else []
    base_results.append({"img": img_path, "people": people, "gt": gt_boxes, "ms": ms, "W": W, "H": H})
    print(f"  base img {i+1}: {len(people)} detected, {ms:.0f}ms")

del base; torch.cuda.empty_cache()

# --- LORA MODEL ---
print("Loading LoRA model...")
lora = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True).eval()
lora = PeftModel.from_pretrained(lora, ADAPTER).eval()

lora_results = []
for i, ex in enumerate(examples):
    img_path = ex["messages"][0]["content"][0]["image"]
    gt = parse_json(ex["messages"][1]["content"])
    gt_boxes = [p["bbox"] for p in gt.get("people", [])] if gt else []
    out, ms, W, H = run_model(lora, processor, img_path, PROMPT_LORA)
    parsed = parse_json(out)
    people = parsed.get("people", []) if parsed else []
    lora_results.append({"img": img_path, "people": people, "gt": gt_boxes, "ms": ms, "W": W, "H": H})
    print(f"  lora img {i+1}: {len(people)} detected, {ms:.0f}ms")

del lora; torch.cuda.empty_cache()

# --- VISUALIZE ---
def iou(a, b):
    ix1,iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2,iy2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    if not inter: return 0.
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)

def mean_iou(gt, pred):
    if not gt or not pred: return 0.
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    cost = np.array([[1-iou(g,p) for p in pred] for g in gt])
    r,c = linear_sum_assignment(cost)
    return float(np.mean([1-cost[i,j] for i,j in zip(r,c)]))

for i in range(5):
    br, lr = base_results[i], lora_results[i]
    W, H = br["W"], br["H"]

    base_iou = mean_iou(br["gt"], [p["bbox"] for p in br["people"] if len(p.get("bbox",[]))==4])
    lora_iou = mean_iou(lr["gt"], [p["bbox"] for p in lr["people"] if len(p.get("bbox",[]))==4])

    base_panel = draw(br["img"], br["people"], br["gt"], f"  BASE  |  {len(br['people'])} detected", W, H)
    lora_panel = draw(lr["img"], lr["people"], lr["gt"], f"  LoRA  |  {len(lr['people'])} detected", W, H)
    base_leg   = make_legend(len(br["gt"]), len(br["people"]), base_iou, br["ms"])
    lora_leg   = make_legend(len(lr["gt"]), len(lr["people"]), lora_iou, lr["ms"])

    ph = base_panel.height
    combined = Image.new("RGB", (480*2 + 8, ph + 60 + 36), "#0f172a")

    # image number label
    num_bar = Image.new("RGB", (968, 36), "#1e293b")
    nd = ImageDraw.Draw(num_bar)
    nd.text((8, 10), f"Image {i+1} of 5  —  GT: {len(br['gt'])} people  (white outline = ground truth)", fill="#64748b")
    combined.paste(num_bar, (0, 0))
    combined.paste(base_panel, (0, 36))
    combined.paste(lora_panel, (488, 36))
    combined.paste(base_leg, (0, 36 + ph))
    combined.paste(lora_leg, (488, 36 + ph))

    out_path = OUT_DIR / f"comparison_{i+1}.png"
    combined.save(out_path)
    print(f"Saved → {out_path}")

print("Done. All 5 comparisons saved to viz_output/5way/")
