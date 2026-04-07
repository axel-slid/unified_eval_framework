#!/usr/bin/env python
"""
run_approach_a_vlm_only.py — Approach A: VLM-only meeting room analysis.

A single inference per image, no CV pre-processing or cropping.
The VLM receives the full meeting room image and a single structured prompt
asking it to simultaneously:
  1. Detect all people (bounding boxes)
  2. Label each person as participant / non-participant
  3. Identify the target speaker (is_target_speaker flag)

Output JSON schema (per image):
  {"people": [
      {"id": "P1", "bbox": [x1, y1, x2, y2],
       "role": "participant" | "non-participant",
       "is_target_speaker": true | false,
       "reason": "<short reason>"},
      ...
  ]}

Results saved to benchmark/results/approach_a_<timestamp>.json

Usage
-----
    cd benchmark/
    python run_approach_a_vlm_only.py
    python run_approach_a_vlm_only.py --vlm smolvlm
    python run_approach_a_vlm_only.py --vlm qwen3vl_4b --images-dir ../people_images
    python run_approach_a_vlm_only.py --max-new-tokens 768
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

PEOPLE_DIR  = ROOT.parent / "people_images"
RESULTS_DIR = ROOT / "results"

# ── Prompt ────────────────────────────────────────────────────────────────────
# Inject image dimensions at runtime so bbox coordinates are unambiguous.
PROMPT_TEMPLATE = """\
You are analyzing a meeting room image ({width}x{height} pixels).

Your tasks:
1. Detect ALL people visible in the image.
2. For each person, label their role:
   - "participant": seated or standing at the table, engaged with the meeting.
   - "non-participant": passing by, in background, not engaged.
3. Identify who is the TARGET SPEAKER — the one person currently talking or \
presenting. Use mouth position, posture, and where others are looking as cues. \
At most one person should be marked is_target_speaker=true. If nobody is \
clearly speaking, set all to false.

Output ONLY valid JSON, no extra text, in this exact schema:
{{"people": [{{"id": "P1", "bbox": [x1, y1, x2, y2], "role": "participant", \
"is_target_speaker": false, "reason": "seated at table, attentive"}}, ...]}}

Bounding boxes must be pixel coordinates [x1, y1, x2, y2] within the image \
({width}x{height}). Do not output any text outside the JSON object."""


# ──────────────────────────────────────────────────────────────────────────────
# JSON parsing
# ──────────────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict | None:
    """Try to extract the first valid JSON object from a (possibly noisy) string."""
    # Direct parse (model may return clean JSON)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    fenced = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    fenced = re.sub(r"\s*```$", "", fenced, flags=re.MULTILINE).strip()
    try:
        return json.loads(fenced)
    except json.JSONDecodeError:
        pass

    # Find the first {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _validate_person(p: dict, img_w: int, img_h: int) -> dict:
    """Normalise and sanity-check a single person dict. Returns cleaned dict."""
    bbox = p.get("bbox", [])
    if len(bbox) == 4:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        # Clamp to image bounds
        x1 = max(0.0, min(x1, img_w))
        y1 = max(0.0, min(y1, img_h))
        x2 = max(0.0, min(x2, img_w))
        y2 = max(0.0, min(y2, img_h))
        bbox = [x1, y1, x2, y2]
    else:
        bbox = None  # malformed

    role = str(p.get("role", "")).lower()
    if role not in ("participant", "non-participant"):
        role = None

    return {
        "id":               p.get("id", "?"),
        "bbox":             bbox,
        "role":             role,
        "is_target_speaker": bool(p.get("is_target_speaker", False)),
        "reason":           str(p.get("reason", "")),
        "bbox_valid":       bbox is not None,
        "role_valid":       role is not None,
    }


def parse_vlm_output(
    raw: str, img_w: int, img_h: int
) -> tuple[list[dict], bool]:
    """
    Parse raw VLM text into a list of validated person dicts.

    Returns (people_list, parse_success).
    """
    parsed = _extract_json(raw)
    if parsed is None:
        return [], False

    people_raw = parsed.get("people", [])
    if not isinstance(people_raw, list):
        return [], False

    people = [_validate_person(p, img_w, img_h) for p in people_raw]
    return people, True


# ──────────────────────────────────────────────────────────────────────────────
# VLM configs  (mirror run_pipeline_people_analysis.py)
# ──────────────────────────────────────────────────────────────────────────────

VLM_CONFIGS = {
    "smolvlm": {
        "class":      "SmolVLMModel",
        "model_path": "/mnt/shared/dils/models/SmolVLM2-2.2B-Instruct",
        "dtype":      "bfloat16",
    },
    "qwen3vl_4b": {
        "class":      "Qwen3VLModel",
        "model_path": "/mnt/shared/dils/models/Qwen3-VL-4B-Instruct",
        "dtype":      "bfloat16",
    },
}


def _make_model_cfg(key: str, vcfg: dict, max_new_tokens: int):
    from config import ModelConfig, GenerationConfig
    return ModelConfig(
        key=key,
        enabled=True,
        cls_name=vcfg["class"],
        model_path=vcfg["model_path"],
        dtype=vcfg["dtype"],
        generation=GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────────────────

def run_approach_a(
    image_paths: list[Path],
    vlm_keys: list[str],
    max_new_tokens: int,
) -> dict:
    """
    Run Approach A on all images for each VLM.
    Returns nested results dict: { vlm_key: { image_stem: {...} } }
    """
    from models import MODEL_REGISTRY

    all_results: dict[str, dict] = {}

    for vlm_key in vlm_keys:
        if vlm_key not in VLM_CONFIGS:
            print(f"[Approach A] Unknown VLM '{vlm_key}', skipping")
            continue

        vcfg = VLM_CONFIGS[vlm_key]
        cls  = MODEL_REGISTRY.get(vcfg["class"])
        if cls is None:
            print(f"[Approach A] {vcfg['class']} not available, skipping")
            continue

        mcfg  = _make_model_cfg(vlm_key, vcfg, max_new_tokens)
        model = cls(mcfg)

        print(f"\n[Approach A] {vlm_key}")
        model.load()

        image_results: dict[str, dict] = {}

        for img_path in image_paths:
            stem = img_path.stem

            # Get image dimensions for the prompt
            with Image.open(img_path) as im:
                img_w, img_h = im.size

            prompt = PROMPT_TEMPLATE.format(width=img_w, height=img_h)

            result = model.run(str(img_path), prompt)

            people, parse_ok = parse_vlm_output(result.response, img_w, img_h)

            n_participants   = sum(1 for p in people if p["role"] == "participant")
            n_non_participants = sum(1 for p in people if p["role"] == "non-participant")
            n_speakers       = sum(1 for p in people if p["is_target_speaker"])
            n_valid_bbox     = sum(1 for p in people if p["bbox_valid"])

            image_results[stem] = {
                "image_path":        str(img_path),
                "image_size":        [img_w, img_h],
                "raw_response":      result.response,
                "parse_success":     parse_ok,
                "latency_ms":        result.latency_ms,
                "n_people":          len(people),
                "n_participants":    n_participants,
                "n_non_participants":n_non_participants,
                "n_target_speakers": n_speakers,
                "n_valid_bboxes":    n_valid_bbox,
                "people":            people,
                "error":             result.error,
            }

            status = (
                f"  {img_path.name}: "
                f"{len(people)} people  "
                f"(parse={'OK' if parse_ok else 'FAIL'}  "
                f"participants={n_participants}  "
                f"speaker={n_speakers}  "
                f"latency={result.latency_ms:.0f}ms)"
            )
            if result.error:
                status += f"  [ERROR: {result.error}]"
            print(status)

        model.unload()
        all_results[vlm_key] = image_results

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Approach A: VLM-only meeting room analysis")
    parser.add_argument("--vlm", nargs="+",
                        default=list(VLM_CONFIGS.keys()),
                        choices=list(VLM_CONFIGS.keys()),
                        help="Which VLMs to run")
    parser.add_argument("--images-dir", default=str(PEOPLE_DIR),
                        help="Directory containing meeting room images")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Max tokens to generate per image (default: 512)")
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    images_dir  = Path(args.images_dir)
    image_paths = (
        sorted(images_dir.glob("*.jpeg")) +
        sorted(images_dir.glob("*.jpg"))  +
        sorted(images_dir.glob("*.png"))
    )
    if not image_paths:
        print(f"No images found in {images_dir}")
        sys.exit(1)
    print(f"Images: {len(image_paths)}  |  VLMs: {args.vlm}")

    results = run_approach_a(image_paths, args.vlm, args.max_new_tokens)

    # ── Aggregate stats per VLM ───────────────────────────────────────────────
    summary: dict[str, dict] = {}
    for vlm_key, img_results in results.items():
        n_images      = len(img_results)
        n_parsed      = sum(1 for r in img_results.values() if r["parse_success"])
        latencies     = [r["latency_ms"] for r in img_results.values() if r["latency_ms"] > 0]
        total_people  = sum(r["n_people"] for r in img_results.values())
        total_speakers= sum(r["n_target_speakers"] for r in img_results.values())
        import numpy as np
        summary[vlm_key] = {
            "n_images":           n_images,
            "parse_success_rate": n_parsed / n_images if n_images else 0.0,
            "mean_latency_ms":    float(np.mean(latencies)) if latencies else 0.0,
            "std_latency_ms":     float(np.std(latencies))  if latencies else 0.0,
            "fps":                1000.0 / float(np.mean(latencies)) if latencies else 0.0,
            "total_people":       total_people,
            "total_target_speakers": total_speakers,
        }
        print(
            f"\n[{vlm_key}] summary: "
            f"parse={summary[vlm_key]['parse_success_rate']:.0%}  "
            f"mean_latency={summary[vlm_key]['mean_latency_ms']:.0f}ms  "
            f"fps={summary[vlm_key]['fps']:.2f}  "
            f"total_people={total_people}"
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"approach_a_{ts}.json"

    payload = {
        "approach":        "A_vlm_only",
        "timestamp":       ts,
        "images_dir":      str(images_dir),
        "images":          [str(p) for p in image_paths],
        "vlm_configs":     VLM_CONFIGS,
        "max_new_tokens":  args.max_new_tokens,
        "summary":         summary,
        "results":         results,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
