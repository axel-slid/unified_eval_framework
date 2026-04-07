#!/usr/bin/env python
"""
finetune/prepare_dataset.py — Download COCO 2017 and build an instruction-tuning
dataset for Qwen3-VL-4B bbox detection.

Strategy
--------
We filter COCO train2017 for images that contain at least one 'person' annotation
AND at least one "indoor office" co-category (chair, dining table, laptop, tv,
keyboard, mouse, bottle, cup, couch, book) as a proxy for meeting-room scenes.
Each image becomes one training example:

  User:    <image> + detection prompt (includes W×H so model sees pixel scale)
  Assistant: {"people": [{"id":"P1","bbox":[x1,y1,x2,y2],"role":"participant",
                           "is_target_speaker":false}, ...]}

Notes on role / is_target_speaker
----------------------------------
COCO has no role or speaker labels.  We default:
  • role = "participant"  (fine-tune only teaches detection; role/speaker heads
                           are added in a later LoRA stage with annotated data)
  • is_target_speaker = false

Outputs
-------
  data/coco_train.jsonl   — training split (90 %)
  data/coco_val.jsonl     — validation split (10 %)
  data/images/            — symlinks to COCO images (or copies if needed)

Usage
-----
  cd finetune/
  python prepare_dataset.py
  python prepare_dataset.py --max-images 5000   # quick smoke-test
  python prepare_dataset.py --no-filter         # all person images, no scene filter
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
DATA_DIR   = ROOT / "data"
IMG_DIR    = DATA_DIR / "images"
ANNO_DIR   = DATA_DIR / "annotations"
COCO_TRAIN = ANNO_DIR / "instances_train2017.json"
COCO_VAL   = ANNO_DIR / "instances_val2017.json"

COCO_ANNO_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
# Base URL for COCO images
COCO_IMG_BASE = "http://images.cocodataset.org/train2017/"

# Indoor/office co-categories used to filter for meeting-room-like images
INDOOR_CATS = {
    "chair", "dining table", "laptop", "tv", "keyboard", "mouse",
    "bottle", "cup", "couch", "book", "monitor", "cell phone",
}

DETECTION_PROMPT = (
    "You are analyzing an image ({width}x{height} pixels).\n\n"
    "Detect ALL people visible in the image and output their bounding boxes.\n\n"
    "Output ONLY valid JSON, no extra text:\n"
    '{{"people": [{{"id": "P1", "bbox": [x1, y1, x2, y2]}}, ...]}}\n\n'
    "Bounding boxes must be pixel coordinates [x1, y1, x2, y2] within the "
    "image ({width}x{height}). Do not output any text outside the JSON object."
)


# ── Download helpers ──────────────────────────────────────────────────────────

def download_annotations():
    if COCO_TRAIN.exists() and COCO_VAL.exists():
        print("[data] Annotations already present, skipping download.")
        return
    ANNO_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "annotations.zip"
    if not zip_path.exists():
        print(f"[data] Downloading COCO annotations ({COCO_ANNO_URL}) ...")
        urllib.request.urlretrieve(COCO_ANNO_URL, zip_path)
    print("[data] Extracting annotations ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_DIR)
    zip_path.unlink()
    print("[data] Annotations ready.")


def download_image(image_id: int, file_name: str) -> Path | None:
    """Download a single COCO image if not already cached. Returns local path."""
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    dest = IMG_DIR / file_name
    if dest.exists():
        return dest
    url = COCO_IMG_BASE + file_name
    try:
        urllib.request.urlretrieve(url, dest)
        return dest
    except Exception as e:
        print(f"  [warn] Failed to download {file_name}: {e}")
        return None


# ── COCO loading + filtering ──────────────────────────────────────────────────

def load_coco(anno_path: Path) -> tuple[dict, dict, dict]:
    """
    Returns:
      cat_id_to_name  — {cat_id: name}
      img_id_to_info  — {img_id: {id, file_name, width, height}}
      img_id_to_anns  — {img_id: [ann, ...]}  (all annotations for that image)
    """
    print(f"[data] Loading {anno_path.name} ...")
    with open(anno_path) as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    img_id_to_info = {img["id"]: img for img in coco["images"]}
    img_id_to_anns: dict[int, list] = {}
    for ann in coco["annotations"]:
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    return cat_id_to_name, img_id_to_info, img_id_to_anns


def filter_images(
    cat_id_to_name: dict,
    img_id_to_info: dict,
    img_id_to_anns: dict,
    apply_indoor_filter: bool,
) -> list[int]:
    """Return image IDs that pass the person + (optional) indoor filter."""
    person_cat_ids = {cid for cid, name in cat_id_to_name.items() if name == "person"}
    indoor_cat_ids = {cid for cid, name in cat_id_to_name.items() if name in INDOOR_CATS}

    kept = []
    for img_id, anns in img_id_to_anns.items():
        ann_cats = {a["category_id"] for a in anns}
        has_person = bool(ann_cats & person_cat_ids)
        if not has_person:
            continue
        if apply_indoor_filter and not (ann_cats & indoor_cat_ids):
            continue
        kept.append(img_id)

    return kept


# ── Build one training example ────────────────────────────────────────────────

def build_example(
    img_info: dict,
    person_anns: list[dict],
    img_local_path: Path,
) -> dict:
    """
    Build one SFT example in the Qwen3-VL chat format expected by TRL SFTTrainer.
    """
    W, H = img_info["width"], img_info["height"]

    people = []
    for i, ann in enumerate(person_anns, start=1):
        # COCO bbox is [x, y, w, h] — convert to [x1, y1, x2, y2]
        x, y, bw, bh = ann["bbox"]
        x1, y1 = round(x), round(y)
        x2, y2 = round(x + bw), round(y + bh)
        # Clamp to image bounds
        x1, x2 = max(0, x1), min(W, x2)
        y1, y2 = max(0, y1), min(H, y2)
        # Skip degenerate boxes
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        people.append({
            "id":   f"P{i}",
            "bbox": [x1, y1, x2, y2],
        })

    if not people:
        return None

    prompt = DETECTION_PROMPT.format(width=W, height=H)
    assistant_reply = json.dumps({"people": people}, separators=(",", ":"))

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(img_local_path)},
                    {"type": "text",  "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": assistant_reply,
            },
        ]
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=None,
                        help="Cap number of images (for quick tests)")
    parser.add_argument("--no-filter", action="store_true",
                        help="Skip indoor co-category filter (use all person images)")
    parser.add_argument("--val-frac", type=float, default=0.10,
                        help="Fraction of data held out for validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip image downloads (use only locally cached images)")
    args = parser.parse_args()

    random.seed(args.seed)
    download_annotations()

    cat_id_to_name, img_id_to_info, img_id_to_anns = load_coco(COCO_TRAIN)
    person_cat_ids = {cid for cid, name in cat_id_to_name.items() if name == "person"}

    img_ids = filter_images(
        cat_id_to_name, img_id_to_info, img_id_to_anns,
        apply_indoor_filter=not args.no_filter,
    )
    random.shuffle(img_ids)
    print(f"[data] {len(img_ids)} images pass filter (indoor_filter={'on' if not args.no_filter else 'off'})")

    if args.max_images:
        img_ids = img_ids[: args.max_images]
        print(f"[data] Capped to {len(img_ids)} images")

    # Split
    cut = int(len(img_ids) * (1 - args.val_frac))
    train_ids, val_ids = img_ids[:cut], img_ids[cut:]
    print(f"[data] train={len(train_ids)}  val={len(val_ids)}")

    def process_split(ids: list[int], split: str):
        out_path = DATA_DIR / f"coco_{split}.jsonl"
        written = skipped = 0
        with open(out_path, "w") as f:
            for img_id in ids:
                info = img_id_to_info[img_id]
                anns = img_id_to_anns.get(img_id, [])
                person_anns = [a for a in anns if a["category_id"] in person_cat_ids]
                if not person_anns:
                    skipped += 1
                    continue

                if args.skip_download:
                    local_path = IMG_DIR / info["file_name"]
                    if not local_path.exists():
                        skipped += 1
                        continue
                else:
                    local_path = download_image(img_id, info["file_name"])
                    if local_path is None:
                        skipped += 1
                        continue

                example = build_example(info, person_anns, local_path)
                if example is None:
                    skipped += 1
                    continue

                f.write(json.dumps(example) + "\n")
                written += 1
                if written % 500 == 0:
                    print(f"  [{split}] {written} written, {skipped} skipped ...")

        print(f"[data] {split}: {written} examples → {out_path}  (skipped {skipped})")
        return out_path

    train_path = process_split(train_ids, "train")
    val_path   = process_split(val_ids,   "val")

    # Write a dataset card
    card = {
        "source":       "COCO 2017 train split",
        "task":         "person bbox detection (VLM instruction tuning)",
        "indoor_filter": not args.no_filter,
        "indoor_cats":  sorted(INDOOR_CATS),
        "train_file":   str(train_path),
        "val_file":     str(val_path),
        "note": (
            "role and is_target_speaker are placeholder values. "
            "A second LoRA stage with domain-annotated data is needed "
            "to learn these fields."
        ),
    }
    (DATA_DIR / "dataset_card.json").write_text(json.dumps(card, indent=2))
    print("[data] Done.")


if __name__ == "__main__":
    main()
