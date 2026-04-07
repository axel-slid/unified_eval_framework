#!/usr/bin/env python
"""
face_detection/run.py — Run MTCNN, RetinaFace, and YOLOv8-Face on an image folder.

Usage
-----
    cd benchmark/face_detection/
    python run.py
    python run.py --images ../../people_images --conf 0.3 --device cpu
    python run.py --models mtcnn retinaface
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from models.mtcnn_face  import MTCNNFaceModel
from models.retinaface  import RetinaFaceModel
from models.yolov8_face import YOLOv8FaceModel
from models.base        import Detection, DetectionResult

DEFAULT_IMAGES = HERE.parent.parent / "people_images"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MODEL_REGISTRY = {
    "mtcnn":      MTCNNFaceModel,
    "retinaface": RetinaFaceModel,
    "yolov8face": YOLOv8FaceModel,
}
MODEL_DISPLAY = {
    "mtcnn":      "MTCNN",
    "retinaface": "RetinaFace",
    "yolov8face": "YOLOv8-Face",
}


# ── visualisation ─────────────────────────────────────────────────────────────

def annotate(image_path: Path, result: DetectionResult, model_label: str) -> np.ndarray:
    pil = PILImage.open(image_path).convert("RGB")
    img = np.array(pil)[:, :, ::-1].copy()   # RGB → BGR
    color = (50, 220, 50)
    for det in result.detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(img, label, (x1 + 1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return img


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images",  default=str(DEFAULT_IMAGES))
    parser.add_argument("--models",  nargs="+", default=list(MODEL_REGISTRY.keys()),
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--conf",    type=float, default=0.25)
    parser.add_argument("--device",  default=None)
    parser.add_argument("--out-dir", default=str(HERE / "results"))
    parser.add_argument("--no-vis",  action="store_true")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    images_dir = Path(args.images)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in images_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images:
        sys.exit(f"No images found in {images_dir}")

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_dir = out_dir / f"vis_{ts}"
    if not args.no_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}  |  Images: {len(images)}  |  Models: {args.models}\n")

    all_results: dict = {}

    for key in args.models:
        label = MODEL_DISPLAY[key]
        print(f"\n{'─'*55}\n  {label}\n{'─'*55}")
        model = MODEL_REGISTRY[key](device=device)
        model.load()

        per_image = []
        for img_path in images:
            res = model.detect(str(img_path), conf_threshold=args.conf)
            per_image.append({
                "image":      img_path.name,
                "n_faces":    len(res.detections),
                "latency_ms": res.latency_ms,
                "error":      res.error,
                "detections": [{"bbox": d.bbox, "confidence": d.confidence}
                               for d in res.detections],
            })
            status = (f"  {img_path.name:<42} {len(res.detections):>2} face(s)"
                      f"  {res.latency_ms:>7.1f} ms")
            if res.error:
                status += f"  ERR: {res.error}"
            print(status)

            if not args.no_vis:
                ann = annotate(img_path, res, label)
                cv2.imwrite(str(vis_dir / f"{img_path.stem}__{key}.jpg"), ann)

        model.unload()

        lats = [r["latency_ms"] for r in per_image if not r["error"]]
        total = sum(r["n_faces"] for r in per_image)
        print(f"\n  Total faces: {total}  |  "
              f"Mean: {np.mean(lats):.1f} ms  |  FPS: {1000/np.mean(lats):.1f}")

        all_results[key] = {
            "label":  label,
            "conf":   args.conf,
            "device": device,
            "summary": {
                "total_faces":     total,
                "mean_latency_ms": float(np.mean(lats)) if lats else 0.0,
                "std_latency_ms":  float(np.std(lats))  if lats else 0.0,
                "fps":             float(1000 / np.mean(lats)) if lats else 0.0,
                "n_images":        len(images),
                "n_errors":        sum(1 for r in per_image if r["error"]),
            },
            "per_image": per_image,
        }

    out_json = out_dir / f"results_{ts}.json"
    out_json.write_text(json.dumps({
        "timestamp":  ts,
        "images_dir": str(images_dir),
        "conf":       args.conf,
        "device":     device,
        "images":     [str(p) for p in images],
        "models":     all_results,
    }, indent=2))

    print(f"\n\n{'='*55}")
    print(f"{'Model':<15} {'Faces':>6} {'Mean ms':>9} {'FPS':>7} {'Errors':>7}")
    print(f"{'─'*55}")
    for key, res in all_results.items():
        s = res["summary"]
        print(f"{res['label']:<15} {s['total_faces']:>6} {s['mean_latency_ms']:>9.1f}"
              f" {s['fps']:>7.1f} {s['n_errors']:>7}")
    print(f"\nJSON  → {out_json}")
    if not args.no_vis:
        print(f"Vis   → {vis_dir}/")


if __name__ == "__main__":
    main()
