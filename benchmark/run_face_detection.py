#!/usr/bin/env python
"""
run_face_detection.py — Run 3 face detection models on the people_images folder.

Models
------
  1. MTCNN       – Multi-task Cascaded CNN (facenet-pytorch)
  2. RetinaFace  – insightface buffalo_sc (ONNX Runtime)
  3. YOLOv8-Face – YOLOv8n fine-tuned on WiderFace

Output
------
  benchmark/results/face_detection_<timestamp>.json   — per-image results
  benchmark/results/face_detection_vis/               — annotated images

Usage
-----
    cd benchmark/
    python run_face_detection.py
    python run_face_detection.py --models mtcnn retinaface
    python run_face_detection.py --conf 0.3 --device cpu
    python run_face_detection.py --images /path/to/images
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models.mtcnn_face import MTCNNFaceModel
from models.retinaface import RetinaFaceModel
from models.yolov8_face import YOLOv8FaceModel
from models.base import DetectionResult

# ── defaults ─────────────────────────────────────────────────────────────────

DEFAULT_IMAGES_DIR = Path(__file__).parent.parent / "people_images"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MODEL_CONFIGS = {
    "mtcnn":      {"cls": MTCNNFaceModel,   "kwargs": {}},
    "retinaface": {"cls": RetinaFaceModel,  "kwargs": {}},
    "yolov8face": {"cls": YOLOv8FaceModel,  "kwargs": {}},
}


# ── helpers ───────────────────────────────────────────────────────────────────

def collect_images(images_dir: Path) -> list[Path]:
    paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    return paths


def draw_detections(image_path: Path, result: DetectionResult, model_name: str) -> np.ndarray:
    """Return a BGR numpy array with bounding boxes drawn."""
    import cv2
    from PIL import Image as PILImage

    # Load via PIL to handle all formats, then convert to BGR for cv2
    pil = PILImage.open(image_path).convert("RGB")
    img = np.array(pil)[:, :, ::-1].copy()  # RGB → BGR

    color = (0, 255, 0)   # green boxes
    for det in result.detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(img, label, (x1 + 1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Overlay model name + face count
    header = f"{model_name}: {len(result.detections)} face(s)  {result.latency_ms:.1f} ms"
    cv2.putText(img, header, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    return img


def save_visualization(
    images: list[Path],
    results: dict[str, DetectionResult],
    vis_dir: Path,
    stem: str,
) -> None:
    """Save one annotated image per model for this image."""
    import cv2

    for model_name, result in results.items():
        annotated = draw_detections(images[0], result, model_name)
        out_path = vis_dir / f"{stem}__{model_name}.jpg"
        cv2.imwrite(str(out_path), annotated)


def run_model_on_images(
    model,
    images: list[Path],
    conf: float,
) -> list[dict]:
    """Run model on all images, return list of per-image result dicts."""
    per_image = []
    for img_path in images:
        result = model.detect(str(img_path), conf_threshold=conf)
        per_image.append({
            "image":      img_path.name,
            "n_faces":    len(result.detections),
            "latency_ms": result.latency_ms,
            "error":      result.error,
            "detections": [
                {
                    "bbox":       d.bbox,
                    "confidence": d.confidence,
                }
                for d in result.detections
            ],
        })
        status = (
            f"  {img_path.name:<40} {len(result.detections):>2} face(s)  "
            f"{result.latency_ms:>7.1f} ms"
        )
        if result.error:
            status += f"  ERROR: {result.error}"
        print(status)
    return per_image


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Face detection benchmark on people_images")
    parser.add_argument(
        "--images",
        default=str(DEFAULT_IMAGES_DIR),
        help=f"Folder of images to run on (default: {DEFAULT_IMAGES_DIR})",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_CONFIGS.keys()),
        choices=list(MODEL_CONFIGS.keys()),
        help="Which models to run",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device: 'cuda' or 'cpu' (default: auto)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results"),
        help="Directory to save results (default: benchmark/results/)",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Skip saving annotated images",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    images_dir = Path(args.images)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device:     {device}")
    print(f"Images dir: {images_dir}")

    images = collect_images(images_dir)
    if not images:
        print(f"No images found in {images_dir}")
        sys.exit(1)
    print(f"Images:     {len(images)} found\n")

    # Visualisation directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_dir = out_dir / f"face_detection_vis_{ts}"
    if not args.no_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}

    for model_key in args.models:
        cfg = MODEL_CONFIGS[model_key]
        print(f"\n{'='*60}")
        print(f"Model: {model_key}")
        print(f"{'='*60}")

        kwargs = dict(cfg["kwargs"])
        kwargs["device"] = device
        model = cfg["cls"](**kwargs)
        model.load()

        per_image = run_model_on_images(model, images, conf=args.conf)
        model.unload()

        latencies = [r["latency_ms"] for r in per_image if not r["error"]]
        total_faces = sum(r["n_faces"] for r in per_image)

        print(f"\n  Total faces detected: {total_faces}")
        if latencies:
            print(f"  Mean latency:         {np.mean(latencies):.1f} ms")
            print(f"  FPS:                  {1000 / np.mean(latencies):.1f}")

        all_results[model_key] = {
            "model_name":      model_key,
            "conf_threshold":  args.conf,
            "device":          device,
            "summary": {
                "total_faces":     total_faces,
                "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
                "std_latency_ms":  float(np.std(latencies))  if latencies else 0.0,
                "fps":             float(1000 / np.mean(latencies)) if latencies else 0.0,
                "n_images":        len(images),
                "n_errors":        sum(1 for r in per_image if r["error"]),
            },
            "per_image": per_image,
        }

        # Save visualisations
        if not args.no_vis:
            import cv2
            from PIL import Image as PILImage

            for img_path in images:
                # Re-run detect for visualization (model is unloaded — use saved results)
                img_data = next(r for r in per_image if r["image"] == img_path.name)
                # Reconstruct a DetectionResult-like object for drawing
                from models.base import Detection, DetectionResult as DR
                detections = [
                    Detection(
                        bbox=d["bbox"],
                        confidence=d["confidence"],
                        class_id=0,
                        class_name="face",
                    )
                    for d in img_data["detections"]
                ]
                result = DR(detections=detections, latency_ms=img_data["latency_ms"])

                annotated = draw_detections(img_path, result, model_key)
                out_path = vis_dir / f"{img_path.stem}__{model_key}.jpg"
                cv2.imwrite(str(out_path), annotated)

    # ── Save JSON results ────────────────────────────────────────────────────
    out_json = out_dir / f"face_detection_{ts}.json"
    payload = {
        "timestamp":  ts,
        "images_dir": str(images_dir),
        "conf":       args.conf,
        "device":     device,
        "n_images":   len(images),
        "images":     [str(p) for p in images],
        "models":     all_results,
    }
    out_json.write_text(json.dumps(payload, indent=2))

    print(f"\n{'='*60}")
    print(f"Results saved → {out_json}")
    if not args.no_vis:
        print(f"Visuals saved → {vis_dir}/")

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'Model':<15} {'Faces':>6} {'Mean ms':>8} {'FPS':>6} {'Errors':>7}")
    print("-" * 50)
    for key, res in all_results.items():
        s = res["summary"]
        print(
            f"{key:<15} {s['total_faces']:>6} {s['mean_latency_ms']:>8.1f} "
            f"{s['fps']:>6.1f} {s['n_errors']:>7}"
        )


if __name__ == "__main__":
    main()
