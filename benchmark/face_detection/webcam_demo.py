#!/usr/bin/env python
"""
webcam_demo.py — Live webcam demo: face detection + Qwen3-VL classification.

Stage 1 (every frame): YOLOv8-Face detects faces; bboxes are dilated 2×.
Stage 2 (every N sec):  Qwen3-VL classifies each face crop as
                        Participant/Non-Participant and Talking/Silent.

Usage
-----
    cd benchmark/face_detection/
    python webcam_demo.py
    python webcam_demo.py --vlm qwen3vl_4b_int8 --vlm-interval 5
    python webcam_demo.py --camera 1 --conf 0.3 --no-vlm
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

HERE       = Path(__file__).parent
BENCH_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BENCH_ROOT))

# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPT_PARTICIPANT = (
    "You are given two images: Image 1 is the full meeting room, Image 2 is a "
    "crop of one detected face from that room. "
    "Using both images, determine: is this person a genuine meeting participant "
    "(seated or standing at the table, engaged in the meeting)? "
    "Answer YES or NO, then give one short reason."
)

PROMPT_TALKING = (
    "You are given two images: Image 1 is the full meeting room, Image 2 is a "
    "crop of one detected face from that room. "
    "Using both images, is this person currently talking or speaking? "
    "Look at their mouth, facial expression, and body posture for cues. "
    "Answer YES or NO, then give one short reason."
)

VLM_CONFIGS = {
    "qwen3vl_4b": {
        "class":      "Qwen3VLModel",
        # On the cluster: /mnt/shared/dils/models/Qwen3-VL-4B-Instruct
        # Locally: HF model ID — will download/cache on first run
        "model_path": "Qwen/Qwen3-VL-4B-Instruct",
        "dtype":      "bfloat16",
        "backend":    "qwen3vl",
    },
    "qwen3vl_4b_int8": {
        "class":      "Qwen3VLModel",
        "model_path": "/mnt/shared/dils/projects/logitech/unified_eval_framework/models/Qwen3-VL-4B-Instruct-int8",
        "dtype":      "bfloat16",
        "backend":    "qwen3vl",
    },
    # ── Gemma 4 E2B via Unsloth ───────────────────────────────────────────────
    "gemma4_e2b_4bit": {
        "class":      "GemmaUnslothModel",
        "model_path": "",   # pulled from gemma_unsloth.QUANT_CONFIGS
        "dtype":      "bfloat16",
        "backend":    "unsloth",
    },
    "gemma4_e2b_bf16": {
        "class":      "GemmaUnslothModel",
        "model_path": "",
        "dtype":      "bfloat16",
        "backend":    "unsloth",
    },
}

# ── Label colors (BGR) ────────────────────────────────────────────────────────
#  participant=True  + talking=True  → green
#  participant=True  + talking=False → cyan
#  participant=False + talking=True  → orange
#  participant=False + talking=False → red
#  unknown/pending                   → gray

def _label_and_color(participant: bool | None, talking: bool | None):
    if participant is None and talking is None:
        return "Classifying...", (160, 160, 160)
    p_str = "Participant" if participant else ("Non-Participant" if participant is False else "?")
    t_str = "Talking"     if talking     else ("Silent"         if talking    is False else "?")
    label = f"{p_str} | {t_str}"
    if participant is True  and talking is True:  color = (0, 200, 0)
    elif participant is True  and talking is False: color = (200, 200, 0)
    elif participant is False and talking is True:  color = (0, 165, 255)
    elif participant is False and talking is False: color = (0, 0, 220)
    else:                                           color = (160, 160, 160)
    return label, color


# ── Geometry helpers ──────────────────────────────────────────────────────────

def dilate_bbox(bbox: list[float], scale: float, W: int, H: int) -> list[float]:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw = (x2 - x1) / 2 * scale
    hh = (y2 - y1) / 2 * scale
    return [
        max(0, cx - hw),
        max(0, cy - hh),
        min(W, cx + hw),
        min(H, cy + hh),
    ]


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def match_vlm_result(
    bbox: list[float],
    cached: list[dict],
    W: int,
    H: int,
) -> dict | None:
    """Match a live bbox to the best cached VLM result.

    Tries IoU first (threshold 0.10); falls back to nearest centroid within
    30% of the frame diagonal so labels persist even after the face moves.
    """
    best, best_iou = None, 0.10
    for c in cached:
        score = iou(bbox, c["dilated_bbox"])
        if score > best_iou:
            best_iou = score
            best = c
    if best:
        return best
    # Centroid fallback
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    diag = (W ** 2 + H ** 2) ** 0.5
    best_dist = diag * 0.30
    for c in cached:
        cb = c["dilated_bbox"]
        ccx, ccy = (cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2
        d = ((cx - ccx) ** 2 + (cy - ccy) ** 2) ** 0.5
        if d < best_dist:
            best_dist = d
            best = c
    return best


def parse_yes_no(text: str) -> bool | None:
    t = text.strip().upper()
    if t.startswith("YES"):  return True
    if t.startswith("NO"):   return False
    if "YES" in t[:20]:      return True
    if "NO"  in t[:20]:      return False
    return None


# ── VLM worker ────────────────────────────────────────────────────────────────

class VLMWorker:
    """Runs Qwen3-VL in a background thread and exposes latest labels."""

    def __init__(self, vlm_key: str, interval: float, model_path: str | None = None):
        self.vlm_key    = vlm_key
        self.interval   = interval
        self.model_path = model_path  # overrides VLM_CONFIGS default when set
        self.model    = None
        self._lock    = threading.Lock()
        # List of {"dilated_bbox": [...], "participant": bool|None, "talking": bool|None}
        self._results: list[dict] = []
        self._pending = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._tmpdir = tempfile.mkdtemp(prefix="webcam_demo_")

    def load_model(self):
        from config import ModelConfig, GenerationConfig

        vcfg       = VLM_CONFIGS[self.vlm_key]
        model_path = self.model_path or vcfg.get("model_path", "")
        backend    = vcfg.get("backend", "qwen3vl")
        mcfg = ModelConfig(
            key=self.vlm_key,
            enabled=True,
            cls_name=vcfg["class"],
            model_path=model_path,
            dtype=vcfg["dtype"],
            generation=GenerationConfig(max_new_tokens=64, do_sample=False),
        )
        if backend == "unsloth":
            from models.gemma_unsloth import GemmaUnslothModel
            self.model = GemmaUnslothModel(mcfg)
        else:
            from models.qwen3vl import Qwen3VLModel
            self.model = Qwen3VLModel(mcfg)
        self.model.load()

    def get_results(self) -> list[dict]:
        with self._lock:
            return list(self._results)

    def submit(self, frame_bgr: np.ndarray, dilated_bboxes: list[list[float]]):
        """Submit a frame for async VLM classification (drops if already pending)."""
        if self._pending or not dilated_bboxes:
            return
        self._pending = True
        t = threading.Thread(
            target=self._classify,
            args=(frame_bgr.copy(), [list(b) for b in dilated_bboxes]),
            daemon=True,
        )
        t.start()

    def _classify(self, frame_bgr: np.ndarray, dilated_bboxes: list[list[float]]):
        try:
            H, W = frame_bgr.shape[:2]
            # Save full frame
            full_path = str(Path(self._tmpdir) / "full_frame.jpg")
            cv2.imwrite(full_path, frame_bgr)

            results = []
            for i, bbox in enumerate(dilated_bboxes):
                x1, y1, x2, y2 = [int(v) for v in bbox]
                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    results.append({"dilated_bbox": bbox, "participant": None, "talking": None})
                    continue

                crop_path = str(Path(self._tmpdir) / f"crop_{i:02d}.jpg")
                cv2.imwrite(crop_path, crop)

                r1 = self.model.run_two_image(full_path, crop_path, PROMPT_PARTICIPANT)
                participant = parse_yes_no(r1.response)

                r2 = self.model.run_two_image(full_path, crop_path, PROMPT_TALKING)
                talking = parse_yes_no(r2.response)

                results.append({
                    "dilated_bbox": bbox,
                    "participant":  participant,
                    "talking":      talking,
                })
                print(
                    f"  [VLM] face {i}: participant={'Y' if participant else 'N' if participant is False else '?'}  "
                    f"talking={'Y' if talking else 'N' if talking is False else '?'}"
                )

            with self._lock:
                self._results = results
        except Exception as e:
            print(f"[VLM] Error: {e}")
        finally:
            self._pending = False


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_face(
    frame: np.ndarray,
    dilated_bbox: list[float],
    label: str,
    color: tuple[int, int, int],
    thickness: int = 2,
):
    x1, y1, x2, y2 = [int(v) for v in dilated_bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Background pill for text
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    font_thick = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thick)
    pad = 4
    tx1, ty1 = x1, max(0, y1 - th - 2 * pad)
    tx2, ty2 = x1 + tw + 2 * pad, y1
    cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), color, -1)
    cv2.putText(
        frame, label,
        (tx1 + pad, ty2 - pad),
        font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Webcam face detection + Qwen3-VL demo")
    parser.add_argument("--camera",       type=int,   default=0,           help="Camera index")
    parser.add_argument("--conf",         type=float, default=0.25,        help="Face detection confidence threshold")
    parser.add_argument("--dilate",       type=float, default=2.0,         help="Bbox dilation factor")
    parser.add_argument("--vlm",          default="qwen3vl_4b",            choices=list(VLM_CONFIGS.keys()))
    parser.add_argument("--model-path",   default=None,
                        help="Override VLM model path/HF repo ID (e.g. 'Qwen/Qwen3-VL-4B-Instruct' or a local dir)")
    parser.add_argument("--vlm-interval", type=float, default=4.0,         help="Seconds between VLM classification runs")
    parser.add_argument("--no-vlm",       action="store_true",             help="Run face detection only (no VLM)")
    parser.add_argument("--device",       default=None,                    help="'cuda', 'cpu', or auto")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load face detector ────────────────────────────────────────────────────
    from models.yolov8_face import YOLOv8FaceModel
    detector = YOLOv8FaceModel(device=device)
    detector.load()

    # ── Load VLM (optional) ───────────────────────────────────────────────────
    worker: VLMWorker | None = None
    if not args.no_vlm:
        worker = VLMWorker(vlm_key=args.vlm, interval=args.vlm_interval, model_path=args.model_path)
        print(f"Loading VLM: {args.vlm} ...")
        worker.load_model()
        print("VLM ready.")

    # ── Open webcam ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera}")
        sys.exit(1)
    print("Webcam open. Press Q to quit.")

    last_vlm_time    = 0.0
    # vlm_cache: list of {"dilated_bbox": [...], "participant": bool|None, "talking": bool|None}
    vlm_cache: list[dict] = []
    last_dilated: list[list[float]] = []  # persisted from last YOLO pass
    _TARGET_MS = 50  # ~20 fps — leaves headroom for VLM thread
    _frame_t   = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame capture failed.")
            break

        H, W = frame.shape[:2]

        # ── Stage 1: detect faces (skip if VLM is busy to reduce GPU pressure) ─
        vlm_busy = worker is not None and worker._pending
        if not vlm_busy:
            result = detector.detect(frame, conf_threshold=args.conf)
            last_dilated = [
                dilate_bbox(det.bbox, args.dilate, W, H)
                for det in result.detections
            ]
        dilated_bboxes = last_dilated

        # ── Stage 2: submit to VLM if interval elapsed ────────────────────────
        now = time.time()
        if worker and dilated_bboxes and (now - last_vlm_time) >= args.vlm_interval:
            worker.submit(frame, dilated_bboxes)
            last_vlm_time = now

        # ── Fetch latest VLM results ──────────────────────────────────────────
        if worker:
            fresh = worker.get_results()
            if fresh:
                vlm_cache = fresh

        # ── Match current bboxes to cached VLM results ────────────────────────
        for bbox in dilated_bboxes:
            best_result = match_vlm_result(bbox, vlm_cache, W, H)
            participant = best_result["participant"] if best_result else None
            talking     = best_result["talking"]     if best_result else None
            label, color = _label_and_color(participant, talking)
            draw_face(frame, bbox, label, color)

        # ── HUD ───────────────────────────────────────────────────────────────
        hud_lines = [
            f"Faces: {len(dilated_bboxes)}",
            f"VLM: {'running' if vlm_busy else 'idle'}" if worker else "VLM: off",
            "Q: quit",
        ]
        for i, line in enumerate(hud_lines):
            cv2.putText(frame, line, (10, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        cv2.imshow("Webcam Demo — Face Detection + Qwen3-VL", frame)

        # ── FPS cap: wait remaining time to hit TARGET_MS per frame ──────────
        elapsed_ms = int((time.time() - _frame_t) * 1000)
        wait_ms    = max(1, _TARGET_MS - elapsed_ms)
        if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
            break
        _frame_t = time.time()

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
