#!/usr/bin/env python
"""
webcam_app.py — Desktop webcam demo: face detection + Qwen3-VL classification.

Usage
-----
    cd benchmark/face_detection/
    python webcam_app.py
    python webcam_app.py --model-path /mnt/shared/dils/models/Qwen3-VL-4B-Instruct
    python webcam_app.py --no-vlm
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image as PILImage, ImageTk, ImageDraw, ImageFont

import customtkinter as ctk

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
        "model_path": "Qwen/Qwen3-VL-4B-Instruct",
        "dtype":      "bfloat16",
        "backend":    "qwen3vl",
    },
    "qwen3vl_4b_int8": {
        "model_path": "/mnt/shared/dils/projects/logitech/unified_eval_framework/models/Qwen3-VL-4B-Instruct-int8",
        "dtype":      "bfloat16",
        "backend":    "qwen3vl",
    },
    # ── Gemma 4 E2B via Unsloth ───────────────────────────────────────────────
    "gemma4_e2b_4bit": {
        "model_path": "",   # pulled from gemma_unsloth.QUANT_CONFIGS
        "dtype":      "bfloat16",
        "backend":    "unsloth",
    },
    "gemma4_e2b_bf16": {
        "model_path": "",
        "dtype":      "bfloat16",
        "backend":    "unsloth",
    },
}

# Label → (hex color, display text)
LABEL_STYLES: dict[tuple, tuple[str, str]] = {
    (True,  True):  ("#22c55e", "Participant · Talking"),
    (True,  False): ("#3b82f6", "Participant · Silent"),
    (False, True):  ("#f97316", "Non-Participant · Talking"),
    (False, False): ("#ef4444", "Non-Participant · Silent"),
}
PENDING_STYLE = ("#6b7280", "Classifying…")


def label_style(participant: bool | None, talking: bool | None) -> tuple[str, str]:
    if participant is None or talking is None:
        return PENDING_STYLE
    return LABEL_STYLES.get((participant, talking), PENDING_STYLE)


def hex_to_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


# ── Geometry ──────────────────────────────────────────────────────────────────

def dilate_bbox(bbox: list[float], scale: float, W: int, H: int) -> list[float]:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw = (x2 - x1) / 2 * scale
    hh = (y2 - y1) / 2 * scale
    return [max(0, cx - hw), max(0, cy - hh), min(W, cx + hw), min(H, cy + hh)]


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
    def __init__(self, vlm_key: str, interval: float, model_path: str | None):
        self.vlm_key    = vlm_key
        self.interval   = interval
        self.model_path = model_path
        self.model      = None
        self._lock      = threading.Lock()
        self._results: list[dict] = []
        self._pending   = False
        self._tmpdir    = tempfile.mkdtemp(prefix="webcam_app_")

    def load_model(self):
        from config import ModelConfig, GenerationConfig

        vcfg    = VLM_CONFIGS[self.vlm_key]
        path    = self.model_path or vcfg.get("model_path", "")
        backend = vcfg.get("backend", "qwen3vl")
        mcfg = ModelConfig(
            key=self.vlm_key, enabled=True,
            cls_name=backend, model_path=path,
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

    @property
    def is_pending(self) -> bool:
        return self._pending

    def submit(self, frame_bgr: np.ndarray, dilated_bboxes: list[list[float]]):
        if self._pending or not dilated_bboxes:
            return
        self._pending = True
        threading.Thread(
            target=self._classify,
            args=(frame_bgr.copy(), [list(b) for b in dilated_bboxes]),
            daemon=True,
        ).start()

    def _classify(self, frame_bgr: np.ndarray, dilated_bboxes: list[list[float]]):
        try:
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
                r2 = self.model.run_two_image(full_path, crop_path, PROMPT_TALKING)
                results.append({
                    "dilated_bbox": bbox,
                    "participant":  parse_yes_no(r1.response),
                    "talking":      parse_yes_no(r2.response),
                })
            with self._lock:
                self._results = results
        except Exception as e:
            print(f"[VLM] Error: {e}")
        finally:
            self._pending = False


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_overlay(frame: np.ndarray, bboxes: list[list[float]], labels: list[dict]) -> np.ndarray:
    out = frame.copy()
    H, W = out.shape[:2]

    for i, bbox in enumerate(bboxes):
        best = match_vlm_result(bbox, labels, W, H)
        participant = best["participant"] if best else None
        talking     = best["talking"]     if best else None
        color_hex, text = label_style(participant, talking)
        color_bgr = hex_to_bgr(color_hex)

        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Rounded-rect bbox (thick)
        cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, 3)

        # Label pill background
        font       = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.55
        thickness  = 1
        (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)
        pad = 6
        pill_x1 = x1
        pill_y2 = y1
        pill_y1 = y1 - th - 2 * pad
        pill_x2 = x1 + tw + 2 * pad
        if pill_y1 < 0:
            pill_y1 = y2
            pill_y2 = y2 + th + 2 * pad

        cv2.rectangle(out, (pill_x1, pill_y1), (pill_x2, pill_y2), color_bgr, -1)
        cv2.putText(
            out, text,
            (pill_x1 + pad, pill_y2 - pad),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )

        # Face index badge (top-right of bbox)
        badge = f"#{i + 1}"
        (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x2 - bw - 8, y1), (x2, y1 + bh + 6), (30, 30, 30), -1)
        cv2.putText(out, badge, (x2 - bw - 4, y1 + bh + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    return out


# ── App ───────────────────────────────────────────────────────────────────────

class WebcamApp(ctk.CTk):
    VIDEO_W = 800
    VIDEO_H = 600

    def __init__(
        self,
        camera_idx: int,
        conf: float,
        dilate: float,
        device: str,
        worker: VLMWorker | None,
        vlm_interval: float,
    ):
        super().__init__()
        self.title("Face Detection + Qwen3-VL")
        self.resizable(False, False)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.conf         = conf
        self.dilate       = dilate
        self.device       = device
        self.worker       = worker
        self.vlm_interval = vlm_interval
        self._last_vlm_t  = 0.0
        self._vlm_cache: list[dict] = []
        self._running     = True
        self._last_dilated: list[list[float]] = []
        self._last_crops: list[PILImage.Image | None] = []

        # ── Load detector ─────────────────────────────────────────────────────
        from models.yolov8_face import YOLOv8FaceModel
        self._detector = YOLOv8FaceModel(device=device)
        self._detector.load()

        # ── Camera (use AVFoundation on macOS for proper permission handling) ──
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(camera_idx, backend)
        if not self._cap.isOpened():
            print(f"[ERROR] Cannot open camera {camera_idx}")
            print("  → System Settings → Privacy & Security → Camera → enable your terminal")
            sys.exit(1)
        # Let the driver pick native resolution; we'll resize in software
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize latency

        self._build_ui()
        self._update()

    _UPDATE_MS = 50  # ~20 fps; leaves headroom for VLM thread on MPS/CPU

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        SIDEBAR_W = 280

        # Root grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0, minsize=SIDEBAR_W)
        self.grid_rowconfigure(0, weight=1)

        # ── Video panel ───────────────────────────────────────────────────────
        video_frame = ctk.CTkFrame(self, corner_radius=12, fg_color="#111827")
        video_frame.grid(row=0, column=0, padx=(12, 6), pady=12, sticky="nsew")
        video_frame.grid_rowconfigure(0, weight=0)
        video_frame.grid_rowconfigure(1, weight=1)
        video_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            video_frame, text="Live Feed",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#9ca3af",
        ).grid(row=0, column=0, pady=(10, 4))

        self._video_label = ctk.CTkLabel(video_frame, text="", corner_radius=8)
        self._video_label.grid(row=1, column=0, padx=10, pady=(0, 10))

        # ── Sidebar ───────────────────────────────────────────────────────────
        sidebar = ctk.CTkFrame(self, corner_radius=12, fg_color="#111827", width=SIDEBAR_W)
        sidebar.grid(row=0, column=1, padx=(6, 12), pady=12, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.grid_columnconfigure(0, weight=1)

        row = 0

        # Title
        ctk.CTkLabel(
            sidebar, text="Detection Status",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=row, column=0, pady=(14, 8), padx=16, sticky="w")
        row += 1

        # Stats row
        stats_frame = ctk.CTkFrame(sidebar, fg_color="#1f2937", corner_radius=8)
        stats_frame.grid(row=row, column=0, padx=12, pady=(0, 10), sticky="ew")
        stats_frame.grid_columnconfigure((0, 1), weight=1)
        row += 1

        ctk.CTkLabel(stats_frame, text="Faces", text_color="#9ca3af",
                     font=ctk.CTkFont(size=11)).grid(row=0, column=0, padx=12, pady=(8, 0))
        ctk.CTkLabel(stats_frame, text="VLM", text_color="#9ca3af",
                     font=ctk.CTkFont(size=11)).grid(row=0, column=1, padx=12, pady=(8, 0))

        self._face_count_var = ctk.StringVar(value="0")
        self._vlm_status_var = ctk.StringVar(value="idle" if self.worker else "off")

        ctk.CTkLabel(stats_frame, textvariable=self._face_count_var,
                     font=ctk.CTkFont(size=28, weight="bold")).grid(row=1, column=0, padx=12, pady=(0, 8))
        self._vlm_badge = ctk.CTkLabel(
            stats_frame, textvariable=self._vlm_status_var,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#22c55e",
        )
        self._vlm_badge.grid(row=1, column=1, padx=12, pady=(0, 8))

        # Separator
        ctk.CTkFrame(sidebar, height=1, fg_color="#374151").grid(
            row=row, column=0, sticky="ew", padx=12, pady=4)
        row += 1

        ctk.CTkLabel(
            sidebar, text="Detected Faces",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#9ca3af",
        ).grid(row=row, column=0, padx=16, pady=(6, 4), sticky="w")
        row += 1

        # Face cards scroll area (using a plain CTkScrollableFrame)
        self._cards_frame = ctk.CTkScrollableFrame(
            sidebar, fg_color="transparent", corner_radius=0,
        )
        self._cards_frame.grid(row=row, column=0, padx=8, pady=(0, 8), sticky="nsew")
        sidebar.grid_rowconfigure(row, weight=1)
        row += 1

        # Separator
        ctk.CTkFrame(sidebar, height=1, fg_color="#374151").grid(
            row=row, column=0, sticky="ew", padx=12, pady=4)
        row += 1

        # Legend
        legend_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        legend_frame.grid(row=row, column=0, padx=12, pady=(4, 12), sticky="ew")
        legend_frame.grid_columnconfigure(1, weight=1)
        row += 1

        for i, (key, (color, text)) in enumerate(LABEL_STYLES.items()):
            dot = ctk.CTkFrame(legend_frame, width=10, height=10,
                               corner_radius=5, fg_color=color)
            dot.grid(row=i, column=0, padx=(0, 6), pady=2)
            ctk.CTkLabel(legend_frame, text=text, anchor="w",
                         font=ctk.CTkFont(size=11), text_color="#d1d5db"
                         ).grid(row=i, column=1, sticky="w")

        self._card_widgets: list[ctk.CTkFrame] = []

    # ── Frame update loop ─────────────────────────────────────────────────────

    def _update(self):
        if not self._running:
            return
        try:
            self._update_inner()
        except Exception as exc:
            print(f"[_update] {type(exc).__name__}: {exc}")
        self.after(self._UPDATE_MS, self._update)  # always reschedule regardless of errors

    def _update_inner(self):
        ret, frame = self._cap.read()
        if not ret:
            print("[camera] frame read failed — check camera permissions")
            return

        H, W = frame.shape[:2]

        # Skip YOLO while VLM is running to reduce GPU/MPS contention.
        # Reuse the last detections so bboxes keep moving smoothly.
        vlm_busy = self.worker is not None and self.worker.is_pending
        if not vlm_busy:
            result  = self._detector.detect(frame, conf_threshold=self.conf)
            dilated = [dilate_bbox(d.bbox, self.dilate, W, H) for d in result.detections]

            # Extract crops from raw frame (before overlay drawing)
            crops: list[PILImage.Image | None] = []
            for bbox in dilated:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                crop_bgr = frame[y1:y2, x1:x2]
                if crop_bgr.size > 0:
                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    crops.append(PILImage.fromarray(crop_rgb))
                else:
                    crops.append(None)

            self._last_dilated = dilated
            self._last_crops   = crops

        dilated = self._last_dilated
        crops   = self._last_crops
        n_faces = len(dilated)

        # Submit to VLM if interval elapsed
        now = time.time()
        if self.worker and dilated and (now - self._last_vlm_t) >= self.vlm_interval:
            self.worker.submit(frame, dilated)
            self._last_vlm_t = now

        # Fetch latest VLM results
        if self.worker:
            fresh = self.worker.get_results()
            if fresh:
                self._vlm_cache = fresh

        # Draw overlay on wide view
        annotated = draw_overlay(frame, dilated, self._vlm_cache)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        pil = PILImage.fromarray(rgb)
        pil = pil.resize((self.VIDEO_W, self.VIDEO_H), PILImage.LANCZOS)
        ctk_img = ctk.CTkImage(light_image=pil, dark_image=pil,
                               size=(self.VIDEO_W, self.VIDEO_H))
        self._video_label.configure(image=ctk_img)
        self._video_label.image = ctk_img  # prevent GC

        # Sidebar stats
        self._face_count_var.set(str(n_faces))
        if self.worker:
            pending = self.worker.is_pending
            self._vlm_status_var.set("running" if pending else "idle")
            self._vlm_badge.configure(text_color="#f59e0b" if pending else "#22c55e")

        self._refresh_cards(dilated, crops, self._vlm_cache)

    # Card thumbnail dimensions (fit inside sidebar width)
    _THUMB_W = 230
    _THUMB_H = 160

    def _refresh_cards(
        self,
        bboxes: list[list[float]],
        crops: list[PILImage.Image | None],
        cached: list[dict],
    ):
        n = len(bboxes)

        # Recreate widget pool only when face count changes
        if len(self._card_widgets) != n:
            for w in self._card_widgets:
                w["frame"].destroy()
            self._card_widgets.clear()

            for i in range(n):
                card_frame = ctk.CTkFrame(
                    self._cards_frame, corner_radius=10, fg_color="#1f2937"
                )
                card_frame.pack(fill="x", padx=6, pady=5)
                card_frame.grid_columnconfigure(0, weight=1)

                img_lbl = ctk.CTkLabel(card_frame, text="", corner_radius=6)
                img_lbl.grid(row=0, column=0, padx=8, pady=(8, 4), sticky="ew")

                title_lbl = ctk.CTkLabel(
                    card_frame, text=f"Face #{i + 1}",
                    font=ctk.CTkFont(size=12, weight="bold"), anchor="w",
                )
                title_lbl.grid(row=1, column=0, padx=12, sticky="w")

                status_lbl = ctk.CTkLabel(
                    card_frame, text="Classifying…",
                    font=ctk.CTkFont(size=11), anchor="w", text_color="#6b7280",
                )
                status_lbl.grid(row=2, column=0, padx=12, pady=(0, 8), sticky="w")

                self._card_widgets.append({
                    "frame":  card_frame,
                    "image":  img_lbl,
                    "title":  title_lbl,
                    "status": status_lbl,
                })

        # Update content of each card
        # Need frame dims for centroid fallback; use the VLM canvas size as proxy
        fw, fh = self.VIDEO_W, self.VIDEO_H
        for i, (bbox, crop) in enumerate(zip(bboxes, crops)):
            widgets = self._card_widgets[i]

            # VLM label
            best        = match_vlm_result(bbox, cached, fw, fh)
            participant = best["participant"] if best else None
            talking     = best["talking"]     if best else None
            color_hex, label_text = label_style(participant, talking)

            widgets["status"].configure(text=label_text, text_color=color_hex)
            widgets["frame"].configure(border_width=2, border_color=color_hex)

            # Crop thumbnail
            if crop is not None:
                cw, ch = crop.size
                scale  = min(self._THUMB_W / cw, self._THUMB_H / ch)
                tw, th = int(cw * scale), int(ch * scale)
                thumb  = crop.resize((tw, th), PILImage.LANCZOS)
                ctk_thumb = ctk.CTkImage(
                    light_image=thumb, dark_image=thumb, size=(tw, th)
                )
                widgets["image"].configure(image=ctk_thumb)
                widgets["image"].image = ctk_thumb

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def on_close(self):
        self._running = False
        self._cap.release()
        self.destroy()


# ── Camera enumeration ────────────────────────────────────────────────────────

def list_cameras() -> list[tuple[int, str]]:
    """Return [(index, label), ...] for all cameras that open successfully."""
    found = []
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    for idx in range(8):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                # Try to get a human-readable name on macOS via system_profiler
                found.append((idx, f"Camera {idx}"))
            cap.release()
    return found


def pick_camera(cameras: list[tuple[int, str]]) -> int:
    """Show a tiny picker window and return chosen index."""
    choice = {"idx": cameras[0][0]}

    picker = ctk.CTk()
    picker.title("Select Camera")
    picker.resizable(False, False)
    ctk.set_appearance_mode("dark")

    ctk.CTkLabel(picker, text="Multiple cameras detected.\nPick one to use:",
                 font=ctk.CTkFont(size=13)).pack(padx=24, pady=(18, 10))

    var = ctk.StringVar(value=f"{cameras[0][0]}: {cameras[0][1]}")
    options = [f"{idx}: {name}" for idx, name in cameras]
    ctk.CTkOptionMenu(picker, values=options, variable=var,
                      width=220).pack(padx=24, pady=(0, 10))

    def confirm():
        choice["idx"] = int(var.get().split(":")[0])
        picker.destroy()

    ctk.CTkButton(picker, text="Open", command=confirm).pack(pady=(0, 18))
    picker.mainloop()
    return choice["idx"]


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Webcam face detection + Qwen3-VL desktop app")
    parser.add_argument("--camera",       type=int,   default=None,
                        help="Camera index (omit to auto-detect / show picker)")
    parser.add_argument("--list-cameras", action="store_true",
                        help="Print available cameras and exit")
    parser.add_argument("--conf",         type=float, default=0.25)
    parser.add_argument("--dilate",       type=float, default=2.0)
    parser.add_argument("--vlm",          default="qwen3vl_4b", choices=list(VLM_CONFIGS.keys()),
                        help=f"VLM to use. Options: {', '.join(VLM_CONFIGS.keys())}")
    parser.add_argument("--model-path",   default=None,
                        help="Override VLM model path or HF repo ID")
    parser.add_argument("--vlm-interval", type=float, default=4.0,
                        help="Seconds between VLM classification runs")
    parser.add_argument("--no-vlm",       action="store_true")
    parser.add_argument("--device",       default=None)
    args = parser.parse_args()

    # ── Camera selection ──────────────────────────────────────────────────────
    cameras = list_cameras()
    if not cameras:
        print("[ERROR] No cameras found. Check permissions and connections.")
        sys.exit(1)

    if args.list_cameras:
        print("Available cameras:")
        for idx, name in cameras:
            print(f"  {idx}: {name}")
        sys.exit(0)

    if args.camera is not None:
        camera_idx = args.camera
    elif len(cameras) == 1:
        camera_idx = cameras[0][0]
        print(f"Using camera {camera_idx}: {cameras[0][1]}")
    else:
        print(f"Found {len(cameras)} cameras: {cameras}")
        ctk.set_appearance_mode("dark")
        camera_idx = pick_camera(cameras)
        print(f"Selected camera {camera_idx}")

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    worker = None
    if not args.no_vlm:
        worker = VLMWorker(
            vlm_key=args.vlm,
            interval=args.vlm_interval,
            model_path=args.model_path,
        )
        print("Loading Qwen3-VL…")
        worker.load_model()
        print("VLM ready.")

    app = WebcamApp(
        camera_idx=camera_idx,
        conf=args.conf,
        dilate=args.dilate,
        device=device,
        worker=worker,
        vlm_interval=args.vlm_interval,
    )
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    main()
