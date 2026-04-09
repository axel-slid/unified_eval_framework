#!/usr/bin/env python
"""
image_demo.py — Upload an image, ask a question, swap VLMs live.

Supported models (selectable in the UI):
  • Gemma 4 E2B — 4-bit / 8-bit / bf16  (via Unsloth FastVisionModel)
  • Gemma 3 4B  — 4-bit / bf16          (via Unsloth FastVisionModel)
  • Qwen3-VL 4B — bfloat16              (via HuggingFace Transformers)

Speed panel shows:
  • Live tokens/sec counter while streaming
  • Latency (ms) and total token count per run
  • Bar chart of last 8 runs (model · tokens/sec)

Usage
-----
    cd benchmark/face_detection/
    python image_demo.py
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk
from PIL import Image as PILImage

HERE       = Path(__file__).parent
BENCH_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BENCH_ROOT))


# ── Model registry ────────────────────────────────────────────────────────────
#   Each entry holds enough info to construct + load the model lazily.

MODEL_REGISTRY: dict[str, dict] = {
    # ── Gemma 4 E2B ──────────────────────────────────────────────────────────
    "gemma4_e2b_4bit": {
        "label":   "Gemma 4 E2B · 4-bit  (~4 GB)",
        "backend": "unsloth",
        "key":     "gemma4_e2b_4bit",
    },
    "gemma4_e2b_8bit": {
        "label":   "Gemma 4 E2B · 8-bit  (~6 GB)",
        "backend": "unsloth",
        "key":     "gemma4_e2b_8bit",
    },
    "gemma4_e2b_bf16": {
        "label":   "Gemma 4 E2B · bf16   (~10 GB)",
        "backend": "unsloth",
        "key":     "gemma4_e2b_bf16",
    },
    # ── Gemma 3 4B ───────────────────────────────────────────────────────────
    "gemma3_4b_4bit": {
        "label":   "Gemma 3 4B · 4-bit   (~4 GB)",
        "backend": "unsloth",
        "key":     "gemma3_4b_4bit",
    },
    "gemma3_4b_bf16": {
        "label":   "Gemma 3 4B · bf16    (~8 GB)",
        "backend": "unsloth",
        "key":     "gemma3_4b_bf16",
    },
    # ── Qwen3-VL (comparison baseline) ───────────────────────────────────────
    "qwen3vl_4b": {
        "label":   "Qwen3-VL 4B · bf16",
        "backend": "qwen3vl",
        "key":     "qwen3vl_4b",
        "model_path": "Qwen/Qwen3-VL-4B-Instruct",
    },
}

MODEL_LABELS = [v["label"] for v in MODEL_REGISTRY.values()]
MODEL_KEYS   = list(MODEL_REGISTRY.keys())

DEFAULT_PROMPT = "Describe what you see in this image in detail."

# ── Colours ───────────────────────────────────────────────────────────────────

C_BG       = "#0f172a"   # outer background
C_CARD     = "#1e293b"   # card background
C_BORDER   = "#334155"   # subtle border
C_TEXT     = "#f1f5f9"
C_MUTED    = "#94a3b8"
C_ACCENT   = "#6366f1"   # indigo — "Run" button
C_SUCCESS  = "#22c55e"
C_WARN     = "#f59e0b"
C_DANGER   = "#ef4444"
C_BAR      = "#6366f1"   # bar chart fill


# ── Speed bar chart (Canvas) ──────────────────────────────────────────────────

class SpeedChart(ctk.CTkCanvas):
    """Simple horizontal bar chart showing tokens/sec for last N runs."""

    MAX_RUNS   = 8
    BAR_H      = 18
    BAR_PAD    = 6
    LABEL_W    = 130
    BAR_MAX_W  = 200
    RIGHT_PAD  = 8

    def __init__(self, parent, **kwargs):
        h = (self.BAR_H + self.BAR_PAD) * self.MAX_RUNS + self.BAR_PAD
        super().__init__(
            parent,
            height=h,
            bg="#0f172a",
            highlightthickness=0,
            **kwargs,
        )
        self._runs: list[dict] = []   # {"label": str, "tps": float, "active": bool}

    def push_run(self, label: str, tps: float, active: bool = False) -> None:
        # If active (streaming in progress), update the last entry in place
        if active and self._runs and self._runs[-1].get("active"):
            self._runs[-1] = {"label": label, "tps": tps, "active": True}
        else:
            # Remove previous active marker
            for r in self._runs:
                r["active"] = False
            self._runs.append({"label": label, "tps": tps, "active": active})
            if len(self._runs) > self.MAX_RUNS:
                self._runs.pop(0)
        self._draw()

    def finish_last(self) -> None:
        if self._runs:
            self._runs[-1]["active"] = False
        self._draw()

    def _draw(self) -> None:
        self.delete("all")
        if not self._runs:
            return

        max_tps = max(r["tps"] for r in self._runs) or 1.0
        W = int(self.winfo_width()) or (self.LABEL_W + self.BAR_MAX_W + self.RIGHT_PAD + 20)

        for i, run in enumerate(self._runs):
            y = self.BAR_PAD + i * (self.BAR_H + self.BAR_PAD)
            bar_area_w = W - self.LABEL_W - self.RIGHT_PAD
            bar_w = int(bar_area_w * min(run["tps"] / max_tps, 1.0))

            # Label
            self.create_text(
                self.LABEL_W - 4, y + self.BAR_H // 2,
                text=run["label"],
                anchor="e",
                fill="#94a3b8" if not run["active"] else "#f1f5f9",
                font=("Helvetica", 9),
            )

            # Bar background
            self.create_rectangle(
                self.LABEL_W, y,
                self.LABEL_W + bar_area_w, y + self.BAR_H,
                fill="#1e293b", outline="",
            )

            # Bar fill
            color = C_WARN if run["active"] else C_BAR
            if bar_w > 0:
                self.create_rectangle(
                    self.LABEL_W, y,
                    self.LABEL_W + bar_w, y + self.BAR_H,
                    fill=color, outline="",
                )

            # tok/s label
            tps_str = f"{run['tps']:.1f} t/s"
            self.create_text(
                self.LABEL_W + bar_w + 4, y + self.BAR_H // 2,
                text=tps_str,
                anchor="w",
                fill=C_MUTED,
                font=("Helvetica", 9),
            )


# ── Model loader (background thread) ─────────────────────────────────────────

class ModelLoader:
    """Manages one loaded model at a time. Thread-safe swap logic."""

    def __init__(self):
        self._model      = None
        self._loaded_key = None
        self._lock       = threading.Lock()

    @property
    def loaded_key(self) -> str | None:
        return self._loaded_key

    @property
    def model(self):
        return self._model

    def load(
        self,
        model_key: str,
        on_progress: callable,
        on_done: callable,
        on_error: callable,
    ) -> None:
        """Start async load in background thread."""
        threading.Thread(
            target=self._load_worker,
            args=(model_key, on_progress, on_done, on_error),
            daemon=True,
        ).start()

    def _load_worker(
        self,
        model_key: str,
        on_progress: callable,
        on_done: callable,
        on_error: callable,
    ) -> None:
        try:
            on_progress("Unloading previous model…")
            with self._lock:
                if self._model is not None:
                    try:
                        self._model.unload()
                    except Exception:
                        pass
                    self._model     = None
                    self._loaded_key = None

            on_progress(f"Loading {MODEL_REGISTRY[model_key]['label']}…")
            model = _build_model(model_key)
            model.load()

            with self._lock:
                self._model      = model
                self._loaded_key = model_key

            on_done(model_key)
        except Exception as e:
            on_error(str(e))

    def unload_all(self) -> None:
        with self._lock:
            if self._model is not None:
                try:
                    self._model.unload()
                except Exception:
                    pass
                self._model     = None
                self._loaded_key = None


def _build_model(key: str):
    """Construct a model object from the registry key."""
    from config import ModelConfig, GenerationConfig

    entry   = MODEL_REGISTRY[key]
    backend = entry["backend"]

    gen = GenerationConfig(max_new_tokens=256, do_sample=False)
    cfg = ModelConfig(
        key=key,
        enabled=True,
        cls_name=backend,
        model_path=entry.get("model_path", ""),
        dtype="bfloat16",
        generation=gen,
    )

    if backend == "unsloth":
        from models.gemma_unsloth import GemmaUnslothModel
        return GemmaUnslothModel(cfg)
    elif backend == "qwen3vl":
        from models.qwen3vl import Qwen3VLModel
        return Qwen3VLModel(cfg)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ── Main app ──────────────────────────────────────────────────────────────────

class ImageDemoApp(ctk.CTk):
    IMG_PANEL_W = 480
    IMG_PANEL_H = 400
    THUMB_MAX   = (460, 380)

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.title("VLM Image Demo — model swapper + speed viz")
        self.resizable(True, True)
        self.configure(fg_color=C_BG)

        self._image_path: str | None = None
        self._loader = ModelLoader()

        # Speed tracking
        self._current_tps   = 0.0
        self._current_tokens = 0
        self._run_history: list[dict] = []

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=0, minsize=self.IMG_PANEL_W)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_image_panel()
        self._build_control_panel()

    def _build_image_panel(self) -> None:
        panel = ctk.CTkFrame(self, fg_color=C_CARD, corner_radius=12)
        panel.grid(row=0, column=0, padx=(12, 6), pady=12, sticky="nsew")
        panel.grid_rowconfigure(1, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            panel, text="Image",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=C_TEXT,
        ).grid(row=0, column=0, pady=(12, 4), padx=16, sticky="w")

        # Drop zone / preview
        self._img_label = ctk.CTkLabel(
            panel,
            text="Click 'Upload Image' or drag & drop",
            text_color=C_MUTED,
            fg_color="#0f172a",
            corner_radius=8,
            width=self.IMG_PANEL_W - 24,
            height=self.IMG_PANEL_H,
        )
        self._img_label.grid(row=1, column=0, padx=12, pady=(0, 8), sticky="nsew")

        ctk.CTkButton(
            panel,
            text="Upload Image",
            fg_color=C_ACCENT,
            hover_color="#4f46e5",
            command=self._pick_image,
            height=36,
        ).grid(row=2, column=0, padx=12, pady=(0, 12), sticky="ew")

    def _build_control_panel(self) -> None:
        panel = ctk.CTkFrame(self, fg_color=C_CARD, corner_radius=12)
        panel.grid(row=0, column=1, padx=(6, 12), pady=12, sticky="nsew")
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(4, weight=1)  # response area expands

        row = 0

        # ── Model selector ────────────────────────────────────────────────────
        ctk.CTkLabel(
            panel, text="Model",
            font=ctk.CTkFont(size=14, weight="bold"), text_color=C_TEXT,
        ).grid(row=row, column=0, padx=16, pady=(14, 4), sticky="w")
        row += 1

        model_row = ctk.CTkFrame(panel, fg_color="transparent")
        model_row.grid(row=row, column=0, padx=12, pady=(0, 8), sticky="ew")
        model_row.grid_columnconfigure(0, weight=1)
        row += 1

        self._model_var = ctk.StringVar(value=MODEL_LABELS[0])
        self._model_menu = ctk.CTkOptionMenu(
            model_row,
            values=MODEL_LABELS,
            variable=self._model_var,
            fg_color="#1e293b",
            button_color="#334155",
            button_hover_color="#475569",
            text_color=C_TEXT,
            height=34,
            dynamic_resizing=False,
        )
        self._model_menu.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        self._load_btn = ctk.CTkButton(
            model_row,
            text="Load",
            width=70,
            fg_color="#334155",
            hover_color="#475569",
            command=self._load_model,
            height=34,
        )
        self._load_btn.grid(row=0, column=1)

        # Model status badge
        self._model_status_var = ctk.StringVar(value="No model loaded")
        self._model_status_lbl = ctk.CTkLabel(
            panel,
            textvariable=self._model_status_var,
            font=ctk.CTkFont(size=11),
            text_color=C_MUTED,
        )
        self._model_status_lbl.grid(row=row, column=0, padx=16, pady=(0, 6), sticky="w")
        row += 1

        _sep(panel, row); row += 1

        # ── Prompt ────────────────────────────────────────────────────────────
        ctk.CTkLabel(
            panel, text="Prompt",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=C_TEXT,
        ).grid(row=row, column=0, padx=16, pady=(8, 4), sticky="w")
        row += 1

        self._prompt_box = ctk.CTkTextbox(
            panel, height=60, fg_color="#0f172a", text_color=C_TEXT,
            border_color=C_BORDER, border_width=1, corner_radius=8,
        )
        self._prompt_box.grid(row=row, column=0, padx=12, pady=(0, 8), sticky="ew")
        self._prompt_box.insert("1.0", DEFAULT_PROMPT)
        row += 1

        self._run_btn = ctk.CTkButton(
            panel,
            text="Run Inference",
            fg_color=C_ACCENT,
            hover_color="#4f46e5",
            command=self._run_inference,
            height=38,
            state="disabled",
        )
        self._run_btn.grid(row=row, column=0, padx=12, pady=(0, 8), sticky="ew")
        row += 1

        _sep(panel, row); row += 1

        # ── Response area ─────────────────────────────────────────────────────
        resp_header = ctk.CTkFrame(panel, fg_color="transparent")
        resp_header.grid(row=row, column=0, padx=12, pady=(8, 2), sticky="ew")
        resp_header.grid_columnconfigure(0, weight=1)
        row += 1

        ctk.CTkLabel(
            resp_header, text="Response",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=C_TEXT,
        ).grid(row=0, column=0, sticky="w")

        # Live speed badge (shown during streaming)
        self._tps_var = ctk.StringVar(value="")
        ctk.CTkLabel(
            resp_header, textvariable=self._tps_var,
            font=ctk.CTkFont(size=12, weight="bold"), text_color=C_WARN,
        ).grid(row=0, column=1, sticky="e")

        self._response_box = ctk.CTkTextbox(
            panel, fg_color="#0f172a", text_color=C_TEXT,
            border_color=C_BORDER, border_width=1, corner_radius=8,
        )
        self._response_box.grid(row=row, column=0, padx=12, pady=(0, 8), sticky="nsew")
        panel.grid_rowconfigure(row, weight=1)
        row += 1

        # Latency / token count row
        metrics_row = ctk.CTkFrame(panel, fg_color="#0f172a", corner_radius=8)
        metrics_row.grid(row=row, column=0, padx=12, pady=(0, 8), sticky="ew")
        metrics_row.grid_columnconfigure((0, 1, 2), weight=1)
        row += 1

        self._latency_var = ctk.StringVar(value="— ms")
        self._tokens_var  = ctk.StringVar(value="— tok")
        self._tps_fin_var = ctk.StringVar(value="— t/s")

        for col, (label, var) in enumerate([
            ("Latency",   self._latency_var),
            ("Tokens",    self._tokens_var),
            ("Avg tok/s", self._tps_fin_var),
        ]):
            ctk.CTkLabel(
                metrics_row, text=label, text_color=C_MUTED,
                font=ctk.CTkFont(size=10),
            ).grid(row=0, column=col, pady=(6, 0), padx=8)
            ctk.CTkLabel(
                metrics_row, textvariable=var,
                font=ctk.CTkFont(size=14, weight="bold"), text_color=C_TEXT,
            ).grid(row=1, column=col, pady=(0, 6), padx=8)

        _sep(panel, row); row += 1

        # ── Speed history chart ───────────────────────────────────────────────
        ctk.CTkLabel(
            panel, text="Speed history  (last 8 runs)",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=C_MUTED,
        ).grid(row=row, column=0, padx=16, pady=(8, 2), sticky="w")
        row += 1

        self._chart = SpeedChart(panel)
        self._chart.grid(row=row, column=0, padx=12, pady=(0, 12), sticky="ew")
        row += 1

    # ── Image upload ──────────────────────────────────────────────────────────

    def _pick_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._set_image(path)

    def _set_image(self, path: str) -> None:
        self._image_path = path
        try:
            img = PILImage.open(path).convert("RGB")
            img.thumbnail(self.THUMB_MAX, PILImage.LANCZOS)
            ctk_img = ctk.CTkImage(
                light_image=img, dark_image=img,
                size=(img.width, img.height),
            )
            self._img_label.configure(image=ctk_img, text="")
            self._img_label.image = ctk_img
        except Exception as e:
            self._img_label.configure(text=f"Error: {e}", image=None)
        self._refresh_run_btn()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        label     = self._model_var.get()
        model_key = MODEL_KEYS[MODEL_LABELS.index(label)]
        if model_key == self._loader.loaded_key:
            self._set_model_status(f"Already loaded: {label}", C_SUCCESS)
            return

        self._load_btn.configure(state="disabled")
        self._run_btn.configure(state="disabled")
        self._set_model_status("Loading…", C_WARN)

        self._loader.load(
            model_key,
            on_progress=lambda msg: self.after(0, self._set_model_status, msg, C_WARN),
            on_done=lambda key: self.after(0, self._on_model_loaded, key),
            on_error=lambda err: self.after(0, self._on_model_error, err),
        )

    def _on_model_loaded(self, key: str) -> None:
        label = MODEL_REGISTRY[key]["label"]
        self._set_model_status(f"Loaded: {label}", C_SUCCESS)
        self._load_btn.configure(state="normal")
        self._refresh_run_btn()

    def _on_model_error(self, err: str) -> None:
        self._set_model_status(f"Error: {err}", C_DANGER)
        self._load_btn.configure(state="normal")

    def _set_model_status(self, msg: str, color: str = C_MUTED) -> None:
        self._model_status_var.set(msg)
        self._model_status_lbl.configure(text_color=color)

    def _refresh_run_btn(self) -> None:
        ready = self._image_path is not None and self._loader.model is not None
        self._run_btn.configure(state="normal" if ready else "disabled")

    # ── Inference ─────────────────────────────────────────────────────────────

    def _run_inference(self) -> None:
        if not self._image_path or not self._loader.model:
            return

        prompt = self._prompt_box.get("1.0", "end").strip() or DEFAULT_PROMPT

        # Clear response area
        self._response_box.configure(state="normal")
        self._response_box.delete("1.0", "end")
        self._tps_var.set("")
        self._latency_var.set("…")
        self._tokens_var.set("…")
        self._tps_fin_var.set("…")

        self._run_btn.configure(state="disabled")
        self._load_btn.configure(state="disabled")

        # Push a live entry into the chart
        loaded_key = self._loader.loaded_key
        short_label = _short_label(loaded_key)
        self._chart.push_run(short_label, 0.0, active=True)
        self._current_tokens = 0

        model = self._loader.model

        # Unsloth models support streaming; Qwen3VL falls back to sync
        if hasattr(model, "run_streaming"):
            model.run_streaming(
                self._image_path,
                prompt,
                token_callback=lambda chunk, tps, n: self.after(
                    0, self._on_token, chunk, tps, n, short_label
                ),
                done_callback=lambda result: self.after(
                    0, self._on_done, result, short_label
                ),
            )
        else:
            # Non-streaming fallback (Qwen3VL etc.) — run in thread
            threading.Thread(
                target=self._sync_infer,
                args=(model, self._image_path, prompt, short_label),
                daemon=True,
            ).start()

    def _sync_infer(self, model, image_path: str, prompt: str, short_label: str) -> None:
        t0 = time.perf_counter()
        result = model.run(image_path, prompt)
        elapsed = time.perf_counter() - t0
        # Approximate token count from decoder
        n_tok = len(result.response.split())
        tps   = n_tok / elapsed if elapsed > 0 else 0.0
        # Push all text at once and call done
        self.after(0, self._response_box.insert, "end", result.response)
        self.after(0, self._on_done, result, short_label, n_tok, tps)

    def _on_token(self, chunk: str, tps: float, n_tokens: int, short_label: str) -> None:
        self._response_box.insert("end", chunk)
        self._response_box.see("end")
        self._current_tps    = tps
        self._current_tokens = n_tokens
        self._tps_var.set(f"{tps:.1f} tok/s")
        # Update live bar
        self._chart.push_run(short_label, tps, active=True)

    def _on_done(
        self,
        result,
        short_label: str,
        n_tokens: int | None = None,
        final_tps: float | None = None,
    ) -> None:
        if n_tokens is None:
            n_tokens = self._current_tokens
        if final_tps is None:
            final_tps = self._current_tps

        self._tps_var.set("")
        self._latency_var.set(f"{result.latency_ms:.0f} ms")
        self._tokens_var.set(f"{n_tokens} tok")
        self._tps_fin_var.set(f"{final_tps:.1f} t/s")

        # Finalise chart bar
        self._chart.push_run(short_label, final_tps, active=False)
        self._chart.finish_last()

        if result.error:
            self._response_box.insert("end", f"\n\n[Error] {result.error}")
            self._response_box.configure(text_color=C_DANGER)
        else:
            self._response_box.configure(text_color=C_TEXT)

        self._run_btn.configure(state="normal")
        self._load_btn.configure(state="normal")

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        self._loader.unload_all()
        self.destroy()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sep(parent, row: int) -> None:
    ctk.CTkFrame(parent, height=1, fg_color=C_BORDER).grid(
        row=row, column=0, sticky="ew", padx=12, pady=2
    )


def _short_label(key: str | None) -> str:
    if key is None:
        return "?"
    mapping = {
        "gemma4_e2b_4bit": "G4-E2B 4b",
        "gemma4_e2b_8bit": "G4-E2B 8b",
        "gemma4_e2b_bf16": "G4-E2B f16",
        "gemma3_4b_4bit":  "G3-4B 4b",
        "gemma3_4b_bf16":  "G3-4B f16",
        "qwen3vl_4b":      "Qwen3VL",
    }
    return mapping.get(key, key[:10])


# ── Entry ─────────────────────────────────────────────────────────────────────

def main() -> None:
    app = ImageDemoApp()
    app.mainloop()


if __name__ == "__main__":
    main()
