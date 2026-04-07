#!/usr/bin/env python
"""
generate_pipeline_report.py — HTML report for the people-analysis pipeline.

Merges per-VLM result JSONs, then shows every room image with:
  - Full room photo
  - CV detection counts (yolo11n / yolo11s / mobilenet_ssd)
  - Per-person cards: crop image + participant/talking verdict from every VLM

Usage:
    cd benchmark
    python generate_pipeline_report.py
    python generate_pipeline_report.py --date 20260406   # filter by date prefix
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

MODEL_ACCENTS = [
    "#58a6ff", "#f78166", "#7ee787", "#d2a8ff",
    "#ffa657", "#79c0ff", "#ff7b72", "#56d364",
]
MODEL_SHORT = {
    "smolvlm":        "SmolVLM2",
    "internvl":       "InternVL3-4B",
    "internvl_int8":  "InternVL3-4B-int8",
    "qwen3vl_4b":     "Qwen3-VL-4B",
    "qwen3vl_4b_int8":"Qwen3-VL-4B-int8",
    "qwen3vl_8b":     "Qwen3-VL-8B",
    "qwen3vl_8b_int8":"Qwen3-VL-8B-int8",
}
VLM_ORDER = list(MODEL_SHORT.keys())


def img_b64(path: str) -> str | None:
    try:
        p = Path(path)
        if not p.exists():
            return None
        ext = p.suffix.lower().strip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")
        return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"
    except Exception:
        return None


def merge_results(date_prefix: str | None) -> dict:
    """
    Load all pipeline_people_*.json files, pick the latest result per VLM key,
    and merge into one unified structure.
    """
    pattern = str(RESULTS_DIR / "pipeline_people_*.json")
    files = sorted(glob.glob(pattern))

    # latest file per vlm_key
    latest: dict[str, dict] = {}
    base_data: dict | None = None  # for cv_results + crop_manifest + images

    for f in files:
        if date_prefix and date_prefix not in Path(f).name:
            continue
        d = json.load(open(f))
        for vlm_key, results in d.get("vlm_results", {}).items():
            latest[vlm_key] = {"results": results, "ts": d["timestamp"]}
        # Use most complete cv_results (one with all 3 detectors)
        if base_data is None or len(d.get("cv_results", {})) > len((base_data or {}).get("cv_results", {})):
            base_data = d

    if base_data is None:
        raise FileNotFoundError("No pipeline_people_*.json files found")

    return {
        "cv_results":    base_data["cv_results"],
        "crop_manifest": base_data["crop_manifest"],
        "images":        base_data["images"],
        "vlm_results":   {k: latest[k]["results"] for k in VLM_ORDER if k in latest},
        "vlm_ts":        {k: latest[k]["ts"] for k in VLM_ORDER if k in latest},
    }


def build_html(data: dict) -> str:
    vlm_keys    = list(data["vlm_results"].keys())
    accents     = {k: MODEL_ACCENTS[i % len(MODEL_ACCENTS)] for i, k in enumerate(vlm_keys)}
    crop_manifest = data["crop_manifest"]
    cv_results    = data["cv_results"]
    image_paths   = data["images"]

    # ── Legend ────────────────────────────────────────────────────────────────
    legend = "".join(
        f"<span style='display:inline-block;margin:0 10px 6px 0;padding:3px 12px;"
        f"border-left:4px solid {accents[k]};background:#111;border-radius:3px;"
        f"font-size:11px;color:#ccc'>{MODEL_SHORT.get(k, k)}</span>"
        for k in vlm_keys
    )

    # ── Summary table: per model, how many participant=Y / talking=Y ──────────
    summary_rows = ""
    for k in vlm_keys:
        vlm_data = data["vlm_results"][k]
        total = part_y = talk_y = parse_err = 0
        for stem, persons in vlm_data.items():
            for p in persons:
                total += 1
                if p.get("participant") is True:   part_y += 1
                if p.get("talking") is True:        talk_y += 1
                if p.get("participant") is None or p.get("talking") is None:
                    parse_err += 1
        summary_rows += (
            f"<tr>"
            f"<td style='border-left:4px solid {accents[k]};padding-left:8px;color:#ccc'>"
            f"{MODEL_SHORT.get(k,k)}</td>"
            f"<td style='color:#888'>{total}</td>"
            f"<td style='color:#3d9'>{part_y} <span style='color:#444'>({part_y/total:.0%})</span></td>"
            f"<td style='color:#58a6ff'>{talk_y} <span style='color:#444'>({talk_y/total:.0%})</span></td>"
            f"<td style='color:{'#e54' if parse_err else '#444'}'>{parse_err}</td>"
            f"</tr>"
        )

    # ── Per-image sections ────────────────────────────────────────────────────
    image_sections = ""
    for img_path in image_paths:
        stem = Path(img_path).stem
        crops = crop_manifest.get(stem, [])

        # full room image
        src = img_b64(img_path)
        room_img = (
            f'<img src="{src}" style="width:100%;max-height:300px;object-fit:cover;'
            f'border-radius:6px;display:block;margin-bottom:10px">'
            if src else
            '<div style="width:100%;height:200px;background:#1a1a1a;border-radius:6px;'
            'margin-bottom:10px;display:flex;align-items:center;justify-content:center;'
            'color:#333;font-size:11px">image not found</div>'
        )

        # CV detection summary
        cv_row = ""
        for det in ["yolo11n", "yolo11s", "mobilenet_ssd"]:
            n = cv_results.get(det, {}).get(stem, {}).get("n_persons", "—")
            cv_row += (
                f"<span style='margin-right:14px;font-size:10px;color:#555'>"
                f"{det}: <span style='color:#aaa'>{n}</span></span>"
            )

        if not crops:
            image_sections += (
                f"<div style='background:#111;border-radius:8px;padding:16px;"
                f"border:1px solid #1e1e1e;margin-bottom:24px'>"
                f"{room_img}"
                f"<div style='font-size:11px;color:#666;margin-bottom:6px'>{Path(img_path).name}</div>"
                f"<div style='margin-bottom:10px'>{cv_row}</div>"
                f"<div style='color:#333;font-size:11px'>No persons detected</div>"
                f"</div>"
            )
            continue

        # Per-person crop grid
        person_cards = ""
        for crop_info in crops:
            crop_path = crop_info["crop_path"]
            pidx      = crop_info["person_idx"]
            conf      = crop_info["confidence"]

            crop_src = img_b64(crop_path)
            crop_img_html = (
                f'<img src="{crop_src}" style="width:100%;height:120px;object-fit:cover;'
                f'border-radius:4px;display:block;margin-bottom:6px">'
                if crop_src else
                '<div style="width:100%;height:120px;background:#1a1a1a;border-radius:4px;margin-bottom:6px"></div>'
            )

            # Per-VLM verdicts
            vlm_blocks = ""
            for k in vlm_keys:
                persons_list = data["vlm_results"][k].get(stem, [])
                p_data = next((p for p in persons_list if p["person_idx"] == pidx), None)

                accent = accents[k]
                if p_data is None:
                    vlm_blocks += (
                        f"<div style='border-left:2px solid {accent}22;padding:4px 6px;"
                        f"margin-bottom:4px;background:#0d0d0d;border-radius:0 3px 3px 0'>"
                        f"<span style='font-size:9px;color:{accent}'>{MODEL_SHORT.get(k,k)}</span>"
                        f"<span style='font-size:9px;color:#333'> — no data</span></div>"
                    )
                    continue

                part  = p_data.get("participant")
                talk  = p_data.get("talking")
                p_resp = p_data.get("participant_response", "")[:80]
                t_resp = p_data.get("talking_response", "")[:80]
                p_lat  = p_data.get("participant_latency", 0)
                t_lat  = p_data.get("talking_latency", 0)
                total_lat = round((p_lat + t_lat))

                part_color = "#3d9" if part else "#e54" if part is False else "#888"
                talk_color = "#58a6ff" if talk else "#e54" if talk is False else "#888"
                part_str   = "PARTICIPANT" if part else "NOT PART." if part is False else "?"
                talk_str   = "TALKING" if talk else "SILENT" if talk is False else "?"

                vlm_blocks += (
                    f"<div style='border-left:2px solid {accent};padding:5px 7px;"
                    f"margin-bottom:4px;background:#0d0d0d;border-radius:0 3px 3px 0'>"
                    f"<div style='font-size:9px;color:{accent};margin-bottom:3px;font-weight:700'>"
                    f"{MODEL_SHORT.get(k,k)}</div>"
                    f"<div style='display:flex;gap:6px;margin-bottom:3px'>"
                    f"<span style='font-size:10px;font-weight:700;color:{part_color}'>{part_str}</span>"
                    f"<span style='font-size:10px;font-weight:700;color:{talk_color}'>{talk_str}</span>"
                    f"<span style='font-size:9px;color:#333;margin-left:auto'>{total_lat}ms</span>"
                    f"</div>"
                    f"<div style='font-size:8px;color:#555;line-height:1.3'>{p_resp}</div>"
                    f"</div>"
                )

            person_cards += (
                f"<div style='background:#0d0d0d;border-radius:6px;padding:10px;"
                f"border:1px solid #1e1e1e;break-inside:avoid'>"
                f"{crop_img_html}"
                f"<div style='font-size:9px;color:#444;margin-bottom:6px'>"
                f"person {pidx:02d} · conf {conf:.2f}</div>"
                f"{vlm_blocks}"
                f"</div>"
            )

        image_sections += (
            f"<div style='background:#111;border-radius:8px;padding:16px;"
            f"border:1px solid #1e1e1e;margin-bottom:32px'>"
            f"<div style='display:flex;gap:16px;margin-bottom:12px'>"
            f"<div style='flex:1;min-width:0'>{room_img}</div>"
            f"<div style='width:200px;flex-shrink:0'>"
            f"<div style='font-size:11px;color:#666;margin-bottom:8px;word-break:break-all'>{Path(img_path).name}</div>"
            f"<div style='font-size:10px;color:#555;margin-bottom:4px'>CV detections:</div>"
            f"<div>{cv_row}</div>"
            f"<div style='margin-top:10px;font-size:10px;color:#555'>{len(crops)} person(s) detected</div>"
            f"</div></div>"
            f"<div style='columns:3;column-gap:12px'>{person_cards}</div>"
            f"</div>"
        )

    ts_range = ", ".join(sorted(set(data["vlm_ts"].values())))

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>People Analysis Pipeline Report</title>
<style>
  body {{ font-family: 'Courier New', monospace; background: #0d0d0d; color: #ccc;
          padding: 2rem; max-width: 1600px; margin: 0 auto; }}
  h1 {{ color: #fff; font-size: 1.3rem; letter-spacing: 3px; text-transform: uppercase; margin-bottom: .3rem; }}
  h2 {{ color: #888; font-size: .85rem; letter-spacing: 2px; text-transform: uppercase;
        border-bottom: 1px solid #222; padding-bottom: .4rem; margin: 2rem 0 1rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; font-size: .8rem; }}
  th {{ background: #161616; color: #888; padding: 6px 10px; text-align: left; border-bottom: 1px solid #222; }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #181818; }}
  .note {{ font-size: 11px; color: #444; margin: .3rem 0 1.2rem; }}
</style>
</head>
<body>
<h1>People Analysis Pipeline</h1>
<p class="note">Two-stage: YOLOv11s detection → full room + crop → VLM classification</p>
<p class="note">Runs: {ts_range} · {len(vlm_keys)} VLMs · {len(image_paths)} images</p>

<div style="margin-bottom:1.2rem">{legend}</div>

<h2>Summary — Participant & Talking Rates</h2>
<table>
  <tr><th>Model</th><th>Total Persons</th><th>Participant=YES</th><th>Talking=YES</th><th>Parse Errors</th></tr>
  {summary_rows}
</table>

<h2>Per-Image Results</h2>
{image_sections}
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260406", help="Filter result files by date prefix")
    args = parser.parse_args()

    print(f"Merging results (date={args.date or 'all'})...")
    data = merge_results(args.date or None)
    print(f"VLMs: {list(data['vlm_results'].keys())}")
    print(f"Images: {len(data['images'])}")

    html = build_html(data)
    out = RESULTS_DIR / "pipeline_people_report.html"
    out.write_text(html)
    print(f"Report → {out}")


if __name__ == "__main__":
    main()
