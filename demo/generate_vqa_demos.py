#!/usr/bin/env python3
"""
generate_vqa_demos.py — Generate bad_demo.html and good_demo.html
showing contrasting VQA performance examples.

Usage:
    cd demo
    python generate_vqa_demos.py
"""

import base64
import sys
from pathlib import Path

IMAGES_DIR = Path(__file__).parent.parent / "benchmark/test_sets/images/captioning"
OUT_DIR = Path(__file__).parent


def _img_b64(path: Path) -> str | None:
    try:
        ext = path.suffix.lower().strip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
    except Exception:
        return None


def _score_color(score: int) -> str:
    if score >= 90: return "#3d9"
    if score >= 70: return "#8bc34a"
    if score >= 50: return "#fa0"
    return "#e54"


# ── Demo data ──────────────────────────────────────────────────────────────────

IMAGES = [
    {"file": "001.jpg", "id": "001"},
    {"file": "002.jpg", "id": "002"},
    {"file": "003.jpg", "id": "003"},
]

# Questions matched to the first 3 captioning images
# (European street scene, plaza with bicycle, other)
QUESTIONS = {
    "001": [
        {
            "id": 1,
            "question": "What prominent structure is visible at the end of the street in the center of the image?",
            "reference_answer": "A tall clock tower is visible at the end of the street. It stands in the center background and is the most prominent landmark in the scene.",
        },
        {
            "id": 2,
            "question": "How many people are walking in the foreground on the left side of the street, closest to the camera?",
            "reference_answer": "There are three people walking closest to the camera on the left side. They are in the foreground near the storefronts.",
        },
        {
            "id": 3,
            "question": "What color is the large awning on the right side of the street?",
            "reference_answer": "The large awning on the right side is blue. It covers a storefront near the lower-right area of the image.",
        },
        {
            "id": 4,
            "question": "Where is the person standing with a bicycle located relative to the center of the street?",
            "reference_answer": "The person with the bicycle is near the center of the street, slightly right of center. They are standing on the tram tracks.",
        },
        {
            "id": 5,
            "question": "What kind of scene is shown, based on the buildings, shops, and people present?",
            "reference_answer": "It is a busy urban pedestrian street scene with storefronts, tram tracks, and many people walking through a city center.",
        },
    ],
    "002": [
        {
            "id": 1,
            "question": "What is the most prominent vehicle in the foreground, and what color is it?",
            "reference_answer": "It is a bicycle, and it is red. The bike is partially visible at the bottom of the image.",
        },
        {
            "id": 2,
            "question": "How many people are walking together near the center-left of the scene?",
            "reference_answer": "Two people are walking together. They are side by side near the center-left area.",
        },
        {
            "id": 3,
            "question": "What large object stands on a pole near the left side of the plaza, and what color is its main face?",
            "reference_answer": "It is a large signboard or advertisement display. Its main face is mostly red.",
        },
        {
            "id": 4,
            "question": "Where is the tall black pole located relative to the red bicycle in the foreground?",
            "reference_answer": "The tall black pole is directly behind the red bicycle and slightly above it in the frame.",
        },
        {
            "id": 5,
            "question": "What type of open public space is shown, and what surrounds it?",
            "reference_answer": "An open pedestrian plaza or town square is shown. It is surrounded by multi-story buildings and has a few people and a bicycle visible.",
        },
    ],
    "003": [
        {
            "id": 1,
            "question": "What is the main subject of this image?",
            "reference_answer": "The main subject is a group of people in an outdoor setting, likely at a public gathering or market.",
        },
        {
            "id": 2,
            "question": "What colors dominate the scene?",
            "reference_answer": "Warm tones such as orange, red, and brown dominate, with some green foliage visible in the background.",
        },
        {
            "id": 3,
            "question": "How many distinct people can be clearly identified in the image?",
            "reference_answer": "At least four distinct people are clearly visible, with others partially visible in the background.",
        },
        {
            "id": 4,
            "question": "What is the approximate time of day suggested by the lighting in the image?",
            "reference_answer": "The warm, directional lighting suggests late afternoon or golden hour, around 4–6 PM.",
        },
        {
            "id": 5,
            "question": "Are there any objects being carried or held by people in the image?",
            "reference_answer": "Yes, at least one person appears to be carrying a bag or basket, consistent with a market or outdoor vendor setting.",
        },
    ],
}

# ── Good model answers (high scores, accurate and specific) ────────────────────
GOOD_ANSWERS = {
    "001": [
        {
            "model_answer": "A tall historic clock tower is visible at the end of the street. It appears to be part of a church or civic building and is the clear focal point of the scene.",
            "score": 100,
            "reason": "The model correctly identifies the clock tower as the prominent structure at the end of the street, matching the reference with relevant detail.",
            "latency_ms": 832,
        },
        {
            "model_answer": "There are 3 people walking in the foreground on the left side of the street closest to the camera.",
            "score": 100,
            "reason": "The model correctly identifies that there are three people walking closest to the camera on the left side of the street.",
            "latency_ms": 806,
        },
        {
            "model_answer": "The large awning on the right side of the street is blue.",
            "score": 100,
            "reason": "The model correctly identifies the large awning on the right side as blue.",
            "latency_ms": 826,
        },
        {
            "model_answer": "The person with the bicycle is slightly right of the center of the street, positioned on the tram tracks.",
            "score": 98,
            "reason": "The answer correctly captures the key location as slightly right of center and includes the tram track detail from the reference.",
            "latency_ms": 801,
        },
        {
            "model_answer": "This is a busy urban shopping street in a city center, with pedestrians, storefronts, and tram tracks visible. The historic buildings and clock tower suggest a European downtown commercial area.",
            "score": 97,
            "reason": "The answer accurately identifies a busy urban pedestrian shopping street with storefronts, tram tracks, and people. The European context is a plausible but non-essential addition.",
            "latency_ms": 1170,
        },
    ],
    "002": [
        {
            "model_answer": "The most prominent vehicle in the foreground is a bicycle. It is red.",
            "score": 100,
            "reason": "The model correctly identifies the foreground vehicle as a bicycle and its color as red.",
            "latency_ms": 747,
        },
        {
            "model_answer": "Two people are walking together near the center-left of the scene.",
            "score": 100,
            "reason": "The model correctly states that two people are walking together near the center-left of the scene.",
            "latency_ms": 957,
        },
        {
            "model_answer": "A large billboard stands on a pole near the left side of the plaza. Its main face is red.",
            "score": 95,
            "reason": "The model correctly identifies the object as a billboard on a pole with a red face, matching the reference closely.",
            "latency_ms": 836,
        },
        {
            "model_answer": "The tall black pole is directly behind the red bicycle in the foreground, slightly above it in the frame.",
            "score": 98,
            "reason": "The answer correctly states that the pole is directly behind the red bicycle and slightly above it, closely matching the reference.",
            "latency_ms": 868,
        },
        {
            "model_answer": "An open pedestrian plaza is shown. It is surrounded by multi-story buildings and has a few people and a bicycle present.",
            "score": 97,
            "reason": "The answer correctly identifies an open pedestrian plaza surrounded by multi-story buildings with people and a bicycle present.",
            "latency_ms": 912,
        },
    ],
    "003": [
        {
            "model_answer": "The main subject is a group of people gathered outdoors, likely at a street market or community event.",
            "score": 95,
            "reason": "The model correctly identifies the main subject as a group of people in an outdoor gathering setting.",
            "latency_ms": 778,
        },
        {
            "model_answer": "Warm tones — orange, red, and brown — dominate the scene. Some green foliage is visible in the upper background.",
            "score": 98,
            "reason": "The model accurately identifies the dominant warm color palette and mentions the green foliage, closely matching the reference.",
            "latency_ms": 843,
        },
        {
            "model_answer": "At least four people are clearly visible, with additional individuals partially visible in the background.",
            "score": 97,
            "reason": "The model correctly identifies at least four distinct people and acknowledges others in the background, consistent with the reference.",
            "latency_ms": 791,
        },
        {
            "model_answer": "The warm, golden-toned directional lighting suggests late afternoon, likely around 4–6 PM.",
            "score": 99,
            "reason": "The model correctly interprets the warm directional light as indicative of golden hour / late afternoon, matching the reference precisely.",
            "latency_ms": 819,
        },
        {
            "model_answer": "Yes, at least one person appears to be carrying a basket or bag, suggesting a market or vendor context.",
            "score": 97,
            "reason": "The model correctly identifies an object being carried and connects it to a market context, consistent with the reference.",
            "latency_ms": 856,
        },
    ],
}

# ── Bad model answers (low scores, hallucinations, vague, wrong) ───────────────
BAD_ANSWERS = {
    "001": [
        {
            "model_answer": "I can see some buildings in the background. There may be a tall structure of some kind, possibly a tower or large building.",
            "score": 25,
            "reason": "The answer is vague and non-committal. It fails to identify the clock tower specifically and does not describe its location at the end of the street.",
            "latency_ms": 315,
        },
        {
            "model_answer": "There are many people walking on the street. It looks like a crowded area with lots of foot traffic.",
            "score": 10,
            "reason": "The answer does not attempt to count the people in the foreground on the left side. It gives a general description instead of answering the specific question.",
            "latency_ms": 298,
        },
        {
            "model_answer": "The awning appears to be green or possibly striped. It is hard to tell from the image.",
            "score": 5,
            "reason": "The model incorrectly identifies the awning color as green. The reference clearly states it is blue. The hedging adds no value.",
            "latency_ms": 312,
        },
        {
            "model_answer": "The person with the bicycle is on the far left side of the street near the buildings.",
            "score": 15,
            "reason": "The model incorrectly places the person on the far left. The reference states they are near the center, slightly right of center on the tram tracks.",
            "latency_ms": 287,
        },
        {
            "model_answer": "It shows a street with some shops and people. It could be any city street. There are buildings.",
            "score": 20,
            "reason": "The answer is too generic and does not mention key elements like tram tracks, the urban pedestrian character, or the city center context. Minimal alignment with reference.",
            "latency_ms": 334,
        },
    ],
    "002": [
        {
            "model_answer": "There is a car or possibly a motorbike in the foreground. It looks dark colored.",
            "score": 0,
            "reason": "The model incorrectly identifies the foreground vehicle as a car or motorbike and gives the wrong color. The reference states it is a red bicycle.",
            "latency_ms": 301,
        },
        {
            "model_answer": "I can see several people in the scene walking around the plaza.",
            "score": 10,
            "reason": "The answer does not specify center-left location or give a count. 'Several' is incorrect — the reference states exactly two people walking together.",
            "latency_ms": 289,
        },
        {
            "model_answer": "There is a pole with a sign near the edge of the image. The sign appears to be white or light-colored.",
            "score": 15,
            "reason": "The model correctly identifies a pole with a sign, but gives the wrong color. The reference states the main face is red, not white.",
            "latency_ms": 318,
        },
        {
            "model_answer": "I don't see a clearly identifiable black pole in this image. There may be some vertical objects in the background.",
            "score": 5,
            "reason": "The model fails to identify the black pole entirely and does not describe its position relative to the bicycle.",
            "latency_ms": 276,
        },
        {
            "model_answer": "The image shows a street or outdoor area. There are buildings and some people visible.",
            "score": 10,
            "reason": "The answer fails to identify the space as a pedestrian plaza or describe the surrounding buildings and bicycle. Extremely generic.",
            "latency_ms": 295,
        },
    ],
    "003": [
        {
            "model_answer": "The image shows a landscape or nature scene. It might be a park or garden.",
            "score": 5,
            "reason": "The model misidentifies the main subject entirely. The reference states it shows a group of people in an outdoor gathering setting, not a landscape.",
            "latency_ms": 308,
        },
        {
            "model_answer": "The image looks mostly grey and dark, without many strong colors.",
            "score": 0,
            "reason": "The model's color description is completely wrong. The reference states warm tones like orange, red, and brown dominate. This is likely a hallucination.",
            "latency_ms": 281,
        },
        {
            "model_answer": "It is difficult to count the people. There might be one or two people in the image.",
            "score": 15,
            "reason": "The model severely under-counts, suggesting one or two people where the reference identifies at least four clearly visible individuals.",
            "latency_ms": 295,
        },
        {
            "model_answer": "The lighting suggests it is midday with harsh overhead sunlight.",
            "score": 10,
            "reason": "The model incorrectly identifies the lighting as midday overhead sunlight. The reference states the warm directional light suggests late afternoon / golden hour.",
            "latency_ms": 312,
        },
        {
            "model_answer": "I do not see anyone carrying objects in the image.",
            "score": 5,
            "reason": "The model incorrectly states no one is carrying objects. The reference identifies at least one person carrying a bag or basket, consistent with a market context.",
            "latency_ms": 278,
        },
    ],
}


# ── HTML generator ─────────────────────────────────────────────────────────────

def build_html(title: str, model_name: str, model_score: float, answers: dict, timestamp: str) -> str:
    gpt_avg = 98.6

    gap = f"{model_score - gpt_avg:+.1f}"
    gap_color = "#3d9" if model_score >= gpt_avg else "#e54"
    model_color = _score_color(model_score)
    gpt_color = _score_color(gpt_avg)

    summary_rows = (
        f"<tr>"
        f"<td>GPT Baseline (gpt-4o)</td>"
        f"<td><span style='color:{gpt_color};font-weight:700'>{gpt_avg}/100</span></td>"
        f"<td style='color:#888'>baseline</td>"
        f"<td>1064ms</td><td>3</td>"
        f"</tr>"
        f"<tr>"
        f"<td>{model_name}</td>"
        f"<td><span style='color:{model_color};font-weight:700'>{model_score:.1f}/100</span></td>"
        f"<td style='color:{gap_color}'>{gap}</td>"
        f"<td>{'310ms' if model_score < 50 else '2547ms'}</td><td>3</td>"
        f"</tr>"
    )

    detail_rows = ""
    for img_meta in IMAGES:
        img_id = img_meta["id"]
        img_path = IMAGES_DIR / img_meta["file"]
        img_src = _img_b64(img_path)
        img_html = (
            f'<img src="{img_src}" style="width:140px;height:105px;object-fit:cover;'
            f'border-radius:4px;border:1px solid #2a2a2a;display:block;margin-bottom:6px">'
            if img_src else ""
        )

        qs = QUESTIONS.get(img_id, [])
        q_list = "".join(
            f"<li style='margin:2px 0;color:#777;font-size:10px'>{q['question']}</li>"
            for q in qs
        )
        q_html = f"<ol style='padding-left:14px;margin:4px 0'>{q_list}</ol>"

        detail_rows += (
            f"<tr>"
            f"<td style='min-width:180px'>{img_html}"
            f"<b style='font-size:11px'>#{img_id}</b>{q_html}</td>"
        )

        # GPT baseline column (always good)
        gpt_ans = GOOD_ANSWERS.get(img_id, [])
        gpt_avg_img = sum(a["score"] for a in gpt_ans) / len(gpt_ans) if gpt_ans else 0
        gpt_color_img = _score_color(gpt_avg_img)
        gpt_breakdown = ""
        for qa in gpt_ans:
            qc = _score_color(qa["score"])
            # Show GPT reference answer as its answer
            ref = QUESTIONS[img_id][qa["id"] - 1]["reference_answer"]
            gpt_breakdown += (
                f"<div style='margin:3px 0;padding:4px 6px;background:#151515;"
                f"border-radius:3px;border-left:2px solid {qc}'>"
                f"<span style='color:{qc};font-weight:600;font-size:11px'>"
                f"Q{qa['id']}: {qa['score']}/100</span>"
                f"<br><span style='color:#aaa;font-size:10px'><b>Answer:</b> {ref}</span>"
                f"<br><small style='color:#888;font-size:10px'>{qa['reason']}</small>"
                f"</div>"
            )
        bar_w_gpt = int(gpt_avg_img)
        detail_rows += (
            f"<td>"
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px'>"
            f"<span style='color:{gpt_color_img};font-weight:700;font-size:14px'>"
            f"{gpt_avg_img:.1f}/100</span>"
            f"<small style='color:#555'>~1050ms</small></div>"
            f"<div style='background:#1a1a1a;border-radius:2px;height:3px;margin-bottom:8px'>"
            f"<div style='width:{bar_w_gpt}%;height:100%;background:{gpt_color_img};border-radius:2px'>"
            f"</div></div>"
            f"{gpt_breakdown}</td>"
        )

        # Model column
        model_ans = answers.get(img_id, [])
        model_avg_img = sum(a["score"] for a in model_ans) / len(model_ans) if model_ans else 0
        model_color_img = _score_color(model_avg_img)
        model_breakdown = ""
        for i, qa in enumerate(model_ans):
            qc = _score_color(qa["score"])
            model_breakdown += (
                f"<div style='margin:3px 0;padding:4px 6px;background:#151515;"
                f"border-radius:3px;border-left:2px solid {qc}'>"
                f"<span style='color:{qc};font-weight:600;font-size:11px'>"
                f"Q{qa['id'] if 'id' in qa else i+1}: {qa['score']}/100</span>"
                f"<br><span style='color:#aaa;font-size:10px'><b>Answer:</b> {qa['model_answer']}</span>"
                f"<br><small style='color:#888;font-size:10px'>{qa['reason']}</small>"
                f"</div>"
            )
        bar_w = int(model_avg_img)
        avg_lat = sum(a["latency_ms"] for a in model_ans) // len(model_ans) if model_ans else 0
        detail_rows += (
            f"<td>"
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px'>"
            f"<span style='color:{model_color_img};font-weight:700;font-size:14px'>"
            f"{model_avg_img:.1f}/100</span>"
            f"<small style='color:#555'>{avg_lat}ms</small></div>"
            f"<div style='background:#1a1a1a;border-radius:2px;height:3px;margin-bottom:8px'>"
            f"<div style='width:{bar_w}%;height:100%;background:{model_color_img};border-radius:2px'>"
            f"</div></div>"
            f"{model_breakdown}</td>"
        )
        detail_rows += "</tr>"

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: 'Courier New', monospace; background: #0d0d0d; color: #ccc; padding: 2rem; }}
  h1 {{ color: #fff; font-size: 1.4rem; letter-spacing: 3px; text-transform: uppercase; }}
  h2 {{ color: #888; font-size: .9rem; letter-spacing: 2px; text-transform: uppercase;
        border-bottom: 1px solid #222; padding-bottom: .4rem; margin-top: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; font-size: .82rem; }}
  th {{ background: #161616; color: #aaa; padding: 7px 12px; text-align: left;
        border-bottom: 1px solid #2a2a2a; }}
  td {{ padding: 10px 12px; border-bottom: 1px solid #1a1a1a; vertical-align: top; }}
  tr:hover td {{ background: #111; }}
  code {{ background: #1a1a1a; padding: 1px 5px; border-radius: 3px; font-size: .8rem; }}
  .tag {{ display:inline-block; background:#1a1a1a; color:#666;
          padding: 2px 10px; border-radius: 3px; font-size: .75rem; }}
  .note {{ font-size: 11px; color: #555; margin-top: 6px; font-style: italic; }}
  .badge {{ display:inline-block; padding: 3px 12px; border-radius: 3px; font-size: .75rem;
            font-weight: 700; letter-spacing: 1px; text-transform: uppercase; margin-left: 12px; }}
</style>
</head>
<body>
<h1>VQA Benchmark Report
  <span class="badge" style="background:{'#1a3a1a;color:#3d9' if model_score >= 80 else '#3a1a1a;color:#e54'}">
    {'GOOD PERFORMANCE' if model_score >= 80 else 'BAD PERFORMANCE'}
  </span>
</h1>
<p class="tag">{timestamp}</p>
<p class="note">5 GPT-generated questions per image · judged by GPT-4o · GPT answers as baseline</p>

<h2>Summary</h2>
<table>
  <tr><th>Model</th><th>Avg Score</th><th>vs GPT Baseline</th><th>Avg Latency</th><th>N images</th></tr>
  {summary_rows}
</table>

<h2>Per-Image Results</h2>
<table>
  <tr>
    <th>Image / Questions</th>
    <th>GPT Baseline (gpt-4o)</th>
    <th>{model_name}</th>
  </tr>
  {detail_rows}
</table>
</body>
</html>"""


def main():
    # Bad demo
    bad_avg = sum(
        sum(a["score"] for a in BAD_ANSWERS[img["id"]]) / len(BAD_ANSWERS[img["id"]])
        for img in IMAGES
    ) / len(IMAGES)

    bad_html = build_html(
        title="VQA Demo — Bad Performance",
        model_name="SmolVLM2-2.2B (bfloat16)",
        model_score=round(bad_avg, 1),
        answers=BAD_ANSWERS,
        timestamp="DEMO — Bad Performance Example",
    )
    bad_path = OUT_DIR / "vqa_demo_bad.html"
    bad_path.write_text(bad_html)
    print(f"Bad demo  → {bad_path}  (avg score: {bad_avg:.1f}/100)")

    # Good demo
    good_avg = sum(
        sum(a["score"] for a in GOOD_ANSWERS[img["id"]]) / len(GOOD_ANSWERS[img["id"]])
        for img in IMAGES
    ) / len(IMAGES)

    good_html = build_html(
        title="VQA Demo — Good Performance",
        model_name="Qwen3-VL-4B-Instruct (bfloat16)",
        model_score=round(good_avg, 1),
        answers=GOOD_ANSWERS,
        timestamp="DEMO — Good Performance Example",
    )
    good_path = OUT_DIR / "vqa_demo_good.html"
    good_path.write_text(good_html)
    print(f"Good demo → {good_path}  (avg score: {good_avg:.1f}/100)")


if __name__ == "__main__":
    main()
