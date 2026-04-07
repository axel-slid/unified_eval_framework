#!/usr/bin/env python3
"""
generate_sample_grid.py — 5x5 HTML grid of randomly sampled finetune training images.
"""

import base64
import json
import random
from pathlib import Path

TRAIN_JSONL = Path(__file__).parent / "data" / "coco_train.jsonl"
OUT_HTML = Path(__file__).parent / "viz_output" / "finetune_sample_grid.html"
SEED = 42
N = 25


def img_b64(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
    except Exception:
        return None


def main() -> None:
    random.seed(SEED)
    with open(TRAIN_JSONL) as f:
        samples = [json.loads(l) for l in f]

    picked = random.sample(samples, N)

    cells = ""
    for s in picked:
        img_path = s["messages"][0]["content"][0]["image"]
        fname = Path(img_path).name
        src = img_b64(img_path)
        if src:
            cells += f"""
<div class="cell">
  <img src="{src}">
  <div class="label">{fname}</div>
</div>"""
        else:
            cells += f'<div class="cell missing">{fname}</div>'

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Qwen3-VL Finetune — Training Sample Grid</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #f9fafb;
    color: #111827;
    padding: 2rem 2.5rem;
  }}
  h1 {{ font-size: 1.2rem; color: #111827; margin-bottom: .25rem; }}
  .subtitle {{ color: #9ca3af; font-size: .8rem; margin-bottom: 1.5rem; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 10px;
    max-width: 960px;
  }}
  .cell {{
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    overflow: hidden;
  }}
  .cell img {{
    width: 100%;
    aspect-ratio: 4/3;
    object-fit: cover;
    display: block;
  }}
  .cell .label {{
    padding: 4px 6px;
    font-size: 9px;
    color: #9ca3af;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .cell.missing {{
    height: 140px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #d1d5db;
    font-size: 11px;
  }}
</style>
</head>
<body>
<h1>Qwen3-VL Finetune — Training Data Sample</h1>
<p class="subtitle">{N} randomly sampled images from {len(samples):,} training examples (COCO 2017, people detection)</p>
<div class="grid">
{cells}
</div>
</body>
</html>"""

    OUT_HTML.parent.mkdir(exist_ok=True)
    OUT_HTML.write_text(html)
    print(f"Saved → {OUT_HTML}")


if __name__ == "__main__":
    main()
