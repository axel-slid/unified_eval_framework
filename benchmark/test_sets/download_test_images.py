#!/usr/bin/env python
"""
download_test_images.py — download N diverse images from Wikimedia Commons.
No API key needed.

Usage:
    python test_sets/download_test_images.py --count 100
    python test_sets/download_test_images.py --count 100 --output test_sets/images/captioning
    python test_sets/download_test_images.py --count 100 --query "cats"
    python test_sets/download_test_images.py --count 100 --delay 2.0   # slower if still hitting limits
"""

import argparse
import json
import random
import time
from pathlib import Path

import httpx

DEFAULT_QUERIES = [
    "street people walking",
    "wild animals",
    "food cooking",
    "city skyline",
    "sports athletes",
    "living room interior",
    "cars traffic",
    "office technology",
    "mountain landscape",
    "market shopping",
]

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}

HEADERS = {
    "User-Agent": "VLMBenchmark/1.0 (research project; https://github.com) httpx/python"
}

MAX_RETRIES = 4


def fetch_wikimedia_images(client: httpx.Client, query: str, count: int) -> list[str]:
    urls = []
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"{query} filetype:bitmap",
        "gsrnamespace": "6",
        "gsrlimit": min(count * 3, 50),
        "prop": "imageinfo",
        "iiprop": "url|mediatype|size",
        "iiurlwidth": 800,
    }
    try:
        r = client.get(WIKIMEDIA_API, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            if len(urls) >= count * 3:
                break
            info = page.get("imageinfo", [{}])[0]
            url = info.get("thumburl") or info.get("url", "")
            ext = Path(url.split("?")[0]).suffix.lower()
            if url and ext in SUPPORTED_EXTS:
                urls.append(url)
    except Exception as e:
        print(f"    Wikimedia API error for '{query}': {e}")
    return urls


def download_with_retry(client: httpx.Client, url: str, delay: float) -> bytes | None:
    """Download with exponential backoff on 429."""
    for attempt in range(MAX_RETRIES):
        try:
            r = client.get(url, timeout=20)
            if r.status_code == 429:
                wait = (2 ** attempt) * delay + random.uniform(1, 3)
                print(f"    Rate limited — waiting {wait:.1f}s (retry {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.content
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait = (2 ** attempt) * delay + random.uniform(1, 3)
                print(f"    Rate limited — waiting {wait:.1f}s (retry {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                return None
        except Exception:
            return None
    return None


def download_images(output_dir: str, count: int, queries: list[str], delay: float) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_query = max(1, count // len(queries))
    downloaded = []
    idx = 1

    print(f"Downloading ~{count} images across {len(queries)} categories")
    print(f"Delay between downloads: {delay}s\n")

    with httpx.Client(headers=HEADERS, follow_redirects=True) as client:
        for query in queries:
            if len(downloaded) >= count:
                break

            print(f"  [{query}] fetching URLs...")
            urls = fetch_wikimedia_images(client, query, per_query)
            time.sleep(1.0)  # pause after API call

            if not urls:
                print(f"  [{query}] no URLs found, skipping")
                continue

            category_count = 0
            for url in urls:
                if len(downloaded) >= count or category_count >= per_query:
                    break

                content = download_with_retry(client, url, delay)
                if content is None:
                    continue

                ext = Path(url.split("?")[0]).suffix.lower().strip(".") or "jpg"
                if ext not in ("jpg", "jpeg", "png"):
                    ext = "jpg"
                if ext == "jpeg":
                    ext = "jpg"

                img_path = output_dir / f"{idx:03d}.{ext}"
                img_path.write_bytes(content)

                downloaded.append(img_path)
                print(f"    [{idx:03d}] {query} → {img_path.name}")
                idx += 1
                category_count += 1

                time.sleep(delay)

    print(f"\nDownloaded {len(downloaded)} images to {output_dir}")
    return downloaded


def generate_test_set(images: list[Path], output_path: str, question: str, rubric: str) -> None:
    output_path = Path(output_path)
    items = []
    for i, img_path in enumerate(images, start=1):
        items.append({
            "id": f"{i:03d}",
            "image": str(img_path),
            "question": question,
            "reference_answer": "",
            "rubric": rubric,
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(items, f, indent=2)
    print(f"Test set JSON saved → {output_path} ({len(items)} items)")


CAPTIONING_QUESTION = "Describe this image in detail."

CAPTIONING_RUBRIC = (
    "Judge the description based on the image provided. "
    "Award 5 if accurate, detailed, mentions main subjects, actions, colors, and setting. "
    "Award 4 if mostly accurate with minor omissions. "
    "Award 3 if correct but vague or missing important elements. "
    "Award 2 if partially correct with notable errors or hallucinations. "
    "Award 1 if mostly wrong, irrelevant, or heavily hallucinated. "
    "Penalize any details clearly not present in the image."
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", default="test_sets/images/captioning")
    parser.add_argument("--test-set", default="test_sets/captioning_100.json")
    parser.add_argument("--query", default=None)
    parser.add_argument("--delay", type=float, default=1.5, help="Seconds between downloads (increase if hitting 429s)")
    args = parser.parse_args()

    queries = [args.query] if args.query else DEFAULT_QUERIES

    images = download_images(
        output_dir=args.output,
        count=args.count,
        queries=queries,
        delay=args.delay,
    )

    if not images:
        print("No images downloaded. Check your network connection.")
        exit(1)

    generate_test_set(
        images=images,
        output_path=args.test_set,
        question=CAPTIONING_QUESTION,
        rubric=CAPTIONING_RUBRIC,
    )

    print(f"\nDone. Run benchmark with:")
    print(f"  CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --test-set {args.test_set}")