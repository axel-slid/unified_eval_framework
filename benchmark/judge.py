from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass

import httpx

from config import JudgeConfig


@dataclass
class JudgeResult:
    score: int        # 0–100
    reason: str
    error: str | None = None


JUDGE_PROMPT = """\
You are an impartial judge evaluating the quality of a vision-language model's response.
You are provided with the image the model was shown.

Question asked to the model:
{question}

Grading rubric:
{rubric}

Reference answer:
{reference_answer}

Model response:
{response}

Score the response from 0 to 100 using this scale:
0–20   - Completely wrong, refused to answer, or heavy hallucination
21–40  - Mostly wrong, minor correct elements
41–60  - Partially correct, missing key details
61–80  - Mostly correct, minor issues or omissions
81–100 - Fully correct and complete, accurate to the image

Base the score on correctness relative to the image, reference answer, and rubric.
Return ONLY valid JSON with no preamble, no markdown fences:
{{"score": <int 0-100>, "reason": "<one sentence explanation>"}}
"""


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def judge(
    question: str,
    rubric: str,
    response: str,
    cfg: JudgeConfig,
    reference_answer: str = "",
    image_path: str | None = None,
) -> JudgeResult:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("    [judge] No OPENAI_API_KEY — scoring skipped, score set to 0")
        return JudgeResult(score=0, reason="No API key set — scoring skipped", error=None)

    if not response.strip():
        return JudgeResult(score=0, reason="Model returned empty response")

    text_content = JUDGE_PROMPT.format(
        question=question,
        rubric=rubric,
        reference_answer=reference_answer,
        response=response,
    )

    if image_path and os.path.exists(image_path):
        b64 = _encode_image(image_path)
        ext = os.path.splitext(image_path)[-1].lower().strip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")
        content = [
            {"type": "text", "text": text_content},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]
    else:
        content = text_content

    payload = {
        "model": cfg.model,
        "max_completion_tokens": cfg.max_tokens,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": content}],
    }

    try:
        r = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=cfg.timeout_seconds,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        parsed = json.loads(text)
        score = int(parsed["score"])
        reason = str(parsed["reason"]).strip()
        if not 0 <= score <= 100:
            raise ValueError(f"Score out of range: {score}")
        if not reason:
            raise ValueError("Empty reason returned by judge")
        return JudgeResult(score=score, reason=reason)
    except httpx.HTTPStatusError as e:
        detail = e.response.text if e.response is not None else str(e)
        return JudgeResult(score=0, reason="", error=f"HTTP error: {detail}")
    except Exception as e:
        return JudgeResult(score=0, reason="", error=str(e))