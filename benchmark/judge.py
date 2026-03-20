"""
judge.py — Claude-as-judge scorer.
Receives a JudgeConfig so all model/token/timeout settings come from the YAML.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import httpx

from config import JudgeConfig


@dataclass
class JudgeResult:
    score: int        # 1–5, or 0 on error
    reason: str
    error: str | None = None


JUDGE_PROMPT = """\
You are an impartial judge evaluating the quality of a vision-language model's response.

Question asked to the model:
{question}

Grading rubric:
{rubric}

Model response:
{response}

Score the response from 1 to 5 using this scale:
1 - Completely wrong or refused to answer
2 - Mostly wrong, minor correct elements
3 - Partially correct, missing key details
4 - Mostly correct, minor issues
5 - Fully correct and complete

Return ONLY valid JSON with no preamble, no markdown fences:
{{"score": <int 1-5>, "reason": "<one sentence explanation>"}}
"""


def judge(question: str, rubric: str, response: str, cfg: JudgeConfig) -> JudgeResult:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return JudgeResult(score=0, reason="", error="ANTHROPIC_API_KEY not set")

    if not response.strip():
        return JudgeResult(score=1, reason="Model returned empty response")

    payload = {
        "model": cfg.model,
        "max_tokens": cfg.max_tokens,
        "messages": [
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    question=question,
                    rubric=rubric,
                    response=response,
                ),
            }
        ],
    }

    try:
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=cfg.timeout_seconds,
        )
        r.raise_for_status()
        text = r.json()["content"][0]["text"].strip()
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        return JudgeResult(score=int(parsed["score"]), reason=parsed["reason"])
    except Exception as e:
        return JudgeResult(score=0, reason="", error=str(e))
