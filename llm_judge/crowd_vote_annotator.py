"""Benchmark-specific annotator for Crowd Vote marketing creativity."""

from __future__ import annotations

import os
import re
from typing import Any, Dict

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.request import Request

from llm_judge.generic_llm_judge_annotator import (
    BACKUP_JUDGE_MODEL,
    _JUDGE_OVERRIDE,
    _call_openrouter_direct,
)

_JUDGE_MODEL_OVERRIDE = os.environ.get("CROWD_VOTE_JUDGE_MODEL_OVERRIDE", "").strip() or None

_PROMPT_TEMPLATE = """You are evaluating a marketing creativity benchmark response.

This local benchmark is a proxy adaptation of a pairwise human-preference benchmark.
Score the single response on four dimensions from 1 to 5.

Brand: {brand}
Category: {category}
Task type: {task_type}
Prompt: {prompt_text}
Response: {response}

Scoring guidance:
- Originality: Is the response surprising, non-cliche, and fresh?
- Brand Relevance: Is it meaningfully tied to this brand rather than generic?
- Creative Potential: Could it become a plausible campaign platform or activation?
- Conciseness: Does it respect the intended short-form constraint for the task?

Return exactly this format:
ORIGINALITY: [1-5]
BRAND_RELEVANCE: [1-5]
CREATIVE_POTENTIAL: [1-5]
CONCISENESS: [1-5]
OVERALL: [1-5]
"""

_SCORE_RE = re.compile(
    r"^(ORIGINALITY|BRAND_RELEVANCE|CREATIVE_POTENTIAL|CONCISENESS|OVERALL):\s*([1-5])\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def _parse_scores(text: str) -> Dict[str, Any]:
    values: Dict[str, float] = {}
    for label, raw in _SCORE_RE.findall((text or "").strip()):
        values[label.lower()] = float(raw)
    required = {
        "originality",
        "brand_relevance",
        "creative_potential",
        "conciseness",
        "overall",
    }
    return {
        **values,
        "parse_rate": 1.0 if required.issubset(values.keys()) else 0.0,
    }


class CrowdVoteAnnotator(Annotator):
    """Structured pointwise proxy for Crowd Vote marketing creativity."""

    def __init__(
        self,
        auto_client: AutoClient,
        judge_model_name: str,
        judge_temperature: float,
        judge_max_new_tokens: int,
        **_: Any,
    ):
        self._auto_client = auto_client
        self.judge_model_name = _JUDGE_MODEL_OVERRIDE or judge_model_name
        self.judge_temperature = judge_temperature
        self.judge_max_new_tokens = judge_max_new_tokens
        # Renamed 2026-04-25 (audit Tier-3 fix): the prefix `crowd_vote_*`
        # was misleading — no actual crowd voting occurs; this is a single
        # LLM-judge proxy. See dropped_metrics.md / launch_plan.md for the
        # paper-fidelity disclosure on this dataset.
        self.name = "marketing_creativity_judge"

    def _call_judge(self, model_name: str, prompt: str) -> str:
        if _JUDGE_OVERRIDE:
            return _call_openrouter_direct(
                _JUDGE_OVERRIDE,
                prompt,
                self.judge_temperature,
                self.judge_max_new_tokens,
            ).strip()

        request = Request(
            model=model_name,
            model_deployment=model_name,
            prompt=prompt,
            temperature=self.judge_temperature,
            max_tokens=self.judge_max_new_tokens,
            num_completions=1,
        )
        result = self._auto_client.make_request(request)
        if not result.success:
            raise RuntimeError(f"Judge call failed for model {model_name}")
        return result.completions[0].text.strip() if result.completions else ""

    def annotate(self, request_state: RequestState) -> Dict[str, Any]:
        assert request_state.result is not None
        completion = request_state.result.completions[0].text.strip() if request_state.result.completions else ""
        extra_data = request_state.instance.extra_data or {}
        prompt = _PROMPT_TEMPLATE.format(
            brand=extra_data.get("brand", ""),
            category=extra_data.get("category", ""),
            task_type=extra_data.get("task_type", ""),
            prompt_text=extra_data.get("prompt_text", request_state.instance.input.text),
            response=completion,
        )

        judge_model = self.judge_model_name
        try:
            raw_judge_text = self._call_judge(judge_model, prompt)
        except Exception:
            judge_model = BACKUP_JUDGE_MODEL
            raw_judge_text = self._call_judge(judge_model, prompt)

        parsed = _parse_scores(raw_judge_text)
        return {
            "judge_model": judge_model,
            "raw_judge_text": raw_judge_text,
            "originality": parsed.get("originality"),
            "brand_relevance": parsed.get("brand_relevance"),
            "creative_potential": parsed.get("creative_potential"),
            "conciseness": parsed.get("conciseness"),
            "overall": parsed.get("overall"),
            "parse_rate": parsed.get("parse_rate", 0.0),
        }
