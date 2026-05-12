"""Benchmark-specific LiveIdeaBench annotator."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.request import Request

from llm_judge.generic_llm_judge_annotator import (
    BACKUP_JUDGE_MODEL,
    _JUDGE_OVERRIDE,
    _call_openrouter_direct,
)

_JUDGE_MODEL_OVERRIDE = os.environ.get("LIVEIDEABENCH_JUDGE_MODEL_OVERRIDE", "").strip() or None

_CRITIC_PROMPT = """You are an extremely demanding scientific reviewer with the highest critical standards, like those at Nature or Science. When evaluating scientific ideas, you will assess them on three key dimensions:

1. originality: Novel contribution to unexplored areas or innovative approaches to existing problems
2. feasibility: Technical implementation and practicality
3. clarity: How well-articulated and easy to understand the idea is

Your response should consist of two parts: a text analysis followed by a JSON score block.

First, provide your brief analysis (less than 100 words) of the idea. Then, for each dimension, provide a score from 1 to 10 where 1-3 = poor, 4-6 = average, 7-10 = excellent.

For example:
```json
{
 "originality": ,
 "feasibility": ,
 "clarity":
}```"""

_DIMENSION_USER_PROMPT = "Please evaluate the following scientific idea:\n\n{idea}"

_FLUENCY_PROMPT = """Here are two ideas submitted to "Good Scientific Ideas" Competition, which both relate to "{keyword}":

# The first idea

{idea_a}

# The second idea

{idea_b}


# Question

Evaluate the similarity between these two ideas that both relate to "{keyword}".
Please choose the best answer:

A. Completely different ideas addressing different problems, despite relating to the same keyword.
B. Different ideas but addressing similar problems.
C. Similar ideas addressing similar or identical problems.
D. Academically identical ideas with the same core approach and problem statement.

ONLY ANSWER A/B/C/D, DO NOT EXPLAIN"""

_JSON_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)
_JSON_BRACE_RE = re.compile(r"(\{.*?\})", re.DOTALL)
_SCORE_RE = re.compile(r'"?(originality|feasibility|clarity)"?\s*:\s*([0-9]+(?:\.[0-9]+)?)', re.IGNORECASE)
_FLUENCY_LABEL_RE = re.compile(r"\b([ABCD])\b")
_FLUENCY_SCORE = {
    "A": 10.0,
    "B": 7.0,
    "C": 4.0,
    "D": 1.0,
}


def _mean(values: List[Optional[float]]) -> Optional[float]:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def _parse_dimension_scores(text: str) -> Dict[str, Any]:
    stripped = (text or "").strip()
    parsed: Dict[str, Optional[float]] = {
        "originality": None,
        "feasibility": None,
        "clarity": None,
    }

    for pattern in (_JSON_CODE_BLOCK_RE, _JSON_BRACE_RE):
        match = pattern.search(stripped)
        if not match:
            continue
        try:
            payload = json.loads(match.group(1))
        except Exception:
            continue
        for key in parsed:
            value = payload.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                parsed[key] = float(value)
        # Use explicit None-check (not truthy `all`): a score of 0.0 from
        # a misbehaving judge is parse-able and would have been caught by
        # the range filter below; we should not retry.
        if all(value is not None for value in parsed.values()):
            break

    if not all(value is not None for value in parsed.values()):
        for label, raw_value in _SCORE_RE.findall(stripped):
            parsed[label.lower()] = float(raw_value)

    # Range-validate: rubric is 1-10. Anything outside that range is
    # rejected so out-of-range stray ints (years, line numbers) cannot
    # masquerade as scores. Scores left as None signal an unparseable
    # judge reply; downstream `_mean` skips Nones.
    valid = all(
        value is not None and 1.0 <= float(value) <= 10.0
        for value in parsed.values()
    )
    for key, value in list(parsed.items()):
        if value is not None and not (1.0 <= float(value) <= 10.0):
            parsed[key] = None
    return {
        **parsed,
        "parse_rate": 1.0 if valid else 0.0,
    }


def _parse_fluency(text: str) -> Dict[str, Any]:
    stripped = (text or "").strip()
    match = _FLUENCY_LABEL_RE.search(stripped)
    if not match:
        return {
            "label": "",
            "score": None,
            "parse_rate": 0.0,
        }
    label = match.group(1).upper()
    return {
        "label": label,
        "score": _FLUENCY_SCORE.get(label),
        "parse_rate": 1.0 if label in _FLUENCY_SCORE else 0.0,
    }


class LiveIdeaBenchAnnotator(Annotator):
    """Restore the benchmark's multi-dimension evaluator in HELM-friendly form."""

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
        self.name = "liveideabench_v2_evaluator"

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

    def _judge_with_fallback(self, prompt: str) -> tuple[str, str]:
        primary_model = self.judge_model_name
        try:
            return primary_model, self._call_judge(primary_model, prompt)
        except Exception:
            return BACKUP_JUDGE_MODEL, self._call_judge(BACKUP_JUDGE_MODEL, prompt)

    def annotate(self, request_state: RequestState) -> Dict[str, Any]:
        assert request_state.result is not None
        extra_data = request_state.instance.extra_data or {}
        keyword = extra_data.get("keyword", "")
        completions = [
            completion.text.strip()
            for completion in request_state.result.completions
            if completion.text and completion.text.strip()
        ]

        dimension_scores: List[Dict[str, Any]] = []
        dimension_raw: List[str] = []
        dimension_models: List[str] = []
        for idea in completions:
            prompt = f"{_CRITIC_PROMPT}\n\n{_DIMENSION_USER_PROMPT.format(idea=idea)}"
            judge_model, raw_text = self._judge_with_fallback(prompt)
            parsed = _parse_dimension_scores(raw_text)
            dimension_scores.append(parsed)
            dimension_raw.append(raw_text)
            dimension_models.append(judge_model)

        fluency_label = ""
        fluency_score: Optional[float] = None
        fluency_parse_rate = 0.0
        fluency_raw_text = ""
        fluency_judge_model = ""
        if len(completions) >= 2:
            fluency_prompt = _FLUENCY_PROMPT.format(
                keyword=keyword,
                idea_a=completions[0],
                idea_b=completions[1],
            )
            fluency_judge_model, fluency_raw_text = self._judge_with_fallback(fluency_prompt)
            fluency = _parse_fluency(fluency_raw_text)
            fluency_label = fluency["label"]
            fluency_score = fluency["score"]
            fluency_parse_rate = float(fluency["parse_rate"])

        mean_originality = _mean([item.get("originality") for item in dimension_scores])
        mean_feasibility = _mean([item.get("feasibility") for item in dimension_scores])
        mean_clarity = _mean([item.get("clarity") for item in dimension_scores])
        # parse_rate is a [0,1] proportion. The previous `_mean(...) or 0.0`
        # collapsed both "no completions" (None) and "all failed" (0.0) to
        # the same literal 0.0 — fine for parse_rate semantics but masks
        # the no-completions branch from any caller that wants to detect
        # "judge was never invoked." Be explicit about the None case.
        _raw_parse_mean = _mean([item.get("parse_rate") for item in dimension_scores])
        dimension_parse_rate = 0.0 if _raw_parse_mean is None else _raw_parse_mean

        overall_components = [
            mean_originality,
            mean_feasibility,
            mean_clarity,
            fluency_score,
        ]
        overall_score = _mean(overall_components)

        return {
            "judge_model": self.judge_model_name,
            "dimension_judge_models": dimension_models,
            "fluency_judge_model": fluency_judge_model,
            "ideas": completions,
            "output_count": len(completions),
            "dimension_scores": dimension_scores,
            "raw_dimension_judgments": dimension_raw,
            "mean_originality": mean_originality,
            "mean_feasibility": mean_feasibility,
            "mean_clarity": mean_clarity,
            "dimension_parse_rate": float(dimension_parse_rate),
            "fluency_label": fluency_label,
            "fluency_score": fluency_score,
            "fluency_parse_rate": float(fluency_parse_rate),
            "raw_fluency_judgment": fluency_raw_text,
            "overall_score": overall_score,
        }
