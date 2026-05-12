"""TinyStories-specific multi-sample teacher-style judge annotator."""

from __future__ import annotations

import json
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


_SYSTEM_PROMPT = "You are an experienced elementary-school teacher evaluating student writing."

_USER_PROMPT = """
The student is given a beginning of a story. The student needs to complete it into a full story.
Evaluate the student's completed story as if you were grading a classroom exercise.

Score each dimension from 1 to 10:
- grammar: correctness and clarity of language
- creativity: originality and imagination of the story
- consistency: how well the completion fits the given beginning and remains coherent

Also estimate the age group of the hypothetical student writer using:
- A: 3 or under
- B: 4-5
- C: 6-7
- D: 8-9
- E: 10-12

Return only valid JSON in this exact format:
{{
  "grammar": 1-10 integer,
  "creativity": 1-10 integer,
  "consistency": 1-10 integer,
  "age_group": "A" | "B" | "C" | "D" | "E"
}}

Story beginning:
***
{story_beginning}
***

Student completion:
***
{completion}
***
""".strip()

_AGE_GROUP_TO_ORDINAL = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0}


def _strip_thinking(text: str) -> str:
    marker = "</think>\n\n"
    marker_pos = text.find(marker)
    if marker_pos != -1:
        return text[marker_pos + len(marker) :]
    return text


def _extract_json_dict(text: str) -> Optional[Dict[str, Any]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None

    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


class TinyStoriesAnnotator(Annotator):
    """Judge multiple TinyStories samples and average teacher-style scores."""

    def __init__(
        self,
        auto_client: AutoClient,
        judge_model_name: str,
        judge_temperature: float,
        judge_max_new_tokens: int,
        max_retries: int = 3,
    ):
        self._auto_client = auto_client
        self.judge_model_name = judge_model_name
        self.judge_temperature = judge_temperature
        self.judge_max_new_tokens = judge_max_new_tokens
        self.max_retries = max_retries
        self.name = "tinystories_judge"

    def _call_judge(self, model_name: str, story_beginning: str, completion: str) -> str:
        prompt = _USER_PROMPT.format(
            story_beginning=story_beginning,
            completion=_strip_thinking(completion),
        )

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
            temperature=self.judge_temperature,
            max_tokens=self.judge_max_new_tokens,
            num_completions=1,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        result = self._auto_client.make_request(request)
        if not result.success:
            raise RuntimeError(f"Judge call failed for model {model_name}")
        return result.completions[0].text.strip()

    def _score_completion(self, model_name: str, story_beginning: str, completion: str) -> Optional[Dict[str, float]]:
        for _ in range(self.max_retries):
            parsed = _extract_json_dict(self._call_judge(model_name, story_beginning, completion))
            if parsed is None:
                continue

            grammar = parsed.get("grammar")
            creativity = parsed.get("creativity")
            consistency = parsed.get("consistency")
            age_group = str(parsed.get("age_group", "")).strip().upper()
            if not all(isinstance(value, int) and 1 <= value <= 10 for value in (grammar, creativity, consistency)):
                continue
            if age_group not in _AGE_GROUP_TO_ORDINAL:
                continue

            return {
                "grammar": float(grammar),
                "creativity": float(creativity),
                "consistency": float(consistency),
                "age_group_ordinal": _AGE_GROUP_TO_ORDINAL[age_group],
            }
        return None

    def annotate(self, request_state: RequestState) -> Dict[str, Any]:
        assert request_state.result is not None

        story_beginning = request_state.instance.extra_data["story_beginning"]
        evaluations: List[Dict[str, float]] = []

        for completion in request_state.result.completions:
            result = None
            try:
                result = self._score_completion(self.judge_model_name, story_beginning, completion.text)
            except Exception:
                result = None

            if result is None and self.judge_model_name != BACKUP_JUDGE_MODEL:
                try:
                    result = self._score_completion(BACKUP_JUDGE_MODEL, story_beginning, completion.text)
                except Exception:
                    result = None

            if result is not None:
                evaluations.append(result)

        valid_count = len(evaluations)
        total_count = len(request_state.result.completions)
        valid_rate = valid_count / total_count if total_count else 0.0

        def _average(metric_name: str) -> float:
            # When zero completions parsed, return -100 sentinel (not 0.0):
            # 1.0 is the rubric minimum, so 0.0 is technically out-of-range
            # but would still pollute downstream means as a low-but-numeric
            # rating. -100 marks "no rating available" per project convention.
            if not evaluations:
                return -100.0
            return sum(item[metric_name] for item in evaluations) / len(evaluations)

        return {
            "tinystories_grammar_score": _average("grammar"),
            "tinystories_creativity_score": _average("creativity"),
            "tinystories_consistency_score": _average("consistency"),
            "tinystories_age_group_ordinal": _average("age_group_ordinal"),
            "tinystories_valid_judge_rate": valid_rate,
            "tinystories_scored_completion_count": float(valid_count),
        }
