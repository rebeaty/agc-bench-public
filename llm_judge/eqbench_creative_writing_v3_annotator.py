"""EQBench Creative Writing v3 rubric annotator."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.request import Request

from llm_judge.generic_llm_judge_annotator import (
    BACKUP_JUDGE_MODEL,
    _JUDGE_OVERRIDE,
    _call_openrouter_direct,
)

_JUDGE_PROMPT_TEMPLATE = """You are an expert in assessing creative writing. Your task is to score the quality of the test model's response below, by several metrics, on a 0-20 scale.

[PROMPT START]

{writing_prompt}

[PROMPT END]

[TEST MODEL RESPONSE]

{test_model_response}

[TEST MODEL RESPONSE END]

[Task]

You are an expert in assessing creative writing. Your task is to score the quality of the test model's response below, by several metrics, on a 0-20 scale.

Scoring notes:

- A scores of 20 represents a masterpiece.

- If the criteria is not relevant to the piece (e.g. criteria relating to dialogue but the piece has none), exclude it from your assessment.

- Do not add any commentary or explanation to the scores section.

- Do not penalize NSFW output; that is the default for some models. Just assess it on its merits.

- Everything within the "TEST MODEL RESPONSE" section was written by the test model. Sometimes models like to write comments on the piece after the piece is concluded; if this happens you should ignore their comments.

- In the output, write the metric names exactly as below so they can be parsed.

- Do not use markdown in your response. Use the designated output format exactly.

- You are to write a comprehensive analysis of the piece, then give your scores.

- For these criteria, lower is better:
{lower_is_better_criteria}

- You are a critic, and your job is to be critical, especially of any failings or amateurish elements.

- Output format is:

[Analysis]

Write your detailed analysis.

[Scores]

Metric 1 name: [Score 0-20]

Metric 2 name: ...

---

Now, rate the supplied model output on the following criteria:

{creative_writing_criteria}
"""

_SCORE_ONLY_PROMPT_TEMPLATE = """Score the quality of the test model's response below on the same 0-20 creative-writing rubric.

[PROMPT START]

{writing_prompt}

[PROMPT END]

[TEST MODEL RESPONSE]

{test_model_response}

[TEST MODEL RESPONSE END]

For these criteria, lower is better:
{lower_is_better_criteria}

Do not write analysis. Output exactly one line per criterion, using the criterion name exactly as written, followed by a colon and a number from 0 to 20.

{creative_writing_criteria}
"""

_CREATIVE_WRITING_CRITERIA = [
    "Adherence to Instructions",
    "Believable Character Actions",
    "Nuanced Characters",
    "Consistent Voice/Tone of Writing",
    "Imagery and Descriptive Quality",
    "Elegant Prose",
    "Emotionally Engaging",
    "Emotionally Complex",
    "Coherent",
    "Meandering",
    "Weak Dialogue",
    "Tell-Don't-Show",
    "Unsurprising or Uncreative",
    "Amateurish",
    "Purple Prose",
    "Overwrought",
    "Incongruent Ending Positivity",
    "Unearned Transformations",
    "Well-earned Lightness or Darkness",
    "Sentences Flow Naturally",
    "Overall Reader Engagement",
    "Overall Impression",
]

_NEGATIVE_CRITERIA = {
    "Unearned Transformations",
    "Incongruent Ending Positivity",
    "Overwrought",
    "Purple Prose",
    "Amateurish",
    "Unsurprising or Uncreative",
    "Tell-Don't-Show",
    "Weak Dialogue",
    "Meandering",
}

_SCORE_RANGE_MAX = 20.0
_JUDGE_MODEL_OVERRIDE = os.environ.get("EQBENCH_CREATIVE_WRITING_V3_JUDGE_MODEL_OVERRIDE", "").strip() or None


def _parse_judge_scores_creative(judge_model_response: str) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    patterns = [
        r"(.*?):\s*(?:Score\s+)?(-?\d+(?:\.\d+)?)",
        r"(.*?):\s*\[(-?\d+(?:\.\d+)?)\]",
    ]
    for pattern in patterns:
        for metric_name, score_text in re.findall(pattern, judge_model_response):
            score = float(score_text)
            # Range-validate: rubric is 0-20. Reject negatives (so the -100
            # sentinel cannot leak in) and out-of-range (year/line-number
            # spillover from chatty judges).
            if 0.0 <= score <= _SCORE_RANGE_MAX:
                scores[metric_name.strip()] = score
    return scores


def _parse_ordered_score_lines(judge_model_response: str) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    values: List[float] = []
    for line in judge_model_response.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = re.fullmatch(r"(?:[-*]\s*)?(?:\d+[.)]\s*)?(-?\d+(?:\.\d+)?)", stripped)
        if not match:
            continue
        score = float(match.group(1))
        if 0.0 <= score <= _SCORE_RANGE_MAX:
            values.append(score)
    if len(values) < len(_CREATIVE_WRITING_CRITERIA) // 2:
        return scores
    for criterion, score in zip(_CREATIVE_WRITING_CRITERIA, values):
        scores[criterion] = score
    return scores


def _invert_if_negative(metric: str, score: float) -> float:
    if metric in _NEGATIVE_CRITERIA:
        return _SCORE_RANGE_MAX - score
    return score


class EQBenchCreativeWritingV3Annotator(Annotator):
    """Upstream-rubric single-piece judge for EQBench Creative Writing v3."""

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
        self.name = "eqbench_creative_writing_v3_rubric"

    def _call_judge(self, model_name: str, prompt: str, max_tokens: int | None = None) -> str:
        max_tokens = max_tokens or self.judge_max_new_tokens
        if _JUDGE_OVERRIDE:
            return _call_openrouter_direct(
                _JUDGE_OVERRIDE,
                prompt,
                self.judge_temperature,
                max_tokens,
            ).strip()

        request = Request(
            model=model_name,
            model_deployment=model_name,
            prompt=prompt,
            temperature=self.judge_temperature,
            max_tokens=max_tokens,
            num_completions=1,
        )
        result = self._auto_client.make_request(request)
        if not result.success:
            raise RuntimeError(f"Judge call failed for model {model_name}")
        return result.completions[0].text.strip() if result.completions else ""

    def annotate(self, request_state: RequestState) -> Dict[str, Any]:
        assert request_state.result is not None
        completion = request_state.result.completions[0].text.strip() if request_state.result.completions else ""
        writing_prompt = request_state.instance.input.text

        prompt = _JUDGE_PROMPT_TEMPLATE.format(
            writing_prompt=writing_prompt,
            test_model_response=completion,
            creative_writing_criteria="\n".join(f"- {criterion}" for criterion in _CREATIVE_WRITING_CRITERIA),
            lower_is_better_criteria=", ".join(sorted(_NEGATIVE_CRITERIA)),
        )

        judge_model = self.judge_model_name
        try:
            raw_judge_text = self._call_judge(judge_model, prompt)
        except Exception:
            judge_model = BACKUP_JUDGE_MODEL
            raw_judge_text = self._call_judge(judge_model, prompt)

        scores = _parse_judge_scores_creative(raw_judge_text)
        score_only_retry_text = ""
        score_only_retry_used = 0.0
        if not scores:
            retry_prompt = _SCORE_ONLY_PROMPT_TEMPLATE.format(
                writing_prompt=writing_prompt,
                test_model_response=completion,
                creative_writing_criteria="\n".join(_CREATIVE_WRITING_CRITERIA),
                lower_is_better_criteria=", ".join(sorted(_NEGATIVE_CRITERIA)),
            )
            try:
                score_only_retry_text = self._call_judge(
                    judge_model,
                    retry_prompt,
                    max_tokens=min(self.judge_max_new_tokens, 2048),
                )
            except Exception:
                score_only_retry_text = ""
            retry_scores = _parse_judge_scores_creative(score_only_retry_text)
            if not retry_scores:
                retry_scores = _parse_ordered_score_lines(score_only_retry_text)
            if retry_scores:
                scores = retry_scores
                score_only_retry_used = 1.0
        adjusted_scores: List[float] = []
        for metric, value in scores.items():
            adjusted = _invert_if_negative(metric, value)
            if 0.0 <= adjusted <= _SCORE_RANGE_MAX:
                adjusted_scores.append(adjusted)

        # -100 sentinel when no criteria parsed: 0.0 is a valid score for
        # this rubric, so silent-zeroing the aggregate would corrupt means.
        if adjusted_scores:
            creative_score = sum(adjusted_scores) / len(adjusted_scores)
            eqbench_score = creative_score * 5.0
        else:
            creative_score = -100
            eqbench_score = -100

        return {
            "judge_model": judge_model,
            "raw_judge_text": raw_judge_text,
            "score_only_retry_text": score_only_retry_text,
            "score_only_retry_used": score_only_retry_used,
            "criteria_scores": scores,
            "criteria_count": float(len(adjusted_scores)),
            "judge_parse_rate": 1.0 if adjusted_scores else 0.0,
            "creative_score_0_20": creative_score,
            "eqbench_creative_score": eqbench_score,
        }
