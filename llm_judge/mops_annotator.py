"""MoPS-specific quality judge annotator."""

import os
import re
from typing import Any, Dict, Optional, Tuple

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.request import Request

from llm_judge.generic_llm_judge_annotator import (
    BACKUP_JUDGE_MODEL,
    _JUDGE_OVERRIDE,
    _call_openrouter_direct,
    _parse_score_robust,
)


_JUDGE_MODEL_OVERRIDE = os.environ.get("MOPS_JUDGE_MODEL_OVERRIDE", "").strip() or None

_PROMPTS = {
    "fascination_score": """Here is a story premise:

{premise}

Now let's give you a score from 0 to 100 to assess to its fascination.

Score 0 indicates that this premise is completely confused, while score 100 indicates that you really want to see the story created based on this premise.

Requirement: just provide a deterministic score and provide a concise and brief explanation, with a blank line between the two.

Score:""",
    "completeness_score": """Here is a story premise:

{premise}

Now let's give you a score from 0 to 100 which represents its completeness level.

Score 0 indicates that it lacks all elements , while score 100 indicates that it has all elements.

Requirement: just provide a deterministic score and provide a concise and brief explanation, with a blank line between the two.

Score:""",
    "originality_score": """Here is a story premise:

{premise}

Now let you give a score from 0 to 100 which represents your level of familiarity with it.

Score 0 indicates that you have seen the exact same premise, while score 100 indicates that you have never seen the same premise at all.

Your score should be based on the assumption that the candidate is at least a complete story premise. Otherwise, you should give a score 0.

Requirement: just provide a deterministic score and provide a concise and brief explanation, with a blank line between the two.

Score:""",
}


def _strip_thinking(text: str) -> str:
    marker = "</think>"
    marker_pos = text.find(marker)
    if marker_pos != -1:
        return text[marker_pos + len(marker) :].strip()
    return text.strip()


def _parse_score_and_explanation(text: str) -> Optional[Tuple[int, str]]:
    cleaned = _strip_thinking(text)
    # The MoPS prompt asks for the score first, then a blank line, then the
    # explanation. Use the robust parser so a chatty rater that prepends
    # reasoning ("Looking at the 25 elements...") doesn't silently grab the
    # wrong integer. Range is the prompt-enforced 0-100 scale.
    score = _parse_score_robust(cleaned, 0, 100)
    if score == -100:
        return None

    # Find where in the cleaned text the parsed score actually appears so we
    # can keep the trailing explanation. Fall back to the full reply if the
    # parser pulled the score from a labeled pattern we can't easily relocate.
    match = re.search(rf"\b{score}\b", cleaned)
    explanation = cleaned[match.end() :].strip(" \n:-") if match else cleaned
    return score, explanation


class MoPSAnnotator(Annotator):
    """Judge MoPS premises on the three paper dimensions without reference leakage."""

    def __init__(
        self,
        auto_client: AutoClient,
        judge_model_name: str,
        judge_temperature: float,
        judge_max_new_tokens: int,
        max_retries: int = 3,
    ):
        self._auto_client = auto_client
        self.judge_model_name = _JUDGE_MODEL_OVERRIDE or judge_model_name
        self.judge_temperature = judge_temperature
        self.judge_max_new_tokens = judge_max_new_tokens
        self.max_retries = max_retries
        self.name = "mops_judge"

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
        return result.completions[0].text.strip()

    def _score_dimension(self, model_name: str, premise: str, metric_name: str) -> Optional[Tuple[int, str]]:
        prompt = _PROMPTS[metric_name].format(premise=_strip_thinking(premise))
        for _ in range(self.max_retries):
            parsed = _parse_score_and_explanation(self._call_judge(model_name, prompt))
            if parsed is not None:
                return parsed
        return None

    def annotate(self, request_state: RequestState) -> Dict[str, Any]:
        assert request_state.result is not None
        premise = request_state.result.completions[0].text.strip()

        if not premise:
            # Empty completion: emit -100 sentinel rather than 0 so downstream
            # aggregation can distinguish "judge unrated" from a real score of 0.
            return {
                "fascination_score": -100.0,
                "completeness_score": -100.0,
                "originality_score": -100.0,
                "mops_quality_score": -100.0,
                "mops_valid_judge_rate": 0.0,
                "mops_judge_reasons": {},
            }

        scores: Dict[str, float] = {}
        reasons: Dict[str, str] = {}
        valid_count = 0
        for metric_name in ("fascination_score", "completeness_score", "originality_score"):
            result = None
            try:
                result = self._score_dimension(self.judge_model_name, premise, metric_name)
            except Exception:
                result = None

            if result is None and self.judge_model_name != BACKUP_JUDGE_MODEL:
                try:
                    result = self._score_dimension(BACKUP_JUDGE_MODEL, premise, metric_name)
                except Exception:
                    result = None

            if result is None:
                # Use -100 sentinel rather than 0.0; the MoPS rubric defines 0
                # as "completely confused" / "lacks all elements", so a silent
                # 0 on parse failure would be indistinguishable from a real
                # bottom-of-scale rating and would systematically drag the
                # mean down.
                scores[metric_name] = -100.0
                reasons[metric_name] = ""
                continue

            score, explanation = result
            scores[metric_name] = float(score)
            reasons[metric_name] = explanation
            valid_count += 1

        # Quality score is only meaningful when all three sub-scores parsed;
        # otherwise mixing real scores with -100 sentinels would produce a
        # garbage average. Emit -100 so the downstream metric aggregator can
        # filter the sentinel out cleanly.
        if valid_count == 3:
            quality_score = (
                sum(scores[name] for name in ("fascination_score", "completeness_score", "originality_score")) / 3.0
            )
        else:
            quality_score = -100.0

        return {
            **scores,
            "mops_quality_score": quality_score,
            "mops_valid_judge_rate": valid_count / 3.0,
            "mops_judge_reasons": reasons,
        }
