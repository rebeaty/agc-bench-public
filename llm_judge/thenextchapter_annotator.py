"""Benchmark-specific no-reference judge for The Next Chapter."""

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
    _infer_rubric_range,
    _parse_score_robust,
)

_BENCHMARK_JUDGE_OVERRIDE = os.environ.get("THENEXTCHAPTER_JUDGE_MODEL_OVERRIDE", "").strip() or None


class TheNextChapterJudgeAnnotator(Annotator):
    """Judge one The Next Chapter quality dimension without exposing references."""

    def __init__(
        self,
        auto_client: AutoClient,
        judge_model_name: str,
        judge_temperature: float,
        judge_max_new_tokens: int,
        metric_name: str,
        rubric: str,
    ):
        self._auto_client = auto_client
        self.judge_model_name = _BENCHMARK_JUDGE_OVERRIDE or judge_model_name
        self.judge_temperature = judge_temperature
        self.judge_max_new_tokens = judge_max_new_tokens
        self.metric_name = metric_name
        self.rubric = rubric
        self.name = f"generic_llm_judge_{metric_name}"
        self._valid_lo, self._valid_hi = _infer_rubric_range(rubric)

    def _call_judge(self, model_name: str, prompt: str) -> int:
        if _JUDGE_OVERRIDE:
            score_text = _call_openrouter_direct(
                _JUDGE_OVERRIDE, prompt, self.judge_temperature, self.judge_max_new_tokens
            ).strip()
        else:
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
            score_text = result.completions[0].text.strip()
        return _parse_score_robust(score_text, self._valid_lo, self._valid_hi)

    def annotate(self, request_state: RequestState) -> Dict[str, Any]:
        assert request_state.result is not None

        completion = request_state.result.completions[0].text.strip()
        condition = ""
        subset = ""
        if request_state.instance.extra_data:
            condition = str(request_state.instance.extra_data.get("condition", "")).strip()
            subset = str(request_state.instance.extra_data.get("subset", "")).strip()
        if not condition:
            condition = request_state.instance.input.text.strip()

        prompt = f"{self.rubric}\n\n"
        if subset:
            prompt += f"Subset: {subset}\n\n"
        prompt += (
            f"Condition:\n{condition}\n\n"
            f"Generated continuation:\n{completion}\n\n"
            "Return only one integer score from 1 to 5."
        )

        try:
            score = self._call_judge(self.judge_model_name, prompt)
        except Exception:
            try:
                score = self._call_judge(BACKUP_JUDGE_MODEL, prompt)
            except Exception:
                score = -100

        return {self.metric_name: score}
