"""POLLUX-specific rubric-aware judge annotator."""

from __future__ import annotations

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
    _parse_score_robust,
)


_SYSTEM_PROMPT = (
    "You are a careful POLLUX judge. Score the generated response only "
    "according to the provided criterion and rubric. Use the POLLUX 0-4 "
    "scale exactly as written in the rubric. Return only the integer score."
)


class POLLUXCreativityAnnotator(Annotator):
    """Judge each POLLUX generation against its own criterion metadata."""

    def __init__(
        self,
        auto_client: AutoClient,
        judge_model_name: str,
        judge_temperature: float,
        judge_max_new_tokens: int,
    ):
        self._auto_client = auto_client
        self.judge_model_name = judge_model_name
        self.judge_temperature = judge_temperature
        self.judge_max_new_tokens = judge_max_new_tokens
        self.name = "generic_llm_judge_pollux_score"

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

    @staticmethod
    def _build_prompt(request_state: RequestState) -> str:
        instance = request_state.instance
        extra_data = instance.extra_data or {}

        instruction = instance.input.text.strip()
        criteria_name = str(extra_data.get("criteria_name", "")).strip()
        criteria_description = str(extra_data.get("criteria_description", "")).strip()
        rubrics = str(extra_data.get("rubrics", "")).strip()
        reference_text = ""
        if instance.references:
            reference_text = instance.references[0].output.text.strip()
        response = request_state.result.completions[0].text.strip()

        prompt = (
            "Evaluate the following POLLUX generation.\n\n"
            f"Instruction:\n{instruction}\n\n"
            f"Criterion name:\n{criteria_name}\n\n"
        )
        if criteria_description:
            prompt += f"Criterion description:\n{criteria_description}\n\n"
        if rubrics:
            prompt += f"Rubric:\n{rubrics}\n\n"
        if reference_text:
            prompt += f"Reference answer:\n{reference_text}\n\n"
        prompt += (
            f"Generated response:\n{response}\n\n"
            "Return only a single integer score from 0 to 4."
        )
        return prompt

    @staticmethod
    def _parse_score(text: str) -> int:
        # POLLUX rubric is fixed at 0-4 (system prompt + rubric both enforce).
        return _parse_score_robust(text, 0, 4)

    def annotate(self, request_state: RequestState) -> Dict[str, Any]:
        assert request_state.result is not None
        prompt = self._build_prompt(request_state)

        try:
            score_text = self._call_judge(self.judge_model_name, prompt)
        except Exception:
            try:
                score_text = self._call_judge(BACKUP_JUDGE_MODEL, prompt)
            except Exception:
                score_text = "0"

        return {"pollux_score": self._parse_score(score_text)}
