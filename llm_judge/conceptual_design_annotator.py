"""Conceptual design judge aligned with the paper's feasibility/novelty/usefulness axes."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.request import Request

from llm_judge.generic_llm_judge_annotator import (
    BACKUP_JUDGE_MODEL,
    _JUDGE_OVERRIDE,
    _call_openrouter_direct,
)

_JUDGE_MODEL_OVERRIDE = os.environ.get("CONCEPTUAL_DESIGN_JUDGE_MODEL_OVERRIDE", "").strip() or None


def _extract_json(text: str) -> Dict[str, Optional[int]]:
    cleaned = (text or "").strip()
    parsed: Dict[str, Optional[int]] = {
        "feasibility": None,
        "novelty": None,
        "usefulness": None,
    }
    if not cleaned:
        return parsed

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(cleaned[start : end + 1])
        except Exception:
            payload = {}
        for key in parsed:
            value = payload.get(key)
            if isinstance(value, (int, float)) and 0 <= int(value) <= 2:
                parsed[key] = int(value)
    return parsed


class ConceptualDesignAnnotator(Annotator):
    """Score the repaired conceptual-design surface on the paper's three expert dimensions."""

    def __init__(
        self,
        auto_client: AutoClient,
        judge_model_name: str,
        judge_temperature: float,
        judge_max_new_tokens: int,
        rubric: str,
        **_: Any,
    ):
        self._auto_client = auto_client
        self.judge_model_name = _JUDGE_MODEL_OVERRIDE or judge_model_name
        self.judge_temperature = judge_temperature
        self.judge_max_new_tokens = judge_max_new_tokens
        self.rubric = rubric
        self.name = "conceptual_design_judge"

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
        prompt = (
            f"{self.rubric}\n\n"
            f"Problem:\n{request_state.instance.input.text}\n\n"
            f"Generated solutions:\n{completion}\n"
        )

        judge_model = self.judge_model_name
        try:
            raw_output = self._call_judge(judge_model, prompt)
        except Exception:
            judge_model = BACKUP_JUDGE_MODEL
            raw_output = self._call_judge(judge_model, prompt)

        scores = _extract_json(raw_output)
        valid = all(value is not None for value in scores.values())
        return {
            "judge_model": judge_model,
            "raw_output": raw_output,
            "feasibility": scores["feasibility"],
            "novelty": scores["novelty"],
            "usefulness": scores["usefulness"],
            "parse_rate": 1.0 if valid else 0.0,
        }
