"""Pron vs Prompt benchmark-specific annotator."""

from __future__ import annotations

import json
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

_JUDGE_MODEL_OVERRIDE = os.environ.get("PRON_VS_PROMPT_JUDGE_MODEL_OVERRIDE", "").strip() or None

_PROMPT_TEMPLATE = """You are an expert literature critic. Evaluate the following creative synopsis for an imaginary movie title using the released Pron vs Prompt literary rubric.

Title: {title}
Synopsis: {synopsis}

Return JSON only with integer scores using these exact keys:
- title_attractiveness: 0-3
- style_attractiveness: 0-3
- theme_attractiveness: 0-3
- title_originality: 0-3
- style_originality: 0-3
- plot_originality: 0-3
- relevance: 0-4
- title_creativity: 0-3
- synopsis_creativity: 0-3
- anthology: 0-3
- readers_opinion: 0-3
- critics_opinion: 0-3
- own_voice: 0-3

Guidance:
- Attractiveness: how engaging the title, style, and story/characters are as literary objects.
- Originality: how surprising and non-cliched the title, style, and plot are.
- Relevance: how well the synopsis uses the title as a creative starting point.
- Creativity: overall creativity of the title and synopsis.
- Criticism block: how likely the text is anthology-worthy, aligned with readers and critics, and indicative of a recognizable voice.

Use only integers in the allowed ranges."""

_EXPECTED_RANGES = {
    "title_attractiveness": (0, 3),
    "style_attractiveness": (0, 3),
    "theme_attractiveness": (0, 3),
    "title_originality": (0, 3),
    "style_originality": (0, 3),
    "plot_originality": (0, 3),
    "relevance": (0, 4),
    "title_creativity": (0, 3),
    "synopsis_creativity": (0, 3),
    "anthology": (0, 3),
    "readers_opinion": (0, 3),
    "critics_opinion": (0, 3),
    "own_voice": (0, 3),
}


def _extract_json_blob(text: str) -> str:
    stripped = (text or "").strip()
    stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
    stripped = re.sub(r"```$", "", stripped).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return stripped[start : end + 1]
    return stripped


def _parse_response(text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(_extract_json_blob(text))
    except json.JSONDecodeError:
        parsed = {}
    result: Dict[str, Any] = {}
    ok = True
    for key, (low, high) in _EXPECTED_RANGES.items():
        value = parsed.get(key)
        if isinstance(value, bool):
            ok = False
            continue
        try:
            ivalue = int(value)
        except (TypeError, ValueError):
            ok = False
            continue
        if not (low <= ivalue <= high):
            ok = False
            continue
        result[key] = ivalue
    result["pron_vs_prompt_parse_rate"] = 1.0 if ok and len(result) == len(_EXPECTED_RANGES) else 0.0
    return result


class PronVsPromptAnnotator(Annotator):
    """Judge Pron vs Prompt outputs with the released literary rubric."""

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
        self.name = "pron_vs_prompt_literary_judge"

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
        prompt = _PROMPT_TEMPLATE.format(title=extra_data.get("title", ""), synopsis=completion)

        judge_model = self.judge_model_name
        try:
            raw_judge_text = self._call_judge(judge_model, prompt)
        except Exception:
            judge_model = BACKUP_JUDGE_MODEL
            raw_judge_text = self._call_judge(judge_model, prompt)

        parsed = _parse_response(raw_judge_text)
        parsed["judge_model"] = judge_model
        parsed["raw_judge_text"] = raw_judge_text
        return parsed
