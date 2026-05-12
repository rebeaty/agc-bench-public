"""Benchmark-specific annotator for CPers."""

from __future__ import annotations

import json
import os
import re
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

_JUDGE_MODEL_OVERRIDE = os.environ.get("CPERS_JUDGE_MODEL_OVERRIDE", "").strip() or None

_PROMPT_TEMPLATE = """You are evaluating the creativity of a Persian literary sentence using the culturally adapted TTCT-style framework from the CPers paper.

Topic: {topic}
Generated Persian text: {response}

Score each question from 1 to 5.

Originality:
1. Does the sentence demonstrate creativity and originality in expression?
2. Does the sentence avoid cliches and overused expressions?
3. Does the sentence contain at least one literary device such as simile, metaphor, hyperbole, or antithesis?

Fluency:
1. Is the sentence grammatically correct?
2. Does the sentence sound natural to Persian readers?
3. Is the sentence appropriate as a literary sentence?

Flexibility:
1. Does the sentence use multiple ideas or layers to express the topic?
2. Does the sentence look at the topic from a fresh perspective?
3. Does the sentence show stylistic or conceptual variety?

Elaboration:
1. Does the sentence use rich and diverse vocabulary?
2. Does the sentence evoke imagery?
3. Does the sentence effectively convey emotion?

Also detect whether each rhetorical device is present:
- simile
- metaphor
- hyperbole
- antithesis

Return exactly one JSON object in this schema:
{{
  "originality": {{"q1": 1, "q2": 1, "q3": 1, "average": 1.0}},
  "fluency": {{"q1": 1, "q2": 1, "q3": 1, "average": 1.0}},
  "flexibility": {{"q1": 1, "q2": 1, "q3": 1, "average": 1.0}},
  "elaboration": {{"q1": 1, "q2": 1, "q3": 1, "average": 1.0}},
  "overall_creativity": 1.0,
  "devices": {{
    "simile": 0,
    "metaphor": 0,
    "hyperbole": 0,
    "antithesis": 0
  }}
}}
"""

_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```|\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Dict[str, Any]:
    stripped = (text or "").strip()
    match = _JSON_RE.search(stripped)
    if not match:
        raise ValueError("No JSON object found")
    payload = match.group(1) if match.group(1) is not None else match.group(0)
    return json.loads(payload)


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _dimension_average(payload: Dict[str, Any], name: str) -> Optional[float]:
    block = payload.get(name)
    if not isinstance(block, dict):
        return None
    avg = _safe_float(block.get("average"))
    if avg is not None:
        return avg
    values = [_safe_float(block.get(f"q{i}")) for i in range(1, 4)]
    numeric = [value for value in values if value is not None]
    if len(numeric) != 3:
        return None
    return sum(numeric) / 3.0


class CPersAnnotator(Annotator):
    """TTCT-style Persian creativity judge with rhetorical-device outputs."""

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
        self.name = "cpers_ttct_judge"

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
            topic=extra_data.get("topic", ""),
            response=completion,
        )

        judge_model = self.judge_model_name
        try:
            raw_judge_text = self._call_judge(judge_model, prompt)
        except Exception:
            judge_model = BACKUP_JUDGE_MODEL
            raw_judge_text = self._call_judge(judge_model, prompt)

        parse_rate = 0.0
        originality = None
        fluency = None
        flexibility = None
        elaboration = None
        overall = None
        simile = None
        metaphor = None
        hyperbole = None
        antithesis = None

        try:
            payload = _extract_json(raw_judge_text)
            originality = _dimension_average(payload, "originality")
            fluency = _dimension_average(payload, "fluency")
            flexibility = _dimension_average(payload, "flexibility")
            elaboration = _dimension_average(payload, "elaboration")
            overall = _safe_float(payload.get("overall_creativity"))
            devices = payload.get("devices", {}) if isinstance(payload.get("devices"), dict) else {}
            simile = _safe_float(devices.get("simile"))
            metaphor = _safe_float(devices.get("metaphor"))
            hyperbole = _safe_float(devices.get("hyperbole"))
            antithesis = _safe_float(devices.get("antithesis"))
            if all(value is not None for value in [originality, fluency, flexibility, elaboration, overall, simile, metaphor, hyperbole, antithesis]):
                parse_rate = 1.0
        except Exception:
            pass

        return {
            "judge_model": judge_model,
            "raw_judge_text": raw_judge_text,
            "originality": originality,
            "fluency": fluency,
            "flexibility": flexibility,
            "elaboration": elaboration,
            "overall_creativity": overall,
            "simile": simile,
            "metaphor": metaphor,
            "hyperbole": hyperbole,
            "antithesis": antithesis,
            "cpers_parse_rate": parse_rate,
        }
