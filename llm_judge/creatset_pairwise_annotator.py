"""Pairwise creativity annotator for CreataSet."""

from __future__ import annotations

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
)

_PAIRWISE_CREATIVITY_PROMPT = """\
你是一个语言创意评估专家。请比较同一条指令下的两个回复，重点考虑：
1. 原创性：是否有独特视角或新想法。
2. 出乎意料性：是否带来惊喜或新鲜感。
3. 价值性：是否有意义、有深度，能启发读者。

你必须只输出“更有创意的回复是：Response 1”或“更有创意的回复是：Response 2”，不要输出其他内容。
"""

_PROMPT_TEMPLATE = """\
[指令]
{instruction}

[Response 1]
{response_1}

[Response 2]
{response_2}

更有创意的回复是：
"""


def _coerce_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _select_baseline(extra_data: Dict[str, Any]) -> Tuple[str, str]:
    baseline_text = _coerce_text(extra_data.get("gen_resp_2"))
    if baseline_text:
        return "gen_resp_2", baseline_text

    fallback_text = _coerce_text(extra_data.get("output"))
    if fallback_text:
        return "output", fallback_text

    return "gen_resp_2", ""


def _parse_verdict(text: str) -> Optional[int]:
    cleaned = " ".join((text or "").strip().split()).upper()
    if not cleaned:
        return None
    matches = re.findall(r"RESPONSE\s*([12])", cleaned)
    if matches:
        return int(matches[-1])
    return None


class CreatSetPairwiseAnnotator(Annotator):
    """Pairwise judge that compares the generated response against a released baseline."""

    def __init__(
        self,
        auto_client: AutoClient,
        judge_model_name: str,
        judge_temperature: float,
        judge_max_new_tokens: int,
        **_: Any,
    ):
        self._auto_client = auto_client
        self.judge_model_name = judge_model_name
        self.judge_temperature = judge_temperature
        self.judge_max_new_tokens = judge_max_new_tokens
        self.name = "creatset_pairwise"

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

        extra_data = request_state.instance.extra_data or {}
        baseline_field, baseline_text = _select_baseline(extra_data)
        candidate_text = request_state.result.completions[0].text.strip() if request_state.result.completions else ""
        instruction = request_state.instance.input.text

        prompt = (
            f"{_PAIRWISE_CREATIVITY_PROMPT}\n"
            f"{_PROMPT_TEMPLATE.format(instruction=instruction, response_1=candidate_text, response_2=baseline_text)}"
        )

        judge_model = self.judge_model_name
        try:
            raw_output = self._call_judge(judge_model, prompt)
        except Exception:
            judge_model = BACKUP_JUDGE_MODEL
            try:
                raw_output = self._call_judge(judge_model, prompt)
            except Exception:
                # Both judge calls failed; record parse-failure rather than crash.
                raw_output = ""

        verdict = _parse_verdict(raw_output)
        return {
            "baseline_field": baseline_field,
            "judge_model": judge_model,
            "raw_output": raw_output,
            "verdict": verdict,
            "parsed": 1.0 if verdict in (1, 2) else 0.0,
        }
