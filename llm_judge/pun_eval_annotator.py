"""PunEval-specific binary pun-detection judge."""

from __future__ import annotations

import ast
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

_BENCHMARK_JUDGE_OVERRIDE = os.environ.get("PUN_EVAL_JUDGE_MODEL_OVERRIDE", "").strip() or None

_DEFINITION = (
    "<*Definition*>\n"
    "Puns are a form of wordplay exploiting different meanings of a word or similar-sounding "
    "words, while non-puns are jokes or statements that don't rely on such linguistic ambiguities."
)

_INSTRUCTION = (
    "<*Instruction*>\n"
    'Determine whether the given Text is a pun. You should either say "The given text is a pun" '
    'or say "The given text is a non-pun". You must output the current status in a parsable JSON '
    'format. An example output looks like:\n{"Choice": "The given text is a XXX"}'
)


def _strip_thinking(text: str) -> str:
    marker = "</think>"
    return text.split(marker, 1)[1].strip() if marker in text else text.strip()


def _extract_json_dict(text: str) -> Optional[Dict[str, Any]]:
    cleaned = _strip_thinking(text or "")
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None

    payload = cleaned[start : end + 1]
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(payload)
        except (ValueError, SyntaxError):
            return None
    return parsed if isinstance(parsed, dict) else None


def _extract_generated_sentence(text: str) -> tuple[str, float]:
    parsed = _extract_json_dict(text)
    if parsed is not None:
        sentence = parsed.get("Sentence")
        if isinstance(sentence, str) and sentence.strip():
            return sentence.strip(), 1.0

    cleaned = _strip_thinking(text or "")
    sentence_match = re.search(r'"Sentence"\s*:\s*"([^"]+)"', cleaned)
    if sentence_match:
        return sentence_match.group(1).strip(), 1.0

    return cleaned, 0.0


def _extract_pun_choice(text: str) -> Optional[int]:
    parsed = _extract_json_dict(text)
    choice = ""
    if parsed is not None:
        choice = str(parsed.get("Choice", "")).strip().lower()
    if not choice:
        choice = _strip_thinking(text or "").lower()

    if "non-pun" in choice:
        return 0
    if "pun" in choice:
        return 1
    return None


class PunEvalPunDetectionAnnotator(Annotator):
    """Run the PunEval Notebook 6 binary pun-detection judge on each generation."""

    def __init__(
        self,
        auto_client: AutoClient,
        judge_model_name: str,
        judge_temperature: float,
        judge_max_new_tokens: int,
    ):
        self._auto_client = auto_client
        self.judge_model_name = _BENCHMARK_JUDGE_OVERRIDE or judge_model_name
        self.judge_temperature = judge_temperature
        self.judge_max_new_tokens = judge_max_new_tokens
        self.name = "pun_eval_judge"

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

    def _annotate_with_model(self, model_name: str, generated_text: str) -> Optional[Dict[str, float]]:
        prompt = (
            f"{_DEFINITION}\n\n"
            f"{_INSTRUCTION}\n\n"
            f"<*Your Response*>\n"
            f"Text: {generated_text}\n"
            "Output:"
        )
        response = self._call_judge(model_name, prompt)
        pun_choice = _extract_pun_choice(response)
        if pun_choice is None:
            return None
        return {
            "pun_detection_rate": float(pun_choice),
            "pun_detection_parsed_rate": 1.0,
        }

    def annotate(self, request_state: RequestState) -> Dict[str, Any]:
        assert request_state.result is not None

        completion = request_state.result.completions[0].text if request_state.result.completions else ""
        generated_text, json_parsed_rate = _extract_generated_sentence(completion)

        result = None
        try:
            result = self._annotate_with_model(self.judge_model_name, generated_text)
        except Exception:
            result = None

        if result is None and self.judge_model_name != BACKUP_JUDGE_MODEL:
            try:
                result = self._annotate_with_model(BACKUP_JUDGE_MODEL, generated_text)
            except Exception:
                result = None

        if result is None:
            # 0 IS a valid pun_detection_rate (means "judge classified as
            # non-pun"), so a silent default of 0.0 on parse failure would
            # systematically bias the metric toward "non-pun" whenever the
            # judge ramble couldn't be parsed. Emit -100 sentinel instead;
            # pun_detection_parsed_rate=0.0 still flags the failure.
            result = {
                "pun_detection_rate": -100.0,
                "pun_detection_parsed_rate": 0.0,
            }

        result["sentence_json_parsed_rate"] = json_parsed_rate
        return result
