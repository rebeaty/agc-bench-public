"""TinyFabulist-specific JSON judge annotator mirroring the upstream evaluator."""

from __future__ import annotations

import json
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


_SYSTEM_PROMPT = (
    "You are an expert literary critic specializing in fables and moral tales. "
    "Your evaluations should be objective, consistent, and based on established "
    "literary standards. Age-appropriateness is a key consideration in your assessment. "
    "Provide your assessment in valid, properly-formatted JSON only. Do not include "
    "any text outside the JSON object. Your response must be parseable by a JSON parser "
    "with no preprocessing. Balance critical analysis with constructive feedback, "
    "focusing on both strengths and weaknesses."
)

_USER_PROMPT = """
Evaluate the following fable according to these specific criteria:

1. **Grammar & Style (1-10)**:
   • 1-3: Significant errors that impede understanding
   • 4-6: Some errors but generally readable
   • 7-10: Clean, polished writing with appropriate language and style for a fable

2. **Creativity & Originality (1-10)**:
   • 1-3: Derivative, predictable, or clichéd
   • 4-6: Contains some original elements but follows familiar patterns
   • 7-10: Fresh perspective, innovative approach while maintaining classic fable structure

3. **Moral Clarity (1-10)**:
   • 1-3: Moral absent, confused, or contradictory
   • 4-6: Moral present but underdeveloped or lacking impact
   • 7-10: Clear, meaningful moral that provides genuine insight

4. **Adherence to Prompt (1-10)**:
   • 1-3: Missing multiple required elements from the prompt
   • 4-6: Incorporates main elements but overlooks some instructions
   • 7-10: Thoroughly addresses all prompt requirements while maintaining narrative cohesion

5. **Age Group Fit**:
   Determine which age group this fable is most appropriate for based on:
   • Vocabulary complexity and sentence structure
   • Conceptual difficulty of the moral lesson
   • Story length and complexity
   • Content appropriateness

Age groups are defined as:
  - A: 3 years or under
  - B: 4-7 years
  - C: 8-11 years
  - D: 12-15 years
  - E: 16 years or above

Format your response as valid JSON with this structure:
{{
    "type": "Fable Evaluation",
    "grammar": <integer 1-10>,
    "creativity": <integer 1-10>,
    "moral_clarity": <integer 1-10>,
    "adherence_to_prompt": <integer 1-10>,
    "best_age_group": "<letter: A, B, C, D, or E>",
    "explanation": [
        "<One sentence explaining grammar & style score>",
        "<One sentence explaining creativity & originality score>",
        "<One sentence explaining moral clarity score>",
        "<One sentence explaining adherence to prompt score>",
        "<One sentence explaining why this fable best fits the chosen age group>"
    ]
}}

Be critical but fair. Ensure your entire evaluation is concise yet informative.

Original Prompt:
{prompt}

Fable:
{fable}
""".strip()

_AGE_GROUPS = ("A", "B", "C", "D", "E")


def _strip_thinking(text: str) -> str:
    marker = "</think>"
    return text.split(marker, 1)[1].strip() if marker in text else text.strip()


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


def _extract_score(parsed: Dict[str, Any], key: str) -> Optional[float]:
    value = parsed.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not 1.0 <= numeric <= 10.0:
        return None
    return numeric


class TinyFabulistAnnotator(Annotator):
    """Call the upstream-style TinyFabulist judge once and parse all dimensions."""

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
        self.name = "tinyfabulist_judge"

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

    def _score_with_model(self, model_name: str, prompt: str) -> Optional[Dict[str, Any]]:
        for _ in range(self.max_retries):
            raw = self._call_judge(model_name, prompt)
            parsed = _extract_json_dict(raw)
            if parsed is None:
                continue

            grammar = _extract_score(parsed, "grammar")
            creativity = _extract_score(parsed, "creativity")
            moral_clarity = _extract_score(parsed, "moral_clarity")
            adherence = _extract_score(parsed, "adherence_to_prompt")
            age_group = str(parsed.get("best_age_group", "")).strip().upper()

            if None in {grammar, creativity, moral_clarity, adherence}:
                continue
            if age_group not in _AGE_GROUPS:
                continue

            mean_score = (grammar + creativity + moral_clarity + adherence) / 4.0
            result: Dict[str, Any] = {
                "grammar_score": grammar,
                "creativity_score": creativity,
                "moral_clarity_score": moral_clarity,
                "adherence_to_prompt_score": adherence,
                "mean_judge_score": mean_score,
                "valid_judge_rate": 1.0,
            }
            for label in _AGE_GROUPS:
                result[f"age_group_{label.lower()}_rate"] = 1.0 if age_group == label else 0.0
            return result
        return None

    def annotate(self, request_state: RequestState) -> Dict[str, Any]:
        assert request_state.result is not None

        completion = _strip_thinking(request_state.result.completions[0].text)
        prompt_text = str((request_state.instance.extra_data or {}).get("prompt", request_state.instance.input.text))
        judge_prompt = _USER_PROMPT.format(prompt=prompt_text, fable=completion)

        result = None
        try:
            result = self._score_with_model(self.judge_model_name, judge_prompt)
        except Exception:
            result = None

        if result is None and self.judge_model_name != BACKUP_JUDGE_MODEL:
            try:
                result = self._score_with_model(BACKUP_JUDGE_MODEL, judge_prompt)
            except Exception:
                result = None

        if result is None:
            # Judge-call/parse failure → -100 sentinel for every score field.
            # 0.0 is below the rubric's valid 1-10 range but inside the
            # numeric domain and would be silently averaged into means; -100
            # is the project-wide convention for "rating unavailable" so it
            # can be filtered downstream. valid_judge_rate stays 0.0 since
            # it is genuinely a 0-1 indicator, not a rubric score.
            result = {
                "grammar_score": -100.0,
                "creativity_score": -100.0,
                "moral_clarity_score": -100.0,
                "adherence_to_prompt_score": -100.0,
                "mean_judge_score": -100.0,
                "valid_judge_rate": 0.0,
            }
            for label in _AGE_GROUPS:
                result[f"age_group_{label.lower()}_rate"] = -100.0

        return result
