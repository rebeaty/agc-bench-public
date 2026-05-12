"""Rebus Puzzle semantic-equivalence annotator."""

from __future__ import annotations

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
from metrics.rebus_puzzle_metric import _extract_answer_field


_DEFAULT_JUDGE_MODEL = "google/gemini-2.5-flash-lite"
_YES_RE = re.compile(r"\b(yes|equivalent|correct)\b", re.IGNORECASE)
_NO_RE = re.compile(r"\b(no|not equivalent|incorrect)\b", re.IGNORECASE)

_SYSTEM_PROMPT = (
    "You are evaluating whether two rebus-puzzle answers should be treated as "
    "the same solution. Consider paraphrases, inflection, and obvious synonymy, "
    "but reject different phrases or different idioms."
)

_USER_PROMPT = """
Determine whether the predicted rebus answer should be accepted as semantically
equivalent to the gold answer.

Gold answer: {gold_answer}
Predicted answer: {predicted_answer}

Return exactly one word:
YES
or
NO
""".strip()


def _parse_yes_no(text: str) -> Optional[float]:
    cleaned = text.strip()
    if _YES_RE.search(cleaned):
        return 1.0
    if _NO_RE.search(cleaned):
        return 0.0
    return None


class RebusPuzzleAnnotator(Annotator):
    """Run a binary semantic-equivalence judge on parsed rebus answers."""

    def __init__(
        self,
        auto_client: AutoClient,
        judge_model_name: str = _DEFAULT_JUDGE_MODEL,
        judge_temperature: float = 0.0,
        judge_max_new_tokens: int = 16,
        max_retries: int = 3,
    ):
        self._auto_client = auto_client
        self.judge_model_name = judge_model_name
        self.judge_temperature = judge_temperature
        self.judge_max_new_tokens = judge_max_new_tokens
        self.max_retries = max_retries
        self.name = "rebus_puzzle_judge"

    def _call_judge(self, model_name: str, gold_answer: str, predicted_answer: str) -> str:
        prompt = _USER_PROMPT.format(
            gold_answer=gold_answer,
            predicted_answer=predicted_answer,
        )

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

    def _score_with_model(self, model_name: str, gold_answer: str, predicted_answer: str) -> Optional[float]:
        for _ in range(self.max_retries):
            raw = self._call_judge(model_name, gold_answer, predicted_answer)
            parsed = _parse_yes_no(raw)
            if parsed is not None:
                return parsed
        return None

    def annotate(self, request_state: RequestState) -> Dict[str, Any]:
        assert request_state.result is not None

        completion = request_state.result.completions[0].text
        predicted_answer, parsed = _extract_answer_field(completion)
        gold_answer = request_state.instance.references[0].output.text if request_state.instance.references else ""

        # No answer parsed from the model output is a legitimate "no" — the
        # model produced no answer field, so it cannot match the gold. This
        # is distinct from a judge-call failure below (which returns -100).
        if not parsed or not predicted_answer:
            return {"rebus_semantic_equivalence": 0.0}

        score = None
        try:
            score = self._score_with_model(self.judge_model_name, gold_answer, predicted_answer)
        except Exception:
            score = None

        if score is None and self.judge_model_name != BACKUP_JUDGE_MODEL:
            try:
                score = self._score_with_model(BACKUP_JUDGE_MODEL, gold_answer, predicted_answer)
            except Exception:
                score = None

        # Judge-call/parse failure → -100 sentinel. 0.0 is a valid score
        # (judge said NO) so we cannot use it to mean "judge failed".
        return {"rebus_semantic_equivalence": float(score) if score is not None else -100.0}
