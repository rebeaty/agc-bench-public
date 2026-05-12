"""Arena-Hard pairwise annotator."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.request import Request

from llm_judge.generic_llm_judge_annotator import (
    BACKUP_JUDGE_MODEL,
    _JUDGE_OVERRIDE,
    _call_openrouter_direct,
)

_OG_ARENA_HARD_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants "
    "to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. "
    "Your job is to evaluate which assistant's answer is better.\n\n"
    "Begin your evaluation by generating your own answer to the prompt. You must provide your answers before "
    "judging any answers.\n\n"
    "When evaluating the assistants' answers, compare both assistants' answers with your answer. You must "
    "identify and correct any mistakes or inaccurate information.\n\n"
    "Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer "
    "correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or "
    "more than one interpretation, it is more helpful and appropriate to ask for clarifications or more "
    "information from the user than providing an answer based on assumptions. Relevant means all parts of the "
    "response closely connect or are appropriate to what is being asked. Concise means the response is clear "
    "and not verbose or excessive.\n\n"
    "Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any "
    "missing important information in the assistants' answers that would be beneficial to include when "
    "responding to the user prompt.\n\n"
    "After providing your explanation, you must output only one of the following choices as your final verdict "
    "with a label:\n\n"
    "1. Assistant A is significantly better: [[A>>B]]\n"
    "2. Assistant A is slightly better: [[A>B]]\n"
    "3. Tie, relatively the same: [[A=B]]\n"
    "4. Assistant B is slightly better: [[B>A]]\n"
    "5. Assistant B is significantly better: [[B>>A]]\n\n"
    'Example output: "My final verdict is tie: [[A=B]]".'
)

_CREATIVE_WRITING_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants "
    "to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. "
    "Your job is to evaluate which assistant's answer is better.\n\n"
    "When evaluating the assistants' answers, compare both assistants' answers. You must identify and correct "
    "any mistakes or inaccurate information.\n\n"
    "Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer "
    "correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or "
    "more than one interpretation, it is more helpful and appropriate to ask for clarifications or more "
    "information from the user than providing an answer based on assumptions. Relevant means all parts of the "
    "response closely connect or are appropriate to what is being asked. Concise means the response is clear "
    "and not verbose or excessive.\n\n"
    "Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any "
    "missing important information in the assistants' answers that would be beneficial to include when "
    "responding to the user prompt.\n\n"
    "After providing your explanation, you must output only one of the following choices as your final verdict "
    "with a label:\n\n"
    "1. Assistant A is significantly better: [[A>>B]]\n"
    "2. Assistant A is slightly better: [[A>B]]\n"
    "3. Tie, relatively the same: [[A=B]]\n"
    "4. Assistant B is slightly better: [[B>A]]\n"
    "5. Assistant B is significantly better: [[B>>A]]\n\n"
    'Example output: "My final verdict is tie: [[A=B]]".'
)

_PROMPT_TEMPLATE = (
    "<|User Prompt|>\n{question}\n\n"
    "<|The Start of Assistant A's Answer|>\n{answer_a}\n"
    "<|The End of Assistant A's Answer|>\n\n"
    "<|The Start of Assistant B's Answer|>\n{answer_b}\n"
    "<|The End of Assistant B's Answer|>"
)

_REGEX_PATTERNS = (r"\[\[([AB<>=]+)\]\]", r"\[([AB<>=]+)\]")
_LABEL_TO_SCORES = {
    "A>B": [1.0],
    "A>>B": [1.0, 1.0, 1.0],
    "A=B": [0.5],
    "A<<B": [0.0, 0.0, 0.0],
    "A<B": [0.0],
    "B>A": [0.0],
    "B>>A": [0.0, 0.0, 0.0],
    "B=A": [0.5],
    "B<<A": [1.0, 1.0, 1.0],
    "B<A": [1.0],
}


def _extract_verdict(text: str) -> Optional[str]:
    upper = (text or "").upper()
    for pattern in _REGEX_PATTERNS:
        matches = [match for match in re.findall(pattern, upper) if match]
        if matches:
            return matches[-1].strip()
    return None


def _invert_scores(scores: List[float]) -> List[float]:
    return [1.0 - score for score in scores]


def _system_prompt_for_category(category: str) -> str:
    if category == "creative_writing":
        return _CREATIVE_WRITING_PROMPT
    return _OG_ARENA_HARD_PROMPT


class ArenaHardPairwiseAnnotator(Annotator):
    """Two-round Arena-Hard pairwise judge."""

    def __init__(
        self,
        auto_client: AutoClient,
        judge_model_name: str,
        judge_temperature: float,
        judge_max_new_tokens: int,
        category: str = "creative_writing",
        **_: Any,
    ):
        self._auto_client = auto_client
        self.judge_model_name = judge_model_name
        self.judge_temperature = judge_temperature
        self.judge_max_new_tokens = judge_max_new_tokens
        self.category = category
        self.name = "arena_hard_pairwise"

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

    def _judge_once(self, question: str, answer_a: str, answer_b: str) -> tuple[Optional[str], str, str]:
        user_prompt = _PROMPT_TEMPLATE.format(
            question=question,
            answer_a=answer_a,
            answer_b=answer_b,
        )
        full_prompt = f"{_system_prompt_for_category(self.category)}\n\n{user_prompt}"

        judge_model = self.judge_model_name
        try:
            raw_output = self._call_judge(judge_model, full_prompt)
        except Exception:
            judge_model = BACKUP_JUDGE_MODEL
            try:
                raw_output = self._call_judge(judge_model, full_prompt)
            except Exception:
                # Both judge calls failed (typical when AGC-Judge is the
                # override AND it can't handle the prompt). Record an empty
                # verdict so the metric flags parse-failure rather than
                # crashing the whole bench.
                raw_output = ""

        return _extract_verdict(raw_output), raw_output, judge_model

    def annotate(self, request_state: RequestState) -> Dict[str, Any]:
        assert request_state.result is not None

        question = request_state.instance.input.text
        baseline_output = request_state.instance.references[0].output.text if request_state.instance.references else ""
        candidate_output = request_state.result.completions[0].text.strip() if request_state.result.completions else ""

        round_1_label, round_1_raw, round_1_model = self._judge_once(
            question=question,
            answer_a=baseline_output,
            answer_b=candidate_output,
        )
        round_2_label, round_2_raw, round_2_model = self._judge_once(
            question=question,
            answer_a=candidate_output,
            answer_b=baseline_output,
        )

        battle_scores: List[float] = []
        if round_1_label in _LABEL_TO_SCORES and round_2_label in _LABEL_TO_SCORES:
            battle_scores = _LABEL_TO_SCORES[round_2_label] + _invert_scores(_LABEL_TO_SCORES[round_1_label])

        return {
            "category": self.category,
            "judge_model_round_1": round_1_model,
            "judge_model_round_2": round_2_model,
            "round_1_label": round_1_label,
            "round_2_label": round_2_label,
            "round_1_raw_output": round_1_raw,
            "round_2_raw_output": round_2_raw,
            "battle_scores": battle_scores,
            "parsed_both": 1.0 if battle_scores else 0.0,
        }
