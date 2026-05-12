"""PoetMT-specific source-aware judge annotator."""

from __future__ import annotations

from typing import Any, Dict

from helm.benchmark.adaptation.request_state import RequestState

from llm_judge.generic_llm_judge_annotator import GenericLLMJudgeAnnotator
from metrics.poetmt_metric import normalize_poetmt_translation


class PoetMTJudgeAnnotator(GenericLLMJudgeAnnotator):
    """Mirror the paper/repo BS/BF/BM prompt shape with the source poem included."""

    def annotate(self, request_state: RequestState) -> Dict[str, Any]:
        assert request_state.result is not None
        completion = request_state.result.completions[0].text
        normalized_translation = normalize_poetmt_translation(completion)

        extra_data = request_state.instance.extra_data or {}
        source_poem = str(extra_data.get("source_poem", "")).strip()
        if not source_poem:
            source_poem = request_state.instance.input.text

        prompt = (
            f"{self.rubric.strip()}\n\n"
            f"Original Chinese poem: {source_poem}\n\n"
            f"English translation: {normalized_translation}\n\n"
            "Evaluation (score only):"
        )

        try:
            score = self._call_judge(self.judge_model_name, prompt)
        except Exception:
            try:
                score = self._call_judge("google/gemini-3-flash-preview", prompt)
            except Exception:
                score = -100

        return {self.metric_name: score}
