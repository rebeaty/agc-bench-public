"""PuzzleWorld final-answer metric."""

from __future__ import annotations

import re
from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


_ANSWER_RE = re.compile(r"answer\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")
_LEADING_PHRASE_RE = re.compile(r"^(the\s+)?(final\s+)?answer\s+(is|=)\s+", re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    marker = "</think>"
    if marker not in text:
        return text.strip()
    return text.split(marker, 1)[1].strip()


def _extract_answer(text: str) -> tuple[str, bool]:
    cleaned = _strip_thinking(text)
    matches = list(_ANSWER_RE.finditer(cleaned))
    if matches:
        answer = matches[-1].group(1).strip()
        had_marker = True
    else:
        answer = ""
        had_marker = False

    lines = answer.splitlines()
    answer = lines[0].strip() if lines else ""
    answer = _LEADING_PHRASE_RE.sub("", answer)
    return answer.strip("`*_\"' .,:;!?"), had_marker


def _normalize(text: str) -> str:
    extracted, _ = _extract_answer(text)
    lowered = extracted.lower()
    lowered = lowered.replace("“", '"').replace("”", '"').replace("’", "'")
    lowered = lowered.strip("`*_\"' .,:;!?")
    return _WHITESPACE_RE.sub(" ", lowered).strip()


class PuzzleWorldAnswerMetric(Metric):
    """Score the extracted final PuzzleWorld answer instead of the whole trace."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None

        completion = request_state.result.completions[0].text
        gold = request_state.instance.references[0].output.text if request_state.instance.references else ""

        extracted_answer, had_marker = _extract_answer(completion)
        normalized_prediction = _normalize(completion)
        normalized_gold = _normalize(gold)

        return [
            Stat(MetricName("puzzleworld_final_answer_accuracy")).add(
                float(bool(normalized_prediction) and normalized_prediction == normalized_gold)
            ),
            Stat(MetricName("puzzleworld_answer_extracted_rate")).add(float(had_marker)),
        ]
