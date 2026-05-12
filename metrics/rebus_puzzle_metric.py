"""Rebus Puzzle exact-answer metric."""

from __future__ import annotations

import json
import re
from typing import List, Optional

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")


def _strip_thinking(text: str) -> str:
    marker = "</think>"
    if marker not in text:
        return text.strip()
    return text.split(marker, 1)[1].strip()


def _extract_answer_field(text: str) -> tuple[str, bool]:
    cleaned = _strip_thinking(text)

    candidate = None
    match = _JSON_BLOCK_RE.search(cleaned)
    if match:
        candidate = match.group(0)
    elif cleaned.startswith("{") and cleaned.endswith("}"):
        candidate = cleaned

    if candidate is None:
        return "", False

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return "", False

    answer = parsed.get("answer")
    return (answer.strip(), True) if isinstance(answer, str) else ("", False)


def _normalize_answer(text: str) -> str:
    lowered = text.lower().strip()
    lowered = lowered.strip("`*_\"' .,:;!?")
    return _WHITESPACE_RE.sub(" ", lowered).strip()


class RebusPuzzleMetric(Metric):
    """Score normalized exact match on the parsed JSON answer field."""

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

        predicted_answer, parsed = _extract_answer_field(completion)
        exact = float(
            parsed and bool(predicted_answer) and _normalize_answer(predicted_answer) == _normalize_answer(gold)
        )

        return [
            Stat(MetricName("rebus_answer_accuracy")).add(exact),
            Stat(MetricName("rebus_answer_parse_rate")).add(float(parsed)),
        ]
