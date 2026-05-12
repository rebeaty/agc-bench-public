"""Benchmark-specific metric for CHIMERA recombination extraction."""

from __future__ import annotations

import json
import re
from typing import List, Tuple

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


_ANSWER_BLOCK_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(text: str) -> Tuple[dict, bool]:
    """Extract the first JSON object from tagged or free-form model output."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    candidate = None
    answer_match = _ANSWER_BLOCK_RE.search(text)
    if answer_match:
        candidate = answer_match.group(1).strip()
    else:
        json_match = _JSON_BLOCK_RE.search(text)
        if json_match:
            candidate = json_match.group(0).strip()

    if not candidate:
        return {}, False

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return {}, False

    return (parsed, True) if isinstance(parsed, dict) else ({}, False)


class RecombinationExtractionMetric(Metric):
    """Parse `<answer>` JSON and score the binary relevance decision."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None

        completion = request_state.result.completions[0].text
        gold_text = request_state.instance.references[0].output.text if request_state.instance.references else "{}"

        predicted_json, predicted_parsed = _extract_json_object(completion)
        gold_json, gold_parsed = _extract_json_object(gold_text)

        predicted_relevant = bool(predicted_json)
        gold_relevant = bool(gold_json)

        return [
            Stat(MetricName("recombination_json_parse_rate")).add(float(predicted_parsed)),
            Stat(MetricName("recombination_classification_accuracy")).add(
                float(predicted_parsed and gold_parsed and predicted_relevant == gold_relevant)
            ),
        ]
