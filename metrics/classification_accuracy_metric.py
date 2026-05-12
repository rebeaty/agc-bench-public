"""Benchmark-specific relevance accuracy for CHIMERA recombination extraction."""

from __future__ import annotations

import json
import re
from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


_ANSWER_BLOCK_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(text: str) -> dict:
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    answer_match = _ANSWER_BLOCK_RE.search(text)
    if answer_match:
        candidate = answer_match.group(1).strip()
    else:
        json_match = _JSON_BLOCK_RE.search(text)
        candidate = json_match.group(0).strip() if json_match else text.strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return {}

    return parsed if isinstance(parsed, dict) else {}


class ClassificationAccuracyMetric(Metric):
    """Score the Level 1 empty-vs-non-empty recombination decision."""

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

        predicted_relevant = bool(_extract_json_object(completion))
        gold_relevant = bool(_extract_json_object(gold_text))

        return [Stat(MetricName("classification_accuracy")).add(float(predicted_relevant == gold_relevant))]
