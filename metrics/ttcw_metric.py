"""TTCW evaluator metrics for expert-agreement prediction."""

from __future__ import annotations

import re
from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


_YES_NO_RE = re.compile(r"\b(Yes|No)\b", re.IGNORECASE)


def _extract_verdict(text: str) -> str | None:
    match = _YES_NO_RE.search(text)
    if match is None:
        return None
    return "Yes" if match.group(1).lower() == "yes" else "No"


class TTCWMetric(Metric):
    """Score majority-label accuracy and agreement against expert votes."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None

        prediction = request_state.result.completions[0].text.strip()
        parsed_verdict = _extract_verdict(prediction)
        majority_label = request_state.instance.extra_data["majority_label"]
        expert_labels: List[str] = request_state.instance.extra_data["expert_labels"]

        valid_binary_response = 1.0 if parsed_verdict is not None else 0.0
        majority_accuracy = 1.0 if parsed_verdict == majority_label else 0.0
        expert_vote_agreement = 0.0
        if parsed_verdict is not None and expert_labels:
            expert_vote_agreement = sum(1 for label in expert_labels if label == parsed_verdict) / len(expert_labels)

        return [
            Stat(MetricName("valid_binary_response")).add(valid_binary_response),
            Stat(MetricName("majority_accuracy")).add(majority_accuracy),
            Stat(MetricName("expert_vote_agreement")).add(expert_vote_agreement),
        ]
