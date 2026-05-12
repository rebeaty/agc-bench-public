"""Benchmark-specific metrics for sequential AidanBench."""

from __future__ import annotations

from typing import Any, Dict, List

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat


def _find_aidanbench_annotation(state: RequestState) -> Dict[str, Any]:
    annotations = state.annotations or {}
    for value in annotations.values():
        if isinstance(value, dict) and "aidanbench_stopping_score" in value:
            return value
    return {}


class AidanBenchMetric(EvaluateInstancesMetric):
    """Expose the sequential AidanBench stopping metrics."""

    def __init__(self, **_: Any):
        super().__init__()

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        eval_states = [state for state in request_states if state.request_mode != "calibration"]
        total = len(eval_states)

        response_count = 0.0
        mean_embedding_novelty = 0.0
        valid_response_rate = 0.0
        coherence_pass_rate = 0.0
        novelty_pass_rate = 0.0
        stopping_score = 0.0
        stopping_ratio = 0.0

        for state in eval_states:
            annotation = _find_aidanbench_annotation(state)
            coherence_passes = [bool(value) for value in annotation.get("aidanbench_coherence_passes", [])]
            novelty_passes = [bool(value) for value in annotation.get("aidanbench_novelty_passes", [])]
            validity_flags = [bool(value) for value in annotation.get("aidanbench_validity_flags", [])]
            expected_responses = int(annotation.get("aidanbench_expected_responses", 0) or 0)
            parsed_count = int(annotation.get("aidanbench_response_count", 0) or 0)
            instance_stopping_score = int(annotation.get("aidanbench_stopping_score", 0) or 0)
            instance_mean_novelty = float(annotation.get("aidanbench_mean_embedding_novelty", 0.0) or 0.0)

            response_count += parsed_count
            mean_embedding_novelty += instance_mean_novelty
            valid_response_rate += (sum(validity_flags) / len(validity_flags)) if validity_flags else 0.0
            coherence_pass_rate += (sum(coherence_passes) / len(coherence_passes)) if coherence_passes else 0.0
            novelty_pass_rate += (sum(novelty_passes) / len(novelty_passes)) if novelty_passes else 0.0
            stopping_score += instance_stopping_score
            stopping_ratio += (instance_stopping_score / expected_responses) if expected_responses else 0.0

        if total:
            response_count /= total
            mean_embedding_novelty /= total
            valid_response_rate /= total
            coherence_pass_rate /= total
            novelty_pass_rate /= total
            stopping_score /= total
            stopping_ratio /= total

        return [
            Stat(MetricName("aidanbench_response_count")).add(response_count),
            Stat(MetricName("aidanbench_mean_embedding_novelty")).add(mean_embedding_novelty),
            Stat(MetricName("aidanbench_valid_response_rate")).add(valid_response_rate),
            Stat(MetricName("aidanbench_coherence_pass_rate")).add(coherence_pass_rate),
            Stat(MetricName("aidanbench_novelty_pass_rate")).add(novelty_pass_rate),
            Stat(MetricName("aidanbench_stopping_score")).add(stopping_score),
            Stat(MetricName("aidanbench_stopping_ratio")).add(stopping_ratio),
            Stat(MetricName("aidanbench_proxy_stopping_score")).add(stopping_score),
            Stat(MetricName("aidanbench_proxy_stopping_ratio")).add(stopping_ratio),
        ]
