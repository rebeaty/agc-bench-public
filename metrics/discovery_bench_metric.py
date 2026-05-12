"""DiscoveryBench HMS-style metric."""

from __future__ import annotations

from typing import Any, Dict, List

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_fraction(value: float) -> float:
    if value > 1.0:
        return value / 100.0
    return max(0.0, value)


def _normalize_relationship(value: float) -> float:
    if value <= 1.0:
        return max(0.0, value) * 100.0
    return max(0.0, value)


def _annotation_for_state(state: RequestState) -> Dict[str, Any]:
    annotations = state.annotations or {}
    for value in annotations.values():
        if isinstance(value, dict) and (
            "discovery_bench_hms_parse_rate" in value
            or "context_f1" in value
            or "variable_f1" in value
        ):
            return value
        if isinstance(value, dict):
            for nested_value in value.values():
                if isinstance(nested_value, dict) and (
                    "discovery_bench_hms_parse_rate" in nested_value
                    or "context_f1" in nested_value
                    or "variable_f1" in nested_value
                ):
                    return nested_value
    return {}


class DiscoveryBenchMetric(EvaluateInstancesMetric):
    """Expose DiscoveryBench HMS-style judge outputs."""

    def __init__(self, **_: Any):
        super().__init__()

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        scored_states = [state for state in request_states if state.request_mode != "calibration"]
        total = len(scored_states)

        hms_scores: List[float] = []
        context_scores: List[float] = []
        variable_scores: List[float] = []
        relationship_scores: List[float] = []
        parse_rates: List[float] = []

        for state in scored_states:
            annotation = _annotation_for_state(state)
            parse_rate = _as_float(annotation.get("discovery_bench_hms_parse_rate", 0.0))
            parse_rates.append(parse_rate)
            if not annotation:
                continue

            context_f1 = _normalize_fraction(_as_float(annotation.get("context_f1", 0.0)))
            variable_f1 = _normalize_fraction(_as_float(annotation.get("variable_f1", 0.0)))
            relationship_accuracy = _normalize_relationship(
                _as_float(annotation.get("relationship_accuracy", 0.0))
            )

            hms = annotation.get("hms")
            if hms is None or hms == "":
                hms = context_f1 * variable_f1 * relationship_accuracy
            else:
                hms = _as_float(hms)

            hms_scores.append(hms)
            context_scores.append(context_f1)
            variable_scores.append(variable_f1)
            relationship_scores.append(relationship_accuracy)

        def _mean(values: List[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        return [
            Stat(MetricName("discovery_bench_hms")).add(_mean(hms_scores)),
            Stat(MetricName("discovery_bench_context_f1")).add(_mean(context_scores)),
            Stat(MetricName("discovery_bench_variable_f1")).add(_mean(variable_scores)),
            Stat(MetricName("discovery_bench_relationship_accuracy")).add(_mean(relationship_scores)),
            Stat(MetricName("discovery_bench_parse_rate")).add(_mean(parse_rates)),
        ]
