"""Benchmark-specific metrics for LiveIdeaBench."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat


def _annotation_for_state(state: RequestState) -> Dict[str, Any]:
    annotations = state.annotations or {}
    annotation = annotations.get("liveideabench_v2_evaluator", {}) or {}
    return annotation if isinstance(annotation, dict) else {}


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


class LiveIdeaBenchMetric(EvaluateInstancesMetric):
    """Expose the LiveIdeaBench five-dimension adaptation."""

    def __init__(self, **_: Any):
        super().__init__()

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        originality: List[float] = []
        feasibility: List[float] = []
        clarity: List[float] = []
        fluency: List[float] = []
        overall: List[float] = []
        dimension_parse_rates: List[float] = []
        fluency_parse_rates: List[float] = []
        output_counts: List[float] = []

        for state in request_states:
            if state.request_mode == "calibration" or state.result is None:
                continue

            annotation = _annotation_for_state(state)
            output_counts.append(float(annotation.get("output_count", 0.0)))
            dimension_parse_rates.append(float(annotation.get("dimension_parse_rate", 0.0)))
            fluency_parse_rates.append(float(annotation.get("fluency_parse_rate", 0.0)))

            if annotation.get("mean_originality") is not None:
                originality.append(float(annotation["mean_originality"]))
            if annotation.get("mean_feasibility") is not None:
                feasibility.append(float(annotation["mean_feasibility"]))
            if annotation.get("mean_clarity") is not None:
                clarity.append(float(annotation["mean_clarity"]))
            if annotation.get("fluency_score") is not None:
                fluency.append(float(annotation["fluency_score"]))
            if annotation.get("overall_score") is not None:
                overall.append(float(annotation["overall_score"]))

        flexibility = float(np.percentile(overall, 30)) if overall else 0.0

        return [
            Stat(MetricName("liveideabench_originality")).add(_mean(originality)),
            Stat(MetricName("liveideabench_feasibility")).add(_mean(feasibility)),
            Stat(MetricName("liveideabench_clarity")).add(_mean(clarity)),
            Stat(MetricName("liveideabench_fluency")).add(_mean(fluency)),
            Stat(MetricName("liveideabench_flexibility")).add(flexibility),
            Stat(MetricName("liveideabench_average")).add(_mean(overall)),
            Stat(MetricName("liveideabench_dimension_parse_rate")).add(_mean(dimension_parse_rates)),
            Stat(MetricName("liveideabench_fluency_parse_rate")).add(_mean(fluency_parse_rates)),
            Stat(MetricName("liveideabench_output_count")).add(_mean(output_counts)),
        ]
