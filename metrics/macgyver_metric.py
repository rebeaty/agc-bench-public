"""MacGyver category metric."""

from __future__ import annotations

from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

_CORRECT = {"A", "B", "C"}
_FEASIBLE = {"A", "B"}


class MacgyverMetric(Metric):
    """Expose paper-style category rates for MacGyver."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        annotations = request_state.annotations or {}
        annotator_output = annotations.get("macgyver_category_judge", {}) or {}
        category = str(annotator_output.get("category", "")).upper()

        stats: List[Stat] = [
            Stat(MetricName("judge_parse_rate")).add(float(annotator_output.get("judge_parse_rate", 0.0))),
            Stat(MetricName("macgyver_correct_rate")).add(1.0 if category in _CORRECT else 0.0),
            Stat(MetricName("macgyver_feasible_solution_rate")).add(1.0 if category in _FEASIBLE else 0.0),
            Stat(MetricName("macgyver_unsolvable_correct_rate")).add(1.0 if category == "C" else 0.0),
        ]
        for label in ["A", "B", "C", "D", "E", "F"]:
            stats.append(Stat(MetricName(f"macgyver_category_{label.lower()}_rate")).add(1.0 if category == label else 0.0))
        return stats
