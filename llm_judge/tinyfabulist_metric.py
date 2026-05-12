"""Metric reader for TinyFabulist judge annotations."""

from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


_METRIC_NAMES = [
    "grammar_score",
    "creativity_score",
    "moral_clarity_score",
    "adherence_to_prompt_score",
    "mean_judge_score",
    "age_group_a_rate",
    "age_group_b_rate",
    "age_group_c_rate",
    "age_group_d_rate",
    "age_group_e_rate",
    "valid_judge_rate",
]


class TinyFabulistMetric(Metric):
    """Expose TinyFabulist judge annotations as HELM stats."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        annotations = request_state.annotations or {}
        annotator_output = annotations.get("tinyfabulist_judge", {}) or {}
        return [
            Stat(MetricName(metric_name)).add(float(annotator_output.get(metric_name, 0.0)))
            for metric_name in _METRIC_NAMES
        ]
