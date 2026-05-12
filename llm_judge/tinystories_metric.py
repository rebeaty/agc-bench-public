"""Metric reader for TinyStories teacher-style judge annotations."""

from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class TinyStoriesMetric(Metric):
    """Expose TinyStories judge annotations as HELM stats."""

    METRICS = (
        "tinystories_grammar_score",
        "tinystories_creativity_score",
        "tinystories_consistency_score",
        "tinystories_age_group_ordinal",
        "tinystories_valid_judge_rate",
        "tinystories_scored_completion_count",
    )

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        annotations = request_state.annotations or {}
        annotator_output = annotations.get("tinystories_judge", {}) or {}
        return [
            Stat(MetricName(metric_name)).add(float(annotator_output.get(metric_name, 0.0)))
            for metric_name in self.METRICS
        ]
