"""Percentile-rank metric for ordinal benchmark scores."""

from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

# Per task description: skip PCA-based Composite Score; focus on LLM-as-a-Judge metrics only.
# Percentile rank is computed from raw judge scores collected by the annotator.
# The annotator must populate request_state.annotations["percentile_rank"] (0–100 scale).


class PercentileRankMetric(Metric):
    """Percentile rank of the generated text among all evaluated responses.

    Reads from annotations populated by the WebNovelBench LLM judge annotator.
    For cross-instance percentile computation, aggregate stat values downstream.
    """

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None

        annotations = request_state.annotations or {}
        score = float(annotations.get("percentile_rank", 0.0))

        return [Stat(MetricName("percentile_rank")).add(score)]
