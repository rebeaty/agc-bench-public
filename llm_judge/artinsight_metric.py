"""Metric reader for ArtInsight judge annotations."""

from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

_METRIC_NAMES = [
    "artinsight_score",
    "artinsight_presumptive_score",
    "artinsight_reductive_score",
    "artinsight_detail_score",
    "artinsight_elements_score",
    "artinsight_misc_deduction",
]


class ArtInsightMetric(Metric):
    """Expose ArtInsight judge annotations as HELM stats."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        annotations = request_state.annotations or {}
        annotator_output = annotations.get("artinsight_judge", {}) or {}
        return [
            Stat(MetricName(metric_name)).add(float(annotator_output.get(metric_name, 0.0)))
            for metric_name in _METRIC_NAMES
        ]
