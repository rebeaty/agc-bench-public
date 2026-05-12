"""Metric reader for HypoBench held-out inference annotations."""

from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


_METRIC_NAMES = [
    "hypobench_ind_accuracy",
    "hypobench_ind_f1",
    "hypobench_ind_parsed_label_rate",
    "hypobench_ood_accuracy",
    "hypobench_ood_f1",
    "hypobench_ood_parsed_label_rate",
    "hypobench_hypothesis_count",
    "hypobench_parsed_hypothesis_rate",
]


class HypoBenchInferenceMetric(Metric):
    """Expose HypoBench inference annotations as HELM stats."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        annotations = request_state.annotations or {}
        annotator_output = annotations.get("hypobench_inference", {}) or {}
        return [
            Stat(MetricName(metric_name)).add(float(annotator_output.get(metric_name, 0.0)))
            for metric_name in _METRIC_NAMES
        ]
