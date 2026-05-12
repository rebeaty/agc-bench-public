"""Metric reader for PunEval pun-detection judge annotations."""

from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

_METRIC_NAMES = [
    "pun_detection_rate",
    "pun_detection_parsed_rate",
    "sentence_json_parsed_rate",
]


class PunEvalMetric(Metric):
    """Expose PunEval judge annotations as HELM stats."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        annotations = request_state.annotations or {}
        annotator_output = annotations.get("pun_eval_judge", {}) or {}
        return [
            Stat(MetricName(metric_name)).add(float(annotator_output.get(metric_name, 0.0)))
            for metric_name in _METRIC_NAMES
        ]
