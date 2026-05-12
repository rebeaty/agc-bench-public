"""Metric reader for BannerRequest400 judge annotations."""

from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

_METRIC_NAMES = [
    "banner_request_400_score",
    "banner_request_400_taa",
    "banner_request_400_lps",
    "banner_request_400_aqs",
    "banner_request_400_ctae",
    "banner_request_400_cpyq",
    "banner_request_400_bis",
    "banner_request_400_blueprint_validity",
    "banner_request_400_render_success",
    "banner_request_400_valid_judge_rate",
]


class BannerRequest400Metric(Metric):
    """Expose BannerRequest400 judge annotations as HELM stats."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        annotations = request_state.annotations or {}
        annotator_output = annotations.get("banner_request_400_judge", {}) or {}
        return [
            Stat(MetricName(metric_name)).add(float(annotator_output.get(metric_name, 0.0)))
            for metric_name in _METRIC_NAMES
        ]
