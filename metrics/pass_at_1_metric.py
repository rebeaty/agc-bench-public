"""Pass@1 metric for exact-answer or executable-solution tasks."""

from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class PassAt1Metric(Metric):
    """Binary metric: 1.0 if the generated code passes all test cases.

    Requires the corresponding annotator to execute the code against test cases
    and populate request_state.annotations["pass"] with a truthy value.
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
        score = 1.0 if annotations.get("pass", False) else 0.0

        return [Stat(MetricName("pass_at_1")).add(score)]
