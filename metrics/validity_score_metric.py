"""Average validity-score metric for graded structured outputs."""

from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

# This metric depends on an external critic model from MineAnyBuild.
# Reference: https://github.com/MineAnyBuild/MineAnyBuild/blob/main/mineanybuild/evaluator.py#L105
# The corresponding annotator must be configured separately to populate
# request_state.annotations["validity_score"] before this metric is called.


class ValidityScoreMetric(Metric):
    """Critic-model-based validity score for MineAnyBuild generated structures.

    Reads from annotations populated by the MineAnyBuild annotator.
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
        score = float(annotations.get("validity_score", 0.0))

        return [Stat(MetricName("validity_score")).add(score)]
