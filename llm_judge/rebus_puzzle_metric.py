"""Metric reader for Rebus Puzzle semantic-equivalence annotations."""

from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class RebusPuzzleJudgeMetric(Metric):
    """Expose rebus semantic-equivalence judge annotations as HELM stats."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        annotations = request_state.annotations or {}
        annotator_output = annotations.get("rebus_puzzle_judge", {}) or {}
        return [
            Stat(MetricName("rebus_semantic_equivalence")).add(
                float(annotator_output.get("rebus_semantic_equivalence", 0.0))
            )
        ]
