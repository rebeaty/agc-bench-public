"""Pairwise creativity preference metric for CreataSet."""

from __future__ import annotations

from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class CreatSetPairwiseMetric(Metric):
    """Score parsed pairwise creativity judgments against the released baseline."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        annotations = request_state.annotations or {}
        annotator_output = annotations.get("creatset_pairwise", {}) or {}

        verdict = annotator_output.get("verdict")
        parsed = float(annotator_output.get("parsed", 0.0))
        candidate_win = 1.0 if verdict == 1 else 0.0
        candidate_loss = 1.0 if verdict == 2 else 0.0

        return [
            Stat(MetricName("creatset_pairwise_parse_rate")).add(parsed),
            Stat(MetricName("creatset_pairwise_win_rate")).add(candidate_win),
            Stat(MetricName("creatset_pairwise_loss_rate")).add(candidate_loss),
        ]
