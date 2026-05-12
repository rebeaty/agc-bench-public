"""EQBench Creative Writing v3 rubric metric."""

from __future__ import annotations

from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class EQBenchCreativeWritingV3Metric(Metric):
    """Reads the benchmark-specific rubric annotation output."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        annotations = request_state.annotations or {}
        annotator_output = annotations.get("eqbench_creative_writing_v3_rubric", {}) or {}
        return [
            Stat(MetricName("judge_parse_rate")).add(float(annotator_output.get("judge_parse_rate", 0.0))),
            Stat(MetricName("criteria_count")).add(float(annotator_output.get("criteria_count", 0.0))),
            Stat(MetricName("creative_score_0_20")).add(float(annotator_output.get("creative_score_0_20", 0.0))),
            Stat(MetricName("eqbench_creative_score")).add(float(annotator_output.get("eqbench_creative_score", 0.0))),
        ]
