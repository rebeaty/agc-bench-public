"""Metric reader for WritingBench checklist-aware judge annotations."""

from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class WritingBenchMetric(Metric):
    """Expose WritingBench judge annotations as HELM stats."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        annotations = request_state.annotations or {}
        annotator_output = annotations.get("writingbench_judge", {}) or {}

        return [
            Stat(MetricName("writingbench_score")).add(
                float(annotator_output.get("writingbench_score", 0.0))
            ),
            Stat(MetricName("writingbench_valid_criteria_rate")).add(
                float(annotator_output.get("writingbench_valid_criteria_rate", 0.0))
            ),
            Stat(MetricName("writingbench_criteria_count")).add(
                float(annotator_output.get("writingbench_criteria_count", 0.0))
            ),
        ]
