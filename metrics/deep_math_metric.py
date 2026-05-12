"""DeepMath benchmark metrics."""

from __future__ import annotations

from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class DeepMathMetric(Metric):
    """Aggregate DeepMath direction and process accuracy."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        annotations = request_state.annotations or {}
        judge = annotations.get("deep_math_judge", {}) or {}
        direction = float(judge.get("deep_math_direction_accuracy", 0.0) or 0.0)
        process = float(judge.get("deep_math_process_accuracy", 0.0) or 0.0)
        valid = float(judge.get("deep_math_valid_judge_rate", 0.0) or 0.0)
        return [
            Stat(MetricName("deep_math_direction_accuracy")).add(direction),
            Stat(MetricName("deep_math_process_accuracy")).add(process),
            Stat(MetricName("deep_math_valid_judge_rate")).add(valid),
        ]
