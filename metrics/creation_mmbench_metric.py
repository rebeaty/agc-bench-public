"""Creation-MMBench metric wrapper."""

from __future__ import annotations

from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class CreationMMBenchMetric(Metric):
    """Reads the benchmark-specific dual-pass judge output."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        annotations = request_state.annotations or {}
        annotator_output = annotations.get("creation_mmbench_judge", {}) or {}
        return [
            Stat(MetricName("judge_parse_rate")).add(float(annotator_output.get("judge_parse_rate", 0.0))),
            Stat(MetricName("creation_mmbench_vfs")).add(float(annotator_output.get("creation_mmbench_vfs", 0.0))),
            Stat(MetricName("creation_mmbench_reward")).add(float(annotator_output.get("creation_mmbench_reward", 0.0))),
            Stat(MetricName("creation_mmbench_dual_eval_gap")).add(float(annotator_output.get("creation_mmbench_dual_eval_gap", 0.0))),
        ]
