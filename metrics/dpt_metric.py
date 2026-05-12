"""DPT metric reader."""

from __future__ import annotations

from typing import List, Optional

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class DptMetric(Metric):
    """Expose the released DPT originality dimensions."""

    def _add_optional(self, name: str, value: Optional[float], stats: List[Stat]) -> None:
        if value is not None:
            stats.append(Stat(MetricName(name)).add(float(value)))

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        annotations = request_state.annotations or {}
        annotator_output = annotations.get("dpt_dimension_judge", {}) or {}
        stats: List[Stat] = [
            Stat(MetricName("dpt_parse_rate")).add(float(annotator_output.get("dpt_parse_rate", 0.0))),
        ]
        self._add_optional("dpt_originality", annotator_output.get("originality"), stats)
        self._add_optional("dpt_uncommon", annotator_output.get("uncommon"), stats)
        self._add_optional("dpt_remote", annotator_output.get("remote"), stats)
        self._add_optional("dpt_clever", annotator_output.get("clever"), stats)
        return stats
