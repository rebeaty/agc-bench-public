"""Benchmark-specific metric for CPers."""

from __future__ import annotations

from typing import List, Optional

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class CPersMetric(Metric):
    """Expose TTCT-style Persian creativity dimensions and rhetorical devices."""

    def _add_optional(self, name: str, value: Optional[float], stats: List[Stat]) -> None:
        if value is not None:
            stats.append(Stat(MetricName(name)).add(float(value)))

    def evaluate_generation(
        self,
        adapter_spec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        annotations = request_state.annotations or {}
        judge = annotations.get("cpers_ttct_judge", {}) or {}
        stats: List[Stat] = [
            Stat(MetricName("cpers_parse_rate")).add(float(judge.get("cpers_parse_rate", 0.0))),
        ]
        self._add_optional("cpers_originality", judge.get("originality"), stats)
        self._add_optional("cpers_fluency", judge.get("fluency"), stats)
        self._add_optional("cpers_flexibility", judge.get("flexibility"), stats)
        self._add_optional("cpers_elaboration", judge.get("elaboration"), stats)
        self._add_optional("cpers_overall_creativity", judge.get("overall_creativity"), stats)
        self._add_optional("cpers_simile_rate", judge.get("simile"), stats)
        self._add_optional("cpers_metaphor_rate", judge.get("metaphor"), stats)
        self._add_optional("cpers_hyperbole_rate", judge.get("hyperbole"), stats)
        self._add_optional("cpers_antithesis_rate", judge.get("antithesis"), stats)
        return stats
