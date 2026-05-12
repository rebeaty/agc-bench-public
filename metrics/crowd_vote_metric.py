"""Benchmark-specific metric for Crowd Vote."""

from __future__ import annotations

from typing import Any, List, Optional

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class CrowdVoteMetric(Metric):
    """Expose structured marketing creativity proxy scores."""

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
        # Stat prefix renamed 2026-04-25: was `crowd_vote_*` — misleading
        # because no actual crowd voting occurs; we use a single LLM judge.
        # The annotation key + prefix now both read `marketing_creativity_judge*`
        # for consistency with what we actually measure.
        judge = annotations.get("marketing_creativity_judge", {}) or {}
        stats: List[Stat] = [
            Stat(MetricName("marketing_creativity_judge_parse_rate")).add(float(judge.get("parse_rate", 0.0))),
        ]
        self._add_optional("marketing_creativity_judge_originality", judge.get("originality"), stats)
        self._add_optional("marketing_creativity_judge_brand_relevance", judge.get("brand_relevance"), stats)
        self._add_optional("marketing_creativity_judge_creative_potential", judge.get("creative_potential"), stats)
        self._add_optional("marketing_creativity_judge_conciseness", judge.get("conciseness"), stats)
        self._add_optional("marketing_creativity_judge_overall", judge.get("overall"), stats)
        return stats
