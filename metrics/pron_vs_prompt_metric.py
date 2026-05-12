"""Pron vs Prompt metric reader."""

from __future__ import annotations

from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class PronVsPromptMetric(Metric):
    """Aggregate released rubric dimensions into group means."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        annotations = request_state.annotations or {}
        scored = annotations.get("pron_vs_prompt_literary_judge", {}) or {}

        def _mean(keys: List[str]) -> float:
            values = [float(scored[key]) for key in keys if key in scored]
            return sum(values) / len(values) if values else 0.0

        return [
            Stat(MetricName("pron_vs_prompt_parse_rate")).add(float(scored.get("pron_vs_prompt_parse_rate", 0.0))),
            Stat(MetricName("pron_vs_prompt_attractiveness")).add(_mean(["title_attractiveness", "style_attractiveness", "theme_attractiveness"])),
            Stat(MetricName("pron_vs_prompt_originality")).add(_mean(["title_originality", "style_originality", "plot_originality"])),
            Stat(MetricName("pron_vs_prompt_relevance")).add(float(scored.get("relevance", 0.0))),
            Stat(MetricName("pron_vs_prompt_creativity")).add(_mean(["title_creativity", "synopsis_creativity"])),
            Stat(MetricName("pron_vs_prompt_criticism")).add(_mean(["anthology", "readers_opinion", "critics_opinion", "own_voice"])),
        ]
