"""Arena-Hard pairwise score aggregation."""

from __future__ import annotations

from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class ArenaHardPairwiseMetric(Metric):
    """Aggregate upstream-style Arena-Hard battle scores."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        annotations = request_state.annotations or {}
        annotator_output = annotations.get("arena_hard_pairwise", {}) or {}
        battle_scores = [float(score) for score in annotator_output.get("battle_scores", [])]

        stats: List[Stat] = [
            Stat(MetricName("arena_hard_parse_rate")).add(float(annotator_output.get("parsed_both", 0.0))),
            Stat(MetricName("arena_hard_battles_per_instance")).add(float(len(battle_scores))),
        ]

        if not battle_scores:
            return stats

        mean_score = sum(battle_scores) / len(battle_scores)
        score_stat = Stat(MetricName("arena_hard_score"))
        win_stat = Stat(MetricName("arena_hard_win"))
        tie_stat = Stat(MetricName("arena_hard_tie"))
        loss_stat = Stat(MetricName("arena_hard_loss"))
        for score in battle_scores:
            score_stat.add(score * 100.0)
            win_stat.add(1.0 if score == 1.0 else 0.0)
            tie_stat.add(1.0 if score == 0.5 else 0.0)
            loss_stat.add(1.0 if score == 0.0 else 0.0)

        stats.append(Stat(MetricName("arena_hard_instance_score")).add(mean_score * 100.0))
        stats.extend([score_stat, win_stat, tie_stat, loss_stat])
        return stats
