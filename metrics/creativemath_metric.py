"""CreativeMath set-level staged evaluation metrics."""

from __future__ import annotations

from typing import List

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat


class CreativeMathMetric(EvaluateInstancesMetric):
    """Aggregate CreativeMath correctness and novelty ratios."""

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        eval_states = [state for state in request_states if state.request_mode != "calibration"]

        total = len(eval_states)
        correctness_yes = 0
        coarse_yes = 0
        fine_yes = 0

        stage1_valid_votes = 0
        stage1_total_votes = 0
        stage2_valid_votes = 0
        stage2_total_votes = 0
        stage3_valid_votes = 0
        stage3_total_votes = 0

        for state in eval_states:
            annotations = state.annotations or {}
            result = annotations.get("creativemath_judge", {}) or {}

            stage1_total_votes += len(result.get("creativemath_correctness_votes", []))
            stage1_valid_votes += int(result.get("creativemath_correctness_valid_votes", 0))
            if result.get("creativemath_correctness_final") == "YES":
                correctness_yes += 1

            stage2_votes = result.get("creativemath_coarse_votes", [])
            stage2_total_votes += len(stage2_votes)
            stage2_valid_votes += int(result.get("creativemath_coarse_valid_votes", 0))
            if result.get("creativemath_coarse_final") == "YES":
                coarse_yes += 1

            stage3_votes = result.get("creativemath_fine_votes", [])
            stage3_total_votes += len(stage3_votes)
            stage3_valid_votes += int(result.get("creativemath_fine_valid_votes", 0))
            if result.get("creativemath_fine_final") == "YES":
                fine_yes += 1

        correctness_ratio = correctness_yes / total if total else 0.0
        novelty_ratio = coarse_yes / total if total else 0.0
        novel_unknown_ratio = fine_yes / total if total else 0.0
        novelty_to_correctness_ratio = coarse_yes / correctness_yes if correctness_yes else 0.0
        novel_unknown_to_novelty_ratio = fine_yes / coarse_yes if coarse_yes else 0.0

        return [
            Stat(MetricName("creativemath_correctness_ratio")).add(correctness_ratio),
            Stat(MetricName("creativemath_novelty_ratio")).add(novelty_ratio),
            Stat(MetricName("creativemath_novel_unknown_ratio")).add(novel_unknown_ratio),
            Stat(MetricName("creativemath_novelty_to_correctness_ratio")).add(novelty_to_correctness_ratio),
            Stat(MetricName("creativemath_novel_unknown_to_novelty_ratio")).add(novel_unknown_to_novelty_ratio),
            Stat(MetricName("creativemath_stage1_valid_judge_rate")).add(
                stage1_valid_votes / stage1_total_votes if stage1_total_votes else 0.0
            ),
            Stat(MetricName("creativemath_stage2_valid_judge_rate")).add(
                stage2_valid_votes / stage2_total_votes if stage2_total_votes else 0.0
            ),
            Stat(MetricName("creativemath_stage3_valid_judge_rate")).add(
                stage3_valid_votes / stage3_total_votes if stage3_total_votes else 0.0
            ),
        ]
