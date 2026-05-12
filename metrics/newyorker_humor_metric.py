"""Task-specific metrics for the New Yorker humor benchmark family."""

from __future__ import annotations

import re
from typing import List, Optional

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat


def _extract_answer_label(text: str, options: str) -> Optional[str]:
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return None

    option_pattern = f"[{re.escape(options)}]"
    answer_matches = re.findall(rf"answer\s*[:\-]?\s*({option_pattern})\b", cleaned, flags=re.IGNORECASE)
    if answer_matches:
        return answer_matches[-1].upper()

    bare_matches = re.findall(rf"\b({option_pattern})\b", cleaned, flags=re.IGNORECASE)
    if bare_matches:
        return bare_matches[-1].upper()

    return None


class NewYorkerHumorMetric(EvaluateInstancesMetric):
    """Score matching or ranking outputs with the upstream answer-letter protocol."""

    def __init__(self, task: str):
        self.task = task
        if self.task not in {"matching", "ranking"}:
            raise ValueError(f"Unsupported newyorker_humor metric task: {task}")

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        scored_request_states = [state for state in request_states if state.request_mode != "calibration"]
        total = len(scored_request_states)
        parsed = 0

        if self.task == "matching":
            correct = 0
            for state in scored_request_states:
                assert state.result is not None
                prediction = state.result.completions[0].text if state.result.completions else ""
                parsed_label = _extract_answer_label(prediction, "ABCDE")
                gold_label = str((state.instance.extra_data or {}).get("gold_label", "")).upper()
                if parsed_label is not None:
                    parsed += 1
                if parsed_label == gold_label:
                    correct += 1

            accuracy = correct / total if total else 0.0
            parsed_rate = parsed / total if total else 0.0
            return [
                Stat(MetricName("accuracy")).add(accuracy),
                Stat(MetricName("parsed_label_rate")).add(parsed_rate),
            ]

        correct_overall = 0
        correct_ny = 0
        total_ny = 0
        correct_crowd = 0
        total_crowd = 0

        for state in scored_request_states:
            assert state.result is not None
            prediction = state.result.completions[0].text if state.result.completions else ""
            parsed_label = _extract_answer_label(prediction, "AB")
            extra_data = state.instance.extra_data or {}
            gold_label = str(extra_data.get("gold_label", "")).upper()
            winner_source = str(extra_data.get("winner_source", ""))
            is_correct = parsed_label == gold_label

            if parsed_label is not None:
                parsed += 1
            if is_correct:
                correct_overall += 1

            if winner_source == "crowd_winner":
                total_crowd += 1
                if is_correct:
                    correct_crowd += 1
            else:
                total_ny += 1
                if is_correct:
                    correct_ny += 1

        accuracy = correct_overall / total if total else 0.0
        accuracy_ny = correct_ny / total_ny if total_ny else 0.0
        accuracy_crowd = correct_crowd / total_crowd if total_crowd else 0.0
        parsed_rate = parsed / total if total else 0.0

        return [
            Stat(MetricName("accuracy")).add(accuracy),
            Stat(MetricName("accuracy_ny")).add(accuracy_ny),
            Stat(MetricName("accuracy_crowd")).add(accuracy_crowd),
            Stat(MetricName("parsed_label_rate")).add(parsed_rate),
        ]
