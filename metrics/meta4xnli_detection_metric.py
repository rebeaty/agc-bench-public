"""Token-level detection metric for Meta4XNLI."""

import re
from typing import List, Optional

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat


def _extract_binary_sequence(text: str) -> Optional[List[int]]:
    cleaned = text.strip()
    if not cleaned:
        return None

    if "labels:" in cleaned.lower():
        cleaned = cleaned.split(":", 1)[1].strip()

    tokens = re.findall(r"\b[01]\b", cleaned)
    if not tokens:
        return None
    return [int(token) for token in tokens]


class Meta4XNLIDetectionMetric(EvaluateInstancesMetric):
    """Score generated binary token labels against the gold detection tags."""

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        true_positive = 0
        false_positive = 0
        false_negative = 0
        correct = 0
        total = 0
        valid_sequences = 0

        for request_state in request_states:
            if request_state.request_mode == "calibration":
                continue
            assert request_state.result is not None

            gold_tags = list((request_state.instance.extra_data or {}).get("gold_tags", []))
            prediction = _extract_binary_sequence(request_state.result.completions[0].text)
            if prediction is None or not gold_tags or len(prediction) != len(gold_tags):
                continue

            valid_sequences += 1
            for pred, gold in zip(prediction, gold_tags):
                total += 1
                if pred == gold:
                    correct += 1
                if pred == 1 and gold == 1:
                    true_positive += 1
                elif pred == 1 and gold == 0:
                    false_positive += 1
                elif pred == 0 and gold == 1:
                    false_negative += 1

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        if precision + recall:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        accuracy = correct / total if total else 0.0
        valid_rate = valid_sequences / len([rs for rs in request_states if rs.request_mode != "calibration"]) if request_states else 0.0

        return [
            Stat(MetricName("token_f1")).add(f1),
            Stat(MetricName("token_accuracy")).add(accuracy),
            Stat(MetricName("valid_label_sequence_rate")).add(valid_rate),
        ]
