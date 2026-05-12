"""GraphEval metrics for decision classification and score correlation."""

from __future__ import annotations

import math
import re
from statistics import mean
from typing import Dict, List, Optional

from scipy.stats import spearmanr

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat


class GraphEvalDecisionMetric(EvaluateInstancesMetric):
    """Parse GraphEval-style review scores and decisions from generated text."""

    LABEL_TO_ID: Dict[str, int] = {
        "Reject": 0,
        "Accept (Poster)": 1,
        "Accept (Spotlight)": 2,
        "Accept (Oral)": 3,
    }

    NORMALIZED_LABELS: Dict[str, str] = {
        "reject": "Reject",
        "poster": "Accept (Poster)",
        "accept (poster)": "Accept (Poster)",
        "spotlight": "Accept (Spotlight)",
        "accept (spotlight)": "Accept (Spotlight)",
        "oral": "Accept (Oral)",
        "accept (oral)": "Accept (Oral)",
    }

    # Longest variants first avoids partial matches.
    DECISION_RE = re.compile(
        r"accept\s*\(\s*spotlight\s*\)|"
        r"accept\s*\(\s*poster\s*\)|"
        r"accept\s*\(\s*oral\s*\)|"
        r"\bspotlight\b|"
        r"\bposter\b|"
        r"\boral\b|"
        r"\breject\b",
        flags=re.IGNORECASE,
    )
    SCORE_RE = re.compile(r"Overall Score\s*\(0-100\)\s*[:=]\s*(\d+)", flags=re.IGNORECASE)

    @classmethod
    def _normalize_label(cls, text: str) -> Optional[str]:
        normalized = re.sub(r"\s+", " ", text.strip()).lower()
        return cls.NORMALIZED_LABELS.get(normalized)

    @classmethod
    def parse_decision(cls, text: str) -> Optional[str]:
        if not text:
            return None

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in reversed(lines):
            match = cls.DECISION_RE.search(line)
            if match:
                label = cls._normalize_label(match.group(0))
                if label:
                    return label

        match = cls.DECISION_RE.search(text)
        if match:
            return cls._normalize_label(match.group(0))
        return None

    @classmethod
    def parse_score(cls, text: str) -> Optional[float]:
        if not text:
            return None
        match = cls.SCORE_RE.search(text)
        if match is None:
            return None
        return float(match.group(1))

    @staticmethod
    def _metric_key(label: str) -> str:
        return (
            label.lower()
            .replace("accept ", "accept_")
            .replace("(", "")
            .replace(")", "")
            .replace(" ", "_")
        )

    @classmethod
    def _collapse_label_for_variant(cls, label: Optional[str], dataset_variant: Optional[str]) -> Optional[str]:
        if label is None:
            return None
        if dataset_variant == "ai_researcher" and label == "Accept (Oral)":
            return "Accept (Spotlight)"
        return label

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        eval_states = [state for state in request_states if state.request_mode != "calibration"]
        if not eval_states:
            return []

        first_extra = eval_states[0].instance.extra_data or {}
        label_set = list(first_extra.get("label_set") or self.LABEL_TO_ID.keys())

        parsed_decisions = 0
        parsed_scores = 0
        correct_predictions = 0

        gold_ids: List[int] = []
        pred_ids: List[Optional[int]] = []
        gold_scores: List[float] = []
        pred_scores: List[float] = []

        per_label_counts = {label: {"tp": 0.0, "fp": 0.0, "fn": 0.0} for label in label_set}

        for request_state in eval_states:
            assert request_state.result is not None

            prediction_text = request_state.result.completions[0].text
            predicted_label = self.parse_decision(prediction_text)
            predicted_score = self.parse_score(prediction_text)
            extra_data = request_state.instance.extra_data or {}
            dataset_variant = extra_data.get("dataset_variant")

            gold_label = extra_data.get("gold_label")
            gold_score = extra_data.get("gold_score")
            if gold_label is None:
                for ref in request_state.instance.references:
                    if getattr(ref, "is_correct", False):
                        gold_label = self._normalize_label(ref.output.text)
                        break
                if gold_label is None and request_state.instance.references:
                    gold_label = self._normalize_label(request_state.instance.references[0].output.text)

            predicted_label = self._collapse_label_for_variant(predicted_label, dataset_variant)
            gold_label = self._collapse_label_for_variant(gold_label, dataset_variant)

            if gold_label is None:
                continue

            if predicted_label is not None:
                parsed_decisions += 1
            if predicted_score is not None and gold_score is not None:
                parsed_scores += 1
                gold_scores.append(float(gold_score))
                pred_scores.append(float(predicted_score))

            if predicted_label == gold_label:
                correct_predictions += 1

            gold_ids.append(self.LABEL_TO_ID[gold_label])
            pred_ids.append(self.LABEL_TO_ID[predicted_label] if predicted_label is not None else None)

            for label in label_set:
                per_label_counts[label]["tp"] += 1.0 if predicted_label == label and gold_label == label else 0.0
                per_label_counts[label]["fp"] += 1.0 if predicted_label == label and gold_label != label else 0.0
                per_label_counts[label]["fn"] += 1.0 if predicted_label != label and gold_label == label else 0.0

        total = len(gold_ids)
        decision_accuracy = correct_predictions / total if total else 0.0
        decision_parsed = parsed_decisions / total if total else 0.0
        score_parsed = parsed_scores / total if total else 0.0

        precisions: List[float] = []
        recalls: List[float] = []
        f1s: List[float] = []
        for label in label_set:
            tp = per_label_counts[label]["tp"]
            fp = per_label_counts[label]["fp"]
            fn = per_label_counts[label]["fn"]
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        spearman_correlation = 0.0
        if len(gold_scores) >= 2:
            corr, _ = spearmanr(gold_scores, pred_scores)
            if corr is not None and not math.isnan(corr):
                spearman_correlation = float(corr)

        stats: List[Stat] = [
            Stat(MetricName("decision_accuracy")).add(decision_accuracy),
            Stat(MetricName("decision_parsed")).add(decision_parsed),
            Stat(MetricName("score_parsed")).add(score_parsed),
            Stat(MetricName("precision_macro")).add(mean(precisions) if precisions else 0.0),
            Stat(MetricName("recall_macro")).add(mean(recalls) if recalls else 0.0),
            Stat(MetricName("f1_macro")).add(mean(f1s) if f1s else 0.0),
            Stat(MetricName("spearman_correlation")).add(spearman_correlation),
        ]

        for label in label_set:
            metric_key = self._metric_key(label)
            stats.extend(
                [
                    Stat(MetricName(f"{metric_key}_tp")).add(per_label_counts[label]["tp"]),
                    Stat(MetricName(f"{metric_key}_fp")).add(per_label_counts[label]["fp"]),
                    Stat(MetricName(f"{metric_key}_fn")).add(per_label_counts[label]["fn"]),
                ]
            )

        return stats
