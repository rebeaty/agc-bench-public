"""BrainTeaser metric with official grouped accuracies."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


def _metric_name(scope: str, name: str) -> str:
    return f"brainteaser_{scope}_{name}"


class BrainteaserMetric(Metric):
    """Reproduce BrainTeaser grouped accuracy reporting."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        golds = [reference for reference in request_state.instance.references if reference.is_correct]
        assert golds

        sorted_completions = sorted(request_state.result.completions, key=lambda x: -x.logprob)
        prediction = sorted_completions[0].text.strip() if sorted_completions else ""
        if request_state.output_mapping is not None:
            if adapter_spec.output_mapping_pattern:
                match = re.search(adapter_spec.output_mapping_pattern, prediction)
                prediction = match.group(0) if match else ""
            prediction = request_state.output_mapping.get(prediction, "")

        correct = 1.0 if prediction and any(gold.output.text == prediction for gold in golds) else 0.0
        return [Stat(MetricName("brainteaser_correct")).add(correct)]

    def derive_per_instance_stats(self, per_instance_stats: Dict) -> List[Stat]:
        groups: Dict[str, Dict[str, Dict[str, float]]] = {
            "sentence": {},
            "wordplay": {},
        }

        for instance, stats in per_instance_stats.items():
            extra_data = instance.extra_data or {}
            subset = extra_data.get("subset")
            variant = extra_data.get("variant")
            base_id = extra_data.get("base_id")
            if subset not in groups or variant not in {"original", "semantic", "context"} or not base_id:
                continue

            correct = 0.0
            for stat in stats:
                if stat.name.name == "brainteaser_correct":
                    correct = float(stat.sum)
                    break
            groups[subset].setdefault(base_id, {"original": 0.0, "semantic": 0.0, "context": 0.0})[variant] = correct

        derived: List[Stat] = []
        for scope in ("sentence", "wordplay", "all"):
            if scope == "all":
                values = list(groups["sentence"].values()) + list(groups["wordplay"].values())
            else:
                values = list(groups[scope].values())
            if not values:
                continue

            count = float(len(values))
            original = sum(item["original"] for item in values) / count
            semantic = sum(item["semantic"] for item in values) / count
            context = sum(item["context"] for item in values) / count
            overall = sum(item["original"] + item["semantic"] + item["context"] for item in values) / (3.0 * count)
            sr_accuracy = sum(1.0 for item in values if item["original"] == 1.0 and item["semantic"] == 1.0) / count
            cr_accuracy = (
                sum(
                    1.0
                    for item in values
                    if item["original"] == 1.0 and item["semantic"] == 1.0 and item["context"] == 1.0
                )
                / count
            )

            derived.extend(
                [
                    Stat(MetricName(_metric_name(scope, "overall_accuracy"))).add(overall),
                    Stat(MetricName(_metric_name(scope, "single_original_accuracy"))).add(original),
                    Stat(MetricName(_metric_name(scope, "single_semantic_accuracy"))).add(semantic),
                    Stat(MetricName(_metric_name(scope, "single_context_accuracy"))).add(context),
                    Stat(MetricName(_metric_name(scope, "sr_accuracy"))).add(sr_accuracy),
                    Stat(MetricName(_metric_name(scope, "cr_accuracy"))).add(cr_accuracy),
                ]
            )

        return derived
