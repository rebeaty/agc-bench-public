"""II-Bench metric mirroring the upstream parsed-answer evaluator."""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


def _extract_option_label(text: str, options: List[str]) -> Optional[str]:
    if not isinstance(text, str):
        return "error"

    matches = re.findall(r"\(([A-F])\)", text, flags=re.IGNORECASE)
    if not matches:
        matches = re.findall(r"\b([A-F])\b", text, flags=re.IGNORECASE)
    if matches:
        counter = Counter(match.upper() for match in matches)
        most_common = counter.most_common()
        max_count = most_common[0][1]
        candidates = [item for item in most_common if item[1] == max_count]
        return candidates[-1][0]

    option_counter: Counter[str] = Counter()
    lowered_text = text.strip().lower()
    for index, option in enumerate(options):
        label = chr(ord("A") + index)
        option_text = option.strip()
        if not option_text:
            continue
        lowered_option = option_text.lower()
        if lowered_option in lowered_text or lowered_text in lowered_option:
            option_counter[label] += 1
    if option_counter:
        most_common = option_counter.most_common()
        max_count = most_common[0][1]
        candidates = [item for item in most_common if item[1] == max_count]
        return candidates[-1][0]

    return None


class IIBenchMetric(Metric):
    """Score II-Bench multiple-choice predictions with upstream parsing rules."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None

        prediction = request_state.result.completions[0].text
        extra_data = request_state.instance.extra_data or {}
        parsed_label = _extract_option_label(prediction, list(extra_data.get("options", [])))
        gold_label = extra_data["gold_label"]

        is_error = parsed_label == "error"
        is_miss = parsed_label is None
        is_correct = parsed_label == gold_label

        return [
            Stat(MetricName("accuracy")).add(1.0 if is_correct else 0.0),
            Stat(MetricName("errors_rate")).add(1.0 if is_error else 0.0),
            Stat(MetricName("miss_rate")).add(1.0 if is_miss else 0.0),
        ]
