"""Task-specific metric for MusicTheoryBench answer selection."""

from __future__ import annotations

import re
from typing import List, Optional

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


_ANSWER_PATTERN = re.compile(r"answer\s*[:\-]?\s*([A-D])\b", flags=re.IGNORECASE)
_BARE_OPTION_PATTERN = re.compile(r"\b([A-D])\b", flags=re.IGNORECASE)


def _extract_option_label(text: str, options: List[str]) -> Optional[str]:
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return None

    answer_matches = _ANSWER_PATTERN.findall(cleaned)
    if answer_matches:
        return answer_matches[-1].upper()

    bare_matches = _BARE_OPTION_PATTERN.findall(cleaned)
    if bare_matches:
        return bare_matches[-1].upper()

    lowered_text = cleaned.lower()
    matched_labels: List[str] = []
    for index, option in enumerate(options):
        option_text = " ".join((option or "").strip().split())
        if not option_text:
            continue
        lowered_option = option_text.lower()
        if lowered_text == lowered_option or lowered_option in lowered_text:
            matched_labels.append(chr(ord("A") + index))

    if len(matched_labels) == 1:
        return matched_labels[0]

    return None


class MusicTheoryBenchMetric(Metric):
    """Score MusicTheoryBench with parsed answer-letter accuracy."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None

        prediction = request_state.result.completions[0].text if request_state.result.completions else ""
        extra_data = request_state.instance.extra_data or {}
        gold_label = str(extra_data.get("gold_label", "")).upper()
        options = list(extra_data.get("options", []))
        parsed_label = _extract_option_label(prediction, options)

        return [
            Stat(MetricName("accuracy")).add(1.0 if parsed_label == gold_label else 0.0),
            Stat(MetricName("parsed_label_rate")).add(1.0 if parsed_label is not None else 0.0),
        ]
