"""Binary label metric for the Humor Transfer benchmark slice."""

from __future__ import annotations

import re
from typing import List, Optional

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


_YES_PATTERN = re.compile(r"\b(yes|funny|humorous)\b", re.IGNORECASE)
_NO_PATTERN = re.compile(r"\b(no|not funny|non-humorous|non humorous)\b", re.IGNORECASE)


def _extract_label(text: str) -> Optional[str]:
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return None
    if _YES_PATTERN.search(cleaned):
        return "Yes"
    if _NO_PATTERN.search(cleaned):
        return "No"
    return None


class HumorTransferMetric(Metric):
    """Score the first extracted binary humor label against the canonical answer."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None

        prediction = request_state.result.completions[0].text if request_state.result.completions else ""
        parsed_label = _extract_label(prediction)
        gold_label = str((request_state.instance.extra_data or {}).get("gold_label", ""))

        return [
            Stat(MetricName("accuracy")).add(1.0 if parsed_label == gold_label else 0.0),
            Stat(MetricName("parsed_label_rate")).add(1.0 if parsed_label is not None else 0.0),
        ]
