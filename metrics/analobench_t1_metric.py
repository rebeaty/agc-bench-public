"""AnaloBench T1 metric mirroring the upstream regex-based evaluator."""

from __future__ import annotations

import re
from typing import List, Optional

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


_OPTION_PATTERN = re.compile(r"(?:\:\s)?([A-D])(?:\.|\s|$)", re.IGNORECASE)


def _extract_option(text: str) -> Optional[str]:
    matches = _OPTION_PATTERN.findall(text or "")
    if not matches:
        return None
    return matches[0].upper()


class AnaloBenchT1Metric(Metric):
    """Score T1 multiple-choice responses with AnaloBench partial credit."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None

        prediction = request_state.result.completions[0].text
        parsed_option = _extract_option(prediction)
        gold_label = request_state.instance.extra_data["gold_label"]

        irrelevant = parsed_option is None
        correct = parsed_option == gold_label if parsed_option is not None else False

        accuracy = 0.25 if irrelevant else (1.0 if correct else 0.0)

        return [
            Stat(MetricName("accuracy")).add(accuracy),
            Stat(MetricName("parsed_option_rate")).add(0.0 if irrelevant else 1.0),
            Stat(MetricName("irrelevant_rate")).add(1.0 if irrelevant else 0.0),
        ]
