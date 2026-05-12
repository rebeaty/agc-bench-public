"""Benchmark-specific preference metric for LitBench."""

import re
from typing import List, Optional

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


def _extract_choice(text: str) -> Optional[str]:
    cleaned = text.strip().upper()
    patterns = [
        r"^\s*([AB])\s*$",
        r"\bPREFERRED\s*[:\-]?\s*([AB])\b",
        r"\bANSWER\s*[:\-]?\s*([AB])\b",
        r"\b([AB])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            return match.group(1)
    return None


class LitBenchMetric(Metric):
    """Parses A/B preference outputs into held-out preference accuracy."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        prediction = request_state.result.completions[0].text.strip()
        parsed = _extract_choice(prediction)
        gold = None
        for reference in request_state.instance.references:
            if "correct" in reference.tags:
                gold = reference.output.text.strip()
                break
        stats: List[Stat] = [Stat(MetricName("litbench_parse_rate")).add(1.0 if parsed else 0.0)]
        if parsed is not None and gold is not None:
            stats.append(Stat(MetricName("litbench_accuracy")).add(1.0 if parsed == gold else 0.0))
        return stats
