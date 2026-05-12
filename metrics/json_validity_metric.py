"""JSON validity metric for benchmarks that require parseable JSON output."""

import json
import re
from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


def _normalize_candidate_json(text: str) -> str:
    stripped = text.strip()
    match = _JSON_FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


class JsonValidityMetric(Metric):
    """Checks whether the generated output is valid JSON."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        completion = _normalize_candidate_json(request_state.result.completions[0].text)

        try:
            json.loads(completion)
            score = 1.0
        except json.JSONDecodeError:
            score = 0.0

        return [Stat(MetricName("json_validity")).add(score)]
