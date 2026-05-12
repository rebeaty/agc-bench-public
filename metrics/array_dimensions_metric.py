"""Validate generated array dimensions for structured-output benchmarks."""

import json
from typing import List, Tuple

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


def _shape(obj) -> Tuple:
    if isinstance(obj, list):
        return (len(obj),) + (_shape(obj[0]) if obj else ())
    return ()


class ArrayDimensionsMetric(Metric):
    """Checks whether the generated array/matrix has the correct shape matching the reference."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        completion = request_state.result.completions[0].text.strip()

        references = request_state.instance.references
        reference_text = references[0].output.text.strip() if references else ""

        try:
            data = json.loads(completion)
            expected = json.loads(reference_text)
            score = 1.0 if _shape(data) == _shape(expected) else 0.0
        except (json.JSONDecodeError, TypeError):
            score = 0.0

        return [Stat(MetricName("array_dimensions")).add(score)]
