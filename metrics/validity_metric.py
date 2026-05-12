"""Validity metric for tasks with parseable boolean correctness signals."""

import ast
from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class ValidityMetric(Metric):
    """Checks syntactic validity of the generated output (Python AST parse)."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        completion = request_state.result.completions[0].text.strip()

        try:
            ast.parse(completion)
            score = 1.0
        except SyntaxError:
            score = 0.0

        return [Stat(MetricName("validity")).add(score)]
