"""Distinct-n diversity metric for generated text."""

from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class DistinctNMetric(Metric):
    """Measures lexical diversity as unique n-grams / total n-grams (distinct-N)."""

    def __init__(self, n: int):
        super().__init__()
        assert n in (1, 2), "n must be 1 or 2"
        self.n = n

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        completion = request_state.result.completions[0].text.strip()

        tokens = completion.lower().split()
        if len(tokens) == 0:
            score = 0.0
        elif self.n == 1:
            ngrams = tokens
            score = len(set(ngrams)) / len(ngrams)
        else:
            ngrams = [tuple(tokens[i : i + 2]) for i in range(len(tokens) - 1)]
            if len(ngrams) == 0:
                score = 0.0
            else:
                score = len(set(ngrams)) / len(ngrams)

        metric_name = f"distinct_{self.n}"
        return [Stat(MetricName(metric_name)).add(score)]
