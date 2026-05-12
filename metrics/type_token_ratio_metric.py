"""Type-token ratio metric for lexical diversity."""

from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class TypeTokenRatioMetric(Metric):
    """Type-token ratio: unique tokens / total tokens, measuring lexical diversity."""

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
            import spacy

            nlp = spacy.load("en_core_web_sm")
            doc = nlp(completion)
            tokens = [token.text.lower() for token in doc if not token.is_space]
        except Exception:
            # Fall back to whitespace tokenization if spaCy is unavailable
            tokens = completion.lower().split()

        score = len(set(tokens)) / len(tokens) if tokens else 0.0

        return [Stat(MetricName("type_token_ratio")).add(score)]
