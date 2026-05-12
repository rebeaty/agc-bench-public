"""Generic METEOR metric for single-output generation benchmarks."""

from __future__ import annotations

import threading
from typing import List

from nltk.translate.meteor_score import single_meteor_score
from nltk.translate.meteor_score import wordnet as nltk_wordnet
from nltk.stem import PorterStemmer

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class _EmptyWordNet:
    def synsets(self, word: str) -> list:
        return []


_meteor_lock = threading.Lock()


class MeteorMetric(Metric):
    """Compute best-reference METEOR for the first completion."""

    def __init__(self) -> None:
        super().__init__()
        self._stemmer = PorterStemmer()
        self._wordnet = nltk_wordnet if self._wordnet_available() else _EmptyWordNet()

    @staticmethod
    def _wordnet_available() -> bool:
        try:
            nltk_wordnet.synsets("benchmark")
            return True
        except LookupError:
            return False

    @staticmethod
    def _clean(text: str) -> str:
        return " ".join((text or "").strip().split())

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None

        prediction = self._clean(request_state.result.completions[0].text if request_state.result.completions else "")
        references = [
            self._clean(reference.output.text)
            for reference in request_state.instance.references
            if self._clean(reference.output.text)
        ]
        if not prediction or not references:
            return [Stat(MetricName("meteor")).add(0.0)]

        prediction_tokens = prediction.split()
        with _meteor_lock:
            best_score = max(
                float(
                    single_meteor_score(
                        reference.split(),
                        prediction_tokens,
                        stemmer=self._stemmer,
                        wordnet=self._wordnet,
                    )
                )
                for reference in references
            )
        return [Stat(MetricName("meteor")).add(best_score)]
