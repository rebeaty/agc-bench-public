"""Automatic similarity metrics used in FutureGen's non-judge evaluation slice."""

from __future__ import annotations

import re
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().split()).lower()


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", _normalize(text)))


class FutureGenSimilarityMetric(Metric):
    """Per-instance Jaccard and TF-IDF cosine similarity against the gold text."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None

        prediction = _normalize(request_state.result.completions[0].text if request_state.result.completions else "")
        references = [
            _normalize(reference.output.text)
            for reference in request_state.instance.references
            if _normalize(reference.output.text)
        ]
        if not prediction or not references:
            return [
                Stat(MetricName("jaccard_similarity")).add(0.0),
                Stat(MetricName("cosine_similarity")).add(0.0),
            ]

        pred_tokens = _tokens(prediction)
        best_jaccard = 0.0
        best_cosine = 0.0

        for reference in references:
            ref_tokens = _tokens(reference)
            union = pred_tokens | ref_tokens
            jaccard = (len(pred_tokens & ref_tokens) / len(union)) if union else 0.0
            best_jaccard = max(best_jaccard, jaccard)

            vectorizer = TfidfVectorizer()
            matrix = vectorizer.fit_transform([prediction, reference])
            cosine = float((matrix[0] @ matrix[1].T).toarray()[0][0])
            if np.isnan(cosine):
                cosine = 0.0
            best_cosine = max(best_cosine, cosine)

        return [
            Stat(MetricName("jaccard_similarity")).add(best_jaccard),
            Stat(MetricName("cosine_similarity")).add(best_cosine),
        ]
