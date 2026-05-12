"""DuanzAI PER metrics for Chinese homophonic pun punchline extraction."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import List

from fuzzywuzzy import fuzz

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


def _clean_prediction(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    cleaned = cleaned.splitlines()[0].strip()
    for prefix in ("答案：", "答案:", "Punchline:", "回答：", "回答:"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    return cleaned


class ChineseHomophonicPunsMetric(Metric):
    """Score punchline extraction with DuanzAI exact/similar-match accuracy."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None

        prediction = _clean_prediction(request_state.result.completions[0].text)
        gold = request_state.instance.references[0].output.text.strip()

        exact = 1.0 if prediction == gold else 0.0
        if exact:
            similar = 1.0
        else:
            similarity_ratio = SequenceMatcher(None, prediction, gold).ratio()
            fuzzy_ratio = fuzz.ratio(prediction, gold) / 100.0
            similar = min(1.0, max(similarity_ratio, fuzzy_ratio))

        return [
            Stat(MetricName("exact_match_accuracy")).add(exact),
            Stat(MetricName("similar_match_accuracy")).add(similar),
        ]
