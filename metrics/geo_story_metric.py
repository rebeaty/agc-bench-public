"""Post-hoc corpus metrics for Geo Story aligned with the upstream repo."""

from __future__ import annotations

import threading
from collections import Counter
from statistics import mean
from typing import List, Set

import nltk
import spacy

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat

_LOCATION_ENTITY_LABELS = {"LOC", "FAC", "GPE"}
_SPACY_MODEL_NAMES = ("en_core_web_trf", "en_core_web_sm")
_NLP = None
_NLP_LOCK = threading.Lock()
_STOPWORDS = None
_STOPWORDS_LOCK = threading.Lock()


def _get_stopwords() -> Set[str]:
    global _STOPWORDS
    if _STOPWORDS is None:
        with _STOPWORDS_LOCK:
            if _STOPWORDS is None:
                try:
                    stopwords = nltk.corpus.stopwords.words("english")
                except LookupError:
                    nltk.download("stopwords", quiet=True)
                    stopwords = nltk.corpus.stopwords.words("english")
                _STOPWORDS = set(stopwords)
    return _STOPWORDS


def _get_nlp():
    global _NLP
    if _NLP is None:
        with _NLP_LOCK:
            if _NLP is None:
                last_error = None
                for model_name in _SPACY_MODEL_NAMES:
                    try:
                        _NLP = spacy.load(model_name)
                        break
                    except Exception as exc:  # pragma: no cover - fallback path
                        last_error = exc
                if _NLP is None:
                    raise RuntimeError(
                        "GeoStoryMetric requires a spaCy English model "
                        f"({', '.join(_SPACY_MODEL_NAMES)})"
                    ) from last_error
    return _NLP


def _normalize_response_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _tokenize_for_uniqueness(text: str, stopwords: Set[str]) -> List[str]:
    return [token for token in text.lower().split() if token and token not in stopwords]


def _informativeness_count(text: str, city_name: str, country_name: str) -> float:
    if not text.strip():
        return 0.0

    nlp = _get_nlp()
    document = nlp(text.replace("||||", "\n"))
    entities = {
        entity.text.strip().strip("the").strip()
        for entity in document.ents
        if entity.label_ in _LOCATION_ENTITY_LABELS and entity.text.strip()
    }
    entities.discard(city_name.strip())
    entities.discard(country_name.strip())
    return float(len(entities))


class GeoStoryMetric(EvaluateInstancesMetric):
    """Compute uniqueness and informativeness over a run's generated outputs."""

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        responses: List[str] = []
        city_names: List[str] = []
        country_names: List[str] = []

        for request_state in request_states:
            if request_state.request_mode == "calibration":
                continue
            assert request_state.result is not None
            response_text = _normalize_response_text(request_state.result.completions[0].text)
            extra_data = request_state.instance.extra_data or {}
            responses.append(response_text)
            city_names.append(str(extra_data.get("city_name", "")))
            country_names.append(str(extra_data.get("country_name", "")))

        if not responses:
            return [
                Stat(MetricName("geo_story_uniqueness")).add(0.0),
                Stat(MetricName("geo_story_informativeness")).add(0.0),
            ]

        stopwords = _get_stopwords()
        tokenized_responses = [_tokenize_for_uniqueness(response, stopwords) for response in responses]
        document_frequency: Counter[str] = Counter()
        for tokens in tokenized_responses:
            for token in set(tokens):
                document_frequency[token] += 1

        total_documents = len(tokenized_responses)
        uniqueness_scores: List[float] = []
        for tokens in tokenized_responses:
            if not tokens:
                uniqueness_scores.append(0.0)
                continue
            score = sum(1.0 / document_frequency[token] for token in tokens) / len(tokens)
            uniqueness_scores.append(score * total_documents)

        informativeness_scores = [
            _informativeness_count(response, city_name, country_name)
            for response, city_name, country_name in zip(responses, city_names, country_names)
        ]

        return [
            Stat(MetricName("geo_story_uniqueness")).add(mean(uniqueness_scores)),
            Stat(MetricName("geo_story_informativeness")).add(mean(informativeness_scores)),
        ]
