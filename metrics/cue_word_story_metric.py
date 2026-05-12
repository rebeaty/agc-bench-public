"""Automatic cue-word-story diagnostics inspired by the paper's metric stack."""

from __future__ import annotations

import re
from statistics import mean
from typing import List, Sequence

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat

_WORD_RE = re.compile(r"[A-Za-z']+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_STOPWORDS = set(ENGLISH_STOP_WORDS)


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def _split_sentences(text: str) -> List[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    return [segment.strip() for segment in _SENTENCE_SPLIT_RE.split(normalized) if segment.strip()]


def _unique_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output


def _dominant_terms(text: str) -> List[str]:
    return _unique_preserve_order(
        token for token in _tokenize(text) if token not in _STOPWORDS and len(token) > 1
    )


def _average_ngram_diversity(tokens: Sequence[str], max_n: int = 5) -> float:
    if not tokens:
        return 0.0

    scores: List[float] = []
    for n in range(1, max_n + 1):
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            continue
        scores.append(len(set(ngrams)) / len(ngrams))
    return mean(scores) if scores else 0.0


def _mean_pairwise_cosine_distance(matrix: np.ndarray) -> float:
    if len(matrix) <= 1:
        return 0.0
    cosine_similarities = cosine_similarity(matrix)
    distances = 1.0 - cosine_similarities
    upper_indices = np.triu_indices(len(matrix), k=1)
    return float(distances[upper_indices].mean()) if upper_indices[0].size else 0.0


class CueWordStoryMetric(EvaluateInstancesMetric):
    """
    Aggregate automatic diagnostics for cue-word stories.

    This keeps the signal lightweight enough for HELM while restoring benchmark-
    shaped checks that are missing from overlap metrics:
    cue-word inclusion, five-sentence compliance, lexical anti-theme overlap,
    average n-gram diversity, and small-slice novelty/surprise proxies derived
    from the paper's formulas.
    """

    def __init__(self, model_name: str = "tfidf_char_3_5") -> None:
        self.model_name = model_name

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        return vectorizer.fit_transform(texts).toarray()

    def _semantic_dispersion(self, text: str) -> float:
        terms = _dominant_terms(text)
        if len(terms) <= 1:
            return 0.0
        return _mean_pairwise_cosine_distance(self._encode(terms))

    def _surprise(self, text: str) -> float:
        sentence_scores = [self._semantic_dispersion(sentence) for sentence in _split_sentences(text)]
        if len(sentence_scores) <= 1:
            return 0.0
        raw_deltas = [
            abs(sentence_scores[index] - sentence_scores[index - 1])
            for index in range(1, len(sentence_scores))
        ]
        return float((2.0 / len(raw_deltas)) * sum(raw_deltas)) if raw_deltas else 0.0

    def _novelty_scores(self, texts: Sequence[str]) -> List[float]:
        story_representations = [
            " ".join(_dominant_terms(text)) or _normalize_text(text)
            for text in texts
        ]
        if len(story_representations) <= 1:
            return [0.0 for _ in story_representations]

        story_matrix = self._encode(story_representations)
        centroid = story_matrix.mean(axis=0, keepdims=True)
        distances = 1.0 - cosine_similarity(story_matrix, centroid).reshape(-1)
        return [float(2.0 * distance) for distance in distances]

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        responses: List[str] = []
        cue_word_coverages: List[float] = []
        all_cue_word_rates: List[float] = []
        five_sentence_rates: List[float] = []
        boring_theme_overlaps: List[float] = []
        ngram_diversities: List[float] = []
        surprise_scores: List[float] = []

        for request_state in request_states:
            if request_state.request_mode == "calibration":
                continue
            assert request_state.result is not None
            if not request_state.result.completions:
                continue

            response_text = _normalize_text(request_state.result.completions[0].text)
            extra_data = request_state.instance.extra_data or {}
            cue_words = [str(word).lower() for word in extra_data.get("cue_words", [])]
            boring_theme = str(extra_data.get("boring_theme", ""))

            responses.append(response_text)
            tokens = _tokenize(response_text)
            token_set = set(tokens)
            sentences = _split_sentences(response_text)

            coverage = (
                sum(1 for cue_word in cue_words if cue_word in token_set) / len(cue_words)
                if cue_words
                else 0.0
            )
            cue_word_coverages.append(coverage)
            all_cue_word_rates.append(1.0 if cue_words and coverage == 1.0 else 0.0)
            five_sentence_rates.append(1.0 if 0 < len(sentences) <= 5 else 0.0)

            theme_terms = set(_dominant_terms(boring_theme))
            response_terms = set(_dominant_terms(response_text))
            overlap = (
                len(theme_terms & response_terms) / len(theme_terms)
                if theme_terms
                else 0.0
            )
            boring_theme_overlaps.append(overlap)

            ngram_diversities.append(_average_ngram_diversity(tokens))
            surprise_scores.append(self._surprise(response_text))

        novelty_scores = self._novelty_scores(responses) if responses else []

        return [
            Stat(MetricName("cue_word_story_cue_word_coverage")).add(
                mean(cue_word_coverages) if cue_word_coverages else 0.0
            ),
            Stat(MetricName("cue_word_story_all_cue_words_rate")).add(
                mean(all_cue_word_rates) if all_cue_word_rates else 0.0
            ),
            Stat(MetricName("cue_word_story_five_sentence_rate")).add(
                mean(five_sentence_rates) if five_sentence_rates else 0.0
            ),
            Stat(MetricName("cue_word_story_boring_theme_overlap")).add(
                mean(boring_theme_overlaps) if boring_theme_overlaps else 0.0
            ),
            Stat(MetricName("cue_word_story_ngram_diversity")).add(
                mean(ngram_diversities) if ngram_diversities else 0.0
            ),
            Stat(MetricName("cue_word_story_novelty")).add(
                mean(novelty_scores) if novelty_scores else 0.0
            ),
            Stat(MetricName("cue_word_story_surprise")).add(
                mean(surprise_scores) if surprise_scores else 0.0
            ),
        ]
