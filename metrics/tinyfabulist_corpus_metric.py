"""Corpus-level TinyFabulist metrics mirroring the upstream automatic evaluator."""

from __future__ import annotations

import re
from typing import List

from nltk.tokenize import wordpunct_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat


_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_VOWEL_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)


def _distinct_n(text: str, n: int) -> float:
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


def _count_sentences(text: str) -> int:
    count = len([segment for segment in re.split(r"[.!?]+", text) if segment.strip()])
    return max(count, 1)


def _count_syllables_in_word(word: str) -> int:
    lowered = word.lower()
    if not lowered:
        return 0
    groups = _VOWEL_RE.findall(lowered)
    syllables = len(groups)
    if lowered.endswith("e") and not lowered.endswith(("le", "ye")) and syllables > 1:
        syllables -= 1
    return max(syllables, 1)


def _flesch_reading_ease(text: str) -> float:
    words = _WORD_RE.findall(text)
    if not words:
        return 0.0

    sentences = _count_sentences(text)
    syllables = sum(_count_syllables_in_word(word) for word in words)
    words_per_sentence = len(words) / sentences
    syllables_per_word = syllables / len(words)
    return 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word


def _compute_self_bleu(texts: List[str]) -> float:
    if len(texts) <= 1:
        return 0.0
    smoothie = SmoothingFunction().method1
    tokenized = [wordpunct_tokenize(text.lower()) for text in texts]
    scores: List[float] = []
    for index, hypothesis in enumerate(tokenized):
        references = tokenized[:index] + tokenized[index + 1 :]
        if not hypothesis or not references:
            continue
        scores.append(sentence_bleu(references, hypothesis, smoothing_function=smoothie))
    return sum(scores) / len(scores) if scores else 0.0


class TinyFabulistCorpusMetric(EvaluateInstancesMetric):
    """Compute the paper's reference-free corpus metrics over generated fables."""

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        fables: List[str] = []
        distinct_1_scores: List[float] = []
        flesch_scores: List[float] = []

        for request_state in request_states:
            if request_state.request_mode == "calibration":
                continue
            assert request_state.result is not None
            completion = request_state.result.completions[0].text.strip()
            if not completion:
                continue
            fables.append(completion)
            distinct_1_scores.append(_distinct_n(completion, 1))
            flesch_scores.append(_flesch_reading_ease(completion))

        self_bleu = _compute_self_bleu(fables)
        distinct_1 = sum(distinct_1_scores) / len(distinct_1_scores) if distinct_1_scores else 0.0
        flesch = sum(flesch_scores) / len(flesch_scores) if flesch_scores else 0.0

        return [
            Stat(MetricName("self_bleu")).add(self_bleu),
            Stat(MetricName("distinct_1")).add(distinct_1),
            Stat(MetricName("flesch_reading_ease")).add(flesch),
        ]
