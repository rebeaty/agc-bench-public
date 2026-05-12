"""Upstream-style automatic metrics for the C3 crosstalk benchmark."""

from collections import Counter
from statistics import mean
from typing import List, Sequence

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.gleu_score import corpus_gleu

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat


def _chars(text: str) -> List[str]:
    # The upstream evaluator scores Chinese continuations at the character level.
    return list(text.strip())


def _ngram_counter(tokens: Sequence[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _rouge_n_f1(reference_tokens: Sequence[str], hypothesis_tokens: Sequence[str], n: int) -> float:
    reference_counts = _ngram_counter(reference_tokens, n)
    hypothesis_counts = _ngram_counter(hypothesis_tokens, n)
    if not reference_counts or not hypothesis_counts:
        return 0.0

    overlap = sum((reference_counts & hypothesis_counts).values())
    if overlap == 0:
        return 0.0

    precision = overlap / sum(hypothesis_counts.values())
    recall = overlap / sum(reference_counts.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_length(reference_tokens: Sequence[str], hypothesis_tokens: Sequence[str]) -> int:
    if not reference_tokens or not hypothesis_tokens:
        return 0

    previous_row = [0] * (len(hypothesis_tokens) + 1)
    for reference_token in reference_tokens:
        current_row = [0]
        for index, hypothesis_token in enumerate(hypothesis_tokens, start=1):
            if reference_token == hypothesis_token:
                current_row.append(previous_row[index - 1] + 1)
            else:
                current_row.append(max(previous_row[index], current_row[-1]))
        previous_row = current_row
    return previous_row[-1]


def _rouge_l_f1(reference_tokens: Sequence[str], hypothesis_tokens: Sequence[str]) -> float:
    if not reference_tokens or not hypothesis_tokens:
        return 0.0

    lcs = _lcs_length(reference_tokens, hypothesis_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(hypothesis_tokens)
    recall = lcs / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _distinct_n(candidates: Sequence[Sequence[str]], n: int) -> float:
    total = 0
    unique = set()

    for tokens in candidates:
        if len(tokens) < n:
            continue
        for index in range(len(tokens) - n + 1):
            ngram = tuple(tokens[index : index + n])
            unique.add(ngram)
            total += 1

    if total == 0:
        return 0.0
    return len(unique) / total


class C3CrosstalkMetric(EvaluateInstancesMetric):
    """Port the released C3 machine metrics into HELM's set-level metric API."""

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        references: List[List[str]] = []
        candidates: List[List[str]] = []

        for request_state in request_states:
            if request_state.request_mode == "calibration":
                continue
            if not request_state.instance.references:
                continue

            assert request_state.result is not None
            generated_tokens = _chars(request_state.result.completions[0].text)
            reference_tokens = _chars(request_state.instance.references[0].output.text)

            candidates.append(generated_tokens)
            references.append(reference_tokens)

        if not references or not candidates:
            metric_names = [
                "bleu_1",
                "bleu_2",
                "bleu_4",
                "gleu",
                "rouge_1",
                "rouge_2",
                "rouge_l",
                "distinct_1",
                "distinct_2",
            ]
            return [Stat(MetricName(metric_name)).add(0.0) for metric_name in metric_names]

        smoothing = SmoothingFunction().method1
        bleu_references = [[reference] for reference in references]

        bleu_1 = corpus_bleu(bleu_references, candidates, weights=(1.0, 0.0, 0.0, 0.0), smoothing_function=smoothing)
        bleu_2 = corpus_bleu(bleu_references, candidates, weights=(0.5, 0.5, 0.0, 0.0), smoothing_function=smoothing)
        bleu_4 = corpus_bleu(
            bleu_references,
            candidates,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing,
        )
        gleu = corpus_gleu(bleu_references, candidates)

        rouge_1 = mean(_rouge_n_f1(reference, candidate, 1) for reference, candidate in zip(references, candidates))
        rouge_2 = mean(_rouge_n_f1(reference, candidate, 2) for reference, candidate in zip(references, candidates))
        rouge_l = mean(_rouge_l_f1(reference, candidate) for reference, candidate in zip(references, candidates))

        return [
            Stat(MetricName("bleu_1")).add(bleu_1),
            Stat(MetricName("bleu_2")).add(bleu_2),
            Stat(MetricName("bleu_4")).add(bleu_4),
            Stat(MetricName("gleu")).add(gleu),
            Stat(MetricName("rouge_1")).add(rouge_1),
            Stat(MetricName("rouge_2")).add(rouge_2),
            Stat(MetricName("rouge_l")).add(rouge_l),
            Stat(MetricName("distinct_1")).add(_distinct_n(candidates, 1)),
            Stat(MetricName("distinct_2")).add(_distinct_n(candidates, 2)),
        ]
