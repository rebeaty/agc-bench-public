"""Automatic SAQA metrics aligned with the paper's overlap-based eval slice."""

from __future__ import annotations

from statistics import mean
from typing import List

from nltk.translate.meteor_score import single_meteor_score
from nltk.translate.meteor_score import wordnet as nltk_wordnet
from nltk.stem import PorterStemmer
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat


class _EmptyWordNet:
    def synsets(self, word: str) -> list:
        return []


class ScienceAnalogiesAutomaticMetric(EvaluateInstancesMetric):
    """
    Benchmark-specific automatic metrics for SAQA.

    The paper's main automatic metric is BLEURT, but the official BLEURT scorer
    is not packaged in this environment. This metric restores the paper's two
    supporting automatic metrics, ROUGE-L and METEOR, and retains BLEU-4 as a
    lexical diagnostic. Each score is computed against the best matching gold
    explanation for the instance and averaged over the evaluated slice.
    """

    def __init__(self) -> None:
        self._bleu = BLEU(effective_order=True)
        self._rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self._stemmer = PorterStemmer()
        self._wordnet = nltk_wordnet if self._wordnet_available() else _EmptyWordNet()

    @staticmethod
    def _wordnet_available() -> bool:
        try:
            nltk_wordnet.synsets("analogy")
            return True
        except LookupError:
            return False

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join((text or "").strip().split())

    def _sentence_bleu(self, prediction: str, references: List[str]) -> float:
        if not prediction or not references:
            return 0.0
        return float(self._bleu.sentence_score(prediction, references).score / 100.0)

    def _rouge_l(self, prediction: str, references: List[str]) -> float:
        if not prediction or not references:
            return 0.0
        return max(
            float(self._rouge.score(reference, prediction)["rougeL"].fmeasure)
            for reference in references
        )

    def _meteor(self, prediction: str, references: List[str]) -> float:
        if not prediction or not references:
            return 0.0
        prediction_tokens = prediction.split()
        return max(
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

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        bleu_scores: List[float] = []
        rouge_scores: List[float] = []
        meteor_scores: List[float] = []
        response_word_counts: List[int] = []

        for request_state in request_states:
            if request_state.request_mode == "calibration":
                continue
            assert request_state.result is not None
            if not request_state.result.completions:
                continue

            prediction = self._clean_text(request_state.result.completions[0].text)
            references = [
                self._clean_text(reference.output.text)
                for reference in request_state.instance.references
                if self._clean_text(reference.output.text)
            ]
            if not prediction or not references:
                continue

            bleu_scores.append(self._sentence_bleu(prediction, references))
            rouge_scores.append(self._rouge_l(prediction, references))
            meteor_scores.append(self._meteor(prediction, references))
            response_word_counts.append(len(prediction.split()))

        return [
            Stat(MetricName("science_analogies_bleu_4")).add(mean(bleu_scores) if bleu_scores else 0.0),
            Stat(MetricName("science_analogies_rouge_l")).add(mean(rouge_scores) if rouge_scores else 0.0),
            Stat(MetricName("science_analogies_meteor")).add(mean(meteor_scores) if meteor_scores else 0.0),
            Stat(MetricName("science_analogies_avg_response_words")).add(
                mean(response_word_counts) if response_word_counts else 0.0
            ),
        ]
