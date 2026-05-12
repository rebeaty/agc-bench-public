"""SCOPE automatic metrics over extracted simile vehicles."""

from __future__ import annotations

import os
import re
import string
from statistics import mean
from typing import List

import torch
from bert_score import score as bert_score
from nltk.translate.bleu_score import corpus_bleu

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat

_WHITESPACE_RE = re.compile(r"\s+")
_CONNECTOR_RE = re.compile(r"\b(?:like|as)\b", re.IGNORECASE)
_LEADING_ARTICLE_RE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)
_TRAILING_PUNCTUATION = " \t\r\n" + string.punctuation + "“”‘’"


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", (text or "").strip())


def _find_case_insensitive(haystack: str, needle: str) -> int:
    return haystack.lower().find(needle.lower())


def _clean_vehicle(text: str) -> str:
    cleaned = _normalize(text).strip(_TRAILING_PUNCTUATION)
    cleaned = _LEADING_ARTICLE_RE.sub("", cleaned)
    return cleaned.strip(_TRAILING_PUNCTUATION)


def _extract_vehicle(prediction: str, extra_data: dict) -> str:
    text = _normalize(prediction)
    if not text:
        return ""

    prefix = _normalize(str(extra_data.get("literal_prefix", "")))
    suffix = _normalize(str(extra_data.get("literal_suffix", "")))

    segment = text
    if prefix:
        prefix_index = _find_case_insensitive(text, prefix)
        if prefix_index != -1:
            segment = text[prefix_index + len(prefix) :].strip()

    if suffix:
        suffix_index = _find_case_insensitive(segment, suffix)
        if suffix_index > 0:
            segment = segment[:suffix_index].strip()

    connector_matches = list(_CONNECTOR_RE.finditer(segment))
    if connector_matches:
        segment = segment[connector_matches[-1].end() :].strip()

    return _clean_vehicle(segment)


class SimileGenerationMetric(EvaluateInstancesMetric):
    """
    Port the released SCOPE automatic eval contract into HELM.

    The paper reports BLEU-1, BLEU-2, and BERTScore on extracted VEHICLE spans
    rather than on the full rewritten sentence. The public repo ships the gold
    vehicle labels in `human_labels.csv`; this metric extracts the generated
    vehicle from the model output and scores it against those references.
    """

    def __init__(self, bert_score_model: str = "roberta-large"):
        self.bert_score_model = bert_score_model

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        bleu_references: List[List[List[str]]] = []
        bleu_candidates: List[List[str]] = []
        bert_candidates: List[str] = []
        bert_references: List[List[str]] = []
        parsed_flags: List[float] = []

        for request_state in request_states:
            if request_state.request_mode == "calibration":
                continue
            assert request_state.result is not None
            if not request_state.result.completions:
                continue

            prediction = request_state.result.completions[0].text
            extra_data = request_state.instance.extra_data or {}
            vehicle_prediction = _extract_vehicle(prediction, extra_data)
            references = [
                _clean_vehicle(reference.output.text)
                for reference in request_state.instance.references
                if _clean_vehicle(reference.output.text)
            ]

            parsed = 1.0 if vehicle_prediction else 0.0
            parsed_flags.append(parsed)
            if not vehicle_prediction or not references:
                continue

            bleu_references.append([reference.split() for reference in references])
            bleu_candidates.append(vehicle_prediction.split())
            bert_candidates.append(vehicle_prediction)
            bert_references.append(references)

        if not bleu_candidates:
            return [
                Stat(MetricName("simile_generation_bleu_1")).add(0.0),
                Stat(MetricName("simile_generation_bleu_2")).add(0.0),
                Stat(MetricName("simile_generation_bert_score")).add(0.0),
                Stat(MetricName("simile_generation_vehicle_parsed_rate")).add(mean(parsed_flags) if parsed_flags else 0.0),
            ]

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        device = os.environ.get("AGC_BERT_SCORE_DEVICE", "").strip()
        if not device:
            device = os.environ.get("BERT_SCORE_DEVICE", "").strip()
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        _, _, bert_f1 = bert_score(
            bert_candidates,
            bert_references,
            lang="en",
            model_type=self.bert_score_model,
            rescale_with_baseline=True,
            verbose=False,
            device=device,
        )

        return [
            Stat(MetricName("simile_generation_bleu_1")).add(
                float(corpus_bleu(bleu_references, bleu_candidates, weights=(1.0, 0.0, 0.0, 0.0)) * 100.0)
            ),
            Stat(MetricName("simile_generation_bleu_2")).add(
                float(corpus_bleu(bleu_references, bleu_candidates, weights=(0.0, 1.0, 0.0, 0.0)) * 100.0)
            ),
            Stat(MetricName("simile_generation_bert_score")).add(float(bert_f1.mean().item())),
            Stat(MetricName("simile_generation_vehicle_parsed_rate")).add(mean(parsed_flags) if parsed_flags else 0.0),
        ]
