"""PoetMT-specific automatic metrics with light translation-only normalization."""

from __future__ import annotations

import re
from statistics import mean
from typing import List, Sequence

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat


_META_LINE_RE = re.compile(
    r"^(?:"
    r"here(?:'s| is)\b|"
    r"of course\b|"
    r"certainly\b|"
    r"sure\b|"
    r"translation\b|"
    r"english translation\b|"
    r"literal translation\b|"
    r"analysis\b|"
    r"explanation\b|"
    r"notes?\b|"
    r"commentary\b|"
    r"pinyin\b|"
    r"option \d+\b|"
    r"version \d+\b"
    r")",
    re.IGNORECASE,
)


def normalize_poetmt_translation(text: str) -> str:
    """Strip common chatty preambles so scoring focuses on the translated poem."""
    cleaned = (text or "").replace("\r\n", "\n").strip()
    if not cleaned:
        return ""

    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```$", "", cleaned).strip()

    label_match = re.search(
        r"(?:^|\n)(?:english\s+)?translation\s*[:：]\s*",
        cleaned,
        flags=re.IGNORECASE,
    )
    if label_match:
        cleaned = cleaned[label_match.end() :].strip()

    lines = []
    kept_any = False
    for raw_line in cleaned.splitlines():
        line = raw_line.strip().strip('"').strip("'").strip()
        if not line:
            if kept_any and lines and lines[-1] != "":
                lines.append("")
            continue
        if _META_LINE_RE.match(line):
            if kept_any:
                break
            continue
        lines.append(line)
        kept_any = True

    normalized = "\n".join(lines).strip()
    return normalized or cleaned


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?|[^\w\s]", text.lower())


class PoetMTAutomaticMetric(EvaluateInstancesMetric):
    """Compute BLEU on lightly normalized poem-only outputs."""

    def __init__(self) -> None:
        self._smoothie = SmoothingFunction().method1

    def _bleu(self, prediction: str, reference: str, weights: Sequence[float]) -> float:
        prediction_tokens = _tokenize(prediction)
        reference_tokens = _tokenize(reference)
        if not prediction_tokens or not reference_tokens:
            return 0.0
        return float(
            sentence_bleu(
                [reference_tokens],
                prediction_tokens,
                weights=weights,
                smoothing_function=self._smoothie,
            )
        )

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        bleu_1_scores: List[float] = []
        bleu_4_scores: List[float] = []

        for request_state in request_states:
            if request_state.request_mode == "calibration":
                continue
            assert request_state.result is not None
            if not request_state.result.completions:
                continue

            prediction = normalize_poetmt_translation(request_state.result.completions[0].text)
            references = [
                reference.output.text.strip()
                for reference in request_state.instance.references
                if reference.output.text.strip()
            ]
            if not prediction or not references:
                continue

            reference = references[0]
            bleu_1_scores.append(self._bleu(prediction, reference, (1.0, 0.0, 0.0, 0.0)))
            bleu_4_scores.append(self._bleu(prediction, reference, (0.25, 0.25, 0.25, 0.25)))

        return [
            Stat(MetricName("bleu_1")).add(mean(bleu_1_scores) if bleu_1_scores else 0.0),
            Stat(MetricName("bleu_4")).add(mean(bleu_4_scores) if bleu_4_scores else 0.0),
        ]
