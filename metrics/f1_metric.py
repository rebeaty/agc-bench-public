"""Benchmark-specific token F1 for CHIMERA recombination extraction."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


_ANSWER_BLOCK_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _canonicalize_text(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    answer_match = _ANSWER_BLOCK_RE.search(text)
    if answer_match:
        candidate = answer_match.group(1).strip()
    else:
        json_match = _JSON_BLOCK_RE.search(text)
        candidate = json_match.group(0).strip() if json_match else text.strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return candidate

    if isinstance(parsed, dict):
        return json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    return candidate


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _TOKEN_RE.findall(prediction.lower())
    ref_tokens = _TOKEN_RE.findall(reference.lower())

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum(min(pred_counts[token], ref_counts[token]) for token in pred_counts)
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


class F1Metric(Metric):
    """Emit token-level F1 against the structured JSON reference."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None

        completion = request_state.result.completions[0].text
        prediction = _canonicalize_text(completion)
        references = [ref.output.text for ref in request_state.instance.references if ref.output.text]

        if not references:
            score = 0.0
        else:
            score = max(_token_f1(prediction, _canonicalize_text(reference)) for reference in references)

        return [Stat(MetricName("f1")).add(float(score))]
