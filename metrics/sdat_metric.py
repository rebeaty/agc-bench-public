"""S-DAT metric using Granite multilingual embeddings and released calibration."""

from __future__ import annotations

import os
import re
import threading
from typing import List, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

_MODEL_NAME = "ibm-granite/granite-embedding-278m-multilingual"

# The paper states that S-DAT linearly calibrates Granite distances to the
# original DAT/GloVe scale. The released paper does not publish coefficients, so
# these are recovered from the official Study 2 rescoring bundle on OSF using
# the released Model_SDAT1_Score targets and the exact Granite CLS-pooling
# configuration from the model card artifacts.
_CALIBRATION_SCALE = 2.979316013661235
_CALIBRATION_BIAS = -13.889735856447274

# Empirical percentile anchors from the released Study 2 S-DAT rescoring
# distribution (N=8,498 valid Model_SDAT1_Score values).
_PERCENTILE_ANCHORS = [
    (0.0, 54.49206349),
    (1.0, 67.30158730),
    (5.0, 72.17222222),
    (10.0, 73.98412698),
    (25.0, 76.44444444),
    (50.0, 79.11111111),
    (75.0, 82.03174603),
    (90.0, 84.87301587),
    (95.0, 86.58730159),
    (99.0, 89.74603175),
    (100.0, 96.09523810),
]

_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•]+|\d+\s*[\).\:-])\s*")
_INLINE_NUMBERING_RE = re.compile(r"(?<!^)\s+(?=\d+\s*[\).\:-]\s*)")
_WHITESPACE_RE = re.compile(r"\s+")


def _metric_device() -> str:
    return os.environ.get("SDAT_DEVICE_OVERRIDE", "cpu")


class _GraniteEncoder:
    """Embedding encoder for S-DAT. Routes through the active embedder
    backend (Gemini by default; see metrics.embedder_factory). The legacy
    Granite-direct path is preserved when AGC_EMBEDDING_BACKEND is forced
    to sentence_transformers / local — in that case we still hit Granite
    via SentenceTransformer if the model is loadable, otherwise fall back
    to whatever the factory hands us. CLS-pool / L2-normalize is delegated
    to the embedder."""

    def __init__(self):
        self._lock = threading.Lock()
        self._embedder = None

    def _load(self):
        if self._embedder is None:
            with self._lock:
                if self._embedder is None:
                    from metrics.embedder_factory import get_embedder
                    self._embedder = get_embedder(_MODEL_NAME)
        return self._embedder

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        embedder = self._load()
        embeddings = embedder.encode(list(texts), convert_to_numpy=True)
        # Defensive L2-normalize in case the backend doesn't return unit vectors.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (embeddings / norms).astype(np.float32)


_ENCODER = _GraniteEncoder()


def _normalize_item(item: str) -> str:
    cleaned = item.replace("\u200b", "")
    cleaned = _LIST_MARKER_RE.sub("", cleaned)
    cleaned = cleaned.strip().strip(".,;:!?()[]{}\"'")
    cleaned = _WHITESPACE_RE.sub(" ", cleaned.lower())
    return cleaned


def _parse_items(text: str) -> List[str]:
    if not text.strip():
        return []

    normalized_text = _INLINE_NUMBERING_RE.sub("\n", text)
    pieces = re.split(r"[\n,;]+", normalized_text)
    items: List[str] = []

    for piece in pieces:
        item = _normalize_item(piece)
        if item:
            items.append(item)
        if len(items) == 10:
            break

    return items


def _is_single_word(item: str) -> bool:
    return bool(re.fullmatch(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", item, re.UNICODE))


def _raw_sdat_score(items: Sequence[str]) -> float:
    if len(items) < 2:
        return 0.0

    embeddings = _ENCODER.encode(items)
    dissimilarities: List[float] = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            dissimilarities.append(1.0 - float(np.dot(embeddings[i], embeddings[j])))
    return float(np.mean(dissimilarities) * 100.0) if dissimilarities else 0.0


def _calibrate_score(raw_score: float) -> float:
    return (_CALIBRATION_SCALE * raw_score) + _CALIBRATION_BIAS


def _estimate_percentile(score: float) -> float:
    if score <= _PERCENTILE_ANCHORS[0][1]:
        return 0.0
    if score >= _PERCENTILE_ANCHORS[-1][1]:
        return 100.0

    for (left_pct, left_score), (right_pct, right_score) in zip(
        _PERCENTILE_ANCHORS,
        _PERCENTILE_ANCHORS[1:],
        strict=False,
    ):
        if left_score <= score <= right_score:
            if right_score == left_score:
                return right_pct
            fraction = (score - left_score) / (right_score - left_score)
            return float(left_pct + fraction * (right_pct - left_pct))

    return 100.0


class SDATMetric(Metric):
    """Score one S-DAT response with Granite embeddings plus released calibration."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None

        completion = request_state.result.completions[0].text.strip()
        items = _parse_items(completion)
        parsed_item_count = len(items)
        unique_item_count = len(dict.fromkeys(items))
        single_word_item_count = sum(1 for item in items if _is_single_word(item))
        exactly_ten_items = 1.0 if parsed_item_count == 10 else 0.0
        minimum_seven_items = 1.0 if parsed_item_count >= 7 else 0.0

        if parsed_item_count >= 7:
            raw_score = _raw_sdat_score(items)
            sdat_score = _calibrate_score(raw_score)
            percentile_estimate = _estimate_percentile(sdat_score)
        else:
            raw_score = 0.0
            sdat_score = 0.0
            percentile_estimate = 0.0

        return [
            Stat(MetricName("sdat_score")).add(float(sdat_score)),
            Stat(MetricName("sdat_raw_score")).add(float(raw_score)),
            Stat(MetricName("sdat_percentile_estimate")).add(float(percentile_estimate)),
            Stat(MetricName("parsed_item_count")).add(float(parsed_item_count)),
            Stat(MetricName("unique_item_count")).add(float(unique_item_count)),
            Stat(MetricName("single_word_item_count")).add(float(single_word_item_count)),
            Stat(MetricName("exactly_ten_items")).add(exactly_ten_items),
            Stat(MetricName("minimum_seven_items")).add(minimum_seven_items),
        ]
