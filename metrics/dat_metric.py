"""Benchmark-specific metric for the Divergent Association Task (DAT)."""

from __future__ import annotations

import os
import re
from itertools import combinations
from typing import Any, Dict, List, Optional

import numpy as np

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat

_DEFAULT_EMBEDDING_PATH = os.environ.get("DAT_GLOVE_PATH", "data/glove/glove.840B.300d.txt")
_WORD_RE = re.compile(r"[^a-zA-Z- ]+")


def _clean_candidate(word: str) -> Optional[str]:
    clean = _WORD_RE.sub("", word or "").strip().lower()
    if len(clean) <= 1:
        return None
    if " " in clean:
        parts = [part for part in clean.split(" ") if part and part != "a"]
        return parts[0] if parts else None
    return clean


def _parse_candidates(text: str) -> List[str]:
    stripped = (text or "").strip()
    if not stripped:
        return []
    if "\n" in stripped and len([line for line in stripped.split("\n") if line.strip()]) >= 10:
        raw = stripped.split("\n")
    elif "," in stripped:
        raw = stripped.split(",")
    elif "*" in stripped:
        raw = stripped.split("*")
    else:
        raw = stripped.split()
    cleaned: List[str] = []
    for token in raw:
        candidate = _clean_candidate(token)
        if candidate:
            cleaned.append(candidate)
        if len(cleaned) >= 10:
            break
    return cleaned


def _embedding_candidates(word: str) -> List[str]:
    clean = re.sub(r"[^a-zA-Z- ]+", "", word or "").strip().lower()
    if len(clean) <= 1:
        return []
    candidates: List[str] = []
    if " " in clean:
        candidates.append(re.sub(r" +", "-", clean))
        candidates.append(re.sub(r" +", "", clean))
    else:
        candidates.append(clean)
        if "-" in clean:
            candidates.append(re.sub(r"-+", "", clean))
    unique: List[str] = []
    for candidate in candidates:
        if candidate and candidate not in unique:
            unique.append(candidate)
    return unique


def _cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(1.0 - np.dot(vec_a, vec_b) / (norm_a * norm_b))


class DATMetric(EvaluateInstancesMetric):
    """Official-style DAT scorer using GloVe vectors."""

    def __init__(self, embedding_path: str = _DEFAULT_EMBEDDING_PATH, minimum_words: int = 7, **_: Any):
        super().__init__()
        self.embedding_path = embedding_path
        self.minimum_words = int(minimum_words)

    def _load_needed_vectors(self, needed_words: List[str]) -> Dict[str, np.ndarray]:
        needed = set(needed_words)
        vectors: Dict[str, np.ndarray] = {}
        if not needed:
            return vectors
        with open(self.embedding_path, "r", encoding="utf8") as handle:
            for line in handle:
                word, *values = line.rstrip("\n").split(" ")
                if word in needed:
                    vectors[word] = np.asarray(values, dtype="float32")
                    if len(vectors) == len(needed):
                        break
        return vectors

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        candidate_lists: List[List[str]] = []
        needed_words: List[str] = []

        for state in request_states:
            if state.request_mode == "calibration" or state.result is None:
                continue
            text = state.result.completions[0].text if state.result.completions else ""
            candidates = _parse_candidates(text)
            candidate_lists.append(candidates)
            for word in candidates:
                needed_words.extend(_embedding_candidates(word))

        vectors = self._load_needed_vectors(needed_words)

        dat_scores: List[float] = []
        valid_counts: List[float] = []
        scoreable: List[float] = []

        for candidates in candidate_lists:
            uniques: List[str] = []
            for candidate in candidates:
                resolved = None
                for option in _embedding_candidates(candidate):
                    if option in vectors:
                        resolved = option
                        break
                if resolved and resolved not in uniques:
                    uniques.append(resolved)
            valid_counts.append(float(len(uniques)))

            if len(uniques) < self.minimum_words:
                scoreable.append(0.0)
                continue

            subset = uniques[: self.minimum_words]
            distances = [
                _cosine_distance(vectors[word_a], vectors[word_b])
                for word_a, word_b in combinations(subset, 2)
            ]
            if not distances:
                scoreable.append(0.0)
                continue

            dat_scores.append(float(np.mean(distances) * 100.0))
            scoreable.append(1.0)

        mean_score = float(np.mean(dat_scores)) if dat_scores else 0.0
        mean_valid = float(np.mean(valid_counts)) if valid_counts else 0.0
        scoreable_rate = float(np.mean(scoreable)) if scoreable else 0.0

        return [
            Stat(MetricName("dat_score")).add(mean_score),
            Stat(MetricName("dat_valid_word_count")).add(mean_valid),
            Stat(MetricName("dat_scoreable_rate")).add(scoreable_rate),
        ]
