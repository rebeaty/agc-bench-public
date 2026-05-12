"""Compute semantic association-distance metrics for divergent-thinking tasks."""

import os
import re
from typing import Dict, List, Optional

import numpy as np

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

_GLOVE_CACHE: Optional[Dict[str, np.ndarray]] = None
_MEAN_VEC: Optional[np.ndarray] = None

_GLOVE_PATHS = [
    os.path.join(os.path.dirname(__file__), "..", "data", "glove.6B.300d.txt"),
    os.path.expanduser("~/data/glove.6B.300d.txt"),
    "/tmp/glove.6B.300d.txt",
]


def _load_glove() -> Dict[str, np.ndarray]:
    global _GLOVE_CACHE, _MEAN_VEC
    if _GLOVE_CACHE is not None:
        return _GLOVE_CACHE

    glove_path: Optional[str] = None
    for p in _GLOVE_PATHS:
        if os.path.exists(p):
            glove_path = p
            break

    if glove_path is None:
        _GLOVE_CACHE = {}
        _MEAN_VEC = np.zeros(300, dtype=np.float32)
        return _GLOVE_CACHE

    vectors: Dict[str, np.ndarray] = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            vectors[word] = vec

    _GLOVE_CACHE = vectors
    _MEAN_VEC = np.mean(list(vectors.values()), axis=0) if vectors else np.zeros(300, dtype=np.float32)
    return _GLOVE_CACHE


def _get_vec(word: str, glove: Dict[str, np.ndarray]) -> np.ndarray:
    global _MEAN_VEC
    vec = glove.get(word.lower())
    if vec is None:
        return _MEAN_VEC if _MEAN_VEC is not None else np.zeros(300, dtype=np.float32)
    return vec


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(1.0 - np.dot(a, b) / denom)


def _parse_chains(text: str) -> List[List[str]]:
    """Extract word lists from 'Chain N: word (reason) → word (reason) → ...' format."""
    chains: List[List[str]] = []
    for line in text.splitlines():
        line = line.strip()
        match = re.match(r"chain\s*\d+\s*:\s*(.+)", line, re.IGNORECASE)
        if not match:
            continue
        chain_text = match.group(1)
        # Split on arrow variants
        steps = re.split(r"→|->|--?>", chain_text)
        words: List[str] = []
        for step in steps:
            step = step.strip()
            # Remove parenthetical reasons: "word (reason)"
            word_match = re.match(r"^\[?(\w[\w'-]*)\]?", step)
            if word_match:
                words.append(word_match.group(1).lower())
        if words:
            chains.append(words)
    return chains


def _chain_association_distance(chain: List[str], glove: Dict[str, np.ndarray]) -> float:
    """Average cumulative cosine distance per position across the chain."""
    if len(chain) < 2:
        return 0.0
    distances: List[float] = []
    for i in range(1, len(chain)):
        cumulative = sum(_cosine_distance(_get_vec(chain[i], glove), _get_vec(chain[j], glove)) for j in range(i))
        distances.append(cumulative / i)
    return sum(distances) / len(distances)


class AssociationDistanceMetric(Metric):
    """
    Semantic association distance using GloVe 6B 300d embeddings.
    Measures how far each new word in a PACE chain ventures from all previous words.
    Higher score = more creative/distant associations.
    """

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        completion = request_state.result.completions[0].text.strip()

        glove = _load_glove()
        chains = _parse_chains(completion)

        if not chains:
            return [Stat(MetricName("association_distance")).add(0.0)]

        chain_scores = [_chain_association_distance(chain, glove) for chain in chains]
        score = sum(chain_scores) / len(chain_scores)

        return [Stat(MetricName("association_distance")).add(score)]
