"""Set-level SentenceBERT proxies for the Conceptual Design benchmark."""

from __future__ import annotations

import re
from statistics import mean
from typing import Any, List, Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat


_SOLUTION_SPLIT_RE = re.compile(r"^\s*(?:\d+[\).\:-]?\s*|[-*]\s*)?")


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _extract_solutions(text: str) -> List[str]:
    cleaned = _normalize_text(text)
    if not cleaned:
        return []

    raw_lines = [line.strip() for line in text.splitlines()]
    solutions: List[str] = []
    for line in raw_lines:
        stripped = _SOLUTION_SPLIT_RE.sub("", line).strip()
        stripped = stripped.strip(" \t\r\n-:;")
        if stripped:
            solutions.append(stripped)
    if solutions:
        return solutions

    segments = [segment.strip() for segment in re.split(r"[。.!?]\s+", cleaned) if segment.strip()]
    return segments if segments else [cleaned]


def _mean_pairwise_cosine_distance(matrix: np.ndarray) -> float:
    if len(matrix) <= 1:
        return 0.0
    cosine_similarities = cosine_similarity(matrix)
    distances = 1.0 - cosine_similarities
    upper = np.triu_indices(len(matrix), k=1)
    return float(distances[upper].mean()) if upper[0].size else 0.0


def _mean_best_similarity(source: np.ndarray, targets: np.ndarray) -> float:
    if len(source) == 0 or len(targets) == 0:
        return 0.0
    sims = cosine_similarity(source, targets)
    return float(sims.max(axis=1).mean())


class ConceptualDesignMetric(EvaluateInstancesMetric):
    """Expose set-level similarity and diversity proxies aligned with the paper."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", **_: Any):
        super().__init__()
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from metrics.embedder_factory import get_embedder
            self._model = get_embedder(self.model_name)
        return self._model

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        embedder = self._get_model()
        embeddings = embedder.encode(list(texts), convert_to_numpy=True)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (embeddings / norms).astype(np.float32)

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        requested = [state for state in request_states if state.request_mode != "calibration" and state.result is not None]

        feasibility: List[float] = []
        novelty: List[float] = []
        usefulness: List[float] = []
        parse_rates: List[float] = []
        generated_counts: List[float] = []
        nearest_ref_similarity: List[float] = []
        within_set_diversity: List[float] = []

        for state in requested:
            annotation = (state.annotations or {}).get("conceptual_design_judge", {}) or {}
            if annotation.get("feasibility") is not None:
                feasibility.append(float(annotation["feasibility"]))
            if annotation.get("novelty") is not None:
                novelty.append(float(annotation["novelty"]))
            if annotation.get("usefulness") is not None:
                usefulness.append(float(annotation["usefulness"]))
            parse_rates.append(float(annotation.get("parse_rate", 0.0)))

            completion_text = state.result.completions[0].text if state.result.completions else ""
            solutions = _extract_solutions(completion_text)
            generated_counts.append(float(len(solutions)))
            if not solutions:
                nearest_ref_similarity.append(0.0)
                within_set_diversity.append(0.0)
                continue

            generated_embeddings = self._encode(solutions)
            within_set_diversity.append(_mean_pairwise_cosine_distance(generated_embeddings))

            references = [
                reference.output.text
                for reference in (state.instance.references or [])
                if reference.output.text.strip()
            ]
            if references:
                reference_embeddings = self._encode(references)
                nearest_ref_similarity.append(_mean_best_similarity(reference_embeddings, generated_embeddings))
            else:
                nearest_ref_similarity.append(0.0)

        return [
            Stat(MetricName("conceptual_design_feasibility")).add(mean(feasibility) if feasibility else 0.0),
            Stat(MetricName("conceptual_design_novelty")).add(mean(novelty) if novelty else 0.0),
            Stat(MetricName("conceptual_design_usefulness")).add(mean(usefulness) if usefulness else 0.0),
            Stat(MetricName("conceptual_design_judge_parse_rate")).add(mean(parse_rates) if parse_rates else 0.0),
            Stat(MetricName("conceptual_design_solution_count")).add(mean(generated_counts) if generated_counts else 0.0),
            Stat(MetricName("conceptual_design_nearest_reference_similarity")).add(
                mean(nearest_ref_similarity) if nearest_ref_similarity else 0.0
            ),
            Stat(MetricName("conceptual_design_within_set_diversity")).add(
                mean(within_set_diversity) if within_set_diversity else 0.0
            ),
        ]
