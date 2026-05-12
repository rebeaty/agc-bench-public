"""MoPS set-level diversity metrics using the upstream embedding + t-SNE recipe."""

import threading
from typing import List, Optional

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from sklearn.manifold import TSNE
import torch
from transformers import AutoModel, AutoTokenizer

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat


class MoPSDiversityMetric(EvaluateInstancesMetric):
    """Compute the MoPS breadth and density scores over a run's generated premises."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        tsne_random_state: int = 42,
        perplexity: int = 50,
        num_bins: int = 10,
    ):
        super().__init__()
        self.model_name = model_name
        self.tsne_random_state = tsne_random_state
        self.perplexity = perplexity
        self.num_bins = num_bins
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModel] = None
        self._lock = threading.Lock()

    def _load_encoder(self):
        """Returns the active embedder. Routes through the embedder factory
        (Gemini by default). Original Auto* signature was (tokenizer, model);
        we now return a single SentenceTransformer-compatible object."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from metrics.embedder_factory import get_embedder
                    self._model = get_embedder(self.model_name)
        return self._model

    def _encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)
        embedder = self._load_encoder()
        embeddings = np.asarray(
            embedder.encode(list(texts), convert_to_numpy=True),
            dtype=np.float32,
        )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return embeddings / norms

    @staticmethod
    def _convex_hull_area(points: np.ndarray) -> float:
        if points.shape[0] < 3:
            return 0.0
        try:
            hull = ConvexHull(points)
        except QhullError:
            return 0.0
        return float(hull.volume)

    def _tsne_reduce(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.shape[0] < 2:
            return np.zeros((embeddings.shape[0], 2))

        effective_perplexity = float(min(self.perplexity, max(1, embeddings.shape[0] - 1)))
        reducer = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            random_state=self.tsne_random_state,
            init="random",
            learning_rate="auto",
        )
        return reducer.fit_transform(embeddings)

    def _density_score(self, points: np.ndarray) -> float:
        if points.shape[0] == 0:
            return 0.0

        x_min = float(np.min(points[:, 0]))
        x_max = float(np.max(points[:, 0]))
        y_min = float(np.min(points[:, 1]))
        y_max = float(np.max(points[:, 1]))

        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5
        if y_min == y_max:
            y_min -= 0.5
            y_max += 0.5

        hist, _, _ = np.histogram2d(
            points[:, 0],
            points[:, 1],
            bins=self.num_bins,
            range=[[x_min, x_max], [y_min, y_max]],
        )
        hist = hist.T

        mask = np.ones_like(hist, dtype=bool)
        for row_idx, row in enumerate(hist):
            non_zero_indices = np.where(row != 0)[0]
            if non_zero_indices.size == 0:
                continue
            first_non_zero = int(non_zero_indices[0])
            last_non_zero = int(non_zero_indices[-1])
            mask[row_idx, first_non_zero : last_non_zero + 1] = False

        unmasked_elements = hist[~mask]
        if unmasked_elements.size == 0:
            return 0.0
        return float(np.std(unmasked_elements.tolist()))

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        premises: List[str] = []
        for request_state in request_states:
            if request_state.request_mode == "calibration":
                continue
            assert request_state.result is not None
            completion = request_state.result.completions[0].text.strip()
            if completion:
                premises.append(completion)

        if not premises:
            return [
                Stat(MetricName("breadth_score")).add(0.0),
                Stat(MetricName("density_score")).add(0.0),
            ]

        embeddings = self._encode(premises)
        points = self._tsne_reduce(embeddings)

        return [
            Stat(MetricName("breadth_score")).add(self._convex_hull_area(points)),
            Stat(MetricName("density_score")).add(self._density_score(points)),
        ]
