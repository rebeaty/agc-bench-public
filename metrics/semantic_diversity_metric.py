"""
SemanticDiversityMetric: per-instance semantic diversity via DSI.

For DAT-style responses (10 unrelated nouns), computes the mean pairwise
cosine distance between word embeddings within a single response.
This is the within-response analog of semantic diversity: higher = more creative.

Reuses segmentation and DSI helpers from creativity_score_metric.py.
"""

import threading
from typing import List, Optional

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

from metrics.creativity_score_metric import _segment_response, _compute_dsi


class SemanticDiversityMetric(Metric):
    """Mean pairwise cosine distance between segment embeddings of a single response.

    Args:
        model_name: SentenceTransformer model identifier.
        task: Segmentation scheme passed to _segment_response.
              Use "dat" for DAT (word lists), "cwt" for sentences, etc.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        task: str = "dat",
    ):
        super().__init__()
        self.model_name = model_name
        self.task = task
        self._model: Optional[object] = None
        self._lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from metrics.embedder_factory import get_embedder
                    self._model = get_embedder(self.model_name)
        return self._model

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        completion = request_state.result.completions[0].text.strip()

        if not completion:
            return [Stat(MetricName("semantic_diversity")).add(0.0)]

        segments = _segment_response(completion, self.task)
        model = self._get_model()
        score = _compute_dsi(segments, model)

        return [Stat(MetricName("semantic_diversity")).add(score)]

