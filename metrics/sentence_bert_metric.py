"""Sentence-BERT similarity metric for reference-based generation tasks."""

import threading
from typing import List

import numpy as np

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class SentenceBertMetric(Metric):
    """Computes sentence-BERT precision, recall, and F1 between prediction and reference.

    Each sentence in the prediction/reference is embedded; precision is the mean
    max-cosine-similarity of each predicted sentence to any reference sentence,
    recall is the mean max-cosine-similarity of each reference sentence to any
    predicted sentence, and F1 is their harmonic mean.
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        super().__init__()
        self._model_name = model_name
        self._model = None
        self._model_lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    from metrics.embedder_factory import get_embedder
                    self._model = get_embedder(self._model_name)
        return self._model

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _sentence_bert_scores(self, prediction: str, reference: str):
        model = self._get_model()

        pred_sents = [s.strip() for s in prediction.split(".") if s.strip()]
        ref_sents = [s.strip() for s in reference.split(".") if s.strip()]

        if not pred_sents:
            pred_sents = [prediction]
        if not ref_sents:
            ref_sents = [reference]

        pred_embs = model.encode(pred_sents, convert_to_numpy=True)
        ref_embs = model.encode(ref_sents, convert_to_numpy=True)

        # precision: for each pred sentence, max similarity to any ref sentence
        precision_scores = []
        for p_emb in pred_embs:
            max_sim = max(self._cosine_similarity(p_emb, r_emb) for r_emb in ref_embs)
            precision_scores.append(max_sim)

        # recall: for each ref sentence, max similarity to any pred sentence
        recall_scores = []
        for r_emb in ref_embs:
            max_sim = max(self._cosine_similarity(r_emb, p_emb) for p_emb in pred_embs)
            recall_scores.append(max_sim)

        precision = float(np.mean(precision_scores))
        recall = float(np.mean(recall_scores))
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        return precision, recall, f1

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None

        prediction = request_state.result.completions[0].text.strip()

        references = [ref.output.text.strip() for ref in request_state.instance.references if ref.output.text.strip()]
        if not references:
            return [
                Stat(MetricName("sentence_bert_precision")).add(0.0),
                Stat(MetricName("sentence_bert_recall")).add(0.0),
                Stat(MetricName("sentence_bert_f1")).add(0.0),
            ]

        reference = references[0]
        precision, recall, f1 = self._sentence_bert_scores(prediction, reference)

        return [
            Stat(MetricName("sentence_bert_precision")).add(precision),
            Stat(MetricName("sentence_bert_recall")).add(recall),
            Stat(MetricName("sentence_bert_f1")).add(f1),
        ]
