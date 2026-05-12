"""Fann or Flop benchmark-specific metric surface."""

from __future__ import annotations

import contextlib
import os
import threading
from typing import List, Optional

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

import transformers.integrations.accelerate as _ta
import transformers.modeling_utils as _mu
from bert_score import BERTScorer


@contextlib.contextmanager
def _noop_init_empty_weights(include_buffers: bool = False):
    yield


_mu.init_empty_weights = _noop_init_empty_weights
_ta.init_empty_weights = _noop_init_empty_weights

_scorer_lock = threading.Lock()


class FannOrFlopMetric(Metric):
    """Emit verse-aware judge stats plus chrF++ and AraBERT BERTScore."""

    def __init__(
        self,
        bert_model_type: str = "aubmindlab/bert-base-arabertv02",
        bert_num_layers: int = 12,
    ):
        super().__init__()
        self.bert_model_type = bert_model_type
        self.bert_num_layers = bert_num_layers
        self._scorer: Optional[BERTScorer] = None

    def _load_scorer(self) -> None:
        with _scorer_lock:
            if self._scorer is not None:
                return
            self._scorer = BERTScorer(
                model_type=self.bert_model_type,
                num_layers=self.bert_num_layers,
                device="cpu",
                lang=None,
            )

    def _compute_bert_score(self, prediction: str, reference: str) -> float:
        self._load_scorer()
        assert self._scorer is not None
        _, _, f1 = self._scorer.score([prediction], [reference])
        return float(f1[0].item())

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        annotations = request_state.annotations or {}
        annotator_output = annotations.get("fann_or_flop_poem_judge", {}) or {}

        stats: List[Stat] = [
            Stat(MetricName("gold_verse_count")).add(float(annotator_output.get("gold_verse_count", 0.0))),
            Stat(MetricName("generated_verse_count")).add(float(annotator_output.get("generated_verse_count", 0.0))),
            Stat(MetricName("aligned_verse_count")).add(float(annotator_output.get("aligned_verse_count", 0.0))),
            Stat(MetricName("verse_parse_rate")).add(float(annotator_output.get("verse_parse_rate", 0.0))),
            Stat(MetricName("judge_parse_rate")).add(float(annotator_output.get("judge_parse_rate", 0.0))),
            Stat(MetricName("faithfulness_score")).add(float(annotator_output.get("faithfulness_score", 0.0))),
            Stat(MetricName("fluency_score")).add(float(annotator_output.get("fluency_score", 0.0))),
            Stat(MetricName("overall_score")).add(float(annotator_output.get("overall_score", 0.0))),
        ]

        assert request_state.result is not None
        prediction = request_state.result.completions[0].text.strip() if request_state.result.completions else ""
        references = [ref.output.text.strip() for ref in request_state.instance.references if ref.output.text.strip()]
        if prediction and references:
            from sacrebleu import sentence_chrf

            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            best_reference = references[0]
            chrfpp = sentence_chrf(prediction, references, word_order=2).score
            bert_score = self._compute_bert_score(prediction, best_reference)
            stats.append(Stat(MetricName("chrfpp")).add(float(chrfpp)))
            stats.append(Stat(MetricName("bert_score")).add(float(bert_score)))

        return stats
