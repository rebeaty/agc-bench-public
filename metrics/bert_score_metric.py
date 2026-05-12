"""BERTScore metric: F1 BERTScore between prediction and gold references."""

import threading
from typing import List, Optional

from bert_score import BERTScorer

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

_scorer_lock = threading.Lock()


class BertScoreMetric(Metric):
    """BERTScore F1 between the first prediction and gold references.

    Produces a stat named 'bert_score'. Uses bert-base-uncased by default.
    Score is the max BERTScore-F1 across all gold references.
    """

    def __init__(self, model_type: str = "bert-base-uncased", device: str = "cpu"):
        self.model_type = model_type
        self.device = device
        self._scorer: Optional[BERTScorer] = None

    def _load_scorer(self) -> None:
        with _scorer_lock:
            if self._scorer is not None:
                return
            import transformers
            # Newer transformers defaults low_cpu_mem_usage=True, which loads weights
            # onto meta device. bert_score then calls model.to(device) which fails on
            # meta tensors. Force eager loading to avoid this.
            _orig = transformers.AutoModel.from_pretrained

            def _eager_from_pretrained(*args, **kwargs):
                kwargs["low_cpu_mem_usage"] = False
                return _orig(*args, **kwargs)

            transformers.AutoModel.from_pretrained = _eager_from_pretrained
            try:
                self._scorer = BERTScorer(model_type=self.model_type, device=self.device)
            finally:
                transformers.AutoModel.from_pretrained = _orig

    def _compute_f1(self, pred: str, ref: str) -> float:
        self._load_scorer()
        assert self._scorer is not None
        _, _, F = self._scorer.score([pred], [ref])
        return F[0].item()

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        references = request_state.instance.references
        if not references:
            return [Stat(MetricName("bert_score")).add(0.0)]

        pred = request_state.result.completions[0].text.strip()
        refs = [ref.output.text for ref in references]

        best_f1 = max(self._compute_f1(pred, ref) for ref in refs)
        return [Stat(MetricName("bert_score")).add(best_f1)]

import contextlib
from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

# Since transformers PR #36963, init_empty_weights is native to transformers and models are
# ALWAYS loaded on meta device. Patching is_accelerate_available no longer prevents this.
# We must replace init_empty_weights with a no-op context manager so bert_score can call
# model.to(device) without hitting the "Cannot copy out of meta tensor" error.
@contextlib.contextmanager
def _noop_init_empty_weights(include_buffers=False):
    yield

import transformers.modeling_utils as _mu
import transformers.integrations.accelerate as _ta
_mu.init_empty_weights = _noop_init_empty_weights
_ta.init_empty_weights = _noop_init_empty_weights


class BertScoreMetric(Metric):
    """Computes BERTScore F1 between prediction and reference using bert-base-uncased."""

    def __init__(self, model_type: str = "bert-base-uncased"):
        super().__init__()
        self._model_type = model_type

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

        if not references or not prediction:
            return [Stat(MetricName("bert_score")).add(0.0)]

        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        import torch
        from bert_score import score as bert_score_fn

        try:
            _, _, F1 = bert_score_fn(
                [prediction],
                [references[0]],
                model_type=self._model_type,
                lang="en",
                verbose=False,
                device="cpu",
            )
            f1_val = float(F1[0].item())
        except (OverflowError, RuntimeError, ValueError, TypeError):
            # transformers>=5 / bert-score 0.3.12 incompatibility: tokenizer.model_max_length
            # can produce OverflowError in enable_truncation. Return NaN; cell completes,
            # other metrics still compute, bert_score backfill happens offline.
            f1_val = float("nan")
        return [Stat(MetricName("bert_score")).add(f1_val)]
