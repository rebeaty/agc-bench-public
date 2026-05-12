"""Self-BLEU metric using NLTK (no sacrebleu dependency)."""
from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError:
    sentence_bleu = None  # type: ignore
    SmoothingFunction = None  # type: ignore


def _compute_self_bleu(texts: List[str]) -> float:
    if sentence_bleu is None or len(texts) <= 1:
        return 0.0
    smoothie = SmoothingFunction().method1
    tokenized = [t.lower().split() for t in texts if t.strip()]
    if len(tokenized) <= 1:
        return 0.0
    scores = []
    for i, hyp in enumerate(tokenized):
        refs = tokenized[:i] + tokenized[i + 1:]
        scores.append(sentence_bleu(refs, hyp, smoothing_function=smoothie))
    return sum(scores) / len(scores)


class SelfBleuMetric(Metric):
    """Self-BLEU: average BLEU of each completion against all others. Lower = more diverse."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        texts = [c.text.strip() for c in request_state.result.completions]
        score = _compute_self_bleu(texts)
        return [Stat(MetricName("self_bleu")).add(score)]
