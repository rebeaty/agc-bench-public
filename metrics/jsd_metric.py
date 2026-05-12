"""Jensen-Shannon divergence metric for distributional outputs."""

from collections import Counter
from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


def _ngram_dist(text: str, n: int):
    tokens = text.lower().split()
    if n == 1:
        grams = tokens
    else:
        grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    total = sum(counts.values())
    return {g: c / total for g, c in counts.items()} if total > 0 else {}


def _jsd(dist1: dict, dist2: dict) -> float:
    import numpy as np

    all_keys = list(set(dist1) | set(dist2))
    if not all_keys:
        return 0.0
    p = [dist1.get(k, 0.0) for k in all_keys]
    q = [dist2.get(k, 0.0) for k in all_keys]
    p_arr = __import__("numpy").array(p)
    q_arr = __import__("numpy").array(q)
    m = 0.5 * (p_arr + q_arr)
    with __import__("numpy").errstate(divide="ignore", invalid="ignore"):
        kl_pm = __import__("numpy").where(p_arr > 0, p_arr * __import__("numpy").log2(p_arr / m), 0.0)
        kl_qm = __import__("numpy").where(q_arr > 0, q_arr * __import__("numpy").log2(q_arr / m), 0.0)
    return float(0.5 * kl_pm.sum() + 0.5 * kl_qm.sum())


class JSDMetric(Metric):
    """Jensen-Shannon divergence between n-gram distributions of completion and reference."""

    def __init__(self, n: int):
        super().__init__()
        assert n in (1, 2), "n must be 1 or 2"
        self.n = n

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        completion = request_state.result.completions[0].text.strip()

        references = request_state.instance.references
        if not references or not completion:
            score = 0.0
        else:
            reference_text = references[0].output.text.strip()
            dist_completion = _ngram_dist(completion, self.n)
            dist_reference = _ngram_dist(reference_text, self.n)
            if not dist_completion or not dist_reference:
                score = 0.0
            else:
                score = _jsd(dist_completion, dist_reference)

        suffix = "unigram" if self.n == 1 else "bigram"
        metric_name = f"jensen_shannon_divergence_{suffix}"
        return [Stat(MetricName(metric_name)).add(score)]
