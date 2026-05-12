"""Generic per-metric LLM-as-Judge metric for the AGC-Bench."""
from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class GenericLLMJudgeMetric(Metric):
    """Reads a single numeric annotation produced by GenericLLMJudgeAnnotator.

    Annotations are keyed by the annotator's ``self.name``, which
    GenericLLMJudgeAnnotator sets to ``f"generic_llm_judge_{metric_name}"``.
    """

    def __init__(self, metric_name: str):
        super().__init__()
        self.metric_name = metric_name

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        annotations = request_state.annotations or {}
        annotator_output = annotations.get(f"generic_llm_judge_{self.metric_name}", {}) or {}
        score = float(annotator_output.get(self.metric_name, 0))
        return [Stat(MetricName(self.metric_name)).add(score)]
