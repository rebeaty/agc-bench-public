"""Mean absolute error metric for numeric prediction tasks."""

from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class MeanAbsoluteErrorMetric(Metric):
    """Computes mean absolute error between predicted and reference numeric values."""

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
        reference_text = references[0].output.text.strip() if references else ""

        try:
            # Try scalar first
            pred = float(completion)
            true = float(reference_text)
            score = abs(pred - true)
        except ValueError:
            try:
                # Try comma-separated lists
                import numpy as np

                preds = [float(x.strip()) for x in completion.split(",") if x.strip()]
                trues = [float(x.strip()) for x in reference_text.split(",") if x.strip()]
                if preds and trues and len(preds) == len(trues):
                    score = float(np.mean(np.abs(np.array(preds) - np.array(trues))))
                else:
                    score = 0.0
            except ValueError:
                score = 0.0

        return [Stat(MetricName("mean_absolute_error")).add(score)]
