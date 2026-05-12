"""Intersection-over-union scoring for structured region or set outputs."""

import json
from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class IoUScoreMetric(Metric):
    """Intersection-over-Union for bounding boxes parsed from completion and reference."""

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
            pred = json.loads(completion)
            true = json.loads(reference_text)
            xi1 = max(pred[0], true[0])
            yi1 = max(pred[1], true[1])
            xi2 = min(pred[2], true[2])
            yi2 = min(pred[3], true[3])
            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            area_p = (pred[2] - pred[0]) * (pred[3] - pred[1])
            area_t = (true[2] - true[0]) * (true[3] - true[1])
            union = area_p + area_t - inter
            score = inter / union if union > 0 else 0.0
        except (json.JSONDecodeError, IndexError, TypeError, ValueError):
            score = 0.0

        return [Stat(MetricName("iou_score")).add(score)]
