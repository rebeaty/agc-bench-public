"""Sudoku-Bench single-shot answer extraction and solve-rate metric."""

from __future__ import annotations

import re
from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


_ANSWER_RE = re.compile(r"<ANSWER>\s*(.*?)\s*</ANSWER>", flags=re.IGNORECASE | re.DOTALL)


def _extract_tagged_answer(text: str) -> tuple[str, bool]:
    match = _ANSWER_RE.search(text or "")
    if match is None:
        return "", False
    digits = "".join(ch for ch in match.group(1) if ch.isdigit())
    return digits, True


class SudokuBenchMetric(Metric):
    """Score single-shot Sudoku-Bench generations by extracted tagged answer."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None

        completion = request_state.result.completions[0].text
        gold = "".join(ch for ch in request_state.instance.references[0].output.text if ch.isdigit())
        prediction, found_tags = _extract_tagged_answer(completion)
        expected_length = len(gold)

        return [
            Stat(MetricName("sudoku_solution_accuracy")).add(float(bool(prediction) and prediction == gold)),
            Stat(MetricName("sudoku_answer_extracted_rate")).add(float(found_tags)),
            Stat(MetricName("sudoku_answer_length_valid_rate")).add(float(bool(prediction) and len(prediction) == expected_length)),
        ]
