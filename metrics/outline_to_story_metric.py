"""Benchmark-specific controllability metric for Outline-to-Story."""

from __future__ import annotations

import re
from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_generated_paragraphs(text: str) -> List[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n+", text.strip()) if paragraph.strip()]
    return paragraphs if paragraphs else ([text.strip()] if text.strip() else [])


class OutlineToStoryMetric(Metric):
    """Track how well each generated paragraph realizes its expected outline events."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None

        generated_story = request_state.result.completions[0].text.strip()
        outline_events = request_state.instance.extra_data.get("outline_events", [])
        if not outline_events:
            return [
                Stat(MetricName("outline_event_recall")).add(0.0),
                Stat(MetricName("outline_paragraph_coverage")).add(0.0),
            ]

        generated_paragraphs = [_normalize(paragraph) for paragraph in _split_generated_paragraphs(generated_story)]
        paragraph_recalls: List[float] = []
        covered_paragraphs = 0

        for paragraph_index, expected_events in enumerate(outline_events):
            normalized_events = [_normalize(event) for event in expected_events if _normalize(event)]
            if paragraph_index >= len(generated_paragraphs) or not normalized_events:
                paragraph_recalls.append(0.0)
                continue

            generated_paragraph = generated_paragraphs[paragraph_index]
            matches = sum(1 for event in normalized_events if event in generated_paragraph)
            recall = matches / len(normalized_events)
            paragraph_recalls.append(recall)
            if matches > 0:
                covered_paragraphs += 1

        outline_event_recall = sum(paragraph_recalls) / len(outline_events)
        outline_paragraph_coverage = covered_paragraphs / len(outline_events)

        return [
            Stat(MetricName("outline_event_recall")).add(outline_event_recall),
            Stat(MetricName("outline_paragraph_coverage")).add(outline_paragraph_coverage),
        ]
