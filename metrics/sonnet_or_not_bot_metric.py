"""Task-specific metric for Sonnet or Not, Bot? form classification."""

from __future__ import annotations

import re
from typing import List, Optional

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat


_ANSWER_PATTERN = re.compile(r"answer\s*[:\-]?\s*([A-J])\b", flags=re.IGNORECASE)
_BARE_OPTION_PATTERN = re.compile(r"\b([A-J])\b", flags=re.IGNORECASE)
_POETIC_FORM_PATTERN = re.compile(r"poetic\s+form\s*[:\-]?\s*([^\n\r.;,]+)", flags=re.IGNORECASE)


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _extract_predicted_form(text: str, possible_forms: List[str]) -> Optional[str]:
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return None

    normalized_forms = [_normalize(form) for form in possible_forms]
    form_by_label = {chr(ord("A") + index): form for index, form in enumerate(possible_forms)}

    answer_matches = _ANSWER_PATTERN.findall(cleaned)
    if answer_matches:
        return form_by_label.get(answer_matches[-1].upper())

    bare_matches = _BARE_OPTION_PATTERN.findall(cleaned)
    if bare_matches:
        return form_by_label.get(bare_matches[-1].upper())

    form_match = _POETIC_FORM_PATTERN.search(cleaned)
    if form_match:
        extracted = _normalize(form_match.group(1))
        matches = [form for form, normalized in zip(possible_forms, normalized_forms) if extracted == normalized]
        if len(matches) == 1:
            return matches[0]

    lowered_text = _normalize(cleaned)
    matches = []
    for form, normalized in zip(possible_forms, normalized_forms):
        if re.search(rf"(?<![a-z]){re.escape(normalized)}(?![a-z])", lowered_text):
            matches.append(form)

    if len(matches) == 1:
        return matches[0]

    return None


class SonnetOrNotBotMetric(EvaluateInstancesMetric):
    """Score parsed form-selection accuracy over the released public-domain set."""

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        scored_request_states = [state for state in request_states if state.request_mode != "calibration"]
        total = len(scored_request_states)
        correct = 0
        parsed = 0

        for state in scored_request_states:
            assert state.result is not None
            prediction = state.result.completions[0].text if state.result.completions else ""
            extra_data = state.instance.extra_data or {}
            possible_forms = list(extra_data.get("possible_forms", []))
            gold_form = str(extra_data.get("gold_form", "")).strip().lower()

            parsed_form = _extract_predicted_form(prediction, possible_forms)
            if parsed_form is not None:
                parsed += 1
            if _normalize(parsed_form or "") == gold_form:
                correct += 1

        accuracy = correct / total if total else 0.0
        parsed_rate = parsed / total if total else 0.0
        return [
            Stat(MetricName("accuracy")).add(accuracy),
            Stat(MetricName("parsed_label_rate")).add(parsed_rate),
        ]
