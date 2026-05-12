"""Task-specific metrics for the CritiCS pairwise judgment benchmark."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat


_QUESTION_1_PATTERN = re.compile(r"1\s*[:\-\)]\s*\[\[\s*([ABC])\s*\]\]", flags=re.IGNORECASE)
_QUESTION_2_PATTERN = re.compile(r"2\s*[:\-\)]\s*\[\[\s*([ABC])\s*\]\]", flags=re.IGNORECASE)
_QUESTION_3_PATTERN = re.compile(r"3\s*[:\-\)]\s*\[\[\s*([ABC])\s*\]\]", flags=re.IGNORECASE)
_QUESTION_4_PATTERN = re.compile(r"4\s*[:\-\)]\s*\[\[\s*(BY|OA|OB|BN|UN)\s*\]\]", flags=re.IGNORECASE)
_ALL_BRACKETED_PATTERN = re.compile(r"\[\[\s*([A-Z]{1,2})\s*\]\]", flags=re.IGNORECASE)

# Loosened fallback patterns (audit fix 2026-04-25): chat-tuned models
# often skip the literal `[[X]]` wrapper and emit something like
# "1) **A**" or "Question 1: B" or "**1.** Story C is more interesting".
# These match the question marker followed within ~50 chars by a bare letter.
_Q1_BARE = re.compile(r"(?:question\s*)?1\s*[:\-\)\.]\s*[^\n]{0,50}?\b([ABC])\b", flags=re.IGNORECASE)
_Q2_BARE = re.compile(r"(?:question\s*)?2\s*[:\-\)\.]\s*[^\n]{0,50}?\b([ABC])\b", flags=re.IGNORECASE)
_Q3_BARE = re.compile(r"(?:question\s*)?3\s*[:\-\)\.]\s*[^\n]{0,50}?\b([ABC])\b", flags=re.IGNORECASE)
_Q4_BARE = re.compile(r"(?:question\s*)?4\s*[:\-\)\.]\s*[^\n]{0,50}?\b(BY|OA|OB|BN|UN)\b", flags=re.IGNORECASE)


def _parse_verdicts(text: str) -> Dict[str, Optional[str]]:
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return {"q1": None, "q2": None, "q3": None, "q4": None}

    def _extract(pattern: re.Pattern[str]) -> Optional[str]:
        match = pattern.search(cleaned)
        return match.group(1).upper() if match else None

    # Strict bracketed format first (paper protocol)
    verdicts = {
        "q1": _extract(_QUESTION_1_PATTERN),
        "q2": _extract(_QUESTION_2_PATTERN),
        "q3": _extract(_QUESTION_3_PATTERN),
        "q4": _extract(_QUESTION_4_PATTERN),
    }

    # Repo-style "all four bracketed somewhere" fallback
    if any(verdicts[key] is None for key in ("q1", "q2", "q3", "q4")):
        bracketed = [match.upper() for match in _ALL_BRACKETED_PATTERN.findall(cleaned)]
        if len(bracketed) >= 4 and all(token in {"A", "B", "C"} for token in bracketed[:3]) and bracketed[3] in {
            "BY",
            "OA",
            "OB",
            "BN",
            "UN",
        }:
            verdicts["q1"] = verdicts["q1"] or bracketed[0]
            verdicts["q2"] = verdicts["q2"] or bracketed[1]
            verdicts["q3"] = verdicts["q3"] or bracketed[2]
            verdicts["q4"] = verdicts["q4"] or bracketed[3]

    # Permissive fallback for chat-tuned outputs without `[[X]]` wrapping
    if verdicts["q1"] is None: verdicts["q1"] = _extract(_Q1_BARE)
    if verdicts["q2"] is None: verdicts["q2"] = _extract(_Q2_BARE)
    if verdicts["q3"] is None: verdicts["q3"] = _extract(_Q3_BARE)
    if verdicts["q4"] is None: verdicts["q4"] = _extract(_Q4_BARE)

    return verdicts
class CriticsStoryMetric(EvaluateInstancesMetric):
    """Score repo-style CritiCS verdict formatting and Q1 agreement."""

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        scored_request_states = [state for state in request_states if state.request_mode != "calibration"]
        total = len(scored_request_states)
        if not total:
            return [
                Stat(MetricName("critics_story_interesting_accuracy")).add(0.0),
                Stat(MetricName("critics_story_interesting_parsed_rate")).add(0.0),
                Stat(MetricName("critics_story_full_format_rate")).add(0.0),
                Stat(MetricName("critics_story_premise_closeness_parsed_rate")).add(0.0),
            ]

        correct_interesting = 0
        parsed_interesting = 0
        parsed_full = 0
        parsed_q4 = 0

        for state in scored_request_states:
            assert state.result is not None
            prediction = state.result.completions[0].text if state.result.completions else ""
            verdicts = _parse_verdicts(prediction)
            gold_label = str((state.instance.extra_data or {}).get("gold_interesting_label", "")).upper()

            if verdicts["q1"] is not None:
                parsed_interesting += 1
            if verdicts["q4"] is not None:
                parsed_q4 += 1
            if all(verdicts[key] is not None for key in ("q1", "q2", "q3", "q4")):
                parsed_full += 1
            if verdicts["q1"] == gold_label:
                correct_interesting += 1

        return [
            Stat(MetricName("critics_story_interesting_accuracy")).add(correct_interesting / total),
            Stat(MetricName("critics_story_interesting_parsed_rate")).add(parsed_interesting / total),
            Stat(MetricName("critics_story_full_format_rate")).add(parsed_full / total),
            Stat(MetricName("critics_story_premise_closeness_parsed_rate")).add(parsed_q4 / total),
        ]
