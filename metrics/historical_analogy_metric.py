"""Wikipedia-backed Pass@1 metric for Historical Analogy."""

from __future__ import annotations

import json
import re
from typing import List, Optional, Set
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat


_THINK_SPLIT_RE = re.compile(r"</think>", flags=re.IGNORECASE)
_LEADING_PREFIX_RE = re.compile(
    r"^(?:historical analog(?:y|ies) events?|analogy event|analogous event|answer|event)\s*[:\-]\s*",
    flags=re.IGNORECASE,
)
_CLAUSE_PREFIXES = (
    "the best analogy is ",
    "the analogous event is ",
    "the historical analogy is ",
    "the analogy is ",
    "my answer is ",
    "answer is ",
    "it is ",
)


def _strip_thinking(text: str) -> str:
    parts = _THINK_SPLIT_RE.split(text or "", maxsplit=1)
    return parts[-1].strip()


def _extract_prediction(text: str) -> Optional[str]:
    cleaned = _strip_thinking(text)
    if not cleaned:
        return None

    for marker in ("Historical Analogies Events:", "Historical Analogy Event:", "Answer:"):
        if marker.lower() in cleaned.lower():
            pattern = re.compile(re.escape(marker), flags=re.IGNORECASE)
            cleaned = pattern.split(cleaned, maxsplit=1)[-1].strip()
            break

    candidate = next((line.strip() for line in cleaned.splitlines() if line.strip()), "")
    candidate = _LEADING_PREFIX_RE.sub("", candidate).strip()
    lowered = candidate.lower()
    for prefix in _CLAUSE_PREFIXES:
        if lowered.startswith(prefix):
            candidate = candidate[len(prefix):].strip()
            lowered = candidate.lower()
            break

    candidate = candidate.strip("`*_\"' ")
    candidate = re.split(r"[.;!?](?:\s|$)", candidate, maxsplit=1)[0].strip()
    return candidate or None


def _normalize_title(text: str) -> str:
    normalized = " ".join((text or "").strip().split())
    normalized = normalized.replace("–", "-").replace("—", "-")
    normalized = normalized.strip("`*_\"' .,:;!?")
    return normalized.casefold()


def _strip_parenthetical(title: str) -> str:
    stripped = re.sub(r"\s*\([^)]*\)", "", title or "").strip()
    return stripped


class HistoricalAnalogyMetric(EvaluateInstancesMetric):
    """
    Reproduce the repo's Pass@1 behavior with Wikipedia title search.

    The upstream `evaluation.py` checks whether any title returned by a
    Wikipedia search for the prediction overlaps the title set returned for the
    gold event. This HELM metric mirrors that logic and also records whether we
    could parse a single event-name prediction from the model output.
    """

    def __init__(self) -> None:
        self._search_cache: dict[str, Set[str]] = {}

    def _wikipedia_titles(self, query: str) -> Set[str]:
        normalized_query = _normalize_title(query)
        if not normalized_query:
            return set()
        if normalized_query in self._search_cache:
            return self._search_cache[normalized_query]

        params = urlencode(
            {
                "action": "opensearch",
                "limit": 10,
                "namespace": 0,
                "format": "json",
                "search": query,
            }
        )
        url = f"https://en.wikipedia.org/w/api.php?{params}"
        request = Request(url, headers={"User-Agent": "agc-bench/1.0"})

        titles: Set[str] = {normalized_query}
        raw_query = " ".join((query or "").strip().split())
        if raw_query:
            titles.add(_normalize_title(_strip_parenthetical(raw_query)))

        try:
            with urlopen(request, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            self._search_cache[normalized_query] = {title for title in titles if title}
            return self._search_cache[normalized_query]

        for title in payload[1] if isinstance(payload, list) and len(payload) > 1 else []:
            normalized_title = _normalize_title(str(title))
            if normalized_title:
                titles.add(normalized_title)
            stripped_title = _normalize_title(_strip_parenthetical(str(title)))
            if stripped_title:
                titles.add(stripped_title)

        self._search_cache[normalized_query] = {title for title in titles if title}
        return self._search_cache[normalized_query]

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        scored_request_states = [state for state in request_states if state.request_mode != "calibration"]
        total = len(scored_request_states)
        parsed = 0
        matched = 0

        for state in scored_request_states:
            assert state.result is not None
            completion = state.result.completions[0].text if state.result.completions else ""
            prediction = _extract_prediction(completion)
            if prediction:
                parsed += 1
            else:
                continue

            extra_data = state.instance.extra_data or {}
            target_event = str(extra_data.get("target_event") or "")
            if not target_event and state.instance.references:
                target_event = state.instance.references[0].output.text

            predicted_titles = self._wikipedia_titles(prediction)
            target_titles = self._wikipedia_titles(target_event)
            if predicted_titles and target_titles and predicted_titles.intersection(target_titles):
                matched += 1

        return [
            Stat(MetricName("historical_analogy_pass_at_1")).add(matched / total if total else 0.0),
            Stat(MetricName("historical_analogy_parsed_event_rate")).add(parsed / total if total else 0.0),
        ]
