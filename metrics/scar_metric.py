"""SCAR mapping metric aligned with the paper's concept/system accuracy story."""

from __future__ import annotations

import ast
import re
from typing import List, Optional, Sequence, Set, Tuple

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat


Pair = Tuple[str, str]


_CODE_BLOCK_RE = re.compile(r"```(?:python|json|text)?\s*([\s\S]*?)```", flags=re.IGNORECASE)
_PAIR_RE = re.compile(
    r"\[\s*['\"`]?([^,\]\[]+?)['\"`]?\s*,\s*['\"`]?([^,\]\[]+?)['\"`]?\s*\]",
    flags=re.DOTALL,
)


def _normalize_term(text: str) -> str:
    text = (
        (text or "")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    text = re.sub(r"^[\s'\"`]+|[\s'\"`]+$", "", text)
    text = re.sub(r"^[\[\]\(\)\{\}<>\s]+|[\[\]\(\)\{\}<>\s]+$", "", text)
    text = re.sub(r"^[,;:.!?]+|[,;:.!?]+$", "", text)
    text = " ".join(text.split())
    return text.casefold()


def _balanced_list_slice(text: str) -> Optional[str]:
    start = text.find("[[")
    if start == -1:
        return None

    depth = 0
    in_quote: Optional[str] = None
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if in_quote is not None:
            if char == "\\":
                escaped = True
            elif char == in_quote:
                in_quote = None
            continue
        if char in {"'", '"'}:
            in_quote = char
            continue
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _pairs_from_sequence(candidate: object) -> Optional[Set[Pair]]:
    if not isinstance(candidate, Sequence) or isinstance(candidate, (str, bytes)):
        return None

    parsed_pairs: List[Pair] = []
    for entry in candidate:
        if not isinstance(entry, Sequence) or isinstance(entry, (str, bytes)) or len(entry) != 2:
            continue
        left = _normalize_term(str(entry[0]))
        right = _normalize_term(str(entry[1]))
        if not left or not right:
            continue
        parsed_pairs.append((left, right))

    return set(parsed_pairs) if parsed_pairs else None


def _parse_pairs(text: str) -> Optional[Set[Pair]]:
    cleaned = (text or "").strip()
    if not cleaned:
        return None

    candidates: List[str] = []
    balanced = _balanced_list_slice(cleaned)
    if balanced:
        candidates.append(balanced)
    candidates.extend(match.group(1).strip() for match in _CODE_BLOCK_RE.finditer(cleaned))
    candidates.append(cleaned)

    seen_candidates = set()
    for candidate in candidates:
        if candidate in seen_candidates:
            continue
        seen_candidates.add(candidate)
        try:
            parsed = ast.literal_eval(candidate)
        except Exception:
            continue
        parsed_pairs = _pairs_from_sequence(parsed)
        if parsed_pairs is not None:
            return parsed_pairs

    regex_pairs = {
        (_normalize_term(left), _normalize_term(right))
        for left, right in _PAIR_RE.findall(cleaned)
        if _normalize_term(left) and _normalize_term(right)
    }
    return regex_pairs or None


def _load_gold_pairs(request_state: RequestState) -> Set[Pair]:
    extra_data = request_state.instance.extra_data or {}
    gold_mappings = extra_data.get("gold_mappings")
    if gold_mappings is not None:
        parsed_pairs = _pairs_from_sequence(gold_mappings)
        if parsed_pairs is not None:
            return parsed_pairs

    for reference in request_state.instance.references:
        parsed_pairs = _parse_pairs(reference.output.text)
        if parsed_pairs is not None:
            return parsed_pairs
    return set()


class SCARMetric(EvaluateInstancesMetric):
    """Score SCAR with mapping-aware concept/system accuracy."""

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        eval_states = [state for state in request_states if state.request_mode != "calibration"]
        total_instances = len(eval_states)

        parsed_instances = 0
        exact_instances = 0
        total_correct_pairs = 0
        total_predicted_pairs = 0
        total_gold_pairs = 0

        for state in eval_states:
            assert state.result is not None

            prediction = state.result.completions[0].text if state.result.completions else ""
            predicted_pairs = _parse_pairs(prediction)
            gold_pairs = _load_gold_pairs(state)

            if predicted_pairs is not None:
                parsed_instances += 1
            else:
                predicted_pairs = set()

            total_correct_pairs += len(predicted_pairs & gold_pairs)
            total_predicted_pairs += len(predicted_pairs)
            total_gold_pairs += len(gold_pairs)

            if predicted_pairs == gold_pairs and gold_pairs:
                exact_instances += 1

        precision = total_correct_pairs / total_predicted_pairs if total_predicted_pairs else 0.0
        recall = total_correct_pairs / total_gold_pairs if total_gold_pairs else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        system_accuracy = exact_instances / total_instances if total_instances else 0.0
        parsed_rate = parsed_instances / total_instances if total_instances else 0.0

        return [
            Stat(MetricName("scar_concept_accuracy")).add(recall),
            Stat(MetricName("scar_system_accuracy")).add(system_accuracy),
            Stat(MetricName("scar_precision")).add(precision),
            Stat(MetricName("scar_recall")).add(recall),
            Stat(MetricName("scar_f1")).add(f1),
            Stat(MetricName("scar_parsed_mapping_rate")).add(parsed_rate),
        ]
