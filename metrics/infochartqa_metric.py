"""InfoChartQA metric mirroring the released qtype-aware checker."""

from __future__ import annotations

import ast
import re
from difflib import SequenceMatcher
from typing import Iterable, List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


_UNIT_MAP = {
    "K": 1e3,
    "k": 1e3,
    "M": 1e6,
    "m": 1e6,
    "million": 1e6,
    "bn": 1e9,
    "Bn": 1e9,
    "b": 1e9,
    "B": 1e9,
    "Billion": 1e9,
    "T": 1e12,
    "%": 1e-2,
    "Cr": 1e8,
    "None": 1.0,
}


def _fuzzy_string_match(answer: str, response: str, threshold: float = 0.8) -> bool:
    answer = answer.lower()
    response = response.lower()
    if answer == response or answer in response:
        return True
    return SequenceMatcher(None, answer, response).ratio() >= threshold


def _contains_number(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def _extract_numbers_keep_order(text: str) -> List[str]:
    matches = []
    for match in re.finditer(r"-?(?:\d+,)+\d+", text):
        if "." not in match.group():
            matches.append((match.start(), match.group()))
    for match in re.finditer(r"-?\d+\.\d+", text):
        if "," not in match.group():
            matches.append((match.start(), match.group()))
    for match in re.finditer(r"(?<![\d.,])-?\d+(?![\d.,])", text):
        value = match.group()
        start = match.start()
        if "." not in value and "," not in value and all(not (start >= s and start < s + len(v)) for s, v in matches):
            matches.append((start, value))
    matches.sort()
    return [value for _, value in matches]


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _is_valid_thousand_separator(text: str, separator: str) -> bool:
    if separator == ",":
        pattern = r"^[-+]?\d{1,3}(,\d{3})*(\.(\d*))?$"
    elif separator == ".":
        pattern = r"^[-+]?\d{1,3}(\.\d{3})*(,(\d*))?$"
    else:
        return False
    return bool(re.match(pattern, text))


def _convert_to_number(text: str) -> float | str:
    value = text.strip()
    if not value:
        return "NoTANumber"
    if value.endswith("."):
        value = value[:-1]
    sign = 1.0
    if value and value[0] in "+-":
        sign = 1.0 if value[0] == "+" else -1.0
        value = value[1:]
    separator, decimal = ",", "."
    if (not _is_valid_thousand_separator(value, ",")) and _is_valid_thousand_separator(value, "."):
        separator, decimal = ".", ","
    elif (not _is_valid_thousand_separator(value, ",")) and (not _is_valid_thousand_separator(value, ".")):
        value = value.split(",")[-1]
    compact = value.replace(separator, "").replace(decimal, ".")
    return sign * float(compact) if _is_number(compact) else "NoTANumber"


def _get_unit(text: str) -> str:
    right = len(text)
    for idx in range(len(text) - 1, -1, -1):
        if text[idx].isalpha() or text[idx] == "%":
            right = idx
            break
    else:
        return "None"

    left = 0
    for idx in range(right, -1, -1):
        if not text[idx].isalpha() and text[idx] != "%":
            left = idx + 1
            break
    unit = text[left : right + 1]
    return unit if unit in _UNIT_MAP else "None"


def _get_numeric_fragment(text: str) -> str:
    left = -1
    right = len(text)
    index = 0
    while index < len(text):
        if not (text[index].isdigit() or text[index] in "+-"):
            index += 1
            continue
        end = index
        while end < len(text) and (text[end].isdigit() or text[end] in ",.+-"):
            end += 1
        left, right = index, end
        index = end
    return text[left:right] if left != -1 else "0"


def _get_unit_and_numeric(text: str) -> tuple[str, str]:
    parts = str(text).split(" ")
    for idx in range(len(parts) - 1, -1, -1):
        part = parts[idx]
        if not _contains_number(part):
            continue
        cleaned = part.replace(" ", "").replace("$", "").replace("\n", "")
        numeric = _get_numeric_fragment(cleaned)
        unit = _get_unit(cleaned)
        if unit == "None" and idx + 1 < len(parts):
            unit = _get_unit(parts[idx + 1])
        return numeric, unit if unit in _UNIT_MAP else "None"
    return "NoTANumber", "None"


def _compare_numeric_value(answer: str, response: str, eps: float = 0.001) -> bool:
    ans_number, ans_unit = _get_unit_and_numeric(answer.replace("\n", " "))
    response_number, response_unit = _get_unit_and_numeric(response.replace("\n", " "))
    ans_value = _convert_to_number(ans_number)
    response_value = _convert_to_number(response_number)
    if ans_value == "NoTANumber" or response_value == "NoTANumber":
        return False

    for unit1 in (ans_unit, "None"):
        for unit2 in (response_unit, "None"):
            expected = ans_value * _UNIT_MAP[unit1]
            actual = response_value * _UNIT_MAP[unit2]
            if abs((expected - actual) / (0.01 + abs(expected))) < eps:
                return True

    for special_case in (100, 1000, 1000000, 1000000000):
        if abs(special_case * ans_value - response_value) / (0.01 + abs(special_case * ans_value)) < eps:
            return True
        if abs(special_case * response_value - ans_value) / (0.01 + abs(ans_value)) < eps:
            return True
    return False


def _compare_value(answer: str, response: str, eps: float = 0.001) -> bool:
    return _compare_numeric_value(answer, response, eps=eps) if _contains_number(answer) else _fuzzy_string_match(answer, response)


def _normalize_list_items(items: Iterable[str]) -> List[str]:
    return [str(item).strip() for item in items]


def _sequence_match_ordered(seq1: List[str], seq2: List[str], fuzzy: bool = False, threshold: float = 0.6) -> bool:
    if len(seq1) != len(seq2):
        return False
    if fuzzy:
        return all(_fuzzy_string_match(left, right, threshold=threshold) for left, right in zip(seq1, seq2))
    return all(left.strip().lower() == right.strip().lower() for left, right in zip(seq1, seq2))


def _sequence_match_unordered(seq1: List[str], seq2: List[str], fuzzy: bool = False, threshold: float = 0.8) -> bool:
    if len(seq1) != len(seq2):
        return False
    if not fuzzy:
        return sorted(item.lower().strip() for item in seq1) == sorted(item.lower().strip() for item in seq2)

    matched_indices = set()
    for left in seq1:
        for idx, right in enumerate(seq2):
            if idx in matched_indices:
                continue
            if _fuzzy_string_match(left, right, threshold=threshold) or left.lower().strip() in right.lower().strip() or right.lower().strip() in left.lower().strip():
                matched_indices.add(idx)
                break
    return len(matched_indices) == len(seq1)


def _parse_multi_choice_response(response: str, all_choices: List[str]) -> str:
    text = " " + response.strip(" ,.!?;:'") + " "
    candidates: List[str] = []
    for choice in all_choices:
        if f"({choice})" in text:
            candidates.append(choice)
    if not candidates:
        for choice in all_choices:
            if f" {choice} " in text:
                candidates.append(choice)
    if not candidates:
        return "None"
    if len(candidates) == 1:
        return candidates[0]
    return max(candidates, key=lambda choice: max(text.rfind(f"({choice})"), text.rfind(f" {choice} ")))


def _multiple_choice_checker(answer: str, response: str) -> bool:
    predicted_items = { _parse_multi_choice_response(item, ["A", "B", "C", "D"]) for item in response.split(",") }
    gold_items = { _parse_multi_choice_response(item, ["A", "B", "C", "D"]) for item in answer.upper().split(",") }
    return predicted_items == gold_items


def _parse_trend_answers(answer: str) -> List[str]:
    stripped = answer.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return [answer]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return [answer]


def _evaluate_answer(answer: str, response: str, qtype: int) -> bool:
    if qtype in {1, 2, 101, 102}:
        return _compare_value(answer, response) if _contains_number(answer) else _fuzzy_string_match(answer, response)
    if qtype in {72, 54}:
        return _compare_value(answer, response)
    if qtype in {10, 11, 12, 50, 51, 52, 110}:
        return _compare_value(answer, response, eps=0.05)
    if qtype in {13, 14, 15, 103, 113}:
        return answer.lower() in response.lower() or _compare_value(answer, response)
    if qtype in {40, 41, 42, 43, 44}:
        normalized_response = response.replace("\n", ",").replace(" ", "")
        normalized_answer = answer.replace(" ", "")
        return _sequence_match_unordered(
            _normalize_list_items(normalized_answer.split(",")),
            _normalize_list_items(normalized_response.split(",")),
            fuzzy=True,
        )
    if qtype in {60, 61, 70, 80, 90}:
        return _fuzzy_string_match(answer, response)
    if qtype == 71:
        normalized_response = response.replace("\n\n", "").replace("\n", ",").replace(" ", "").replace("<", ",").replace(">", ",")
        if normalized_response.count(":") == 1:
            normalized_response = normalized_response.split(":", 1)[1]
        normalized_answer = answer.replace(" ", "")
        return _sequence_match_ordered(
            _normalize_list_items(normalized_answer.split(",")),
            _normalize_list_items(normalized_response.split(",")),
            fuzzy=True,
        )
    if qtype == 30:
        normalized_response = _normalize_list_items(response.split(","))
        return any(
            _sequence_match_ordered(_normalize_list_items(candidate.split(",")), normalized_response, fuzzy=True)
            for candidate in _parse_trend_answers(answer)
        )
    if qtype in {202, 300, 1919810, 1919811, 1919812}:
        return _multiple_choice_checker(answer, response)
    return _compare_value(answer, response)


class InfoChartQAMetric(Metric):
    """Score InfoChartQA responses with the released qtype-aware checker."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None

        prediction = str(request_state.result.completions[0].text or "").replace("\xa0", " ").strip()
        gold = request_state.instance.references[0].output.text if request_state.instance.references else ""
        qtype = int((request_state.instance.extra_data or {}).get("question_type_id", 1))
        accuracy = 1.0 if prediction and _evaluate_answer(str(gold), prediction, qtype) else 0.0

        return [Stat(MetricName("infochartqa_accuracy")).add(accuracy)]
