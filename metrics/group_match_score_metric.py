"""Group-match scoring utilities for structured categorical outputs."""

import re
from typing import Iterable, List, Sequence

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class GroupMatchScoreMetric(Metric):
    """
    Group-aware evaluator for Only Connect Wall task 1.

    This mirrors the upstream OCW evaluation more closely than the original
    token-set Jaccard implementation by:
    - parsing four predicted groups from free-form model text
    - ignoring group order and clue order inside each group
    - reporting exact group matches and full-wall solves
    - surfacing formatting / hallucination signals used in the OCW baseline
    """

    _GROUP_PREFIX_RE = re.compile(r"^\s*group\s*\d+\s*:?\s*", flags=re.IGNORECASE)
    _CONNECTION_RE = re.compile(r"\.\s*connection\s*:.*$", flags=re.IGNORECASE)
    _CLUES_RE = re.compile(r"Clues:\s*(.*?)\n\nSolved wall:", flags=re.IGNORECASE | re.DOTALL)

    @classmethod
    def _normalize_token(cls, token: str) -> str:
        token = token.strip().lower()
        token = re.sub(r"\s+", " ", token)
        token = token.strip(".,;:!?\"'`()[]{}")
        return token

    @classmethod
    def _normalize_group(cls, group: Iterable[str]) -> tuple[str, ...]:
        return tuple(sorted(cls._normalize_token(token) for token in group if cls._normalize_token(token)))

    @classmethod
    def _extract_clues(cls, prompt: str) -> set[str]:
        match = cls._CLUES_RE.search(prompt)
        if not match:
            return set()
        return {
            cls._normalize_token(token)
            for token in match.group(1).split(",")
            if cls._normalize_token(token)
        }

    @classmethod
    def _parse_groups(cls, text: str) -> List[List[str]]:
        cleaned_lines = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line = cls._GROUP_PREFIX_RE.sub("", line)
            line = cls._CONNECTION_RE.sub("", line)
            cleaned_lines.append(line)

        groups = cleaned_lines[:4]
        groups += [""] * (4 - len(groups))

        parsed_groups: List[List[str]] = []
        for line in groups:
            members = [cls._normalize_token(token) for token in line.split(",")]
            members = [token for token in members if token]
            members = members[:4]
            members += [""] * (4 - len(members))
            parsed_groups.append(members)
        return parsed_groups

    @classmethod
    def _count_exact_group_matches(
        cls, predicted_groups: Sequence[Sequence[str]], reference_groups: Sequence[Sequence[str]]
    ) -> int:
        remaining_reference_groups = [cls._normalize_group(group) for group in reference_groups]
        correct_groups = 0

        for predicted_group in predicted_groups:
            normalized_predicted_group = cls._normalize_group(predicted_group)
            for reference_index, normalized_reference_group in enumerate(remaining_reference_groups):
                if normalized_predicted_group == normalized_reference_group:
                    correct_groups += 1
                    remaining_reference_groups.pop(reference_index)
                    break

        return correct_groups

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
        predicted_groups = self._parse_groups(completion)
        reference_groups = self._parse_groups(reference_text)

        correct_groups = self._count_exact_group_matches(predicted_groups, reference_groups)
        full_wall = 1.0 if correct_groups == 4 else 0.0
        group_match_score = correct_groups / 4.0

        clue_inventory = self._extract_clues(request_state.instance.input.text)
        flattened_predictions = [token for group in predicted_groups for token in group if token]
        hallucinated_tokens = [token for token in flattened_predictions if clue_inventory and token not in clue_inventory]
        empty_slots = sum(1 for group in predicted_groups for token in group if not token)

        stats = [
            Stat(MetricName("correct_groups")).add(float(correct_groups)),
            Stat(MetricName("full_wall")).add(full_wall),
            Stat(MetricName("group_match_score")).add(group_match_score),
            Stat(MetricName("hallucinated_words")).add(float(len(hallucinated_tokens))),
            Stat(MetricName("empty_group_slots")).add(float(empty_slots)),
        ]

        return stats
