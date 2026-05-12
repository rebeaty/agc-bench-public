"""Automatic metrics for Unfun Corpus based on the upstream evaluation scripts."""

import string
from typing import List

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat


def _normalize(text: str) -> str:
    return " ".join(text.lower().split()).replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("—", "-").replace("…", "...")


def _clean_generated_headline(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    cleaned = cleaned.splitlines()[0].strip()
    return cleaned.strip('"').strip()


def _find_edit_distance(humor: str, non_humor: str) -> int:
    humor = _normalize(humor).translate(str.maketrans("", "", string.punctuation))
    non_humor = _normalize(non_humor).translate(str.maketrans("", "", string.punctuation))

    humor_tokens = humor.split()
    non_humor_tokens = non_humor.split()

    len_humor = len(humor_tokens)
    len_non_humor = len(non_humor_tokens)

    minimum_edits = [[0] * (len_non_humor + 1) for _ in range(len_humor + 1)]
    for i in range(1, len_humor + 1):
        minimum_edits[i][0] = i
    for j in range(1, len_non_humor + 1):
        minimum_edits[0][j] = j

    for i in range(1, len_humor + 1):
        for j in range(1, len_non_humor + 1):
            if humor_tokens[i - 1] == non_humor_tokens[j - 1]:
                minimum_edits[i][j] = minimum_edits[i - 1][j - 1]
            else:
                minimum_edits[i][j] = 1 + min(
                    minimum_edits[i - 1][j - 1],
                    minimum_edits[i - 1][j],
                    minimum_edits[i][j - 1],
                )

    return minimum_edits[len_humor][len_non_humor]


class UnfunAutomaticMetric(EvaluateInstancesMetric):
    """Expose upstream-style automatic metrics for generated unfunned headlines."""

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        outputs: List[str] = []
        edit_distances: List[int] = []

        for request_state in request_states:
            if request_state.request_mode == "calibration":
                continue
            assert request_state.result is not None
            generated = _clean_generated_headline(request_state.result.completions[0].text)
            if not generated:
                continue

            outputs.append(generated)
            satire_headline = str((request_state.instance.extra_data or {}).get("satirical_headline", ""))
            if satire_headline:
                edit_distances.append(_find_edit_distance(satire_headline, generated))

        all_tokens = []
        for output in outputs:
            all_tokens.extend(_normalize(output).split())

        unique_tokens = set(all_tokens)
        ttr = (len(unique_tokens) / len(all_tokens)) if all_tokens else 0.0
        mean_edit_distance = (sum(edit_distances) / len(edit_distances)) if edit_distances else 0.0

        return [
            Stat(MetricName("edit_distance")).add(mean_edit_distance),
            Stat(MetricName("lexical_diversity_ttr")).add(ttr),
        ]
