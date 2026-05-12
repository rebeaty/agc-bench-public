"""MCQ classification metric that normalizes markdown in predictions.

HELM's `MultipleChoiceClassificationMetric` does
    pred = sorted_completions[0].text.strip()
then compares to gold via macro/micro F1. Chat-tuned models wrap their
answer in markdown (`**A**`, `**Sim**`), pre-pend the option letter to
the answer text (`A. Sim`, `D. D`), or follow the answer with explanatory
text (`C\\n\\nThe process...`). All three patterns broke literal F1 comparison
and the optional `output_mapping` lookup.

This subclass overrides only the prediction extraction step. Scoring
semantics (F1) are unchanged. Order of operations:
  1. Strip markdown (`**`, `__`, headers, etc.)
  2. Pull the leading single letter (A-E) if the response begins with one
  3. If a single letter was found AND output_mapping has it, use the mapped value
  4. Otherwise, also try the full normalized text against output_mapping
  5. Fall back to whichever survives (letter / mapped / raw)
"""
from __future__ import annotations

import re
from typing import List, Optional

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.classification_metrics import MultipleChoiceClassificationMetric
from helm.common.request import GeneratedOutput
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat
from sklearn.metrics import f1_score

from metrics.markdown_normalizer import normalize_markdown

# Matches a leading MCQ letter, optionally followed by a punctuation
# delimiter and the option text or an explanation. Examples:
#   "A"          -> A
#   "A. Sim"     -> A
#   "D. D"       -> D
#   "C\n\nThe..."-> C
#   "C) The..."  -> C
#   "answer: B"  -> B (handled by the `answer:` strip below before this)
_LEADING_LETTER = re.compile(r"^\s*([A-Ea-e])(?:[\.\):\-\s]|$)")
_ANSWER_PREFIX = re.compile(r"^\s*(?:answer|resposta|respuesta|ответ)\s*[:\-]?\s*", re.IGNORECASE)


def _extract_leading_letter(text: str) -> Optional[str]:
    """Return the upper-case A-E letter starting `text`, or None."""
    if not text:
        return None
    # Strip a chatty 'answer:'/'resposta:' prefix some models emit.
    cleaned = _ANSWER_PREFIX.sub("", text).lstrip()
    m = _LEADING_LETTER.match(cleaned)
    return m.group(1).upper() if m else None


class MarkdownNormalizedMCQClassificationMetric(MultipleChoiceClassificationMetric):
    """MCQ classification with markdown-strip and leading-letter extraction."""

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        y_pred: List[str] = []
        y_true: List[str] = []
        for request_state in request_states:
            if request_state.request_mode == "calibration":
                raise ValueError(
                    "MarkdownNormalizedMCQClassificationMetric does not support calibration requests"
                )
            golds = [r for r in request_state.instance.references if r.is_correct]
            assert len(golds) > 0, "MCQ classification expects at least one correct reference"
            assert request_state.result is not None
            sorted_completions: List[GeneratedOutput] = sorted(
                request_state.result.completions, key=lambda x: -x.logprob
            )
            raw = sorted_completions[0].text or ""
            normalized = normalize_markdown(raw).strip()
            # Try to recover the answer in priority order:
            #   1. leading letter via output_mapping (e.g., "A" -> "Sim")
            #   2. leading letter directly (gold itself is a letter, e.g., "D")
            #   3. full normalized text via output_mapping (e.g., "Sim" -> "Sim")
            #   4. full normalized text raw
            letter = _extract_leading_letter(normalized)
            mapping = request_state.output_mapping or {}
            if letter is not None and letter in mapping:
                pred = mapping[letter]
            elif letter is not None:
                pred = letter
            elif normalized in mapping:
                pred = mapping[normalized]
            else:
                pred = normalized

            y_true.append(golds[0].output.text)
            y_pred.append(pred)
        return [
            Stat(MetricName("classification_macro_f1")).add(
                f1_score(y_pred=y_pred, y_true=y_true, average="macro")
            ),
            Stat(MetricName("classification_micro_f1")).add(
                f1_score(y_pred=y_pred, y_true=y_true, average="micro")
            ),
        ]
