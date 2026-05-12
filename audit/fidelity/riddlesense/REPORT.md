# riddlesense fidelity audit

**Tier:** 1
**Confidence:** high
**Recommendation:** keep_as_is (optional: log accuracy explicitly alongside F1)

## Paper / repo audited
- Paper: Lin et al. 2021, ACL-Findings (https://aclanthology.org/2021.findings-acl.131/)
- Repo: https://github.com/INK-USC/RiddleSense
- Dataset: INK-USC/riddle_sense (HF)

## Implementation audited
- scenarios/riddlesense_scenario.py — `Question: {q}\nA. ... E. ...` joint MCQ; instruction "Choose the single best answer. Respond with only one letter: A, B, C, D, or E." `output_prefix="Answer: "`; zero-shot.
- run_specs/riddlesense_run_specs.py — `MultipleChoiceClassificationMetric` (macro/micro F1).
- HF validation split = 1,021 instances; sampling cap = 200 (random, seed 20260421). Test labels hidden.

## Deviations found
- [LOW] **Metric**: paper reports accuracy as primary; HELM uses macro/micro F1. For single-label 5-way MCQ, micro-F1 ≈ accuracy (acceptable proxy). Macro-F1 is acceptable extra. Recommend logging accuracy explicitly.
- [LOW] **Validation substitution**: test labels hidden, validation used as eval split. Documented in scenario docstring.

## Notes
Reference handling correct: gold `answerKey` letter; all 5 choices materialized as Reference with CORRECT_TAG on gold; deterministic letter match via `output_mapping_pattern=r"\b([ABCDE])\b"`. No LLM judge needed. PROMPT, INSTANCE COUNT, JUDGE/REFERENCE all faithful. Tier 1.
