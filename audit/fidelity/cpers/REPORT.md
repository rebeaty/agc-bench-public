# cpers fidelity audit

**Tier:** 3
**Confidence:** high
**Recommendation:** discuss (rework or drop)

## Paper / repo audited
- Paper: arXiv:2509.18401 / EMNLP 2025 Findings ("Evaluating the Creativity of LLMs in Persian Literary Text Generation")

## Implementation audited
- scenarios/cpers_scenario.py — Persian template "درباره {topic} یک متن ادبی در یک جمله بنویس" matching paper's single-sentence constrained-generation task.
- metrics/cpers_metric.py — `CPersMetric` exposes correct TTCT + rhetorical-device stats keyed off a `cpers_ttct_judge` annotation, but **not wired into registry** and no annotator is registered.
- registry: declares `bleu_4`, `rouge_l`, `llm_judge_quality` (GPT-4o, `judge_prompt: null`).

## Deviations found
- [HIGH] **Topic set diverges**: scenario = {love, longing, friendship, hope, despair}; paper = {love, longing, friendship, nature, wisdom}. Two topics substituted without justification.
- [HIGH] **Metric mismatch**: paper uses TTCT dimensions (originality, fluency, flexibility, elaboration) + rhetorical-device counts (simile, metaphor, antithesis, hyperbole) via a hybrid Claude-3.7-Sonnet + GPT-4o judge. Registry uses BLEU/ROUGE (paper uses neither) + generic `llm_judge_quality`.
- [HIGH] References empty so BLEU/ROUGE cannot compute.
- [HIGH] Single-judge (GPT-4o) vs paper's dual-judge (Claude 3.7 Sonnet + GPT-4o) with ICC validation.
- [MEDIUM] Instance count: 5 topics × 100 trials = 500. Paper uses repeated sampling rather than the 4,371-entry corpus; protocol shape reasonable but trial count not paper-anchored.

## Notes
Major rework required. Recommended: drop BLEU/ROUGE, register CPersMetric with annotator computing TTCT + rhetorical-device scores, supply concrete judge prompts, add Claude 3.7 Sonnet alongside GPT-4o, restore paper topics (nature, wisdom in place of hope, despair). In current state should not be reported.
