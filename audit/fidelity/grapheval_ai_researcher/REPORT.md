# grapheval_ai_researcher fidelity audit

**Tier:** 1
**Confidence:** high
**Recommendation:** keep_as_is (optional: expand evaluation set or add small-N caveat)

## Paper / repo audited
- Paper: GraphEval (Feng et al., arXiv:2503.12600); dataset originates from Si et al. 2024 (AI-Researcher).
- Repo: https://github.com/ulab-uiuc/GraphEval (Data/AI_Researcher, Baselines/Prompt/basic_prompt.txt)

## Implementation audited
- scenarios/grapheval_ai_researcher_scenario.py — uses verbatim `basic_prompt.txt` from repo: 6 evaluation dimensions, 4-class decision standards, 4 few-shot examples, output format ("Overall Score (0-100)= {score}\n{decision}"). Title/abstract substitution preserved.
- metrics/grapheval_decision_metric.py — implements `GraphEvalDecisionMetric` reproducing paper's reported metrics: decision accuracy, macro precision/recall/F1 over four labels, Spearman correlation between parsed scores and gold (mean human rating × 10).
- 9 test items + 56 train; scenario uses test split only.

## Deviations found
- [LOW] Dataset is empirically 3-class (no Oral in test); scenario header documents this and `label_set` reflects 3 classes while metric still supports 4-way — acceptable.
- [LOW] Inference: `_use_defaults` (paper/repo do not specify decoding params).
- [LOW] **n=9 is unstable for Spearman / macro-F1** — small-N caveat warranted.

## Notes
Reference is historical peer-review decision; gold score = mean of human ratings × 10. No LLM judge needed — paper's LLaMa-405B judge is for viewpoint factuality inside graph pipeline, not decision scoring. Replicates direct-prompting baseline faithfully. Tier 1.
