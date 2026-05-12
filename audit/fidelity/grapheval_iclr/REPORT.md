# grapheval_iclr fidelity audit

**Tier:** 1
**Confidence:** high
**Recommendation:** keep_as_is (minor optional polish)

## Paper / repo audited
- Paper: GraphEval (arXiv 2503.12600)
- Repo: https://github.com/ulab-uiuc/GraphEval (Baselines/Prompt/basic_prompt.txt; Data/ICLR_Dataset)

## Implementation audited
- scenarios/grapheval_iclr_scenario.py — downloads `ICLR_test_set.jsonl` directly from upstream (50 test items per docstring); embeds verbatim `basic_prompt.txt` system prompt — all 6 dimensions, decision rubric, prior-distribution note, and 4 in-prompt few-shot examples. Only `{title}`/`{abstract}` interpolated.
- metrics/grapheval_decision_metric.py (`GraphEvalDecisionMetric`): parses `Overall Score (0-100)= N` line and decision token (full label or bare `poster|oral|spotlight|reject`); computes `decision_accuracy`, macro precision/recall/F1, Spearman of parsed score vs mean human rating × 10, parse-rate diagnostics, per-class TP/FP/FN.
- Inference: benchmark defaults (paper/repo do not specify decoding params).

## Deviations found
- [LOW] `gold_score = mean(ratings) * 10` to scale onto 0–100; paper does not formally define a Spearman-vs-score metric for ICLR slice, so `spearman_correlation` is auxiliary diagnostic.
- [LOW] Internal `LABEL_TO_ID` orders Spotlight=2, Oral=3 (cosmetic; ordering does not affect macro-F1 or accuracy).
- [INFO] Inference defaults rather than paper-specified decoding (unspecified in paper).

## Notes
Prompt and dataset source are byte-faithful to upstream. Primary metrics (`decision_accuracy`, `f1_macro`) mirror the paper's evaluation; auxiliary metrics are clearly diagnostic. No judge model invoked — scoring is purely formula-based against gold labels. Tier 1.
