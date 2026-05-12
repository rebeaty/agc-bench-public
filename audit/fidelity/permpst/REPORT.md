# permpst fidelity audit

**Tier:** 1
**Confidence:** high
**Recommendation:** keep_as_is

## Paper / repo audited
- Paper: PerSE (arXiv:2310.03304) — Personalized Story Evaluation, Oct 2023
- Repo: facebookresearch/perse

## Implementation audited
- scenarios/permpst_scenario.py — uses prompt[0] from official PerSE-released `review.valid.c{k}.jsonl` files (dataset ships pre-formatted prompts) verbatim. Default k=1 (one historical plot/review/score from same critic, then a new plot to score) matches paper's main config.
- metrics/permpst_score_metric.py — reports Pearson, Spearman, Kendall-Tau correlation between predicted and ground-truth Score (1-10), plus exact-match `score_accuracy` and `valid_score_json_rate`. JSON parsing (`_check_json`, `_get_valid_entry`) ported from official PerSE scorer including `ast.literal_eval` fallback and bracket/quote repair. Malformed outputs dropped from correlation.
- ~915 instances (full validation file `review.valid.c1.jsonl`, 92 unique reviewers).

## Deviations found
- [LOW] Inference: T=0.7, max_tokens=512 — paper does not specify; T=0.7 moderately high for structured-JSON regression task and may slightly inflate JSON-parse failures vs greedy. `valid_score_json_rate` surfaces this.

## Notes
Reference is ground-truth reviewer score parsed from dataset's `completion` JSON, stored as one CORRECT_TAG Reference. No LLM judge — regression against human reviewer scores. Faithful port: official prompts, official-derived parser, correct correlation metrics, full validation split, correct reference. Tier 1.
