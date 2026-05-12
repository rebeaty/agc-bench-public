# schnovel fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** high
**Recommendation:** patch_with_position_shuffle_lower_T_and_explicit_accuracy

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/schnovel_run_specs.py`:
>
> - **MetricSpec(s):** `metrics.markdown_normalized_classification_metric.MarkdownNormalizedMCQClassificationMetric`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: Lin et al., arXiv:2409.16605 (AISD@ACL 2025) — SchNovel pairwise novelty assessment.
- Repo: huggingface.co/datasets/ethannlin/SchNovel

## Implementation audited
- scenarios/schnovel_scenario.py — reproduces verbatim paper's Zero-Shot prompt (Appendix A.2): 4-step rubric, "current research landscape in 2024" framing, instruction to "only state which paper is more novel (e.g., 1...; 2...)." Paper1/Paper2 title+abstract substituted directly.
- run_specs/schnovel_run_specs.py — `MultipleChoiceClassificationMetric` (macro/micro F1).
- 15,000 pairs (6 fields × 2,500). Default `field="all"` loads all 15K; subsampled to 200 via `schnovel_cap200_selection.json` (not visibly stratified by year-gap).

## Deviations found
- [HIGH] **No per-gap or per-field breakdown**: paper primary metric is **Accuracy** broken down by year-gap and field. HELM emits only F1. Paper's central analysis not reproduced.
- [MEDIUM] **Inference**: T=0.7, max_tokens=512 for single-token "1"/"2" answer. CLAUDE.md guidance is 0.0–0.3 for classification; current setting likely inflates variance.
- [LOW] **No stratified subsampling**: 200-cap should ideally stratify by year-gap to recover paper's main analysis.

## Notes
Newly observed positional bias: lines 119-123 always present paper1 (more recent) as the correct answer in slot "1". A model with position bias toward "1" will inflate accuracy. Same class of bug as proparalogy (which was fixed) but never flagged here. Suggested fix: deterministically shuffle the (paper1, paper2) ordering by sample hash and retag CORRECT_TAG. The paper does randomize order in its eval protocol (Appendix A.2 wording is order-agnostic).

For balanced binary labels, micro-F1 ≈ accuracy (acceptable proxy). Tier 2 — fixes: (1) shuffle paper position per instance with seeded RNG, (2) lower T to ≤0.3, (3) surface accuracy explicitly with year-gap × field breakdown, (4) stratify any subsample by year-gap.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, high, patch_with_lower_temperature_and_explicit_accuracy
- Now:   Tier 2, high, patch_with_position_shuffle_lower_T_and_explicit_accuracy
- Delta: regressed slightly (added newly identified positional-bias issue at lines 119-123 that mirrors the proparalogy bug; recommend the same shuffle fix)
