# liveideabench fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** high
**Recommendation:** keep_as_is_with_caveat (single-judge substitution acceptable; reconcile stale registry rubric line vs. annotator)

## Paper / repo audited
- Paper: https://arxiv.org/abs/2412.17596 (Ruan et al., 2024) — note: skim (abstract; confirms 5 dimensions: originality, feasibility, fluency, flexibility, clarity; multi-model judge panel)
- Repo: https://github.com/x66ccff/liveideabench — note: skim (scenario header references `utils/prompts.json` with idea_prompt and critic_prompt verbatim)

## Implementation audited
- scenarios/liveideabench_scenario.py — Uses verbatim `idea_prompt.description` from upstream `utils/prompts.json` (lines 39–48: "I'll be submitting your next responses to a 'Good Scientific Idea' expert review panel...100 words total..."). Loads 1,180 keywords from column B of `keywordsEverywhere20241216.xlsx` (rows 3+). 22-domain coverage matches paper.
- metrics/liveideabench_metric.py — `LiveIdeaBenchMetric` aggregates judge annotations from `liveideabench_v2_evaluator` annotator. Reports originality, feasibility, clarity, fluency, flexibility (30th percentile of overall scores), average, plus parse-rate diagnostics. Faithful to paper's 5-dimension scheme + Guilford-style flexibility.
- registry_metrics.yaml (lines 1592–1600): single `llm_judge_creativity` metric — `openai/gpt-4`, T=0.0, max_new_tokens=512, `judge_prompt: null`. **This is the stale single-rubric registration; the actual annotator (`liveideabench_v2_evaluator`) implements the upstream 5-dimension rubric and pairwise fluency stage.**
- registry_inference.yaml (lines 373–375): `_use_defaults: true`. Run-spec layer reportedly pins T=0.7, max_tokens=220, num_outputs=2 to enable the fluency pairwise stage.

## Deviations found
- [MEDIUM] C. Judge protocol: Upstream uses a dynamic multi-model critic panel sampled from `config.py`. AGC implementation uses a single LLM judge (with fallback). Defensible (HELM canonicalization), but means scores are not directly leaderboard-comparable.
- [LOW] C. Registry hygiene: registry_metrics.yaml exposes only a generic `llm_judge_creativity` line, while the actual scoring is performed by a dedicated annotator implementing the 5-dimension + pairwise-fluency rubric. The registry block is stale relative to the annotator/metric pair.
- [LOW] C. Judge model: registry pins `openai/gpt-4`; the run-spec apparently uses `openai/gpt-4.1-mini`. Minor inconsistency.
- [LOW] D. Generation config: registry says `_use_defaults: true` but run-specs override with T=0.7, max_tokens=220, num_outputs=2 (needed for the pairwise fluency stage). Should be documented at registry level for transparency.
- [info] A. Dataset: 1,180 keywords across 22 domains, matches paper exactly.
- [info] B. Prompt fidelity: idea_prompt and critic_prompt copied verbatim from `utils/prompts.json`.

## Notes
This is a high-fidelity port — verbatim prompts, faithful 5-dimension rubric, exact keyword set, paper-accurate fluency pairwise A/B/C/D mapping (A=10, B=7, C=4, D=1). The single-judge substitution is the only substantive deviation, and it is a deliberate AGC-wide canonicalization rather than a bug. Recommended cleanup: replace the `llm_judge_creativity` placeholder in registry_metrics.yaml with explicit metric entries that mirror the annotator outputs (originality, feasibility, clarity, fluency, flexibility, average), and pin the gen-time config in registry_inference.yaml.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, high, keep_as_is (single-judge substitution acceptable; reconcile registry rubric)
- Now:   Tier 2, high, keep_as_is_with_caveat
- Delta: confirmed
