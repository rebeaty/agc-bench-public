# proparalogy fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** high
**Recommendation:** keep_as_is (positional bias bug has been fixed)

## Paper / repo audited
- Paper: Sultan, Bitton, Yosef, & Shahaf (2024). *ParallelPARC: A Scalable Pipeline for Generating Natural-Language Analogies*. NAACL 2024. arXiv:2403.01139 — skim
- Repo: orensul/ParallelPARC — skim (gold_test_set CSVs)

## Implementation audited
- scenarios/proparalogy_scenario.py — reads `gold_set_analogies_w_challenging_distractors_w_randoms.csv`. Lines 106-121 now do **per-instance positional shuffle** with sha256-derived deterministic seed from `sample_id`; CORRECT_TAG attached to whichever label (A/B/C) ends up holding the correct option (lines 132-139). Bug from 2026-04-25 audit is fixed (a comment on line 106 calls out the fix explicitly).
- registry_metrics.yaml: classification_macro_f1 + classification_micro_f1 via MultipleChoiceClassificationMetric
- registry_inference.yaml: `_use_defaults: true` → T=0.7, 512 tokens (T=0 with short max_tokens would be more appropriate for 3-choice MCQ; not a regression but worth pinning).

## Deviations found
- [LOW] C. Metric: paper reports accuracy; HELM uses MultipleChoiceClassificationMetric (macro/micro F1). Acceptable substitute for balanced 3-class MCQ.
- [LOW] D. Generation config: `_use_defaults` (T=0.7, 512 tokens). For deterministic 3-choice MCQ, T=0 with max_tokens=4 would be tighter. Not a fidelity defect but a tuning miss.
- [info] A. 310 analogies in paper gold test; subsampling (if any) is downstream of scenario.
- [info] Positional bias bug from prior audit RESOLVED (lines 106-121 shuffle with seeded RNG).

## Notes
The previously HIGH positional-bias bug has been fixed at the scenario level via per-instance seeded shuffling (deterministic, reproducible, paper-faithful). Implementation now matches the paper's MCQ protocol. Optional polish: pin T=0 and max_tokens=4 in registry_inference.yaml to reflect MCQ-style scoring.

## Compared to prior audit (2026-04-25)
- Prior: Tier 3, high, discuss (positional bias bug must be fixed)
- Now:   Tier 2, high, keep_as_is
- Delta: lifted (Tier 3 → Tier 2) — bug fixed in scenarios/proparalogy_scenario.py lines 106-121, fix is dated 2026-04-25 in code comment
