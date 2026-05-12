# nyt_connections fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** medium-high
**Recommendation:** keep_as_is_with_minor_fixes (note prompt provenance, set T=0.5, document instance scope)

## Paper / repo audited
- Paper: Loredo Lopez, McDonald & Emami (COLING 2025), "NYT-Connections: A Deceptively Simple Text Classification Task" (arXiv:2412.01621v3)
- Data: tm21cy/NYT-Connections HF dataset (358 puzzles)

## Implementation audited
- scenarios/nyt_connections_scenario.py — terse prompt adapted from `lechmazur/nyt-connections` (community), not paper's IO/CoT/CoT-SC prompts (Appendix Figs. 6-8) which include detailed background, instructions, and an example game.
- run_specs/nyt_connections_run_specs.py — single One-Try-equivalent setting; T=0.7.
- metrics/group_match_score_metric.py — order-invariant token-set equality, 0.25 per correct group, continuous group_match_score.

## Deviations found
- [HIGH] **Prompt source mismatch**: HELM uses community prompt rather than paper's verbose IO/CoT/CoT-SC prompts. Missing system-2 framing.
- [MEDIUM] **Configurations collapsed**: paper defines 3 settings (One Try / No Hints / Full Hints with up to 4 retries; scoring 0/25/50/75/100). HELM evaluates only One-Try-equivalent.
- [LOW] **Temperature drift**: T=0.7 vs paper's 0.5 (non-LLaMA) / 0.6 (LLaMA).
- [LOW] **Instance scope**: full 358 vs paper's 100 median-difficulty subset (broader, undocumented).
- [LOW] Metric docstring still says OCW (repurposed); no fidelity issue but cosmetic.

## Notes
No judge needed (formula-based exact match). Tier 2 due to prompt provenance and missing CoT/multi-attempt configs. Update metric docstring; consider T=0.5 to match paper.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, medium-high, keep_as_is_with_minor_fixes
- Now:   Tier 2, medium-high, keep_as_is_with_minor_fixes
- Delta: confirmed (verified scenarios/nyt_connections_scenario.py lines 42-46 prompt is from lechmazur/nyt-connections community repo not paper Appendix; line 53 loads `train` split as test; registry exposes `group_match_score` via metrics.group_match_score_metric.GroupMatchScoreMetric; inference `_use_defaults` → T=0.7 not paper's 0.5)
