# ocw fidelity audit

**Tier:** 1
**Confidence:** high
**Recommendation:** keep_as_is (verify 556 vs 618 eval pool count)

## Paper / repo audited
- Paper: Naeini et al., "Large Language Models are Fixated by Red Herrings..." (NeurIPS 2023, arXiv 2306.11167)
- Repo: TaatiTeam/OCW

## Implementation audited
- scenarios/ocw_scenario.py — system+user prompt verbatim from upstream `notebooks/run_openai.ipynb`: Round 3 framing, red-herring warning, `Clues: ...`, `Solved wall:` suffix.
- run_specs/ocw_run_specs.py — zero-shot (`max_train_instances=0`), T=0.0, max_tokens=144, no stop sequences. HELM's generic in-context adapter would inject misleading `n/a` train fillers, so zero-shot was chosen as strictest safe variant.
- metrics/group_match_score_metric.py — mirrors `evaluate_only_connect.py`: strips "Group N:" prefixes and trailing "Connection:" annotations, splits on newlines/commas, normalizes case/whitespace/punctuation, performs order-invariant set matching with one-to-one assignment. Reports `correct_groups` (0–4), `full_wall` binary, `group_match_score` (correct/4), plus auxiliary `hallucinated_words` and `empty_group_slots`.
- 618 puzzles (62 train / 62 val / 494 test). Sampling manifest reports `total_eval_instances=556` capped to 200.

## Deviations found
- [LOW] **Few-shot variants not reproduced** (paper's zero-shot baseline preserved).
- [INFO] **Eval pool 556 vs 618**: doesn't match documented split combo; only 200 sampled IDs actually evaluated. Worth verifying.
- [INFO] No partial-credit Jaccard; aligns with paper's exact-set matching.

## Notes
References built from `groups.group_i.gt_words`. No LLM judge — pure formula-based set comparison, matching paper's Task 1 evaluation. Task 2 (Connections) excluded here, handled by separate `ocw_connections` benchmark. HIGH FIDELITY — Tier 1.
