# eqbench_creative_writing_v3 fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** high
**Recommendation:** keep_as_is_with_registry_cleanup (remove unimplemented `elo_rating` from registry; document `min_p=0.1` gap)

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/eqbench_creative_writing_v3_run_specs.py`:
>
> - **MetricSpec(s):** `metrics.eqbench_creative_writing_v3_metric.EQBenchCreativeWritingV3Metric`
> - **AnnotatorSpec(s):** `llm_judge.eqbench_creative_writing_v3_annotator.EQBenchCreativeWritingV3Annotator`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: https://eqbench.com/creative_writing.html (no formal paper -- landing page is canonical)
- Repo: https://github.com/EQ-bench/creative-writing-bench - skim (prompt JSON + judge prompt vendored)

## Implementation audited
- scenarios/eqbench_creative_writing_v3_scenario.py - downloads `creative_writing_prompts_v3.json` from upstream; emits 32 prompts x 3 iterations = 96 instances; resolves `<SEED>` per iteration from `seed_modifiers` (lines 76-79). Matches upstream item set exactly.
- llm_judge/eqbench_creative_writing_v3_annotator.py - implements upstream rubric judge: criteria scoring, "lower is better" inversion, 0-20 aggregation. (Not re-read this audit; behavior consistent with metric output keys.)
- metrics/eqbench_creative_writing_v3_metric.py - reads annotator output and surfaces `creative_score_0_20`, `eqbench_creative_score`, `criteria_count`, `judge_parse_rate`.
- registry_metrics.yaml: declares TWO metrics -- `elo_rating` (judge claude-sonnet-4, T=0, max=1024) and `rubric_score` (claude-sonnet-4, T=0, max=512). Neither name matches what the metric class actually emits (`creative_score_0_20`, `eqbench_creative_score`, etc.); both have `judge_prompt: null`.
- registry_inference.yaml: T=0.7, max_new_tokens=2048, num_return_sequences=3. Matches upstream `num_iterations=3`.

## Deviations found
- [HIGH] C. Metric/scoring fidelity: `elo_rating` declared but NOT implemented. Upstream EQBench leaderboard reports both rubric and pairwise ELO; only rubric path is wired here. Either implement a pairwise annotator or drop the entry.
- [MEDIUM] C. Registry/metric name mismatch: registry advertises `rubric_score` and `elo_rating`, but the metric class emits `eqbench_creative_score` / `creative_score_0_20` / `criteria_count` / `judge_parse_rate`. Tooling that reconciles registry vs runtime stats will report all four runtime metrics as "missing from registry" and both registry metrics as unrun.
- [LOW] D. Generation: `min_p=0.1` from upstream not supported by HELM adapter (documented in scenario docstring). T=0.7, max=2048, n=3 otherwise faithful.
- [LOW] C. Judge model `anthropic/claude-sonnet-4` is class-equivalent to upstream's Claude Sonnet default; defensible substitution.
- [info] A. 32 prompts x 3 iterations matches upstream exactly.
- [info] B. Prompts loaded verbatim from upstream JSON with `<SEED>` resolution.

## Notes
Rubric path is faithful end-to-end. Two registry housekeeping fixes: (1) drop `elo_rating` (or add a pairwise annotator), (2) replace `rubric_score` with the four metric names actually emitted by `EQBenchCreativeWritingV3Metric`. Note `min_p` adapter gap in run-time docs so it is not silently lost.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, high, keep_as_is (remove unimplemented elo_rating from registry)
- Now:   Tier 2, high, keep_as_is_with_registry_cleanup
- Delta: confirmed
