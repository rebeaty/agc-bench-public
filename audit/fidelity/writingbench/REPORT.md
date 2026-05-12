# writingbench fidelity audit

**Tier:** 1
**Confidence:** high
**Recommendation:** keep_as_is (reconcile judge model between registry and run_spec)

## Paper / repo audited
- Paper: Wu et al., "WritingBench: A Comprehensive Benchmark for Generative Writing" (arXiv:2503.05244)
- Repo: https://github.com/X-PLUG/WritingBench

## Implementation audited
- scenarios/writingbench_scenario.py — passes `query` directly as user text with no system prompt; matches repo `generate_response.py` (`messages=[{"role":"user","content":query}]`).
- run_specs/writingbench_run_specs.py — adapter uses empty instructions/prefixes, effectively zero-shot raw query.
- llm_judge/writingbench_annotator.py — reproduces paper Appendix C.6 rubric verbatim (1-2 / 3-4 / 5-6 / 7-8 / 9-10 bands, JSON `{score, reason}`) with paper's "expert evaluator" system message.
- llm_judge/writingbench_metric.py — emits `writingbench_score` (mean of per-criterion 1-10 scores across 5 checklist criteria), `writingbench_valid_criteria_rate`, `writingbench_criteria_count`.
- 1,000 queries (6 domains, 100 subdomains; 555 EN / 445 ZH). Eval cap: 200 random.

## Deviations found
- [MEDIUM] **Judge model reconciliation**: manifest + registry declare `anthropic/claude-sonnet-4`; run_spec default is `google/gemini-2.5-flash-lite` (env-overrideable via `WRITINGBENCH_JUDGE_MODEL_OVERRIDE`). Paper validates `claude-3-5-sonnet` plus a finetuned 7B critic.
- [LOW] **Per-domain breakdowns not surfaced as separate stats** (minor gap).
- [LOW] **Inference config drift**: T=0.7, top_p=0.8, top_k=20, max_tokens=16000 in paper README; run_spec omits `top_k` and `top_p`.

## Notes
PROMPT, METRIC, INSTANCE COUNT, JUDGE/REFERENCE all faithful. No reference targets (open-ended generation). Tier 1 because core implementation tracks paper closely; only deviations are judge-model registry/run_spec mismatch and minor inference parameter drift.
