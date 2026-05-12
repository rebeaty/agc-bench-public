# arena_hard_creative fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** high
**Recommendation:** patch_with_judge_metric_module (implement `metrics/arena_hard_creative_metric.py` + populate `judge_prompt`; consider GPT-4.1 + Gemini-2.5 ensemble per v2.0 README)

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/arena_hard_creative_run_specs.py`:
>
> - **MetricSpec(s):** `metrics.arena_hard_pairwise_metric.ArenaHardPairwiseMetric`
> - **AnnotatorSpec(s):** `llm_judge.arena_hard_pairwise_annotator.ArenaHardPairwiseAnnotator`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: https://arxiv.org/abs/2406.11939 (Li et al., Arena-Hard / BenchBuilder) - skim (abstract only)
- Repo: https://github.com/lmarena/arena-hard-auto - read (README)

## Implementation audited
- scenarios/arena_hard_creative_scenario.py - downloads `data/arena-hard-v2.0/question.jsonl` and `gemini-2.0-flash-001.jsonl` baseline live; filters `category == "creative_writing"`; passes prompt verbatim; baseline attached as `Reference(Output(text=baseline_output))` keyed by `uid` (lines 22-92). Instance count dynamic (~250 per v2.0 README).
- metrics/arena_hard_creative_metric.py - MISSING (`has_metric_file: false`).
- registry_metrics.yaml: single `win_rate` metric, `type: llm_judge`, `judge_model_name: openai/gpt-4-turbo`, `judge_prompt: null`, T=0.0, max=1024, `in_helm: false`.
- registry_inference.yaml: `_use_defaults: true` -> T=0.7, max_new_tokens=512, n=1.

## Deviations found
- [HIGH] C. Metric/scoring fidelity: metric module missing AND `judge_prompt: null`. Upstream pairwise CoT prompt (two A/B orderings, `[[A>>B]]/[[A>B]]/[[A=B]]/[[B>A]]/[[B>>A]]` regex, position-swap tie handling, bootstrap CIs from `show_result.py`) is not encoded. End-to-end win-rate scoring cannot run.
- [MEDIUM] C. Judge model: registry pins `openai/gpt-4-turbo`; the v2.0 README recommends a GPT-4.1 + Gemini-2.5 ensemble for the creative_writing category specifically. Single judge is defensible but off-spec.
- [LOW] D. Generation config: max_new_tokens=512 default may truncate creative responses relative to the (full-length) gemini baseline answer, biasing win-rate. Bumping to >=1024 would be safer.
- [LOW] A. Dataset/instance source: upstream files fetched live, not pinned to a commit hash -- reproducibility risk.
- [info] A. The category filter and baseline-by-uid join are correct.
- [info] B. Prompts pass through verbatim (no template wrapping).

## Notes
Data ingestion and reference handling are faithful. The blocker is missing judge plumbing -- no metric file and a null `judge_prompt` mean win-rate cannot be computed at evaluation time. Fix: port upstream `gen_judgment.py` system prompt verbatim into `judge_prompt`, write `metrics/arena_hard_creative_metric.py` that runs the two-game A/B swap, parses the `[[A>>B]]`-style verdict, and aggregates win_rate with bootstrap CIs (mirror `show_result.py`). Bump max_new_tokens for the model under test, and pin upstream JSONL files to a commit hash.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, high, patch_with_judge_metric_module
- Now:   Tier 2, high, patch_with_judge_metric_module
- Delta: confirmed
