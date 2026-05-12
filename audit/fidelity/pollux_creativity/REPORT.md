# pollux_creativity fidelity audit (re-audit 2026-05-04)

**Tier:** 3
**Confidence:** high
**Recommendation:** discuss (fix scale 0-4 vs 0-1; populate per-criterion judge prompts; clarify metric structure)

## Paper / repo audited
- Paper: https://arxiv.org/abs/2505.24616 ("Eye of Judgement: Dissecting the Evaluation of Russian-speaking LLMs with POLLUX") — "skim" (relied on scenario header; paper reports 35 task types, 2,115 expert-authored prompts, 161,076 evaluation samples across 7 models, LLM-as-judge with detailed criteria on 0-4 scale)
- Repo: https://huggingface.co/datasets/ai-forever/POLLUX — "skim" (verified parquet path used by scenario via the `refs%2Fconvert%2Fparquet` URL)

## Implementation audited
- scenarios/pollux_creativity_scenario.py — Streams the converted-parquet shard `default/test/0000.parquet` directly with pandas (lines 92–95), filters to 7 Russian creative task types (lines 70–78), deduplicates by `instruction` (multiple model outputs per prompt are collapsed). Each instance carries the original `criteria_name`, `criteria_score` (expert annotation), `difficulty`, `domain`, `rubrics` in extra_data. Reference is `reference_answer` if non-empty (most creative tasks lack a single gold).
- No metric file (uses LLM-judge annotator).
- registry_metrics.yaml (line 2105): registers two judges `llm_judge_creativity` and `llm_judge_originality`, both with judge_model_name=`openai/gpt-4`, T=0.0, max_new_tokens=512, judge_prompt=null. `in_helm: false`.
- registry_inference.yaml (line 520): `_use_defaults: true`.

## Deviations found
- [HIGH] C. Metric/scoring fidelity (scale): paper uses a 0-4 scale per criterion; registered judges have no rubric (`judge_prompt: null`) so the actual scale produced is unknown. Without the per-criterion rubrics from the POLLUX dataset (each instance carries its own `rubrics` field), the judge cannot reproduce the paper's scoring. Need to inject the per-instance `rubrics` into the judge prompt at runtime.
- [HIGH] C. Metric/scoring fidelity (criteria coverage): paper evaluates 8+ creativity-related criteria (Креативность, Драматургия, Выразительность диалога, Качество рифмы, Литературные акценты, Соблюдение образа персонажа, Размер стиха, Попадание в жанр) — see scenario lines 81–90. Registry collapses this into just two judges (creativity + originality), losing the per-criterion granularity that is the entire point of POLLUX.
- [MEDIUM] A. Dataset/instance source: dedup-by-instruction is correct, but mixing 7 task types into one undifferentiated unit makes the per-task-type comparison the paper relies on impossible without splitting the run. Recommend a `task_type` arg to allow per-type runs, or split into 7 RunSpecs.
- [MEDIUM] B. Prompt fidelity: instruction is passed verbatim from the dataset (good), no system prompt added — appropriate since instructions are self-contained Russian.
- [LOW] D. Generation configuration: defaults; for creative Russian text generation, T=0.7+ would better match the paper's contestant settings.

## Notes
The scenario is engineered well — streaming a converted-parquet shard, dedup, criterion-rich extra_data — but the metric registry is fundamentally underspecified. The fix is two-pronged: (1) write per-criterion judge prompts that consume the per-instance `rubrics` field and emit a 0-4 score; (2) split the criteria into the paper's panel rather than collapsing to "creativity + originality". Tier 3 retained until those judge prompts exist.

## Compared to prior audit (2026-04-25)
- Prior: Tier 3, high, discuss (fix scale, judge model, metric structure before reporting)
- Now:   Tier 3, high, discuss
- Delta: confirmed
