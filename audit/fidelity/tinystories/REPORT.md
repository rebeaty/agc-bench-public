# tinystories fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** high
**Recommendation:** keep_as_is (judge-model swap is benchmark-wide standardization; populate the 4 judge prompts with paper's grammar/creativity/consistency rubric)

## Paper / repo audited
- Paper: https://arxiv.org/abs/2305.07759 (Eldan & Li, "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?", ICLR 2024) — "skim" (paper evaluates story completions on Grammar, Creativity, Consistency on 1-10 scale + age-group estimation, judge model GPT-4)
- Repo: https://huggingface.co/datasets/roneneldan/TinyStories — "skim" (verified `Evaluation prompts.yaml` containing 44 story beginnings; this is the official test set)

## Implementation audited
- scenarios/tinystories_scenario.py — Downloads `Evaluation prompts.yaml` from HF mirror at `resolve/main/`. Loads 44 story beginnings. Prompt at lines 68–73 frames the task: "The student is given the beginning of a story and needs to complete it into a full story. Write the rest of the story in simple language that a young child could understand." with the beginning bracketed by `***`. No references attached (LLM-judge eval).
- No metric file in `metrics/` — but registry references `llm_judge.tinystories_metric.TinyStoriesMetric` for the formula-based stats.
- registry_metrics.yaml (line 2885): registers 4 LLM-judge axes (`grammar_score`, `creativity_score`, `consistency_score`, `age_group_ordinal`) with `google/gemini-2.5-flash-lite`, T=0.0, max_new_tokens=128, judge_prompt=null. Plus 2 formula-based stats (`valid_judge_rate`, `scored_completion_count`) wired to `llm_judge.tinystories_metric.TinyStoriesMetric`.
- registry_inference.yaml (line 682): `_use_defaults: true`.

## Deviations found
- [LOW] C. Metric/scoring fidelity: paper uses GPT-4 as judge; we substitute `google/gemini-2.5-flash-lite`. This is the documented benchmark-wide judge standardization choice — defensible.
- [MEDIUM] C. Metric/scoring fidelity (rubric): the 4 judge axes have `judge_prompt: null`. Paper's rubric is explicit ("Grammar: 8/10, Creativity: 7/10, Consistency: 7/10" + age group) — needs to be populated in the yaml or in `llm_judge.tinystories_metric.TinyStoriesMetric` for reproducibility.
- [LOW] B. Prompt fidelity: prompt template is paraphrased ("The student is given the beginning..."); paper's exact GPT-4 evaluation harness uses a similar but not identical wrapper. Spirit preserved.
- [LOW] A. Dataset/instance source: 44 prompts, matches paper's official eval set.
- [info] D. Generation configuration: defaults; for child-language story completion, T=0.7+ would better fit creative-completion intent — defaults may be too low.

## Notes
This is a clean reproduction of the TinyStories evaluation harness. The judge swap (GPT-4 → Gemini Flash Lite) is the benchmark-wide standardization convention and is fine as long as it's footnoted. The two real fixes: (1) populate the 4 `judge_prompt` fields with paper's "Grammar/Creativity/Consistency on 1-10 scale + estimate age group" rubric; (2) confirm `llm_judge.tinystories_metric.TinyStoriesMetric` parses the judge output format the rubric specifies. Tier 2 confirmed.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, high, keep_as_is (judge swap is benchmark-wide standardization)
- Now:   Tier 2, high, keep_as_is
- Delta: confirmed
