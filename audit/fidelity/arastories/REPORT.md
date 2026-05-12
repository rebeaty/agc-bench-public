# arastories fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** medium-high
**Recommendation:** patch_with_judge_rubric (populate `judge_prompt` for each of the 5 LLM-judge dimensions; metric .py file optional)

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/arastories_run_specs.py`:
>
> - **MetricSpec(s):** `llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric`
> - **AnnotatorSpec(s):** `llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: https://arxiv.org/abs/2407.07551 (Arabic Automatic Story Generation with LLMs, ArabicNLP 2024) — note: skim (abstract only; PDF body returned binary)
- Repo: https://github.com/UBC-NLP/arastories — note: skim (README + dir listing; src/evaluate.py 404 on raw)

## Implementation audited
- scenarios/arastories_scenario.py — Downloads the UBC-NLP/arastories repo zip and parses three CSVs (MSA: 996 rows, Egyptian: 1000, Moroccan: 1000) for the `Prompt` field. Default `dialect="all"` produces ~2,996 instances. Reference-free (`references=[]`); reference story stored only in `extra_data`. Prompts passed verbatim from the released CSVs.
- metrics/arastories_metric.py — does not exist (consistent with `has_metric_file=false`); evaluation depends on registry-driven LLM-judge metrics only.
- registry_metrics.yaml (lines 129–165): five `llm_judge` metrics — fluency, coherence, following_instructions, consistency, variety — each `judge_model_name: openai/gpt-4-0125-preview`, `judge_temperature: 0.0`, `judge_max_new_tokens: 256`. **All five `judge_prompt` fields are `null`.**
- registry_inference.yaml (lines 46–48): `_use_defaults: true` — paper does not specify gen-time T/max-tokens.

## Deviations found
- [MEDIUM] C. Metric/scoring fidelity: All 5 `judge_prompt` fields are `null`. The 5-dimension rubric (1–5 scale) is documented in the scenario docstring but no concrete prompt template is wired in, so the run cannot reproduce paper primary LLM-judge scores until prompts are populated.
- [LOW] C. Judge model: registry pins `openai/gpt-4-0125-preview` (paper-faithful) rather than AGC's canonical Gemini judge — downstream re-routing may be needed.
- [LOW] D. Generation config: paper-unspecified, so `_use_defaults` is acceptable.
- [info] A. Dataset/instance source: matches the paper repo's three formatted CSVs exactly.
- [info] B. Prompt fidelity: prompts reused verbatim from released CSVs (paper's own GPT-4-generated prompts).

## Notes
Scenario itself is faithful — same dataset files, same dialects, no paraphrasing, references-free as the paper specifies. The blocking gap is purely in the registry: the 5 judge prompts are null, so the LLM-judge evaluation cannot run end-to-end. Fix: write a concise rubric template per dimension (prompt + generated story → integer 1–5 with brief justification) and paste into the five `judge_prompt` fields. No scenario code changes needed.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, high, patch_with_judge_rubric (implement `metrics/arastories_metric.py` + populate `judge_prompt` fields)
- Now:   Tier 2, medium-high, patch_with_judge_rubric (populate `judge_prompt` fields; metric .py file is optional)
- Delta: confirmed (slightly softened: a metric file may not be needed if LLM-judge dispatcher aggregates natively)
